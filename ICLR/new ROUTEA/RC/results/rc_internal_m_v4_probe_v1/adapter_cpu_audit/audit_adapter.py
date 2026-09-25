"""Read-only CPU decomposition of the frozen PRE_REAL V4 adapter.

No labels, fitting, reference tokens, encoder forward or GPU required.
Compute residual energy via the 16-dimensional up-projection Gram matrix.
"""
import hashlib
import json
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'results/rc_prellm_m_adapter_v2/input_manifest.json'
CP = ROOT / 'results/rc_internal_m_condition_scale_v4/PRE_REAL/snapshots/0128.pt'
AUTH = ROOT / 'registry/rc_internal_m_condition_scale_v4_authority_20260924.json'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}

def stats(x):
    a = np.asarray(x, dtype=float)
    return {'min': float(a.min()), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'max': float(a.max())}

m = json.loads(MANIFEST.read_text())
a = json.loads(AUTH.read_text())
cp = torch.load(CP, map_location='cpu', weights_only=True)
assert cp['step'] == 128 and cp['arm'] == 'PRE_REAL'
s = cp['adapter']
down, bias = s['down.weight'], s['down.bias']
up, ubias = s['up.weight'].double(), s['up.bias'].double()
scale = float(s['residual_scale'])
gram = (up.T @ up) * scale ** 2
cross = (up.T @ ubias) * scale ** 2
benergy = float(ubias.square().sum()) * scale ** 2

def energy(h):
    return torch.einsum('...i,ij,...j->...', h, gram, h)

rows = []
for split, source_rows in [('train', m['train_rows']), ('probe', m['probe_rows'])]:
    for row in source_rows:
        path = ROOT / 'results/rc_prellm_m_adapter_v2/encoder_cache' / row['query_id'] / 'payload.pt'
        cache = torch.load(path, map_location='cpu', weights_only=True)
        x = cache['merged'].float()
        xn = F.layer_norm(x, (x.shape[-1],))
        content = F.linear(xn, down[:, :-1], bias)
        h0 = F.gelu(content).double()
        z = (torch.tensor(row['M'], dtype=torch.float32).clamp_min(float(s['mass_epsilon'])).log()
             - s['mass_log_mean']) / s['mass_log_std']
        cond = (z * a['condition_gain'])[:, None] * down[:, -1][None, :]
        native_energy = float(x.double().square().sum(dim=1).mean())
        records = []
        for start in range(0, 128, 16):
            h = F.gelu(content[None, :, :] + cond[start:start+16, None, :]).double()
            dh = h - h0[None, :, :]
            total = energy(dh).mean(dim=1)
            common = energy(dh.mean(dim=1))
            spatial = torch.clamp(total-common, min=0)
            residual = energy(h).mean(dim=1) + 2 * torch.einsum('bpi,i->b', h, cross)/x.shape[0] + benergy
            for j in range(h.shape[0]):
                k = start + j
                records.append({'candidate_position': k, 'M': row['M'][k], 'z': float(z[k]),
                    'condition_to_content_preactivation_rms': float(cond[k].square().mean().sqrt() / content.square().mean().sqrt()),
                    'residual_to_source_rms': float((residual[j]/native_energy).sqrt()),
                    'M_specific_residual_to_source_rms': float((total[j]/native_energy).sqrt()),
                    'M_specific_spatial_energy_share': float(spatial[j]/total[j]) if total[j] > 0 else 0.0,
                    'M_specific_common_energy_share': float(common[j]/total[j]) if total[j] > 0 else 0.0,
                    'M_specific_energy': float(total[j]), 'M_specific_spatial_energy': float(spatial[j])})
        # Direct 3584-D verification of Gram decomposition, at fixed candidate 0.
        exact = F.linear(F.gelu(content+cond[0])-F.gelu(content), s['up.weight']) * scale
        direct = float(exact.double().square().sum(1).mean())
        inferred = records[0]['M_specific_energy']
        assert abs(direct-inferred) <= max(1e-8, direct*2e-5), (direct, inferred)
        rows.append({'query_id': row['query_id'], 'split': split, 'patch_count': len(x),
                     'cache': sha(path), 'source_rms_per_coordinate': float(x.square().mean().sqrt()),
                     'content_preactivation_rms': float(content.square().mean().sqrt()),
                     'direct_gram_relative_error': abs(direct-inferred)/max(direct,1e-12),
                     'candidates': records})
        print(split, row['query_id'], 'done', flush=True)
        del cache, x, xn

summary = {}
for split in ['train', 'probe']:
    subset = [r for r in rows if r['split'] == split]
    pairs = [p for r in subset for p in r['candidates']]
    keys = ['z', 'condition_to_content_preactivation_rms', 'residual_to_source_rms',
            'M_specific_residual_to_source_rms', 'M_specific_spatial_energy_share',
            'M_specific_common_energy_share']
    summary[split] = {'queries': len(subset), 'pairs': len(pairs),
        **{k: stats([p[k] for p in pairs]) for k in keys},
        'energy_weighted_M_specific_spatial_share': sum(p['M_specific_spatial_energy'] for p in pairs)/sum(p['M_specific_energy'] for p in pairs),
        'fraction_pairs_common_energy_above_90pct': sum(p['M_specific_common_energy_share'] > .9 for p in pairs)/len(pairs)}
result = {'status': 'FROZEN_ADAPTER_CPU_DECOMPOSITION_PASS', 'cpu_threads': 2,
    'new_training': False, 'new_encoder_or_roma_forwards': 0, 'label_reads': 0,
    'condition_gain': a['condition_gain'], 'sources': [sha(MANIFEST), sha(CP), sha(AUTH), sha(Path(__file__))],
    'definition': 'delta_M(p)=residual(p,M)-residual(p,standardized_zero); decompose delta into patch-mean vector plus patch-varying residual, before BF16 cast and before LLM. Energy uses original up-projection Gram matrix; these are not output token effects.',
    'summary': summary, 'rows': rows}
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
