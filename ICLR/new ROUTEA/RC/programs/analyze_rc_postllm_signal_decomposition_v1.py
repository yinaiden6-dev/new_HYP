#!/usr/bin/env python3
"""Frozen POST_REAL causal residual/assignment decomposition, CPU only.

Existing TRAIN16 + opened PROBE8, fixed original C128 and joint INTERNAL3 head.
No training, threshold choice, model selection, LLM, RoMa, or new token encoding.
Per-candidate JSON/NPZ checkpoints make short bounded invocations resumable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from rc_postllm_m_backend_v1 import load_projection, predict_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_postllm_m_signal_decomposition_v1'
POST = ROOT/'results/rc_postllm_m_v1'
AUTH = ROOT/'registry/rc_postllm_m_v1_authority_20260924.json'
MANIFEST = ROOT/'results/rc_prellm_m_adapter_v2/input_manifest.json'
SNAPSHOT = POST/'POST_REAL/snapshots/0128.pt'
TOKEN_ARMS = ['CONSTANT', 'NATIVE', 'COMMON_ONLY', 'SPATIAL_ONLY', 'RECOMPOSED']
SCORE_ARMS = TOKEN_ARMS + ['NATIVE_FIXED_ARGMAX', 'COMMON_FIXED_ARGMAX']

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def checked(binding):
    path = Path(binding['path'])
    if sha(path) != binding['sha256']:
        raise ValueError(f'source SHA mismatch: {path}')
    return path

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def ratio(num, den):
    return float(num/den) if den > 0 else None

def parse_rows(value, n):
    if value == 'all':
        return list(range(n))
    ids = []
    for part in value.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            ids.extend(range(lo, hi+1))
        else:
            ids.append(int(part))
    if len(ids) != len(set(ids)) or any(i < 0 or i >= n for i in ids):
        raise ValueError('rows must be unique indices in frozen TRAIN16 then PROBE8 order')
    return sorted(ids)

def decision(row, scores, head):
    L = np.asarray(scores, np.float64)
    raw = np.asarray(row['raw_scores'], np.float64)
    w = int(row['winner_index'])
    idx = np.asarray(row['challenger_positions'], np.int64)
    assert list(idx) == [k for k in range(128) if k != w]
    features = np.stack(((raw[idx]-raw[w])/max(float(raw.std()), 1e-12),
        (L[idx]-L[w])/(np.abs(L[idx])+abs(L[w])+1e-12), np.ones(127)), axis=1)
    logits = features @ np.asarray(head, np.float64)
    k = int(np.argmax(logits))
    picked = int(idx[k]) if logits[k] > 0 else w
    return {'theta': list(map(float, head)), 'feature_order': ['raw_standardized_delta','sym_L','bias'],
        'logits': logits.tolist(), 'challenger_positions': idx.tolist(), 'hold_logit': 0.0,
        'prediction_position': picked, 'prediction_identity': row['candidate_identities'][picked],
        'switched': picked != w, 'best_challenger_position': int(idx[k]),
        'best_challenger_logit': float(logits[k])}

def reference_stats(indices, nref):
    counts = torch.bincount(indices, minlength=nref).double()
    present = counts[counts > 0]
    p = present / len(indices)
    entropy = float(-(p*p.log()).sum())
    return {'unique_reference_tokens': int(len(present)), 'reference_token_count': int(nref),
        'query_patch_count': int(len(indices)), 'reference_fraction_covered': len(present)/nref,
        'unique_over_max_possible': len(present)/min(nref, len(indices)),
        'largest_many_to_one_fraction': float(p.max()),
        'effective_reference_tokens': math.exp(entropy)}

def projected(model, cache, image_hidden):
    full = cache['hidden'].clone()
    full[cache['image_mask']] = image_hidden
    return predict_projection(model, {**cache, 'hidden': full}, None, 0.0)

def prep_protocol():
    a = read(AUTH)
    m = read(MANIFEST)
    sources = [bind(AUTH), bind(MANIFEST), bind(SNAPSHOT),
        bind(ROOT/'programs/rc_postllm_m_backend_v1.py'),
        bind(ROOT/'programs/rc_prellm_m_scaled_adapter_v4.py'),
        bind(ROOT/'programs/rc_prellm_m_adapter_v1.py'), bind(Path(__file__))]
    protocol = {'schema': 1, 'status': 'FROZEN_POST_RESIDUAL_DECOMPOSITION', 'sources': sources,
        'model': 'POST_REAL128', 'head': 'same original joint INTERNAL3; no refit',
        'candidate_source': 'original ColNomic natural C128',
        'row_order': [r['query_id'] for r in m['train_rows']+m['probe_rows']],
        'token_arms': TOKEN_ARMS, 'score_arms': SCORE_ARMS,
        'definition': 'a0=actualBF16_adapter(hidden,z=0); aM=actualBF16_adapter(hidden,M); delta=float32(aM)-float32(a0); common=mean_patch(delta); spatial=delta-common; each intervened hidden recastBF16 before exact frozen projector.',
        'fixed_argmax': 'Use each query patch best reference-token index under SAME POST_REAL adapter at constant M; read native/common cosine at that fixed index, without rematching.',
        'label_policy': 'Labels from already opened POST result only join after score calculation; no selection by outcome.',
        'new_training': False, 'new_llm_or_roma_forwards': 0, 'cpu_only': True}
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json') != protocol:
        raise ValueError('protocol/code changed since prior checkpoint; use a new version')
    write(OUT/'protocol.json', protocol)
    return a, m, bind(OUT/'protocol.json')

def summary():
    completed = sorted((OUT/'queries').glob('*/result.json'))
    if not completed:
        return
    labels = {x['query_id']: x for x in read(POST/'result.json')['rows']}
    allrows = [read(p) for p in completed]
    totals = {}
    for split in ['train', 'probe']:
        rows = [r for r in allrows if r['split'] == split]
        if not rows:
            continue
        subtotal = {}
        for arm in SCORE_ARMS:
            corr, rescue, broken, changed = [], [], [], []
            for rec in rows:
                q = rec['query_id']; lab = labels[q]
                good = rec['arms'][arm]['decision']['prediction_identity'] == lab['identity']
                raw = lab['correct']['RAW']
                if good: corr.append(q)
                if good and not raw: rescue.append(q)
                if raw and not good: broken.append(q)
                if rec['arms'][arm]['decision']['prediction_identity'] != lab['selected']['POST_REAL']:
                    changed.append(q)
            subtotal[arm] = {'n': len(rows), 'correct': len(corr), 'rescues': rescue,
                'breaks': broken, 'changed_vs_sealed_GPU_POST_REAL': changed}
        totals[split] = subtotal
    write(OUT/'summary.json', {'status': 'ALL24_COMPLETE' if len(allrows)==24 else 'PARTIAL_ROWS_COMPLETE',
        'completed_queries': len(allrows), 'total_queries': 24, 'summary': totals,
        'source_results': [bind(p) for p in completed], 'opened_label_source': bind(POST/'result.json'),
        'native_CPU_GPU_decision_disagreements': sum(not r['cpu_gpu_anchors']['NATIVE']['same_decision'] for r in allrows),
        'constant_CPU_GPU_decision_disagreements': sum(not r['cpu_gpu_anchors']['CONSTANT']['same_decision'] for r in allrows),
        'new_training': False, 'llm_or_roma_forwards': 0,
        'scope': 'frozen diagnostic on opened TRAIN16/PROBE8; not independent accuracy or refit'} )

@torch.no_grad()
def process_row(index, row, split, model, small, head, protocol, started, budget):
    folder = OUT/'queries'/row['query_id']
    final = folder/'result.json'
    if final.exists():
        if read(final)['protocol'] != protocol:
            raise ValueError('row protocol mismatch')
        return True
    sealpath = ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/row['query_id']/'validation.json'
    seal = read(sealpath)
    cache = torch.load(checked(seal['payload']), map_location='cpu', weights_only=True)
    hidden = cache['hidden'][cache['image_mask']]
    assert hidden.dtype == torch.bfloat16 and hidden.ndim == 2
    small.conditioning = 'constant'
    constant_hidden = small(hidden, row['M'][0])
    constant_tokens = projected(model, cache, constant_hidden)
    constant_unit = F.normalize(constant_tokens.double(), dim=-1)
    small.conditioning = 'real'
    gpu_dir = POST/'POST_REAL'/('endpoints/0128' if split == 'train' else 'probe')
    gpu = {'NATIVE': read(gpu_dir/'native'/(row['query_id']+'.json')),
           'CONSTANT': read(gpu_dir/'constant'/(row['query_id']+'.json'))}
    completed = []
    rowstarted = time.monotonic()
    for i in range(128):
        path = folder/'candidates'/f'{i:04d}.json'
        if path.exists():
            rec = read(path)
            if rec['protocol'] != protocol or rec['query_id'] != row['query_id'] or rec['candidate_position'] != i:
                raise ValueError('candidate resume mismatch')
            checked(rec['patch_evidence'])
            completed.append(rec)
            continue
        if time.monotonic()-started >= budget:
            write(folder/'progress.json', {'query_id': row['query_id'], 'row_index': index,
                'candidates': len(completed), 'total': 128, 'status': 'PARTIAL_BUDGET', 'protocol': protocol})
            return False
        tick = time.monotonic()
        refpath = checked(row['reference_tokens'][i])
        refpayload = torch.load(refpath, map_location='cpu', weights_only=True)
        ref = F.normalize(refpayload['tokens'].double(), dim=-1)
        actual = small(hidden, row['M'][i])
        delta = actual.float()-constant_hidden.float()
        common = delta.mean(0, keepdim=True)
        spatial = delta-common
        worlds = {'NATIVE': actual,
            'COMMON_ONLY': (constant_hidden.float()+common).to(torch.bfloat16),
            'SPATIAL_ONLY': (constant_hidden.float()+spatial).to(torch.bfloat16),
            'RECOMPOSED': (constant_hidden.float()+common+spatial).to(torch.bfloat16)}
        reconstruction_error = float((worlds['RECOMPOSED'].float()-actual.float()).abs().max())
        if not torch.equal(worlds['RECOMPOSED'], actual):
            raise ValueError(f'BF16 hidden recomposition failed: {reconstruction_error}')
        total = float(delta.double().square().sum(1).mean())
        common_energy = float(common.double().square().sum())
        spatial_energy = float(spatial.double().square().sum(1).mean())
        closure = abs(total-common_energy-spatial_energy)
        if closure > max(1e-9, total*1e-5):
            raise ValueError('float residual orthogonality closure failed')
        tokens = {'CONSTANT': constant_tokens}
        for arm in ['NATIVE','COMMON_ONLY','SPATIAL_ONLY']:
            tokens[arm] = projected(model, cache, worlds[arm])
        # Same exact hidden tensor, deterministic same projector: no redundant GEMM.
        tokens['RECOMPOSED'] = tokens['NATIVE']
        units = {arm: F.normalize(t.double(), dim=-1) for arm,t in tokens.items()}
        best, locations, stats = {}, {}, {}
        for arm in TOKEN_ARMS:
            if arm == 'RECOMPOSED':
                best[arm], locations[arm] = best['NATIVE'], locations['NATIVE']
            else:
                best[arm], locations[arm] = (units[arm] @ ref.T).max(dim=1)
            cosine = (units[arm]*constant_unit).sum(1)
            stats[arm] = {**reference_stats(locations[arm], len(ref)),
                'argmax_changed_fraction_vs_constant': float((locations[arm] != locations['CONSTANT']).double().mean()),
                'mean_token_cosine_vs_constant': float(cosine.mean()),
                'minimum_token_cosine_vs_constant': float(cosine.min()),
                'mean_normalized_token_change_norm': float((units[arm]-constant_unit).norm(dim=1).mean())}
        fixed_ref = ref[locations['CONSTANT']]
        fixed_native = (units['NATIVE']*fixed_ref).sum(1)
        fixed_common = (units['COMMON_ONLY']*fixed_ref).sum(1)
        gain_native = best['NATIVE']-fixed_native
        gain_common = best['COMMON_ONLY']-fixed_common
        if float(gain_native.min()) < -1e-12 or float(gain_common.min()) < -1e-12:
            raise ValueError('free MaxSim must dominate fixed-constant correspondence')
        scores = {arm: float(best[arm].mean()) for arm in TOKEN_ARMS}
        scores.update(NATIVE_FIXED_ARGMAX=float(fixed_native.mean()), COMMON_FIXED_ARGMAX=float(fixed_common.mean()))
        payload_path = folder/'patch_evidence'/f'{i:04d}.npz'
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = payload_path.with_name(payload_path.name+f'.tmp.{os.getpid()}')
        with tmp.open('wb') as f:
            np.savez_compressed(f, token_arms=np.asarray(TOKEN_ARMS),
                maxsim_by_patch=np.stack([best[k].numpy() for k in TOKEN_ARMS]),
                best_reference_token=np.stack([locations[k].numpy() for k in TOKEN_ARMS]),
                fixed_native_by_patch=fixed_native.numpy(), fixed_common_by_patch=fixed_common.numpy(),
                M_specific_hidden_delta_norm=delta.double().norm(dim=1).numpy(),
                M_specific_hidden_spatial_norm=spatial.double().norm(dim=1).numpy(),
                native_unit_token_change_norm=(units['NATIVE']-constant_unit).norm(dim=1).numpy(),
                common_unit_token_change_norm=(units['COMMON_ONLY']-constant_unit).norm(dim=1).numpy(),
                common_hidden_vector=common.squeeze(0).numpy())
        os.replace(tmp, payload_path)
        rec = {'protocol': protocol, 'query_id': row['query_id'], 'candidate_position': i,
            'candidate_id': row['candidate_ids'][i], 'M': row['M'][i], 'scores': scores,
            'reference_binding': row['reference_tokens'][i], 'patch_evidence': bind(payload_path),
            'token_and_matching_stats': stats,
            'residual_decomposition': {'actual_BF16_delta_energy': total, 'common_energy': common_energy,
                'spatial_energy': spatial_energy, 'common_fraction': ratio(common_energy,total),
                'spatial_fraction': ratio(spatial_energy,total), 'float_energy_closure_error': closure,
                'BF16_recomposed_max_abs_error': reconstruction_error,
                'recomposed_projection_reused_from_identical_native_hidden': True,
                'native_hidden_sha256': tensor_sha(actual)},
            'assignment_gain': {'native_mean': float(gain_native.mean()), 'common_mean': float(gain_common.mean()),
                'native_min': float(gain_native.min()), 'common_min': float(gain_common.min())},
            'cpu_gpu_content_abs_error': {arm: abs(scores[arm]-gpu[arm]['L'][i]) for arm in gpu},
            'candidate_seconds': time.monotonic()-tick}
        write(path, rec); completed.append(rec)
        if (i+1)%16 == 0:
            print(json.dumps({'query_id':row['query_id'],'row':index,'candidates':i+1,'seconds':time.monotonic()-rowstarted}),flush=True)
    arms = {arm: {'L': [rec['scores'][arm] for rec in completed]} for arm in SCORE_ARMS}
    for arm in SCORE_ARMS:
        arms[arm]['decision'] = decision(row, arms[arm]['L'], head)
    assert arms['NATIVE']['L'] == arms['RECOMPOSED']['L']
    anchors = {}
    for arm in gpu:
        sealed = gpu[arm]['decision']; current = arms[arm]['decision']
        anchors[arm] = {'source':bind(gpu_dir/('native' if arm=='NATIVE' else 'constant')/(row['query_id']+'.json')),
            'max_L_abs_error':float(np.max(np.abs(np.asarray(arms[arm]['L'])-gpu[arm]['L']))),
            'max_logit_abs_error':float(np.max(np.abs(np.asarray(current['logits'])-sealed['logits']))),
            'same_decision':current['prediction_identity']==sealed['prediction_identity'] and current['switched']==sealed['switched'],
            'sealed_GPU_prediction':sealed['prediction_identity'], 'CPU_prediction':current['prediction_identity']}
    common_sum = sum(x['residual_decomposition']['common_energy'] for x in completed)
    total_sum = sum(x['residual_decomposition']['actual_BF16_delta_energy'] for x in completed)
    write(final, {'status':'FROZEN_POST_DECOMPOSITION_ROW_COMPLETE','protocol':protocol,
        'query_id':row['query_id'],'row_index':index,'split':split,'candidate_ids':row['candidate_ids'],
        'raw_scores':row['raw_scores'],'M':row['M'],'winner_index':row['winner_index'],
        'arms':arms,'cpu_gpu_anchors':anchors,'cache_binding':seal['payload'],
        'query_patch_count':len(hidden),'energy_weighted_common_share':ratio(common_sum,total_sum),
        'candidate_seconds_sum':sum(x['candidate_seconds'] for x in completed),
        'candidate_results':[bind(folder/'candidates'/f'{i:04d}.json') for i in range(128)],
        'new_training':False,'llm_or_roma_forwards':0,
        'interpretation':'CPU frozen intervention, not GPU-bitexact. Common hidden direction can induce patch-dependent rotations after normalization; this does not establish spatial attention.'})
    print(json.dumps({'row_complete':index,'query_id':row['query_id'],'CPU_GPU_anchors':anchors,
        'candidate_seconds_sum':sum(x['candidate_seconds'] for x in completed)}),flush=True)
    summary()
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',default='all',help='all, or zero-based indices/ranges in TRAIN16 then PROBE8 order')
    ap.add_argument('--budget',type=float,default=480,help='wall-clock seconds, checkpoint at candidate boundary')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args()
    if not 1 <= args.threads <= 2 or args.budget <= 0:
        ap.error('threads must be1..2 and budget positive')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.summarize_only:
        summary();return
    started=time.monotonic()
    a,m,protocol=prep_protocol()
    allrows=m['train_rows']+m['probe_rows']
    selected=parse_rows(args.rows,len(allrows))
    cp=torch.load(SNAPSHOT,map_location='cpu',weights_only=True)
    assert cp['step']==128 and cp['arm']=='POST_REAL'
    small=ScaledQualityResidualAdapter(hidden_size=3584,bottleneck=a['bottleneck'],condition_gain=a['condition_gain'])
    small.load_state_dict(cp['adapter']);small.eval().requires_grad_(False)
    model=load_projection(a,'cpu')
    write(OUT/'projection_loading.json',model.report)
    for index in selected:
        done=process_row(index,allrows[index],'train' if index<16 else 'probe',model,small,
            cp['head'].double().tolist(),protocol,started,args.budget)
        if not done:
            print(json.dumps({'status':'PARTIAL_BUDGET','row':index,'elapsed_seconds':time.monotonic()-started}),flush=True)
            return
    summary()
    print(json.dumps({'status':'SELECTED_ROWS_COMPLETE','rows':selected,'elapsed_seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':
    main()
