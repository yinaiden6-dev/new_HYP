#!/usr/bin/env python3
"""Read-only CPU arithmetic on sealed POST128 caches/weights; no new fit.

The numerator/norm split is a specified sequential algebraic decomposition,
not an order-invariant causal allocation. Actual saved BF16-path scores anchor
the calculation. The pilot and full-H593 programs/artifacts remain untouched.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from rc_postllm_m_backend_v1 import load_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_existing_evidence_attribution_v1/internal'
SOURCE = ROOT / 'results/rc_postllm_m_signal_decomposition_v2'
QID = 'H593-90acde9b0567a472f4232120'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    authp = ROOT/'registry/rc_postllm_m_v1_authority_20260924.json'
    a = read(authp); mp = checked(a['manifest']); manifest = read(mp)
    n = manifest['mass_normalization']
    row = next(r for r in manifest['probe_rows'] if r['query_id'] == QID)
    sealp = ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/QID/'validation.json'
    seal = read(sealp); payloadp = checked(seal['payload'])
    cache = torch.load(payloadp, map_location='cpu', weights_only=True)
    snapshot = ROOT/'results/rc_postllm_m_v1/POST_REAL/snapshots/0128.pt'
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    adapter = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=16, conditioning='real',
        condition_gain=a['condition_gain'], mass_log_mean=n['log_mean'], mass_log_std=n['log_std'],
        mass_epsilon=n['epsilon'], residual_scale=a['residual_scale'])
    adapter.load_state_dict(state['adapter']); adapter.eval()
    model = load_projection(a, device='cpu')
    h = cache['hidden'][cache['image_mask']]
    t = F.linear(F.layer_norm(h.float(), (3584,)), adapter.down.weight[:, :-1], adapter.down.bias)
    wm = adapter.down.weight[:, -1]
    variation = (t-t.mean(0)).square().sum(1).mean().sqrt()
    adapter.conditioning = 'constant'; h0 = adapter(h, 0.)
    f0 = cache['hidden'].clone(); f0[cache['image_mask']] = h0
    p0 = model(f0)[cache['image_mask']].double()
    results = {}; hidden_dirs = []; projected_dirs = []; sources = []
    for name, pos in [('target', 87), ('RAW_wrong_winner', 104)]:
        base = SOURCE/'queries'/QID
        recordp = base/'candidates'/f'{pos:04d}.json'; rec = read(recordp)
        patchp = checked(rec['patch_evidence']); arrays = np.load(patchp)
        sources += [bind(recordp), bind(patchp), row['reference_tokens'][pos]]
        d = torch.tensor(arrays['common_hidden_vector']); hidden_dirs.append(d.double())
        # Ideal linear part of the frozen BF16-parameter projector, evaluated
        # in FP64; actual p0/p1 below retain the native BF16 GEMMs and LoRA.
        projected = F.linear(d.double(), model.weight.double()) + F.linear(
            F.linear(d.double(), model.lora_a.double()), model.lora_b.double())*model.scaling
        projected_dirs.append(projected)
        f1 = cache['hidden'].clone(); f1[cache['image_mask']] = (h0.float()+d).to(torch.bfloat16)
        p1 = model(f1)[cache['image_mask']].double()
        ref = torch.load(checked(row['reference_tokens'][pos]), map_location='cpu', weights_only=True)['tokens'].double()
        ref = F.normalize(ref, dim=-1); old_idx = torch.tensor(arrays['best_reference_token'][0])
        r = ref[old_idx]; norm0 = p0.norm(dim=1); norm1 = p1.norm(dim=1)
        numerator = ((p1-p0)*r).sum(1)/norm0
        normalization = (p1*r).sum(1)*(1/norm1-1/norm0)
        exact = (F.normalize(p1, dim=-1)*r).sum(1)-(F.normalize(p0, dim=-1)*r).sum(1)
        assert float((numerator+normalization-exact).abs().max()) < 1e-12
        actual = arrays['fixed_common_by_patch']-arrays['maxsim_by_patch'][0]
        c = a['condition_gain']*(math.log(max(row['M'][pos], n['epsilon']))-n['log_mean'])/n['log_std']
        b = t+wm*c; response = F.gelu(b)-F.gelu(t)
        native_change = arrays['maxsim_by_patch'][1]-arrays['maxsim_by_patch'][0]
        fixed_native_change = arrays['fixed_native_by_patch']-arrays['maxsim_by_patch'][0]
        record = dict(position=pos, M=row['M'][pos], scores=rec['scores'],
            native_L_gain=float(native_change.mean()),
            native_L_gain_percent=100*float(native_change.mean())/rec['scores']['CONSTANT'],
            native_patch_score_increase_fraction=float((native_change>0).mean()),
            fixed_native_patch_score_increase_fraction=float((fixed_native_change>0).mean()),
            native_token_rotation_mean_degrees=float(np.degrees(np.arccos(np.clip(
                1-arrays['native_unit_token_change_norm']**2/2, -1, 1))).mean()),
            hidden_common_norm=float(d.norm()), projected_common_norm=float(projected.norm()),
            old_projection_norm_mean=float(norm0.mean()), new_projection_norm_mean=float(norm1.mean()),
            numerator_at_old_norm_mean_gain=float(numerator.mean()),
            normalization_after_numerator_mean_gain=float(normalization.mean()),
            sum_FP64_normalized_fixed_score_gain=float(exact.mean()),
            actual_BF16_then_FP64_fixed_score_gain=float(actual.mean()),
            BF16_normalization_rounding_residual=float(actual.mean()-exact.mean()),
            numerator_gain_per_projected_shift_norm=float(numerator.mean()/projected.norm()),
            old_matched_reference_alignment_to_unit_shift_mean=float((r@projected).mean()/projected.norm()),
            standardized_log_M_before_gain=c/a['condition_gain'], scaled_log_M=c,
            bottleneck_condition_shift_norm=float((wm*c).norm()),
            condition_shift_over_content_patch_variation=float((wm*c).norm()/variation),
            ideal_bottleneck_response_common_energy_fraction=float(response.mean(0).square().sum()/response.square().sum(1).mean()),
            bottleneck_same_saturated_region_fraction=float((((t>3)&(b>3))|((t < -3)&(b < -3))).float().mean()),
            bottleneck_sign_change_fraction=float(((t>0)!=(b>0)).float().mean()))
        npz = OUT/f'{name}_projection_arithmetic.npz'
        np.savez_compressed(npz, constant_projected=p0.numpy(), common_projected=p1.numpy(),
            old_matched_reference_unit=r.numpy(), old_reference_indices=old_idx.numpy(),
            numerator_at_old_norm=numerator.numpy(), norm_adjustment=normalization.numpy(),
            actual_BF16path_fixed_change=actual, hidden_common=d.numpy(), projected_common=projected.numpy(),
            bottleneck_content=t.numpy(), bottleneck_condition=(wm*c).numpy())
        record['arithmetic_arrays'] = bind(npz); results[name] = record
    # Outcome-blind descriptive direction chosen from first frozen TRAIN query;
    # this is not a newly trained decision model and has no predicted outcomes.
    D = []; z = []; npz_sources = []
    for item in manifest['train_rows']+manifest['probe_rows']:
        folder = SOURCE/'queries'/item['query_id']
        for pos in range(128):
            evidence = read(folder/'candidates'/f'{pos:04d}.json')['patch_evidence']
            with np.load(checked(evidence)) as array:
                D.append(array['common_hidden_vector'].copy())
            npz_sources.append(evidence)
            z.append((math.log(max(item['M'][pos], n['epsilon']))-n['log_mean'])/n['log_std'])
    D = np.asarray(D, np.float64); z = np.asarray(z)
    selected = int(np.linalg.norm(D[:128], axis=1).argmax())
    direction = D[selected]/np.linalg.norm(D[selected]); coordinate = D@direction
    norms = np.linalg.norm(D, axis=1); cos = abs(coordinate)/np.maximum(norms, 1e-30)
    descriptive = dict(pairs=len(D), first_train_query=manifest['train_rows'][0]['query_id'],
        reference_candidate_position=selected,
        direction_rule='Largest common-vector norm among first frozen TRAIN query 128 candidates; no outcomes used',
        energy_on_one_direction=float((coordinate**2).sum()/(D**2).sum()),
        abs_direction_cosine_quantiles=np.quantile(cos, [0,.05,.5,.95,1]).tolist(),
        signed_coordinate_vs_standardized_log_M_correlation=float(np.corrcoef(coordinate,z)[0,1]),
        scope='Descriptive frozen adapter geometry only; not an all24 decision ablation or new fit',
        source_bindings=npz_sources)
    output = dict(status='EXISTING_POST128_INTERNAL_ARITHMETIC_RECOMPUTED', query_id=QID,
        sources=[bind(__file__),bind(authp),bind(mp),bind(snapshot),bind(sealp),seal['payload']]+sources,
        projector=model.report, candidate_results=results,
        hidden_common_direction_cosine=float(F.cosine_similarity(hidden_dirs[0], hidden_dirs[1], dim=0)),
        projected_common_direction_cosine=float(F.cosine_similarity(projected_dirs[0], projected_dirs[1], dim=0)),
        projected_shift_norm_ratio_target_wrong=float(projected_dirs[0].norm()/projected_dirs[1].norm()),
        numerator_sensitivity_ratio_target_wrong=results['target']['numerator_gain_per_projected_shift_norm']/results['RAW_wrong_winner']['numerator_gain_per_projected_shift_norm'],
        shared_direction_geometry_all24=descriptive,
        adapter_design=dict(condition_gain=a['condition_gain'], bottleneck=16,
            constant_bottleneck_mean_norm=float(t.norm(dim=1).mean()),
            constant_bottleneck_patch_variation_rms_norm=float(variation),
            learned_M_column_norm=float(wm.norm()),
            caution='Gain scales the learned condition input. These weights were trained with that gain; no gain1 retraining/counterfactual here. Cannot attribute dominance causally to gain alone.'),
        numerical_contract='Actual BF16 frozen projector outputs; FP64 ideal L2 algebra compared with saved BF16 then FP64 normalization. Residual explicitly retained.',
        decomposition_order='First change projected numerator at old norm; then adjust new norm. This allocation depends on order.',
        limitations=['Two-candidate arithmetic explains cefdinir at this endpoint only.',
            'Near-collinear directions and broad score increases do not identify semantic regions or visual cues.',
            'All24 direction statistics describe an existing learned model, not all24 causal efficacy of replacing it by one linear direction.',
            'No raw encoder information-theoretic absence or necessity inference.'],
        new_training=False, new_GPU_jobs=0, new_LLM_or_RoMa_forwards=0)
    (OUT/'result.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ['status','hidden_common_direction_cosine','projected_common_direction_cosine',
        'projected_shift_norm_ratio_target_wrong','numerator_sensitivity_ratio_target_wrong']},indent=2))


if __name__ == '__main__':
    main()
