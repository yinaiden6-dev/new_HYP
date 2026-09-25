#!/usr/bin/env python3
"""H593 five-fold frozen POST_REAL common/patch-deviation/assignment replay.

Derived from the qualified 24-image v2 kernel, retaining its actual BF16
hidden difference, FP32 orthogonal decomposition, frozen BF16 projection,
FP64 normalization/MaxSim, and per-candidate resumable patch evidence.
Workers never open held labels. A separate join independently checks patches.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from rc_postllm_m_backend_v1 import load_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
from run_rc_postllm_h593_v1 import Inputs
from analyze_rc_postllm_signal_decomposition_v2 import (
    bind, checked, read, write, tensor_sha, ratio, decision, reference_stats,
    projected, TOKEN_ARMS, SCORE_ARMS, RECOMPOSITION_LIMITS,
)
ROOT = Path(__file__).resolve().parents[1]
POST = ROOT/'results/rc_postllm_h593_v1'
AUTH = ROOT/'registry/rc_postllm_h593_authority_v1_20260924.json'
OUT = ROOT/'results/rc_postllm_h593_decomposition_v1'
INPUTS = None

@torch.no_grad()
def process_row(index, row, split, model, small, head, protocol, started, budget):
    folder = OUT/'queries'/row['query_id']
    final = folder/'result.json'
    if final.exists():
        if read(final)['protocol'] != protocol:
            raise ValueError('row protocol mismatch')
        return True
    cache = INPUTS.cache_cpu(row)
    seal = {'payload': INPUTS.query_sources[row['query_id']]['payload']}
    hidden = cache['hidden'][cache['image_mask']]
    assert hidden.dtype == torch.bfloat16 and hidden.ndim == 2
    small.conditioning = 'constant'
    constant_hidden = small(hidden, row['M'][0])
    constant_tokens = projected(model, cache, constant_hidden)
    constant_unit = F.normalize(constant_tokens.double(), dim=-1)
    small.conditioning = 'real'
    gpu_dir = POST/f"fold{row['oof_fold']}"/'POST_REAL/held'
    gpu = {'NATIVE': read(gpu_dir/'native'/(row['query_id']+'.json')),
           'CONSTANT': read(gpu_dir/'constant'/(row['query_id']+'.json'))}
    refs = INPUTS.refs(row)
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
        ref = refs[i]
        actual = small(hidden, row['M'][i])
        delta = actual.float()-constant_hidden.float()
        common = delta.mean(0, keepdim=True)
        spatial = delta-common
        worlds = {'NATIVE': actual,
            'COMMON_ONLY': (constant_hidden.float()+common).to(torch.bfloat16),
            'SPATIAL_ONLY': (constant_hidden.float()+spatial).to(torch.bfloat16),
            'RECOMPOSED': (constant_hidden.float()+common+spatial).to(torch.bfloat16)}
        reconstruction_error = float((worlds['RECOMPOSED'].float()-actual.float()).abs().max())
        hidden_exact=torch.equal(worlds['RECOMPOSED'],actual)
        if reconstruction_error > RECOMPOSITION_LIMITS['hidden_max_abs']:
            raise ValueError(f'BF16 hidden recomposition exceeds frozen numerical limit: {reconstruction_error}')
        total = float(delta.double().square().sum(1).mean())
        common_energy = float(common.double().square().sum())
        spatial_energy = float(spatial.double().square().sum(1).mean())
        closure = abs(total-common_energy-spatial_energy)
        if closure > max(1e-9, total*1e-5):
            raise ValueError('float residual orthogonality closure failed')
        tokens = {'CONSTANT': constant_tokens}
        for arm in ['NATIVE','COMMON_ONLY','SPATIAL_ONLY','RECOMPOSED']:
            tokens[arm] = projected(model, cache, worlds[arm])
        recomposed_token_error=float((tokens['RECOMPOSED'].float()-tokens['NATIVE'].float()).abs().max())
        if recomposed_token_error > RECOMPOSITION_LIMITS['token_max_abs']:
            raise ValueError(f'independent recomposed token mismatch: {recomposed_token_error}')
        units = {arm: F.normalize(t.double(), dim=-1) for arm,t in tokens.items()}
        best, locations, stats = {}, {}, {}
        for arm in TOKEN_ARMS:
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
        recomposed_L_error=abs(scores['RECOMPOSED']-scores['NATIVE'])
        if recomposed_L_error > RECOMPOSITION_LIMITS['L_max_abs']:
            raise ValueError(f'independent recomposed score mismatch: {recomposed_L_error}')
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
                'BF16_recomposed_exact_equal':bool(hidden_exact),
                'recomposed_projection_independently_evaluated':True,
                'recomposed_token_max_abs_error':recomposed_token_error,
                'recomposed_L_abs_error':recomposed_L_error,
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
    recomposed_logit_error=float(np.max(np.abs(np.asarray(arms['NATIVE']['decision']['logits'])-
        np.asarray(arms['RECOMPOSED']['decision']['logits']))))
    recomposed_same_decision=(arms['NATIVE']['decision']['prediction_identity']==arms['RECOMPOSED']['decision']['prediction_identity']
        and arms['NATIVE']['decision']['switched']==arms['RECOMPOSED']['decision']['switched'])
    if recomposed_logit_error > RECOMPOSITION_LIMITS['logit_max_abs'] or not recomposed_same_decision:
        raise ValueError(f'independent recomposed action mismatch: {recomposed_logit_error}; same={recomposed_same_decision}')
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
        'query_id':row['query_id'],'row_index':index,'split':split,'fold':row['oof_fold'],'candidate_ids':row['candidate_ids'],
        'raw_scores':row['raw_scores'],'M':row['M'],'winner_index':row['winner_index'],
        'arms':arms,'cpu_gpu_anchors':anchors,'cache_binding':seal['payload'],
        'recomposition_validation':{'limits':RECOMPOSITION_LIMITS,
            'max_hidden_abs_error':max(x['residual_decomposition']['BF16_recomposed_max_abs_error'] for x in completed),
            'hidden_nonexact_pairs':sum(not x['residual_decomposition']['BF16_recomposed_exact_equal'] for x in completed),
            'max_token_abs_error':max(x['residual_decomposition']['recomposed_token_max_abs_error'] for x in completed),
            'max_L_abs_error':max(x['residual_decomposition']['recomposed_L_abs_error'] for x in completed),
            'max_logit_abs_error':recomposed_logit_error,'same_decision':recomposed_same_decision,
            'all128_recomposed_projector_and_MaxSim_calls_independent':True},
        'query_patch_count':len(hidden),'energy_weighted_common_share':ratio(common_sum,total_sum),
        'candidate_seconds_sum':sum(x['candidate_seconds'] for x in completed),
        'candidate_results':[bind(folder/'candidates'/f'{i:04d}.json') for i in range(128)],
        'new_training':False,'llm_or_roma_forwards':0,
        'interpretation':'CPU frozen intervention, not GPU-bitexact. Common hidden direction can induce patch-dependent rotations after normalization; this does not establish spatial attention.'})
    print(json.dumps({'row_complete':index,'query_id':row['query_id'],'CPU_GPU_anchors':anchors,
        'candidate_seconds_sum':sum(x['candidate_seconds'] for x in completed)}),flush=True)
    return True


def prepare():
    """Freeze outcome-independent ordering and the five original fold endpoints."""
    a = read(AUTH)
    manifest = read(checked(a['manifest']))
    folds = {}
    endpoints = {}
    for f in range(5):
        fd = manifest['folds'][str(f)]
        assert not set(fd['held_query_ids']) & set(fd['train_all_query_ids'])
        for q in fd['held_query_ids']:
            assert q not in folds
            folds[q] = f
        path = POST/f'fold{f}/POST_REAL/fit_validation.json'
        seal = read(path)
        assert seal['status'] == 'H593_FIXED_EIGHT_PASSES_COMPLETE'
        assert seal['authority'] == bind(AUTH) and seal['passes'] == 8
        checked(seal['snapshot'])
        endpoints[str(f)] = dict(fit_validation=bind(path), snapshot=seal['snapshot'])
    ids = [r['query_id'] for r in manifest['rows']]
    assert len(ids) == len(set(ids)) == len(folds) == 593 and set(ids) == set(folds)
    sources = [bind(Path(__file__)), bind(ROOT/'programs/audit_rc_postllm_h593_decomposition_v1.py'),
        bind(ROOT/'slurm/rc_postllm_h593_decomposition_v1.sbatch'),
        bind(ROOT/'programs/dispatch_rc_postllm_h593_decomposition_v1.py'),
        bind(ROOT/'plan/RC_POSTLLM_H593_DECOMPOSITION_V1_20260925.md')]
    for name in ['analyze_rc_postllm_signal_decomposition_v2.py',
                 'rc_postllm_m_backend_v1.py', 'rc_prellm_m_adapter_v1.py',
                 'rc_prellm_m_scaled_adapter_v4.py', 'run_rc_postllm_h593_v1.py',
                 'run_rc_internal_m_condition_scale_v4.py', 'run_rc_prellm_m_pilot_v1.py']:
        sources.append(bind(ROOT/'programs'/name))
    protocol = dict(schema=1, status='FROZEN_H593_POSTLLM_DECOMPOSITION',
        authority=bind(AUTH), manifest=a['manifest'], endpoints=endpoints,
        code_sources=sources, rows=[dict(query_id=q, fold=folds[q], index=i) for i,q in enumerate(ids)],
        token_arms=TOKEN_ARMS, score_arms=SCORE_ARMS, shards=50,
        definition='Same qualified v2 actual BF16 delta: common=mean_patch(hM-h0); deviation=delta-common. All arms independently project; native is never aliased to recomposed.',
        fixed_argmax='Same final POST_REAL at constant M supplies the original per-patch reference index.',
        recomposition_limits=RECOMPOSITION_LIMITS,
        anchor_policy='Report all CPU versus sealed GPU errors and decision differences. Any difference blocks claiming that the decomposition exactly accounts for GPU478; do not retune on outcomes.',
        label_policy='Workers use label-free manifest and predictions; join labels only after independent patch/score/axis checks.',
        source_result=bind(POST/'result.json'), scope='Opened H593 original grouped OOF, original ColNomic natural C128, fixed INTERNAL3 head, no direct M feature',
        training=False, new_encoder_or_RoMa_forwards=0, cpu_only=True)
    path = OUT/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Frozen protocol changed; use a new version')
    write(path, protocol)
    return protocol, bind(path)


def worker(shard, pilot, budget, threads):
    global INPUTS
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    protocol = read(OUT/'protocol.json')
    pb = bind(OUT/'protocol.json')
    for b in protocol['code_sources']:
        checked(b)
    a = read(checked(protocol['authority']))
    manifest = read(checked(protocol['manifest']))
    selected = [0] if pilot else list(range(shard, 593, protocol['shards']))
    assert pilot or 0 <= shard < protocol['shards']
    INPUTS = Inputs(protocol['authority'], 'cpu', cpu_gb=4, gpu_gb=2)
    model = load_projection(a, 'cpu')
    write(OUT/'loading'/f'{"pilot" if pilot else shard}.json', model.report)
    active_fold, small, head = None, None, None
    for index in selected:
        record = protocol['rows'][index]
        row = dict(manifest['rows'][index], oof_fold=record['fold'])
        assert row['query_id'] == record['query_id']
        folder = OUT/'queries'/row['query_id']
        folder.mkdir(parents=True, exist_ok=True)
        lock = (folder/'worker.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (folder/'result.json').exists():
                assert read(folder/'result.json')['protocol'] == pb
                continue
            if time.monotonic()-started >= budget-30:
                return False
            f = row['oof_fold']
            if f != active_fold:
                cp = torch.load(checked(protocol['endpoints'][str(f)]['snapshot']), map_location='cpu', weights_only=True)
                assert cp['arm'] == 'POST_REAL'
                assert cp['step'] == 8 * len(manifest['folds'][str(f)]['train_query_ids'])
                assert cp['authority'] == protocol['authority'] and cp['input_scope']['manifest'] == protocol['manifest']
                small = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=a['bottleneck'],
                    condition_gain=a['condition_gain'], residual_scale=a['residual_scale'])
                small.load_state_dict(cp['adapter'])
                small.eval().requires_grad_(False)
                head = cp['head'].double().tolist()
                active_fold = f
            if not process_row(index, row, 'held', model, small, head, pb, started, budget-20):
                return False
        finally:
            lock.close()
    write(OUT/'shards'/f'{"pilot" if pilot else shard}.json', dict(status='SELECTED_ROWS_COMPLETE',
        protocol=pb, indices=selected, elapsed_seconds=time.monotonic()-started,
        query_results=[bind(OUT/'queries'/protocol['rows'][i]['query_id']/'result.json') for i in selected],
        input_cache_stats=INPUTS.stats(), held_label_reads=0))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'worker'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--budget', type=float, default=460)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.command == 'prepare':
        p, b = prepare()
        print(json.dumps(dict(status=p['status'], queries=len(p['rows']), protocol=b)))
    else:
        done = worker(args.shard, args.pilot, args.budget, args.threads)
        print(json.dumps(dict(status='DONE' if done else 'PARTIAL_BUDGET', shard=args.shard, pilot=args.pilot)), flush=True)
        if not done:
            sys.exit(75)


if __name__ == '__main__':
    main()
