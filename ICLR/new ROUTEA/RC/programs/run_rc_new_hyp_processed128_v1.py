#!/usr/bin/env python3
"""Frozen processed128 regression: unchanged RAW/RoMa loops and fixed heads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import materialize_rc_original7_train128_token_raw_v1 as R
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC

OUT = ROOT / 'results/rc_new_hyp_processed128_regression_v1'
AUTH = ROOT / 'registry/rc_new_hyp_processed128_authority_v1_20260916.json'
read, write, bind, checked, need = M.read, M.write, M.bind, M.checked, M.need


def guard(stage):
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    a = read(AUTH)
    need(a['status'] == 'PROCESSED128_FROZEN_REGRESSION_AUTHORIZED', 'AUTHORIZED')
    allowed = set()
    for b in a['sources'].values():
        allowed.add(Path(checked(b)).resolve())
    for b in a['operator_sources'].values():
        p = Path(b['path'])
        if stage == 'raw' or p.stat().st_size < 16 * (1 << 20):
            checked(b)
        allowed.add(p.resolve())
    for sources in a['heads'].values():
        for b in sources.values():
            allowed.add(Path(checked(b)).resolve())
    profile = read(checked(a['sources']['roma_profile']))
    for b in profile['sources'].values():
        p = Path(b['path'])
        if stage == 'roma' or p.stat().st_size < 16 * (1 << 20):
            checked(b)
        allowed.add(p.resolve())
    for tree in profile['source_trees'].values():
        for b in tree['python_files']:
            allowed.add(Path(checked(b)).resolve())
    def audit(event, args):
        if event == 'socket.connect':
            need(not isinstance(args[1], tuple), 'OFFLINE_RUNTIME')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        name = str(p).lower()
        need(not any(s in name for s in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi',
                 '/reports/', 'rc_opened_', 'gisc_prerecall_universe')), 'PROTECTED_READ')
        if ROOT / 'results' in p.parents:
            need(OUT in p.parents or p in allowed, 'UNLISTED_RESULT:' + name)
    sys.addaudithook(audit)
    need(read(checked(a['sources']['metadata_validation']))['status'] == 'PROCESSED128_METADATA_IMAGE_INTEGRITY_PASS', 'METADATA')
    return a


def workers(shard):
    need(shard in range(16), 'SHARD_RANGE')
    records = read(OUT / 'worker_manifest.json')['records'][8 * shard:8 * (shard + 1)]
    need(len(records) == 8, 'EIGHT_QUERIES')
    return records


def save(kind, shard, payload):
    folder = OUT / kind / f'shard{shard:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'payload.pt'
    R.plain_only(payload)
    # An interrupted write cannot masquerade as a completed payload.
    tmp = folder / ('.partial-' + str(os.getpid()))
    torch.save(payload, tmp)
    os.link(tmp, path); tmp.unlink()
    write(folder / 'receipt.json', dict(authority=bind(AUTH), payload=bind(path), query_count=8))


def load(kind, shard, validated=True):
    folder = OUT / kind / f'shard{shard:02d}'
    receipt = read(folder / 'receipt.json')
    need(receipt['authority'] == bind(AUTH), 'STAGE_AUTHORITY')
    if validated:
        v = read(folder / 'validation.json')
        need(v['payload'] == receipt['payload'] and v['status'] == 'PROCESSED_' + kind.upper() + '_CPU_PASS', 'STAGE_VALIDATED')
    return torch.load(checked(receipt['payload']), map_location='cpu', mmap=True, weights_only=True), receipt['payload']


def child(stage, shard):
    subprocess.run([sys.executable, __file__, stage, '--shard', str(shard)],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)


def raw(shard, a):
    pre = read(checked(a['sources']['old_preflight']))
    need(R.runtime_libraries() == pre['runtime_libraries'], 'FROZEN_RAW_RUNTIME')
    compute, digest = M.compile_raw(R)
    need(digest == a['raw_numeric_loop_sha256'], 'UNCHANGED_RAW_NUMERIC_LOOP')
    with torch.inference_mode():
        records, references, gpu, gallery = compute(workers(shard), pre)
    manifest = read(OUT / 'gallery_manifest.json')
    expected = manifest['records']
    need(gallery['corrected_mapping_sha256'] == manifest['corrected_mapping_sha256'], 'GALLERY_BINDING')
    for physical, ref in references.items():
        need(ref['source_path'] == expected[physical]['image_path'], 'GALLERY_PHYSICAL_ORDER')
    save('raw', shard, dict(records=records, references=references, authority=bind(AUTH),
         GPU_runtime=gpu, gallery=gallery, runtime_libraries=R.runtime_libraries(),
         target_reads=0, training_updates=0))
    child('verify-raw', shard)


def verify_raw(shard, a):
    payload, binding = load('raw', shard, False)
    gallery = read(OUT / 'gallery_manifest.json')['records']
    labels = [r['identity'] for r in gallery]
    need(len(payload['records']) == 8 and len(labels) == 5413 and len(set(labels)) == 5412, 'COMPLETE_AXES')
    for r, w in zip(payload['records'], workers(shard), strict=True):
        need(r['query_id'] == w['query_id'] and r['execution_ordinal'] == w['execution_ordinal'], 'QUERY_AXIS')
        need(r['query_source_path'] == w['query_image_path'], 'QUERY_PATH')
        need(R.sha(Path(w['query_image_path'])) == w['source_image_sha256'] == r['query_source_sha256'], 'QUERY_BYTES')
        for name in ('query_tokens', 'template_tokens'):
            need(R.tensor_sha(r[name]) == r[name + '_sha256'], 'QUERY_TOKEN_HASH')
        scores = r['raw_physical_scores']; seen, ordered = set(), []
        need(scores.shape == (5413,) and scores.dtype == torch.float64 and torch.isfinite(scores).all(), 'RAW_SCORES')
        for p in sorted(range(5413), key=lambda p: (-float(scores[p]), p)):
            if labels[p] not in seen:
                seen.add(labels[p]); ordered.append(p)
        need(ordered == r['raw_ranked_physical_rows'] and ordered[:128] == r['candidate_ranked_physical_rows'], 'INDEPENDENT_FULLRANK')
        need(sorted(ordered[:128]) == r['candidate_physical_rows'], 'NATURAL_C128_AXIS')
        need(torch.equal(scores[r['candidate_physical_rows']], r['candidate_raw_scores']), 'C128_SCORES')
    for physical, r in payload['references'].items():
        need(r['source_path'] == gallery[physical]['image_path'], 'REFERENCE_PATH')
        need(R.tensor_sha(r['tokens']) == r['tokens_sha256'], 'REFERENCE_TOKENS')
        need(R.sha(Path(r['source_path'])) == r['source_image_sha256'], 'REFERENCE_BYTES')
        need(r['tokens'].shape == (r['grid_shape'][0] * r['grid_shape'][1], 128), 'REFERENCE_GRID')
    write(OUT / 'raw' / f'shard{shard:02d}' / 'validation.json',
          dict(status='PROCESSED_RAW_CPU_PASS', authority=bind(AUTH), payload=binding, query_count=8, target_reads=0))


def roma(shard, a):
    token, raw_binding = load('raw', shard)
    module = M.loadmodule(M.ROMA_SOURCE, 'processed_frozen_roma')
    compute, digest = M.compile_roma(module)
    need(digest == a['roma_numeric_loop_sha256'], 'UNCHANGED_ROMA_LOOP')
    with torch.inference_mode():
        records = compute(token, read(a['sources']['roma_profile']['path']))
    save('roma', shard, dict(records=records, raw=raw_binding, authority=bind(AUTH), training_updates=0, target_reads=0))
    child('verify-roma', shard)
    predict(shard, a)


def verify_roma(shard, a):
    module = M.loadmodule(M.ROMA_SOURCE, 'processed_roma_validator')
    raw, raw_binding = load('raw', shard)
    payload, binding = load('roma', shard, False)
    need(payload['raw'] == raw_binding and len(payload['records']) == len(raw['records']) == 8, 'RAW_ROMA_BINDING')
    checks = 0
    for q, r in zip(raw['records'], payload['records'], strict=True):
        need(q['query_id'] == r['query_id'] and q['candidate_physical_rows'] == r['candidate_physical_rows'], 'QUERY_CANDIDATE_AXIS')
        need(len(r['candidates']) == 128, 'ALL128')
        for j, c in enumerate(r['candidates']):
            ref = raw['references'][q['candidate_physical_rows'][j]]
            need(c['candidate_position'] == j and c['physical_row'] == ref['physical_row'] and c['reference_tokens_sha256'] == ref['tokens_sha256'], 'CANDIDATE_BINDING')
            for w, count in ((c['query_visibility'], len(q['query_tokens'])), (c['reference_visibility'], len(ref['tokens']))):
                need(w.dtype == torch.float64 and w.shape == (count,) and torch.isfinite(w).all() and ((w >= 0) & (w <= 1)).all(), 'VISIBILITY_DOMAIN')
            scores = module.replay_c4(q['query_tokens'], ref['tokens'], c['query_visibility'], c['reference_visibility'])
            need(all(float(value).hex() == float(c['old_scores'][k]).hex() for k, value in scores.items()), 'INDEPENDENT_FP64_C4_BITS')
            checks += 4
    write(OUT / 'roma' / f'shard{shard:02d}' / 'validation.json',
          dict(status='PROCESSED_ROMA_CPU_PASS', authority=bind(AUTH), payload=binding, C4_checks=checks, target_reads=0))


def predict(shard, a):
    raw, rb = load('raw', shard)
    roma_payload, mb = load('roma', shard)
    heads = {m: torch.tensor([float.fromhex(h) for h in read(checked(s['head']))['theta_hex']], dtype=torch.float64) for m, s in a['heads'].items()}
    predictions, maximum_error = [], 0.0
    for q, r in zip(raw['records'], roma_payload['records']):
        axis = q['candidate_physical_rows']; winner = axis.index(q['candidate_ranked_physical_rows'][0])
        challengers = [p for p in range(128) if p != winner]
        scores = [c['old_scores'] for c in r['candidates']]
        outputs = {}
        for mode in ('REAL', 'CBIND'):
            ids = list(range(128)) if mode == 'REAL' else [(i + 64) % 128 for i in range(128)]
            evidence = {i: scores[ids[i]] for i in range(128)}
            x = torch.stack([FC.candidate_feature(q['candidate_raw_scores'].tolist(), evidence, c, winner) for c in challengers])
            for model, theta in heads.items():
                features = x[:, :1] if model == 'RAW2_CE' else x
                z = features @ theta[:-1] + theta[-1]
                k = int(z.argmax()); selected = axis[challengers[k]] if z[k] > 0 else axis[winner]
                independent = np.sum(features.numpy() * theta[:-1].numpy(), axis=1) + float(theta[-1])
                error = float(np.abs(independent - z.numpy()).max()); maximum_error = max(maximum_error, error)
                need(error < 2e-10, 'NUMPY_HEAD_LOGITS')
                j = int(np.argmax(independent)); selected_check = axis[challengers[j]] if independent[j] > 0 else axis[winner]
                need(selected_check == selected, 'NUMPY_HEAD_ACTION')
                outputs[model if mode == 'REAL' else model + '_CBIND'] = dict(selected=selected, logits_hex=[float(t).hex() for t in z])
        predictions.append(dict(query_id=q['query_id'], execution_ordinal=q['execution_ordinal'], raw_selected=axis[winner], raw_ranked_physical_rows=q['raw_ranked_physical_rows'], models=outputs))
    folder = OUT / 'predictions' / f'shard{shard:02d}'
    write(folder / 'payload.json', dict(authority=bind(AUTH), raw=rb, roma=mb, records=predictions, target_reads=0, training_updates=0))
    write(folder / 'validation.json', dict(status='PROCESSED_FROZEN_HEAD_NUMPY_ACTION_PASS', authority=bind(AUTH), payload=bind(folder / 'payload.json'), max_abs_error=maximum_error, target_reads=0))
    print(json.dumps(dict(stage='PREDICTIONS_SEALED', shard=shard, queries=8, outcomes_joined=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['raw', 'verify-raw', 'roma', 'verify-roma', 'predict'])
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(17)
    authority = guard(args.stage)
    globals()[args.stage.replace('-', '_')](args.shard, authority)
