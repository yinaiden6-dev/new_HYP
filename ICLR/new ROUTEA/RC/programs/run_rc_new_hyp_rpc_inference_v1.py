#!/usr/bin/env python3
"""Outcome-blind RPC RAW, unchanged RoMa, and fixed-head prediction stages."""
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
import run_romav2_colnomic_sealed_source_e0_v2 as S
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC

OUT = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'
AUTH = ROOT / 'registry/rc_new_hyp_rpc_inference_authority_v1_20260913.json'
read, write, bind, checked, need = M.read, M.write, M.bind, M.checked, M.need


def guard(stage):
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    a = read(AUTH)
    allowed = set()
    for b in a['sources'].values(): allowed.add(Path(checked(b)).resolve())
    for b in a['operator_sources'].values():
        p = Path(b['path'])
        # CPU replay and RoMa do not load encoder weights or the full RAW gallery.
        # Their producer already pins those sources; consumed stage payloads are rehashed below.
        if stage == 'raw' or p.stat().st_size < 16 * (1 << 20): checked(b)
        allowed.add(p.resolve())
    for sources in a['heads'].values():
        for b in sources.values(): allowed.add(Path(checked(b)).resolve())
    profile = read(checked(a['sources']['roma_profile']))
    for b in profile['sources'].values():
        p = Path(b['path'])
        if stage == 'roma' or p.stat().st_size < 16 * (1 << 20): checked(b)
        allowed.add(p.resolve())
    for tree in profile['source_trees'].values():
        for b in tree['python_files']: allowed.add(Path(checked(b)).resolve())
    def audit(event, args):
        if event == 'socket.connect': need(not isinstance(args[1], tuple), 'OFFLINE_RUNTIME')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        name = str(p).lower()
        need(not any(s in name for s in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', '/reports/', 'rc_opened_', 'rc_new_hyp_grozi120_external')), 'PROTECTED_READ')
        if ROOT / 'results' in p.parents:
            own = OUT in p.parents
            need(own or p in allowed, 'UNLISTED_RESULT:' + name)
    sys.addaudithook(audit)
    need(read(checked(a['sources']['reference_validation']))['status'] == 'RPC200_REFERENCE_CPU_INTEGRITY_PASS', 'REFERENCE_QUALIFIED')
    return a


def workers(shard):
    need(shard in range(75), 'SHARD_RANGE')
    records = read(OUT / 'worker_manifest.json')['records'][8 * shard:8 * (shard + 1)]
    need(len(records) == 8, 'FROZEN_EIGHT_QUERY_SHARD')
    return records


def save(kind, shard, payload):
    folder = OUT / kind / f'shard{shard:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'payload.pt'
    R.plain_only(payload)
    with path.open('xb') as f: torch.save(payload, f)
    write(folder / 'receipt.json', dict(authority=bind(AUTH), payload=bind(path), query_count=8))


def load(kind, shard, validated=True):
    folder = OUT / kind / f'shard{shard:02d}'
    receipt = read(folder / 'receipt.json')
    need(receipt['authority'] == bind(AUTH), 'STAGE_AUTHORITY')
    if validated:
        v = read(folder / 'validation.json')
        need(v['payload'] == receipt['payload'] and v['status'] == 'RPC_' + kind.upper() + '_CPU_PASS', 'STAGE_VALIDATED')
    return torch.load(checked(receipt['payload']), map_location='cpu', mmap=True, weights_only=True), receipt['payload']


def rank(scores, labels):
    seen, ordered = set(), []
    for p in torch.argsort(scores, descending=True, stable=True).tolist():
        if labels[p] not in seen:
            seen.add(labels[p]); ordered.append(p)
    need(len(ordered) == 200, 'COMPLETE_IDENTITY_RANK')
    return ordered


def raw(shard, a):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    pre = read(checked(a['sources']['old_preflight']))
    need(R.runtime_libraries() == pre['runtime_libraries'], 'FROZEN_RUNTIME')
    device = torch.device('cuda')
    enrolled = torch.load(checked(a['sources']['reference_payload']), map_location='cpu', mmap=True, weights_only=True)['records']
    passages = [r['passage_emb'] for r in enrolled]
    newrefs = {r['physical_row']: r for r in enrolled}
    encoder = ColQwen2_5.from_pretrained(str(R.MODEL), torch_dtype=torch.bfloat16).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(R.MODEL))
    labels = [r['identity'] for r in read(OUT / 'gallery_manifest.json')['records']]
    need(len(passages) == len(labels) == len(set(labels)) == 200, 'INDEPENDENT_RPC200_GALLERY')
    images = {r['asset_id']: r for r in read(OUT / 'download_receipt.json')['files']}
    records, needed = [], set()
    with torch.inference_mode():
        for worker in workers(shard):
            source = dict(query_id=worker['query_id'], execution_ordinal=worker['execution_ordinal'],
                          query_source_path=worker['image_path'], query_source_sha256=images[worker['query_id']]['image_sha256'],
                          track='new_difficult_train')  # Frozen enum selects EXIF-oriented processor frame only.
            image, template, grid, metadata = R.encode_query(source, processor, encoder, device)
            scores = S.full_gallery_scores(torch.cat((image, template)), passages, device, batch_size=16)
            need(scores.dtype == torch.float64 and scores.shape == (200,) and torch.isfinite(scores).all(), 'RAW_FULL200')
            ordered = rank(scores, labels)
            top = S.top128_rows(scores, labels)
            need(ordered[:128] == top, 'NATURAL_C128')
            axis = sorted(top); needed.update(axis)
            records.append(dict(**source, **metadata, dataset='RPC600-external', processor_input_frame='EXIF_ORIENTED_BEFORE_RESIZE',
                  query_grid_shape=grid, query_tokens=image, query_tokens_sha256=R.tensor_sha(image),
                  template_tokens=template, template_tokens_sha256=R.tensor_sha(template),
                  raw_physical_scores=scores, raw_ranked_physical_rows=ordered,
                  candidate_ranked_physical_rows=top, candidate_physical_rows=axis,
                  candidate_raw_scores=scores[axis].clone().contiguous()))
            print(json.dumps(dict(event='RPC_RAW_READY', query_id=worker['query_id'], candidates=128)), flush=True)
        references = {}
        for physical in sorted(needed):
            r = newrefs[physical]
            references[physical] = {k: r[k] for k in ('physical_row', 'source_path', 'source_image_sha256', 'grid_shape', 'tokens', 'tokens_sha256')}
    save('raw', shard, dict(records=records, references=references, authority=bind(AUTH), gallery_reference_payload=a['sources']['reference_payload'],
         GPU_runtime=R.gpu_runtime(), runtime_libraries=R.runtime_libraries(), external_query_count=8, target_reads=0, training_updates=0))
    subprocess.run([sys.executable, __file__, 'verify-raw', '--shard', str(shard)], check=True)


def verify_raw(shard, a):
    payload, binding = load('raw', shard, False)
    labels = [r['identity'] for r in read(OUT / 'gallery_manifest.json')['records']]
    need(len(payload['records']) == 8, 'COMPLETE_RAW_SHARD')
    for r, w in zip(payload['records'], workers(shard), strict=True):
        need(r['query_id'] == w['query_id'] and r['query_source_path'] == w['image_path'], 'QUERY_AXIS')
        need(R.sha(Path(w['image_path'])) == r['query_source_sha256'], 'QUERY_BYTES')
        need(R.tensor_sha(r['query_tokens']) == r['query_tokens_sha256'], 'QUERY_TOKENS')
        need(R.tensor_sha(r['template_tokens']) == r['template_tokens_sha256'], 'TEMPLATE_TOKENS')
        scores = r['raw_physical_scores']; seen, ordered = set(), []
        for p in sorted(range(200), key=lambda p: (-float(scores[p]), p)):
            if labels[p] not in seen: seen.add(labels[p]); ordered.append(p)
        need(ordered == r['raw_ranked_physical_rows'] and ordered[:128] == r['candidate_ranked_physical_rows'], 'INDEPENDENT_FULLRANK')
        need(sorted(ordered[:128]) == r['candidate_physical_rows'], 'CANDIDATE_PHYSICAL_ORDER')
    for r in payload['references'].values():
        need(R.tensor_sha(r['tokens']) == r['tokens_sha256'], 'REFERENCE_TOKENS')
        need(R.sha(Path(r['source_path'])) == r['source_image_sha256'], 'REFERENCE_BYTES')
        need(r['tokens'].shape == (r['grid_shape'][0] * r['grid_shape'][1], 128), 'REFERENCE_GRID')
    write(OUT / 'raw' / f'shard{shard:02d}' / 'validation.json', dict(status='RPC_RAW_CPU_PASS', authority=bind(AUTH), payload=binding, query_count=8, target_reads=0))


def roma(shard, a):
    token, raw_binding = load('raw', shard)
    module = M.loadmodule(M.ROMA_SOURCE, 'rpc_frozen_roma')
    compute, checksum = M.compile_roma(module)
    need(checksum == a['roma_numeric_loop_sha256'], 'UNCHANGED_ROMA_LOOP')
    with torch.inference_mode(): records = compute(token, read(a['sources']['roma_profile']['path']))
    save('roma', shard, dict(records=records, raw=raw_binding, authority=bind(AUTH), training_updates=0, target_reads=0))
    subprocess.run([sys.executable, __file__, 'verify-roma', '--shard', str(shard)], check=True)
    predict(shard, a)


def verify_roma(shard, a):
    module = M.loadmodule(M.ROMA_SOURCE, 'rpc_roma_validator')
    raw, raw_binding = load('raw', shard)
    payload, binding = load('roma', shard, False)
    need(payload['raw'] == raw_binding and len(payload['records']) == len(raw['records']) == 8, 'RAW_ROMA_BINDING')
    checks = 0
    for q, r in zip(raw['records'], payload['records']):
        need(q['query_id'] == r['query_id'] and q['candidate_physical_rows'] == r['candidate_physical_rows'], 'QUERY_CANDIDATE_AXIS')
        need(len(r['candidates']) == 128, 'ALL128')
        for j, c in enumerate(r['candidates']):
            ref = raw['references'][q['candidate_physical_rows'][j]]
            need(c['candidate_position'] == j and c['physical_row'] == ref['physical_row'] and c['reference_tokens_sha256'] == ref['tokens_sha256'], 'CANDIDATE_BINDING')
            scores = module.replay_c4(q['query_tokens'], ref['tokens'], c['query_visibility'], c['reference_visibility'])
            need(all(float(value).hex() == float(c['old_scores'][k]).hex() for k, value in scores.items()), 'INDEPENDENT_FP64_C4_BITS')
            checks += 4
    write(OUT / 'roma' / f'shard{shard:02d}' / 'validation.json', dict(status='RPC_ROMA_CPU_PASS', authority=bind(AUTH), payload=binding, C4_checks=checks, target_reads=0))


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
    write(folder / 'validation.json', dict(status='RPC_FROZEN_HEAD_NUMPY_ACTION_PASS', authority=bind(AUTH), payload=bind(folder / 'payload.json'), max_abs_error=maximum_error, target_reads=0))
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
    {'raw': raw, 'verify-raw': verify_raw, 'roma': roma, 'verify-roma': verify_roma, 'predict': predict}[args.stage](args.shard, authority)
