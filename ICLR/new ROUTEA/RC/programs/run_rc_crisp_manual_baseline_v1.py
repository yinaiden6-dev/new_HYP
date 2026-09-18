#!/usr/bin/env python3
"""Fixed CRISP image-to-image adaptation and preregistered untrained heads."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC

OUT = ROOT / 'results/rc_crisp_manual_baseline_v1'
AUTH = ROOT / 'registry/rc_crisp_manual_baseline_authority_v1_20260918.json'
UPSTREAM = ROOT / 'isolated/crisp_baseline_20260918_v1/upstream_crisp.py'
PROGRAM = Path(__file__).resolve()
LAUNCHER = ROOT / 'slurm/rc_crisp_manual_baseline_v1.sbatch'
H593 = ROOT / 'results/rc_new_hyp593_oof5_v1'
RPC = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'
MANUAL = dict(ZERO_HEAD=[0.] * 7, MANUAL_EQUAL=[1.] * 6 + [0.],
              MANUAL_BALANCED=[1.] + [.2] * 5 + [0.])
SCORED = ('PATCH_MAXSIM', 'CRISP', 'VISIBILITY_DIRECT')
MODELS = ('RAW', *SCORED, *MANUAL, 'COST1', 'CE')
read, write, bind, checked, need = M.read, M.write, M.bind, M.checked, M.need


def source(folder):
    receipt = read(folder / 'receipt.json')
    result = dict(payload=receipt['payload'], receipt=bind(folder / 'receipt.json'),
                  validation=bind(folder / 'validation.json'))
    validate_envelope(result)
    return result


def validate_envelope(b):
    receipt = read(checked(b['receipt']))
    validation = read(checked(b['validation']))
    need(receipt['payload'] == validation['payload'] == b['payload'], 'SOURCE_ENVELOPE')
    need(validation.get('receipt', b['receipt']) == b['receipt'], 'VALIDATED_RECEIPT')
    need(validation['status'].endswith(('CPU_REPLAY_PASS', '_CPU_PASS')), 'CPU_VALIDATED_SOURCE')


def prepare():
    need(not AUTH.exists(), 'IMMUTABLE_AUTHORITY')
    pre = read(OUT / 'preflight.json')
    need(pre['status'] == 'CRISP_MANUAL_SYNTHETIC_PASS' and pre['program'] == bind(PROGRAM), 'PREFLIGHT_CURRENT')
    upstream = UPSTREAM.read_bytes()
    need(hashlib.sha1(b'blob ' + str(len(upstream)).encode() + b'\0' + upstream).hexdigest()
         == 'fa2c71e533c477cf805f2bf119a9d3c9160b61c9', 'OFFICIAL_GIT_BLOB')
    panels = {}
    for panel, root, manifest in [('H593', H593, H593 / 'metadata/worker_manifest.json'),
                                   ('RPC600', RPC, RPC / 'worker_manifest.json')]:
        rows = []
        envelopes = {}
        def cached_source(kind, shard):
            key = (kind, shard)
            if key not in envelopes:
                envelopes[key] = source(root / kind / f'shard{shard:02d}')
            return envelopes[key]
        for w in read(manifest)['records']:
            if 'reuse' in w:
                z = w['reuse']
                raw, roma = z['token_raw'], z['roma']
                validate_envelope(raw); validate_envelope(roma)
                index, expected = z['record_index'], z['source_query_id']
            else:
                ordinal = w.get('missing_ordinal', w['execution_ordinal'])
                raw, roma = cached_source('raw', ordinal // 8), cached_source('roma', ordinal // 8)
                index, expected = ordinal % 8, w['query_id']
            rows.append(dict(query_id=w['query_id'], execution_ordinal=w['execution_ordinal'],
                             source_query_id=expected, source_index=index,
                             source_image_sha256=w.get('source_image_sha256'), raw=raw, roma=roma))
        need(len(rows) == (593 if panel == 'H593' else 600), 'COMPLETE_PANEL')
        need([r['execution_ordinal'] for r in rows] == list(range(len(rows))), 'WORKER_ORDER')
        path = OUT / f'{panel}_workers.json'
        write(path, dict(panel=panel, source_manifest=bind(manifest), labels_included=False, records=rows))
        panels[panel] = dict(worker=bind(path), count=len(rows))
    postjoin = dict(
        H593=dict(curator=bind(H593 / 'metadata/curator_roles.json'),
                  gallery=bind(ROOT / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'),
                  comparator=bind(ROOT / 'results/rc_six_cause_isolation_v1/loss_binding/result.json'),
                  head_names=dict(COST1='COST1', CE='ALL_CE'), expected_raw=426,
                  expected_COST1=481, expected_CE=486, expected_recall=570,
                  lineage='Grouped five-fold OOF heads, training only inside each fold'),
        RPC600=dict(curator=bind(RPC / 'curator_roles.json'), gallery=bind(RPC / 'gallery_manifest.json'),
                    comparator=bind(RPC / 'result.json'), head_names=dict(COST1='COST1', CE='CE'),
                    expected_raw=192, expected_COST1=207, expected_CE=217, expected_recall=600,
                    lineage='Frozen full-H593 heads transferred to the already-opened RPC600 panel'))
    # expected_recall is checked against the original report at join, not assumed for claims.
    postjoin['RPC600'].pop('expected_recall')
    write(AUTH, dict(status='FIXED_CACHE_UNTRAINED_BASELINES_AUTHORIZED',
         user_request='Try CRISP and manually assigned head weights; no training',
         sources=dict(program=bind(PROGRAM), launcher=bind(LAUNCHER), helper=bind(Path(M.__file__)),
                      features=bind(Path(FC.__file__)), upstream=bind(UPSTREAM),
                      upstream_provenance=bind(UPSTREAM.parent / 'source.json'),
                      plan=bind(ROOT / 'plan/RC_CRISP_MANUAL_HEAD_BASELINE_V1_20260918.md'),
                      preflight=bind(OUT / 'preflight.json')),
         panels=panels, postjoin=postjoin, manual_theta=MANUAL, batch_size=8,
         candidate_source='Unchanged natural RAW C128; no target insertion',
         task='Image-to-image CRISP adaptation, not original text-to-flowchart reproduction',
         image_tokens_only=True, extra_normalization=False, training_updates=0,
         bootstrap_seed=20260918, bootstrap_draws=100000, shard_size=8, shards_per_panel=75))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH), panels={k:v['count'] for k,v in panels.items()})))


def guard(worker=False):
    a = read(AUTH)
    need(a['manual_theta'] == MANUAL, 'FIXED_MANUAL_COEFFICIENTS')
    allowed = {Path(checked(b)).resolve() for b in a['sources'].values()}
    allowed.add(AUTH.resolve())
    for info in a['panels'].values():
        w = checked(info['worker']); allowed.add(w.resolve())
        for r in read(w)['records']:
            for kind in ('raw', 'roma'):
                allowed.update(Path(b['path']).resolve() for b in r[kind].values())
    if worker:
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        def audit(event, args):
            if event == 'socket.connect':
                need(not isinstance(args[1], tuple), 'OFFLINE_WORKER')
            if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(args[0])).resolve(); name = str(p).lower()
            need(not any(s in name for s in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi',
                 '/reports/', 'rc_opened_', 'gisc_prerecall_universe')), 'NO_LABEL_OR_PROTECTED_READ')
            if ROOT / 'results' in p.parents:
                need(p in allowed or OUT in p.parents, 'UNLISTED_RESULT:' + name)
        sys.addaudithook(audit)
    return a


def numpy_scores(q, p):
    """Independent high precision reference for ordinary active image tokens."""
    q = np.asarray(q, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    q = q[np.abs(q).sum(1) > 1e-6]
    p = p[np.linalg.norm(p, axis=1) > 1e-6]
    need(len(q) > 0 and len(p) > 1, 'ACTIVE_TOKENS')
    sim = q @ p.T
    d = np.linalg.norm(p - p.mean(0), axis=1)
    weights = d / max(d.sum(), 1e-8) * len(p)
    weighted = sim * weights
    valid = weights > 0
    need(valid.any(), 'NONDEGENERATE_REFERENCE')
    forward = weighted[:, valid].max(1) - weighted.sum(1) / valid.sum()
    return float(forward.mean() + sim.max(0).mean()), float(sim.max(1).mean())


def patch_maxsim(q, refs, device, batch_size=8):
    q = q[q.abs().sum(-1) > 1e-6].to(device)
    out = []
    for start in range(0, len(refs), batch_size):
        p = pad_sequence(refs[start:start + batch_size], batch_first=True).to(device)
        active = p.abs().sum(-1) > 1e-6
        need(bool(active.any(1).all()), 'REFERENCE_HAS_ACTIVE_TOKENS')
        sim = torch.einsum('nd,csd->cns', q, p)
        sim.masked_fill_(~active[:, None, :], -torch.inf)
        out.extend(sim.max(-1).values.mean(-1).cpu().tolist())
    return out


def action(axis, raw_selected, logits):
    challengers = [p for p in axis if p != raw_selected]
    k = max(range(len(challengers)), key=lambda i: logits[i])
    return challengers[k] if logits[k] > 0 else raw_selected


def replay(r):
    axis = r['candidate_physical_rows']; raw = r['selected']['RAW']
    need(axis == sorted(set(axis)) and len(axis) == 128 and raw in axis, 'NATURAL_AXIS')
    need(r['raw_ranked_physical_rows'][0] == raw and sorted(r['raw_ranked_physical_rows'][:128]) == axis, 'RAW_AXIS')
    for m in SCORED:
        scores = r['scores'][m]
        need(len(scores) == 128 and np.isfinite(scores).all(), 'FINITE_SCORES')
        j = max(range(128), key=lambda i: scores[i])
        need(axis[j] == r['selected'][m], 'INDEPENDENT_SCORE_ACTION')
    x = np.asarray(r['features'], dtype=np.float64)
    need(x.shape == (127, 6) and np.isfinite(x).all(), 'SIX_FEATURES')
    for m, theta in MANUAL.items():
        expected = np.sum(x * theta[:-1], axis=1) + theta[-1]
        z = np.asarray(r['logits'][m])
        need(np.max(np.abs(expected-z)) < 2e-10, 'INDEPENDENT_HEAD_LOGITS')
        need(action(axis, raw, expected) == action(axis, raw, z) == r['selected'][m], 'INDEPENDENT_HEAD_ACTION')
    need(r['selected']['ZERO_HEAD'] == raw, 'ZERO_IS_RAW')


def preflight():
    upstream = M.loadmodule(UPSTREAM, 'official_crisp_preflight')
    torch.manual_seed(17); errors = []
    for negative in (False, True):
        qs = [torch.randn(n, 13) for n in (3, 7)]
        ps = [torch.randn(n, 13) for n in (4, 9, 5)]
        if negative:
            qs = [v.abs() for v in qs]; ps = [-v.abs() for v in ps]
        scores = upstream.crisp_score(qs, ps, torch.device('cpu'), 2).numpy()
        for i, q in enumerate(qs):
            matched = patch_maxsim(q, ps, torch.device('cpu'), 2)
            for j, p in enumerate(ps):
                ref, maxsim = numpy_scores(q.numpy(), p.numpy())
                errors.extend((abs(float(scores[i,j])-ref), abs(matched[j]-maxsim)))
    need(max(errors) < 1e-5, 'NUMPY_CRISP_AND_MAXSIM')
    axis = list(range(128)); x = np.zeros((127,6))
    x[1,0] = x[2,0] = 1
    raw = 0
    for theta in MANUAL.values():
        z = (torch.tensor(x) @ torch.tensor(theta[:-1], dtype=torch.float64) + theta[-1]).tolist()
        need(action(axis, raw, z) == (0 if max(z) <= 0 else 2), 'HOLD_AND_STABLE_TIE')
    write(OUT / 'preflight.json', dict(status='CRISP_MANUAL_SYNTHETIC_PASS', program=bind(PROGRAM),
         official_source=bind(UPSTREAM), maximum_numpy_error=max(errors), scalar_checks=len(errors),
         negative_similarity=True, unequal_lengths=True, HOLD_and_tie_checks=True,
         natural_outcomes_read=0, natural_tensor_reads=0, manual_theta=MANUAL))
    print(json.dumps(dict(status='SYNTHETIC_PASS', max_error=max(errors))))


def worker(shard):
    a = guard(worker=True)
    need(shard in range(150), 'SHARD_RANGE')
    panel = 'H593' if shard < 75 else 'RPC600'; local = shard % 75
    workers = read(a['panels'][panel]['worker']['path'])['records'][local*8:(local+1)*8]
    device = torch.device('cuda'); need(torch.cuda.is_available(), 'GPU_REQUIRED')
    official = M.loadmodule(UPSTREAM, 'official_crisp_runtime')
    cache = {}; records = []; start = time.perf_counter(); checks = []
    def load(b):
        key = b['payload']['sha256']
        if key not in cache:
            validate_envelope(b)
            cache[key] = torch.load(checked(b['payload']), map_location='cpu', weights_only=True, mmap=True)
        return cache[key]
    with torch.inference_mode():
        for w in workers:
            raw, roma = load(w['raw']), load(w['roma'])
            q, r = raw['records'][w['source_index']], roma['records'][w['source_index']]
            need(q['query_id'] == r['query_id'] == w['source_query_id'], 'SOURCE_QUERY')
            need(q['query_source_sha256'] == r['query_source_sha256'], 'SOURCE_IMAGE')
            if w['source_image_sha256'] is not None:
                need(q['query_source_sha256'] == w['source_image_sha256'], 'MANIFEST_IMAGE_SHA')
            axis = q['candidate_physical_rows']
            need(axis == r['candidate_physical_rows'] == sorted(set(axis)) and len(axis) == 128, 'C128_AXIS')
            need(torch.equal(q['candidate_raw_scores'], r['candidate_raw_scores']), 'RAW_SCORE_BINDING')
            refs = []
            for j, c in enumerate(r['candidates']):
                ref = raw['references'][axis[j]]
                need(c['candidate_position'] == j and c['physical_row'] == axis[j], 'CANDIDATE_POSITION')
                need(c['reference_tokens_sha256'] == ref['tokens_sha256'], 'REFERENCE_TOKEN_BINDING')
                refs.append(ref['tokens'].float())
            need(len(refs) == 128, 'ALL_REFERENCES')
            query = q['query_tokens'].float()
            scores = dict(CRISP=official.crisp_score([query], refs, device, a['batch_size'])[0].tolist(),
                          PATCH_MAXSIM=patch_maxsim(query, refs, device, a['batch_size']),
                          VISIBILITY_DIRECT=[float(c['old_scores']['real_score']) for c in r['candidates']])
            crisp_check, maxsim_check = numpy_scores(query.numpy(), refs[0].numpy())
            error = max(abs(crisp_check-scores['CRISP'][0]), abs(maxsim_check-scores['PATCH_MAXSIM'][0]))
            need(error < 2e-5, 'NATURAL_NUMPY_SCORER_CHECK'); checks.append(error)
            winner = axis.index(q['candidate_ranked_physical_rows'][0])
            raw_selected = axis[winner]
            ev = {j:c['old_scores'] for j,c in enumerate(r['candidates'])}
            x = torch.stack([FC.candidate_feature(q['candidate_raw_scores'].tolist(), ev, c, winner)
                             for c in range(128) if c != winner])
            selected = dict(RAW=raw_selected)
            for m, values in scores.items():
                selected[m] = axis[int(np.argmax(values))]
            logits = {}
            for m, values in MANUAL.items():
                theta = torch.tensor(values, dtype=torch.float64)
                logits[m] = (x @ theta[:-1] + theta[-1]).tolist()
                selected[m] = action(axis, raw_selected, logits[m])
            row = dict(query_id=w['query_id'], execution_ordinal=w['execution_ordinal'],
                       source_image_sha256=q['query_source_sha256'], candidate_physical_rows=axis,
                       raw_ranked_physical_rows=q['raw_ranked_physical_rows'], scores=scores,
                       selected=selected, features=x.tolist(), logits=logits)
            replay(row); records.append(row)
            print(json.dumps(dict(event='PREDICTION_READY', panel=panel, query_id=w['query_id'],
                  elapsed_seconds=time.perf_counter()-start, labels_read=0)), flush=True)
    folder = OUT / 'predictions' / f'shard{shard:03d}'
    write(folder / 'payload.json', dict(authority=bind(AUTH), panel=panel, global_shard=shard,
          records=records, elapsed_seconds=time.perf_counter()-start, GPU=torch.cuda.get_device_name(),
          torch_version=str(torch.__version__), target_reads=0, training_updates=0))
    write(folder / 'validation.json', dict(status='CRISP_MANUAL_NUMPY_ACTION_PASS', authority=bind(AUTH),
          payload=bind(folder / 'payload.json'), count=len(records), max_numpy_score_error=max(checks),
          independent_candidate_checks=len(checks), target_reads=0))


def comparison(rows, model, baseline, a):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['component']].append(r)
    ns, ds = [], []
    for _, rr in sorted(groups.items()):
        ns.append(len(rr)); ds.append(sum(int(r['correct'][model])-int(r['correct'][baseline]) for r in rr))
    ns, ds = np.array(ns), np.array(ds)
    rng = np.random.default_rng(a['bootstrap_seed']); boots = []
    for start in range(0, a['bootstrap_draws'], 2000):
        ix = rng.integers(len(ns), size=(min(2000, a['bootstrap_draws']-start), len(ns)))
        boots.append(ds[ix].sum(1)/ns[ix].sum(1))
    rescue = sum(r['correct'][model] and not r['correct'][baseline] for r in rows)
    breaks = sum(not r['correct'][model] and r['correct'][baseline] for r in rows)
    need(rescue-breaks == int(ds.sum()), 'PAIRED_COUNTS')
    return dict(model=model, baseline=baseline, rescues=rescue, breaks=breaks, net=rescue-breaks,
                grouped_bootstrap95=np.quantile(np.concatenate(boots), [.025,.975]).tolist(),
                groups=len(ns), draws=a['bootstrap_draws'], seed=a['bootstrap_seed'])


def join():
    a = guard(); predictions = collections.defaultdict(list); validations = []
    for shard in range(150):
        folder = OUT / 'predictions' / f'shard{shard:03d}'
        v = read(folder / 'validation.json')
        need(v['status'] == 'CRISP_MANUAL_NUMPY_ACTION_PASS' and v['authority'] == bind(AUTH), 'ALL_SHARDS_VALID')
        p = read(checked(v['payload']))
        need(p['authority'] == bind(AUTH) and p['target_reads'] == p['training_updates'] == 0, 'SEALED_BLIND_PREDICTIONS')
        need(p['global_shard'] == shard and len(p['records']) == (1 if shard == 74 else 8), 'SHARD_COMPLETENESS')
        for r in p['records']:
            replay(r)
        predictions[p['panel']].extend(p['records']); validations.append(bind(folder / 'validation.json'))
    for panel, info in a['panels'].items():
        need([r['query_id'] for r in predictions[panel]] == [w['query_id'] for w in read(checked(info['worker']))['records']], 'COMPLETE_QUERY_ORDER')
    write(OUT / 'all_predictions_prejoin_seal.json', dict(authority=bind(AUTH), validations=validations,
          counts={p:len(rr) for p,rr in predictions.items()}, independent_action_replay=True))
    # First label and original-result reads occur only after the complete new predictions are sealed.
    result = {}; lines = ['# CRISP 与手填参数头：固定缓存对照', '',
        'CRISP 是 image-to-image 适配；所有模型使用既有 RAW 自然 C128，未训练或调参。', '',
        'H593 的 COST1/CE 为原五折 OOF；RPC600 为原 full-H593 冻结头，RPC 已打开，不是新外部确认。', '']
    for panel, pp in predictions.items():
        meta = a['postjoin'][panel]
        gallery = read(checked(meta['gallery']))['records']; labels = [r['identity'] for r in gallery]
        roles = {r['query_id']:r for r in read(checked(meta['curator']))['records']}
        old = {r['query_id']:r for r in read(checked(meta['comparator']))['rows']}
        need(set(roles) == set(old) == {r['query_id'] for r in pp}, 'JOIN_QUERY_SETS')
        rows = []
        for p in pp:
            qid = p['query_id']; role = roles[qid]
            correct = {m:labels[s] == role['identity'] for m,s in p['selected'].items()}
            need(correct['RAW'] == old[qid]['correct']['RAW'], 'HISTORICAL_RAW_REPLAY')
            for name, oldname in meta['head_names'].items():
                correct[name] = old[qid]['correct'][oldname]
            recall = role['identity'] in [labels[s] for s in p['candidate_physical_rows']]
            need(recall == old[qid]['target_in_C128'], 'HISTORICAL_C128_REPLAY')
            rows.append(dict(query_id=qid, identity=role['identity'], component=role['component'],
                             selected=p['selected'], correct=correct, target_in_C128=recall))
        counts = {m:sum(r['correct'][m] for r in rows) for m in MODELS}
        need(counts['RAW'] == meta['expected_raw'] and counts['COST1'] == meta['expected_COST1']
             and counts['CE'] == meta['expected_CE'], 'FROZEN_BASELINE_COUNTS')
        recall = sum(r['target_in_C128'] for r in rows)
        if 'expected_recall' in meta:
            need(recall == meta['expected_recall'], 'C128_RECALL')
        comparisons = {base:{m:comparison(rows,m,base,a) for m in MODELS if m != base}
                       for base in ('RAW','PATCH_MAXSIM','COST1','CE')}
        result[panel] = dict(counts=counts, queries=len(rows), target_recall_C128=recall,
                             comparisons=comparisons, rows=rows, head_lineage=meta['lineage'])
        lines += [f'## {panel}', '', f"自然 C128 包含 target：{recall}/{len(rows)}。", '',
                  '| 方法 | 正确 | 准确率 | 相对 RAW 救回 | 改错 | 净增 |', '|---|---:|---:|---:|---:|---:|']
        for m in MODELS:
            c = comparisons['RAW'].get(m, dict(rescues=0,breaks=0,net=0))
            lines.append(f"| {m} | {counts[m]}/{len(rows)} | {counts[m]/len(rows):.2%} | {c['rescues']} | {c['breaks']} | {c['net']:+d} |")
        lines += ['', '固定手填权重不是手工参数族的最优值；把训练后的参数抄入程序仍是训练模型。', '']
        with (OUT / f'{panel}_per_query.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['query_id','identity','component','target_in_C128',*MODELS])
            writer.writeheader()
            writer.writerows({**{k:r[k] for k in ('query_id','identity','component','target_in_C128')},
                              **{m:int(r['correct'][m]) for m in MODELS}} for r in rows)
    write(OUT / 'result.json', dict(status='CRISP_MANUAL_BASELINE_COMPLETE', authority=bind(AUTH),
         panels=result, training_updates=0, manual_theta=MANUAL, formal_GO_claimed=False))
    with (OUT / 'report_zh.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    write(OUT / 'result_validation.json', dict(status='ALL_SEALS_COUNTS_AND_ACTIONS_PASS',
         result=bind(OUT / 'result.json'), queries=1193, shards=150, authority=bind(AUTH)))
    print(json.dumps({p:{k:v[k] for k in ('counts','queries','target_recall_C128')} for p,v in result.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight','prepare','worker','join'])
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest'); torch.manual_seed(17)
    if args.stage == 'worker':
        worker(args.shard)
    else:
        globals()[args.stage]()
