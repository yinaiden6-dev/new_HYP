#!/usr/bin/env python3
"""Independent NumPy audit and final H593 OOF join, after all scores are sealed.

No fitting, checkpoint selection, candidate insertion, or head threshold tuning
occurs here. Curator labels are opened only after all five folds, three trained
arms, and REAL inference interventions have passed score and source checks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / 'registry/rc_postllm_h593_authority_v1_20260924.json'
OUT = ROOT / 'results/rc_postllm_h593_v1'
REPORT = ROOT / 'reports/REPORT_POSTLLM_H593_V1_20260924.md'
ARMS = ('POST_REAL', 'POST_CONSTANT', 'POST_SHUFFLED')
KINDS = ('INTERNAL3', 'ADDITIVE4', 'PRODUCT5')
FEATURES = dict(INTERNAL3=['raw_standardized_delta', 'sym_L', 'bias'],
    ADDITIVE4=['raw_standardized_delta', 'sym_M', 'sym_L', 'bias'],
    PRODUCT5=['raw_standardized_delta', 'sym_ML', 'sym_M', 'sym_L', 'bias'])


def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).absolute(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


class Checker:
    def __init__(self):
        self.seen = {}
    def __call__(self, b):
        p = Path(b['path']); stat = p.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        key = (str(p.absolute()), b['sha256'])
        if key not in self.seen:
            need(bind(p) == b, 'SOURCE_SHA_DRIFT:' + str(p))
            self.seen[key] = signature
        need(self.seen[key] == signature, 'SOURCE_CHANGED_DURING_JOIN:' + str(p))
        return p


def write(path, value):
    path = Path(path); text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        need(path.read_text() == text, 'IMMUTABLE_JOIN_OUTPUT:' + str(path)); return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text); os.replace(tmp, path)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def numpy_decision(row, content, theta, kind='INTERNAL3'):
    raw = np.asarray(row['raw_scores'], dtype=np.float64)
    mass = np.asarray(row['M'], dtype=np.float64)
    content = np.asarray(content, dtype=np.float64); theta = np.asarray(theta, dtype=np.float64)
    need(raw.shape == mass.shape == content.shape == (128,), 'FULL128_INPUTS')
    need(np.isfinite(raw).all() and np.isfinite(mass).all() and np.isfinite(content).all()
         and np.isfinite(theta).all(), 'FINITE_SCORE_INPUTS')
    need(kind in FEATURES and theta.shape == (len(FEATURES[kind]),), 'HEAD_DIMENSION')
    winner = int(row['winner_index']); challengers = row['challenger_positions']
    need(challengers == [i for i in range(128) if i != winner], 'CHALLENGER_ORDER')
    idx = np.asarray(challengers)
    def sym(x):
        return (x[idx] - x[winner]) / (np.abs(x[idx]) + abs(x[winner]) + 1e-12)
    columns = [(raw[idx] - raw[winner]) / max(float(raw.std(ddof=0)), 1e-12)]
    if kind == 'PRODUCT5':
        columns.append(sym(mass * content))
    if kind != 'INTERNAL3':
        columns.append(sym(mass))
    columns.extend([sym(content), np.ones(127, dtype=np.float64)])
    logits = np.stack(columns, axis=1) @ theta
    position = challengers[int(np.argmax(logits))] if float(logits.max()) > 0 else winner
    scores = np.zeros(128, dtype=np.float64); scores[idx] = logits
    return dict(prediction_position=position, prediction_id=row['candidate_ids'][position],
        prediction_identity=row['candidate_identities'][position], switched=position != winner,
        challenger_positions=challengers, logits=logits.tolist(), scores128=scores.tolist(),
        theta=theta.tolist(), features=FEATURES[kind])


def compare_decision(actual, expected):
    need(np.array_equal(np.asarray(actual['theta']), np.asarray(expected['theta'])), 'EXACT_HEAD_COEFFICIENTS')
    for key in ('prediction_position', 'prediction_id', 'prediction_identity', 'switched', 'challenger_positions', 'features'):
        need(actual[key] == expected[key], 'INDEPENDENT_ACTION:' + key)
    err = 0.
    for key, shape in (('logits', (127,)), ('scores128', (128,))):
        x = np.asarray(actual[key], dtype=np.float64); y = np.asarray(expected[key], dtype=np.float64)
        need(x.shape == y.shape == shape and np.isfinite(x).all(), 'DECISION_VECTOR_SHAPE')
        err = max(err, float(np.abs(x - y).max()))
    need(err <= 1e-10, 'INDEPENDENT_NUMPY_SCORE_ERROR')
    return err


def required_seals(out):
    """Fail before labels when any fold/arm or endpoint is incomplete."""
    paths = []
    for f in range(5):
        d = Path(out) / f'fold{f}'
        paths.append(d / 'warm/validation.json')
        for arm in ARMS:
            paths.extend([d / arm / 'fit_validation.json',
                d / arm / 'train_endpoint/validation-000-of-001.json',
                d / arm / 'held/validation-000-of-001.json'])
    for p in paths:
        need(p.is_file(), 'INCOMPLETE_BEFORE_LABEL_JOIN:' + str(p))
    return paths


def prelabel_audit(authority_path, out):
    import torch
    out = Path(out); check = Checker(); a = read(authority_path); ab = bind(authority_path)
    for b in a.get('code_sources', []):
        check(b)
    required = required_seals(out)
    manifest = read(check(a['manifest'])); rows = {r['query_id']: r for r in manifest['rows']}
    need(len(rows) == 593, 'ALL593_ROWS')
    held_predictions = {}; training_summary = {}; seals = []; maxerr = 0.
    cache_content = {}; cache_payloads = {}
    for f in range(5):
        fd = manifest['folds'][str(f)]; train_ids = fd['train_query_ids']; held_ids = fd['held_query_ids']
        need(set(train_ids).isdisjoint(held_ids), 'TRAIN_HELD_DISJOINT')
        scope = dict(manifest=a['manifest'], labels=fd['train_labels'],
            train_axis_sha256=object_sha(train_ids), held_axis_sha256=object_sha(held_ids))
        labels = {r['query_id']: r for r in read(check(fd['train_labels']))['records']}
        need(set(labels) == set(fd['train_all_query_ids']) and set(train_ids) <= set(labels), 'TRAIN_ONLY_ROLE_SCOPE')
        warmdir = out / f'fold{f}/warm'; ws = read(warmdir / 'validation.json')
        need(ws['authority'] == ab and ws['input_scope'] == scope and ws['held_label_reads'] == 0, 'WARM_SCOPE')
        heads = {}
        for kind in KINDS:
            h = read(check(ws['heads'][kind]))
            need(h['authority'] == ab and h['input_scope'] == scope and h['kind'] == kind
                 and h['fit_queries'] == train_ids and h['held_label_reads'] == 0
                 and not h['old_pilot_head_reused'] and h['feature_order'] == FEATURES[kind], 'FRESH_TRAIN_HEAD')
            heads[kind] = h
        baseline_l0 = {}; train_decisions = {arm: {} for arm in ARMS}; first_initial = None
        for arm in ARMS:
            directory = out / f'fold{f}' / arm; fit = read(directory / 'fit_validation.json')
            need(fit['authority'] == ab and fit['input_scope'] == scope and fit['arm'] == arm
                 and fit['steps'] == 8 * len(train_ids) and fit['passes'] == 8
                 and fit['train_queries'] == len(train_ids) and fit['held_label_reads'] == 0
                 and not fit['old_pilot_parameters_reused'] and not fit['direct_M_in_head'], 'FIXED_FINAL_FIT')
            state = torch.load(check(fit['snapshot']), map_location='cpu', weights_only=True)
            need(state['authority'] == ab and state['input_scope'] == scope and state['arm'] == arm
                 and state['step'] == fit['steps'] and state['warmstart'] == ws['heads']['INTERNAL3']
                 and state['normalization'] == fd['mass_normalization'], 'SNAPSHOT_SOURCE_AND_FOLD')
            initial_path = directory / 'initial.pt'
            initial = torch.load(check(bind(initial_path)), map_location='cpu', weights_only=True)
            need(initial['authority'] == ab and initial['input_scope'] == scope and initial['arm'] == arm
                 and initial['step'] == 0 and not initial['pilot_checkpoint_reused']
                 and initial['warmstart'] == ws['heads']['INTERNAL3']
                 and initial['head'].double().tolist() == heads['INTERNAL3']['theta'], 'FRESH_INITIAL_HEAD')
            need(not bool(initial['adapter']['up.weight'].any()) and not bool(initial['adapter']['up.bias'].any()), 'ZERO_INITIAL_ADAPTER_OUTPUT')
            if first_initial is None:
                first_initial = initial['adapter']
            else:
                need(first_initial.keys() == initial['adapter'].keys() and
                     all(torch.equal(first_initial[k], initial['adapter'][k]) for k in first_initial), 'MATCHED_INITIAL_ADAPTER_PARAMETERS')
            theta = state['head'].double().tolist()
            for split, query_ids, folder in (('train', train_ids, 'train_endpoint'), ('held', held_ids, 'held')):
                ep = directory / folder / 'validation-000-of-001.json'; seal = read(ep)
                modes = ['native', 'constant', 'shuffled'] if split == 'held' and arm == 'POST_REAL' else ['native']
                need(seal['status'] == 'FULL_C128_ENDPOINT_SHARD_COMPLETE' and seal['authority'] == ab
                     and seal['input_scope'] == scope and seal['snapshot'] == fit['snapshot']
                     and seal['split'] == split and seal['arm'] == arm and seal['fold'] == f
                     and seal['shard_index'] == 0 and seal['shard_count'] == 1
                     and seal['query_ids'] == query_ids and seal['modes'] == modes
                     and seal['held_label_reads'] == 0, 'COMPLETE_SEALED_ENDPOINT')
                expected_paths = [str((directory / folder / mode / (q + '.json')).absolute()) for q in query_ids for mode in modes]
                need([b['path'] for b in seal['predictions']] == expected_paths, 'EXACT_ALL_PREDICTION_FILES')
                for b in seal['predictions']:
                    rec = read(check(b)); q = rec['query_id']; row = rows[q]; mode = rec['intervention']
                    need(q in query_ids and mode in modes and rec['authority'] == ab
                         and rec['input_scope'] == scope and rec['snapshot'] == fit['snapshot']
                         and rec['split'] == split and rec['fold'] == f and rec['arm'] == arm
                         and rec['held_label_reads'] == rec['training_updates'] == 0
                         and not rec['direct_M_in_head'], 'PREDICTION_SCOPE')
                    for key in ('candidate_ids', 'candidate_identities', 'raw_scores', 'M', 'winner_index'):
                        need(rec[key] == row[key], 'FIXED_CANDIDATE_INPUT:' + key)
                    hs = rec['hidden_source']; hs_seal = read(check(hs['validation']))
                    need(hs['candidate_axis_sha256'] == object_sha(row['candidate_ids'])
                         and hs_seal['payload'] == hs['payload'] and 'PASS' in hs_seal['status']
                         and hs_seal['image_sha256'] == row['source_image_sha256']
                         and not hs_seal['held_labels_read'], 'HIDDEN_SOURCE_AXIS')
                    payload_path = check(hs['payload'])
                    if q not in cache_content:
                        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
                        need(payload['query_id'] == q and payload['hidden'].shape[-1] == 3584, 'ACTUAL_POSTLLM_HIDDEN_SOURCE')
                        cache_content[q] = torch.as_tensor(payload['fresh_L0'], dtype=torch.float64).tolist()
                        cache_payloads[q] = hs['payload']; del payload
                    else:
                        need(cache_payloads[q] == hs['payload'], 'SAME_QUERY_CACHE_ACROSS_FOLDS')
                    if row['hidden_validation']:
                        need(hs['validation'] == row['hidden_validation'], 'LEGACY_CACHE_BINDING')
                    else:
                        need(hs_seal['authority'] == ab, 'NEW_CACHE_AUTHORITY')
                    native = numpy_decision(row, rec['L'], theta)
                    maxerr = max(maxerr, compare_decision(rec['decision'], native))
                    l0 = rec['L0']; need(len(l0) == 128 and np.isfinite(l0).all()
                        and l0 == cache_content[q], 'BASELINE_FULL128_BOUND_TO_HIDDEN_CACHE')
                    if q in baseline_l0:
                        need(l0 == baseline_l0[q], 'SHARED_BASELINE_CONTENT_ALL_ARMS')
                    else:
                        baseline_l0[q] = l0
                    external = {}
                    if mode == 'native':
                        need(set(rec['external']) == set(KINDS), 'ALL_EXTERNAL_CONTROLS')
                        for kind in KINDS:
                            d = numpy_decision(row, l0, heads[kind]['theta'], kind)
                            maxerr = max(maxerr, compare_decision(rec['external'][kind], d)); external['EXTERNAL_' + kind] = d
                        if split == 'train':
                            for kind in KINDS:
                                maxerr = max(maxerr, compare_decision(heads[kind]['predictions'][q], external['EXTERNAL_' + kind]))
                    else:
                        need(not rec['external'], 'INTERVENTION_HAS_NO_REFIT')
                    if split == 'held':
                        item = held_predictions.setdefault(q, dict(query_id=q, fold=f, decisions={}))
                        need(item['fold'] == f, 'SINGLE_OOF_FOLD')
                        name = arm if mode == 'native' else arm + '_' + mode.upper()
                        need(name not in item['decisions'], 'NO_DUPLICATE_PREDICTION')
                        item['decisions'][name] = native
                        for name, decision in external.items():
                            if name in item['decisions']:
                                need(item['decisions'][name] == decision, 'EXTERNAL_CONTROL_IDENTICAL_BETWEEN_ARMS')
                            item['decisions'][name] = decision
                    else:
                        train_decisions[arm][q] = native
                seals.append(bind(ep))
        training_summary[str(f)] = {arm: dict(n=len(train_ids),
            correct=sum(d['prediction_identity'] == labels[q]['target_id'] for q, d in pp.items()))
            for arm, pp in train_decisions.items()}
    expected_names = set(ARMS) | {'POST_REAL_CONSTANT', 'POST_REAL_SHUFFLED'} | {'EXTERNAL_' + k for k in KINDS}
    need(set(held_predictions) == set(rows) and all(set(v['decisions']) == expected_names for v in held_predictions.values()), 'ALL593_ALL8_DECISIONS')
    preseal = dict(status='ALL593_PREDICTIONS_INDEPENDENT_NUMPY_PASS_BEFORE_LABEL_JOIN', authority=ab,
        manifest=a['manifest'], required_seals=[bind(p) for p in required], endpoint_seals=seals,
        query_count=593, model_decisions=593 * len(expected_names), max_logit_error=maxerr,
        training_summary=training_summary, curator_label_reads=0, independent_numpy=True,
        checked_unique_files=len(check.seen))
    write(out / 'all_predictions_prelabel_seal.json', preseal)
    return a, manifest, held_predictions, preseal, check


def paired_interval(rows, model, baseline, samples=10000):
    groups = sorted({r['component'] for r in rows}); by = {g: [r for r in rows if r['component'] == g] for g in groups}
    totals = np.asarray([len(by[g]) for g in groups], dtype=np.float64)
    diffs = np.asarray([sum(int(r['correct'][model]) - int(r['correct'][baseline]) for r in by[g]) for g in groups], dtype=np.float64)
    rng = np.random.default_rng(20260924); index = rng.integers(0, len(groups), size=(samples, len(groups)))
    pooled = diffs[index].sum(1) / totals[index].sum(1)
    macro = (diffs / totals)[index].mean(1)
    return dict(baseline=baseline, groups=len(groups), samples=samples, seed=20260924,
        pooled_accuracy_difference=float(diffs.sum() / totals.sum()),
        group_resampled_pooled_95ci=np.quantile(pooled, [.025, .975]).tolist(),
        equal_group_accuracy_difference=float((diffs / totals).mean()),
        equal_group_95ci=np.quantile(macro, [.025, .975]).tolist(),
        interpretation='Development OOF paired group bootstrap; not an untouched confirmation or automatic GO test')


def join(authority_path=AUTH, out=OUT, report=REPORT):
    out = Path(out); a, manifest, predictions, preseal, check = prelabel_audit(authority_path, out)
    # This is intentionally the first and only curator access in the program.
    roles = {r['query_id']: r for r in read(check(manifest['held_join_source']))['records']}
    need(set(roles) == set(predictions), 'EXACT_CURATOR_QUERY_SET')
    byid = {r['query_id']: r for r in manifest['rows']}; rows = []
    train_roles = {f: read(check(fd['train_labels']))['records'] for f, fd in manifest['folds'].items()}
    for source in manifest['rows']:
        q = source['query_id']; p = predictions[q]; f = p['fold']; role = roles[q]
        tr = train_roles[str(f)]
        need(role['outer_fold'] == f and role['identity'] not in {r['target_id'] for r in tr}
             and role['component'] not in {r['component'] for r in tr}, 'HELD_IDENTITY_AND_COMPONENT_DISJOINT')
        target_positions = [j for j, x in enumerate(source['candidate_identities']) if x == role['identity']]
        need(len(target_positions) <= 1, 'HELD_TARGET_MULTIPLICITY')
        selected = {'RAW': source['candidate_identities'][source['winner_index']]}
        selected.update({k: v['prediction_identity'] for k, v in p['decisions'].items()})
        rows.append(dict(query_id=q, execution_ordinal=source['execution_ordinal'], fold=f,
            identity=role['identity'], component=role['component'], target_in_C128=bool(target_positions),
            target_positions=target_positions, selected=selected,
            correct={k: v == role['identity'] for k, v in selected.items()},
            decisions=p['decisions']))
    need([r['execution_ordinal'] for r in rows] == list(range(593)), 'ALL593_ONCE_ORIGINAL_ORDER')
    need(sum(r['correct']['RAW'] for r in rows) == 426 and sum(r['target_in_C128'] for r in rows) == 570, 'ORIGINAL_H593_BASELINE_AND_RECALL')
    totals = {}
    for name in rows[0]['correct']:
        rescues = [r['query_id'] for r in rows if r['correct'][name] and not r['correct']['RAW']]
        breaks = [r['query_id'] for r in rows if not r['correct'][name] and r['correct']['RAW']]
        totals[name] = dict(correct=sum(r['correct'][name] for r in rows), n=593, rescue=len(rescues), breaks=len(breaks),
            net_gain=len(rescues)-len(breaks), rescue_query_ids=rescues, break_query_ids=breaks,
            changed_decisions=sum(r['selected'][name] != r['selected']['RAW'] for r in rows),
            by_fold={str(f): dict(n=sum(r['fold'] == f for r in rows), correct=sum(r['correct'][name] for r in rows if r['fold'] == f)) for f in range(5)})
        if name != 'RAW':
            totals[name]['paired_vs_RAW'] = paired_interval(rows, name, 'RAW')
    contrasts = {baseline: paired_interval(rows, 'POST_REAL', baseline) for baseline in
        ('POST_CONSTANT', 'POST_SHUFFLED', 'POST_REAL_CONSTANT', 'POST_REAL_SHUFFLED', 'EXTERNAL_INTERNAL3', 'EXTERNAL_ADDITIVE4', 'EXTERNAL_PRODUCT5')}
    result = dict(status='H593_POSTLLM_FIVEFOLD_INDEPENDENT_JOIN_COMPLETE', authority=bind(authority_path),
        prelabel_seal=bind(out / 'all_predictions_prelabel_seal.json'), manifest=a['manifest'],
        candidate_source='Original ColNomic natural C128; no target insertion; all 23 absent-target held queries retained',
        scope='Grouped five-fold OOF on opened H593 development population, fresh fold-only heads/adapters, fixed eight TRAIN passes',
        totals=totals, POST_REAL_paired_contrasts=contrasts, train_endpoint_summary=preseal['training_summary'],
        target_in_C128=570, n=593, rows=rows,
        history_context='Historical NATIVE7 COST1 481/593 is context only, not a newly replayed same-source baseline',
        interpretation='Internal effectiveness does not require beating the external M head; matched constant/shuffled controls determine M-specific attribution')
    write(out / 'result.json', result)
    validation = dict(status='ALL593_POSTLLM_NUMPY_AXES_SNAPSHOTS_ACTIONS_AND_LABELS_PASS',
        authority=bind(authority_path), result=bind(out / 'result.json'), prelabel_seal=result['prelabel_seal'],
        queries=593, candidate_pairs=75904, raw_correct=426, target_present=570,
        max_logit_error=preseal['max_logit_error'], held_join_source=manifest['held_join_source'],
        no_training_in_join=True, same_zero_threshold_all_arms=True, old_pilot_parameters_reused=False)
    write(out / 'validation.json', validation)
    lines = ['# 后 LLM 内部 M 改造：H593 原五折结果', '',
        '原 ColNomic 自然 C128；冻结编码器与检索投影；仅训练后 LLM 适配器和不直读 M 的 INTERNAL3 头。每折从零输出适配器与本折 TRAIN 头开始，固定8轮；没有复用旧 TRAIN16 的参数。', '',
        '| 方法 | 正确 / 593 | 救回 RAW | 误伤 RAW | 净增 |', '|---|---:|---:|---:|---:|']
    lines.extend(f"| {k} | {v['correct']} | {v['rescue']} | {v['breaks']} | {v['net_gain']:+d} |" for k, v in totals.items())
    lines += ['', 'POST_REAL_CONSTANT / SHUFFLED 为同一个训练完的 POST_REAL 模型仅在推理时更换内部 M；POST_CONSTANT / SHUFFLED 为等预算单独训练的对照。EXTERNAL_* 为本轮各折用实际 fresh_L0 从 TRAIN 训练的输出端头，不能与历史481混称同一次重放。', '',
        '全部593张都计入，23张 target 不在自然 C128。RAW426、召回570与原来源一致。先逐项验证全部128分、127 challenger logits、HOLD=0、strict>0切换、候选轴和快照，再读取held curator标签；独立NumPy最大分数差为 ' + format(preseal['max_logit_error'], '.3g') + '。', '',
        '配对分组 bootstrap 区间、逐折结果、逐查询候选 logits、救回/误伤清单与内部M对照均在result.json中。该面板此前已反复用于开发，不能称为新外部确认；内部有效不以超过外部头为必要条件。', '',
        '[完整结果](../results/rc_postllm_h593_v1/result.json) · [独立验证](../results/rc_postllm_h593_v1/validation.json) · [读标签前封存](../results/rc_postllm_h593_v1/all_predictions_prelabel_seal.json)', '']
    text = '\n'.join(lines); report = Path(report)
    if report.exists():
        need(report.read_text() == text, 'IMMUTABLE_FINAL_REPORT')
    else:
        report.write_text(text)
    print(json.dumps(dict(status=validation['status'], result=validation['result'],
        totals={k: {x: v[x] for x in ('correct', 'n', 'rescue', 'breaks')} for k, v in totals.items()})), flush=True)
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, default=AUTH)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    join(args.authority, args.out, args.report)
