#!/usr/bin/env python3
"""Opened-label, exact six-feature capacity diagnostics; never train a head."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'results/rc_opened_h593_rank_capacity_v1'
AUTH = ROOT / 'registry/rc_opened_h593_rank_capacity_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_OPENED_H593_RANK_CAPACITY_V1_20260921.md'
REPORT = ROOT / 'reports/REPORT_OPENED_H593_RANK_CAPACITY_V1_20260921.md'
PARENT = ROOT / 'results/rc_h593_gap_curve_v1'
BASE = ROOT / 'results/rc_six_cause_isolation_v1/loss_binding'


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(source):
    need(bind(source['path']) == source, 'SHA256:' + source['path'])
    return Path(source['path'])


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    parent_validation = read(PARENT / 'validation.json')
    need(parent_validation['status'] == 'GAP_CURVE_ALL_COUNTS_PASS', 'PARENT_VALIDATED')
    checked(parent_validation['result'])
    parent = read(checked(parent_validation['authority']))
    code = dict(program=Path(__file__), plan=PLAN,
                core=ROOT / 'src/rc_aslo_xf/h593_rank_capacity_v1.py',
                independent=ROOT / 'programs/validate_rc_opened_h593_rank_capacity_v1.py',
                launcher=ROOT / 'slurm/rc_opened_h593_rank_capacity_v1.sbatch')
    folds = {}
    for fold in range(5):
        v = read(PARENT / f'fold{fold}/validation.json')
        iv = read(PARENT / f'fold{fold}/independent_validation.json')
        b = read(BASE / f'fold{fold}/validation.json')
        need(iv['passed'] and iv['payload'] == v['payload'], 'PARENT_FOLD')
        need(b['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS', 'BASE_FOLD')
        checked(v['payload']); checked(b['payload'])
        folds[str(fold)] = dict(payload=v['payload'], independent=bind(PARENT / f'fold{fold}/independent_validation.json'),
                                validation=bind(PARENT / f'fold{fold}/validation.json'),
                                base_payload=b['payload'], base_validation=bind(BASE / f'fold{fold}/validation.json'))
    write(AUTH, dict(status='OPENED_H593_RANK_CAPACITY_AUTHORIZED',
                     code_sources={k: bind(v) for k, v in code.items()},
                     parent_authority=parent_validation['authority'], public_sources=parent['public_sources'],
                     parent_result=parent_validation['result'], parent_validation=bind(PARENT / 'validation.json'),
                     features=parent['features'], folds=folds, expected_counts=dict(all=593, RAW_correct=426,
                     target_present=570, ranking_population=144, original_top_correct=104, ranking_blocked=40),
                     source_of_labels='already opened parent H593 result; diagnostic oracle only',
                     model_fits=0, new_accuracy_results=False, certificate_use_in_training=False,
                     protected_sources=['D1-MI', 'formal392', 'external datasets'],
                     scope='per-query capacity plus same-fold preserve-and-add-one ranking capacity'))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage):
    a = read(AUTH)
    need(a['status'] == 'OPENED_H593_RANK_CAPACITY_AUTHORIZED', 'AUTHORITY')
    for source in a['code_sources'].values():
        checked(source)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM')
    if stage in ('audit', 'verify'):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allowed = set()
    def collect(node):
        if isinstance(node, dict):
            if set(node) == {'path', 'sha256'}:
                allowed.add(Path(node['path']).resolve())
            else:
                for value in node.values():
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
    collect(a)
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        s = str(path).lower()
        need(not any(x in s for x in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/')), 'PROTECTED_READ')
        if ROOT / 'results' in path.parents:
            need(OUT in path.parents or path in allowed, 'UNLISTED_RESULT:' + s)
        if ROOT / 'reports' in path.parents:
            need(stage == 'publish' and (path == REPORT or path == REPORT.with_name('.' + REPORT.name + '.tmp')), 'REPORT_ACCESS')
    sys.addaudithook(audit)
    if stage not in ('preflight',):
        pf = read(OUT / 'preflight.json')
        need(pf['authority'] == bind(AUTH) and pf['status'] == 'SYNTHETIC_CAPACITY_PASS', 'PREFLIGHT')
    return a


def inputs(a):
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    result = read(checked(a['parent_result']))
    need(read(checked(a['parent_validation']))['result'] == a['parent_result'], 'PARENT_SEAL')
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    split = read(checked(a['public_sources']['split']))['folds']
    predictions, base_theta = {}, {}
    for fold in range(5):
        s = a['folds'][str(fold)]
        p = read(checked(s['payload'])); b = read(checked(s['base_payload']))
        v, iv, bv = (read(checked(s[k])) for k in ('validation', 'independent', 'base_validation'))
        need(v['payload'] == iv['payload'] == s['payload'] and iv['passed'] and bv['payload'] == s['base_payload'], 'FOLD_SEALS')
        need(set(r['query_id'] for r in p['predictions']) == set(split[fold]['heldout_query_ids']), 'FOLD_AXIS')
        bp = {r['query_id']: r for r in b['predictions']}
        base_theta[fold] = b['parameters']['COST1'][:6]
        for r in p['predictions']:
            need(r['models']['COST1_FULL'] == bp[r['query_id']]['models']['COST1'], 'ORIGINAL_LOGITS_PARITY')
            need(r['query_id'] not in predictions, 'DUPLICATE_PREDICTION')
            predictions[r['query_id']] = r
    features = {}
    for s in a['features']:
        receipt, v = read(checked(s['receipt'])), read(checked(s['validation']))
        need(receipt['payload'] == v['payload'] == s['payload'] and v['status'] == 'H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS', 'FEATURE_SEALS')
        for r in torch.load(checked(s['payload']), weights_only=True, map_location='cpu')['records']:
            need(r['query_id'] not in features, 'DUPLICATE_FEATURE')
            x = r['modes']['REAL']['X']
            need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6) and bool(torch.isfinite(x).all()), 'FP64_127_6')
            features[r['query_id']] = r
    need(len(predictions) == len(features) == len(result['rows']) == 593, 'ALL593')
    items, meta, retainers = {}, {}, {f: [] for f in range(5)}
    for r in sorted(result['rows'], key=lambda v: v['query_id']):
        q = r['query_id']; p = predictions[q]; f = features[q]
        for key in ('winner', 'challenger_positions', 'candidate_physical_rows'):
            need(p[key] == f[key], 'FEATURE_CANDIDATE_AXIS')
        axis, cs = p['candidate_physical_rows'], p['challenger_positions']
        positions = [j for j, physical in enumerate(axis) if labels[physical] == r['identity']]
        raw = labels[axis[p['winner']]] == r['identity']
        need(raw == r['correct']['RAW'] and bool(positions) == r['target_in_C128'], 'TARGET_MEMBERSHIP')
        if raw or not positions:
            continue
        need(len(positions) == 1, 'ONE_TARGET_FOR_RANKING_DIAGNOSTIC')
        target = cs.index(positions[0])
        z = [float.fromhex(v) for v in p['models']['COST1_FULL']['logits_hex']]
        top = max(range(127), key=z.__getitem__)
        items[q] = dict(query_id=q, x=f['modes']['REAL']['X'].tolist(), target_index=target)
        meta[q] = dict(query_id=q, original_query_id=r['original_query_id'], fold=r['fold'],
                       component=r['component'], identity=r['identity'], original_top_correct=top == target,
                       GAP_BIAS2_correct=r['correct']['GAP_BIAS2'], original_positive_SWITCH=z[top] > 0)
        if top == target:
            retainers[r['fold']].append(q)
    counts = dict(all=len(result['rows']), RAW_correct=sum(r['correct']['RAW'] for r in result['rows']),
                  target_present=sum(r['target_in_C128'] for r in result['rows']), ranking_population=len(items),
                  original_top_correct=sum(len(v) for v in retainers.values()),
                  ranking_blocked=sum(not r['original_top_correct'] for r in meta.values()))
    need(counts == a['expected_counts'], 'POPULATION_COUNTS')
    need(all(not r['GAP_BIAS2_correct'] for r in meta.values() if not r['original_top_correct']), 'ALL40_REMAIN_WRONG')
    return items, meta, retainers, base_theta


def summarize(meta, query, shared):
    blocked = [q for q in meta if not meta[q]['original_top_correct']]
    return dict(all144=dict(Counter(c['status'] for c in query.values())),
                blocked40=dict(Counter(query[q]['status'] for q in blocked)),
                shared40=dict(Counter(c['certificate']['status'] for c in shared.values())),
                cross40=dict(Counter(query[q]['status'] + '__' + shared[q]['certificate']['status'] for q in blocked)),
                by_fold={str(f): dict(query=dict(Counter(query[q]['status'] for q in meta if meta[q]['fold'] == f)),
                                     blocked=dict(Counter(query[q]['status'] for q in blocked if meta[q]['fold'] == f)),
                                     shared=dict(Counter(shared[q]['certificate']['status'] for q in blocked if meta[q]['fold'] == f))) for f in range(5)})


def run(a):
    from rc_aslo_xf.h593_rank_capacity_v1 import query_certificate, make_constraints, solve_constraints
    from validate_rc_opened_h593_rank_capacity_v1 import verify_query, verify_system
    start = time.monotonic()
    items, meta, retainers, base_theta = inputs(a)
    baseline = {}
    for fold in range(5):
        baseline[str(fold)] = verify_system([items[q] for q in retainers[fold]],
                                            dict(status='STRICT_FEASIBLE', theta_hex=base_theta[fold]))
    checkpoint = OUT / 'checkpoint.json'
    progress = read(checkpoint) if checkpoint.exists() else dict(authority=bind(AUTH), query={}, shared={})
    need(progress['authority'] == bind(AUTH), 'CHECKPOINT_AUTHORITY')
    for i, (q, item) in enumerate(items.items()):
        if q not in progress['query']:
            progress['query'][q] = query_certificate(item)
        verify_query(item['x'], item['target_index'], progress['query'][q])
        write(checkpoint, progress)
        if (i + 1) % 12 == 0:
            print(json.dumps(dict(stage='per_query', completed=i + 1, total=144, elapsed_s=round(time.monotonic()-start, 2))), flush=True)
    for i, q in enumerate(q for q in items if not meta[q]['original_top_correct']):
        retained = retainers[meta[q]['fold']]
        members = [items[k] for k in retained] + [items[q]]
        if q not in progress['shared']:
            exact, metadata = make_constraints(members)
            progress['shared'][q] = dict(retained_query_ids=retained, certificate=solve_constraints(exact, metadata))
        need(progress['shared'][q]['retained_query_ids'] == retained, 'SHARED_MEMBERS')
        verify_system(members, progress['shared'][q]['certificate'])
        write(checkpoint, progress)
        print(json.dumps(dict(stage='shared_extension', completed=i + 1, total=40,
                              status=progress['shared'][q]['certificate']['status'], elapsed_s=round(time.monotonic()-start, 2))), flush=True)
    summary = summarize(meta, progress['query'], progress['shared'])
    payload = dict(status='OPENED_H593_RANK_CAPACITY_CERTIFICATES_SEALED', authority=bind(AUTH),
                   source=a['parent_result'], metadata=meta, query_certificates=progress['query'],
                   shared_certificates=progress['shared'], baseline_exact_verifications=baseline,
                   summary=summary, model_fits=0, new_accuracy_results=False,
                   elapsed_s=time.monotonic()-start,
                   evidence_level='Opened H593 label-aware oracle capacity diagnostic; no learned model')
    write(OUT / 'result.json', payload)
    subprocess.run([sys.executable, __file__, 'verify'], check=True)
    print(json.dumps(dict(status='COMPLETE_AND_INDEPENDENTLY_VERIFIED', summary=summary)), flush=True)


def verify(a):
    from validate_rc_opened_h593_rank_capacity_v1 import verify_query, verify_system
    result = read(OUT / 'result.json')
    need(result['authority'] == bind(AUTH) and result['source'] == a['parent_result'], 'RESULT_AUTHORITY')
    items, meta, retainers, base_theta = inputs(a)
    need(result['metadata'] == meta, 'INDEPENDENT_INPUT_METADATA')
    baseline = {str(f): verify_system([items[q] for q in retainers[f]],
                                      dict(status='STRICT_FEASIBLE', theta_hex=base_theta[f])) for f in range(5)}
    need(baseline == result['baseline_exact_verifications'], 'ORIGINAL_WEIGHT_CERTIFICATES')
    need(set(result['query_certificates']) == set(items), '144_QUERY_COVERAGE')
    need(set(result['shared_certificates']) == {q for q in items if not meta[q]['original_top_correct']}, '40_SHARED_COVERAGE')
    checks = {}
    for q, item in items.items():
        checks[q] = verify_query(item['x'], item['target_index'], result['query_certificates'][q])
    shared_checks = {}
    for q, record in result['shared_certificates'].items():
        retained = retainers[meta[q]['fold']]
        need(record['retained_query_ids'] == retained, 'EXACT_SHARED_POPULATION')
        shared_checks[q] = verify_system([items[k] for k in retained] + [items[q]], record['certificate'])
    need(summarize(meta, result['query_certificates'], result['shared_certificates']) == result['summary'], 'INDEPENDENT_COUNTS')
    write(OUT / 'validation.json', dict(status='OPENED_H593_RANK_CAPACITY_EXACT_VALIDATION_PASS',
          authority=bind(AUTH), result=bind(OUT / 'result.json'), all_input_seals_checked=True,
          query_checks=checks, shared_checks=shared_checks, baseline_checks=baseline,
          unresolved_queries=sum(not v['proved'] for v in checks.values()),
          unresolved_shared=sum(not v['proved'] for v in shared_checks.values()), model_fits=0))


def publish(a):
    v = read(OUT / 'validation.json')
    need(v['status'] == 'OPENED_H593_RANK_CAPACITY_EXACT_VALIDATION_PASS' and v['authority'] == bind(AUTH), 'VALIDATED')
    r = read(checked(v['result']))
    rows = []
    for q, meta in r['metadata'].items():
        rows.append(dict(**meta, individual_status=r['query_certificates'][q]['status'],
                         shared_status=r['shared_certificates'][q]['certificate']['status'] if q in r['shared_certificates'] else 'BASELINE_RETAINED'))
    with (OUT / 'per_query.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# H593六项特征：候选排序容量定位', '',
             '本轮没有训练或评估新模型。原COST1=481、CE=486、GAP_BIAS2=492均保持；下表是已打开标签下的数学容量诊断，不是新准确率。', '',
             '全部593：426个RAW正确，23个RAW错误且target缺失，144个RAW错误且target在C128。对全部144检查完整127挑战者；其中104个原最高挑战者正确，40个错误。', '',
             '| 范围 | 精确判定 | 数量 |', '|---|---|---:|']
    for scope in ('all144', 'blocked40', 'shared40'):
        for status, count in sorted(r['summary'][scope].items()):
            lines.append(f'| {scope} | {status} | {count} |')
    lines += ['', '## 40个排序障碍的两层交叉', '', '| 单query与同折保留后扩展 | 数量 |', '|---|---:|']
    lines += [f'| {key} | {count} |' for key, count in sorted(r['summary']['cross40'].items())]
    lines += ['', 'LINEAR_SEPARABLE表示存在query专用六维线性方向使target严格第一；EXACT_CONVEX_HULL表示不能得到唯一线性第一，不排除平分，也不证明非线性无用。EXACT_FEATURE_COLLISION只限制六维pointwise评分。UNRESOLVED不作原因结论。', '',
              '共享扩展按原头所属折分别进行：保留本折原正确最高挑战者，再逐个添加一张当前错误。STRICT_FEASIBLE只证明该组排名可以由一个共享六维方向同时实现；EXACT_INFEASIBLE只证明保留该集合时有冲突。允许损失后的群体净增仍可能，保留条件不是晋级门。多个分别可行的扩展不能合并理解为可同时救回。', '',
              '所有结论由原FP64端点的Fraction精确运算复核，浮点solver仅找witness或凸组合。独立验证器在新进程重读全部输入SHA与证书，没有调用优化器。证书参数不得用于训练、初始化、阈值选择或预测成绩。', '',
              '现有证据仍不能仅凭“单例可分”断言统计接口信息充分，更不能据此证明跨组可学习性。需结合共享冲突与后续TRAIN-only学习读出判断。', '',
              '证据：`results/rc_opened_h593_rank_capacity_v1/result.json`、`validation.json`、`per_query.csv`。协议：`plan/RC_OPENED_H593_RANK_CAPACITY_V1_20260921.md`。', '']
    REPORT.write_text('\n'.join(lines))
    print(json.dumps(dict(status='PUBLISHED', report=str(REPORT), summary=r['summary'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'preflight', 'audit', 'verify', 'publish'))
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
    else:
        a = guard(stage)
        if stage == 'preflight':
            from rc_aslo_xf.h593_rank_capacity_v1 import _selftest as self_test
            from validate_rc_opened_h593_rank_capacity_v1 import self_test as independent_test
            write(OUT / 'preflight.json', dict(status='SYNTHETIC_CAPACITY_PASS', authority=bind(AUTH),
                  core=self_test(), independent=independent_test(), natural_solves=0))
            print('SYNTHETIC_CAPACITY_PASS', flush=True)
        elif stage == 'audit':
            run(a)
        elif stage == 'verify':
            verify(a)
        else:
            publish(a)
