#!/usr/bin/env python3
"""Conditional competition evidence with a matched curvature control."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
OUT = ROOT / 'results/rc_h593_conditional_gap_v1'
AUTH = ROOT / 'registry/rc_h593_conditional_gap_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_CONDITIONAL_GAP_V1_20260921.md'
LAUNCH = ROOT / 'slurm/rc_h593_conditional_gap_v1.sbatch'
PARENT = ROOT / 'results/rc_h593_raw_incumbent_gate_v1'
ARMS = ('GAP_RAW_INTERACT4', 'GAP_CURVE4')


def need(condition, message):
    if not condition:
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


def checked(binding):
    need(bind(binding['path']) == binding, 'SHA256:' + binding['path'])
    return Path(binding['path'])


def hx(values):
    return [float(x).hex() for x in values]


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    parent_auth = ROOT / 'registry/rc_h593_raw_incumbent_gate_authority_v1_20260921.json'
    old = read(parent_auth)
    old_validation = read(PARENT / 'validation.json')
    need(old_validation['status'] == 'RAW_INCUMBENT_ALL_COUNTS_PASS' and
         old_validation['result'] == bind(PARENT / 'result.json'), 'PARENT_COMPLETE')
    source_paths = dict(program=Path(__file__), plan=PLAN, launcher=LAUNCH,
                        core=ROOT / 'src/rc_aslo_xf/h593_conditional_gap_v1.py',
                        shared_gate=ROOT / 'src/rc_aslo_xf/h593_s_bias_competition_v1.py',
                        incumbent_gate=ROOT / 'src/rc_aslo_xf/h593_raw_incumbent_gate_v1.py',
                        shared_independent=ROOT / 'programs/validate_rc_h593_s_bias_competition_v1.py',
                        feature_formula=ROOT / 'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
                        tests=ROOT / 'tests/test_h593_conditional_gap_v1.py',
                        independent_validator=ROOT / 'programs/validate_rc_h593_conditional_gap_v1.py')
    folds = {}
    for f in range(5):
        folder = PARENT / f'fold{f}'
        prior = old['fold_sources'][str(f)]
        folds[str(f)] = dict(parent_payload=bind(folder / 'payload.json'),
                            parent_validation=bind(folder / 'validation.json'),
                            parent_independent_validation=bind(folder / 'independent_validation.json'),
                            train_roles=prior['train_roles'],
                            inner_origin_payload=prior['inner_origin_payload'])
    write(AUTH, dict(status='H593_CONDITIONAL_GAP_AUTHORIZED', parent=bind(parent_auth),
                     code_sources={k: bind(v) for k, v in source_paths.items()},
                     public_sources=old['public_sources'], fold_sources=folds, features=old['features'],
                     join_sources=dict(parent_result=bind(PARENT / 'result.json'),
                                       parent_validation=bind(PARENT / 'validation.json'),
                                       curator=old['join_sources']['curator']),
                     arms=list(ARMS), primary='GAP_RAW_INTERACT4', primary_control='GAP_BIAS2',
                     structural_control='GAP_RAWNEAR3', capacity_control='GAP_CURVE4',
                     precision='float64', outer_folds=5, inner_heads_reused=20,
                     fit_rule='same logistic slopes and exact conditional net-gain bias; add one training-RMS-scaled basis r*d or d*d with a free coefficient',
                     no_encoder_forwards=True, heldout_label_reads_during_fit=0,
                     automatic_deployment_change=False, full_binary64_plane_optimum_claimed=False,
                     evidence_level='Opened H593 nested grouped development validation'))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    need(a['status'] == 'H593_CONDITIONAL_GAP_AUTHORIZED', 'AUTHORITY_STATUS')
    for binding in a['code_sources'].values():
        checked(binding)
    need(a['code_sources']['program'] == bind(__file__), 'PINNED_PROGRAM')
    if stage != 'preflight':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage in ('fit', 'verify'):
        for item in a['features']:
            allow.update(Path(b['path']).resolve() for b in item.values())
        need(fold in range(5), 'FOLD_RANGE')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join', 'join-verify'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for item in a['features']:
            allow.update(Path(b['path']).resolve() for b in item.values())
        for item in a['fold_sources'].values():
            allow.update(Path(b['path']).resolve() for b in item.values())

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        need(not any(t in s for t in ('d1-mi', 'd1_mi', 'formal392', '/grozi/',
                                     '/target_join/', 'gisc_prerecall_universe', '/reports/', 'rc_opened_')),
             'PROTECTED_READ')
        if 'curator_roles' in s:
            need(stage in ('join', 'join-verify'), 'LABELS_ONLY_AFTER_SEALS')
        if ROOT / 'results' in p.parents:
            own = OUT in p.parents
            if own and stage in ('fit', 'verify', 'preflight'):
                first = p.relative_to(OUT).parts[0]
                own = first in ('preflight.json', '.preflight.json.tmp', f'fold{fold}')
            need(own or p in allow, 'UNLISTED_RESULT:' + s)

    sys.addaudithook(audit)
    if stage != 'preflight':
        pf = read(OUT / 'preflight.json')
        need(pf['status'] == 'CONDITIONAL_GAP_SYNTHETIC_PASS' and pf['authority'] == bind(AUTH),
             'PREFLIGHT')
    return a


def top_gap(logits):
    need(len(logits) == 127 and all(np.isfinite(logits)), '127_FINITE_LOGITS')
    top = max(range(len(logits)), key=logits.__getitem__)
    second = max(logits[k] for k in range(len(logits)) if k != top)
    return top, float(logits[top] - second)


def select(logits, record):
    j = max(range(len(logits)), key=logits.__getitem__)
    pos = record['challenger_positions'][j] if logits[j] > 0 else record['winner']
    return record['candidate_physical_rows'][pos]


def parent_fold(a, fold):
    source = a['fold_sources'][str(fold)]
    v = read(checked(source['parent_validation']))
    iv = read(checked(source['parent_independent_validation']))
    need(v['status'] == 'RAW_INCUMBENT_FRESH_REPLAY_PASS' and
         iv['status'] == 'RAW_INCUMBENT_INDEPENDENT_PASS' and iv['passed'] and
         v['payload'] == iv['payload'] == source['parent_payload'] and
         v['authority'] == iv['authority'] == a['parent'], 'PARENT_FOLD_VALIDATED')
    p = read(checked(source['parent_payload']))
    need(p['fold'] == fold and p['authority'] == a['parent'] and p['heldout_label_reads'] == 0,
         'PARENT_FOLD_BINDING')
    origin = read(checked(source['inner_origin_payload']))
    parent_authority = read(checked(a['parent']))
    parent_sources = parent_authority['fold_sources'][str(fold)]
    need(p['parent_payload'] == parent_sources['parent_payload'] and
         source['inner_origin_payload'] == parent_sources['inner_origin_payload'], 'INNER_ORIGIN_BINDING')
    splits = read(checked(a['public_sources']['split']))['folds']
    train = set(splits[fold]['train_query_ids']); held = set(splits[fold]['heldout_query_ids'])
    roles = {r['query_id']: r for r in read(checked(source['train_roles']))['records']}
    need(set(roles) == train == set(p['train_query_ids']) and not train & held, 'OUTER_SPLIT')
    need({r['query_id'] for r in p['predictions']} == held, 'OUTER_COVERAGE')
    need({r['query_id'] for r in p['calibration']} == train and len(p['calibration']) == len(train), 'INNER_COVERAGE')
    for h in origin['inner_heads']:
        it, ih = set(h['train_query_ids']), set(h['heldout_query_ids'])
        need(not it & ih and it | ih == train and ih == set(splits[h['inner_fold']]['heldout_query_ids']), 'INNER_SPLIT')
        for key in ('identity', 'component'):
            need(not {roles[q][key] for q in it} & {roles[q][key] for q in ih}, 'INNER_GROUP_DISJOINT')
    return p


def feature_rows(a):
    import torch
    rows = {}
    for source in a['features']:
        rec, val = read(checked(source['receipt'])), read(checked(source['validation']))
        need(rec['payload'] == val['payload'] == source['payload'] and
             val['status'] == 'H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS', 'FEATURE_SEALS')
        for r in torch.load(checked(source['payload']), map_location='cpu', weights_only=True)['records']:
            need(r['query_id'] not in rows, 'UNIQUE_QUERY')
            x = r['modes']['REAL']['X']; raw = x[:, 0]
            need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6), 'FEATURE_SCHEMA')
            need(bool(torch.isfinite(raw).all()) and bool((raw <= 0).all()), 'RAW_NONPOSITIVE')
            need(torch.equal(raw.view(torch.int64), r['modes']['CBIND']['X'][:, 0].contiguous().view(torch.int64)), 'RAW_GEOMETRY_INDEPENDENT')
            rows[r['query_id']] = r
    need(len(rows) == 593, 'ALL_593_FEATURES')
    return rows


def raw_inputs(feature, top):
    values = feature['modes']['REAL']['X'][:, 0].tolist()
    nearest = max(range(127), key=values.__getitem__)
    return dict(r_pair=float(values[top]), r_near=float(values[nearest]), raw_nearest_index=nearest)


def projected(records, name):
    key = 'r_near'
    return [dict(r, h_raw=r[key]) for r in records]


def compute(a, fold):
    from rc_aslo_xf.h593_conditional_gap_v1 import fit_head, apply_head
    from rc_aslo_xf.h593_raw_incumbent_gate_v1 import fit_head as fit_old
    parent = parent_fold(a, fold)
    features = feature_rows(a)
    calibration = copy.deepcopy(parent['calibration'])
    for r in calibration:
        z = list(map(float.fromhex, r['logits_hex']))
        top, gap = top_gap(z)
        need(top == r['original_top'] and z[top].hex() == float(r['m']).hex(), 'CALIBRATION_TOP')
        need(gap.hex() == float(r['d']).hex(), 'CALIBRATION_GAP')
        inputs = raw_inputs(features[r['query_id']], top)
        need(all(r[k] == value for k, value in inputs.items()), 'SEALED_RAW_INPUTS')
    # Reproduce the incumbent under the identical optimizer and exact bias rule.
    control = fit_old([dict(r, h_raw=0.0) for r in calibration], 'GAP_BIAS2')
    old = parent['matched_control']
    for key in ('beta_hex','bias_hex','disabled','training_net_gain','training_rescues','training_breaks'):
        need(control[key] == old[key], 'MATCHED_CONTROL_' + key)
    need(control['optimization']['theta_hex'] == old['optimization']['theta_hex'], 'CONTROL_FIT_BITS')
    structural = fit_old(projected(calibration, 'GAP_RAWNEAR3'), 'GAP_RAWNEAR3')
    need(structural == parent['parameters']['GAP_RAWNEAR3'], 'STRUCTURAL_CONTROL_FULL_PARITY')
    parameters = {name:fit_head(projected(calibration, name), name) for name in ARMS}
    for name, h in parameters.items():
        print(json.dumps(dict(event='HEAD_FIT_DONE', fold=fold, arm=name, gamma=h['gamma'],
                              beta=h['beta'], eta=h['eta'], scale=h['scale'], bias=h['bias'], inner_net=h['training_net_gain'])), flush=True)
    predictions = copy.deepcopy(parent['predictions'])
    for p in predictions:
        feature = features[p['query_id']]
        for key in ('candidate_physical_rows','challenger_positions','winner'):
            need(p[key] == feature[key], 'FEATURE_AXIS_' + key)
        z = list(map(float.fromhex, p['models']['COST1_FULL']['logits_hex']))
        top, gap = top_gap(z)
        inputs = raw_inputs(feature, top)
        need(all(p[k] == value for k, value in inputs.items()), 'PREDICT_RAW_INPUT_PARITY')
        for name in ARMS:
            h = p['r_near']
            out = apply_head(z, h, gap, parameters[name])
            p['models'][name] = dict(logits_hex=hx(out), selected=select(out, p))
    return dict(status='CONDITIONAL_GAP_OUTER_SEALED',authority=bind(AUTH),fold=fold,
                parent_payload=a['fold_sources'][str(fold)]['parent_payload'],
                train_query_ids=parent['train_query_ids'],parameters=parameters,
                calibration=calibration,predictions=predictions, matched_control=control, structural_control=structural,
                gallery_mapping_sha256=parent['gallery_mapping_sha256'],
                heldout_label_reads=0,encoder_forwards=0,base_training_updates=0,inner_heads_reused=4)


def fit(a, fold, replay=False):
    folder = OUT / f'fold{fold}'
    if not replay and (folder / 'validation.json').exists():
        v = read(folder / 'validation.json')
        need(v['authority'] == bind(AUTH) and v['payload'] == bind(folder / 'payload.json'), 'RESUME_SEAL')
    else:
        start = time.perf_counter()
        p = compute(a, fold)
        if replay:
            need(p == read(folder / 'payload.json'), 'FRESH_FIT_EXACT_REPLAY')
            write(folder / 'validation.json', dict(status='CONDITIONAL_GAP_FRESH_REPLAY_PASS',
                                                  authority=bind(AUTH), payload=bind(folder / 'payload.json'),
                                                  heldout_label_reads=0))
            return
        write(folder / 'payload.json', p)
        write(folder / 'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-start))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'programs/validate_rc_h593_conditional_gap_v1.py'),
                    '--fold', str(fold)], check=True)


def groupstats(rows, base, new):
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    need(len(groups) == 64, 'ALL64_COMPONENTS')
    delta = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return dict(n_components=len(groups),
                rescue=sum(r['correct'][new] and not r['correct'][base] for r in rows),
                loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                changed=sum(r['selected'][new] != r['selected'][base] for r in rows),
                equal_component_difference=float(delta.mean()),
                bootstrap95=list(map(float, np.quantile(boot, [.025, .975]))))


def joined(a):
    ps, seals = [], []
    for f in range(5):
        folder = OUT / f'fold{f}'
        v, iv = read(folder / 'validation.json'), read(folder / 'independent_validation.json')
        need(v['status'] == 'CONDITIONAL_GAP_FRESH_REPLAY_PASS' and
             iv['status'] == 'CONDITIONAL_GAP_INDEPENDENT_PASS' and iv['passed'] and
             v['authority'] == iv['authority'] == bind(AUTH) and v['payload'] == iv['payload'], 'ALL_VALIDATORS_PASS')
        p = read(checked(v['payload']))
        need(p['authority'] == bind(AUTH) and p['fold'] == f, 'SEALED_FOLD')
        ps.append(p)
        seals += [bind(folder / 'validation.json'), bind(folder / 'independent_validation.json')]
    write(OUT / 'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), validations=seals))
    previous = read(checked(a['join_sources']['parent_result']))
    prior_validation = read(checked(a['join_sources']['parent_validation']))
    need(prior_validation['result'] == a['join_sources']['parent_result'], 'PARENT_RESULT_SEALED')
    old_rows = {r['query_id']: r for r in previous['rows']}
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    # Full RAW ranks are read only after the prediction seals, for MRR accounting.
    import torch
    raw_rows = {}
    for src in a['features']:
        receipt, validation = read(checked(src['receipt'])), read(checked(src['validation']))
        need(receipt['payload'] == validation['payload'] == src['payload'] and
             validation['status'] == 'H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS', 'FEATURE_SEALS')
        for r in torch.load(checked(src['payload']), map_location='cpu', weights_only=True)['records']:
            raw_rows[r['query_id']] = r['raw_ranked_physical_rows']
    rows = []
    for p in ps:
        train = [roles[q] for q in p['train_query_ids']]
        for pred in p['predictions']:
            q = pred['query_id']
            role = roles[q]
            need(role['outer_fold'] == p['fold'], 'OUTER_FOLD')
            for key in ('identity', 'component', 'source_image_sha256'):
                need(role[key] not in {t[key] for t in train}, 'OUTER_GROUP_IMAGE_DISJOINT')
            row = copy.deepcopy(old_rows[q])
            for name, model in pred['models'].items():
                s = model['selected']
                if name not in ARMS:
                    need(row['selected'][name] == s and row['correct'][name] == (labels[s] == role['identity']),
                         'PARENT_PREDICTION_PARITY')
                    continue
                row['selected'][name] = s
                row['correct'][name] = labels[s] == role['identity']
                ranked = raw_rows[q]
                tr = row['ranks']['RAW']
                row['ranks'][name] = 1 if row['correct'][name] else tr + int(ranked.index(s)+1 > tr)
            rows.append(row)
    rows.sort(key=lambda r: r['query_id'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'ALL593')
    need([sum(r['fold'] == f for r in rows) for f in range(5)] == [119,118,119,119,118], 'FOLD_SIZES')
    summary = {m: dict(correct=sum(r['correct'][m] for r in rows),
                       MRR=float(np.mean([1/r['ranks'][m] for r in rows])),
                       switches=sum(r['selected'][m] != r['selected']['RAW'] for r in rows),
                       fold_correct={str(f): sum(r['correct'][m] for r in rows if r['fold'] == f) for f in range(5)},
                       against_RAW=groupstats(rows, 'RAW', m), against_COST1=groupstats(rows, 'COST1_FULL', m))
               for m in rows[0]['correct']}
    need([summary[m]['correct'] for m in ('RAW','COST1_FULL','CE_FULL','ZERO_S','S_NET1')] ==
         [426,481,486,487,485], 'HISTORICAL_BASELINES')
    need(sum(not r['target_in_C128'] for r in rows) == 23, 'RECALL_MISSES_INCLUDED')
    need(all(r['selected']['BIAS1'] == r['selected']['GAP_NET1'] for r in rows),
         'BIAS_ONLY_REPRODUCES_PREVIOUS_CONSTANT_LIFT_ACTIONS')
    need(summary['GAP_BIAS2']['correct']==492, 'INCUMBENT492_PARITY')
    comparisons = {base+'__to__'+new: groupstats(rows,base,new)
                   for new in ARMS for base in ('GAP_BIAS2','GAP_RAWNEAR3','COST1_FULL','BIAS1','CE_FULL','ZERO_S')}
    comparisons['GAP_CURVE4__to__GAP_RAW_INTERACT4'] = groupstats(rows,'GAP_CURVE4','GAP_RAW_INTERACT4')
    need(summary['GAP_RAWNEAR3']['correct'] == 482 and summary['GAP_RAWPAIR3']['correct'] == 486, 'STRUCTURAL_HISTORY_PARITY')
    primary_vs_incumbent = comparisons['GAP_BIAS2__to__GAP_RAW_INTERACT4']
    primary_vs_curve = comparisons['GAP_CURVE4__to__GAP_RAW_INTERACT4']
    signals = dict(primary_net_vs492=primary_vs_incumbent['rescue']-primary_vs_incumbent['loss'],
                   primary_net_vs_curve=primary_vs_curve['rescue']-primary_vs_curve['loss'],
                   primary_count_improvement=summary['GAP_RAW_INTERACT4']['correct'] > 492,
                   interaction_development_signal=all(c['rescue'] > c['loss'] and
                       c['equal_component_difference'] > 0 for c in (primary_vs_incumbent, primary_vs_curve)),
                   both_descriptive_intervals_positive=all(c['bootstrap95'][0] > 0 for c in (primary_vs_incumbent, primary_vs_curve)),
                   formal_external_GO=False)
    return dict(status='H593_CONDITIONAL_GAP_COMPLETE',authority=bind(AUTH),population=593,
                primary='GAP_RAW_INTERACT4',primary_control='GAP_BIAS2',capacity_control='GAP_CURVE4',
                structural_control='GAP_RAWNEAR3',summary=summary,comparisons=comparisons,signals=signals,
                parameters={str(p['fold']):p['parameters'] for p in ps},rows=rows,
                evidence_level=a['evidence_level'],automatic_deployment_change=False,
                limits=['Opened development set; method choice informed by previous outer results.',
                        'Same logistic slope fit and exact conditional bias as GAP_BIAS2; no joint empirical optimum claim.',
                        'Original ranking and positive SWITCH scores are locked.',
                        'A conditional basis tests a restricted interaction; this does not establish universal geometry calibration.',
                        'Bootstrap fixes current OOF predictions and does not refit; it omits shared-training and adaptive-development uncertainty.'])


def join(a, replay=False):
    r = joined(a)
    if replay:
        need(r == read(OUT / 'result.json'), 'FRESH_JOIN_REPLAY')
        write(OUT / 'validation.json', dict(status='CONDITIONAL_GAP_ALL_COUNTS_PASS',
                                            authority=bind(AUTH), result=bind(OUT / 'result.json'), queries=593))
        return
    write(OUT / 'result.json', r)
    lines = ['# H593 条件化竞争读出：嵌套五折开发结果', '',
             '同一 COST1、自然 RAW C128、593 张分母；23 张 target 不在候选内。原排序及已有 SWITCH 冻结。', '',
             '| 模型 | 正确/593 | 对 COST1 救/损 | SWITCH |', '|---|---:|---:|---:|']
    for m, s in r['summary'].items():
        c = s['against_COST1']
        lines.append(f"| {m} | {s['correct']} | {c['rescue']}/{c['loss']} | {s['switches']} |")
    lines += ['', '主臂GAP_RAW_INTERACT4对GAP_BIAS2（492）；同四参数对照GAP_CURVE4；结构对照GAP_RAWNEAR3（482）。原COST1排序和正SWITCH冻结。',
              '新增一个训练HOLD行RMS缩放的交互项r*d或平方项d*d，新增系数自由；既有m/r/d、监督、损失与精确偏置校准不变。', '']
    for name, c in r['comparisons'].items():
        lines.append(f"- {name}: {c['rescue']} 救/{c['loss']} 损；等组差 {c['equal_component_difference']:.6f}，bootstrap95% {c['bootstrap95']}。")
    lines += ['', '预定信号核算：' + json.dumps(r['signals'], ensure_ascii=False),
              '区间仅描述当前开发数据，不校正历史反复选择。', '', 'H593 已打开；方法设计参考历史开发结果，本轮不构成新的外部确认，不自动替换部署。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, __file__, 'join-verify'], check=True)
    print(json.dumps(dict(status=r['status'], counts={m:s['correct'] for m,s in r['summary'].items()})), flush=True)


def preflight(a):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
    subprocess.run([sys.executable, str(ROOT / 'tests/test_h593_conditional_gap_v1.py')], env=env, check=True)
    write(OUT / 'preflight.json', dict(status='CONDITIONAL_GAP_SYNTHETIC_PASS', authority=bind(AUTH),
                                      natural_updates=0, scipy_version=__import__('scipy').__version__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','preflight','fit','verify','join','join-verify'])
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        a = guard(args.stage, args.fold)
        if args.stage == 'preflight':
            preflight(a)
        elif args.stage in ('join','join-verify'):
            join(a, args.stage == 'join-verify')
        else:
            fit(a, args.fold, args.stage == 'verify')
