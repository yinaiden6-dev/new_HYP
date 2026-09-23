#!/usr/bin/env python3
"""Frozen nested calibration of COST1 HOLD actions: S bias and competition."""
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
OUT = ROOT / 'results/rc_h593_s_bias_competition_v1'
AUTH = ROOT / 'registry/rc_h593_s_bias_competition_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_S_BIAS_COMPETITION_V1_20260921.md'
LAUNCH = ROOT / 'slurm/rc_h593_s_bias_competition_v1.sbatch'
PARENT = ROOT / 'results/rc_h593_s_net_hold_optimization_v1'
ARMS = ('BIAS1', 'S_FIXED_BIAS2', 'S_BIAS2', 'GAP_BIAS2', 'S_GAP3')


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
    parent_auth = ROOT / 'registry/rc_h593_s_net_hold_optimization_authority_v1_20260920.json'
    old = read(parent_auth)
    old_validation = read(PARENT / 'validation.json')
    need(old_validation['status'] == 'S_NET_ALL_FOLDS_ACTION_LABEL_COUNTS_PASS' and
         old_validation['result'] == bind(PARENT / 'result.json'), 'PARENT_COMPLETE')
    source_paths = dict(program=Path(__file__), plan=PLAN, launcher=LAUNCH,
                        core=ROOT / 'src/rc_aslo_xf/h593_s_bias_competition_v1.py',
                        tests=ROOT / 'tests/test_h593_s_bias_competition_v1.py',
                        independent_validator=ROOT / 'programs/validate_rc_h593_s_bias_competition_v1.py')
    folds = {}
    for f in range(5):
        folder = PARENT / f'fold{f}'
        folds[str(f)] = dict(parent_payload=bind(folder / 'payload.json'),
                            parent_validation=bind(folder / 'validation.json'),
                            parent_independent_validation=bind(folder / 'independent_validation.json'),
                            train_roles=old['fold_sources'][str(f)]['train_roles'])
    write(AUTH, dict(status='H593_S_BIAS_COMPETITION_AUTHORIZED', parent=bind(parent_auth),
                     code_sources={k: bind(v) for k, v in source_paths.items()},
                     public_sources={k: old['public_sources'][k] for k in ('split', 'gallery')},
                     fold_sources=folds, features=old['features'],
                     join_sources=dict(parent_result=bind(PARENT / 'result.json'),
                                       parent_validation=bind(PARENT / 'validation.json'),
                                       curator=old['join_sources']['curator']),
                     arms=list(ARMS), primary='S_GAP3', primary_control='S_BIAS2',
                     precision='float64', outer_folds=5, inner_heads_reused=20,
                     fit_rule='paired logistic plus L2 for slopes, then exact conditional net-gain bias',
                     l2=.001, maxiter=2000, ftol=1e-12, gtol=1e-8,
                     no_encoder_forwards=True, heldout_label_reads_during_fit=0,
                     automatic_deployment_change=False,
                     evidence_level='Opened H593 nested grouped development validation'))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    need(a['status'] == 'H593_S_BIAS_COMPETITION_AUTHORIZED', 'AUTHORITY_STATUS')
    for binding in a['code_sources'].values():
        checked(binding)
    need(a['code_sources']['program'] == bind(__file__), 'PINNED_PROGRAM')
    if stage != 'preflight':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage in ('fit', 'verify'):
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
        need(pf['status'] == 'S_BIAS_COMPETITION_SYNTHETIC_PASS' and pf['authority'] == bind(AUTH),
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
    need(v['status'] == 'S_NET_NESTED_FRESH_REPLAY_NUMPY_PASS' and
         iv['status'] == 'S_NET_INDEPENDENT_NESTED_EXHAUSTIVE_OPTIMALITY_PASS' and iv['passed'] and
         v['payload'] == iv['payload'] == source['parent_payload'] and
         v['authority'] == iv['authority'] == a['parent'], 'PARENT_FOLD_VALIDATED')
    p = read(checked(source['parent_payload']))
    need(p['fold'] == fold and p['authority'] == a['parent'] and p['heldout_label_reads'] == 0,
         'PARENT_FOLD_BINDING')
    splits = read(checked(a['public_sources']['split']))['folds']
    train = set(splits[fold]['train_query_ids'])
    held = set(splits[fold]['heldout_query_ids'])
    roles = {r['query_id']: r for r in read(checked(source['train_roles']))['records']}
    need(set(roles) == train == set(p['train_query_ids']) and not train & held, 'OUTER_SPLIT')
    need({r['query_id'] for r in p['predictions']} == held, 'OUTER_PREDICTION_COVERAGE')
    need({r['query_id'] for r in p['calibration']} == train and len(p['calibration']) == len(train),
         'INNER_CALIBRATION_COVERAGE')
    for h in p['inner_heads']:
        it, ih = set(h['train_query_ids']), set(h['heldout_query_ids'])
        need(not it & ih and it | ih == train and ih == set(splits[h['inner_fold']]['heldout_query_ids']),
             'INNER_SPLIT')
        for key in ('identity', 'component'):
            need(not {roles[q][key] for q in it} & {roles[q][key] for q in ih}, 'INNER_GROUP_DISJOINT')
    return p


def compute(a, fold):
    from rc_aslo_xf.h593_s_bias_competition_v1 import fit_head, calibrate_bias, apply_head
    parent = parent_fold(a, fold)
    calibration = copy.deepcopy(parent['calibration'])
    for r in calibration:
        z = list(map(float.fromhex, r['logits_hex']))
        top, gap = top_gap(z)
        need(top == r['original_top'] and z[top].hex() == float(r['m']).hex(), 'CALIBRATION_TOP_BITS')
        need(r['delta'] == int(r['top_correct']) - int(r['raw_correct']), 'CALIBRATION_DELTA')
        r['d'] = gap
    parameters = {}
    for name in ARMS:
        if name == 'S_FIXED_BIAS2':
            alpha = float.fromhex(parent['parameters']['S_NET1']['alpha_hex'])
            parameters[name] = calibrate_bias(calibration, alpha, 0.)
            parameters[name]['fixed_s_source'] = dict(model='S_NET1', alpha_hex=alpha.hex(),
                                                     parent_payload=a['fold_sources'][str(fold)]['parent_payload'])
        else:
            parameters[name] = fit_head(calibration, name)
        print(json.dumps(dict(event='HEAD_FIT_DONE', fold=fold, arm=name,
                              disabled=parameters[name]['disabled'],
                              alpha=parameters[name]['alpha_hex'], beta=parameters[name]['beta_hex'],
                              bias=parameters[name]['bias_hex'])), flush=True)
    predictions = copy.deepcopy(parent['predictions'])
    for p in predictions:
        z = list(map(float.fromhex, p['models']['COST1_FULL']['logits_hex']))
        top, gap = top_gap(z)
        need(top == p['original_top'], 'OUTER_TOP')
        p['d'] = gap
        for name in ARMS:
            out = apply_head(z, p['h_s'], gap, parameters[name])
            p['models'][name] = dict(logits_hex=hx(out), selected=select(out, p))
    return dict(status='S_BIAS_COMPETITION_OUTER_SEALED', authority=bind(AUTH), fold=fold,
                parent_payload=a['fold_sources'][str(fold)]['parent_payload'],
                train_query_ids=parent['train_query_ids'], parameters=parameters,
                calibration=calibration, predictions=predictions,
                gallery_mapping_sha256=parent['gallery_mapping_sha256'],
                heldout_label_reads=0, encoder_forwards=0, base_training_updates=0,
                inner_heads_reused=4)


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
            write(folder / 'validation.json', dict(status='S_BIAS_COMPETITION_FRESH_REPLAY_PASS',
                                                  authority=bind(AUTH), payload=bind(folder / 'payload.json'),
                                                  heldout_label_reads=0))
            return
        write(folder / 'payload.json', p)
        write(folder / 'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-start))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'programs/validate_rc_h593_s_bias_competition_v1.py'),
                    '--fold', str(fold)], check=True)


def groupstats(rows, base, new):
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    delta = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return dict(rescue=sum(r['correct'][new] and not r['correct'][base] for r in rows),
                loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                changed=sum(r['selected'][new] != r['selected'][base] for r in rows),
                equal_component_difference=float(delta.mean()),
                bootstrap95=list(map(float, np.quantile(boot, [.025, .975]))))


def joined(a):
    ps, seals = [], []
    for f in range(5):
        folder = OUT / f'fold{f}'
        v, iv = read(folder / 'validation.json'), read(folder / 'independent_validation.json')
        need(v['status'] == 'S_BIAS_COMPETITION_FRESH_REPLAY_PASS' and
             iv['status'] == 'S_BIAS_COMPETITION_INDEPENDENT_PASS' and iv['passed'] and
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
    comparisons = {base+'__to__'+new: groupstats(rows, base, new)
                   for base, new in [('S_BIAS2','S_GAP3'), ('GAP_BIAS2','S_GAP3'),
                                     ('S_NET1','S_FIXED_BIAS2'), ('S_FIXED_BIAS2','S_BIAS2'),
                                     ('S_NET1','S_GAP3'), ('CE_FULL','S_GAP3'), ('ZERO_S','S_GAP3')]}
    return dict(status='H593_S_BIAS_COMPETITION_COMPLETE', authority=bind(AUTH), population=593,
                primary='S_GAP3', primary_control='S_BIAS2', summary=summary, comparisons=comparisons,
                bias_only_action_parity_with_GAP_NET1=True,
                parameters={str(p['fold']): p['parameters'] for p in ps}, rows=rows,
                evidence_level=a['evidence_level'], automatic_deployment_change=False,
                limits=['Opened development set; not independent external confirmation.',
                        'Exact bias optimum is conditional on slopes; slopes use a convex logistic surrogate.',
                        'Original top challenger and all positive SWITCH actions are locked.',
                        'S_FIXED_BIAS2 isolates bias from the changed slope training rule.',
                        'Inner three-fold to outer four-fold score-scale transport is not guaranteed.'])


def join(a, replay=False):
    r = joined(a)
    if replay:
        need(r == read(OUT / 'result.json'), 'FRESH_JOIN_REPLAY')
        write(OUT / 'validation.json', dict(status='S_BIAS_COMPETITION_ALL_COUNTS_PASS',
                                            authority=bind(AUTH), result=bind(OUT / 'result.json'), queries=593))
        return
    write(OUT / 'result.json', r)
    lines = ['# H593 S 偏置与候选竞争：嵌套五折开发结果', '',
             '同一 COST1、自然 RAW C128、593 张分母；23 张 target 不在候选内。原排序及已有 SWITCH 冻结。', '',
             '| 模型 | 正确/593 | 对 COST1 救/损 | SWITCH |', '|---|---:|---:|---:|']
    for m, s in r['summary'].items():
        c = s['against_COST1']
        lines.append(f"| {m} | {s['correct']} | {c['rescue']}/{c['loss']} | {s['switches']} |")
    lines += ['', '主比较：S_GAP3 对 S_BIAS2。S_FIXED_BIAS2 固定上轮 S 系数、仅拟合偏置。',
              '斜率采用固定正则化 logistic 目标；随后偏置精确优化内层净增。不是三参数净增的联合全局最优。', '']
    for name, c in r['comparisons'].items():
        lines.append(f"- {name}: {c['rescue']} 救/{c['loss']} 损；等组差 {c['equal_component_difference']:.6f}，bootstrap95% {c['bootstrap95']}。")
    lines += ['', 'H593 已打开；方法设计参考历史开发结果，本轮不构成新的外部确认，不自动替换部署。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, __file__, 'join-verify'], check=True)
    print(json.dumps(dict(status=r['status'], counts={m:s['correct'] for m,s in r['summary'].items()})), flush=True)


def preflight(a):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
    subprocess.run([sys.executable, str(ROOT / 'tests/test_h593_s_bias_competition_v1.py')], env=env, check=True)
    write(OUT / 'preflight.json', dict(status='S_BIAS_COMPETITION_SYNTHETIC_PASS', authority=bind(AUTH),
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
