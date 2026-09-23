#!/usr/bin/env python3
"""TRAIN-only conditional challenger ranking; no new HOLD/SWITCH action."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import run_rc_h593_gap_curve_v1 as U

ROOT = U.ROOT
sys.path.insert(0, str(ROOT / 'src'))
read, write, bind, checked, need, hx = U.read, U.write, U.bind, U.checked, U.need, U.hx
OUT = ROOT / 'results/rc_h593_conditional_rank_v1'
AUTH = ROOT / 'registry/rc_h593_conditional_rank_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_CONDITIONAL_RANK_V1_20260921.md'
REPORT = ROOT / 'reports/REPORT_H593_CONDITIONAL_RANK_V1_20260921.md'
BASE = ROOT / 'results/rc_six_cause_isolation_v1/loss_binding'
PARENT = ROOT / 'results/rc_h593_gap_curve_v1'
MODELS = ('COST1_FULL', 'CE_FULL', 'RANK_ONLY6')


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    prior_path = ROOT / 'registry/rc_h593_gap_curve_authority_v1_20260921.json'
    prior = read(prior_path)
    validated = read(PARENT / 'validation.json')
    need(validated['status'] == 'GAP_CURVE_ALL_COUNTS_PASS', 'PARENT_COMPLETE')
    checked(validated['result'])
    training_source = read(ROOT / 'registry/rc_h593_s_net_hold_optimization_authority_v1_20260920.json')
    codes = dict(program=Path(__file__), shared_io_loader=Path(U.__file__), plan=PLAN,
                 core=ROOT / 'src/rc_aslo_xf/h593_conditional_rank_v1.py',
                 independent=ROOT / 'programs/validate_rc_h593_conditional_rank_v1.py',
                 launcher=ROOT / 'slurm/rc_h593_conditional_rank_v1.sbatch')
    folds = {}
    for f in range(5):
        v = read(BASE / f'fold{f}/validation.json')
        need(v['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS', 'BASE_VALIDATED')
        checked(v['payload'])
        folds[str(f)] = dict(base_payload=v['payload'], base_validation=bind(BASE / f'fold{f}/validation.json'),
                            train_roles=prior['fold_sources'][str(f)]['train_roles'])
    public = dict(prior['public_sources'], worker=training_source['public_sources']['worker'])
    write(AUTH, dict(status='H593_CONDITIONAL_RANK_AUTHORIZED', code_sources={k: bind(v) for k, v in codes.items()},
          public_sources=public, features=prior['features'], fold_sources=folds,
          join_sources=dict(parent_result=validated['result'], parent_validation=bind(PARENT / 'validation.json'),
                            curator=prior['join_sources']['curator']), primary='RANK_ONLY6', primary_control='CE_FULL',
          secondary_control='COST1_FULL', parameter_count=6, bias=0, precision='float64',
          optimizer=dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000, initialization='zero'),
          objective='sum conditional127 CE on RAWwrong TRAIN divided by ALL recall-present TRAIN count',
          evidence_level='Opened H593 grouped fivefold development ranking-only experiment',
          action_calibration=False, deployment_change=False, encoder_forwards=0))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    for b in a['code_sources'].values():
        checked(b)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM_SHA')
    if stage not in ('preflight', 'publish'):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    for sources in a['features']:
        allow.update(Path(b['path']).resolve() for b in sources.values())
    if stage in ('fit', 'verify'):
        need(fold in range(5), 'FOLD_RANGE')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join', 'join-verify', 'publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for sources in a['fold_sources'].values():
            allow.update(Path(b['path']).resolve() for b in sources.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve(); s = str(path).lower()
        need(not any(t in s for t in ('rc_opened_', 'd1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/')), 'PROTECTED_READ')
        if 'curator_roles' in s:
            need(stage in ('join', 'join-verify', 'publish'), 'NO_OUTER_LABELS_BEFORE_SEALS')
        if ROOT / 'reports' in path.parents:
            need(stage == 'publish' and path == REPORT, 'NO_REPORT_ACCESS')
        if ROOT / 'results' in path.parents:
            own = OUT in path.parents
            if own and stage in ('fit', 'verify', 'preflight'):
                part = path.relative_to(OUT).parts[0]
                own = part in ('preflight.json', '.preflight.json.tmp', f'fold{fold}')
            need(own or path in allow, 'UNLISTED_RESULT:' + s)
    sys.addaudithook(audit)
    if stage != 'preflight':
        p = read(OUT / 'preflight.json')
        need(p['authority'] == bind(AUTH) and p['status'] == 'CONDITIONAL_RANK_SYNTHETIC_PASS', 'PREFLIGHT')
    return a


def features(a):
    rows = sorted(U.feature_rows(a).values(), key=lambda r: r['execution_ordinal'])
    worker = read(checked(a['public_sources']['worker']))['records']
    need(len(rows) == len(worker) == 593, 'ALL593')
    for r, w in zip(rows, worker):
        need(all(r[k] == w[k] for k in ('query_id', 'execution_ordinal', 'source_image_sha256')), 'WORKER_FEATURE_AXIS')
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    return rows, labels


def target(r, identity, labels):
    positions = [i for i, p in enumerate(r['candidate_physical_rows']) if labels[p] == identity]
    need(len(positions) <= 1, 'UNIQUE_TARGET_IDENTITY')
    if not positions:
        return -2
    return -1 if positions[0] == r['winner'] else r['challenger_positions'].index(positions[0])


def train_inputs(a, fold):
    import torch
    rows, labels = features(a)
    split = read(checked(a['public_sources']['split']))['folds'][fold]
    roles = {r['query_id']: r for r in read(checked(a['fold_sources'][str(fold)]['train_roles']))['records']}
    train_ids, held_ids = set(split['train_query_ids']), set(split['heldout_query_ids'])
    need(set(roles) == train_ids and not train_ids & held_ids, 'TRAIN_ONLY_ROLES')
    train = [r for r in rows if r['query_id'] in train_ids]
    held = [r for r in rows if r['query_id'] in held_ids]
    need(not {r['source_image_sha256'] for r in train} & {r['source_image_sha256'] for r in held}, 'OUTER_SOURCE_DISJOINT')
    keep = [r for r in train if target(r, roles[r['query_id']]['identity'], labels) >= -1]
    y = torch.tensor([target(r, roles[r['query_id']]['identity'], labels) for r in keep], dtype=torch.long)
    x = torch.stack([r['modes']['REAL']['X'] for r in keep])
    sources = a['fold_sources'][str(fold)]
    old = read(checked(sources['base_payload']))
    validation = read(checked(sources['base_validation']))
    need(validation['payload'] == sources['base_payload'] and validation['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS', 'BASE_SEALS')
    need(set(old['train_query_ids']) == train_ids, 'SAME_TRAIN_POPULATION')
    return train, held, keep, x, y, roles, labels, old


def compute(a, fold):
    import torch
    from rc_aslo_xf.h593_conditional_rank_v1 import fit_rank, loss_terms
    train, held, keep, x, y, roles, labels, old = train_inputs(a, fold)
    theta = fit_rank(x, y)
    oldp = {r['query_id']: r for r in old['predictions']}
    params = dict(RANK_ONLY6=hx(theta), CE_FULL=old['parameters']['ALL_CE'], COST1_FULL=old['parameters']['COST1'])
    train_metrics = {}
    for name, par in params.items():
        t = torch.tensor([float.fromhex(v) for v in par], dtype=torch.float64)
        z = x @ t[:6] + (t[6] if len(t) == 7 else 0.0)
        joint, rank, action = loss_terms(z, y)
        positive = y >= 0
        train_metrics[name] = dict(joint128_CE=float(joint), conditional_rank_loss_total_N=float(rank),
                                   action_loss=float(action), target_top=int(((z.argmax(1) == y) & positive).sum()))
    xx = torch.stack([r['modes']['REAL']['X'] for r in held])
    zs = {}
    for name, par in params.items():
        t = torch.tensor([float.fromhex(v) for v in par], dtype=torch.float64)
        zs[name] = xx @ t[:6] + (t[6] if len(t) == 7 else 0.0)
    predictions = []
    for i, r in enumerate(held):
        models = {}
        for name, z in zs.items():
            values = hx(z[i]); top = int(z[i].argmax())
            if name != 'RANK_ONLY6':
                prior = oldp[r['query_id']]['models']['ALL_CE' if name == 'CE_FULL' else 'COST1']
                need(values == prior['logits_hex'], 'ALL_CONTROL_LOGITS_BIT_PARITY')
            models[name] = dict(logits_hex=values, top_index=top,
                                top_physical=r['candidate_physical_rows'][r['challenger_positions'][top]])
        predictions.append(dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'],
                           winner=r['winner'], candidate_physical_rows=r['candidate_physical_rows'],
                           challenger_positions=r['challenger_positions'], models=models))
    positive_ids = [r['query_id'] for r, v in zip(keep, y) if int(v) >= 0]
    return dict(status='CONDITIONAL_RANK_FOLD_SEALED', authority=bind(AUTH), fold=fold, parameters=params,
                train_query_ids=[r['query_id'] for r in train], effective_train_query_ids=[r['query_id'] for r in keep],
                ranking_positive_query_ids=positive_ids, train_counts=dict(all=len(train), present=len(keep),
                ranking_positive=len(positive_ids), raw_correct=len(keep)-len(positive_ids), excluded_target_absent=len(train)-len(keep),
                positive_identities=len({roles[q]['identity'] for q in positive_ids}),
                positive_components=len({roles[q]['component'] for q in positive_ids})),
                train_metrics=train_metrics, predictions=predictions, heldout_label_reads=0,
                training_updates=2000, control_training_updates=0, action_trained=False, new_accuracy_result=False)


def fit(a, fold, replay=False):
    import torch
    from validate_rc_h593_conditional_rank_v1 import independent_loss_gradient, validate_logits
    folder = OUT / f'fold{fold}'
    if not replay and (folder / 'validation.json').exists():
        v = read(folder / 'validation.json')
        need(v['authority'] == bind(AUTH) and v['payload'] == bind(folder / 'payload.json'), 'RESUME_VALIDATED_FOLD')
        return
    started = time.monotonic()
    p = compute(a, fold)
    if not replay:
        write(folder / 'payload.json', p)
        write(folder / 'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'], fit_and_prediction_seconds=time.monotonic()-started))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
        print(json.dumps(dict(event='FOLD_VALIDATED', fold=fold, train=p['train_counts'])), flush=True)
        return
    need(p == read(folder / 'payload.json'), 'FRESH_FIT_PARAMETERS_PREDICTIONS_REPLAY')
    train, held, keep, x, y, roles, labels, old = train_inputs(a, fold)
    xx = torch.stack([r['modes']['REAL']['X'] for r in held]).numpy()
    checks = {}
    for name in MODELS:
        params = p['parameters'][name]
        if name == 'RANK_ONLY6':
            checks[name] = validate_logits(xx, params, [r['models'][name]['logits_hex'] for r in p['predictions']])
        else:
            t = np.array([float.fromhex(v) for v in params]); z = np.sum(xx*t[:6], axis=2)+t[6]
            want = np.array([[float.fromhex(v) for v in r['models'][name]['logits_hex']] for r in p['predictions']])
            error = float(np.max(abs(z-want)))
            need(error < 2e-10 and np.array_equal(z.argmax(1), want.argmax(1)), 'CONTROL_NUMPY_LOGITS')
            checks[name] = dict(logit_count=int(z.size), max_abs_error=error)
    theta = np.array([float.fromhex(v) for v in p['parameters']['RANK_ONLY6']])
    loss, gradient = independent_loss_gradient(theta, x.numpy(), y.numpy())
    from rc_aslo_xf.h593_conditional_rank_v1 import loss_terms
    t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    rank = loss_terms(x @ t, y)[1]
    tg = torch.autograd.grad(rank, t)[0].detach().numpy()
    rank_value = float(rank.detach())
    need(abs(rank_value-loss) < 2e-10 and float(np.max(abs(tg-gradient))) < 2e-10, 'INDEPENDENT_LOSS_GRADIENT')
    write(folder / 'validation.json', dict(status='CONDITIONAL_RANK_FRESH_REPLAY_NUMPY_PASS',
          authority=bind(AUTH), payload=bind(folder / 'payload.json'), heldout_label_reads=0,
          independent_logit_checks=checks, rank_loss_error=abs(rank_value-loss),
          rank_gradient_max_error=float(np.max(abs(tg-gradient)))))


def join(a, replay=False):
    payloads, seals = [], []
    for f in range(5):
        folder = OUT / f'fold{f}'
        v = read(folder / 'validation.json')
        need(v['status'] == 'CONDITIONAL_RANK_FRESH_REPLAY_NUMPY_PASS' and v['authority'] == bind(AUTH), 'ALL_FOLDS_VALIDATED')
        p = read(checked(v['payload'])); need(p['fold'] == f and p['authority'] == bind(AUTH), 'FOLD_BINDING')
        payloads.append(p); seals.append(bind(folder / 'validation.json'))
    write(OUT / 'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), validations=seals))
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    prior = read(checked(a['join_sources']['parent_result']))
    need(read(checked(a['join_sources']['parent_validation']))['result'] == a['join_sources']['parent_result'], 'PARENT_JOIN_SEAL')
    oldrows = {r['query_id']: r for r in prior['rows']}
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    rows = []
    for p in payloads:
        training_roles = read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records']
        for prediction in p['predictions']:
            q = prediction['query_id']; role = roles[q]
            need(role['outer_fold'] == p['fold'], 'OUTER_FOLD')
            for key in ('identity', 'component'):
                need(role[key] not in {r[key] for r in training_roles}, 'OUTER_GROUP_DISJOINT')
            axis = prediction['candidate_physical_rows']; identity = role['identity']
            raw = labels[axis[prediction['winner']]] == identity
            present = any(labels[v] == identity for v in axis)
            need(raw == oldrows[q]['correct']['RAW'] and present == oldrows[q]['target_in_C128'], 'BASE_MEMBERSHIP_PARITY')
            top, rank, selected = {}, {}, {}
            for name, model in prediction['models'].items():
                z = [float.fromhex(v) for v in model['logits_hex']]
                order = sorted(range(127), key=lambda j: (-z[j], j))
                physical = axis[prediction['challenger_positions'][order[0]]]
                need(physical == model['top_physical'] and order[0] == model['top_index'], 'TOP_RECOMPUTED')
                selected[name] = physical; top[name] = labels[physical] == identity
                rank[name] = next((i+1 for i,j in enumerate(order) if labels[axis[prediction['challenger_positions'][j]]] == identity), None)
            rows.append(dict(query_id=q, original_query_id=role['original_query_id'], component=role['component'], fold=p['fold'],
                             RAW_correct=raw, target_present=present, ranking_eligible=not raw and present,
                             target_is_top=top, target_challenger_rank=rank, top_physical=selected,
                             raw_or_top_oracle={m: raw or top[m] for m in MODELS}))
    rows.sort(key=lambda r:r['query_id'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'ALL593_JOIN')
    eligible = [r for r in rows if r['ranking_eligible']]
    need(len(eligible) == 144 and sum(r['RAW_correct'] for r in rows) == 426 and sum(r['target_present'] for r in rows) == 570, 'ALL_POPULATIONS')
    summary = {m:dict(target_top_among144=sum(r['target_is_top'][m] for r in eligible),
                      raw_or_top_oracle_among593=sum(r['raw_or_top_oracle'][m] for r in rows),
                      by_fold={str(f):sum(r['target_is_top'][m] for r in eligible if r['fold']==f) for f in range(5)}) for m in MODELS}
    for m in ('CE_FULL','COST1_FULL'):
        need(summary[m]['target_top_among144'] == 104 and summary[m]['raw_or_top_oracle_among593'] == 530, 'FROZEN_SORTING_BASELINES')
    comparisons = {}
    for baseline in ('CE_FULL','COST1_FULL'):
        groups = defaultdict(list)
        for r in rows:
            groups[r['component']].append(int(r['raw_or_top_oracle']['RANK_ONLY6'])-int(r['raw_or_top_oracle'][baseline]))
        need(len(groups) == 64, '64_COMPONENTS')
        d = np.array([np.mean(values) for _,values in sorted(groups.items())]); rng=np.random.default_rng(20260920)
        boot = d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
        gained=sum(r['target_is_top']['RANK_ONLY6'] and not r['target_is_top'][baseline] for r in eligible)
        lost=sum(not r['target_is_top']['RANK_ONLY6'] and r['target_is_top'][baseline] for r in eligible)
        comparisons[baseline] = dict(ranking_gained=gained, ranking_lost=lost, ranking_net=gained-lost,
               equal_component_oracle_difference=float(d.mean()), bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),
               primary_positive_internal_signal=gained>lost and d.mean()>0)
    output=dict(status='H593_CONDITIONAL_RANK_ALL593_COMPLETE', authority=bind(AUTH), summary=summary, comparisons=comparisons,
                train_counts={str(p['fold']):p['train_counts'] for p in payloads},
                train_metrics={str(p['fold']):p['train_metrics'] for p in payloads}, rows=rows,
                source=a['join_sources']['parent_result'], evidence_level='Opened H593 grouped OOF ranking-only diagnostic',
                new_deployed_accuracy=None, action_trained=False, external_GO=False,
                limits=['RAW-or-top is an oracle candidate availability count, not prediction accuracy.',
                        'Component bootstrap conditions on frozen predictions and does not include repeated-development selection.',
                        'All full593 deployment metrics remain those of the existing models until an independently calibrated action is tested.'])
    if replay:
        need(output == read(OUT / 'result.json'), 'FRESH_JOIN_EXACT_REPLAY')
        write(OUT / 'validation.json', dict(status='CONDITIONAL_RANK_ALL_COUNTS_PASS', authority=bind(AUTH), result=bind(OUT / 'result.json'), fold_validations=seals))
    else:
        write(OUT / 'result.json',output)
        subprocess.run([sys.executable,__file__,'join-verify'],check=True)
        print(json.dumps(dict(status='COMPLETE',summary=summary,comparisons=comparisons)),flush=True)


def publish():
    v=read(OUT/'validation.json');need(v['status']=='CONDITIONAL_RANK_ALL_COUNTS_PASS','RESULT_VALIDATED')
    r=read(checked(v['result']))
    lines=['# H593：纯条件候选排序结果','',
           '同六维特征、自然RAW C128、五折分组OOF。新RANK_ONLY6仅移除联合CE的行动项；不训练新HOLD/SWITCH门。以下是排序诊断，不是新最终准确率。','',
           '| 排序头 | 144例中的正确最高挑战者 | 全593的RAW或最高挑战者含正确答案（oracle） |','|---|---:|---:|']
    for m,s in r['summary'].items():lines.append(f"| {m} | {s['target_top_among144']} | {s['raw_or_top_oracle_among593']} |")
    lines+=['','| 相对对照 | 排序变正确 | 排序变错误 | 净增 | 组件等权差 | 95%区间 |','|---|---:|---:|---:|---:|---|']
    for m,s in r['comparisons'].items():lines.append(f"| {m} | {s['ranking_gained']} | {s['ranking_lost']} | {s['ranking_net']} | {s['equal_component_oracle_difference']:.6f} | {s['bootstrap95']} |")
    lines+=['','原最终成绩仍为COST1=481/593、CE=486/593、GAP_BIAS2=492/593。oracle列不能用作准确率，RANK_ONLY6分数正负不用于切换。', '',
            '条件排序loss仍除以完整recall-present TRAIN行数N；RAW正确行贡献0。各折有效样本和TRAIN损失/排名保存在result.json。40个旧排序错例没有被选作训练集；拟合进程禁止读取容量证书和外层标签。', '',
            '分组区间基于64组件的固定预测，不涵盖共享训练集或反复开发的选择不确定性。正结果只支持内部排序信号，负结果不证明信息完全缺失。', '',
            '结果及逐query：`results/rc_h593_conditional_rank_v1/result.json`；完整复算验证：`validation.json`。','']
    REPORT.write_text('\n'.join(lines));print(REPORT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','preflight','fit-all','fit','verify','join','join-verify','publish'));parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='fit-all':
        need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
        for f in range(5):subprocess.run([sys.executable,__file__,'fit','--fold',str(f)],check=True)
        subprocess.run([sys.executable,__file__,'join'],check=True)
    else:
        a=guard(args.stage,args.fold)
        if args.stage=='publish':publish()
        else:
            import torch
            torch.set_num_threads(8);torch.set_num_interop_threads(1)
            if args.stage=='preflight':
                from rc_aslo_xf.h593_conditional_rank_v1 import self_test
                from validate_rc_h593_conditional_rank_v1 import self_test as independent_test
                write(OUT/'preflight.json',dict(status='CONDITIONAL_RANK_SYNTHETIC_PASS',authority=bind(AUTH),core=self_test(),independent=independent_test(),natural_updates=0))
                print('CONDITIONAL_RANK_SYNTHETIC_PASS',flush=True)
            elif args.stage in ('fit','verify'):fit(a,args.fold,args.stage=='verify')
            else:join(a,args.stage=='join-verify')
