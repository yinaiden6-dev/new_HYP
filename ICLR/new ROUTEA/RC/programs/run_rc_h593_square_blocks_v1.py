#!/usr/bin/env python3
"""Prespecified square-block refits under original full CE; frozen full13 control."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import run_rc_h593_conditional_rank_v1 as C
import run_rc_h593_rank_action_v1 as R

ROOT = C.ROOT
read, write, bind, checked, need, hx = C.read, C.write, C.bind, C.checked, C.need, C.hx
OUT = ROOT / 'results/rc_h593_square_blocks_v1'
AUTH = ROOT / 'registry/rc_h593_square_blocks_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_SQUARE_BLOCKS_V1_20260921.md'
REPORT = ROOT / 'reports/REPORT_H593_SQUARE_BLOCKS_V1_20260921.md'
ARMS = ('JOINT_SQUARE_CE12', 'RAW_SQUARE_CE8')
CONTROLS = ('COST1_FULL', 'CE_FULL', 'GAP_BIAS2', 'DIAG_CE13')
MODELS = ('RAW',) + CONTROLS + ARMS


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    parent_auth = ROOT/'registry/rc_h593_joint_diagonal_authority_v1_20260921.json'
    old = read(parent_auth)
    base = ROOT/'results/rc_h593_joint_diagonal_v1'
    v = read(base/'validation.json')
    need(v['status']=='JOINT_DIAGONAL_ALL_COUNTS_PASS' and v['authority']==bind(parent_auth),'PARENT_VALIDATED')
    checked(v['result'])
    codes = dict(old['code_sources'])
    for name, path in dict(program=Path(__file__), plan=PLAN,
        core=ROOT/'src/rc_aslo_xf/h593_square_blocks_v1.py',
        independent=ROOT/'programs/validate_rc_h593_square_blocks_v1.py',
        parent_core=ROOT/'src/rc_aslo_xf/h593_joint_diagonal_v1.py',
        parent_validator=ROOT/'programs/validate_rc_h593_joint_diagonal_v1.py',
        launcher=ROOT/'slurm/rc_h593_square_blocks_v1.sbatch').items():
        codes[name]=bind(path)
    for b in codes.values(): checked(b)
    folds = {}
    for f in range(5):
        fv=read(base/f'fold{f}/validation.json')
        need(fv['status']=='JOINT_DIAGONAL_FRESH_NUMPY_PASS' and fv['authority']==bind(parent_auth),'PARENT_FOLD_SEAL')
        folds[str(f)]=dict(old['fold_sources'][str(f)],
            diagonal_payload=fv['payload'],diagonal_validation=bind(base/f'fold{f}/validation.json'))
        for b in folds[str(f)].values(): checked(b)
    write(AUTH,dict(status='H593_SQUARE_BLOCKS_AUTHORIZED',parent_authority=bind(parent_auth),
        code_sources=codes,public_sources=old['public_sources'],features=old['features'],fold_sources=folds,
        join_sources=dict(parent_result=v['result'],parent_validation=bind(base/'validation.json'),curator=old['join_sources']['curator']),
        arms=list(ARMS),primary='JOINT_SQUARE_CE12',primary_control='RAW_SQUARE_CE8',
        strong_control='DIAG_CE13',frozen_controls=list(CONTROLS),storage_coordinates=13,
        effective_parameters=dict(JOINT_SQUARE_CE12=12,RAW_SQUARE_CE8=8),
        feature_masks=dict(JOINT_SQUARE_CE12_disabled=[6],RAW_SQUARE_CE8_disabled=[7,8,9,10,11]),
        optimizer=dict(name='AdamW',lr=.03,weight_decay=.001,steps=2000,initialization='zero'),
        objective='Original FULL128 CE, HOLD logit=0',gate='max challenger >0 SWITCH else HOLD',
        joint_curvature_signal='Primary beats CE_FULL and RAW_SQUARE_CE8 in count and component mean',
        performance_signal='Primary beats DIAG_CE13 and GAP_BIAS2 in count and component mean',
        evidence_level='Opened H593 grouped fivefold exploratory square-block refits',
        external_GO=False,deployment_change=False,encoder_forwards=0))
    print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    checked(a['parent_authority'])
    for b in a['code_sources'].values(): checked(b)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM_SHA')
    if stage not in ('preflight', 'publish'):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bs in a['features']: allow.update(Path(b['path']).resolve() for b in bs.values())
    if stage in ('fit', 'verify'):
        need(fold in range(5), 'FOLD_RANGE')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join', 'join-verify', 'publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for bs in a['fold_sources'].values(): allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        need(not any(x in str(p).lower() for x in ('rc_opened_', 'd1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/')), 'PROTECTED_READ')
        if 'curator_roles' in str(p): need(stage in ('join', 'join-verify', 'publish'), 'NO_HELD_LABELS')
        if ROOT/'reports' in p.parents: need(stage == 'publish' and p == REPORT, 'NO_REPORT_READ')
        if ROOT/'results' in p.parents:
            own = OUT in p.parents
            if own and stage in ('preflight', 'fit', 'verify'):
                own = p.relative_to(OUT).parts[0] in ('preflight.json', '.preflight.json.tmp', f'fold{fold}')
            need(own or p in allow, 'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    if stage != 'preflight':
        v = read(OUT/'preflight.json')
        need(v['status'] == 'SQUARE_BLOCKS_SYNTHETIC_PASS' and v['authority'] == bind(AUTH), 'PREFLIGHT')
    return a


def compute(a, fold):
    import torch
    from rc_aslo_xf.h593_square_blocks_v1 import expand, fit, objective
    train, held, keep, x, y, roles, labels, old = C.train_inputs(a, fold)
    source = a['fold_sources'][str(fold)]
    gv = read(checked(source['gap_validation']))
    need(gv['status'] == 'GAP_CURVE_FRESH_REPLAY_PASS' and gv['payload'] == source['gap_payload'], 'GAP_SEAL')
    gap = read(checked(source['gap_payload']))
    need(set(gap['train_query_ids']) == {r['query_id'] for r in train}, 'GAP_SAME_TRAIN')
    gp = {r['query_id']: r for r in gap['predictions']}
    op = {r['query_id']: r for r in old['predictions']}
    parameters = {m: hx(fit(x, y, m)) for m in ARMS}
    parameters.update(COST1_FULL=old['parameters']['COST1'], CE_FULL=old['parameters']['ALL_CE'])
    dv=read(checked(source['diagonal_validation']))
    need(dv['status']=='JOINT_DIAGONAL_FRESH_NUMPY_PASS' and dv['payload']==source['diagonal_payload'] and dv['authority']==a['parent_authority'],'DIAG_SOURCE_SEAL')
    diagonal=read(checked(source['diagonal_payload']))
    need(set(diagonal['train_query_ids'])=={r['query_id'] for r in train},'DIAG_SAME_TRAIN')
    parameters['DIAG_CE13']=diagonal['parameters']['DIAG_CE13']
    dp={r['query_id']:r for r in diagonal['predictions']}
    xx = torch.stack([r['modes']['REAL']['X'] for r in held])
    held_phi, train_phi = expand(xx), expand(x)
    scores, metrics = {}, {}
    for m, par in parameters.items():
        t = torch.tensor([float.fromhex(v) for v in par], dtype=torch.float64)
        ph, pt = (held_phi, train_phi) if len(par)==13 else (xx, x)
        scores[m] = ph @ t[:-1] + t[-1]
        z = pt @ t[:-1] + t[-1]
        metrics[m] = dict(CE=float(objective(z, y, 'DIAG_CE13')),
            COST1=float(objective(z, y, 'DIAG_COST113')),
            actual_correct=int(((torch.where(z.max(1).values > 0, z.argmax(1), -1)) == y).sum()),
            target_top_RAWwrong=int(((z.argmax(1) == y) & (y >= 0)).sum()))
    predictions = []
    for i, r in enumerate(held):
        q = r['query_id']
        for k in ('winner', 'challenger_positions', 'candidate_physical_rows', 'execution_ordinal'):
            need(r[k] == gp[q][k], 'CONTROL_AXIS')
        models = {'GAP_BIAS2': gp[q]['models']['GAP_BIAS2']}
        for m, zs in scores.items():
            z = zs[i]
            top = int(z.argmax())
            pos = r['challenger_positions'][top] if z[top] > 0 else r['winner']
            models[m] = dict(logits_hex=hx(z), top_index=top, selected=r['candidate_physical_rows'][pos])
            if m in ('COST1_FULL', 'CE_FULL'):
                legacy = op[q]['models']['COST1' if m == 'COST1_FULL' else 'ALL_CE']
                need(models[m]['logits_hex'] == legacy['logits_hex'] == gp[q]['models'][m]['logits_hex'], 'ALL_CONTROL_LOGITS_BIT_PARITY')
                need(models[m]['selected'] == legacy['selected'] == gp[q]['models'][m]['selected'], 'OLD_ACTION_PARITY')
        for k in ('winner','challenger_positions','candidate_physical_rows','execution_ordinal'):
            need(r[k]==dp[q][k],'DIAG_CONTROL_AXIS')
        need(models['DIAG_CE13']==dp[q]['models']['DIAG_CE13'],'DIAG_FULL_PARAMETER_AND_LOGIT_PARITY')
        predictions.append(dict(query_id=q, execution_ordinal=r['execution_ordinal'], winner=r['winner'],
            candidate_physical_rows=r['candidate_physical_rows'], challenger_positions=r['challenger_positions'], models=models))
    return dict(status='SQUARE_BLOCKS_FOLD_SEALED', authority=bind(AUTH), fold=fold,
        parameters=parameters, predictions=predictions, train_metrics=metrics,
        train_query_ids=[r['query_id'] for r in train], effective_train_query_ids=[r['query_id'] for r in keep],
        train_counts=dict(total=len(train), present=len(keep), absent=len(train)-len(keep), raw_correct=int((y==-1).sum())),
        heldout_label_reads=0, new_fits=2, updates_per_fit=2000, old_control_updates=0)


def fit(a, fold, replay=False):
    import torch
    from rc_aslo_xf.h593_square_blocks_v1 import expand, objective, training_loss, active_mask
    from validate_rc_h593_square_blocks_v1 import values_gradient, logits_check
    folder = OUT/f'fold{fold}'
    if not replay and (folder/'validation.json').exists():
        v = read(folder/'validation.json')
        need(v['status'] == 'SQUARE_BLOCKS_FRESH_NUMPY_PASS' and v['authority'] == bind(AUTH) and v['payload'] == bind(folder/'payload.json'), 'RESUME_SEAL')
        return
    started = time.monotonic()
    p = compute(a, fold)
    if not replay:
        write(folder/'payload.json', p)
        write(folder/'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'], fit_seconds=time.monotonic()-started))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
        print(dict(event='FOLD_VALIDATED', fold=fold), flush=True)
        return
    need(p == read(folder/'payload.json'), 'FRESH_FIT_BIT_EXACT')
    train, held, keep, x, y, roles, labels, old = C.train_inputs(a, fold)
    xx = torch.stack([r['modes']['REAL']['X'] for r in held]).numpy()
    checks = {m: logits_check(xx, par, p['predictions'], m) for m, par in p['parameters'].items()}
    gradients = {}
    for m in ARMS:
        t = torch.tensor([float.fromhex(v) for v in p['parameters'][m]], dtype=torch.float64, requires_grad=True)
        need(bool((t.detach()[active_mask(m)==0]==0).all()),'DISABLED_COORDINATES_EXACT_ZERO')
        loss = training_loss(t, expand(x), y, m)
        grad = torch.autograd.grad(loss, t)[0].numpy()
        lv, gv = values_gradient(t.detach().numpy(), x.numpy(), y.numpy(), m)
        le, ge = abs(float(loss.detach())-lv), float(np.max(abs(grad-gv)))
        need(le < 2e-10 and ge < 2e-10, 'INDEPENDENT_OBJECTIVE_GRADIENT')
        gradients[m] = dict(loss_error=le, gradient_error=ge)
    write(folder/'validation.json', dict(status='SQUARE_BLOCKS_FRESH_NUMPY_PASS', authority=bind(AUTH),
        payload=bind(folder/'payload.json'), logits=checks, gradients=gradients,
        heldout_label_reads=0, fresh_refits=2))


def join(a, replay=False):
    ps, seals = [], []
    for f in range(5):
        v = read(OUT/f'fold{f}/validation.json')
        need(v['status'] == 'SQUARE_BLOCKS_FRESH_NUMPY_PASS' and v['authority'] == bind(AUTH), 'FIVE_SEALS')
        p = read(checked(v['payload']))
        need(p['fold'] == f and p['authority'] == bind(AUTH), 'FOLD_SEAL')
        ps.append(p); seals.append(bind(OUT/f'fold{f}/validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), validations=seals))
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    worker = {r['query_id']: r for r in read(checked(a['public_sources']['worker']))['records']}
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    pv = read(checked(a['join_sources']['parent_validation']))
    need(pv['result'] == a['join_sources']['parent_result'], 'PARENT_JOIN')
    prior = {r['query_id']: r for r in read(checked(pv['result']))['rows']}
    rows = []
    for p in ps:
        training = read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records']
        need({r['query_id'] for r in training} == set(p['train_query_ids']), 'TRAIN_IDS')
        for pred in p['predictions']:
            q = pred['query_id']; role = roles[q]
            need(role['outer_fold'] == p['fold'], 'FOLD_ASSIGNMENT')
            for k in ('identity', 'component'): need(role[k] not in {r[k] for r in training}, 'GROUP_DISJOINT')
            need(worker[q]['source_image_sha256'] not in {worker[r['query_id']]['source_image_sha256'] for r in training}, 'IMAGE_DISJOINT')
            axis = pred['candidate_physical_rows']
            selected = dict(RAW=axis[pred['winner']], **{m: pred['models'][m]['selected'] for m in CONTROLS+ARMS})
            correct = {m: labels[v] == role['identity'] for m,v in selected.items()}
            present = any(labels[v] == role['identity'] for v in axis)
            need(present == prior[q]['target_in_C128'], 'PRESENT_PARITY')
            for m in ('RAW',)+CONTROLS:
                need(selected[m] == prior[q]['selected'][m] and correct[m] == prior[q]['correct'][m], 'BASELINE_UNCHANGED')
            top_correct = {}
            for m in ('COST1_FULL', 'CE_FULL', 'DIAG_CE13')+ARMS:
                z = np.array([float.fromhex(v) for v in pred['models'][m]['logits_hex']])
                top = int(z.argmax()); pos = pred['challenger_positions'][top]
                top_correct[m] = labels[axis[pos]] == role['identity']
                action_pos = pos if z[top] > 0 else pred['winner']
                need(axis[action_pos] == selected[m], 'JOIN_ACTUAL_ACTION')
            rows.append(dict(query_id=q, original_query_id=role['original_query_id'], component=role['component'], fold=p['fold'],
                target_in_C128=present, selected=selected, correct=correct, top_correct=top_correct,
                original40_ranking_blocked=prior[q]['original40_ranking_blocked']))
    rows.sort(key=lambda r:r['query_id'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593 and sum(r['target_in_C128'] for r in rows) == 570, '593_RECALL570')
    summary = {}
    for m in MODELS:
        rescue = sum(r['correct'][m] and not r['correct']['RAW'] for r in rows)
        loss = sum(not r['correct'][m] and r['correct']['RAW'] for r in rows)
        summary[m] = dict(correct=sum(r['correct'][m] for r in rows), total=593, rescue_vs_RAW=rescue,
            break_vs_RAW=loss, net_vs_RAW=rescue-loss, raw_correct_loss_rate=loss/426,
            switches=sum(r['selected'][m] != r['selected']['RAW'] for r in rows),
            by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)},
            old40_correct=sum(r['correct'][m] for r in rows if r['original40_ranking_blocked']))
        if m in ('COST1_FULL','CE_FULL','DIAG_CE13')+ARMS:
            summary[m]['target_top_of_144'] = sum(r['top_correct'][m] for r in rows if r['target_in_C128'] and not r['correct']['RAW'])
        need(summary[m]['correct'] == 426+rescue-loss, 'RESCUE_LOSS_ACCOUNTING')
    for m,n in [('RAW',426),('COST1_FULL',481),('CE_FULL',486),('GAP_BIAS2',492),('DIAG_CE13',493)]: need(summary[m]['correct']==n,'BASELINE_COUNTS')
    comparisons = {m:{b:R.compare(rows,b,m) for b in ('RAW',)+CONTROLS} for m in ARMS}
    signal = all(comparisons['JOINT_SQUARE_CE12'][b]['net']>0 and comparisons['JOINT_SQUARE_CE12'][b]['equal_component_difference']>0 for b in ('DIAG_CE13','GAP_BIAS2'))
    block_comparison=R.compare(rows,'RAW_SQUARE_CE8','JOINT_SQUARE_CE12')
    curvature=block_comparison['net']>0 and block_comparison['equal_component_difference']>0 and comparisons['JOINT_SQUARE_CE12']['CE_FULL']['net']>0 and comparisons['JOINT_SQUARE_CE12']['CE_FULL']['equal_component_difference']>0
    output = dict(status='H593_SQUARE_BLOCKS_ALL593_COMPLETE', authority=bind(AUTH), primary='JOINT_SQUARE_CE12',
        summary=summary, comparisons=comparisons, rows=rows, primary_internal_performance_signal=signal,
        joint_curvature_signal=curvature, block_comparison=block_comparison,
        evidence_level=a['evidence_level'], external_GO=False, automatic_deployment_change=False,
        training='FULL target-present TRAIN only, same loss and optimizer; full593 heldout denominator',
        intervals='Conditional on opened OOF predictions; not corrected for repeated development or overlapping TRAIN')
    if replay:
        need(output == read(OUT/'result.json'), 'INDEPENDENT_JOIN_REPLAY')
        write(OUT/'validation.json', dict(status='SQUARE_BLOCKS_ALL_COUNTS_PASS', authority=bind(AUTH),
            result=bind(OUT/'result.json'), fold_validations=seals))
    else:
        write(OUT/'result.json', output)
        subprocess.run([sys.executable, __file__, 'join-verify'], check=True)
        print(dict(status='COMPLETE', summary=summary, primary_signal=signal), flush=True)


def publish():
    v=read(OUT/'validation.json'); need(v['status']=='SQUARE_BLOCKS_ALL_COUNTS_PASS','VALIDATED')
    r=read(checked(v['result']))
    lines=['# H593：RAW与联合证据平方的重训四格对照','',
        '同六项原输入、自然RAW C128、FULL CE、五折分组OOF。JOINT_SQUARE_CE12为预定主臂，RAW_SQUARE_CE8为曲率对照；原CE7与完整DIAG_CE13封存不变。有效参数8/12，使用13坐标存储和固定mask，不声明等容量。','',
        '|模型|正确/593|对RAW救回/损失|正确最高挑战者/144|原40最终救回|',
        '|---|---:|---:|---:|---:|']
    for m,s in r['summary'].items():
        lines.append(f"|{m}|{s['correct']}|{s['rescue_vs_RAW']}/{s['break_vs_RAW']}|{s.get('target_top_of_144','—')}|{s['old40_correct']}|")
    lines+=['','|新头/对照|救回|损失|净增|组件差|95%区间|','|---|---:|---:|---:|---:|---|']
    for m,bs in r['comparisons'].items():
        for b,s in bs.items():
            lines.append(f"|{m}/{b}|{s['rescue']}|{s['loss']}|{s['net']}|{s['equal_component_difference']:.6f}|{s['bootstrap95']}|")
    bc=r['block_comparison']
    lines += ['',f"主臂对RAW平方对照：{bc['rescue']}救{bc['loss']}损，净增{bc['net']}。",'',
        f"预定联合曲率开发信号：{r['joint_curvature_signal']}；超过493与492的性能信号：{r['primary_internal_performance_signal']}。",'',
        '五折新进程重训重放、独立NumPy损失/梯度/动作验证；原对照逐位不变。23张候选缺失保留593分母。已打开开发集，区间未校正反复选择，不是外部GO；不自动替换模型。','',
        '结果：`results/rc_h593_square_blocks_v1/result.json`及各折/汇总validation.json。','']
    REPORT.write_text('\n'.join(lines)); print(REPORT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','preflight','fit-all','fit','verify','join','join-verify','publish'))
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'fit-all':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        for f in range(5): subprocess.run([sys.executable, __file__, 'fit', '--fold', str(f)], check=True)
        subprocess.run([sys.executable, __file__, 'join'], check=True)
    else:
        a = guard(args.stage, args.fold)
        if args.stage == 'publish': publish()
        else:
            import torch
            torch.set_num_threads(8); torch.set_num_interop_threads(1)
            if args.stage == 'preflight':
                from validate_rc_h593_square_blocks_v1 import self_test
                v = self_test(); v['authority'] = bind(AUTH)
                write(OUT/'preflight.json', v); print(v)
            elif args.stage in ('fit','verify'): fit(a, args.fold, args.stage=='verify')
            else: join(a, args.stage=='join-verify')
