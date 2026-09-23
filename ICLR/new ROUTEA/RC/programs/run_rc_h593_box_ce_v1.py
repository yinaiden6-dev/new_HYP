#!/usr/bin/env python3
"""H593 fixed-feature FULL CE optimization, independently certified on a box."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import run_rc_h593_ce_continuation_v1 as P
import rc_h593_box_ce_core_v1 as K
from validate_rc_h593_box_ce_v1 import design,verify,logits_check

ROOT=P.ROOT
read,write,bind,checked,need,hx=P.read,P.write,P.bind,P.checked,P.need,P.hx
C,R=P.C,P.R
OUT=ROOT/'results/rc_h593_box_ce_v1'
AUTH=ROOT/'registry/rc_h593_box_ce_authority_v1_20260921.json'
PLAN=ROOT/'plan/RC_H593_BOX_CE_V1_20260921.md'
REPORT=ROOT/'reports/REPORT_H593_BOX_CE_V1_20260921.md'
ARMS=('CONTENT_BOX_CE18','DIAG_BOX_CE13')
CONTROLS=P.CONTROLS+P.ARMS
MODELS=('RAW',)+CONTROLS+ARMS
START={'CONTENT_BOX_CE18':'CONTENT_CONT_CE18','DIAG_BOX_CE13':'DIAG_CONT_CE13'}


def prepare():
    need(not AUTH.exists(),'AUTHORITY_ALREADY_FROZEN')
    a=read(P.AUTH);v=read(P.OUT/'validation.json')
    need(v['status']=='CE_CONTINUATION_ALL_COUNTS_PASS' and v['authority']==bind(P.AUTH),'PARENT_SEAL')
    checked(v['result']);codes=dict(a['code_sources'])
    for name,path in dict(program=Path(__file__),plan=PLAN,core=ROOT/'programs/rc_h593_box_ce_core_v1.py',
        independent=ROOT/'programs/validate_rc_h593_box_ce_v1.py',parent_program=Path(P.__file__),
        continuation_validator=ROOT/'programs/validate_rc_h593_ce_continuation_v1.py',
        launcher=ROOT/'slurm/rc_h593_box_ce_v1.sbatch',join_launcher=ROOT/'slurm/rc_h593_box_ce_join_v1.sbatch',
        exact_interval_core=ROOT/'programs/rc_convex_loss_cause_core_v1.py',
        certificate_provenance=ROOT/'programs/run_rc_paired_ce_optimization_isolation_v1.py').items():codes[name]=bind(path)
    for b in codes.values():checked(b)
    folds={k:dict(s) for k,s in a['fold_sources'].items()}
    for f in range(5):
        fv=read(P.OUT/f'fold{f}/validation.json')
        need(fv['status']=='CE_CONTINUATION_FRESH_NUMPY_PASS' and fv['authority']==bind(P.AUTH),'OLD_FOLD_SEAL')
        pp=read(checked(fv['payload']))
        for m in START.values():need(max(abs(float.fromhex(x)) for x in pp['parameters'][m])<=64,'INITIAL_INSIDE_BOX')
        folds[str(f)].update(continuous_payload=fv['payload'],continuous_validation=bind(P.OUT/f'fold{f}/validation.json'))
    a.update(status='H593_BOX_CE_AUTHORIZED',continuation_authority=bind(P.AUTH),code_sources=codes,fold_sources=folds,
        join_sources=dict(parent_result=v['result'],parent_validation=bind(P.OUT/'validation.json'),curator=a['join_sources']['curator']),
        arms=list(ARMS),primary=ARMS[0],primary_control='CONTENT_CONT_CE18',strong_control='DIAG_CE13',
        parameter_counts=dict(CONTENT_BOX_CE18=18,DIAG_BOX_CE13=13),
        feature_order='Original fixed 18/13 basis including bias; HOLD=0',
        optimizer=dict(name='L-BFGS-B',bounds=[-64,64],maxiter=2000,maxls=50,ftol=1e-14,gtol=1e-10,initialization=START),
        objective='Original unregularized FULL128 CE; no extra penalty. This is NOT the full decoupled AdamW procedure.',
        certificate=dict(kind='exact dyadic simplex Fenchel lower bound; mpmath interval objective',precision=50,denominator=2**52,gap_tolerance=1e-6),
        signal='CONTENT_BOX_CE18 beats frozen496 and original CE486 in correct count and equal-component mean',
        evidence_level='Opened H593 grouped-fivefold diagnostic; fixed box is a restriction; no unbounded optimum claim',
        coefficient_constraints=True,external_GO=False,deployment_change=False,encoder_forwards=0)
    a.pop('midpoint_sources',None)
    write(AUTH,a);print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage,fold=None):
    a=read(AUTH)
    for key in ('parent_authority','diagonal_authority','content_authority','content_model_authority','continuation_authority'):checked(a[key])
    for b in a['code_sources'].values():checked(b)
    need(a['code_sources']['program']==bind(__file__),'PROGRAM_SHA')
    if stage not in ('preflight','publish'):need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for seq in ('features','content_features'):
        for bs in a[seq]:allow.update(Path(b['path']).resolve() for b in bs.values())
    if stage in ('fit','verify'):
        need(fold in range(5),'FOLD_RANGE');allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join','join-verify','publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for bs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        need(not any(x in str(p).lower() for x in ('rc_opened_','d1-mi','d1_mi','formal392','/grozi/','/target_join/')),'PROTECTED_READ')
        if 'curator_roles' in str(p):need(stage in ('join','join-verify','publish'),'NO_HELD_LABELS')
        if ROOT/'reports' in p.parents:need(stage=='publish' and p==REPORT,'NO_REPORT_READ')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('preflight','fit','verify'):own=p.relative_to(OUT).parts[0] in ('preflight.json','.preflight.json.tmp',f'fold{fold}')
            need(own or p in allow,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    if stage!='preflight':
        v=read(OUT/'preflight.json');need(v['status']=='BOX_CE_SYNTHETIC_PASS' and v['authority']==bind(AUTH),'PREFLIGHT')
    return a


def inputs(a,f):
    train,held,keep,x,y,roles,labels,old=C.train_inputs(a,f)
    aug=P.content_inputs(a,train+held)
    x=np.stack([aug[r['query_id']].numpy() for r in keep]);xx=np.stack([aug[r['query_id']].numpy() for r in held])
    s=a['fold_sources'][str(f)];v=read(checked(s['continuous_validation']));p=read(checked(s['continuous_payload']))
    need(v['status']=='CE_CONTINUATION_FRESH_NUMPY_PASS' and v['authority']==a['continuation_authority'] and v['payload']==s['continuous_payload'],'SOURCE_SEAL')
    need(p['fold']==f and p['train_query_ids']==[r['query_id'] for r in train] and p['effective_train_query_ids']==[r['query_id'] for r in keep],'SAME_TRAIN_IDS')
    need([r['query_id'] for r in p['predictions']]==[r['query_id'] for r in held],'SAME_HELD_AXIS')
    return train,held,keep,x,y.numpy()+1,xx,p


def fit(a,f,replay=False):
    folder=OUT/f'fold{f}'
    if not replay and (folder/'validation.json').exists():
        v=read(folder/'validation.json');need(v['status']=='BOX_CE_EXACT_REPLAY_PASS' and v['authority']==bind(AUTH) and v['payload']==bind(folder/'payload.json'),'RESUME_SEAL');return
    started=time.monotonic();train,held,keep,x,y,xx,old=inputs(a,f)
    if replay:
        p=read(folder/'payload.json');need(p['authority']==bind(AUTH) and p['fold']==f,'OWN_BINDING')
        need(p['train_query_ids']==old['train_query_ids'] and p['effective_train_query_ids']==old['effective_train_query_ids'],'TRAIN_PARITY')
        checks={};certs={}
        from validate_rc_h593_ce_continuation_v1 import logits_check as old_logits_check
        for m in ARMS:
            theta=np.array([float.fromhex(v) for v in p['parameters'][m]])
            need(p['solver_records'][m]['initial_parameters']==old['parameters'][START[m]],'INITIALIZATION_PARITY')
            certs[m]=verify(theta,design(x,m),y,p['solver_records'][m])
            checks[m]=logits_check(xx,p['parameters'][m],p['predictions'],m)
        for m in CONTROLS:
            for row,prev in zip(p['predictions'],old['predictions']):need(row['models'][m]==prev['models'][m],'CONTROL_BIT_EXACT')
            if m!='GAP_BIAS2':checks[m]=old_logits_check(xx,old['parameters'][m],p['predictions'],m)
        write(folder/'validation.json',dict(status='BOX_CE_EXACT_REPLAY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),
            fold=f,logits=checks,certificates=certs,heldout_label_reads=0,natural_refits=0))
        print(dict(event='FOLD_CERTIFICATE_VERIFIED',fold=f,certificates=certs),flush=True);return
    parameters=dict(old['parameters']);records={};zs={}
    for m in ARMS:
        initial=np.array([float.fromhex(v) for v in old['parameters'][START[m]]])
        theta,records[m]=K.fit(design(x,m),y,initial);parameters[m]=hx(theta)
        zs[m]=design(xx,m)[:,1:]@theta
        print(dict(event='FIT_CERTIFIED',fold=f,model=m,initial_CE=records[m]['initial_CE'],final_CE=records[m]['final_CE'],gap=records[m]['certificate']['gap_upper_float'],solver=records[m]['solver']),flush=True)
    preds=[]
    for i,(r,prev) in enumerate(zip(held,old['predictions'])):
        for key in ('query_id','execution_ordinal','winner','challenger_positions','candidate_physical_rows'):need(r[key]==prev[key],'AXIS_PARITY')
        models={m:prev['models'][m] for m in CONTROLS}
        for m in ARMS:
            z=zs[m][i];top=int(z.argmax());pos=r['challenger_positions'][top] if z[top]>0 else r['winner']
            models[m]=dict(logits_hex=hx(z),top_index=top,selected=r['candidate_physical_rows'][pos])
        preds.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],winner=r['winner'],
            challenger_positions=r['challenger_positions'],candidate_physical_rows=r['candidate_physical_rows'],models=models))
    write(folder/'payload.json',dict(status='BOX_CE_FOLD_SEALED',authority=bind(AUTH),fold=f,
        parameters=parameters,solver_records=records,predictions=preds,train_query_ids=[r['query_id'] for r in train],
        effective_train_query_ids=[r['query_id'] for r in keep],heldout_label_reads=0,new_fits=2,old_control_updates=0))
    write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],fit_and_certificate_seconds=time.monotonic()-started))
    subprocess.run([sys.executable,__file__,'verify','--fold',str(f)],check=True)


def join(a,replay=False):
    ps=[];seals=[]
    for f in range(5):
        v=read(OUT/f'fold{f}/validation.json');need(v['status']=='BOX_CE_EXACT_REPLAY_PASS' and v['authority']==bind(AUTH),'FIVE_SEALS')
        p=read(checked(v['payload']));need(p['fold']==f and p['authority']==bind(AUTH),'FOLD_AXIS')
        ps.append(p);seals.append(bind(OUT/f'fold{f}/validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=seals))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    worker={r['query_id']:r for r in read(checked(a['public_sources']['worker']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    pv=read(checked(a['join_sources']['parent_validation']));need(pv['result']==a['join_sources']['parent_result'],'PARENT_JOIN')
    parent=read(checked(pv['result']));prior={r['query_id']:r for r in parent['rows']};rows=[]
    for p in ps:
        training=read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records']
        need({r['query_id'] for r in training}==set(p['train_query_ids']),'TRAIN_IDS')
        for pred in p['predictions']:
            q=pred['query_id'];role=roles[q];axis=pred['candidate_physical_rows']
            need(role['outer_fold']==p['fold'],'FOLD_ASSIGNMENT')
            for k in ('identity','component'):need(role[k] not in {r[k] for r in training},'GROUP_DISJOINT')
            need(worker[q]['source_image_sha256'] not in {worker[r['query_id']]['source_image_sha256'] for r in training},'IMAGE_DISJOINT')
            selected=dict(RAW=axis[pred['winner']],**{m:pred['models'][m]['selected'] for m in CONTROLS+ARMS})
            correct={m:labels[v]==role['identity'] for m,v in selected.items()}
            present=any(labels[v]==role['identity'] for v in axis);need(present==prior[q]['target_in_C128'],'RECALL_PARITY')
            for m in ('RAW',)+CONTROLS:need(selected[m]==prior[q]['selected'][m] and correct[m]==prior[q]['correct'][m],'CONTROL_UNCHANGED')
            tops={}
            for m in CONTROLS+ARMS:
                if m=='GAP_BIAS2':continue
                z=np.array([float.fromhex(v) for v in pred['models'][m]['logits_hex']]);top=int(z.argmax());pos=pred['challenger_positions'][top]
                tops[m]=labels[axis[pos]]==role['identity'];need(axis[pos if z[top]>0 else pred['winner']]==selected[m],'ACTUAL_ACTION')
            rows.append(dict(query_id=q,original_query_id=role['original_query_id'],component=role['component'],fold=p['fold'],
                target_in_C128=present,selected=selected,correct=correct,top_correct=tops,original40_ranking_blocked=prior[q]['original40_ranking_blocked']))
    rows.sort(key=lambda r:r['query_id'])
    need(len(rows)==len({r['query_id'] for r in rows})==593 and sum(r['target_in_C128'] for r in rows)==570,'593_RECALL570')
    summary={}
    for m in MODELS:
        rescue=sum(r['correct'][m] and not r['correct']['RAW'] for r in rows);loss=sum(not r['correct'][m] and r['correct']['RAW'] for r in rows)
        s=dict(correct=sum(r['correct'][m] for r in rows),total=593,rescue_vs_RAW=rescue,break_vs_RAW=loss,net_vs_RAW=rescue-loss,
            raw_correct_loss_rate=loss/426,switches=sum(r['selected'][m]!=r['selected']['RAW'] for r in rows),
            by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)},
            old40_correct=sum(r['correct'][m] for r in rows if r['original40_ranking_blocked']))
        if m not in ('RAW','GAP_BIAS2'):s['target_top_of_144']=sum(r['top_correct'][m] for r in rows if r['target_in_C128'] and not r['correct']['RAW'])
        need(s['correct']==426+rescue-loss,'ACCOUNTING');summary[m]=s
        if m in parent['summary']:need(s==parent['summary'][m],'FROZEN_SUMMARY')
    comparisons={m:{b:R.compare(rows,b,m) for b in ('RAW',)+CONTROLS} for m in ARMS}
    signal=all(comparisons[ARMS[0]][b]['net']>0 and comparisons[ARMS[0]][b]['equal_component_difference']>0 for b in ('CONTENT_CONT_CE18','CE_FULL'))
    certs=[dict(fold=p['fold'],model=m,initial_CE=p['solver_records'][m]['initial_CE'],final_CE=p['solver_records'][m]['final_CE'],
        gap=p['solver_records'][m]['certificate']['gap_upper_float'],certified=p['solver_records'][m]['certificate']['certified_box_gap_le_1e_6'],boundary_coordinates=p['solver_records'][m]['boundary_coordinates']) for p in ps for m in ARMS]
    output=dict(status='H593_BOX_CE_ALL593_COMPLETE',authority=bind(AUTH),summary=summary,comparisons=comparisons,rows=rows,
        optimization_certificates=certs,all_box_optima_certified=all(c['certified'] for c in certs),primary_internal_performance_signal=signal,
        evidence_level=a['evidence_level'],external_GO=False,automatic_deployment_change=False,
        intervals='Opened development paired intervals; no repeated-selection or shared-training correction')
    if replay:
        need(output==read(OUT/'result.json'),'INDEPENDENT_JOIN_REPLAY')
        write(OUT/'validation.json',dict(status='BOX_CE_ALL_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=seals))
    else:
        write(OUT/'result.json',output);subprocess.run([sys.executable,__file__,'join-verify'],check=True)
        print(dict(status='COMPLETE',summary=summary,certificates=certs,primary_signal=signal),flush=True)


def publish():
    v=read(OUT/'validation.json');need(v['status']=='BOX_CE_ALL_COUNTS_PASS','VALIDATED');r=read(checked(v['result']))
    lines=['# H593：固定特征、原CE的有界凸优化','',
        '同一五折、自然RAW C128，全部593计分，23张候选缺失保留。固定13/18参数结构与HOLD/SWITCH规则，原FULL CE无附加惩罚，参数范围[-64,64]。仅对该范围的FP64特征端点给出最优值证书，不等于无界最优、AdamW算法等价或泛化保证。','',
        '|头|正确/593|对RAW救回/损失|最高挑战者/144|原40最终救回|','|---|---:|---:|---:|---:|']
    for m,s in r['summary'].items():lines.append(f"|{m}|{s['correct']}|{s['rescue_vs_RAW']}/{s['break_vs_RAW']}|{s.get('target_top_of_144','—')}|{s['old40_correct']}|")
    lines+=['','|模型/对照|救回|损失|净增|组件95%区间|','|---|---:|---:|---:|---|']
    for m,cs in r['comparisons'].items():
        for b,c in cs.items():lines.append(f"|{m}/{b}|{c['rescue']}|{c['loss']}|{c['net']}|{c['bootstrap95']}|")
    lines+=['','|折/头|TRAIN CE原值|求解后|证书差距上界|达到1e-6|边界坐标|','|---|---:|---:|---:|---|---|']
    for c in r['optimization_certificates']:lines.append(f"|{c['fold']}/{c['model']}|{c['initial_CE']:.9f}|{c['final_CE']:.9f}|{c['gap']:.3g}|{c['certified']}|{c['boundary_coordinates']}|")
    lines+=['',f"主性能信号：{r['primary_internal_performance_signal']}；全部有界CE近最优证书成立：{r['all_box_optima_certified']}。",'',
        'H593已反复用于开发；原COST1仍为论文主模型。本试验不自动替换模型，不声明外部GO或ownership。','']
    REPORT.write_text('\n'.join(lines));print(REPORT)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','preflight','fit','verify','join','join-verify','publish'));ap.add_argument('--fold',type=int);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.fold)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if args.stage=='preflight':
            from validate_rc_h593_box_ce_v1 import self_test
            v=self_test();v['authority']=bind(AUTH);write(OUT/'preflight.json',v);print(v)
        elif args.stage in ('fit','verify'):fit(a,args.fold,args.stage=='verify')
        elif args.stage in ('join','join-verify'):join(a,args.stage=='join-verify')
        else:publish()
