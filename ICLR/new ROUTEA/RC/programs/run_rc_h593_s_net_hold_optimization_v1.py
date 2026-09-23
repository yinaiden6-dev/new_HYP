#!/usr/bin/env python3
"""Nested group calibration of frozen COST1 HOLD actions on H593."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import run_rc_six_cause_loss_binding_v1 as L
from rc_aslo_xf.h593_hold_net_gain_v1 import fit_hold_net_gain, apply_hold_lift

OUT=ROOT/'results/rc_h593_s_net_hold_optimization_v1'
AUTH=ROOT/'registry/rc_h593_s_net_hold_optimization_authority_v1_20260920.json'
PLAN=ROOT/'plan/RC_H593_S_NET_HOLD_OPTIMIZATION_V1_20260920.md'
LAUNCH=ROOT/'slurm/rc_h593_s_net_hold_optimization_v1.sbatch'
ARMS=('GAP_NET1','S_NET1','S_SAFE1')
read,write,need,bind,checked=L.read,L.write,L.need,L.bind,L.checked

def hx(values):
    return [float(v).hex() for v in values]

def prepare():
    parent_path=ROOT/'registry/rc_h593_six_feature_ablation_authority_v1_20260920.json'
    parent=read(parent_path)
    sources=dict(parent['code_sources'])
    sources.update(program=bind(__file__),launcher=bind(LAUNCH),plan=bind(PLAN),
                   exact_optimizer=bind(ROOT/'src/rc_aslo_xf/h593_hold_net_gain_v1.py'),
                   optimizer_tests=bind(ROOT/'tests/test_h593_hold_net_gain_v1.py'),
                   independent_validator=bind(ROOT/'programs/validate_rc_h593_s_net_hold_independent_v1.py'))
    folds={}
    for f in range(5):
        folder=ROOT/'results/rc_h593_six_feature_ablation_v1'/f'fold{f}'
        v=read(folder/'validation.json')
        need(v['payload']==bind(folder/'payload.json'),'OLD_ABLATION_SEALED')
        folds[str(f)]=dict(parent['fold_sources'][str(f)],ablation_payload=bind(folder/'payload.json'),
                          ablation_validation=bind(folder/'validation.json'))
    public=dict(parent['public_sources'])
    public['gallery']=bind(ROOT/'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json')
    write(AUTH,dict(status='H593_S_NET_HOLD_OPTIMIZATION_AUTHORIZED',parent=bind(parent_path),
                    code_sources=sources,public_sources=public,features=parent['features'],fold_sources=folds,
                    join_sources=parent['join_sources'],primary='S_NET1',controls=list(ARMS),
                    outer_folds=5,inner_folds_per_outer=4,inner_heads=20,steps=2000,precision='float64',
                    no_encoder_forwards=True,automatic_deployment_change=False,
                    evidence_level='Opened H593 nested grouped development validation'))
    print(json.dumps(dict(status='PREPARED',authority=bind(AUTH))),flush=True)

def guard(stage,fold):
    a=read(AUTH)
    need(a['status']=='H593_S_NET_HOLD_OPTIMIZATION_AUTHORIZED','AUTH_STATUS')
    for source in a['code_sources'].values():checked(source)
    need(a['code_sources']['program']==bind(__file__),'PINNED_PROGRAM')
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for d in a['features']:allow.update(Path(b['path']).resolve() for b in d.values())
    if stage in ('fit','verify'):
        need(fold in range(5),'FOLD')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join','join-verify'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for d in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in d.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('d1-mi','d1_mi','formal392','/grozi/','/target_join/',
                                     'gisc_prerecall_universe','/reports/','rc_opened_')),'PROTECTED_READ')
        if 'curator_roles' in s:need(stage in ('join','join-verify'),'LABELS_AFTER_SEALS')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('fit','verify'):
                part=p.relative_to(OUT).parts[0]
                own=part=='preflight.json' or part==f'fold{fold}'
            need(own or p in allow,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':
        pf=read(OUT/'preflight.json')
        need(pf['authority']==bind(AUTH) and pf['status']=='S_NET_SYNTHETIC_PREFLIGHT_PASS','PREFLIGHT')
    return a

def features(a):
    rows=[]
    for source in a['features']:
        rec=read(checked(source['receipt']));v=read(checked(source['validation']))
        need(v['payload']==rec['payload']==source['payload'],'FEATURE_SEALS')
        need(v['status']=='H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS','FEATURE_VALIDATION')
        rows.extend(torch.load(checked(source['payload']),map_location='cpu',weights_only=True)['records'])
    rows.sort(key=lambda r:r['execution_ordinal'])
    workers=read(checked(a['public_sources']['worker']))['records']
    need(len(rows)==len(workers)==593 and len({r['query_id'] for r in rows})==593,'COMPLETE593')
    for r,w in zip(rows,workers):
        need(all(r[k]==w[k] for k in ('query_id','execution_ordinal','source_image_sha256')),'WORKER_JOIN')
    g=read(checked(a['public_sources']['gallery']))
    labels={r['physical_row']:r['identity'] for r in g['records']}
    return rows,labels,g['corrected_mapping_sha256']

def target(r,identity,labels):
    positions=[i for i,p in enumerate(r['candidate_physical_rows']) if labels[p]==identity]
    need(len(positions)<=1,'UNIQUE_IDENTITY')
    if not positions:return -2
    return -1 if positions[0]==r['winner'] else r['challenger_positions'].index(positions[0])

def select(z,r):
    k=int(np.argmax(z));p=r['challenger_positions'][k] if float(z[k])>0 else r['winner']
    return r['candidate_physical_rows'][p]

def compute(a,fold):
    rows,labels,mapping=features(a);byid={r['query_id']:r for r in rows}
    splits=read(checked(a['public_sources']['split']))['folds']
    sources=a['fold_sources'][str(fold)]
    roles={r['query_id']:r for r in read(checked(sources['train_roles']))['records']}
    train_ids=set(splits[fold]['train_query_ids']);held_ids=set(splits[fold]['heldout_query_ids'])
    need(set(roles)==train_ids and not train_ids&held_ids,'OUTER_DISJOINT')
    train=[r for r in rows if r['query_id'] in train_ids]
    held=[r for r in rows if r['query_id'] in held_ids]
    old=read(checked(sources['full_payload']))
    full_validation=read(checked(sources['full_validation']))
    need(full_validation['payload']==sources['full_payload'] and
         full_validation['status']=='SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS','OLD_FULL_VALIDATED')
    av=read(checked(sources['ablation_validation']))
    need(av['payload']==sources['ablation_payload'] and av['status']=='H593_ABLATION_FRESH_REPLAY_NUMPY_PASS','ABLATION_VALIDATED')
    ablation=read(checked(sources['ablation_payload']))
    old_predictions={r['query_id']:r for r in old['predictions']}
    ablated_predictions={r['query_id']:r for r in ablation['predictions']}
    calibration=[];inner_heads=[];timing=[]
    for inner in range(5):
        if inner==fold:continue
        inner_held_ids=set(splits[inner]['heldout_query_ids'])
        need(inner_held_ids<=train_ids,'INNER_WITHIN_OUTER_TRAIN')
        fitting=[r for r in train if r['query_id'] not in inner_held_ids]
        validating=[r for r in train if r['query_id'] in inner_held_ids]
        for key in ('identity','component'):
            need(not {roles[r['query_id']][key] for r in fitting}&{roles[r['query_id']][key] for r in validating},'INNER_GROUP_DISJOINT')
        need(not {r['source_image_sha256'] for r in fitting}&{r['source_image_sha256'] for r in validating},'INNER_IMAGE_DISJOINT')
        effective=[r for r in fitting if target(r,roles[r['query_id']]['identity'],labels)>=-1]
        x=torch.stack([r['modes']['REAL']['X'] for r in effective])
        y=torch.tensor([target(r,roles[r['query_id']]['identity'],labels) for r in effective])
        need(x.dtype==torch.float64 and x.shape[1:]==(127,6),'FP64_SCHEMA')
        torch.manual_seed(17);start=time.perf_counter();theta=L.train(x,y,'COST1')
        seconds=time.perf_counter()-start;timing.append(dict(inner=inner,seconds=seconds))
        h=torch.stack([r['modes']['REAL']['X'] for r in validating]);zz=h@theta[:-1]+theta[-1]
        inner_heads.append(dict(inner_fold=inner,theta_hex=hx(theta),
                                train_query_ids=[r['query_id'] for r in fitting],
                                effective_train_query_ids=[r['query_id'] for r in effective],
                                heldout_query_ids=[r['query_id'] for r in validating]))
        for r,z in zip(validating,zz):
            j=int(z.argmax());q=r['query_id'];identity=roles[q]['identity']
            raw=r['candidate_physical_rows'][r['winner']]
            challenger=r['candidate_physical_rows'][r['challenger_positions'][j]]
            xs=float(r['modes']['REAL']['X'][j,1]);need(-1<=xs<=1,'S_BOUNDED')
            calibration.append(dict(query_id=q,inner_fold=inner,m=float(z[j]),h_s=max(xs,0.),
                                    delta=int(labels[challenger]==identity)-int(labels[raw]==identity),
                                    raw_correct=labels[raw]==identity,top_correct=labels[challenger]==identity,
                                    original_top=j,logits_hex=hx(z)))
        print(json.dumps(dict(event='INNER_FIT_DONE',outer_fold=fold,inner_fold=inner,seconds=seconds)),flush=True)
    calibration.sort(key=lambda r:byid[r['query_id']]['execution_ordinal'])
    need({r['query_id'] for r in calibration}==train_ids and len(calibration)==len(train),'INNER_COVERAGE')
    parameters={}
    for name in ARMS:
        records=[dict(m=r['m'],h=1. if name=='GAP_NET1' else r['h_s'],delta=r['delta']) for r in calibration]
        parameters[name]=fit_hold_net_gain(records,zero_break=name=='S_SAFE1')
    full=torch.tensor([float.fromhex(v) for v in old['parameters']['COST1']],dtype=torch.float64)
    x=torch.stack([r['modes']['REAL']['X'] for r in held]);allz=x@full[:-1]+full[-1]
    predictions=[]
    for r,z in zip(held,allz):
        q=r['query_id'];need(hx(z)==old_predictions[q]['models']['COST1']['logits_hex'],'FULL_LOGITS_BIT_PARITY')
        j=int(z.argmax());hs=max(float(r['modes']['REAL']['X'][j,1]),0.)
        models=dict(COST1_FULL=old_predictions[q]['models']['COST1'],CE_FULL=old_predictions[q]['models']['ALL_CE'],
                    ZERO_S=ablated_predictions[q]['models']['COST1_ZERO_S'])
        for name in ARMS:
            alpha=float.fromhex(parameters[name]['alpha_hex'])
            changed=apply_hold_lift(tuple(map(float,z)),1. if name=='GAP_NET1' else hs,alpha)
            models[name]=dict(logits_hex=hx(changed),selected=select(changed,r))
        predictions.append(dict(query_id=q,execution_ordinal=r['execution_ordinal'],
                                candidate_physical_rows=r['candidate_physical_rows'],challenger_positions=r['challenger_positions'],
                                winner=r['winner'],original_top=j,h_s=hs,models=models))
    payload=dict(status='S_NET_OUTER_FOLD_SEALED',authority=bind(AUTH),fold=fold,
                 train_query_ids=[r['query_id'] for r in train],parameters=parameters,inner_heads=inner_heads,
                 calibration=calibration,predictions=predictions,gallery_mapping_sha256=mapping,
                 heldout_label_reads=0,inner_training_updates=8000)
    return payload,rows,labels,timing

def fit(a,fold,replay=False):
    folder=OUT/f'fold{fold}'
    if not replay and (folder/'validation.json').exists():
        v=read(folder/'validation.json')
        need(v['authority']==bind(AUTH) and v['payload']==bind(folder/'payload.json'),'RESUME_MATCH')
        subprocess.run([sys.executable,str(ROOT/'programs/validate_rc_h593_s_net_hold_independent_v1.py'),'--fold',str(fold)],check=True)
        print('ALREADY_VALIDATED',flush=True);return
    p,rows,labels,timing=compute(a,fold)
    if not replay:
        write(folder/'payload.json',p)
        write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],fits=timing,threads=torch.get_num_threads()))
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True)
        subprocess.run([sys.executable,str(ROOT/'programs/validate_rc_h593_s_net_hold_independent_v1.py'),'--fold',str(fold)],check=True)
        return
    need(p==read(folder/'payload.json'),'FRESH_INNER_FIT_CALIBRATION_ALL_LOGIT_BITS')
    byid={r['query_id']:r for r in rows};maxerr=0.;checks=0
    for inner in p['inner_heads']:
        theta=np.array([float.fromhex(v) for v in inner['theta_hex']])
        for r in p['calibration']:
            if r['inner_fold']!=inner['inner_fold']:continue
            x=byid[r['query_id']]['modes']['REAL']['X'].numpy()
            z=np.sum(x*theta[:-1],axis=1)+theta[-1]
            expected=np.array([float.fromhex(v) for v in r['logits_hex']])
            error=float(np.max(np.abs(z-expected)));need(error<2e-10,'NUMPY_INNER_LOGITS')
            need(int(z.argmax())==r['original_top'] and (z.max()>0)==(r['m']>0),'NUMPY_INNER_DECISION')
            maxerr=max(maxerr,error);checks+=len(z)
    for pred in p['predictions']:
        r=byid[pred['query_id']];base=pred['models']['COST1_FULL']['logits_hex'];z=[float.fromhex(v) for v in base]
        for name in ARMS:
            vals=pred['models'][name]['logits_hex'];out=[float.fromhex(v) for v in vals]
            need(select(out,r)==pred['models'][name]['selected'],'INDEPENDENT_ACTION')
            j=pred['original_top'];alpha=float.fromhex(p['parameters'][name]['alpha_hex'])
            h=1. if name=='GAP_NET1' else pred['h_s']
            expected=list(z)
            if alpha!=0 and z[j]<=0:expected[j]=float(z[j]+float(alpha*h))
            need(hx(expected)==vals,'INDEPENDENT_LIFT_BITS')
            need(int(np.argmax(out))==j,'ORIGINAL_TOP_PRESERVED')
            if z[j]>0:need(vals==base,'ORIGINAL_SWITCH_BITS_LOCKED')
            need(all(vals[k]==base[k] for k in range(127) if k!=j),'NON_TOP_BITS_LOCKED')
    write(folder/'validation.json',dict(status='S_NET_NESTED_FRESH_REPLAY_NUMPY_PASS',authority=bind(AUTH),
                                        payload=bind(folder/'payload.json'),inner_logit_checks=checks,max_abs_error=maxerr,
                                        original_switch_and_top_preserved=True,heldout_label_reads=0))

def groupstats(rows,base,new):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(int(r['correct'][new])-int(r['correct'][base]))
    delta=np.array([np.mean(v) for _,v in sorted(groups.items())])
    rng=np.random.default_rng(20260920);boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    return dict(rescue=sum(r['correct'][new] and not r['correct'][base] for r in rows),
                loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                changed=sum(r['selected'][new]!=r['selected'][base] for r in rows),
                equal_component_difference=float(delta.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))

def joined(a):
    ps=[];bindings=[]
    for f in range(5):
        path=OUT/f'fold{f}/validation.json';v=read(path)
        need(v['status']=='S_NET_NESTED_FRESH_REPLAY_NUMPY_PASS' and v['authority']==bind(AUTH),'ALL_FOLDS_VALIDATED')
        p=read(checked(v['payload']));need(p['authority']==bind(AUTH) and p['fold']==f,'FOLD_BINDING')
        independent_path=OUT/f'fold{f}/independent_validation.json';ind=read(independent_path)
        need(ind['payload']==v['payload'] and ind['authority']==bind(AUTH) and ind.get('passed') is True,'INDEPENDENT_OPTIMALITY')
        ps.append(p);bindings.append(bind(path))
        bindings.append(bind(independent_path))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),fold_validations=bindings))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    features_,labels,_=features(a);byid={r['query_id']:r for r in features_};rows=[]
    for p in ps:
        training=[roles[q] for q in p['train_query_ids']]
        for pred in p['predictions']:
            q=pred['query_id'];role=roles[q];r=byid[q]
            need(role['outer_fold']==p['fold'],'OUTER_FOLD_JOIN')
            for key in ('identity','component','source_image_sha256'):
                need(role[key] not in {t[key] for t in training},'OUTER_GROUP_IMAGE_DISJOINT')
            raw=r['candidate_physical_rows'][r['winner']]
            selected=dict(RAW=raw,**{m:v['selected'] for m,v in pred['models'].items()})
            correct={m:labels[s]==role['identity'] for m,s in selected.items()}
            ranked=r['raw_ranked_physical_rows'];tr=next(i+1 for i,s in enumerate(ranked) if labels[s]==role['identity'])
            ranks={m:1 if correct[m] else tr+int(ranked.index(s)+1>tr) for m,s in selected.items()}
            rows.append(dict(query_id=q,original_query_id=role['original_query_id'],fold=p['fold'],
                             identity=role['identity'],component=role['component'],target_in_C128=target(r,role['identity'],labels)!=-2,
                             selected=selected,correct=correct,ranks=ranks))
    rows.sort(key=lambda r:r['query_id']);need(len(rows)==len({r['query_id'] for r in rows})==593,'ALL593_JOIN')
    summary={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=float(np.mean([1/r['ranks'][m] for r in rows])),
                    switches=sum(r['selected'][m]!=r['selected']['RAW'] for r in rows),
                    fold_correct={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)},
                    against_RAW=groupstats(rows,'RAW',m),against_COST1=groupstats(rows,'COST1_FULL',m))
             for m in rows[0]['correct']}
    need([summary[m]['correct'] for m in ('RAW','COST1_FULL','CE_FULL','ZERO_S')]==[426,481,486,487],'BASELINE_PARITY')
    comparisons={base+'__to__'+new:groupstats(rows,base,new) for base,new in
                 [('COST1_FULL','S_NET1'),('GAP_NET1','S_NET1'),('S_SAFE1','S_NET1'),('CE_FULL','S_NET1'),('ZERO_S','S_NET1')]}
    return dict(status='H593_S_NET_HOLD_OPTIMIZATION_COMPLETE',authority=bind(AUTH),population=593,
                recall_present=sum(r['target_in_C128'] for r in rows),primary='S_NET1',summary=summary,comparisons=comparisons,
                parameters={str(p['fold']):p['parameters'] for p in ps},rows=rows,
                evidence_level='Opened H593 nested grouped development validation',automatic_deployment_change=False,
                limits=['Original challenger ranking and positive SWITCH locked; cannot fix wrong original top.',
                        'Inner three-fold training to outer four-fold model score-scale transport is tested, not guaranteed.',
                        'No outer heldout labels used for parameter fitting; historical observations informed method choice.'])

def join(a,replay=False):
    result=joined(a)
    if replay:
        need(result==read(OUT/'result.json'),'FRESH_LABEL_JOIN_COUNTS_STATISTICS_REPLAY')
        write(OUT/'validation.json',dict(status='S_NET_ALL_FOLDS_ACTION_LABEL_COUNTS_PASS',authority=bind(AUTH),
                                         result=bind(OUT/'result.json'),queries=593,old_baseline_parity=True,
                                         group_statistics='deterministic replay of shared helper'))
        return
    write(OUT/'result.json',result)
    lines=['# H593 S 条件净收益放行：嵌套五折开发结果','',
           '原RAW C128，原COST1排序与已有SWITCH冻结；4个内层留组预测学习每外层一个alpha。23张候选缺失保留在593分母。','',
           '| 模型 | 正确/593 | 对COST1救/损 | SWITCH |','|---|---:|---:|---:|']
    for m,s in result['summary'].items():
        c=s['against_COST1'];lines.append(f"| {m} | {s['correct']} | {c['rescue']}/{c['loss']} | {s['switches']} |")
    lines+=['','主要配对比较：','']
    for name,c in result['comparisons'].items():lines.append(f"- {name}: {c['rescue']}救/{c['loss']}损；等组差{c['equal_component_difference']:.6f}，bootstrap95% {c['bootstrap95']}。")
    lines+=['','H593已打开，本次不自动换部署，不称新外部确认。参数/内层逐候选分数/选择证书保存在fold*/payload.json。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,__file__,'join-verify'],check=True)
    print(json.dumps(dict(status=result['status'],counts={m:s['correct'] for m,s in result['summary'].items()})),flush=True)

def preflight(a):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
    subprocess.run([sys.executable,str(ROOT/'tests/test_h593_hold_net_gain_v1.py')],env=env,check=True)
    # Real data are deliberately not consumed in the synthetic preflight.
    write(OUT/'preflight.json',dict(status='S_NET_SYNTHETIC_PREFLIGHT_PASS',authority=bind(AUTH),natural_updates=0))
    print('S_NET_SYNTHETIC_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','fit','verify','join','join-verify']);p.add_argument('--fold',type=int)
    args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.fold)
        if args.stage=='preflight':preflight(a)
        elif args.stage in ('join','join-verify'):join(a,args.stage=='join-verify')
        else:fit(a,args.fold,args.stage=='verify')
