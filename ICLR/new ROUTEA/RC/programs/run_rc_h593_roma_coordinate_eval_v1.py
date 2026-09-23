#!/usr/bin/env python3
"""Matched H593 five-fold frozen-head and refit evaluation of coordinate precision."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_six_cause_loss_binding_v1 as L
import run_rc_h593_roma_coordinate_precision_v1 as C
H=L.H
read,write,bind,checked,need=C.read,C.write,C.bind,C.checked,C.need
OUT=ROOT/'results/rc_h593_roma_coordinate_eval_v1'
AUTH=ROOT/'registry/rc_h593_roma_coordinate_eval_authority_v1_20260922.json'
REPORT=ROOT/'reports/REPORT_H593_ROMA_COORDINATE_PRECISION_V1_20260922.md'
LEVELS=tuple(C.LEVELS);KINDS=('COST1','CE')
PARENT=ROOT/'registry/rc_h593_six_feature_ablation_authority_v1_20260920.json'
LAUNCH=ROOT/'slurm/rc_h593_roma_coordinate_eval_v1.sbatch'


def hx(x):return [float(v).hex() for v in x]


def choose(z,r):
    k=int(z.argmax());pos=r['challenger_positions'][k] if float(z[k])>0 else r['winner']
    return r['candidate_physical_rows'][pos]


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY');p=read(PARENT)
    for b in p['code_sources'].values():checked(b)
    codes=dict(p['code_sources']);codes.update(program=bind(__file__),launcher=bind(LAUNCH),coordinate_plan=bind(C.PLAN),coordinate_program=bind(C.__file__),coordinate_hook=bind(ROOT/'programs/rc_roma_coordinate_precision_v1.py'))
    public=dict(p['public_sources']);public['gallery']=bind(ROOT/'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json')
    write(AUTH,dict(status='ROMA_COORDINATE_EVAL_AUTHORIZED',parent=bind(PARENT),coordinate_authority=bind(C.AUTH),code_sources=codes,
        public_sources=public,features=p['features'],fold_sources=p['fold_sources'],join_sources=p['join_sources'],
        levels=list(LEVELS),losses=list(KINDS),primary='COST1',training='Original 2000-step AdamW lr=.03 wd=.001 FP64, original order; native fit must bit-replay sealed heads',
        fixed='Original per-fold COST1/CE weights applied to all coordinate levels',refit='Same retrieval labels and folds; no heldout labels until all five prediction seals',
        evidence_level='Opened H593 grouped OOF5 mechanism ablation; 593 denominator, natural RAW C128',external_GO=False))
    print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage,fold):
    a=read(AUTH);need(a['status']=='ROMA_COORDINATE_EVAL_AUTHORIZED','AUTHORITY');checked(a['parent']);checked(a['coordinate_authority'])
    for b in a['code_sources'].values():checked(b)
    if stage not in ('preflight','publish'):need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bs in a['features']:allow.update(Path(b['path']).resolve() for b in bs.values())
    if stage in ('fit','verify'):
        need(fold in range(5),'FOLD');allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join','join-verify','publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for bs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('d1-mi','d1_mi','formal392','/grozi/','/target_join/','rc_opened_')),'PROTECTED')
        if 'curator_roles' in s:need(stage in ('join','join-verify'),'HELD_LABELS_AFTER_SEALS')
        if ROOT/'reports' in p.parents:need(stage=='publish' and p==REPORT,'NO_REPORT_READ')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('fit','verify','preflight'):
                first=p.relative_to(OUT).parts[0];own=first in ('preflight.json',f'.preflight.json.{os.getpid()}.tmp',f'fold{fold}')
            coord=C.OUT in p.parents and p.parent.name.startswith('query') and (p.name in ('payload.json','validation.json') or (stage=='diagnose' and p.name.startswith('part') and p.suffix=='.pt'))
            need(own or coord or p in allow,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':
        p=read(OUT/'preflight.json');need(p['status']=='COORDINATE_EVAL_SYNTHETIC_PASS' and p['authority']==bind(AUTH),'PREFLIGHT')
    return a


def features(a):
    for bs in a['features']:
        for b in bs.values():checked(b)
    old,_=H.features();rows=[];sources=[]
    for r in old:
        d=C.OUT/f"query{r['execution_ordinal']:03d}";v=read(d/'validation.json');p=read(checked(v['payload']))
        need(v['status']=='ROMA_COORDINATE_QUERY_PASS' and v['authority']==p['authority']==a['coordinate_authority'],'COORDINATE_SEAL')
        for k in ('query_id','execution_ordinal','source_image_sha256','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions'):need(p[k]==r[k],'ORIGINAL_AXIS:'+k)
        need(set(p['modes'])==set(LEVELS),'FOUR_LEVELS')
        for mode in LEVELS:
            x=torch.tensor(p['modes'][mode]['X'],dtype=torch.float64);need(x.shape==(127,6) and bool(torch.isfinite(x).all()),'FP64_127x6')
            p['modes'][mode]['X']=x
        need(torch.equal(p['modes']['NATIVE']['X'],r['modes']['REAL']['X']),'NATIVE_FEATURE_BITS')
        rows.append(p);sources.append(bind(d/'validation.json'))
    gallery=read(checked(a['public_sources']['gallery']));labels={r['physical_row']:r['identity'] for r in gallery['records']}
    need(len(rows)==593 and len({r['query_id'] for r in rows})==593,'COMPLETE593')
    return rows,labels,sources


def compute(a,fold):
    rows,labels,sources=features(a);f=a['fold_sources'][str(fold)]
    old=read(checked(f['full_payload']));v=read(checked(f['full_validation']))
    need(v['status']=='SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS' and v['payload']==f['full_payload'],'ORIGINAL_HEAD_QUALIFIED')
    roles={r['query_id']:r for r in read(checked(f['train_roles']))['records']};sp=read(checked(a['public_sources']['split']))['folds'][fold]
    trids=set(sp['train_query_ids']);teids=set(sp['heldout_query_ids']);need(set(roles)==trids and not trids&teids,'DISJOINT_FOLD')
    train=[r for r in rows if r['query_id'] in trids];held=[r for r in rows if r['query_id'] in teids]
    need(not {r['source_image_sha256'] for r in train}&{r['source_image_sha256'] for r in held},'DISJOINT_IMAGES')
    need([r['query_id'] for r in train]==old['train_query_ids'],'OLD_TRAIN_ORDER')
    targets=[H.target_position(r,roles[r['query_id']]['identity'],labels) for r in train];idx=[i for i,t in enumerate(targets) if t>=-1];y=torch.tensor([targets[i] for i in idx])
    ce=read(checked(f['ce_payload']));effective=[train[i]['query_id'] for i in idx]
    need(effective==ce['models']['ALL_CE']['query_ids'],'EFFECTIVE_TRAIN_ORDER')
    params={};zs={};timing=[]
    for level in LEVELS:
        x=torch.stack([train[i]['modes'][level]['X'] for i in idx]);xx=torch.stack([r['modes'][level]['X'] for r in held])
        for kind in KINDS:
            key='ALL_CE' if kind=='CE' else kind;t=torch.tensor([float.fromhex(v) for v in old['parameters'][key]],dtype=torch.float64)
            frozen=f'{kind}_FROZEN_{level}';params[frozen]=dict(theta_hex=hx(t),level=level,method='FROZEN');zs[frozen]=xx@t[:-1]+t[-1]
            start=time.perf_counter();torch.manual_seed(17);t=L.train(x,y,kind)
            if level=='NATIVE':need(hx(t)==old['parameters'][key],'NATIVE_FIT_BITS:'+kind)
            name=f'{kind}_REFIT_{level}';params[name]=dict(theta_hex=hx(t),level=level,method='REFIT');zs[name]=xx@t[:-1]+t[-1]
            timing.append(dict(model=name,seconds=time.perf_counter()-start));print(dict(event='HEAD_DONE',fold=fold,**timing[-1]),flush=True)
    oldpred={r['query_id']:r for r in old['predictions']};preds=[]
    for j,r in enumerate(held):
        models={m:dict(logits_hex=hx(z[j]),selected=choose(z[j],r)) for m,z in zs.items()}
        for kind in KINDS:
            for method in ('FROZEN','REFIT'):need(models[f'{kind}_{method}_NATIVE']==oldpred[r['query_id']]['models']['ALL_CE' if kind=='CE' else kind],'NATIVE_LOGIT_ACTION_BITS')
        preds.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],models=models))
    return dict(status='COORDINATE_EVAL_FOLD_SEALED',authority=bind(AUTH),fold=fold,train_query_ids=[r['query_id'] for r in train],effective_train_query_ids=effective,
        target_absent_train=len(train)-len(idx),parameters=params,predictions=preds,coordinate_validations=sources,heldout_label_reads=0,native_exact=True),rows,timing


def fit(a,fold,replay=False):
    d=OUT/f'fold{fold}'
    if not replay and (d/'validation.json').exists():
        v=read(d/'validation.json');need(v['authority']==bind(AUTH) and v['payload']==bind(d/'payload.json') and v['status']=='COORDINATE_EVAL_FRESH_NUMPY_PASS','RESUME');return
    p,rows,timing=compute(a,fold)
    if not replay:
        write(d/'payload.json',p);write(d/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],fits=timing))
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True);return
    need(p==read(d/'payload.json'),'FRESH_FIT_BITS');byid={r['query_id']:r for r in rows};maximum=0.;checks=0
    for pred in p['predictions']:
        r=byid[pred['query_id']]
        for m,par in p['parameters'].items():
            t=np.array([float.fromhex(v) for v in par['theta_hex']]);x=r['modes'][par['level']]['X'].numpy();z=np.sum(x*t[:-1],axis=1)+t[-1]
            expected=pred['models'][m];err=float(abs(z-np.array([float.fromhex(v) for v in expected['logits_hex']])).max());maximum=max(maximum,err);checks+=len(z)
            need(err<2e-10 and choose(z,r)==expected['selected'],'NUMPY_LOGIT_ACTION')
    write(d/'validation.json',dict(status='COORDINATE_EVAL_FRESH_NUMPY_PASS',authority=bind(AUTH),payload=bind(d/'payload.json'),logit_checks=checks,max_abs_error=maximum,heldout_label_reads=0))


def comparison(rows,base,new):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(int(r['correct'][new])-int(r['correct'][base]))
    need(len(groups)==64,'COMPONENT64');d=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260922)
    boot=d[rng.integers(0,64,size=(10000,64))].mean(1)
    gain=[r['query_id'] for r in rows if r['correct'][new] and not r['correct'][base]];loss=[r['query_id'] for r in rows if r['correct'][base] and not r['correct'][new]]
    return dict(baseline=base,new=new,rescue=len(gain),loss=len(loss),net=len(gain)-len(loss),changed=sum(r['selected'][new]!=r['selected'][base] for r in rows),gained_query_ids=gain,lost_query_ids=loss,equal_component_difference=float(d.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))


def diagnose(a):
    totals={m:dict(pairs=0,calls=0,elements=0,changed=0,squared_error_sum=0.,max_abs_normalized=0.,query_map_l1=0.,reference_map_l1=0.,final_AB_RMS=0.,final_BA_RMS=0.) for m in LEVELS[1:]}
    terminal=0;seals=[]
    for i in range(593):
        d=C.OUT/f'query{i:03d}';v=read(d/'validation.json');p=read(checked(v['payload']))
        need(v['status']=='ROMA_COORDINATE_QUERY_PASS' and v['authority']==p['authority']==a['coordinate_authority'] and p['execution_ordinal']==i,'DIAGNOSTIC_SOURCE')
        count=0
        for b in p['parts']:
            part=torch.load(checked(b),map_location='cpu',weights_only=True);need(part['authority']==a['coordinate_authority'] and part['query_id']==p['query_id'],'PART_SOURCE')
            for pair in part['pairs']:
                count+=1;terminal+=pair['arms']['NATIVE']['diagnostics']['terminal_coordinate_elements_changed']
                for m in LEVELS[1:]:
                    z=pair['arms'][m]['diagnostics'];t=totals[m];need(len(z['calls'])==12,'ALL12_CALLS');t['pairs']+=1
                    for c in z['calls']:
                        need(c['max_abs_normalized']<=1/C.LEVELS[m]+2e-7,'BOUND')
                        for k in ('elements','changed','squared_error_sum'):t[k]+=c[k]
                        t['max_abs_normalized']=max(t['max_abs_normalized'],c['max_abs_normalized']);t['calls']+=1
                    for k in ('query_map_l1','reference_map_l1'):t[k]+=z[k]
                    for direction in ('AB','BA'):t[f'final_{direction}_RMS']+=z['final_warp_RMS_normalized']['warp_'+direction]
        need(count==128,'ALL128_DIAGNOSTICS');seals.append(bind(d/'validation.json'))
        if i%50==0:print(dict(event='DIAGNOSTICS',queries=i+1),flush=True)
    for t in totals.values():
        need(t['pairs']==593*128 and t['calls']==593*128*12,'COMPLETE_DIAGNOSTICS')
        t['input_RMS_normalized']=(t['squared_error_sum']/t['elements'])**.5;t['changed_fraction']=t['changed']/t['elements']
        for k in ('query_map_l1','reference_map_l1','final_AB_RMS','final_BA_RMS'):t[k]/=t['pairs']
    write(OUT/'coordinate_diagnostics.json',dict(status='COORDINATE_DIAGNOSTICS_COMPLETE',authority=bind(AUTH),totals=totals,query_validations=seals,terminal_coordinate_elements_changed=terminal,label_reads=0))


def join(a,replay=False):
    ps=[];seals=[]
    for f in range(5):
        d=OUT/f'fold{f}';v=read(d/'validation.json');p=read(checked(v['payload']))
        need(v['status']=='COORDINATE_EVAL_FRESH_NUMPY_PASS' and p['fold']==f and v['authority']==p['authority']==bind(AUTH),'ALL5_SEALED');ps.append(p);seals.append(bind(d/'validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=seals))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']};features_,labels,src=features(a);fm={r['query_id']:r for r in features_}
    ov=read(checked(a['join_sources']['old_validation']));need(ov['result']==a['join_sources']['old_result'],'OLD_JOIN_SEAL');old={r['query_id']:r for r in read(checked(a['join_sources']['old_result']))['rows']};rows=[]
    for p in ps:
        train=read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records'];ids={r['identity'] for r in train};components={r['component'] for r in train}
        for pred in p['predictions']:
            q=pred['query_id'];role=roles[q];r=fm[q];need(role['outer_fold']==p['fold'] and role['identity'] not in ids and role['component'] not in components,'GROUP_DISJOINT')
            selected=dict(RAW=r['candidate_physical_rows'][r['winner']],**{m:z['selected'] for m,z in pred['models'].items()});correct={m:labels[s]==role['identity'] for m,s in selected.items()}
            rank=r['raw_ranked_physical_rows'];target=next(i+1 for i,s in enumerate(rank) if labels[s]==role['identity']);ranks={m:1 if correct[m] else target+int(rank.index(s)+1>target) for m,s in selected.items()}
            for new,prior in (('RAW','RAW'),('COST1_FROZEN_NATIVE','COST1'),('CE_FROZEN_NATIVE','ALL_CE')):need(correct[new]==old[q]['correct'][prior],'BASELINE_PARITY')
            rows.append(dict(query_id=q,execution_ordinal=r['execution_ordinal'],original_query_id=role['original_query_id'],fold=p['fold'],identity=role['identity'],component=role['component'],target_in_C128=target<=128,selected=selected,correct=correct,ranks=ranks))
    rows.sort(key=lambda r:r['execution_ordinal']);need(len(rows)==593 and [r['execution_ordinal'] for r in rows]==list(range(593)),'ALL593_ONCE');need(sum(r['target_in_C128'] for r in rows)==570,'RECALL570')
    summary={};comparisons={}
    for m in rows[0]['selected']:
        vsraw=comparison(rows,'RAW',m);summary[m]=dict(correct=sum(r['correct'][m] for r in rows),total=593,MRR=float(np.mean([1/r['ranks'][m] for r in rows])),rescue_vs_RAW=vsraw['rescue'],break_vs_RAW=vsraw['loss'],net_vs_RAW=vsraw['net'],by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)})
        if m!='RAW':comparisons[m]=comparison(rows,m.split('_')[0]+'_FROZEN_NATIVE',m)
    need(summary['RAW']['correct']==426 and summary['COST1_FROZEN_NATIVE']['correct']==481 and summary['CE_FROZEN_NATIVE']['correct']==486,'ORIGINAL_COUNTS')
    diagnostics=read(OUT/'coordinate_diagnostics.json');need(diagnostics['status']=='COORDINATE_DIAGNOSTICS_COMPLETE' and diagnostics['authority']==bind(AUTH) and diagnostics['query_validations']==src,'DIAGNOSTICS_SEAL')
    p=dict(status='COORDINATE_EVAL_ALL593_COMPLETE',authority=bind(AUTH),summary=summary,comparisons=comparisons,rows=rows,coordinate_validations=src,diagnostics=bind(OUT/'coordinate_diagnostics.json'),candidate_recall=570,total=593,evidence_level=a['evidence_level'],external_GO=False)
    if replay:
        need(p==read(OUT/'result.json'),'INDEPENDENT_JOIN_REPLAY');write(OUT/'validation.json',dict(status='COORDINATE_EVAL_ALL_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=seals));return
    write(OUT/'result.json',p);subprocess.run([sys.executable,__file__,'join-verify'],check=True)
    print(dict(status=p['status'],summary=summary),flush=True)


def preflight():
    x=torch.tensor([[.2,-.5,.8],[.7,.7,-.4],[-1.,-.3,-.2]],dtype=torch.float64);y=torch.tensor([-1,1,2]);z=x.numpy()
    cost=np.mean([np.logaddexp(0,z[0].max()),np.logaddexp(0,-z[1,1])+np.logaddexp(0,max(z[1,0],z[1,2])),np.logaddexp(0,-z[2,2])+np.logaddexp(0,max(z[2,0],z[2,1]))])
    need(abs(float(L.unit(x,y))-cost)<1e-14,'COST1_SCALAR')
    full=np.column_stack([np.zeros(3),z]);mx=full.max(1);ce=np.mean(mx+np.log(np.exp(full-mx[:,None]).sum(1))-full[np.arange(3),y.numpy()+1])
    need(abs(float(L.N.R.objective(x,y+1))-ce)<1e-14,'CE_SCALAR')
    r=dict(candidate_physical_rows=[8,1,7],challenger_positions=[0,2],winner=1)
    need(choose(np.array([0.,0.]),r)==1 and choose(np.array([2.,2.]),r)==8,'HOLD_TIE_AXIS')
    write(OUT/'preflight.json',dict(status='COORDINATE_EVAL_SYNTHETIC_PASS',authority=bind(AUTH),checks=['COST1_scalar','CE_scalar','HOLD_zero','candidate_tie'],natural_updates=0));print('COORDINATE_EVAL_SYNTHETIC_PASS')


def publish():
    v=read(OUT/'validation.json');need(v['status']=='COORDINATE_EVAL_ALL_COUNTS_PASS','VALIDATED');p=read(checked(v['result']))
    lines=['# RoMa匹配坐标精度消融：H593','', '原生precise连续坐标，以及在每次细化输入对x/y取整到256、64、16格。双向、两阶段均干预；1280像素下每轴最大取整位移2.5、10、40px。最终输出仍有连续残差。本实验测内部坐标采样精度的敏感性，不测有真值的匹配误差。','', '完整593图，原五折、RAW C128、冻结内容编码器。候选召回570/593，缺失23张计入分母。COST1为主，CE辅助。FROZEN固定原参数；REFIT按原2000步训练。NATIVE特征、参数、全部logits逐位核对，heldout标签在预测封存后读取。','', '|头/坐标精度|正确/593|MRR|对RAW救回/损失|对同损失原生头净增|组件95%区间|','|---|---:|---:|---:|---:|---|']
    for m,s in p['summary'].items():
        c=p['comparisons'].get(m,{});lines.append(f"|{m}|{s['correct']}|{s['MRR']:.6f}|{s['rescue_vs_RAW']}/{s['break_vs_RAW']}|{c.get('net',0)}|{c.get('bootstrap95','—')}|")
    ds=read(checked(p['diagnostics']))['totals'];lines+=['','|粒度|输入每轴扰动上界|输入RMS|query权重L1|reference权重L1|最终AB/BA坐标RMS变化|','|---|---:|---:|---:|---:|---|']
    for m,s in ds.items():lines.append(f"|{m}|{s['max_abs_normalized']:.7f}|{s['input_RMS_normalized']:.7f}|{s['query_map_l1']:.7f}|{s['reference_map_l1']:.7f}|{s['final_AB_RMS']:.7f}/{s['final_BA_RMS']:.7f}|")
    lines+=['','坐标量均为归一化坐标；最终坐标RMS表示相对原生的变化，不是真值误差。','', 'H593已反复用于开发；不据此宣称外部GO、ownership或最优粒度。最终warp取整且不重新计算overlap的负对照不改变评分，因为评分只读取overlap派生权重。','',f"原始结果：{OUT/'result.json'}",''];REPORT.write_text('\n'.join(lines));print(REPORT)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','preflight','fit','verify','diagnose','join','join-verify','publish'));ap.add_argument('--fold',type=int);args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.fold)
        if args.stage=='preflight':preflight()
        elif args.stage=='diagnose':diagnose(a)
        elif args.stage in ('fit','verify'):fit(a,args.fold,args.stage=='verify')
        elif args.stage in ('join','join-verify'):join(a,args.stage=='join-verify')
        else:publish()
