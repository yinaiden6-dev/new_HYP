#!/usr/bin/env python3
"""Fixed original COST1 fold heads on cached stage/path interventions, CPU only."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import run_rc_h593_quality_operator_eval_v1 as E
import rc_roma_m_inside_v1 as I
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature,FEATURE_NAMES
read,write,save,bind,checked=C.read,C.write,C.save,C.bind,C.checked
OUT=ROOT/'results/rc_h593_subset41_frozen_paths_v1'
AUTH=ROOT/'registry/rc_h593_subset41_frozen_paths_authority_v1_20260923.json'
SNAP=ROOT/'results/rc_h593_existing_subset_analysis_v1/snapshot.json'
GPU_NEW=0
ARMS=tuple([f'native/{s}' for s in ('COARSE','LR4','LR2','LR1','HR4','HR2','HR1')]
    +[f'visual/{s}/HR1' for s in ('Q_GRAY','R_GRAY','Q_LOWPASS','R_LOWPASS','Q_SHUFFLE','R_SHUFFLE')]
    +[f'inside/{b}/{s}' for b in I.BRANCHES if b!='A1J1P1' for s in ('COARSE','HR1')]
    +[f'coordinate/{s}' for s in ('GRID256','GRID64','GRID16')])
MODES=('M_ONLY','FULL_UV')
NATIVE='FULL_UV/native/HR1'
LAUNCH=ROOT/'slurm/rc_h593_subset41_frozen_paths_v1.sbatch'
CONTROL=ROOT/'slurm/rc_h593_subset41_frozen_paths_control_v1.sbatch'

def prepare():
    assert not AUTH.exists()
    snapshot=read(SNAP);assert snapshot['indices']==list(range(41))
    parent=read(E.AUTH);split=read(checked(parent['public_sources']['split']))
    workers=read(C.WORKERS)['records'];heads={};bindings=[]
    wanted={workers[i]['query_id'] for i in snapshot['indices']}
    for fold in range(5):
        d=E.OUT/f'fold{fold}';v=read(d/'validation.json');p=read(checked(v['payload']))
        assert v['status']=='QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS' and p['fold']==fold and p['authority']==bind(E.AUTH)
        held=set(split['folds'][fold]['heldout_query_ids']);train=set(p['train_query_ids']);assert not held&train
        predictions={x['query_id']:x for x in p['predictions']}
        for q in sorted(wanted&held):
            assert q not in heads
            heads[q]=dict(fold=fold,theta_hex=p['parameters']['COST1_FROZEN_NATIVE']['theta_hex'],
                native=predictions[q]['models']['COST1_FROZEN_NATIVE'],
                source=v['payload'],original_operator_predictions={k:z for k,z in predictions[q]['models'].items() if k.startswith('COST1_FROZEN_')})
        bindings.extend([bind(d/'validation.json'),v['payload']])
    assert set(heads)==wanted
    write(OUT/'heads.json',dict(heads=heads,source_bindings=bindings,split=parent['public_sources']['split'],training=False))
    inside=read(ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json')
    sources=[Path(__file__),LAUNCH,CONTROL,ROOT/'plan/RC_H593_SUBSET41_FROZEN_PATHS_V1_20260923.md',Path(C.__file__),Path(E.__file__),Path(I.__file__),
      ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py']
    write(AUTH,dict(status='SUBSET41_FIXED_COST1_PATH_REPLAY_AUTHORIZED',sources=[bind(p) for p in sources],snapshot=bind(SNAP),heads=bind(OUT/'heads.json'),
        workers=bind(C.WORKERS),arms=list(ARMS),modes=list(MODES),queries=41,candidates=128,
        original_model='Original sealed COST1 head of each query held-out fold, not M1Q0R0 refit or global 32 head',
        M_ONLY='Replace M with stage/intervention mass; pool original native u/v and original weighted MaxSim. Recompute all dependent six head features. Control masses use the new maps with the original roll convention.',
        FULL_UV='Replace both maps, recompute original four scores and six features; original RAW axis, head and zero threshold unchanged',
        input_scope='38 cached stage/intervention arms; no new RoMa or encoder forward, no fit or threshold choice',
        predictor_boundary='Internal paths fix native geometry and native refiner evidence; not removal of the full matcher',
        old_result=inside['old_result'],old_validation=inside['old_validation'],gallery=inside['gallery'],
        chunk_seconds=430,max_chunks=12,exploratory=True,full593_replaced=False,new_GPU_forwards=0))
    # Nontrivial independent arithmetic check, including a zero-mass case.
    gen=torch.Generator().manual_seed(20260923)
    sim=torch.randn(7,9,generator=gen,dtype=torch.float64)
    u=torch.rand(7,generator=gen,dtype=torch.float64);v=torch.rand(9,generator=gen,dtype=torch.float64)
    uv=(u*.7,v*.9);loc=locals_torch(sim,v)
    for mode in MODES:
        a=pair_scores(sim,u,v,*uv,mode,loc);b=independent(sim.numpy(),u.numpy(),v.numpy(),uv[0].numpy(),uv[1].numpy(),mode)
        assert max(abs(a[k]-b[k]) for k in C.M.SCORE_KEYS)<1e-13
    assert all(x==0 for x in pair_scores(sim,u,v,torch.zeros_like(u),v,'M_ONLY',loc).values())
    write(OUT/'preflight.json',dict(status='FIXED_PATH_FORMULAS_NUMPY_PASS',authority=bind(AUTH),checks=['new_mass_native_pool','full_changed_maps','zero_mass'],arms=len(ARMS)))

def guard(stage,index=None):
    a=read(AUTH)
    for b in [*a['sources'],a['snapshot'],a['heads'],a['workers']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    if stage in ('worker','verify'):assert index in read(SNAP)['indices']
    allowed={Path(a[k]['path']).resolve() for k in ('snapshot','heads','workers')}
    if index is not None:
        w=read(C.WORKERS)['records'][index]
        allowed.update(Path(b['path']).resolve() for f in ('raw','roma') for b in w[f].values())
    if stage=='join':allowed.update(Path(a[k]['path']).resolve() for k in ('old_result','old_validation','gallery'))
    roots=[ROOT/'results'/s for s in ('rc_h593_m_inside_v1','rc_h593_m_visual_origin_v1','rc_h593_roma_coordinate_precision_v2','rc_h593_quality_operator_v1')]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        assert not any(x in s for x in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/'))
        if ROOT/'reports' in p.parents:assert stage=='join' and p.name=='REPORT_H593_SUBSET41_FROZEN_PATHS_20260923.md'
        if ROOT/'results' in p.parents:
            permitted=p in allowed or OUT in p.parents
            if index is not None:permitted=permitted or any(d/f'query{index:03d}' in p.parents for d in roots)
            assert permitted,s
    sys.addaudithook(audit)
    return a

def locals_torch(sim,v):
    return (sim*v[None]).max(1).values,(sim*v.roll(max(1,len(v)//2))[None]).max(1).values

def pair_scores(sim,nu,nv,u,v,mode,native_local):
    if mode=='M_ONLY':pool,ref=nu,nv;local,rolled=native_local
    else:pool,ref=u,v;local,rolled=locals_torch(sim,ref)
    control=pool.roll(max(1,len(pool)//2))
    mass=torch.sqrt(u.mean()*v.mean())
    qm=torch.sqrt(u.roll(max(1,len(u)//2)).mean()*v.mean())
    rm=torch.sqrt(u.mean()*v.roll(max(1,len(v)//2)).mean())
    scores=(mass*(pool*local).sum()/pool.sum().clamp_min(1e-12),mass,
            qm*(control*local).sum()/control.sum().clamp_min(1e-12),rm*(pool*rolled).sum()/pool.sum().clamp_min(1e-12))
    return dict(zip(C.M.SCORE_KEYS,map(float,scores)))

def independent(sim,nu,nv,u,v,mode,native_local=None):
    pool,ref=(nu,nv) if mode=='M_ONLY' else (u,v)
    uc=np.roll(pool,max(1,len(pool)//2));vr=np.roll(ref,max(1,len(ref)//2))
    if mode=='M_ONLY' and native_local is not None:local,rolled=native_local
    else:local=np.max(sim*ref[None],axis=1);rolled=np.max(sim*vr[None],axis=1)
    mass=np.sqrt(u.mean()*v.mean());qm=np.sqrt(np.roll(u,max(1,len(u)//2)).mean()*v.mean());rm=np.sqrt(u.mean()*np.roll(v,max(1,len(v)//2)).mean())
    scores=(mass*np.sum(pool*local)/max(pool.sum(),1e-12),mass,qm*np.sum(uc*local)/max(uc.sum(),1e-12),rm*np.sum(pool*rolled)/max(pool.sum(),1e-12))
    return dict(zip(C.M.SCORE_KEYS,map(float,scores)))

def inputs(a,index):
    selected=next(r for r in read(SNAP)['sources'] if r['index']==index)
    payloads={f:read(checked(b['payload'])) for f,b in selected['sources'].items()}
    parts={f:[] for f in payloads}
    for f,p in payloads.items():
        for b in p['parts']:parts[f].extend(torch.load(checked(b),map_location='cpu',weights_only=True)['pairs'])
        assert len(parts[f])==128 and [v['candidate_position'] for v in parts[f]]==list(range(128))
    manifest=read(checked(selected['sources']['inside']['paths_payload']))
    w=read(C.WORKERS)['records'][index];q,old,refs=C.load_input(w)
    for f,p in payloads.items():assert p['query_id']==w['query_id'] and p['candidate_physical_rows']==q['candidate_physical_rows']
    op=read(checked(payloads['visual']['operator_source']))
    return w,q,old,refs,parts,manifest,op

def maps(pos,parts,inside):
    result={}
    for arm in ARMS:
        bits=arm.split('/');family=bits[0]
        if family=='native':z=parts['visual'][pos]['arms']['NATIVE']['stages'][bits[1]];u,v=z['AB']['weights'],z['BA']['weights']
        elif family=='visual':z=parts['visual'][pos]['arms'][bits[1]]['stages'][bits[2]];u,v=z['AB']['weights'],z['BA']['weights']
        elif family=='coordinate':z=parts['coordinate'][pos]['arms'][bits[1]];u,v=z['query_visibility'],z['reference_visibility']
        else:
            def side(i):
                z=inside['sides'][i]
                return z['coarse'][bits[1]]['weights'] if bits[2]=='COARSE' else z['stages'][bits[2]]['branches'][bits[1]]['weights']
            u,v=side(0),side(1)
        assert u.dtype==v.dtype==torch.float64
        result[arm]=(u,v)
    return result

def worker(index):
    a=guard('worker',index);folder=OUT/f'query{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'validation.json').exists():checked(read(folder/'validation.json')['payload']);return
    started=time.monotonic();w,q,old,refs,parts,manifest,op=inputs(a,index)
    qnorm=F.normalize(q['query_tokens'].to(torch.float64),dim=1)
    for pos,physical in enumerate(q['candidate_physical_rows']):
        dest=folder/f'pair{pos:03d}.pt'
        if dest.exists():continue
        if time.monotonic()-started>a['chunk_seconds']:break
        ref=refs[physical];prior=old['candidates'][pos]
        assert ref['tokens_sha256']==prior['reference_tokens_sha256'] and C.M.token_sha(ref['tokens'])==ref['tokens_sha256']
        z=torch.load(checked(manifest['sources'][pos]),map_location='cpu',weights_only=True)
        assert z['query_sha']==q['query_source_sha256'] and z['reference_sha']==ref['source_image_sha256']
        sim=qnorm@F.normalize(ref['tokens'].to(torch.float64),dim=1).T
        nu,nv=prior['query_visibility'],prior['reference_visibility'];native_local=locals_torch(sim,nv)
        variants=maps(pos,parts,z);scores={};maximum=0.;checks=0
        # Independent NumPy reduction, retaining original Torch matrix product.
        # Matrix product/native scores additionally checked against sealed prior data.
        nsim=sim.numpy();nnu=nu.numpy();nnv=nv.numpy()
        nlocal=(np.max(nsim*nnv[None],axis=1),np.max(nsim*np.roll(nnv,max(1,len(nnv)//2))[None],axis=1))
        for arm,(u,v) in variants.items():
            for mode in MODES:
                name=mode+'/'+arm;scores[name]=pair_scores(sim,nu,nv,u,v,mode,native_local)
                verify=independent(nsim,nnu,nnv,u.numpy(),v.numpy(),mode,nlocal)
                err=max(abs(scores[name][k]-verify[k]) for k in C.M.SCORE_KEYS);maximum=max(maximum,err);assert err<2e-10
                checks+=4
                if arm=='native/HR1':
                    assert all(float(scores[name][k]).hex()==float(prior['old_scores'][k]).hex() for k in C.M.SCORE_KEYS),('NATIVE_SCORE_BITS',index,pos,mode)
                if mode=='FULL_UV' and arm.startswith('coordinate/'):
                    expected=parts['coordinate'][pos]['arms'][arm.split('/')[1]]['scores']
                    assert all(float(scores[name][k]).hex()==float(expected[k]).hex() for k in C.M.SCORE_KEYS),('COORDINATE_SCORE_BITS',index,pos,arm)
        save(dest,dict(authority=bind(AUTH),query_id=w['query_id'],index=index,position=pos,physical_row=int(physical),scores=scores,
            max_numpy_error=maximum,independent_scalar_checks=checks,inside_source=manifest['sources'][pos],reference_tokens_sha256=ref['tokens_sha256']))
        if (pos+1)%16==0:print(dict(event='PAIRS_READY',index=index,pairs=pos+1,seconds=time.monotonic()-started),flush=True)
    ready=len(list(folder.glob('pair*.pt')))==128
    if ready:
        pairdata=[torch.load(folder/f'pair{i:03d}.pt',map_location='cpu',weights_only=True) for i in range(128)]
        raw=list(map(float,q['candidate_raw_scores']));row={k:op[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','winner','challenger_positions')}
        head=read(OUT/'heads.json')['heads'][w['query_id']];theta=torch.tensor([float.fromhex(x) for x in head['theta_hex']],dtype=torch.float64)
        models={};feature_rows={};parity=0
        for name in pairdata[0]['scores']:
            evidence={i:p['scores'][name] for i,p in enumerate(pairdata)}
            x=torch.stack([candidate_feature(raw,evidence,c,row['winner']) for c in row['challenger_positions']]);zs=x@theta[:-1]+theta[-1]
            pred=dict(logits_hex=E.hx(zs),selected=E.choose(zs,row))
            nz=np.sum(x.numpy()*theta[:-1].numpy(),axis=1)+float(theta[-1]);assert np.max(abs(nz-zs.numpy()))<2e-10 and E.choose(nz,row)==pred['selected']
            models[name]=pred;feature_rows[name]=x
            if name.endswith('/native/HR1'):
                assert torch.equal(x,torch.tensor(op['modes']['NATIVE']['X'],dtype=torch.float64))
                assert pred==head['native'],('NATIVE_LOGIT_BITS',index,name);parity+=127
        # Previously sealed CPU operator controls under exactly the same head.
        for mode,values in op['modes'].items():
            name='OPERATOR/'+mode;x=torch.tensor(values['X'],dtype=torch.float64);zs=x@theta[:-1]+theta[-1]
            pred=dict(logits_hex=E.hx(zs),selected=E.choose(zs,row));assert pred==head['original_operator_predictions']['COST1_FROZEN_'+mode]
            models[name]=pred;feature_rows[name]=x
        tensor=folder/'features.pt';value=dict(authority=bind(AUTH),row=row,theta=theta,X=feature_rows,label_reads=0)
        if tensor.exists():
            before=torch.load(tensor,map_location='cpu',weights_only=True)
            assert before['authority']==value['authority'] and before['row']==row and torch.equal(before['theta'],theta)
            assert before['X'].keys()==feature_rows.keys() and all(torch.equal(before['X'][k],v) for k,v in feature_rows.items())
        else:save(tensor,value)
        write(folder/'payload.json',dict(status='SUBSET41_FIXED_PATH_PREDICTIONS_SEALED',authority=bind(AUTH),index=index,row=row,fold=head['fold'],head=head,
             models=models,features=bind(tensor),pairs=[bind(folder/f'pair{i:03d}.pt') for i in range(128)],label_reads=0,new_GPU_forwards=0,training=False))
        subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)
    write(folder/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",dict(status='SUBSET41_CPU_NORMAL_EXIT',index=index,complete=ready,seconds=time.monotonic()-started,pairs=len(list(folder.glob('pair*.pt')))))

def verify(index):
    guard('verify',index);d=OUT/f'query{index:03d}';p=read(d/'payload.json');v=torch.load(checked(p['features']),map_location='cpu',weights_only=True)
    assert p['authority']==v['authority']==bind(AUTH) and p['row']==v['row'] and p['label_reads']==0
    theta=np.asarray([float.fromhex(x) for x in p['head']['theta_hex']]);maximum=0.;count=0
    for name,z in p['models'].items():
        x=v['X'][name].numpy();assert x.shape==(127,6) and np.isfinite(x).all()
        pred=np.sum(x*theta[:-1],axis=1)+theta[-1];expected=np.asarray([float.fromhex(x) for x in z['logits_hex']])
        maximum=max(maximum,float(abs(pred-expected).max()));assert maximum<2e-10 and E.choose(pred,p['row'])==z['selected'];count+=127
    assert p['models'][NATIVE]==p['models']['M_ONLY/native/HR1']==p['head']['native']
    pairs=[torch.load(checked(b),map_location='cpu',weights_only=True) for b in p['pairs']]
    assert [z['position'] for z in pairs]==list(range(128)) and [z['physical_row'] for z in pairs]==p['row']['candidate_physical_rows']
    write(d/'validation.json',dict(status='SUBSET41_COST1_FIXED_PATH_CPU_PASS',authority=bind(AUTH),payload=bind(d/'payload.json'),index=index,
       max_numpy_logit_error=maximum,logits_checked=count,scalars_checked=sum(z['independent_scalar_checks'] for z in pairs),
       max_numpy_scalar_error=max(z['max_numpy_error'] for z in pairs),native_all127_logits_bit_exact=True,training=False,new_GPU_forwards=0))

def join():
    a=guard('join');observed=[]
    for i in read(SNAP)['indices']:
        d=OUT/f'query{i:03d}';v=read(d/'validation.json');assert v['status']=='SUBSET41_COST1_FIXED_PATH_CPU_PASS' and v['authority']==bind(AUTH)
        observed.append(read(checked(v['payload'])))
    write(OUT/'prediction_seal.json',dict(authority=bind(AUTH),queries=41,validations=[bind(OUT/f'query{i:03d}/validation.json') for i in read(SNAP)['indices']]))
    old={r['query_id']:r for r in read(checked(a['old_result']))['rows']};labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    rows=[]
    for p in observed:
        q=p['row']['query_id'];r=old[q];axis=p['row']['candidate_physical_rows'];target=next((i for i,x in enumerate(axis) if labels[x]==r['identity']),None)
        assert r['fold']==p['fold'] and p['models'][NATIVE]['selected']==r['selected']['COST1_FROZEN_NATIVE']
        details={};v=torch.load(checked(p['features']),map_location='cpu',weights_only=True);theta=v['theta']
        for name,pred in p['models'].items():
            z=torch.tensor([float.fromhex(x) for x in pred['logits_hex']],dtype=torch.float64);scores=torch.zeros(128,dtype=torch.float64);scores[p['row']['challenger_positions']]=z
            detail=dict(selected=pred['selected'],correct=labels[pred['selected']]==r['identity'],switched=pred['selected']!=axis[p['row']['winner']])
            if target is not None:
                wrong=scores.clone();wrong[target]=-torch.inf;j=int(wrong.argmax());detail.update(target_logit=float(scores[target]),strongest_wrong_logit=float(wrong[j]),target_margin=float(scores[target]-wrong[j]),strongest_wrong=axis[j])
                def contributions(pos):
                    if pos==p['row']['winner']:return torch.zeros(7,dtype=torch.float64)
                    k=p['row']['challenger_positions'].index(pos);return torch.cat([v['X'][name][k]*theta[:-1],theta[-1:]])
                term=contributions(target)-contributions(j);assert abs(float(term.sum())-detail['target_margin'])<2e-10
                detail['target_margin_contributions']=dict(zip((*FEATURE_NAMES,'bias'),map(float,term)))
            details[name]=detail
        rows.append(dict(query_id=q,original_query_id=r['original_query_id'],index=p['index'],fold=r['fold'],component=r['component'],raw_correct=r['correct']['RAW'],native_correct=r['correct']['COST1_FROZEN_NATIVE'],target_in_C128=target is not None,models=details))
    summaries={};rescues=[r for r in rows if r['native_correct'] and not r['raw_correct']]
    assert len(rows)==41 and sum(r['raw_correct'] for r in rows)==32 and sum(r['native_correct'] for r in rows)==35 and len(rescues)==3
    for name in rows[0]['models']:
        summaries[name]=dict(correct=sum(r['models'][name]['correct'] for r in rows),retained_original_rescues=sum(r['models'][name]['correct'] for r in rescues),
            rescue_vs_native=sum(r['models'][name]['correct'] and not r['native_correct'] for r in rows),break_vs_native=sum(not r['models'][name]['correct'] and r['native_correct'] for r in rows),
            rescue_vs_raw=sum(r['models'][name]['correct'] and not r['raw_correct'] for r in rows),break_vs_raw=sum(not r['models'][name]['correct'] and r['raw_correct'] for r in rows),
            changed_vs_native=sum(r['models'][name]['selected']!=r['models'][NATIVE]['selected'] for r in rows))
    result=dict(status='SUBSET41_FIXED_COST1_PATH_JOIN_PASS',authority=bind(AUTH),seal=bind(OUT/'prediction_seal.json'),queries=41,groups=len({r['component'] for r in rows}),
       head='Original per-fold frozen COST1',candidate_source='Natural RAW C128',summary=summaries,rows=rows,original_rescue_ids=[r['original_query_id'] for r in rescues],
       new_GPU_forwards=0,training=False,exploratory=True,external_GO=False,full593_result_replaced=False,
       interpretation='Conditional fixed-head interventions; changed inputs may cause distribution shift. M_ONLY keeps original spatial pooling; FULL_UV also changes spatial weighting. No parameter selection or retraining.')
    write(OUT/'result.json',result);write(OUT/'validation.json',dict(status=result['status'],authority=bind(AUTH),result=bind(OUT/'result.json'),all_queries_validated=True))
    lines=['# 41张共同子集：原COST1冻结折头回放','',f"RAW 32/41；原COST1 35/41，原纠错：{result['original_rescue_ids']}。每张使用原held-out折参数，候选、零阈值、HOLD/SWITCH不变。原生全部127 logits逐位复现；无新GPU前向、无训练。",
      '', 'M_ONLY替换整体质量、保留原空间汇聚；FULL_UV替换完整两侧权重。下表正确数均经过原COST1小头，不能与此前M×内容直接排序混用。','',
      '|缓存通路|仅替换M：正确/保留原3救回|完整u/v：正确/保留原3救回|完整u/v相对原头救/损|','|---|---:|---:|---:|']
    for arm in ARMS:
        m=summaries['M_ONLY/'+arm];v=summaries['FULL_UV/'+arm]
        lines.append(f"|{arm}|{m['correct']}/41；{m['retained_original_rescues']}/3|{v['correct']}/41；{v['retained_original_rescues']}/3|{v['rescue_vs_native']}/{v['break_vs_native']}|")
    lines+=['','逐候选127 logits、全部6列输入、4个评分、输入来源及原3例的margin逐项贡献均已保存在本结果目录。',
      '', '这是按缓存完成情况选取的已打开41张/30组探索，不是全593或新的独立测试。完整u/v干预沿用原参数，下降也可能包含分布变化；内部通路保持原生几何及细化证据，不证明整个匹配器不可替代。']
    (ROOT/'reports/REPORT_H593_SUBSET41_FROZEN_PATHS_20260923.md').write_text('\n'.join(lines)+'\n')
    print(dict(status=result['status'],rescues=result['original_rescue_ids'],summary=summaries),flush=True)

def control(previous=None):
    # Scheduler-only: no model/data reads except completed validation envelopes.
    assert os.environ.get('SLURM_JOB_ID');a=read(AUTH)
    for b in a['sources']:checked(b)
    if previous:
        text=subprocess.run(['sacct','-X','-n','-P','-j',previous,'--format=JobID,State,ExitCode'],capture_output=True,text=True,check=True).stdout
        lines=[x.split('|') for x in text.splitlines() if x.strip()]
        assert lines and all(x[1:3]==['COMPLETED','0:0'] for x in lines),('PREVIOUS_FAILED',text)
    pilot=OUT/'query000/validation.json'
    if pilot.exists():assert read(pilot)['status']=='SUBSET41_COST1_FIXED_PATH_CPU_PASS'
    attempts=Counter()
    for p in (OUT/'waves').glob('*.json'):
        for i in read(p)['indices']:attempts[i]+=1
    missing=[]
    for i in read(SNAP)['indices'] if pilot.exists() else [0]:
        p=OUT/f'query{i:03d}/validation.json'
        if p.exists():v=read(p);assert v['status']=='SUBSET41_COST1_FIXED_PATH_CPU_PASS' and v['authority']==bind(AUTH);checked(v['payload'])
        else:assert attempts[i]<a['max_chunks'];missing.append(i)
    if not missing:
        subprocess.run([sys.executable,__file__,'join'],check=True);return
    args=['sbatch','--parsable','--hold','--array='+','.join(map(str,missing))+'%40',str(LAUNCH),'worker']
    job=subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip().split(';')[0]
    spool=OUT/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scontrol','write','batch_script',job,str(spool)],capture_output=True,text=True,check=True);assert spool.read_bytes()==LAUNCH.read_bytes()
    callback=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+job,'--kill-on-invalid-dep=yes',str(CONTROL),'control',job],capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
    write(OUT/'waves'/f'{len(list((OUT/"waves").glob("*.json"))):03d}.json',dict(job=job,callback=callback,indices=missing,authority=bind(AUTH),spool=bind(spool)))
    subprocess.run(['scontrol','release',job],check=True)
    print(dict(submitted=job,callback=callback,remaining=len(missing)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','worker','verify','join','control'));ap.add_argument('--index',type=int);ap.add_argument('--previous');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='worker':worker(args.index)
    elif args.stage=='verify':verify(args.index)
    elif args.stage=='join':join()
    else:control(args.previous)
