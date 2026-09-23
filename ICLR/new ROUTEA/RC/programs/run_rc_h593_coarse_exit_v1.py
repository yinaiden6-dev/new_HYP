#!/usr/bin/env python3
"""Real early-exit qualification, cached enlarged replay, and bounded Slurm chain."""
import argparse
from collections import Counter
import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import run_rc_h593_subset41_frozen_paths_v1 as S
import run_rc_h593_quality_operator_eval_v1 as E
import rc_roma_coarse_exit_v1 as X
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
read,write,save,bind,checked=C.read,C.write,C.save,C.bind,C.checked
OUT=ROOT/'results/rc_h593_coarse_exit_v1'
AUTH=ROOT/'registry/rc_h593_coarse_exit_authority_v1_20260923.json'
GPU=ROOT/'slurm/rc_h593_coarse_exit_gpu_v1.sbatch'
CPU=ROOT/'slurm/rc_h593_coarse_exit_cpu_v1.sbatch'
VIS=ROOT/'results/rc_h593_m_visual_origin_v1'
PLAN=ROOT/'plan/RC_H593_COARSE_EXIT_V1_20260923.md'

def prepare():
    assert not AUTH.exists()
    workers=read(C.WORKERS)['records'];records=[]
    for i,w in enumerate(workers):
        src=next((d for d in (VIS,C.OUT) if (d/f'query{i:03d}/validation.json').exists()),None)
        if src is None:continue
        vp=src/f'query{i:03d}/validation.json';v=read(vp);p=read(checked(v['payload']))
        assert v['status'] in ('M_VISUAL_ORIGIN_QUERY_PASS','ROMA_COORDINATE_QUERY_PASS')
        assert p['query_id']==w['query_id'] and p['execution_ordinal']==i
        op=ROOT/f'results/rc_h593_quality_operator_v1/query{i:03d}/validation.json';ov=read(op)
        records.append(dict(index=i,query_id=w['query_id'],kind='maps' if src==VIS else 'logits',validation=bind(vp),payload=v['payload'],operator_validation=bind(op),operator=ov['payload']))
    assert [r['index'] for r in records]==list(range(70)), 'FREEZE_EXPECTED70_OR_REVIEW_SNAPSHOT'
    ea=read(E.AUTH);split=read(checked(ea['public_sources']['split']));heads={}
    wanted={r['query_id'] for r in records}
    for fold in range(5):
        v=read(E.OUT/f'fold{fold}/validation.json');p=read(checked(v['payload']))
        held=set(split['folds'][fold]['heldout_query_ids']);assert not held&set(p['train_query_ids'])
        for pred in p['predictions']:
            q=pred['query_id']
            if q in wanted:
                assert q in held and q not in heads
                heads[q]=dict(fold=fold,theta_hex=p['parameters']['COST1_FROZEN_NATIVE']['theta_hex'],native=pred['models']['COST1_FROZEN_NATIVE'],source=v['payload'])
    assert set(heads)==wanted
    write(OUT/'heads.json',dict(heads=heads,split=ea['public_sources']['split']))
    write(OUT/'cache_snapshot.json',dict(records=records,indices=list(range(70)),query_selection='All validated complete coarse-stage caches at freeze; no label filtering',full593_target=True))
    source=[Path(__file__),Path(X.__file__),Path(C.__file__),Path(S.__file__),Path(E.__file__),GPU,CPU,PLAN,ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py']
    prev=read(S.AUTH)
    write(AUTH,dict(status='COARSE_EXIT_AND_CACHED70_AUTHORIZED',sources=[bind(p) for p in source],snapshot=bind(OUT/'cache_snapshot.json'),heads=bind(OUT/'heads.json'),workers=bind(C.WORKERS),profile=bind(C.PROFILE),
        old_result=prev['old_result'],old_validation=prev['old_validation'],gallery=prev['gallery'],prior41_result=bind(S.OUT/'result.json'),
        probe_queries=[0,18,36,54],probe_candidates=[0,127],timing_repetitions=3,chunk_seconds=430,max_chunks=12,max_parallel=50,
        cached_queries=70,primary='FULL_COARSE',comparator='Original per-query held-out COST1 NATIVE',parameters_updated=False,threshold_changed=False,
        compute='Only8 label-blind pairs rerun full/early for parity and timings; remaining GPU use decodes cached confidence only',full593_result_replaced=False,external_GO=False))
    # Readout includes independent NumPy checks during every query, not only a pilot.
    write(OUT/'preflight.json',dict(status='SOURCES_AND_LABEL_BLIND70_FROZEN',authority=bind(AUTH),maps=sum(r['kind']=='maps' for r in records),logits=sum(r['kind']=='logits' for r in records)))

def guard(stage,index=None):
    a=read(AUTH)
    for b in [*a['sources'],a['snapshot'],a['heads'],a['workers'],a['profile']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID')
    if stage in ('probe','worker'):
        forbidden={Path(a[k]['path']).resolve() for k in ('old_result','old_validation','gallery','prior41_result')}
        def audit(event,args):
            if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
            assert p not in forbidden and not any(t in s for t in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/','/reports/'))
        sys.addaudithook(audit)
    if stage=='worker':
        assert not torch.cuda.is_available()
        v=read(OUT/'probe_validation.json');assert v['status']=='COARSE_EARLY_EXIT_BITS_AND_ZERO_REFINER_PASS' and v['authority']==bind(AUTH);checked(v['payload'])
        assert index in read(a['snapshot']['path'])['indices']
    return a

def snapshot(a):return {r['index']:r for r in read(checked(a['snapshot']))['records']}
def parts(record):
    v=read(checked(record['validation']));p=read(checked(record['payload']));assert v['payload']==record['payload']
    return p

def context(a,index):
    w=read(checked(a['workers']))['records'][index];q,old,refs=C.load_input(w)
    profile=read(a['profile']['path']);core=C.M.legacy_core(profile)
    qp=Path(q['query_source_path']);assert bind(qp)['sha256']==q['query_source_sha256']
    qg,qmeta=C.M.geometry(qp,q['query_source_sha256'],'early-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
    return w,q,old,refs,profile,core,qg,qmeta

def refgeom(profile,ref):
    p=Path(ref['source_path']);assert bind(p)['sha256']==ref['source_image_sha256']
    return C.M.geometry(p,ref['source_image_sha256'],'early-reference',ref['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])

def load_maps(record):
    p=parts(record);answer={}
    for b in p['parts']:
        for pair in torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)['pairs']:
            z=pair['arms']['NATIVE']['stages']['COARSE'];pos=pair['candidate_position']
            answer[pos]=(pair['physical_row'],z['AB']['weights'],z['BA']['weights'])
    assert sorted(answer)==list(range(128));return p,answer

def probe(a):
    assert torch.cuda.is_available()
    if (OUT/'probe_validation.json').exists():return
    started=time.monotonic();snap=snapshot(a);profile=read(a['profile']['path']);C.check_profile(profile)
    model=C.M.gpu_model(profile);device=str(next(model.parameters()).device)
    summaries=[]
    for index in a['probe_queries']:
        todo=[pos for pos in a['probe_candidates'] if not (OUT/f'probe/q{index:03d}_p{pos:03d}.json').exists()]
        if not todo:continue
        w,q,old,refs,profile,core,qg,qmeta=context(a,index);p,cached=load_maps(snap[index]);qi=core.oriented(Path(q['query_source_path']))
        for pos in todo:
            if time.monotonic()-started>a['chunk_seconds']:return
            physical=q['candidate_physical_rows'][pos];ref=refs[physical];rg,rmeta=refgeom(profile,ref);ri=core.oriented(Path(ref['source_path']))
            calls=X.Calls(model)
            full=model.match(qi,ri);full_counts=dict(calls.counts);matcher={k:v.cpu().clone() for k,v in calls.coarse.items()}
            assert full_counts==dict(descriptor=2,matcher=1,fine_features=4,refiner_4=4,refiner_2=4,refiner_1=4),full_counts
            nu=core.cell_means(full['overlap_AB'][0,...,0].cpu(),qg);nv=core.cell_means(full['overlap_BA'][0,...,0].cpu(),rg)
            assert C.M.bit_equal(nu,old['candidates'][pos]['query_visibility']) and C.M.bit_equal(nv,old['candidates'][pos]['reference_visibility'])
            calls.reset();early=X.match_coarse(model,qi,ri);early_counts=dict(calls.counts);calls.close()
            assert early_counts==dict(descriptor=2,matcher=1),early_counts
            assert all(torch.equal(early[k].cpu(),v) for k,v in matcher.items()),'MATCHER_OUTPUT_BITS'
            u=core.cell_means(early['overlap_AB'][0,...,0].cpu(),qg);v=core.cell_means(early['overlap_BA'][0,...,0].cpu(),rg)
            assert cached[pos][0]==physical and C.M.bit_equal(u,cached[pos][1]) and C.M.bit_equal(v,cached[pos][2]),'CACHED_COARSE_MAP_BITS'
            evidence=OUT/f'probe/q{index:03d}_p{pos:03d}.pt'
            save(evidence,dict(index=index,position=pos,physical_row=physical,query_source_sha=q['query_source_sha256'],reference_source_sha=ref['source_image_sha256'],query_geometry=qmeta,reference_geometry=rmeta,
                matcher=matcher,u=u,v=v,native_u=nu,native_v=nv,full_calls=full_counts,early_calls=early_counts))
            del full,early,calls
            timed=[]
            for rep in range(a['timing_repetitions']):
                for name in (('FULL','EARLY') if rep%2==0 else ('EARLY','FULL')):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                    output=model.match(qi,ri) if name=='FULL' else X.match_coarse(model,qi,ri)
                    torch.cuda.synchronize();seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated();del output
                    timed.append(dict(arm=name,repetition=rep,seconds=seconds,peak_allocated_bytes=peak))
            write(evidence.with_suffix('.json'),dict(status='EXACT_EARLY_PAIR_PASS',authority=bind(AUTH),index=index,position=pos,intermediates=bind(evidence),timings=timed,full_calls=full_counts,early_calls=early_counts))
            print(dict(event='EARLY_PROBE_PAIR',index=index,pos=pos,seconds=time.monotonic()-started),flush=True)
    # Decode only already-sealed coarse confidence. Never a RoMa/encoder forward.
    for index,record in snap.items():
        if record['kind']!='logits':continue
        dest=OUT/f'decoded/query{index:03d}.pt'
        if dest.exists():continue
        if time.monotonic()-started>a['chunk_seconds']:return
        p=parts(record);overlaps=[]
        for b in p['parts']:
            for pair in torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)['pairs']:
                c=pair['coarse_matcher']
                # Original observer uses GPU sigmoid on [...,0], then CPU cell_means.
                z={side:c['confidence_'+side].to(device)[0,...,0].sigmoid().cpu() for side in ('AB','BA')}
                overlaps.append(dict(position=pair['candidate_position'],physical_row=pair['physical_row'],overlaps=z))
        assert [r['position'] for r in overlaps]==list(range(128))
        save(dest,dict(authority=bind(AUTH),index=index,query_id=record['query_id'],source=record['payload'],records=overlaps,new_model_forwards=0))
        print(dict(event='CACHED_SIGMOID_ONLY',index=index,seconds=time.monotonic()-started),flush=True)
    for i in a['probe_queries']:
        for pos in a['probe_candidates']:summaries.append(read(OUT/f'probe/q{i:03d}_p{pos:03d}.json'))
    times={k:[t['seconds'] for r in summaries for t in r['timings'] if t['arm']==k] for k in ('FULL','EARLY')}
    per_pair=[statistics.median(t['seconds'] for t in r['timings'] if t['arm']=='FULL')/statistics.median(t['seconds'] for t in r['timings'] if t['arm']=='EARLY') for r in summaries]
    payload=dict(status='COARSE_EARLY_EXIT_BITS_AND_ZERO_REFINER_PASS',authority=bind(AUTH),pairs=summaries,decoded=[bind(OUT/f'decoded/query{i:03d}.pt') for i,r in snap.items() if r['kind']=='logits'],
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,timing_seconds=times,median_seconds={k:statistics.median(v) for k,v in times.items()},paired_speedup_median=statistics.median(per_pair),
        speedup_is_not_end_to_end=True,model_load_decode_and_head_excluded=True,parameters_updated=False)
    write(OUT/'probe_payload.json',payload);write(OUT/'probe_validation.json',dict(status=payload['status'],authority=bind(AUTH),payload=bind(OUT/'probe_payload.json'),exact_pairs=len(summaries),zero_refiner=True))

def worker(a,index):
    d=OUT/f'query{index:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'validation.json').exists():checked(read(d/'validation.json')['payload']);return
    started=time.monotonic();record=snapshot(a)[index];w,q,old,refs,profile,core,qg,qmeta=context(a,index)
    source=parts(record)
    if record['kind']=='maps':_,weights=load_maps(record);decoded=None
    else:decoded=torch.load(OUT/f'decoded/query{index:03d}.pt',map_location='cpu',weights_only=True);assert decoded['source']==record['payload'];weights=None
    op=read(checked(record['operator']));assert op['query_id']==w['query_id'] and op['candidate_physical_rows']==q['candidate_physical_rows']
    qnorm=F.normalize(q['query_tokens'].to(torch.float64),dim=1)
    for pos,physical in enumerate(q['candidate_physical_rows']):
        dest=d/f'pair{pos:03d}.pt'
        if dest.exists():continue
        if time.monotonic()-started>a['chunk_seconds']:return
        ref=refs[physical];prior=old['candidates'][pos]
        assert C.M.token_sha(ref['tokens'])==ref['tokens_sha256']==prior['reference_tokens_sha256']
        if weights is not None:
            phys,u,v=weights[pos];assert phys==physical
        else:
            z=decoded['records'][pos];assert z['position']==pos and z['physical_row']==physical
            rg,_=refgeom(profile,ref);u=core.cell_means(z['overlaps']['AB'],qg);v=core.cell_means(z['overlaps']['BA'],rg)
        sim=qnorm@F.normalize(ref['tokens'].to(torch.float64),dim=1).T;nu,nv=prior['query_visibility'],prior['reference_visibility']
        loc=S.locals_torch(sim,nv);scores={};err=0.
        for name,uu,vv in [('NATIVE',nu,nv),('FULL_COARSE',u,v)]:
            vals=S.pair_scores(sim,nu,nv,uu,vv,'FULL_UV',loc);ind=S.independent(sim.numpy(),nu.numpy(),nv.numpy(),uu.numpy(),vv.numpy(),'FULL_UV')
            err=max(err,max(abs(vals[k]-ind[k]) for k in vals));assert err<2e-10
            if name=='NATIVE':assert all(float(vals[k]).hex()==float(prior['old_scores'][k]).hex() for k in vals),'NATIVE_SCORE_BITS'
            scores[name]=vals
        save(dest,dict(authority=bind(AUTH),query_id=w['query_id'],position=pos,physical_row=physical,u=u,v=v,scores=scores,max_numpy_error=err,reference_tokens_sha256=ref['tokens_sha256']))
    pairs=[torch.load(d/f'pair{i:03d}.pt',map_location='cpu',weights_only=True) for i in range(128)]
    head=read(checked(a['heads']))['heads'][w['query_id']];theta=torch.tensor([float.fromhex(x) for x in head['theta_hex']],dtype=torch.float64)
    row={k:op[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','winner','challenger_positions')};features={};models={};max_error=0.
    for name in ('NATIVE','FULL_COARSE'):
        evidence={i:p['scores'][name] for i,p in enumerate(pairs)}
        xx=torch.stack([candidate_feature(list(map(float,q['candidate_raw_scores'])),evidence,c,row['winner']) for c in row['challenger_positions']]);zs=xx@theta[:-1]+theta[-1]
        nz=np.sum(xx.numpy()*theta[:-1].numpy(),axis=1)+float(theta[-1]);max_error=max(max_error,float(np.max(abs(nz-zs.numpy()))));assert max_error<2e-10
        pred=dict(logits_hex=E.hx(zs),selected=E.choose(zs,row));assert E.choose(nz,row)==pred['selected']
        if name=='NATIVE':assert pred==head['native'] and torch.equal(xx,torch.tensor(op['modes']['NATIVE']['X'],dtype=torch.float64))
        features[name]=xx;models[name]=pred
    save(d/'features.pt',dict(row=row,theta=theta,X=features,authority=bind(AUTH)))
    write(d/'payload.json',dict(authority=bind(AUTH),index=index,row=row,fold=head['fold'],head=head,models=models,features=bind(d/'features.pt'),pairs=[bind(d/f'pair{i:03d}.pt') for i in range(128)],cache_source=record,label_reads=0,parameters_updated=False))
    write(d/'validation.json',dict(status='COARSE_FIXED_COST1_CPU_PASS',authority=bind(AUTH),payload=bind(d/'payload.json'),max_numpy_logit_error=max_error,max_numpy_scalar_error=max(p['max_numpy_error'] for p in pairs),native127_bit_exact=True))
    print(dict(status='COARSE_QUERY_PASS',index=index,seconds=time.monotonic()-started),flush=True)

def join(a):
    snap=snapshot(a);pred=[]
    for i in snap:
        v=read(OUT/f'query{i:03d}/validation.json');assert v['status']=='COARSE_FIXED_COST1_CPU_PASS' and v['authority']==bind(AUTH);pred.append(read(checked(v['payload'])))
    write(OUT/'prediction_seal.json',dict(authority=bind(AUTH),payloads=[bind(OUT/f'query{i:03d}/payload.json') for i in snap]))
    old={r['query_id']:r for r in read(checked(a['old_result']))['rows']};labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    previous={r['query_id']:r for r in read(checked(a['prior41_result']))['rows']};rows=[]
    for p in pred:
        q=p['row']['query_id'];original=old[q];axis=p['row']['candidate_physical_rows'];target=original['identity'];result={}
        assert original['fold']==p['fold'] and p['models']['NATIVE']['selected']==original['selected']['COST1_FROZEN_NATIVE']
        for name,m in p['models'].items():
            z=np.array([float.fromhex(s) for s in m['logits_hex']]);assert E.choose(z,p['row'])==m['selected']
            result[name]=dict(selected=m['selected'],correct=labels[m['selected']]==target)
        if q in previous:
            prior=previous[q]['models']['FULL_UV/native/COARSE'];assert result['FULL_COARSE']['selected']==prior['selected']
            oldp=read(S.OUT/f"query{p['index']:03d}/payload.json");assert oldp['models']['FULL_UV/native/COARSE']==p['models']['FULL_COARSE'],'PRIOR41_COARSE_BITS'
        rows.append(dict(index=p['index'],query_id=q,original_query_id=original['original_query_id'],component=original['component'],fold=p['fold'],raw_correct=original['correct']['RAW'],target_in_C128=any(labels[x]==target for x in axis),models=result))
    summary={}
    for panel,subset in [('ALL70',rows),('PREVIOUS41',[r for r in rows if r['index']<41]),('ADDED29',[r for r in rows if r['index']>=41])]:
        base=np.array([r['models']['NATIVE']['correct'] for r in subset]);new=np.array([r['models']['FULL_COARSE']['correct'] for r in subset]);raw=np.array([r['raw_correct'] for r in subset]);groups=sorted({r['component'] for r in subset})
        bygroup=np.array([[sum(int(r['models']['FULL_COARSE']['correct'])-int(r['models']['NATIVE']['correct']) for r in subset if r['component']==g),sum(r['component']==g for r in subset)] for g in groups])
        rng=np.random.default_rng(20260923);sample=bygroup[rng.integers(len(groups),size=(10000,len(groups)))].sum(1);ci=np.quantile(sample[:,0]/sample[:,1],[.025,.975]).tolist()
        summary[panel]=dict(n=len(subset),groups=len(groups),raw=int(raw.sum()),native=int(base.sum()),coarse=int(new.sum()),rescue_vs_native=int((new&~base).sum()),break_vs_native=int((~new&base).sum()),rescue_vs_raw=int((new&~raw).sum()),break_vs_raw=int((~new&raw).sum()),target_absent=sum(not r['target_in_C128'] for r in subset),group_bootstrap_accuracy_delta_ci95=ci,changed_decisions=sum(r['models']['NATIVE']['selected']!=r['models']['FULL_COARSE']['selected'] for r in subset))
    payload=dict(status='COARSE_CACHED70_RECOUNT_PASS',authority=bind(AUTH),summary=summary,rows=rows,probe=bind(OUT/'probe_payload.json'),training=False,exploratory=True,full593_replaced=False)
    write(OUT/'result.json',payload);write(OUT/'validation.json',dict(status=payload['status'],result=bind(OUT/'result.json'),all70_queries_validated=True))
    timing=read(OUT/'probe_payload.json')
    lines=['# 粗阶段真实提前退出与70张冻结头回放','',f"匹配阶段GPU实测中位耗时：{timing['median_seconds']}；逐对加速比中位数：{timing['paired_speedup_median']:.3f}。这是模型加载及PIL解码以外的匹配计算，不是端到端检索耗时。8对原matcher输出及token权重逐位一致，fine_features/refiner调用为0。",'', '|面板|RAW|原COST1|粗阶段COST1|对原头救/损|','|---|---:|---:|---:|---:|']
    for k,s in summary.items():lines.append(f"|{k}|{s['raw']}/{s['n']}|{s['native']}/{s['n']}|{s['coarse']}/{s['n']}|{s['rescue_vs_native']}/{s['break_vs_native']}|")
    lines+=['','原生全部127 logits逐位复现，前41张粗阶段logits亦与已封存回放逐位一致。原五折头、C128、阈值固定，无训练。70张依缓存完整性选择，属于已打开H593的探索子集；29张新增子集单列，不能当全593或独立外部GO。分组重采样区间见result.json；若无观察到的损失，区间退化不构成总体无损证明。']
    (ROOT/'reports/REPORT_H593_COARSE_EXIT_V1_20260923.md').write_text('\n'.join(lines)+'\n')
    print(payload['summary'],flush=True)

def submit(launcher,stage,indices=None,dependency=None):
    args=['sbatch','--parsable','--hold']
    if indices is not None:args+=['--array='+','.join(map(str,indices))+'%50']
    if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    args += [str(launcher),stage]
    job=subprocess.run(args,capture_output=True,text=True,check=True).stdout.strip().split(';')[0];assert job.isdigit()
    spool=OUT/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scontrol','write','batch_script',job,str(spool)],capture_output=True,text=True,check=True);assert spool.read_bytes()==launcher.read_bytes()
    return job,args,bind(spool)

def control(a,previous):
    lock=(OUT/'control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'validation.json').exists():return
    if previous:
        z=subprocess.run(['sacct','-X','-n','-P','-j',previous,'--format=State,ExitCode'],capture_output=True,text=True,check=True).stdout.splitlines()
        assert z and all(line.split('|')[:2]==['COMPLETED','0:0'] for line in z),z
    waves=[read(p) for p in sorted((OUT/'waves').glob('*.json'))];counts=Counter(i for w in waves for i in w.get('indices',[]));probes=sum(w['stage']=='probe' for w in waves)
    if not (OUT/'probe_validation.json').exists():
        assert probes<a['max_chunks'];stage='probe';indices=None;launcher=GPU
    else:
        complete={i for i in snapshot(a) if (OUT/f'query{i:03d}/validation.json').exists()}
        if len(complete)==70:join(a);return
        indices=[0] if 0 not in complete else sorted(set(snapshot(a))-complete,key=lambda i:(counts[i],i))[:50]
        assert all(counts[i]<a['max_chunks'] for i in indices)
        stage='worker';launcher=CPU
    job,args,spool=submit(launcher,stage,indices)
    callback,cargs,cspool=submit(CPU,'control:'+job,dependency=job)
    write(OUT/'waves'/f'{len(waves):04d}.json',dict(job_id=job,stage=stage,indices=indices or [],args=args,spool=spool,callback=callback,callback_args=cargs,callback_spool=cspool,authority=bind(AUTH)))
    for j in (callback,job):subprocess.run(['scontrol','release',j],capture_output=True,text=True,check=True)
    print(dict(submitted=job,stage=stage,indices=indices,callback=callback),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        stage=args.stage.split(':')[0];a=guard(stage,args.index)
        if stage=='probe':probe(a)
        elif stage=='worker':worker(a,args.index)
        elif stage=='join':join(a)
        elif stage=='control':control(a,args.stage.partition(':')[2] or None)
        else:raise ValueError(stage)
