#!/usr/bin/env python3
"""Full-C128 M-origin experiments, independent verification and bounded continuation."""
import argparse
import collections
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_roma_visual_origin_v1 as V
from rc_roma_exact_feature_cache_v2 import equal_tree

OUT = ROOT/'results/rc_h593_m_visual_origin_v1'
AUTH = ROOT/'registry/rc_h593_m_visual_origin_authority_v1_20260922.json'
PLAN = ROOT/'plan/RC_H593_M_VISUAL_ORIGIN_V1_20260922.md'
GPU = ROOT/'slurm/rc_h593_m_visual_origin_v1.sbatch'
CPU = ROOT/'slurm/rc_h593_m_visual_origin_control_v1.sbatch'
read, write, save, bind, checked = C.read, C.write, C.save, C.bind, C.checked


def prepare():
    assert not AUTH.exists()
    test = V.self_test(); old = read(C.AUTH)
    sources = [Path(__file__), Path(V.__file__), Path(C.__file__),
        ROOT/'programs/rc_roma_exact_feature_cache_v2.py', PLAN, GPU, CPU]
    evaluation = ROOT/'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json'
    ea = read(evaluation)
    write(AUTH, dict(status='M_VISUAL_ORIGIN_AUTHORIZED',
        user_authorization='2026-09-22 user explicitly requested localization of M visual source',
        sources=[bind(p) for p in sources], profile=old['profile'], workers=old['workers'],
        operator_authority=ea['operator_authority'], gallery=ea['public_sources']['gallery'],
        old_result=bind(ROOT/'results/rc_h593_quality_operator_eval_v1/result.json'),
        old_validation=bind(ROOT/'results/rc_h593_quality_operator_eval_v1/validation.json'),
        arms=list(V.ARMS), stages=list(V.STAGES), queries=593, candidates=128,
        chunk_seconds=600, part_pairs=8, max_chunks=32, max_parallel=46,
        control='Only RoMa image tensors modified; ColNomic content scores and RAW candidates fixed',
        scope='Opened H593 descriptive intervention, no fitting or accuracy-selected configurations',
        external_GO=False))
    test['authority'] = bind(AUTH); write(OUT/'preflight.json', test)
    print(dict(status='PREPARED', authority=bind(AUTH)), flush=True)


def guard(stage, index):
    a = read(AUTH); assert a['status'] == 'M_VISUAL_ORIGIN_AUTHORIZED'
    for b in a['sources']: checked(b)
    checked(a['profile']); workers = read(checked(a['workers']))['records']
    p = read(OUT/'preflight.json'); assert p['authority'] == bind(AUTH) and p['status'] == 'VISUAL_ORIGIN_TRANSFORMS_PASS'
    allowed = {Path(a['workers']['path']).resolve(), Path(a['profile']['path']).resolve()}
    if stage in ('worker','verify'):
        assert index in range(593) and os.environ.get('SLURM_JOB_ID')
        for kind in ('raw','roma'):
            allowed.update(Path(b['path']).resolve() for b in workers[index][kind].values())
        allowed.update((ROOT/f'results/rc_h593_quality_operator_v1/query{index:03d}'/n).resolve() for n in ('validation.json','payload.json'))
        if index:
            val = read(OUT/'query000/validation.json'); checked(val['payload'])
            assert val['authority'] == bind(AUTH) and val['status'] == 'M_VISUAL_ORIGIN_QUERY_PASS'
    if stage == 'join':
        allowed.update(Path(a[k]['path']).resolve() for k in ('gallery','old_result','old_validation'))
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve(); s = str(p).lower()
        assert not any(k in s for k in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/')), s
        if stage in ('worker','verify'):
            assert '/reports/' not in s, s
        if ROOT/'results' in p.parents:
            assert OUT in p.parents or p in allowed, s
    sys.addaudithook(audit)
    return a, workers


def worker(a, w, index):
    folder = OUT/f'query{index:03d}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'validation.json').exists():
        checked(read(folder/'validation.json')['payload']); return
    started = time.monotonic(); q, old, refs = C.load_input(w)
    profile = read(C.PROFILE); C.check_profile(profile)
    core = C.M.legacy_core(profile); model = C.M.gpu_model(profile)
    assert model.threshold is None and model.H_lr == model.W_lr == 800 and model.H_hr == model.W_hr == 1280 and model.bidirectional
    opval = read(ROOT/f'results/rc_h593_quality_operator_v1/query{index:03d}/validation.json')
    operator = read(checked(opval['payload'])); assert operator['query_id'] == w['query_id'] and operator['authority'] == a['operator_authority']
    qp = Path(q['query_source_path']); assert bind(qp)['sha256'] == q['query_source_sha256']
    qgeom, qmeta = C.M.geometry(qp, q['query_source_sha256'], 'm-origin-query', q['query_grid_shape'], q['processor_input_frame'], profile['sources']['processor']['sha256'])
    qi = core.oriented(qp); input_sources = {}; current = {}
    def sink(image_sha, tag, lr, hr):
        key = image_sha+'_'+tag
        if key in input_sources: return
        path = OUT/'inputs'/f'{key}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                value = dict(authority=bind(AUTH), image_sha256=image_sha, transform=tag,
                    original_image=current[image_sha], low_resolution_shape=list(lr.shape), high_resolution_shape=list(hr.shape),
                    low_resolution_thumbnail64=torch.nn.functional.interpolate(lr, size=(64,64), mode='area').cpu(),
                    high_resolution_thumbnail64=torch.nn.functional.interpolate(hr, size=(64,64), mode='area').cpu())
                save(path,value)
            else:
                value=torch.load(path,map_location='cpu',weights_only=True)
                assert value['authority']==bind(AUTH) and value['image_sha256']==image_sha and value['transform']==tag
        input_sources[key] = bind(path)
    current[q['query_source_sha256']] = dict(path=str(qp),sha256=q['query_source_sha256'])
    observer = V.Observer(model, core, sink)
    torch.cuda.reset_peak_memory_stats(); parts = []
    try:
        for part in range(16):
            path = folder/f'part{part:02d}.pt'
            if path.exists():
                value = torch.load(path, map_location='cpu', weights_only=True)
                assert value['authority'] == bind(AUTH) and value['query_id'] == w['query_id'] and value['part'] == part
                parts.append(bind(path)); continue
            if time.monotonic()-started > a['chunk_seconds']:
                write(folder/f"chunk_{os.environ['SLURM_JOB_ID']}_{part:02d}.json",dict(status='M_ORIGIN_NORMAL_PARTIAL',authority=bind(AUTH),completed_parts=len(parts),index=index))
                print(dict(status='PARTIAL', index=index, parts=len(parts)),flush=True); return
            pairs = []
            for pos in range(part*8,(part+1)*8):
                physical = q['candidate_physical_rows'][pos]; ref = refs[physical]; prior = old['candidates'][pos]
                assert physical == prior['physical_row'] and ref['tokens_sha256'] == prior['reference_tokens_sha256']
                rp = Path(ref['source_path']); assert bind(rp)['sha256'] == ref['source_image_sha256']
                current[ref['source_image_sha256']] = dict(path=str(rp),sha256=ref['source_image_sha256'])
                rg, rmeta = C.M.geometry(rp, ref['source_image_sha256'], 'm-origin-reference', ref['grid_shape'], 'DECODED_RAW_BEFORE_EXIF', profile['sources']['processor']['sha256'])
                ri = core.oriented(rp); arms = {}
                for arm in V.ARMS:
                    observer.select(arm,qgeom,rg,q['query_source_sha256'],ref['source_image_sha256'])
                    if index == pos == 0 and arm == 'NATIVE':
                        observer.close(); plain = model.match(qi,ri)
                        observer = V.Observer(model,core,sink)
                        observer.select(arm,qgeom,rg,q['query_source_sha256'],ref['source_image_sha256'])
                    torch.cuda.synchronize(); t = time.monotonic(); pred = model.match(qi,ri); torch.cuda.synchronize()
                    seconds = time.monotonic()-t
                    if index == pos == 0 and arm == 'NATIVE':
                        assert equal_tree(plain,pred), 'INSTRUMENTATION_CHANGED_NATIVE'; del plain
                        write(folder/'instrumentation_parity.json',dict(status='M_ORIGIN_FULL_DENSE_NATIVE_BITS_PASS',authority=bind(AUTH)))
                    stages = observer.stages
                    for stage, sides in stages.items():
                        sides['M'] = float(torch.sqrt(sides['AB']['weights'].mean()*sides['BA']['weights'].mean()))
                    if arm == 'NATIVE':
                        assert C.M.bit_equal(stages['HR1']['AB']['weights'],prior['query_visibility'])
                        assert C.M.bit_equal(stages['HR1']['BA']['weights'],prior['reference_visibility'])
                        assert stages['HR1']['M'].hex() == float(prior['old_scores']['visibility_mass']).hex(), 'NATIVE_M'
                    arms[arm] = dict(stages=stages,seconds=seconds)
                    del pred
                pairs.append(dict(candidate_position=pos,physical_row=physical,
                    reference_image=dict(path=str(rp),sha256=ref['source_image_sha256']),reference_geometry=rmeta,
                    free_content=float(operator['modes']['M0Q0R0']['scores'][pos]['real_score']),arms=arms,
                    inputs={key:source for key,source in input_sources.items() if key.split('_',1)[0] in (q['query_source_sha256'],ref['source_image_sha256'])}))
            save(path,dict(authority=bind(AUTH),query_id=w['query_id'],part=part,pairs=pairs))
            parts.append(bind(path))
            print(dict(event='M_ORIGIN_PART_SAVED',index=index,pairs=(part+1)*8,seconds=time.monotonic()-started),flush=True)
    finally:
        observer.close()
    # Build one compact score table without copying the stage tensors into JSON.
    scores = {arm:{stage:[] for stage in V.STAGES} for arm in V.ARMS}
    for source in parts:
        value = torch.load(checked(source),map_location='cpu',weights_only=True)
        for pair in value['pairs']:
            input_sources.update(pair['inputs'])
            for arm in V.ARMS:
                for stage in V.STAGES: scores[arm][stage].append(pair['arms'][arm]['stages'][stage]['M'])
    write(folder/'payload.json',dict(status='M_ORIGIN_ALL128_SEALED',authority=bind(AUTH),
        query_id=w['query_id'],execution_ordinal=index,query_image=current[q['query_source_sha256']],query_geometry=qmeta,
        candidate_physical_rows=q['candidate_physical_rows'],raw_winner=operator['winner'],
        parts=parts,masses=scores,operator_source=opval['payload'],input_manifest=input_sources,
        label_reads=0,model_updates=0))
    write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,
        descriptor_cache=dict(observer.stats),gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)


def verify(a,w,index):
    import numpy as np
    folder=OUT/f'query{index:03d}'; p=read(folder/'payload.json')
    assert p['authority']==bind(AUTH) and p['query_id']==w['query_id']
    q,old,refs=C.load_input(w); positions=[]; error=0.;count=0
    for source in p['parts']:
        part=torch.load(checked(source),map_location='cpu',weights_only=True)
        assert part['authority']==bind(AUTH) and part['query_id']==w['query_id']
        for pair in part['pairs']:
            pos=pair['candidate_position']; positions.append(pos)
            assert pair['physical_row']==q['candidate_physical_rows'][pos]
            assert set(pair['arms'])==set(V.ARMS)
            for arm in V.ARMS:
                assert set(pair['arms'][arm]['stages'])==set(V.STAGES)
                for stage,sides in pair['arms'][arm]['stages'].items():
                    u=sides['AB']['weights'];v=sides['BA']['weights']
                    assert u.dtype==v.dtype==torch.float64 and len(u)==len(q['query_tokens']) and len(v)==len(refs[pair['physical_row']]['tokens'])
                    for x in (u,v): assert bool(torch.isfinite(x).all()) and bool(((x>=0)&(x<=1)).all())
                    m=math.sqrt(float(np.mean(u.numpy()))*float(np.mean(v.numpy())))
                    err=abs(m-sides['M']);error=max(error,err);assert err<2e-12
                    assert sides['M']==p['masses'][arm][stage][pos];count+=1
                if arm=='NATIVE':
                    native=pair['arms'][arm]['stages']['HR1'];prior=old['candidates'][pos]
                    assert C.M.bit_equal(native['AB']['weights'],prior['query_visibility']) and C.M.bit_equal(native['BA']['weights'],prior['reference_visibility'])
                    assert native['M']==float(prior['old_scores']['visibility_mass'])
    assert positions==list(range(128)) and p['candidate_physical_rows']==q['candidate_physical_rows']
    if index==0: assert read(folder/'instrumentation_parity.json')['status']=='M_ORIGIN_FULL_DENSE_NATIVE_BITS_PASS'
    for source in p['input_manifest'].values(): checked(source)
    write(folder/'validation.json',dict(status='M_VISUAL_ORIGIN_QUERY_PASS',authority=bind(AUTH),
        payload=bind(folder/'payload.json'),candidates=128,independent_masses=count,max_error=error,
        native_final_bit_exact=True,label_reads=0,model_updates=0))
    print(dict(status='QUERY_PASS',index=index,masses=count),flush=True)


def control(a, workers):
    assert os.environ.get('SLURM_JOB_ID')
    (OUT/'dispatch').mkdir(parents=True,exist_ok=True)
    waves=sorted((OUT/'dispatch').glob('wave*.json')); attempts=collections.Counter()
    # Each controller is submitted afterany its preceding array; no background polling.
    if waves:
        wave=read(waves[-1]); job=wave['job_id']
        proc=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'],capture_output=True,text=True,check=True)
        statuses=[line.split('|')[:3] for line in proc.stdout.splitlines() if line.strip()]
        assert statuses and all(state=='COMPLETED' and code=='0:0' for _,state,code in statuses), ('PREVIOUS_WAVE_NOT_NORMAL',statuses)
    for path in waves:
        for i in read(path)['indices']:attempts[i]+=1
    complete=[]
    for i in range(593):
        path=OUT/f'query{i:03d}/validation.json'
        if path.exists():
            v=read(path);assert v['status']=='M_VISUAL_ORIGIN_QUERY_PASS' and v['authority']==bind(AUTH);checked(v['payload']);complete.append(i)
    if len(complete)==593:
        args=['sbatch','--parsable',str(CPU),'join']
        job=subprocess.run(args,capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
        write(OUT/'join_submitted.json',dict(job_id=job,authority=bind(AUTH)));return
    todo=[0] if 0 not in complete else sorted((i for i in range(1,593) if i not in complete),key=lambda i:(attempts[i],i))[:46]
    assert all(attempts[i]<a['max_chunks'] for i in todo), 'MAX_CHUNKS_REACHED'
    args=['sbatch','--parsable','--hold','--array='+','.join(map(str,todo))+'%46']
    if todo==[0]:args+=['--partition=dev_accelerated']
    args += [str(GPU)]
    job=subprocess.run(args,capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
    assert job.isdigit()
    follow=None
    try:
        spool=OUT/'dispatch'/f'spool_{job}.sh'
        subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True,text=True)
        assert spool.read_bytes()==GPU.read_bytes(), 'SUBMITTED_SCRIPT_DRIFT'
        follow=subprocess.run(['sbatch','--parsable','--dependency=afterany:'+job,str(CPU),'control'],capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
        assert follow.isdigit()
        write(OUT/'dispatch'/f'wave{len(waves):04d}.json',dict(job_id=job,afterany_controller=follow,indices=todo,
            authority=bind(AUTH),script=bind(spool),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True)
    except BaseException:
        subprocess.run(['scancel',job],check=False)
        if follow:subprocess.run(['scancel',follow],check=False)
        raise
    print(dict(status='M_ORIGIN_WAVE_SUBMITTED',job_id=job,afterany_controller=follow,indices=todo,complete=len(complete)),flush=True)


def join(a):
    # All 593 observation seals are validated before reading identity labels.
    observed=[];sources=[]
    for i in range(593):
        v=read(OUT/f'query{i:03d}/validation.json');p=read(checked(v['payload']))
        assert v['authority']==p['authority']==bind(AUTH) and v['status']=='M_VISUAL_ORIGIN_QUERY_PASS'
        observed.append(p);sources.append(bind(OUT/f'query{i:03d}/validation.json'))
    oldval=read(checked(a['old_validation']));assert oldval['result']==a['old_result']
    old={r['query_id']:r for r in read(checked(a['old_result']))['rows']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    rows=[]
    for p in observed:
        r=old[p['query_id']];axis=p['candidate_physical_rows'];target=[i for i,v in enumerate(axis) if labels[v]==r['identity']]
        assert len(target)<=1 and bool(target)==r['target_in_C128']
        row=dict(query_id=p['query_id'],execution_ordinal=p['execution_ordinal'],fold=r['fold'],component=r['component'],
            raw_correct=r['correct']['RAW'],simple_correct=r['correct']['COST1_REFIT_M1Q0R0'],target_in_C128=bool(target),arms={})
        for arm in V.ARMS:
            row['arms'][arm]={}
            for stage in V.STAGES:
                ms=p['masses'][arm][stage];best=max(range(128),key=lambda i:(ms[i],-axis[i]));win=p['raw_winner']
                record=dict(direct_correct=best in target,selected_physical_row=axis[best],candidate_mean=sum(ms)/128)
                if target:
                    t=target[0];wrong=[i for i in range(128) if i!=t]
                    strongest=max(ms[i] for i in wrong)
                    record.update(target_M=ms[t],raw_M=ms[win],target_minus_raw=ms[t]-ms[win],
                        target_minus_strongest_wrong=ms[t]-strongest,
                        normalized_target_vs_wrong=(ms[t]-strongest)/(abs(ms[t])+abs(strongest)+1e-12),
                        normalized_target_vs_raw=(ms[t]-ms[win])/(abs(ms[t])+abs(ms[win])+1e-12),
                        target_rank=1+sum(ms[i]>ms[t] for i in wrong))
                row['arms'][arm][stage]=record
        rows.append(row)
    groups={'ALL593':rows,'RAW_CORRECT':[r for r in rows if r['raw_correct']],
        'RAW_WRONG_RECALL_PRESENT':[r for r in rows if not r['raw_correct'] and r['target_in_C128']],
        'SIMPLE_RESCUES':[r for r in rows if r['simple_correct'] and not r['raw_correct']]}
    summary={}
    for group,rs in groups.items():
        summary[group]={'n':len(rs),'arms':{}}
        for arm in V.ARMS:
            summary[group]['arms'][arm]={}
            for stage in V.STAGES:
                items=[r['arms'][arm][stage] for r in rs];valid=[x for x in items if 'target_M'in x]
                summary[group]['arms'][arm][stage]=dict(direct_correct=sum(x['direct_correct'] for x in items),
                    target_gt_strongest_wrong=sum(x['target_minus_strongest_wrong']>0 for x in valid),
                    mean_normalized_target_gap=(sum(x['normalized_target_vs_wrong'] for x in valid)/len(valid)) if valid else None,
                    mean_target_gap=(sum(x['target_minus_strongest_wrong'] for x in valid)/len(valid)) if valid else None)
    assert summary['ALL593']['arms']['NATIVE']['HR1']['direct_correct']==379
    write(OUT/'result.json',dict(status='M_VISUAL_ORIGIN_ALL593_COMPLETE',authority=bind(AUTH),query_validations=sources,
        summary=summary,rows=rows,external_GO=False,scope=a['scope']))
    write(OUT/'validation.json',dict(status='M_ORIGIN_JOIN_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),queries=593))
    print(dict(status='ALL593_COMPLETE',result=bind(OUT/'result.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','worker','verify','control','join'));parser.add_argument('--index',type=int)
    args=parser.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        authority,workers=guard(args.stage,args.index)
        if args.stage=='worker':worker(authority,workers[args.index],args.index)
        elif args.stage=='verify':verify(authority,workers[args.index],args.index)
        elif args.stage=='control':control(authority,workers)
        else:join(authority)
