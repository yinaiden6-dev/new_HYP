#!/usr/bin/env python3
"""Frozen full-C128 property accounting, with bounded CPU context replay.

Historical phase/amplitude GPU outputs are reused, never mixed with new CPU
context cells in a causal contrast. The two stages have separate native anchors.
No training, image encoding, matching, refinement, LLM or GPU forward is run.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
OUT=RC/'results/rc_m_fine_c128_attribution_v1'
OLD=RC/'results/rc_roma_position_factor_v1'
BRIDGE=RC/'results/rc_m_causal128_attribution_chain_v1/bridge'
PHASE=RC/'results/rc_m_phase_external_internal_bridge_v2'
INDICES=[0,1,2,4,5,6,7,8,9,21,32]
MAIN=INDICES[:8]
CASES=[9,21,32]
DIRECTIONS=['X_PLUS','X_MINUS','Y_PLUS','Y_MINUS']
SHIFTS={'X_PLUS':(0,4),'X_MINUS':(0,-4),'Y_PLUS':(4,0),'Y_MINUS':(-4,0)}
CONTEXT=['NATIVE']+[d+'__'+b for d in DIRECTIONS for b in ['001','110','111']]
PHASE_ARMS=['NATIVE','ZERO','AMP_FLAT','AMP_PERMUTE']+[f'{scope}_{d}' for d in DIRECTIONS for scope in ['GLOBAL','LOCAL']]
MODELS=['NATIVE7','M_FREE','POST']


def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())
def checked(b):
    assert bind(b['path'])=={k:b[k] for k in ['path','sha256']},('SOURCE_DRIFT',b['path'])
    return Path(b['path'])
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text,('IMMUTABLE',str(p));return
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(text)
    try:os.link(tmp,p)
    finally:tmp.unlink(missing_ok=True)
def save(p,d,torch):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    try:torch.save(d,tmp);os.link(tmp,p)
    finally:tmp.unlink(missing_ok=True)
def configure():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    return torch
def emit(**x):print(json.dumps(x,allow_nan=False),flush=True)
def scalar_mass(branch):
    means=[math.fsum(x['weights'].tolist())/x['weights'].numel() for x in branch]
    return math.sqrt(means[0]*means[1])


def prepare(root,torch):
    old=read(OLD/'manifest.json');ov=read(OLD/'validation.json');bp=read(BRIDGE/'protocol.json')
    assert old['indices']==INDICES and old['outcome_independent_panel']==MAIN and old['selected_rescue_audit']==CASES
    assert ov['status']=='POSITION_FACTOR_INDEPENDENT_CPU_AND_PARITY_PASS'
    inputs={r['index']:r for r in read(checked(bp['inputs']))['rows']}
    oldphase=read(PHASE/'protocol.json');pv=read(PHASE/'validation.json');assert 'PASS' in pv['status']
    phaseitems={r['index']:r for r in read(checked(oldphase['inputs']))['rows']}
    queries=[];pairs=[]
    for ordinal,index in enumerate(INDICES):
        item=inputs[index];w=next(w for w in old['workers'] if w['execution_ordinal']==index)
        folder=OLD/f'query{index:03d}';gv=read(folder/'gpu_validation.json');cv=read(folder/'cpu_validation.json')
        cp=read(checked(cv['payload']));assert cp['query_id']==item['query_id'] and len(gv['pairs'])==128
        axis=item['post_row']['candidate_ids'];assert axis==gv['candidate_physical_rows']==cp['row']['candidate_physical_rows']
        assert np.array_equal(cp['theta'],item['external']['native_theta'])
        masses={a:[cp['scores']['M_ONLY/'+a][str(pos)]['visibility_mass'] for pos in range(128)] for a in PHASE_ARMS}
        reuse=None
        if index in MAIN:
            pi=MAIN.index(index);rv=read(PHASE/'queries'/f'{pi:02d}'/'independent_validation.json')
            rr=read(checked(rv['result']));assert rr['query_id']==item['query_id']
            for a in PHASE_ARMS:
                olda=a if a=='NATIVE' else a+'/FULL'
                assert np.max(abs(np.asarray(masses[a])-phaseitems[index]['masses'][olda]))<1e-14
            reuse=dict(validation=bind(PHASE/'queries'/f'{pi:02d}'/'independent_validation.json'),result=rv['result'])
        rp=torch.load(checked(w['roma']['payload']),weights_only=True,mmap=True,map_location='cpu')
        r=rp['records'][w['source_index']];assert r['query_id']==w['source_query_id'] and r['candidate_physical_rows']==axis
        for pos,b in enumerate(gv['pairs']):
            assert Path(b['path']).exists()
            # Capture/descriptor hashes are checked by the bounded worker. The
            # original validated pair receipt freezes their existing bindings.
            pairs.append(dict(pair_index=len(pairs),index=index,ordinal=ordinal,position=pos,physical_row=axis[pos],
                old_pair=b,geometry_metadata=[r['query_geometry'],r['candidates'][pos]['reference_geometry']],
                image_shas=[r['query_source_sha256'],r['candidates'][pos]['reference_image_sha256']]))
        queries.append(dict(index=index,ordinal=ordinal,query_id=item['query_id'],cohort='MAIN8' if index in MAIN else 'SELECTED_CASES3',
            item=item,phase_M=masses,phase_reuse=reuse,old_gpu_validation=bind(folder/'gpu_validation.json'),
            old_cpu_validation=bind(folder/'cpu_validation.json'),old_cpu_payload=cv['payload'],
            original_bridge=bind(BRIDGE/'queries'/f'{index:03d}'/'result.json')))
        del rp
    write(root/'inputs.json',dict(queries=queries,pairs=pairs,labels_included=False))
    import rc_m_context_binding_v1 as C
    code=[Path(__file__),Path(C.__file__),RC/'programs/probe_rc_roma_property_restore_cpu_v3.py',
          RC/'programs/run_rc_m_phase_newgroups_cpu_v1.py',RC/'programs/rc_roma_shared_native_cache_v1.py',
          RC/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py',
          RC.parents[2]/'third_party/RoMaV2/src/romav2/dpt.py']
    code+=[Path(x['path']) for x in bp['code_sources']]
    p=dict(status='FINE_C128_PROPERTY_PROTOCOL_FROZEN',version=1,inputs=bind(root/'inputs.json'),
        old_manifest=bind(OLD/'manifest.json'),old_validation=bind(OLD/'validation.json'),
        phase_protocol=bind(PHASE/'protocol.json'),phase_validation=bind(PHASE/'validation.json'),
        bridge_protocol=bind(BRIDGE/'protocol.json'),bridge_validation=bind(BRIDGE/'validation.json'),
        label_source=bp['original_label_source'],head_state=read(OLD/'pilot_validation.json')['head_state'],
        code_sources=[bind(x) for x in dict.fromkeys(code)],
        indices=INDICES,main=MAIN,cases=CASES,queries=11,candidates=128,pairs=1408,shards=50,
        phase_arms=PHASE_ARMS,context_arms=CONTEXT,directions=SHIFTS,threads=4,torch_version=str(torch.__version__),
        phase_backend='Reuse sealed historical GPU DPT+sigmoid+FP64 pooling outputs for every phase/amplitude cell; no new RoMa forward.',
        context_backend='All thirteen context cells use frozen CPU DPT on original retained A/J/P; own same-CPU NATIVE anchor. Never form a mixed-backend causal contrast.',
        fine_property_terminology='Fine means finer mechanistic property decomposition at the COARSE DPT endpoint, not RoMa fine-matching/refiner-stage inference.',
        native_parity='Repeat CPU native confidence exactly for the first formal pair; keep CPU versus historical GPU confidence/M drift descriptive for every pair.',
        primary_context='Four-direction mean of (000+111-001-110)/2, each cell scored over every natural C128 candidate. Also retain 110-000,001-000,111-000.',
        property_contrasts='Native-amplitude GLOBAL minus LOCAL phase coherence, NATIVE minus AMP_PERMUTE and AMP_FLAT amplitude interventions, matched context alignment. Conditional effects overlap, not additive unique shares.',
        cohort_policy='Eight historically outcome-independent group representatives are primary; three historically selected original rescues remain case diagnostics, never pooled as an unbiased sample.',
        mediator='M-only: original local weighting shape/content fixed; all M-scaled descendants updated together; POST uses real M with frozen original held-fold adapter/head.',
        scope='Opened historical11, complete natural C128. Not all128 queries, not all593 and not untouched confirmation. No inference of universal necessity, unique cause, or all original benefit attribution.',
        new_gpu_or_encoder_matcher_LLM_forwards=0,new_training=0,all_intermediate_outputs_retained=True)
    write(root/'protocol.json',p)
    write(root/'preparation_validation.json',dict(status='FINE_C128_PREPARED',protocol=bind(root/'protocol.json'),
        queries=11,candidates_per_query=128,pairs=1408,phase_candidate_arms_reused=16896,context_candidate_arms=18304,
        phase_full_query_reuse=8,phase_POST_query_completion=3,labels_used_for_new_selection=False))
    emit(status='FINE_C128_PREPARED')


def guard(root,torch):
    p=read(root/'protocol.json');assert p['status']=='FINE_C128_PROPERTY_PROTOCOL_FROZEN'
    assert p['threads']==torch.get_num_threads()==4 and p['torch_version']==str(torch.__version__)
    for b in p['code_sources']:checked(b)
    for k in ['inputs','old_manifest','old_validation','phase_protocol','phase_validation','bridge_protocol','bridge_validation','head_state']:checked(p[k])
    return p,bind(root/'protocol.json'),read(p['inputs']['path'])
def original_arms(q):
    old=read(checked(q['original_bridge']))
    return old['worlds']['ORIGINAL_HR1'],old['head']
def all_scores(item,masses,content,head):
    from run_rc_m_phase_bridge_v2 import external
    from analyze_rc_postllm_signal_decomposition_v2 import decision
    out={}
    for a,m in masses.items():
        v=np.asarray(content[a]);post=decision(item['post_row'],v,head);z=np.zeros(128)
        z[item['post_row']['challenger_positions']]=post['logits']
        out[a]=dict(M=m,L=v.tolist(),POST=dict(logits128=z.tolist(),prediction_position=post['prediction_position'],
            switched=post['prediction_position']!=item['post_row']['winner_index']),external=external(item['external'],m))
    return out
def independent_scores(item,worlds,head):
    ext=item['external'];w=ext['winner'];cs=ext['challengers'];raw=np.asarray(item['post_row']['raw_scores']);maximum=0.
    def sym(a,b):return (a-b)/(abs(a)+abs(b)+1e-12)
    for arm,d in worlds.items():
        m=np.asarray(d['M']);L=np.asarray(d['L']);free=np.asarray(ext['free_content'])
        s,q,r=(np.asarray(ext['native_scores'])*(m/np.asarray(ext['original_M']))[:,None]).T
        lc=s/np.maximum(m,1e-12);z={k:np.zeros(128) for k in MODELS}
        for j,c in enumerate(cs):
            rawcol=ext['native_X'][j][0]
            x=[rawcol,sym(s[c],s[w]),sym(m[c],m[w]),sym(lc[c],lc[w]),sym(s[c]-q[c],s[w]-q[w]),sym(s[c]-r[c],s[w]-r[w])]
            y=[rawcol,sym(m[c]*free[c],m[w]*free[w]),sym(m[c],m[w]),sym(free[c],free[w]),0.,0.]
            for model,features,theta in [('NATIVE7',x,ext['native_theta']),('M_FREE',y,ext['simple_theta'])]:
                z[model][c]=math.fsum(a*b for a,b in zip(features,theta[:-1]))+theta[-1]
            v=[(raw[c]-raw[w])/max(float(raw.std()),1e-12),sym(L[c],L[w]),1.]
            z['POST'][c]=math.fsum(a*b for a,b in zip(v,head))
        for model in MODELS:
            actual=d['POST'] if model=='POST' else d['external'][model]
            e=float(np.max(abs(z[model]-actual['logits128'])));maximum=max(maximum,e);assert e<2e-10
            best=max(cs,key=lambda k:z[model][k]);pred=best if z[model][best]>0 else w
            assert pred==actual['prediction_position']
    return maximum


def score_worker(root,index,budget,torch,phase=False):
    p,pb,inputs=guard(root,torch);assert 0<=index<11;q=inputs['queries'][index];item=q['item']
    family='phase' if phase else 'context_bridge';folder=root/family/f"query{q['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json');assert v['protocol']==pb;checked(v['result']);return 0
        deadline=time.monotonic()+budget;origin,head=original_arms(q);patches=[];reuse=None
        if phase and q['phase_reuse']:
            reuse=q['phase_reuse'];checked(reuse['validation']);old=read(checked(reuse['result']))
            masses=q['phase_M'];content={a:old['worlds'][a if a=='NATIVE' else a+'/FULL']['L'] for a in PHASE_ARMS}
            arms=all_scores(item,masses,content,head)
            for a in PHASE_ARMS:
                source=old['worlds'][a if a=='NATIVE' else a+'/FULL']
                for model in MODELS:
                    one=source['POST'] if model=='POST' else source['external'][model]
                    other=arms[a]['POST'] if model=='POST' else arms[a]['external'][model]
                    assert np.max(abs(np.asarray(one['logits128'])-other['logits128']))<2e-10
        else:
            if phase:masses=q['phase_M'];arm_names=PHASE_ARMS
            else:
                vp=root/'context_validation.json'
                if not vp.exists():return 75
                mv=read(vp);assert mv['status']=='FINE_C128_CONTEXT_MASSES_PASS' and mv['protocol']==pb
                mr=read(checked(mv['result']));masses=next(x['M'] for x in mr['rows'] if x['index']==q['index']);arm_names=CONTEXT
            import run_rc_m_causal128_frozen_bridge_v1 as B
            context=B.Context(read(checked(p['bridge_protocol'])));adapter,h,cache,refs=context.load(item);assert h==head
            values=[]
            for pos in range(128):
                path=folder/'patch'/f'{pos:03d}.npz';ms=[masses[a][pos] for a in arm_names]
                if path.exists():
                    with np.load(path) as z:
                        assert z['worlds'].tolist()==arm_names and np.array_equal(z['M'],ms)
                        v=z['maxsim'].copy();assert np.array_equal(v.mean(1),z['L'])
                else:
                    if deadline-time.monotonic()<30:return 75
                    v,loc=context.candidate(adapter,cache,refs[pos],ms)
                    B.npz_once(path,worlds=np.asarray(arm_names),M=np.asarray(ms),maxsim=v,argmax=loc,L=v.mean(1))
                values.append(v.mean(1));patches.append(bind(path))
            matrix=np.asarray(values).T;content={a:matrix[k].tolist() for k,a in enumerate(arm_names)}
            arms=all_scores(item,masses,content,head)
        arms['ORIGINAL_HR1']=origin;maximum=independent_scores(item,arms,head)
        row=dict(protocol=pb,index=q['index'],query_id=q['query_id'],cohort=q['cohort'],fold=item['fold'],
            component=item['component'],axis=item['post_row']['candidate_ids'],winner=item['post_row']['winner_index'],
            worlds=arms,head=head,patches=patches,legacy_reuse=reuse,original_anchor=q['original_bridge'],
            full_C128_predictions_computed=True,scope=p['scope'],mediator=p['mediator'])
        write(folder/'result.json',row);write(folder/'validation.json',dict(status='FINE_C128_FROZEN_SCORES_PASS',protocol=pb,
            result=bind(folder/'result.json'),maximum_numpy_error=maximum,worlds=len(arms),candidates=128,
            new_POST_forwards=not bool(reuse),new_RoMa_forwards=False))
        emit(stage='SCORES_COMPLETE',family=family,index=q['index']);return 0


def unpack(x,torch):
    if isinstance(x,dict):return x['tensor'].to(dtype=getattr(torch,x['original_dtype'])).clone()
    return x.clone()
def context_worker(root,shard,shards,budget,limit,torch):
    p,pb,inputs=guard(root,torch);assert 0<=shard<shards==p['shards'];deadline=time.monotonic()+budget;completed=0
    if limit:assert shard==0 and limit==1,'PILOT_IS_FIXED_TO_FIRST_PAIR'
    if limit and (root/'pilot_validation.json').exists():
        oldpilot=read(root/'pilot_validation.json');assert oldpilot['protocol']==pb
        checked(oldpilot['pair']);return 0
    import rc_m_context_binding_v1 as C
    import run_rc_m_phase_newgroups_cpu_v1 as G
    from rc_roma_shared_native_cache_v1 import ExactPool
    head=None
    for w in inputs['pairs'][shard::shards]:
        folder=root/'context_pairs'/f"{w['pair_index']:05d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            vp=folder/'validation.json'
            if vp.exists():
                previous=read(vp);assert previous['protocol']==pb
                if limit:
                    assert w['pair_index']==0 and previous['status']=='FINE_C128_CONTEXT_PAIR_PASS'
                    assert [a['arm'] for a in previous['arms']]==CONTEXT
                    for a in previous['arms']:checked(a['payload'])
                    native=torch.load(previous['arms'][0]['payload']['path'],weights_only=True,mmap=True,map_location='cpu')
                    assert all(s['native_repeat_bit_exact'] is True for s in native['sides'])
                    write(root/'pilot_validation.json',dict(status='FINE_C128_CONTEXT_PILOT_PASS',protocol=pb,pair=bind(vp),new_pairs=0,
                        native_repeat_bit_exact=True,recovered_completed_first_pair=True))
                    return 0
                continue
            if deadline-time.monotonic()<45:return 75
            old=torch.load(checked(w['old_pair']),weights_only=True,mmap=True,map_location='cpu')
            assert (old['index'],old['position'],old['physical_row'])==(w['index'],w['position'],w['physical_row'])
            cap=torch.load(checked(old['capture']),weights_only=True,mmap=True,map_location='cpu')
            assert [cap['query_sha'],cap['reference_sha']]==w['image_shas']
            fs=[torch.load(checked(b),weights_only=True,mmap=True,map_location='cpu')['features'] for b in cap['descriptor_sources']]
            cap2={'sides':[{k:unpack(s[k],torch) for k in ['J','P']} for s in cap['sides']]}
            if head is None:head=C.load_head(p,torch)
            gs=G.geometries(w);pool=ExactPool();inv=folder/'input_invariants.json'
            if not inv.exists():write(inv,dict(protocol=pb,capture=old['capture'],descriptors=cap['descriptor_sources'],records=C.invariants(fs,cap2,torch)))
            else:assert read(inv)['protocol']==pb
            records=[];native_conf=None;native_ln=[]
            with torch.inference_mode():
                for side in range(2):
                    baseline_sum=fs[side][-1]+cap2['sides'][side]['J']+cap2['sides'][side]['P']
                    native_ln.append(head.norm(baseline_sum.reshape(1,-1,1024)).reshape_as(baseline_sum))
            for arm in CONTEXT:
                path=folder/(arm+'.pt');receipt=folder/(arm+'.json')
                if receipt.exists():
                    rec=read(receipt);assert rec['protocol']==pb;checked(rec['payload']);records.append(rec)
                    if arm=='NATIVE':native_conf=[s['confidence'] for s in torch.load(path,weights_only=True,mmap=True,map_location='cpu')['sides']]
                    continue
                if deadline-time.monotonic()<30:return 75
                tick=time.monotonic();sides=[];shift,bits=C.recipe(arm)
                with torch.inference_mode():
                    for side in range(2):
                        aa,j,pp=C.transform(fs[side],cap2['sides'][side]['J'],cap2['sides'][side]['P'],arm,torch)
                        summed=aa[-1]+j+pp;native_sum=fs[side][-1]+cap2['sides'][side]['J']+cap2['sides'][side]['P']
                        if bits=='111':assert torch.equal(summed,C.roll(native_sum,shift,torch))
                        if bits[-1]=='0':assert torch.equal(pp,cap2['sides'][side]['P'])
                        algebra,means=C.pointwise(aa,j,pp,summed,torch)
                        ln_capture=[]
                        def hook(module,args,out):ln_capture.append(out.detach())
                        token=head.norm.register_forward_hook(hook)
                        try:pred=head([aa[0],summed],img_A=None,img_B=None)
                        finally:token.remove()
                        assert len(ln_capture)==4 and torch.equal(ln_capture[2],ln_capture[3])
                        actual_ln=ln_capture[2].reshape_as(summed)
                        assert torch.equal(actual_ln,head.norm(summed.reshape(1,-1,1024)).reshape_as(summed))
                        commutator=None
                        if bits=='111':
                            commutator=float((actual_ln-C.roll(native_ln[side],shift,torch)).abs().max());assert commutator==0
                        lncos=(actual_ln.double()*native_ln[side].double()).sum(-1)/(actual_ln.double().norm(dim=-1)*native_ln[side].double().norm(dim=-1)).clamp_min(1e-30)
                        conf=pred[...,2:].clone()
                        assert bool(torch.isfinite(conf).all())
                        repeat=None
                        if arm=='NATIVE' and w['pair_index']==0:
                            again=head([aa[0],summed],img_A=None,img_B=None)[...,2:]
                            assert torch.equal(conf,again),'CPU_NATIVE_REPEAT_FAILED';repeat=True
                        weights=pool(conf[0,...,0].sigmoid(),gs[side]);error=C.independent_pool(conf,gs[side],weights)
                        grid=None
                        if bits=='111':
                            delta=torch.roll(conf,(-4*shift[0],-4*shift[1]),dims=(1,2)).double()-native_conf[side].double()
                            grid=dict(confidence_logit_RMS=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()))
                        gpu=old['branches']['NATIVE'][side]
                        sides.append(dict(confidence=conf,weights=weights,preLN=algebra,preLN_means=means,independent_pool_error=error,
                            LN_to_native_cosine=lncos,LN_to_native_cosine_mean=float(lncos.mean()),pointwise_LN_common_roll_max_error=commutator,
                            dense_sigmoid_mean=float(conf.sigmoid().double().mean()),token_pool_mean=float(weights.mean()),
                            native_repeat_bit_exact=repeat,common_grid_diagnostic=grid,
                            native_cpu_gpu_confidence_RMS=float((conf.double()-gpu['confidence'].double()).square().mean().sqrt()) if arm=='NATIVE' else None))
                mass=scalar_mass(sides);assert 0<mass<=1
                if arm=='NATIVE':native_conf=[s['confidence'] for s in sides]
                value=dict(protocol=pb,pair_index=w['pair_index'],index=w['index'],position=w['position'],physical_row=w['physical_row'],
                    arm=arm,M=mass,sides=sides,capture=old['capture'],descriptors=cap['descriptor_sources'],
                    recipe=dict(bits=bits,shift=list(shift)),seconds=time.monotonic()-tick,
                    old_GPU_native_M=scalar_mass(old['branches']['NATIVE']) if arm=='NATIVE' else None)
                if path.exists():
                    prior=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
                    assert prior['protocol']==pb and prior['M']==mass
                    assert all(torch.equal(x['confidence'],y['confidence']) and torch.equal(x['weights'],y['weights']) for x,y in zip(prior['sides'],sides))
                else:save(path,value,torch)
                rec=dict(protocol=pb,arm=arm,payload=bind(path),M=mass);write(receipt,rec);records.append(rec)
                emit(stage='CONTEXT_ARM',pair=w['pair_index'],arm=arm,seconds=value['seconds'])
            write(vp,dict(status='FINE_C128_CONTEXT_PAIR_PASS',protocol=pb,pair_index=w['pair_index'],index=w['index'],position=w['position'],
                arms=records,input_invariants=bind(inv),source=w['old_pair']))
            completed+=1
            if limit and completed>=limit:
                write(root/'pilot_validation.json',dict(status='FINE_C128_CONTEXT_PILOT_PASS',protocol=pb,pair=bind(vp),new_pairs=completed,
                    native_repeat_bit_exact=w['pair_index']==0,elapsed_seconds=budget-(deadline-time.monotonic())))
                return 0
    return 0


def context_join(root,torch):
    p,pb,inputs=guard(root,torch);sources=[];rows={q['index']:dict(index=q['index'],query_id=q['query_id'],M={a:[] for a in CONTEXT}) for q in inputs['queries']};maximum=0.;native_drift=[]
    for w in inputs['pairs']:
        vpath=root/'context_pairs'/f"{w['pair_index']:05d}"/'validation.json'
        if not vpath.exists():return 75
        v=read(vpath);assert v['protocol']==pb and v['status']=='FINE_C128_CONTEXT_PAIR_PASS'
        assert [a['arm'] for a in v['arms']]==CONTEXT;checked(v['input_invariants'])
        for a in v['arms']:
            x=torch.load(checked(a['payload']),weights_only=True,mmap=True,map_location='cpu')
            assert x['protocol']==pb and (x['index'],x['position'])==(w['index'],w['position'])
            mass=scalar_mass(x['sides']);e=abs(mass-x['M']);maximum=max(maximum,e);assert e<2e-12 and mass==a['M']
            rows[w['index']]['M'][a['arm']].append(mass)
            if a['arm']=='NATIVE':native_drift.append(dict(index=w['index'],position=w['position'],cpu_M=mass,gpu_M=x['old_GPU_native_M']))
        sources.append(bind(vpath))
    assert all(len(v)==128 for row in rows.values() for v in row['M'].values())
    write(root/'context_masses.json',dict(protocol=pb,rows=list(rows.values()),native_cpu_GPU_drift=native_drift,sources=sources))
    write(root/'context_validation.json',dict(status='FINE_C128_CONTEXT_MASSES_PASS',protocol=pb,result=bind(root/'context_masses.json'),
        pairs=1408,arms=13,maximum_mass_error=maximum,backend='CPU',old_GPU_phase_worlds_not_mixed=True))
    emit(status='FINE_C128_CONTEXT_MASSES_PASS');return 0


def aggregate(rows,model,arm,cohort):
    rs=[r for r in rows if r['cohort']==cohort];raw={r['index'] for r in rs if r['raw_correct']}
    cor={r['index'] for r in rs if r['correct'][model][arm]};hr={r['index'] for r in rs if r['correct'][model]['ORIGINAL_HR1']}
    native={r['index'] for r in rs if r['correct'][model]['NATIVE']};resc=cor-raw;brk=raw-cor;original=hr-raw
    return dict(total=len(rs),target_present=sum(r['target_present'] for r in rs),RAW_correct=len(raw),correct=len(cor),
        rescues=sorted(resc),breaks=sorted(brk),net=len(resc)-len(brk),original_HR1_rescues=sorted(original),
        original_rescues_retained=sorted(resc&original),original_rescues_lost=sorted(original-resc),new_rescues=sorted(resc-original),
        original_breaks_repaired=sorted((raw-hr)-brk),new_breaks_vs_HR1=sorted(brk-(raw-hr)),
        own_native_correct_to_wrong=sorted(native-cor),own_native_wrong_to_correct=sorted(cor-native))

def continuous_contrasts(phase):
    """Coefficients apply to each completed world, never to a new score/model."""
    if phase:
        out={'phase_GLOBAL_minus_LOCAL':{a:sign/4 for d in DIRECTIONS for a,sign in [(f'GLOBAL_{d}',1),(f'LOCAL_{d}',-1)]},
             'amplitude_NATIVE_minus_PERMUTE':{'NATIVE':1.,'AMP_PERMUTE':-1.},
             'amplitude_NATIVE_minus_FLAT':{'NATIVE':1.,'AMP_FLAT':-1.}}
    else:
        out={'context_matched_alignment':{'NATIVE':.5,**{d+'__'+b:sign/8 for d in DIRECTIONS for b,sign in [('111',1),('001',-1),('110',-1)]}},
             'context_P_fixed_move_AJ':{'NATIVE':-1.,**{d+'__110':.25 for d in DIRECTIONS}},
             'context_P_only_moved':{'NATIVE':-1.,**{d+'__001':.25 for d in DIRECTIONS}},
             'context_all_moved_grid_control':{'NATIVE':-1.,**{d+'__111':.25 for d in DIRECTIONS}}}
    out['own_native_minus_ORIGINAL_HR1']={'NATIVE':1.,'ORIGINAL_HR1':-1.}
    assert all(abs(math.fsum(x.values()))<1e-15 for x in out.values())
    return out

def group_stats(pairs):
    grouped={}
    for group,value in pairs:grouped.setdefault(str(group),[]).append(value)
    values=np.asarray([math.fsum(v)/len(v) for v in grouped.values()],float)
    if not len(values):return dict(groups=0,mean=None,exploratory_bootstrap95=None)
    rng=np.random.default_rng(270928);boot=rng.choice(values,(4000,len(values)),replace=True).mean(1)
    return dict(groups=len(values),queries=len(pairs),mean=float(values.mean()),positive_groups=int((values>0).sum()),
        exploratory_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),group_values={k:math.fsum(v)/len(v) for k,v in grouped.items()},
        inference='Opened small mechanism panel; unadjusted exploratory group bootstrap, not confirmation or equivalence.')

def continuous_summary(rows,coefficients):
    answer={}
    for cohort in ['MAIN8','SELECTED_CASES3']:
        rr=[r for r in rows if r['cohort']==cohort]
        metrics=sorted({m for r in rr for m in r['measurements']})
        answer[cohort]={name:{metric:group_stats([(r['component'],r['continuous_effects'][name][metric]) for r in rr
            if r['continuous_effects'][name].get(metric) is not None]) for metric in metrics} for name in coefficients}
    return answer

def accounting(root,torch,phase):
    p,pb,inputs=guard(root,torch);family='phase' if phase else 'context_bridge';arms=PHASE_ARMS if phase else CONTEXT;worlds=arms+['ORIGINAL_HR1']
    folders=[root/family/f"query{q['index']:03d}" for q in inputs['queries']]
    if not all((f/'validation.json').exists() for f in folders):return None
    labels={r['query_id']:r for r in read(checked(p['label_source']))['rows']};rows=[];sources=[];maximum=0.;coefficients=continuous_contrasts(phase)
    for q,folder in zip(inputs['queries'],folders):
        v=read(folder/'validation.json');assert v['protocol']==pb and v['status']=='FINE_C128_FROZEN_SCORES_PASS'
        d=read(checked(v['result']));assert list(d['worlds'].keys())==sorted(worlds)
        if d['patches']:
            assert len(d['patches'])==128
            for pos,source in enumerate(d['patches']):
                with np.load(checked(source)) as z:
                    assert z['worlds'].tolist()==arms and np.array_equal(z['M'],[d['worlds'][a]['M'][pos] for a in arms])
                    means=z['maxsim'].mean(1)
                    assert np.array_equal(means,z['L']) and np.array_equal(means,[d['worlds'][a]['L'][pos] for a in arms])
                    assert np.issubdtype(z['argmax'].dtype,np.integer) and np.all(z['argmax']>=0)
        else:
            assert phase and q['index'] in MAIN and d['legacy_reuse']==q['phase_reuse']
            prior=read(checked(d['legacy_reuse']['result']))
            for a in arms:assert prior['worlds'][a if a=='NATIVE' else a+'/FULL']['L']==d['worlds'][a]['L']
        maximum=max(maximum,independent_scores(q['item'],d['worlds'],d['head']));label=labels[q['query_id']]
        targets=[k for k,x in enumerate(q['item']['post_row']['candidate_identities']) if x==label['identity']];assert len(targets)<=1
        row=dict(index=q['index'],query_id=q['query_id'],original_query_id=label['original_query_id'],cohort=q['cohort'],component=label['component'],
            target_present=bool(targets),target_position=targets[0] if targets else None,raw_correct=d['winner'] in targets,
            correct={},prediction={},switched={},target_vs_strongest_wrong_margin={},scores_source=v['result'],measurements={})
        # Evaluation-only fixed competitor: highest RAW-scoring non-target,
        # chosen once and reused under every property intervention.
        raw=q['item']['post_row']['raw_scores'];wrong=max((k for k in range(128) if k not in targets),key=lambda k:raw[k])
        row['fixed_wrong_position']=wrong
        for field in ['M','logM','POST_content']:
            for role in ['target','fixed_wrong','target_wrong_gap']:
                row['measurements'][field+'_'+role]={}
            for a in worlds:
                x=np.asarray(d['worlds'][a]['L' if field=='POST_content' else 'M'])
                if field=='logM':x=np.log(x)
                row['measurements'][field+'_fixed_wrong'][a]=float(x[wrong])
                row['measurements'][field+'_target'][a]=float(x[targets[0]]) if targets else None
                row['measurements'][field+'_target_wrong_gap'][a]=float(x[targets[0]]-x[wrong]) if targets else None
        for model in MODELS:
            row['correct'][model]={};row['prediction'][model]={};row['switched'][model]={};row['target_vs_strongest_wrong_margin'][model]={}
            for metric in ['target_action','wrong_action','target_wrong_action_gap','strongest_wrong_margin','correct_indicator','RAW_rescue_indicator','RAW_break_indicator']:
                row['measurements'][model+'_'+metric]={}
            for a in worlds:
                x=d['worlds'][a]['POST'] if model=='POST' else d['worlds'][a]['external'][model]
                pred=x['prediction_position'];row['prediction'][model][a]=pred;row['correct'][model][a]=pred in targets;row['switched'][model][a]=pred!=d['winner']
                z=x['logits128'];row['target_vs_strongest_wrong_margin'][model][a]=z[targets[0]]-max(z[k] for k in range(128) if k not in targets) if targets else None
                for metric,value in dict(target_action=z[targets[0]] if targets else None,wrong_action=z[wrong],
                    target_wrong_action_gap=z[targets[0]]-z[wrong] if targets else None,
                    strongest_wrong_margin=row['target_vs_strongest_wrong_margin'][model][a],
                    correct_indicator=float(pred in targets),RAW_rescue_indicator=float(not row['raw_correct'] and pred in targets),
                    RAW_break_indicator=float(row['raw_correct'] and pred not in targets)).items():row['measurements'][model+'_'+metric][a]=value
        row['continuous_effects']={name:{metric:(math.fsum(c*values[a] for a,c in coef.items()) if all(values[a] is not None for a in coef) else None)
            for metric,values in row['measurements'].items()} for name,coef in coefficients.items()}
        rows.append(row);sources += [bind(folder/'validation.json'),v['result']]
    summaries={cohort:{m:{a:aggregate(rows,m,a,cohort) for a in worlds} for m in MODELS} for cohort in ['MAIN8','SELECTED_CASES3']}
    comparisons=([(f'phase_{d}',f'LOCAL_{d}',f'GLOBAL_{d}') for d in DIRECTIONS]+[('amplitude_permute','AMP_PERMUTE','NATIVE'),('amplitude_flat','AMP_FLAT','NATIVE')]
                 if phase else [(f'{d}_{b}',d+'__'+b,'NATIVE') for d in DIRECTIONS for b in ['001','110','111']])
    changes={}
    for name,off,on in comparisons:
        changes[name]={}
        for cohort in ['MAIN8','SELECTED_CASES3']:
            rr=[r for r in rows if r['cohort']==cohort];changes[name][cohort]={}
            for m in MODELS:
                recovered=[r['index'] for r in rr if not r['correct'][m][off] and r['correct'][m][on]]
                lost=[r['index'] for r in rr if r['correct'][m][off] and not r['correct'][m][on]]
                original=[r['index'] for r in rr if not r['raw_correct'] and r['correct'][m]['ORIGINAL_HR1']]
                changes[name][cohort][m]=dict(off=off,on=on,recovered=recovered,lost=lost,net=len(recovered)-len(lost),
                    original_rescues_recovered=sorted(set(recovered)&set(original)),original_rescues_lost=sorted(set(lost)&set(original)))
    return dict(protocol=pb,family=family,rows=rows,summary=summaries,conditional_changes=changes,sources=sources,
        contrast_coefficients=coefficients,continuous_summary=continuous_summary(rows,coefficients),
        fixed_wrong_policy='Highest RAW score among non-target natural-C128 candidates, fixed before comparing interventions; target labels enter accounting only.',
        indicator_contrast_policy='Contrasts of completed-world 0/1 outcomes are directional expectations, not a new model, integer rescue set, or unique causal share.',
        max_independent_logit_error=maximum,scope=p['scope'],mediator=p['mediator'],
        zero_denominator='If a primary cohort has no original rescue, recovery fraction is undefined; case3 cannot replace its denominator.',
        directional_policy='Every full-C128 directional world is reported. Mean effects are directional expectations, not an averaged-score new model or union of rescued queries.')
def phase_join(root,torch):
    data=accounting(root,torch,True)
    if data is None:return 75
    write(root/'phase_accounting.json',data)
    write(root/'phase_validation.json',dict(status='FINE_C128_PHASE_ACCOUNTING_PASS',protocol=bind(root/'protocol.json'),
        result=bind(root/'phase_accounting.json'),queries=11,candidates=128,phase_worlds=12,max_error=data['max_independent_logit_error']))
    emit(status='FINE_C128_PHASE_ACCOUNTING_PASS');return 0
def join(root,torch):
    if not (root/'phase_validation.json').exists():return 75
    pv=read(root/'phase_validation.json');assert pv['status']=='FINE_C128_PHASE_ACCOUNTING_PASS'
    phase=read(checked(pv['result']));context=accounting(root,torch,False)
    if context is None:return 75
    write(root/'context_accounting.json',context)
    result=dict(status='FINE_C128_THREE_PROPERTY_ACCOUNTING_COMPLETE',protocol=bind(root/'protocol.json'),
        phase=phase,context=context,phase_validation=bind(root/'phase_validation.json'),context_validation=bind(root/'context_validation.json'),
        scope=context['scope'],not_unique_or_additive_contribution=True,no_full128_or593_extrapolation=True)
    write(root/'result.json',result)
    lines=['# Full-C128 fine-property accounting on the existing 11-query panel','',
        'Here fine-property means finer mechanistic properties of the coarse DPT endpoint. This experiment does not run the RoMa fine-matching/refiner stage.','',
        'MAIN8 comprises historical outcome-independent group representatives; SELECTED_CASES3 was historically selected for original rescues and remains descriptive. Natural C128 is complete for every query. Original HR1, historical GPU coarse native, and new CPU context native are separate anchors.','',
        'The primary context contrast compares aligned against misaligned A/J and P with matched absolute placements: (000+111−001−110)/2, averaged over four prespecified inverse shifts. Phase coherence compares GLOBAL against LOCAL under native amplitude. Amplitude compares NATIVE against the frozen PERMUTE and FLAT interventions. Conditional effects are not unique causal shares.','',
        '| Family/cohort | Contrast | Measurement | Mean group effect | Exploratory 95% interval | Groups |',
        '|---|---|---|---:|---|---:|']
    for family,data in [('phase',phase),('context',context)]:
        for cohort,contrasts in data['continuous_summary'].items():
            for contrast,metrics in contrasts.items():
                for key in ['M_target_wrong_gap','logM_target_wrong_gap','POST_content_target_wrong_gap','NATIVE7_target_wrong_action_gap','M_FREE_target_wrong_action_gap','POST_target_wrong_action_gap']:
                    s=metrics[key];ci=s['exploratory_bootstrap95'];interval='undefined' if ci is None else f"[{ci[0]:.7g}, {ci[1]:.7g}]"
                    mean='undefined' if s['mean'] is None else f"{s['mean']:.7g}"
                    lines.append(f"| {family}/{cohort} | {contrast} | {key} | {mean} | {interval} | {s['groups']} |")
    lines+=['','| Family/cohort/model | World | Correct | RAW rescue | RAW break | Original rescues retained |','|---|---|---:|---:|---:|---:|']
    for family,data in [('phase',phase),('context',context)]:
        for cohort,models in data['summary'].items():
            for m,worlds in models.items():
                for a,s in worlds.items():lines.append(f"| {family}/{cohort}/{m} | {a} | {s['correct']}/{s['total']} | {len(s['rescues'])} | {len(s['breaks'])} | {len(s['original_rescues_retained'])}/{len(s['original_HR1_rescues'])} |")
    lines+=['','A displayed 0/0 retention count is undefined, not a zero effect. Raw sets, per-query predictions and strongest-wrong margins are saved. Conditional recovery sets can overlap and cannot be summed into unique causal shares.','',
        'Phase/amplitude cells reuse the same historical GPU outputs. Context cells use the same frozen CPU backend and their own native. No cross-backend factorial contrast is formed. Only M and its algebraic descendants propagate to fixed external/POST models.','']
    report=root/'report.md';s='\n'.join(lines)
    if report.exists():assert report.read_text()==s
    else:report.write_text(s)
    write(root/'validation.json',dict(status='FINE_C128_THREE_PROPERTY_ACCOUNTING_PASS',protocol=bind(root/'protocol.json'),
        result=bind(root/'result.json'),report=bind(report),phase=bind(root/'phase_validation.json'),context=bind(root/'context_validation.json'),
        queries=11,candidates=128,main_queries=8,selected_case_queries=3,max_error=max(phase['max_independent_logit_error'],context['max_independent_logit_error']),
        all_original128_or593_benefit_closed=False))
    emit(status='FINE_C128_THREE_PROPERTY_ACCOUNTING_PASS');return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','phase-worker','phase-join','worker','context-join','bridge-worker','join'])
    ap.add_argument('--root',type=Path,default=OUT);ap.add_argument('--index',type=int,default=0)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=50)
    ap.add_argument('--budget',type=float,default=430);ap.add_argument('--limit-pairs',type=int,default=0);a=ap.parse_args();t=configure()
    if a.command=='prepare':prepare(a.root,t)
    elif a.command=='phase-worker':sys.exit(score_worker(a.root,a.index,a.budget,t,True))
    elif a.command=='bridge-worker':sys.exit(score_worker(a.root,a.index,a.budget,t,False))
    elif a.command=='worker':sys.exit(context_worker(a.root,a.shard,a.shards,a.budget,a.limit_pairs,t))
    elif a.command=='phase-join':sys.exit(phase_join(a.root,t))
    elif a.command=='context-join':sys.exit(context_join(a.root,t))
    else:sys.exit(join(a.root,t))
