#!/usr/bin/env python3
"""Cached-input identification probes with explicit Gram/local-binding invariants.

Local binding means named pointwise inner products, not all semantic binding.
Common channel-rotation controls measure frozen decoder basis sensitivity.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
from run_rc_m_conditional_prediction_v1 import read,bind,write,boot
OUT=RC/'results/rc_m_structure_binding_isolation_v1'
BRIDGE=RC/'results/rc_m_causal128_attribution_chain_v1/bridge'
MAIN=[0,1,2,4,5,6,7,8]
EVENT=[9,21,32,41,44,55,82,84,110,116,118]
ANGLE=.5
ARMS=['NATIVE']+[f'{a}_{s}' for a in ['Q_P','Q_C','Q_ALL','S_P','Q_AFTER_S','S_AFTER_Q'] for s in ['PLUS','MINUS']]

def checked(b):
    assert bind(b['path'])=={k:b[k] for k in ['path','sha256']},('SOURCE_CHANGED',b['path']);return Path(b['path'])
def unpack(x,t):
    return x['tensor'].to(getattr(t,x['original_dtype'])) if isinstance(x,dict) else x
def configure():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    return torch
def emit(**x):print(json.dumps(x,allow_nan=False),flush=True)
def save(p,x,t):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');t.save(x,tmp);os.replace(tmp,p)

def rotate(x,angle,t):
    # Every disjoint four-channel block is rotated in the plane
    # (1,-1,0,0)/sqrt2,(0,0,1,-1)/sqrt2. Q is orthogonal and fixes ones.
    y=x.double().reshape(*x.shape[:-1],-1,4);z=y.clone();s=math.sin(angle);c=math.cos(angle);rt=math.sqrt(2.)
    a=(y[...,0]-y[...,1])/rt;b=(y[...,2]-y[...,3])/rt
    da=a*(c-1)-b*s;db=a*s+b*(c-1)
    z[...,0]+=da/rt;z[...,1]-=da/rt;z[...,2]+=db/rt;z[...,3]-=db/rt
    return z.reshape_as(x)
def basis(content,t):
    vectors=[t.ones_like(content[0],dtype=t.float64)]+[x.double() for x in content]
    qs=[]
    for x in vectors:
        v=x.clone()
        # Reorthogonalization controls nearly dependent content channels.
        for _ in range(2):
            for q in qs:v=v-(v*q).sum(-1,keepdim=True)*q
        n=v.norm(dim=-1,keepdim=True);qs.append(t.where(n>1e-10,v/n.clamp_min(1e-30),t.zeros_like(v)))
    return qs
def project(x,qs):return sum((x*q).sum(-1,keepdim=True)*q for q in qs)
def local_preserve(p,qs,angle,t):
    # Restore P's exact local content projections and residual norm after Q.
    x=p.double();parallel=project(x,qs);r=x-parallel;proposal=rotate(x,angle,t);v=proposal-project(proposal,qs)
    # A second orthogonal projection limits floating residuals.
    v=v-project(v,qs);nv=v.norm(dim=-1,keepdim=True);nr=r.norm(dim=-1,keepdim=True)
    replacement=v*nr/nv.clamp_min(1e-30)
    return parallel+t.where(nv>1e-12,replacement,r)
def transform(fs,j,p,arm,t):
    if arm=='NATIVE':return [x.clone() for x in fs],j.clone(),p.clone()
    name,sign=arm.rsplit('_',1);angle=ANGLE*(1 if sign=='PLUS' else -1)
    qs=basis([*fs,j],t)
    if name=='Q_P':pp=rotate(p,angle,t)
    elif name=='S_P':pp=local_preserve(p,qs,angle,t)
    elif name=='Q_AFTER_S':pp=rotate(local_preserve(p,qs,angle,t),angle,t)
    elif name=='S_AFTER_Q':pp=local_preserve(rotate(p,angle,t),qs,angle,t)
    elif name=='Q_ALL':return [rotate(x,angle,t).to(x.dtype) for x in fs],rotate(j,angle,t).to(j.dtype),rotate(p,angle,t).to(p.dtype)
    elif name=='Q_C':return [rotate(x,angle,t).to(x.dtype) for x in fs],rotate(j,angle,t).to(j.dtype),p.clone()
    else:raise ValueError(arm)
    return [x.clone() for x in fs],j.clone(),pp.to(p.dtype)

def checks(fs,j,p,aa,jj,pp,arm,t):
    x=p.double();y=pp.double();xn=x.norm(dim=-1);yn=y.norm(dim=-1)
    normerr=float(((xn-yn).abs()/xn.clamp_min(1e-12)).max());assert normerr<3e-6
    meanerr=float(((x.mean(-1)-y.mean(-1)).abs()/x.square().mean(-1).sqrt().clamp_min(1e-12)).max());assert meanerr<3e-6
    ix=t.linspace(0,x.numel()//x.shape[-1]-1,64).long();u=x.reshape(-1,x.shape[-1])[ix];v=y.reshape(-1,y.shape[-1])[ix]
    gram=(u@u.T-v@v.T);gerr=float(gram.abs().max()/((u.norm(dim=-1).max()**2).clamp_min(1e-12)))
    local=[]
    for c,d in zip([*fs,j],[*aa,jj]):
        c=c.double();d=d.double();err=((x*c).sum(-1)-(y*d).sum(-1)).abs()/(xn*c.norm(dim=-1)).clamp_min(1e-12)
        local.append(float(err.max()))
    original_sum=fs[-1].double()+j.double()+x;new_sum=aa[-1].double()+jj.double()+y
    var0=original_sum.var(-1,unbiased=False);var1=new_sum.var(-1,unbiased=False)
    varerr=float(((var1-var0).abs()/var0.clamp_min(1e-10)).max())
    if arm=='NATIVE' or arm.startswith('Q_P_') or arm.startswith('Q_C_') or arm.startswith('Q_ALL_'):assert gerr<3e-6
    if arm=='NATIVE' or arm.startswith('S_P_') or arm.startswith('Q_ALL_'):assert max(local)<3e-6 and varerr<5e-6
    joint_check=None
    if arm.startswith('Q_AFTER_S_'):
        sign=1 if arm.endswith('PLUS') else -1
        base=local_preserve(p,basis([*fs,j],t),sign*ANGLE,t).reshape(-1,x.shape[-1])[ix]
        joint_check=float((base@base.T-v@v.T).abs().max()/base.norm(dim=-1).max().square().clamp_min(1e-12));assert joint_check<3e-6
    if arm.startswith('S_AFTER_Q_'):
        sign=1 if arm.endswith('PLUS') else -1;base=rotate(p,sign*ANGLE,t)
        joint_check=max(float((((base-y)*c.double()).sum(-1).abs()/(base.norm(dim=-1)*c.double().norm(dim=-1)).clamp_min(1e-12)).max()) for c in [*fs,j])
        assert joint_check<3e-6
    return dict(norm_relative_max=normerr,mean_scaled_max=meanerr,selected64_P_Gram_relative_max=gerr,
        conditional_joint_invariant_error=joint_check,
        local_content_inner_product_scaled_max=local,sum_LN_variance_relative_max=varerr,
        perturbation_RMS=float((x-y).square().mean().sqrt()),
        P_SHA=__import__('hashlib').sha256(pp.contiguous().view(t.uint8).numpy().tobytes()).hexdigest())

def prepare(t):
    import run_rc_m_fine_c128_attribution_v1 as F
    roots=[RC/'results/rc_m_fine_c128_attribution_v1',RC/'results/rc_m_upstream_event_closure_v1']
    bp=read(BRIDGE/'protocol.json');items={x['index']:x for x in read(checked(bp['inputs']))['rows']}
    labels={x['query_id']:x for x in read(RC/'results/rc_h593_quality_operator_eval_v1/result.json')['rows']}
    full={};qs={};sources=[bind(BRIDGE/'protocol.json'),bp['inputs']]
    for root in roots:
        d=read(root/'inputs.json');sources.append(bind(root/'inputs.json'))
        for w in d['pairs']:
            w['source_root']=str(root);full[w['index'],w['position']]=w
    workers=[];cases=[]
    for index in sorted(set(k[0] for k in full)):
        item=items[index];identity=labels[item['query_id']]['identity'];targets=[i for i,v in enumerate(item['post_row']['candidate_identities']) if v==identity]
        if not targets:continue
        assert len(targets)==1;target=targets[0];wrong=[i for i in range(128) if i!=target];c=np.asarray(item['external']['free_content'])
        opponents={'C_NEAREST':min(wrong,key=lambda i:(abs(c[i]-c[target]),item['post_row']['candidate_ids'][i])),
            'C_STRONGEST':max(wrong,key=lambda i:(c[i],-item['post_row']['candidate_ids'][i])),
            'M_STRONGEST':max(wrong,key=lambda i:(item['external']['original_M'][i],-item['post_row']['candidate_ids'][i]))}
        cohort='MAIN8' if index in MAIN else 'ORIGINAL_CHANGED11'
        for pos in sorted({target,*opponents.values()}):
            w=full[index,pos];old=t.load(checked(w['old_pair']),map_location='cpu',weights_only=True,mmap=True)
            cap=t.load(checked(old['capture']),map_location='cpu',weights_only=True,mmap=True)
            workers.append(dict(index=index,position=pos,physical_row=w['physical_row'],capture=old['capture'],features=cap['descriptor_sources'],
                geometry_metadata=w['geometry_metadata'],image_shas=w['image_shas'],cohort=cohort,
                cpu_native=bind(Path(w['source_root'])/'context_pairs'/f"{w['pair_index']:05d}"/'NATIVE.pt')))
        cases.append(dict(index=index,query_id=item['query_id'],component=item['component'],cohort=cohort,target=target,opponents=opponents,
            signed_C_gap={k:float(c[target]-c[v]) for k,v in opponents.items()},item=item))
    # Additional already-captured disjoint-development groups: only their two frozen candidates.
    root=RC/'results/rc_m_causal128_attribution_chain_v1/phase_replication';old=read(root/'protocol.json');sources.append(bind(root/'protocol.json'))
    for w in old['workers']:
        index=w['index']
        if index in {x['index'] for x in cases}:continue
        item=items[index]
        for pair in w['pairs']:
            val=read(root/f'query{index:03d}'/f"pair{pair['position']:03d}"/'validation.json')
            features=[bind(d['output_path']) if d.get('encoding_needed') else d['payload'] for d in pair['descriptors']]
            workers.append(dict(index=index,position=pair['position'],physical_row=pair['physical_row'],capture=val['capture'],features=features,
                geometry_metadata=pair['geometry_metadata'],image_shas=pair['image_shas'],cohort='ADDITIONAL_GROUPS'))
        cases.append(dict(index=index,query_id=item['query_id'],component=item['component'],cohort='ADDITIONAL_GROUPS',target=w['target_position'],
            opponents={'C_STRONGEST':w['fixed_wrong_position']},signed_C_gap={'C_STRONGEST':w['free_content_scores'][w['target_position']]-w['free_content_scores'][w['fixed_wrong_position']]},item=item))
    for i,w in enumerate(workers):w['ordinal']=i
    write(OUT/'inputs.json',dict(workers=workers,cases=cases))
    head=read(roots[0]/'protocol.json')['head_state']
    codes=[Path(__file__),RC/'programs/run_rc_m_conditional_prediction_v1.py',Path(F.__file__),RC/'programs/rc_m_context_binding_v1.py',
        RC/'programs/run_rc_m_causal128_frozen_bridge_v1.py',RC/'programs/run_rc_m_phase_newgroups_cpu_v1.py',
        RC/'programs/rc_roma_shared_native_cache_v1.py',RC.parents[2]/'third_party/RoMaV2/src/romav2/dpt.py']
    write(OUT/'protocol.json',dict(status='STRUCTURE_BINDING_OPERATIONAL_ISOLATION_FROZEN',sources=sources,code_sources=[bind(x) for x in codes],
        inputs=bind(OUT/'inputs.json'),head_state=head,bridge_protocol=bind(BRIDGE/'protocol.json'),arms=ARMS,angle=ANGLE,shards=16,
        queries=len(cases),candidate_pairs=len(workers),torch_version=str(t.__version__),threads=4,
        source_scope='Existing19 full-C128 captures plus already-captured9-group paired replication, overlap deduplicated; excludes the target-missing control. No missing encoder/matcher stage is recomputed.',
        operational_structure='P row Gram under a common orthogonal channel rotation; all patch norms and means also held.',
        operational_binding='Pointwise P dot A0,A1,J,ones plus P norm; these determine sum mean/variance but NOT all semantic content binding.',
        contrasts=['local-binding-restored S_P vs NATIVE: residual effect beyond these local statistics',
            '(NATIVE+Q_ALL-Q_P-Q_C)/2: relative alignment interaction with decoder basis control',
            'Q_AFTER_S-S_P vs Q_P-NATIVE; S_AFTER_Q-Q_P vs S_P-NATIVE; retain order interaction'],
        source_components_not_independent='J and P originate in the same upstream matching process; equal summed inputs are indistinguishable to DPT.',
        decoder_basis_control='Q_ALL preserves joint Gram and local inner products but can change learned channel response. Nonzero Q_ALL forbids attributing Q_P to semantic binding alone.',
        inference_scope='Fixed candidate contrasts only, not new complete-C128 accuracy or fraction of all rescues. Frozen POST adapter consumes absolute M; terminal head receives RAW and content only.',
        cohorts='MAIN8 controls, ORIGINAL_CHANGED11 selected events, ADDITIONAL_GROUPS analyzed separately; already-opened data.',
        new_GPU=0,new_encoder_matcher_LLM=0,new_training=0,unique_causal_percentages=False,
        intermediates='Source tensors+deterministic transform recipe+changedP SHA; dense confidence, FP64 weights, LN scales/local products, and POST patch maxsim/argmax retained.'))
    emit(stage='PREPARED',queries=len(cases),pairs=len(workers),arms=len(ARMS))

def guard(t):
    p=read(OUT/'protocol.json');assert str(t.__version__)==p['torch_version'] and t.get_num_threads()==4
    for b in p['code_sources']+[p['inputs'],p['bridge_protocol']]:checked(b)
    return p,bind(OUT/'protocol.json'),read(checked(p['inputs']))
def worker(index,budget,t,pilot=False):
    import rc_m_context_binding_v1 as C
    import run_rc_m_phase_newgroups_cpu_v1 as G
    import run_rc_m_causal128_frozen_bridge_v1 as B
    from rc_roma_shared_native_cache_v1 import ExactPool
    p,pb,inp=guard(t);deadline=time.monotonic()+budget
    if not pilot:assert read(OUT/'pilot_validation.json')['status']=='ISOLATION_NATIVE_AND_INVARIANTS_PASS'
    head=C.load_head(p,t);ctx=B.Context(read(checked(p['bridge_protocol'])));cases={x['index']:x for x in inp['cases']}
    selected=inp['workers'][:1] if pilot else inp['workers'][index::p['shards']]
    for w in selected:
        folder=OUT/'pairs'/f"{w['ordinal']:03d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'validation.json').exists():continue
            if time.monotonic()>deadline-25:return 75
            cap=t.load(checked(w['capture']),map_location='cpu',weights_only=True,mmap=True)
            fs=[t.load(checked(b),map_location='cpu',weights_only=True,mmap=True)['features'] for b in w['features']]
            geometry=G.geometries(w);pool=ExactPool();receipts=[]
            for arm in ARMS:
                rp=folder/f'{arm}.json';path=folder/f'{arm}.pt'
                if rp.exists():z=read(rp);assert z['protocol']==pb;checked(z['payload']);receipts.append(z);continue
                if time.monotonic()>deadline-25:return 75
                started=time.monotonic();sides=[]
                with t.inference_mode():
                    for side in range(2):
                        j=unpack(cap['sides'][side]['J'],t);P=unpack(cap['sides'][side]['P'],t)
                        aa,jj,pp=transform(fs[side],j,P,arm,t);audit=checks(fs[side],j,P,aa,jj,pp,arm,t)
                        summed=aa[-1]+jj+pp;out=head([aa[0],summed],img_A=None,img_B=None);conf=out[...,2:].clone()
                        anchor_error=None
                        if arm=='NATIVE' and w.get('cpu_native'):
                            anchor=t.load(checked(w['cpu_native']),map_location='cpu',weights_only=True,mmap=True)['sides'][side]['confidence']
                            anchor_error=float((conf-anchor).abs().max());assert anchor_error==0,('CPU_NATIVE_ANCHOR',w['ordinal'],side,anchor_error)
                        if arm=='NATIVE' and pilot:
                            again=head([fs[side][0],fs[side][-1]+j+P],img_A=None,img_B=None)[...,2:]
                            assert t.equal(conf,again)
                        weights=pool(conf[0,...,0].sigmoid(),geometry[side]);err=C.independent_pool(conf,geometry[side],weights)
                        local=t.stack([(pp.double()*v.double()).sum(-1) for v in [*aa,jj]],0)
                        sides.append(dict(confidence=conf,weights=weights,input_audit=audit,local_inner_products=local,
                            P_norm=pp.double().norm(dim=-1),sum_mean=summed.double().mean(-1),sum_variance=summed.double().var(-1,unbiased=False),independent_pool_error=err,cpu_anchor_error=anchor_error))
                mass=math.sqrt(float(sides[0]['weights'].mean())*float(sides[1]['weights'].mean()));assert 0<mass<=1
                save(path,dict(protocol=pb,index=w['index'],position=w['position'],arm=arm,M=mass,sides=sides,capture=w['capture']),t)
                z=dict(protocol=pb,arm=arm,M=mass,payload=bind(path),seconds=time.monotonic()-started);write(rp,z);receipts.append(z)
                emit(stage='ISOLATION_ARM',pair=w['ordinal'],arm=arm,M=mass)
            if time.monotonic()>deadline-25:return 75
            adapter,h,cache,refs=ctx.load(cases[w['index']]['item']);values,loc=ctx.candidate(adapter,cache,refs[w['position']],[x['M'] for x in receipts])
            patch=folder/'post_patch.npz'
            with patch.open('wb') as stream:np.savez_compressed(stream,worlds=np.array(ARMS),M=np.array([x['M'] for x in receipts]),maxsim=values,argmax=loc,L=values.mean(1))
            write(folder/'validation.json',dict(status='ISOLATION_PAIR_COMPLETE',protocol=pb,worker=w,arms=receipts,post_patch=bind(patch),POST_L=values.mean(1).tolist()))
    if pilot:
        v=read(OUT/'pairs/000/validation.json');assert len(v['arms'])==len(ARMS)
        write(OUT/'pilot_validation.json',dict(status='ISOLATION_NATIVE_AND_INVARIANTS_PASS',protocol=pb,pair=bind(OUT/'pairs/000/validation.json'),self_test=bind(OUT/'self_test.json')))
    return 0

def join(t):
    p,pb,inp=guard(t);pairs={};source=[];maximum=0.
    for w in inp['workers']:
        f=OUT/'pairs'/f"{w['ordinal']:03d}";v=read(f/'validation.json');assert v['protocol']==pb;record={}
        with np.load(checked(v['post_patch'])) as z:
            assert z['worlds'].tolist()==ARMS;L=z['maxsim'].mean(1);assert np.array_equal(L,z['L'])
        for k,r in enumerate(v['arms']):
            x=t.load(checked(r['payload']),map_location='cpu',weights_only=True,mmap=True)
            M=float(np.sqrt(np.mean(x['sides'][0]['weights'].numpy())*np.mean(x['sides'][1]['weights'].numpy())))
            maximum=max(maximum,abs(M-r['M']));assert abs(M-r['M'])<2e-12
            record[r['arm']]=dict(logM=math.log(M),POST_content=float(L[k]),input_audits=[s['input_audit'] for s in x['sides']])
        pairs[w['index'],w['position']]=record;source.append(bind(f/'validation.json'))
    rows=[]
    for q in inp['cases']:
        for opponent,pos in q['opponents'].items():
            a=pairs[q['index'],q['target']];b=pairs[q['index'],pos];metrics={}
            for metric in ['logM','POST_content']:
                y={arm:a[arm][metric]-b[arm][metric] for arm in ARMS};contrasts={}
                for name in ['binding_interaction','local_stats_preserved_effect','common_basis_effect','B_after_S_interaction','S_after_B_interaction','order_effect']:
                    values=[]
                    for sign in ['PLUS','MINUS']:
                        Q=y['Q_P_'+sign];C=y['Q_C_'+sign];both=y['Q_ALL_'+sign];S=y['S_P_'+sign];BS=y['Q_AFTER_S_'+sign];SB=y['S_AFTER_Q_'+sign];N=y['NATIVE']
                        values.append(dict(binding_interaction=(N+both-Q-C)/2,local_stats_preserved_effect=S-N,common_basis_effect=both-N,
                            B_after_S_interaction=(BS-S)-(Q-N),S_after_B_interaction=(SB-Q)-(S-N),order_effect=SB-BS)[name])
                    contrasts[name]=float(np.mean(values))
                metrics[metric]=dict(world_gaps=y,contrasts=contrasts,target={a0:a[a0][metric] for a0 in ARMS},wrong={a0:b[a0][metric] for a0 in ARMS})
            rows.append(dict(index=q['index'],query_id=q['query_id'],component=q['component'],cohort=q['cohort'],opponent=opponent,target=q['target'],wrong=pos,
                signed_C_gap=q['signed_C_gap'][opponent],metrics=metrics))
    summaries={}
    for cohort in ['MAIN8','ORIGINAL_CHANGED11','ADDITIONAL_GROUPS']:
        summaries[cohort]={}
        for opponent in ['C_NEAREST','C_STRONGEST','M_STRONGEST']:
            selected=[r for r in rows if r['cohort']==cohort and r['opponent']==opponent]
            if not selected:continue
            summaries[cohort][opponent]={metric:{name:boot([(r['component'],r['metrics'][metric]['contrasts'][name]) for r in selected])
                for name in selected[0]['metrics'][metric]['contrasts']} for metric in ['logM','POST_content']}
    write(OUT/'result.json',dict(status='STRUCTURE_BINDING_OPERATIONAL_ISOLATION_COMPLETE',protocol=pb,rows=rows,summaries=summaries,sources=source,
        caveats=['Local binding invariants are not all semantic binding.','Q_ALL controls channel-basis sensitivity; Q_P alone does not isolate binding.',
                 'Effects are conditional, signed and potentially order-dependent; no unique causal percentages.','Candidate-pair contrasts do not establish complete-C128 accuracy.']))
    write(OUT/'validation.json',dict(status='STRUCTURE_BINDING_INDEPENDENT_M_AND_PATCH_PASS',result=bind(OUT/'result.json'),pairs=len(pairs),maximum_M_recount_error=maximum))
    lines=['# P structure and local content binding: operational isolation','',p['source_scope'],'',
        'Local binding means per-patch inner products/norm/normalization scale. It does not mean the full semantic relation has been preserved. Q_ALL explicitly tests frozen decoder basis sensitivity. No unique share or new accuracy is inferred.','',
        '| Cohort | Opponent | Endpoint | Contrast | Mean | Exploratory group95% |','|---|---|---|---|---:|---|']
    for c,dd in summaries.items():
        for o,ee in dd.items():
            for m,ff in ee.items():
                for name,v in ff.items():lines.append(f"| {c} | {o} | {m} | {name} | {v['group_equal_mean']:.8f} | {v['exploratory_group95']} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

def test(t):
    g=t.Generator().manual_seed(20260930);fs=[t.randn(1,4,4,16,generator=g) for _ in range(2)];j=t.randn(1,4,4,16,generator=g);p=t.randn(1,4,4,16,generator=g)
    audits={}
    for arm in ARMS:
        a,b,c=transform(fs,j,p,arm,t);audits[arm]=checks(fs,j,p,a,b,c,arm,t)
    q=rotate(t.eye(16,dtype=t.float64),ANGLE,t)
    assert float((q@q.T-t.eye(16)).abs().max())<1e-12 and float((q.sum(-1)-1).abs().max())<1e-12
    restored=rotate(rotate(p,ANGLE,t),-ANGLE,t);assert float((restored-p.double()).abs().max())<1e-12
    # Detect degenerate-content failure without assuming full rank.
    deg=[fs[0],fs[0]];a,b,c=transform(deg,fs[0],p,'S_P_PLUS',t);checks(deg,fs[0],p,a,b,c,'S_P_PLUS',t)
    assert audits['S_P_PLUS']['selected64_P_Gram_relative_max']>1e-5
    write(OUT/'self_test.json',dict(status='GRAM_LOCAL_BINDING_ALGEBRA_PASS',audits=audits,orthogonal_Q_max_error=float((q@q.T-t.eye(16)).abs().max())))
    emit(stage='TEST_PASS')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','test','pilot','work','join']);a.add_argument('--index',type=int,default=0);a.add_argument('--budget',type=int,default=420);x=a.parse_args();t=configure()
    if x.stage=='test':test(t)
    elif x.stage=='prepare':prepare(t)
    elif x.stage=='join':join(t)
    else:sys.exit(worker(x.index,x.budget,t,x.stage=='pilot'))
