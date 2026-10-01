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
OUT=RC/'results/rc_m_structure_binding_isolation_v2'
OLD=RC/'results/rc_m_structure_binding_isolation_v1'
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
    old=read(OLD/'protocol.json'); inp=read(checked(old['inputs']))
    reused=[];rerun=[]
    # Inputs are identical in value; only avoid casting rotated features back to BF16.
    for w in inp['workers']:
        complete=(OLD/'pairs'/f"{w['ordinal']:03d}"/'validation.json').exists()
        # Verify source precision before declaring a former result unchanged.
        if complete:
            fs=[t.load(checked(b),map_location='cpu',weights_only=True,mmap=True)['features'] for b in w['features']]
            cap=t.load(checked(w['capture']),map_location='cpu',weights_only=True,mmap=True)
            dtypes=[x.dtype for side in fs for x in side]+[unpack(cap['sides'][i][k],t).dtype for i in range(2) for k in ['J','P']]
            assert all(x==t.float32 for x in dtypes),('REUSE_REQUIRES_FP32',w['ordinal'],dtypes)
            reused.append(w['ordinal'])
        else:rerun.append(w['ordinal'])
    assert reused==list(range(63)) and rerun==list(range(63,77))
    write(OUT/'inputs.json',inp)
    codes=old['code_sources']+[bind(__file__)]
    protocol=dict(old,inputs=bind(OUT/'inputs.json'),code_sources=codes,parent_protocol=bind(OLD/'protocol.json'),
        numeric_repair='All intervention input tensors promoted losslessly from stored BF16 values to FP32; transforms evaluated in FP64 and emitted FP32. All13 arms including NATIVE use this dtype. Original native dtype replay retained separately as a numeric control.',
        frozen_invariant_tolerances_unchanged=True,reused_pair_ordinals=reused,recompute_pair_ordinals=rerun,pilot_ordinal=63)
    write(OUT/'protocol.json',protocol);pb=bind(OUT/'protocol.json')
    for ordinal in reused:
        src=OLD/'pairs'/f'{ordinal:03d}';dest=OUT/'pairs'/f'{ordinal:03d}'
        v=read(src/'validation.json');assert v['protocol']==bind(OLD/'protocol.json')
        receipts=[]
        for arm in ARMS:
            row=read(src/f'{arm}.json');checked(row['payload'])
            r=dict(row,protocol=pb,reused_from=bind(src/f'{arm}.json'))
            write(dest/f'{arm}.json',r);receipts.append(r)
        checked(v['post_patch'])
        write(dest/'validation.json',dict(v,protocol=pb,arms=receipts,reused_from=bind(src/'validation.json')))
    emit(stage='PRECISION_REPAIR_PREPARED',reuse_pairs=len(reused),rerun_pairs=len(rerun),arms=len(ARMS))

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
    selected=[inp['workers'][p['pilot_ordinal']]] if pilot else inp['workers'][index::p['shards']]
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
                        source_j=unpack(cap['sides'][side]['J'],t);source_P=unpack(cap['sides'][side]['P'],t)
                        f32=[x.float() for x in fs[side]];j=source_j.float();P=source_P.float()
                        aa,jj,pp=transform(f32,j,P,arm,t);audit=checks(f32,j,P,aa,jj,pp,arm,t)
                        summed=aa[-1]+jj+pp;out=head([aa[0],summed],img_A=None,img_B=None);conf=out[...,2:].clone()
                        anchor_error=None;precision_native=None
                        if arm=='NATIVE' and w.get('cpu_native'):
                            anchor=t.load(checked(w['cpu_native']),map_location='cpu',weights_only=True,mmap=True)['sides'][side]['confidence']
                            anchor_error=float((conf-anchor).abs().max());assert anchor_error==0,('CPU_NATIVE_ANCHOR',w['ordinal'],side,anchor_error)
                        if arm=='NATIVE':
                            original=head([fs[side][0],fs[side][-1]+source_j+source_P],img_A=None,img_B=None)[...,2:]
                            prior=OLD/'pairs'/f"{w['ordinal']:03d}"/'NATIVE.json'
                            if prior.exists():
                                original_saved=t.load(checked(read(prior)['payload']),map_location='cpu',weights_only=True,mmap=True)['sides'][side]['confidence']
                                assert t.equal(original,original_saved),('ORIGINAL_DTYPE_NATIVE_DRIFT',w['ordinal'],side)
                            ow=pool(original[0,...,0].sigmoid(),geometry[side])
                            precision_native=dict(source_dtypes=[str(x.dtype) for x in [*fs[side],source_j,source_P]],
                                original_confidence=original.clone(),original_weights=ow,
                                FP32_vs_source_max_confidence=float((conf-original).abs().max()))
                            if pilot:
                                again=head([f32[0],f32[-1]+j+P],img_A=None,img_B=None)[...,2:]
                                assert t.equal(conf,again)
                        weights=pool(conf[0,...,0].sigmoid(),geometry[side]);err=C.independent_pool(conf,geometry[side],weights)
                        local=t.stack([(pp.double()*v.double()).sum(-1) for v in [*aa,jj]],0)
                        sides.append(dict(confidence=conf,weights=weights,input_audit=audit,local_inner_products=local,
                            P_norm=pp.double().norm(dim=-1),sum_mean=summed.double().mean(-1),sum_variance=summed.double().var(-1,unbiased=False),independent_pool_error=err,cpu_anchor_error=anchor_error,precision_native=precision_native))
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
        vp=OUT/'pairs'/f"{p['pilot_ordinal']:03d}"/'validation.json';v=read(vp);assert len(v['arms'])==len(ARMS)
        write(OUT/'pilot_validation.json',dict(status='ISOLATION_NATIVE_AND_INVARIANTS_PASS',protocol=pb,pair=bind(vp),self_test=bind(OUT/'self_test.json')))
    return 0

def join(t):
    p,pb,inp=guard(t);pairs={};source=[];maximum=0.;precision=[]
    for w in inp['workers']:
        f=OUT/'pairs'/f"{w['ordinal']:03d}";v=read(f/'validation.json');assert v['protocol']==pb;record={}
        with np.load(checked(v['post_patch'])) as z:
            assert z['worlds'].tolist()==ARMS;L=z['maxsim'].mean(1);assert np.array_equal(L,z['L'])
        for k,r in enumerate(v['arms']):
            x=t.load(checked(r['payload']),map_location='cpu',weights_only=True,mmap=True)
            M=float(np.sqrt(np.mean(x['sides'][0]['weights'].numpy())*np.mean(x['sides'][1]['weights'].numpy())))
            maximum=max(maximum,abs(M-r['M']));assert abs(M-r['M'])<2e-12
            if r['arm']=='NATIVE' and x['sides'][0].get('precision_native'):
                ss=[side['precision_native'] for side in x['sides']]
                before=math.sqrt(float(ss[0]['original_weights'].mean())*float(ss[1]['original_weights'].mean()))
                precision.append(dict(ordinal=w['ordinal'],index=w['index'],position=w['position'],source_M=before,FP32_M=M,
                    logM_difference=math.log(M/before),max_confidence_difference=max(v['FP32_vs_source_max_confidence'] for v in ss)))
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
    write(OUT/'result.json',dict(status='STRUCTURE_BINDING_OPERATIONAL_ISOLATION_COMPLETE',protocol=pb,rows=rows,summaries=summaries,sources=source,native_precision_controls=precision,
        caveats=['Repaired additional groups use a common FP32 input convention for all arms; original source-dtype native differences are reported, not counted as property effects.','Local binding invariants are not all semantic binding.','Q_ALL controls channel-basis sensitivity; Q_P alone does not isolate binding.',
                 'Effects are conditional, signed and potentially order-dependent; no unique causal percentages.','Candidate-pair contrasts do not establish complete-C128 accuracy.']))
    write(OUT/'validation.json',dict(status='STRUCTURE_BINDING_INDEPENDENT_M_AND_PATCH_PASS',result=bind(OUT/'result.json'),pairs=len(pairs),maximum_M_recount_error=maximum))
    lines=['# P structure and local content binding: operational isolation','',p['source_scope'],'',
        'All new arms use FP32 input tensors, with original source-dtype native controls saved in result.json. Local binding means per-patch inner products/norm/normalization scale. It does not mean the full semantic relation has been preserved. Q_ALL explicitly tests frozen decoder basis sensitivity. No unique share or new accuracy is inferred.','',
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
    stored=[x.bfloat16() for x in fs];promoted=[x.float() for x in stored]
    mixed={}
    for arm in ARMS:
        a,b,c=transform(promoted,j,p,arm,t)
        mixed[arm]=checks(promoted,j,p,a,b,c,arm,t)
    old_a,old_j,old_p=transform(stored,j,p,'Q_ALL_PLUS',t)
    try:
        checks(stored,j,p,old_a,old_j,old_p,'Q_ALL_PLUS',t)
    except AssertionError:
        old_quantization_rejected=True
    else:
        raise AssertionError('Regression must reproduce the source-dtype quantization failure')
    write(OUT/'self_test.json',dict(status='GRAM_LOCAL_BINDING_ALGEBRA_PASS',audits=audits,mixed_precision_audits=mixed,
        old_quantization_rejected=old_quantization_rejected,orthogonal_Q_max_error=float((q@q.T-t.eye(16)).abs().max())))
    emit(stage='TEST_PASS')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','test','pilot','work','join']);a.add_argument('--index',type=int,default=0);a.add_argument('--budget',type=int,default=420);x=a.parse_args();t=configure()
    if x.stage=='test':test(t)
    elif x.stage=='prepare':prepare(t)
    elif x.stage=='join':join(t)
    else:sys.exit(worker(x.index,x.budget,t,x.stage=='pilot'))
