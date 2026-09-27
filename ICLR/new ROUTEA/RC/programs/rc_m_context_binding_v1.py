#!/usr/bin/env python3
"""Bounded A/J/P context registration intervention with cached CPU inputs.

No encoder/matcher, new candidates, fitting or GPU computation. Existing A0/A1,
J and P are rolled on a fixed grid, then the frozen DPT is rerun. A downstream
stage propagates the same M worlds through frozen external and POST models.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
SOURCE=RC/'results/rc_m_causal128_attribution_chain_v1/phase_replication'
BRIDGE=RC/'results/rc_m_causal128_attribution_chain_v1/bridge'
OUT=RC/'results/rc_m_context_binding_v1'
PLAN=RC/'plan/RC_M_CONTEXT_BINDING_V1_20260927.md'
DIRECTIONS={'X_PLUS':(0,4),'X_MINUS':(0,-4),'Y_PLUS':(4,0),'Y_MINUS':(-4,0)}
BITS=[''.join(map(str,x)) for x in itertools.product((0,1),repeat=3)]
ARMS=['NATIVE']+[d+'__'+b for d in DIRECTIONS for b in BITS if b!='000']
INDICES=[78,82,93,96,100,101,104,108,118]


def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())
def checked(b):
    assert bind(b['path'])==b,('BINDING_CHANGED',b['path']);return Path(b['path'])
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text,('IMMUTABLE_OUTPUT',str(p));return
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(text)
    try:os.link(temp,p)
    except FileExistsError:assert p.read_text()==text
    finally:temp.unlink(missing_ok=True)
def save(p,value,torch):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(value,temp);os.link(temp,p)
    finally:temp.unlink(missing_ok=True)
def configure():
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    return torch
def emit(**x):print(json.dumps(x,allow_nan=False),flush=True)
def recipe(arm):
    if arm=='NATIVE':return (0,0),'000'
    direction,bits=arm.split('__');return DIRECTIONS[direction],bits
def roll(x,shift,torch):return torch.roll(x,shifts=shift,dims=(1,2))
def transform(fs,j,p,arm,torch):
    shift,bits=recipe(arm)
    a=[roll(x,shift,torch) if bits[0]=='1' else x.clone() for x in fs]
    jj=roll(j,shift,torch) if bits[1]=='1' else j.clone()
    pp=roll(p,shift,torch) if bits[2]=='1' else p.clone()
    return a,jj,pp
def scalar_effects(values):
    results=[]
    for direction in DIRECTIONS:
        y={b:values['NATIVE' if b=='000' else direction+'__'+b] for b in BITS}
        i0=y['011']-y['010']-y['001']+y['000']
        i1=y['111']-y['110']-y['101']+y['100']
        results.append(dict(aligned_vs_misaligned=(y['000']+y['111']-y['001']-y['110'])/2,
            P_fixed_context_shift=y['110']-y['000'],common_grid_shift=y['111']-y['000'],
            P_only_shift=y['001']-y['000'],J_P_interaction_A_fixed=i0,
            J_P_interaction_A_shifted=i1,A_J_P_interaction=i1-i0))
    return {k:math.fsum(v[k] for v in results)/4 for k in results[0]}
def group_stats(values):
    a=np.asarray(values,np.float64);rng=np.random.default_rng(20260927)
    bt=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
    return dict(groups=len(a),mean=float(a.mean()),positive_groups=int((a>0).sum()),
                negative_groups=int((a<0).sum()),exploratory_bootstrap95=np.quantile(bt,[.025,.975]).tolist())


def prepare(root,torch):
    old=read(SOURCE/'protocol.json');v=read(SOURCE/'validation.json')
    assert v['status']=='PHASE_NEWGROUP_REPLICATION_JOIN_PASS' and v['groups']==9
    assert old['indices']==INDICES and v['protocol']==bind(SOURCE/'protocol.json')
    source=read(BRIDGE/'protocol.json');bv=read(BRIDGE/'validation.json')
    assert bv['status']=='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'
    for binding in source['code_sources']:checked(binding)
    workers=[];queries=[]
    for w in old['workers']:
        qdir=SOURCE/f"query{w['index']:03d}"
        qv=read(qdir/'validation.json');assert qv['target_present']
        queries.append({k:w[k] for k in ['index','query_id','group','fold','axis','target_position','fixed_wrong_position']})
        for pair in sorted(w['pairs'],key=lambda x:x['position']):
            pd=qdir/f"pair{pair['position']:03d}";pv=read(pd/'validation.json');checked(pv['capture'])
            desc=[]
            for d in pair['descriptors']:
                f=Path(d['output_path']) if d.get('encoding_needed') else checked(d['payload'])
                assert f.exists(),'ALL_A_DESCRIPTORS_MUST_ALREADY_EXIST'
                desc.append(bind(f))
            workers.append(dict(shard=len(workers),index=w['index'],query_id=w['query_id'],component=w['group'],
                fold=w['fold'],axis=w['axis'],position=pair['position'],physical_row=pair['physical_row'],
                role=pair['role'],target_position=w['target_position'],fixed_wrong_position=w['fixed_wrong_position'],
                capture=pv['capture'],features=desc,source_pair_validation=bind(pd/'validation.json'),
                source_native=next(x['payload'] for x in pv['arms'] if x['arm']=='NATIVE'),
                geometry_metadata=pair['geometry_metadata'],image_shas=pair['image_shas']))
    assert len(workers)==18
    import probe_rc_roma_property_restore_cpu_v3 as h
    code=[Path(__file__),PLAN,RC/'programs/run_rc_m_phase_newgroups_cpu_v1.py',
          Path(h.__file__),RC/'programs/rc_roma_shared_native_cache_v1.py',
          RC/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py',
          RC/'programs/run_rc_m_causal128_frozen_bridge_v1.py',
          RC/'programs/run_rc_m_causal128_newgroup_property_bridge_v1.py',h.ROMA/'dpt.py']
    write(root/'self_test.json',self_test(torch))
    p=dict(status='CONTEXT_BINDING_PROTOCOL_FROZEN',version=1,code_sources=[bind(x) for x in code],
        source_protocol=bind(SOURCE/'protocol.json'),source_validation=bind(SOURCE/'validation.json'),
        bridge_protocol=bind(BRIDGE/'protocol.json'),bridge_validation=bind(BRIDGE/'validation.json'),
        inherited_bridge_code_sources=source['code_sources'],
        bridge_inputs=source['inputs'],head_state=old['head_state'],queries=queries,workers=workers,
        directions={k:list(v) for k,v in DIRECTIONS.items()},arms=ARMS,grid=[50,50],coarse_shift=4,
        threads=4,torch_version=str(torch.__version__),native_timing=bind(root/'native_timing.json'),
        self_test=bind(root/'self_test.json'),native_hook_validation=bind(root/'native_input_hook_validation.json'),
        permutation_definition='Same cyclic grid shift on A0 AND A1 iff A bit=1, J iff J bit=1, P iff P bit=1. Bits ordered A,J,P. Both query/reference sides receive the same direction; four directions are within-group replicates.',
        full_Gram_guarantee='Exact row permutation algebraically preserves full channel Gram, field-vector multiset and toroidal autocorrelation. Verify full inverse permutation bit-exact, all-channel P FFT amplitude, fixed 32-channel Gram projection.',
        primary='Average across four directions of [g000+g111-g001-g110]/2 for target-minus-fixed-wrong logM; also separate target/wrong and same frozen external/POST propagation.',
        native_scope='NATIVE is unchanged cached COARSE DPT input on CPU, not the original system final HR1 M. All registered contrasts compare this same coarse endpoint.',
        bridge_scope='M-only propagation: frozen external local weighting shape and content stay original while M and its algebraic descendants change; POST receives altered M with same hidden/reference tokens. This is not total propagation of every altered upstream weight field.',
        subsidiary=list(scalar_effects({a:0. for a in ARMS})),
        pointwise='DPT receives A0 and rounded A1+J+P. Save per-patch centered variances/covariances, context-P cosine, true FP32 summed-input LN outputs and rounding closure. Common roll must commute with sum and pointwise LN; full DPT need not commute.',
        boundary='Already opened phase9 mechanism-development panel, not full128 or untouched replication. Frozen latent input registration is not a transformed-photo intervention. DPT convolution/padding/resize and token pooling can contribute; common-roll control quantifies but does not eliminate them. J/P with equal summed input are structurally indistinguishable at this decoder.',
        no_unique_semantic_or_causal_claim=True,no_full_C128_accuracy=True,new_encoder_or_matcher_forwards=0,
        new_gpu_forwards=0,new_training=0,intermediate_retention='Keep A/J/P source bindings plus exact grid indices; retain per-arm per-patch algebra, FP32 LN_sum tensors, full confidence, FP64 weights, M, and downstream maxsim/argmax. No destructive cache pruning.')
    write(root/'protocol.json',p)
    write(root/'preparation_validation.json',dict(status='CONTEXT_BINDING_PREPARED',protocol=bind(root/'protocol.json'),pairs=18,groups=9,arms=29,
        new_non_native_DPT_outputs_read=False,input_permutation_algebra_may_be_checked_before_freeze=True))
    emit(status='CONTEXT_BINDING_PREPARED',pairs=18,arms=29)


def guard(root,torch):
    p=read(root/'protocol.json');assert p['status']=='CONTEXT_BINDING_PROTOCOL_FROZEN'
    assert p['arms']==ARMS and p['threads']==4 and p['torch_version']==str(torch.__version__)
    for b in p['code_sources']:checked(b)
    for b in p['inherited_bridge_code_sources']:checked(b)
    for key in ['source_protocol','source_validation','bridge_protocol','bridge_validation','bridge_inputs','head_state','self_test','native_hook_validation','native_timing']:checked(p[key])
    return p,bind(root/'protocol.json')
def load_head(p,torch):
    import probe_rc_roma_property_restore_cpu_v3 as h
    DPT=h.load_cpu_dpt(torch)
    state=torch.load(checked(p['head_state']),map_location='cpu',weights_only=True,mmap=True)
    head=DPT(dim_in=1024,out_dim=3,pos_embed=False,feature_only=False,down_ratio=4).eval().requires_grad_(False)
    head.load_state_dict(state['head'],strict=True);assert not head.pos_embed
    return head
def load_inputs(w,torch):
    cap=torch.load(checked(w['capture']),map_location='cpu',weights_only=True,mmap=True)
    fs=[torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)['features'] for b in w['features']]
    assert all(len(x)==2 for x in fs)
    for side in range(2):
        assert all(tuple(x.shape)==(1,50,50,1024) for x in [*fs[side],cap['sides'][side]['J'],cap['sides'][side]['P']])
    return fs,cap
def tensor_sha(x,torch):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def invariants(fs,cap,torch):
    records=[];channels=torch.linspace(0,1023,32).long()
    for side in range(2):
        fields={'A0':fs[side][0],'A1':fs[side][1],'J':cap['sides'][side]['J'],'P':cap['sides'][side]['P']}
        for name,x in fields.items():
            x=x.clone();small=x[...,channels].reshape(-1,32).double();gram=small.T@small
            spectrum=torch.fft.rfftn(x.double(),dim=(1,2)).abs() if name=='P' else None
            for direction,shift in DIRECTIONS.items():
                y=roll(x,shift,torch);assert torch.equal(roll(y,tuple(-v for v in shift),torch),x)
                ss=y[...,channels].reshape(-1,32).double();err=float((ss.T@ss-gram).abs().max())
                rel=err/max(1.,float(gram.abs().max()));assert rel<1e-12
                fftrel=None
                if spectrum is not None:
                    fftrel=float((torch.fft.rfftn(y.double(),dim=(1,2)).abs()-spectrum).abs().max())/max(1.,float(spectrum.max()))
                    assert fftrel<1e-12
                records.append(dict(side=side,field=name,direction=direction,source_tensor_sha256=tensor_sha(x,torch),
                    inverse_roll_bit_exact=True,selected_Gram_channels=channels.tolist(),selected_Gram_relative_error=rel,
                    full_P_FFT_amplitude_relative_error=fftrel))
    return records


def pointwise(a,j,p,actual_sum,torch):
    aa=a[-1].double();jj=j.double();pp=p.double();xs=[aa,jj,pp]
    centered=[x-x.mean(-1,keepdim=True) for x in xs]
    var=[x.square().mean(-1) for x in centered]
    cov=[(centered[i]*centered[k]).mean(-1) for i,k in [(0,1),(0,2),(1,2)]]
    exact_sum=aa+jj+pp;cc=exact_sum-exact_sum.mean(-1,keepdim=True)
    expected=var[0]+var[1]+var[2]+2*(cov[0]+cov[1]+cov[2])
    residual=cc.square().mean(-1)-expected
    assert float(residual.abs().max())<1e-9*max(1.,float(expected.abs().max()))
    rounded=actual_sum.double();rc=rounded-rounded.mean(-1,keepdim=True)
    ctx=aa+jj;cos=(ctx*pp).sum(-1)/(ctx.norm(dim=-1)*pp.norm(dim=-1)).clamp_min(1e-30)
    data=dict(var_A1=var[0],var_J=var[1],var_P=var[2],cov_AJ=cov[0],cov_AP=cov[1],cov_JP=cov[2],
        var_sum_exact=cc.square().mean(-1),var_sum_actual=rc.square().mean(-1),
        variance_identity_residual=residual,actual_sum_rounding_variance_residual=rc.square().mean(-1)-expected,
        context_P_cosine=cos,norm_A1=aa.norm(dim=-1),norm_J=jj.norm(dim=-1),norm_P=pp.norm(dim=-1),
        norm_context=ctx.norm(dim=-1),norm_sum=rounded.norm(dim=-1))
    return data,{k:float(v.mean()) for k,v in data.items()}


def independent_pool(conf,geometry,weights):
    """Literal NumPy integer rectangles, separate from ExactPool implementation."""
    image=conf[0,...,0].sigmoid().double().numpy();height,width=image.shape;pooled=[]
    for box,valid in zip(geometry.cell_boxes_xyxy,geometry.valid_patch_mask):
        if not bool(valid):pooled.append(0.);continue
        x0,y0,x1,y1=map(float,box)
        left=max(0,min(width-1,math.floor(x0*width)));top=max(0,min(height-1,math.floor(y0*height)))
        right=max(left+1,min(width,math.ceil(x1*width)));bottom=max(top+1,min(height,math.ceil(y1*height)))
        pooled.append(float(image[top:bottom,left:right].mean()))
    error=float(np.max(abs(np.asarray(pooled)-weights.numpy())))
    assert error<2e-12,'INDEPENDENT_POOL_ERROR'
    return error


def worker(root,shard,budget,torch):
    p,pb=guard(root,torch);assert 0<=shard<len(p['workers']);w=p['workers'][shard]
    folder=root/'pairs'/f'{shard:03d}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json');assert v['protocol']==pb
            for a in v['arms']:checked(a['payload']);checked(a['intermediates'])
            return 0
        started=time.monotonic();deadline=started+budget
        fs,cap=load_inputs(w,torch);head=load_head(p,torch)
        assert cap['protocol']==p['source_protocol']
        assert cap['pair']['position']==w['position'] and cap['pair']['physical_row']==w['physical_row']
        assert cap['pair']['image_shas']==w['image_shas']
        import run_rc_m_phase_newgroups_cpu_v1 as old
        from rc_roma_shared_native_cache_v1 import ExactPool
        gs=old.geometries(w);pool=ExactPool()
        invpath=folder/'input_invariants.json'
        if invpath.exists():assert read(invpath)['protocol']==pb
        else:write(invpath,dict(protocol=pb,capture=w['capture'],features=w['features'],records=invariants(fs,cap,torch)))
        native_ln=[]
        with torch.inference_mode():
            for side in range(2):
                s=fs[side][-1]+cap['sides'][side]['J']+cap['sides'][side]['P']
                native_ln.append(head.norm(s.reshape(1,-1,1024)).reshape_as(s))
        records=[]
        for arm in ARMS:
            path=folder/(arm+'.pt');lnpath=folder/(arm+'.ln.pt');receipt=folder/(arm+'.json')
            if receipt.exists():
                r=read(receipt);assert r['protocol']==pb and r['arm']==arm;checked(r['payload']);checked(r['intermediates']);records.append(r);continue
            if deadline-time.monotonic()<30:return 75
            tick=time.monotonic();sides=[];lns=[];shift,bits=recipe(arm)
            with torch.inference_mode():
                for side in range(2):
                    a,j,pp=transform(fs[side],cap['sides'][side]['J'],cap['sides'][side]['P'],arm,torch)
                    summed=a[-1]+j+pp
                    if bits=='111':
                        native_sum=fs[side][-1]+cap['sides'][side]['J']+cap['sides'][side]['P']
                        assert torch.equal(summed,roll(native_sum,shift,torch)),'COMMON_SUM_NOT_EQUIVARIANT'
                    algebra,means=pointwise(a,j,pp,summed,torch)
                    ln_capture=[]
                    def hook(module,args,out):ln_capture.append(out.detach())
                    token=head.norm.register_forward_hook(hook)
                    try:pred=head([a[0],summed],img_A=None,img_B=None)
                    finally:token.remove()
                    assert len(ln_capture)==4 and torch.equal(ln_capture[2],ln_capture[3])
                    actual_ln=ln_capture[2].reshape_as(summed).clone();assert actual_ln.dtype==torch.float32
                    assert torch.equal(actual_ln,head.norm(summed.reshape(1,-1,1024)).reshape_as(summed))
                    commutator=0.
                    if bits=='111':
                        commutator=float((actual_ln-roll(native_ln[side],shift,torch)).abs().max())
                        assert commutator==0,'POINTWISE_LN_NOT_EQUIVARIANT'
                    conf=pred[...,2:].clone();assert bool(torch.isfinite(conf).all())
                    if arm=='NATIVE':assert torch.equal(conf,cap['sides'][side]['confidence']),'NATIVE_CPU_REPLAY_DRIFT'
                    weights=pool(conf[0,...,0].sigmoid(),gs[side]);dense=float(conf.sigmoid().double().mean())
                    pool_error=independent_pool(conf,gs[side],weights)
                    lncos=(actual_ln.double()*native_ln[side].double()).sum(-1)/(actual_ln.double().norm(dim=-1)*native_ln[side].double().norm(dim=-1)).clamp_min(1e-30)
                    boundary=None
                    if bits=='111':
                        aligned=torch.roll(conf,shifts=(-4*shift[0],-4*shift[1]),dims=(1,2))
                        diff=aligned.double()-cap['sides'][side]['confidence'].double()
                        boundary=dict(nominal_dense_shift=[4*v for v in shift],inverse_rolled_confidence_RMS=float(diff.square().mean().sqrt()),
                            inverse_rolled_confidence_max_abs=float(diff.abs().max()),not_assumed_translation_equivariant=True)
                    sides.append(dict(confidence=conf,weights=weights,preLN=algebra,preLN_means=means,
                        dense_sigmoid_mean=dense,token_pool_mean=float(weights.mean()),
                        independent_pool_max_error=pool_error,LN_to_native_cosine=lncos,LN_to_native_cosine_mean=float(lncos.mean()),
                        pointwise_LN_common_roll_max_error=commutator if bits=='111' else None,common_grid_diagnostic=boundary))
                    lns.append(actual_ln)
            mass=float(torch.sqrt(sides[0]['weights'].mean()*sides[1]['weights'].mean()))
            assert 0<mass<=1
            lp=dict(protocol=pb,shard=shard,index=w['index'],position=w['position'],arm=arm,LN_sum=lns)
            if lnpath.exists():
                prior=torch.load(lnpath,map_location='cpu',weights_only=True,mmap=True)
                assert prior['protocol']==pb and all(torch.equal(x,y) for x,y in zip(prior['LN_sum'],lns))
            else:save(lnpath,lp,torch)
            value=dict(protocol=pb,shard=shard,index=w['index'],query_id=w['query_id'],position=w['position'],physical_row=w['physical_row'],
                arm=arm,M=mass,sides=sides,intermediates=bind(lnpath),capture=w['capture'],features=w['features'],
                recipe=dict(bits=bits,shift=list(shift),grid=[50,50]),seconds=time.monotonic()-tick)
            if path.exists():
                prior=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                assert prior['protocol']==pb and prior['M']==mass and all(torch.equal(x['confidence'],y['confidence']) for x,y in zip(prior['sides'],sides))
            else:save(path,value,torch)
            record=dict(protocol=pb,arm=arm,payload=bind(path),intermediates=bind(lnpath),M=mass)
            write(receipt,record);records.append(record);emit(stage='CONTEXT_ARM_SAVED',shard=shard,arm=arm,seconds=value['seconds'])
        write(folder/'validation.json',dict(status='CONTEXT_BINDING_PAIR_PASS',protocol=pb,shard=shard,index=w['index'],
            query_id=w['query_id'],position=w['position'],component=w['component'],arms=records,input_invariants=bind(invpath)))
        return 0


def independent_pair(root,p,pb,w,torch):
    folder=root/'pairs'/f"{w['shard']:03d}";v=read(folder/'validation.json')
    assert v['status']=='CONTEXT_BINDING_PAIR_PASS' and v['protocol']==pb and v['position']==w['position']
    checked(v['input_invariants']);rows={};maximum=0.
    for record in v['arms']:
        x=torch.load(checked(record['payload']),weights_only=True,mmap=True,map_location='cpu');checked(record['intermediates'])
        assert x['protocol']==pb and x['arm']==record['arm'] and x['position']==w['position'] and x['physical_row']==w['physical_row']
        means=[math.fsum(s['weights'].tolist())/s['weights'].numel() for s in x['sides']]
        mass=math.sqrt(means[0]*means[1]);error=abs(mass-x['M']);assert error<2e-12;maximum=max(maximum,error)
        rows[x['arm']]=dict(M=mass,logM=math.log(mass),payload=record['payload'],
            dense_M=math.sqrt(math.prod(s['dense_sigmoid_mean'] for s in x['sides'])),
            sides=[dict(preLN_means=s['preLN_means'],dense_sigmoid_mean=s['dense_sigmoid_mean'],token_pool_mean=s['token_pool_mean'],
                        independent_pool_max_error=s['independent_pool_max_error'],LN_to_native_cosine_mean=s['LN_to_native_cosine_mean'],
                        common_grid_diagnostic=s['common_grid_diagnostic']) for s in x['sides']])
    assert list(rows)==ARMS
    return rows,maximum


def join(root,torch):
    p,pb=guard(root,torch);pairs={};seals=[];maximum=0.
    for w in p['workers']:
        v=root/'pairs'/f"{w['shard']:03d}"/'validation.json'
        if not v.exists():return 75
        pair,error=independent_pair(root,p,pb,w,torch);pairs[(w['index'],w['role'])]=pair;maximum=max(maximum,error);seals.append(bind(v))
    rows=[]
    for q in p['queries']:
        t=pairs[(q['index'],'target')];w=pairs[(q['index'],'fixed_wrong')]
        metrics={role:{arm:values[arm]['logM'] for arm in ARMS} for role,values in [('target_logM',t),('fixed_wrong_logM',w)]}
        metrics['logM_gap']={arm:t[arm]['logM']-w[arm]['logM'] for arm in ARMS}
        metrics['dense_logM_gap']={arm:math.log(t[arm]['dense_M'])-math.log(w[arm]['dense_M']) for arm in ARMS}
        for role,values in [('target',t),('fixed_wrong',w)]:
            for side in range(2):
                for name in ['cov_AJ','cov_AP','cov_JP','var_sum_actual','context_P_cosine']:
                    metrics[f'{role}_side{side}_{name}']={a:values[a]['sides'][side]['preLN_means'][name] for a in ARMS}
                metrics[f'{role}_side{side}_LN_to_native_cosine']={a:values[a]['sides'][side]['LN_to_native_cosine_mean'] for a in ARMS}
        rows.append(dict(index=q['index'],query_id=q['query_id'],component=q['group'],
                         target=t,fixed_wrong=w,effects={name:scalar_effects(vals) for name,vals in metrics.items()}))
    keys=list(rows[0]['effects']);contrasts=list(rows[0]['effects'][keys[0]])
    summary={m:{c:group_stats([r['effects'][m][c] for r in rows]) for c in contrasts} for m in keys}
    result=dict(status='CONTEXT_BINDING_UPSTREAM_COMPLETE',protocol=pb,groups=9,pairs=18,arms=29,rows=rows,summary=summary,
                primary=p['primary'],boundary=p['boundary'],new_encoder_or_matcher_forwards=0,new_gpu_forwards=0,new_training=0)
    write(root/'upstream_result.json',result)
    write(root/'upstream_validation.json',dict(status='CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS',protocol=pb,
        pair_validations=seals,result=bind(root/'upstream_result.json'),groups=9,pairs=18,candidate_arms=522,
        independent_M_max_error=maximum,full_C128_accuracy=False))
    emit(status='CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS');return 0


def bridge_worker(root,budget,torch,index=None):
    p,pb=guard(root,torch)
    if not (root/'upstream_validation.json').exists():return 75
    up=read(root/'upstream_validation.json');assert up['status']=='CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS' and up['protocol']==pb
    import run_rc_m_causal128_frozen_bridge_v1 as B
    import run_rc_m_causal128_newgroup_property_bridge_v1 as NB
    bp=read(checked(p['bridge_protocol']));context=B.Context(bp)
    items={r['index']:r for r in read(checked(p['bridge_inputs']))['rows']};deadline=time.monotonic()+budget
    queries=p['queries'] if index is None else [p['queries'][index]]
    for q in queries:
        folder=root/'bridge'/f"query{q['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json');assert v['protocol']==pb;checked(v['result']);continue
        if deadline-time.monotonic()<25:return 75
        item=items[q['index']];assert item['query_id']==q['query_id'] and item['post_row']['candidate_ids']==q['axis']
        adapter,head,cache,refs=context.load(item);records=[];patches=[];bypos={}
        ws=sorted([w for w in p['workers'] if w['index']==q['index']],key=lambda w:w['position'])
        for w in ws:
            v=read(root/'pairs'/f"{w['shard']:03d}"/'validation.json');ms=[r['M'] for r in v['arms']]
            pos=w['position'];patch=folder/f'pair{pos:03d}.npz'
            if patch.exists():
                with np.load(patch) as z:
                    assert z['worlds'].tolist()==ARMS and np.array_equal(z['M'],ms)
                    vals=z['maxsim'].copy();locations=z['argmax'].copy()
            else:
                vals,locations=context.candidate(adapter,cache,refs[pos],ms)
                B.npz_once(patch,worlds=np.asarray(ARMS),M=np.asarray(ms),maxsim=vals,argmax=locations,L=vals.mean(1))
            patches.append(bind(patch));records.extend(r['payload'] for r in v['arms']);bypos[pos]=dict(M=ms,L=vals.mean(1))
        original=RC/'results/rc_postllm_h593_decomposition_v1/queries'/q['query_id']/'result.json'
        L=read(original)['arms']['NATIVE']['L'];arms={};individual={}
        for i,arm in enumerate(ARMS):
            ms=np.asarray(item['masses']['ORIGINAL_HR1']).copy();ls=np.asarray(L).copy()
            for pos in bypos:ms[pos]=bypos[pos]['M'][i];ls[pos]=bypos[pos]['L'][i]
            arms[arm],individual[arm]=NB.score_metrics(item,ms,ls,head,q['target_position'],q['fixed_wrong_position'])
        row=dict(status='CONTEXT_BINDING_BRIDGE_QUERY_COMPLETE',protocol=pb,index=q['index'],query_id=q['query_id'],component=q['group'],
            fold=q['fold'],candidate_physical_rows=q['axis'],target_position=q['target_position'],fixed_wrong_position=q['fixed_wrong_position'],
            RAW_winner=item['post_row']['winner_index'],head=head,snapshot=bp['endpoints'][str(q['fold'])]['snapshot'],
            original_POST=bind(original),saved_arm_inputs=records,patch_outputs=patches,arms=arms,individual_endpoints=individual,
            full_C128_predictions_computed=False,anchor_policy='Only fixed target/wrong change; if RAW winner is outside pair hold its original M/L. HOLD is zero.')
        error=NB.audit_row(row,item,ARMS)
        write(folder/'result.json',row);write(folder/'validation.json',dict(status='CONTEXT_BINDING_BRIDGE_QUERY_INDEPENDENT_PASS',
            protocol=pb,result=bind(folder/'result.json'),independent_numpy_max_error=error))
        emit(stage='CONTEXT_BINDING_BRIDGE_QUERY_PASS',index=q['index'])
    return 0


def bridge_join(root,torch):
    p,pb=guard(root,torch);up=read(root/'upstream_validation.json');assert up['status']=='CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS'
    source=read(checked(up['result']));uprows={r['index']:r for r in source['rows']}
    import run_rc_m_causal128_frozen_bridge_v1 as B
    import run_rc_m_causal128_newgroup_property_bridge_v1 as NB
    bp=read(checked(p['bridge_protocol']))
    items={r['index']:r for r in read(checked(p['bridge_inputs']))['rows']};rows=[];seals=[];maximum=0.;closure=0.
    for q in p['queries']:
        path=root/'bridge'/f"query{q['index']:03d}"/'validation.json'
        if not path.exists():return 75
        v=read(path);assert v['protocol']==pb and v['status']=='CONTEXT_BINDING_BRIDGE_QUERY_INDEPENDENT_PASS'
        row=read(checked(v['result']));assert row['index']==q['index'] and row['query_id']==q['query_id']
        assert row['protocol']==pb and row['component']==q['group'] and row['fold']==q['fold']
        assert row['candidate_physical_rows']==q['axis']
        assert row['target_position']==q['target_position'] and row['fixed_wrong_position']==q['fixed_wrong_position']
        assert row['snapshot']==bp['endpoints'][str(q['fold'])]['snapshot']
        snap=torch.load(checked(row['snapshot']),map_location='cpu',weights_only=True,mmap=True)
        assert row['head']==snap['head'].double().tolist() and snap['arm']=='POST_REAL'
        maximum=max(maximum,NB.audit_row(row,items[q['index']],ARMS))
        row['effects']={m:scalar_effects({a:row['arms'][a][m] for a in ARMS}) for m in NB.GAP_METRICS}
        for c,x in row['effects']['logM_gap'].items():closure=max(closure,abs(x-uprows[q['index']]['effects']['logM_gap'][c]))
        rows.append(row);seals.append(bind(path))
    assert closure<2e-10
    names=list(rows[0]['effects']['logM_gap']);summary={m:{c:group_stats([r['effects'][m][c] for r in rows]) for c in names} for m in NB.GAP_METRICS}
    individual={role:{m:{c:group_stats([scalar_effects({a:r['individual_endpoints'][a][role][m] for a in ARMS})[c] for r in rows])
        for c in names} for m in NB.SIDE_METRICS} for role in ['target','fixed_wrong']}
    result=dict(status='CONTEXT_BINDING_FROZEN_PROPAGATION_COMPLETE',protocol=pb,groups=9,pairs=18,arms=29,
        upstream=bind(root/'upstream_result.json'),rows=rows,gap_summary=summary,individual_summary=individual,
        same_frozen_models=True,new_parameters=0,new_encoder_matcher_or_LLM_forwards=0,new_gpu_forwards=0,
        full_C128_predictions_computed=False,unique_causality_claim=False,boundary=p['boundary'])
    write(root/'result.json',result)
    text=['# A/J/P 循环配准：性质与冻结外部／POST 传播','',p['boundary'],'',
          '主要配准对比：四方向内平均 [000+111−001−110]/2；其余单项及交互全部预先冻结。','',
          '| 对比 | logM gap | 外部NATIVE7 gap | 外部M_FREE gap | POST内容gap | POST action gap |','|---|---:|---:|---:|---:|---:|']
    for c in names:text.append('| '+c+' | '+' | '.join(f"{summary[m][c]['mean']:+.8f}" for m in ['logM_gap','NATIVE7_action_gap','M_FREE_action_gap','POST_content_gap','POST_action_gap'])+' |')
    text+=['','所有逐patch方差/交叉项、LN_sum、DPT置信度、池化weights及POST patch maxsim/argmax均保留。',
           '相位图的内部结构、输入配准与解码器网格响应不能通过该有限实验宣称唯一因果；目标/固定错误配对结果不是完整C128检索准确率。','']
    report=root/'report.md';rendered='\n'.join(text)
    if report.exists():assert report.read_text()==rendered
    else:report.write_text(rendered)
    write(root/'validation.json',dict(status='CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS',protocol=pb,
        upstream_validation=bind(root/'upstream_validation.json'),bridge_query_validations=seals,
        result=bind(root/'result.json'),report=bind(report),groups=9,candidate_arms=522,
        independent_action_max_error=maximum,upstream_logM_contrast_closure_max_error=closure,new_gpu_forwards=0))
    emit(status='CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS');return 0


def self_test(torch):
    torch.manual_seed(13);fs=[torch.randn(1,10,12,16) for _ in range(2)];j=torch.randn_like(fs[0]);p=torch.randn_like(j)
    for arm in ARMS:
        a,jj,pp=transform(fs,j,p,arm,torch);shift,bits=recipe(arm)
        for original,changed,bit in [(fs[0],a[0],bits[0]),(fs[1],a[1],bits[0]),(j,jj,bits[1]),(p,pp,bits[2])]:
            recovered=roll(changed,tuple(-x for x in shift),torch) if bit=='1' else changed
            assert torch.equal(original,recovered)
        if bits=='111':assert torch.equal(a[-1]+jj+pp,roll(fs[-1]+j+p,shift,torch))
        if bits[2]=='0':assert torch.equal(pp,p)
        pointwise(a,jj,pp,a[-1]+jj+pp,torch)
    # Purely additive placement effects cancel in the primary matched contrast.
    v={}
    for arm in ARMS:
        shift,bits=recipe(arm);v[arm]=3*int(bits[0])+5*int(bits[1])+7*int(bits[2])
    assert scalar_effects(v)['aligned_vs_misaligned']==0
    # The decoder sees A0 and SUM; exact compensating J/P changes cannot be identified there.
    a=torch.ones(2,3);j=torch.full_like(a,2);p=torch.full_like(a,4);delta=torch.ones_like(a)
    assert torch.equal(a+j+p,a+(j+delta)+(p-delta))
    return dict(status='CONTEXT_BINDING_ALGEBRA_SELF_TEST_PASS',arms=29,A_both_levels=True,
        inverse_roll_exact=True,P_fixed_branches_exact=True,common_sum_equivariance=True,
        variance_decomposition=True,matched_additive_control_cancels=True,same_SUM_nonidentifiability=True,
        actual_new_intervention_data_read=False)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','self-test','worker','join','bridge-worker','bridge-join'])
    a.add_argument('--root',type=Path,default=OUT);a.add_argument('--shard','--pair-index',dest='shard',type=int)
    a.add_argument('--index',type=int,choices=range(9));a.add_argument('--budget',type=float,default=430)
    args=a.parse_args();torch=configure();root=args.root.resolve()
    if args.stage=='self-test':emit(**self_test(torch));return 0
    if args.stage=='prepare':prepare(root,torch);return 0
    if args.stage=='worker':return worker(root,args.shard,args.budget,torch)
    if args.stage=='join':return join(root,torch)
    if args.stage=='bridge-worker':return bridge_worker(root,args.budget,torch,args.index)
    return bridge_join(root,torch)


if __name__=='__main__':raise SystemExit(main())
