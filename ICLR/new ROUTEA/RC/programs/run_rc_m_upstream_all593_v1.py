#!/usr/bin/env python3
"""Sealed H593 upstream expansion: one input capture, all historical properties.

No GPU, retrieval-network training, or new POST forwards. Each causal contrast
uses a common FP32 input convention and CPU DPT; source-dtype drift is separate.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
sys.path[:0] = [str(RC/'programs'), str(RC/'src')]
OUT = RC/'results/rc_m_upstream_all593_v1'
REPLICATION = RC/'results/rc_m593_cached_replication_fix_v1'
OLD_ISO = RC/'results/rc_m_structure_binding_isolation_v2'
PLAN = RC/'plan/RC_M_UPSTREAM_ALL593_EXPANSION_20261009.md'
SHARDS = 50


def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''): h.update(block)
    return dict(path=str(p), sha256=h.hexdigest())
_verified = set()
def checked(b):
    key = (b['path'], b['sha256'])
    if key not in _verified:
        assert bind(b['path'])['sha256'] == b['sha256'], ('SOURCE_DRIFT', b['path'])
        _verified.add(key)
    return Path(b['path'])
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n'
    tmp = p.with_name(p.name+f'.{os.getpid()}.tmp'); tmp.write_text(text); os.replace(tmp,p)
def save(p,x,t):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp'); t.save(x,tmp); os.replace(tmp,p)
def emit(**x): print(json.dumps(x,allow_nan=False),flush=True)
def configure():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    return torch
def families():
    import rc_roma_position_factor_v1 as pos
    import rc_roma_property_restore_v2 as restore
    import rc_m_context_binding_v1 as context
    import run_rc_m_structure_binding_isolation_v2 as iso
    return dict(phase=list(dict.fromkeys([*pos.ARMS,*restore.ARMS])),
                context=list(context.ARMS),isolation=list(iso.ARMS))
def worlds():
    return ['NATIVE']+[f'{f}/{a}' for f,aa in families().items() for a in aa if a!='NATIVE']
def guard():
    a=read(OUT/'authority.json')
    for b in a['code_sources']:checked(b)
    return a
def protocol():
    guard();p=read(OUT/'protocol.json');assert p['worlds']==worlds()
    return p,bind(OUT/'protocol.json')
def source_roots():
    return [RC/'cache/rc_h593_shared_native_v1/images',
            RC/'results/rc_roma_position_factor_v1/images',
            RC/'results/rc_m_fine_c128_extension_v1/images',
            RC/'results/rc_m_causal128_attribution_chain_v1/phase_replication/images']


def prepare(t,budget):
    guard();deadline=time.monotonic()+budget
    if (OUT/'protocol.json').exists():protocol();return 0
    legacy=read(RC/'results/rc_m_causal128_attribution_chain_v1/phase_replication/protocol.json')
    pv=read(REPLICATION/'validation.json');rep=read(checked(pv['result']))
    assert rep['queries']==593 and pv['status']=='M593_SEALS_AND_OLD128_REPLAY_PASS'
    rp=read(REPLICATION/'protocol.json');cache=read(checked(rp['cache']))
    workers=read(RC/'results/rc_crisp_manual_baseline_v1/H593_workers.json')['records']
    oldiso=read(checked(read(OLD_ISO/'protocol.json')['inputs']))
    isomap={(w['index'],w['position']):w for w in oldiso['workers']}
    # Existing complete-C128 captures and fixed-pair CPU captures are both reused.
    image_catalog={x['sha256']:x for x in read(RC/'cache/rc_h593_shared_native_v1/catalog.json')['images']}
    qseals=[];images={};oldpair_sources={};last_payload=None;last_data=None
    for i in range(593):
        qp=OUT/'inputs'/f'{i:03d}.json'
        if qp.exists():
            q=read(qp);qseals.append(bind(qp))
            for d in q['images']:images[d['image']['sha256']]=d
            continue
        if time.monotonic()>deadline-30:return 75
        row=cache['rows'][i];w=workers[i]
        assert w['execution_ordinal']==row['execution_ordinal']==i and w['query_id']==row['query_id']
        b=w['roma']['payload']
        if b['path']!=last_payload:
            last_data=t.load(checked(b),map_location='cpu',weights_only=True,mmap=True);last_payload=b['path']
        orig=last_data['records'][w['source_index']]
        assert orig['query_id']==w['source_query_id'] and orig['candidate_physical_rows']==row['axis']
        im_path=RC/f'results/rc_h593_m_inside_v1/query{i:03d}/inside_manifest.json'
        im=read(im_path);assert len(im['records'])==128 and im['query_id']==row['query_id']
        repq=read(checked(rep['query_receipts'][i]))['row']
        assert repq['index']==i and repq['axis']==row['axis']
        oldgv=None
        for base in [RC/'results/rc_roma_position_factor_v1',RC/'results/rc_m_fine_c128_extension_v1']:
            f=base/f'query{i:03d}/gpu_validation.json'
            if f.exists():oldgv=read(f);break
        qi=[];pairs=[]
        allsha=[orig['query_source_sha256']]+[z['reference_image_sha256'] for z in orig['candidates']]
        for sha in dict.fromkeys(allsha):
            image=image_catalog[sha];desc=None
            for root in source_roots():
                receipt=root/f'{sha}.json'
                if receipt.exists():
                    d=read(receipt)
                    if 'payload' in d and Path(d['payload']['path']).exists():
                        desc=dict(receipt=bind(receipt),payload=d['payload']);break
                tensor=root/f'{sha}.pt'
                if tensor.exists():desc=dict(payload=bind(tensor));break
            d=dict(image=image,descriptor=desc);images[sha]=d;qi.append(d)
        for pos,c in enumerate(orig['candidates']):
            old=isomap.get((i,pos));capture=None;features=None;iso_reuse=None
            if old:
                capture=old['capture'];features=old['features']
                iso_reuse=str(OLD_ISO/'pairs'/f"{old['ordinal']:03d}")
            elif oldgv:
                pair=t.load(checked(oldgv['pairs'][pos]),weights_only=True,mmap=True,map_location='cpu')
                if 'capture' in pair:
                    cap=t.load(checked(pair['capture']),weights_only=True,mmap=True,map_location='cpu')
                    capture=pair['capture'];features=cap.get('descriptor_sources',cap.get('features'))
                    if features is None:
                        features=[cap[k] for k in ['query_features','reference_features']] if 'query_features' in cap else None
                    if features is None:
                        # Historical descriptors are pinned in the capture under these keys.
                        features=[cap[k] for k in ['query_binding','reference_binding']] if 'query_binding' in cap else None
                    if features is None:capture=None
            pairs.append(dict(index=i,position=pos,physical_row=row['axis'][pos],
                image_shas=[allsha[0],allsha[pos+1]],geometry_metadata=[orig['query_geometry'],c['reference_geometry']],
                old_capture=capture,old_features=features,old_isolation=iso_reuse,
                old_gpu_M=im['records'][pos]['branches']['A1J1P1']['COARSE']))
        q=dict(index=i,query_id=row['query_id'],component=repq['component'],fold=repq['fold'],
            axis=row['axis'],target=repq['target_position'],images=qi,pairs=pairs,
            sources=[w['roma']['payload'],bind(im_path),rep['query_receipts'][i]],
            fixed_wrong=repq['fixed_wrong_position'],original_row=row)
        write(qp,q);qseals.append(bind(qp));emit(stage='INPUT_MANIFEST',index=i)
    old=read(OLD_ISO/'protocol.json')
    p=dict(status='ALL593_UPSTREAM_PROTOCOL_FROZEN',authority=bind(OUT/'authority.json'),
           queries=593,candidates=128,pairs=593*128,shards=SHARDS,worlds=worlds(),families=families(),
        query_inputs=qseals,images=list(images.values()),checkpoint=legacy['checkpoint'],
           profile=legacy['profile'],head_state=legacy['head_state'],radius=legacy['radius'],
           old_isolation=bind(OLD_ISO/'protocol.json'),replication=bind(REPLICATION/'protocol.json'),
           upstream_replication=pv['result'],torch_version=str(t.__version__),threads=4,
           primary='Group-equal target minus fixed wrong logM; C_NEAREST,C_STRONGEST,M_STRONGEST separately.',
           source_dtype='Lossless promotion to FP32; original-dtype native control retained.',
           no_new_GPU=True,no_new_POST=True,unique_cause_claim=False,opened_H593=True,
           full_C128=True,retention='Captured tensors, confidence logits, FP64 weights, transforms, invariants, source SHA, external scores')
    assert len(p['worlds'])==60
    write(OUT/'protocol.json',p)
    write(OUT/'inventory.json',dict(queries=593,target_present=sum(read(x['path'])['target'] is not None for x in qseals),
         pairs=p['pairs'],worlds=60,candidate_worlds=p['pairs']*60,unique_images=len(images),
         descriptors_available=sum(x['descriptor'] is not None for x in images.values()),
         completed_reuse=['inside593','visual593','coordinate593','entry_external_replication593'],
         new_properties=['phase_and_restoration20','registration29','structure_binding13'],
         source_isolation_pairs=len(oldiso['workers']),new_POST=False))
    return 0


class Runtime:
    def __init__(self,p,t):
        import run_rc_m_phase_newgroups_cpu_v1 as G
        import probe_rc_roma_property_restore_cpu_v3 as H
        self.p=p;self.t=t;self.G=G
        DPT=H.load_cpu_dpt(t)
        state=t.load(checked(p['head_state']),map_location='cpu',weights_only=True,mmap=True)
        self.head=DPT(dim_in=1024,out_dim=3,pos_embed=False,feature_only=False,down_ratio=4).eval().requires_grad_(False)
        self.head.load_state_dict(state['head'],strict=True)
        self.omega=state['omega'];self.scale=float(state['scale']);self.matcher=None;self.encoder=None
        self.image_info={x['image']['sha256']:x for x in p['images']}
        self.memory={}
    def descriptor(self,sha):
        if sha in self.memory:return self.memory[sha]
        d=self.image_info[sha];folder=OUT/'images';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{sha}.pt';receipt=folder/f'{sha}.json'
        if d['descriptor']:
            b=d['descriptor']['payload'];z=self.t.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
        else:
            with (folder/f'{sha}.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if receipt.exists():b=read(receipt)['payload'];z=self.t.load(checked(b),weights_only=True,mmap=True,map_location='cpu')
                else:
                    if self.encoder is None:self.encoder=self.G.encoder(self.p,self.t)
                    z=self.G.encode_image(self.encoder,d['image'],self.t);save(path,z,self.t);b=bind(path)
                    write(receipt,dict(payload=b,image=d['image'],profile=self.p['profile']))
        assert z.get('profile',self.p['profile'])==self.p['profile']
        fs=z['features'];assert len(fs)==2 and all(tuple(x.shape)==(1,50,50,1024) for x in fs)
        self.memory[sha]=(fs,b)
        if len(self.memory)>4:self.memory.pop(next(iter(self.memory)))
        return fs,b
    def inputs(self,w,folder,pilot=False):
        import run_rc_m_structure_binding_isolation_v2 as I
        t=self.t;receipt=folder/'capture.json'
        if w['old_capture'] and w['old_features']:
            b=w['old_capture'];cap=t.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
            fs=[t.load(checked(v),map_location='cpu',weights_only=True,mmap=True)['features'] for v in w['old_features']]
            fb=w['old_features']
        elif receipt.exists():
            r=read(receipt);b=r['payload'];fb=r['features'];cap=t.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
            fs=[t.load(checked(v),map_location='cpu',weights_only=True,mmap=True)['features'] for v in fb]
        else:
            zs=[self.descriptor(s) for s in w['image_shas']];fs=[z[0] for z in zs];fb=[z[1] for z in zs]
            if self.matcher is None:
                mod=importlib.import_module('romav2.matcher');m=mod.Matcher(mod.Matcher.Cfg()).cpu().eval().requires_grad_(False)
                state=t.load(checked(self.p['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
                m.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('matcher.')},strict=True)
                self.matcher=(m,mod)
            cap=self.G.capture(*self.matcher,fs,t)
            if pilot:
                repeated=self.G.capture(*self.matcher,fs,t)
                for left,right in zip(cap['sides'],repeated['sides']):
                    for key in ['J','P','confidence','warp']:
                        assert t.equal(left[key],right[key]),('MATCHER_REPEAT',key)
                cap['CPU_repeat_exact']=True
            cap.update(index=w['index'],position=w['position'],features=fb,image_shas=w['image_shas'])
            save(folder/'capture.pt',cap,t);b=bind(folder/'capture.pt')
            write(receipt,dict(payload=b,features=fb,image_shas=w['image_shas']))
        native=[(fs[s],I.unpack(cap['sides'][s]['J'],t),I.unpack(cap['sides'][s]['P'],t)) for s in range(2)]
        return native,cap,dict(capture=b,features=fb)


def transform(fs,j,p,world,runtime,side):
    import rc_roma_position_factor_v1 as P
    import rc_roma_property_restore_v2 as R
    import rc_m_context_binding_v1 as C
    import run_rc_m_structure_binding_isolation_v2 as I
    t=runtime.t
    if world=='NATIVE':return fs,j,p,dict(kind='native')
    family,arm=world.split('/',1)
    if family=='phase':
        function=R.transform if arm in R.ARMS else P.transform
        pp,audit=function(p,runtime.omega,runtime.scale,arm,runtime.p['radius'][side],side)
        return fs,j,pp,audit
    if family=='context':
        aa,jj,pp=C.transform(fs,j,p,arm,t);shift,bits=C.recipe(arm)
        # Reversible permutation, finite precision exact; validates preservation
        # without treating grid equivariance as guaranteed by the frozen DPT.
        for x,y,bit in zip([*fs,j,p],[*aa,jj,pp],[bits[0],bits[0],bits[1],bits[2]]):
            inv=t.roll(y,shifts=tuple(-z for z in shift),dims=(1,2)) if bit=='1' else y
            assert t.equal(x,inv),'CONTEXT_INVERSE'
        return aa,jj,pp,dict(kind='grid_permutation',shift=shift,bits=bits,inverse_exact=True)
    aa,jj,pp=I.transform(fs,j,p,arm,t)
    return aa,jj,pp,I.checks(fs,j,p,aa,jj,pp,arm,t)


def pair(w,p,pb,runtime,deadline,pilot=False):
    import rc_m_context_binding_v1 as C
    import run_rc_m_phase_newgroups_cpu_v1 as G
    from rc_roma_shared_native_cache_v1 import ExactPool
    t=runtime.t;folder=OUT/'pairs'/f"{w['index']:03d}"/f"{w['position']:03d}"
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            v=read(folder/'validation.json');assert v['protocol']==pb;checked(v['result']);return True
        if time.monotonic()>deadline-45:return False
        native,cap,sources=runtime.inputs(w,folder,pilot);geometry=G.geometries(w);pool=ExactPool();records=[]
        for world in p['worlds']:
            key=world.replace('/','__');receipt=folder/(key+'.json')
            if receipt.exists():
                d=read(receipt);assert d['protocol']==pb and d['sources']==sources;checked(d['payload']);records.append(d);continue
            if time.monotonic()>deadline-35:return False
            started=time.monotonic();old=None
            if w.get('old_isolation') and (world=='NATIVE' or world.startswith('isolation/')):
                oldarm=world.split('/',1)[-1];rp=Path(w['old_isolation'])/(oldarm+'.json')
                if rp.exists():
                    d=read(rp);z=t.load(checked(d['payload']),map_location='cpu',weights_only=True,mmap=True)
                    assert z['capture']==sources['capture'] and z['index']==w['index'] and z['position']==w['position']
                    old=(z,d,bind(rp))
            sides=[]
            with t.inference_mode():
                for s in range(2):
                    ff,j0,p0=native[s];fs=[x.float() for x in ff];j=j0.float();pp=p0.float()
                    aa,jj,pt,audit=transform(fs,j,pp,world,runtime,s)
                    if old is not None:conf=old[0]['sides'][s]['confidence']
                    else:conf=runtime.head([aa[0],aa[-1]+jj+pt],img_A=None,img_B=None)[...,2:].clone()
                    assert t.isfinite(conf).all()
                    weights=pool(conf[0,...,0].sigmoid(),geometry[s]);error=C.independent_pool(conf,geometry[s],weights)
                    control=None
                    if world=='NATIVE':
                        orig=runtime.head([ff[0],ff[-1]+j0+p0],img_A=None,img_B=None)[...,2:].clone()
                        ow=pool(orig[0,...,0].sigmoid(),geometry[s]);control=dict(weights=ow,confidence=orig,
                            max_confidence_difference=float((orig-conf).abs().max()),source_dtypes=[str(x.dtype) for x in [*ff,j0,p0]])
                        if pilot:
                            fresh=runtime.head([fs[0],fs[-1]+j+pp],img_A=None,img_B=None)[...,2:]
                            assert t.equal(conf,fresh),'PILOT_NATIVE_REPLAY'
                            repeated=runtime.head([fs[0],fs[-1]+j+pp],img_A=None,img_B=None)[...,2:]
                            assert t.equal(fresh,repeated),'PILOT_REPEAT'
                    audit['changed_P_sha256']=hashlib.sha256(pt.contiguous().view(t.uint8).numpy().tobytes()).hexdigest()
                    sides.append(dict(confidence=conf,weights=weights,invariants=audit,pooling_error=error,source_native=control))
            mass=math.sqrt(float(sides[0]['weights'].mean())*float(sides[1]['weights'].mean()))
            assert 0<mass<=1
            if old is not None:assert abs(mass-old[1]['M'])<2e-12,'LEGACY_M_REPLAY'
            if old is None:
                path=folder/(key+'.pt');save(path,dict(protocol=pb,world=world,index=w['index'],position=w['position'],
                    image_shas=w['image_shas'],sources=sources,M=mass,sides=sides),t);payload=bind(path)
            else:payload=old[1]['payload']
            result=dict(protocol=pb,world=world,M=mass,payload=payload,sources=sources,
                        seconds=time.monotonic()-started,reused_from=None if old is None else old[2],
                        invariants=[s['invariants'] for s in sides],pooling_error=max(s['pooling_error'] for s in sides))
            if world=='NATIVE':
                om=math.sqrt(float(sides[0]['source_native']['weights'].mean())*float(sides[1]['source_native']['weights'].mean()))
                result['precision_control']=dict(original_dtype_M=om,fp32_M=mass,prior_gpu_M=w['old_gpu_M'],
                    fp32_minus_original_logM=math.log(mass/om),original_minus_gpu_logM=math.log(om/w['old_gpu_M']))
            write(receipt,result);records.append(result)
            emit(stage='WORLD_SAVED',index=w['index'],position=w['position'],world=world,M=mass)
        data=dict(protocol=pb,index=w['index'],position=w['position'],physical_row=w['physical_row'],
                  sources=sources,worlds={r['world']:r for r in records})
        write(folder/'result.json',data)
        write(folder/'validation.json',dict(status='ALL593_UPSTREAM_PAIR_PASS',protocol=pb,result=bind(folder/'result.json'),
             worlds=len(records),max_pool_error=max(r['pooling_error'] for r in records)))
        return True


def run_pairs(t,shard,budget,pilot):
    p,pb=protocol();deadline=time.monotonic()+budget
    assert 0<=shard<SHARDS
    if not pilot:
        v=read(OUT/'pilot_validation.json');assert v['status']=='ALL593_UPSTREAM_PILOT_PASS' and v['protocol']==pb
    runtime=Runtime(p,t)
    # One old exact-FP32 capture, one additional BF16-source capture: inherited
    # numeric issue must be exercised before full automatic expansion.
    if pilot:
        old=read(checked(read(OLD_ISO/'protocol.json')['inputs']))['workers']
        work=[(old[k]['index'],old[k]['position']) for k in [0,63]]
        fresh=read(checked(p['query_inputs'][128]))
        work.append((128,next(w['position'] for w in fresh['pairs'] if w['old_capture'] is None)))
    else:work=[divmod(n,128) for n in range(shard,593*128,SHARDS)]
    current=None;query=None;completed=0
    for i,pos in work:
        if current!=i:query=read(checked(p['query_inputs'][i]));current=i
        if not pair(query['pairs'][pos],p,pb,runtime,deadline,pilot):return 75
        completed+=1
    if pilot:
        write(OUT/'pilot_validation.json',dict(status='ALL593_UPSTREAM_PILOT_PASS',protocol=pb,
            pairs=work,worlds_per_pair=60,native_repeat_exact=True,legacy_isolation_recount=True,
            tests=bind(OUT/'self_test.json')))
    else:write(OUT/'shards'/f'{shard:02d}.json',dict(status='UPSTREAM_SHARD_PASS',protocol=pb,shard=shard,pairs=completed))
    return 0


def bootstrap(rows):
    g=defaultdict(list)
    for group,x in rows:g[group].append(float(x))
    values=np.array([np.mean(g[k]) for k in sorted(g)])
    if len(values)==0:return dict(queries=0,groups=0,mean=None,group95=None)
    rng=np.random.default_rng(20261009);b=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(queries=sum(map(len,g.values())),groups=len(g),mean=float(values.mean()),group95=np.quantile(b,[.025,.975]).tolist())


def contrasts(y):
    out={};dirs=['X_PLUS','X_MINUS','Y_PLUS','Y_MINUS']
    get=lambda arm:y['NATIVE' if arm=='NATIVE' else 'phase/'+arm]
    G=np.mean([get('GLOBAL_'+d) for d in dirs]);L=np.mean([get('LOCAL_'+d) for d in dirs])
    AG=np.mean([get('AMP_PERMUTE__GLOBAL_'+d) for d in dirs]);AL=np.mean([get('AMP_PERMUTE__LOCAL_'+d) for d in dirs])
    out.update(phase_native_G_minus_L=G-L,phase_amp_permuted_G_minus_L=AG-AL,
               amplitude_restored_under_global=G-AG,amplitude_restored_under_local=L-AL,
               phase_amplitude_interaction=(AG-AL)-(G-L),whole_P_native_minus_zero=y['NATIVE']-get('ZERO'),
               amplitude_native_minus_flat=y['NATIVE']-get('AMP_FLAT'),amplitude_native_minus_permuted=y['NATIVE']-get('AMP_PERMUTE'))
    cs=[]
    for d in dirs:
        z=lambda bits:y['NATIVE' if bits=='000' else 'context/'+d+'__'+bits]
        cs.append((z('000')+z('111')-z('001')-z('110'))/2)
        for bits in ['001','010','100','011','101','110','111']:out['context_'+d+'_'+bits]=z(bits)-z('000')
    out['context_registration_interaction_mean4']=float(np.mean(cs))
    for name in ['binding_interaction','local_stats_preserved','common_basis','B_after_S','S_after_B','order']:
        vals=[]
        for sign in ['PLUS','MINUS']:
            z=lambda a:y['isolation/'+a+'_'+sign]
            n=y['NATIVE'];q=z('Q_P');c=z('Q_C');both=z('Q_ALL');s=z('S_P');bs=z('Q_AFTER_S');sb=z('S_AFTER_Q')
            vals.append(dict(binding_interaction=(n+both-q-c)/2,local_stats_preserved=s-n,common_basis=both-n,
                             B_after_S=(bs-s)-(q-n),S_after_B=(sb-q)-(s-n),order=sb-bs)[name])
        out['isolation_'+name]=float(np.mean(vals))
    return {k:float(v) for k,v in out.items()}


_summary_sources = {}
def query_summary(t,i,p,pb):
    import run_rc_m593_cached_replication_fix_v1 as E
    q=read(checked(p['query_inputs'][i]));row=q['original_row'];masses={a:[] for a in p['worlds']};seals=[];maxerr=0.
    for pos in range(128):
        folder=OUT/'pairs'/f'{i:03d}'/f'{pos:03d}';v=read(folder/'validation.json');z=read(checked(v['result']))
        assert v['protocol']==z['protocol']==pb and z['physical_row']==q['axis'][pos] and z['position']==pos
        seals.append(bind(folder/'validation.json'))
        for a in p['worlds']:
            r=z['worlds'][a];payload=t.load(checked(r['payload']),weights_only=True,mmap=True,map_location='cpu')
            rec=math.sqrt(float(np.mean(payload['sides'][0]['weights'].numpy()))*float(np.mean(payload['sides'][1]['weights'].numpy())))
            err=abs(rec-r['M']);maxerr=max(maxerr,err);assert err<2e-12
            masses[a].append(r['M'])
    if not _summary_sources:
        rp=read(checked(p['replication']))
        _summary_sources.update(cache=read(checked(rp['cache'])),authority=read(checked(rp['simple_authority'])))
    cache=_summary_sources['cache'];auth=_summary_sources['authority']
    src=auth['sources'][i]['payload'];ns=read(checked(src))['modes']['NATIVE']['scores']
    native_scores=[[z[k] for k in ('real_score','query_control_score','reference_control_score')] for z in ns]
    theta=np.array([float.fromhex(x) for x in cache['native_parameters'][q['fold']]['COST1']])
    stheta=np.array([float.fromhex(x) for x in cache['prior_product_parameters'][q['fold']]['COST1']])
    scores={a:E.scores(row,m,native_scores,theta,stheta)[0] for a,m in masses.items()}
    # Stored scalar formula is independently checked against the established
    # scalar-loop implementation for every complete 128-candidate world.
    scalar_error=0.;cs=row['challengers'];win=row['winner']
    for a,m in masses.items():
        _,x=E.scores(row,m,native_scores,theta,stheta)
        sym=lambda x,y:(x-y)/(abs(x)+abs(y)+1e-12)
        free=row['free_content']
        for k,c in enumerate(cs):
            scalar_error=max(scalar_error,abs(math.fsum(float(v)*float(w) for v,w in zip(x[k],theta[:-1]))+theta[-1]-scores[a]['NATIVE7']['logits128'][c]))
            sx=[x[k,0],sym(m[c]*free[c],m[win]*free[win]),sym(m[c],m[win]),sym(free[c],free[win]),0.,0.]
            scalar_error=max(scalar_error,abs(math.fsum(float(v)*float(w) for v,w in zip(sx,stheta[:-1]))+stheta[-1]-scores[a]['M_FREE']['logits128'][c]))
    assert scalar_error<2e-10
    target=q['target'];opponents={};gaps={}
    if target is not None:
        wrong=[j for j in range(128) if j!=target];free=row['free_content']
        opponents=dict(C_STRONGEST=max(wrong,key=lambda j:(free[j],-row['axis'][j])),
                       C_NEAREST=min(wrong,key=lambda j:(abs(free[j]-free[target]),row['axis'][j])),
                       M_STRONGEST=max(wrong,key=lambda j:(row['mass'][j],-row['axis'][j])))
        for opponent,pos in opponents.items():
            end={'logM':{a:math.log(m[target])-math.log(m[pos]) for a,m in masses.items()}}
            for model in ['NATIVE7','M_FREE']:end[model]={a:z[model]['logits128'][target]-z[model]['logits128'][pos] for a,z in scores.items()}
            gaps[opponent]={metric:dict(worlds=y,contrasts=contrasts(y)) for metric,y in end.items()}
    answer=dict(protocol=pb,index=i,query_id=q['query_id'],component=q['component'],fold=q['fold'],axis=q['axis'],target=target,
                source_queries=p['query_inputs'][i],pair_validations=seals,M=masses,scores=scores,fixed_opponents=opponents,gaps=gaps,
                mass_recount_error=maxerr,scalar_score_error=scalar_error,raw_correct=row['winner']==target)
    write(OUT/'queries'/f'{i:03d}.json',answer)
    return answer


def join(t,budget):
    p,pb=protocol();deadline=time.monotonic()+budget
    for s in range(SHARDS):
        v=read(OUT/'shards'/f'{s:02d}.json');assert v['status']=='UPSTREAM_SHARD_PASS' and v['protocol']==pb
    seals=[]
    for i in range(593):
        path=OUT/'queries'/f'{i:03d}.json'
        if path.exists():q=read(path);assert q['protocol']==pb
        else:
            if time.monotonic()>deadline-90:return 75
            q=query_summary(t,i,p,pb)
        seals.append(bind(path));emit(stage='QUERY_AUDITED',index=i)
    rows=[read(s['path']) for s in seals];frontgroups={r['component'] for r in rows[:128]}
    panels={'ALL593':rows,'FRONT128':rows[:128],'TAIL465':rows[128:],
            'TAIL_NEW_COMPONENTS':[r for r in rows[128:] if r['component'] not in frontgroups],
            'TAIL_SHARED_COMPONENTS':[r for r in rows[128:] if r['component'] in frontgroups]}
    summaries={}
    for name,rr in panels.items():
        present=[r for r in rr if r['target'] is not None];summaries[name]=dict(queries=len(rr),target_present=len(present),groups=len({r['component'] for r in rr}),contrasts={},decisions={})
        for opponent in ['C_STRONGEST','C_NEAREST','M_STRONGEST']:
            summaries[name]['contrasts'][opponent]={}
            for metric in ['logM','NATIVE7','M_FREE']:
                summaries[name]['contrasts'][opponent][metric]={k:bootstrap([(r['component'],r['gaps'][opponent][metric]['contrasts'][k]) for r in present]) for k in contrasts({a:0. for a in p['worlds']})}
        for model in ['NATIVE7','M_FREE']:
            dec={}
            for a in p['worlds']:
                raw=[r['raw_correct'] for r in rr];before=[r['scores']['NATIVE'][model]['prediction_position']==r['target'] for r in rr]
                after=[r['scores'][a][model]['prediction_position']==r['target'] for r in rr]
                dec[a]=dict(correct=sum(after),raw_correct=sum(raw),rescue=sum(not x and y for x,y in zip(raw,after)),
                            break_RAW=sum(x and not y for x,y in zip(raw,after)),rescue_retained=sum(not x and b and c for x,b,c in zip(raw,before,after)),
                            native_rescue=sum(not x and b for x,b in zip(raw,before)),break_recovered=sum(x and not b and c for x,b,c in zip(raw,before,after)))
            summaries[name]['decisions'][model]=dec
    result=dict(status='ALL593_UPSTREAM_PROPERTIES_AND_EXTERNAL_COMPLETE',protocol=pb,queries=593,target_present=570,
                pairs=593*128,worlds=60,panels=summaries,query_receipts=seals,POST_expanded=False,unique_cause=False)
    assert sum(r['target'] is not None for r in rows)==570
    write(OUT/'result.json',result);write(OUT/'validation.json',dict(status='ALL593_UPSTREAM_RECOUNT_PASS',result=bind(OUT/'result.json'),
         queries=593,pairs=593*128,candidate_worlds=593*128*60,max_M_error=max(r['mass_recount_error'] for r in rows),max_score_error=max(r['scalar_score_error'] for r in rows)))
    lines=['# H593 upstream properties: final scoped result','',
           '593 original queries, natural C128, frozen NATIVE7/M_FREE, original five-fold heads. CPU causal baselines are separate from historical GPU anchors. No new POST result or unique causal percentages.',
           '', '| Panel | Endpoint | Contrast | Mean | Exploratory component interval |','|---|---|---|---:|---|']
    for panel,s in summaries.items():
        for metric,cc in s['contrasts']['C_STRONGEST'].items():
            for c,v in cc.items():lines.append(f"| {panel} | {metric} | {c} | {v['mean']} | {v['group95']} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');return 0


def self_test(t):
    import rc_roma_property_restore_v2 as R
    import run_rc_m_structure_binding_isolation_v2 as I
    import rc_m_context_binding_v1 as C
    g=t.Generator().manual_seed(1009);fs=[t.randn(1,8,8,16,generator=g) for _ in range(2)]
    j=t.randn(1,8,8,16,generator=g);p=t.randn(1,8,8,16,generator=g)
    for a in I.ARMS:
        aa,jj,pp=I.transform(fs,j,p,a,t);I.checks(fs,j,p,aa,jj,pp,a,t)
    for a in C.ARMS:
        aa,jj,pp=C.transform(fs,j,p,a,t);shift,bits=C.recipe(a)
        for x,y,bit in zip([*fs,j,p],[*aa,jj,pp],[bits[0],bits[0],bits[1],bits[2]]):
            assert t.equal(t.roll(y,tuple(-z for z in shift),(1,2)) if bit=='1' else y,x)
    algebra=R.self_test();assert len(worlds())==60 and len(set(worlds()))==60
    assert all(v==0 for v in contrasts({a:1. for a in worlds()}).values())
    write(OUT/'self_test.json',dict(status='ALL593_UPSTREAM_ALGEBRA_PASS',phase=algebra,worlds=worlds(),
         isolation_original_invariants=True,context_inverse_exact=True,zero_effect_contrasts=True))
    return 0


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','self-test','pilot','worker','join'])
    a.add_argument('--shard',type=int,default=0);a.add_argument('--budget',type=float,default=430);o=a.parse_args();t=configure()
    if o.mode=='self-test':return self_test(t)
    if o.mode=='prepare':return prepare(t,o.budget)
    if o.mode=='join':return join(t,o.budget)
    return run_pairs(t,o.shard,o.budget,o.mode=='pilot')
if __name__=='__main__':sys.exit(main())
