#!/usr/bin/env python3
"""Shared native-input prerequisite and gated continuation of existing analyses."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_roma_shared_native_cache_v1 as S
from rc_roma_exact_feature_cache_v2 import equal_tree
OUT=ROOT/'cache/rc_h593_shared_native_v1'
AUTH=ROOT/'registry/rc_h593_shared_native_cache_authority_v2_20260922.json'
GPU=ROOT/'slurm/rc_h593_shared_native_cache_v2.sbatch'
CPU=ROOT/'slurm/rc_h593_shared_native_control_v2.sbatch'
read,write,save,bind,checked=C.read,C.write,C.save,C.bind,C.checked
FAMILIES={
 'inside':dict(array='5157691',callback='5157692',folder=ROOT/'results/rc_h593_m_inside_v1',script=ROOT/'programs/run_rc_h593_m_inside_v1.py'),
 'visual':dict(array='5157672',callback='5157673',folder=ROOT/'results/rc_h593_m_visual_origin_v1',script=ROOT/'programs/run_rc_h593_m_visual_origin_v1.py'),
 'coordinate':dict(array='5157625',callback='5157626',folder=C.OUT,script=ROOT/'programs/run_rc_h593_roma_coordinate_cache_v2.py')}


def prepare():
    assert not AUTH.exists()
    preflight=ROOT/'reports/h593_shared_cache_preflight_v2_20260922.json'
    assert read(preflight)['status']=='SHARED_CACHE_CPU_CHECKS_PASS'
    source=ROOT/'results/rc_h593_feature_fusion_cache_v1'
    validation=read(source/'catalog_validation.json');cat=read(checked(validation['catalog']))
    images={}
    for entry in cat['images'].values():
        item=entry['item'];images.setdefault(item['image_sha256'],dict(path=item['path'],sha256=item['image_sha256']))
    sources=[Path(__file__),Path(S.__file__),Path(C.__file__),Path(C.M.__file__),
      ROOT/'programs/rc_roma_exact_feature_cache_v2.py',ROOT/'plan/RC_H593_SHARED_NATIVE_CACHE_V2_20260922.md',GPU,CPU,
      preflight,*(x['script'] for x in FAMILIES.values())]
    write(OUT/'catalog.json',dict(images=[images[k] for k in sorted(images)],source=validation['catalog'],
        raw_unique_images=len(images),pooled_entries=len(cat['images'])))
    write(AUTH,dict(status='SHARED_NATIVE_PREREQUISITE_AUTHORIZED',sources=[bind(p) for p in sources],
        catalog=bind(OUT/'catalog.json'),profile=bind(C.PROFILE),workers=bind(C.WORKERS),
        parent_authorities=[bind(C.AUTH),bind(ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json'),
          bind(ROOT/'registry/rc_h593_m_visual_origin_authority_v1_20260922.json')],
        data_first=True,chunk_seconds=600,max_chunks=32,cache_dtype='Actual native floating dtype; no conversion',supersedes=bind(ROOT/'registry/rc_h593_shared_native_cache_authority_v1_20260922.json'),
        consumer_max_parallel=46,scope='Exact shared coarse descriptors and integer pooling boxes; no scientific protocol change',
        user_authorization='User explicitly prioritized completing reusable prerequisite data before downstream work'))
    print(dict(status='PREPARED',images=len(images),estimated_coarse_bytes_range=[len(images)*10240000,len(images)*20480000]),flush=True)


def authority():
    a=read(AUTH)
    for b in [*a['sources'],a['catalog'],a['profile'],a['workers'],*a['parent_authorities']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID')
    return a


def image_lr(model,core,item):
    assert bind(item['path'])['sha256']==item['sha256']
    image=core.oriented(Path(item['path']))
    t=model._load_image(image)
    lr=F.interpolate(t,size=(800,800),mode='bicubic',align_corners=False,antialias=True)
    return image,lr


def produce_image(model,core,item):
    receipt=OUT/'images'/f"{item['sha256']}.json"
    if receipt.exists():return read(receipt)
    _,lr=image_lr(model,core,item);key=S.tensor_key(lr);path=OUT/'coarse'/f'{key}.pt'
    if not path.exists():
        with torch.inference_mode():values=model.f(lr)
        specs=[dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for v in values]
        write(OUT/'native_coarse_specification.json',dict(specs=specs,authority=bind(AUTH)))
        assert len(values)==2 and all(v.shape==(1,50,50,1024) and v.dtype in (torch.float16,torch.bfloat16,torch.float32) for v in values),specs
        save(path,dict(input_key=key,features=[v.detach().cpu().clone() for v in values],profile=bind(C.PROFILE)))
        reread=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert equal_tree(values,[v.to(lr.device) for v in reread['features']]),'COARSE_DISK_ROUNDTRIP'
    value=dict(status='NATIVE_COARSE_IMAGE_PASS',image=item,input_key=key,payload=bind(path),authority=bind(AUTH))
    write(receipt,value)
    return value


@torch.inference_mode()
def qualify(model,core):
    if (OUT/'qualification.json').exists():
        q=read(OUT/'qualification.json');assert q['status']=='SHARED_NATIVE_FULL128_PASS' and q['authority']==bind(AUTH)
        return
    w=read(C.WORKERS)['records'][0];q,old,refs=C.load_input(w)
    qi=dict(path=q['query_source_path'],sha256=q['query_source_sha256'])
    if not (OUT/'fine_storage_probe.json').exists():
        _,lr=image_lr(model,core,qi)
        original=model._load_image(core.oriented(Path(qi['path'])))
        hr=F.interpolate(original,size=(1280,1280),mode='bicubic',align_corners=False,antialias=True)
        fine_rows=[]
        for name,x in [('LR',lr),('HR',hr)]:
            torch.cuda.synchronize();start=time.monotonic();v=model.refiner_features(x);torch.cuda.synchronize();compute=time.monotonic()-start
            path=OUT/'probes'/f'fine_{name}.pt';start=time.monotonic()
            if not path.exists():save(path,{k:t.cpu().clone() for k,t in v.items()})
            stored=time.monotonic()-start;start=time.monotonic()
            loaded=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            restored={k:t.to(x.device) for k,t in loaded.items()};torch.cuda.synchronize();read_time=time.monotonic()-start
            assert equal_tree(v,restored)
            fine_rows.append(dict(stage=name,compute_seconds=compute,write_seconds=stored,warm_read_to_gpu_seconds=read_time,
                bytes=path.stat().st_size,shapes={str(k):list(t.shape) for k,t in v.items()}))
            del v,loaded,restored
        write(OUT/'fine_storage_probe.json',dict(rows=fine_rows,scope='One natural image, warm filesystem read; no full fine cache commissioned'))
        del lr,hr,original
    items=[qi]+[dict(path=refs[p]['source_path'],sha256=refs[p]['source_image_sha256']) for p in q['candidate_physical_rows']]
    bindings={}
    for item in items:
        r=produce_image(model,core,item);bindings[r['input_key']]=r['payload']
    reader=S.CoarseReader(model,bindings,checked);pool=S.ExactPool()
    query_image=core.oriented(Path(qi['path']))
    profile=read(C.PROFILE)
    qgeom,_=C.M.geometry(Path(qi['path']),qi['sha256'],'shared-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
    timings=dict(fresh=0.,cached=0.,old_pool=0.,cached_bounds_pool=0.);checks=0
    try:
        for pos,physical in enumerate(q['candidate_physical_rows']):
            r=refs[physical];rp=Path(r['source_path']);ri=core.oriented(rp)
            rg,_=C.M.geometry(rp,r['source_image_sha256'],'shared-ref',r['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])
            reader.enabled=False;torch.cuda.synchronize();start=time.monotonic();fresh=model.match(query_image,ri);torch.cuda.synchronize();timings['fresh']+=time.monotonic()-start
            reader.enabled=True;torch.cuda.synchronize();start=time.monotonic();cached=model.match(query_image,ri);torch.cuda.synchronize();timings['cached']+=time.monotonic()-start
            assert equal_tree(fresh,cached),('DENSE_FORWARD_PARITY',pos)
            for direction,g,key in [('AB',qgeom,'query_visibility'),('BA',rg,'reference_visibility')]:
                overlap=cached['overlap_'+direction][0,...,0].cpu()
                start=time.monotonic();expected=core.cell_means(overlap,g);timings['old_pool']+=time.monotonic()-start
                pool(overlap,g)  # Build boxes before measuring repeated use.
                start=time.monotonic();actual=pool(overlap,g);timings['cached_bounds_pool']+=time.monotonic()-start
                assert equal_tree(actual,expected) and equal_tree(actual,old['candidates'][pos][key]);checks+=1
            if pos==0:
                for size in (200,400,800,320,640,1280):
                    field=torch.linspace(0,1,size*size).reshape(size,size)
                    for geom in (qgeom,rg):assert equal_tree(pool(field,geom),core.cell_means(field,geom));checks+=1
            if pos%16==15:print(dict(event='SHARED_NATIVE_PARITY',pairs=pos+1,seconds=timings),flush=True)
            del fresh,cached
    finally:reader.close()
    # Do not expand a cache whose observed complete read path is slower.
    result=dict(status='SHARED_NATIVE_FULL128_PASS' if timings['cached']<timings['fresh'] else 'SHARED_NATIVE_NO_SPEED_BENEFIT',
        authority=bind(AUTH),pairs=128,pooling_checks=checks,timings=timings,reader=dict(reader.stats),
        scope='First natural full-C128 qualification; fresh first, cached second; warm filesystem; no global speed guarantee',
        full_dense_bit_exact=True,old_token_weights_bit_exact=True,labels_read=0)
    write(OUT/'qualification.json',result);assert result['status']=='SHARED_NATIVE_FULL128_PASS',result


def build(a):
    if (OUT/'ready.json').exists():return
    started=time.monotonic();profile=read(C.PROFILE);C.check_profile(profile)
    core=C.M.legacy_core(profile);model=C.M.gpu_model(profile)
    qualify(model,core)
    images=read(OUT/'catalog.json')['images'];done=0
    for item in images:
        if not (OUT/'images'/f"{item['sha256']}.json").exists():
            if time.monotonic()-started>a['chunk_seconds']:
                write(OUT/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",dict(status='SHARED_CACHE_NORMAL_PARTIAL',done=done,total=len(images)));return
            produce_image(model,core,item)
        done+=1
        if done%128==0:print(dict(event='SHARED_NATIVE_IMAGES',done=done,total=len(images),seconds=time.monotonic()-started),flush=True)
    write(OUT/'all_images_produced.json',dict(authority=bind(AUTH),images=len(images)))


def seal():
    images=read(OUT/'catalog.json')['images'];bindings={};receipts=[]
    for item in images:
        p=OUT/'images'/f"{item['sha256']}.json";r=read(p)
        assert r['status']=='NATIVE_COARSE_IMAGE_PASS' and r['image']['sha256']==item['sha256'] and r['authority']==bind(AUTH)
        if r['input_key'] not in bindings:checked(r['payload']);bindings[r['input_key']]=r['payload']
        else:assert bindings[r['input_key']]==r['payload']
        receipts.append(bind(p))
    write(OUT/'ready.json',dict(status='SHARED_NATIVE_ALL_IMAGES_PASS',authority=bind(AUTH),qualification=bind(OUT/'qualification.json'),
        images=len(images),unique_inputs=len(bindings),features=bindings,image_receipts=receipts))


def consumer(family,index):
    ready=read(OUT/'ready.json');assert ready['status']=='SHARED_NATIVE_ALL_IMAGES_PASS' and ready['authority']==bind(AUTH)
    qualification=read(checked(ready['qualification']))
    assert qualification['status']=='SHARED_NATIVE_FULL128_PASS' and qualification['authority']==bind(AUTH)
    readers=[];pools=[]
    factory=C.M.gpu_model;legacy=C.M.legacy_core
    def model_factory(profile):
        model=factory(profile);readers.append(S.CoarseReader(model,ready['features'],checked));return model
    def core_factory(profile):
        core=legacy(profile);pool=S.ExactPool();core.cell_means=pool;pools.append(pool);return core
    C.M.gpu_model=model_factory;C.M.legacy_core=core_factory
    already=(FAMILIES[family]['folder']/f'query{index:03d}/validation.json').exists()
    if family=='coordinate':
        import run_rc_h593_roma_coordinate_cache_v2 as D
        D.worker(D.checked_authority(),index)
    elif family=='inside':
        import run_rc_h593_m_inside_v1 as D
        D.configure(index);a,ws=D.U.guard('worker',index);D.U.worker(a,ws[index],index)
    else:
        import run_rc_h593_m_visual_origin_v1 as D
        a,ws=D.guard('worker',index);D.worker(a,ws[index],index)
    # Existing protocols keep their immutable partial files and original validators.
    write(OUT/'consumers'/family/f"{os.environ['SLURM_JOB_ID']}_{index}.json",dict(authority=bind(AUTH),ready=bind(OUT/'ready.json'),
        index=index,previously_complete=already,readers=[dict(r.stats) for r in readers],pools=[dict(p.stats) for p in pools],
        complete=(FAMILIES[family]['folder']/f'query{index:03d}/validation.json').exists()))


def cmd(args):return subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True,timeout=60).stdout


def submit(stage,indices=None,family=None,partition=None):
    args=['sbatch','--parsable','--hold']
    if indices is not None:args+=['--array='+','.join(map(str,indices))+'%46']
    if partition:args+=['--partition='+partition]
    args += [str(GPU),stage]+([family] if family else [])
    job=cmd(args).strip().split(';')[0];assert job.isdigit()
    callback=None
    try:
        spool=OUT/'dispatch'/f'spool_{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
        cmd(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==GPU.read_bytes()
        callback=cmd(['sbatch','--parsable','--dependency=afterany:'+job,str(CPU),'control',family or 'build',job]).strip().split(';')[0]
        write(OUT/'dispatch'/f'{job}.json',dict(job=job,callback=callback,stage=stage,family=family,indices=indices,authority=bind(AUTH),spool=bind(spool)))
        cmd(['scontrol','release',job]);print(dict(submitted=job,callback=callback,family=family,indices=indices),flush=True)
    except BaseException:
        cmd(['scancel',job])
        if callback:cmd(['scancel',callback])
        raise


def control(a,family,previous):
    if previous:
        lines=cmd(['sacct','-X','-n','-P','-j',previous,'--format=JobID,State,ExitCode']).splitlines()
        assert lines and all(x.split('|')[1:3]==['COMPLETED','0:0'] for x in lines if x.strip()),lines
    if family=='build':
        if not (OUT/'all_images_produced.json').exists():
            assert len(list((OUT/'chunks').glob('*.json')))<a['max_chunks']
            return submit('build',partition='dev_accelerated')
        if not (OUT/'ready.json').exists():seal()
        # The existing running shards were allowed to finish. Only held pending
        # jobs from the explicitly deferred families are replaced.
        rows=cmd(['squeue','-r','-u','ap7811','-h','-o','%i|%T|%r']).splitlines()
        owned={x['array'] for x in FAMILIES.values()}|{x['callback'] for x in FAMILIES.values()}
        replaced=[]
        for line in rows:
            job,state,reason=line.split('|',2)
            if job.split('_')[0] not in owned:continue
            assert state=='PENDING' and reason=='JobHeldUser',(job,state,reason)
            cmd(['scancel',job]);replaced.append(job)
        write(OUT/'cutover.json',dict(authority=bind(AUTH),ready=bind(OUT/'ready.json'),old_pending_replaced=replaced,
            retained_original_results=True))
        for name in FAMILIES:control(a,name,None)
        return
    data=FAMILIES[family];valid=set()
    for i in range(593):
        p=data['folder']/f'query{i:03d}/validation.json'
        if p.exists():
            v=read(p);checked(v['payload']);assert v['status'] in ('M_VISUAL_ORIGIN_QUERY_PASS','ROMA_COORDINATE_QUERY_PASS');valid.add(i)
            expected=C.AUTH if family=='coordinate' else ROOT/f'registry/rc_h593_m_{"inside" if family=="inside" else "visual_origin"}_authority_v1_20260922.json'
            assert v['authority']==bind(expected)
            if family=='inside':
                extra=read(p.with_name('inside_validation.json'));checked(extra['payload'])
                assert extra['status']=='M_INSIDE_PATHS_PASS' and extra['authority']==v['authority']
    attempts={i:0 for i in range(593)}
    for p in (OUT/'dispatch').glob('*.json'):
        wave=read(p)
        if wave.get('family')==family:
            for i in wave['indices']:attempts[i]+=1
    todo=[i for i in range(593) if i not in valid][:3 if family=='inside' else 46]
    assert all(attempts[i]<32 for i in todo),'CONSUMER_CHUNK_LIMIT'
    if todo:return submit('consumer',todo,family,partition='dev_accelerated' if family=='inside' else 'accelerated')
    # Original analysis/join programs retain the old science and label gates.
    if family=='coordinate':
        job=cmd(['sbatch','--parsable',str(ROOT/'slurm/rc_h593_roma_coordinate_cache_dispatch_v1.sbatch'),'run']).strip()
    else:
        launcher=ROOT/('slurm/rc_h593_m_inside_control_v1.sbatch' if family=='inside' else 'slurm/rc_h593_m_visual_origin_control_v1.sbatch')
        job=cmd(['sbatch','--parsable',str(launcher),'join']).strip()
    write(OUT/f'{family}_join_submitted.json',dict(job=job,authority=bind(AUTH)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','build','consumer','control'));p.add_argument('family',nargs='?',default='build');p.add_argument('previous',nargs='?');p.add_argument('--index',type=int)
    args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=authority()
        if args.stage=='build':build(a)
        elif args.stage=='consumer':consumer(args.family,args.index)
        else:control(a,args.family,args.previous)
