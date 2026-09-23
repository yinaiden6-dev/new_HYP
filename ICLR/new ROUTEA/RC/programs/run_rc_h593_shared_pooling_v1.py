#!/usr/bin/env python3
"""Apply only byte-qualified pooling-bound reuse to the unchanged science runners."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_roma_shared_native_cache_v1 as S
OUT=ROOT/'cache/rc_h593_shared_pooling_v1'
AUTH=ROOT/'registry/rc_h593_shared_pooling_cache_authority_v1_20260922.json'
GPU=ROOT/'slurm/rc_h593_shared_pooling_v1.sbatch'
CPU=ROOT/'slurm/rc_h593_shared_pooling_control_v1.sbatch'
QUAL=ROOT/'cache/rc_h593_shared_native_v1/qualification.json'
read,write,bind,checked=C.read,C.write,C.bind,C.checked
FAMILIES={
 'inside':dict(array='5157691',callback='5157692',folder=ROOT/'results/rc_h593_m_inside_v1',script=ROOT/'programs/run_rc_h593_m_inside_v1.py'),
 'visual':dict(array='5157672',callback='5157673',folder=ROOT/'results/rc_h593_m_visual_origin_v1',script=ROOT/'programs/run_rc_h593_m_visual_origin_v1.py'),
 'coordinate':dict(array='5157625',callback='5157626',folder=C.OUT,script=ROOT/'programs/run_rc_h593_roma_coordinate_cache_v2.py')}


def prepare():
    assert not AUTH.exists()
    q=read(QUAL)
    assert q['status']=='SHARED_NATIVE_NO_SPEED_BENEFIT' and q['pairs']==128 and q['pooling_checks']==268
    assert q['full_dense_bit_exact'] and q['old_token_weights_bit_exact']
    assert q['timings']['cached_bounds_pool']<q['timings']['old_pool']
    prior=read(checked(q['authority']))
    for b in prior['sources']:checked(b)
    preflight=ROOT/'reports/h593_shared_pooling_preflight_20260922.json'
    assert read(preflight)['status']=='EXACT_POOLING_LAUNCHERS_PASS'
    sources=[Path(__file__),Path(S.__file__),Path(C.__file__),Path(C.M.__file__),
        GPU,CPU,ROOT/'plan/RC_H593_SHARED_POOLING_V1_20260922.md',preflight,*[x['script'] for x in FAMILIES.values()]]
    write(AUTH,dict(status='EXACT_POOLING_EXECUTION_AUTHORIZED',sources=[bind(p) for p in sources],
        profile=bind(C.PROFILE),workers=bind(C.WORKERS),qualification=bind(QUAL),
        parent_authorities=prior['parent_authorities'],
        scope='Cache integer boxes only; original FP64 slice means, descriptors, pairing and scientific outputs unchanged',
        full_descriptor_expansion=False,consumer_max_parallel=46,max_attempts=32,
        user_authorization='Complete reusable prerequisite first; apply measured exact acceleration and resume existing experiments'))
    write(OUT/'ready.json',dict(status='EXACT_POOLING_READY',authority=bind(AUTH),qualification=bind(QUAL),
        pooled_seconds=q['timings']['cached_bounds_pool'],original_seconds=q['timings']['old_pool'],
        full_descriptor_expansion=False,existing_tokens_and_features_reused=True))
    print(dict(status='EXACT_POOLING_READY',authority=bind(AUTH)),flush=True)


def authority():
    a=read(AUTH)
    for b in [*a['sources'],a['profile'],a['workers'],a['qualification'],*a['parent_authorities']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID')
    return a


def consumer(family,index):
    ready=read(OUT/'ready.json');assert ready['status']=='EXACT_POOLING_READY' and ready['authority']==bind(AUTH)
    qualification=read(checked(ready['qualification']))
    assert qualification['full_dense_bit_exact'] and qualification['old_token_weights_bit_exact']
    pools=[];legacy=C.M.legacy_core
    def core_factory(profile):
        core=legacy(profile);pool=S.ExactPool();core.cell_means=pool;pools.append(pool);return core
    C.M.legacy_core=core_factory
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
        index=index,previously_complete=already,native_features_unchanged=True,pools=[dict(p.stats) for p in pools],
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
        callback=cmd(['sbatch','--parsable','--dependency=afterany:'+job,str(CPU),'control',family or 'start',job]).strip().split(';')[0]
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
    if family=='start':
        assert read(OUT/'ready.json')['authority']==bind(AUTH)
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
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','consumer','control'));p.add_argument('family',nargs='?',default='start');p.add_argument('previous',nargs='?');p.add_argument('--index',type=int)
    args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=authority()
        if args.stage=='consumer':consumer(args.family,args.index)
        else:control(a,args.family,args.previous)
