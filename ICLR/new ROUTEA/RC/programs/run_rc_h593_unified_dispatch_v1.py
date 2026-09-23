#!/usr/bin/env python3
"""Bounded, provenance-checked transition to one GPU producer and CPU consumers."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import socket
import time

ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'programs'/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
M=module('unified_legacy_dispatch','run_rc_h593_m_priority_dispatch_v1.py');P=M.P
B=module('unified_dev_placement','run_rc_h593_dev_backfill_20260923_v1.py')
OUT=ROOT/'results/rc_h593_unified_acquisition_v1'
CACHE=ROOT/'cache/rc_h593_unified_acquisition_v1'
AUTH=ROOT/'registry/rc_h593_unified_dispatch_authority_v1_20260923.json'
GPU=ROOT/'slurm/rc_h593_unified_acquisition_v1.sbatch'
CPU=ROOT/'slurm/rc_h593_unified_export_v1.sbatch'
EXEC_AUTH=ROOT/'registry/rc_h593_unified_acquisition_authority_v1_20260923.json'
FAMILIES=('inside','visual','coordinate')
read,write,bind,checked,command=P.read,P.write,P.bind,P.checked,P.command

def status(name,**kw):
    write(OUT/'status.json',dict(status=name,time_utc=P.now(),host=socket.gethostname(),pid=os.getpid(),**kw),replace=True)

def selection(done,ready,active,exported,first):
    if first not in exported:
        return ('export' if first in ready else 'collect',[first]) if first not in active else ('collect',[])
    export=[i for i in sorted(ready) if i not in exported and i not in active]
    if export:return 'export',export[:46]
    return 'collect',[i for i in range(593) if i not in done and i not in ready and i not in active][:max(0,46-len(active))]

def prepare():
    assert not AUTH.exists()
    for b in read(EXEC_AUTH)['sources']:checked(b)
    assert selection(set(),set(),set(),set(),70)==('collect',[70])
    assert selection(set(),{70},set(),set(),70)==('export',[70])
    assert selection(set(),{70},{70},set(),70)[1]==[]
    k,rows=selection({0},{70},set(),{70},70)
    assert k=='collect' and len(rows)==46 and 0 not in rows and 70 not in rows
    done={f:{i for i in range(593) if P.complete(dict(family=f,index=i))} for f in FAMILIES}
    first=next(i for i in range(593) if all(i not in done[f] for f in FAMILIES))
    write(AUTH,dict(status='UNIFIED_DISPATCH_AUTHORIZED',sources=[bind(p) for p in (Path(__file__),Path(M.__file__),Path(P.__file__),Path(B.__file__),EXEC_AUTH,GPU,CPU)],
        first_complete_query=first,selection='First natural ordinal missing all three families, without outcomes',
        maximum_gpu_tasks=46,maximum_cpu_tasks=46,maximum_query_chunks=32,maximum_days=3,
        qualification='First8 exact registered fields, then full first-query export and original validators before broad release',
        science_unchanged=True,old_jobs=bind(OUT/'transition_hold.json'),pilot=bind(OUT/'pilot_submission.json')))
    write(OUT/'preflight.json',dict(status='UNIFIED_DEPENDENCY_AND_SELECTION_PASS',authority=bind(AUTH),first_query=first,existing={f:len(x) for f,x in done.items()}))

def submit(stage,indices,partition=None):
    assert stage in ('collect','export') and 0<len(indices)<=46
    launcher=GPU if stage=='collect' else CPU
    partition=partition or ('accelerated' if stage=='collect' else 'cpuonly')
    args=['sbatch','--parsable','--hold','--partition='+partition,'--array='+','.join(map(str,indices))+'%46',str(launcher)]
    if stage=='collect':args.append('collect')
    number=len(list((OUT/'waves').glob('*.json')));assert number<2048
    job=command(args).stdout.strip().split(';')[0];assert job.isdigit()
    try:
        spool=OUT/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
        command(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==launcher.read_bytes()
        info=P.jobinfo(job);assert info['Partition']==partition and info['CPUs/Task']=='8'
        assert ('gres/gpu=1' in info['ReqTRES']) if stage=='collect' else ('gres/gpu' not in info['ReqTRES'])
        assert info['TimeLimit']==('00:15:00' if stage=='collect' else '00:10:00')
        write(OUT/'waves'/f'{number:04d}.json',dict(job=job,stage=stage,indices=indices,authority=bind(AUTH),args=args,spool=bind(spool),job_info=info))
        command(['scontrol','release',job]);assert P.jobinfo(job).get('Reason')!='JobHeldUser'
    except BaseException:
        command(['scancel',job],check=False);raise
    print(dict(submitted=job,stage=stage,indices=indices),flush=True)

def retire_legacy():
    marker=OUT/'legacy_retired.json'
    if marker.exists():return
    old=read(OUT/'transition_hold.json');queue={r['job']:r for r in P.queue()};cancelled=[]
    for r in old['held']:
        if r['job'] not in queue:continue
        row=queue[r['job']];assert row['state']=='PENDING' and row['reason']=='JobHeldUser'
        assert row['name']==r['name']
        command(['scancel',r['job']]);cancelled.append(r['job'])
    write(marker,dict(time_utc=P.now(),cancelled=cancelled,preserved_outputs=True,authority=bind(AUTH)))

def placement_setup():
    old=B.OUT/'backfill_process.json';proc=read(old)
    assert proc['host']==socket.gethostname()
    path=Path(f"/proc/{proc['pid']}/cmdline")
    if path.exists() and path.read_bytes():
        argv=[x.decode() for x in path.read_bytes().split(b'\0') if x]
        assert argv==proc['command'] and path.stat().st_uid==os.getuid()
        os.kill(proc['pid'],signal.SIGTERM)
        for _ in range(50):
            if not path.exists() or not path.read_bytes():break
            time.sleep(.1)
        assert not path.exists() or not path.read_bytes()
    B.OUT=OUT/'placement';B.OUT.mkdir(parents=True,exist_ok=True)
    original=B.owned
    def owned():
        scope=original()
        for p in (OUT/'waves').glob('*.json'):
            w=read(p)
            if w['stage']=='collect':
                for i in w['indices']:scope[f"{w['job']}_{i}"]='unified'
        return scope
    def candidates(rows,scope,turn):
        dev=[r for r in rows if r['partition'].startswith('dev_')]
        free=max(0,min(4-len(dev),3-sum(r['partition']=='dev_accelerated' for r in dev)))
        order=['unified','fusion','inside','visual','coordinate']
        order=order[turn%5:]+order[:turn%5]
        pool=[r for r in rows if r['job'] in scope and r['name'] in B.NAMES[scope[r['job']]] and r['state']=='PENDING' and r['partition']=='accelerated'
              and r['reason'] not in ('JobHeldUser','JobHeldAdmin','Dependency','DependencyNeverSatisfied')]
        return sorted(pool,key=lambda r:(order.index(scope[r['job']]),int(r['job'].split('_')[-1])))[:free]
    B.owned=owned;B.candidates=candidates;B.NAMES['unified']={'h593_collect'}
    write(OUT/'placement_takeover.json',dict(stopped=proc,preserved_fusion_placement=True,authority=bind(AUTH)))

def cpu_placement():
    rows=P.queue();dev=[r for r in rows if r['partition'].startswith('dev_')]
    if len(dev)>=4 or any(r['partition']=='dev_cpuonly' for r in dev):return
    owned={f"{w['job']}_{i}" for p in (OUT/'waves').glob('*.json') if (w:=read(p))['stage']=='export' for i in w['indices']}
    for r in rows:
        if r['job'] not in owned or r['state']!='PENDING' or r['partition']!='cpuonly' or r['reason'] in ('JobHeldUser','Dependency','DependencyNeverSatisfied'):continue
        before=P.jobinfo(r['job']);assert before['JobName']=='h593_export' and 'gres/gpu' not in before['ReqTRES']
        command(['scontrol','update','JobId='+r['job'],'Partition=dev_cpuonly'])
        after=P.jobinfo(r['job']);assert after['Partition']=='dev_cpuonly'
        for k in B.FIELDS:assert before.get(k)==after.get(k)
        B.record(dict(event='CPU_EXPORT_DEV_VERIFIED',job=r['job'],before=before,after=after));return

def settle(rows):
    queue={r['job'] for r in rows};waves=[read(p) for p in sorted((OUT/'waves').glob('*.json'))]
    active=set();pending=[];attempts=Counter()
    for w in waves:
        for i in w['indices']:
            key=f"{w['job']}_{i}";marker=OUT/'retired'/f'{key}.json'
            if w['stage']=='collect':attempts[i]+=1
            if marker.exists():continue
            if key in queue:active.add(i)
            else:pending.append((w,i,key,marker))
    acc=M.accounting(sorted({w['job'] for w,i,key,marker in pending}))
    for w,i,key,marker in pending:
        row=acc.get(key)
        if not row or row['state'] in ('PENDING','RUNNING','COMPLETING'):active.add(i);continue
        assert row['state']=='COMPLETED' and row['exit']=='0:0',('UNIFIED_WORKER_FAILED',key,row)
        if w['stage']=='collect':
            receipt=CACHE/f'query{i:03d}'/'chunks'/f"{row['raw']}.json"
            value=read(receipt);assert value['status']=='UNIFIED_NORMAL_EXIT' and value['index']==i
        else:
            receipt=CACHE/f'query{i:03d}'/'export_validation.json'
            value=read(receipt);assert value['status']=='UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS'
            for b in value['validations'].values():checked(b)
        write(marker,dict(accounting=row,receipt=bind(receipt),stage=w['stage'],index=i))
    return active,attempts

def joins(done):
    M.OUT=OUT;M.AUTH=AUTH;M.joins(done)
    dest=OUT/'coordinate_cpu_join.json'
    if len(done['coordinate'])==593 and not dest.exists():
        launcher=ROOT/'slurm/rc_h593_coordinate_cpu_chain_v1.sbatch'
        job=command(['sbatch','--parsable',str(launcher),'control']).stdout.strip().split(';')[0]
        assert job.isdigit();write(dest,dict(job=job,launcher=bind(launcher),all593_validated=True))

def tick(a):
    pilot=read(OUT/'pilot_submission.json')['job'];q=CACHE/'qualification.json'
    row=M.accounting([pilot]).get(pilot)
    if not q.exists():
        assert not row or row['state'] in ('PENDING','RUNNING','COMPLETING'),('QUALIFICATION_FAILED',row)
        status('WAITING_UNIFIED_FIRST8_QUALIFICATION',pilot=pilot);return False
    qual=read(q);assert qual['status']=='UNIFIED_FIRST8_REGISTERED_FIELDS_BIT_EXACT' and qual['authority']==bind(EXEC_AUTH)
    if not row or row['state'] in ('PENDING','RUNNING','COMPLETING'):
        status('WAITING_QUALIFICATION_JOB_EXIT',pilot=pilot);return False
    assert row['state']=='COMPLETED' and row['exit']=='0:0'
    rows=P.queue();active,attempts=settle(rows)
    for r in read(OUT/'transition_hold.json')['retained_running']:
        if any(x['job']==r['job'] for x in rows):active.add(int(r['job'].split('_')[-1]))
    done={f:{i for i in range(593) if P.complete(dict(family=f,index=i))} for f in FAMILIES}
    ready={int(p.parent.name[5:]) for p in CACHE.glob('query[0-9][0-9][0-9]/ready.json')}
    exported={int(p.parent.name[5:]) for p in CACHE.glob('query[0-9][0-9][0-9]/export_validation.json')}
    first=a['first_complete_query']
    if first in exported and first not in active:
        assert all(first in done[f] for f in FAMILIES)
        for f in FAMILIES:P.complete(dict(family=f,index=first),True)
        retire_legacy()
    else:exported.discard(first)
    stage,indices=selection(set.intersection(*done.values()),ready,active,exported,first)
    for i in indices:assert attempts[i]<32 or stage=='export',('MAX_QUERY_CHUNKS',i)
    if indices:submit(stage,indices)
    if first in exported:joins(done)
    complete=all(len(v)==593 for v in done.values())
    status('UNIFIED_ACQUISITION_AND_CPU_EXPORT_COMPLETE' if complete else 'UNIFIED_DISPATCH_ACTIVE',pilot=pilot,first_query=first,
           first_export_validated=first in exported,counts={f:len(v) for f,v in done.items()},active=len(active),new_stage=stage,new_indices=indices,
           complete_gpu_capsules=len(ready),cpu_exports=len(exported),legacy_retired=(OUT/'legacy_retired.json').exists())
    return complete

def watch():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a=read(AUTH)
    for b in a['sources']:checked(b)
    placement_setup();deadline=time.monotonic()+a['maximum_days']*86400;turn=0
    try:
        while time.monotonic()<deadline:
            done=tick(a);B.tick(turn);cpu_placement()
            if done:return
            turn+=1;time.sleep(20)
        status('UNIFIED_DISPATCH_72H_LIMIT',running_jobs_unchanged=True)
    except BaseException as e:
        status('UNIFIED_STOPPED_REQUIRES_REPAIR',error=repr(e),no_failed_jobs_retried=True);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','watch'));arg=ap.parse_args()
    prepare() if arg.stage=='prepare' else watch()
