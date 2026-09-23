#!/usr/bin/env python3
"""Scheduler-only four-GPU packing of unchanged H593 scientific programs."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parents[2]
OUT=ROOT/'results/rc_h593_packed_gpu_v1'
AUTH=ROOT/'registry/rc_h593_packed_gpu_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_h593_packed_gpu_v1.sbatch'
PLAN=ROOT/'plan/RC_H593_PACKED_GPU_EXECUTION_V1_20260923.md'
POOL=ROOT/'cache/rc_h593_shared_pooling_v1'
FD=ROOT/'results/rc_h593_feature_fusion_dispatch_v3'
READOUT=ROOT/'results/rc_h593_m_path_readout_v1'
PY=WS/'.venv-romav2/bin/python'
FAMILIES=('inside','visual','coordinate','fusion')
FOLDERS={k:ROOT/'results'/v for k,v in dict(inside='rc_h593_m_inside_v1',visual='rc_h593_m_visual_origin_v1',coordinate='rc_h593_roma_coordinate_precision_v2',fusion='rc_h593_feature_fusion_train_v2').items()}
AUTHORITIES={k:ROOT/'registry'/v for k,v in dict(inside='rc_h593_m_inside_authority_v1_20260922.json',visual='rc_h593_m_visual_origin_authority_v1_20260922.json',coordinate='rc_h593_roma_coordinate_precision_authority_v2_20260922.json',fusion='rc_h593_feature_fusion_train_authority_v2_20260922.json').items()}
STATUSES=dict(inside='M_VISUAL_ORIGIN_QUERY_PASS',visual='M_VISUAL_ORIGIN_QUERY_PASS',coordinate='ROMA_COORDINATE_QUERY_PASS',fusion='FUSION_FOLD_NUMPY_READOUT_PASS')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return dict(path=str(p),sha256=h.hexdigest())
def checked(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def write(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists() and not replace:assert p.read_text()==text,p;return
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');tmp.write_text(text)
    if replace:os.replace(tmp,p)
    else:os.link(tmp,p);tmp.unlink()
def command(args,check=True):
    p=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,capture_output=True,timeout=45)
    if check and p.returncode:raise RuntimeError(str(args)+': '+p.stderr)
    return p
def fields(text):return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)',text))
def jobinfo(job):return fields(command(['scontrol','show','job','-o',job]).stdout)
def queue():
    text=command(['squeue','-r','-u','ap7811','-h','-o','%i|%j|%P|%T|%r']).stdout
    return [dict(zip(('job','name','partition','state','reason'),s.split('|',4))) for s in text.splitlines() if s]
def status(state,**kw):write(OUT/'status.json',dict(time_utc=now(),pid=os.getpid(),host=socket.gethostname(),status=state,**kw),replace=True)
def folder(t):return FOLDERS[t['family']]/(('fit' if t['family']=='fusion' else 'query')+f"{t['index']:03d}")
def taskkey(t):return t['family']+f"_{t['index']:03d}"
def complete(t,hash_payload=False):
    p=folder(t)/'validation.json'
    if not p.exists():return False
    v=read(p);assert v['status']==STATUSES[t['family']] and v['authority']==bind(AUTHORITIES[t['family']]),p
    assert Path(v['payload']['path']).is_file()
    if hash_payload:checked(v['payload'])
    if t['family']=='inside':
        x=read(p.with_name('inside_validation.json'));assert x['status']=='M_INSIDE_PATHS_PASS' and x['authority']==v['authority']
        if hash_payload:checked(x['payload'])
    return True
def counts():return {k:sum(complete(dict(family=k,index=i)) for i in range(200 if k=='fusion' else 593)) for k in FAMILIES}
def child_command(t):
    if t['family']=='fusion':return [str(PY),str(ROOT/'programs/run_rc_h593_feature_fusion_train_v2.py'),'fit','--index',str(t['index'])]
    return [str(PY),str(ROOT/'programs/run_rc_h593_shared_pooling_v1.py'),'consumer',t['family'],'--index',str(t['index'])]
def guard():
    a=read(AUTH);assert a['status']=='PACKED_GPU_EXECUTION_AUTHORIZED'
    for b in a['sources']:checked(b)
    return a

def prepare():
    assert not AUTH.exists() and PY.is_file()
    ps=read(POOL/'ready.json');assert ps['status']=='EXACT_POOLING_READY';checked(ps['authority'])
    sources=[Path(__file__),LAUNCH,PLAN,Path(ps['authority']['path']),*AUTHORITIES.values()]
    for b in read(ps['authority']['path'])['sources']:checked(b);sources.append(Path(b['path']))
    fa=read(AUTHORITIES['fusion'])
    for b in fa['sources'] if 'sources' in fa else fa['code_sources'].values():checked(b);sources.append(Path(b['path']))
    assert len(fa['configs'])==200
    write(AUTH,dict(status='PACKED_GPU_EXECUTION_AUTHORIZED',sources=[bind(p) for p in dict.fromkeys(sources)],families=list(FAMILIES),parent_pool=ps['authority'],pilot_tasks=4,gpus=4,cpus=32,memory_gb=256,allocation_minutes=55,pilot_minutes=30,maximum_hours=72,maximum_allocations=128,scope='Execution packing only; fresh unchanged child processes, original outputs/validators/folds/candidates; no heldout label reads by scheduler'))
    write(OUT/'preflight.json',dict(status='PACKED_GPU_STATIC_PASS',authority=bind(AUTH),families=list(FAMILIES),commands={k:child_command(dict(family=k,index=17)) for k in FAMILIES}))
    print('PACKED_GPU_PREPARED',flush=True)

def owned():
    gpu={};cpu={}
    for p in (POOL/'dispatch').glob('*.json'):
        d=read(p)
        if d.get('family') in FAMILIES:gpu[d['job']]=d['family'];cpu[d['callback']]='control'
    for p in FD.glob('wave[0-9][0-9][0-9][0-9].json'):gpu[read(p)['job_id']]='fusion'
    for p in (READOUT/'dispatch').glob('*.json'):
        d=read(p)
        if d.get('stage')=='follow':cpu[d['job']]='readout'
    return gpu,cpu

def pause_legacy():
    assert not (OUT/'legacy.json').exists();gpu,cpu=owned();held=[];running=[]
    for r in queue():
        root=r['job'].split('_')[0];kind=gpu.get(root);role=cpu.get(r['job'])
        if kind is None and role is None:continue
        if r['state']=='RUNNING':
            assert role is None,'WAIT_FOR_RUNNING_CPU_CONTROLLER';running.append(dict(**r,family=kind));continue
        assert r['state']=='PENDING',r
        before=jobinfo(r['job'])
        if before['JobState']!='PENDING':raise AssertionError('QUEUE_CHANGED_RETRY')
        assert before['UserId'].startswith('ap7811(')
        assert r['reason']!='JobHeldUser','PREEXISTING_HOLD_REQUIRES_SEPARATE_SCOPE'
        command(['scontrol','hold',r['job']]);after=jobinfo(r['job']);assert after['Reason']=='JobHeldUser'
        held.append(dict(**r,family=kind,role=role,before=before,after=after))
        write(OUT/'legacy_pause_progress.json',dict(held=held,running=running),replace=True)
    current=read(FD/'status.json');pid=current['pid'];cmdline=Path(f'/proc/{pid}/cmdline')
    assert cmdline.exists() and b'dispatch_rc_h593_feature_fusion_v3.py' in cmdline.read_bytes(),'WATCHER_HOST_MISMATCH'
    write(OUT/'legacy.json',dict(authority=bind(AUTH),held=held,running=running,watcher=dict(pid=pid,host=socket.gethostname(),cmdline=cmdline.read_bytes().decode(errors='replace'))))
    return read(OUT/'legacy.json')

def release_legacy():
    legacy=read(OUT/'legacy.json') if (OUT/'legacy.json').exists() else read(OUT/'legacy_pause_progress.json')
    for r in legacy['held']:
        p=command(['scontrol','show','job','-o',r['job']],check=False)
        if p.returncode:continue
        d=fields(p.stdout)
        if d.get('JobState')=='PENDING' and d.get('Reason')=='JobHeldUser':command(['scontrol','release',r['job']])
    status('PILOT_FAILED_LEGACY_RELEASED')

def choose_tasks(limit,exclude=()):
    used=set(exclude);attempts=Counter()
    for p in (OUT/'batches').glob('*/manifest.json'):
        for t in read(p)['tasks']:attempts[taskkey(t)]+=1
    for p in FD.glob('wave[0-9][0-9][0-9][0-9].json'):
        for i in read(p)['indices']:attempts['fusion_'+f'{i:03d}']+=1
    pools={k:sorted((dict(family=k,index=i) for i in range(200 if k=='fusion' else 593) if taskkey(dict(family=k,index=i)) not in used and not complete(dict(family=k,index=i))),key=lambda t:(attempts[taskkey(t)],t['index'])) for k in FAMILIES}
    tasks=[]
    while len(tasks)<limit and any(pools.values()):
        for k in FAMILIES:
            if pools[k] and len(tasks)<limit:tasks.append(pools[k].pop(0))
    assert len({taskkey(t) for t in tasks})==len(tasks)
    return tasks

def submit_batch(tasks,pilot=False,dependency=None):
    number=len(list((OUT/'batches').glob('*/manifest.json')));assert number<128
    d=OUT/'batches'/f'{number:03d}';minutes=30 if pilot else 55
    manifest=dict(authority=bind(AUTH),number=number,pilot=pilot,tasks=tasks,minutes=minutes,new_start_budget_seconds=minutes*60-950)
    write(d/'manifest.json',manifest)
    args=['sbatch','--parsable','--hold','--time='+f'00:{minutes:02d}:00']
    if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    args +=[str(LAUNCH),str(d/'manifest.json')]
    job=command(args).stdout.strip().split(';')[0];assert job.isdigit()
    try:
        spool=d/'spool.sh';command(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==LAUNCH.read_bytes()
        write(d/'submission.json',dict(job=job,manifest=bind(d/'manifest.json'),spool=bind(spool),authority=bind(AUTH),dependency=dependency))
        command(['scontrol','release',job])
    except BaseException:command(['scancel',job],check=False);raise
    print(json.dumps(dict(submitted=job,pilot=pilot,tasks=tasks),ensure_ascii=False),flush=True);return job

def start():
    guard()
    try:
        legacy=pause_legacy()
        excluded={r['family']+'_'+f"{int(r['job'].split('_')[1]):03d}" for r in legacy['running']}
        tasks=choose_tasks(4,excluded);assert {t['family'] for t in tasks}==set(FAMILIES)
        dep=':'.join(r['job'] for r in legacy['running']) or None
        job=submit_batch(tasks,True,dep);status('PILOT_SUBMITTED',job=job,tasks=tasks,legacy_pending_held=len(legacy['held']))
    except BaseException:
        if (OUT/'legacy_pause_progress.json').exists():release_legacy()
        raise

def worker(manifest,lane):
    guard();assert os.environ.get('SLURM_JOB_ID')
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_GPU_PER_STEP'
    props=torch.cuda.get_device_properties(0);gpu=os.environ.get('SLURM_STEP_GPUS');assert gpu,'REAL_SLURM_GPU_STEP_REQUIRED'
    d=Path(manifest).parent;m=read(manifest);start=read(d/'started.json')['monotonic'];job=os.environ['SLURM_JOB_ID']
    device=dict(lane=lane,job=job,step=os.environ.get('SLURM_STEP_ID'),host=socket.gethostname(),step_gpus=gpu,cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),device_name=props.name,uuid=str(getattr(props,'uuid','')))
    write(d/f'lane{lane}_device.json',device)
    tmp=Path(os.environ.get('SLURM_TMPDIR','/tmp'))/f'h593-pack-{job}-{lane}'
    for name in ('tmp','xdg','inductor','triton'):(tmp/name).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TMPDIR=str(tmp/'tmp'),XDG_CACHE_HOME=str(tmp/'xdg'),TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'),TRITON_CACHE_DIR=str(tmp/'triton'))
    assert 'SLURM_ARRAY_JOB_ID' not in env and 'SLURM_ARRAY_TASK_ID' not in env
    while time.monotonic()-start<m['new_start_budget_seconds']:
        with (d/'claim.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=read(d/'claims.json') if (d/'claims.json').exists() else {'claimed':[]}
            candidates=[m['tasks'][lane]] if m['pilot'] else m['tasks']
            remaining=[t for t in candidates if taskkey(t) not in state['claimed']]
            if not remaining:break
            t=remaining[0];state['claimed'].append(taskkey(t));write(d/'claims.json',state,replace=True)
        key=taskkey(t);started=time.monotonic();argv=child_command(t)
        with (d/(key+'.out')).open('w') as out,(d/(key+'.err')).open('w') as err:
            p=subprocess.run(argv,cwd=ROOT,env=env,stdout=out,stderr=err,timeout=850)
        assert p.returncode==0,(key,p.returncode)
        done=complete(t,True);source=None
        if t['family']=='fusion':
            source=folder(t)/'chunks'/f'{job}_{t["index"]}.json';r=read(source);assert r['status']=='FUSION_CHUNK_NORMAL_EXIT' and r['config_index']==t['index'] and r['job_id']==job
            assert r['authority']==bind(AUTHORITIES['fusion']);checked(r['checkpoint'])
        else:
            source=POOL/'consumers'/t['family']/f'{job}_{t["index"]}.json';r=read(source);assert r['index']==t['index'] and r['complete']==done
        write(d/(key+'.json'),dict(status='PACKED_TASK_NORMAL_EXIT',task=t,job=job,lane=lane,command=argv,seconds=time.monotonic()-started,complete=done,original_receipt=bind(source)))

def allocation(manifest):
    guard();assert os.environ.get('SLURM_JOB_ID')
    d=Path(manifest).parent;m=read(manifest);assert m['authority']==bind(AUTH)
    write(d/'started.json',dict(job=os.environ['SLURM_JOB_ID'],time_utc=now(),monotonic=time.monotonic()))
    procs=[];handles=[]
    for lane in range(4):
        out=(d/f'lane{lane}.out').open('w');err=(d/f'lane{lane}.err').open('w');handles.extend([out,err])
        argv=['srun','--exclusive','--exact','--nodes=1','--ntasks=1','--cpus-per-task=8','--gres=gpu:1','--mem=64G','--gpu-bind=single:1',str(PY),str(Path(__file__)),'worker','--manifest',str(manifest),'--lane',str(lane)]
        procs.append(subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err))
    codes=[p.wait() for p in procs]
    for h in handles:h.close()
    assert codes==[0,0,0,0],codes
    devices=[read(d/f'lane{k}_device.json') for k in range(4)];assert len({(v['host'],v['step_gpus']) for v in devices})==4
    completed=[]
    for t in m['tasks']:
        path=d/(taskkey(t)+'.json')
        if path.exists():
            r=read(path);assert r['status']=='PACKED_TASK_NORMAL_EXIT' and r['task']==t;checked(r['original_receipt']);completed.append(bind(path))
    if m['pilot']:
        assert len(completed)==4
        for t in m['tasks']:
            if t['family']!='fusion':assert complete(t,True),'PILOT_FULL_C128_REQUIRED'
    assert completed,'NO_WORK_COMPLETED'
    write(d/'validation.json',dict(status='PACKED_FOUR_GPU_PASS',authority=bind(AUTH),job=os.environ['SLURM_JOB_ID'],manifest=bind(manifest),devices=devices,task_receipts=completed,seconds=time.monotonic()-read(d/'started.json')['monotonic']))
    print('PACKED_FOUR_GPU_PASS',flush=True)

def cutover(pilot):
    legacy=read(OUT/'legacy.json');w=legacy['watcher'];assert w['host']==socket.gethostname()
    p=Path(f'/proc/{w["pid"]}/cmdline')
    if p.exists():assert p.read_bytes().decode(errors='replace')==w['cmdline'];os.kill(w['pid'],signal.SIGTERM)
    readout=[];cancelled=[]
    # Repoint protected downstream dependencies before canceling any predecessor.
    ordered=sorted(legacy['held'],key=lambda r:r['role']!='readout')
    for r in ordered:
        q=command(['scontrol','show','job','-o',r['job']],check=False)
        if q.returncode:continue
        state=fields(q.stdout);assert state['JobState']=='PENDING' and state['Reason']=='JobHeldUser',state
        if r['role']=='readout':
            command(['scontrol','update','JobId='+r['job'],'Dependency=afterok:'+pilot]);after=jobinfo(r['job']);assert after['Reason']=='JobHeldUser' and (after['Dependency']=='(null)' or pilot in after['Dependency']);readout.append(r['job'])
        else:command(['scancel',r['job']]);cancelled.append(r['job'])
    write(OUT/'cutover.json',dict(authority=bind(AUTH),pilot=pilot,cancelled_pending=cancelled,held_readout=readout,old_watcher_stopped=w,scientific_outputs_preserved=True))
    write(OUT/'old_fusion_status_at_cutover.json',read(FD/'status.json'))
    write(FD/'status.json',dict(time_utc=now(),pid=w['pid'],status='REPLACED_BY_PACKED_GPU_EXECUTION',new_status_path=str(OUT/'status.json'),cutover=bind(OUT/'cutover.json')),replace=True)

def finish_ready(cs):
    if cs['inside']==593 and not (OUT/'inside_ready.json').exists():
        for job in read(OUT/'cutover.json')['held_readout']:command(['scontrol','release',job])
        write(OUT/'inside_ready.json',dict(time_utc=now(),readout_released=True))
    for family in FAMILIES:
        total=200 if family=='fusion' else 593
        dest=OUT/(family+'_join.json')
        if cs[family]!=total or dest.exists():continue
        if family=='fusion':args=[ROOT/'slurm/rc_h593_feature_fusion_train_join_v2.sbatch']
        elif family=='coordinate':args=[ROOT/'slurm/rc_h593_roma_coordinate_cache_dispatch_v1.sbatch','run']
        else:args=[ROOT/f'slurm/rc_h593_m_{"inside" if family=="inside" else "visual_origin"}_control_v1.sbatch','join']
        job=command(['sbatch','--parsable',*args]).stdout.strip().split(';')[0];assert job.isdigit()
        write(dest,dict(job=job,time_utc=now(),arguments=list(map(str,args))))

def watch():
    a=guard();lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+a['maximum_hours']*3600;errors=0
    while time.monotonic()<deadline:
        try:
            batches=sorted((OUT/'batches').glob('*/submission.json'));assert batches
            latest=read(batches[-1]);job=latest['job'];d=batches[-1].parent
            active={r['job'].split('_')[0] for r in queue()}
            if job in active:status('PACKED_ALLOCATION_QUEUED_OR_RUNNING',job=job,batch=d.name);time.sleep(15);continue
            acc=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode']).stdout.strip().splitlines()
            if not acc:time.sleep(15);continue
            if not all(s.split('|')[1:3]==['COMPLETED','0:0'] for s in acc) or not (d/'validation.json').exists():
                if not (OUT/'cutover.json').exists():release_legacy()
                else:status('STOPPED_PACKED_JOB_FAILURE',job=job,accounting=acc)
                return
            v=read(d/'validation.json');assert v['status']=='PACKED_FOUR_GPU_PASS' and v['authority']==bind(AUTH);checked(v['manifest'])
            for b in v['task_receipts']:checked(b)
            if not (OUT/'cutover.json').exists():cutover(job)
            cs=counts();finish_ready(cs);tasks=choose_tasks(32)
            if not tasks:status('ALL_PRODUCERS_COMPLETE_FINAL_JOINS_SUBMITTED',counts=cs);return
            new=submit_batch(tasks);status('PACKED_CONTINUATION_SUBMITTED',job=new,counts=cs);errors=0
        except Exception as e:
            errors+=1;status('SCHEDULER_ERROR',error=str(e),consecutive=errors)
            if errors>=3:return
        time.sleep(15)
    status('STOPPED_72H_LIMIT')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','start','allocation','worker','watch','rollback'));p.add_argument('--manifest',type=Path);p.add_argument('--lane',type=int);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='start':start()
    elif args.stage=='allocation':allocation(args.manifest)
    elif args.stage=='worker':worker(args.manifest,args.lane)
    elif args.stage=='watch':watch()
    else:release_legacy()
