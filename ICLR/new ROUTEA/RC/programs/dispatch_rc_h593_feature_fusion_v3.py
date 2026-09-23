#!/usr/bin/env python3
"""Bounded scheduler-only continuation for the already authorized fusion branch."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'results/rc_h593_feature_fusion_cache_v1'
TRAIN=ROOT/'results/rc_h593_feature_fusion_train_v2'
OUT=ROOT/'results/rc_h593_feature_fusion_dispatch_v3'
AUTH=ROOT/'registry/rc_h593_feature_fusion_dispatch_authority_v3_20260922.json'


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bound(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def checked(b):
    assert bound(Path(b['path']))==b,b['path'];return Path(b['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,indent=2,ensure_ascii=False)+'\n'
    if p.exists():assert p.read_text()==text;return
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');tmp.write_text(text);os.link(tmp,p);tmp.unlink()
def event(kind,**kw):
    value=dict(time_utc=now(),event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)
def command(args,check=True):
    p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=45)
    if check and p.returncode:raise RuntimeError(p.stderr.strip())
    return p


def prepare():
    assert not AUTH.exists()
    write(AUTH,dict(status='FUSION_AUTOMATIC_CONTINUATION_AUTHORIZED',sources=[bound(Path(__file__))],
        cache_authority=bound(ROOT/'registry/rc_h593_feature_fusion_cache_authority_v1_20260922.json'),
        train_authority=bound(ROOT/'registry/rc_h593_feature_fusion_train_authority_v2_20260922.json'),
        catalog_submission=bound(CACHE/'catalog_submitted.json'),benchmark_submission=bound(TRAIN/'benchmark_submitted.json'),
        maximum_hours=48,maximum_parallel=46,max_chunks_per_config=128,config_count=200,
        first_natural_training_gate='Only config0 in first wave; normal checkpoint receipt required before wider release',
        user_scope='Continue feature-fusion branch automatically after job5157186; keep original scientific programs frozen',
        dev_boost='Only this branch catalog, benchmark, feature extraction, cache join, final join; pending jobs only; shared dev cap4',
        temporary_backfill_pause='May pause helper4138063 for at most20min to let the two short fusion prerequisites acquire dev; never pause a Slurm computation'))


def queue():
    p=command(['squeue','-h','-r','-u','ap7811','-o','%i|%j|%P|%T|%r'])
    return [dict(zip(('job','name','partition','state','reason'),line.split('|',4))) for line in p.stdout.splitlines() if line.strip()]


def boost(rows,jobs):
    occupied=sum(any(p.startswith('dev_') for p in r['partition'].split(',')) for r in rows)
    if occupied>=4:return
    for job,name,gpu in jobs:
        candidates=[r for r in rows if (r['job']==job or r['job'].startswith(job+'_')) and r['name']==name and r['state']=='PENDING' and not r['partition'].startswith('dev_')]
        for row in candidates[:4-occupied]:
            text=command(['scontrol','show','job','-o',row['job']]).stdout
            d=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)',text))
            assert d['UserId'].startswith('ap7811(') and d['JobName']==name
            if d['JobState']!='PENDING':continue
            assert ('gpu' in d['ReqTRES'])==gpu
            dest='dev_accelerated' if gpu else 'dev_cpuonly'
            p=command(['scontrol','update','JobId='+row['job'],'Partition='+dest],check=False)
            if p.returncode:
                event('DEV_SLOT_BUSY',job=row['job'],message=p.stderr.strip());return
            text=command(['scontrol','show','job','-o',row['job']]).stdout
            after=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)',text))
            assert after['Partition']==dest
            for k in ('TimeLimit','ReqTRES','Dependency'):assert d.get(k)==after.get(k)
            event('DEV_PLACEMENT_VERIFIED',job=row['job'],partition=dest);occupied+=1
            if occupied>=4:return


def submit(p,args,metadata):
    if p.exists():
        previous=read(p);assert previous['args']==args;return previous['job_id']
    proc=command(['sbatch','--parsable',*args]);job=proc.stdout.strip().split(';')[0];assert job.isdigit()
    write(p,dict(job_id=job,args=args,authority=bound(AUTH),**metadata));event('SUBMITTED',job=job,**metadata);return job


def status(value):
    p=OUT/'status.json';tmp=OUT/f'.status.{os.getpid()}.tmp';tmp.write_text(json.dumps(dict(time_utc=now(),pid=os.getpid(),**value),indent=2)+'\n');os.replace(tmp,p)


def watch():
    a=read(AUTH)
    for b in [*a['sources'],a['cache_authority'],a['train_authority'],a['catalog_submission'],a['benchmark_submission']]:checked(b)
    train_auth=read(a['train_authority']['path']);assert len(train_auth['configs'])==200
    OUT.mkdir(exist_ok=True,parents=True);lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    catalog=read(CACHE/'catalog_submitted.json')['job_id'];bench=read(TRAIN/'benchmark_submitted.json')['job_id']
    paused=False;helper=4138063;pause_start=time.monotonic();deadline=time.monotonic()+48*3600;errors=0
    helperpath=Path(f'/proc/{helper}/cmdline')
    if helperpath.exists() and b'migrate_h593_operator_5157222_dev_cpuonly_v1.py' in helperpath.read_bytes():
        os.kill(helper,signal.SIGSTOP);paused=True;event('CPU_BACKFILL_TEMPORARILY_PAUSED',pid=helper,maximum_seconds=1200)
    def resume():
        nonlocal paused
        if paused:
            if helperpath.exists() and b'migrate_h593_operator_5157222_dev_cpuonly_v1.py' in helperpath.read_bytes():os.kill(helper,signal.SIGCONT)
            paused=False;event('CPU_BACKFILL_RESUMED',pid=helper)
    try:
        while time.monotonic()<deadline:
            try:
                rows=queue();jobs=[(catalog,'h593_fcat',False),(bench,'h593_fbench',True)]
                if (CACHE/'extract_submitted.json').exists():jobs.append((read(CACHE/'extract_submitted.json')['job_id'],'h593_fcache',True))
                if (CACHE/'join_submitted.json').exists():jobs.append((read(CACHE/'join_submitted.json')['job_id'],'h593_fcat',False))
                if (OUT/'join_submitted.json').exists():jobs.append((read(OUT/'join_submitted.json')['job_id'],'h593_fjoin',False))
                boost(rows,jobs)
                if paused and ((CACHE/'catalog_validation.json').exists() and (TRAIN/'benchmark_validation.json').exists() or time.monotonic()-pause_start>1200):resume()
                for job,_,_ in jobs:
                    if any(r['job']==job or r['job'].startswith(job+'_') for r in rows):continue
                    acc=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'],check=False).stdout
                    bad=[line for line in acc.splitlines() if any(s in line for s in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'))]
                    if bad:raise AssertionError('UPSTREAM_JOB_FAILED:'+str(bad))
                if (TRAIN/'validation.json').exists():
                    v=read(TRAIN/'validation.json');checked(v['result']);assert v['status']=='FUSION_JOIN_COUNTS_AND_GROUPS_PASS'
                    status(dict(status='COMPLETE',validation=bound(TRAIN/'validation.json')));event('ALL_FUSION_CONFIGS_COMPLETE');return
                if not (CACHE/'ready.json').exists() or not (TRAIN/'benchmark_validation.json').exists():
                    status(dict(status='WAITING_FOR_FEATURES_AND_TRAINING_CAPACITY',catalog_job=catalog,benchmark_job=bench,
                        features_ready=(CACHE/'ready.json').exists(),capacity_ready=(TRAIN/'benchmark_validation.json').exists(),cpu_helper_paused=paused))
                    errors=0;time.sleep(15);continue
                ready=read(CACHE/'ready.json');capacity=read(TRAIN/'benchmark_validation.json')
                assert ready['status']=='FUSION_ALL593_FEATURE_CACHE_PASS' and ready['authority']==a['cache_authority']
                assert capacity['status']=='FUSION_TRAINING_CAPACITY_PASS' and capacity['authority']==a['train_authority'];checked(capacity['payload'])
                complete=set()
                for i in range(200):
                    p=TRAIN/f'fit{i:03d}/validation.json'
                    if p.exists():
                        v=read(p);assert v['status']=='FUSION_FOLD_NUMPY_READOUT_PASS' and v['authority']==a['train_authority'];checked(v['payload']);complete.add(i)
                waves=sorted(OUT.glob('wave[0-9][0-9][0-9][0-9].json'));active=[];attempts={i:0 for i in range(200)}
                for path in waves:
                    wave=read(path);job=wave['job_id']
                    for i in wave['indices']:attempts[i]+=1
                    if any(r['job']==job or r['job'].startswith(job+'_') for r in rows):active.append(job);continue
                    for i in wave['indices']:
                        receipt=TRAIN/f'fit{i:03d}/chunks/{job}_{i}.json'
                        if not receipt.exists():
                            acc=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'],check=False).stdout
                            if any(s in acc for s in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')):raise AssertionError('TRAIN_WAVE_FAILED:'+job)
                            # Accounting or completion seal may lag the queue.
                            active.append(job);break
                        rec=read(receipt);assert rec['status']=='FUSION_CHUNK_NORMAL_EXIT' and rec['authority']==a['train_authority']
                if len(complete)==200:
                    join=submit(OUT/'join_submitted.json',[str(ROOT/'slurm/rc_h593_feature_fusion_train_join_v2.sbatch')],dict(stage='ALL200_JOIN'))
                    status(dict(status='FINAL_JOIN_QUEUED',job=join,complete_configs=200));errors=0;time.sleep(15);continue
                if not active:
                    todo=[0] if not waves else sorted((i for i in range(200) if i not in complete),key=lambda i:(attempts[i],i))[:46]
                    assert all(attempts[i]<128 for i in todo),'MAXIMUM_CHUNKS_REACHED'
                    args=['--array='+','.join(map(str,todo))+'%46',str(ROOT/'slurm/rc_h593_feature_fusion_train_v2.sbatch'),'fit']
                    job=submit(OUT/f'wave{len(waves):04d}.json',args,dict(stage='FUSION_TRAIN_OR_RESUME',indices=todo))
                    active=[job]
                status(dict(status='TRAINING_OR_PREDICTING',active_arrays=active,complete_configs=len(complete),total_configs=200))
                errors=0
            except AssertionError as exc:
                status(dict(status='STOPPED_VALIDATION_FAILURE',error=str(exc)));event('STOPPED_VALIDATION_FAILURE',error=str(exc));raise
            except Exception as exc:
                errors+=1;event('TRANSIENT_ERROR',error=str(exc),consecutive=errors)
                if errors>=5:status(dict(status='STOPPED_REPEATED_ERRORS',error=str(exc)));raise
            time.sleep(15)
        status(dict(status='STOPPED_48H_LIMIT'))
    finally:resume()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','watch'));args=ap.parse_args()
    prepare() if args.stage=='prepare' else watch()
