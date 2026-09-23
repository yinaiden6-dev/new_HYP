#!/usr/bin/env python3
"""Small bounded rolling queue; never mutates other experiments."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_pair_quality_v1'
AUTH=ROOT/'registry/rc_h593_pair_quality_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_h593_pair_quality_v1.sbatch'

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout.strip()
def submit(stage,index=None):
    # No duplicate dev/ordinary copies. Slurm picks one eligible partition.
    args=['/usr/bin/sbatch','--parsable','--partition=accelerated,dev_accelerated',str(LAUNCH),stage]
    if index is not None:args.append(str(index))
    job=command(args).split(';')[0]
    if not job.isdigit():raise RuntimeError('invalid job '+job)
    return dict(job_id=job,stage=stage,index=index,args=args)
def state(job):
    p=subprocess.run(['/usr/bin/squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=45)
    if p.returncode and 'Invalid job id' not in p.stderr and 'Invalid job' not in p.stderr:raise RuntimeError(p.stderr)
    s=p.stdout.strip()
    if s:return s.splitlines()[0]
    s=command(['/usr/bin/sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'])
    for line in s.splitlines():
        p=line.split('|')
        if p[0]==job:return 'COMPLETED' if p[1]=='COMPLETED' and p[2]=='0:0' else p[1]+':'+p[2]
    return 'ACCOUNTING_WAIT'
def main():
    import fcntl
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'dispatcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=read(AUTH)
    for b in auth['sources']:
        if hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()!=b['sha256']:raise RuntimeError('source drift')
    ledger=OUT/'dispatch.json'
    d=read(ledger) if ledger.exists() else dict(benchmark=None,active=[],chunks={},join=None,complete=[])
    start=time.monotonic()
    while time.monotonic()-start<3*86400:
        try:
            if d['benchmark'] is None:
                d['benchmark']=submit('benchmark');write(ledger,d)
            s=state(d['benchmark']['job_id'])
            if s not in ('COMPLETED','PENDING','RUNNING','COMPLETING','CONFIGURING','ACCOUNTING_WAIT'):raise RuntimeError('benchmark failed '+s)
            if s=='COMPLETED':
                if not (OUT/'benchmark_validation.json').exists():raise RuntimeError('qualification missing')
                active=[]
                for j in d['active']:
                    js=state(j['job_id'])
                    if js=='COMPLETED':
                        if (OUT/f"fit{j['index']:02d}/validation.json").exists():d['complete'].append(j['index'])
                    elif js in ('PENDING','RUNNING','COMPLETING','CONFIGURING','ACCOUNTING_WAIT'):active.append(j)
                    else:raise RuntimeError('fit failed '+str(j)+' '+js)
                d['active']=active;write(ledger,d)
                while len(d['active'])<4 and len(set(d['complete']))<30:
                    occupied={j['index'] for j in d['active']}|set(d['complete'])
                    # Finish earlier configs before opening further work.
                    index=next(i for i in range(30) if i not in occupied)
                    attempts=d['chunks'].get(str(index),0)
                    if attempts>=128:raise RuntimeError('chunk budget '+str(index))
                    j=submit('fit',index);d['active'].append(j);d['chunks'][str(index)]=attempts+1;write(ledger,d)
                if len(set(d['complete']))==30:
                    if d['join'] is None:d['join']=submit('join');write(ledger,d)
                    js=state(d['join']['job_id'])
                    if js=='COMPLETED':
                        if not (OUT/'validation.json').exists():raise RuntimeError('join missing validation')
                        write(OUT/'dispatch_status.json',dict(status='COMPLETE',join=d['join']));return
            write(OUT/'dispatch_status.json',dict(status='BENCHMARK_OR_TRAIN',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                benchmark=d['benchmark'],active=d['active'],complete=len(set(d['complete'])),total=30,pid=os.getpid()))
        except Exception as e:
            text=str(e)
            if any(x in text for x in ('QOSMaxSubmitJobPerUserLimit','temporarily unavailable','Socket timed out')):
                write(OUT/'dispatch_status.json',dict(status='RETRY_SCHEDULER',error=text,pid=os.getpid()))
            else:
                write(OUT/'dispatch_status.json',dict(status='STOPPED_ERROR',error=text,pid=os.getpid()));raise
        time.sleep(30)
    write(OUT/'dispatch_status.json',dict(status='STOPPED_TIME_BUDGET',pid=os.getpid()))
if __name__=='__main__':main()
