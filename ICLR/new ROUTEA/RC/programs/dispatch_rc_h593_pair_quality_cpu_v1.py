#!/usr/bin/env python3
"""CPU-only bounded continuation of the sealed six-arm experiment."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_pair_quality_cpu_v1'
AUTH=ROOT/'registry/rc_h593_pair_quality_cpu_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_h593_pair_quality_cpu_v1.sbatch'
WAITING=('PENDING','RUNNING','COMPLETING','CONFIGURING','ACCOUNTING_WAIT')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(exist_ok=True,parents=True)
    t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout.strip()
def submit(stage,index=None):
    args=['/usr/bin/sbatch','--parsable','--partition=dev_cpuonly',str(LAUNCH),stage]
    if index is not None:args.append(str(index))
    job=command(args).split(';')[0]
    if not job.isdigit():raise RuntimeError('invalid job '+job)
    return dict(job_id=job,stage=stage,index=index,args=args)
def state(job):
    p=subprocess.run(['/usr/bin/squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=45)
    if p.returncode and 'Invalid job' not in p.stderr:raise RuntimeError(p.stderr)
    s=p.stdout.strip()
    if s:return s.splitlines()[0]
    s=command(['/usr/bin/sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'])
    for line in s.splitlines():
        parts=line.split('|')
        if parts[0]==job:
            return 'COMPLETED' if parts[1]=='COMPLETED' and parts[2]=='0:0' else parts[1]+':'+parts[2]
    return 'ACCOUNTING_WAIT'
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'dispatcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=read(AUTH)
    if auth['execution_device']!='cpu' or auth['capacity_gate_required']:raise RuntimeError('CPU authority required')
    for b in auth['sources']:
        if hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()!=b['sha256']:
            from rc_h593_pair_quality_execution_source_compat_v1 import approved_execution_source_change
            if not approved_execution_source_change(b):raise RuntimeError('source drift')
    total=len(auth['configs']);parallel=auth['max_parallel']
    ledger=OUT/'dispatch.json'
    d=read(ledger) if ledger.exists() else dict(active=[],chunks={},join=None,complete=[],partition='dev_cpuonly')
    d['partition']='dev_cpuonly'
    start=time.monotonic()
    while time.monotonic()-start<3*86400:
        try:
            active=[]
            for j in d['active']:
                js=state(j['job_id'])
                if js=='COMPLETED':
                    if (OUT/f"fit{j['index']:02d}/validation.json").exists():d['complete'].append(j['index'])
                elif js in WAITING:active.append(j)
                else:raise RuntimeError('fit failed '+str(j)+' '+js)
            d['active']=active;write(ledger,d)
            while len(d['active'])<parallel and len(set(d['complete']))<total:
                occupied={j['index'] for j in d['active']}|set(d['complete'])
                available=[i for i in range(total) if i not in occupied]
                if not available:break
                index=available[0];attempts=d['chunks'].get(str(index),0)
                if attempts>=auth['max_chunks_per_config']:raise RuntimeError('chunk budget '+str(index))
                j=submit('fit',index);d['active'].append(j);d['chunks'][str(index)]=attempts+1;write(ledger,d)
            if len(set(d['complete']))==total:
                if d['join'] is None:d['join']=submit('join');write(ledger,d)
                js=state(d['join']['job_id'])
                if js=='COMPLETED':
                    if not (OUT/'validation.json').exists():raise RuntimeError('join missing validation')
                    write(OUT/'dispatch_status.json',dict(status='COMPLETE',join=d['join'],partition='dev_cpuonly'));return
                if js not in WAITING:raise RuntimeError('join failed '+js)
            write(OUT/'dispatch_status.json',dict(status='CPU_TRAIN_OR_JOIN',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                active=d['active'],complete=len(set(d['complete'])),total=total,pid=os.getpid(),partition='dev_cpuonly',capacity_probe=False))
        except Exception as e:
            error=str(e)
            if any(x in error for x in ('QOSMaxSubmitJobPerUserLimit','temporarily unavailable','Socket timed out')):
                write(OUT/'dispatch_status.json',dict(status='RETRY_SCHEDULER',error=error,pid=os.getpid(),partition='dev_cpuonly'))
            else:
                write(OUT/'dispatch_status.json',dict(status='STOPPED_ERROR',error=error,pid=os.getpid(),partition='dev_cpuonly'));raise
        time.sleep(30)
    write(OUT/'dispatch_status.json',dict(status='STOPPED_TIME_BUDGET',pid=os.getpid(),partition='dev_cpuonly'))
if __name__=='__main__':main()
