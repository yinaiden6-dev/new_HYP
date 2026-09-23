#!/usr/bin/env python3
"""Bounded one-job continuation for the first-fold mass teacher diagnostic."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_fold0_mass_teacher_v1'
AUTH=ROOT/'registry/rc_fold0_mass_teacher_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_fold0_mass_teacher_v1.sbatch'
WAITING=('PENDING','RUNNING','COMPLETING','CONFIGURING','ACCOUNTING_WAIT')


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def run(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout.strip()
def state(job):
    p=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=45)
    if p.stdout.strip():return p.stdout.strip().splitlines()[0]
    if p.returncode and 'Invalid job' not in p.stderr:raise RuntimeError(p.stderr)
    for line in run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode']).splitlines():
        q=line.split('|')
        if q[0]==job:return 'COMPLETED' if q[1]=='COMPLETED' and q[2]=='0:0' else q[1]+':'+q[2]
    return 'ACCOUNTING_WAIT'
def main():
    lock=(OUT/'dispatcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a=read(AUTH)
    for b in a['sources']:
        assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
    path=OUT/'dispatch.json';d=read(path) if path.exists() else dict(active=None,chunks=0,history=[])
    started=time.monotonic()
    while time.monotonic()-started<3*86400:
        try:
            if d['active']:
                j=d['active'];s=state(j['job_id'])
                if s=='COMPLETED':
                    if j['stage']=='fit':assert read(OUT/'chunks'/f"{j['job_id']}.json")['status']=='M_TEACHER_NORMAL_CHUNK'
                    else:assert read(OUT/'validation.json')['status']=='M_TEACHER_NATIVE_PARITY_AND_NUMPY_PASS'
                    d['history'].append(j);d['active']=None;write(path,d)
                elif s not in WAITING:raise RuntimeError('JOB_FAILED:'+str(j)+':'+s)
            if not d['active']:
                if (OUT/'validation.json').exists():
                    write(OUT/'status.json',dict(status='COMPLETE',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),history=d['history']));return
                stage='join' if (OUT/'fit_validation.json').exists() else 'fit'
                if stage=='fit' and d['chunks']>=a['max_chunks']:raise RuntimeError('CHUNK_BUDGET')
                args=['sbatch','--parsable','--partition=dev_cpuonly',str(LAUNCH),stage]
                job=run(args).split(';')[0];assert job.isdigit()
                d['active']=dict(job_id=job,stage=stage,args=args)
                if stage=='fit':d['chunks']+=1
                write(path,d)
            write(OUT/'status.json',dict(status='TEACHER_DIAGNOSTIC_RUNNING',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    pid=os.getpid(),host=socket.gethostname(),active=d['active'],chunks=d['chunks'],partition='dev_cpuonly',only_fold=0,head_training=False))
        except Exception as e:
            error=str(e);retry=any(x in error for x in ('QOSMaxSubmitJobPerUserLimit','Socket timed out','temporarily unavailable'))
            write(OUT/'status.json',dict(status='RETRY_SCHEDULER' if retry else 'STOPPED_ERROR',error=error,pid=os.getpid(),host=socket.gethostname()))
            if not retry:raise
        time.sleep(30)
    write(OUT/'status.json',dict(status='STOPPED_TIME_BUDGET',pid=os.getpid()))


if __name__=='__main__':main()
