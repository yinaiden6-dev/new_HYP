#!/usr/bin/env python3
"""Bounded placement helper for only the user-authorized 5157112_[32-46]."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_quality_operator_dispatch_v1/dev_migration_5157112'
ARRAY='5157112';USER='ap7811';DEST='dev_cpuonly';LIMIT=4
ALLOWED={f'{ARRAY}_{i}' for i in range(32,47)}
FIELDS=('JobState','Reason','Partition','TimeLimit','NumCPUs','CPUs/Task','MinMemoryNode','ReqTRES','Dependency')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(kind,**value):
    item=dict(time_utc=now(),event=kind,**value)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
    print(json.dumps(item,ensure_ascii=False),flush=True)


def command(args,check=True):
    p=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if check and p.returncode:raise RuntimeError('COMMAND_FAILED: '+p.stderr.strip())
    return p


def parse_queue(value):
    rows=[]
    for line in value.splitlines():
        if not line.strip():continue
        job,partition,state,qos,reason=line.split('|',4)
        rows.append(dict(job=job.strip(),partition=partition.strip(),state=state.strip(),qos=qos.strip(),reason=reason.strip()))
    return rows


def snapshot():
    return parse_queue(command(['/usr/bin/squeue','-h','-r','-u',USER,'-o','%i|%P|%T|%q|%R']).stdout)


def next_jobs(rows):
    occupied=sum(r['qos']=='dev' or any(p.startswith('dev_') for p in r['partition'].split(',')) for r in rows)
    candidates=sorted((r['job'] for r in rows if r['job'] in ALLOWED and r['partition']=='cpuonly' and r['state']=='PENDING'),key=lambda v:int(v.rsplit('_',1)[1]))
    return candidates[:max(0,LIMIT-occupied)],occupied,candidates


def details(job):
    assert job in ALLOWED,'JOB_SCOPE'
    raw=command(['/usr/bin/scontrol','show','job',job,'-o']).stdout
    d=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)',raw))
    assert d.get('ArrayJobId')==ARRAY and d.get('ArrayTaskId')==job.rsplit('_',1)[1],'ARRAY_IDENTITY'
    assert d.get('UserId','').startswith(USER+'(') and d.get('JobName')=='h593_operator','OWNER_NAME'
    assert d.get('TimeLimit')=='00:05:00' and d.get('CPUs/Task')=='8' and d.get('MinMemoryNode')=='32G','RESOURCE_CONTRACT'
    assert 'gpu' not in d.get('ReqTRES','').lower(),'NO_GPU'
    return d


def write_status(status,rows,moved,**extra):
    p=OUT/'status.json';tmp=OUT/f'.status.{os.getpid()}.tmp'
    item=dict(time_utc=now(),status=status,pid=os.getpid(),array_id=ARRAY,indices=[32,46],destination=DEST,dev_submit_limit=LIMIT,poll_seconds=30,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),migrated_by_helper=sorted(moved),
              target_jobs=[r for r in rows if r['job'] in ALLOWED],**extra)
    tmp.write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def run():
    assert pwd.getpwuid(os.getuid()).pw_name==USER,'USER_SCOPE'
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'helper.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    deadline=time.monotonic()+24*3600;moved=set();rows=[];errors=0
    event('STARTED',pid=os.getpid(),array=ARRAY,indices=[32,46],maximum_hours=24)
    while time.monotonic()<deadline:
        try:
            rows=snapshot();todo,occupied,pending=next_jobs(rows)
            if not pending:
                accounting=command(['/usr/bin/sacct','-X','-n','-P','-j',ARRAY,'--format=JobID,State,Partition'],check=False)
                write_status('NO_CPUONLY_PENDING_TARGETS',rows,moved,accounting=accounting.stdout)
                event('PLACEMENT_FINISHED',experiment_completion_claimed=False);return
            write_status('ACTIVE',rows,moved,dev_slots_occupied=occupied,pending_migration=len(pending))
            for job in todo:
                before=details(job)
                if before['JobState']!='PENDING' or before['Partition']!='cpuonly':
                    event('STATE_CHANGED_SKIP',job=job,before={k:before.get(k) for k in FIELDS});continue
                # Recheck shared dev occupancy before each mutation; scheduler enforces races.
                if next_jobs(snapshot())[1]>=LIMIT:break
                p=command(['/usr/bin/scontrol','update','JobId='+job,'Partition='+DEST],check=False)
                if p.returncode:
                    event('MIGRATION_REJECTED_RETRY_LATER',job=job,returncode=p.returncode,message=p.stderr.strip());break
                after=details(job)
                assert after['Partition']==DEST,'PARTITION_NOT_CHANGED'
                for k in ('TimeLimit','CPUs/Task','MinMemoryNode','Dependency'):
                    assert before.get(k)==after.get(k),'RESOURCE_OR_DEPENDENCY_CHANGED:'+k
                if after.get('Reason','').startswith('JobHeld'):
                    command(['/usr/bin/scontrol','release',job])
                    after=details(job)
                    assert after['Partition']==DEST and not after.get('Reason','').startswith('JobHeld'),'RELEASE_FAILED'
                moved.add(job);event('MIGRATED_VERIFIED',job=job,before={k:before.get(k) for k in FIELDS},after={k:after.get(k) for k in FIELDS})
            if todo:write_status('ACTIVE',snapshot(),moved)
            errors=0
        except AssertionError as exc:
            write_status('STOPPED_CONTRACT_CHECK',rows,moved,error=str(exc));event('STOPPED_CONTRACT_CHECK',error=str(exc));raise
        except Exception as exc:
            errors+=1;event('TRANSIENT_ERROR',consecutive=errors,error=str(exc))
            if errors>=5:
                write_status('STOPPED_REPEATED_ERRORS',rows,moved,error=str(exc));raise
        time.sleep(30)
    write_status('STOPPED_24_HOUR_LIMIT',rows,moved);event('STOPPED_24_HOUR_LIMIT')


def self_test():
    rows=parse_queue('5157112_34|cpuonly|PENDING|normal|JobHeldUser\n5157112_32|cpuonly|PENDING|normal|JobHeldUser\n5157112_31|cpuonly|PENDING|normal|Priority\n5157112_47|cpuonly|PENDING|normal|Priority\n5156927_32|cpuonly|PENDING|normal|Priority\n')
    assert next_jobs(rows)[0]==['5157112_32','5157112_34']
    rows+=parse_queue('1|dev_cpuonly|RUNNING|normal|None\n2|dev_accelerated|PENDING|normal|Priority\n3|dev_cpuonly|PENDING|normal|Priority')
    assert next_jobs(rows)[0]==['5157112_32']
    rows+=parse_queue('4|dev_cpuonly|PENDING|normal|Priority')
    assert not next_jobs(rows)[0]
    assert ALLOWED=={f'5157112_{i}' for i in range(32,47)}
    print('PURE_CPU_REPLACEMENT_SCOPE_AND_DEV_CAP_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('self-test','run'));args=p.parse_args()
    self_test() if args.stage=='self-test' else run()
