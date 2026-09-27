#!/usr/bin/env python3
"""Place this exact chain in CPU/dev spare slots; never cancel or resubmit."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_token_competition_f128_v2/dev_chain_placement'
# User's accelerated + idle GPU amendments exclude train tasks 2..14.
JOBS = {**{f'5167552_{i}':'token128_evidence' for i in range(16)},
        **{f'5167554_{i}':'token128_train' for i in range(2)},
        '5167553':'token128_verify', '5167555':'token128_join'}
TARGET = 'cpuonly,dev_cpuonly'
BLOCKED = ('JobHeldUser', 'JobHeldAdmin', 'Dependency', 'DependencyNeverSatisfied')
# Slurm can satisfy/clear Dependency between the two reads. The update command
# never writes Dependency; record both values without treating progress as drift.
UNCHANGED = ('TimeLimit', 'ReqTRES', 'CPUs/Task', 'Requeue')


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def run(cmd, check=True):
    return subprocess.run(cmd, text=True, capture_output=True, check=check, timeout=30)


def queue():
    raw = run(['squeue', '-h', '-r', '-u', 'ap7811', '-o', '%i|%j|%P|%T|%r']).stdout
    return [dict(zip(('job', 'name', 'partition', 'state', 'reason'), line.split('|', 4)))
            for line in raw.splitlines() if line]


def info(job):
    raw = run(['scontrol', 'show', 'job', '-o', job]).stdout
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)', raw))


def candidates(rows):
    occupied = sum(any(p.startswith('dev_') for p in r['partition'].split(',')) for r in rows)
    possible = [r for r in rows if r['job'] in JOBS and r['name'] == JOBS[r['job']]
                and r['state'] == 'PENDING' and r['partition'] == 'cpuonly'
                and r['reason'] not in BLOCKED]
    possible.sort(key=lambda r: tuple(map(int,r['job'].split('_'))))
    return possible[:max(0, 4-occupied)]


def event(value):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), **value})+'\n')


def status(value):
    p=OUT/'status.json'; temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps({'utc':now(), 'pid':os.getpid(), 'host':socket.gethostname(),
        'scope':sorted(JOBS), 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **value}, indent=2)+'\n')
    os.replace(temp,p)


def move(job, target):
    before=info(job)
    assert job in JOBS and before['UserId'].startswith('ap7811(') and before['JobName']==JOBS[job]
    if '_' in job:
        parent,index=job.split('_')
        assert before['ArrayJobId']==parent and before['ArrayTaskId']==index
    else: assert before['JobId']==job
    if before['JobState']!='PENDING': return
    if target==TARGET and (before['Partition']!='cpuonly' or before['Reason'] in BLOCKED): return
    if target=='cpuonly' and (before['Reason'] not in BLOCKED or 'dev_cpuonly' not in before['Partition']): return
    assert before['TimeLimit']=='00:10:00' and before['CPUs/Task']=='8'
    assert before['ReqTRES']=='cpu=8,mem=32G,node=1,billing=8'
    result=run(['scontrol','update','JobId='+job,'Partition='+target],check=False)
    if result.returncode:
        event({'event':'SCHEDULER_RETRY','job':job,'stderr':result.stderr.strip()}); return False
    after=info(job)
    # A pending task can be dispatched during verification; Slurm then reports
    # only the selected partition instead of the submitted alternatives.
    assert after['Partition'] and set(after['Partition'].split(',')) <= set(target.split(','))
    for field in UNCHANGED: assert before.get(field)==after.get(field),field
    event({'event':'MIGRATED_VERIFIED','job':job,'before':before,'after':after})
    return True


def tick():
    # Waiting dependencies should not reserve the four dev submission slots.
    for row in queue():
        if (row['job'] in JOBS and row['state']=='PENDING' and row['reason'] in BLOCKED
                and 'dev_cpuonly' in row['partition'].split(',')):
            move(row['job'],'cpuonly')
    for row in candidates(queue()):
        if row['job'] not in {r['job'] for r in candidates(queue())}: break
        if move(row['job'],TARGET) is False: break
    rows=queue(); own=[r for r in rows if r['job'] in JOBS]
    remaining=[r for r in own if r['state']=='PENDING' and r['partition']=='cpuonly']
    status({'status':'DEV_PLACEMENT_ACTIVE' if remaining else 'DEV_PLACEMENT_COMPLETE',
            'remaining_cpu_pending':len(remaining),'jobs':own})
    return bool(remaining)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:
        rows=[dict(job=f'5167552_{i}',name='token128_evidence',partition='cpuonly',state='PENDING',reason='Priority') for i in range(16)]
        rows += [dict(job=str(i),name='other',partition='cpuonly,dev_cpuonly',state='PENDING',reason='Dependency') for i in range(2)]
        assert [r['job'] for r in candidates(rows)]==['5167552_0','5167552_1']
        rows[0]['state']='RUNNING';rows[1]['reason']='JobHeldUser'
        assert [r['job'] for r in candidates(rows)]==['5167552_2','5167552_3']
        rows[2]['partition']='dev_cpuonly'
        assert [r['job'] for r in candidates(rows)]==['5167552_3']
        rows[3]['partition']='dev_cpuonly'; assert not candidates(rows)
        assert not candidates([dict(job='999999_0',name='token128_train',partition='cpuonly',state='PENDING',reason='Priority')])
        assert candidates([dict(job='5167553',name='token128_verify',partition='cpuonly',state='PENDING',reason='Priority')])[0]['job']=='5167553'
        assert not candidates([dict(job='5167554_0',name='token128_train',partition='cpuonly',state='PENDING',reason='Dependency')])
        assert not candidates([dict(job='5167554_4',name='token128_train',partition='cpuonly',state='PENDING',reason='Priority')])
        assert not candidates([dict(job='5167554_2',name='token128_train',partition='cpuonly',state='PENDING',reason='Priority')])
        assert not candidates([dict(job='5167554_3',name='token128_train',partition='cpuonly',state='PENDING',reason='Priority')])
        print('EXACT_CHAIN_PENDING_ONLY_DEV_CAP_PASS');return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'placement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+7*24*3600; errors=0
        while time.monotonic()<deadline:
            try:
                if not tick(): return
                errors=0
            except AssertionError as exc:
                status({'status':'STOPPED_CONTRACT','error':str(exc)});raise
            except Exception as exc:
                errors+=1;event({'event':'TRANSIENT_ERROR','count':errors,'error':str(exc)})
                if errors>=5:
                    status({'status':'STOPPED_ERRORS','error':str(exc)});raise
            time.sleep(30)
        status({'status':'STOPPED_7DAY_LIMIT'})


if __name__=='__main__': main()
