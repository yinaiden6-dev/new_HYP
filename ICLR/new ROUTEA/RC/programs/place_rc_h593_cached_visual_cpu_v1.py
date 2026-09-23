#!/usr/bin/env python3
"""One dev-CPU backfill slot for the recorded visual replay and join jobs."""
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_cached_visual_trend_v1'

def run(args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=40)
    assert p.returncode==0,(args,p.stderr)
    return p.stdout

def info(job):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',run(['scontrol','show','job','-o',job])))

def main():
    lock=(OUT/'placement.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=json.loads((OUT/'submission.json').read_text())
    owned={receipt['remaining']['job']+'_'+str(i) for i in range(42,55)}|{receipt['join']['job']}
    deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline and not (OUT/'validation.json').exists():
        text=run(['squeue','-r','-h','-u','ap7811','-o','%i|%j|%P|%T|%r'])
        rows=[dict(zip(('job','name','partition','state','reason'),x.split('|'))) for x in text.splitlines() if x]
        active=[r for r in rows if r['job'] in owned]
        if not active:return
        dev=[r for r in rows if r['partition'].startswith('dev_')]
        if len(dev)<4 and not any(r['partition']=='dev_cpuonly' for r in dev):
            for r in active:
                if r['state']!='PENDING' or r['partition']!='cpuonly' or r['reason'] in ('Dependency','DependencyNeverSatisfied','JobHeldUser','JobHeldAdmin'):continue
                b=info(r['job']);assert b['JobName']=='h55_vtrend' and b['UserId'].startswith('ap7811(') and 'gres/gpu' not in b['ReqTRES']
                if b['JobState']!='PENDING':continue
                run(['scontrol','update','JobId='+r['job'],'Partition=dev_cpuonly'])
                a=info(r['job']);assert a['Partition']=='dev_cpuonly'
                for k in ('ReqTRES','CPUs/Task','TimeLimit','Dependency'):assert b.get(k)==a.get(k)
                with (OUT/'placement_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),job=r['job'],before=b,after=a))+'\n')
                break
        p=OUT/'placement_status.json';t=p.with_name('.placement.tmp')
        t.write_text(json.dumps(dict(host=socket.gethostname(),pid=os.getpid(),time=time.time(),active=active,status='CPU_ONLY_BACKFILL'))+'\n');os.replace(t,p)
        time.sleep(30)

if __name__=='__main__':main()
