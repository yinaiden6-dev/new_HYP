#!/usr/bin/env python3
"""Give one authorized probe the next slot, restoring only our own holds."""
import fcntl
import importlib.util
import os
from pathlib import Path
import socket
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scheduler_io',ROOT/'programs/run_rc_h593_packed_gpu_v2.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
OUT=ROOT/'results/rc_h593_pair_quality_priority_5159145'
TARGET='5159145'
NAMES={'h593_ftrain','h593_collect','h593_uqual'}
FIELDS=('ReqTRES','TimeLimit','CPUs/Task','Dependency','TresPerNode','Partition')

def status(s,**kwargs):
    P.write(OUT/'status.json',dict(status=s,time_utc=P.now(),pid=os.getpid(),host=socket.gethostname(),**kwargs),replace=True)

def restore(reason):
    for p in sorted((OUT/'holds').glob('*.json')):
        dest=OUT/'restored'/p.name
        if dest.exists():continue
        saved=P.read(p);job=saved['job'];res=P.command(['scontrol','show','job','-o',job],check=False)
        after=None
        if res.returncode==0:
            info=P.fields(res.stdout)
            if info.get('JobState')=='PENDING' and info.get('Reason')=='JobHeldUser':
                P.command(['scontrol','release',job]);after=P.jobinfo(job)
                assert after.get('Reason')!='JobHeldUser'
                for k in FIELDS:assert after.get(k)==saved['before'].get(k),(job,k)
        P.write(dest,dict(job=job,reason=reason,after=after,time_utc=P.now()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+6*3600
    try:
        while time.monotonic()<deadline:
            info=P.jobinfo(TARGET)
            assert info.get('UserId','').startswith('ap7811(') and info.get('JobName')=='h593_pairq'
            if info['JobState']!='PENDING':
                restore('TARGET_'+info['JobState']);status('RESTORED_AFTER_TARGET_STATE',target=info);return
            if info.get('Reason') in ('JobHeldUser','JobHeldAdmin','DependencyNeverSatisfied'):
                restore('TARGET_NOT_ELIGIBLE');status('STOPPED_TARGET_NOT_ELIGIBLE',target=info);return
            rows=P.queue()
            for row in rows:
                if row['name'] not in NAMES or row['state']!='PENDING' or 'accelerated' not in row['partition']:continue
                if row['reason'] in ('JobHeldUser','JobHeldAdmin','Dependency','DependencyNeverSatisfied'):continue
                job=row['job'];before=P.jobinfo(job)
                if before.get('JobState')!='PENDING' or before.get('Reason') in ('JobHeldUser','JobHeldAdmin'):continue
                assert before.get('UserId','').startswith('ap7811(') and before['JobName'] in NAMES
                assert before.get('Account')=='hk-project-p0025545'
                record=OUT/'holds'/f'{job}.json'
                if not record.exists():P.write(record,dict(job=job,before=before))
                res=P.command(['scontrol','hold',job],check=False)
                after=P.jobinfo(job)
                if after.get('JobState')!='PENDING':continue
                assert res.returncode==0 and after.get('Reason')=='JobHeldUser',(job,res.stderr,after)
                for k in FIELDS:assert after.get(k)==before.get(k),(job,k)
            status('WAITING_FOR_PRIORITY_PROBE',target=P.jobinfo(TARGET),temporary_holds=len(list((OUT/'holds').glob('*.json'))))
            time.sleep(15)
        restore('SIX_HOUR_LIMIT');status('RESTORED_TIME_LIMIT')
    except BaseException as e:
        restore('SUPERVISOR_ERROR');status('RESTORED_ERROR',error=repr(e));raise

if __name__=='__main__':main()
