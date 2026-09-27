#!/usr/bin/env python3
"""Give only this sealed attribution chain spare dev eligibility; submit nothing."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

RC=Path(__file__).resolve().parents[1]
ROOT=RC/'results/rc_m_causal128_attribution_chain_v1'
STATUS=ROOT/'dev_refill.json'
BLOCKED={'Dependency','DependencyNeverSatisfied','JobHeldUser','JobHeldAdmin'}
FIELDS=('NumCPUs','NumNodes','CPUs/Task','ReqTRES','TimeLimit','MinMemoryNode','Requeue')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cmd(*args):return subprocess.run(args,capture_output=True,text=True,timeout=25)
def has_dev(p):return any(x.startswith('dev_') for x in p.split(','))
def parse(s):return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',s))
def put(d):
    d['updated_at']=datetime.now(timezone.utc).isoformat()
    t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,STATUS)
def queue():
    r=cmd('squeue','-r','-h','-u','ap7811','-o','%i|%P|%T|%r')
    if r.returncode:raise RuntimeError(r.stderr)
    return [dict(zip(['job','partition','state','reason'],x.split('|',3))) for x in r.stdout.splitlines() if x]
def whitelist():
    p=read(ROOT/'protocol.json');s=read(ROOT/'submission.json')
    assert s['status']=='ALL_STAGES_SUBMITTED'
    assert s['protocol']['sha256']==sha(ROOT/'protocol.json')
    out={}
    for d in p['stages']:
        job=s['stages'][d['name']]['job_id']
        array=d.get('array')
        if array:
            m=re.fullmatch(r'0-(\d+)%\d+',array);assert m,array
            ids=[job+'_'+str(i) for i in range(int(m.group(1))+1)]
        else:ids=[job]
        for j in ids:out[j]=d
    return out,sha(ROOT/'protocol.json'),sha(ROOT/'submission.json')
def eligible(rows,allowed):
    slots=max(0,4-sum(has_dev(r['partition']) for r in rows))
    by={r['job']:r for r in rows}
    return [j for j in allowed if j in by and by[j]['state']=='PENDING' and by[j]['partition']=='cpuonly'
            and by[j]['reason'] not in BLOCKED][:slots]
def watch():
    with (ROOT/'dev_refill.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allowed,ph,sh=whitelist()
        state=dict(pid=os.getpid(),host=socket.gethostname(),allowed_jobs=list(allowed),migrations=[],
                   protocol_sha256=ph,submission_sha256=sh,poll_seconds=60,new_jobs_submitted=0)
        end=time.monotonic()+72*3600;errors=0
        while time.monotonic()<end:
            try:
                assert sha(ROOT/'protocol.json')==ph and sha(ROOT/'submission.json')==sh,'CHAIN_CHANGED'
                for j in eligible(queue(),allowed):
                    if j not in eligible(queue(),allowed):continue
                    before=cmd('scontrol','show','job',j,'-o')
                    if before.returncode:continue
                    b=parse(before.stdout);d=allowed[j]
                    if b.get('JobState')!='PENDING' or b.get('Partition')!='cpuonly':continue
                    assert b.get('UserId','').startswith('ap7811(')
                    assert b['Comment']=='M_CAUSAL128:'+d['name']
                    assert int(b['NumCPUs'])==d['cpu'] and b['MinMemoryNode']==d['mem'] and b['TimeLimit']=='00:10:00'
                    u=cmd('scontrol','update','JobId='+j,'Partition=dev_cpuonly,cpuonly')
                    if u.returncode:
                        state['last_update_error']={'job':j,'error':u.stderr or u.stdout}
                        if 'QOS' in u.stderr+u.stdout:break
                        raise RuntimeError(u.stderr)
                    after=cmd('scontrol','show','job',j,'-o')
                    if after.returncode:raise RuntimeError('POST_UPDATE_VERIFICATION_UNAVAILABLE: '+j)
                    a=parse(after.stdout)
                    assert all(b.get(k)==a.get(k) for k in FIELDS),(j,b,a)
                    assert set(a['Partition'].split(',')) <= {'dev_cpuonly','cpuonly'}
                    assert 'dev_cpuonly' in a['Partition'].split(',') or a['JobState']!='PENDING'
                    state['migrations'].append(dict(job=j,before=b,after=a,time_utc=datetime.now(timezone.utc).isoformat()))
                rows=queue();state['queue']=[r for r in rows if r['job'] in allowed]
                state['dev_occupancy']=sum(has_dev(r['partition']) for r in rows)
                if not state['queue']:
                    state['status']='ALL_CHAIN_JOBS_TERMINAL';put(state);return
                state['status']='REFILL_ACTIVE';put(state);errors=0
            except AssertionError as error:
                state.update(status='STOPPED_VERIFICATION_FAILURE',error=str(error));put(state);raise
            except Exception as error:
                errors+=1;state.update(status='SCHEDULER_RETRY',error=repr(error),consecutive_errors=errors);put(state)
                if errors>=5:raise
            time.sleep(60)
        state['status']='STOPPED_72H_LIMIT';put(state)

def test():
    allowed=['1','2','3','4','5']
    rows=[dict(job=j,partition='cpuonly',state='PENDING',reason='Priority') for j in allowed]
    assert eligible(rows,allowed)==allowed[:4]
    rows[0]['reason']='Dependency';rows[1]['state']='RUNNING'
    assert eligible(rows,allowed)==allowed[2:]
    rows+=[dict(job='other',partition='dev_cpuonly',state='RUNNING',reason='None')]*4
    assert eligible(rows,allowed)==[]
    assert has_dev('dev_accelerated,accelerated')
    print('CAUSAL128_SCOPED_DEV_REFILL_TEST_PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['watch','test']);a=p.parse_args()
    test() if a.action=='test' else watch()
