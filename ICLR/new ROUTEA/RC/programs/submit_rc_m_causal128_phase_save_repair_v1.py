#!/usr/bin/env python3
"""Resume only the failed pilot-save branch; preserve all sealed scientific code."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

RC = Path(__file__).resolve().parents[1]
BASE = RC/'results/rc_m_causal128_attribution_chain_v1'
OUT = BASE/'phase_save_repair_v1'
PYTHON = RC.parents[2]/'.venv-colpali/bin/python'
PARENT = RC/'programs/submit_rc_m_causal128_attribution_chain_v1.py'
RECOVER = RC/'programs/repair_rc_m_phase_newgroups_pilot_save_v1.py'
SCRIPT = RC/'slurm/rc_m_causal128_phase_save_repair_v1.sbatch'

def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return {'path':str(p),'sha256':h.hexdigest()}
def put(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def stages():
    return [dict(name='pilot_repair',deps=[],mem='32G',restarts=4),
            dict(name='phase_replication',deps=['pilot_repair'],array='0-8%9',mem='32G',restarts=8),
            dict(name='phase_join',deps=['phase_replication'],mem='16G',restarts=8),
            dict(name='final_summary',deps=['phase_join'],mem='16G',restarts=4),
            dict(name='health',deps=['pilot_repair','phase_replication','phase_join','final_summary'],dependency='afterany',mem='8G',restarts=2)]
def verify():
    p=read(OUT/'protocol.json')
    for b in p['sources']: assert bind(b['path'])==b,('REPAIR_SOURCE_CHANGED',b['path'])
    return p
def prepare():
    # Recovery preparation must already have sealed the saved-forward inputs.
    recovery_manifest=BASE/'phase_replication/pilot_save_repair_manifest.json'
    assert recovery_manifest.is_file(), recovery_manifest
    source=[Path(__file__),SCRIPT,PARENT,RECOVER,BASE/'protocol.json',BASE/'submission.json',
            BASE/'phase_replication/protocol.json',recovery_manifest]
    for rel in ['paths/cache_validation.json','paths/validation.json','bridge/validation.json',
                'bridge/joint_pairs/validation.json','pair_structure/validation.json']:
        f=BASE/rel;assert f.is_file(),f;source.append(f)
    p=dict(schema='CAUSAL128_PILOT_SAVE_REPAIR_CHAIN_V1',sources=[bind(x) for x in source],
           stages=stages(),original_failed_job='5167604',original_cancelled_jobs=['5167605','5167606','5167607'],
           completed_science_reused=True,scientific_protocol_changed=False,compute_device='CPU',
           scheduling={'partition':'accelerated','gpu_per_task':1,'cpus':4,'time_limit':'00:10:00'})
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==p,'REPAIR_PROTOCOL_ALREADY_FROZEN'
    else:put(OUT/'protocol.json',p)
    print('CAUSAL128_SAVE_REPAIR_CHAIN_PREPARED')
def submit():
    p=verify();path=OUT/'submission.json'
    with (OUT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(path) if path.exists() else dict(protocol=bind(OUT/'protocol.json'),stages={})
        assert state['protocol']==bind(OUT/'protocol.json')
        for d in p['stages']:
            name=d['name'];prior=state['stages'].get(name)
            if prior:
                assert prior.get('job_id'),'UNCERTAIN_SUBMISSION_REQUIRES_RECONCILIATION';continue
            args=['sbatch','--parsable','--kill-on-invalid-dep=yes','--mem='+d['mem'],
                  '--comment=M_CAUSAL128_SAVE_REPAIR:'+name]
            if d.get('array'):args+=['--array='+d['array']]
            if d['deps']:
                args+=['--dependency='+d.get('dependency','afterok')+':'+':'.join(state['stages'][x]['job_id'] for x in d['deps'])]
            args+=[str(SCRIPT),name,str(d['restarts'])]
            state['stages'][name]=dict(status='SUBMITTING',argv=args);put(path,state)
            result=subprocess.run(args,capture_output=True,text=True,timeout=40)
            if result.returncode:
                state['stages'][name].update(status='SUBMISSION_REJECTED',error=result.stderr);put(path,state)
                raise RuntimeError(result.stderr)
            match=re.fullmatch(r'(\d+)(?:;[^\s]+)?',result.stdout.strip());assert match,'UNCERTAIN_SBATCH_RESPONSE'
            state['stages'][name].update(status='SUBMITTED',job_id=match.group(1));put(path,state)
            print(json.dumps({'stage':name,'job_id':match.group(1)}),flush=True)
        state['status']='ALL_REPAIR_STAGES_SUBMITTED';put(path,state)
def run(name):
    verify()
    if name=='pilot_repair':return subprocess.call([str(PYTHON),str(RECOVER),'recover'])
    if name in ('phase_replication','phase_join','final_summary'):
        return subprocess.call([str(PYTHON),str(PARENT),'run',name])
    assert name=='health',name
    ids=[v['job_id'] for v in read(OUT/'submission.json')['stages'].values()]
    q=subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,Partition,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=30)
    receipts={}
    for rel in ['phase_replication/pilot_validation.json','phase_replication/validation.json','final_summary/validation.json']:
        p=BASE/rel;receipts[rel]={'present':p.exists()}
        if p.exists():receipts[rel].update(status=read(p).get('status'),source=bind(p))
    put(OUT/'health.json',dict(status='CAUSAL128_SAVE_REPAIR_HEALTH_SNAPSHOT',time_utc=datetime.now(timezone.utc).isoformat(),
                              accounting=q.stdout.splitlines(),accounting_error=q.stderr,validations=receipts))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','run']);p.add_argument('stage',nargs='?');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit()
    else:sys.exit(run(a.stage))
