#!/usr/bin/env python3
"""Submit resumable CPU chains; preserve exact batch spool and scheduler receipt."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

RC=Path(__file__).resolve().parents[1]
LAUNCHER=RC/'slurm/rc_m_identification_20260930.sbatch'
def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def main(family):
    root=RC/'results'/('rc_m_conditional_prediction_v1' if family=='held' else 'rc_m_structure_binding_isolation_v1')
    assert (root/'protocol.json').exists()
    with (root/'submit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'submission.json';v=read(path) if path.exists() else dict(status='SUBMITTING',family=family,stages={})
        stages=([('fold0','fold',None,[],'dev_cpuonly,cpuonly'),('fold_rest','fold','1-4%4',[],'cpuonly'),('join','join',None,['fold0','fold_rest'],'cpuonly')]
            if family=='held' else [('pilot','pilot',None,[],'dev_cpuonly,cpuonly'),('work','work','0-15%16',['pilot'],'cpuonly'),('join','join',None,['work'],'cpuonly')])
        for name,stage,array,deps,part in stages:
            if name in v['stages']:continue
            args=['sbatch','--parsable','--partition='+part,'--job-name='+('Mnll_' if family=='held' else 'Miso_')+name]
            if array:args+=['--array='+array]
            if deps:args+=['--dependency=afterok:'+':'.join(v['stages'][d]['job_id'] for d in deps)]
            args+=[str(LAUNCHER),family,stage]
            r=subprocess.run(args,capture_output=True,text=True)
            if r.returncode: v['error']=r.stderr;save(path,v);raise RuntimeError(r.stderr)
            job=r.stdout.strip().split(';')[0];assert job.isdigit()
            v['stages'][name]=dict(job_id=job,stage=stage,array=array,dependencies=[v['stages'][d]['job_id'] for d in deps],args=args);save(path,v)
            print(name,job,flush=True)
        for name,x in v['stages'].items():
            state=subprocess.check_output(['scontrol','show','job',x['job_id'],'-o'],text=True)
            assert 'TimeLimit=00:10:00' in state and 'gres/gpu' not in state
            spool=root/f"submitted_{x['job_id']}.sbatch"
            if not spool.exists():subprocess.run(['scontrol','write','batch_script',x['job_id'],str(spool)],check=True,capture_output=True)
            assert spool.read_bytes()==LAUNCHER.read_bytes()
            x['scheduler']=state.strip();x['submitted_script_sha256']=hashlib.sha256(spool.read_bytes()).hexdigest()
        v['status']='SUBMITTED_AND_SPOOL_VERIFIED';save(path,v)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('family',choices=['held','upstream']);main(a.parse_args().family)
