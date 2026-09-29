#!/usr/bin/env python3
"""Attach the CPU upstream-registration extension to existing collection."""
import json
import os
from pathlib import Path
import subprocess
import run_rc_m_upstream_event_closure_v1 as U

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def main():
    ex=U.read(U.E.OUT/'submission.json');assert ex['status']=='SUBMITTED_AND_VERIFIED'
    launcher=U.RC/'slurm/rc_m_upstream_event_closure_v1.sbatch'
    source=[Path(__file__),Path(U.__file__),launcher,U.E.OUT/'protocol.json',U.RC/'plan/RC_M_UPSTREAM_IDENTITY_CLOSURE_20260929.md']
    U.write(U.OUT/'launch_contract.json',dict(sources=[U.bind(p) for p in source],stage_scope='CPU-only completion of registration for new8; all19 analyzed with fullC128, same-world source validation.'))
    path=U.OUT/'submission.json';record=U.read(path) if path.exists() else dict(status='SUBMITTING',stages={})
    for name,count,dependencies in [('prepare',1,[ex['stages']['collect']['job_id']]),('context',8,['prepare']),('masses',1,['context']),('bridge',8,['masses']),('analyze',1,['bridge',ex['stages']['join']['job_id'],'5168069'])]:
        if name in record['stages']:continue
        ids=[record['stages'][x]['job_id'] if x in record['stages'] else x for x in dependencies]
        args=['sbatch','--parsable','--job-name=Mup19_'+name,'--dependency=afterok:'+':'.join(ids)]
        if count>1:args+=['--array=0-7%8']
        args+=[str(launcher),name]
        x=subprocess.run(args,capture_output=True,text=True)
        if x.returncode:record['error']=x.stderr;save(path,record);raise RuntimeError(x.stderr)
        job=x.stdout.strip().split(';')[0];assert job.isdigit()
        record['stages'][name]=dict(job_id=job,count=count,dependencies=ids,args=args);save(path,record);print(name,job,flush=True)
    checks=[]
    for name,s in record['stages'].items():
        txt=subprocess.check_output(['scontrol','show','job',s['job_id'],'-o'],text=True)
        assert 'TimeLimit=00:10:00' in txt and 'gres/gpu' not in txt
        for d in s['dependencies']:assert d in txt
        spool=U.OUT/('submitted_'+s['job_id']+'.sbatch')
        subprocess.run(['scontrol','write','batch_script',s['job_id'],str(spool)],check=True,capture_output=True)
        assert spool.read_bytes()==launcher.read_bytes()
        checks.append(dict(stage=name,job_id=s['job_id'],scheduler=txt.strip(),spool_exact=True))
    record['status']='SUBMITTED_AND_VERIFIED';save(path,record)
    U.write(U.OUT/'submission_validation.json',dict(status='SUBMITTED_AND_VERIFIED',checks=checks))

if __name__=='__main__':main()
