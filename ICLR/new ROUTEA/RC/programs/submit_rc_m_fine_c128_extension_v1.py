#!/usr/bin/env python3
"""Submit bounded collection and frozen downstream replay, without monitoring."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import run_rc_m_fine_c128_extension_v1 as E

STAGES=[('pilot',1,[],True),('collect',8,['pilot'],True),('replay',8,['collect'],False),('join',1,['replay'],False)]

def save(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)

def main():
    p,pb=E.guard();path=E.OUT/'submission.json'
    record=E.read(path) if path.exists() else dict(protocol=pb,status='SUBMITTING',utc=datetime.now(timezone.utc).isoformat(),stages={})
    assert record['protocol']==pb
    launcher=E.RC/'slurm/rc_m_fine_c128_extension_v1.sbatch'
    for name,count,deps,gpu in STAGES:
        if name in record['stages']:continue
        partition='dev_accelerated,accelerated' if name=='pilot' else ('accelerated' if gpu else ('dev_cpuonly,cpuonly' if name=='join' else 'cpuonly'))
        args=['sbatch','--parsable','--job-name=M19_'+name,'--partition='+partition,'--cpus-per-task='+('8' if gpu else '4'),'--mem='+('64G' if gpu else '32G')]
        if gpu:args+=['--gres=gpu:1']
        if count>1:args+=['--array=0-7%8']
        ids=[record['stages'][d]['job_id'] for d in deps]
        if ids:args+=['--dependency=afterok:'+':'.join(ids)]
        args+=[str(launcher),name]
        x=subprocess.run(args,text=True,capture_output=True)
        if x.returncode:
            record['last_error']=dict(stage=name,stderr=x.stderr,args=args);save(path,record);raise RuntimeError(x.stderr)
        job=x.stdout.strip().split(';')[0];assert job.isdigit()
        record['stages'][name]=dict(job_id=job,dependencies=ids,count=count,partition=partition,gpu=gpu,args=args)
        save(path,record);print(name,job,flush=True)
    checks=[]
    for name,s in record['stages'].items():
        job=s['job_id'];txt=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'TimeLimit=00:10:00' in txt
        for dep in s['dependencies']:assert dep in txt
        if s['gpu']:assert 'gres/gpu=1' in txt or 'gres/gpu:' in txt
        dst=E.OUT/('submitted_'+job+'.sbatch')
        subprocess.run(['scontrol','write','batch_script',job,str(dst)],check=True,capture_output=True)
        assert dst.read_bytes()==launcher.read_bytes()
        checks.append(dict(stage=name,job_id=job,scheduler=txt.strip(),spool_exact=True))
    record['status']='SUBMITTED_AND_VERIFIED';save(path,record)
    E.write(E.OUT/'submission_validation.json',dict(protocol=pb,status='SUBMITTED_AND_VERIFIED',checks=checks))

if __name__=='__main__':main()
