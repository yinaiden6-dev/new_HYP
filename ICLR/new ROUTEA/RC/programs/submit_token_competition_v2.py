#!/usr/bin/env python3
"""Idempotent submission of the authorized five-stage native-token chain."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

RC=Path(__file__).resolve().parents[1]
OUT=RC/'results/rc_token_competition_f128_v2'
SCRIPT=RC/'slurm/token_competition_v2.sbatch'


def bind(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def check(binding):
    if bind(binding['path'])['sha256']!=binding['sha256']:raise ValueError('Source drift: '+binding['path'])


def stages():
    return [('pilot',None,None,'cpuonly,dev_cpuonly'),
            ('evidence','pilot','0-15%16','cpuonly'),
            ('verify','evidence',None,'cpuonly,dev_cpuonly'),
            ('train','verify','0-14%15','cpuonly'),
            ('join','train',None,'cpuonly,dev_cpuonly')]


def command(name,predecessor,array,partition):
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--partition='+partition,
         '--job-name=token128_'+name,'--comment=TOKEN_COMPETITION_F128_V2:'+name]
    if predecessor:cmd+=['--dependency=afterok:'+predecessor]
    if array:cmd+=['--array='+array]
    return cmd+[str(SCRIPT),name]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    p=read(OUT/'protocol.json')
    prep=read(OUT/'preparation_validation.json')
    if prep['status']!='TOKEN_COMPETITION_PREPARATION_PASS' or prep['protocol']!=bind(OUT/'protocol.json'):
        raise ValueError('Frozen preparation missing')
    for b in p['code_sources'].values():check(b)
    module=read(OUT/'module_engineering.json')
    if not module.get('status','').endswith('PASS'):raise ValueError('Model engineering not passed')
    for source,sha in module['source_sha256'].items():
        check({'path':source,'sha256':sha})
    launch=read(OUT/'launcher_engineering.json')
    if launch['status']!='TOKEN_COMPETITION_LAUNCHER_PASS' or launch['source']!=bind(SCRIPT):
        raise ValueError('Actual launcher branch validation missing')
    check(launch['checker'])
    source_checks=read(OUT/'source_engineering.json')
    if source_checks['status']!='TOKEN_COMPETITION_SOURCES_COMPILE_PASS' or source_checks['protocol']!=bind(OUT/'protocol.json'):
        raise ValueError('Source compilation/binding validation missing')
    sb=OUT/'submission.json'
    with (OUT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(sb) if sb.exists() else {'status':'PREPARED','protocol':bind(OUT/'protocol.json'),
           'script':bind(SCRIPT),'stages':{},'population':128,'candidate_count':128,
           'baseline_fits_reused':5,'new_reader_fits':15,'new_backbone_forwards':0,'gpus':0}
        if state['protocol']!=bind(OUT/'protocol.json') or state['script']!=bind(SCRIPT):
            raise ValueError('Existing submission differs; will not duplicate')
        jobs={}
        for name,dep,array,partition in stages():
            if name in state['stages']:
                jobs[name]=state['stages'][name]['job_id'];continue
            cmd=command(name,jobs[dep] if dep else None,array,partition)
            if a.dry_run:
                print(json.dumps({'stage':name,'command':cmd}),flush=True);jobs[name]='DRY_'+name;continue
            result=subprocess.run(cmd,capture_output=True,text=True)
            if result.returncode:
                state['status']='PARTIAL_SUBMISSION';state['last_error']={'stage':name,'stderr':result.stderr}
                write(sb,state);raise RuntimeError(result.stderr)
            job=result.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job):raise ValueError('Unexpected sbatch response: '+result.stdout)
            jobs[name]=job
            state['stages'][name]={'job_id':job,'predecessor':jobs[dep] if dep else None,
                'array':array,'partition':partition,'command':cmd,
                'submitted_utc':datetime.now(timezone.utc).isoformat()}
            state['status']='SUBMITTING';write(sb,state)
            print(json.dumps({'stage':name,'job_id':job,'afterok':jobs[dep] if dep else None}),flush=True)
        if not a.dry_run:
            state['status']='COMPLETE_CHAIN_SUBMITTED_WAITING_ENGINEERING_PILOT';write(sb,state)


if __name__=='__main__':main()
