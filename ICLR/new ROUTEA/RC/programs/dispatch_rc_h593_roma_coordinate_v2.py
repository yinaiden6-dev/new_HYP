#!/usr/bin/env python3
"""Bounded, afterok-only waves. Never launch more than 46 coordinate tasks per wave."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_rc_h593_roma_coordinate_eval_v2 as E
C=E.C;ROOT=E.ROOT
AUTH=ROOT/'registry/rc_h593_roma_coordinate_dispatch_authority_v2_20260922.json'
OUT=ROOT/'results/rc_h593_roma_coordinate_dispatch_v2'
LAUNCH=ROOT/'slurm/rc_h593_roma_coordinate_dispatch_v2.sbatch'


def next_indices(valid):return [i for i in range(1,593) if i not in valid][:46]


def prepare():
    assert next_indices({0})==list(range(1,47)) and next_indices(set(range(590)))==[590,591,592] and not next_indices(set(range(593)))
    E.write(AUTH,dict(status='COORDINATE_DISPATCH_AUTHORIZED',coordinate=E.bind(C.AUTH),evaluation=E.bind(E.AUTH),
        sources=[E.bind(__file__),E.bind(LAUNCH),E.bind(ROOT/'slurm/rc_h593_roma_coordinate_precision_v2.sbatch'),E.bind(E.LAUNCH)],
        total_queries=593,max_wave_tasks=46,dependency='afterok only; prerequisite failure cancels next task',automatic_retries=0))
    print('DISPATCH_WAVES_PREFLIGHT_PASS',flush=True)


def submit(args):
    p=subprocess.run(['/usr/bin/sbatch','--parsable',*args],cwd=ROOT,capture_output=True,text=True)
    E.need(p.returncode==0,'SBATCH:'+p.stderr);job=p.stdout.strip().split(';')[0];E.need(job.isdigit(),'SBATCH_ID');return job


def run():
    E.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');a=E.read(AUTH)
    for b in [a['coordinate'],a['evaluation'],*a['sources']]:E.checked(b)
    valid=set()
    for i in range(593):
        p=C.OUT/f'query{i:03d}/validation.json'
        if not p.exists():continue
        v=E.read(p);E.need(v['status']=='ROMA_COORDINATE_QUERY_PASS' and v['authority']==a['coordinate'],'VALIDATION_AUTHORITY');E.checked(v['payload']);valid.add(i)
    E.need(0 in valid,'PILOT_REQUIRED');todo=next_indices(valid);record=OUT/f"after{len(valid):03d}.json"
    E.need(not record.exists(),'ALREADY_DISPATCHED')
    if todo:
        job=submit(['--array='+','.join(map(str,todo))+'%46',str(ROOT/'slurm/rc_h593_roma_coordinate_precision_v2.sbatch')])
        E.write(record.with_name(record.stem+'_workers.json'),dict(authority=E.bind(AUTH),job_id=job,indices=todo))
        callback=submit(['--dependency=afterok:'+job,'--kill-on-invalid-dep=yes',str(LAUNCH)])
        value=dict(authority=E.bind(AUTH),validated_queries=len(valid),worker_job=job,indices=todo,next_dispatch_job=callback)
    else:
        fits=submit(['--array=0-4%5',str(E.LAUNCH),'fit']);E.write(OUT/'fit_submitted.json',dict(authority=E.bind(AUTH),job_id=fits))
        diagnostics=submit([str(E.LAUNCH),'diagnose']);E.write(OUT/'diagnostics_submitted.json',dict(authority=E.bind(AUTH),job_id=diagnostics))
        joined=submit(['--dependency=afterok:'+fits+':'+diagnostics,'--kill-on-invalid-dep=yes',str(E.LAUNCH),'join'])
        value=dict(authority=E.bind(AUTH),validated_queries=593,fit_job=fits,diagnostics_job=diagnostics,join_job=joined)
    E.write(record,value);print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));args=p.parse_args()
    prepare() if args.stage=='prepare' else run()
