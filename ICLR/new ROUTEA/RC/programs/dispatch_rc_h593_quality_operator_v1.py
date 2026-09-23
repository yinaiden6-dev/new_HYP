#!/usr/bin/env python3
"""Release CPU operator waves only after qualified predecessors, then evaluate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_rc_h593_quality_operator_eval_v1 as E
C=E.C;ROOT=E.ROOT
AUTH=ROOT/'registry/rc_h593_quality_operator_dispatch_authority_v1_20260922.json'
OUT=ROOT/'results/rc_h593_quality_operator_dispatch_v1'
LAUNCH=ROOT/'slurm/rc_h593_quality_operator_dispatch_v1.sbatch'


def next_indices(valid):return [i for i in range(1,75) if i not in valid][:46]


def prepare():
    E.need(not AUTH.exists(),'NEW_AUTHORITY')
    E.need(next_indices({0})==list(range(1,47)),'FIRST_WAVE')
    E.need(next_indices(set(range(47)))==list(range(47,75)),'SECOND_WAVE')
    E.need(not next_indices(set(range(75))),'COMPLETE')
    E.write(AUTH,dict(status='QUALITY_OPERATOR_DISPATCH_AUTHORIZED',operator=E.bind(C.AUTH),evaluation=E.bind(E.AUTH),
        sources=[E.bind(__file__),E.bind(LAUNCH),E.bind(C.LAUNCH),E.bind(E.LAUNCH)],
        total_shards=75,total_queries=593,max_wave_tasks=46,partition='cpuonly',gpu_count=0,
        dependency='afterok plus qualified shard seals; failed predecessor cancels callback',automatic_retries=0))
    print(dict(status='QUALITY_OPERATOR_DISPATCH_PREFLIGHT_PASS',authority=E.bind(AUTH)),flush=True)


def submit_once(path,args,value):
    if path.exists():
        previous=E.read(path)
        E.need(previous['authority']==E.bind(AUTH) and previous['arguments']==args,'SUBMISSION_RECEIPT')
        return previous['job_id']
    p=subprocess.run(['/usr/bin/sbatch','--parsable',*args],cwd=ROOT,capture_output=True,text=True)
    E.need(p.returncode==0,'SBATCH:'+p.stderr);job=p.stdout.strip().split(';')[0];E.need(job.isdigit(),'SBATCH_ID')
    E.write(path,dict(authority=E.bind(AUTH),job_id=job,arguments=args,**value));return job


def run():
    E.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');a=E.read(AUTH)
    E.need(a['status']=='QUALITY_OPERATOR_DISPATCH_AUTHORIZED','AUTHORITY')
    for b in [a['operator'],a['evaluation'],*a['sources']]:E.checked(b)
    valid=set();count=0
    for i in range(75):
        path=C.OUT/f'shard{i:02d}/validation.json'
        if not path.exists():continue
        v=E.read(path);E.need(v['status']=='QUALITY_OPERATOR_SHARD_PASS' and v['authority']==a['operator'],'SHARD_QUALIFIED')
        E.need(v['query_count']==len(v['queries']),'SHARD_COUNT')
        for seal in v['queries']:
            q=E.read(E.checked(seal));E.need(q['status']=='QUALITY_OPERATOR_QUERY_PASS' and q['authority']==a['operator'],'QUERY_QUALIFIED');E.checked(q['payload'])
        count+=v['query_count'];valid.add(i)
    E.need(0 in valid,'PILOT_REQUIRED');todo=next_indices(valid);record=OUT/f'after{len(valid):02d}.json'
    if record.exists():
        E.need(E.read(record)['authority']==E.bind(AUTH),'RECORDED_AUTHORITY');print(E.read(record),flush=True);return
    if todo:
        workers=submit_once(record.with_name(record.stem+'_workers.json'),['--array='+','.join(map(str,todo))+'%46',str(C.LAUNCH)],dict(shards=todo))
        callback=submit_once(record.with_name(record.stem+'_callback.json'),['--dependency=afterok:'+workers,'--kill-on-invalid-dep=yes',str(LAUNCH)],{})
        value=dict(validated_shards=len(valid),validated_queries=count,worker_job=workers,shards=todo,next_dispatch_job=callback)
    else:
        E.need(count==593,'ALL593')
        fits=submit_once(OUT/'fit_submitted.json',['--array=0-4%5',str(E.LAUNCH),'fit'],{})
        diagnostics=submit_once(OUT/'diagnostics_submitted.json',[str(E.LAUNCH),'diagnose'],{})
        joined=submit_once(OUT/'join_submitted.json',['--dependency=afterok:'+fits+':'+diagnostics,'--kill-on-invalid-dep=yes',str(E.LAUNCH),'join'],{})
        value=dict(validated_shards=len(valid),validated_queries=count,fit_job=fits,diagnostics_job=diagnostics,join_job=joined)
    E.write(record,dict(authority=E.bind(AUTH),**value));print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));args=p.parse_args()
    prepare() if args.stage=='prepare' else run()
