#!/usr/bin/env python3
"""Keep coordinate GPU acquisitions; submit all downstream readouts CPU-only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import torch
import run_rc_h593_shared_pooling_v1 as P

ROOT=P.ROOT
OUT=ROOT/'results/rc_h593_coordinate_cpu_chain_v1'
AUTH=ROOT/'registry/rc_h593_coordinate_cpu_chain_authority_v1_20260923.json'
CONTROL=ROOT/'slurm/rc_h593_coordinate_cpu_chain_v1.sbatch'
EVAL=ROOT/'slurm/rc_h593_roma_coordinate_eval_cpu_v1.sbatch'
read,write,bind,checked=P.read,P.write,P.bind,P.checked

def prepare():
    assert not AUTH.exists()
    old=ROOT/'slurm/rc_h593_roma_coordinate_eval_v2.sbatch'
    # Submission-only changes. Executed scientific shell body is identical.
    assert EVAL.read_text().split('set -euo pipefail',1)[1]==old.read_text().split('set -euo pipefail',1)[1]
    assert '#SBATCH --gres=' not in EVAL.read_text() and '#SBATCH --partition=cpuonly' in EVAL.read_text()
    write(AUTH,dict(status='COORDINATE_DATA_FIRST_CPU_READOUT_AUTHORIZED',
       sources=[bind(p) for p in (Path(__file__),CONTROL,EVAL,old,Path(P.__file__),P.AUTH,
         ROOT/'registry/rc_h593_roma_coordinate_eval_authority_v2_20260922.json')],
       scope='Original GPU acquisitions and candidate/seal rules; CPU-only fit, diagnose, join and publish',
       user_instruction='Complete required GPU intermediate data once, then CPU consumers',science_unchanged=True))

def submit(stage,dependency=None,array=None):
    dest=OUT/(stage+'_submitted.json')
    if dest.exists():return read(dest)['job']
    args=['sbatch','--parsable','--hold']
    if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    if array:args+=['--array='+array]
    args += [str(EVAL),stage]
    job=P.cmd(args).strip().split(';')[0];assert job.isdigit()
    spool=OUT/(job+'_spool.sh');spool.parent.mkdir(parents=True,exist_ok=True)
    P.cmd(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==EVAL.read_bytes()
    raw=P.cmd(['scontrol','show','job','-o',job]);assert 'Partition=cpuonly' in raw and 'gres/gpu' not in raw
    write(dest,dict(job=job,args=args,authority=bind(AUTH),spool=bind(spool),job_info=raw))
    P.cmd(['scontrol','release',job]);return job

def control(previous):
    assert os.environ.get('SLURM_JOB_ID')
    a=read(AUTH)
    for b in a['sources']:checked(b)
    pa=P.authority()
    if previous:
        lines=P.cmd(['sacct','-X','-n','-P','-j',previous,'--format=JobID,State,ExitCode']).splitlines()
        assert lines and all(x.split('|')[1:3]==['COMPLETED','0:0'] for x in lines if x.strip()),lines
    complete=0
    for i in range(593):
        p=P.C.OUT/f'query{i:03d}'/'validation.json'
        if not p.exists():continue
        v=read(p);assert v['status']=='ROMA_COORDINATE_QUERY_PASS' and v['authority']==bind(P.C.AUTH)
        checked(v['payload']);complete+=1
    if complete<593:
        # Original controller still deduplicates all complete queries, preserves
        # chunks, exact pooling and the original GPU launcher, and caps attempts.
        P.CPU=CONTROL
        return P.control(pa,'coordinate',None)
    fits=submit('fit',array='0-4%5');diagnostics=submit('diagnose')
    joined=submit('join',dependency=fits+':'+diagnostics)
    write(OUT/'all593_cpu_readout_submitted.json',dict(authority=bind(AUTH),validated=593,
          fit=fits,diagnose=diagnostics,join=joined,GPU_requested=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','control'));ap.add_argument('family',nargs='?',default='coordinate');ap.add_argument('previous',nargs='?');a=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    prepare() if a.stage=='prepare' else control(a.previous)
