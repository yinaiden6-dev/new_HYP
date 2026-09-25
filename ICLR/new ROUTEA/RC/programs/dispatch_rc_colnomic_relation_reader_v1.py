#!/usr/bin/env python3
"""Idempotent short CPU jobs for the authorized cache-only relation pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_colnomic_relation_reader_v1 as W


def submit(name,role,dependencies=(),array=None):
    dest=W.OUT/'dispatch'/f'{name}.json'
    if dest.exists():
        record=W.read(dest);W.need(record['authority']==W.bind(W.AUTH),'DISPATCH_AUTHORITY')
        return record['job_id']
    cmd=['sbatch','--parsable','--job-name=cnrel_'+name]
    # Do not fill dev submission quota with dependent work: capacity/fit worker
    # self-dispatches the next stage only after producing its qualification.
    cmd+=['--partition=cpuonly' if role=='join' else '--partition=dev_cpuonly,cpuonly']
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(dependencies),'--kill-on-invalid-dep=yes']
    if array is not None:cmd+=['--array='+array]
    cmd+=[str(ROOT/'slurm/rc_colnomic_relation_reader_v1.sbatch'),role]
    proc=subprocess.run(cmd,text=True,capture_output=True)
    if proc.returncode:
        W.write(W.OUT/'dispatch'/f'{name}.failure.json',dict(command=cmd,stdout=proc.stdout,stderr=proc.stderr),mutable=True)
        raise RuntimeError(proc.stderr)
    job=proc.stdout.strip().split(';')[0];W.need(job.isdigit(),'JOB_ID_PARSE')
    W.write(dest,dict(authority=W.bind(W.AUTH),job_id=job,command=cmd,stage=role))
    W.emit('SUBMITTED',name=name,job_id=job,role=role)
    return job


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','after_capacity','after_fit']);args=p.parse_args()
    W.load_contract()
    directory=W.OUT/'dispatch';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.action=='start':
            submit('capacity','capacity');return
        if args.action=='after_capacity':
            W.need(W.read(W.OUT/'capacity_validation.json')['status']=='FULL_C128_CPU_CAPACITY_PASS','CAPACITY_REQUIRED')
            # Capacity occupies one of four dev submission slots until it exits.
            # Three small jobs run first. The BASE arm is inexpensive and runs
            # inside the capacity job's remaining budget before this dispatch.
            for idx,arm in enumerate(W.C.ARMS):
                if arm=='BASE':continue
                submit(arm.lower(),'fit',array=str(idx))
            return
        complete=all((W.OUT/arm/'validation.json').exists() for arm in W.C.ARMS)
        if complete:submit('join','join')
        else:W.emit('WAITING_FOR_OTHER_ARMS',sealed=[a for a in W.C.ARMS if (W.OUT/a/'validation.json').exists()])


if __name__=='__main__':
    main()
