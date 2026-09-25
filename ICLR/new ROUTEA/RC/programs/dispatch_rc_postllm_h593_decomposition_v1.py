#!/usr/bin/env python3
"""Bounded CPU-only dispatch for frozen H593 POST attribution."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from analyze_rc_postllm_signal_decomposition_v2 import bind, checked, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_postllm_h593_decomposition_v1'
SBATCH = ROOT/'slurm/rc_postllm_h593_decomposition_v1.sbatch'
PY = '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
KERNEL = ROOT/'programs/analyze_rc_postllm_h593_decomposition_v1.py'
AUDIT = ROOT/'programs/audit_rc_postllm_h593_decomposition_v1.py'


def submit(role, *, dependency=None, array=None, partition='cpuonly'):
    command = ['sbatch','--parsable',f'--job-name=post_attr_{role}',f'--partition={partition}']
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
        command.append('--kill-on-invalid-dep=yes')
    if array:
        command.append(f'--array={array}')
    command.extend([str(SBATCH),role])
    result = subprocess.run(command,text=True,capture_output=True,check=True)
    job = result.stdout.strip().split(';')[0]
    assert job.isdigit(),result.stdout
    spool = OUT/'dispatch'/f'{role}-{job}.submitted.sbatch'
    subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True,text=True)
    if spool.read_bytes() != SBATCH.read_bytes():
        raise RuntimeError('Submitted batch differs from frozen script; inspect before releasing dependency')
    live = subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    receipt = dict(job_id=job,role=role,command=command,script=bind(SBATCH),submitted_script=bind(spool),live=live)
    write(OUT/'dispatch'/f'{role}-{job}.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


def chain(initial):
    OUT.joinpath('dispatch').mkdir(parents=True,exist_ok=True)
    lock = (OUT/'dispatch/worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT/'dispatch/chain.json'
    state = read(path) if path.exists() else dict(protocol=bind(OUT/'protocol.json'))
    assert state['protocol'] == bind(OUT/'protocol.json')
    protocol = read(OUT/'protocol.json')
    for b in protocol['code_sources']:
        checked(b)
    if initial:
        if 'pilot' not in state:
            state['pilot'] = submit('pilot',partition='dev_cpuonly,cpuonly')
            write(path,state)
        if 'release' not in state:
            state['release'] = submit('release',dependency=state['pilot']['job_id'],partition='dev_cpuonly,cpuonly')
            write(path,state)
    else:
        pilot = read(OUT/'pilot_validation.json')
        assert pilot['status'] == 'FIRST_FULL_C128_PATCH_SCORE_AND_ACTION_PASS'
        assert pilot['protocol'] == state['protocol']
        checked(pilot['audit'])
        if 'array' not in state:
            state['array'] = submit('array',array='0-49%50')
            write(path,state)
        if 'join' not in state:
            state['join'] = submit('join',dependency=state['array']['job_id'],partition='cpuonly')
            write(path,state)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['submit','release','pilot','array','join'])
    p.add_argument('--index',type=int,default=0)
    args = p.parse_args()
    if args.command in ['submit','release']:
        chain(args.command=='submit')
        return
    if args.command in ['pilot','array']:
        cmd = [PY,'-u',str(KERNEL),'worker','--shard',str(args.index),'--threads','4','--budget','420']
        if args.command == 'pilot':cmd.append('--pilot')
        rc = subprocess.run(cmd).returncode
        if rc:
            sys.exit(rc)
        if args.command == 'array':return
    cmd = [PY,'-u',str(AUDIT),'--budget','460']
    if args.command == 'pilot':cmd.append('--pilot')
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == '__main__':
    main()
