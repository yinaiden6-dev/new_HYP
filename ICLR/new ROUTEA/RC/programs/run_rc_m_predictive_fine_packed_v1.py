#!/usr/bin/env python3
"""Execution-only packaging of unchanged independent context shard workers."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import submit_rc_m_predictive_and_fine_c128_v1 as C


def command(shard):
    return [sys.executable, '-u', str(C.FINE_CODE), 'worker', '--shard', str(shard),
            '--shards', '50', '--budget', '430']


def execute(shard):
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    path = C.RC/'logs'/f"M2-packed-{os.environ['SLURM_JOB_ID']}-shard{shard:02d}.out"
    with path.open('a') as log:
        process = subprocess.run(command(shard), env=env, stdout=log, stderr=subprocess.STDOUT)
    return dict(shard=shard, exit_code=process.returncode)


def main():
    C.guard()
    p = C.read(C.ROOT/'packed_execution_plan.json')
    assert p['scientific_protocol'] == C.bind(C.ROOT/'protocol.json')
    assert p['wrapper'] == C.bind(__file__)
    task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    shards = p['groups'][task]
    assert len(shards) <= 13 and int(os.environ['SLURM_CPUS_PER_TASK']) >= 4*len(shards)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as pool:
        records = list(pool.map(execute, shards))
    receipt = dict(status='PACKED_SLICE', job=os.environ['SLURM_JOB_ID'], task=task,
                   restart=int(os.environ.get('SLURM_RESTART_COUNT', '0')), children=records)
    C.save(C.ROOT/f"packed_slices/{receipt['job']}-{receipt['restart']}.json", receipt)
    codes = [r['exit_code'] for r in records]
    if any(c not in (0, 75) for c in codes):
        print(json.dumps(receipt), flush=True)
        return 1
    print(json.dumps(receipt), flush=True)
    return 75 if 75 in codes else 0


if __name__ == '__main__':
    raise SystemExit(main())
