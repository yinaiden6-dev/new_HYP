#!/usr/bin/env python3
"""Submit the frozen Task 2 DAG once; all work reuses CPU-readable caches."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_simple_external_transfer_v1'
LAUNCH = ROOT / 'slurm/rc_simple_external_transfer_v1.sbatch'
AUTH = ROOT / 'registry/rc_simple_external_transfer_authority_v1_20260924.json'


def main():
    p = OUT / 'jobs.json'
    assert not p.exists(), 'Submission record exists: inspect it before any resubmission'
    a = json.loads(AUTH.read_text())
    b = a['code_sources']['launcher']
    assert hashlib.sha256(LAUNCH.read_bytes()).hexdigest() == b['sha256']
    records = dict(authority=str(AUTH), authorized_task=2, submitted=[], created_unix=time.time())

    def submit(name, args, options=(), deps=()):
        cmd = ['sbatch', '--parsable', '--job-name=' + name, *options]
        if deps:
            cmd += ['--dependency=afterok:' + ':'.join(deps)]
        cmd += [str(LAUNCH), *args]
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        row = dict(job_id=job, name=name, command=cmd, dependencies=list(deps))
        records['submitted'].append(row)
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        return job

    # Only these two scalar jobs can enter the limited development partition.
    source = submit('simp_src', ['source', 'fit'], ['--partition=cpuonly,dev_cpuonly'])
    inputs = submit('simp_inputs', ['replay', 'preflight-all'], ['--partition=cpuonly,dev_cpuonly'])
    pilot = submit('simp_pilot', ['replay', 'array'], ['--array=0,60%2'], [source, inputs])
    replay = submit('simp_replay', ['replay', 'array'], ['--array=1-59,61-127%32'], [pilot])
    join = submit('simp_join', ['replay', 'join'], [], [replay])
    verify = submit('simp_verify', ['replay', 'verify'], [], [join])
    records['status'] = 'TASK2_DAG_SUBMITTED_NOT_RESULTS'
    records['completion_job'] = verify
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
