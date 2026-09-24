"""Submit authorized CPU -> GPU pilot only after the prior Git push is verified."""
import datetime
import json
from pathlib import Path
import subprocess

import run_rc_internal_m_condition_scale_v4 as P


def main():
    a = P.read(P.AUTH)
    P.check_sources(a)
    previous = P.read(P.OUT / 'prior_round_github_receipt.json')
    P.need(previous['commit'] == previous['remote_commit'] and previous['clean_worktree'], 'PUSH_FIRST')
    review = P.read(P.OUT / 'prefreeze_tests.json')
    P.need(review['status'] == 'PREFREEZE_INDEPENDENT_TESTS_PASS' and review['returncode'] == 0, 'TESTS_FIRST')
    for binding in review['bindings']:
        P.checked(binding)
    preflight = P.read(P.OUT / 'preflight.json')
    P.need(preflight['status'] == 'JOINT_FULL_C128_VJP_AND_NO_DIRECT_M_PASS'
           and preflight['authority'] == P.bind(P.AUTH), 'PREFLIGHT_FIRST')
    path = P.OUT / 'submission.json'
    record = P.read(path) if path.exists() else dict(authority=P.bind(P.AUTH),
             git_receipt=P.bind(P.OUT / 'prior_round_github_receipt.json'),
             submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[])
    P.need(record['authority'] == P.bind(P.AUTH), 'EXISTING_SUBMISSION_AUTHORITY')
    for stage in ('cpu', 'pilot'):
        if any(j['stage'] == stage for j in record['jobs']):
            continue
        command = ['sbatch', '--parsable']
        if stage == 'cpu':
            command += ['--partition=cpuonly,dev_cpuonly', '--gres=none', '--mem=32G', '--job-name=cn_Mscale4_cpu']
            arm = ''
        else:
            cpu_job = next(j['job_id'] for j in record['jobs'] if j['stage'] == 'cpu')
            command += ['--partition=dev_accelerated,accelerated', '--dependency=afterok:' + cpu_job, '--job-name=cn_Mscale4_pilot']
            arm = 'PRE_REAL'
        command += [str(P.LAUNCH), stage, arm]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        P.need(result.returncode == 0, 'SUBMISSION_FAILED:' + result.stderr)
        job_id = result.stdout.strip().split(';')[0]
        P.need(job_id.isdigit(), 'INVALID_SUBMITTED_JOB')
        record['jobs'].append(dict(stage=stage, arm=arm, job_id=job_id, command=command,
                                   launcher=P.bind(P.LAUNCH)))
        P.write(path, record)
    record['status'] = 'CPU_AND_DEPENDENT_GPU_PILOT_SUBMITTED'
    P.write(path, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
