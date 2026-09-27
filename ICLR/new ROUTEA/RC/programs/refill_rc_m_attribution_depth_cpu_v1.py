#!/usr/bin/env python3
"""One-shot dev CPU refill for the explicitly recorded attribution jobs only."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1] / 'results/rc_m_attribution_depth_v1'


def read(path):
    return json.loads(path.read_text())


def main():
    replacement = read(ROOT / 'downstream_cpu_replacement.json')
    chain = read(ROOT / 'cpu_postprocess_chain.json')
    assert replacement['status'] == 'CPU_REPLACEMENT_AND_DEPENDENCY_VERIFIED'
    assert chain['status'] == 'CPU_POSTPROCESS_CHAIN_VERIFIED'
    allowed = {replacement['replacement'], *(v['job'] for v in chain['stages'].values())}
    lines = subprocess.check_output(
        ['squeue', '-h', '-r', '-u', 'ap7811', '-o', '%i|%P|%T|%R'], text=True
    ).splitlines()
    rows = [line.split('|', 3) for line in lines]
    # CPU and GPU development partitions share the submission allowance.
    occupied = sum(any(part.startswith('dev_') for part in row[1].split(',')) for row in rows)
    slots = max(0, 4 - occupied)
    changes = []
    for job, partition, state, reason in rows:
        if slots == 0:
            break
        if job.split('_')[0] not in allowed or state != 'PENDING':
            continue
        if 'dev_cpuonly' in partition or 'Dependency' in reason:
            continue
        before = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        if 'JobState=PENDING' not in before or 'Dependency=(null)' not in before:
            continue
        assert 'gres/gpu=1' not in before and 'cpu=4' in before and 'mem=32G' in before
        proc = subprocess.run(
            ['scontrol', 'update', 'JobId=' + job, 'Partition=dev_cpuonly,cpuonly'],
            capture_output=True, text=True,
        )
        after = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        changes.append(dict(job=job, before=before, after=after,
                            returncode=proc.returncode, stderr=proc.stderr))
        if proc.returncode == 0:
            assert 'Partition=dev_cpuonly,cpuonly' in after or 'Partition=dev_cpuonly ' in after
            slots -= 1
    record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  allowed_jobs=sorted(allowed), previous_dev_submissions=occupied, changes=changes)
    with (ROOT / 'cpu_dev_refill.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(dict(previous_dev_submissions=occupied,
                          changes=[dict(job=x['job'], returncode=x['returncode']) for x in changes])))


if __name__ == '__main__':
    main()
