#!/usr/bin/env python3
"""Submit only the 14 source-precision affected pairs, with fail-closed join."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_structure_binding_isolation_v2'
LAUNCHER = RC / 'slurm/rc_m_isolation_precision_repair_20261001.sbatch'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    assert (ROOT / 'protocol.json').exists()
    with (ROOT / 'submit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = ROOT / 'submission.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'status': 'SUBMITTING', 'replaces_failed_array': '5171646',
            'replaces_blocked_join': '5171647', 'stages': {}}
        for name, array, deps, partition in [
            ('pilot', None, [], 'dev_cpuonly,cpuonly'),
            ('work', '0-12,15%14', ['pilot'], 'cpuonly'),
            ('join', None, ['work'], 'cpuonly'),
        ]:
            if name in state['stages']:
                continue
            args = ['sbatch', '--parsable', '--partition=' + partition,
                    '--job-name=Miso2_' + name]
            if array:
                args += ['--array=' + array]
            if deps:
                args += ['--dependency=afterok:' + ':'.join(state['stages'][k]['job_id'] for k in deps)]
            args += [str(LAUNCHER), name]
            job = subprocess.check_output(args, text=True).strip().split(';')[0]
            assert job.isdigit()
            state['stages'][name] = dict(job_id=job, args=args, array=array, dependencies=deps)
            save(path, state)
            print(name, job, flush=True)
        for name, entry in state['stages'].items():
            job = entry['job_id']
            output = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
            assert 'TimeLimit=00:10:00' in output and 'gres/gpu' not in output
            spool = ROOT / f'submitted_{job}.sbatch'
            if not spool.exists():
                subprocess.run(['scontrol', 'write', 'batch_script', job, str(spool)], check=True, capture_output=True)
            assert spool.read_bytes() == LAUNCHER.read_bytes()
            entry.update(scheduler=output.strip(), spool_sha256=hashlib.sha256(spool.read_bytes()).hexdigest())
        state['status'] = 'SUBMITTED_AND_SPOOL_VERIFIED'
        save(path, state)


if __name__ == '__main__':
    main()
