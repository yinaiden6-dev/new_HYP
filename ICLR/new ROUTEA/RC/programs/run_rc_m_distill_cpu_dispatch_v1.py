#!/usr/bin/env python3
"""Execution-only CPU routing for the frozen M distillation experiment."""
import hashlib
import json
import os
import inspect
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / 'registry/rc_m_distill_cpu_dispatch_v1_20260924.json'


def checked(binding):
    p = Path(binding['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == binding['sha256'], str(p)
    return p


def routed_command(command, authority):
    if not isinstance(command, (tuple, list)) or not command or Path(str(command[0])).name != 'sbatch':
        return command
    original = authority['original_launcher']['path']
    replacement = authority['launcher']['path']
    assert original in command, ('UNEXPECTED_SBATCH', command)
    assert not any(str(x).startswith('--partition') for x in command), 'EXPLICIT_PARTITION_REVIEW_REQUIRED'
    return [command[0], '--partition=cpuonly'] + [replacement if x == original else x for x in command[1:]]


def main():
    a = json.loads(AUTH.read_text())
    for key in ('program', 'launcher', 'original_program', 'original_launcher', 'scientific_authority'):
        checked(a[key])
    original_run = subprocess.run
    # The full457 path has no probe by design. Its original progress print
    # indexes an absent key after saving a valid snapshot. Change only logging;
    # use the original module globals so configure() still binds output/authority.
    import run_rc_m_distill_full_repair_v1 as core
    body = inspect.getsource(core.run)
    old = "'probe':test['mean_query_metrics']"
    new = "'probe':test.get('mean_query_metrics', test.get('status'))"
    assert body.count(old) == 1
    exec(compile(body.replace(old, new), str(checked(a['original_core'])), 'exec'), core.__dict__)

    def run(command, *args, **kwargs):
        routed = routed_command(command, a)
        if routed == command:
            return original_run(command, *args, **kwargs)
        # Record every successful child immediately; retain useful scheduler errors.
        checked_flag = kwargs.pop('check', False)
        result = original_run(routed, *args, check=False, **kwargs)
        directory = ROOT / 'results/rc_m_distill_generalization_v1/cpu_dispatch'
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(command=routed, returncode=result.returncode, stdout=result.stdout,
                      stderr=result.stderr, parent=os.environ.get('SLURM_JOB_ID'))
        (directory / f'{time.time_ns()}_{os.getpid()}.json').write_text(json.dumps(record, indent=2))
        if checked_flag and result.returncode:
            raise RuntimeError(f'SBATCH_FAILED: {result.stderr}')
        return result

    subprocess.run = run
    runpy.run_path(str(checked(a['original_program'])), run_name='__main__')


if __name__ == '__main__':
    main()
