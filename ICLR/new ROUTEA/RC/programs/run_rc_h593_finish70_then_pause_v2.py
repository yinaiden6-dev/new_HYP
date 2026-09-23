#!/usr/bin/env python3
"""Finish only query 70 with the original workers; never expand GPU sampling."""
import fcntl
import importlib.util
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_unified_dispatch', ROOT / 'programs/run_rc_h593_unified_dispatch_v1.py')
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
OUT = ROOT / 'results/rc_h593_finish70_then_pause_v1'
AUTH = ROOT / 'registry/rc_h593_finish70_then_pause_authority_v2_20260923.json'


def select_only70(done, ready, active, exported, first):
    assert first == 70
    if 70 in exported or 70 in active:
        return 'collect', []
    return ('export' if 70 in ready else 'collect'), [70]


def status(name, **kwargs):
    U.write(OUT / 'status.json', dict(status=name, time_utc=U.P.now(),
        host=socket.gethostname(), pid=os.getpid(), **kwargs), replace=True)


def main():
    a = U.read(AUTH)
    for b in a['sources']:
        U.checked(b)
    old = U.read(U.AUTH)
    for b in old['sources']:
        U.checked(b)
    assert old['first_complete_query'] == 70
    legacy_out = U.OUT
    lock = (OUT / 'watch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The previous dispatch cannot advance past its cancelled last wave:
    # either its accounting asserts CANCELLED or it waits for the array row.
    # Do not rewrite its immutable ledger to pretend the worker completed.
    for record in a['cancelled_pending']:
        assert record['before']['JobState'] == 'PENDING'
        assert record['before']['JobName'] == 'h593_collect'
        assert record['before']['ArrayTaskId'] == '70'
        assert all(r['job'] != record['job'] for r in U.P.queue())
    U.OUT = OUT / 'dispatch'
    U.OUT.mkdir(parents=True, exist_ok=True)
    for name in ('pilot_submission.json', 'transition_hold.json'):
        U.write(U.OUT / name, U.read(legacy_out / name))
    U.AUTH = AUTH
    U.GPU = ROOT / 'slurm/rc_h593_unified_duplicate_resume_v1.sbatch'
    U.selection = select_only70
    U.retire_legacy = lambda: None
    U.joins = lambda done: None
    original_submit = U.submit

    def submit(stage, indices, partition=None):
        assert indices == [70]
        return original_submit(stage, indices, partition or (
            'dev_accelerated' if stage == 'collect' else 'cpuonly'))

    U.submit = submit
    deadline = time.monotonic() + 72 * 3600
    try:
        while time.monotonic() < deadline:
            U.tick(old)
            U.cpu_placement()
            latest = U.read(U.OUT / 'status.json')
            if latest.get('first_export_validated'):
                v = U.CACHE / 'query070/export_validation.json'
                value = U.read(v)
                assert value['status'] == 'UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS'
                for binding in value['validations'].values():
                    U.checked(binding)
                status('INDEX70_VALIDATED_COLLECTION_PAUSED', validation=U.bind(v),
                    subsequent_gpu_collection=False, gpu_fusion_still_held=True,
                    cpu_experiments_unchanged=True, counts=latest['counts'])
                U.status('UNIFIED_PAUSED_AFTER_INDEX70', first_query=70,
                    first_export_validated=True, counts=latest['counts'],
                    pause_authority=U.bind(AUTH), resume_requires_user_request=True)
                return
            status('FINISHING_INDEX70_ONLY', original_status=latest,
                saved_pairs=len(list((U.CACHE / 'query070').glob('pair*.pt'))))
            time.sleep(20)
        status('PAUSED_CONTROLLER_TIME_LIMIT', outputs_preserved=True)
    except BaseException as exc:
        status('PAUSED_CONTROLLER_NEEDS_REPAIR', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
