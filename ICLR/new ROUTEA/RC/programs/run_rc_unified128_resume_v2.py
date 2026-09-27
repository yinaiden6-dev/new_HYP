#!/usr/bin/env python3
"""Execution-only resumption of the frozen natural first128 acquisition."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / 'registry/rc_unified128_resume_v2_20260927.json'
OLD128 = ROOT / 'results/rc_h128_unified_resume_v1/authority.json'
CACHE = ROOT / 'cache/rc_h593_unified_acquisition_v1'
CHUNK = ROOT / 'programs/run_rc_unified128_collect_chunk_v2.py'
GPU = ROOT / 'slurm/rc_unified128_resume_gpu_v2.sbatch'
FAMILIES = {'inside': 'rc_h593_m_inside_v1', 'visual': 'rc_h593_m_visual_origin_v1',
            'coordinate': 'rc_h593_roma_coordinate_precision_v2'}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(binding):
    assert bind(binding['path']) == binding, ('SOURCE_HASH_DRIFT', binding['path'])
    return Path(binding['path'])


def validate_old():
    old = read(OLD128)
    assert old['status'] == 'UNIFIED128_RESUME_AUTHORIZED'
    assert old['query_cap'] == 128 and old['max_parallel'] == 46 and old['max_restarts'] == 31
    assert old['indices'] == list(range(41, 70)) + list(range(71, 128))
    assert old['already_all_families_complete'] == list(range(41)) + [70]
    for binding in [old['original_authority'], old['pilot_export'], *old['sources']]:
        checked(binding)
    scientific = read(old['original_authority']['path'])
    for binding in [*scientific['sources'], scientific['workers'], scientific['profile']]:
        checked(binding)
    pilot = read(old['pilot_export']['path'])
    assert pilot['status'] == 'UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS'
    for binding in pilot['validations'].values():
        checked(binding)
    return old


def complete(index, verify=False):
    for name in FAMILIES.values():
        path = ROOT / 'results' / name / f'query{index:03d}/validation.json'
        if not path.exists():
            return False
        if verify:
            validation = read(path)
            assert validation['status'].endswith('PASS')
            checked(validation['payload'])
    return True


def guard():
    authority = read(AUTH)
    assert authority['status'] == 'UNIFIED128_RESUME_V2_AUTHORIZED'
    for binding in [authority['original128_authority'], *authority['sources']]:
        checked(binding)
    old = validate_old()
    assert authority['indices'] == old['indices']
    assert authority['query_cap'] == 128 and authority['max_restarts'] == 31
    return authority


def prepare():
    assert not AUTH.exists(), 'RESUME_AUTHORITY_ALREADY_EXISTS'
    old = validate_old()
    from run_rc_unified128_collect_chunk_v2 import selftest
    selftest()
    locks = []
    handles = []
    try:
        for name in ('rc_h593_unified_acquisition_v1', 'rc_h593_finish70_then_pause_v1'):
            path = ROOT / 'results' / name / 'watch.lock'
            assert path.exists(), ('EXPECTED_WATCH_LOCK_MISSING', str(path))
            handle = path.open('a+')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append({'path': str(path), 'status': 'EXCLUSIVE_LOCK_ACQUIRED'})
        done = [index for index in range(128) if complete(index, verify=True)]
        assert set(old['already_all_families_complete']) <= set(done)
        sources = [Path(__file__), CHUNK, GPU,
                   ROOT / 'programs/run_rc_h593_unified_duplicate_resume_v1.py',
                   ROOT / 'registry/rc_h593_unified_duplicate_resume_authority_v1_20260923.json',
                   ROOT / 'results/rc_h128_unified_resume_v1/export_pack_authority.json',
                   ROOT / 'programs/run_rc_unified128_export_pack_v1.py',
                   ROOT / 'slurm/rc_unified128_export_pack_v1.sbatch']
        authority = dict(status='UNIFIED128_RESUME_V2_AUTHORIZED',
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            user_request='71张缓存?现在先启动原来暂停的统一采样',
            original128_authority=bind(OLD128), sources=[bind(path) for path in sources],
            indices=old['indices'], already_complete_at_resume=done,
            query_cap=128, max_parallel=46, max_restarts=31,
            scientific_changes=0, held_fusion_unchanged=True,
            repair='Replay identical-pair native matcher when internal hooks are required; verify replay outputs bit-exact; distinct restart receipts',
            original_validators_unchanged=True, prior_watch_locks=locks,
            synthetic_regression='UNIFIED128_DUPLICATE_HOOK_AND_RESTART_SELFTEST_PASS')
        text = json.dumps(authority, indent=2, ensure_ascii=False) + '\n'
        temporary = AUTH.with_name('.' + AUTH.name + f'.{os.getpid()}.tmp')
        try:
            with temporary.open('x') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, AUTH)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        for handle in handles:
            handle.close()
    guard()
    print({'status': 'UNIFIED128_RESUME_V2_PREPARED', 'authority': str(AUTH),
           'additional_queries': len([i for i in old['indices'] if i not in done]),
           'already_all_families_complete': len(done)}, flush=True)


def collect(index):
    authority = guard()
    assert os.environ.get('SLURM_JOB_ID')
    assert index in authority['indices']
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    assert 0 <= restart <= authority['max_restarts']
    if complete(index, verify=True):
        return 0
    ready = CACHE / f'query{index:03d}/ready.json'
    if ready.exists():
        checked(read(ready)['manifest'])
        return 0
    result = subprocess.run([str(ROOT.parents[2] / '.venv-romav2/bin/python'),
                             str(CHUNK), str(index)])
    if result.returncode:
        return result.returncode
    if ready.exists():
        checked(read(ready)['manifest'])
        return 0
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'collect'))
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        sys.exit(collect(args.index))
