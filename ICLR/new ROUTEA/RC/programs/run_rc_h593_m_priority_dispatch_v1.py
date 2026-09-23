#!/usr/bin/env python3
"""Prioritize unchanged M-origin experiments with short accelerated arrays."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('packed_parent', ROOT/'programs/run_rc_h593_packed_gpu_v2.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
OUT = ROOT/'results/rc_h593_m_priority_dispatch_v1'
AUTH = ROOT/'registry/rc_h593_m_priority_dispatch_authority_v1_20260923.json'
PLAN = ROOT/'plan/RC_H593_M_PRIORITY_DISPATCH_V1_20260923.md'
LAUNCH = ROOT/'slurm/rc_h593_shared_pooling_v1.sbatch'
FAMILIES = ('inside', 'visual')
read, write, bind, checked, command = P.read, P.write, P.bind, P.checked, P.command


def event(state, **kw):
    write(OUT/'status.json', dict(time_utc=P.now(), host=socket.gethostname(), pid=os.getpid(), status=state, **kw), replace=True)


def choose(done, active, excluded, capacity, attempts):
    pools = {f: [i for i in range(593) if i not in done[f] and (f, i) not in active and (f, i) not in excluded] for f in FAMILIES}
    answer = []
    while len(answer) < capacity and any(pools.values()):
        for f in ('inside', 'inside', 'visual'):
            if pools[f] and len(answer) < capacity:
                i = pools[f].pop(0)
                assert attempts[f, i] < 32, ('CHUNK_LIMIT', f, i)
                answer.append((f, i))
    assert len(answer) == len(set(answer))
    assert not set(answer) & (set(active) | set(excluded))
    return answer


def prepare():
    assert not AUTH.exists()
    sources = [Path(__file__), PLAN, LAUNCH, Path(P.__file__), P.AUTH, P.POOL/'ready.json',
               ROOT/'registry/rc_h593_shared_pooling_cache_authority_v1_20260922.json',
               ROOT/'registry/rc_h593_m_path_readout_authority_v1_20260922.json',
               *[P.AUTHORITIES[f] for f in FAMILIES]]
    P.guard()
    done = {f: set() for f in FAMILIES}
    selected = choose(done, set(), {('inside', 0), ('visual', 0)}, 42, Counter())
    assert len(selected) == 42 and Counter(f for f, _ in selected) == {'inside': 28, 'visual': 14}
    assert choose({f: set(range(593)) for f in FAMILIES}, set(), set(), 46, Counter()) == []
    write(AUTH, dict(status='M_PRIORITY_SHORT_ARRAYS_AUTHORIZED', sources=[bind(p) for p in sources],
                    families=list(FAMILIES), partition='accelerated', gpus_per_task=1, cpus_per_task=8,
                    memory_gb=64, minutes=12, maximum_active_gpus=46, maximum_days=14,
                    maximum_attempts_per_query=32, maximum_submissions=2048,
                    readout_job='5158704', deferred=['coordinate', 'fusion'],
                    scientific_change=False, candidate_count=128, queries=593,
                    user_instruction='Prioritize why M is useful; restore original multiple short GPU shards in accelerated',
                    running_packed_policy='Let current allocation finish, exclude all its manifest tasks until exit',
                    no_new_packed_allocations=True))
    write(OUT/'preflight.json', dict(status='M_PRIORITY_SELECTION_PASS', authority=bind(AUTH),
          capacity_including_four_legacy_gpus=46, short_tasks=42, inside=28, visual=14,
          duplicate_and_running_exclusion=True, scientific_programs_unchanged=True))


def guard():
    a = read(AUTH)
    assert a['status'] == 'M_PRIORITY_SHORT_ARRAYS_AUTHORIZED'
    for b in a['sources']: checked(b)
    return a


def take_over():
    guard()
    proc = read(P.OUT/'supervisor_process.json')
    assert proc['host'] == socket.gethostname(), 'OLD_SUPERVISOR_DIFFERENT_HOST'
    pid = proc['pid']; path = Path(f'/proc/{pid}/cmdline')
    before = read(P.OUT/'status.json')
    if path.exists():
        argv = [s.decode() for s in path.read_bytes().split(b'\0') if s]
        assert argv == proc['command'], ('PID_COMMAND_CHANGED', argv)
        assert path.stat().st_uid == os.getuid()
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            if not path.exists() or not path.read_bytes(): break
            time.sleep(.1)
        assert not path.exists() or not path.read_bytes(), 'OLD_SUPERVISOR_STILL_RUNNING'
    lock = (P.OUT/'watch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active_roots = {r['job'].split('_')[0]: r for r in P.queue()}
    retained = []
    for source in sorted((P.OUT/'batches').glob('*/submission.json')):
        d = read(source); row = active_roots.get(d['job'])
        if row is None: continue
        if row['state'] == 'PENDING':
            info = P.jobinfo(d['job']); assert info['JobState'] == 'PENDING' and info['JobName'] == 'h593_pack'
            command(['scancel', d['job']])
        else:
            assert row['state'] in ('RUNNING', 'COMPLETING'), row
            retained.append(dict(job=d['job'], manifest=d['manifest'], submission=bind(source)))
    readout = P.jobinfo('5158704')
    assert readout['JobState'] == 'PENDING' and readout['Reason'] == 'JobHeldUser'
    write(OUT/'takeover.json', dict(authority=bind(AUTH), time_utc=P.now(), stopped_supervisor=proc,
          prior_status=before, retained_running=retained, readout='5158704',
          fusion_old_arrays_kept_held=True, cache_and_outputs_preserved=True))
    event('TAKEOVER_READY', retained_running=[x['job'] for x in retained])


def submit(family, indices):
    number = len(list((OUT/'waves').glob('*.json')))
    assert number < 2048 and family in FAMILIES and 0 < len(indices) <= 46
    args = ['sbatch', '--parsable', '--hold', '--partition=accelerated', '--time=00:12:00',
            '--job-name=h593_m_'+('in' if family == 'inside' else 'vis'),
            '--array='+','.join(map(str, indices))+'%46', str(LAUNCH), 'consumer', family]
    job = command(args).stdout.strip().split(';')[0]; assert job.isdigit()
    try:
        spool = OUT/'spools'/f'{job}.sh'; spool.parent.mkdir(parents=True, exist_ok=True)
        command(['scontrol', 'write', 'batch_script', job, str(spool)])
        assert spool.read_bytes() == LAUNCH.read_bytes()
        info = P.jobinfo(job)
        assert info['Partition'] == 'accelerated' and info['TimeLimit'] == '00:12:00'
        assert info['NumCPUs'] == '8' and 'gres/gpu=1' in info['ReqTRES'] and 'mem=64G' in info['ReqTRES']
        write(OUT/'waves'/f'{number:04d}.json', dict(job=job, family=family, indices=indices,
              authority=bind(AUTH), arguments=args, spool=bind(spool), verified_job=info))
        command(['scontrol', 'release', job])
        after = P.jobinfo(job)
        assert after['Partition'] == 'accelerated' and after['TimeLimit'] == '00:12:00'
        assert after.get('Reason') != 'JobHeldUser'
    except BaseException:
        command(['scancel', job], check=False); raise
    print(json.dumps(dict(submitted=job, family=family, indices=indices)), flush=True)


def accounting(jobs):
    if not jobs: return {}
    output = command(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs), '--format=JobID,JobIDRaw,State,ExitCode']).stdout
    return {p[0]: dict(raw=p[1], state=p[2], exit=p[3]) for line in output.splitlines() if len(p := line.strip().split('|')) >= 4}


def joins(done):
    for family in FAMILIES:
        dest = OUT/f'{family}_join.json'
        if len(done[family]) != 593 or dest.exists(): continue
        launcher = ROOT/('slurm/rc_h593_m_inside_control_v1.sbatch' if family == 'inside' else 'slurm/rc_h593_m_visual_origin_control_v1.sbatch')
        job = command(['sbatch', '--parsable', str(launcher), 'join']).stdout.strip().split(';')[0]
        assert job.isdigit(); write(dest, dict(job=job, authority=bind(AUTH), launch=bind(launcher)))
    if len(done['inside']) == 593 and not (OUT/'readout_released.json').exists():
        info = P.jobinfo('5158704')
        assert info['JobState'] == 'PENDING' and info['Reason'] == 'JobHeldUser'
        assert info['Dependency'] == '(null)'
        command(['scontrol', 'release', '5158704'])
        after = P.jobinfo('5158704'); assert after.get('Reason') != 'JobHeldUser'
        write(OUT/'readout_released.json', dict(job='5158704', time_utc=P.now(), prerequisite='ALL593_INSIDE_VALIDATED'))


def tick():
    takeover = read(OUT/'takeover.json'); rows = P.queue(); queued = {r['job'] for r in rows}
    roots = {x.split('_')[0] for x in queued}; excluded = set(); packed_gpus = 0
    for item in takeover['retained_running']:
        marker = OUT/'retired'/f"packed_{item['job']}.json"
        if item['job'] in roots:
            packed_gpus += 4
            for t in read(checked(item['manifest']))['tasks']:
                if t['family'] in FAMILIES: excluded.add((t['family'], t['index']))
        elif not marker.exists():
            acc = accounting([item['job']]).get(item['job'])
            assert acc and acc['state'] == 'COMPLETED' and acc['exit'] == '0:0', ('LEGACY_ALLOCATION_FAILED', acc)
            val = Path(item['manifest']['path']).with_name('validation.json')
            assert read(val)['status'] == 'PACKED_FOUR_GPU_PASS'
            write(marker, dict(accounting=acc, validation=bind(val)))
    waves = [read(p) for p in sorted((OUT/'waves').glob('*.json'))]
    active = set(); unsettled = []; attempts = Counter()
    # Preserve the original conservative chunk cap across prior schedulers.
    for path in (P.POOL/'dispatch').glob('*.json'):
        d = read(path)
        if d.get('family') in FAMILIES:
            for i in d['indices']: attempts[d['family'], i] += 1
    for path in (P.OUT/'batches').glob('*/manifest.json'):
        for t in read(path)['tasks']:
            if t['family'] in FAMILIES: attempts[t['family'], t['index']] += 1
    for w in waves:
        for i in w['indices']:
            key = f"{w['job']}_{i}"; pair = (w['family'], i); attempts[pair] += 1
            marker = OUT/'retired'/f'{key}.json'
            if key in queued: active.add(pair)
            elif not marker.exists(): unsettled.append((w, i, key, marker))
    acc = accounting(sorted({w['job'] for w, _, _, _ in unsettled}))
    for w, i, key, marker in unsettled:
        row = acc.get(key)
        if not row or row['state'] in ('PENDING', 'RUNNING', 'COMPLETING'):
            active.add((w['family'], i)); continue
        assert row['state'] == 'COMPLETED' and row['exit'] == '0:0', ('SHORT_JOB_FAILED', key, row)
        receipt = P.POOL/'consumers'/w['family']/f"{row['raw']}_{i}.json"
        r = read(receipt); assert r['index'] == i and r['authority'] == bind(ROOT/'registry/rc_h593_shared_pooling_cache_authority_v1_20260922.json')
        if r['complete']: assert P.complete(dict(family=w['family'], index=i), True)
        write(marker, dict(task=dict(family=w['family'], index=i), accounting=row, receipt=bind(receipt)))
    done = {f: {i for i in range(593) if P.complete(dict(family=f, index=i))} for f in FAMILIES}
    assert len(active) + packed_gpus <= 46
    tasks = choose(done, active, excluded, 46-len(active)-packed_gpus, attempts)
    for family in FAMILIES:
        indices = [i for f, i in tasks if f == family]
        if indices: submit(family, indices)
    joins(done)
    validations = [P.FOLDERS[f]/'validation.json' for f in FAMILIES] + [P.READOUT/'validation.json']
    if all(p.exists() for p in validations):
        expected = ['M_ORIGIN_JOIN_COUNTS_PASS', 'M_ORIGIN_JOIN_COUNTS_PASS', 'M_PATH_READOUT_COUNTS_PASS']
        for p, status in zip(validations, expected):
            v = read(p); assert v['status'] == status; checked(v['result'])
        event('M_ORIGIN_AND_RETRIEVAL_COMPLETE', validations=[bind(p) for p in validations]); return True
    event('M_PRIORITY_ACTIVE', counts={f: len(v) for f, v in done.items()},
          existing_short_tasks=len(active), newly_submitted=len(tasks), legacy_packed_gpus=packed_gpus,
          maximum_gpu_concurrency=46, readout_released=(OUT/'readout_released.json').exists(), deferred=['coordinate', 'fusion'])
    return False


def watch():
    a = guard(); lock = (OUT/'watch.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep the old supervisor lock for the whole handoff, preventing its restart.
    old_lock = (P.OUT/'watch.lock').open('a+'); fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()+a['maximum_days']*86400
    while time.monotonic() < deadline:
        try:
            if tick(): return
        except Exception as e:
            event('STOPPED_ERROR', error=repr(e)); raise
        time.sleep(30)
    event('STOPPED_14_DAY_LIMIT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'takeover', 'watch'])
    args = parser.parse_args()
    {'prepare': prepare, 'takeover': take_over, 'watch': watch}[args.stage]()
