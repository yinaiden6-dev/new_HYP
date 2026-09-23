#!/usr/bin/env python3
"""Reserve new GPU starts for unified acquisition; preserve running work."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_unified_priority_v1'
FUSION = ROOT / 'results/rc_h593_feature_fusion_dispatch_v3'
UNIFIED = ROOT / 'results/rc_h593_unified_acquisition_v1'
AUTH = ROOT / 'registry/rc_h593_unified_priority_authority_v1_20260923.json'
FIELDS = ('ReqTRES', 'TimeLimit', 'CPUs/Task', 'TresPerNode', 'Dependency')

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.' + p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v, indent=2) + '\n'); os.replace(tmp, p)
def run(args, check=True):
    p = subprocess.run(args, text=True, capture_output=True, timeout=40)
    if check and p.returncode: raise RuntimeError(str(args) + ': ' + p.stderr)
    return p
def info(job):
    s = run(['scontrol', 'show', 'job', '-o', job]).stdout
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)', s))
def queue():
    text = run(['squeue', '-h', '-r', '-u', 'ap7811', '-o', '%i|%j|%P|%T|%r']).stdout
    return [dict(zip(('job', 'name', 'partition', 'state', 'reason'), line.split('|', 4))) for line in text.splitlines() if line]
def scope():
    return {f"{w['job_id']}_{i}" for p in FUSION.glob('wave[0-9][0-9][0-9][0-9].json') for w in [read(p)] for i in w['indices']}
def status(name, **kw):
    write(OUT / 'status.json', dict(status=name, time_utc=now(), pid=os.getpid(), host=socket.gethostname(), **kw))
def unchanged(a, b):
    assert all(a.get(k) == b.get(k) for k in FIELDS), (a, b)

def tick():
    owned = scope()
    for row in queue():
        if row['job'] not in owned or row['name'] != 'h593_ftrain' or row['state'] != 'PENDING': continue
        job = row['job']; path = OUT / 'held' / f'{job}.json'
        before = info(job)
        assert before['UserId'].startswith('ap7811(') and before['JobName'] == 'h593_ftrain'
        if before['JobState'] != 'PENDING': continue
        if before['Reason'] == 'JobHeldUser' and not path.exists(): continue
        if before['Reason'] == 'JobHeldAdmin': continue
        rec = read(path) if path.exists() else dict(job=job, before=before, time_utc=now(), held_by_priority=False)
        if before['Reason'] != 'JobHeldUser':
            write(path, rec)
            proc = run(['scontrol', 'hold', job], check=False)
            after = info(job)
            if after['JobState'] in ('RUNNING', 'COMPLETING'):
                rec['race_running_preserved'] = after; write(path, rec); continue
            if proc.returncode: raise RuntimeError(proc.stderr)
            assert after['JobState'] == 'PENDING' and after['Reason'] == 'JobHeldUser'
            unchanged(before, after)
            rec['held_by_priority'] = True; rec['held'] = after; write(path, rec)
        # Held dev entries consume dev queue slots; move only these pending jobs back.
        current = info(job)
        if current['Partition'] == 'dev_accelerated':
            assert current['JobState'] == 'PENDING' and current['Reason'] == 'JobHeldUser'
            run(['scontrol', 'update', 'JobId=' + job, 'Partition=accelerated'])
            after = info(job); assert after['Partition'] == 'accelerated' and after['Reason'] == 'JobHeldUser'
            unchanged(current, after); rec['partition_after'] = after; write(path, rec)
    rows = queue()
    status('UNIFIED_ACQUISITION_PRIORITY_ACTIVE',
           held_fusion=sum(r['job'] in owned and r['reason'] == 'JobHeldUser' for r in rows),
           running_fusion=[r['job'] for r in rows if r['job'] in owned and r['state'] == 'RUNNING'],
           unified_jobs=[r for r in rows if r['name'] in ('h593_collect', 'h593_export')],
           cpu_quality_jobs_unchanged=[r for r in rows if r['name'] == 'h593_pqcpu'])

def restore(reason):
    rows = {r['job']: r for r in queue()}; released = []
    for p in (OUT / 'held').glob('*.json'):
        rec = read(p); job = rec['job']
        if not rec.get('held_by_priority') or job not in rows: continue
        before = info(job)
        if before['JobState'] != 'PENDING' or before['Reason'] != 'JobHeldUser': continue
        assert before['UserId'].startswith('ap7811(') and before['JobName'] == 'h593_ftrain'
        run(['scontrol', 'release', job]); after = info(job)
        assert after.get('Reason') != 'JobHeldUser'; unchanged(before, after)
        rec['released'] = after; rec['release_reason'] = reason; write(p, rec); released.append(job)
    status('PRIORITY_ENDED_FUSION_RELEASED', reason=reason, released=released)

def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'release':
        write(OUT / 'release_requested.json', dict(time_utc=now())); return
    lock = (OUT / 'watch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a = read(AUTH)
    for b in a['sources']:
        assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest() == b['sha256'], b['path']
    if mode == 'once': tick(); return
    deadline = time.monotonic() + 72 * 3600
    errors = 0
    while time.monotonic() < deadline:
        try:
            if (OUT / 'release_requested.json').exists(): restore('USER_RELEASE'); return
            us = read(UNIFIED / 'status.json')['status']
            if us == 'UNIFIED_ACQUISITION_AND_CPU_EXPORT_COMPLETE': restore('UNIFIED_COMPLETE'); return
            if us.startswith('UNIFIED_STOPPED') or us.endswith('72H_LIMIT'):
                restore('UNIFIED_CONTROLLER_STOPPED:' + us); return
            tick(); errors = 0
        except Exception as exc:
            errors += 1; status('PRIORITY_RETRY', error=repr(exc), consecutive_errors=errors)
            if errors >= 5:
                restore('PRIORITY_ERROR:' + repr(exc)); raise
        time.sleep(20)
    restore('PRIORITY_72H_LIMIT')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('once', 'watch', 'release'))
    main(p.parse_args().mode)
