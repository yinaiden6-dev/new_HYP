#!/usr/bin/env python3
"""Place only recorded H593 pending GPU tasks in spare dev slots; no resubmits."""
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
OUT = ROOT/'results/rc_h593_dev_reuse_20260923_v1'
FIELDS = ('TimeLimit', 'ReqTRES', 'CPUs/Task', 'TresPerNode', 'Dependency')
NAMES = {'inside': {'h593_m_in'}, 'visual': {'h593_m_vis'},
         'fusion': {'h593_ftrain'}, 'coordinate': {'h593_pool'}}

def read(p): return json.loads(Path(p).read_text())
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def run(args, check=True):
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=40)
def queue():
    s = run(['squeue', '-h', '-r', '-u', 'ap7811', '-o', '%i|%j|%P|%T|%r']).stdout
    return [dict(zip(('job', 'name', 'partition', 'state', 'reason'), x.split('|', 4))) for x in s.splitlines() if x]
def info(job):
    s = run(['scontrol', 'show', 'job', '-o', job]).stdout
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)', s))
def owned():
    result = {}
    for name in ('rc_h593_m_priority_dispatch_v1', 'rc_h593_m_stage_reuse_dispatch_v1'):
        for p in (ROOT/'results'/name/'waves').glob('*.json'):
            w = read(p)
            for i in w['indices']: result[f"{w['job']}_{i}"] = w['family']
    for p in (ROOT/'results/rc_h593_feature_fusion_dispatch_v3').glob('wave[0-9][0-9][0-9][0-9].json'):
        w = read(p)
        for i in w['indices']: result[f"{w['job_id']}_{i}"] = 'fusion'
    for p in (ROOT/'cache/rc_h593_shared_pooling_v1/dispatch').glob('*.json'):
        w = read(p)
        if w.get('family') == 'coordinate':
            for i in w['indices']: result[f"{w['job']}_{i}"] = 'coordinate'
    return result
def candidates(rows, scope, turn):
    dev = [r for r in rows if any(p.startswith('dev_') for p in r['partition'].split(','))]
    gpu = [r for r in dev if r['partition'] == 'dev_accelerated']
    free = max(0, min(4-len(dev), 3-len(gpu)))
    order = ['inside', 'visual', 'fusion', 'coordinate']
    order = order[turn % 4:] + order[:turn % 4]
    pool = [r for r in rows if r['job'] in scope and r['name'] in NAMES[scope[r['job']]]
            and r['state'] == 'PENDING' and r['partition'] == 'accelerated'
            and r['reason'] not in ('JobHeldUser', 'JobHeldAdmin', 'Dependency', 'DependencyNeverSatisfied')]
    pool.sort(key=lambda r: (order.index(scope[r['job']]), int(r['job'].split('_')[-1])))
    return pool[:free]
def record(v):
    with (OUT/'backfill_events.jsonl').open('a') as f: f.write(json.dumps(dict(time_utc=now(), **v))+'\n')
def tick(turn):
    rows = queue(); scope = owned(); moved = []
    for row in candidates(rows, scope, turn):
        if not candidates(queue(), scope, turn): break
        b = info(row['job'])
        assert b['UserId'].startswith('ap7811(') and b['JobName'] in NAMES[scope[row['job']]]
        if b['JobState'] != 'PENDING' or b['Partition'] != 'accelerated': continue
        assert b['TimeLimit'] in ('00:12:00', '00:13:00', '00:15:00')
        assert b['CPUs/Task'] == '8' and 'gres/gpu=1' in b['ReqTRES'] and 'mem=64G' in b['ReqTRES']
        proc = run(['scontrol', 'update', 'JobId='+row['job'], 'Partition=dev_accelerated'], check=False)
        if proc.returncode:
            record(dict(event='SCHEDULER_RETRY', job=row['job'], message=proc.stderr.strip())); break
        a = info(row['job']); assert a['Partition'] == 'dev_accelerated'
        for k in FIELDS: assert b.get(k) == a.get(k), k
        moved.append(row['job'])
        record(dict(event='MIGRATED_VERIFIED', job=row['job'], family=scope[row['job']], before=b, after=a))
    value = dict(time_utc=now(), host=socket.gethostname(), pid=os.getpid(), status='DEV_BACKFILL_ACTIVE',
                 turn=turn, moved=moved, scope_count=len(scope), dev_jobs=[r for r in queue() if r['partition'].startswith('dev_')],
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    p = OUT/'.backfill_status.tmp'; p.write_text(json.dumps(value, indent=2)+'\n'); p.replace(OUT/'backfill_status.json')
def watch():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'backfill.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()+24*3600
    turn = 0; errors = 0
    while time.monotonic() < deadline:
        try: tick(turn); errors = 0
        except AssertionError as exc:
            record(dict(event='STOPPED_CONTRACT', error=str(exc))); raise
        except Exception as exc:
            errors += 1; record(dict(event='TRANSIENT_ERROR', error=str(exc), count=errors))
            if errors >= 5: raise
        turn += 1; time.sleep(20)
    record(dict(event='STOPPED_24H_PLACEMENT_LIMIT', running_jobs_unchanged=True))
def test():
    rows = [dict(job=f'1_{i}', name='h593_m_in', partition='accelerated', state='PENDING', reason='Priority') for i in range(5)]
    scope = {r['job']: 'inside' for r in rows}
    assert len(candidates(rows, scope, 0)) == 3
    rows += [dict(job='2', name='other', partition='dev_cpuonly', state='RUNNING', reason='None')]
    assert len(candidates(rows, scope, 0)) == 3
    rows += [dict(job=f'3_{i}', name='other', partition='dev_accelerated', state='PENDING', reason='Priority') for i in range(3)]
    assert not candidates(rows, scope, 0)
    rows = rows[:5]; rows[0]['reason'] = 'JobHeldUser'; rows[1]['state'] = 'RUNNING'; rows[2]['name'] = 'other'
    assert [r['job'] for r in candidates(rows, scope, 0)] == ['1_3', '1_4']
    print('DEV_BACKFILL_SCOPE_CAP_AND_RUNNING_EXCLUSION_PASS')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('test', 'watch')); args = ap.parse_args()
    test() if args.stage == 'test' else watch()
