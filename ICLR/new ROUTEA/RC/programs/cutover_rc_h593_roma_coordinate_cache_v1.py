#!/usr/bin/env python3
"""Qualified cache rollout, restricted to pending 5157033_59..92 and 5157034."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_h593_roma_coordinate_cache_cutover_v1'
AUTH = ROOT/'registry/rc_h593_roma_coordinate_cache_cutover_authority_v1_20260922.json'
EXEC_AUTH = ROOT/'registry/rc_h593_roma_coordinate_cache_authority_v2_20260922.json'
QUALIFICATION = ROOT/'results/rc_h593_roma_coordinate_cache_v2/validation.json'
DATA = ROOT/'results/rc_h593_roma_coordinate_precision_v2'
WORKER = ROOT/'slurm/rc_h593_roma_coordinate_cache_v2.sbatch'
LAUNCH = ROOT/'slurm/rc_h593_roma_coordinate_cache_cutover_v1.sbatch'
DISPATCH_PROGRAM = ROOT/'programs/dispatch_rc_h593_roma_coordinate_cache_v1.py'
DISPATCH_LAUNCH = ROOT/'slurm/rc_h593_roma_coordinate_cache_dispatch_v1.sbatch'
DISPATCH_AUTH = ROOT/'registry/rc_h593_roma_coordinate_cache_dispatch_authority_v1_20260922.json'
PYTHON = ROOT.parents[2]/'.venv-romav2/bin/python'


def read(p): return json.loads(Path(p).read_text())
def bind(p): return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def checked(b):
    assert bind(Path(b['path'])) == b, b['path']
    return Path(b['path'])
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if p.exists():
        assert p.read_text() == text, str(p)
        return
    tmp = p.with_name('.'+p.name+f'.{os.getpid()}.tmp'); tmp.write_text(text); os.link(tmp, p); tmp.unlink()
def event(kind, **kw):
    value = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=kind, **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(value, flush=True)
def command(args, check=True):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=check, timeout=45)
def show(job):
    p = command(['scontrol', 'show', 'job', '-o', job], check=False)
    if p.returncode: return None
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)', p.stdout))
def owned(job, d):
    assert job == '5157034' or job in {f'5157033_{i}' for i in range(59,93)}
    assert d['UserId'].startswith('ap7811(')
    assert d['JobName'] == ('h593_coord_next' if job == '5157034' else 'h593_coord2')
def qualified(index):
    p = DATA/f'query{index:03d}/validation.json'
    if not p.exists(): return False
    v = read(p); assert v['status'] == 'ROMA_COORDINATE_QUERY_PASS'; checked(v['payload'])
    assert v['authority'] == read(EXEC_AUTH)['scientific_authority']
    return True
def submit(path, args):
    if path.exists(): return read(path)['job_id']
    p = command(['sbatch', '--parsable', *args]); job = p.stdout.strip().split(';')[0]
    assert job.isdigit()
    write(path, dict(job_id=job, args=args, authority=bind(AUTH))); return job
def release(held):
    for job in held:
        d = show(job)
        if d and d['JobState'] == 'PENDING':
            owned(job, d)
            p = command(['scontrol', 'release', job], check=False)
            event('OLD_TASK_RELEASED', job=job, returncode=p.returncode, message=p.stderr.strip())


def prepare():
    assert not AUTH.exists()
    files = [Path(__file__), WORKER, LAUNCH, DISPATCH_PROGRAM, DISPATCH_LAUNCH]
    write(AUTH, dict(status='EXACT_CACHE_STAGED_CUTOVER_AUTHORIZED', execution=bind(EXEC_AUTH),
                     sources=[bind(p) for p in files], qualification_job='5157565',
                     pending_scope=[f'5157033_{i}' for i in range(59,93)]+['5157034'],
                     stages='Qualification afterok -> hold scoped pending old jobs -> one natural cached worker -> afterany verifier -> cancel held pending old jobs and continue cached waves',
                     failure='First cached worker failure releases held original jobs; never cancels running tasks',
                     user_authorization='2026-09-22 改; cache-only acceleration with original completed data retained'))


def start():
    assert not (OUT/'transition.json').exists(), 'ALREADY_STARTED'
    held = []; scheduled = False; job = None
    try:
        # Pause the callback before changing the pending worker set.
        for job in ['5157034', *(f'5157033_{i}' for i in range(59,93))]:
            d = show(job)
            if not d: continue
            owned(job, d)
            if d['JobState'] != 'PENDING' or d.get('Priority') == '0': continue
            command(['scontrol', 'hold', job])
            held.append(job)
            after = show(job); assert after and after['JobState'] == 'PENDING' and after['Priority'] == '0'
            for k in ('Partition', 'TimeLimit', 'ReqTRES', 'Dependency'): assert d.get(k) == after.get(k)
            event('PENDING_OLD_TASK_HELD', job=job)
        todo = [i for i in range(59,93) if f'5157033_{i}' in held and not qualified(i)]
        if not todo:
            release(held); write(OUT/'no_cutover.json', dict(status='NO_PENDING_UNQUALIFIED_TASKS', held=held)); return
        index = todo[0]
        job = submit(OUT/'first_worker_submitted.json', ['--hold', '--partition=dev_accelerated', f'--array={index}%1', str(WORKER)])
        write(OUT/'transition.json', dict(authority=bind(AUTH), held=held, first_index=index, first_worker_job=job))
        follow = submit(OUT/'finish_submitted.json', ['--dependency=afterany:'+job, '--kill-on-invalid-dep=yes', str(LAUNCH), 'finish'])
        command(['scontrol', 'release', job])
        scheduled = True
        event('FIRST_NATURAL_CACHED_WORKER_SUBMITTED', index=index, job=job, afterany_verifier=follow)
    finally:
        if not scheduled:
            if job:
                command(['scancel', job], check=False)
                if (OUT/'finish_submitted.json').exists():
                    # afterany restores the originals only after this writer
                    # has stopped, including an ambiguous release response.
                    event('FIRST_WORKER_ABORTED_AWAITING_ROLLBACK', job=job)
                else:
                    d = show(job)
                    assert d is None or d['JobState'] in ('CANCELLED', 'COMPLETING')
                    release(held)
            else:
                release(held)


def finish():
    t = read(OUT/'transition.json'); held = t['held']; committed = False; retired = []; next_job = None
    try:
        index = t['first_index']
        receipt = DATA/f'query{index:03d}/exact_cache_execution.json'
        if not qualified(index) or not receipt.exists():
            write(OUT/'rollback.json', dict(status='FIRST_CACHED_WORKER_NOT_QUALIFIED_ORIGINAL_QUEUE_RESTORED', first_index=index))
            event('FIRST_WORKER_FAILED_RESTORE_ORIGINAL_QUEUE', index=index); return
        execution = read(receipt); assert execution['status'] == 'QUALIFIED_EXACT_CACHE_QUERY_COMPLETE'
        assert execution['execution']['execution_authority'] == bind(EXEC_AUTH)
        if not DISPATCH_AUTH.exists(): command([str(PYTHON), str(DISPATCH_PROGRAM), 'prepare'])
        # Prepare the new chain under hold before retiring the original queue.
        next_job = submit(OUT/'cached_dispatch_submitted.json', ['--hold', '--dependency=afterany:5157033', '--kill-on-invalid-dep=yes', str(DISPATCH_LAUNCH)])
        for job in held:
            d = show(job)
            if not d or d['JobState'] in ('COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT'): continue
            owned(job, d)
            assert d['JobState'] == 'PENDING' and d['Priority'] == '0', ('REFUSE_CANCEL_NONHELD', job, d['JobState'])
            command(['scancel', job]); retired.append(job)
        for job in retired:
            d = show(job); assert d is None or d['JobState'] in ('CANCELLED', 'COMPLETING'), ('CANCEL_NOT_CONFIRMED', job)
        command(['scontrol', 'release', next_job])
        d = show(next_job); assert d is None or d['JobState'] != 'PENDING' or d.get('Reason') != 'JobHeldUser'
        committed = True
        write(OUT/'complete.json', dict(status='COORDINATE_CACHE_CUTOVER_COMPLETE', authority=bind(AUTH), qualification=bind(QUALIFICATION),
              first_natural_query=bind(receipt), retired_pending_jobs=retired, cached_dispatch_job=next_job,
              preserved_completed_queries=sum(qualified(i) for i in range(593))))
        event('CACHED_CHAIN_RELEASED', next_dispatch=next_job, retired_pending=len(retired))
    finally:
        if not committed:
            release(held)
            if next_job:
                if retired:
                    # Complete the handoff after any surviving original tasks
                    # end; qualified queries are skipped by the new dispatcher.
                    command(['scontrol', 'release', next_job])
                    event('PARTIAL_RETIREMENT_RECOVERED_BY_AFTERANY', job=next_job, retired=retired)
                else:
                    command(['scancel', next_job], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'start', 'finish')); args = p.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        def interrupted(signum, frame):
            raise RuntimeError('INTERRUPTED_SIGNAL_' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        assert os.environ.get('SLURM_JOB_ID')
        a = read(AUTH)
        for b in [a['execution'], *a['sources']]: checked(b)
        q = read(QUALIFICATION); assert q['status'] == 'ROMA_FULL128_EXACT_CACHE_PASS' and q['authority'] == a['execution']; checked(q['payload'])
        start() if args.stage == 'start' else finish()
