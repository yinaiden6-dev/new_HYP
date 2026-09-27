#!/usr/bin/env python3
"""One authorized, resume-aware completion/publication/seed-confirmation chain.

No held evaluation is opened until the independently validated join exists.
An explicit pre-result gate permits at most one subsequent confirmation round.
Operational receipts live outside the experiment and Git export trees.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
PY = WS / '.venv-colpali/bin/python'
FIRST = RC / 'results/rc_token_competition_f128_v2'
CONFIRM = RC / 'results/rc_token_competition_f128_confirm_v1'
OPS = WS / 'operations/token128_completion_20260927'
REPO = WS / 'github_exports/new_HYP_token128_publication_20260927'
PUB = WS / 'github_exports/token_publication_records'
PROGRAMS = RC / 'programs'
PASS_FIRST = 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS'
PASS_NEXT = 'TOKEN_COMPETITION_F128_CONFIRM_ALL30_INDEPENDENT_JOIN_PASS'
GATE_PASS = 'TOKEN_COMPETITION_F128_SEED_CONFIRMATION_GATE_PASS'
GATE_NO = 'TOKEN_COMPETITION_F128_SEED_CONFIRMATION_GATE_NO_GO'
PROTOCOL_SHA = 'f7b9349ca2e268de497a5a3693beefd3ab3c7e57a51478e384da952761987143'
SOURCES = [
    'programs/complete_token128_pipeline_20260927.py',
    'programs/analyze_token_competition_completed_v2.py',
    'programs/decide_token_competition_promotion_v1.py',
    'programs/publish_token_competition_results_v2.py',
    'programs/prepare_token_competition_confirm_v1.py',
    'programs/submit_token_competition_confirm_v1.py',
    'programs/token_competition_confirm_common_v1.py',
    'programs/token_competition_confirm_data_v1.py',
    'programs/run_token_competition_confirm_v1.py',
    'programs/run_token_competition_confirm_stage_v1.py',
    'programs/join_token_competition_confirm_v1.py',
    'programs/analyze_token_competition_confirm_v1.py',
    'programs/check_token_competition_confirm_launcher_v1.py',
    'programs/place_token128_chain_dev_20260927.py',
    'slurm/token_competition_confirm_v1.sbatch',
    'plan/RC_TOKEN_COMPETITION_F128_CONFIRMATION_AND_PUBLICATION_20260927.md',
    'results/rc_token_competition_f128_v2/promotion_policy.json',
]


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def check(binding):
    if bind(binding['path']) != binding:
        raise ValueError('Bound artifact changed: ' + binding['path'])


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def event(**value):
    with (OPS / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), **value}, ensure_ascii=False) + '\n')


def state(status, **values):
    put(OPS / 'status.json', {'utc': now(), 'status': status,
        'pid': os.getpid(), 'host': socket.gethostname(), **values})


def verify_contract():
    manifest = read(OPS / 'execution_manifest.json')
    if manifest['allowed_followup_rounds'] != 1 or manifest['repository'] != str(REPO):
        raise ValueError('Execution scope changed')
    for binding in manifest['sources'].values(): check(binding)
    if bind(FIRST / 'protocol.json')['sha256'] != PROTOCOL_SHA:
        raise ValueError('Original scientific protocol changed')
    return manifest


def run_step(name, arguments, artifacts, timeout=1800):
    """Persist successful outputs; restarts do not rewrite an earlier analysis."""
    verify_contract()
    receipt = OPS / 'steps' / (name + '.json')
    if receipt.exists():
        value = read(receipt)
        if value['command'] != arguments:
            raise ValueError('Completed command changed: ' + name)
        for output in value['outputs']: check(output)
        return value
    state('EXECUTING_' + name.upper(), command=arguments)
    log = OPS / 'logs' / (name + '.txt')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write('\n' + now() + '\n')
        f.flush()
        result = subprocess.run(arguments, cwd=RC, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; inspect {log}')
    value = {'status': 'STEP_VERIFIED', 'utc': now(), 'command': arguments,
             'outputs': [bind(p) for p in artifacts], 'log': str(log)}
    put(receipt, value)
    event(event='STEP_VERIFIED', step=name, outputs=value['outputs'])
    return value


def publication(root, name):
    publisher = str(PROGRAMS / 'publish_token_competition_results_v2.py')
    common = [str(PY), publisher, '--root', str(root), '--repo', str(REPO)]
    receipt = PUB / root.name / 'publication_receipt.json'
    # --push prepares its own new snapshot and rewrites prepared.json. A prior
    # successful push is the durable checkpoint; mutable preparation metadata
    # must not invalidate it when this controller restarts.
    push_step = OPS / 'steps' / (name + '_push.json')
    if not push_step.exists():
        run_step(name + '_prepare', common + ['--prepare-only'], [])
    run_step(name + '_push', common + ['--push'], [receipt], timeout=3600)
    value = read(receipt)
    if (value.get('status') != 'TOKEN_COMPETITION_GITHUB_PUBLICATION_VERIFIED' or
        value.get('remote_verified') is not True or value['commit'] != value['remote_main'] or
        value.get('experiment_root') != str(root) or
        value.get('scientific_validation_sha256') != bind(root/'validation.json')['sha256']):
        raise ValueError('Remote publication was not verified')
    return value


def completed(root, expected):
    path = root / 'validation.json'
    if not path.exists(): return False
    value = read(path)
    if value.get('status') != expected:
        raise ValueError('Unexpected final validation: ' + str(value.get('status')))
    return True


def progress(root):
    rows = []
    for path in sorted(root.glob('fold*/seed*/*/progress.json')):
        value = read(path)
        rows.append({'fit': str(path.parent.relative_to(root)), 'status': value.get('status'),
                     'stage': value.get('stage'), 'epoch': value.get('epoch'),
                     'updates': value.get('updates')})
    return {'complete': sum(r['status'] == 'COMPLETE' for r in rows), 'fits': rows}


def job_failure(join_id):
    result = subprocess.run(['sacct', '-X', '-j', str(join_id), '-n', '-P',
                             '--format=JobIDRaw,State,ExitCode'],
                            capture_output=True, text=True, timeout=30, check=True)
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 3 or fields[0] != str(join_id): continue
        status = fields[1].split()[0].rstrip('+')
        if status in {'FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL',
                      'DEADLINE', 'TIMEOUT'}:
            raise RuntimeError('Final join stopped: ' + line)
        return status
    return None


def dev_placement(submission):
    """Reuse audited pending-only placement with this chain's exact CPU IDs."""
    path = PROGRAMS / 'place_token128_chain_dev_20260927.py'
    spec = importlib.util.spec_from_file_location('token_confirmation_dev_placement', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = submission['stages']
    module.JOBS = {f"{jobs['train']['job_id']}_{i}": 'token128c_train' for i in range(30)}
    module.JOBS[jobs['join']['job_id']] = 'token128c_join'
    module.OUT = OPS / 'confirmation_dev_placement'
    module.OUT.mkdir(parents=True, exist_ok=True)
    return module


def wait_for(root, expected, join_id, deadline, placement=None):
    last_accounting = 0.0
    transient_errors = 0
    completed_without_gate = 0
    while not completed(root, expected):
        verify_contract()
        if time.monotonic() > deadline:
            raise TimeoutError('Completion controller reached its registered seven-day limit')
        try:
            if placement is not None:
                placement.tick()
            if time.monotonic() - last_accounting > 120:
                scheduler_status = job_failure(join_id)
                last_accounting = time.monotonic()
                completed_without_gate = (completed_without_gate + 1
                                          if scheduler_status == 'COMPLETED' else 0)
                # Allow a second accounting interval for shared-file visibility.
                if completed_without_gate >= 2 and not completed(root, expected):
                    raise RuntimeError('Join completed without its final validation: ' + str(join_id))
            transient_errors = 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            transient_errors += 1
            event(event='SCHEDULER_RETRY', count=transient_errors, error=repr(exc))
            if transient_errors >= 5:
                raise RuntimeError('Scheduler queries failed five consecutive times') from exc
        state('WAITING_FOR_INDEPENDENT_JOIN', experiment=str(root), join_job=join_id,
              progress=progress(root), scheduler_query_errors=transient_errors)
        time.sleep(30)
    event(event='INDEPENDENT_JOIN_PRESENT', experiment=str(root), validation=bind(root/'validation.json'))


def pipeline():
    manifest = verify_contract()
    sealed_at = dt.datetime.fromisoformat(manifest['utc'])
    elapsed = (dt.datetime.now(dt.timezone.utc) - sealed_at).total_seconds()
    deadline = time.monotonic() + max(0.0, 7 * 86400 - elapsed)
    wait_for(FIRST, PASS_FIRST, '5167555', deadline)
    run_step('initial_analysis', [str(PY), str(PROGRAMS/'analyze_token_competition_completed_v2.py'),
        '--root', str(FIRST)], [FIRST/'analysis/analysis_summary.json'])
    run_step('promotion_decision', [str(PY), str(PROGRAMS/'decide_token_competition_promotion_v1.py'),
        '--root', str(FIRST)], [FIRST/'analysis/promotion_decision.json'])
    first_pub = publication(FIRST, 'initial_publication')
    gate_path = FIRST/'analysis/promotion_decision.json'
    gate = read(gate_path)
    if gate['status'] == GATE_NO:
        state('COMPLETE_NO_FOLLOWUP_GATE', initial_publication=first_pub,
              promotion=bind(gate_path), new_experiments_submitted=False)
        return
    if gate['status'] != GATE_PASS:
        raise ValueError('Only the registered positive gate permits confirmation')
    run_step('confirmation_prepare', [str(PY), str(PROGRAMS/'prepare_token_competition_confirm_v1.py'),
        '--gate', str(gate_path)], [CONFIRM/'protocol.json', CONFIRM/'preparation_validation.json',
        CONFIRM/'source_engineering.json', CONFIRM/'launcher_engineering.json'])
    run_step('confirmation_submission_dryrun', [str(PY), str(PROGRAMS/'submit_token_competition_confirm_v1.py'),
        '--dry-run'], [])
    run_step('confirmation_submission', [str(PY), str(PROGRAMS/'submit_token_competition_confirm_v1.py')],
        [CONFIRM/'submission.json'])
    submitted = read(CONFIRM/'submission.json')
    if submitted['status'] != 'COMPLETE_CONFIRMATION_CHAIN_SUBMITTED':
        raise ValueError('Incomplete confirmation dependency chain')
    if submitted['new_reader_fits'] != 30 or submitted['seeds'] != [1, 2]:
        raise ValueError('Unexpected confirmation scope')
    placement = dev_placement(submitted)
    wait_for(CONFIRM, PASS_NEXT, submitted['stages']['join']['job_id'], deadline, placement)
    run_step('confirmation_analysis', [str(PY), str(PROGRAMS/'analyze_token_competition_confirm_v1.py'),
        '--root', str(CONFIRM)], [CONFIRM/'analysis/analysis_summary.json'])
    final_pub = publication(CONFIRM, 'confirmation_publication')
    state('COMPLETE_INITIAL_AND_CONFIRMATION_PUBLISHED', initial_publication=first_pub,
          confirmation_publication=final_pub, followup_rounds=1, further_submission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true', help='pin reviewed orchestration sources without starting it')
    parser.add_argument('--check', action='store_true', help='verify source bindings without effects')
    args = parser.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    if args.seal:
        path = OPS/'execution_manifest.json'
        if path.exists():
            verify_contract(); print('EXISTING_EXECUTION_MANIFEST_VERIFIED'); return
        if (FIRST/'validation.json').exists():
            raise ValueError('Seal workflow before the first final outcome is available')
        manifest = {'status': 'AUTHORIZED_COMPLETION_PIPELINE_SEALED', 'utc': now(),
            'allowed_followup_rounds': 1, 'repository': str(REPO),
            'remote': 'https://github.com/yinaiden6-dev/new_HYP.git',
            'sources': {name: bind(RC/name) for name in SOURCES},
            'policy': bind(FIRST/'promotion_policy.json'),
            'first_result_observed': False, 'auto_push_authorized': True,
            'user_instruction': 'Analyze completed results, publish them, proceed to one next round only if the gate passes, and publish that completed result.'}
        put(path, manifest); verify_contract(); print('EXECUTION_MANIFEST_SEALED'); return
    if args.check:
        verify_contract(); print('EXECUTION_MANIFEST_PASS'); return
    with (OPS/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pipeline()
        except Exception as exc:
            state('STOPPED_FOR_REVIEW', error=repr(exc))
            event(event='STOPPED_FOR_REVIEW', error=repr(exc))
            raise


if __name__ == '__main__': main()
