#!/usr/bin/env python3
"""Versioned H593 POST experiment dispatch; no model/label computation here."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_postllm_h593_v1'
AUTH = ROOT / 'registry/rc_postllm_h593_authority_v1_20260924.json'
LAUNCH = ROOT / 'slurm/rc_postllm_h593_v1.sbatch'
PY = ROOT.parents[2] / '.venv-colpali/bin/python'
ARMS = ['POST_REAL', 'POST_CONSTANT', 'POST_SHUFFLED']


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked(b):
    if bind(b['path']) != b:
        raise ValueError(f'source SHA changed: {b["path"]}')
    return Path(b['path'])


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.tmp.{os.getpid()}')
    t.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    os.replace(t, p)


def freeze():
    if AUTH.exists():
        guard()
        print(json.dumps({'authority': bind(AUTH), 'already_frozen': True}))
        return
    m = read(OUT / 'input_manifest.json')
    intake = read(OUT / 'intake_audit.json')
    validation = read(OUT / 'input_manifest_validation.json')
    old = read(ROOT / 'registry/rc_postllm_m_v1_authority_20260924.json')
    new_files = [
        Path(__file__), LAUNCH,
        ROOT / 'plan/RC_POSTLLM_H593_FULL_EXPANSION_V1_20260924.md',
        ROOT / 'programs/prepare_rc_postllm_h593_v1.py',
        ROOT / 'programs/cache_rc_postllm_h593_v1.py',
        ROOT / 'programs/run_rc_postllm_h593_v1.py',
        ROOT / 'programs/join_rc_postllm_h593_v1.py',
        ROOT / 'tests/test_postllm_h593_v1.py',
        ROOT / 'tests/test_postllm_h593_join_v1.py',
    ]
    shared = [
        'rc_postllm_m_backend_v1.py', 'rc_prellm_m_scaled_adapter_v4.py',
        'rc_prellm_m_adapter_v1.py', 'run_rc_internal_m_condition_scale_v4.py',
        'run_rc_prellm_m_pilot_v1.py', 'run_rc_internal_m_v4_probe_v1.py',
    ]
    a = dict(
        schema_version=1, status='H593_POSTLLM_GROUPED_FIVEFOLD_AUTHORIZED',
        user_authorization='2026-09-24 先把内部改造，从8扩展到全量。',
        manifest=bind(OUT / 'input_manifest.json'),
        intake=bind(OUT / 'intake_audit.json'),
        input_validation=bind(OUT / 'input_manifest_validation.json'),
        model=old['model'], model_source_validation=old['model_source_validation'],
        attention='sdpa', processor_backend='torchvision',
        epochs=8, passes=8, seed=17, bottleneck=16, residual_scale=0.1,
        condition_gain=math.sqrt(3584), condition_shift=1,
        adapter_lr=3e-4, head_lr=0.03, weight_decay=1e-3, clip_norm=1.,
        warmstart_steps=2000, external_steps=2000, candidates=128,
        arms=ARMS, folds=5, queries=593, natural_target_recall=570,
        direct_M_in_head=False, checkpoint_every=16,
        worker_budget=400, max_requeues=48, cache_shards=50,
        evaluation_policy='Final fixed eight-pass endpoint; each original held query counted once, including target-absent rows. Opened development OOF, not untouched confirmation.',
        initialization='Fresh seed17 zero-output residual plus fold-TRAIN-only zero-init INTERNAL3 warmstart; no trained pilot state reused.',
        diagnostic_interventions=['constant', 'shuffled'],
        source_model=bind(ROOT / 'registry/rc_postllm_m_v1_authority_20260924.json'),
        code_sources=[bind(p) for p in new_files] + [bind(ROOT / 'programs' / s) for s in shared],
    )
    # Complete fold and dataset invariants are produced by the data preparer;
    # this dispatcher pins their bytes rather than inferring new folds.
    if (len(m['rows']) != 593 or len(m['folds']) != 5
            or validation['status'] != 'H593_MANIFEST_593_C128_FIVEFOLD_PASS'
            or validation['manifest'] != a['manifest']
            or not validation['train_and_held_images_disjoint']
            or not intake['all_required_token_files_sha_verified']
            or intake['legacy_hidden_reusable'] + intake['missing_query_hidden'] != 593):
        raise ValueError('complete H593 input qualification required')
    held = [q for fd in m['folds'].values() for q in fd['held_query_ids']]
    if len(held) != 593 or len(set(held)) != 593:
        raise ValueError('each original held query must occur exactly once')
    for fd in m['folds'].values():
        if set(fd['held_query_ids']) & set(fd['train_all_query_ids']):
            raise ValueError('TRAIN/held query overlap')
        checked(fd['train_labels'])
    write(AUTH, a)
    print(json.dumps({'authority': bind(AUTH), 'status': a['status']}))


def guard():
    a = read(AUTH)
    for b in a['code_sources'] + [a['manifest'], a['intake'], a['input_validation']]:
        checked(b)
    return a


def job(name, role, *, dependencies=(), array=None, cpu=False, pilot=False):
    path = OUT / 'dispatch' / (name + '.json')
    if path.exists():
        receipt = read(path)
        if receipt['authority'] != bind(AUTH):
            raise ValueError('dispatch authority mismatch')
        return receipt['job_id']
    cmd = ['sbatch', '--parsable', '--job-name=post593_' + name]
    if cpu:
        cmd += ['--partition=cpuonly', '--gres=none', '--mem=16G']
    elif pilot:
        cmd += ['--partition=dev_accelerated,accelerated', '--mem=64G']
    elif role == 'cache':
        cmd += ['--mem=64G']
    if array is not None:
        cmd += ['--array=' + array]
    if dependencies:
        cmd += ['--dependency=afterok:' + ':'.join(dependencies)]
    cmd += [str(LAUNCH), role]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        # Keep a successful parent eligible for same-ID requeue if a later
        # scheduler submission fails after an earlier child was submitted.
        # The existing receipt makes that earlier submission idempotent.
        write(OUT / 'dispatch' / (name + '.submission_failure.json'),
              dict(name=name, command=cmd, returncode=result.returncode,
                   stdout=result.stdout, stderr=result.stderr, authority=bind(AUTH)))
        print(json.dumps({'submission_retry_required': name, 'stderr': result.stderr}),
              file=sys.stderr, flush=True)
        raise SystemExit(75)
    jid = result.stdout.strip().split(';')[0]
    if not jid.isdigit():
        raise ValueError('unexpected sbatch result: ' + result.stdout)
    write(path, dict(job_id=jid, command=cmd, authority=bind(AUTH), role=role,
                     array=array, dependencies=list(dependencies)))
    print(json.dumps({'submitted': name, 'job_id': jid, 'role': role}), flush=True)
    return jid


def advance(stage):
    a = guard()
    d = OUT / 'dispatch'
    d.mkdir(parents=True, exist_ok=True)
    with (d / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if stage == 'submit':
            job('cache_pilot', 'cache_pilot', pilot=True)
        elif stage == 'after_cache_pilot':
            require_status(OUT / 'encoder_cache/pilot_validation.json', 'H593_QUERY_HIDDEN_PILOT_PASS')
            parent = read(d / 'cache_pilot.json')['job_id']
            cache = job('cache', 'cache', dependencies=[parent], array='0-49%50')
            job('cache_validate', 'cache_validate', dependencies=[cache], cpu=True)
        elif stage == 'after_cache_validate':
            seal = require_status(OUT / 'encoder_cache/validation.json', 'H593_QUERY_HIDDEN_ALL593_PASS')
            if seal['count'] != 593 or seal['missing']:
                raise ValueError('all 593 hidden caches required')
            cache = read(d / 'cache_validate.json')['job_id']
            warm = job('warm', 'warm', dependencies=[cache], array='0-4%5', cpu=True)
            job('train_pilot', 'train_pilot', dependencies=[cache, warm], pilot=True)
        elif stage == 'after_train_pilot':
            seal = require_status(OUT / 'fold0/POST_REAL/pilot_validation.json', 'H593_POST_FULL128_UPDATE_RESUME_PASS')
            if seal['candidate_count'] != 128 or seal['step'] != 1:
                raise ValueError('full-C128 first update qualification required')
            for fold in range(5):
                warm = require_status(OUT / f'fold{fold}/warm/validation.json', 'FOLD_FRESH_WARM_AND_EXTERNAL_HEADS_PASS')
                for b in warm['heads'].values():
                    checked(b)
            parent = read(d / 'train_pilot.json')['job_id']
            train = job('train', 'train', dependencies=[parent], array='0-14%15')
            job('join', 'join', dependencies=[train], cpu=True)
        else:
            raise ValueError(stage)


def require_status(path, status):
    seal = read(path)
    if seal['status'] != status or seal['authority'] != bind(AUTH):
        raise ValueError('qualification required: ' + str(path))
    return seal


def worker(role, index, budget):
    guard()
    base = [str(PY), '-u']
    encoder = str(ROOT / 'programs/cache_rc_postllm_h593_v1.py')
    trainer = str(ROOT / 'programs/run_rc_postllm_h593_v1.py')
    follow = None
    if role in ('cache', 'cache_pilot', 'cache_validate'):
        stage = {'cache': 'cache', 'cache_pilot': 'pilot', 'cache_validate': 'validate'}[role]
        cmd = base + [encoder, stage, '--shard', str(index), '--shards', '50', '--budget', str(budget)]
        follow = {'cache_pilot': 'after_cache_pilot', 'cache_validate': 'after_cache_validate'}.get(role)
    elif role == 'warm':
        cmd = base + [trainer, 'warm', '--fold', str(index), '--device', 'cpu', '--cpu-cache-gb', '4', '--budget', str(budget)]
    elif role == 'train_pilot':
        cmd = base + [trainer, 'pilot', '--fold', '0', '--arm', ARMS[0], '--budget', str(budget)]
        follow = 'after_train_pilot'
    elif role == 'train':
        if not 0 <= index < 15:
            raise ValueError('expected 0..14 training array')
        cmd = base + [trainer, 'worker', '--fold', str(index // 3), '--arm', ARMS[index % 3], '--budget', str(budget)]
    elif role == 'join':
        cmd = base + [str(ROOT / 'programs/join_rc_postllm_h593_v1.py')]
    else:
        raise ValueError(role)
    result = subprocess.run(cmd)
    if result.returncode:
        raise SystemExit(result.returncode)
    if follow:
        advance(follow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'submit', 'after_cache_pilot', 'after_cache_validate', 'after_train_pilot', 'worker', 'status'])
    ap.add_argument('--role')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--budget', type=float, default=400)
    args = ap.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'worker':
        worker(args.role, args.index, args.budget)
    elif args.stage == 'status':
        jobs = {p.stem: read(p) for p in sorted((OUT / 'dispatch').glob('*.json'))
                if not p.name.endswith('.submission_failure.json')}
        intake = read(OUT / 'intake_audit.json')
        new_cache = list((OUT / 'encoder_cache').glob('H593-*/validation.json'))
        fits = []
        for fold in range(5):
            for arm in ARMS:
                d = OUT / f'fold{fold}' / arm
                fit = read(d / 'fit_validation.json') if (d / 'fit_validation.json').exists() else None
                steps = sorted((d / 'steps').glob('*.json'))
                fits.append(dict(fold=fold, arm=arm, fit_complete=fit is not None,
                                 completed_updates=fit['steps'] if fit else (int(steps[-1].stem) if steps else 0),
                                 held_native_predictions=len(list((d / 'held/native').glob('*.json')))))
        print(json.dumps(dict(jobs=jobs, cached_query_hidden=intake['legacy_hidden_reusable']+len(new_cache),
                              required_hidden=593, models=fits,
                              joined=(OUT / 'validation.json').exists()), ensure_ascii=False))
    else:
        advance(args.stage)


if __name__ == '__main__':
    main()
