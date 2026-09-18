#!/usr/bin/env python3
"""Measure existing training functions; timing-only parameters never become models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_six_cause_loss_binding_v1 as L
H, M = L.H, L.H.M
OUT = ROOT / 'results/rc_head_training_time_v1_20260918'
AUTH = ROOT / 'registry/rc_head_training_time_authority_v1_20260918.json'
ORIGINAL = ROOT / 'registry/rc_new_hyp_external_head_freeze_authority_v1_20260913.json'
SIZES = (32, 128, 593)
MODELS = ('COST1', 'CE')


def prepare():
    original = M.read(ORIGINAL)
    for b in original['code_sources'].values():
        M.checked(b)
    sources = dict(program=M.bind(Path(__file__).resolve()),
        launcher=M.bind(ROOT / 'slurm/rc_head_training_time_v1.sbatch'),
        historical_authority=M.bind(ORIGINAL),
        gallery=M.bind(ROOT / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'),
        historical_log=M.bind(ROOT / 'logs/rc_original7_train128_readout-5139291.out'),
        historical_training_closure=M.bind(ROOT / 'results/rc_original7_train128_readout_v1/training_input_closure.json'),
        plan=M.bind(ROOT / 'plan/RC_HEAD_TRAINING_TIME_SCALING_V1_20260918.md'))
    sources.update({'original_' + k:v for k,v in original['code_sources'].items()})
    sources.update({'public_' + k:v for k,v in original['public_sources'].items()})
    sources['feature_authority'] = M.bind(ROOT / 'registry/rc_new_hyp593_feature_authority_v1_20260911.json')
    for model in MODELS:
        sources[model + '_frozen_head'] = M.bind(ROOT / f'results/rc_new_hyp_external_head_freeze_v1/{model}/head.json')
    M.write(AUTH, dict(status='TIMING_ONLY_AUTHORIZED', sources=sources, features=original['features'],
            pool_sizes=list(SIZES), models=list(MODELS), repeats=3,
            selection='Prefix of the existing outcome-blind H593 execution order; nested 32/128/593 pools',
            loss_eligibility='Original rule: omit undefined positive loss for target absent from natural C128',
            initialization='zero', steps=2000, precision='float64', threads=8,
            optimizer='Original AdamW lr=0.03 weight_decay=0.001',
            timer='perf_counter around the unchanged L.train call; CPU synchronous; warmed process',
            excluded='Queue, interpreter startup, cache I/O, feature construction, encoder/RoMa inference and evaluation',
            warmup='One original 2000-step call per model on the 32-query pool, excluded from the fit',
            result_scope='Runtime benchmark, not a new accuracy experiment or model replacement',
            fit='Separate affine OLS T(N)=a+b*N per model, using three median times; N is effective training queries',
            extrapolation=False))
    print(json.dumps(dict(status='TIMING_PREPARED', authority=M.bind(AUTH))))


def guard():
    a = M.read(AUTH)
    M.need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allowed = {AUTH.resolve()}
    for b in a['sources'].values():
        allowed.add(M.checked(b).resolve())
    for bs in a['features']:
        allowed.update(M.checked(b).resolve() for b in bs.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve(); text = str(p).lower()
        M.need(not any(x in text for x in ('d1_mi', 'd1-mi', 'formal392', '/reports/', '/target_join/', 'gisc_prerecall_universe')), 'PROTECTED_READ')
        if ROOT / 'results' in p.parents:
            M.need(p in allowed or OUT in p.parents, 'UNLISTED_RESULT:' + text)
    sys.addaudithook(audit)
    return a


def run():
    a = guard()
    start = time.perf_counter()
    rows, _ = H.features()
    roles = {r['query_id']:r for r in M.read(a['sources']['public_curator']['path'])['records']}
    gallery = M.read(a['sources']['gallery']['path'])
    labels = [r['identity'] for r in gallery['records']]
    data = {n:H.batch(rows[:n], roles, labels) for n in SIZES}
    meta = {}
    for n in SIZES:
        eligible = [r['query_id'] for r in rows[:n] if H.target_position(r,roles[r['query_id']]['identity'],labels) >= -1]
        meta[str(n)] = dict(pool_size=n, effective_queries=len(data[n]['y']),
                            absent_from_C128=data[n]['target_absent_count'],
                            query_ids=[r['query_id'] for r in rows[:n]], train_query_ids=eligible,
                            X_sha256=hashlib.sha256(data[n]['X'].numpy().tobytes()).hexdigest(),
                            y_sha256=hashlib.sha256(data[n]['y'].numpy().tobytes()).hexdigest())
    M.need(meta['593']['effective_queries'] == 570 and meta['593']['absent_from_C128'] == 23, 'FULL_H593_ELIGIBILITY')
    expected = {m:M.read(a['sources'][m+'_frozen_head']['path']) for m in MODELS}
    for m in MODELS:
        M.need(meta['593']['train_query_ids'] == expected[m]['train_query_ids'], 'EXACT_HISTORICAL_FULL_TRAIN_ORDER')
    runtime = dict(hostname=platform.node(), platform=platform.platform(), cpu_model=next(
        (s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown'),
        python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'), job_id=os.environ['SLURM_JOB_ID'],
        partition=os.environ.get('SLURM_JOB_PARTITION'), training_device='CPU', GPU_used=False)
    M.write(OUT / 'data_closure.json', dict(authority=M.bind(AUTH), pools=meta, runtime=runtime,
            cache_load_and_batch_seconds=time.perf_counter()-start, evaluation_reads=0))
    warmup = []
    for m in MODELS:
        t = time.perf_counter(); theta = L.train(data[32]['X'], data[32]['y'], m)
        warmup.append(dict(model=m, seconds=time.perf_counter()-t, steps=2000))
    records = []; parameters = {}
    configurations = [(n,m) for n in SIZES for m in MODELS]
    for repeat in range(a['repeats']):
        # Rotate the complete configuration order; every scale moves earlier/later.
        order = configurations[2*repeat:] + configurations[:2*repeat]
        for n,m in order:
            begin_cpu = time.process_time(); begin = time.perf_counter()
            theta = L.train(data[n]['X'], data[n]['y'], m)
            seconds = time.perf_counter()-begin; cpu_seconds = time.process_time()-begin_cpu
            values = [float(t).hex() for t in theta]
            key = f'{n}_{m}'
            if key in parameters:
                M.need(parameters[key] == values, 'REPEAT_PARAMETER_BITS')
            parameters[key] = values
            historical_error = None
            if n == 593:
                ref = np.array([float.fromhex(t) for t in expected[m]['theta_hex']])
                historical_error = float(abs(theta.numpy()-ref).max())
                M.need(historical_error < 2e-10, 'HISTORICAL_FULL_HEAD_PARAMETER_PARITY')
            record = dict(repeat=repeat, model=m, pool_size=n, effective_queries=len(data[n]['y']),
                absent_from_C128=data[n]['target_absent_count'], steps=2000,
                fit_seconds=seconds, cpu_seconds=cpu_seconds,
                theta_sha256=hashlib.sha256(theta.numpy().tobytes()).hexdigest(),
                historical_head_max_abs_error=historical_error)
            M.write(OUT / f'timing_{repeat}_{n}_{m}.json', record)
            records.append(record)
            print(json.dumps(dict(event='TIMED_FIT_COMPLETE', **record)), flush=True)
    M.write(OUT / 'result.json', dict(status='HEAD_TRAINING_TIMING_COMPLETE', authority=M.bind(AUTH),
        runtime=runtime, pools=meta, records=records, warmup=warmup, repeats=a['repeats'],
        timed_optimizer_updates=len(records)*2000, warmup_optimizer_updates=4000,
        external_queries_read=0, accuracy_evaluation=False, deployment_changed=False,
        parameters_saved_as_models=False, note='Only parameter hashes recorded; existing heads remain unchanged'))
    for m in MODELS:
        M.checked(a['sources'][m+'_frozen_head'])
    M.write(OUT / 'validation.json', dict(status='TIMING_REPEATS_AND_FROZEN_HEAD_PARITY_PASS',
        result=M.bind(OUT / 'result.json'), timed_fits=len(records), expected_fits=18,
        same_node_and_threads=True, old_head_files_unchanged=True,
        within_config_parameter_bits_equal=True, external_queries_read=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare','run']); args = p.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.manual_seed(17)
    globals()[args.stage]()
