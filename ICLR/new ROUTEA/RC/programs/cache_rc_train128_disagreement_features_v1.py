#!/usr/bin/env python3
"""TRAIN128-only content/visibility disagreement cache; no labels or fitting.

The primary process uses Torch max/gather. A fresh validator recomputes every
matrix, selects indices in NumPy, and independently pools the extracted vectors.
Both preserve the existing FP64 sum-then-divide reduction. math.fsum provides
an additional independent pooling check; it is not substituted into the cache.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as TF

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
SOURCE = ROOT / 'results/rc_original7_train128_inputs_v1/validation.json'
SOURCE_SHA = '770b9f9ce46f431c7bbe8479e320934c9d1dacdf23f3c3eea650f2812c1a77bf'
WORKER = ROOT / 'results/rc_original7_train128_manifest_v1/worker_manifest.json'
WORKER_SHA = '50d894c9643ca1ef200c18ba79cf700b598be15c5ea322010853c201fc0f22f7'
LEGACY_SCORE = ROOT / 'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py'
LEGACY_SCORE_SHA = 'fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c'
OUT = ROOT / 'results/rc_train128_disagreement_features_v1'
LAUNCHER = ROOT / 'slurm/rc_train128_disagreement_features_v1_dev_cpuonly_20m.sbatch'
AUTHORITY = ROOT / 'registry/rc_train128_disagreement_cache_authority_v1_20260910.json'
KEYS = ('F', 'A', 'D', 'mean_wr', 'mean_wq')
C4_KEYS = ('real_score', 'visibility_mass', 'query_control_score', 'reference_control_score')
DEADLINE = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
BLOCKED = []
RESULT_ALLOW = {SOURCE.resolve(), WORKER.resolve()}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    s = str(p).lower()
    forbidden = any(v in s for v in ('curator', 'role_shards', 'target_join', 'oracle',
                    'direction_capacity', 'angle_capacity', 'd1_mi', 'd1-mi', '/grozi/'))
    # The source qualification carries private pointers, but none are followed.
    if ROOT / 'results' in p.parents:
        allowed_output = OUT in p.parents or (OUT.parent / (OUT.name + '_preflight')) in p.parents
        forbidden |= p not in RESULT_ALLOW and not allowed_output
    if forbidden:
        BLOCKED.append(str(p))
        raise PermissionError('TRAIN_CACHE_INPUT_BOUNDARY:' + str(p))


sys.addaudithook(audit)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(binding):
    path = Path(binding['path'])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    need(sha(path) == binding['sha256'], 'SOURCE_HASH_DRIFT:' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == content, 'IMMUTABLE_OUTPUT_EXISTS:' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.partial.' + str(os.getpid()))
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def hx(value):
    return float(value).hex()


def tsha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode('ascii') +
        json.dumps(list(value.shape), separators=(',', ':')).encode('ascii') +
        value.view(torch.uint8).numpy().tobytes()).hexdigest()


def mapsha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'CUDA_used': False}


def authority_check(path, stage):
    need(os.environ.get('SLURM_JOB_ID'), 'NATURAL_CACHE_REQUIRES_SLURM')
    need(datetime.now(timezone.utc) < DEADLINE, 'USER_RESEARCH_DEADLINE_REACHED')
    authority = read(path)
    need(authority['status'] == 'NEW_HYP_TRAIN128_DISAGREEMENT_CACHE_AUTHORIZED', 'AUTHORITY_STATUS')
    need(stage in authority['allowed_stages'], 'STAGE_NOT_AUTHORIZED')
    need(authority['cutoff_UTC'] == DEADLINE.isoformat(), 'AUTHORITY_CUTOFF')
    need(authority['query_count'] == 128, 'AUTHORITY_QUERY_COUNT')
    expected = {'cache_program': PROGRAM, 'cache_launcher': LAUNCHER,
                'source_validation': SOURCE, 'source_worker_manifest': WORKER}
    for key, p in expected.items():
        need(authority['source_bindings'][key] == bind(p), 'AUTHORITY_BINDING:' + key)
    for binding in authority['source_bindings'].values():
        checked(binding)
    return bind(path)


def source_inventory():
    need(sha(SOURCE) == SOURCE_SHA and sha(WORKER) == WORKER_SHA, 'QUALIFIED_SOURCE_PIN')
    src, worker = read(SOURCE), read(WORKER)
    need(src['status'] == 'ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS', 'SOURCE_STATUS')
    need(src['query_count'] == 128 and src['candidate_occurrences'] == 16384,
         'COMPLETE_TRAIN128_C128_SOURCE')
    need(src['independent_C4_scalar_checks'] == 65536 and src['all32_original_input_bits_exact'],
         'PRIOR_C4_AND_ORIGINAL32_QUALIFICATION')
    for key in ('curator_reads', 'EVAL_result_reads', 'training_updates', 'head_predictions',
                'candidate_insertions', 'forbidden_read_attempts'):
        need(src[key] == 0, 'SOURCE_BOUNDARY:' + key)
    need(src['worker_manifest'] == bind(WORKER), 'WORKER_SOURCE_BINDING')
    need(worker['role'] == 'TRAIN' and worker['query_count'] == 128, 'WORKER_TRAIN128')
    rows = worker['records']
    need([r['execution_ordinal'] for r in rows] == list(range(128)), 'WORKER_EXECUTION_AXIS')
    need(len({r['query_id'] for r in rows}) == len({r['source_image_sha256'] for r in rows}) == 128,
         'DISTINCT_TRAIN128_IMAGES')
    for family, directory in (('raw_shards', 'rc_original7_train128_token_raw_v1'),
                              ('roma_shards', 'rc_original7_train128_roma_v1')):
        need([s['shard'] for s in src[family]] == list(range(16)), 'COMPLETE_SHARD_AXIS:' + family)
        for item in src[family]:
            for key, name in (('payload', 'payload.pt'), ('receipt', 'receipt.json'),
                              ('validation', 'validation.json')):
                expected = (ROOT / 'results' / directory / ('shard%02d' % item['shard']) / name).resolve()
                need(Path(item[key]['path']).resolve() == expected, 'FROZEN_TRAIN_SOURCE_PATH')
                RESULT_ALLOW.add(expected)
                checked(item[key])
    return src, rows


def verify_shard_envelope(src, shard):
    raw_source, roma_source = src['raw_shards'][shard], src['roma_shards'][shard]
    loaded = []
    for kind, source in (('TOKEN_RAW', raw_source), ('ROMA', roma_source)):
        for key in ('payload', 'receipt', 'validation'):
            checked(source[key])
        receipt, validation = read(source['receipt']['path']), read(source['validation']['path'])
        need(receipt['payload'] == validation['payload'] == source['payload'], 'SHARD_PAYLOAD_CHAIN')
        need(validation['receipt'] == source['receipt'], 'SHARD_RECEIPT_CHAIN')
        need(receipt['authority'] == validation['authority'] == src['authority'], 'SOURCE_AUTHORITY_CHAIN')
        need(receipt['status'] == 'RC_ORIGINAL7_TRAIN128_' + kind + '_SHARD_READY', 'SHARD_READY')
        need(validation['status'] == 'RC_ORIGINAL7_TRAIN128_' + kind + '_CPU_REPLAY_PASS', 'SHARD_REPLAY_PASS')
        loaded.append(torch.load(source['payload']['path'], map_location='cpu', weights_only=True))
    raw, roma = loaded
    need(raw['shard'] == roma['shard'] == shard and len(raw['records']) == len(roma['records']) == 8,
         'EXACT_SHARD_HEADER')
    need(roma['token_source'] == {k: raw_source[k] for k in ('payload', 'receipt', 'validation')},
         'ROMA_TOKEN_SOURCE_CHAIN')
    return raw, roma


def checked_pairs(src, workers, shard):
    raw, roma = verify_shard_envelope(src, shard)
    refs = {int(k): v for k, v in raw['references'].items()}
    for ref in refs.values():
        need(tsha(ref['tokens']) == ref['tokens_sha256'], 'REFERENCE_TOKEN_BYTES')
    for offset, (qrow, rrow) in enumerate(zip(raw['records'], roma['records'], strict=True)):
        ordinal = shard * 8 + offset
        worker = workers[ordinal]
        need(qrow['execution_ordinal'] == rrow['execution_ordinal'] == ordinal, 'QUERY_EXECUTION_ORDER')
        need(qrow['query_id'] == rrow['query_id'] == worker['query_id'], 'OPAQUE_QUERY_JOIN')
        need(qrow['query_source_sha256'] == rrow['query_source_sha256'] == worker['source_image_sha256'],
             'QUERY_IMAGE_BINDING')
        q = qrow['query_tokens']
        need(tsha(q) == qrow['query_tokens_sha256'] == rrow['query_tokens_sha256'], 'QUERY_TOKEN_BYTES')
        need(q.dtype == torch.float16 and q.ndim == 2 and q.shape[1] == 128, 'QUERY_TOKEN_DOMAIN')
        axis = list(qrow['candidate_physical_rows'])
        need(axis == sorted(axis) and len(set(axis)) == 128 and axis == list(rrow['candidate_physical_rows']),
             'FULL_ORIGINAL_C128_AXIS')
        need(tsha(qrow['candidate_raw_scores']) == tsha(rrow['candidate_raw_scores']), 'ORIGINAL_RAW_BITS')
        need(len(rrow['candidates']) == 128, 'COMPLETE_CANDIDATES')
        for position, cand in enumerate(rrow['candidates']):
            ref = refs[axis[position]]
            r, wq, wr = ref['tokens'], cand['query_visibility'], cand['reference_visibility']
            need(cand['candidate_position'] == position and cand['physical_row'] == axis[position],
                 'PHYSICAL_POSITION_BINDING')
            need(cand['reference_tokens_sha256'] == ref['tokens_sha256'], 'REFERENCE_TOKEN_BINDING')
            need(r.dtype == torch.float16 and r.ndim == 2 and r.shape[1] == 128, 'REFERENCE_TOKEN_DOMAIN')
            for w, length in ((wq, len(q)), (wr, len(r))):
                need(w.dtype == torch.float64 and tuple(w.shape) == (length,) and
                     bool(torch.isfinite(w).all()) and bool(((w >= 0) & (w <= 1)).all()), 'WEIGHT_DOMAIN')
            need(mapsha(wq) == cand['query_map_sha256'] == cand['old_scores']['query_map_sha256'] and
                 mapsha(wr) == cand['reference_map_sha256'] == cand['old_scores']['reference_map_sha256'],
                 'SAVED_VISIBILITY_BYTES')
            need(cand['control_shifts'] == {'query': max(1, len(wq)//2), 'reference': max(1, len(wr)//2)},
                 'FROZEN_CONTROL_AXIS_SHIFTS')
            yield ordinal, position, qrow, cand, q, r, wq, wr


def c4_from_similarity(sim, wq, wr):
    def score(a, b):
        local = (sim * b[None]).max(1).values
        mass = torch.sqrt(a.mean() * b.mean())
        return mass * (a * local).sum() / a.sum().clamp_min(1e-12), mass
    real, mass = score(wq, wr)
    query, _ = score(torch.roll(wq, max(1, len(wq)//2)), wr)
    reference, _ = score(wq, torch.roll(wr, max(1, len(wr)//2)))
    return [float(v) for v in (real, mass, query, reference)]


def primary(sim, wq, wr):
    den = wq.sum().clamp_min(1e-12)
    free = sim.max(1).values
    idx = (sim * wr[None]).max(1).indices
    aligned = sim.gather(1, idx[:, None])[:, 0]
    f = (wq * free).sum() / den
    a = (wq * aligned).sum() / den
    return [float(v) for v in (f, a, f-a, wr.mean(), wq.mean())]


def independent(sim, wq, wr):
    # NumPy owns index selection and row extraction, rather than Torch max/gather.
    matrix = sim.numpy()
    rows = np.arange(matrix.shape[0], dtype=np.int64)
    free_positions = np.argmax(matrix, axis=1)
    weighted_matrix = np.multiply(matrix, wr.numpy()[None, :])
    weighted_positions = np.argmax(weighted_matrix, axis=1)
    free = torch.from_numpy(matrix[rows, free_positions].copy())
    aligned = torch.from_numpy(matrix[rows, weighted_positions].copy())
    denominator = torch.clamp(torch.sum(wq), min=1e-12)
    f = torch.sum(torch.mul(free, wq)) / denominator
    a = torch.sum(torch.mul(aligned, wq)) / denominator
    values = [float(f), float(a), float(f-a), float(torch.mean(wr)), float(torch.mean(wq))]
    fsum_den = max(math.fsum(wq.tolist()), 1e-12)
    fsum_f = math.fsum(torch.mul(free, wq).tolist()) / fsum_den
    fsum_a = math.fsum(torch.mul(aligned, wq).tolist()) / fsum_den
    errors = [abs(float(f)-fsum_f), abs(float(a)-fsum_a)]
    need(max(errors) <= 64*np.finfo(np.float64).eps, 'INDEPENDENT_FSUM_POOLING_DISAGREEMENT')
    need(values[2] >= -64*np.finfo(np.float64).eps, 'FREE_MATCH_DOMINATES_ALIGNED_CONTENT')
    return values, max(errors)


def compute(src, workers, replay=False, expected=None):
    features = np.empty((128, 128, 5), dtype=np.float64)
    c4 = np.empty((128, 128, 4), dtype=np.float64)
    axes = np.empty((128, 128), dtype=np.int64)
    meta = []
    max_pool_error = 0.0
    for shard in range(16):
        for ordinal, pos, qrow, cand, q, r, wq, wr in checked_pairs(src, workers, shard):
            sim = TF.normalize(q.to(torch.float64), dim=1) @ TF.normalize(r.to(torch.float64), dim=1).T
            if replay:
                values, error = independent(sim, wq, wr)
                max_pool_error = max(max_pool_error, error)
            else:
                values = primary(sim, wq, wr)
            features[ordinal, pos] = values
            c4[ordinal, pos] = c4_from_similarity(sim, wq, wr)
            need([hx(v) for v in c4[ordinal, pos]] == [hx(cand['old_scores'][k]) for k in C4_KEYS],
                 'SOURCE_C4_BINARY64_MISMATCH:%d:%d' % (ordinal, pos))
            axes[ordinal, pos] = cand['physical_row']
            if replay:
                need(features[ordinal, pos].tobytes() == expected['features'][ordinal, pos].tobytes(),
                     'FRESH_NUMPY_INDEX_FEATURE_BITS:%d:%d' % (ordinal, pos))
                need(c4[ordinal, pos].tobytes() == expected['c4'][ordinal, pos].tobytes(),
                     'FRESH_C4_BITS:%d:%d' % (ordinal, pos))
            if pos == 0:
                meta.append({'query_id': qrow['query_id'], 'execution_ordinal': ordinal,
                             'source_image_sha256': qrow['query_source_sha256'],
                             'query_tokens_sha256': qrow['query_tokens_sha256'],
                             'query_grid_shape': list(qrow['query_grid_shape']),
                             'candidate_physical_rows': list(qrow['candidate_physical_rows'])})
        print(json.dumps({'stage': 'validate' if replay else 'cache', 'completed_shards': shard+1,
                          'candidate_occurrences': (shard+1)*1024}), flush=True)
    need(len(meta) == 128 and np.isfinite(features).all() and np.isfinite(c4).all() and not BLOCKED,
         'WHOLE_TRAIN128_CACHE_VALID')
    arrays = {'features': features, 'c4': c4, 'candidate_physical_rows': axes,
              'query_ids': np.array([r['query_id'] for r in meta], dtype='U32'),
              'execution_ordinals': np.arange(128, dtype=np.int64)}
    arrays.update({key: features[:, :, i].copy() for i, key in enumerate(KEYS)})
    return arrays, meta, max_pool_error


def validate(authority_path, nonce):
    need(nonce and os.environ.get('RC_TRAIN_DISAGREEMENT_REPLAY_NONCE') == nonce, 'FRESH_VALIDATOR_NONCE')
    authority = authority_check(authority_path, 'validate')
    src, workers = source_inventory()
    manifest = read(OUT / 'manifest.json')
    need(manifest['authority'] == authority and manifest['program'] == bind(PROGRAM), 'CACHE_AUTHORITY_PROGRAM')
    need(manifest['source_validation'] == bind(SOURCE), 'CACHE_QUALIFIED_SOURCE')
    cache_path = checked(manifest['cache'])
    with np.load(cache_path, allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    got, meta, error = compute(src, workers, replay=True, expected=expected)
    for key in expected:
        need(expected[key].dtype == got[key].dtype and expected[key].shape == got[key].shape and
             expected[key].tobytes() == got[key].tobytes(), 'COMPLETE_CACHE_REPLAY:' + key)
    need(meta == manifest['records'], 'ALL_QUERY_AXES_AND_TOKEN_BINDINGS_REPLAYED')
    result = {'status': 'TRAIN128_DISAGREEMENT_CACHE_INDEPENDENT_REPLAY_PASS',
              'authority': authority, 'program': bind(PROGRAM), 'manifest': bind(OUT / 'manifest.json'),
              'cache': bind(cache_path), 'source_validation': bind(SOURCE), 'fresh_process_nonce': nonce,
              'fresh_process_PID': os.getpid(), 'parent_PID': os.getppid(), 'runtime': runtime(),
              'query_count': 128, 'candidate_occurrences': 16384, 'feature_scalar_bit_checks': 81920,
              'source_C4_scalar_bit_checks': 65536, 'independent_F_A_D_scalar_checks': 49152,
              'independent_fsum_F_A_checks': 32768, 'maximum_fsum_absolute_error': error,
              'all_original_axes_unchanged': True, 'argmax_tie_rule': 'first physical token position',
              'independent_index_operator': 'numpy.argmax plus NumPy row indexing',
              'pooling': 'FP64 weighted sum, then divide by clamped FP64 weight sum',
              'curator_reads': 0, 'EVAL_reads': 0, 'head_predictions': 0,
              'training_updates': 0, 'model_forward_calls': 0, 'forbidden_read_attempts': len(BLOCKED)}
    write_json(OUT / 'validation.json', result)
    print(json.dumps({'status': result['status'], 'validation': bind(OUT / 'validation.json')}), flush=True)


def cache(authority_path):
    authority = authority_check(authority_path, 'cache')
    need(not (OUT / 'manifest.json').exists(), 'CACHE_ALREADY_SEALED_USE_VALIDATE')
    started = time.monotonic()
    src, workers = source_inventory()
    data, meta, _ = compute(src, workers)
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    write_bytes(OUT / 'features.npz', stream.getvalue())
    value = {'status': 'TRAIN128_DISAGREEMENT_CACHE_SEALED_PENDING_REPLAY', 'authority': authority,
             'program': bind(PROGRAM), 'source_validation': bind(SOURCE), 'worker_manifest': bind(WORKER),
             'raw_shards': src['raw_shards'], 'roma_shards': src['roma_shards'],
             'cache': bind(OUT / 'features.npz'), 'feature_names': list(KEYS), 'C4_names': list(C4_KEYS),
             'feature_shape': [128, 128, 5], 'C4_shape': [128, 128, 4], 'dtype': 'float64',
             'candidate_occurrences': 16384, 'source_C4_scalar_bit_checks': 65536,
             'records': meta, 'runtime': runtime(), 'elapsed_seconds': time.monotonic()-started,
             'formulas': {'F': 'sum(wq * max_j cos) / clamp_min(sum(wq), 1e-12)',
                          'A': 'sum(wq * cos[i, argmax_j(wr[j]*cos[i,j])]) / clamp_min(sum(wq), 1e-12)',
                          'D': 'F - A', 'mean_wr': 'mean(wr)', 'mean_wq': 'mean(wq)'},
             'original_C4_operator': 'mass * sum(wq * max_j(wr*cos)) / clamp_min(sum(wq), 1e-12)',
             'control_means_recomputed_after_roll': True, 'argmax_ties': 'first token position',
             'curator_reads': 0, 'EVAL_reads': 0, 'head_predictions': 0,
             'training_updates': 0, 'model_forward_calls': 0, 'forbidden_read_attempts': len(BLOCKED)}
    write_json(OUT / 'manifest.json', value)
    nonce = uuid.uuid4().hex
    env = dict(os.environ, RC_TRAIN_DISAGREEMENT_REPLAY_NONCE=nonce)
    subprocess.run([sys.executable, str(PROGRAM), '--phase', 'validate', '--authority', str(authority_path),
                    '--nonce', nonce], env=env, check=True)
    need(read(OUT / 'validation.json')['status'] == 'TRAIN128_DISAGREEMENT_CACHE_INDEPENDENT_REPLAY_PASS',
         'FRESH_VALIDATOR_PASS_REQUIRED')
    write_json(OUT / 'result.json', {'status': 'TRAIN128_DISAGREEMENT_CACHE_READY',
               'authority': authority, 'manifest': bind(OUT / 'manifest.json'),
               'cache': bind(OUT / 'features.npz'), 'validation': bind(OUT / 'validation.json'),
               'query_count': 128, 'candidate_occurrences': 16384,
               'claim_level': 'TRAIN_ONLY_INPUT_FEATURE_CACHE_NO_IDENTITY_RESULT',
               'training_updates': 0, 'head_predictions': 0, 'EVAL_reads': 0, 'curator_reads': 0})
    print(json.dumps({'status': 'TRAIN128_DISAGREEMENT_CACHE_READY', 'result': bind(OUT / 'result.json')}), flush=True)


def preflight():
    # Source hashes are checked without deserializing any natural tensor payload.
    source_inventory()
    need(sha(LEGACY_SCORE) == LEGACY_SCORE_SHA, 'LEGACY_SCORE_SOURCE_PIN')
    tree = ast.parse(LEGACY_SCORE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    namespace = {'torch': torch, 'F': TF}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LEGACY_SCORE), 'exec'), namespace)
    literal = namespace['score']
    q = torch.tensor([[1., -2., 3.], [0., 0., 0.], [-1., .5, -2.]], dtype=torch.float16)
    r = torch.tensor([[1., 1., -1.], [-2., 1., 2.], [1., 1., -1.], [0., 0., 0.]], dtype=torch.float16)
    weights = [(torch.tensor([.3, .8, .05], dtype=torch.float64), torch.tensor([.1, .7, .1, 0.], dtype=torch.float64)),
               (torch.zeros(3, dtype=torch.float64), torch.ones(4, dtype=torch.float64)),
               (torch.ones(3, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)),
               (torch.full((3,), 1e-16, dtype=torch.float64), torch.ones(4, dtype=torch.float64))]
    for wq, wr in weights:
        sim = TF.normalize(q.to(torch.float64), dim=1) @ TF.normalize(r.to(torch.float64), dim=1).T
        p, (i, _) = primary(sim, wq, wr), independent(sim, wq, wr)
        need([hx(v) for v in p] == [hx(v) for v in i], 'SYNTHETIC_INDEPENDENT_ARITHMETIC')
        real, mass, _ = literal(q, r, wq, wr)
        legacy = [real, mass, literal(q, r, wq.roll(max(1, len(wq)//2)), wr)[0],
                  literal(q, r, wq, wr.roll(max(1, len(wr)//2)))[0]]
        need([hx(v) for v in c4_from_similarity(sim, wq, wr)] == [hx(v) for v in legacy],
             'SYNTHETIC_LITERAL_C4_BITS')
    result = {'status': 'TRAIN128_DISAGREEMENT_CACHE_PREFLIGHT_PASS', 'program': bind(PROGRAM),
              'launcher': bind(LAUNCHER), 'source_validation': bind(SOURCE), 'worker_manifest': bind(WORKER),
              'legacy_score_source': bind(LEGACY_SCORE), 'runtime': runtime(), 'source_file_hash_checks': 96,
              'synthetic_feature_scalar_bit_checks': 20, 'synthetic_C4_scalar_bit_checks': 16,
              'synthetic_cases': ['mixed signed cosine and tie', 'zero query mass', 'zero reference mass',
                                  'query denominator clamp'], 'natural_tensor_payload_reads': 0,
              'EVAL_reads': 0, 'curator_reads': 0, 'training_updates': 0, 'head_predictions': 0}
    path = OUT.parent / (OUT.name + '_preflight') / sha(PROGRAM) / 'result.json'
    write_json(path, result)
    print(json.dumps({'status': result['status'], 'preflight': bind(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'cache', 'validate'), required=True)
    parser.add_argument('--authority', type=Path, default=AUTHORITY)
    parser.add_argument('--nonce', default='')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        if args.phase == 'preflight':
            preflight()
        elif args.phase == 'cache':
            cache(args.authority)
        else:
            validate(args.authority, args.nonce)


if __name__ == '__main__':
    main()
