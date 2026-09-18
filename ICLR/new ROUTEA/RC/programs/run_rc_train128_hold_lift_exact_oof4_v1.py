#!/usr/bin/env python3
"""Exact TRAIN-only OOF HOLD calibration with immutable prior fold BASE7.

This is post-hoc TRAIN development. No optimizer, backbone call, base refit,
EVAL access, or modification of a BASE SWITCH is permitted.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import uuid

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC

OUT = ROOT / 'results/rc_train128_hold_lift_exact_oof4_v1'
PREFLIGHT = ROOT / 'results/rc_train128_hold_lift_exact_oof4_v1_preflight'
PARENT = ROOT / 'results/rc_train128_disagreement_oof4_v1'
OLD = ROOT / 'programs/run_rc_train128_disagreement_oof4_v1.py'
OLD_SHA = 'd7302eca66c7c6469a7f5ef99a77afeae8af4b25c3a253a1bdcbb2da7af3a715'
OLD_VALIDATION_SHA = '43ccd7e30a50e52f4c4783d4d17304f9b4ec205bac1ca09674e2a82fca20d8c0'
OLD_RESULT_SHA = '9fdf72f68ab3b1444c1961a9d842b4a8d9267bdfebaee30a081f7b7947a79d0d'
INPUT = ROOT / 'results/rc_original7_train128_inputs_v1'
META = ROOT / 'results/rc_original7_train128_manifest_v1'
CACHE = ROOT / 'results/rc_train128_disagreement_features_v1'
PLAN = ROOT / 'plan/RC_TRAIN128_HOLD_LIFT_EXACT_OOF4_V1_20260910.md'
AUTH = ROOT / 'registry/rc_train128_hold_lift_exact_oof4_authority_v1_20260910.json'
LAUNCH = ROOT / 'slurm/rc_train128_hold_lift_exact_oof4_v1_dev_cpuonly_20m.sbatch'
GALLERY_IMAGES = ROOT.parents[2] / 'dailymed/data/box_flat_20000_images/data/raw_images'
MODELS = ('BASE7', 'GAP_HOLD1', 'CONTENT_HOLD1')
ARM = 'C_PAIRED'
MAX_BITS = 0x7fefffffffffffff
DEADLINE = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
PHASE, FOLD = None, None
HASH_DEPTH = 0
BLOCKED = []
CONTRACT = {
    'scope': 'post-hoc TRAIN128 grouped OOF only, prior fold BASE7 retained exactly',
    'models': list(MODELS), 'fold_count': 4, 'query_count': 128, 'group_count': 32,
    'prior_OOF_program_sha256': OLD_SHA, 'new_parameters_per_fold_per_arm': 1,
    'base_switch': 'return all original logits and original action unchanged',
    'base_hold': 'lift only original highest challenger; physical row ties',
    'GAP_HOLD1_h': '1', 'CONTENT_HOLD1_h': 'max(sym(F_top,F_RAW),0)',
    'canonical_lift': 'fl(m + fl(alpha*h)); zero remains HOLD',
    'alpha_domain': 'all finite nonnegative binary64 values',
    'alpha_zero': 'return original logits unchanged, including signed zero',
    'fit': 'retain every training BASE-correct query, maximize new correct training queries, then smallest alpha',
    'threshold': 'first positive canonical lifted logit using monotone positive binary64 bit search',
    'no_finite_crossing': 'None, never infinity',
    'safe_upper': 'minimum predecessor of crossing among BASE-correct training HOLDs; maxfinite if none',
    'alpha': 'maximum crossing among feasible target-is-original-top rescues; zero if none',
    'dtype': 'float64', 'optimizer_steps': 0, 'base_training_steps': 0,
    'roles': 'only current fold train_roles during solving; heldout roles after all fresh seals',
    'gate': ['CONTENT correct images strictly above BASE and GAP',
             'CONTENT equal-group accuracy strictly above BASE and GAP',
             'CONTENT preserves every BASE-correct heldout image'],
    'target_absent': 'retain every TRAIN and heldout image; no positive identity is inserted',
    'claim': 'retrieval-supervised content-assisted HOLD calibration, not new candidate ranking',
    'EVAL_access': False,
}
STATISTICS_CONTRACT = {
    'paired_comparisons': [['CONTENT_HOLD1', 'BASE7'], ['CONTENT_HOLD1', 'GAP_HOLD1'], ['GAP_HOLD1', 'BASE7']],
    'unit': 'source group; equally weighted group mean accuracy difference',
    'bootstrap_draws': 10000, 'bootstrap_seed': 20260910,
    'bootstrap_generator': 'numpy.default_rng PCG64; one shared group-index draw array for all contrasts',
    'bootstrap_interval': 'percentile 95%, numpy.quantile method linear',
    'signflip': 'exact two-sided inclusive tail, abs(sum signed group differences)>=abs(observed sum)',
    'arithmetic': 'rational group differences, common-denominator integer DP; zero groups retained exactly',
    'p_value_is_gate': False,
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    s = str(p).lower()
    hard = any(token in s for token in ('/grozi/', 'd1-mi', 'd1_mi', '/target_join/', '/role_shards/',
                  '/rc_opened_', '/rc_original7_eval', 'direction_capacity', 'angle_capacity', 'origin_linear_projection'))
    if hard:
        BLOCKED.append(str(p))
        raise PermissionError('EVAL_OR_PROTECTED_INPUT_FORBIDDEN')
    if HASH_DEPTH:
        return  # Byte binding only; no role or evaluation payload deserialization.
    forbidden = False
    if ROOT / 'results' in p.parents:
        allowed = (OUT in p.parents or PREFLIGHT in p.parents or CACHE in p.parents or
                   p in (INPUT / 'validation.json', INPUT / 'feature_records.json',
                         META / 'worker_manifest.json', PARENT / 'fold_manifest.json', PARENT / 'validation.json'))
        if PARENT in p.parents and p.parent.name.startswith('fold'):
            allowed = p.name in ('validation.json', 'seal.json', 'parameters.json', 'predictions.json')
            if p.name == 'train_roles.json':
                allowed = PHASE in ('solve-fold', 'validate-fold') and p.parent == PARENT / ('fold%02d' % FOLD)
            if p.name == 'heldout_roles.json':
                allowed = PHASE in ('join', 'validate-join')
        forbidden = not allowed
        if PHASE == 'preflight':
            forbidden |= p == INPUT / 'feature_records.json' or CACHE in p.parents or PARENT in p.parents
    if forbidden:
        BLOCKED.append(str(p))
        raise PermissionError('HOLD_LIFT_TRAIN_FOLD_READ_BOUNDARY:' + str(p))


def sha(path):
    global HASH_DEPTH
    HASH_DEPTH += 1
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    finally:
        HASH_DEPTH -= 1


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(binding):
    p = Path(binding['path'])
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    need(sha(p) == binding['sha256'], 'ARTIFACT_HASH_DRIFT:' + str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'APPEND_ONLY_OUTPUT:' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def hx(value):
    return float(value).hex()


def matrix_hex(value):
    return [[hx(v) for v in row] for row in value]


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'CUDA_used': False, 'flush_denormal': False}


def reused_operators():
    need(sha(OLD) == OLD_SHA, 'FROZEN_PARENT_SOURCE')
    names = {'qualified_rows', 'cache_rows', 'gallery_labels', 'exact_group_signflip', 'paired_group_statistics'}
    nodes = [n for n in ast.parse(OLD.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names, 'BOUND_INPUT_AND_STATS_OPERATORS')
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OLD), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def static_sources():
    paths = {'program': PROGRAM, 'launcher': LAUNCH, 'plan': PLAN, 'prior_program': OLD,
             'prior_root_validation': PARENT / 'validation.json', 'prior_result_bytes': PARENT / 'result.json',
             'prior_authority': ROOT / 'registry/rc_train128_disagreement_oof4_execution_authority_v1_20260910.json',
             'fold_manifest': PARENT / 'fold_manifest.json', 'input_validation': INPUT / 'validation.json',
             'input_features': INPUT / 'feature_records.json', 'worker_manifest': META / 'worker_manifest.json',
             'cache_manifest': CACHE / 'manifest.json', 'cache_validation': CACHE / 'validation.json',
             'cache_features': CACHE / 'features.npz', 'cache_result': CACHE / 'result.json',
             'cache_program': ROOT / 'programs/cache_rc_train128_disagreement_features_v1.py',
             'cache_authority': ROOT / 'registry/rc_train128_disagreement_cache_authority_v1_20260910.json',
             'feature_core': Path(FC.__file__),
             'gallery_identity_program': ROOT / 'src/rc_aslo_xf/gallery_identity_repair.py',
             'gallery_identity_registry': ROOT / 'registry/gallery_identity_repair_v1.json',
             'gallery_identity_contract': ROOT / 'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'}
    for fold in range(4):
        for name in ('train_roles', 'heldout_roles', 'validation', 'seal', 'parameters', 'predictions'):
            paths['prior_fold%d_%s' % (fold, name)] = PARENT / ('fold%02d' % fold) / (name+'.json')
    return {name: bind(path) for name, path in paths.items()}


def require_authority():
    need(os.environ.get('SLURM_JOB_ID') and datetime.now(timezone.utc) < DEADLINE, 'SLURM_AND_DEADLINE_REQUIRED')
    authority = read(AUTH)
    need(authority['status'] == 'TRAIN128_HOLD_LIFT_EXACT_OOF4_AUTHORIZED' and authority['contract'] == CONTRACT,
         'AUTHORITY_CONTRACT')
    need(PHASE in authority['allowed_stages'] and authority['cutoff_UTC'] == DEADLINE.isoformat(), 'AUTHORIZED_PHASE')
    need(authority['source_bindings'] == static_sources(), 'ALL_SOURCE_BINDINGS')
    preflight = read(checked(authority['preflight']))
    need(preflight['status'] == 'TRAIN128_HOLD_LIFT_EXACT_OOF4_SYNTHETIC_PREFLIGHT_PASS' and
         preflight['program'] == bind(PROGRAM) and preflight['contract'] == CONTRACT and preflight['runtime'] == runtime(),
         'FROZEN_PREFLIGHT')
    need(sha(PARENT / 'validation.json') == OLD_VALIDATION_SHA and sha(PARENT / 'result.json') == OLD_RESULT_SHA,
         'EXACT_PRIOR_OOF_QUALIFICATION')
    parent = read(PARENT / 'validation.json')
    need(parent['status'] == 'TRAIN128_DISAGREEMENT_OOF4_ALL_FOLDS_AND_JOIN_REPLAY_PASS' and
         parent['result'] == bind(PARENT / 'result.json') and parent['program'] == bind(OLD) and
         parent['all_heldout_logit_checks'] == 128*127*3 and parent['EVAL_reads'] == parent['forbidden_read_attempts'] == 0,
         'PRIOR_ALL_FOUR_FOLDS_VALIDATED')
    return bind(AUTH)


def from_bits(bits):
    need(0 <= bits <= MAX_BITS, 'NONNEGATIVE_FINITE_BINARY64_BITS')
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def bits_of(value):
    need(math.isfinite(value) and value >= 0, 'NONNEGATIVE_FINITE_ALPHA')
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def canonical(m, alpha, h):
    product = float(alpha) * float(h)
    return float(m) + product


def threshold(m, h):
    need(math.isfinite(m) and m <= 0 and math.isfinite(h) and 0 <= h <= 1, 'THRESHOLD_DOMAIN')
    if h == 0 or canonical(m, from_bits(MAX_BITS), h) <= 0:
        return None
    lo, hi = 0, MAX_BITS
    while lo+1 < hi:
        mid = (lo+hi)//2
        if canonical(m, from_bits(mid), h) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def threshold_independent(m, h):
    need(math.isfinite(m) and m <= 0 and math.isfinite(h) and 0 <= h <= 1, 'INDEPENDENT_THRESHOLD_DOMAIN')
    if h == 0:
        return None
    def positive(bits):
        if bits > MAX_BITS:
            return True  # Exclusive search sentinel; never a candidate parameter.
        alpha = np.array([bits], dtype='>u8').view('>f8')[0]
        with np.errstate(under='ignore'):
            product = np.multiply(np.float64(alpha), np.float64(h))
            score = np.add(np.float64(m), product)
        return bool(score > 0)
    left, right = 0, MAX_BITS+1
    while left < right:
        middle = left+(right-left)//2
        if positive(middle):
            right = middle
        else:
            left = middle+1
    return left if left <= MAX_BITS else None


def verify_threshold(m, h, crossing):
    if crossing is None:
        need(h == 0 or canonical(m, from_bits(MAX_BITS), h) <= 0, 'NO_FINITE_CROSSING_CERTIFICATE')
        return
    need(1 <= crossing <= MAX_BITS, 'STRICTLY_POSITIVE_CROSSING_ALPHA')
    for bits, wanted in ((crossing, True), (crossing-1, False)):
        alpha = from_bits(bits)
        scalar = canonical(m, alpha, h)
        tensor = torch.add(torch.tensor(m, dtype=torch.float64), torch.mul(torch.tensor(alpha, dtype=torch.float64),
                            torch.tensor(h, dtype=torch.float64)))
        need(math.isfinite(scalar) and hx(scalar) == hx(tensor) and (scalar > 0) == wanted,
             'EXACT_BINARY64_CROSSING_AND_PREDECESSOR')


def top_index(logits, row):
    candidates, axis = row['challenger_positions'], row['candidate_physical_rows']
    return max(range(len(candidates)), key=lambda j: (float(logits[j]), -axis[candidates[j]]))


def lifted(logits, row, alpha, h):
    need(math.isfinite(alpha) and alpha >= 0 and 0 <= h <= 1 and math.isfinite(h), 'LIFT_DOMAIN')
    index = top_index(logits, row)
    if float(logits[index]) > 0 or alpha == 0 or h == 0:
        return logits.clone()
    answer = logits.clone()
    # Two actual eager FP64 operators: no fused multiply-add and no reassociation.
    answer[index] = torch.add(logits[index], torch.mul(logits.new_tensor(alpha), logits.new_tensor(h)))
    need(torch.isfinite(answer).all(), 'FINITE_LIFTED_LOGITS')
    need(hx(answer[index]) == hx(canonical(float(logits[index]), alpha, h)), 'SCALAR_TENSOR_CANONICAL_MATCH')
    return answer


def state(row, weight, bias):
    logits = row['X'] @ weight + bias
    need(torch.isfinite(logits).all(), 'FINITE_PRIOR_BASE_LOGITS')
    index = top_index(logits, row)
    m = float(logits[index])
    top = row['challenger_positions'][index]
    final = top if m > 0 else row['base_winner_position']
    return {'logits': logits, 'top_index': index, 'top_position': top, 'm': m, 'final_position': final,
            'h_content': max(float(row['dF'][index]), 0.)}


def solve_training(rows, weight, bias, model, independent=False):
    finder = threshold_independent if independent else threshold
    ledger, upper = [], MAX_BITS
    for row in rows:
        current = state(row, weight, bias)
        correct = current['final_position'] == row['target_position']
        h = 1. if model == 'GAP_HOLD1' else current['h_content']
        need(math.isfinite(h) and 0 <= h <= 1, 'BOUNDED_CONTENT_SYMMETRIC_CONTRAST')
        crossing = finder(current['m'], h) if current['m'] <= 0 else None
        if current['m'] <= 0:
            verify_threshold(current['m'], h, crossing)
        protect = correct and current['m'] <= 0
        rescue = not correct and current['m'] <= 0 and current['top_position'] == row['target_position']
        if protect and crossing is not None:
            upper = min(upper, crossing-1)
        ledger.append({'query_id': row['query_id'], 'execution_ordinal': row['execution_ordinal'],
                       'BASE_correct': correct, 'BASE_action': 'SWITCH' if current['m'] > 0 else 'HOLD',
                       'original_top_position': current['top_position'], 'margin_binary64': hx(current['m']),
                       'h_binary64': hx(h), 'crossing_bits': crossing,
                       'crossing_alpha_binary64': hx(from_bits(crossing)) if crossing is not None else None,
                       'protected_correct_HOLD': protect, 'target_is_original_top_on_wrong_HOLD': rescue})
    feasible = [r for r in ledger if r['target_is_original_top_on_wrong_HOLD'] and
                r['crossing_bits'] is not None and r['crossing_bits'] <= upper]
    chosen = max((r['crossing_bits'] for r in feasible), default=0)
    alpha = from_bits(chosen)
    need(chosen <= upper, 'TRAIN_SAFE_ALPHA_DOMAIN')
    gains, losses = [], []
    for row, original in zip(rows, ledger, strict=True):
        current = state(row, weight, bias)
        values = lifted(current['logits'], row, alpha, float.fromhex(original['h_binary64']))
        index = top_index(values, row)
        final = row['challenger_positions'][index] if float(values[index]) > 0 else row['base_winner_position']
        now_correct = final == row['target_position']
        if original['BASE_correct'] and not now_correct:
            losses.append(row['query_id'])
        if not original['BASE_correct'] and now_correct:
            gains.append(row['query_id'])
    need(not losses and set(gains) == {r['query_id'] for r in feasible}, 'GLOBAL_EMPIRICAL_OPTIMUM_AND_ALL_TRAIN_BASE_CORRECT_RETAINED')
    if chosen:
        need(any(r['crossing_bits'] == chosen for r in feasible), 'SMALLEST_MAXIMIZER_WITNESS')
    return {'alpha_binary64': hx(alpha), 'alpha_bits': chosen, 'safe_upper_bits': upper,
            'safe_upper_binary64': hx(from_bits(upper)), 'training_queries': len(rows),
            'training_BASE_correct': sum(r['BASE_correct'] for r in ledger),
            'maximal_training_rescue_count': len(feasible), 'training_rescue_query_ids': gains,
            'training_BASE_break_count': 0, 'all_training_BASE_correct_retained': True,
            'optimizer_steps': 0, 'parameter_count': 1, 'threshold_ledger': ledger}


def serialize_prediction(row, logits):
    index = top_index(logits, row)
    cs, axis = row['challenger_positions'], row['candidate_physical_rows']
    final = cs[index] if float(logits[index]) > 0 else row['base_winner_position']
    return {'all127_logits_binary64': [hx(v) for v in logits], 'best_challenger_position': cs[index],
            'best_logit_binary64': hx(logits[index]), 'final_position': final, 'final_physical_row': axis[final],
            'action': 'SWITCH' if float(logits[index]) > 0 else 'HOLD'}


def predict(rows, weight, bias, parameters):
    predictions = []
    for row in rows:
        original = state(row, weight, bias)
        models = {'BASE7': serialize_prediction(row, original['logits'])}
        for model in MODELS[1:]:
            alpha = float.fromhex(parameters[model]['alpha_binary64'])
            h = 1. if model == 'GAP_HOLD1' else original['h_content']
            values = lifted(original['logits'], row, alpha, h)
            models[model] = serialize_prediction(row, values)
            for j in range(127):
                if j != original['top_index'] or original['m'] > 0 or alpha == 0 or h == 0:
                    need(hx(values[j]) == hx(original['logits'][j]), 'UNCHANGED_BASE_SWITCH_ZERO_ALPHA_AND_OTHER126_LOGITS')
        predictions.append({'execution_ordinal': row['execution_ordinal'], 'query_id': row['query_id'],
            'source_image_sha256': row['source_image_sha256'], 'candidate_physical_rows': row['candidate_physical_rows'],
            'base_winner_position': row['base_winner_position'], 'predictions': models})
    return predictions


def load_fold(fold):
    operators = reused_operators()
    rows = operators['qualified_rows']()
    cache = operators['cache_rows'](rows)
    manifest = read(PARENT / 'fold_manifest.json')
    directory = PARENT / ('fold%02d' % fold)
    val = read(directory / 'validation.json')
    seal = read(checked(val['seal']))
    parent_root = read(PARENT / 'validation.json')
    need(val['status'] == 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and val['fold'] == fold and
         val['program'] == parent_root['program'] and val['authority'] == parent_root['authority'] and
         val['heldout_label_reads'] == val['EVAL_reads'] == val['forbidden_read_attempts'] == 0, 'PRIOR_BASE_FOLD_QUALIFICATION')
    need(seal['fold'] == fold and seal['authority'] == val['authority'] and seal['program'] == val['program'], 'PRIOR_BASE_SEAL_LINEAGE')
    base = read(checked(seal['parameters']))['BASE7']
    old_predictions = read(checked(seal['predictions']))
    weight = torch.tensor(list(map(float.fromhex, base['weight_binary64'])), dtype=torch.float64)
    bias = float.fromhex(base['bias_binary64'])
    need(weight.shape == (6,) and torch.isfinite(weight).all() and math.isfinite(bias), 'EXACT_SIX_PLUS_BIAS_PARAMETERS')
    roles_path = checked(manifest['role_projections']['fold%d_train_roles.json' % fold])
    need(roles_path == (directory / 'train_roles.json').resolve(), 'SAME_FOLD_TRAIN_ROLES')
    roles = read(roles_path)['records']
    rolemap = {r['query_id']: r for r in roles}
    need(set(rolemap) == {r['query_id'] for r in manifest['records'] if r['fold'] != fold} and
         len({r['group'] for r in roles}) == 24, 'ALL24_TRAINING_GROUPS_ONLY')
    labels, mapping = operators['gallery_labels']()
    train, held = [], []
    for row, assignment in zip(rows, manifest['records'], strict=True):
        need(row['query_id'] == assignment['query_id'], 'OLD_FOLD_AXIS')
        if assignment['fold'] == fold:
            need(row['query_id'] not in rolemap, 'HELDOUT_ROLES_NOT_IN_TRAIN')
            held.append(row)
        else:
            identity = rolemap[row['query_id']]['identity']
            targets = [p for p, physical in enumerate(row['candidate_physical_rows']) if labels[physical] == identity]
            need(len(targets) <= 1, 'NATURAL_C128_IDENTITY_DEDUP')
            row['target_position'] = targets[0] if targets else None
            train.append(row)
    need(len(train)+len(held) == 128 and len(old_predictions) == len(held), 'ALL128_RETAINED')
    for row, previous in zip(held, old_predictions, strict=True):
        need(row['query_id'] == previous['query_id'] and row['execution_ordinal'] == previous['execution_ordinal'] and
             row['candidate_physical_rows'] == previous['candidate_physical_rows'], 'PRIOR_HELDOUT_AXIS')
        need(serialize_prediction(row, state(row, weight, bias)['logits']) == previous['predictions']['BASE7'],
             'PRIOR_BASE_ALL127_LOGITS_AND_ACTION_EXACT')
    closure = {'cache': cache, 'prior_fold_validation': bind(directory / 'validation.json'),
               'prior_fold_seal': bind(directory / 'seal.json'), 'prior_parameters': seal['parameters'],
               'prior_predictions': seal['predictions'], 'train_roles': bind(roles_path),
               'fold_manifest': bind(PARENT / 'fold_manifest.json'), 'gallery_mapping_sha256': mapping,
               'training_ordinals': [r['execution_ordinal'] for r in train],
               'heldout_ordinals': [r['execution_ordinal'] for r in held]}
    return train, held, weight, bias, base, closure


def fresh(phase, fold=None):
    nonce = uuid.uuid4().hex
    env = dict(os.environ, RC_HOLD_LIFT_FRESH_NONCE=nonce)
    command = [sys.executable, str(PROGRAM), '--phase', phase, '--nonce', nonce]
    if fold is not None:
        command.extend(['--fold', str(fold)])
    subprocess.run(command, env=env, check=True)


def solve_fold(fold, replay=False, nonce=None):
    authority = require_authority()
    if replay:
        need(nonce and os.environ.get('RC_HOLD_LIFT_FRESH_NONCE') == nonce, 'FRESH_INDEPENDENT_SOLVER_NONCE')
    train, held, weight, bias, base, closure = load_fold(fold)
    parameters = {'BASE7': base}
    parameters.update({model: solve_training(train, weight, bias, model, independent=replay) for model in MODELS[1:]})
    predictions = predict(held, weight, bias, parameters)
    directory = OUT / ('fold%02d' % fold)
    if replay:
        seal = read(directory / 'seal.json')
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['closure'] == closure,
             'SAME_FRESH_SOLVER_INPUTS')
        need(parameters == read(checked(seal['parameters'])), 'INDEPENDENT_THRESHOLD_SOLVER_PARAMETERS_AND_CERTIFICATES_EXACT')
        need(predictions == read(checked(seal['predictions'])), 'FRESH_ALL127_LOGIT_PREDICTION_REPLAY')
        need(os.getpid() != seal['process_PID'], 'DISTINCT_SOLVER_PROCESS')
        write(directory / 'validation.json', {'status': 'HOLD_LIFT_OOF4_FRESH_EXACT_SOLVER_REPLAY_PASS',
              'authority': authority, 'program': bind(PROGRAM), 'fold': fold, 'seal': bind(directory / 'seal.json'),
              'fresh_process_nonce': nonce, 'process_PID': os.getpid(), 'runtime': runtime(),
              'independent_solver': 'NumPy separate multiply/add and independent lower-bound binary64 search',
              'new_parameters_replayed': 2, 'heldout_logit_checks': len(held)*127*3,
              'prior_BASE_logit_parity_checks': len(held)*127, 'all_training_BASE_correct_retained': True,
              'heldout_label_reads': 0, 'EVAL_reads': 0, 'optimizer_steps': 0, 'base_refits': 0,
              'forbidden_read_attempts': len(BLOCKED)})
    else:
        write(directory / 'parameters.json', parameters)
        write(directory / 'predictions.json', predictions)
        write(directory / 'seal.json', {'status': 'HOLD_LIFT_OOF4_PARAMETERS_PREDICTIONS_SEALED_PREJOIN',
              'authority': authority, 'program': bind(PROGRAM), 'fold': fold, 'closure': closure,
              'parameters': bind(directory / 'parameters.json'), 'predictions': bind(directory / 'predictions.json'),
              'process_PID': os.getpid(), 'runtime': runtime(), 'heldout_label_reads': 0, 'EVAL_reads': 0,
              'optimizer_steps': 0, 'base_refits': 0, 'forbidden_read_attempts': len(BLOCKED)})
        fresh('validate-fold', fold)
    print(json.dumps({'phase': PHASE, 'fold': fold, 'status': 'PASS',
                      'alphas': {m: parameters[m]['alpha_binary64'] for m in MODELS[1:]}}), flush=True)


def joined_result():
    authority = require_authority()
    manifest = read(PARENT / 'fold_manifest.json')
    predictions, fold_sources = [], {}
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        val = read(directory / 'validation.json')
        need(val['status'] == 'HOLD_LIFT_OOF4_FRESH_EXACT_SOLVER_REPLAY_PASS' and val['fold'] == fold and
             val['authority'] == authority and val['program'] == bind(PROGRAM) and
             val['heldout_label_reads'] == val['EVAL_reads'] == val['optimizer_steps'] == val['base_refits'] == val['forbidden_read_attempts'] == 0,
             'ALL_FRESH_FOLD_GATES_BEFORE_HELDOUT_ROLE_READ')
        seal = read(checked(val['seal']))
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['fold'] == fold,
             'SEALED_FOLD_LINEAGE')
        checked(seal['parameters'])
        rows = read(checked(seal['predictions']))
        expected = [r['execution_ordinal'] for r in manifest['records'] if r['fold'] == fold]
        need([r['execution_ordinal'] for r in rows] == expected and val['heldout_logit_checks'] == len(rows)*127*3,
             'EXACT_HELDOUT_GROUP_COVERAGE')
        predictions.extend(dict(r, fold=fold) for r in rows)
        fold_sources[str(fold)] = bind(directory / 'validation.json')
    need(sorted(r['execution_ordinal'] for r in predictions) == list(range(128)), 'ALL128_SEALED_BEFORE_LABELS')
    roles = {}
    for fold in range(4):
        path = checked(manifest['role_projections']['fold%d_heldout_roles.json' % fold])
        need(path == (PARENT / ('fold%02d' % fold) / 'heldout_roles.json').resolve(), 'EXACT_OLD_HELDOUT_ROLES')
        for role in read(path)['records']:
            need(role['query_id'] not in roles, 'DISJOINT_HELDOUT_ROLE_GROUPS')
            roles[role['query_id']] = role
    operators = reused_operators()
    labels, mapping = operators['gallery_labels']()
    rows = []
    for pred in sorted(predictions, key=lambda r: r['execution_ordinal']):
        role = roles[pred['query_id']]
        need(pred['execution_ordinal'] == role['execution_ordinal'] and pred['source_image_sha256'] == role['source_image_sha256'],
             'POST_SEAL_HELDOUT_ROLE_JOIN')
        axis, identity = pred['candidate_physical_rows'], role['identity']
        correct = {m: labels[pred['predictions'][m]['final_physical_row']] == identity for m in MODELS}
        rows.append({'query_id': pred['query_id'], 'original_query_id': role['original_query_id'],
                     'execution_ordinal': pred['execution_ordinal'], 'group': role['group'], 'identity': identity,
                     'fold': pred['fold'], 'RAW_correct': labels[axis[pred['base_winner_position']]] == identity,
                     'target_in_natural_C128': identity in {labels[p] for p in axis}, 'correct': correct})
    groups = sorted({r['group'] for r in rows})
    need(len(groups) == len({r['identity'] for r in rows}) == 32, 'ALL32_GROUPS_IDENTITIES')
    scores, means = {}, {}
    for model in MODELS:
        per_group = {g: Fraction(sum(r['correct'][model] for r in rows if r['group'] == g), sum(r['group'] == g for r in rows)) for g in groups}
        average = sum(per_group.values(), Fraction(0))/32
        means[model] = average
        scores[model] = {'correct': sum(r['correct'][model] for r in rows),
                         'rescue_vs_RAW': sum(r['correct'][model] and not r['RAW_correct'] for r in rows),
                         'break_vs_RAW': sum(not r['correct'][model] and r['RAW_correct'] for r in rows),
                         'equal_group_accuracy': float(average), 'equal_group_accuracy_fraction': str(average),
                         'group_metrics': {g: {'correct': sum(r['correct'][model] for r in rows if r['group'] == g),
                             'count': sum(r['group'] == g for r in rows), 'accuracy_fraction': str(value)} for g, value in per_group.items()}}
    retained = {m: [r['original_query_id'] for r in rows if r['correct']['BASE7'] and not r['correct'][m]] for m in MODELS[1:]}
    gates = {f'count_strictly_above_{m}': scores['CONTENT_HOLD1']['correct'] > scores[m]['correct'] for m in ('BASE7', 'GAP_HOLD1')}
    gates.update({f'equal_group_strictly_above_{m}': means['CONTENT_HOLD1'] > means[m] for m in ('BASE7', 'GAP_HOLD1')})
    gates['all_BASE_correct_retained'] = not retained['CONTENT_HOLD1']
    return {'status': 'TRAIN128_HOLD_LIFT_EXACT_OOF4_JOINED', 'authority': authority, 'program': bind(PROGRAM),
            'contract': CONTRACT, 'posthoc_TRAIN_reuse': True, 'prior_OOF_validation': bind(PARENT / 'validation.json'),
            'fold_validations': fold_sources, 'gallery_mapping_sha256': mapping,
            'query_count': 128, 'group_count': 32, 'identity_count': 32,
            'RAW_correct': sum(r['RAW_correct'] for r in rows), 'target_absent_count': sum(not r['target_in_natural_C128'] for r in rows),
            'scores': scores, 'BASE_correct_losses': retained,
            'fold_metrics': {str(f): {'count': sum(r['fold'] == f for r in rows),
                 'RAW_correct': sum(r['RAW_correct'] for r in rows if r['fold'] == f),
                 'correct': {m: sum(r['correct'][m] for r in rows if r['fold'] == f) for m in MODELS}} for f in range(4)},
            'paired_group_statistics': operators['paired_group_statistics'](rows, groups),
            'gates': gates, 'OOF_gate_pass': all(gates.values()),
            'next_stage': 'SEPARATE_FINAL_FIT_AUTHORITY_ELIGIBLE' if all(gates.values()) else 'STOP_THIS_FAMILY_WITHOUT_EVAL_ACCESS',
            'rows': rows, 'EVAL_reads': 0, 'optimizer_steps': 0, 'base_refits': 0,
            'final_ec7_fit_performed': False, 'forbidden_read_attempts': len(BLOCKED)}


def join(replay=False, nonce=None):
    if replay:
        need(nonce and os.environ.get('RC_HOLD_LIFT_FRESH_NONCE') == nonce, 'FRESH_JOIN_NONCE')
    result = joined_result()
    if replay:
        need(read(OUT / 'result.json') == result, 'FRESH_JOIN_COUNTS_GROUP_STATS_GATES_EXACT')
        write(OUT / 'validation.json', {'status': 'TRAIN128_HOLD_LIFT_EXACT_OOF4_ALL_FOLDS_JOIN_REPLAY_PASS',
              'authority': bind(AUTH), 'program': bind(PROGRAM), 'result': bind(OUT / 'result.json'),
              'process_PID': os.getpid(), 'fresh_process_nonce': nonce, 'query_count': 128, 'group_count': 32,
              'all_heldout_logit_checks': 128*127*3, 'OOF_gate_pass': result['OOF_gate_pass'],
              'EVAL_reads': 0, 'optimizer_steps': 0, 'base_refits': 0, 'forbidden_read_attempts': len(BLOCKED)})
    else:
        write(OUT / 'result.json', result)
        fresh('validate-join')
    print(json.dumps({'phase': PHASE, 'gate': result['OOF_gate_pass'],
                      'correct': {m: result['scores'][m]['correct'] for m in MODELS}}), flush=True)


def preflight():
    operators = reused_operators()
    tiny = from_bits(1)
    cases = [(0., 1.), (-0., 1.), (-tiny, 1.), (-1., 1.), (-1., .5),
             (-tiny, tiny), (0., tiny), (-1., 0.), (-from_bits(MAX_BITS), 1.)]
    for m, h in cases:
        one, two = threshold(m, h), threshold_independent(m, h)
        need(one == two, 'TWO_EXACT_THRESHOLD_SEARCHES')
        verify_threshold(m, h, one)
    row = {'candidate_physical_rows': [8, 7, 2, 9], 'challenger_positions': [1, 2, 3], 'base_winner_position': 0}
    logits = torch.tensor([-1., -1., -2.], dtype=torch.float64)
    need(top_index(logits, row) == 1, 'PHYSICAL_ROW_TIE')
    promoted = lifted(logits, row, 2., 1.)
    need([hx(x) for x in promoted] == [hx(-1.), hx(1.), hx(-2.)], 'ONLY_ORIGINAL_TOP_LIFTED')
    negative_zero = torch.tensor([-0., -1., -2.], dtype=torch.float64)
    switched = torch.tensor([.5, -1., -2.], dtype=torch.float64)
    for x, alpha, h in ((negative_zero, 0., 1.), (negative_zero, 3., 0.), (switched, from_bits(MAX_BITS), 1.)):
        need([hx(v) for v in lifted(x, row, alpha, h)] == [hx(v) for v in x], 'EXACT_ZERO_ALPHA_ZERO_H_BASE_SWITCH')
    weight = torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.float64)
    def synthetic(identifier, target, margin=-1., df=1.):
        x = torch.zeros((127, 6), dtype=torch.float64)
        x[:, 0] = -3.
        x[0, 0] = margin
        return {'query_id': identifier, 'execution_ordinal': 0, 'candidate_physical_rows': list(range(128)),
                'challenger_positions': list(range(1, 128)), 'base_winner_position': 0,
                'target_position': target, 'X': x, 'dF': torch.full((127,), df, dtype=torch.float64)}
    protect, rescue = synthetic('protect', 0), synthetic('rescue', 1)
    for model in MODELS[1:]:
        fixed = solve_training([protect, rescue], weight, 0., model)
        need(fixed['alpha_bits'] == fixed['maximal_training_rescue_count'] == 0, 'SAME_THRESHOLD_PROTECT_RESCUE_INCOMPATIBLE')
        a = solve_training([synthetic('rescue1', 1, -.5), synthetic('rescue2', 1, -1.)], weight, 0., model)
        b = solve_training([synthetic('rescue1', 1, -.5), synthetic('rescue2', 1, -1.)], weight, 0., model, independent=True)
        need(a == b and a['maximal_training_rescue_count'] == 2 and a['alpha_bits'] == threshold(-1., 1.),
             'MAXIMUM_RESCUES_SMALLEST_SHARED_BINARY64_PARAMETER')
    z = solve_training([synthetic('zero-content', 1, -1., 0.)], weight, 0., 'CONTENT_HOLD1')
    need(z['alpha_bits'] == 0 and z['maximal_training_rescue_count'] == 0, 'ZERO_CONTENT_EXACT_FALLBACK')
    stats_rows = [{'group': 'g%d' % (i//2), 'original_query_id': 's%d' % i,
                   'correct': {m: i%2 == 0 for m in MODELS}} for i in range(6)]
    stats = operators['paired_group_statistics'](stats_rows, ['g0', 'g1', 'g2'])
    need(all(x['net_correct_images'] == 0 and x['exact_two_sided_group_signflip']['two_sided_p'] == 1.
             for x in stats['contrasts'].values()), 'FIXED_GROUP_STATISTICS_NEW_ARM_NAMES')
    value = {'status': 'TRAIN128_HOLD_LIFT_EXACT_OOF4_SYNTHETIC_PREFLIGHT_PASS',
             'program': bind(PROGRAM), 'launcher': bind(LAUNCH), 'plan': bind(PLAN), 'prior_program': bind(OLD),
             'contract': CONTRACT, 'statistics_contract': STATISTICS_CONTRACT, 'runtime': runtime(),
             'threshold_domain_cases': len(cases), 'natural_feature_reads': 0, 'natural_solver_calls': 0,
             'training_updates': 0, 'base_refits': 0, 'EVAL_reads': 0, 'forbidden_read_attempts': len(BLOCKED)}
    path = PREFLIGHT / sha(PROGRAM) / 'result.json'
    write(path, value)
    print(json.dumps({'status': value['status'], 'preflight': bind(path)}), flush=True)


def main():
    global PHASE, FOLD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'solve-fold', 'validate-fold', 'join', 'validate-join', 'run'), required=True)
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--nonce')
    args = parser.parse_args()
    PHASE, FOLD = args.phase, args.fold
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    sys.addaudithook(audit)
    with torch.inference_mode():
        if PHASE == 'preflight':
            preflight()
        elif PHASE in ('solve-fold', 'validate-fold'):
            need(FOLD is not None, 'FOLD_REQUIRED')
            solve_fold(FOLD, PHASE == 'validate-fold', args.nonce)
        elif PHASE in ('join', 'validate-join'):
            join(PHASE == 'validate-join', args.nonce)
        else:
            require_authority()
            for fold in range(4):
                subprocess.run([sys.executable, str(PROGRAM), '--phase', 'solve-fold', '--fold', str(fold)], check=True)
            subprocess.run([sys.executable, str(PROGRAM), '--phase', 'join'], check=True)


if __name__ == '__main__':
    main()
