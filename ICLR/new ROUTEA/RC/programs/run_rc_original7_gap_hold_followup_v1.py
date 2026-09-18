#!/usr/bin/env python3
"""Exploratory fixed ORIGINAL7 plus one exact TRAIN128 GAP HOLD calibration.

The earlier content-primary OOF gate failed. This separately authorized GAP
control follow-up is performance calibration, not evidence for content HYP.
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

OUT = ROOT / 'results/rc_original7_gap_hold_followup_v1'
PREFLIGHT = ROOT / 'results/rc_original7_gap_hold_followup_v1_preflight'
PLAN = ROOT / 'plan/RC_ORIGINAL7_GAP_HOLD_EXPLORATORY_FOLLOWUP_V1_20260910.md'
AUTH = ROOT / 'registry/rc_original7_gap_hold_followup_authority_v1_20260910.json'
LAUNCH = ROOT / 'slurm/rc_original7_gap_hold_followup_v1_dev_cpuonly_20m.sbatch'
SOLVER = ROOT / 'programs/run_rc_train128_hold_lift_exact_oof4_v1.py'
SOLVER_SHA = 'ba8e58d7ded20686483a05cbc5102bd361b2912ac2e4d4bcd857fd68f662ee71'
OLD_INPUT = ROOT / 'programs/run_rc_train128_disagreement_oof4_v1.py'
OLD_INPUT_SHA = 'd7302eca66c7c6469a7f5ef99a77afeae8af4b25c3a253a1bdcbb2da7af3a715'
INPUT = ROOT / 'results/rc_original7_train128_inputs_v1'
META = ROOT / 'results/rc_original7_train128_manifest_v1'
HEAD = ROOT / 'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
HEAD_SHA = '42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b'
BASE_PARAMETER_SHA = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
GALLERY_IMAGES = ROOT.parents[2] / 'dailymed/data/box_flat_20000_images/data/raw_images'
MAX_BITS = 0x7fefffffffffffff
DEADLINE = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
PHASE = None
HASH_DEPTH = 0
BLOCKED = []
ARM = 'C_PAIRED'
CONTRACT = {
    'scope': 'exploratory performance follow-up of GAP control after CONTENT OOF primary failed',
    'model': 'ORIGINAL7_ec7_plus_single_GAP_HOLD_alpha', 'base_parameter_sha256': BASE_PARAMETER_SHA,
    'TRAIN_queries': 128, 'new_parameters': 1, 'base_refits': 0, 'optimizer_steps': 0,
    'fit': 'same exact GAP solver on all TRAIN128; preserve all TRAIN BASE correct, maximize rescue, minimum binary64 alpha',
    'h': 'constant 1; no F or new evidence features',
    'base_switch': 'unchanged complete logits and action',
    'base_hold': 'only lift original highest challenger by fl(m+fl(alpha*1)); physical row ties',
    'alpha_zero': 'exact original logits including signed zero',
    'alpha_source': 'one full-TRAIN fit shared by both EVAL32 and EVAL128',
    'no_EVAL_coefficient_selection': True, 'EVAL_is_already_opened': True,
    'claim': 'exploratory joint-model decision calibration, not new content HYP proof',
}


def bootstrap_bound_functions():
    text = SOLVER.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != SOLVER_SHA:
        raise RuntimeError('EXACT_SOLVER_SOURCE_PIN')
    names = {'need', 'sha', 'bind', 'checked', 'read', 'write', 'hx', 'matrix_hex', 'runtime',
             'from_bits', 'bits_of', 'canonical', 'threshold', 'threshold_independent',
             'verify_threshold', 'top_index', 'lifted', 'solve_training', 'serialize_prediction'}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError('COMPLETE_EXACT_SOLVER_AST')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOLVER), 'exec'), globals())


bootstrap_bound_functions()


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    text = str(path).lower()
    if any(token in text for token in ('/grozi/', 'd1_mi', 'd1-mi', 'direction_capacity', 'angle_capacity',
                                       '/rc_opened_', 'origin_linear_projection')):
        BLOCKED.append(str(path))
        raise PermissionError('PROTECTED_OR_ORACLE_INPUT_FORBIDDEN')
    # Evaluation paths unlock only after fit validation; roles unlock only
    # after the complete prediction seal and its fresh independent replay.
    if ROOT / 'results' in path.parents:
        allowed = OUT in path.parents or PREFLIGHT in path.parents or path in (
            INPUT / 'validation.json', INPUT / 'feature_records.json', META / 'worker_manifest.json',
            META / 'independent_metadata_validation.json', META / 'curator_roles.json')
        if path in EVAL_PATHS:
            allowed = EVAL_RELEASED and PHASE in ('predict', 'validate-predictions', 'join', 'validate-result')
        if path in POST_PATHS:
            allowed = POST_RELEASED and PHASE in ('join', 'validate-result')
        forbidden = not allowed
        if not HASH_DEPTH and PHASE == 'preflight':
            forbidden |= path in (INPUT / 'feature_records.json', META / 'curator_roles.json')
        if forbidden:
            BLOCKED.append(str(path))
            raise PermissionError('FIT_ONLY_TRAIN_INPUT_BOUNDARY:' + str(path))


def input_operators():
    need(sha(OLD_INPUT) == OLD_INPUT_SHA, 'BOUND_QUALIFIED_TRAIN_INPUT_OPERATORS')
    names = {'qualified_rows', 'gallery_labels'}
    nodes = [n for n in ast.parse(OLD_INPUT.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names, 'QUALIFIED_INPUT_AST')
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OLD_INPUT), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def fixed_head():
    need(sha(HEAD) == HEAD_SHA, 'FROZEN_EC7_HEAD_SEAL_BYTES')
    seal = read(HEAD)
    need(seal['family'] == 'NATIVE7' and seal['arm'] == ARM and seal['parameter_sha256'] == BASE_PARAMETER_SHA,
         'EXACT_ORIGINAL7_HEAD')
    weight = torch.tensor(list(map(float.fromhex, seal['weight_binary64'])), dtype=torch.float64)
    bias = float.fromhex(seal['bias_binary64'])
    values = {'weight': [float(x) for x in weight], 'bias': bias}
    actual = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    need(actual == BASE_PARAMETER_SHA and weight.shape == (6,), 'EXACT_EC7_PARAMETER_VALUES')
    return weight, bias, {'weight_binary64': seal['weight_binary64'], 'bias_binary64': seal['bias_binary64'],
                          'parameter_sha256': BASE_PARAMETER_SHA, 'source_seal': bind(HEAD)}


def state(row, weight, bias):
    # GAP has h=1. No content/F input is read or fabricated for this solver.
    logits = row['X'] @ weight + bias
    need(torch.isfinite(logits).all(), 'FINITE_ORIGINAL7_LOGITS')
    index = top_index(logits, row)
    margin = float(logits[index])
    top = row['challenger_positions'][index]
    return {'logits': logits, 'top_index': index, 'top_position': top, 'm': margin,
            'final_position': top if margin > 0 else row['base_winner_position']}


def training_sources():
    paths = {'program': PROGRAM, 'launcher': LAUNCH, 'plan': PLAN, 'solver_program': SOLVER,
             'qualified_input_program': OLD_INPUT, 'original7_head_seal': HEAD,
             'train_input_validation': INPUT / 'validation.json', 'train_features': INPUT / 'feature_records.json',
             'train_worker': META / 'worker_manifest.json', 'train_curator': META / 'curator_roles.json',
             'train_metadata_validation': META / 'independent_metadata_validation.json',
             'feature_core': Path(FC.__file__), 'eval_helper': EVAL_HELPER, 'evaluation_metric_core': EVAL_CORE,
             'gallery_identity_program': ROOT / 'src/rc_aslo_xf/gallery_identity_repair.py',
             'gallery_identity_registry': ROOT / 'registry/gallery_identity_repair_v1.json',
             'gallery_identity_contract': ROOT / 'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'}
    return {name: bind(path) for name, path in paths.items()}


def require_authority():
    need(os.environ.get('SLURM_JOB_ID') and datetime.now(timezone.utc) < DEADLINE, 'SLURM_AND_DEADLINE_REQUIRED')
    authority = read(AUTH)
    need(authority['status'] == 'ORIGINAL7_GAP_HOLD_EXPLORATORY_FOLLOWUP_AUTHORIZED' and
         authority['contract'] == CONTRACT and authority['cutoff_UTC'] == DEADLINE.isoformat(), 'AUTHORITY_CONTRACT')
    need(PHASE in authority['allowed_stages'], 'PHASE_NOT_AUTHORIZED')
    need(authority['training_sources'] == training_sources(), 'TRAIN_SOURCE_BINDINGS')
    preflight = read(checked(authority['preflight']))
    need(preflight['status'] == 'ORIGINAL7_GAP_HOLD_FOLLOWUP_SYNTHETIC_PREFLIGHT_PASS' and
         preflight['program'] == bind(PROGRAM) and preflight['contract'] == CONTRACT and
         preflight['runtime'] == runtime() and preflight['eval_helper'] == bind(EVAL_HELPER), 'FROZEN_PREFLIGHT')
    return bind(AUTH)


def train_rows():
    need(sha(INPUT / 'validation.json') == '770b9f9ce46f431c7bbe8479e320934c9d1dacdf23f3c3eea650f2812c1a77bf',
         'TRAIN_INPUT_QUALIFICATION_PIN')
    need(sha(META / 'curator_roles.json') == '240be57ffe824e8e029e83052a8b608166782edfdf1df8327bec164621de8ca1',
         'TRAIN_CURATOR_PIN')
    ops = input_operators()
    rows = ops['qualified_rows']()
    roles = read(META / 'curator_roles.json')
    metadata = read(META / 'independent_metadata_validation.json')
    need(metadata['status'] == 'TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS' and
         metadata['curator_ledger'] == bind(META / 'curator_roles.json') and
         metadata['worker_manifest'] == bind(META / 'worker_manifest.json'), 'TRAIN_METADATA_CHAIN')
    need(roles['status'] == 'CURATOR_ONLY_ORIGINAL7_TRAIN128_ROLES_FROZEN' and roles['training_labels_only'] is True,
         'TRAIN_LABELS_ONLY')
    need(len(rows) == len(roles['records']) == 128, 'ALL128_TRAIN_IMAGES')
    labels, mapping = ops['gallery_labels']()
    for row, role in zip(rows, roles['records'], strict=True):
        need(row['query_id'] == role['query_id'] and row['execution_ordinal'] == role['execution_ordinal'] and
             row['source_image_sha256'] == role['source_image_sha256'] and role['role'] == 'TRAIN', 'TRAIN_ROLE_IMAGE_AXIS')
        positions = [p for p, physical in enumerate(row['candidate_physical_rows']) if labels[physical] == role['identity']]
        need(len(positions) <= 1, 'NATURAL_C128_NO_TARGET_INSERTION')
        row['target_position'] = positions[0] if positions else None
        row['original_query_id'] = role['original_query_id']
    return rows, {'input_validation': bind(INPUT / 'validation.json'), 'train_curator': bind(META / 'curator_roles.json'),
                  'train_metadata_validation': bind(META / 'independent_metadata_validation.json'),
                  'gallery_mapping_sha256': mapping, 'query_count': 128}


def train_predictions(rows, weight, bias, alpha):
    results = []
    for row in rows:
        base = state(row, weight, bias)
        values = lifted(base['logits'], row, alpha, 1.)
        original = serialize_prediction(row, base['logits'])
        adjusted = serialize_prediction(row, values)
        results.append({'query_id': row['query_id'], 'execution_ordinal': row['execution_ordinal'],
                        'BASE7': original, 'GAP_HOLD1': adjusted})
    return results


def fresh(phase):
    nonce = uuid.uuid4().hex
    env = dict(os.environ, RC_ORIGINAL7_GAP_FRESH_NONCE=nonce)
    subprocess.run([sys.executable, str(PROGRAM), '--phase', phase, '--nonce', nonce], env=env, check=True)


def fit(replay=False, nonce=None):
    authority = require_authority()
    if replay:
        need(nonce and os.environ.get('RC_ORIGINAL7_GAP_FRESH_NONCE') == nonce, 'FRESH_SOLVER_NONCE')
    weight, bias, base = fixed_head()
    rows, closure = train_rows()
    solver = solve_training(rows, weight, bias, 'GAP_HOLD1', independent=replay)
    parameters = {'status': 'ORIGINAL7_GAP_SINGLE_ALPHA_TRAIN_FIT', 'base': base,
                  'alpha_binary64': solver['alpha_binary64'], 'alpha_bits': solver['alpha_bits'],
                  'solver': solver, 'contract': CONTRACT, 'new_parameter_count': 1,
                  'base_refits': 0, 'optimizer_steps': 0, 'EVAL_reads': 0}
    predictions = train_predictions(rows, weight, bias, float.fromhex(solver['alpha_binary64']))
    if replay:
        seal = read(OUT / 'fit_seal.json')
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['closure'] == closure,
             'IDENTICAL_FRESH_FIT_INPUTS')
        need(parameters == read(checked(seal['parameters'])), 'INDEPENDENT_EXACT_FULL_TRAIN_SOLVER_REPLAY')
        need(predictions == read(checked(seal['train_predictions'])), 'TRAIN_ALL127_ACTION_LOGIT_REPLAY')
        need(os.getpid() != seal['process_PID'], 'FRESH_SOLVER_PROCESS')
        write(OUT / 'fit_validation.json', {'status': 'ORIGINAL7_GAP_FULL_TRAIN_FRESH_EXACT_SOLVER_PASS',
              'authority': authority, 'program': bind(PROGRAM), 'seal': bind(OUT / 'fit_seal.json'),
              'parameters': seal['parameters'], 'fresh_process_nonce': nonce, 'process_PID': os.getpid(),
              'new_parameter_count': 1, 'train_query_count': 128, 'train_logit_replay_count': 128*127*2,
              'all_TRAIN_original_correct_retained': True, 'alpha_binary64': solver['alpha_binary64'],
              'base_parameter_sha256': BASE_PARAMETER_SHA, 'optimizer_steps': 0, 'base_refits': 0,
              'EVAL_reads': 0, 'forbidden_read_attempts': len(BLOCKED), 'runtime': runtime()})
    else:
        write(OUT / 'parameters.json', parameters)
        write(OUT / 'train_predictions.json', predictions)
        write(OUT / 'fit_seal.json', {'status': 'ORIGINAL7_GAP_TRAIN_PARAMETERS_SEALED_BEFORE_EVAL',
              'authority': authority, 'program': bind(PROGRAM), 'closure': closure,
              'parameters': bind(OUT / 'parameters.json'), 'train_predictions': bind(OUT / 'train_predictions.json'),
              'process_PID': os.getpid(), 'runtime': runtime(), 'EVAL_reads': 0,
              'optimizer_steps': 0, 'base_refits': 0, 'forbidden_read_attempts': len(BLOCKED)})
        fresh('validate-fit')
    print(json.dumps({'phase': PHASE, 'status': 'PASS', 'alpha_binary64': solver['alpha_binary64'],
                      'TRAIN_rescues': solver['maximal_training_rescue_count']}), flush=True)


def preflight():
    fixed_head()
    public, post = evaluation_sources()
    need(len(public) == 73 and len(post) == 7, 'EVAL_INTERFACE_BINDING_COUNTS')
    for margin, h in ((0., 1.), (-0., 1.), (-from_bits(1), 1.), (-1., 1.), (-from_bits(MAX_BITS), 1.)):
        one, two = threshold(margin, h), threshold_independent(margin, h)
        need(one == two, 'SAME_BOUND_EXACT_SOLVER')
        verify_threshold(margin, h, one)
    # No dF/F field: this caller must remain GAP-only.
    row = {'query_id': 'synthetic', 'execution_ordinal': 0, 'candidate_physical_rows': list(range(128)),
           'base_winner_position': 0, 'challenger_positions': list(range(1, 128)), 'target_position': 1,
           'X': torch.full((127, 6), -1., dtype=torch.float64)}
    weight = torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.float64)
    primary = solve_training([row], weight, 0., 'GAP_HOLD1')
    independent = solve_training([row], weight, 0., 'GAP_HOLD1', independent=True)
    need(primary == independent and primary['maximal_training_rescue_count'] == 1, 'GAP_ONLY_NO_CONTENT_FIELD_SOLVER')
    protected = dict(row, query_id='protected', target_position=0)
    blocked = solve_training([row, protected], weight, 0., 'GAP_HOLD1')
    need(blocked['alpha_bits'] == 0 and blocked['maximal_training_rescue_count'] == 0, 'FULL_TRAIN_CORRECT_SET_HARD_CONSTRAINT')
    value = {'status': 'ORIGINAL7_GAP_HOLD_FOLLOWUP_SYNTHETIC_PREFLIGHT_PASS',
             'program': bind(PROGRAM), 'launcher': bind(LAUNCH), 'plan': bind(PLAN), 'solver_program': bind(SOLVER),
             'head_seal': bind(HEAD), 'base_parameter_sha256': BASE_PARAMETER_SHA,
             'eval_helper': bind(EVAL_HELPER), 'evaluation_metric_core': bind(EVAL_CORE),
             'contract': CONTRACT, 'runtime': runtime(), 'natural_feature_reads': 0, 'natural_solver_calls': 0,
             'EVAL_reads': 0, 'base_refits': 0, 'optimizer_steps': 0, 'forbidden_read_attempts': len(BLOCKED)}
    path = PREFLIGHT / sha(PROGRAM) / 'result.json'
    write(path, value)
    print(json.dumps({'status': value['status'], 'preflight': bind(path)}), flush=True)


def main():
    global PHASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'fit', 'validate-fit', 'predict', 'validate-predictions', 'join', 'validate-result', 'run'), required=True)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    PHASE = args.phase
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    sys.addaudithook(audit)
    with torch.inference_mode():
        if PHASE == 'preflight':
            preflight()
        elif PHASE in ('fit', 'validate-fit'):
            fit(PHASE == 'validate-fit', args.nonce)
        elif PHASE in ('predict', 'validate-predictions'):
            predict_all(PHASE == 'validate-predictions', args.nonce)
        elif PHASE in ('join', 'validate-result'):
            join(PHASE == 'validate-result', args.nonce)
        else:
            require_authority()
            for stage in ('fit', 'predict', 'join'):
                subprocess.run([sys.executable, str(PROGRAM), '--phase', stage], check=True)


EVAL_HELPER = ROOT / 'programs/rc_original7_gap_hold_eval_v1.py'
exec(compile(EVAL_HELPER.read_text(), str(EVAL_HELPER), 'exec'), globals())


if __name__ == '__main__':
    main()
