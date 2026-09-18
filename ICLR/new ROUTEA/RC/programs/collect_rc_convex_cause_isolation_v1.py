#!/usr/bin/env python3
"""Collect certified loss bounds and independently read out the sealed TRAIN head.

Only EVAL32 and EVAL128 are scored. Opened-cone parameters are never scored.
The collector performs no optimization or training, and requires Slurm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P

TRAIN = ROOT / 'results/rc_convex_train_loss_cause_v1'
UNIT = TRAIN / 'unit_cost96'
DIAG = ROOT / 'results/rc_opened_convex_loss_rescue_cones_v1'
PREV = ROOT / 'results/rc_fixed_panels_train269_group_risk_v1'
OUT = ROOT / 'results/rc_opened_convex_cause_readout_v1'
AUTH = ROOT / 'registry/rc_convex_cause_isolation_authority_v1_20260912.json'
REPORT = ROOT / 'reports/REPORT_CONVEX_CAUSE_ISOLATION_V1_20260912.md'
CORE = ROOT / 'programs/rc_convex_loss_cause_core_v1.py'
MODELS = ('ORIGINAL7', 'TRAIN_CONVEX7', 'TRAIN_UNIT_COST7')
HEX7 = ('0x1.01ffd7d7241b3p+0', '-0x1.e3db1bf7526f8p+1',
        '0x1.6ddd66d0df5d3p+2', '0x1.7a67f045c5bd2p+2',
        '-0x1.3095a38e39e50p-3', '-0x1.e8ea4dd00f544p-3',
        '-0x1.7a55fbee7993ap+0')
EC7 = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'


def need(value, label):
    if not value:
        raise RuntimeError(label)


def path_of(binding):
    path = Path(binding['path'])
    return path if path.is_absolute() else ROOT / path


def checked(binding):
    path = path_of(binding)
    need(M.sha(path) == binding['sha256'], 'SOURCE_DRIFT:' + str(path))
    return path


def equal_binding(left, right):
    return path_of(left).resolve() == path_of(right).resolve() and left['sha256'] == right['sha256']


def indexed(rows, key='query_id'):
    result = {row[key]: row for row in rows}
    need(len(result) == len(rows), 'DUPLICATE_' + key)
    return result


def action(logits, challengers, winner):
    need(len(logits) == len(challengers) and len(logits) > 0, 'ACTION_DIMENSIONS')
    best = max(range(len(logits)), key=lambda i: float(logits[i]))
    return challengers[best] if float(logits[best]) > 0 else winner


def moved_rank(raw, selected, target):
    order = [selected] + [row for row in raw if row != selected]
    need(len(order) == len(raw) == len(set(order)) and set(order) == set(raw), 'MOVE_TO_FRONT_PERMUTATION')
    return order.index(target) + 1


def interval_record(low, high):
    low, high = Fraction(low), Fraction(high)
    need(low <= high, 'ORDERED_EXACT_INTERVAL')
    return {'lower_rational': str(low), 'upper_rational': str(high),
            'lower_display': float(low), 'upper_display': float(high)}


def solver_interval(solver):
    need(Fraction(solver['bound_rational']) == 64, 'FIXED_BOX64')
    need(solver['unbounded_global_optimality_claimed'] is False, 'NO_UNBOUNDED_GLOBAL_CLAIM')
    upper = solver['upper_interval']
    need(upper is not None, 'CERTIFIED_FEASIBLE_UPPER_BOUND_REQUIRED')
    return Fraction(solver['lower_bound_rational']), Fraction(upper['upper_rational'])


def risk_comparison(old, global_bounds, cone_bounds):
    old_l, old_u = old
    global_l, global_u = global_bounds
    cone_l, cone_u = cone_bounds
    if cone_l > old_u:
        status = 'BOX_CONE_ALL_ACTIONS_COST_MORE_THAN_ORIGINAL'
    elif cone_u < old_l and cone_l > global_u:
        status = 'BOX_LOSS_REDUCTION_AND_RESCUE_COMPATIBLE_BUT_NOT_NEAR_OPTIMUM'
    else:
        status = 'INTERVAL_OVERLAP_UNRESOLVED'
    return {'status': status,
            'cone_minus_box_optimum': interval_record(cone_l - global_u, cone_u - global_l),
            'cone_minus_original': interval_record(cone_l - old_u, cone_u - old_l),
            'all_cone_points_worse_than_original_certified': cone_l > old_u,
            'strict_action_with_loss_reduction_exists_certified': cone_u < old_l,
            'positive_risk_separation_from_box_optimum_certified': cone_l > global_u}


def tensor_bits_equal(left, right):
    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and left.shape == right.shape
            and left.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            == right.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def summarize(rows):
    names = list(rows[0]['correct'])
    scores = {}
    for name in names:
        correct = sum(row['correct'][name] for row in rows)
        mrr = sum((Fraction(1, row['ranks'][name]) for row in rows), Fraction(0)) / len(rows)
        scores[name] = {'correct': correct, 'queries': len(rows), 'accuracy': correct / len(rows),
                        'MRR': float(mrr), 'MRR_rational': str(mrr),
                        'switch_count': sum(row['selected_physical_rows'][name] != row['selected_physical_rows']['RAW'] for row in rows)}
    comparisons = {}
    for base, new in (('RAW', 'ORIGINAL7'), ('RAW', 'TRAIN_CONVEX7'), ('ORIGINAL7', 'TRAIN_CONVEX7'),
                      ('RAW', 'TRAIN_UNIT_COST7'), ('ORIGINAL7', 'TRAIN_UNIT_COST7'),
                      ('TRAIN_CONVEX7', 'TRAIN_UNIT_COST7'),
                      ('ORIGINAL7_CBIND', 'TRAIN_CONVEX7_CBIND'), ('TRAIN_CONVEX7_CBIND', 'TRAIN_CONVEX7'),
                      ('ORIGINAL7_CBIND', 'TRAIN_UNIT_COST7_CBIND'), ('TRAIN_UNIT_COST7_CBIND', 'TRAIN_UNIT_COST7')):
        rescue = [row['query_id'] for row in rows if not row['correct'][base] and row['correct'][new]]
        loss = [row['query_id'] for row in rows if row['correct'][base] and not row['correct'][new]]
        groups = defaultdict(list)
        for row in rows:
            groups[row['component']].append(int(row['correct'][new]) - int(row['correct'][base]))
        means = {key: Fraction(sum(values), len(values)) for key, values in groups.items()}
        comparisons[base + '__to__' + new] = {
            'rescue': len(rescue), 'loss': len(loss), 'net': len(rescue) - len(loss),
            'rescue_query_ids': rescue, 'loss_query_ids': loss,
            'all_old_correct_preserved': not loss,
            'equal_component_difference_rational': str(sum(means.values(), Fraction(0)) / len(means)),
            'positive_components': sum(value > 0 for value in means.values()),
            'negative_components': sum(value < 0 for value in means.values()), 'components': len(means)}
    binding = {}
    for name in MODELS:
        rescued = [row for row in rows if not row['correct']['RAW'] and row['correct'][name]]
        retained = sum(row['correct'][name + '_CBIND'] for row in rescued)
        binding[name] = {'REAL_rescues_over_RAW': len(rescued), 'CBIND_retained_REAL_rescues': retained,
                         'CBIND_removed_REAL_rescues': len(rescued) - retained,
                         'REAL_minus_CBIND_correct': scores[name]['correct'] - scores[name + '_CBIND']['correct']}
    return {'population': len(rows), 'recall_C128': sum(row['target_in_C128'] for row in rows),
            'scores': scores, 'comparisons': comparisons, 'C_BIND_rescue_retention': binding, 'rows': rows}


def checked_search(folder, authority_binding, problem_binding):
    search = M.read(folder / 'search.json')
    validation = M.read(folder / 'validation.json')
    need(search['status'] == 'CONVEX_CAUSE_SEARCH_COMPLETED', 'SEARCH_COMPLETE')
    need(validation['status'] == 'CONVEX_CAUSE_EXACT_BOUNDS_REPLAY_PASS', 'FRESH_EXACT_BOUND_REPLAY_PASS')
    need(equal_binding(validation['search'], M.bind(folder / 'search.json')), 'VALIDATION_SEARCH_BINDING')
    need(equal_binding(search['authority'], authority_binding) and equal_binding(validation['authority'], authority_binding), 'SEARCH_AUTHORITY')
    need(equal_binding(search['train_problem'], problem_binding), 'IDENTICAL_TRAIN_PROBLEM')
    need(equal_binding(validation['core'], M.bind(CORE)), 'BOUND_CORE_BINDING')
    need(validation['solver_calls'] == validation['original_head_training_updates'] == 0, 'VALIDATOR_NO_OPTIMIZATION')
    need(isinstance(validation.get('fresh_nonce'), str) and len(validation['fresh_nonce']) == 32, 'FRESH_BOUND_VALIDATOR_NONCE')
    need(search['original_head_training_updates'] == search['new_Adam_updates'] == 0, 'ORIGINAL_HEAD_NOT_RETRAINED')
    need(search['AdamW_decay_not_reinterpreted_as_L2'] is True, 'LOGGED_LOSS_SCOPE')
    lower, upper = solver_interval(search['solver'])
    proof = validation['bounds']
    need(proof['status'] == 'EXACT_RATIONAL_CUT_AND_INTERVAL_BOUND_REPLAY_PASS'
         and proof['solver_calls'] == 0, 'BOUND_PROOF_STATUS')
    need(Fraction(proof['lower_bound_rational']) == lower
         and Fraction(proof['upper_bound_rational']) == upper, 'BOUND_PROOF_EXACT_INTERVAL')
    return search, validation


def main():
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    torch.set_num_threads(1)
    authority = M.read(AUTH)
    ab = M.bind(AUTH)
    expected_eval = {'result': PREV / 'result.json', 'result_validation': PREV / 'result_validation.json',
                     'eval_labels': PREV / 'eval_curator_roles.json', 'panel_manifest': PREV / 'panel_manifest.json'}
    need(set(authority['evaluation_sources']) == set(expected_eval), 'EXACT_EVALUATION_SOURCE_KEYS')
    for name, path in expected_eval.items():
        need(path_of(authority['evaluation_sources'][name]).resolve() == path.resolve(), 'EVALUATION_SOURCE_PATH_' + name)
    labels_open = False
    evaluation_paths = {path.resolve() for path in expected_eval.values()}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        text = str(path).lower()
        need(not any(token in text for token in ('d1-mi', 'd1_mi', 'grozi', 'gisc_prerecall_universe', '/target_join/', 'formal_private')), 'PROTECTED_INPUT')
        need(labels_open or (path not in evaluation_paths and 'curator_roles' not in text), 'PREDICTION_SEAL_BEFORE_LABELS')

    sys.addaudithook(audit)
    for binding in authority['sources'].values():
        checked(binding)
    need(any(equal_binding(binding, M.bind(__file__)) for binding in authority['sources'].values()), 'COLLECTOR_AUTHORITY_PIN')
    preflight = M.read(TRAIN / 'preflight.json')
    need(preflight['status'] == 'CONVEX_CAUSE_PIPELINE_PREFLIGHT_PASS' and equal_binding(preflight['authority'], ab), 'QUALIFIED_PREFLIGHT')
    problem_binding = M.bind(TRAIN / 'train_problem.json')
    problem = M.read(TRAIN / 'train_problem.json')
    need(tuple(problem['old_theta_binary64']) == HEX7, 'FROZEN_ORIGINAL_PARAMETER_BITS')
    train_search, train_validation = checked_search(TRAIN / 'fit', ab, problem_binding)
    need(train_search['stage'] == 'train' and train_search['solver']['diagnostic_only'] is False, 'TRAIN_ONLY_SOLUTION')
    need(train_search['solver']['theta_float_is_exact_primal'] is True, 'BINARY64_TRAIN_PRIMAL')
    seal = M.read(TRAIN / 'prediction_seal.json')
    need(seal['status'] == 'CONVEX_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED', 'ALL160_PREJOIN_PREDICTION_SEAL')
    need(equal_binding(seal['authority'], ab) and equal_binding(seal['train_search'], M.bind(TRAIN / 'fit/search.json')) and equal_binding(seal['train_search_validation'], M.bind(TRAIN / 'fit/validation.json')), 'PREDICTION_SEAL_BINDINGS')
    need(seal['heldout_label_reads'] == seal['original_head_training_updates'] == 0, 'SEALED_TRAIN_NO_EVAL_LABELS')
    need(path_of(seal['payload']).resolve() == (TRAIN / 'predictions.pt').resolve(), 'EXACT_PREDICTION_PAYLOAD')
    checked(seal['payload'])
    unit_fit = M.read(UNIT / 'fit.json')
    unit_validation = M.read(UNIT / 'fit_validation.json')
    need(unit_validation['status'] == 'UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS', 'UNIT_COST_FRESH_FIT_REPLAY')
    need(equal_binding(unit_validation['fit'], M.bind(UNIT / 'fit.json'))
         and equal_binding(unit_validation['authority'], ab)
         and equal_binding(unit_fit['authority'], ab), 'UNIT_COST_FIT_BINDINGS')
    need(equal_binding(unit_validation['input_seal'], M.bind(TRAIN / 'input_seal.json'))
         and equal_binding(unit_fit['input_seal'], M.bind(TRAIN / 'input_seal.json')), 'UNIT_COST_IDENTICAL_MIXED96')
    need(isinstance(unit_validation.get('fresh_nonce'), str) and len(unit_validation['fresh_nonce']) == 32, 'UNIT_COST_FRESH_NONCE')
    need(unit_validation['heldout_label_reads'] == 0, 'UNIT_COST_FIT_NO_HELDOUT_LABELS')
    for item in (unit_fit, unit_validation):
        need(item['original_head_training_updates'] == 0 and item['new_head_training_updates'] == 2000, 'UNIT_COST_ONLY_NEW2000_UPDATES')
    unit_seal = M.read(UNIT / 'prediction_seal.json')
    need(unit_seal['status'] == 'UNIT_COST96_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED', 'UNIT_COST_ALL160_PREJOIN_SEAL')
    need(equal_binding(unit_seal['authority'], ab)
         and equal_binding(unit_seal['unit_cost_fit'], M.bind(UNIT / 'fit.json'))
         and equal_binding(unit_seal['unit_cost_fit_validation'], M.bind(UNIT / 'fit_validation.json')), 'UNIT_COST_PREDICTION_SEAL_BINDINGS')
    need(unit_seal['heldout_label_reads'] == unit_seal['original_head_training_updates'] == 0
         and unit_seal['new_head_training_updates'] == 2000, 'UNIT_COST_PREJOIN_ATTESTATIONS')
    need(path_of(unit_seal['payload']).resolve() == (UNIT / 'predictions.pt').resolve(), 'UNIT_COST_EXACT_PAYLOAD_PATH')
    checked(unit_seal['payload'])
    manifest = M.read(checked(authority['diagnostic_sources']['manifest']))
    need(manifest['cone_count'] == len(manifest['cones']) == 6, 'SIX_FIXED_CONES')
    need(manifest['train_sealed_before_eval_cone_reads'] and manifest['existing_witnesses_all_exactly_revalidated'], 'CONES_PROVENANCE')
    cone_searches = []
    for index, entry in enumerate(manifest['cones']):
        need(entry['index'] == index and entry['cone_id'] == f'cone{index:02d}', 'FIXED_CONE_ORDER')
        binding = entry.get('input', entry.get('binding', entry))
        checked(binding)
        search, validation = checked_search(DIAG / 'solves' / f'cone{index:02d}', ab, problem_binding)
        need(search['stage'] == 'diagnose' and search['cone_index'] == index and search['cone_id'] == entry['cone_id'], 'CONE_SEARCH_ID')
        need(equal_binding(search['cone_input'], binding), 'CONE_INPUT_BINDING')
        need(search['label_aware'] and search['not_a_trainable_model'] and search['solver']['diagnostic_only'], 'QUARANTINED_CONE_ONLY')
        need(search['baseline_objective'] == train_search['baseline_objective'], 'SAME_ORIGINAL_LOSS_INTERVAL')
        need(Fraction(search['solver']['conflict_threshold_rational']) ==
             Fraction(search['baseline_objective']['upper_rational']), 'CONE_THRESHOLD_IS_ORIGINAL_LOSS_UPPER')
        cone_searches.append((entry, search, validation))
    payload = torch.load(checked(seal['payload']), weights_only=True, map_location='cpu')
    need(equal_binding(payload['authority'], ab) and payload['heldout_label_reads'] == payload['original_head_training_updates'] == 0, 'PAYLOAD_PROVENANCE')
    need(equal_binding(payload['train_search'], seal['train_search']) and equal_binding(payload['train_search_validation'], seal['train_search_validation']), 'PAYLOAD_TRAIN_SEAL')
    unit_payload = torch.load(checked(unit_seal['payload']), weights_only=True, map_location='cpu')
    need(equal_binding(unit_payload['authority'], ab)
         and equal_binding(unit_payload['unit_cost_fit'], unit_seal['unit_cost_fit'])
         and equal_binding(unit_payload['unit_cost_fit_validation'], unit_seal['unit_cost_fit_validation']), 'UNIT_COST_PAYLOAD_BINDINGS')
    need(unit_payload['heldout_label_reads'] == unit_payload['original_head_training_updates'] == 0
         and unit_payload['new_head_training_updates'] == 2000, 'UNIT_COST_PAYLOAD_ATTESTATIONS')
    previous_receipt = M.read(PREV / 'fits/receipt.json')
    previous_fit_validation = M.read(PREV / 'fits/validation.json')
    need(previous_fit_validation['status'] == 'FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS', 'PRIOR_FIT_REPLAY')
    need(equal_binding(previous_fit_validation['receipt'], M.bind(PREV / 'fits/receipt.json')) and equal_binding(previous_fit_validation['payload'], previous_receipt['payload']), 'PRIOR_FIT_SEALS')
    need(equal_binding(payload['previous_payload'], previous_receipt['payload']), 'PRIOR_PAYLOAD_LINEAGE')
    need(equal_binding(unit_payload['previous_payload'], previous_receipt['payload']), 'UNIT_COST_PRIOR_PAYLOAD_LINEAGE')
    old_payload = torch.load(checked(previous_receipt['payload']), weights_only=True, map_location='cpu')
    predictions = indexed(payload['predictions'])
    unit_predictions = indexed(unit_payload['predictions'])
    old_predictions = indexed(old_payload['predictions'])
    need(len(predictions) == len(old_predictions) == len(unit_predictions) == 160
         and set(predictions) == set(old_predictions) == set(unit_predictions), 'EXACT160_PREVIOUS_POPULATION')
    need(set(payload['parameters']) == {'ORIGINAL7', 'TRAIN_CONVEX7'}
         and set(unit_payload['parameters']) == {'ORIGINAL7', 'TRAIN_UNIT_COST7'}, 'ONLY_DECLARED_TRAIN_MODELS')
    need(tensor_bits_equal(payload['parameters']['ORIGINAL7']['theta'], unit_payload['parameters']['ORIGINAL7']['theta']), 'DUAL_PAYLOAD_ORIGINAL_BITS')
    payload['parameters']['TRAIN_UNIT_COST7'] = unit_payload['parameters']['TRAIN_UNIT_COST7']
    for query_id, prediction in predictions.items():
        unit_prediction = unit_predictions[query_id]
        need(set(prediction['models']) == {'ORIGINAL7', 'TRAIN_CONVEX7'}
             and set(unit_prediction['models']) == {'ORIGINAL7', 'TRAIN_UNIT_COST7'}, 'ONLY_DECLARED_PREDICTION_MODELS')
        for key in ('query_id', 'execution_ordinal', 'candidate_physical_rows', 'raw_ranked_physical_rows', 'winner', 'challenger_positions'):
            need(prediction[key] == unit_prediction[key], 'UNIT_COST_SAME_EVALUATION_AXIS')
        for mode in ('REAL', 'CBIND'):
            need(prediction['models']['ORIGINAL7'][mode]['selected_position'] == unit_prediction['models']['ORIGINAL7'][mode]['selected_position']
                 and tensor_bits_equal(prediction['models']['ORIGINAL7'][mode]['logits'], unit_prediction['models']['ORIGINAL7'][mode]['logits']), 'UNIT_COST_ORIGINAL_PREDICTION_BIT_PARITY')
        prediction['models']['TRAIN_UNIT_COST7'] = unit_prediction['models']['TRAIN_UNIT_COST7']
    theta = {}
    for model in MODELS:
        parameters = payload['parameters'][model]
        need(set(parameters) == {'theta'}, 'NO_CONE_OR_EXTRA_MODEL_PARAMETERS')
        value = parameters['theta']
        need(value.dtype == torch.float64 and tuple(value.shape) == (7,) and bool(torch.isfinite(value).all()), 'SEVEN_FINITE_PARAMETERS')
        theta[model] = value.numpy()
    need(tuple(float(v).hex() for v in theta['ORIGINAL7']) == HEX7, 'ORIGINAL_EC7_UNCHANGED')
    need(tuple(float(v).hex() for v in theta['TRAIN_CONVEX7']) == tuple(train_search['solver']['theta_binary64']), 'SEALED_TRAIN_PARAMETERS')
    need(tuple(float(v).hex() for v in theta['TRAIN_UNIT_COST7']) == tuple(unit_fit['theta_binary64']), 'SEALED_UNIT_COST_PARAMETERS')
    need(tensor_bits_equal(payload['parameters']['ORIGINAL7']['theta'], old_payload['parameters']['ORIGINAL7']['theta']), 'OLD_PARAMETER_BIT_PARITY')
    features, feature_sources = P.features()
    need(payload['feature_sources'] == feature_sources == old_payload['feature_sources'] == unit_payload['feature_sources'], 'SAME_VALIDATED_FROZEN_FEATURES')
    fi = indexed(features)
    chosen = {}
    maximum_error = 0.0
    for query_id, prediction in predictions.items():
        feature, old = fi[query_id], old_predictions[query_id]
        for key in ('query_id', 'execution_ordinal', 'candidate_physical_rows', 'raw_ranked_physical_rows', 'winner', 'challenger_positions'):
            need(prediction[key] == feature[key] == old[key], 'UNCHANGED_FEATURE_AXIS_' + key)
        axis = feature['candidate_physical_rows']
        raw = feature['raw_ranked_physical_rows']
        challengers = feature['challenger_positions']
        need(len(axis) == 128 and axis == sorted(set(axis)) and set(axis) == set(raw[:128]), 'FULL_NATURAL_RAW_C128')
        need(len(raw) == len(set(raw)) and axis[feature['winner']] == raw[0], 'FULL_GALLERY_RAW_WINNER')
        need(challengers == [i for i in range(128) if i != feature['winner']], 'ALL127_CHALLENGERS')
        need(set(prediction['models']) == set(MODELS), 'NO_DIAGNOSTIC_PREDICTIONS')
        selected = {'RAW': raw[0]}
        for model in MODELS:
            need(set(prediction['models'][model]) == {'REAL', 'CBIND'}, 'REAL_CBIND_PREDICTIONS')
            for mode in ('REAL', 'CBIND'):
                report = prediction['models'][model][mode]
                x = feature['modes'][mode]['X']
                need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6), 'FEATURE127X6')
                logits = np.asarray(x.numpy(), dtype=np.float64) @ theta[model][:6] + theta[model][6]
                observed = report['logits']
                need(observed.dtype == torch.float64 and tuple(observed.shape) == (127,), 'LOGIT127')
                need(np.isfinite(logits).all() and bool(torch.isfinite(observed).all()), 'FINITE_LOGITS')
                error = float(np.max(np.abs(logits - observed.numpy())))
                maximum_error = max(maximum_error, error)
                need(error <= 1e-12, 'INDEPENDENT_NUMPY_LOGIT_REPLAY')
                position = action(logits, challengers, feature['winner'])
                need(position == report['selected_position'], 'INDEPENDENT_ZERO_HOLD_FIRST_MAX')
                if model == 'ORIGINAL7':
                    need(position == old['models'][model][mode]['selected_position'] and tensor_bits_equal(observed, old['models'][model][mode]['logits']), 'OLD_PREDICTION_BIT_PARITY')
                selected[model if mode == 'REAL' else model + '_CBIND'] = axis[position]
        chosen[query_id] = selected
    # No outcome or query-target identity has been read before the seals and replay.
    labels_open = True
    for binding in authority['evaluation_sources'].values():
        checked(binding)
    previous_result = M.read(expected_eval['result'])
    previous_validation = M.read(expected_eval['result_validation'])
    need(previous_validation['status'] == 'FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS', 'PRIOR_READOUT_VALIDATED')
    need(equal_binding(previous_validation['result'], authority['evaluation_sources']['result']), 'PRIOR_RESULT_HASH')
    need(equal_binding(previous_validation['payload'], previous_receipt['payload']), 'PRIOR_READOUT_PAYLOAD')
    need(previous_validation['original7_parameter_sha256'] == EC7 and tuple(previous_validation['original7_binary64']) == HEX7, 'PRIOR_EC7_IDENTITY')
    split = M.read(expected_eval['panel_manifest'])
    roles = indexed(M.read(expected_eval['eval_labels'])['records'])
    need(set(split['panels']) == {'EVAL32', 'EVAL128'} and split['selection_used_outcomes'] is False, 'FROZEN_PANEL_MEMBERSHIP')
    ids = [q for panel in ('EVAL32', 'EVAL128') for q in split['panels'][panel]]
    need(len(ids) == len(set(ids)) == 160 and set(ids) == set(predictions) == set(roles), 'EXACT_LABEL_POPULATION160')
    from rc_aslo_xf.conditional_rep_sources import build_gallery_source
    labels = build_gallery_source(verify_cache_file_sha256=True).corrected_identities
    panels = {}
    for panel in ('EVAL32', 'EVAL128'):
        query_ids = split['panels'][panel]
        need(len(query_ids) == {'EVAL32': 32, 'EVAL128': 128}[panel], 'EXACT_PANEL_SIZE')
        prior_rows = indexed(previous_result['panels'][panel]['rows'])
        rows = []
        for query_id in query_ids:
            feature, role, selected = fi[query_id], roles[query_id], chosen[query_id]
            need(feature['execution_ordinal'] == role['execution_ordinal'] and feature['source_image_sha256'] == role['source_image_sha256'], 'ROLE_FEATURE_IMAGE_BINDING')
            raw = feature['raw_ranked_physical_rows']
            targets = [physical for physical in raw if labels[physical] == role['identity']]
            need(len(targets) == 1, 'UNIQUE_FULLGALLERY_TARGET')
            target = targets[0]
            ranks = {name: moved_rank(raw, physical, target) for name, physical in selected.items()}
            correct = {name: physical == target for name, physical in selected.items()}
            row = {'query_id': query_id, 'original_query_id': role['original_query_id'],
                   'component': role['component'], 'group': role['group'],
                   'target_in_C128': target in feature['candidate_physical_rows'],
                   'correct': correct, 'ranks': ranks, 'selected_physical_rows': selected}
            prior = prior_rows[query_id]
            for field in ('original_query_id', 'component', 'group', 'target_in_C128'):
                need(row[field] == prior[field], 'PRIOR_ROW_LINEAGE_' + field)
            for field in ('correct', 'ranks', 'selected_physical_rows'):
                for name in ('RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND'):
                    need(row[field][name] == prior[field][name], 'ORIGINAL_PER_QUERY_READOUT_PARITY')
            rows.append(row)
        panels[panel] = summarize(rows)
        need(panels[panel]['scores']['ORIGINAL7']['correct'] == {'EVAL32': 28, 'EVAL128': 99}[panel], 'FROZEN28_99_BASELINE')
    baseline = train_search['baseline_objective']
    old_bounds = (Fraction(baseline['lower_rational']), Fraction(baseline['upper_rational']))
    global_bounds = solver_interval(train_search['solver'])
    optimization_gap = interval_record(old_bounds[0] - global_bounds[1], old_bounds[1] - global_bounds[0])
    cones = []
    for entry, search, validation in cone_searches:
        record = {'cone_id': entry['cone_id'], 'source_system_id': entry['source_system_id'],
                  'scope': 'PRESERVE_OLD28_AND_NEW99_PLUS_ONE' if entry['index'] < 5 else 'OLD_EVAL32_FIXED29_CORRECT_SET_ALLOWING_LOSSES',
                  'risk_interval': interval_record(*solver_interval(search['solver'])),
                  'solver_status': search['solver']['status'],
                  'box_active_coordinates': search['solver']['box_active_coordinates'],
                  'search': M.bind(DIAG / 'solves' / entry['cone_id'] / 'search.json'),
                  'validation': M.bind(DIAG / 'solves' / entry['cone_id'] / 'validation.json'),
                  'new_model_accuracy': None, 'coefficients_used_as_model': False}
        record.update(risk_comparison(old_bounds, global_bounds, solver_interval(search['solver'])))
        cones.append(record)
    result = {'status': 'CONVEX_CAUSE_ISOLATION_OPENED_DEVELOPMENT_READOUT_COMPLETE',
              'authority': ab, 'program': M.bind(__file__), 'prediction_seal': M.bind(TRAIN / 'prediction_seal.json'),
              'payload': seal['payload'], 'train_search': M.bind(TRAIN / 'fit/search.json'),
              'unit_cost_prediction_seal': M.bind(UNIT / 'prediction_seal.json'),
              'unit_cost_payload': unit_seal['payload'],
              'unit_cost_fit': M.bind(UNIT / 'fit.json'),
              'unit_cost_fit_validation': M.bind(UNIT / 'fit_validation.json'),
              'train_bound_validation': M.bind(TRAIN / 'fit/validation.json'),
              'diagnostic_manifest': authority['diagnostic_sources']['manifest'],
              'previous_result': authority['evaluation_sources']['result'],
              'previous_result_validation': authority['evaluation_sources']['result_validation'],
              'original7_parameter_sha256': EC7, 'original7_binary64': list(HEX7),
              'original_objective_interval': interval_record(*old_bounds),
              'box_optimum_interval': interval_record(*global_bounds),
              'original_minus_box_optimum': optimization_gap,
              'strict_positive_original_surrogate_gap_certified': Fraction(optimization_gap['lower_rational']) > 0,
              'train_solver_status': train_search['solver']['status'],
              'train_box_active_coordinates': train_search['solver']['box_active_coordinates'],
              'box': {'all_seven_parameters': [-64, 64], 'unbounded_global_optimality_claimed': False},
              'training_population': {'PAIR': 64, 'FULL': 32, 'unique_images': 96},
              'objective': 'Original recorded PAIR weighted BCE mean plus FULL SIGN max-softplus mean; no added L2 penalty',
              'bound_comparison_objective': 'Original cost4 objective only; no subtraction of losses from different objectives',
              'matched_objective_control': {'model': 'TRAIN_UNIT_COST7',
                  'only_objective_change': 'Original negative-cost factor 4 changed to 1 in PAIR and FULL terms',
                  'same_AdamW_updates': 2000, 'same_original_mixed96_features': True,
                  'original_ec7_retrained': False, 'new_loss_used_for_optimization_gap': False},
              'AdamW_decoupled_decay_not_equivalent_to_this_unregularized_objective': True,
              'head_class': 'Original frozen six features plus intercept; seven parameters',
              'candidate_source': 'Original RAW natural C128, all127 challengers, HOLD0 or max-logit SWITCH',
              'panels': panels, 'diagnostic_cones': cones,
              'joint_all_old127_correct_preserved': all(panels[p]['comparisons']['ORIGINAL7__to__TRAIN_CONVEX7']['loss'] == 0 for p in panels),
              'new_head_net_gain_by_panel': {p: panels[p]['comparisons']['ORIGINAL7__to__TRAIN_CONVEX7']['net'] > 0 for p in panels},
              'unit_cost_intervention_net_gain_by_panel': {p: panels[p]['comparisons']['ORIGINAL7__to__TRAIN_UNIT_COST7']['net'] > 0 for p in panels},
              'unit_cost_causal_readout_by_panel': {p: ('COST_INTERVENTION_NET_GAIN_ON_OPENED_PANEL' if panels[p]['comparisons']['ORIGINAL7__to__TRAIN_UNIT_COST7']['net'] > 0 else 'COST_INTERVENTION_NO_POSITIVE_NET_GAIN_ON_OPENED_PANEL') for p in panels},
              'current_model_success_criterion': 'Report net gain and losses explicitly; strict-retention cones are separate mathematical diagnoses',
              'evaluation_labels_opened_after_prediction_seal_and_independent_replay': True,
              'independently_checked_logits': 160 * 3 * 2 * 127,
              'maximum_absolute_numpy_logit_error': maximum_error,
              'original_logits_and_actions_bit_equal_previous_payload': True,
              'new_593_accuracy_computed': False, 'diagnostic_coefficients_scored_as_models': False,
              'original_head_training_updates': 0, 'collector_optimizer_calls': 0,
              'historically_opened_development': True, 'external_confirmation': False,
              'deployment_changed': False, 'HYP_GO_claimed': False,
              'limitations': ['All risk bounds refer only to the fixed box [-64,64]^7 and supplied real feature endpoints.',
                  'A gap to the original logged loss does not identify the decoupled AdamW algorithm with L2-regularized minimization.',
                  'The matched cost4-to1 intervention is evaluated by per-panel net gain and losses; its numerical loss is not compared against the cost4 convex objective.',
                  'Cone optima use closure A theta >= 0; existing strict witnesses make the strict-region infima equal by continuity.',
                  'Five cones preserve old28 and new99; the sixth is one fixed old32 29-correct cone allowing losses, not all possible net-gain cones.',
                  'Positive empirical-risk separation is not proof of inherent information loss or external generalization.',
                  'Unresolved bound overlap or a resource stop is not an infeasibility certificate.']}
    M.write(OUT / 'result.json', result)
    validation = {'status': 'CONVEX_CAUSE_INDEPENDENT_ACTION_RANK_AND_BOUND_COMPARISON_PASS',
                  'result': M.bind(OUT / 'result.json'), 'authority': ab, 'program': M.bind(__file__),
                  'prediction_seal': result['prediction_seal'], 'payload': seal['payload'],
                  'unit_cost_prediction_seal': result['unit_cost_prediction_seal'], 'unit_cost_payload': unit_seal['payload'],
                  'bound_validation_count': 7, 'independently_checked_logits': result['independently_checked_logits'],
                  'maximum_absolute_logit_error': maximum_error, 'logit_tolerance': 1e-12,
                  'MRR_method': 'Literal full-gallery move-to-front followed by exact Fraction reciprocal ranks',
                  'action_rule': 'Earliest challenger on equal maxima; HOLD when maximum <= 0',
                  'original_ec7_unchanged': True, 'original_logits_bit_equal_previous_payload': True,
                  'labels_opened_after_seal_and_independent_prediction_replay': True,
                  'optimizer_calls': 0, 'new_593_accuracy_computed': False,
                  'diagnostic_coefficients_used_as_models': False, 'slurm_job_id': os.environ['SLURM_JOB_ID']}
    M.write(OUT / 'independent_validation.json', validation)
    write_report(result)
    print(json.dumps({'status': result['status'], 'result': M.bind(OUT / 'result.json'),
                      'original_loss_gap': optimization_gap,
                      'panels': {p: {'old': panels[p]['scores']['ORIGINAL7']['correct'],
                                     'new': panels[p]['scores']['TRAIN_CONVEX7']['correct'],
                                     'unit_cost': panels[p]['scores']['TRAIN_UNIT_COST7']['correct'],
                                     'comparison': panels[p]['comparisons']['ORIGINAL7__to__TRAIN_CONVEX7'],
                                     'unit_cost_comparison': panels[p]['comparisons']['ORIGINAL7__to__TRAIN_UNIT_COST7']} for p in panels}}), flush=True)


def write_report(result):
    gap = result['original_minus_box_optimum']
    lines = ['# 原 ec7 平台原因隔离：固定盒内的经验损失与实际动作', '',
             '本轮只读既有 ec7 参数，原模型没有重新训练。TRAIN_CONVEX7 优化原 PAIR64＋FULL32 的记录损失；TRAIN_UNIT_COST7 仅将负项成本4改为1，保留同一混合训练集和 AdamW 2000步。两组全部160条预测封存并独立重放后才读取已打开的 EVAL 标签。', '',
             '| 面板 | RAW | 原 ec7 | TRAIN_CONVEX7 | 对 ec7 净增（救回／损失） | TRAIN_UNIT_COST7 | 对 ec7 净增（救回／损失） |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('EVAL32', 'EVAL128'):
        panel = result['panels'][name]
        score, comparison = panel['scores'], panel['comparisons']['ORIGINAL7__to__TRAIN_CONVEX7']
        unit = panel['comparisons']['ORIGINAL7__to__TRAIN_UNIT_COST7']
        lines.append(f"| {name} | {score['RAW']['correct']}/{panel['population']} | {score['ORIGINAL7']['correct']}/{panel['population']} | {score['TRAIN_CONVEX7']['correct']}/{panel['population']} | {comparison['net']:+d}（{comparison['rescue']}／{comparison['loss']}） | {score['TRAIN_UNIT_COST7']['correct']}/{panel['population']} | {unit['net']:+d}（{unit['rescue']}／{unit['loss']}） |")
    lines += ['', '候选绑定破坏后的 C_BIND 正确数（原 ec7／TRAIN_CONVEX7／TRAIN_UNIT_COST7）：' + '；'.join(
        p + '：' + '／'.join(str(result['panels'][p]['scores'][m + '_CBIND']['correct']) for m in MODELS)
        for p in ('EVAL32', 'EVAL128')) + '。', '',
        '成本干预只有在对应面板相对原 ec7 净增为正时才记为有效内部干预：' + '；'.join(
            p + '：' + ('有正净增' if result['unit_cost_intervention_net_gain_by_panel'][p] else '无正净增，不能据此认定成本是已解决的原因')
            for p in ('EVAL32', 'EVAL128')) + '。两面板仍是已打开的开发结果。']
    lines += ['', '两面板都使用原 RAW 自然 C128、原六维特征、127 challenger 与 HOLD0／最大正 logit SWITCH。MRR 由完整图库 move-to-front 排列逐条计算，精确分数保存在结果文件。净增与损失分别报告；数学保持约束不替代当前净增准则。', '',
              f"原 ec7 的记录损失与固定盒内最优损失之差，已证区间为 [{gap['lower_display']:.12g}, {gap['upper_display']:.12g}]。TRAIN 求解状态为 `{result['train_solver_status']}`，位于盒边界的参数坐标为 `{result['train_box_active_coordinates']}`。", '',
              '**全部优化界只覆盖 [-64,64]^7，且只分析原成本4目标。** 原训练采用 decoupled AdamW weight decay；本轮优化原来记录的未加 L2 损失，不能声称两者具有同一个显式正则化目标。成本1与成本4的损失数值不相减。低训练损失能否解释 plateau，要结合上表实际动作和下表约束风险差。', '',
              '| 隔离系统 | 数学范围 | 相对盒内最优风险差区间 | 相对原 ec7 的关系 |',
              '| --- | --- | --- | --- |']
    translations = {'BOX_CONE_ALL_ACTIONS_COST_MORE_THAN_ORIGINAL': '该盒内全部满足动作要求的参数都比原 ec7 损失高',
                    'BOX_LOSS_REDUCTION_AND_RESCUE_COMPATIBLE_BUT_NOT_NEAR_OPTIMUM': '可以同时降低原损失并满足动作，但与盒内最优有正风险间隔',
                    'INTERVAL_OVERLAP_UNRESOLVED': '区间尚未分离，保留未决'}
    for cone in result['diagnostic_cones']:
        diff = cone['cone_minus_box_optimum']
        scope = '保旧28＋新99并修一例' if cone['scope'].startswith('PRESERVE') else '旧32固定29正确集合，允许损失'
        lines.append(f"| {cone['cone_id']}：{cone['source_system_id']} | {scope} | [{diff['lower_display']:.9g}, {diff['upper_display']:.9g}] | {translations[cone['status']]} |")
    lines += ['', '以上六组系数只用于已知标签的数学诊断，没有计算新模型准确率，也不进入训练、阈值或部署。前五个锥分别可行，不表示同一参数同时修好五例；第六个仅覆盖既有一个29正确集合。严格可行 witness 使闭包与严格动作区域的损失下确界相同，但不提供泛化保证。', '',
              '所有七份求解结果均绑定到独立精确界重放；成本1模型有独立训练参数重放。collector 没有调用优化器。NumPy 独立计算三头 REAL／C_BIND 共121,920个 logit，原 ec7 与历史 payload 逐bit一致。没有新的593总体准确率、外部确认或 HYP GO。', '',
              f"结果：`{(OUT / 'result.json').relative_to(ROOT)}`。独立读出验证：`{(OUT / 'independent_validation.json').relative_to(ROOT)}`。", '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x') as handle:
        handle.write('\n'.join(lines))


def self_test():
    need(action([0., 0.], [2, 3], 1) == 1, 'ZERO_HOLD_TEST')
    need(action([1., 1.], [2, 3], 1) == 2, 'FIRST_MAX_TEST')
    need(moved_rank([4, 2, 9], 9, 2) == 3 and moved_rank([4, 2, 9], 2, 2) == 1, 'LITERAL_RANK_TEST')
    r = risk_comparison((Fraction(10), Fraction(11)), (Fraction(2), Fraction(3)), (Fraction(12), Fraction(13)))
    need(r['all_cone_points_worse_than_original_certified'], 'STRICT_CONFLICT_TEST')
    r = risk_comparison((Fraction(10), Fraction(11)), (Fraction(2), Fraction(3)), (Fraction(5), Fraction(6)))
    need(r['status'] == 'BOX_LOSS_REDUCTION_AND_RESCUE_COMPATIBLE_BUT_NOT_NEAR_OPTIMUM', 'COMPATIBLE_BUT_SUBOPTIMAL_TEST')
    r = risk_comparison((Fraction(10), Fraction(11)), (Fraction(2), Fraction(3)), (Fraction(3), Fraction(12)))
    need(r['status'] == 'INTERVAL_OVERLAP_UNRESOLVED' and not r['positive_risk_separation_from_box_optimum_certified'], 'BOUNDARY_UNRESOLVED_TEST')
    rows = []
    for i in range(2):
        correctness = {'RAW': i == 0, 'ORIGINAL7': i == 0, 'ORIGINAL7_CBIND': False,
                       'TRAIN_CONVEX7': i == 1, 'TRAIN_CONVEX7_CBIND': False,
                       'TRAIN_UNIT_COST7': True, 'TRAIN_UNIT_COST7_CBIND': False}
        rows.append({'query_id': str(i), 'component': 'same', 'target_in_C128': True,
                     'correct': correctness, 'ranks': {k: 1 if v else 2 for k, v in correctness.items()},
                     'selected_physical_rows': {k: int(v) for k, v in correctness.items()}})
    summary = summarize(rows)
    change = summary['comparisons']['ORIGINAL7__to__TRAIN_CONVEX7']
    need((change['rescue'], change['loss'], change['net'], change['all_old_correct_preserved']) == (1, 1, 0, False), 'NET_AND_RETENTION_SEPARATE_TEST')
    need(summary['scores']['TRAIN_CONVEX7']['MRR_rational'] == '3/4', 'EXACT_MRR_TEST')
    need(summary['comparisons']['ORIGINAL7__to__TRAIN_UNIT_COST7']['net'] == 1
         and summary['scores']['TRAIN_UNIT_COST7']['MRR_rational'] == '1', 'THIRD_MODEL_NET_AND_MRR_TEST')
    print('CONVEX_CAUSE_COLLECTOR_SYNTHETIC_LOGIC_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main()
