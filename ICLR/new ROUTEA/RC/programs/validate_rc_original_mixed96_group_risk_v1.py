#!/usr/bin/env python3
"""Independent MIXED96 action/rank audit; no training and no early label reads."""
import math
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P

OUT = ROOT / 'results/rc_original_mixed96_group_risk_v1'
PREV = ROOT / 'results/rc_fixed_panels_train269_group_risk_v1'
AUTH = ROOT / 'registry/rc_original_mixed96_group_risk_authority_v1_20260911.json'
PREV_AUTH = ROOT / 'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json'
MODELS = ('ORIGINAL7', 'GROUP_MIXED96')
SCORE_MODELS = {'RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND', 'GROUP_MIXED96', 'GROUP_MIXED96_CBIND'}
HEX7 = ('0x1.01ffd7d7241b3p+0', '-0x1.e3db1bf7526f8p+1',
        '0x1.6ddd66d0df5d3p+2', '0x1.7a67f045c5bd2p+2',
        '-0x1.3095a38e39e50p-3', '-0x1.e8ea4dd00f544p-3',
        '-0x1.7a55fbee7993ap+0')
EC7 = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'


def need(value, label):
    if not value:
        raise RuntimeError(label)


def index(rows, key):
    result = {r[key]: r for r in rows}
    need(len(result) == len(rows), 'UNIQUE_' + key)
    return result


def close(actual, expected, label):
    need(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-12, label)


def fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def bit_equal(a, b):
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype and a.shape == b.shape
            and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes())


def action(logits, challengers, winner):
    j = max(range(len(logits)), key=lambda i: float(logits[i]))
    return challengers[j] if float(logits[j]) > 0 else winner


def comparison(rows, base, new):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    means = {g: Fraction(sum(v), len(v)) for g, v in sorted(grouped.items())}
    ds = np.asarray([float(v) for v in means.values()], dtype=np.float64)
    rng = np.random.default_rng(20260911)
    samples = ds[rng.integers(0, len(ds), size=(10000, len(ds)))].mean(axis=1)
    summary = dict(rescue=sum(not r['correct'][base] and r['correct'][new] for r in rows),
                   loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                   equal_component_difference=float(sum(means.values(), Fraction()) / len(means)),
                   component_bootstrap95=[float(v) for v in np.quantile(samples, [.025, .975])],
                   positive_components=sum(v > 0 for v in means.values()),
                   negative_components=sum(v < 0 for v in means.values()), components=len(means))
    return summary, {g: fraction(v) for g, v in means.items()}


def main():
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    need(M.datetime.now(M.timezone.utc) < M.DEADLINE, 'USER_CUTOFF')
    authority = M.read(AUTH)
    need(authority['status'] == 'ORIGINAL_MIXED96_GROUP_RISK_AUTHORIZED', 'AUTHORITY_STATUS')
    pins = authority['sources']
    evaluation = authority['evaluation_sources']
    expected_public = {
        'program': ROOT / 'programs/run_rc_original_mixed96_group_risk_v1.py',
        'validator': Path(__file__).resolve(),
        'reporter': ROOT / 'programs/report_rc_original_mixed96_group_risk_v1.py',
        'freezer': ROOT / 'programs/freeze_rc_original_mixed96_group_risk_v1.py',
        'input_helper': ROOT / 'programs/materialize_rc_new_hyp593_inputs_v1.py',
        'feature_helper': ROOT / 'programs/run_rc_new_hyp593_oof5_v1.py',
        'legacy_runner': ROOT / 'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py',
        'original_seal': ROOT / 'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
        'feature_authority': ROOT / 'registry/rc_new_hyp593_feature_authority_v1_20260911.json',
        'previous_authority': PREV_AUTH,
        'previous_payload': PREV / 'fits/payload.pt',
        'previous_receipt': PREV / 'fits/receipt.json',
        'previous_fit_validation': PREV / 'fits/validation.json',
        'panel_manifest': PREV / 'panel_manifest.json',
        'train_roles': OUT / 'train_roles.json',
        'train_inputs': OUT / 'train_inputs.pt',
        'train_input_seal': OUT / 'train_input_seal.json',
        'plan': ROOT / 'plan/RC_ORIGINAL_MIXED96_GROUP_RISK_V1_20260911.md',
        'launcher': ROOT / 'slurm/rc_original_mixed96_group_risk_v1.sbatch',
    }
    expected_eval = {'eval_labels': PREV / 'eval_curator_roles.json',
                     'previous_result': PREV / 'result.json',
                     'previous_result_validation': PREV / 'result_validation.json'}
    need(set(pins) == set(expected_public) and set(evaluation) == set(expected_eval), 'EXACT_AUTHORITY_SOURCE_KEYS')
    for key, path in expected_public.items():
        need(Path(pins[key]['path']) == path, 'PUBLIC_SOURCE_PATH_' + key)
    for key, path in expected_eval.items():
        need(Path(evaluation[key]['path']) == path, 'EVAL_SOURCE_PATH_' + key)
    public_paths = {Path(b['path']).absolute() for b in pins.values()}
    eval_paths = {Path(b['path']).absolute() for b in evaluation.values()}
    need(not public_paths & eval_paths, 'EVAL_SOURCE_PATHS_AND_SEPARATION')
    need(len(public_paths) == len(pins) and len(eval_paths) == len(evaluation), 'UNIQUE_SOURCE_PATHS')

    # This hook is installed before hashing any source, so hashing cannot bypass it.
    labels_open = False
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        s = str(p).lower()
        need(not any(x in s for x in ('rc_opened_', 'd1-mi', 'd1_mi', 'grozi', 'gisc_prerecall_universe', '/target_join/')), 'PROTECTED_INPUT')
        if not labels_open:
            need(p not in eval_paths and 'curator_roles' not in s and p.name not in ('result.json', 'result_validation.json') and '/reports/' not in s, 'FIT_SEAL_BEFORE_EVAL_READ')
        for blocked in (P.OUT / 'fits', P.OUT / 'roles', ROOT / 'results/rc_h593_group_risk_strong_base_v1'):
            need(blocked not in p.parents, 'NO_PRIOR593_MODEL_OR_FOLD_LABEL_READ')
    sys.addaudithook(audit)
    for b in pins.values():
        M.checked(b)
    by_public_path = {Path(b['path']): b for b in pins.values()}
    by_eval_path = {Path(b['path']): b for b in evaluation.values()}
    ab = M.bind(AUTH)
    preflight = M.read(OUT / 'preflight.json')
    need(preflight['status'] == 'ORIGINAL_MIXED96_GROUP_PREFLIGHT_PASS' and preflight['authority'] == ab, 'PREFLIGHT_BINDING')
    fitval = M.read(OUT / 'fits/validation.json')
    receipt = M.read(OUT / 'fits/receipt.json')
    need(fitval['status'] == 'ORIGINAL_MIXED96_GROUP_FRESH_REPLAY_PASS', 'FIT_REPLAY_PASS')
    need(fitval['payload'] == receipt['payload'] and fitval['receipt'] == M.bind(OUT / 'fits/receipt.json'), 'FIT_RECEIPT_BINDING')
    need(Path(receipt['payload']['path']) == OUT / 'fits/payload.pt', 'ACTIVE_FIT_PAYLOAD')
    need(fitval['authority'] == receipt['authority'] == ab, 'FIT_AUTHORITY')
    need(fitval['original_head_training_updates'] == 0, 'NO_ORIGINAL_HEAD_REFIT')
    need(fitval['heldout_label_reads'] == 0, 'FIT_NO_EVAL_LABEL_READS')
    need(isinstance(fitval.get('nonce'), str) and len(fitval['nonce']) == 32, 'FRESH_REPLAY_NONCE')
    payload = torch.load(M.checked(receipt['payload']), map_location='cpu', weights_only=True)
    need(payload['authority'] == ab, 'PAYLOAD_AUTHORITY')
    need(payload['train_input_seal'] == by_public_path[OUT / 'train_input_seal.json'], 'PAYLOAD_TRAIN_INPUT_SEAL')
    need(payload['train_roles'] == by_public_path[OUT / 'train_roles.json'], 'PAYLOAD_TRAIN_ROLES')
    need(payload['previous_source_payload'] == by_public_path[PREV / 'fits/payload.pt'], 'PAYLOAD_PREVIOUS_SOURCE')
    need(payload['original_head_training_updates'] == 0 and payload['heldout_label_reads'] == 0, 'PAYLOAD_NO_ORIGINAL_REFIT_OR_EVAL_READ')
    need(payload['new_head_training_updates'] == 2000 and payload['old_predictions_copied'] is True, 'NEW_FIT_ONLY_2000_UPDATES')
    need(payload['finite_training'] == dict(input_finite=True, finite_loss_step_count=2000, finite_gradient_step_count=2000, finite_parameter_state_count=2001, all_finite=True), 'EVERY_TRAIN_UPDATE_FINITE')
    need(payload['function_change'] == preflight['function_change'], 'PREFLIGHT_EXACT_FUNCTION_CHANGE')
    need(payload['function_change']['replacement_count'] == 2 and payload['function_change']['PAIR_FULL_mixture'] == [1, 1], 'ONLY_TWO_WITHIN_POOL_MEAN_EDITS')
    need(preflight['natural_training_updates'] == preflight['original_head_training_updates'] == 0, 'PREFLIGHT_NO_NATURAL_FIT')
    train_seal = M.read(OUT / 'train_input_seal.json')
    need(train_seal['status'] == 'ORIGINAL_MIXED96_INPUTS_FROZEN', 'TRAIN_INPUTS_FROZEN')
    need(train_seal['payload'] == by_public_path[OUT / 'train_inputs.pt'] and train_seal['train_roles'] == payload['train_roles'], 'STAGED_TRAIN_INPUT_BINDINGS')
    need(train_seal['original_head_training_updates'] == 0, 'STAGING_NO_ORIGINAL_REFIT')
    need(train_seal['counts'] == dict(PAIR=64, FULL=32, unique_images=96, identities=32), 'STAGED_TRAIN96_COUNTS')
    need(train_seal['feature_sha256'] == preflight['original_train_feature_sha256'], 'PREFLIGHT_TRAIN_FEATURE_DIGEST')
    for b in list(train_seal['sources'].values()) + list(train_seal['full_shards'].values()) + train_seal['training_role_sources']:
        M.checked(b)
    need(len(train_seal['training_role_sources']) == 96, 'ALL96_ORIGINAL_TRAIN_ROLE_BINDINGS')
    staged = torch.load(M.checked(train_seal['payload']), map_location='cpu', weights_only=True)
    split = M.read(PREV / 'panel_manifest.json')
    need(set(split['panels']) == {'EVAL32', 'EVAL128'} and split['selection_used_outcomes'] is False, 'FIXED_PANEL_SELECTION')
    train_doc = M.read(OUT / 'train_roles.json')
    train = index(train_doc['records'], 'query_id')
    need(train_doc['evaluation_labels_included'] is False, 'TRAIN_ROLE_NO_EVAL_LABELS')
    need(len(train) == 96 and len({r['source_image_sha256'] for r in train.values()}) == 96, 'TRAIN96_UNIQUE_IMAGES')
    need(len({r['identity'] for r in train.values()}) == len({r['group'] for r in train.values()}) == len({r['component'] for r in train.values()}) == 32, 'TRAIN32_IDENTITIES_GROUPS_COMPONENTS')
    need(Counter(r['pool'] for r in train.values()) == {'PAIR': 64, 'FULL': 32}, 'ORIGINAL_PAIR64_FULL32')
    need(all(r['group'] == r['supergroup'] for r in train.values()), 'ORIGINAL_SUPERGROUP_LABELS')
    need(len({(r['identity'], r['group'], r['component']) for r in train.values()}) == 32, 'IDENTITY_GROUP_COMPONENT_BIJECTION')
    pool_counts = {pool: dict(Counter(r['group'] for r in train.values() if r['pool'] == pool)) for pool in ('PAIR', 'FULL')}
    need({p: len(c) for p, c in pool_counts.items()} == {'PAIR': 20, 'FULL': 12}, 'PAIR20_FULL12_GROUP_COUNTS')
    need(not set(pool_counts['PAIR']) & set(pool_counts['FULL']), 'PAIR_FULL_COMPONENT_SEPARATION')
    need(train_seal['group_counts'] == {'PAIR': 20, 'FULL': 12} and train_seal['group_image_counts'] == pool_counts, 'STAGED_POOL_GROUP_COUNTS')
    for pool, data in (('PAIR', staged['pair']['records']), ('FULL', staged['full'])):
        pool_roles = [r for r in train.values() if r['pool'] == pool]
        need(len(pool_roles) == len(data) == {'PAIR': 64, 'FULL': 32}[pool], 'STAGED_POOL_COUNT')
        need([r['original_query_id'] for r in pool_roles] == [r['query_id'] for r in data], 'STAGED_POOL_QUERY_ORDER')
        need([r['legacy_execution_ordinal'] for r in pool_roles] == [r['execution_ordinal'] for r in data], 'STAGED_POOL_EXECUTION_ORDER')
        weights = staged['weights'][pool]
        expected = np.array([1 / (len(pool_counts[pool]) * pool_counts[pool][r['group']]) for r in pool_roles], dtype=np.float64)
        need(isinstance(weights, torch.Tensor) and weights.dtype == torch.float64 and np.array_equal(weights.numpy(), expected), 'EXACT_WITHIN_POOL_EQUAL_GROUP_WEIGHTS')
        close(weights.sum(), 1.0, 'POOL_WEIGHT_SUM_ONE')
    oldv = M.read(PREV / 'fits/validation.json')
    oldr = M.read(PREV / 'fits/receipt.json')
    need(oldv['status'] == 'FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS', 'PREVIOUS_FIT_VALIDATED')
    need(oldv['receipt'] == by_public_path[PREV / 'fits/receipt.json'] and oldv['payload'] == oldr['payload'] == payload['previous_source_payload'], 'PREVIOUS_FIT_CHAIN')
    need(oldv['authority'] == oldr['authority'] == by_public_path[PREV_AUTH], 'PREVIOUS_FIT_AUTHORITY')
    previous = torch.load(M.checked(payload['previous_source_payload']), map_location='cpu', weights_only=True)
    need(previous['authority'] == by_public_path[PREV_AUTH], 'PREVIOUS_PAYLOAD_AUTHORITY')
    need(set(payload['parameters']) == set(MODELS), 'EXACT_MODEL_SET')
    theta = {}
    for model in MODELS:
        t = payload['parameters'][model]['theta']
        need(isinstance(t, torch.Tensor) and t.dtype == torch.float64 and tuple(t.shape) == (7,) and bool(torch.isfinite(t).all()), 'THETA_' + model)
        need('beta' not in payload['parameters'][model], 'SEVEN_PARAMETER_ARCHITECTURE')
        theta[model] = t.numpy()
    seal = M.read(ROOT / 'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json')
    need(seal['parameter_sha256'] == EC7 and tuple(seal['weight_binary64'] + [seal['bias_binary64']]) == HEX7, 'FROZEN_EC7_SEAL_HEX')
    need(tuple(float(v).hex() for v in theta['ORIGINAL7']) == HEX7, 'ORIGINAL7_EXACT_BINARY64')
    need(bit_equal(payload['parameters']['ORIGINAL7']['theta'], previous['parameters']['ORIGINAL7']['theta']), 'ORIGINAL7_THETA_BIT_EQUAL_PRIOR')
    features, feature_sources = P.features()
    need(payload['feature_sources'] == previous['feature_sources'] == feature_sources, 'VALIDATED_FEATURE_SOURCE_BINDINGS')
    fi = index(features, 'query_id')
    qids = [q for qs in split['panels'].values() for q in qs]
    pred = index(payload['predictions'], 'query_id')
    oldpred = index(previous['predictions'], 'query_id')
    need(len(qids) == len(set(qids)) == len(pred) == len(oldpred) == 160 and set(qids) == set(pred) == set(oldpred), 'ALL160_FIXED_PREDICTIONS')
    need(not set(train) & set(qids), 'TRAIN_EVAL_QUERY_SEPARATION_PREJOIN')
    chosen, max_error, parity_logits = {}, 0.0, 0
    need(action([0.0, 0.0], [2, 3], 1) == 1 and action([1.0, 1.0], [2, 3], 1) == 2, 'PYTHON_TIE_AND_HOLD_ZERO')
    for q in qids:
        f, p, old = fi[q], pred[q], oldpred[q]
        for k in ('query_id', 'execution_ordinal', 'candidate_physical_rows', 'raw_ranked_physical_rows', 'winner', 'challenger_positions'):
            need(p[k] == old[k] == f[k], 'PRED_FEATURE_PRIOR_AXIS_' + k)
        axis, raw, challengers = list(f['candidate_physical_rows']), list(f['raw_ranked_physical_rows']), list(f['challenger_positions'])
        need(len(axis) == 128 and axis == sorted(set(axis)), 'SORTED_FULL_C128')
        need(len(raw) == len(set(raw)) and set(axis) == set(raw[:128]) and axis[f['winner']] == raw[0], 'FULLGALLERY_RAW_C128')
        need(challengers == [j for j in range(128) if j != f['winner']], '127_CHALLENGERS_IN_ORDER')
        need(set(p['models']) == set(MODELS), 'PRED_MODEL_SET')
        selected = {'RAW': raw[0]}
        for model in MODELS:
            need(set(p['models'][model]) == {'REAL', 'CBIND'}, 'PRED_MODE_SET')
            for mode in ('REAL', 'CBIND'):
                x = f['modes'][mode]['X']
                need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6), 'FEATURE127X6')
                z = np.asarray(x.numpy(), dtype=np.float64) @ theta[model][:6] + theta[model][6]
                reported = p['models'][model][mode]
                observed = reported['logits']
                need(isinstance(observed, torch.Tensor) and observed.dtype == torch.float64 and tuple(observed.shape) == (127,), '127_LOGITS')
                need(np.isfinite(z).all() and np.isfinite(observed.numpy()).all(), 'FINITE_LOGITS')
                error = float(np.max(np.abs(z - observed.numpy())))
                max_error = max(max_error, error)
                need(error <= 1e-12, 'NUMPY_LOGIT_REPLAY')
                position = action(z, challengers, f['winner'])
                need(position == reported['selected_position'], 'INDEPENDENT_HOLD_SWITCH')
                if model == 'ORIGINAL7':
                    prior = old['models'][model][mode]
                    need(bit_equal(observed, prior['logits']) and position == prior['selected_position'], 'ORIGINAL7_LOGITS_ACTION_BIT_EQUAL_PRIOR_PRELABEL')
                    parity_logits += len(z)
                selected[model if mode == 'REAL' else model + '_CBIND'] = axis[position]
        chosen[q] = selected

    # Labels and previous observed outcomes are opened only after all replay checks.
    labels_open = True
    for b in evaluation.values():
        M.checked(b)
    previous_result = M.read(PREV / 'result.json')
    previous_validation = M.read(PREV / 'result_validation.json')
    need(previous_validation['status'] == 'FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS', 'PREVIOUS_RESULT_INDEPENDENT_PASS')
    need(previous_validation['result'] == by_eval_path[PREV / 'result.json'] and previous_validation['authority'] == by_public_path[PREV_AUTH], 'PREVIOUS_RESULT_CHAIN')
    need(previous_validation['payload'] == payload['previous_source_payload'] and previous_validation['fit_validation'] == by_public_path[PREV / 'fits/validation.json'], 'PREVIOUS_VALIDATION_FIT_CHAIN')
    need(previous_result['authority'] == by_public_path[PREV_AUTH] and previous_result['fit_validation'] == by_public_path[PREV / 'fits/validation.json'], 'PREVIOUS_RESULT_FIT_BINDING')
    previous_authority = M.read(PREV_AUTH)
    need(previous_validation['public_sources'] == previous_authority['sources'] and previous_validation['evaluation_sources'] == previous_authority['evaluation_sources'], 'PREVIOUS_VALIDATION_SOURCE_BINDINGS')
    for collection in ('sources', 'evaluation_sources'):
        for b in previous_authority[collection].values():
            M.checked(b)
    need(previous_validation['program'] == previous_authority['sources']['validator'], 'PREVIOUS_VALIDATOR_PROGRAM_BINDING')
    need(previous_validation['evaluation_labels_opened_after_fit_seal'] is True, 'PREVIOUS_READ_ORDER')
    need(previous_validation['historically_opened_development'] is True and previous_validation['external_confirmation'] is False and previous_validation['deployment_changed'] is False, 'PREVIOUS_EVIDENCE_BOUNDARY')
    need(previous_validation['historic_input_parity'] == M.bind(PREV / 'historic_input_parity.json'), 'PREVIOUS_HISTORIC_INPUT_PARITY_BINDING')
    historic = M.read(PREV / 'historic_input_parity.json')
    need(historic['status'] == 'FIXED269_HISTORIC_INPUT_PARITY_PASS' and historic['all_pass'] is True, 'PREVIOUS_HISTORIC_INPUT_PARITY_PASS')
    need(historic['authority'] == by_public_path[PREV_AUTH] and historic['program'] == previous_authority['sources']['historic_input_validator'], 'PREVIOUS_HISTORIC_PARITY_AUTHORITY')
    need(len(historic['rows']) == 160 and {r['query_id'] for r in historic['rows']} == set(qids) and all(all(r['checks'].values()) for r in historic['rows']), 'ALL160_PREVIOUS_HISTORIC_INPUTS_CHECKED')
    roles = index(M.read(PREV / 'eval_curator_roles.json')['records'], 'query_id')
    need(set(roles) == set(qids), 'EXACT_EVAL_LABEL_SET')
    allroles = index(M.read(P.OUT / 'metadata/curator_roles.json')['records'], 'query_id')
    safe = ('query_id', 'original_query_id', 'execution_ordinal', 'source_image_sha256', 'identity', 'group', 'component')
    for q, r in list(train.items()) + list(roles.items()):
        need(all(r[k] == allroles[q][k] for k in safe), 'CURATOR_ROLE_PROVENANCE')
        need(fi[q]['source_image_sha256'] == r['source_image_sha256'] and fi[q]['execution_ordinal'] == r['execution_ordinal'], 'ROLE_FEATURE_IMAGE_BINDING')
    overlaps = {}
    for panel, qs in split['panels'].items():
        need(len(qs) == {'EVAL32': 32, 'EVAL128': 128}[panel], 'PANEL_COUNT')
        overlaps[panel] = {}
        for k in ('query_id', 'source_image_sha256', 'identity', 'group', 'component'):
            overlaps[panel][k] = len({r[k] for r in train.values()} & {roles[q][k] for q in qs})
            need(overlaps[panel][k] == 0, 'TRAIN_EVAL_DISJOINT_' + panel + '_' + k)
    for field in ('query_id', 'source_image_sha256', 'identity', 'group', 'component'):
        need(not ({roles[q][field] for q in split['panels']['EVAL32']} & {roles[q][field] for q in split['panels']['EVAL128']}), 'EVAL_PANELS_MUTUALLY_DISJOINT_' + field)
    from rc_aslo_xf.conditional_rep_sources import build_gallery_source
    labels = build_gallery_source(verify_cache_file_sha256=True).corrected_identities
    result = M.read(OUT / 'result.json')
    need(result['status'] == 'ORIGINAL_MIXED96_GROUP_FIXED_PANEL_DEVELOPMENT_COMPLETE', 'RESULT_STATUS')
    need(result['authority'] == ab and result['fit_validation'] == M.bind(OUT / 'fits/validation.json'), 'RESULT_BINDINGS')
    need(result['previous_result_validation'] == by_eval_path[PREV / 'result_validation.json'], 'RESULT_PREVIOUS_VALIDATION_BINDING')
    need(result['historically_opened_development'] is True and result['external_confirmation'] is False and result['deployment_changed'] is False, 'EVIDENCE_BOUNDARY')
    need(result['original_head_training_updates'] == 0 and result['new_head_training_updates'] == 2000 and result['old_predictions_copied'] is True, 'RESULT_NEW_HEAD_ONLY_SCOPE')
    need(set(result['panels']) == {'EVAL32', 'EVAL128'}, 'RESULT_PANELS')
    summaries = {}
    for panel, qs in split['panels'].items():
        output = result['panels'][panel]
        prior_rows = index(previous_result['panels'][panel]['rows'], 'query_id')
        need(set(prior_rows) == set(qs), 'PREVIOUS_PANEL_MEMBERSHIP')
        need(output['population'] == len(qs) and len(output['rows']) == len(qs), 'RESULT_PANEL_POPULATION')
        rows = []
        for q in qs:
            role, f, selected = roles[q], fi[q], chosen[q]
            raw = list(f['raw_ranked_physical_rows'])
            targets = [p for p in raw if labels[p] == role['identity']]
            need(len(targets) == 1, 'UNIQUE_FULLGALLERY_TARGET')
            target = targets[0]
            ranks, correct = {}, {}
            for key, physical in selected.items():
                ranked = raw if key == 'RAW' else [physical] + [p for p in raw if p != physical]
                need(len(ranked) == len(raw) and len(set(ranked)) == len(raw), 'FULLGALLERY_PERMUTATION')
                ranks[key] = ranked.index(target) + 1
                correct[key] = physical == target
            row = dict(query_id=q, original_query_id=role['original_query_id'], component=role['component'], group=role['group'], target_in_C128=target in f['candidate_physical_rows'], correct=correct, ranks=ranks, selected_physical_rows=selected)
            rows.append(row)
            old = prior_rows[q]
            for field in ('query_id', 'original_query_id', 'component', 'group', 'target_in_C128'):
                need(row[field] == old[field], 'PREVIOUS_ROW_METADATA_' + field)
            for key in ('RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND'):
                for field in ('correct', 'ranks', 'selected_physical_rows'):
                    need(row[field][key] == old[field][key], 'PREVIOUS_BASELINE_PERQUERY_' + field)
        need(output['rows'] == rows, 'EVERY_RESULT_ROW_SELECTION_CORRECT_RANK')
        recall = sum(r['target_in_C128'] for r in rows)
        need(output['recall_C128'] == previous_result['panels'][panel]['recall_C128'] == recall, 'C128_RECALL_COUNT')
        scores = {}
        need(set(output['scores']) == SCORE_MODELS, 'ALL_FIVE_SCORE_ARMS')
        for model in rows[0]['correct']:
            count = sum(r['correct'][model] for r in rows)
            mrr = sum((Fraction(1, r['ranks'][model]) for r in rows), Fraction()) / len(rows)
            need(output['scores'][model]['correct'] == count, 'EXACT_TOP1_COUNT')
            close(output['scores'][model]['MRR'], mrr, 'EXACT_FRACTION_MRR')
            scores[model] = dict(correct=count, MRR=float(mrr), MRR_exact=fraction(mrr))
        need((scores['RAW']['correct'], scores['ORIGINAL7']['correct']) == {'EVAL32': (25, 28), 'EVAL128': (88, 99)}[panel], 'ORIGINAL_PANEL_BASELINE_COUNTS')
        need(set(output['comparisons']) == {'ORIGINAL7__to__GROUP_MIXED96'}, 'SINGLE_ORIGINAL_TO_GROUP_COMPARISON')
        summary, means = comparison(rows, 'ORIGINAL7', 'GROUP_MIXED96')
        observed = output['comparisons']['ORIGINAL7__to__GROUP_MIXED96']
        need(set(observed) == set(summary), 'COMPARISON_FIELDS')
        for k, v in summary.items():
            if k == 'component_bootstrap95':
                need(len(observed[k]) == 2, 'BOOTSTRAP_CI_SHAPE')
                for x, y in zip(observed[k], v):
                    close(x, y, 'INDEPENDENT_COMPONENT_BOOTSTRAP95')
            elif k == 'equal_component_difference':
                close(observed[k], v, 'EXACT_COMPONENT_MEAN')
            else:
                need(observed[k] == v, 'COMPARISON_EXACT_' + k)
        summaries[panel] = dict(population=len(rows), recall_C128=recall, scores=scores,
                               comparisons={'ORIGINAL7__to__GROUP_MIXED96': dict(summary, component_mean_differences_exact=means)},
                               old_ec7_perquery_parity_count=len(rows))
    validation = dict(status='ORIGINAL_MIXED96_GROUP_INDEPENDENT_RESULT_PASS',
        result=M.bind(OUT / 'result.json'), program=M.bind(Path(__file__)),
        authority=ab, fit_validation=M.bind(OUT / 'fits/validation.json'), payload=receipt['payload'],
        train_input_seal=by_public_path[OUT / 'train_input_seal.json'],
        previous_result_validation=by_eval_path[PREV / 'result_validation.json'],
        previous_source_payload=payload['previous_source_payload'], public_sources=pins, evaluation_sources=evaluation,
        evaluation_labels_opened_after_fit_seal=True, original_head_training_updates=0,
        original7_parameter_sha256=EC7, original7_binary64=list(HEX7),
        original7_prelabel_bit_equal_logits=parity_logits,
        independent_logit_method='NumPy float64 X @ theta[:6] + theta[6]',
        maximum_absolute_logit_error=max_error, logit_tolerance=1e-12,
        independently_checked_logits=160 * 2 * 2 * 127,
        action_tie_rule='Python max earliest challenger; HOLD when maximum <= 0',
        rank_method='literal selected-physical-row move to front of full RAW gallery',
        MRR_method='exact fractions followed by float comparison at absolute tolerance 1e-12',
        train_eval_overlaps=overlaps, counts=dict(train=96, PAIR=64, FULL=32, train_groups=32, EVAL32=32, EVAL128=128),
        pool_group_image_counts=pool_counts, panels=summaries,
        historically_opened_development=True, external_confirmation=False, deployment_changed=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    M.write(OUT / 'result_validation.json', validation)
    print(validation['status'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    main()
