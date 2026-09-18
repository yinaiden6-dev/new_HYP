#!/usr/bin/env python3
"""Frozen NATIVE7 group interventions and fixed-refit decomposition, no training."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_frozen_group_effect_native64_v1'
PLAN = ROOT / 'plan/RC_FROZEN_GROUP_EFFECT_DECOMPOSITION_V1_20260909.md'
PARENT = ROOT / 'results/rc_product_response_factorial_v1'
WORKER = ROOT / 'programs/run_rc_product_response_factorial_v1.py'
AUTH = ROOT / 'registry/rc_product_response_factorial_authority_v1_20260909.json'
PINS = {
    'programs/run_rc_product_response_factorial_v1.py': '3d8190a481de2dc4bf76a5be7a1d9c795d1faec8471271c309fe7bff54bf23f3',
    'registry/rc_product_response_factorial_authority_v1_20260909.json': '30109e4a00633970a3e2278edd1a0451c73d6fa08285dc0a328447bacca44b90',
    'results/rc_product_response_factorial_v1/result.json': 'e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf',
    'results/rc_product_response_factorial_v1/parameters.json': '019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43',
    'results/rc_product_response_factorial_v1/eval_prejoin.json': '3ed9dd108c5e50c8ba9d84fdef1f10a06589301f19b3a07d950cb4f76afe4011',
    'results/rc_product_response_factorial_v1/eval_prejoin_seal.json': '571dd9b1b50605018f2ef7051f20add51ec75b17287fe2f649fa36d8f69127ea',
    'results/rc_product_response_factorial_v1/independent_validation.json': 'f91c01caac1824ceb8cde358a9c22f95b94fe0d846965b35815c66cc778ed3bb',
    'results/rc_product_response_factorial_v1/input_closure.json': 'ee1a9fc70aff90c361e8b8b336400b90a07cad000267d99eb1a6fd32f316279a',
}
GROUPS = {'DROP_S': ((1,), 'RESPONSE6'), 'DROP_QR': ((4, 5), 'PRODUCT5'),
          'DROP_S_QR': ((1, 4, 5), 'JOINT4')}
MODES = ('REAL', 'CBIND')
CONDITIONS = ('ORIGINAL7', *GROUPS, 'REFIT_RESPONSE6', 'REFIT_PRODUCT5', 'REFIT_JOINT4')
P = H = None


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def init():
    global P, H
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    for relative, expected in PINS.items():
        need(sha(ROOT / relative) == expected, 'SOURCE_PIN:' + relative)
    need(PLAN.is_file(), 'SHARED_PLAN_REQUIRED')
    spec = importlib.util.spec_from_file_location('frozen_factorial_inputs_for_group_effect', WORKER)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    P.setup()
    H = P.H
    need(P.sources() == H.read(AUTH)['sources'], 'FROZEN_FACTORIAL_AUTHORITY')
    return {'source_pins': PINS, 'program': H.binding(__file__), 'plan': H.binding(PLAN),
            'parent_authority': H.binding(AUTH), 'training_updates': 0,
            'conditions': list(CONDITIONS), 'groups': {k: {'removed_columns': list(v[0]), 'refit': v[1]} for k, v in GROUPS.items()}}


def predict(row, values):
    need(values.shape == (127,), 'ALL127_LOGITS')
    need(bool(__import__('torch').isfinite(values).all()), 'FINITE_PREDICTION')
    cs, axis = row['challenger_positions'], row['candidate_physical_rows']
    i = max(range(127), key=lambda j: (float(values[j]), -int(axis[int(cs[j])])) )
    return {'final_position': int(cs[i]) if float(values[i]) > 0 else int(row['base_winner_position']),
            'decision': 'SWITCH' if float(values[i]) > 0 else 'HOLD',
            'all127_logits_binary64': [H.hx(v) for v in values]}


def prepare():
    pure, pair, train, evals, entries, labels, closure = P.prepare()
    H.BARRIER.outcomes.update(str((PARENT / f).resolve()) for f in ('result.json', 'independent_validation.json'))
    need(closure == H.read(PARENT / 'input_closure.json'), 'PARENT_INPUT_CLOSURE_REBUILT')
    params = H.read(PARENT / 'parameters.json')
    old_prejoin = H.read(PARENT / 'eval_prejoin.json')
    old_seal = H.read(PARENT / 'eval_prejoin_seal.json')
    need(old_seal['parameters_sha256'] == H.sha(PARENT / 'parameters.json') and
         old_seal['eval_prejoin_sha256'] == H.sha(PARENT / 'eval_prejoin.json'), 'PARENT_PREJOIN_SEAL')
    return pure, train, evals, entries, labels, closure, params, old_prejoin


def calculate(train, evals, params, old_prejoin):
    import torch
    old = {r['query_id']: r for r in old_prejoin}
    need(len(old) == 64, 'PARENT_FULL64_AXIS')
    full_w, full_b = P.tensors(params['ORIGINAL7'])
    records = []
    for row in sorted(train + evals, key=lambda r: r['execution_ordinal']):
        rec = {'query_id': row['query_id'], 'execution_ordinal': row['execution_ordinal'],
               'data_split_role': row['data_split_role'], 'candidate_physical_rows': row['candidate_physical_rows'],
               'challenger_positions': row['challenger_positions'], 'base_winner_position': int(row['base_winner_position']),
               'base_scores_binary64': [H.hx(v) for v in row['base_scores']], 'modes': {}}
        for mode in MODES:
            x = row['real_native_features' if mode == 'REAL' else 'cbind_native_features']['C_PAIRED']
            need(x.shape == (127, 6) and x.dtype == torch.float64, 'ORIGINAL_SIX_COLUMN_AXIS')
            original = x @ full_w + full_b
            pred = {'ORIGINAL7': predict(row, original)}
            need(pred['ORIGINAL7'] == old[row['query_id']]['predictions']['ORIGINAL7'][mode], 'ORIGINAL_ALL_LOGITS_BITS')
            parts = {}
            for condition, (removed, refit) in GROUPS.items():
                modified = x.clone()
                modified[:, list(removed)] = 0.0
                fixed = modified @ full_w + full_b
                pred[condition] = predict(row, fixed)
                refit_w, refit_b = P.tensors(params[refit])
                kept = list(P.COLUMNS[refit])
                refit_values = row[P.keys(refit)[0 if mode == 'REAL' else 1]]['C_PAIRED'] @ refit_w + refit_b
                pred['REFIT_' + refit] = predict(row, refit_values)
                need(pred['REFIT_' + refit] == old[row['query_id']]['predictions'][refit][mode], 'REFIT_ALL_LOGITS_BITS')
                direct = (x[:, list(removed)] * full_w[list(removed)]).sum(1)
                shared = (x[:, kept] * (full_w[kept] - refit_w)).sum(1)
                bias_delta = full_b - refit_b
                arithmetic_sum = (direct + shared) + bias_delta
                observed_delta = original - refit_values
                residual = observed_delta - arithmetic_sum
                removal_delta = original - fixed
                compensation_delta = fixed - refit_values
                # These are explanatory terms, not a substitute for recomputing complete logits.
                for value in (direct, shared, arithmetic_sum, observed_delta, residual, removal_delta, compensation_delta):
                    need(bool(torch.isfinite(value).all()), 'FINITE_DECOMPOSITION')
                need(float(residual.abs().max()) <= 1e-12 * max(1.0, float(original.abs().max()), float(refit_values.abs().max())), 'EXCESSIVE_FLOAT_RECOMPOSITION_RESIDUAL')
                parts[condition] = {'refit_model': refit, 'removed_columns': list(removed), 'kept_columns': kept,
                    'direct_original_group_binary64': [H.hx(v) for v in direct],
                    'shared_weight_change_binary64': [H.hx(v) for v in shared], 'bias_change_binary64': H.hx(bias_delta),
                    'actual_original_minus_refit_binary64': [H.hx(v) for v in observed_delta],
                    'arithmetic_recomposition_binary64': [H.hx(v) for v in arithmetic_sum],
                    'floating_recomposition_residual_binary64': [H.hx(v) for v in residual],
                    'actual_original_minus_frozen_drop_binary64': [H.hx(v) for v in removal_delta],
                    'actual_frozen_drop_minus_refit_binary64': [H.hx(v) for v in compensation_delta]}
            rec['modes'][mode] = {'original_feature_sha256': H.tensor_sha(x),
                'features_binary64': [[H.hx(v) for v in line] for line in x], 'predictions': pred, 'decompositions': parts}
        records.append(rec)
    need(len(records) == 64 and len({r['query_id'] for r in records}) == 64, 'ALL64_UNIQUE_PREDICTIONS')
    need(H.BARRIER.blocked == 0 and not H.BARRIER.released, 'NO_EARLY_EVAL_READ')
    return records


def literal_action(row, prediction):
    logits = [float.fromhex(v) for v in prediction['all127_logits_binary64']]
    cs = [int(v) for v in row['challenger_positions']]
    axis = [int(v) for v in row['candidate_physical_rows']]
    index = sorted(range(len(cs)), key=lambda j: (-logits[j], axis[cs[j]]))[0]
    final = cs[index] if logits[index] > 0.0 else int(row['base_winner_position'])
    decision = 'SWITCH' if logits[index] > 0.0 else 'HOLD'
    need(final == prediction['final_position'] and decision == prediction['decision'], 'LITERAL_ACTION_REPLAY')
    base = int(row['base_winner_position']); target = int(row['target_position'])
    return {'query_id': row['query_id'], 'execution_ordinal': int(row['execution_ordinal']),
            'supergroup': row['supergroup'], 'base_winner': base, 'target_position': target,
            'proposed_challenger': cs[index], 'switch_logit': logits[index], 'decision': decision,
            'final_position': final, 'base_correct': base == target, 'final_correct': final == target,
            'wrong_to_wrong': base != target and final != target and final != base}


def metric(actions):
    return {'query_count': len(actions), 'base_top1': sum(a['base_correct'] for a in actions),
        'final_top1': sum(a['final_correct'] for a in actions),
        'rescue': sum(not a['base_correct'] and a['final_correct'] for a in actions),
        'break': sum(a['base_correct'] and not a['final_correct'] for a in actions),
        'switch_count': sum(a['decision'] == 'SWITCH' for a in actions),
        'wrong_to_wrong': sum(a['wrong_to_wrong'] for a in actions)}


def compare(new, base):
    need([a['query_id'] for a in new] == [a['query_id'] for a in base], 'PAIRED_QUERY_AXIS')
    positive = sorted(a['query_id'] for a, b in zip(new, base) if a['final_correct'] and not b['final_correct'])
    negative = sorted(a['query_id'] for a, b in zip(new, base) if b['final_correct'] and not a['final_correct'])
    groups = {}
    for a, b in zip(new, base):
        g = groups.setdefault(a['supergroup'], {'queries': 0, 'new_correct': 0, 'base_correct': 0})
        g['queries'] += 1; g['new_correct'] += int(a['final_correct']); g['base_correct'] += int(b['final_correct'])
    for g in groups.values(): g['net'] = g['new_correct'] - g['base_correct']
    old_rescues = {b['query_id'] for b in base if not b['base_correct'] and b['final_correct']}
    new_correct = {a['query_id'] for a in new if a['final_correct']}
    return {'rescue': len(positive), 'break': len(negative), 'net': len(positive) - len(negative),
        'rescue_query_ids': positive, 'break_query_ids': negative, 'supergroups': groups,
        'baseline_rescues_retained': len(old_rescues & new_correct),
        'baseline_rescues_lost_query_ids': sorted(old_rescues - new_correct),
        'group_balanced_accuracy_difference': sum(g['net'] / g['queries'] for g in groups.values()) / len(groups)}


def summarize(records, pure, train, evals, entries, labels, params):
    seal = H.read(OUT / 'prejoin_seal.json')
    need(seal['predictions_sha256'] == H.sha(OUT / 'all64_prejoin.json') and
         seal['parameters_sha256'] == H.sha(OUT / 'frozen_parameters.json') and H.BARRIER.blocked == 0, 'PREJOIN_SEAL_REQUIRED')
    H.BARRIER.release()
    joined = H.role_join(evals, entries, labels)
    old = H.read(PARENT / 'result.json'); val = H.read(PARENT / 'independent_validation.json')
    need(val['result_sha256'] == H.sha(PARENT / 'result.json') and all(val['checks'].values()), 'PARENT_INDEPENDENT_VALIDATION')
    by_id = {r['query_id']: r for r in records}
    summaries = {}; actions = {}; comparisons = {}; query_facts = []
    for role, rows in (('TRAIN', train), ('EVAL', joined)):
        need(len(rows) == 32, 'ALL32_ROLE_AXIS')
        actions[role] = {}; summaries[role] = {}; comparisons[role] = {}
        for mode in MODES:
            actions[role][mode] = {}; summaries[role][mode] = {}; comparisons[role][mode] = {}
            for condition in CONDITIONS:
                aa = [literal_action(row, by_id[row['query_id']]['modes'][mode]['predictions'][condition]) for row in rows]
                actions[role][mode][condition] = aa; summaries[role][mode][condition] = metric(aa)
                if condition == 'ORIGINAL7' or condition.startswith('REFIT_'):
                    name = condition.removeprefix('REFIT_'); w, b = P.tensors(params[name])
                    source_actions = pure['actions'](w, b, rows, P.family(name), 'C_PAIRED', control=mode == 'CBIND')
                    need(H.encode(source_actions) == H.encode(old['actions'][role][name][mode]), 'ALL_TRAIN_EVAL_PARENT_ACTION_FIELDS')
                    for a, source in zip(aa, source_actions):
                        need(all(a[k] == source[k] for k in a if k != 'supergroup'), 'LITERAL_SOURCE_ACTION_FIELDS')
                    need(all(summaries[role][mode][condition][k] == old['metrics'][role][name][mode][k] for k in summaries[role][mode][condition]), 'PARENT_SUMMARIES')
            for condition, (_, refit) in GROUPS.items():
                for new, base in ((condition, 'ORIGINAL7'), ('REFIT_' + refit, condition), ('REFIT_' + refit, 'ORIGINAL7')):
                    comparisons[role][mode][new + '_vs_' + base] = compare(actions[role][mode][new], actions[role][mode][base])
        for row in rows:
            q = {'query_id': row['query_id'], 'role': role, 'supergroup': row['supergroup'], 'target_position': row['target_position'],
                 'base_winner_position': row['base_winner_position'], 'conditions': {}}
            cs = row['challenger_positions']; target = row['target_position']; base = row['base_winner_position']
            for condition in CONDITIONS:
                a = next(a for a in actions[role]['REAL'][condition] if a['query_id'] == row['query_id'])
                z = [float.fromhex(v) for v in by_id[row['query_id']]['modes']['REAL']['predictions'][condition]['all127_logits_binary64']]
                if target == base:
                    margin = -max(z); target_logit = None; wrong_logit = max(z)
                else:
                    ti = cs.index(target); target_logit = z[ti]; wrong_logit = max(v for i, v in enumerate(z) if i != ti)
                    margin = min(target_logit, target_logit - wrong_logit)
                q['conditions'][condition] = {**a, 'correct_action_margin': margin,
                    'target_logit': target_logit, 'strongest_wrong_challenger_logit': wrong_logit}
            query_facts.append(q)
    residual = max(abs(float.fromhex(v)) for r in records for m in MODES for d in r['modes'][m]['decompositions'].values() for v in d['floating_recomposition_residual_binary64'])
    return {'status': 'FROZEN_NATIVE64_GROUP_EFFECT_DECOMPOSITION_COMPLETE', 'theory_name': 'new HYP',
        'sources_sha256': H.sha(OUT / 'sources.json'), 'prejoin_seal_sha256': H.sha(OUT / 'prejoin_seal.json'),
        'metrics': summaries, 'comparisons': comparisons, 'actions': actions, 'all64_query_facts': query_facts,
        'focus_query_facts': [q for q in query_facts if q['query_id'] in ('OUTCOME-0618', 'OUTCOME-0212')],
        'maximum_absolute_float_recomposition_residual': residual,
        'checks': {'all_original_and_refit_parameters_predictions_bound': True, 'all_parent_TRAIN_EVAL_REAL_CBIND_actions_exact': True,
                   'all127_reselected_with_same_arithmetic_and_threshold': True, 'all64_logits_sealed_before_EVAL_roles': True,
                   'literal_action_metrics_independently_checked': True},
        'candidate_source': 'original RAW full-gallery C128; all127 challenger HOLD/SWITCH; NATIVE7 head lineage',
        'task_supervision': 'original retrieval identity/pair labels only; no spatial task annotations',
        'training_updates': 0, 'new_encoder_or_RoMa_forwards': 0, 'HYP_GO_claimed': False, 'model_adopted': False,
        'limits': ['Internal fixed-head computation intervention and fixed-refit comparison on previously opened FULL64.',
                   'Grouped feature zeroing does not remove all quality or content information from other features.',
                   'Direct/shared/bias decomposition has the reported floating arithmetic residual; predictions use full original matmul.',
                   'Refit differences describe existing optimized heads, not independent population confirmation or new training.']}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=('produce', 'validate'), required=True)
    validate = p.parse_args().phase == 'validate'
    sources = init()
    need(OUT.exists() if validate else not OUT.exists(), 'APPEND_ONLY_OUTPUT_STATE')
    pure, train, evals, entries, labels, closure, params, old_prejoin = prepare()
    records = calculate(train, evals, params, old_prejoin)
    values = {'sources.json': sources, 'input_closure.json': closure, 'frozen_parameters.json': params, 'all64_prejoin.json': records}
    for filename, value in values.items():
        if validate: need(H.read(OUT / filename) == value, 'FRESH_REBUILD:' + filename)
        else: H.atomic(OUT / filename, value)
    if not validate:
        H.atomic(OUT / 'prejoin_seal.json', {'parameters_sha256': H.sha(OUT / 'frozen_parameters.json'),
            'predictions_sha256': H.sha(OUT / 'all64_prejoin.json'), 'conditions': list(CONDITIONS),
            'FULL_queries': 64, 'EVAL_queries': 32, 'runtime_EVAL_role_reads': 0, 'training_updates': 0})
    result = summarize(records, pure, train, evals, entries, labels, params)
    if validate:
        need(H.read(OUT / 'result.json') == result, 'FRESH_COMPLETE_RESULT_REPLAY')
        H.atomic(OUT / 'independent_validation.json', {'status': 'FROZEN_NATIVE64_GROUP_EFFECT_FRESH_REEXECUTION_PASS',
            'result_sha256': H.sha(OUT / 'result.json'), 'checks': result['checks'],
            'scope': 'fresh-process original-input rebuild, all-logit recomputation, literal action/metric checks; no training'})
    else: H.atomic(OUT / 'result.json', result)
    print(json.dumps({'status': result['status'], 'validation': validate, 'EVAL_REAL': result['metrics']['EVAL']['REAL'],
                      'residual': result['maximum_absolute_float_recomposition_residual']}), flush=True)


if __name__ == '__main__':
    main()
