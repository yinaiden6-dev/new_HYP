#!/usr/bin/env python3
"""Exhaustive FULL TRAIN group deletions; four fixed heads and unchanged PAIR64."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, statistics, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc_train_group_jackknife_stability_v1'
OUT = ROOT / 'results' / NAME
PREFLIGHT = ROOT / 'results' / (NAME + '_preflight_v2') / 'result.json'
AUTH = ROOT / 'registry/rc_train_group_jackknife_stability_authority_v1_20260909.json'
PLAN = ROOT / 'plan/RC_TRAIN_GROUP_JACKKNIFE_STABILITY_V1_20260909.md'
LAUNCH = ROOT / 'slurm/rc_train_group_jackknife_stability_v1_dev_cpuonly_59m.sbatch'
PARENT = ROOT / 'results/rc_product_response_factorial_v1'
WORKER = ROOT / 'programs/run_rc_product_response_factorial_v1.py'
PARENT_AUTH = ROOT / 'registry/rc_product_response_factorial_authority_v1_20260909.json'
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
HEADS = ('JOINT4', 'PRODUCT5', 'RESPONSE6', 'ORIGINAL7')
EDGES = (('PRODUCT5', 'JOINT4'), ('ORIGINAL7', 'RESPONSE6'),
         ('RESPONSE6', 'JOINT4'), ('ORIGINAL7', 'PRODUCT5'))
MODES = ('REAL', 'CBIND')
P = H = None


def need(ok, message):
    if not bool(ok): raise RuntimeError(message)


def setup():
    global P, H
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    for relative, digest in PINS.items():
        need(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, 'SOURCE_PIN:' + relative)
    spec = importlib.util.spec_from_file_location('frozen_factorial_for_group_jackknife', WORKER)
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    P.setup(); H = P.H
    need(P.sources() == H.read(PARENT_AUTH)['sources'], 'PARENT_AUTHORITY')
    return {'program': H.binding(__file__), 'plan': H.binding(PLAN), 'launcher': H.binding(LAUNCH),
        'parent_source_pins': PINS, 'parent_authority': H.binding(PARENT_AUTH),
        'split_qualification': H.binding(P.U.SPLIT), 'research_extension': H.binding(P.U.EXT),
        'post_submission_continuation': H.binding(P.CONTINUATION)}


def prepare():
    pure, pair, train, evals, entries, labels, parent = P.prepare()
    H.BARRIER.outcomes.update(str((PARENT / f).resolve()) for f in ('result.json', 'independent_validation.json'))
    H.BARRIER.outcomes.update(str((OUT / f).resolve()) for f in ('result.json', 'independent_validation.json'))
    need(parent == H.read(PARENT / 'input_closure.json'), 'UNCHANGED_PARENT_INPUT_CLOSURE')
    split = H.read(P.U.SPLIT)
    need(split['status'] == 'RC_SHARED_QUERY_TARGET_PRIOR_SPLIT_QUALIFICATION_V1_PASS', 'QUALIFIED_SPLIT')
    for role, n, groups in (('PAIR64', 64, 20), ('FULL_TRAIN32', 32, 12), ('FULL_EVAL32', 32, 11)):
        need(split['counts'][role]['query_id'] == n and split['counts'][role]['supergroup'] == groups, 'QUALIFIED_GROUP_COUNT')
    need(all(v == 0 for row in split['overlap_counts'].values() for v in row.values()), 'NO_SPLIT_OVERLAP')
    groups = sorted({r['supergroup'] for r in train})
    need(len(groups) == 12 and all(isinstance(g, str) for g in groups), 'EXACT_12_TRAIN_GROUPS')
    need([r['execution_ordinal'] for r in train] == sorted(r['execution_ordinal'] for r in train), 'ORIGINAL_TRAIN_ORDER')
    records = []
    for index, group in enumerate([None, *groups]):
        removed = [r for r in train if group is not None and r['supergroup'] == group]
        kept = [r for r in train if group is None or r['supergroup'] != group]
        need(kept and (len(removed) > 0 if group is not None else not removed), 'NONEMPTY_EXHAUSTIVE_GROUP_DELETION')
        records.append({'condition': 'FULL_TRAIN' if group is None else 'DROP_GROUP_' + str(index).zfill(2),
            'removed_supergroup': group, 'removed_query_ids': [r['query_id'] for r in removed],
            'removed_execution_ordinals': [r['execution_ordinal'] for r in removed],
            'retained_training_query_ids': [r['query_id'] for r in kept],
            'retained_training_execution_ordinals': [r['execution_ordinal'] for r in kept],
            'remaining_TRAIN_count': len(kept), 'PAIR_count': 64})
    need(len(records) == 13 and sum(len(x['removed_query_ids']) for x in records[1:]) == 32, 'EXHAUSTIVE_13_CONDITIONS')
    return pure, pair, train, evals, entries, labels, {'parent_input_closure': parent,
        'conditions': records, 'sorted_TRAIN_supergroups': groups, 'PAIR_order_unchanged': parent['parent_input_closure']['pair_order'],
        'expected_fit_count': 52, 'head_order': list(HEADS), 'EVAL_role_reads': 0}


def fit(pure, pair, train, manifest):
    expected = H.read(PARENT / 'parameters.json')
    text = (ROOT / H.PINS['old_runner'][0]).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'train_head')
    digest = hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
    results = {}; started = time.monotonic(); fit_count = 0
    for spec in manifest['conditions']:
        P.U.deadline()
        condition = spec['condition']; kept = set(spec['retained_training_execution_ordinals'])
        rows = [r for r in train if r['execution_ordinal'] in kept]
        need([r['execution_ordinal'] for r in rows] == spec['retained_training_execution_ordinals'], 'REMAINING_TRAIN_ORDER')
        results[condition] = {}
        for name in HEADS:
            head, loss, finite = pure['train_head'](pair, rows, P.family(name), 'C_PAIRED')
            w = head.weight.detach().flatten(); b = float(head.bias.detach())
            result = {'weight_binary64': [H.hx(v) for v in w], 'bias_binary64': H.hx(b),
                'parameter_sha256': pure['parameter_sha'](w, b), 'parameter_count': len(P.COLUMNS[name]) + 1,
                'original_feature_columns': list(P.COLUMNS[name]), 'loss_total_last_recorded': loss[0],
                'PAIR_loss_last_recorded': loss[1], 'FULL_loss_last_recorded': loss[2], 'finite_training': finite,
                'unchanged_training_function_sha256': digest}
            need(finite['all_finite'] and finite['finite_loss_step_count'] == 2000, 'EXACT_FINITE_2000_STEPS')
            if condition == 'FULL_TRAIN': need(result == expected[name], 'EXACT_ALL_FOUR_BASELINE_PARAMETERS')
            results[condition][name] = result; fit_count += 1
            print(json.dumps({'event': 'TRAIN_GROUP_JACKKNIFE_HEAD_FIT', 'condition': condition, 'head': name,
                'fit': fit_count, 'of': 52, 'elapsed_seconds': time.monotonic() - started,
                'parameter_sha256': result['parameter_sha256']}), flush=True)
    need(fit_count == 52 and not H.BARRIER.released and H.BARRIER.blocked == 0, 'ALL52_FITS_BEFORE_EVAL')
    return results


def predictions(params, train, evals, manifest):
    rows = sorted(train + evals, key=lambda r: r['execution_ordinal'])
    values = {}
    for spec in manifest['conditions']:
        condition = spec['condition']
        values[condition] = [{'query_id': r['query_id'], 'execution_ordinal': r['execution_ordinal'],
            'predictions': {name: {mode: P.predict(r, params[condition][name], name, mode) for mode in MODES} for name in HEADS}} for r in rows]
    need(values['FULL_TRAIN'] == H.read(PARENT / 'eval_prejoin.json'), 'EXACT_BASELINE_FULL64_PREDICTIONS')
    need(H.BARRIER.blocked == 0 and not H.BARRIER.released, 'ALL_PREDICTIONS_BEFORE_EVAL')
    return values


def literal_metric(rows, predictions_for_rows, actions):
    """Compute selection and ranking from sealed hex logits, independently of source actions."""
    details = []
    for row, pred, actual in zip(rows, predictions_for_rows, actions, strict=True):
        cs = [int(x) for x in row['challenger_positions']]; axis = [int(x) for x in row['candidate_physical_rows']]
        zs = [float.fromhex(x) for x in pred['all127_logits_binary64']]
        index = sorted(range(127), key=lambda i: (-zs[i], axis[cs[i]]))[0]
        best = cs[index]; winner = int(row['base_winner_position']); target = int(row['target_position'])
        switch = zs[index] > 0; final = best if switch else winner
        need(final == pred['final_position'] and pred['decision'] == ('SWITCH' if switch else 'HOLD'), 'LITERAL_SEALED_ACTION')
        raw = [float(x) for x in row['base_scores']]
        base_order = sorted(range(128), key=lambda i: (-raw[i], axis[i]))
        final_order = ([best] + [i for i in base_order if i != best]) if switch else base_order
        v = {'query_id': row['query_id'], 'base_correct': winner == target, 'final_correct': final == target,
             'final_position': final, 'proposed_challenger': best, 'switch_logit': zs[index],
             'base_target_rank': base_order.index(target) + 1, 'final_target_rank': final_order.index(target) + 1,
             'decision': 'SWITCH' if switch else 'HOLD', 'wrong_to_wrong': winner != target and final != target and final != winner}
        need(all(v[k] == actual[k] for k in v), 'INDEPENDENT_ACTION_AND_RANK_FIELDS')
        details.append(v)
    n = len(details); base = sum(r['base_correct'] for r in details); final = sum(r['final_correct'] for r in details)
    return {'query_count': n, 'base_top1': base, 'final_top1': final, 'base_R@1': base / n, 'final_R@1': final / n,
        'base_MRR': sum(1 / r['base_target_rank'] for r in details) / n,
        'final_MRR': sum(1 / r['final_target_rank'] for r in details) / n,
        'rescue': sum(not r['base_correct'] and r['final_correct'] for r in details),
        'break': sum(r['base_correct'] and not r['final_correct'] for r in details),
        'retained_correct': sum(r['base_correct'] and r['final_correct'] for r in details),
        'retained_wrong': sum(not r['base_correct'] and not r['final_correct'] for r in details),
        'wrong_to_wrong': sum(r['wrong_to_wrong'] for r in details),
        'switch_count': sum(r['decision'] == 'SWITCH' for r in details), 'hold_count': sum(r['decision'] == 'HOLD' for r in details)}


def distribution(values):
    need(len(values) == 12, 'ONLY12_DELETION_CONDITIONS_IN_SENSITIVITY')
    return {'count': 12, 'min': min(values), 'median': statistics.median(values), 'max': max(values),
        'positive': sum(v > 0 for v in values), 'zero': sum(v == 0 for v in values), 'negative': sum(v < 0 for v in values)}


def checked_paired(new, base, rows):
    value = P.paired(new, base, rows)
    new_correct = {a['query_id'] for a in new if a['final_correct']}
    base_correct = {a['query_id'] for a in base if a['final_correct']}
    rescued, broken = new_correct - base_correct, base_correct - new_correct
    need(set(value['rescue_query_ids']) == rescued and set(value['break_query_ids']) == broken, 'INDEPENDENT_PAIRED_CORRECT_SETS')
    need((value['rescue'], value['break'], value['net']) == (len(rescued), len(broken), len(rescued) - len(broken)), 'INDEPENDENT_PAIRED_COUNTS')
    groups = {g: {r['query_id'] for r in rows if r['supergroup'] == g} for g in sorted({r['supergroup'] for r in rows})}
    independent = {g: {'queries': len(ids), 'new_correct': len(ids & new_correct), 'base_correct': len(ids & base_correct),
                      'net': len(ids & new_correct) - len(ids & base_correct)} for g, ids in groups.items()}
    need(independent == value['supergroups'], 'INDEPENDENT_SUPERGROUP_COUNTS')
    average = math.fsum(v['net'] / v['queries'] for v in independent.values()) / len(independent)
    need(math.isclose(average, value['group_balanced_accuracy_difference'], rel_tol=0, abs_tol=1e-15), 'INDEPENDENT_GROUP_BALANCED_AVERAGE')
    need(value['groups_improved'] == sum(v['net'] > 0 for v in independent.values()) and
         value['groups_harmed'] == sum(v['net'] < 0 for v in independent.values()), 'INDEPENDENT_GROUP_DIRECTIONS')
    return value


def summarize(params, preds, pure, train, evals, entries, labels, manifest):
    seal = H.read(OUT / 'eval_prejoin_seal.json')
    need(seal['parameters_sha256'] == H.sha(OUT / 'parameters.json') and
         seal['predictions_sha256'] == H.sha(OUT / 'all_conditions_prejoin.json') and H.BARRIER.blocked == 0, 'ALL_CONDITIONS_SEALED')
    H.BARRIER.release(); joined = H.role_join(evals, entries, labels)
    need(len({r['supergroup'] for r in joined}) == 11, 'EVAL_11_GROUPS_AFTER_SEAL')
    need(all(not ({r[k] for r in train} & {r[k] for r in joined}) for k in ('query_id', 'target_identity', 'supergroup')), 'TRAIN_EVAL_DISJOINT')
    previous = H.read(PARENT / 'result.json'); validation = H.read(PARENT / 'independent_validation.json')
    need(validation['result_sha256'] == H.sha(PARENT / 'result.json') and all(validation['checks'].values()), 'VALIDATED_PARENT')
    actions = {}; metrics = {}; edges = {}; within = {}; retained_metrics = {}
    for spec in manifest['conditions']:
        condition = spec['condition']; by_id = {r['query_id']: r for r in preds[condition]}
        actions[condition] = {}; metrics[condition] = {}; edges[condition] = {}; within[condition] = {}; retained_metrics[condition] = {}
        for role, rows in (('TRAIN', train), ('EVAL', joined)):
            actions[condition][role] = {}; metrics[condition][role] = {}; edges[condition][role] = {}; within[condition][role] = {}
            for mode in MODES:
                actions[condition][role][mode] = {}; metrics[condition][role][mode] = {}; within[condition][role][mode] = {}
                for name in HEADS:
                    w, b = P.tensors(params[condition][name])
                    aa = pure['actions'](w, b, rows, P.family(name), 'C_PAIRED', control=mode == 'CBIND')
                    pp = [by_id[r['query_id']]['predictions'][name][mode] for r in rows]
                    independent = literal_metric(rows, pp, aa)
                    need(independent == pure['summary'](aa), 'INDEPENDENT_METRIC_ARITHMETIC')
                    if condition == 'FULL_TRAIN':
                        need(H.encode(aa) == H.encode(previous['actions'][role][name][mode]), 'ALL_PARENT_ACTION_FIELDS_EXACT')
                        need(independent == previous['metrics'][role][name][mode], 'PARENT_METRICS_EXACT')
                    actions[condition][role][mode][name] = aa; metrics[condition][role][mode][name] = independent
                    within[condition][role][mode][name] = checked_paired(aa, actions['FULL_TRAIN'][role][mode][name], rows)
                edges[condition][role][mode] = {a + '_vs_' + b: checked_paired(actions[condition][role][mode][a], actions[condition][role][mode][b], rows) for a, b in EDGES}
        kept = set(spec['retained_training_query_ids'])
        for mode in MODES:
            retained_metrics[condition][mode] = {name: pure['summary']([a for a in actions[condition]['TRAIN'][mode][name] if a['query_id'] in kept]) for name in HEADS}
    deleted = [s['condition'] for s in manifest['conditions'][1:]]
    sensitivity = {}; frequencies = {}
    for role in ('TRAIN', 'EVAL'):
        sensitivity[role] = {}; frequencies[role] = {}
        for mode in MODES:
            sensitivity[role][mode] = {'head_accuracy_counts': {}, 'within_head_vs_FULL_TRAIN': {}, 'factorial_edges': {}}
            frequencies[role][mode] = {}
            for name in HEADS:
                sensitivity[role][mode]['head_accuracy_counts'][name] = distribution([metrics[c][role][mode][name]['final_top1'] for c in deleted])
                sensitivity[role][mode]['within_head_vs_FULL_TRAIN'][name] = {
                    'query_net': distribution([within[c][role][mode][name]['net'] for c in deleted]),
                    'group_balanced_accuracy_difference': distribution([within[c][role][mode][name]['group_balanced_accuracy_difference'] for c in deleted])}
                qq = []
                for index, base in enumerate(actions['FULL_TRAIN'][role][mode][name]):
                    variants = [actions[c][role][mode][name][index] for c in deleted]
                    need(all(a['query_id'] == base['query_id'] for a in variants), 'QUERY_FREQUENCY_AXIS')
                    correct = sum(a['final_correct'] for a in variants)
                    same_decision = sum(a['decision'] == base['decision'] for a in variants)
                    same_final = sum(a['final_physical_row'] == base['final_physical_row'] for a in variants)
                    same_identity = sum(labels[a['final_physical_row']] == labels[base['final_physical_row']] for a in variants)
                    qq.append({'query_id': base['query_id'], 'baseline_final_correct': base['final_correct'],
                        'baseline_decision': base['decision'], 'baseline_final_physical_row': base['final_physical_row'],
                        'baseline_final_reference_identity': labels[base['final_physical_row']],
                        'deletion_condition_count': 12, 'correct_count': correct, 'correct_frequency': correct / 12,
                        'decision_agreement_count': same_decision, 'decision_agreement_rate': same_decision / 12,
                        'final_physical_candidate_agreement_count': same_final, 'final_physical_candidate_agreement_rate': same_final / 12,
                        'final_reference_identity_agreement_count': same_identity, 'final_reference_identity_agreement_rate': same_identity / 12,
                        'incorrect_when_baseline_correct_count': sum(base['final_correct'] and not a['final_correct'] for a in variants),
                        'correct_when_baseline_incorrect_count': sum(not base['final_correct'] and a['final_correct'] for a in variants)})
                frequencies[role][mode][name] = qq
            for a, b in EDGES:
                key = a + '_vs_' + b
                sensitivity[role][mode]['factorial_edges'][key] = {
                    'query_net': distribution([edges[c][role][mode][key]['net'] for c in deleted]),
                    'group_balanced_accuracy_difference': distribution([edges[c][role][mode][key]['group_balanced_accuracy_difference'] for c in deleted])}
    return {'status': 'TRAIN_GROUP_JACKKNIFE_DESCRIPTIVE_STABILITY_COMPLETE', 'theory_name': 'new HYP',
        'authority_sha256': H.sha(AUTH), 'parameters_sha256': H.sha(OUT / 'parameters.json'),
        'eval_prejoin_seal_sha256': H.sha(OUT / 'eval_prejoin_seal.json'), 'input_closure_sha256': H.sha(OUT / 'input_closure.json'),
        'conditions': manifest['conditions'], 'metrics': metrics, 'actions': actions,
        'remaining_TRAIN_subset_metrics': retained_metrics, 'same_head_vs_FULL_TRAIN': within, 'factorial_comparisons': edges,
        'deletion_condition_sensitivity': sensitivity, 'query_correctness_and_agreement': frequencies,
        'baseline_condition': 'FULL_TRAIN', 'deletion_condition_count': 12, 'fits_per_execution': 52, 'steps_per_fit': 2000,
        'split_counts': {'PAIR': 64, 'FULL_TRAIN': 32, 'EVAL': 32}, 'supergroup_counts': {'PAIR': 20, 'FULL_TRAIN': 12, 'EVAL': 11},
        'candidate_source': 'original RAW full-gallery C128, all127 challenger zero-threshold HOLD/SWITCH',
        'task_supervision': 'retrieval identity and original PAIR labels only; no task spatial annotations',
        'new_encoder_or_RoMa_forwards': 0, 'best_deletion_selected': False, 'model_adopted': False, 'HYP_GO_claimed': False,
        'checks': {'all52_fits_original_optimizer': True, 'all13_conditions_sealed_before_EVAL': True,
                   'all_FULL_TRAIN_parameters_predictions_actions_exact': True, 'independent_literal_actions_and_metrics': True,
                   'independent_paired_sets_and_group_metrics': True,
                   'exact_12_training_groups_exhaustive_deletion': True, 'complete_PAIR_order_unchanged': True},
        'limits': ['Twelve highly dependent training perturbations, not twelve independent external test sets.',
                   'All deletion summaries exclude FULL_TRAIN baseline and use denominator12.',
                   'TRAIN metrics use the complete original TRAIN32; remaining_TRAIN_subset_metrics separately use each fitted subset.',
                   'Descriptive sensitivity only; no p values, favorable deletion selection, deployment, or population reliability proof.',
                   'Deleted FULL groups have unequal sizes; remaining FULL mean loss renormalizes retained rows while PAIR loss stays fixed.',
                   'All20 PAIR groups remain unperturbed; this does not test sensitivity to the PAIR supervision pool.',
                   'Final physical candidate and corrected reference identity agreement are separate from HOLD/SWITCH agreement.']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('preflight', 'freeze', 'run', 'validate'), required=True)
    phase = parser.parse_args().phase; sources = setup()
    if phase == 'freeze':
        need(not AUTH.exists() and not OUT.exists(), 'APPEND_ONLY_FREEZE')
        pre = H.read(PREFLIGHT); need(pre['status'] == 'TRAIN_GROUP_JACKKNIFE_PREFLIGHT_PASS' and pre['sources'] == sources, 'MATCHED_PREFLIGHT')
        H.atomic(AUTH, {'status': 'TRAIN_GROUP_JACKKNIFE_STABILITY_AUTHORIZED', 'sources': sources, 'preflight': H.binding(PREFLIGHT),
            'condition_manifest': pre['input_closure']['conditions'], 'fits_per_execution': 52, 'fresh_validation_fit_count': 52,
            'head_order': list(HEADS), 'factorial_edges': [list(x) for x in EDGES], 'steps': 2000, 'seed': 17,
            'loss': 'unchanged PAIR loss plus remaining TRAIN mean sign loss', 'cutoff_UTC': H.read(P.U.EXT)['cutoff_UTC']})
        print(json.dumps({'authority_sha256': H.sha(AUTH)}), flush=True); return
    pure, pair, train, evals, entries, labels, manifest = prepare()
    if phase == 'preflight':
        need(H.BARRIER.blocked == 0 and not H.BARRIER.released, 'NO_EARLY_EVAL_PREFLIGHT')
        expected = H.read(PARENT / 'parameters.json')
        training_actions = {}
        for mode in MODES:
            training_actions[mode] = {}
            for name in HEADS:
                w, b = P.tensors(expected[name])
                aa = pure['actions'](w, b, train, P.family(name), 'C_PAIRED', control=mode == 'CBIND')
                pp = [P.predict(r, expected[name], name, mode) for r in train]
                need(literal_metric(train, pp, aa) == pure['summary'](aa), 'TRAIN_ONLY_LITERAL_METRIC_PREFLIGHT')
                training_actions[mode][name] = aa
            for a, b in EDGES: checked_paired(training_actions[mode][a], training_actions[mode][b], train)
        need(H.BARRIER.blocked == 0 and not H.BARRIER.released, 'TRAIN_ONLY_SMOKE_NO_EVAL')
        H.atomic(PREFLIGHT, {'status': 'TRAIN_GROUP_JACKKNIFE_PREFLIGHT_PASS', 'sources': sources,
            'input_closure': manifest, 'training_updates': 0, 'runtime_EVAL_target_reads': 0,
            'checks': {'parent_native_subset_features_exact': True, 'all12_training_groups_and_13_conditions': True,
                'original_PAIR_and_remaining_TRAIN_order': True, 'split_qualification_20_12_11': True,
                'TRAIN_only_existing_head_literal_actions_metrics_and_edges': True}})
        print(json.dumps({'status': 'TRAIN_GROUP_JACKKNIFE_PREFLIGHT_PASS', 'condition_count': 13,
            'removed_group_sizes': [len(c['removed_query_ids']) for c in manifest['conditions'][1:]], 'fits_per_execution': 52}), flush=True); return
    authority = H.read(AUTH)
    need(authority['sources'] == sources and authority['condition_manifest'] == manifest['conditions'], 'FROZEN_AUTHORITY')
    validate = phase == 'validate'; need(OUT.exists() if validate else not OUT.exists(), 'APPEND_ONLY_OUTPUT')
    if validate: need(H.read(OUT / 'input_closure.json') == manifest, 'FRESH_INPUT_CLOSURE')
    else: H.atomic(OUT / 'input_closure.json', manifest)
    params = fit(pure, pair, train, manifest)
    if validate: need(H.read(OUT / 'parameters.json') == params, 'FRESH_52_PARAMETER_RETRAINS')
    else: H.atomic(OUT / 'parameters.json', params)
    preds = predictions(params, train, evals, manifest)
    if validate: need(H.read(OUT / 'all_conditions_prejoin.json') == preds, 'FRESH_ALL13_PREDICTIONS')
    else:
        H.atomic(OUT / 'all_conditions_prejoin.json', preds)
        H.atomic(OUT / 'eval_prejoin_seal.json', {'parameters_sha256': H.sha(OUT / 'parameters.json'),
            'predictions_sha256': H.sha(OUT / 'all_conditions_prejoin.json'), 'condition_count': 13, 'fits': 52,
            'FULL_query_count': 64, 'EVAL_query_count': 32, 'candidate_count': 128, 'runtime_EVAL_target_reads': 0})
    result = summarize(params, preds, pure, train, evals, entries, labels, manifest)
    if validate:
        need(H.read(OUT / 'result.json') == result, 'INDEPENDENT_COMPLETE_RESULT')
        H.atomic(OUT / 'independent_validation.json', {'status': 'TRAIN_GROUP_JACKKNIFE_INDEPENDENT_REEXECUTION_PASS',
            'result_sha256': H.sha(OUT / 'result.json'), 'checks': result['checks'], 'fit_count': 52,
            'scope': 'fresh-process complete52 retraining, exact parameter/logit replay, independent literal action metrics'})
    else: H.atomic(OUT / 'result.json', result)
    print(json.dumps({'status': result['status'], 'validation': validate, 'EVAL_REAL_sensitivity': result['deletion_condition_sensitivity']['EVAL']['REAL']}), flush=True)


if __name__ == '__main__': main()
