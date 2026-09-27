#!/usr/bin/env python3
"""Independent descriptive-M audit from native per-query operator receipts.

Does not import the audited analysis, perform fitting, choose thresholds, or
run either encoder or matcher. All original experiment artifacts are read-only.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_m_scalar_headroom_audit_20260927_v1'
TOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(ref):
    assert binding(ref['path']) == ref, ref
    return read(ref['path'])


def main():
    result_path = OUT / 'result.json'
    reported = read(result_path)
    assert reported['status'] == 'H593_M_SCALAR_DESCRIPTIVE_AUDIT_PASS'
    assert reported['new_training_updates'] == reported['new_encoder_or_matcher_forwards'] == 0
    assert binding(reported['program']['path']) == reported['program']
    roles = {r['query_id']: r for r in verified(reported['sources']['curator'])['records']}
    gallery = {r['physical_row']: r['identity'] for r in verified(reported['sources']['gallery'])['records']}
    old = {r['query_id']: r for r in verified(reported['sources']['old_result'])['rows']}
    expected = {r['query_id']: r for r in reported['records']}
    assert len(expected) == len(old) == 593
    cache = verified(reported['sources']['cache'])
    cached = {r['query_id']: r for r in cache['rows']}
    assert set(cached) == set(old)

    predictions = {}
    source_receipts = []
    for fold in range(5):
        receipt_path = ROOT / f'results/rc_six_cause_isolation_v1/loss_binding/fold{fold}/validation.json'
        receipt = read(receipt_path)
        assert receipt['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS'
        payload = verified(receipt['payload'])
        assert receipt['payload'] == reported['sources']['folds'][fold]
        for row in payload['predictions']:
            qid = row['query_id']
            assert qid not in predictions and qid not in payload['train_query_ids']
            predictions[qid] = (fold, row['models']['COST1'])
        source_receipts.append(binding(receipt_path))
    assert set(predictions) == set(expected)

    records = []
    checked_fields = 0
    max_feature_error = 0.0
    for index in range(593):
        receipt_path = ROOT / f'results/rc_h593_quality_operator_v1/query{index:03d}/validation.json'
        receipt = read(receipt_path)
        assert receipt['status'] == 'QUALITY_OPERATOR_QUERY_PASS' and receipt['native_exact'] is True
        payload = verified(receipt['payload'])
        assert payload['authority'] == receipt['authority']
        assert payload['execution_ordinal'] == index
        qid = payload['query_id']
        role = roles[qid]
        fold, pred = predictions[qid]
        axis = payload['candidate_physical_rows']
        raw_idx = payload['winner']
        challengers = payload['challenger_positions']
        masses = [float(x['visibility_mass']) for x in payload['modes']['NATIVE']['scores']]
        assert len(axis) == len(set(axis)) == len(masses) == 128
        assert set(challengers) == set(range(128)) - {raw_idx}
        assert all(math.isfinite(m) and m >= 0 for m in masses)
        assert cached[qid]['axis'] == axis
        assert cached[qid]['mass'] == masses
        assert cached[qid]['winner'] == raw_idx and cached[qid]['challengers'] == challengers
        for j, cidx in enumerate(challengers):
            dm = (masses[cidx] - masses[raw_idx]) / (masses[cidx] + masses[raw_idx] + TOL)
            delta = abs(dm - payload['modes']['NATIVE']['X'][j][2])
            assert delta < 2e-10
            max_feature_error = max(max_feature_error, delta)

        target_idxs = [i for i, physical in enumerate(axis) if gallery[physical] == role['identity']]
        assert len(target_idxs) <= 1
        target_idx = target_idxs[0] if target_idxs else None
        logits = [float.fromhex(v) for v in pred['logits_hex']]
        assert len(logits) == 127
        best = max(range(127), key=logits.__getitem__)
        top_idx = challengers[best]
        final_idx = top_idx if logits[best] > 0 else raw_idx
        assert axis[final_idx] == pred['selected']
        mpick = max(range(128), key=masses.__getitem__)
        row = {
            'query_id': qid, 'display_id': role['original_query_id'],
            'fold': fold, 'component': old[qid]['component'],
            'raw_correct': target_idx == raw_idx, 'cost1_correct': target_idx == final_idx,
            'target_in_C128': target_idx is not None,
            'raw_physical': axis[raw_idx], 'final_physical': axis[final_idx],
            'target_physical': axis[target_idx] if target_idx is not None else None,
            'max_M_physical': axis[mpick], 'max_M_correct': target_idx == mpick,
            'max_M_tie_count': sum(abs(m - masses[mpick]) <= TOL for m in masses),
            'action': 'HOLD' if raw_idx == final_idx else 'SWITCH',
            'best_challenger_correct': target_idx == top_idx,
            'max_logit': logits[best],
            'target_logit': 0.0 if target_idx == raw_idx else logits[challengers.index(target_idx)] if target_idx is not None else None,
        }
        assert row['raw_correct'] == old[qid]['correct']['RAW']
        assert row['cost1_correct'] == old[qid]['correct']['COST1']
        assert row['target_in_C128'] == old[qid]['target_in_C128']
        if target_idx is not None:
            target = masses[target_idx]
            wrong = [m for i, m in enumerate(masses) if i != target_idx]
            row.update({
                'target_M': target, 'raw_M': masses[raw_idx], 'final_M': masses[final_idx],
                'target_minus_final_M': target - masses[final_idx],
                'target_minus_strongest_wrong_M': target - max(wrong),
                'M_strict_rank': 1 + sum(m > target + TOL for m in wrong),
                'M_ties_with_wrong': sum(abs(m - target) <= TOL for m in wrong),
                'M_unique_first': target > max(wrong) + TOL,
                'M_above_final': target > masses[final_idx] + TOL,
                'head_challenger_rank': None if target_idx == raw_idx else 1 + sum(z > logits[challengers.index(target_idx)] + TOL for z in logits),
            })
        assert row == expected[qid], (qid, {k: (v, expected[qid].get(k)) for k, v in row.items() if v != expected[qid].get(k)})
        checked_fields += len(row)
        records.append(row)
        source_receipts.append(binding(receipt_path))

    failed = [r for r in records if not r['cost1_correct']]
    present = [r for r in failed if r['target_in_C128']]
    counts = {
        'population': len(records), 'raw_correct': sum(r['raw_correct'] for r in records),
        'cost1_correct': sum(r['cost1_correct'] for r in records),
        'cost1_failures': len(failed), 'failures_target_present': len(present),
        'failures_target_absent': len(failed) - len(present),
        'failures_M_above_final': sum(r['M_above_final'] for r in present),
        'failures_M_unique_first': sum(r['M_unique_first'] for r in present),
        'failures_M_rank_at_most_5': sum(r['M_strict_rank'] <= 5 for r in present),
        'failures_M_rank_at_most_10': sum(r['M_strict_rank'] <= 10 for r in present),
        'failures_M_tied_first': sum(r['M_strict_rank'] == 1 and not r['M_unique_first'] for r in present),
        'failures_target_top_challenger_but_held': sum(r['best_challenger_correct'] and r['action'] == 'HOLD' for r in failed),
        'max_M_fixed_rule_correct': sum(r['max_M_correct'] for r in records),
        'max_M_rescues_vs_cost1': sum(r['max_M_correct'] and not r['cost1_correct'] for r in records),
        'max_M_breaks_vs_cost1': sum(not r['max_M_correct'] and r['cost1_correct'] for r in records),
        'cost1_or_max_M_oracle_correct': sum(r['max_M_correct'] or r['cost1_correct'] for r in records),
    }
    assert counts == reported['counts'], (counts, reported['counts'])
    special = {r['display_id']: r for r in records if r['display_id'] in {'DIFFICULT-0011', 'OUTCOME-0477'}}
    assert special['DIFFICULT-0011']['M_strict_rank'] == 19
    assert special['OUTCOME-0477']['M_strict_rank'] == 1
    opportunities = [r for r in present if r['M_unique_first']]
    opportunity_groups = {
        'queries': len(opportunities),
        'components': len({r['component'] for r in opportunities}),
        'fold_counts': dict(sorted(Counter(r['fold'] for r in opportunities).items())),
        'original_actions': dict(Counter(r['action'] for r in opportunities)),
    }
    assert opportunity_groups == {
        'queries': 40, 'components': 20,
        'fold_counts': {0: 4, 1: 11, 2: 7, 3: 9, 4: 9},
        'original_actions': {'HOLD': 33, 'SWITCH': 7},
    }, opportunity_groups
    validation = {
        'status': 'H593_M_SCALAR_HEADROOM_INDEPENDENT_PASS', 'result': binding(result_path),
        'verifier': binding(__file__), 'independent_inputs': '593 original native quality-operator query payloads and original five COST1 fold payloads',
        'counts': counts, 'queries_verified': 593, 'native_masses_verified': 593 * 128,
        'per_query_fields_verified': checked_fields, 'native_M_feature_max_abs_error': max_feature_error,
        'source_validation_receipts': source_receipts,
        'cases': special, 'forty_M_first_failures': opportunity_groups,
        'new_training_updates': 0, 'new_encoder_or_matcher_forwards': 0,
        'scientific_limit': 'Descriptive opened-development evidence only; 521 is an answer-guided union, not learned accuracy or a general capacity bound.',
    }
    (OUT / 'independent_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': validation['status'], 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
