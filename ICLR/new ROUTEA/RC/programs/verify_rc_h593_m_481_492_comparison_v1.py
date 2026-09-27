#!/usr/bin/env python3
"""Independent verifier of the descriptive 481/492/M join; never trains."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_h593_m_481_492_comparison_20260927_v1'


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(binding):
    assert bound(binding['path']) == binding, binding['path']
    return read(binding['path'])


def keyed(rows):
    answer = {r['query_id']: r for r in rows}
    assert len(answer) == len(rows)
    return answer


def close(actual, expected, message, atol=2e-10):
    assert np.allclose(actual, expected, rtol=0, atol=atol), message


def scalar_cell(value):
    return '' if value is None else str(value)


def main():
    summary = read(OUT / 'summary.json')
    records = read(OUT / 'cases593.json')
    cases = keyed(records)
    candidates = keyed([json.loads(line) for line in (OUT / 'all_candidates.jsonl').read_text().splitlines()])
    terms = keyed(read(OUT / 'score_decomposition.json'))
    sources = summary['sources']
    audit = checked(sources['M_audit'])
    audit_records = keyed(audit['records'])
    cache = keyed(checked(sources['cache'])['rows'])
    gallery = {r['physical_row']: r['identity'] for r in checked(sources['gallery'])['records']}
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(curator_path)['records'])
    gap_result = checked(sources['GAP_result'])
    gap_result_rows = keyed(gap_result['rows'])
    original_result_path = RC / 'results/rc_six_cause_isolation_v1/loss_binding/result.json'
    assert checked(sources['M_validation'])['result'] == sources['M_audit']
    assert checked(sources['GAP_validation'])['result'] == sources['GAP_result']
    assert all(len(x) == 593 for x in [cases, candidates, cache, audit_records, curator, gap_result_rows])
    assert all(set(x) == set(cases) for x in [candidates, cache, audit_records, curator, gap_result_rows])
    assert len(terms) == 570
    native_predictions = {}
    gap_predictions = {}
    gate_parameters = {}
    native_parameters = {}
    native_bindings = []
    for fold in range(5):
        native_path = RC / f'results/rc_six_cause_isolation_v1/loss_binding/fold{fold}/payload.json'
        gap_path = RC / f'results/rc_h593_s_bias_competition_v1/fold{fold}/payload.json'
        native = read(native_path)
        gap = read(gap_path)
        assert bound(gap_path) in sources['GAP_folds']
        assert read(gap_path.with_name('validation.json'))['payload'] == bound(gap_path)
        native_bindings.append(bound(native_path))
        native_parameters[fold] = np.asarray([float.fromhex(x) for x in native['parameters']['COST1']])
        gate_parameters[fold] = gap['parameters']['GAP_BIAS2']
        for pool, payload in [(native_predictions, native), (gap_predictions, gap)]:
            train = set(payload['train_query_ids'])
            for p in payload['predictions']:
                q = p['query_id']
                assert q not in pool and q not in train
                assert curator[q]['outer_fold'] == fold
                pool[q] = p
    assert set(native_predictions) == set(gap_predictions) == set(cases)
    total_transitions = Counter()
    failures = {481: Counter(), 492: Counter()}
    tiers = ['M_FIRST', 'M_FIRST_TIED', 'M_RANK_2_5', 'M_RANK_6_10', 'M_RANK_11_128', 'TARGET_ABSENT']
    transitions = ['BOTH_CORRECT', 'RESCUED_BY_492', 'BROKEN_BY_492', 'BOTH_WRONG']
    matrix = {t: Counter() for t in tiers}
    first_errors = []
    replay_error = 0.0
    unchanged_switches = 0
    absent = 0
    for q, row in cases.items():
        source = cache[q]
        annotation = curator[q]
        pred = gap_predictions[q]
        axis = source['axis']
        assert len(axis) == len(set(axis)) == 128
        assert axis == pred['candidate_physical_rows'] == candidates[q]['physical_axis']
        winner = source['winner']
        challengers = source['challengers']
        assert challengers == pred['challenger_positions']
        assert set(challengers) == set(range(128)) - {winner}
        target_slots = [j for j, physical in enumerate(axis) if gallery[physical] == annotation['identity']]
        assert len(target_slots) <= 1
        target = target_slots[0] if target_slots else None
        assert row['target_identity'] == annotation['identity'], (q, 'correct identity must remain available outside C128')
        assert row['target_in_C128'] == (target is not None)
        assert row['target_physical'] == (axis[target] if target is not None else None)
        assert candidates[q]['target_physical'] == row['target_physical']
        assert row['display_id'] == annotation['original_query_id']
        assert row['component'] == annotation['component'] and row['fold'] == annotation['outer_fold']
        mass = np.asarray(source['mass'], dtype=np.float64)
        assert mass.shape == (128,) and np.isfinite(mass).all()
        assert np.array_equal(mass, candidates[q]['M0'])
        assert np.array_equal(source['free_content'], candidates[q]['L_free'])
        assert len(candidates[q]['L_free']) == 128
        p0 = native_predictions[q]['models']['COST1']
        sealed = np.asarray([float.fromhex(x) for x in p0['logits_hex']])
        zgap_sealed = np.asarray([float.fromhex(x) for x in pred['models']['GAP_BIAS2']['logits_hex']])
        assert p0['logits_hex'] == pred['models']['COST1_FULL']['logits_hex']
        theta = native_parameters[annotation['outer_fold']]
        X = np.asarray(source['native_X'])
        assert X.shape == (127, 6)
        recomputed = X @ theta[:6] + theta[6]
        replay_error = max(replay_error, float(np.max(np.abs(recomputed - sealed))))
        close(recomputed, sealed, (q, 'native score replay'))
        top_index = max(range(127), key=lambda j: sealed[j])
        runner_value = max(sealed[j] for j in range(127) if j != top_index)
        gate = gate_parameters[annotation['outer_fold']]
        assert gate['alpha'] == 0
        gate_score = float(sealed[top_index] + gate['beta'] * (sealed[top_index] - runner_value) + gate['bias'])
        gap_replay = sealed.copy()
        if sealed[top_index] <= 0 and gate_score > 0 and not gate['disabled']:
            gap_replay[top_index] = gate_score
        assert np.array_equal(gap_replay, zgap_sealed), (q, 'all127 exact GAP replay')
        original_top_slot = challengers[top_index]
        correctness = {}
        full_scores = {}
        for model, z, score_key in [(481, sealed, 'COST1_481_scores'), (492, zgap_sealed, 'GAP_BIAS2_492_scores')]:
            values = np.zeros(128)
            values[challengers] = z
            full_scores[model] = values
            assert np.array_equal(values, candidates[q][score_key])
            decision_index = max(range(127), key=lambda j: z[j])
            selected_slot = challengers[decision_index] if z[decision_index] > 0 else winner
            selected_physical = axis[selected_slot]
            frozen_key = 'COST1_FULL' if model == 481 else 'GAP_BIAS2'
            assert selected_physical == gap_result_rows[q]['selected'][frozen_key] == pred['models'][frozen_key]['selected']
            if model == 481:
                assert selected_physical == p0['selected']
            correct = gallery[selected_physical] == annotation['identity']
            correctness[model] = correct
            assert correct == gap_result_rows[q]['correct'][frozen_key] == row[f'model{model}_correct']
            assert row[f'model{model}_physical'] == selected_physical
            assert row[f'model{model}_identity'] == gallery[selected_physical]
            assert row[f'model{model}_action'] == ('HOLD' if selected_slot == winner else 'SWITCH')
            assert row[f'model{model}_selected_M'] == mass[selected_slot]
            assert row[f'model{model}_selected_M_rank'] == 1 + int(np.sum(mass > mass[selected_slot] + 1e-12))
            assert row[f'model{model}_selected_score'] == values[selected_slot]
            assert row[f'model{model}_target_score'] == (values[target] if target is not None else None)
            if correct:
                failure = 'CORRECT'
            elif target is None:
                failure = 'TARGET_ABSENT'
            elif target == winner:
                failure = 'RAW_CORRECT_BROKEN'
            elif target == original_top_slot:
                failure = 'TARGET_TOP_BUT_HOLD'
            else:
                failure = 'CHALLENGER_RANKING_BLOCKED'
            assert row[f'model{model}_failure'] == failure
            if not correct:
                failures[model][failure] += 1
        assert row['RAW_physical'] == candidates[q]['HOLD_physical'] == axis[winner]
        assert row['RAW_identity'] == gallery[axis[winner]] and row['RAW_correct'] == (target == winner)
        assert row['head_top_challenger_physical'] == axis[original_top_slot]
        assert row['head_top_challenger_correct'] == (target == original_top_slot)
        assert row['gate_evaluated_for_HOLD'] == (sealed[top_index] <= 0)
        assert row['model492_gate_if_original_HOLD'] == gate_score
        assert row['challenger_gap'] == sealed[top_index] - runner_value
        if sealed[top_index] > 0:
            assert np.array_equal(sealed, zgap_sealed)
            unchanged_switches += 1
        mtop = int(np.argmax(mass))
        assert row['max_M_physical'] == axis[mtop] and row['max_M_identity'] == gallery[axis[mtop]]
        assert row['max_M'] == mass[mtop]
        if target is None:
            absent += 1
            assert row['target_M'] is None and row['target_M_rank'] is None
            assert not row['target_M_unique_first']
            tier = 'TARGET_ABSENT'
            assert q not in terms
        else:
            wrong_mass = np.delete(mass, target)
            rank = 1 + int(np.sum(wrong_mass > mass[target] + 1e-12))
            ties = int(np.sum(np.abs(wrong_mass - mass[target]) <= 1e-12))
            unique = rank == 1 and ties == 0
            assert row['target_M'] == mass[target] and row['target_M_rank'] == rank
            assert row['M_ties_with_wrong'] == ties and row['target_M_unique_first'] == unique
            assert row['target_minus_strongest_wrong_M'] == mass[target] - max(wrong_mass)
            tier = 'M_FIRST' if unique else 'M_FIRST_TIED' if rank == 1 else 'M_RANK_2_5' if rank <= 5 else 'M_RANK_6_10' if rank <= 10 else 'M_RANK_11_128'
            score = full_scores[481]
            rival = max([j for j in range(128) if j != target], key=lambda j: score[j])
            expanded_terms = np.zeros((128, 7))
            expanded_terms[challengers, :6] = X * theta[:6]
            expanded_terms[challengers, 6] = theta[6]
            difference = expanded_terms[target] - expanded_terms[rival]
            term_record = terms[q]
            assert term_record['baseline481_rival'] == row['baseline481_strongest_wrong_physical'] == axis[rival]
            assert term_record['rival_is_HOLD'] == (rival == winner)
            close(term_record['target_minus_rival_score'], score[target] - score[rival], (q, 'score margin'))
            close(math.fsum(difference.tolist()), score[target] - score[rival], (q, 'seven component sum'))
            for name, value in zip(['RAW', 'S', 'M', 'L', 'Q', 'R', 'bias'], difference):
                close(term_record['signed_terms'][name], value, (q, name))
                close(row[f'target_minus_rival_term_{name}'], value, (q, name, 'table'))
            if unique and not correctness[492]:
                first_errors.append(row)
        assert row['M_tier'] == tier
        transition = {(True, True): 'BOTH_CORRECT', (False, True): 'RESCUED_BY_492', (True, False): 'BROKEN_BY_492', (False, False): 'BOTH_WRONG'}[correctness[481], correctness[492]]
        assert row['transition'] == transition
        total_transitions[transition] += 1
        matrix[tier][transition] += 1
    expected_first_types = {'TARGET_TOP_BUT_HOLD': 18, 'CHALLENGER_RANKING_BLOCKED': 10, 'RAW_CORRECT_BROKEN': 1}
    assert len(first_errors) == 29 and dict(Counter(r['model492_failure'] for r in first_errors)) == expected_first_types
    assert absent == 23 and unchanged_switches == 78
    assert sum(r['model481_correct'] for r in records) == 481
    assert sum(r['model492_correct'] for r in records) == 492
    assert total_transitions == {'BOTH_CORRECT': 476, 'RESCUED_BY_492': 16, 'BROKEN_BY_492': 5, 'BOTH_WRONG': 96}
    assert summary['transitions'] == total_transitions
    for model in [481, 492]:
        assert summary[f'failure{model}'] == failures[model]
    assert summary['M_rank_matrix'] == {tier: {transition: matrix[tier][transition] for transition in transitions} for tier in tiers}
    assert summary['M_first_492_error_types'] == expected_first_types
    assert summary['M_first_new_492_breaks'] == ['OUTCOME-0721']
    assert set(summary['M_first_492_error_ids']) == {r['display_id'] for r in first_errors}
    csv_inputs = {
        'cases593.csv': records,
        'M_first_but_492_wrong_29.csv': first_errors,
        'M_lower_and_492_wrong.csv': [r for r in records if not r['model492_correct'] and r['target_in_C128'] and not r['target_M_unique_first']],
        '492_rescues_and_breaks_21.csv': [r for r in records if r['transition'] in ['RESCUED_BY_492', 'BROKEN_BY_492']],
    }
    csv_cells = 0
    for filename, expected in csv_inputs.items():
        with (OUT / filename).open(newline='') as f:
            table = list(csv.DictReader(f))
        assert len(table) == len(expected), filename
        lookup = keyed(expected)
        assert {r['query_id'] for r in table} == set(lookup)
        for row in table:
            for key, value in row.items():
                assert value == scalar_cell(lookup[row['query_id']].get(key)), (filename, row['query_id'], key)
                csv_cells += 1
    page = (OUT / 'index.html').read_text()
    html_rows = json.loads(page.split('const DATA=', 1)[1].split(';const esc=', 1)[0])
    assert html_rows == records
    assert summary['new_training_updates'] == summary['new_encoder_or_matcher_forwards'] == 0
    artifacts = ['summary.json', 'cases593.json', 'all_candidates.jsonl', 'score_decomposition.json', 'index.html', *csv_inputs]
    output = {
        'status': 'H593_M_481_492_INDEPENDENT_VALIDATION_PASS',
        'queries_verified': 593,
        'candidate_rows_verified': 593 * 128,
        'challenger_scores_verified_per_model': 593 * 127,
        'HOLD_zero_scores_verified_per_model': 593,
        'target_absent_queries_with_identity_retained': absent,
        'original_SWITCHEs_unchanged': unchanged_switches,
        'correct': {'COST1_481': 481, 'GAP_BIAS2_492': 492},
        'transitions': dict(total_transitions),
        'M_first_492_error_types': expected_first_types,
        'csv_cells_verified': csv_cells,
        'csv_row_counts': {k: len(v) for k, v in csv_inputs.items()},
        'score_decompositions_verified': len(terms),
        'maximum_native_logit_replay_error': replay_error,
        'GAP_full127_score_replay': 'BIT_EXACT',
        'artifacts': {name: bound(OUT / name) for name in artifacts},
        'independent_original_inputs': {'curator': bound(curator_path), 'native_COST1_result': bound(original_result_path), 'native_COST1_folds': native_bindings, 'GAP_folds': sources['GAP_folds'], 'GAP_result': sources['GAP_result'], 'cache': sources['cache'], 'gallery': sources['gallery']},
        'verifier': bound(__file__),
        'scientific_scope': 'Read-only reconstruction of previously opened H593 results; no new model, fitting, or external confirmation.',
        'issues': [],
    }
    (OUT / 'validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['status', 'queries_verified', 'candidate_rows_verified', 'transitions', 'M_first_492_error_types', 'csv_row_counts', 'maximum_native_logit_replay_error']}, indent=2))


if __name__ == '__main__':
    main()
