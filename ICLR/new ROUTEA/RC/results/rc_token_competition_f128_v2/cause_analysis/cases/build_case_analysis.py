#!/usr/bin/env python3
"""Read-only primary fixed-zero case audit; write only this reporting directory."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10), (a, b)


def run(root):
    out = root / 'cause_analysis/cases'
    out.mkdir(parents=True, exist_ok=True)
    files = ['protocol.json', 'validation.json', 'joined_predictions.json',
             'candidate_predictions.json', 'analysis/analysis_summary.json',
             'analysis/per_query_diagnostics.csv']
    before = {name: sha(root / name) for name in files}
    docs = {name: json.loads((root / name).read_text()) for name in files if name.endswith('.json')}
    validation = docs['validation.json']
    analysis = docs['analysis/analysis_summary.json']
    assert validation['status'] == 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS'
    assert analysis['status'] == 'TOKEN_COMPETITION_ANALYSIS_PASS'
    for name in ['joined_predictions.json', 'candidate_predictions.json']:
        assert before[name] == validation['artifacts'][name]['sha256']
    assert before['protocol.json'] == validation['protocol']['sha256']
    assert before['validation.json'] == analysis['validation']['sha256']
    assert before['analysis/per_query_diagnostics.csv'] == analysis['outputs']['per_query_diagnostics.csv']['sha256']
    source = [r for r in csv.DictReader((root / 'analysis/per_query_diagnostics.csv').open())
              if r['operating_point'] == 'zero']
    assert len(source) == 384
    joined = {(r['model'], r['query_id']): r for r in docs['joined_predictions.json']['records']
              if r['operating_point'] == 'zero'}
    candidates = defaultdict(list)
    for r in docs['candidate_predictions.json']['records']:
        candidates[(r['model'], r['query_id'])].append(r)
    assert len(candidates) == 384
    rows, candidate_support = [], []
    for d in source:
        arm, query = d['model'], d['query_id']
        cs = candidates[(arm, query)]
        assert len(cs) == len({c['physical_row'] for c in cs}) == 128
        by_row = {c['physical_row']: c for c in cs}
        assert sorted(c['position'] for c in cs) == list(range(128))
        for c in cs:
            close(c['logit'], c['zero_base'] + c['residual'])
        raw = next(c for c in cs if c['raw_anchor'])
        close(raw['logit'], 0.0)
        close(raw['residual'], 0.0)
        raw_row = raw['physical_row']
        # Frozen decision: highest challenger above zero, else RAW; tie by row.
        def choose(field):
            challenge = sorted((c for c in cs if not c['raw_anchor']),
                               key=lambda c: (-c[field], c['physical_row']))[0]
            return challenge if challenge[field] > 0 else raw
        base_selected, selected = choose('zero_base'), choose('logit')
        bp, pred = joined[('B_CAL', query)], joined[(arm, query)]
        assert base_selected['physical_row'] == bp['selected'] == int(d['B_CAL_selected'])
        assert selected['physical_row'] == pred['selected'] == int(d['selected'])
        target_row = int(d['target_physical_row']) if d['target_in_c128'] == 'True' else None
        target = by_row[target_row] if target_row is not None else None
        correct = bool(target is not None and selected['physical_row'] == target_row)
        base_correct = bool(target is not None and base_selected['physical_row'] == target_row)
        assert correct == pred['correct'] == (d['correct'] == 'True')
        assert base_correct == bp['correct'] == (d['B_CAL_correct'] == 'True')
        wrong = [c for c in cs if c['physical_row'] != target_row]
        base_wrong = min(wrong, key=lambda c: (-c['zero_base'], c['physical_row']))
        final_wrong = min(wrong, key=lambda c: (-c['logit'], c['physical_row']))
        action = 'HOLD' if selected['physical_row'] == raw_row else 'SWITCH'
        assert action == d['action']
        if correct:
            failure = 'correct'
        elif target is None:
            failure = 'target_absent_from_C128'
        elif action == 'HOLD':
            failure = 'target_present_but_HOLD_wrong_RAW'
        elif target_row == raw_row:
            failure = 'SWITCH_away_from_correct_RAW'
        elif target['logit'] > 0:
            failure = 'target_beats_RAW_but_wrong_challenger_wins'
        else:
            failure = 'target_not_above_RAW_and_wrong_challenger_wins'
        row = {
            'query_id': query, 'identity': d['identity'], 'fold': int(d['fold']),
            'seed': 0, 'model': arm, 'operating_point': 'fixed0',
            'target_in_C128': target is not None, 'target_physical_row': target_row,
            'target_B_CAL_rank': int(d['target_B_CAL_rank']) if target else None,
            'target_final_rank': sorted(cs, key=lambda c: (-c['logit'], c['physical_row'])).index(target) + 1 if target else None,
            'raw_physical_row': raw_row, 'B_CAL_selected': bp['selected'],
            'B_CAL_correct': base_correct, 'selected': pred['selected'], 'correct': correct,
            'action': action, 'changed_vs_B_CAL': pred['selected'] != bp['selected'],
            'transition_vs_B_CAL': ('rescue' if correct and not base_correct else
                                    'break' if base_correct and not correct else
                                    'wrong_to_different_wrong' if pred['selected'] != bp['selected'] and not correct else
                                    'correct_unchanged' if correct else 'wrong_unchanged'),
            'failure_class': failure,
            'target_base_logit': target['zero_base'] if target else None,
            'target_residual': target['residual'] if target else None,
            'target_final_logit': target['logit'] if target else None,
            'base_strongest_wrong_row': base_wrong['physical_row'],
            'base_strongest_wrong_logit': base_wrong['zero_base'],
            'final_strongest_wrong_row': final_wrong['physical_row'],
            'final_strongest_wrong_base_logit': final_wrong['zero_base'],
            'final_strongest_wrong_residual': final_wrong['residual'],
            'final_strongest_wrong_logit': final_wrong['logit'],
            'strongest_wrong_changed': base_wrong['physical_row'] != final_wrong['physical_row'],
            'target_vs_base_strongest_wrong_margin': target['zero_base'] - base_wrong['zero_base'] if target else None,
            'target_vs_final_strongest_wrong_margin': target['logit'] - final_wrong['logit'] if target else None,
            'base_margin_against_final_wrong': target['zero_base'] - final_wrong['zero_base'] if target else None,
            'residual_margin_against_final_wrong': target['residual'] - final_wrong['residual'] if target else None,
            'main_epoch': int(d['main_epoch']), 'evidence': 'observed_candidate_score_arithmetic',
        }
        row['margin_change_vs_B_CAL'] = (row['target_vs_final_strongest_wrong_margin'] -
                                       row['target_vs_base_strongest_wrong_margin']) if target else None
        if target:
            close(row['base_margin_against_final_wrong'] + row['residual_margin_against_final_wrong'],
                  row['target_vs_final_strongest_wrong_margin'])
        rows.append(row)
        support_ids = {raw_row, bp['selected'], pred['selected'],
                       base_wrong['physical_row'], final_wrong['physical_row']}
        if target:
            support_ids.add(target_row)
        candidate_support.append({'query_id': query, 'model': arm, 'target_physical_row': target_row,
                                  'records': [by_row[k] for k in sorted(support_ids)]})
    keyed = {(r['model'], r['query_id']): r for r in rows}
    summary = {
        'status': 'PRIMARY_FIXED0_CASE_ARITHMETIC_PASS',
        'panel': 'F128 opened development, original grouped five folds, seed0',
        'candidate_source': 'unchanged frozen natural ColNomic C128',
        'baseline': 'SAME F128 seed0 phase-correct frozen 18-dimensional B_CAL',
        'scoring': 'B_CAL zero_base plus native-token D_g minus D_RAW residual; RAW exactly0',
        'operating_point': 'fixed0 primary only',
        'evidence_level': 'post-join descriptive score decomposition; no visual causal attribution or fixed-weight rival ablation',
        'source_bindings': {n: {'path': str(root / n), 'sha256': h} for n, h in before.items()},
        'analysis_code': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
        'arms': {},
    }
    for arm in ARMS:
        ar = [r for r in rows if r['model'] == arm]
        summary['arms'][arm] = {
            'n': len(ar), 'correct': sum(r['correct'] for r in ar),
            'changed_decisions': sum(r['changed_vs_B_CAL'] for r in ar),
            'transition_counts': dict(Counter(r['transition_vs_B_CAL'] for r in ar)),
            'failure_counts': dict(Counter(r['failure_class'] for r in ar if not r['correct'])),
            'absent_failure_actions': dict(Counter(r['action'] for r in ar if not r['target_in_C128'])),
            'rescues': [r['query_id'] for r in ar if r['transition_vs_B_CAL'] == 'rescue'],
            'breaks': [r['query_id'] for r in ar if r['transition_vs_B_CAL'] == 'break'],
        }
    differences, pair_summary, loss_details = [], [], []
    queries = sorted({r['query_id'] for r in rows})
    for old, new in [(ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]), (ARMS[0], ARMS[2])]:
        pair = []
        for q in queries:
            o, n = keyed[old, q], keyed[new, q]
            if o['selected'] == n['selected']:
                continue
            d = {'query_id': q, 'identity': n['identity'], 'fold': n['fold'], 'old_model': old,
                 'new_model': new, 'target_physical_row': n['target_physical_row'],
                 'target_B_CAL_rank': n['target_B_CAL_rank'],
                 'old_selected': o['selected'], 'new_selected': n['selected'],
                 'old_correct': o['correct'], 'new_correct': n['correct'],
                 'old_action': o['action'], 'new_action': n['action'],
                 'old_target_final_logit': o['target_final_logit'],
                 'new_target_final_logit': n['target_final_logit'],
                 'old_target_vs_strongest_wrong_margin': o['target_vs_final_strongest_wrong_margin'],
                 'new_target_vs_strongest_wrong_margin': n['target_vs_final_strongest_wrong_margin'],
                 'transition': 'gain' if n['correct'] and not o['correct'] else
                               'loss' if o['correct'] and not n['correct'] else 'wrong_to_different_wrong'}
            pair.append(d)
            if old == ARMS[1] and new == ARMS[2] and d['transition'] == 'loss':
                loss_details.append({'comparison': d, 'all_arm_rows': [keyed[a, q] for a in ARMS],
                                     'candidate_support': [r for r in candidate_support if r['query_id'] == q]})
        differences.extend(pair)
        pair_summary.append({'old_model': old, 'new_model': new, 'changed_decisions': len(pair),
                             'transitions': dict(Counter(d['transition'] for d in pair))})
    summary['between_arms'] = pair_summary
    rank_rows = []
    for arm in ARMS:
        for label, lo, hi in [('rank1', 1, 1), ('rank2', 2, 2), ('rank3_5', 3, 5),
                              ('rank6_128', 6, 128), ('absent', None, None)]:
            rr = [r for r in rows if r['model'] == arm and
                  (r['target_B_CAL_rank'] is None if lo is None else
                   r['target_B_CAL_rank'] is not None and lo <= r['target_B_CAL_rank'] <= hi)]
            rank_rows.append({'model': arm, 'target_B_CAL_rank_stratum': label, 'queries': len(rr),
                              'B_CAL_correct': sum(r['B_CAL_correct'] for r in rr),
                              'correct': sum(r['correct'] for r in rr),
                              'rescues': sum(r['transition_vs_B_CAL'] == 'rescue' for r in rr),
                              'breaks': sum(r['transition_vs_B_CAL'] == 'break' for r in rr)})
    summary['target_rank_depth'] = rank_rows
    assert len(loss_details) == 2
    assert summary['arms'][ARMS[0]]['rescues'] == summary['arms'][ARMS[1]]['rescues']
    assert summary['arms'][ARMS[1]]['breaks'] == summary['arms'][ARMS[2]]['breaks']
    assert before == {name: sha(root / name) for name in files}
    summary['source_hashes_unchanged_after_analysis'] = True
    csvout(out / 'primary_fixed0_cases.csv', rows)
    csvout(out / 'rescues_and_breaks.csv', [r for r in rows if r['transition_vs_B_CAL'] in ('rescue', 'break')])
    csvout(out / 'errors.csv', [r for r in rows if not r['correct']])
    csvout(out / 'between_arm_decision_changes.csv', differences)
    csvout(out / 'target_rank_depth.csv', rank_rows)
    dump(out / 'multi_specific_losses.json', loss_details)
    dump(out / 'candidate_score_support.json', candidate_support)
    output_names = ['primary_fixed0_cases.csv', 'rescues_and_breaks.csv', 'errors.csv',
                    'between_arm_decision_changes.csv', 'target_rank_depth.csv',
                    'multi_specific_losses.json', 'candidate_score_support.json']
    summary['outputs'] = {n: {'path': str(out / n), 'sha256': sha(out / n)} for n in output_names}
    dump(out / 'case_summary.json', summary)
    print(json.dumps({'status': summary['status'], 'rows': len(rows), 'multi_losses': len(loss_details),
                      'out': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root.resolve())
