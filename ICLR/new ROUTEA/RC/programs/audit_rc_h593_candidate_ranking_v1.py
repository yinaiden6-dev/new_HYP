#!/usr/bin/env python3
"""Read sealed scores; separate candidate ordering from HOLD calibration. No fits."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

from audit_rc_h593_population_opportunities_v1 import bind, checked, read

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_h593_gap_curve_v1'
POPULATION = ROOT / 'results/rc_h593_population_opportunities_v1'
OUT = ROOT / 'results/rc_h593_candidate_ranking_v1'
MODELS = ('COST1_FULL', 'CE_FULL', 'ZERO_S')


def shift(rows, key, baseline):
    return dict(improved=sum(r[key] < r[baseline] for r in rows),
                same=sum(r[key] == r[baseline] for r in rows),
                worse=sum(r[key] > r[baseline] for r in rows))


def main():
    validation = read(SOURCE / 'validation.json')
    assert validation['status'] == 'GAP_CURVE_ALL_COUNTS_PASS'
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    gallery = read(checked(authority['public_sources']['gallery']))
    identities = {r['physical_row']: r['identity'] for r in gallery['records']}
    population = read(POPULATION / 'result.json')
    assert population['source'] == validation['result']
    with (POPULATION / 'all593_categories.csv').open() as file:
        categories = {r['query_id']: r['category'] for r in csv.DictReader(file)}
    predictions, seals = {}, []
    for fold in range(5):
        v = read(SOURCE / f'fold{fold}/validation.json')
        iv = read(SOURCE / f'fold{fold}/independent_validation.json')
        assert iv['passed'] and v['payload'] == iv['payload']
        payload = read(checked(v['payload']))
        for p in payload['predictions']:
            assert p['query_id'] not in predictions
            predictions[p['query_id']] = p
        seals.append(v['payload'])
    assert len(predictions) == len(categories) == len(result['rows']) == 593
    rows = []
    for r in result['rows']:
        p = predictions[r['query_id']]
        axis = p['candidate_physical_rows']
        challengers = p['challenger_positions']
        assert len(axis) == 128 and len(challengers) == 127
        assert sorted(challengers + [p['winner']]) == list(range(128))
        target_positions = [j for j, physical in enumerate(axis) if identities[physical] == r['identity']]
        assert bool(target_positions) == r['target_in_C128']
        raw_correct = identities[axis[p['winner']]] == r['identity']
        assert raw_correct == r['correct']['RAW']
        row = dict(query_id=r['query_id'], original_query_id=r['original_query_id'],
                   identity=r['identity'], component=r['component'], fold=r['fold'],
                   category=categories[r['query_id']], target_in_C128=bool(target_positions),
                   target_physical_candidate_count=len(target_positions), RAW_correct=raw_correct,
                   RAW_target_rank=r['ranks']['RAW'], GAP_BIAS2_correct=r['correct']['GAP_BIAS2'])
        for model in MODELS:
            z = [float.fromhex(x) for x in p['models'][model]['logits_hex']]
            assert len(z) == 127
            order = sorted(range(127), key=lambda j: (-z[j], j))
            top = order[0]
            selected = axis[challengers[top] if z[top] > 0.0 else p['winner']]
            assert selected == p['models'][model]['selected'] == r['selected'][model]
            correct = identities[selected] == r['identity']
            assert correct == r['correct'][model]
            target_indices = [j for j, pos in enumerate(challengers) if pos in target_positions]
            target_rank = next((i + 1 for i, j in enumerate(order) if j in target_indices), None)
            scores = [0.0] * 128
            for j, pos in enumerate(challengers):
                scores[pos] = z[j]
            combined = sorted(range(128), key=lambda j: (-scores[j], j))
            unified_rank = next((i + 1 for i, pos in enumerate(combined) if pos in target_positions), None)
            target_score = max((z[j] for j in target_indices), default=None)
            fields = dict(correct=correct, top_identity=identities[axis[challengers[top]]],
                          top_is_target=identities[axis[challengers[top]]] == r['identity'],
                          top_score=z[top], target_challenger_rank=target_rank,
                          target_unified128_diagnostic_rank=unified_rank,
                          target_score=target_score,
                          target_ties=sum(scores[j] == target_score for j in range(128)
                                          if j not in target_positions) if target_score is not None else None)
            row.update({f'{model}_{k}': v for k, v in fields.items()})
        rows.append(row)
    subset = [r for r in rows if r['category'] == 'wrong_RAW_and_challenger']
    assert len(subset) == 28
    assert all(r['target_physical_candidate_count'] == 1 for r in subset)
    assert all(r[f'{m}_target_ties'] == 0 for r in subset for m in MODELS)
    eligible = [r for r in rows if r['target_in_C128'] and not r['RAW_correct']]
    ranking_blocked = [r for r in eligible if not r['COST1_FULL_top_is_target']]
    assert len(eligible) == 144 and len(ranking_blocked) == 40
    assert Counter(r['category'] for r in ranking_blocked) == dict(wrong_RAW_and_challenger=28, locked_wrong_SWITCH=12)
    remaining_present = [r for r in rows if r['target_in_C128'] and not r['GAP_BIAS2_correct']]
    mechanism_counts = dict(wrong_challenger=sum(not r['RAW_correct'] and not r['COST1_FULL_top_is_target'] for r in remaining_present),
                            correct_challenger_not_selected=sum(not r['RAW_correct'] and r['COST1_FULL_top_is_target'] for r in remaining_present),
                            RAW_correct_wrong_SWITCH=sum(r['RAW_correct'] for r in remaining_present))
    assert mechanism_counts == dict(wrong_challenger=40, correct_challenger_not_selected=30, RAW_correct_wrong_SWITCH=8)
    summary = {}
    for model in MODELS:
        ranks = [r[f'{model}_target_challenger_rank'] for r in subset]
        summary[model] = dict(
            all593_correct=sum(r[f'{model}_correct'] for r in rows),
            all593_vs492=dict(rescue=sum(r[f'{model}_correct'] and not r['GAP_BIAS2_correct'] for r in rows),
                              loss=sum(not r[f'{model}_correct'] and r['GAP_BIAS2_correct'] for r in rows)),
            target_top_challenger_among_RAW_wrong_present=sum(r[f'{model}_top_is_target'] for r in eligible),
            top_challenger_correctness_vs_COST1=dict(gained=sum(r[f'{model}_top_is_target'] and not r['COST1_FULL_top_is_target'] for r in eligible),
                                                    lost=sum(not r[f'{model}_top_is_target'] and r['COST1_FULL_top_is_target'] for r in eligible)),
            any_RAW_or_top_challenger_correct_oracle=sum(r['RAW_correct'] or r[f'{model}_top_is_target'] for r in rows),
            subset28_correct=sum(r[f'{model}_correct'] for r in subset),
            subset28_target_top_challenger=sum(r[f'{model}_top_is_target'] for r in subset),
            subset28_challenger_rank_shift=shift(subset, f'{model}_target_challenger_rank', 'COST1_FULL_target_challenger_rank'),
            subset28_unified_diagnostic_rank_shift=shift(subset, f'{model}_target_unified128_diagnostic_rank', 'COST1_FULL_target_unified128_diagnostic_rank'),
            subset28_top_identity_unchanged=sum(r[f'{model}_top_identity'] == r['COST1_FULL_top_identity'] for r in subset),
            subset28_challenger_rank=dict(min=min(ranks), median=median(ranks), max=max(ranks),
                                         second=sum(rank == 2 for rank in ranks), top4=sum(rank <= 4 for rank in ranks)),
            subset28_top_target_but_HOLD=[dict(query_id=r['query_id'], original_query_id=r['original_query_id'], score=r[f'{model}_top_score'])
                                        for r in subset if r[f'{model}_top_is_target'] and not r[f'{model}_correct']])
    report = dict(status='SEALED_H593_CANDIDATE_RANKING_RECOUNT_PASS',
                  source=validation['result'], authority=validation['authority'], payloads=seals,
                  population_categories=bind(POPULATION / 'all593_categories.csv'), program=bind(__file__),
                  evidence_level='Post-hoc diagnostic on opened H593 grouped OOF development predictions',
                  all593_queries=len(rows), RAW_wrong_target_present_queries=len(eligible),
                  GAP_BIAS2_remaining78_mechanism_counts=mechanism_counts,
                  all40_ranking_blocked_original_categories=dict(Counter(r['category'] for r in ranking_blocked)),
                  subset28_identities=len({r['identity'] for r in subset}),
                  subset28_components=len({r['component'] for r in subset}),
                  subset28_by_fold=dict(Counter(r['fold'] for r in subset)), models=summary,
                  COST1_subset28_unified_rank_vs_RAW=shift(subset, 'COST1_FULL_target_unified128_diagnostic_rank', 'RAW_target_rank'),
                  model_fits=0, external_GO=False,
                  limits=['Unified128 rank is a score diagnostic, not deployed pipeline MRR.',
                          'Subset28 was selected from GAP_BIAS2 errors; its metrics are not an unbiased model comparison.',
                          'Oracle counts are action-space capacities, not achievable predictions.',
                          'Ranking results do not isolate frozen-token information, summary-feature compression, or learning failure.'])
    OUT.mkdir(exist_ok=True)
    for filename, data in (('all593_candidate_diagnostics.csv', rows), ('subset28_candidate_diagnostics.csv', subset), ('all40_ranking_blocked.csv', ranking_blocked)):
        with (OUT / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(data)
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
