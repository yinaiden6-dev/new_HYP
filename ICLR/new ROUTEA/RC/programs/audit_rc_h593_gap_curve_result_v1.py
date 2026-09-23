#!/usr/bin/env python3
"""Read completed CURVE3 results; decompose decisions without any fitting."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_gap_curve_v1'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def score(m, raw, gap, head, drop_extra_raw=False):
    raw = raw if head['mode'] == 'GAP_CURVE4' and not drop_extra_raw else 0.0
    value = (m + head['gamma'] * raw) + head['beta'] * gap
    if head.get('eta', 0.0) != 0.0:
        value += head['eta'] * ((gap * gap) / head['scale'])
    return value + head['bias']


def main():
    validation = read(OUT / 'validation.json')
    assert validation['status'] == 'GAP_CURVE_ALL_COUNTS_PASS'
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    for b in authority['code_sources'].values():
        checked(b)
    rows = result['rows']
    by_query = {r['query_id']: r for r in rows}
    assert len(rows) == len(by_query) == 593
    assert len({r['component'] for r in rows}) == 64
    assert sum(not r['target_in_C128'] for r in rows) == 23
    gallery = read(checked(authority['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    comparisons = {}
    for base in ('GAP_BIAS2', 'GAP_CURVE4'):
        counts = Counter()
        rescues, losses = [], []
        for r in rows:
            new, old, raw = (r['correct'][name] for name in ('GAP_CURVE3', base, 'RAW'))
            if new and not old:
                counts['repaired_RAW_break' if raw else 'new_RAW_rescue'] += 1
                rescues.append(r['original_query_id'])
            if old and not new:
                counts['new_RAW_break' if raw else 'lost_RAW_rescue'] += 1
                losses.append(r['original_query_id'])
        expected = result['comparisons'][base + '__to__GAP_CURVE3']
        assert expected['rescue'] == len(rescues) and expected['loss'] == len(losses)
        comparisons[base] = dict(counts=counts, rescues=rescues, losses=losses)
    folds, cases, diagnostic = [], [], []
    for fold in range(5):
        v = read(OUT / f'fold{fold}/validation.json')
        iv = read(OUT / f'fold{fold}/independent_validation.json')
        assert v['payload'] == iv['payload'] and iv['passed']
        p = read(checked(v['payload']))
        new, linear, curve = p['parameters']['GAP_CURVE3'], p['matched_control'], p['structural_control']
        roles = {r['query_id']: r for r in read(checked(authority['fold_sources'][str(fold)]['train_roles']))['records']}
        opportunities = [r for r in p['calibration'] if r['m'] <= 0 and r['delta'] == 1]
        risks = [r for r in p['calibration'] if r['m'] <= 0 and r['delta'] == -1]
        heads = dict(GAP_BIAS2=linear, GAP_CURVE4=curve, GAP_CURVE3=new)
        folds.append(dict(fold=fold, positive_HOLD_count=len(opportunities), negative_HOLD_count=len(risks),
                          positive_HOLD_components=len({roles[r['query_id']]['component'] for r in opportunities}),
                          inner_net={k:h['training_net_gain'] for k,h in heads.items()},
                          outer_correct={k:result['summary'][k]['fold_correct'][str(fold)] for k in heads},
                          fitted_loss={k:h['optimization']['fitted_surrogate_loss'] for k,h in heads.items()},
                          parameters={k:{f:h[f] for f in ('gamma','beta','eta','bias','scale') if f in h} for k,h in heads.items()}))
        for pred in p['predictions']:
            row = by_query[pred['query_id']]
            z = [float.fromhex(x) for x in pred['models']['COST1_FULL']['logits_hex']]
            top = max(range(127), key=z.__getitem__)
            m, raw, d = z[top], pred['r_near'], pred['d']
            gates = {k:score(m, raw, d, h) for k,h in heads.items()}
            expected = list(z)
            if m <= 0 and not new['disabled'] and gates['GAP_CURVE3'] > 0:
                expected[top] = gates['GAP_CURVE3']
            assert [x.hex() for x in expected] == pred['models']['GAP_CURVE3']['logits_hex']
            if any(row['selected']['GAP_CURVE3'] != row['selected'][base] for base in ('GAP_BIAS2','GAP_CURVE4')):
                cases.append(dict(query_id=row['query_id'], original_query_id=row['original_query_id'], fold=fold,
                                  RAW_correct=row['correct']['RAW'], linear492_correct=row['correct']['GAP_BIAS2'],
                                  curve489_correct=row['correct']['GAP_CURVE4'], new490_correct=row['correct']['GAP_CURVE3'],
                                  m=m, raw_margin=raw, gap=d, linear_gate=gates['GAP_BIAS2'],
                                  curve4_gate=gates['GAP_CURVE4'], curve3_gate=gates['GAP_CURVE3'],
                                  removal_term=-curve['gamma']*raw,
                                  beta_refit_term=(new['beta']-curve['beta'])*d,
                                  eta_refit_term=(new['eta']-curve['eta'])*((d*d)/new['scale']),
                                  bias_refit_term=new['bias']-curve['bias']))
            direct = score(m, raw, d, curve, drop_extra_raw=True)
            position = pred['challenger_positions'][top] if m > 0 or (not curve['disabled'] and direct > 0) else pred['winner']
            diagnostic.append(dict(correct=labels[pred['candidate_physical_rows'][position]] == row['identity'],
                                   original=row['correct']['GAP_CURVE4']))
    counts = dict(correct=sum(r['correct'] for r in diagnostic),
                  rescues=sum(r['correct'] and not r['original'] for r in diagnostic),
                  losses=sum(r['original'] and not r['correct'] for r in diagnostic))
    audit = dict(status='GAP_CURVE_POST_JOIN_ANALYSIS_PASS', result=validation['result'],
                 authority=validation['authority'], program=bind(__file__), comparisons=comparisons, folds=folds,
                 frozen_CURVE4_drop_raw_diagnostic=counts,
                 diagnostic_scope='Post-hoc gamma=0 intervention with all other CURVE4 coefficients fixed; no refitting, not a validated selected alternative.',
                 natural_fits=0, automatic_model_replacement=False)
    (OUT/'post_join_analysis.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'changed_cases.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    print(json.dumps(dict(status=audit['status'], comparisons=comparisons, diagnostic=counts,
                         training_support=[(f['positive_HOLD_count'],f['negative_HOLD_count'],f['positive_HOLD_components']) for f in folds]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
