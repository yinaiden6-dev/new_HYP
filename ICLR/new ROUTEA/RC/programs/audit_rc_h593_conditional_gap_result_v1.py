#!/usr/bin/env python3
"""Post-join, label-open descriptive audit. No fitting or model selection."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_conditional_gap_v1'
ARMS = ('GAP_RAW_INTERACT4', 'GAP_CURVE4')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(b):
    assert binding(b['path']) == b
    return Path(b['path'])


def literal(m, raw, gap, head, drop_eta=False):
    u = (m + head['gamma'] * raw) + head['beta'] * gap
    eta = head.get('eta', 0.0)
    term = 0.0
    if eta != 0.0 and not drop_eta:
        phi = (raw * gap if head['mode'] == ARMS[0] else gap * gap) / head['scale']
        term = eta * phi
        u = u + term
    return u + head['bias'], term


def paired(rows, base, arm):
    rescue = [r['original_query_id'] for r in rows if r['correct'][arm] and not r['correct'][base]]
    loss = [r['original_query_id'] for r in rows if r['correct'][base] and not r['correct'][arm]]
    return dict(rescues=rescue, losses=loss, net=len(rescue)-len(loss))


def main():
    validation = read(OUT / 'validation.json')
    assert validation['status'] == 'CONDITIONAL_GAP_ALL_COUNTS_PASS'
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    for b in authority['code_sources'].values():
        checked(b)
    gallery = read(checked(authority['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    rows = result['rows']
    assert len(rows) == len({r['query_id'] for r in rows}) == 593
    assert len({r['component'] for r in rows}) == 64
    assert sum(not r['target_in_C128'] for r in rows) == 23
    by_query = {r['query_id']: r for r in rows}
    fits, counterfactual, details = [], defaultdict(list), []
    cases = []
    selected_cases = {'OUTCOME-0341', 'OUTCOME-0568', 'OUTCOME-0721',
                      'OUTCOME-0722', 'OUTCOME-0379', 'DIFFICULT-0105'}
    for f in range(5):
        v = read(OUT / f'fold{f}/validation.json')
        iv = read(OUT / f'fold{f}/independent_validation.json')
        assert v['payload'] == iv['payload'] and iv['passed']
        p = read(checked(v['payload']))
        all_heads = dict(GAP_BIAS2=p['matched_control'], GAP_RAWNEAR3=p['structural_control'], **p['parameters'])
        for arm, h in all_heads.items():
            fits.append(dict(fold=f, arm=arm,
                             coefficients={k: h[k] for k in ('gamma','beta','eta','bias','scale') if k in h},
                             inner_net=h['training_net_gain'],
                             inner_rescue=h['training_rescues'], inner_loss=h['training_breaks'],
                             fitted_logistic=h['optimization']['fitted_surrogate_loss'],
                             gradient_residual=h['optimization']['projected_gradient_inf_norm']))
        for pred in p['predictions']:
            row = by_query[pred['query_id']]
            z = [float.fromhex(x) for x in pred['models']['COST1_FULL']['logits_hex']]
            top = max(range(len(z)), key=z.__getitem__)
            m, raw, gap = z[top], pred['r_near'], pred['d']
            targets = dict(raw=pred['candidate_physical_rows'][pred['winner']],
                           challenger=pred['candidate_physical_rows'][pred['challenger_positions'][top]])
            gates = {}
            for arm, head in all_heads.items():
                score, term = literal(m, raw, gap, head)
                gates[arm] = dict(score=score, extra_term=term, correct=row['correct'][arm])
            if row['original_query_id'] in selected_cases:
                cases.append(dict(query=row['original_query_id'], fold=f, m=m, r=raw, d=gap, gates=gates))
            for arm in ARMS:
                head = p['parameters'][arm]
                score, _ = literal(m, raw, gap, head, drop_eta=True)
                selected = targets['challenger'] if m > 0.0 or (not head['disabled'] and score > 0.0) else targets['raw']
                counterfactual[arm].append(dict(query=row['original_query_id'],
                                                full=row['correct'][arm],
                                                drop_eta=labels[selected] == row['identity']))
                if row['selected'][arm] != row['selected']['GAP_BIAS2']:
                    details.append(dict(query_id=row['query_id'], original_query_id=row['original_query_id'],
                                        fold=f, identity=row['identity'], arm=arm,
                                        RAW_correct=row['correct']['RAW'],
                                        incumbent492_correct=row['correct']['GAP_BIAS2'],
                                        new_correct=row['correct'][arm],
                                        m=m, r=raw, d=gap,
                                        new_gate=gates[arm]['score'], extra_term=gates[arm]['extra_term']))
    for arm, summary in result['summary'].items():
        assert sum(r['correct'][arm] for r in rows) == summary['correct']
    contrasts = {}
    for arm, rr in counterfactual.items():
        assert len(rr) == 593
        contrasts[arm] = dict(full=sum(r['full'] for r in rr), drop_eta_keep_other_fitted_parameters=sum(r['drop_eta'] for r in rr),
                             added_term_rescues=sum(r['full'] and not r['drop_eta'] for r in rr),
                             added_term_losses=sum(r['drop_eta'] and not r['full'] for r in rr),
                             scope='Post-hoc frozen-coordinate intervention; not an independently trained or validated alternative model.')
    report = dict(status='CONDITIONAL_GAP_POST_JOIN_AUDIT_PASS', result=validation['result'],
                  authority=validation['authority'], program=binding(__file__),
                  rows=593, components=64, C128_misses=23,
                  comparisons={base+'__to__'+arm:paired(rows, base, arm) for arm in ARMS for base in ('GAP_BIAS2','GAP_RAWNEAR3')},
                  fits=fits, fixed_coordinate_diagnostics=contrasts, illustrative_cases=cases,
                  trained_new_models=0, external_GO=False)
    (OUT / 'post_join_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (OUT / 'changed_cases.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader();writer.writerows(details)
    print(json.dumps(dict(status=report['status'], diagnostics=contrasts),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
