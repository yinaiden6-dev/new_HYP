#!/usr/bin/env python3
"""Descriptive full-candidate M audit; no fitting, threshold selection or inference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_m_scalar_headroom_audit_20260927_v1'
EPS = 1e-12


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b, b['path']
    return read(b['path'])


def main():
    auth_path = ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
    auth = read(auth_path)
    cache_validation_path = ROOT / 'results/rc_h593_simple_explanations_v1/cache_validation.json'
    cv = read(cache_validation_path)
    assert cv['status'] == 'CACHE_PASS' and cv['authority'] == bind(auth_path)
    cache = checked(cv['payload'])
    old = checked(auth['join']['old_result'])
    validation = checked(auth['join']['old_validation'])
    assert validation['result'] == auth['join']['old_result']
    roles = {r['query_id']: r for r in checked(auth['join']['curator'])['records']}
    labels = {r['physical_row']: r['identity'] for r in checked(auth['public']['gallery'])['records']}
    original = {r['query_id']: r for r in old['rows']}
    predictions = {}
    folds = []
    for f in range(5):
        p = ROOT / f'results/rc_six_cause_isolation_v1/loss_binding/fold{f}/payload.json'
        folds.append(bind(p))
        for r in read(p)['predictions']:
            assert r['query_id'] not in predictions
            predictions[r['query_id']] = r['models']['COST1']
    records = []
    maximum_feature_error = 0.
    for r in cache['rows']:
        q = r['query_id']; role = roles[q]; o = original[q]; pred = predictions[q]
        axis = r['axis']; w = r['winner']; c = r['challengers']
        mass = np.asarray(r['mass'], dtype=np.float64)
        assert mass.shape == (128,) and np.isfinite(mass).all() and np.all(mass >= 0)
        relative_m = (mass[c] - mass[w]) / (abs(mass[c]) + abs(mass[w]) + 1e-12)
        err = float(np.max(abs(relative_m - np.asarray(r['native_X'])[:, 2])))
        maximum_feature_error = max(maximum_feature_error, err)
        assert err < 2e-10
        targets = [j for j, physical in enumerate(axis) if labels[physical] == role['identity']]
        assert len(targets) <= 1
        t = targets[0] if targets else None
        assert bool(targets) == o['target_in_C128']
        assert (t == w) == o['correct']['RAW']
        final = axis.index(pred['selected'])
        assert (t == final) == o['correct']['COST1']
        z = np.asarray([float.fromhex(v) for v in pred['logits_hex']])
        bestc = c[int(z.argmax())]
        assert final == (bestc if z.max() > 0 else w)
        mpick = int(np.argmax(mass))  # fixed physical-axis tie break, not selected from labels
        item = dict(query_id=q, display_id=role['original_query_id'], fold=o['fold'],
                    component=o['component'], raw_correct=bool(t == w), cost1_correct=bool(t == final),
                    target_in_C128=bool(targets), raw_physical=axis[w], final_physical=axis[final],
                    target_physical=axis[t] if t is not None else None,
                    max_M_physical=axis[mpick], max_M_correct=bool(t == mpick),
                    max_M_tie_count=int(np.sum(abs(mass - mass[mpick]) <= EPS)),
                    action='HOLD' if final == w else 'SWITCH',
                    best_challenger_correct=bool(t == bestc), max_logit=float(z.max()),
                    target_logit=0. if t == w else float(z[c.index(t)]) if t is not None else None)
        if t is not None:
            wrong = np.asarray([j for j in range(128) if j != t]); target_m = float(mass[t])
            item.update(target_M=target_m, raw_M=float(mass[w]), final_M=float(mass[final]),
                        target_minus_final_M=float(mass[t] - mass[final]),
                        target_minus_strongest_wrong_M=float(mass[t] - np.max(mass[wrong])),
                        M_strict_rank=1 + int(np.sum(mass[wrong] > target_m + EPS)),
                        M_ties_with_wrong=int(np.sum(abs(mass[wrong] - target_m) <= EPS)),
                        M_unique_first=bool(mass[t] > np.max(mass[wrong]) + EPS),
                        M_above_final=bool(mass[t] > mass[final] + EPS),
                        head_challenger_rank=None if t == w else 1 + int(np.sum(z > z[c.index(t)] + EPS)))
        records.append(item)
    assert len(records) == len({r['query_id'] for r in records}) == 593
    assert sum(r['raw_correct'] for r in records) == 426
    assert sum(r['cost1_correct'] for r in records) == 481
    failed = [r for r in records if not r['cost1_correct']]
    eligible = [r for r in failed if r['target_in_C128']]
    tally = dict(population=593, raw_correct=426, cost1_correct=481, cost1_failures=len(failed),
                 failures_target_present=len(eligible), failures_target_absent=len(failed)-len(eligible),
                 failures_M_above_final=sum(r['M_above_final'] for r in eligible),
                 failures_M_unique_first=sum(r['M_unique_first'] for r in eligible),
                 failures_M_rank_at_most_5=sum(r['M_strict_rank'] <= 5 for r in eligible),
                 failures_M_rank_at_most_10=sum(r['M_strict_rank'] <= 10 for r in eligible),
                 failures_M_tied_first=sum(r['M_strict_rank'] == 1 and not r['M_unique_first'] for r in eligible),
                 failures_target_top_challenger_but_held=sum(r['best_challenger_correct'] and r['action']=='HOLD' for r in failed),
                 max_M_fixed_rule_correct=sum(r['max_M_correct'] for r in records),
                 max_M_rescues_vs_cost1=sum(r['max_M_correct'] and not r['cost1_correct'] for r in records),
                 max_M_breaks_vs_cost1=sum(not r['max_M_correct'] and r['cost1_correct'] for r in records),
                 cost1_or_max_M_oracle_correct=sum(r['max_M_correct'] or r['cost1_correct'] for r in records))
    result = dict(status='H593_M_SCALAR_DESCRIPTIVE_AUDIT_PASS', counts=tally,
                  maximum_native_M_feature_error=maximum_feature_error, comparison_tolerance=EPS,
                  scope='All593 original grouped OOF COST1 and natural ColNomic C128; opened development diagnostics',
                  oracle_warning='Answer-guided union is not deployable accuracy, a learned gain, or the ceiling of all M-based heads.',
                  L_warning='No L-based capacity claim here. Native S/M and free full-reference L0 must remain distinct.',
                  new_training_updates=0, new_encoder_or_matcher_forwards=0,
                  sources=dict(authority=bind(auth_path), cache_validation=bind(cache_validation_path),
                               cache=cv['payload'], old_result=auth['join']['old_result'],
                               curator=auth['join']['curator'], gallery=auth['public']['gallery'], folds=folds),
                  program=bind(__file__), records=records)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / 'per_query.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(tally, ensure_ascii=False, indent=2))
    print(json.dumps([r for r in records if r['display_id'] in ('DIFFICULT-0011','OUTCOME-0477')],indent=2))


if __name__ == '__main__':
    main()
