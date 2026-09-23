#!/usr/bin/env python3
"""Post-join independent recount and fixed-parameter mechanism diagnostics.

Does not train or select a model. Diagnostics were added after seeing the
five-arm result and must not be presented as untouched confirmation.
"""
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_s_bias_competition_v1'
AUTH = ROOT / 'registry/rc_h593_s_bias_competition_authority_v1_20260921.json'
ARMS = ('BIAS1', 'S_FIXED_BIAS2', 'S_BIAS2', 'GAP_BIAS2', 'S_GAP3')


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def checked(source):
    assert bind(source['path']) == source
    return source['path']


def pairs(rows, base, new):
    rescued = [r for r in rows if r['correct'][new] and not r['correct'][base]]
    lost = [r for r in rows if r['correct'][base] and not r['correct'][new]]
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    means = np.array([np.mean(values) for _, values in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    samples = rng.integers(0, len(means), size=(10000, len(means)))
    interval = list(map(float, np.quantile(means[samples].mean(1), [.025, .975])))
    return dict(rescue=len(rescued), loss=len(lost), net=len(rescued)-len(lost),
                changed=sum(r['selected'][base] != r['selected'][new] for r in rows),
                equal_component_difference=float(means.mean()), bootstrap95=interval,
                rescue_query_ids=[r['original_query_id'] for r in rescued],
                loss_query_ids=[r['original_query_id'] for r in lost],
                fold_net={str(f): sum(int(r['correct'][new])-int(r['correct'][base])
                                     for r in rows if r['fold'] == f) for f in range(5)})


def selected(record, model):
    z = [float.fromhex(v) for v in record['models'][model]['logits_hex']]
    top = max(range(127), key=z.__getitem__)
    pos = record['challenger_positions'][top] if z[top] > 0 else record['winner']
    return record['candidate_physical_rows'][pos]


def main():
    a = read(AUTH)
    v = read(OUT / 'validation.json')
    r = read(OUT / 'result.json')
    assert v['status'] == 'S_BIAS_COMPETITION_ALL_COUNTS_PASS'
    assert v['authority'] == r['authority'] == bind(AUTH)
    assert v['result'] == bind(OUT / 'result.json')
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {x['physical_row']: x['identity'] for x in gallery['records']}
    roles = {x['query_id']: x for x in read(checked(a['join_sources']['curator']))['records']}
    joined = {x['query_id']: x for x in r['rows']}
    rows, records, original_switches = [], [], 0
    for f in range(5):
        path = OUT / f'fold{f}'
        fv, iv = read(path / 'validation.json'), read(path / 'independent_validation.json')
        assert fv['authority'] == iv['authority'] == bind(AUTH)
        assert fv['payload'] == iv['payload'] == bind(path / 'payload.json')
        assert fv['status'] == 'S_BIAS_COMPETITION_FRESH_REPLAY_PASS'
        assert iv['passed'] and iv['status'] == 'S_BIAS_COMPETITION_INDEPENDENT_PASS'
        payload = read(path / 'payload.json')
        for p in payload['predictions']:
            q = p['query_id']
            role = roles[q]
            assert role['outer_fold'] == f
            base = [float.fromhex(x) for x in p['models']['COST1_FULL']['logits_hex']]
            top = max(range(127), key=base.__getitem__)
            m = base[top]
            d = m - max(base[i] for i in range(127) if i != top)
            assert d.hex() == float(p['d']).hex()
            decisions = dict(RAW=p['candidate_physical_rows'][p['winner']])
            for name, model in p['models'].items():
                decisions[name] = selected(p, name)
                assert decisions[name] == model['selected']
            if m > 0:
                original_switches += 1
                assert all(p['models'][name]['logits_hex'] == p['models']['COST1_FULL']['logits_hex']
                           for name in ARMS)
            # Frozen-coordinate interventions, no refitting/threshold selection.
            for name, source, remove in (
                ('GAP_DROP_D_KEEP_BIAS', 'GAP_BIAS2', 'd'),
                ('GAP_DROP_BIAS_KEEP_D', 'GAP_BIAS2', 'bias'),
                ('S_GAP_DROP_EXTRA_S', 'S_GAP3', 'extra_s'),
            ):
                head = payload['parameters'][source]
                alpha, beta, bias = (float.fromhex(head[k+'_hex']) for k in ('alpha','beta','bias'))
                if remove == 'd': beta = 0.
                if remove == 'bias': bias = 0.
                if remove == 'extra_s': alpha = 0.
                g = ((m + alpha*p['h_s']) + beta*d) + bias
                switch = m > 0 or (not head['disabled'] and m <= 0 and g > 0)
                pos = p['challenger_positions'][top] if switch else p['winner']
                decisions[name] = p['candidate_physical_rows'][pos]
            correct = {name: labels[s] == role['identity'] for name, s in decisions.items()}
            for name in r['summary']:
                assert joined[q]['selected'][name] == decisions[name]
                assert joined[q]['correct'][name] == correct[name]
            row = dict(query_id=q, original_query_id=role['original_query_id'],
                       fold=f, component=role['component'], identity=role['identity'],
                       selected=decisions, correct=correct)
            rows.append(row)
            records.append(dict(query_id=q, original_query_id=role['original_query_id'], fold=f,
                                m=m, h_s=p['h_s'], d=d,
                                gate_scores={name: ((m+head['alpha']*p['h_s'])+head['beta']*d)+head['bias']
                                             for name,head in payload['parameters'].items()},
                                correct=correct))
    rows.sort(key=lambda x: x['query_id'])
    assert len(rows) == len({x['query_id'] for x in rows}) == 593
    assert len({x['component'] for x in rows}) == 64
    assert original_switches == 78
    counts = {name: sum(x['correct'][name] for x in rows) for name in rows[0]['correct']}
    assert all(counts[name] == r['summary'][name]['correct'] for name in r['summary'])
    comparisons = {base+'__to__GAP_BIAS2': pairs(rows, base, 'GAP_BIAS2')
                   for base in ('RAW','COST1_FULL','BIAS1','S_NET1','CE_FULL','ZERO_S','S_GAP3',
                                'GAP_DROP_D_KEEP_BIAS','GAP_DROP_BIAS_KEEP_D')}
    comparisons['S_GAP_DROP_EXTRA_S__to__S_GAP3'] = pairs(rows,'S_GAP_DROP_EXTRA_S','S_GAP3')
    for base, key in (('RAW','against_RAW'),('COST1_FULL','against_COST1')):
        actual = comparisons[base+'__to__GAP_BIAS2']
        expected = r['summary']['GAP_BIAS2'][key]
        assert all(actual[k] == expected[k] for k in expected)
    first = comparisons['COST1_FULL__to__GAP_BIAS2']
    result = dict(status='S_BIAS_COMPETITION_POST_JOIN_INDEPENDENT_RECOUNT_PASS',
                  code=bind(__file__), authority=bind(AUTH), source_result=bind(OUT/'result.json'),
                  queries=593, components=64, original_switches_preserved=original_switches,
                  parameter_updates=0, counts=counts, comparisons=comparisons,
                  fold_correct={str(f): {m: sum(x['correct'][m] for x in rows if x['fold']==f)
                                        for m in counts} for f in range(5)},
                  changed_correctness_records=[x for x in records if
                      x['original_query_id'] in set(first['rescue_query_ids']+first['loss_query_ids'])],
                  evidence_level='Opened H593 grouped nested OOF development; best arm among five',
                  interventions_are_posthoc=True,
                  limits=['Coordinate-drop diagnostics use frozen parameters; no refit.',
                          'GAP_BIAS2 keeps S in the original COST1 score. Only its additional alpha is zero.',
                          'Main preregistered arm S_GAP3 is 486; do not rename 492 as main-arm confirmation.',
                          'Equal-component bootstrap is descriptive and not corrected for selecting best arm.'])
    target = OUT/'post_join_analysis.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=result['status'], counts=counts, comparisons=comparisons),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
