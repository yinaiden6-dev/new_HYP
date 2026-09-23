#!/usr/bin/env python3
"""Post-join recount and training/outer decision trace; no fitting or selection."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_gap_net2_v1'
AUTH = ROOT / 'registry/rc_h593_gap_net2_authority_v1_20260921.json'
OLD, NEW, BASE = 'GAP_BIAS2', 'GAP_NET2', 'COST1_FULL'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(source):
    assert binding(source['path']) == source, source['path']
    return source['path']


def gate(m, d, head):
    assert head['alpha'] == 0
    return ((m + 0.0) + float.fromhex(head['beta_hex']) * d) + float.fromhex(head['bias_hex'])


def changes(records):
    changed = [r for r in records if r['old_switch'] != r['new_switch']]
    return dict(changed=len(changed),
                rescue=sum(r['accuracy_delta'] == 1 for r in changed),
                loss=sum(r['accuracy_delta'] == -1 for r in changed),
                neutral=sum(r['accuracy_delta'] == 0 for r in changed),
                net=sum(r['accuracy_delta'] for r in changed),
                component_net={c: sum(r['accuracy_delta'] for r in changed if r['component'] == c)
                               for c in sorted({r['component'] for r in changed})},
                records=changed)


def comparison(rows, base, new):
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    means = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    boot = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(1)
    return dict(rescue=sum(r['correct'][new] and not r['correct'][base] for r in rows),
                loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                changed=sum(r['selected'][new] != r['selected'][base] for r in rows),
                equal_component_difference=float(means.mean()),
                bootstrap95=list(map(float, np.quantile(boot, [.025, .975]))))


def main():
    a, result, v = read(AUTH), read(OUT / 'result.json'), read(OUT / 'validation.json')
    assert result['status'] == 'H593_GAP_NET2_COMPLETE'
    assert v['status'] == 'GAP_NET2_ALL_COUNTS_PASS'
    assert v['authority'] == result['authority'] == binding(AUTH)
    assert v['result'] == binding(OUT / 'result.json')
    for section in ('code_sources', 'public_sources', 'join_sources'):
        for source in a[section].values():
            checked(source)
    checked(a['parent'])
    prelabel = read(OUT / 'all_predictions_prelabel_seal.json')
    assert prelabel['authority'] == binding(AUTH)
    assert len(prelabel['validations']) == 10
    for source in prelabel['validations']:
        checked(source)
    gallery = {r['physical_row']: r['identity']
               for r in read(a['public_sources']['gallery']['path'])['records']}
    roles = {r['query_id']: r for r in read(a['join_sources']['curator']['path'])['records']}
    parent_result = read(a['join_sources']['parent_result']['path'])
    parent_rows = {r['query_id']: r for r in parent_result['rows']}
    joined = {r['query_id']: r for r in result['rows']}
    rows, folds, traces, switches, candidates_missing = [], [], [], 0, 0
    for f in range(5):
        for source in a['fold_sources'][str(f)].values():
            checked(source)
        directory = OUT / f'fold{f}'
        p, fv, iv = (read(directory / x) for x in ('payload.json', 'validation.json', 'independent_validation.json'))
        assert fv['status'] == 'GAP_NET2_FRESH_REPLAY_PASS'
        assert iv['status'] == 'GAP_NET2_INDEPENDENT_PASS' and iv['passed']
        assert fv['payload'] == iv['payload'] == binding(directory / 'payload.json')
        assert fv['authority'] == iv['authority'] == p['authority'] == binding(AUTH)
        assert p['heldout_label_reads'] == p['encoder_forwards'] == p['base_training_updates'] == 0
        parent = read(checked(p['parent_payload']))
        prior = {r['query_id']: r for r in parent['predictions']}
        assert p['calibration'] == parent['calibration']
        assert p['train_query_ids'] == parent['train_query_ids']
        old, new = parent['parameters'][OLD], p['parameters'][NEW]
        inner_records = []
        for cal in p['calibration']:
            m, d = cal['m'], cal['d']
            before, after = gate(m, d, old), gate(m, d, new)
            os = m <= 0 and not old['disabled'] and before > 0
            ns = m <= 0 and not new['disabled'] and after > 0
            role = roles[cal['query_id']]
            inner_records.append(dict(query_id=cal['query_id'], original_query_id=role['original_query_id'],
                                      component=role['component'], identity=role['identity'],
                                      m=m, d=d, old_gate=before, new_gate=after, old_switch=os, new_switch=ns,
                                      correctness_delta_if_switched=cal['delta'],
                                      accuracy_delta=(int(ns)-int(os))*cal['delta']))
        inner = changes(inner_records)
        assert inner['net'] == new['training_net_gain'] - old['training_net_gain']
        outer_records = []
        for pred in p['predictions']:
            q = pred['query_id']; role = roles[q]
            assert role['outer_fold'] == f and q not in p['train_query_ids']
            assert {k: value for k, value in pred.items() if k != 'models'} == {
                k: value for k, value in prior[q].items() if k != 'models'}
            assert {k: value for k, value in pred['models'].items() if k != NEW} == prior[q]['models']
            base = list(map(float.fromhex, pred['models'][BASE]['logits_hex']))
            top = max(range(127), key=base.__getitem__)
            m = base[top]; d = m-max(z for j, z in enumerate(base) if j != top)
            assert top == pred['original_top'] and d.hex() == float(pred['d']).hex()
            decisions = {'RAW': pred['candidate_physical_rows'][pred['winner']]}
            for name, model in pred['models'].items():
                z = list(map(float.fromhex, model['logits_hex']))
                j = max(range(127), key=z.__getitem__)
                pos = pred['challenger_positions'][j] if z[j] > 0 else pred['winner']
                decisions[name] = pred['candidate_physical_rows'][pos]
                assert decisions[name] == model['selected']
            before, after = gate(m, d, old), gate(m, d, new)
            expected = list(pred['models'][BASE]['logits_hex'])
            if m <= 0 and not new['disabled'] and after > 0:
                expected[top] = after.hex()
            assert expected == pred['models'][NEW]['logits_hex']
            if m > 0:
                switches += 1
                assert pred['models'][NEW] == pred['models'][BASE]
            if new['search']['incumbent_retained']:
                assert pred['models'][NEW] == pred['models'][OLD]
            correct = {name: gallery[s] == role['identity'] for name, s in decisions.items()}
            present = any(gallery[s] == role['identity'] for s in pred['candidate_physical_rows'])
            candidates_missing += int(not present)
            assert present == joined[q]['target_in_C128']
            assert correct == joined[q]['correct'] and decisions == joined[q]['selected']
            assert {k:v for k,v in correct.items() if k != NEW} == parent_rows[q]['correct']
            rows.append(dict(query_id=q, fold=f, component=role['component'], selected=decisions, correct=correct))
            trace = dict(query_id=q, original_query_id=role['original_query_id'], fold=f,
                         identity=role['identity'], component=role['component'], m=m, d=d,
                         old_gate=before, new_gate=after,
                         slope_contribution_change=(new['beta']-old['beta'])*d,
                         bias_change=new['bias']-old['bias'],
                         new_bias_crossing=-(m+new['beta']*d),
                         raw_identity=gallery[decisions['RAW']],
                         challenger_identity=gallery[pred['candidate_physical_rows'][pred['challenger_positions'][top]]],
                         old_switch=decisions[OLD] != decisions['RAW'],
                         new_switch=decisions[NEW] != decisions['RAW'],
                         correct=correct, accuracy_delta=int(correct[NEW])-int(correct[OLD]))
            outer_records.append(trace)
            if trace['old_switch'] != trace['new_switch'] or (correct[BASE] and not correct[OLD]):
                traces.append(trace)
        folds.append(dict(fold=f, queries=len(outer_records),
                          old_beta=old['beta'], new_beta=new['beta'], old_bias=old['bias'], new_bias=new['bias'],
                          old_inner_net=old['training_net_gain'], new_inner_net=new['training_net_gain'],
                          candidate_count=new['search']['candidate_count'],
                          incumbent_retained=new['search']['incumbent_retained'],
                          runtime=read(directory/'runtime.json'), inner_changes=inner,
                          outer_changes=changes(outer_records)))
    assert len(rows) == len({r['query_id'] for r in rows}) == 593
    assert len({r['component'] for r in rows}) == 64
    assert switches == 78 and candidates_missing == 23
    counts = {name: sum(r['correct'][name] for r in rows) for name in result['summary']}
    assert all(counts[name] == summary['correct'] for name, summary in result['summary'].items())
    for key, expected in result['comparisons'].items():
        base, new = key.split('__to__')
        assert comparison(rows, base, new) == expected
    for base, key in [('RAW', 'against_RAW'), (BASE, 'against_COST1')]:
        assert comparison(rows, base, NEW) == result['summary'][NEW][key]
    for f in range(5):
        assert sum(r['correct'][NEW] for r in rows if r['fold'] == f) == result['summary'][NEW]['fold_correct'][str(f)]
    output = dict(status='GAP_NET2_POST_JOIN_INDEPENDENT_RECOUNT_PASS', code=binding(__file__),
                  authority=binding(AUTH), result=binding(OUT/'result.json'),
                  queries=593, components=64, original_switches_preserved=switches,
                  target_missing=candidates_missing, parameter_updates=0, counts=counts,
                  comparisons=result['comparisons'], folds=folds, outer_case_traces=traces,
                  limits=['Post-join analysis, not another model or parameter selection.',
                          'Inner calibration sets overlap; their gains must not be pooled as unique-query accuracy.',
                          'Accuracy/actions/group comparisons independently recounted; full-ranking MRR uses existing join replay.',
                          'Opened H593 development, not independent external confirmation.'])
    (OUT/'post_join_analysis.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'], counts=counts,
                         comparison=output['comparisons'][OLD+'__to__'+NEW],
                         folds=[dict(fold=f['fold'],inner_net=f['inner_changes']['net'],outer_net=f['outer_changes']['net'])
                                for f in folds]), indent=2))


if __name__ == '__main__':
    main()
