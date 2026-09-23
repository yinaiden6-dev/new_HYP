#!/usr/bin/env python3
"""Independent input-index, optimizer, threshold and all-score validation.

Numerical helpers come from the frozen previous independent validator, not
the trainer. Its legacy h_s/alpha fields are aliases for signed h_raw/gamma
here; the helpers accept finite signed inputs and do not impose S semantics.
"""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import validate_rc_h593_s_bias_competition_v1 as independent


def alias_rows(records, field):
    return [dict(r, h_s=r[field]) for r in records]


def alias_head(head):
    h = copy.deepcopy(head)
    h['alpha'], h['alpha_hex'] = h['gamma'], h['gamma_hex']
    h['optimization']['free_features'] = ['h_s' if k == 'h_raw' else k
                                           for k in h['optimization']['free_features']]
    return h


def numerical_check(records, field, head):
    rows, h = alias_rows(records, field), alias_head(head)
    optimization = independent.independent_surrogate(rows, 'S_GAP3', h)
    bias = independent.independent_bias(rows, h['alpha'], h['beta'])
    assert bias['bias_hex'] == h['bias_hex'] and bias['disabled'] == h['disabled']
    assert bias['certificates'] == h['calibration_certificate']['certificates']
    for key, val in bias['counts'].items():
        assert head['training_' + key] == val
    return dict(optimization=optimization, counts=bias['counts'],
                distinct_tested_biases=bias['distinct_tested_biases'])


def validate(fold):
    import torch
    import run_rc_h593_raw_incumbent_gate_v1 as runner
    a = runner.guard('verify', fold)
    read, checked, bind = runner.read, runner.checked, runner.bind
    assert a['code_sources']['independent_validator'] == bind(__file__)
    parent = runner.parent_fold(a, fold)  # IO/split checks only; no fitting.
    path = runner.OUT / f'fold{fold}/payload.json'
    p = read(path)
    assert p['status'] == 'RAW_INCUMBENT_OUTER_SEALED'
    assert p['authority'] == bind(runner.AUTH) and p['fold'] == fold
    assert p['parent_payload'] == a['fold_sources'][str(fold)]['parent_payload']
    assert p['train_query_ids'] == parent['train_query_ids']
    assert p['heldout_label_reads'] == p['encoder_forwards'] == p['base_training_updates'] == 0
    assert p['gallery_mapping_sha256'] == parent['gallery_mapping_sha256']
    # Independent extraction from sealed tensors; never call runner.raw_inputs.
    features = {}
    for source in a['features']:
        rec, val = read(checked(source['receipt'])), read(checked(source['validation']))
        assert rec['payload'] == val['payload'] == source['payload']
        assert val['status'] == 'H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS'
        for row in torch.load(checked(source['payload']), map_location='cpu', weights_only=True)['records']:
            assert row['query_id'] not in features
            features[row['query_id']] = row
    assert len(features) == 593

    def check_inputs(row):
        feature = features[row['query_id']]
        tensor = feature['modes']['REAL']['X']
        assert tensor.dtype == torch.float64 and tuple(tensor.shape) == (127, 6)
        raw = [independent.finite(tensor[j, 0]) for j in range(127)]
        assert all(x <= 0 for x in raw)
        assert all(raw[j].hex() == float(feature['modes']['CBIND']['X'][j, 0]).hex()
                   for j in range(127))
        order = sorted(range(127), key=lambda j: (-raw[j], j))
        assert row['raw_nearest_index'] == order[0]
        assert float(row['r_pair']).hex() == raw[row['original_top']].hex()
        assert float(row['r_near']).hex() == raw[order[0]].hex()
        assert row['r_pair'] <= row['r_near'] <= 0

    assert len(p['calibration']) == len(parent['calibration'])
    for row, old in zip(p['calibration'], parent['calibration']):
        assert all(row[k] == v for k, v in old.items())
        check_inputs(row)
        z = independent.logits(row); top, d = independent.top_and_gap(z)
        assert top == row['original_top'] and z[top].hex() == float(row['m']).hex()
        assert d.hex() == float(row['d']).hex()
    control, old = p['matched_control'], parent['parameters']['GAP_BIAS2']
    assert control['gamma'] == 0
    for key in ('beta_hex', 'bias_hex', 'disabled', 'training_net_gain', 'training_rescues', 'training_breaks'):
        assert control[key] == old[key]
    assert control['optimization']['theta_hex'] == old['optimization']['theta_hex']
    independent.independent_surrogate([dict(r, h_s=0.0) for r in p['calibration']],
                                      'GAP_BIAS2', alias_head(control))
    assert set(p['parameters']) == set(runner.ARMS)
    fits = {}
    for name in runner.ARMS:
        h = p['parameters'][name]
        for key in ('gamma', 'beta', 'bias'):
            assert independent.finite(h[key]).hex() == h[key+'_hex']
        assert h['gamma'] >= 0 and h['beta'] >= 0
        fits[name] = numerical_check(p['calibration'],
                                    'r_pair' if name == 'GAP_RAWPAIR3' else 'r_near', h)
    old_rows = {r['query_id']:r for r in parent['predictions']}
    assert len(p['predictions']) == len(old_rows)
    assert {r['query_id'] for r in p['predictions']} == set(old_rows)
    checked_scores = switches = 0
    for row in p['predictions']:
        old = old_rows[row['query_id']]
        assert all(row[k] == v for k,v in old.items() if k != 'models')
        assert all(row['models'][k] == v for k,v in old['models'].items())
        assert set(row['models']) == set(old['models']) | set(runner.ARMS)
        check_inputs(row)
        feature = features[row['query_id']]
        for key in ('candidate_physical_rows', 'challenger_positions', 'winner'):
            assert row[key] == feature[key]
        z = independent.logits(old['models']['COST1_FULL'])
        top, gap = independent.top_and_gap(z)
        switches += z[top] > 0
        for name in runner.ARMS:
            head = p['parameters'][name]
            expected = list(z)
            raw = row['r_pair'] if name == 'GAP_RAWPAIR3' else row['r_near']
            if z[top] <= 0 and not head['disabled']:
                score = ((z[top] + head['gamma']*raw) + head['beta']*gap) + head['bias']
                assert independent.finite(score) == score
                if score > 0:
                    expected[top] = score
            assert [v.hex() for v in expected] == row['models'][name]['logits_hex']
            pos = row['challenger_positions'][top] if expected[top] > 0 else row['winner']
            assert row['models'][name]['selected'] == row['candidate_physical_rows'][pos]
            assert max(range(127), key=expected.__getitem__) == top
            checked_scores += 127
    report = dict(status='RAW_INCUMBENT_INDEPENDENT_PASS', passed=True,
                  authority=bind(runner.AUTH), payload=bind(path), fold=fold,
                  calibration_queries=len(p['calibration']), heldout_queries=len(p['predictions']),
                  input_readout_checks=len(p['calibration'])+len(p['predictions']),
                  new_model_score_checks=checked_scores, original_switches=switches,
                  matched_GAP_BIAS2_exact=True, heldout_label_reads=0, fits=fits)
    runner.write(runner.OUT/f'fold{fold}/independent_validation.json', report)
    print('RAW_INCUMBENT_INDEPENDENT_PASS', fold, checked_scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    validate(parser.parse_args().fold)
