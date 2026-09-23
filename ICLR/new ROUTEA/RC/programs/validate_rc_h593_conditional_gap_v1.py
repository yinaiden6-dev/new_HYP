#!/usr/bin/env python3
"""Independent conditional-gap input, feature, KKT, threshold and score audit.

No trainer numerical helpers are imported. Only frozen data bindings and
runner IO/access checks are reused; all score arithmetic is reconstructed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('GAP_RAW_INTERACT4', 'GAP_CURVE4')
FEATURES = {'GAP_BIAS2': ('d',), 'GAP_RAWNEAR3': ('h_raw', 'd'),
            'GAP_RAW_INTERACT4': ('h_raw', 'd', 'phi'),
            'GAP_CURVE4': ('h_raw', 'd', 'phi')}


def finite(value):
    value = float(value)
    assert math.isfinite(value)
    return value


def logits(row):
    z = [finite(float.fromhex(v)) for v in row['logits_hex']]
    assert len(z) == 127
    return z


def top_gap(z):
    top = max(range(len(z)), key=z.__getitem__)
    second = max(v for i, v in enumerate(z) if i != top)
    return top, float(z[top] - second)


def raw_phi(raw, gap, mode):
    if mode == 'GAP_RAW_INTERACT4':
        return finite(raw * gap)
    if mode == 'GAP_CURVE4':
        return finite(gap * gap)
    return 0.0


def scale_for(records, mode):
    values = [raw_phi(r['r_near'], r['d'], mode) for r in records if r['m'] <= 0.0]
    x = np.asarray(values, dtype=np.float64)
    fallback = len(x) == 0 or bool(np.all(x == 0.0))
    scale = 1.0 if fallback else float(np.sqrt(np.mean(x*x, dtype=np.float64)))
    assert scale > 0.0
    return finite(scale), values


def control_alias(head):
    """Add neutral fourth-coordinate fields only to a local verifier copy."""
    return dict(head, eta=0.0, eta_hex=0.0.hex(), scale=1.0, scale_hex=1.0.hex())


def feature_value(raw, gap, mode, scale):
    assert scale > 0.0
    return finite(raw_phi(raw, gap, mode) / scale)


def prebias(row, head):
    gamma = finite(float.fromhex(head['gamma_hex']))
    beta = finite(float.fromhex(head['beta_hex']))
    eta = finite(float.fromhex(head['eta_hex']))
    scale = finite(float.fromhex(head['scale_hex']))
    raw = 0.0 if head['mode'] == 'GAP_BIAS2' else row['r_near']
    u = finite(finite(row['m'] + finite(gamma * raw)) + finite(beta * row['d']))
    if eta != 0.0:
        phi = feature_value(row['r_near'], row['d'], head['mode'], scale)
        u = finite(u + finite(eta * phi))
    return u


def independent_bias(records, head):
    """Exact fixed-slope net optimum via independent endpoint enumeration."""
    candidates = {0.0}
    certificates = []
    for i, row in enumerate(records):
        if row['m'] > 0.0:
            continue
        u = prebias(row, head)
        crossing = math.nextafter(-u, math.inf)
        if math.isfinite(crossing):
            predecessor = math.nextafter(crossing, -math.inf)
            assert float(u + crossing) > 0 and float(u + predecessor) <= 0
            candidates.update((crossing, predecessor))
        else:
            crossing = predecessor = None
        certificates.append(dict(index=i, u_hex=u.hex(), delta=row['delta'],
                                 crossing_hex=None if crossing is None else crossing.hex(),
                                 predecessor_hex=None if predecessor is None else predecessor.hex()))
    best = (0, 0, 0.0, 0.0)
    bias = 0.0
    counts = dict(net_gain=0, changed_holds=0, rescues=0, breaks=0, both_wrong=0)
    for candidate in sorted(candidates):
        scores = [float(float.fromhex(c['u_hex']) + candidate) for c in certificates]
        if not all(math.isfinite(s) for s in scores):
            continue
        deltas = [c['delta'] for c, s in zip(certificates, scores) if s > 0.0]
        key = (sum(deltas), -len(deltas), -abs(candidate), -candidate)
        if sum(deltas) > 0 and key > best:
            best, bias = key, candidate
            counts = dict(net_gain=sum(deltas), changed_holds=len(deltas),
                          rescues=deltas.count(1), breaks=deltas.count(-1),
                          both_wrong=deltas.count(0))
    assert head['bias_hex'] == bias.hex()
    assert head['disabled'] == (counts['net_gain'] <= 0)
    for key, value in counts.items():
        assert head['training_' + key] == value
    cert = head['calibration_certificate']
    assert cert['status'] == 'EXACT_FINITE_BINARY64_BIAS_NET_GAIN_OPTIMUM_FIXED_SLOPES'
    assert cert['certificates'] == certificates
    assert cert['selected_counts'] == counts
    digest = hashlib.sha256(json.dumps(certificates, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    assert cert['breakpoints_sha256'] == digest
    return dict(**counts, independent_bias_candidates=len(candidates), bias_hex=bias.hex())


def independent_surrogate(records, head):
    mode = head['mode']
    features = FEATURES[mode]
    opt = head['optimization']
    assert opt['method'] == 'L-BFGS-B' and opt['success']
    assert opt['free_features'] == list(features) + ['bias']
    assert opt['l2'] == .001 and opt['maxiter'] == 2000
    assert opt['ftol'] == 1e-12 and opt['gtol'] == 1e-8
    assert opt['initialization'] == 'all zero'
    assert opt['joint_empirical_net_optimum_claimed'] is False
    theta = np.array([finite(float.fromhex(x)) for x in opt['theta_hex']], dtype=np.float64)
    assert len(theta) == len(features) + 1
    weights = dict(zip(features, theta[:-1]))
    for feature, key in (('h_raw', 'gamma'), ('d', 'beta'), ('phi', 'eta')):
        assert float(weights.get(feature, 0.0)).hex() == head[key + '_hex']
    constrained = 1 if mode == 'GAP_BIAS2' else 2
    assert np.all(theta[:constrained] >= 0.0)
    assert float(theta[-1]).hex() == opt['surrogate_bias_hex']
    scale = float.fromhex(head['scale_hex'])
    rows = [r for r in records if r['m'] <= 0.0 and r['delta'] != 0]
    assert len(rows) == opt['informative_hold_count']
    assert opt['neutral_holds_omitted_from_surrogate'] == sum(
        r['m'] <= 0.0 and r['delta'] == 0 for r in records)
    design_rows = []
    for r in rows:
        values = dict(h_raw=r['r_near'], d=r['d'],
                      phi=feature_value(r['r_near'], r['d'], mode, scale))
        design_rows.append([values[k] for k in features] + [1.0])
    x = np.asarray(design_rows, dtype=np.float64).reshape(len(rows), len(theta))
    m = np.asarray([r['m'] for r in rows], dtype=np.float64)
    y = np.asarray([r['delta'] for r in rows], dtype=np.float64)

    def evaluate(v):
        if not len(rows):
            return .0005 * float(v @ v), .001 * v
        yz = y * (m + x @ v)
        residual = -y * np.exp(-np.logaddexp(0.0, yz))
        return (float(np.logaddexp(0.0, -yz).mean() + .0005 * (v @ v)),
                x.T @ residual / len(rows) + .001 * v)

    initial, _ = evaluate(np.zeros(len(theta)))
    final, gradient = evaluate(theta)
    expected = np.array([float.fromhex(x) for x in opt['gradient_hex']])
    assert np.allclose(gradient, expected, rtol=0.0, atol=1e-11)
    pg = gradient.copy()
    for j in range(constrained):
        if theta[j] <= 0.0 and pg[j] > 0.0:
            pg[j] = 0.0
    norm = float(np.max(np.abs(pg), initial=0.0))
    assert norm <= 1e-5 and abs(norm - opt['projected_gradient_inf_norm']) <= 1e-11
    assert abs(initial - opt['initial_surrogate_loss']) <= 1e-11
    assert abs(final - opt['fitted_surrogate_loss']) <= 1e-11 and final <= initial + 1e-10
    calibrated = theta.copy()
    calibrated[-1] = float.fromhex(head['bias_hex'])
    after, _ = evaluate(calibrated)
    assert abs(after - opt['calibrated_bias_surrogate_loss']) <= 1e-11
    return dict(informative_holds=len(rows), fitted_surrogate_loss=final,
                projected_gradient_inf_norm=norm)


def numerical_check(records, head):
    for key in ('gamma', 'beta', 'eta', 'bias', 'scale'):
        assert finite(head[key]).hex() == head[key + '_hex']
    assert head['gamma'] >= 0 and head['beta'] >= 0 and head['scale'] > 0
    if head['mode'] in ARMS:
        scale, values = scale_for(records, head['mode'])
        assert head['scale_hex'] == scale.hex()
        info = head['training_scale']
        assert info['count'] == len(values)
        assert info['raw_phi_hex'] == [v.hex() for v in values]
        assert info['mode'] == head['mode']
        assert info['scale_hex'] == scale.hex() and float(info['scale']).hex() == scale.hex()
        assert info['empty_or_allzero_fallback'] == (not values or all(v == 0.0 for v in values))
        assert info['centered'] is False and info['old_inputs_rescaled'] is False
        assert head['optimization']['bounds'] == [[0.0, None], [0.0, None], [None, None], [None, None]]
    else:
        assert head['eta'] == 0.0 and head['scale'] == 1.0
    return dict(optimization=independent_surrogate(records, head),
                calibration=independent_bias(records, head), scale_hex=head['scale_hex'])


def validate(fold):
    import torch
    sys.path.insert(0, str(ROOT / 'programs'))
    import run_rc_h593_conditional_gap_v1 as runner
    a = runner.guard('verify', fold)
    read, checked, bind = runner.read, runner.checked, runner.bind
    assert a['code_sources']['independent_validator'] == bind(__file__)
    parent = runner.parent_fold(a, fold)
    path = runner.OUT / f'fold{fold}/payload.json'
    p = read(path)
    assert p['status'] == 'CONDITIONAL_GAP_OUTER_SEALED'
    assert p['authority'] == bind(runner.AUTH) and p['fold'] == fold
    assert p['parent_payload'] == a['fold_sources'][str(fold)]['parent_payload']
    assert p['train_query_ids'] == parent['train_query_ids']
    assert p['heldout_label_reads'] == p['encoder_forwards'] == p['base_training_updates'] == 0
    assert p['gallery_mapping_sha256'] == parent['gallery_mapping_sha256']
    features = {}
    for source in a['features']:
        receipt, validation = read(checked(source['receipt'])), read(checked(source['validation']))
        assert receipt['payload'] == validation['payload'] == source['payload']
        assert validation['status'] == 'H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS'
        for row in torch.load(checked(source['payload']), map_location='cpu', weights_only=True)['records']:
            assert row['query_id'] not in features
            features[row['query_id']] = row
    assert len(features) == 593

    def check_inputs(row):
        feature = features[row['query_id']]
        tensor = feature['modes']['REAL']['X']
        assert tensor.dtype == torch.float64 and tuple(tensor.shape) == (127, 6)
        raw = [finite(tensor[j, 0]) for j in range(127)]
        assert all(v <= 0 for v in raw)
        assert all(raw[j].hex() == float(feature['modes']['CBIND']['X'][j, 0]).hex()
                   for j in range(127))
        order = sorted(range(127), key=lambda j: (-raw[j], j))
        assert row['raw_nearest_index'] == order[0]
        assert float(row['r_pair']).hex() == raw[row['original_top']].hex()
        assert float(row['r_near']).hex() == raw[order[0]].hex()
        assert row['r_pair'] <= row['r_near'] <= 0

    assert len(p['calibration']) == len(parent['calibration'])
    for row, old in zip(p['calibration'], parent['calibration']):
        assert row == old
        check_inputs(row)
        z = logits(row)
        top, gap = top_gap(z)
        assert top == row['original_top'] and z[top].hex() == float(row['m']).hex()
        assert gap.hex() == float(row['d']).hex()
        assert row['delta'] == int(row['top_correct']) - int(row['raw_correct'])
    controls = {}
    for key, old in (('matched_control', parent['matched_control']),
                     ('structural_control', parent['parameters']['GAP_RAWNEAR3'])):
        control = p[key]
        assert control == old, key
        controls[key] = numerical_check(p['calibration'], control_alias(control))
    assert set(p['parameters']) == set(ARMS) == set(runner.ARMS)
    fits = {}
    for name in ARMS:
        assert p['parameters'][name]['mode'] == name
        fits[name] = numerical_check(p['calibration'], p['parameters'][name])
    old_rows = {r['query_id']: r for r in parent['predictions']}
    assert len(p['predictions']) == len(old_rows)
    assert {r['query_id'] for r in p['predictions']} == set(old_rows)
    checked_scores = switches = 0
    for row in p['predictions']:
        old = old_rows[row['query_id']]
        assert all(row[k] == v for k, v in old.items() if k != 'models')
        assert all(row['models'][k] == v for k, v in old['models'].items())
        assert set(row['models']) == set(old['models']) | set(ARMS)
        check_inputs(row)
        feature = features[row['query_id']]
        for key in ('candidate_physical_rows', 'challenger_positions', 'winner'):
            assert row[key] == feature[key]
        z = logits(old['models']['COST1_FULL'])
        top, gap = top_gap(z)
        assert top == row['original_top'] and gap.hex() == float(row['d']).hex()
        switches += z[top] > 0.0
        gates = dict(p['parameters'], GAP_BIAS2=control_alias(p['matched_control']),
                     GAP_RAWNEAR3=control_alias(p['structural_control']))
        for name, head in gates.items():
            expected = list(z)
            if z[top] <= 0.0 and not head['disabled']:
                inputs = dict(m=z[top], r_near=row['r_near'], d=gap)
                score = finite(prebias(inputs, head) + float.fromhex(head['bias_hex']))
                if score > 0.0:
                    expected[top] = score
            assert [v.hex() for v in expected] == row['models'][name]['logits_hex']
            position = row['challenger_positions'][top] if expected[top] > 0.0 else row['winner']
            assert row['models'][name]['selected'] == row['candidate_physical_rows'][position]
            assert max(range(127), key=expected.__getitem__) == top
            checked_scores += 127
    report = dict(status='CONDITIONAL_GAP_INDEPENDENT_PASS', passed=True,
                  authority=bind(runner.AUTH), payload=bind(path), fold=fold,
                  calibration_queries=len(p['calibration']), heldout_queries=len(p['predictions']),
                  input_readout_checks=len(p['calibration']) + len(p['predictions']),
                  new_model_score_checks=checked_scores, original_switches=switches,
                  matched_GAP_BIAS2_exact=True, structural_RAWNEAR3_exact=True,
                  scale_fitted_on_inner_holds_only=True, heldout_label_reads=0,
                  fits=fits, controls=controls)
    runner.write(runner.OUT / f'fold{fold}/independent_validation.json', report)
    print('CONDITIONAL_GAP_INDEPENDENT_PASS', fold, checked_scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    validate(parser.parse_args().fold)
