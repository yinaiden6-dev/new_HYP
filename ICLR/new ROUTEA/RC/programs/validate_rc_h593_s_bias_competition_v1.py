#!/usr/bin/env python3
"""Independent FP64 HOLD-gate replay and fixed-slope bias certificate.

No imports of fitting, calibration, or gate implementations. Parent inner OOF
scores are reused only after checking their already sealed independent audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_s_bias_competition_v1'
AUTH = ROOT / 'registry/rc_h593_s_bias_competition_authority_v1_20260921.json'
ARMS = ('BIAS1', 'S_FIXED_BIAS2', 'S_BIAS2', 'GAP_BIAS2', 'S_GAP3')
FEATURES = {'BIAS1': (), 'S_BIAS2': ('h_s',), 'GAP_BIAS2': ('d',),
            'S_GAP3': ('h_s', 'd')}


def finite(x):
    x = float(x)
    assert math.isfinite(x)
    return x


def logits(record):
    z = [finite(float.fromhex(v)) for v in record['logits_hex']]
    assert len(z) == 127
    return z


def top_and_gap(z):
    top = max(range(len(z)), key=z.__getitem__)
    runnerup = max(z[i] for i in range(len(z)) if i != top)
    return top, float(z[top] - runnerup)


def offset(row, alpha, beta):
    # Literal scalar operations; no reassociation or fused multiply-add.
    ah = finite(alpha * row['h_s'])
    bd = finite(beta * row['d'])
    return finite(finite(row['m'] + ah) + bd)


def independent_bias(records, alpha, beta):
    """Enumerate every change endpoint, predecessor and zero, independently.

    Within each constant-action interval the bias nearest zero is one of these
    candidates. The disabled action is compared explicitly, so calibration
    cannot introduce changes with nonpositive empirical net gain.
    """
    pairs = [(r, offset(r, alpha, beta)) for r in records if r['m'] <= 0.0]
    candidates = {0.0}
    certificates = []
    for i, row in enumerate(records):
        if row['m'] > 0.0:
            continue
        u = offset(row, alpha, beta)
        boundary = math.nextafter(-u, math.inf)
        if math.isfinite(boundary):
            predecessor = math.nextafter(boundary, -math.inf)
            assert float(u + boundary) > 0 and float(u + predecessor) <= 0
            candidates.update((boundary, predecessor))
        else:
            boundary = predecessor = None
        certificates.append(dict(index=i, u_hex=u.hex(), delta=row['delta'],
                                 crossing_hex=None if boundary is None else boundary.hex(),
                                 predecessor_hex=None if predecessor is None else predecessor.hex()))
    best = (0, 0, 0.0, 0.0)
    selected_bias = 0.0
    counts = dict(net_gain=0, changed_holds=0, rescues=0, breaks=0, both_wrong=0)
    feasible = 0
    for bias in sorted(candidates):
        scores = [float(u + bias) for _, u in pairs]
        if not all(math.isfinite(v) for v in scores):
            continue
        feasible += 1
        deltas = [row['delta'] for (row, _), score in zip(pairs, scores) if score > 0.0]
        key = (sum(deltas), -len(deltas), -abs(bias), -bias)
        if sum(deltas) > 0 and key > best:
            best = key
            selected_bias = bias
            counts = dict(net_gain=sum(deltas), changed_holds=len(deltas),
                          rescues=deltas.count(1), breaks=deltas.count(-1),
                          both_wrong=deltas.count(0))
    return dict(bias_hex=selected_bias.hex(), disabled=counts['net_gain'] <= 0,
                counts=counts, certificates=certificates,
                distinct_tested_biases=len(candidates), feasible_tested_biases=feasible)


def independent_surrogate(records, mode, head):
    """Evaluate the pinned convex surrogate and KKT residual independently."""
    import numpy as np
    opt = head['optimization']
    features = FEATURES[mode]
    assert opt['method'] == 'L-BFGS-B' and opt['success']
    assert opt['free_features'] == list(features) + ['bias']
    assert opt['l2'] == .001 and opt['maxiter'] == 2000
    assert opt['ftol'] == 1e-12 and opt['gtol'] == 1e-8
    assert opt['initialization'] == 'all zero'
    assert opt['joint_empirical_net_optimum_claimed'] is False
    theta = np.array([finite(float.fromhex(x)) for x in opt['theta_hex']], dtype=np.float64)
    assert len(theta) == len(features) + 1 and np.all(theta[:-1] >= 0.0)
    weights = dict(zip(features, theta[:-1]))
    assert float(weights.get('h_s', 0.0)).hex() == head['alpha_hex']
    assert float(weights.get('d', 0.0)).hex() == head['beta_hex']
    assert float(theta[-1]).hex() == opt['surrogate_bias_hex']
    rows = [r for r in records if r['m'] <= 0.0 and r['delta'] != 0]
    assert len(rows) == opt['informative_hold_count']
    assert opt['neutral_holds_omitted_from_surrogate'] == sum(
        r['m'] <= 0.0 and r['delta'] == 0 for r in records)
    x = np.array([[r[k] for k in features] + [1.0] for r in rows], dtype=np.float64)
    x = x.reshape(len(rows), len(theta))
    m = np.array([r['m'] for r in rows], dtype=np.float64)
    y = np.array([r['delta'] for r in rows], dtype=np.float64)

    def evaluate(v):
        if not len(rows):
            return .0005 * float(v @ v), .001 * v
        z = m + x @ v
        yz = y * z
        # exp(-logaddexp(0, yz)) is stable sigmoid(-yz), using neither
        # scipy.special.expit nor any trainer helper.
        residual = -y * np.exp(-np.logaddexp(0.0, yz))
        loss = float(np.logaddexp(0.0, -yz).mean() + .0005 * (v @ v))
        return loss, x.T @ residual / len(rows) + .001 * v

    initial, _ = evaluate(np.zeros(len(theta), dtype=np.float64))
    final, gradient = evaluate(theta)
    expected_gradient = np.array([float.fromhex(v) for v in opt['gradient_hex']])
    assert np.allclose(gradient, expected_gradient, rtol=0.0, atol=1e-11)
    projected = gradient.copy()
    for j in range(len(features)):
        if theta[j] <= 0.0 and projected[j] > 0.0:
            projected[j] = 0.0
    pg = float(np.max(np.abs(projected), initial=0.0))
    assert pg <= 1e-5, (mode, 'projected gradient', pg)
    assert abs(pg - opt['projected_gradient_inf_norm']) <= 1e-11
    assert abs(initial - opt['initial_surrogate_loss']) <= 1e-11
    assert abs(final - opt['fitted_surrogate_loss']) <= 1e-11
    assert final <= initial + 1e-10
    calibrated = theta.copy()
    calibrated[-1] = float.fromhex(head['bias_hex'])
    loss_after, _ = evaluate(calibrated)
    assert abs(loss_after - opt['calibrated_bias_surrogate_loss']) <= 1e-11
    return dict(informative_holds=len(rows), fitted_surrogate_loss=final,
                projected_gradient_inf_norm=pg)


def validate(fold):
    # Import runner before its filesystem audit barrier, and use only its IO
    # and authorization helpers. No core fitting/calibration/gate calls.
    sys.path.insert(0, str(ROOT / 'programs'))
    import run_rc_h593_s_bias_competition_v1 as runner
    a = runner.guard('verify', fold)
    read, checked, bind = runner.read, runner.checked, runner.bind
    p_path = OUT / f'fold{fold}/payload.json'
    p = read(p_path)
    assert p['status'] == 'S_BIAS_COMPETITION_OUTER_SEALED'
    assert p['authority'] == bind(AUTH) and p['fold'] == fold
    assert a['code_sources']['independent_validator'] == bind(__file__)
    assert p['heldout_label_reads'] == 0
    sources = a['fold_sources'][str(fold)]
    parent = read(checked(sources['parent_payload']))
    assert p['parent_payload'] == sources['parent_payload']
    pv = read(checked(sources['parent_validation']))
    pi = read(checked(sources['parent_independent_validation']))
    assert pv['payload'] == pi['payload'] == sources['parent_payload']
    assert pv['status'] == 'S_NET_NESTED_FRESH_REPLAY_NUMPY_PASS'
    assert pi['status'] == 'S_NET_INDEPENDENT_NESTED_EXHAUSTIVE_OPTIMALITY_PASS'
    assert pi['passed'] and pi['heldout_label_reads'] == 0
    assert parent['heldout_label_reads'] == 0 and parent['fold'] == fold
    assert parent['authority'] == pv['authority'] == pi['authority']
    assert p['gallery_mapping_sha256'] == parent['gallery_mapping_sha256']
    splits = read(checked(a['public_sources']['split']))['folds']
    train = set(splits[fold]['train_query_ids'])
    held = set(splits[fold]['heldout_query_ids'])
    assert train.isdisjoint(held)
    assert set(p['train_query_ids']) == set(parent['train_query_ids']) == train
    roles = {r['query_id']: r for r in read(checked(sources['train_roles']))['records']}
    assert len(roles) == len(train) and set(roles) == train
    seen = set()
    for inner in parent['inner_heads']:
        inner_held = set(inner['heldout_query_ids'])
        inner_train = set(inner['train_query_ids'])
        assert inner['inner_fold'] != fold
        assert inner_held == set(splits[inner['inner_fold']]['heldout_query_ids'])
        assert inner_train == train - inner_held
        assert inner_train.isdisjoint(held) and not seen.intersection(inner_held)
        assert set(inner['effective_train_query_ids']) <= inner_train
        for key in ('identity', 'component'):
            assert {roles[q][key] for q in inner_train}.isdisjoint(
                {roles[q][key] for q in inner_held})
        seen.update(inner_held)
    assert seen == train
    assert len(p['calibration']) == len(parent['calibration']) == len(train)
    assert {r['query_id'] for r in p['calibration']} == train
    for row, old in zip(p['calibration'], parent['calibration']):
        assert all(row[k] == value for k, value in old.items())
        z = logits(old)
        j, d = top_and_gap(z)
        assert j == row['original_top']
        assert float(row['m']).hex() == z[j].hex()
        assert float(row['d']).hex() == d.hex() and d >= 0.0
        assert finite(row['h_s']) >= 0.0
        assert row['delta'] == int(row['top_correct']) - int(row['raw_correct'])
    assert set(p['parameters']) == set(ARMS)
    fits = {}
    for name in ARMS:
        head = p['parameters'][name]
        alpha, beta, bias = [finite(float.fromhex(head[k + '_hex']))
                             for k in ('alpha', 'beta', 'bias')]
        assert alpha >= 0.0 and beta >= 0.0
        for k, value in zip(('alpha', 'beta', 'bias'), (alpha, beta, bias)):
            assert float(head[k]).hex() == value.hex()
        if name == 'S_FIXED_BIAS2':
            assert head['alpha_hex'] == parent['parameters']['S_NET1']['alpha_hex']
            assert beta == 0.0
            assert head['fixed_s_source'] == dict(model='S_NET1', alpha_hex=alpha.hex(),
                                                parent_payload=sources['parent_payload'])
            optimization = dict(status='FROZEN_PARENT_S_NET1_SLOPE')
        else:
            optimization = independent_surrogate(p['calibration'], name, head)
        fit = independent_bias(p['calibration'], alpha, beta)
        assert fit['bias_hex'] == head['bias_hex'] and fit['disabled'] == head['disabled']
        for key, value in fit['counts'].items():
            assert head['training_' + key] == value, (name, key)
        cert = head['calibration_certificate']
        assert cert['status'] == 'EXACT_FINITE_BINARY64_BIAS_NET_GAIN_OPTIMUM_FIXED_SLOPES'
        assert cert['certificates'] == fit['certificates']
        digest = hashlib.sha256(json.dumps(fit['certificates'], sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        assert cert['breakpoints_sha256'] == digest
        assert cert['selected_counts'] == fit['counts']
        fits[name] = dict(alpha_hex=alpha.hex(), beta_hex=beta.hex(), bias_hex=bias.hex(),
                          disabled=fit['disabled'], **fit['counts'],
                          independent_bias_candidates=fit['distinct_tested_biases'],
                          optimization=optimization)
    assert len(p['predictions']) == len(parent['predictions']) == len(held)
    assert {r['query_id'] for r in p['predictions']} == held
    originals = {r['query_id']: r for r in parent['predictions']}
    checked_scores = 0
    old_switches = 0
    for row in p['predictions']:
        old = originals[row['query_id']]
        for key, value in old.items():
            if key != 'models':
                assert row[key] == value
        for name, value in old['models'].items():
            assert row['models'][name] == value
        assert set(row['models']) == set(old['models']) | set(ARMS)
        z = logits(old['models']['COST1_FULL'])
        top, d = top_and_gap(z)
        assert row['original_top'] == top
        if 'd' in row:
            assert float(row['d']).hex() == d.hex()
        assert len(row['candidate_physical_rows']) == 128
        assert len(row['challenger_positions']) == 127
        assert set(row['challenger_positions']) | {row['winner']} == set(range(128))
        old_switches += z[top] > 0.0
        for name, head in p['parameters'].items():
            expected = list(z)
            if z[top] <= 0.0 and not head['disabled']:
                gate_row = dict(m=z[top], h_s=row['h_s'], d=d)
                score = finite(offset(gate_row, float.fromhex(head['alpha_hex']),
                                      float.fromhex(head['beta_hex'])) + float.fromhex(head['bias_hex']))
                if score > 0.0:
                    expected[top] = score
            actual = row['models'][name]
            assert actual['logits_hex'] == [v.hex() for v in expected], (row['query_id'], name)
            assert max(range(127), key=expected.__getitem__) == top
            position = row['challenger_positions'][top] if expected[top] > 0 else row['winner']
            assert row['candidate_physical_rows'][position] == actual['selected']
            checked_scores += 127
    output = dict(status='S_BIAS_COMPETITION_INDEPENDENT_PASS', passed=True,
                  authority=bind(AUTH), payload=bind(p_path), fold=fold,
                  parent_payload=sources['parent_payload'], calibration_queries=len(train),
                  heldout_queries=len(held), inner_group_disjointness_pass=True,
                  new_model_score_checks=checked_scores, original_switches=old_switches,
                  original_switches_and_rank_locked=True, heldout_label_reads=0, fits=fits,
                  empirical_net_optimality_scope='bias only at frozen slopes; no joint optimum claim')
    runner.write(OUT / f'fold{fold}/independent_validation.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    validate(parser.parse_args().fold)
