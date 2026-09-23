#!/usr/bin/env python3
"""Independent three-parameter gap-curvature audit; RAW input is ignored.

Uses the frozen previous independent validator's endpoint enumeration and
scalar operations, never the fitting module. New 3D derivatives are evaluated
here, with exactly one nonnegative coordinate and a canonical positive-zero
RAW input for the primary gate.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import validate_rc_h593_conditional_gap_v1 as independent


def canonical_rows(records):
    return [dict(r, r_near=0.0) for r in records]


def alias_head(head):
    # The new term remains d²/scale. The old verifier identifies that feature
    # by its former mode name; no trained coefficient or score is modified.
    return dict(head, mode='GAP_CURVE4')


def surrogate_check(records, head):
    opt = head['optimization']
    assert opt['method'] == 'L-BFGS-B' and opt['success']
    assert opt['free_features'] == opt['design_features'] == ['d', 'phi', 'bias']
    assert opt['coefficient_order'] == ['beta', 'eta', 'bias']
    assert opt['bounds'] == [[0.0, None], [None, None], [None, None]]
    assert opt['l2'] == .001 and opt['maxiter'] == 2000
    assert opt['ftol'] == 1e-12 and opt['gtol'] == 1e-8
    assert opt['initialization'] == 'all zero'
    assert opt['joint_empirical_net_optimum_claimed'] is False
    theta = np.asarray([independent.finite(float.fromhex(x)) for x in opt['theta_hex']], dtype=np.float64)
    assert theta.shape == (3,) and theta[0] >= 0.0
    assert float(theta[0]).hex() == head['beta_hex']
    assert float(theta[1]).hex() == head['eta_hex']
    assert float(theta[2]).hex() == opt['surrogate_bias_hex']
    rows = [r for r in records if r['m'] <= 0.0 and r['delta'] != 0]
    assert len(rows) == opt['informative_hold_count']
    assert opt['neutral_holds_omitted_from_surrogate'] == sum(
        r['m'] <= 0.0 and r['delta'] == 0 for r in records)
    scale = float.fromhex(head['scale_hex'])
    x = np.asarray([[r['d'], independent.finite(float(r['d'] * r['d']) / scale), 1.0]
                    for r in rows], dtype=np.float64).reshape(len(rows), 3)
    m = np.asarray([r['m'] for r in rows], dtype=np.float64)
    y = np.asarray([r['delta'] for r in rows], dtype=np.float64)

    def evaluate(v):
        if not len(rows):
            return .0005 * float(v @ v), .001 * v
        yz = y * (m + x @ v)
        residual = -y * np.exp(-np.logaddexp(0.0, yz))
        return (float(np.logaddexp(0.0, -yz).mean() + .0005 * (v @ v)),
                x.T @ residual / len(rows) + .001 * v)

    initial, _ = evaluate(np.zeros(3))
    final, gradient = evaluate(theta)
    expected = np.asarray([float.fromhex(v) for v in opt['gradient_hex']])
    assert np.allclose(gradient, expected, rtol=0.0, atol=1e-11)
    projected = gradient.copy()
    if theta[0] <= 0.0 and projected[0] > 0.0:
        projected[0] = 0.0
    norm = float(np.max(np.abs(projected)))
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
    assert head['mode'] == 'GAP_CURVE3'
    for key in ('gamma', 'beta', 'eta', 'bias', 'scale'):
        assert independent.finite(head[key]).hex() == head[key + '_hex']
    assert head['gamma_hex'] == 0.0.hex() and head['frozen_gamma'] == 0.0
    assert head['ignored_raw_input'] is True
    assert head['beta'] >= 0.0 and head['scale'] > 0.0
    canonical = canonical_rows(records)
    scale, values = independent.scale_for(canonical, 'GAP_CURVE4')
    info = head['training_scale']
    assert head['scale_hex'] == scale.hex()
    assert info['mode'] == 'GAP_CURVE3' and info['count'] == len(values)
    assert info['scale_hex'] == scale.hex() and float(info['scale']).hex() == scale.hex()
    assert info['raw_phi_hex'] == [v.hex() for v in values]
    assert info['empty_or_allzero_fallback'] == (not values or all(v == 0.0 for v in values))
    assert info['centered'] is False and info['old_inputs_rescaled'] is False
    return dict(optimization=surrogate_check(canonical, head),
                calibration=independent.independent_bias(canonical, alias_head(head)),
                scale_hex=scale.hex())


def validate(fold):
    import run_rc_h593_gap_curve_v1 as runner
    a = runner.guard('verify', fold)
    read, checked, bind = runner.read, runner.checked, runner.bind
    assert a['code_sources']['independent_validator'] == bind(__file__)
    assert bind(independent.__file__) in a['code_sources'].values()
    parent = runner.parent_fold(a, fold)
    path = runner.OUT / f'fold{fold}/payload.json'
    p = read(path)
    assert p['status'] == 'GAP_CURVE_OUTER_SEALED'
    assert p['authority'] == bind(runner.AUTH) and p['fold'] == fold
    assert p['parent_payload'] == a['fold_sources'][str(fold)]['parent_payload']
    assert p['train_query_ids'] == parent['train_query_ids']
    assert p['heldout_label_reads'] == p['encoder_forwards'] == p['base_training_updates'] == 0
    assert p['gallery_mapping_sha256'] == parent['gallery_mapping_sha256']
    assert p['calibration'] == parent['calibration']
    assert set(r['query_id'] for r in p['calibration']) == set(p['train_query_ids'])
    assert len(p['calibration']) == len(set(p['train_query_ids']))
    for row in p['calibration']:
        z = independent.logits(row)
        top, gap = independent.top_gap(z)
        assert top == row['original_top'] and z[top].hex() == float(row['m']).hex()
        assert gap.hex() == float(row['d']).hex()
        assert row['delta'] == int(row['top_correct']) - int(row['raw_correct'])
    assert p['matched_control'] == parent['matched_control']
    assert p['structural_control'] == parent['parameters']['GAP_CURVE4']
    controls = dict(
        matched_control=independent.numerical_check(
            p['calibration'], independent.control_alias(p['matched_control'])),
        structural_control=independent.numerical_check(p['calibration'], p['structural_control']))
    assert set(p['parameters']) == set(runner.ARMS) == {'GAP_CURVE3'}
    head = p['parameters']['GAP_CURVE3']
    fit = numerical_check(p['calibration'], head)
    # Exactly the same normalized d² coordinate as the four-parameter control.
    assert head['scale_hex'] == p['structural_control']['scale_hex']
    for key in ('count', 'raw_phi_hex', 'empty_or_allzero_fallback', 'centered', 'old_inputs_rescaled'):
        assert head['training_scale'][key] == p['structural_control']['training_scale'][key]
    # These are nested convex surrogate families at the same feature scale and
    # regularizer. This check intentionally precedes the net-bias objective.
    loss2 = controls['matched_control']['optimization']['fitted_surrogate_loss']
    loss3 = fit['optimization']['fitted_surrogate_loss']
    loss4 = controls['structural_control']['optimization']['fitted_surrogate_loss']
    assert loss4 <= loss3 + 1e-6 and loss3 <= loss2 + 1e-6
    originals = {r['query_id']: r for r in parent['predictions']}
    assert len(p['predictions']) == len(originals)
    assert {r['query_id'] for r in p['predictions']} == set(originals)
    scores_checked = switches = 0
    for row in p['predictions']:
        old = originals[row['query_id']]
        assert all(row[k] == value for k, value in old.items() if k != 'models')
        assert all(row['models'][k] == value for k, value in old['models'].items())
        assert set(row['models']) == set(old['models']) | {'GAP_CURVE3'}
        assert len(row['candidate_physical_rows']) == 128
        assert len(row['challenger_positions']) == 127
        assert set(row['challenger_positions']) | {row['winner']} == set(range(128))
        z = independent.logits(old['models']['COST1_FULL'])
        top, gap = independent.top_gap(z)
        assert top == row['original_top'] and gap.hex() == float(row['d']).hex()
        switches += z[top] > 0.0
        gates = dict(GAP_CURVE3=alias_head(head),
                     GAP_BIAS2=independent.control_alias(p['matched_control']),
                     GAP_CURVE4=p['structural_control'])
        for name, gate in gates.items():
            expected = list(z)
            if z[top] <= 0.0 and not gate['disabled']:
                raw = 0.0 if name == 'GAP_CURVE3' else row['r_near']
                inputs = dict(m=z[top], r_near=raw, d=gap)
                score = independent.finite(independent.prebias(inputs, gate) +
                                           float.fromhex(gate['bias_hex']))
                if score > 0.0:
                    expected[top] = score
            assert [v.hex() for v in expected] == row['models'][name]['logits_hex']
            position = row['challenger_positions'][top] if expected[top] > 0.0 else row['winner']
            assert row['models'][name]['selected'] == row['candidate_physical_rows'][position]
            assert max(range(127), key=expected.__getitem__) == top
            scores_checked += 127
    report = dict(status='GAP_CURVE_INDEPENDENT_PASS', passed=True,
                  authority=bind(runner.AUTH), payload=bind(path), fold=fold,
                  calibration_queries=len(p['calibration']), heldout_queries=len(p['predictions']),
                  new_model_score_checks=scores_checked, original_switches=switches,
                  matched_GAP_BIAS2_exact=True, structural_GAP_CURVE4_exact=True,
                  scale_fitted_on_inner_holds_only=True, same_curve_coordinate_as_structural=True,
                  canonical_positive_zero_RAW_input=True, heldout_label_reads=0,
                  nested_surrogate_family_check=dict(GAP_BIAS2=loss2, GAP_CURVE3=loss3,
                                                     GAP_CURVE4=loss4, tolerance=1e-6),
                  fits={'GAP_CURVE3': fit}, controls=controls)
    runner.write(runner.OUT / f'fold{fold}/independent_validation.json', report)
    print('GAP_CURVE_INDEPENDENT_PASS', fold, scores_checked, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    validate(parser.parse_args().fold)
