#!/usr/bin/env python3
"""Independent NumPy validation of a three-parameter, target-free action.

No producer/core imports, data loaders, label joins, optimizer or natural fits.
Public APIs:
  independent_action_features(z127, x127x6) -> (top_index, [gap, raw_gap])
  independent_scales(features[N,2]) -> two RMS values (zero becomes one)
  independent_loss_gradient(theta3, features, delta, scales) -> loss, gradient
  independent_action_scores(features, head) -> N scores
  validate_training(features, delta, head) -> numerical validation report
  validate_actions(logits, x, head, winner_positions, challenger_positions,
                   candidate_physical_rows, saved_scores_hex, saved_selected,
                   saved_top_indices=None, saved_features_hex=None) -> report

Head fields are theta_hex[3], scale_hex[2]; optional theta/scale float copies
must agree exactly. The score is explicitly ((d/s_d)*a + (r/s_r)*c) + b.
Every complete calibration row enters RMS and the loss denominator. Only
delta=+/-1 enters the logistic numerator; delta=0 remains neutral. The L2
term is .001/2 times the full three-coordinate squared norm, including bias.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


L2 = 0.001
TOLERANCE = 2e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _features(features):
    value = np.asarray(features, dtype=np.float64)
    _require(value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == 2,
             'features must have nonempty shape [N,2]')
    _require(np.isfinite(value).all(), 'features must be finite')
    _require(np.all(value[:, 0] >= 0.0), 'rank gap must be nonnegative')
    _require(np.all(value[:, 1] <= 0.0), 'selected challenger RAW gap must be nonpositive')
    return value


def _theta(theta):
    value = np.asarray(theta, dtype=np.float64)
    _require(value.shape == (3,) and np.isfinite(value).all(),
             'theta must contain three finite weights')
    return value


def _scales(scales):
    value = np.asarray(scales, dtype=np.float64)
    _require(value.shape == (2,) and np.isfinite(value).all()
             and np.all(value > 0.0), 'scales must contain two positive finite values')
    return value


def _delta(delta, count):
    value = np.asarray(delta)
    _require(value.shape == (count,) and np.issubdtype(value.dtype, np.number)
             and not np.issubdtype(value.dtype, np.bool_), 'delta must be a numeric vector of length N')
    _require(np.isfinite(value).all() and np.isin(value, (-1, 0, 1)).all(),
             'delta must belong to {-1,0,+1}')
    return value.astype(np.float64)


def _hex_vector(values, count, name):
    _require(isinstance(values, (tuple, list)) and len(values) == count
             and all(isinstance(v, str) for v in values), f'{name} must contain {count} hexadecimal strings')
    try:
        result = np.array([float.fromhex(v) for v in values], dtype=np.float64)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'invalid hexadecimal {name}') from error
    _require(np.isfinite(result).all(), f'{name} must be finite')
    return result


def _head(head):
    _require(isinstance(head, dict), 'head must be a mapping')
    theta = _theta(_hex_vector(head.get('theta_hex'), 3, 'theta_hex'))
    scales = _scales(_hex_vector(head.get('scale_hex'), 2, 'scale_hex'))
    for key, expected in (('theta', theta), ('scale', scales)):
        if key in head:
            actual = np.asarray(head[key], dtype=np.float64)
            _require(actual.shape == expected.shape and np.array_equal(actual, expected),
                     f'{key} numeric copy disagrees with hexadecimal values')
    return theta, scales


def independent_action_features(logits, x):
    """First-maximum challenger, its gap to runner-up, and its OWN RAW gap."""
    logits = np.asarray(logits, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _require(logits.shape == (127,) and np.isfinite(logits).all(), 'expected 127 finite rank logits')
    _require(x.shape == (127, 6) and np.isfinite(x).all(), 'expected finite [127,6] original features')
    _require(np.all(x[:, 0] <= 0.0), 'all candidate RAW gaps must be nonpositive')
    # Sorting supplies an independent implementation of the first-max rule,
    # including the second maximum when multiple candidates share the maximum.
    order = sorted(range(127), key=lambda j: (-float(logits[j]), j))
    top = order[0]
    gap = float(logits[top]) - float(logits[order[1]])
    result = np.array([gap, float(x[top, 0])], dtype=np.float64)
    _features(result[None, :])
    return top, result


def independent_scales(features):
    """RMS over ALL rows including neutral rows; no centering or clipping."""
    features = _features(features)
    with np.errstate(over='ignore', invalid='ignore'):
        squared = features * features
        scales = np.sqrt(np.mean(squared, axis=0, dtype=np.float64))
    _require(np.isfinite(squared).all() and np.isfinite(scales).all(), 'RMS computation overflowed')
    scales[scales == 0.0] = 1.0
    return _scales(scales)


def _scores(theta, features, scales):
    theta, features, scales = _theta(theta), _features(features), _scales(scales)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = features / scales
        # Do not use matmul/FMA: the positive-zero boundary is part of the rule.
        first = normalized[:, 0] * theta[0]
        second = normalized[:, 1] * theta[1]
        scores = (first + second) + theta[2]
    _require(np.isfinite(normalized).all() and np.isfinite(scores).all(), 'nonfinite action score')
    return scores, normalized


def independent_action_scores(features, head):
    theta, scales = _head(head)
    return _scores(theta, features, scales)[0]


def independent_loss_gradient(theta, features, delta, scales):
    """ALL-N logistic numerator plus L2, with an analytic three-weight gradient."""
    theta = _theta(theta)
    scores, normalized = _scores(theta, features, scales)
    delta = _delta(delta, len(scores))
    active = delta != 0.0
    arguments = -delta[active] * scores[active]
    # Stable sigmoid; numpy.where would evaluate both overflow-prone branches.
    probabilities = np.empty_like(arguments)
    nonnegative = arguments >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-arguments[nonnegative]))
    exponentials = np.exp(arguments[~nonnegative])
    probabilities[~nonnegative] = exponentials / (1.0 + exponentials)
    residual = -delta[active] * probabilities
    design = np.column_stack((normalized[active], np.ones(int(active.sum()), dtype=np.float64)))
    with np.errstate(over='ignore', invalid='ignore'):
        loss = float(np.logaddexp(0.0, arguments).sum(dtype=np.float64) / len(scores)
                     + (L2 / 2.0) * np.sum(theta * theta, dtype=np.float64))
        gradient = np.sum(design * residual[:, None], axis=0, dtype=np.float64) / len(scores) + L2 * theta
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'nonfinite logistic loss or gradient')
    return loss, gradient


def validate_training(features, delta, head):
    """Replay TRAIN-derived RMS and final objective, without fitting a model."""
    theta, scales = _head(head)
    expected = independent_scales(features)
    _require(np.array_equal(expected, scales), 'saved RMS differs from all-row TRAIN RMS')
    value, gradient = independent_loss_gradient(theta, features, delta, scales)
    diagnostics = head.get('diagnostics', {})
    if 'fitted_loss' in diagnostics:
        _require(abs(float(diagnostics['fitted_loss']) - value) < TOLERANCE, 'saved fitted loss differs')
    if 'gradient_hex' in diagnostics:
        saved_gradient = _hex_vector(diagnostics['gradient_hex'], 3, 'gradient_hex')
        _require(np.max(abs(saved_gradient - gradient)) < TOLERANCE, 'saved gradient differs')
    delta = _delta(delta, len(features))
    if not np.any(delta != 0):
        _require(np.array_equal(theta, np.zeros(3)), 'all-neutral fallback must use zero parameters')
    return dict(status='RANK_ACTION_TRAIN_NUMPY_PASS', calibration_rows=len(features),
                positive_rows=int(np.sum(delta == 1)), negative_rows=int(np.sum(delta == -1)),
                neutral_rows=int(np.sum(delta == 0)), all_row_RMS_exact=True,
                logistic_denominator=len(features), objective=value,
                gradient_hex=[float(v).hex() for v in gradient],
                gradient_inf_norm=float(np.max(abs(gradient))), natural_fits=0)


def _integer_array(value, shape, name):
    array = np.asarray(value)
    _require(array.shape == shape and np.issubdtype(array.dtype, np.integer)
             and not np.issubdtype(array.dtype, np.bool_), f'{name} must have integer shape {shape}')
    return array


def validate_actions(logits, x, head, winner_positions, challenger_positions,
                     candidate_physical_rows, saved_scores_hex, saved_selected,
                     saved_top_indices=None, saved_features_hex=None):
    """Rebuild features and actual RAW/top choice; no target identity is read.

    logits: [N,127], x: [N,127,6], winner_positions: [N] positions on C128,
    challenger_positions: [N,127] mappings into candidate_physical_rows[N,128].
    saved_selected contains the selected physical gallery row, not a position.
    saved_scores_hex is one hexadecimal action score per query.
    """
    logits, x = np.asarray(logits, dtype=np.float64), np.asarray(x, dtype=np.float64)
    _require(logits.ndim == 2 and len(logits) > 0 and logits.shape[1] == 127,
             'logits must have nonempty shape [N,127]')
    count = len(logits)
    _require(x.shape == (count, 127, 6), 'features must have shape [N,127,6]')
    winners = _integer_array(winner_positions, (count,), 'winner_positions')
    challengers = _integer_array(challenger_positions, (count, 127), 'challenger_positions')
    physical = _integer_array(candidate_physical_rows, (count, 128), 'candidate_physical_rows')
    selected = _integer_array(saved_selected, (count,), 'saved_selected')
    _require(np.all((winners >= 0) & (winners < 128)), 'RAW winner position out of range')
    features, tops = [], []
    for row in range(count):
        _require(list(challengers[row]) == [j for j in range(128) if j != int(winners[row])],
                 'challenger axis must be the canonical C128 axis excluding RAW')
        _require(len(set(map(int, physical[row]))) == 128, 'candidate physical axis must be unique')
        top, feature = independent_action_features(logits[row], x[row])
        tops.append(top); features.append(feature)
    features, tops = np.asarray(features), np.asarray(tops, dtype=np.int64)
    scores = independent_action_scores(features, head)
    saved_scores = _hex_vector(saved_scores_hex, count, 'saved_scores_hex')
    error = float(np.max(abs(scores - saved_scores)))
    _require(error < TOLERANCE, 'independent action scores differ')
    _require(np.array_equal(scores > 0.0, saved_scores > 0.0), 'strict-positive action differs at zero boundary')
    if saved_top_indices is not None:
        saved_tops = _integer_array(saved_top_indices, (count,), 'saved_top_indices')
        _require(np.array_equal(saved_tops, tops), 'saved first-maximum challenger differs')
    if saved_features_hex is not None:
        _require(isinstance(saved_features_hex, (list, tuple)) and len(saved_features_hex) == count,
                 'saved feature batch must match N')
        saved_features = np.array([_hex_vector(r, 2, 'feature_hex') for r in saved_features_hex])
        _require(np.array_equal(features, saved_features), 'saved gap/selected-RAW features differ')
    top_positions = challengers[np.arange(count), tops]
    chosen_positions = np.where(scores > 0.0, top_positions, winners)
    expected_selected = physical[np.arange(count), chosen_positions]
    _require(np.array_equal(expected_selected, selected), 'selected physical reference differs')
    return dict(status='RANK_ACTION_PREDICTIONS_NUMPY_PASS', query_count=count,
                challenger_score_count=int(logits.size), max_abs_action_score_error=error,
                switches=int(np.sum(scores > 0.0)), holds=int(np.sum(scores <= 0.0)),
                exact_zero_scores=int(np.sum(scores == 0.0)),
                top_indices=[int(v) for v in tops], selected_physical_rows=[int(v) for v in expected_selected],
                raw_or_top_only=True, target_identity_reads=0,
                candidate_tie_rule='first maximum on canonical challenger axis',
                action_rule='strict score > 0 selects top; otherwise RAW')


def self_test():
    rng = np.random.default_rng(20260921)
    features = np.column_stack((rng.uniform(.1, 3., 9), -rng.uniform(.1, 2., 9)))
    delta = np.array([1, -1, 0, 1, 0, -1, -1, 1, 0])
    theta = np.array([.35, -.2, -.45])
    scales = independent_scales(features)
    loss, gradient = independent_loss_gradient(theta, features, delta, scales)
    checks = 0
    numeric = np.zeros(3)
    step = 1e-5
    for j in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step; minus[j] -= step
        numeric[j] = (independent_loss_gradient(plus, features, delta, scales)[0]
                      - independent_loss_gradient(minus, features, delta, scales)[0]) / (2*step)
    maximum_error = float(np.max(abs(gradient - numeric)))
    _require(maximum_error < 2e-9, 'finite-difference action gradient differs')
    checks += 1
    active = delta != 0
    subloss, subgradient = independent_loss_gradient(theta, features[active], delta[active], scales)
    regularization = L2/2 * float(theta @ theta)
    ratio = float(active.mean())
    _require(abs((loss-regularization) - ratio*(subloss-regularization)) < 1e-14
             and np.max(abs((gradient-L2*theta) - ratio*(subgradient-L2*theta))) < 1e-14,
             'neutral rows must remain in logistic denominator')
    checks += 1
    changed = features.copy(); changed[~active, 0] *= 100.; changed[~active, 1] *= 100.
    changed_loss, changed_gradient = independent_loss_gradient(theta, changed, delta, scales)
    _require(changed_loss == loss and np.array_equal(changed_gradient, gradient),
             'neutral rows changed numerator at fixed scales')
    _require(not np.array_equal(independent_scales(changed), scales), 'RMS must include neutral rows')
    checks += 1
    neutral_loss, neutral_gradient = independent_loss_gradient(theta, features, np.zeros(9), scales)
    _require(abs(neutral_loss-regularization) < 1e-15 and np.array_equal(neutral_gradient, L2*theta),
             'all-neutral data loss must vanish, leaving L2 including bias')
    checks += 1
    _require(np.array_equal(independent_scales(np.zeros((2,2))), np.ones(2)), 'zero-RMS fallback differs')
    manual_rms = independent_scales(np.array([[3.,-4.],[0.,0.]]))
    _require(np.array_equal(manual_rms, np.sqrt(np.array([4.5,8.]))), 'manual all-row RMS differs')
    checks += 1
    logits = np.zeros(127); logits[5] = logits[9] = 2.
    x = np.zeros((127,6)); x[:,0] = -2.; x[0,0] = -.001; x[5,0] = -.75
    top, pair = independent_action_features(logits, x)
    shifted_top, shifted_pair = independent_action_features(logits + 7., x)
    _require(top == shifted_top == 5 and np.array_equal(pair, [0.,-.75])
             and np.array_equal(pair, shifted_pair), 'tie/shift/selected challenger RAW gap differs')
    logits[5] = 3.
    _require(np.array_equal(independent_action_features(logits,x)[1], [1.,-.75]), 'runner-up gap differs')
    checks += 1
    def head(weights, rms):
        return dict(theta_hex=[float(v).hex() for v in weights], scale_hex=[float(v).hex() for v in rms])
    zero_head = head(np.zeros(3), scales)
    zero_head['diagnostics'] = dict(fitted_loss=0., gradient_hex=[0.0.hex()]*3)
    validated = validate_training(features, np.zeros(9), zero_head)
    _require(validated['neutral_rows'] == 9 and validated['objective'] == 0., 'all-neutral validation failed')
    checks += 1
    case_features = np.array([[1.,0.],[0.,-1.],[1.,-1.]])
    case_head = head([1.,1.,0.], [1.,1.])
    case_scores = independent_action_scores(case_features, case_head)
    _require(np.array_equal(case_scores, [1.,-1.,0.]), 'strict-zero score fixture differs')
    case_logits = np.zeros((3,127)); case_logits[:,0] = [1.,0.,1.]
    case_x = np.zeros((3,127,6)); case_x[:,0,0] = [0.,-1.,-1.]
    winner = np.array([0,64,127])
    challengers = np.array([[j for j in range(128) if j != w] for w in winner])
    physical = np.tile(np.arange(1000,1128), (3,1))
    expected = np.array([1001,1064,1127])
    saved_scores = [float(v).hex() for v in case_scores]
    report = validate_actions(case_logits,case_x,case_head,winner,challengers,physical,
                              saved_scores,expected,[0,0,0],
                              [[float(v).hex() for v in row] for row in case_features])
    _require(report['switches'] == 1 and report['holds'] == 2 and report['exact_zero_scores'] == 1,
             'strict-positive RAW/top selection failed')
    checks += 1
    corrupted = list(saved_scores); corrupted[2] = float(np.nextafter(0.,1.)).hex()
    try:
        validate_actions(case_logits,case_x,case_head,winner,challengers,physical,corrupted,expected)
    except ValueError:
        checks += 1
    else:
        raise AssertionError('action change hidden inside numeric tolerance accepted')
    bad_scale = head([0.,0.,0.],[1.,0.])
    bad_inputs = [lambda: independent_action_scores(features,bad_scale),
                  lambda: independent_action_features(logits[:126],x),
                  lambda: independent_loss_gradient(theta,features,np.full(9,2),scales),
                  lambda: independent_scales(np.array([[np.inf,-1.]])),
                  lambda: independent_action_features(np.full(127,np.nan),x)]
    for call in bad_inputs:
        try:
            call()
        except ValueError:
            checks += 1
        else:
            raise AssertionError('malformed or nonfinite input accepted')
    extreme = head([0.,0.,1000.], scales)
    extreme_loss, extreme_gradient = independent_loss_gradient([0.,0.,1000.],features,delta,scales)
    _require(np.isfinite(independent_action_scores(features,extreme)).all()
             and np.isfinite(extreme_loss) and np.isfinite(extreme_gradient).all(),
             'stable extreme-score arithmetic failed')
    checks += 1
    return dict(status='RANK_ACTION_INDEPENDENT_SYNTHETIC_PASS', checks=checks,
                max_finite_difference_gradient_error=maximum_error,
                no_core_imports=True, natural_data_reads=0, model_fits=0,
                denominator_all_rows=True, RMS_includes_neutral=True,
                strict_positive_boundary_verified=True, first_maximum_tie_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True), flush=True)
