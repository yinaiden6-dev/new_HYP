#!/usr/bin/env python3
"""Independent NumPy checks for a six-weight, challenger-only rank head.

The conditional 127-way loss contributes zero for RAW-correct queries, while
its denominator remains ALL recall-present training queries. This module
never applies a positive-logit action threshold or produces HOLD/SWITCH.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _features(x):
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim == 3 and x.shape[0] > 0 and x.shape[1:] == (127, 6),
             'x must have shape (positive N, 127, 6)')
    _require(np.isfinite(x).all(), 'nonfinite feature endpoint')
    return x


def _theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    _require(theta.shape == (6,) and np.isfinite(theta).all(),
             'theta must contain six finite FP64 weights and no bias')
    return theta


def independent_loss_gradient(theta, x, y):
    """Return conditional CE/N and its six-coordinate analytic derivative.

    y=-1 identifies a correct RAW incumbent. Otherwise y is the target's
    challenger index in 0..126. The derivative excludes AdamW's decoupled
    weight decay, which is an optimizer operation, not this loss's gradient.
    """
    x, theta = _features(x), _theta(theta)
    y = np.asarray(y)
    _require(y.shape == (len(x),) and np.issubdtype(y.dtype, np.integer)
             and not np.issubdtype(y.dtype, np.bool_), 'y must be an integer vector of length N')
    _require(np.all((y >= -1) & (y < 127)), 'y must be -1 or a challenger index in 0..126')
    active = y >= 0
    if not np.any(active):
        return 0.0, np.zeros(6, dtype=np.float64)
    xa, ya = x[active], y[active]
    z = np.sum(xa * theta[None, None, :], axis=2, dtype=np.float64)
    _require(np.isfinite(z).all(), 'nonfinite rank logit')
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    normalizers = np.sum(exp_shifted, axis=1, dtype=np.float64)
    # Subtract the shifted target score, avoiding cancellation of a common
    # large offset. Only active rows enter the numerator; denominator is N.
    losses = np.log(normalizers) - shifted[np.arange(len(ya)), ya]
    loss = float(np.sum(losses, dtype=np.float64) / len(x))
    residual = exp_shifted / normalizers[:, None]
    residual[np.arange(len(ya)), ya] -= 1.0
    gradient = np.sum(residual[:, :, None] * xa, axis=(0, 1), dtype=np.float64) / len(x)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'nonfinite conditional objective')
    return loss, gradient


def validate_logits(x, theta_hex, logits_hex_batch):
    """Recompute every challenger score and the first-maximum top index.

    Scores are compared at absolute tolerance strictly below 2e-10; candidate
    indices must agree exactly. There is deliberately no threshold at zero.
    """
    x = _features(x)
    _require(isinstance(theta_hex, (list, tuple)) and len(theta_hex) == 6,
             'expected six hexadecimal weights')
    _require(all(isinstance(v, str) for v in theta_hex), 'weight hex values must be strings')
    try:
        theta = _theta([float.fromhex(v) for v in theta_hex])
    except (ValueError, TypeError) as error:
        raise ValueError('invalid hexadecimal weight') from error
    _require(isinstance(logits_hex_batch, (list, tuple)) and len(logits_hex_batch) == len(x),
             'logit batch must match N')
    saved = []
    for row in logits_hex_batch:
        _require(isinstance(row, (list, tuple)) and len(row) == 127
                 and all(isinstance(v, str) for v in row), 'expected 127 hexadecimal scores per query')
        try:
            saved.append([float.fromhex(v) for v in row])
        except (ValueError, TypeError) as error:
            raise ValueError('invalid hexadecimal score') from error
    saved = np.asarray(saved, dtype=np.float64)
    _require(np.isfinite(saved).all(), 'nonfinite saved challenger logit')
    replay = np.sum(x * theta[None, None, :], axis=2, dtype=np.float64)
    _require(np.isfinite(replay).all(), 'nonfinite independent challenger logit')
    error = float(np.max(np.abs(replay - saved)))
    _require(error < 2e-10, 'independent score replay exceeds absolute tolerance')
    top = np.argmax(replay, axis=1)
    saved_top = np.argmax(saved, axis=1)
    _require(np.array_equal(top, saved_top), 'independent top challenger index differs')
    tied = np.sum(np.sum(replay == np.max(replay, axis=1, keepdims=True), axis=1) > 1)
    return dict(status='CONDITIONAL_RANK_LOGITS_INDEPENDENT_PASS',
                query_count=len(x), logit_count=int(replay.size),
                max_abs_error=error, top_indices=[int(v) for v in top],
                exact_top_tie_queries=int(tied), candidate_tie_rule='first maximum in frozen challenger axis',
                rank_only=True, action_decisions_generated=False)


def self_test():
    rng = np.random.default_rng(20260921)
    x = rng.normal(size=(7, 127, 6))
    y = np.array([-1, 4, 60, -1, 3, 126, 0], dtype=np.int64)
    theta = np.array([.2, -.1, .4, -.3, .6, .15], dtype=np.float64)
    checks = 0
    loss, gradient = independent_loss_gradient(theta, x, y)
    numeric = np.zeros(6)
    step = 1e-5
    for j in range(6):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        numeric[j] = (independent_loss_gradient(plus, x, y)[0] -
                      independent_loss_gradient(minus, x, y)[0]) / (2*step)
    _require(np.max(np.abs(numeric-gradient)) < 2e-9, 'central difference gradient mismatch')
    checks += 1
    # RAW-correct inputs do not enter either numerator or gradient.
    perturbed = x.copy()
    perturbed[y == -1] = rng.normal(size=perturbed[y == -1].shape) * 1e4
    changed_loss, changed_gradient = independent_loss_gradient(theta, perturbed, y)
    _require(loss == changed_loss and np.array_equal(gradient, changed_gradient), 'RAW-correct row affected ranking loss')
    checks += 1
    active = y >= 0
    reduced_loss, reduced_gradient = independent_loss_gradient(theta, x[active], y[active])
    factor = int(active.sum()) / len(y)
    _require(abs(loss - reduced_loss*factor) < 1e-14
             and np.max(np.abs(gradient - reduced_gradient*factor)) < 1e-14,
             'denominator is not all recall-present queries')
    checks += 1
    zero_loss, zero_gradient = independent_loss_gradient(theta, x, np.full(len(x), -1, dtype=np.int64))
    _require(zero_loss == 0.0 and np.array_equal(zero_gradient, np.zeros(6)), 'all RAW-correct loss must be zero')
    checks += 1
    # Add a query-dependent vector equally to every challenger: conditional
    # rank probabilities and derivatives are invariant to this common shift.
    shifted_x = x + rng.normal(size=(len(x), 1, 6))
    shift_loss, shift_gradient = independent_loss_gradient(theta, shifted_x, y)
    _require(abs(shift_loss-loss) < 1e-13 and np.max(np.abs(shift_gradient-gradient)) < 1e-13,
             'common challenger offset changed conditional rank objective')
    checks += 1
    # Full C128 CE decomposes into conditional rank and challenger-set-versus-HOLD
    # terms. RAW-correct queries contribute only to the latter.
    bias = -.7
    z = np.sum(x * theta[None, None, :], axis=2) + bias
    maximum = np.max(z, axis=1)
    log_partition = maximum + np.log(np.sum(np.exp(z-maximum[:, None]), axis=1))
    action = np.where(active, np.logaddexp(0.0, -log_partition), np.logaddexp(0.0, log_partition))
    full = np.logaddexp(0.0, log_partition)
    full[active] -= z[active, y[active]]
    _require(abs(float(full.mean()) - (loss + float(action.mean()))) < 1e-13,
             'full C128 CE decomposition mismatch')
    checks += 1
    within = np.exp(z-log_partition[:, None])
    expected_feature = np.sum(within[:, :, None]*x, axis=1)
    challenger_probability = np.exp(-np.logaddexp(0.0, -log_partition))
    full_gradient_rows = challenger_probability[:, None]*expected_feature
    full_gradient_rows[active] -= x[active, y[active]]
    action_gradient = np.mean((challenger_probability-active.astype(np.float64))[:, None]
                              * expected_feature, axis=0)
    _require(np.max(np.abs(full_gradient_rows.mean(axis=0)-gradient-action_gradient)) < 1e-13,
             'full C128 CE gradient decomposition mismatch')
    checks += 1
    replay = np.sum(x * theta[None, None, :], axis=2)
    encoded = [[float(v).hex() for v in row] for row in replay]
    report = validate_logits(x, [float(v).hex() for v in theta], encoded)
    _require(report['max_abs_error'] == 0.0 and report['logit_count'] == 7*127, 'logit exact replay')
    checks += 1
    bad_scores = [list(row) for row in encoded]
    bad_scores[0][0] = (float.fromhex(bad_scores[0][0]) + 1e-5).hex()
    try:
        validate_logits(x, [float(v).hex() for v in theta], bad_scores)
    except ValueError:
        checks += 1
    else:
        raise AssertionError('corrupted saved score was accepted')
    # Near-identical scores can satisfy the numeric tolerance but still choose
    # a different winner; this must fail the explicit argmax check.
    flat = np.zeros((1, 127, 6))
    tie_scores = [[0.0.hex()] * 127]
    tie_scores[0][1] = (1e-12).hex()
    try:
        validate_logits(flat, [0.0.hex()] * 6, tie_scores)
    except ValueError:
        checks += 1
    else:
        raise AssertionError('changed top candidate hidden inside tolerance was accepted')
    return dict(status='CONDITIONAL_RANK_INDEPENDENT_SYNTHETIC_PASS', checks=checks,
                natural_data_reads=0, model_fits=0, max_gradient_difference=float(np.max(np.abs(numeric-gradient))),
                full_CE_decomposition_verified=True, all_present_query_denominator_verified=True,
                no_HOLD_SWITCH_action_generated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True), flush=True)
