#!/usr/bin/env python3
"""Independent strongest-wrong conditional ranking objective and score replay.

Dependency: frozen programs/validate_rc_h593_conditional_rank_v1.py, used only
for its independent six-dimensional logit/argmax replay. No trainer or core
objective is imported. The output is challenger ranking, never HOLD/SWITCH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validate_rc_h593_conditional_rank_v1 import validate_logits as _linear_logits

LOGIT_VALIDATOR_DEPENDENCY = str(Path(__file__).resolve().with_name('validate_rc_h593_conditional_rank_v1.py'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_loss_gradient(theta, x, y):
    """Mean-over-ALL-N softplus(max wrong score - target score), and gradient.

    RAW-correct rows y=-1 contribute zero. At exact wrong-score ties, this
    reports the symmetric subgradient that equally averages all tied wrong
    features (the semantics of torch.amax, not indexed torch.max).
    """
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y)
    _require(x.ndim == 3 and x.shape[0] > 0 and x.shape[1:] == (127, 6)
             and np.isfinite(x).all(), 'x must be finite FP64 with shape (positive N,127,6)')
    _require(theta.shape == (6,) and np.isfinite(theta).all(), 'expected six finite weights without bias')
    _require(y.shape == (len(x),) and np.issubdtype(y.dtype, np.integer)
             and not np.issubdtype(y.dtype, np.bool_), 'y must be an integer vector of length N')
    _require(np.all((y >= -1) & (y < 127)), 'y must be RAW(-1) or a valid challenger index')
    active = y >= 0
    if not np.any(active):
        return 0.0, np.zeros(6, dtype=np.float64)
    features, targets = x[active], y[active]
    z = np.sum(features*theta[None, None, :], axis=2, dtype=np.float64)
    _require(np.isfinite(z).all(), 'nonfinite rank logit')
    target_scores = z[np.arange(len(targets)), targets]
    wrong_scores = z.copy()
    wrong_scores[np.arange(len(targets)), targets] = -np.inf
    maximum = np.max(wrong_scores, axis=1)
    margin = maximum-target_scores
    _require(np.isfinite(margin).all(), 'nonfinite strongest-wrong margin')
    tied = wrong_scores == maximum[:, None]
    counts = np.sum(tied, axis=1)
    _require(np.all(counts >= 1) and not np.any(tied[np.arange(len(targets)), targets]),
             'target was not excluded from strongest-wrong competition')
    wrong_mean = np.sum(features*tied[:, :, None], axis=1, dtype=np.float64)/counts[:, None]
    target_features = features[np.arange(len(targets)), targets]
    # sigmoid(margin), stable for both large positive and negative margins.
    probability = np.exp(-np.logaddexp(0.0, -margin))
    loss = float(np.sum(np.logaddexp(0.0, margin), dtype=np.float64)/len(x))
    gradient = np.sum(probability[:, None]*(wrong_mean-target_features), axis=0, dtype=np.float64)/len(x)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'nonfinite hard-rank loss or gradient')
    return loss, gradient


def validate_logits(x, theta_hex, logits_hex_batch):
    report = _linear_logits(x, theta_hex, logits_hex_batch)
    report['status'] = 'HARD_RANK_LOGITS_INDEPENDENT_PASS'
    return report


def self_test():
    rng = np.random.default_rng(20260921)
    x = rng.normal(size=(7, 127, 6))
    y = np.array([-1, 4, 60, -1, 3, 126, 0], dtype=np.int64)
    theta = np.array([.2, -.1, .4, -.3, .6, .15], dtype=np.float64)
    checks = 0
    loss, gradient = independent_loss_gradient(theta, x, y)
    # Finite differences are evaluated away from max ties, where the gradient
    # is unique. Tie behavior is checked by explicit formulas below.
    active = y >= 0
    scores = np.sum(x[active]*theta, axis=2)
    scores[np.arange(active.sum()), y[active]] = -np.inf
    sorted_scores = np.sort(scores, axis=1)
    _require(np.min(sorted_scores[:, -1]-sorted_scores[:, -2]) > 1e-3, 'synthetic finite difference point is too close to a tie')
    numerical = np.zeros(6)
    for j in range(6):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numerical[j] = (independent_loss_gradient(plus, x, y)[0]-
                        independent_loss_gradient(minus, x, y)[0])/2e-5
    maximum_error = float(np.max(np.abs(numerical-gradient)))
    _require(maximum_error < 2e-9, 'hard-rank finite difference gradient mismatch')
    checks += 1
    zero_loss, zero_gradient = independent_loss_gradient(np.zeros(6), x, y)
    expected = np.zeros(6)
    for row, target in zip(x, y):
        if target >= 0:
            others = np.concatenate([row[:target], row[target+1:]], axis=0)
            expected += .5*(others.mean(axis=0)-row[target])/len(x)
    _require(abs(zero_loss-active.sum()*np.log(2)/len(x)) < 1e-14
             and np.max(abs(zero_gradient-expected)) < 1e-14,
             'zero initialization did not split gradient across all 126 wrong ties')
    checks += 1
    tied_x = np.zeros((1, 127, 6))
    tied_x[0, :, 0] = -5.
    tied_x[0, 0, :2] = [0., 3.]
    tied_x[0, 1, :2] = [1., 2.]
    tied_x[0, 2, :2] = [1., -4.]
    one_axis = np.array([1., 0., 0., 0., 0., 0.])
    value, derivative = independent_loss_gradient(one_axis, tied_x, np.array([0]),)
    expected = np.array([1., -4., 0., 0., 0., 0.])*np.exp(-np.logaddexp(0., -1.))
    _require(abs(value-np.logaddexp(0., 1.)) < 1e-14 and np.max(abs(derivative-expected)) < 1e-14,
             'nonzero two-way tie did not average tied wrong features')
    checks += 1
    # Target is the unique highest scorer; it must nevertheless be masked out
    # of max_wrong, producing negative margin rather than an artificial zero.
    tied_x[0, 0, :2] = [2., 10.]
    tied_x[0, 1, :2] = [1., 0.]
    tied_x[0, 2, :2] = [1., 2.]
    value, derivative = independent_loss_gradient(one_axis, tied_x, np.array([0]))
    expected = np.array([-1., -9., 0., 0., 0., 0.])*np.exp(-np.logaddexp(0., 1.))
    _require(abs(value-np.logaddexp(0., -1.)) < 1e-14 and np.max(abs(derivative-expected)) < 1e-14,
             'target incorrectly participates in strongest-wrong max')
    checks += 1
    altered = x.copy()
    altered[~active] = rng.normal(size=altered[~active].shape)*1e4
    altered_loss, altered_gradient = independent_loss_gradient(theta, altered, y)
    _require(altered_loss == loss and np.array_equal(altered_gradient, gradient), 'RAW-correct rows contribute to objective')
    checks += 1
    active_loss, active_gradient = independent_loss_gradient(theta, x[active], y[active])
    fraction = int(active.sum())/len(x)
    _require(abs(loss-active_loss*fraction) < 1e-14
             and np.max(abs(gradient-active_gradient*fraction)) < 1e-14, 'denominator is not all present queries')
    checks += 1
    raw_loss, raw_gradient = independent_loss_gradient(theta, x, np.full(len(x), -1, dtype=np.int64))
    _require(raw_loss == 0.0 and np.array_equal(raw_gradient, np.zeros(6)), 'all-RAW objective must be zero')
    checks += 1
    shifted = x+rng.normal(size=(len(x), 1, 6))
    shift_loss, shift_gradient = independent_loss_gradient(theta, shifted, y)
    _require(abs(shift_loss-loss) < 1e-13 and np.max(abs(shift_gradient-gradient)) < 1e-13,
             'common candidate feature/score translation changed margin objective')
    checks += 1
    # Both sigmoid extremes must remain finite without exponent overflow.
    extreme = np.zeros((2, 127, 6))
    extreme[0, 1:, 0] = 1e3
    extreme[1, 0, 0] = 1e3
    value, derivative = independent_loss_gradient(one_axis, extreme, np.array([0, 0]))
    _require(np.isfinite(value) and np.isfinite(derivative).all()
             and abs(value-500.) < 1e-12 and abs(derivative[0]-500.) < 1e-12,
             'unstable sigmoid or softplus at extreme margins')
    checks += 1
    z = np.sum(x*theta, axis=2)
    report = validate_logits(x, [float(v).hex() for v in theta],
                             [[float(v).hex() for v in row] for row in z])
    _require(report['max_abs_error'] == 0.0 and report['logit_count'] == len(x)*127
             and report['action_decisions_generated'] is False, 'independent score replay failed')
    checks += 1
    return dict(status='HARD_RANK_INDEPENDENT_SYNTHETIC_PASS', checks=checks,
                natural_data_reads=0, model_fits=0, max_gradient_difference=maximum_error,
                uniform_exact_tie_subgradient_verified=True, target_excluded=True,
                all_present_query_denominator_verified=True, no_HOLD_SWITCH_action_generated=True,
                logit_validator_dependency=LOGIT_VALIDATOR_DEPENDENCY)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True), flush=True)
