#!/usr/bin/env python3
"""Independent NumPy quadratic challenger-ranking validation.

Original six coordinates are unchanged. DIAG12 adds their squares;
FULL_QUAD27 adds the 21 products x_i*x_j, i<=j in lexical order. Neither
basis includes a constant, centering, scaling, learned encoder or action.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

MODES = {'DIAG12': 12, 'FULL_QUAD27': 27}
PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def basis(x, mode):
    """Return an unscaled binary64 basis, preserving the last-axis order."""
    _require(mode in MODES, 'unknown quadratic rank basis')
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim >= 1 and x.shape[-1] == 6, 'last input dimension must contain six original coordinates')
    _require(np.isfinite(x).all(), 'nonfinite original feature')
    if mode == 'DIAG12':
        additions = x*x
    else:
        additions = np.stack([x[..., i]*x[..., j] for i, j in PAIRS], axis=-1)
    result = np.concatenate([x, additions], axis=-1)
    _require(result.shape[-1] == MODES[mode] and np.isfinite(result).all(), 'nonfinite or malformed quadratic basis')
    return result


def _inputs(theta, x, mode):
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim == 3 and x.shape[0] > 0 and x.shape[1:] == (127, 6),
             'x must have shape (positive N,127,6)')
    design = basis(x, mode)
    theta = np.asarray(theta, dtype=np.float64)
    _require(theta.shape == (MODES[mode],) and np.isfinite(theta).all(),
             'weight count must match the selected basis, without bias')
    return design, theta


def independent_loss_gradient(theta, x, y, mode):
    """Conditional 127-way CE / ALL present-query N and its derivative.

    RAW-correct y=-1 rows contribute zero. No AdamW decay term is added to
    the reported gradient: decoupled weight decay is an optimizer operation.
    """
    design, theta = _inputs(theta, x, mode)
    y = np.asarray(y)
    _require(y.shape == (len(design),) and np.issubdtype(y.dtype, np.integer)
             and not np.issubdtype(y.dtype, np.bool_), 'y must be an integer vector of length N')
    _require(np.all((y >= -1) & (y < 127)), 'target must be RAW(-1) or one of 127 challenger indices')
    active = y >= 0
    if not np.any(active):
        return 0.0, np.zeros(len(theta), dtype=np.float64)
    features, targets = design[active], y[active]
    scores = np.sum(features*theta[None, None, :], axis=2, dtype=np.float64)
    _require(np.isfinite(scores).all(), 'nonfinite rank score')
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    mass = np.sum(exponentials, axis=1, dtype=np.float64)
    losses = np.log(mass) - shifted[np.arange(len(targets)), targets]
    loss = float(np.sum(losses, dtype=np.float64)/len(design))
    residual = exponentials/mass[:, None]
    residual[np.arange(len(targets)), targets] -= 1.0
    gradient = np.sum(residual[:, :, None]*features, axis=(0, 1), dtype=np.float64)/len(design)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'nonfinite conditional objective')
    return loss, gradient


def validate_logits(x, theta_hex, logits_hex_batch, mode):
    """Independently replay all scores and first-maximum candidate indices."""
    _require(mode in MODES, 'unknown quadratic rank mode')
    _require(isinstance(theta_hex, (list, tuple)) and len(theta_hex) == MODES[mode]
             and all(isinstance(v, str) for v in theta_hex), 'invalid hexadecimal parameter vector')
    try:
        theta = [float.fromhex(v) for v in theta_hex]
    except ValueError as error:
        raise ValueError('invalid hexadecimal weight') from error
    design, theta = _inputs(theta, x, mode)
    _require(isinstance(logits_hex_batch, (list, tuple)) and len(logits_hex_batch) == len(design),
             'saved score batch must match N')
    saved = []
    for row in logits_hex_batch:
        _require(isinstance(row, (list, tuple)) and len(row) == 127
                 and all(isinstance(v, str) for v in row), 'expected 127 hexadecimal scores per query')
        try:
            saved.append([float.fromhex(v) for v in row])
        except ValueError as error:
            raise ValueError('invalid hexadecimal score') from error
    saved = np.asarray(saved, dtype=np.float64)
    replay = np.sum(design*theta[None, None, :], axis=2, dtype=np.float64)
    _require(np.isfinite(saved).all() and np.isfinite(replay).all(), 'nonfinite saved or replayed scores')
    error = float(np.max(np.abs(saved-replay)))
    _require(error < 2e-10, 'independent rank score replay exceeds absolute tolerance')
    tops = np.argmax(replay, axis=1)
    _require(np.array_equal(tops, np.argmax(saved, axis=1)), 'independent top candidate differs')
    ties = np.sum(np.sum(replay == np.max(replay, axis=1, keepdims=True), axis=1) > 1)
    return dict(status='QUADRATIC_RANK_LOGITS_INDEPENDENT_PASS', mode=mode,
                query_count=len(design), feature_dimension=MODES[mode], logit_count=int(replay.size),
                max_abs_error=error, top_indices=[int(v) for v in tops],
                exact_top_tie_queries=int(ties), rank_only=True, action_decisions_generated=False,
                candidate_tie_rule='first maximum in frozen challenger axis')


def self_test():
    checks = 0
    point = np.array([1., -2., 3., -4., 5., -6.])
    diag_expected = np.array([1., -2., 3., -4., 5., -6., 1., 4., 9., 16., 25., 36.])
    full_expected = np.array([1., -2., 3., -4., 5., -6.,
                              1., -2., 3., -4., 5., -6.,
                              4., -6., 8., -10., 12.,
                              9., -12., 15., -18.,
                              16., -20., 24., 25., -30., 36.])
    _require(np.array_equal(basis(point, 'DIAG12'), diag_expected), 'manual diagonal basis mismatch')
    _require(np.array_equal(basis(point, 'FULL_QUAD27'), full_expected), 'manual product order mismatch')
    checks += 2
    for mode in MODES:
        _require(np.array_equal(basis(np.zeros(6), mode), np.zeros(MODES[mode])), 'basis contains an intercept')
        checks += 1
    rng = np.random.default_rng(20260921)
    x = rng.normal(scale=.4, size=(7, 127, 6))
    y = np.array([-1, 4, 60, -1, 3, 126, 0], dtype=np.int64)
    max_gradient_difference = 0.0

    def score_loss(scores):
        active = y >= 0
        scores = scores[active]
        shifted = scores - scores.max(axis=1, keepdims=True)
        return float(np.sum(np.log(np.exp(shifted).sum(1))-
                            shifted[np.arange(active.sum()), y[active]])/len(y))

    for mode, dimension in MODES.items():
        theta = rng.normal(scale=.2, size=dimension)
        loss, gradient = independent_loss_gradient(theta, x, y, mode)
        numeric = np.zeros(dimension)
        for j in range(dimension):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numeric[j] = (independent_loss_gradient(plus, x, y, mode)[0]-
                          independent_loss_gradient(minus, x, y, mode)[0])/2e-5
        error = float(np.max(np.abs(gradient-numeric)))
        max_gradient_difference = max(max_gradient_difference, error)
        _require(error < 2e-9, 'quadratic gradient central difference mismatch')
        checks += 1
        altered = x.copy()
        altered[y == -1] = rng.normal(size=altered[y == -1].shape)*1e4
        value, derivative = independent_loss_gradient(theta, altered, y, mode)
        _require(value == loss and np.array_equal(derivative, gradient), 'RAW-correct row changed loss or gradient')
        checks += 1
        active = y >= 0
        smaller, derivative = independent_loss_gradient(theta, x[active], y[active], mode)
        fraction = int(active.sum())/len(y)
        _require(abs(loss-smaller*fraction) < 1e-14 and np.max(abs(gradient-derivative*fraction)) < 1e-14,
                 'conditional objective uses wrong denominator')
        checks += 1
        zero_loss, zero_gradient = independent_loss_gradient(theta, x, np.full(len(x), -1, dtype=np.int64), mode)
        _require(zero_loss == 0.0 and np.array_equal(zero_gradient, np.zeros(dimension)), 'all-RAW loss is not zero')
        checks += 1
        scores = np.sum(basis(x, mode)*theta, axis=2)
        shifted_scores = scores + rng.normal(scale=25, size=(len(x), 1))
        _require(abs(score_loss(shifted_scores)-loss) < 1e-13, 'common score shift changed conditional CE')
        _require(np.array_equal(scores.argmax(1), shifted_scores.argmax(1)), 'common score shift changed top')
        checks += 1
        # Raw-coordinate translation is NOT invariant at fixed quadratic
        # weights. With the algebraically adjusted linear coefficients, it
        # represents the same score up to one constant, as tested here.
        translation = rng.normal(scale=.2, size=6)
        translated_theta = theta.copy()
        pairs = tuple((j, j) for j in range(6)) if mode == 'DIAG12' else PAIRS
        for weight, (i, j) in zip(theta[6:], pairs):
            translated_theta[i] -= weight*translation[j]
            translated_theta[j] -= weight*translation[i]
        transformed = np.sum(basis(x+translation, mode)*translated_theta, axis=2)
        difference = transformed-scores
        _require(np.max(np.abs(difference-difference[0, 0])) < 1e-13,
                 'translated quadratic function family is inconsistent')
        _require(abs(score_loss(transformed)-loss) < 1e-13, 'reparameterized coordinate translation changed CE')
        checks += 1
        encoded = [[float(v).hex() for v in row] for row in scores]
        replay = validate_logits(x, [float(v).hex() for v in theta], encoded, mode)
        _require(replay['max_abs_error'] == 0.0 and replay['logit_count'] == len(x)*127, 'quadratic score replay')
        checks += 1
        corrupt = [list(row) for row in encoded]
        corrupt[0][0] = (float.fromhex(corrupt[0][0])+1e-5).hex()
        try:
            validate_logits(x, [float(v).hex() for v in theta], corrupt, mode)
        except ValueError:
            checks += 1
        else:
            raise AssertionError('corrupt quadratic logit accepted')
        tie_scores = [[0.0.hex()]*127]
        tie_scores[0][1] = (1e-12).hex()
        try:
            validate_logits(np.zeros((1, 127, 6)), [0.0.hex()]*dimension, tie_scores, mode)
        except ValueError:
            checks += 1
        else:
            raise AssertionError('changed argmax hidden inside score tolerance was accepted')
    # Embed the diagonal basis into the full quadratic family exactly; mixed
    # product weights are zero. This is function-class nesting, not a claim
    # about finite-iteration AdamW minima or held-out performance ordering.
    diagonal_theta = rng.normal(scale=.2, size=12)
    full_theta = np.zeros(27)
    full_theta[:6] = diagonal_theta[:6]
    positions = list(range(6))
    for j in range(6):
        index = 6+PAIRS.index((j, j))
        full_theta[index] = diagonal_theta[6+j]
        positions.append(index)
    diagonal_loss, diagonal_gradient = independent_loss_gradient(diagonal_theta, x, y, 'DIAG12')
    full_loss, full_gradient = independent_loss_gradient(full_theta, x, y, 'FULL_QUAD27')
    _require(abs(diagonal_loss-full_loss) < 1e-14
             and np.max(abs(diagonal_gradient-full_gradient[positions])) < 1e-13,
             'diagonal-in-full loss or restricted derivative mismatch')
    scores12 = np.sum(basis(x, 'DIAG12')*diagonal_theta, axis=2)
    scores27 = np.sum(basis(x, 'FULL_QUAD27')*full_theta, axis=2)
    _require(np.max(abs(scores12-scores27)) < 1e-14, 'diagonal-in-full score mismatch')
    checks += 1
    return dict(status='QUADRATIC_RANK_INDEPENDENT_SYNTHETIC_PASS', checks=checks,
                natural_data_reads=0, model_fits=0, max_gradient_difference=max_gradient_difference,
                original_six_features_preserved=True, all_present_query_denominator_verified=True,
                diagonal_family_nested_in_full=True, original_input_translation_requires_reparameterization=True,
                no_HOLD_SWITCH_action_generated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True), flush=True)
