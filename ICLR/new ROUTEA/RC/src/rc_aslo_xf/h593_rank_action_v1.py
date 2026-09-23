"""A fixed three-parameter action recipe for frozen challenger ranking heads.

Features are the top1-minus-top2 gap over all 127 challenger logits and the
selected challenger's signed RAW gap from original X[top,0].  Each coordinate
is divided by its TRAIN calibration RMS, including neutral rows; no centering
or clipping is performed.  The action score is evaluated in explicit FP64
order as ((d/sd)*theta0 + (r/sr)*theta1) + theta2.  SWITCH iff score > 0.

The loss uses informative delta in {-1,+1}, divided by ALL calibration rows,
plus .001/2 times the squared norm of all three parameters, including bias.
Neutral delta=0 rows contribute zero to the numerator but remain in both the
denominator and RMS estimates.  Optimization is a single zero-initialized,
unconstrained L-BFGS-B run.  There is no empirical-net bias sweep or grid.

No files, datasets, model encoders, or evaluation labels are accessed here.
"""
import numpy as np
from scipy.optimize import minimize


CHALLENGERS = 127
ORIGINAL_DIMENSION = 6
PARAMETERS = 3
L2 = 0.001
MAXITER = 2000
FTOL = 1e-12
GTOL = 1e-8


def _features(features):
    try:
        values = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('features must be a finite FP64 [N,2] array') from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError('features must have shape [positive N,2]')
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0.0):
        raise ValueError('features must be finite and challenger gaps nonnegative')
    # The second feature is signed.  Its original RAW-margin sign must not
    # be removed or clipped when constructing or using the action design.
    return values


def _delta(delta, n):
    values = np.asarray(delta)
    if values.shape != (n,) or values.dtype.kind not in 'iuf':
        raise ValueError('delta must be a numeric vector of length N')
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, (-1., 0., 1.)).all():
        raise ValueError('delta values must be exactly -1, 0, or +1')
    return values


def _theta(theta):
    values = np.asarray(theta, dtype=np.float64)
    if values.shape != (PARAMETERS,) or not np.isfinite(values).all():
        raise ValueError('theta must contain exactly three finite parameters')
    return values


def _scales(scales):
    values = np.asarray(scales, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('scales must contain two finite strictly positive RMS values')
    return values


def action_features(z127, x127x6):
    """Return (first top index, FP64 [challenger gap, selected RAW gap])."""
    z = np.asarray(z127, dtype=np.float64)
    x = np.asarray(x127x6, dtype=np.float64)
    if z.shape != (CHALLENGERS,) or x.shape != (CHALLENGERS, ORIGINAL_DIMENSION):
        raise ValueError('expected 127 logits and original [127,6] feature rows')
    if not np.isfinite(z).all() or not np.isfinite(x).all():
        raise ValueError('rank logits and original features must be finite')
    top = int(np.argmax(z))
    second = max(float(v) for i, v in enumerate(z) if i != top)
    with np.errstate(over='raise', invalid='raise'):
        try:
            gap = np.float64(z[top]) - np.float64(second)
        except FloatingPointError as exc:
            raise ValueError('challenger gap overflowed FP64') from exc
    result = np.asarray([gap, x[top, 0]], dtype=np.float64)
    _features(result[None, :])
    return top, result


def fit_rms(features):
    """FP64 sqrt(mean(feature²)) on ALL rows; zero coordinates use one."""
    values = _features(features)
    with np.errstate(over='raise', invalid='raise'):
        try:
            scales = np.sqrt(np.mean(values * values, axis=0, dtype=np.float64))
        except FloatingPointError as exc:
            raise ValueError('RMS computation overflowed FP64') from exc
    scales = np.where(scales == 0.0, np.float64(1.0), scales)
    return _scales(scales)


def _normalized_features(features, scales):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            normalized = features / scales
        except FloatingPointError as exc:
            raise ValueError('normalized action features are not finite') from exc
    if not np.isfinite(normalized).all():
        raise ValueError('normalized action features are not finite')
    return normalized


def _scores(normalized, theta):
    with np.errstate(over='raise', invalid='raise'):
        try:
            result = (normalized[:, 0] * theta[0] + normalized[:, 1] * theta[1]) + theta[2]
        except FloatingPointError as exc:
            raise ValueError('action score overflowed FP64') from exc
    if not np.isfinite(result).all():
        raise ValueError('nonfinite action scores')
    return result


def _evaluate(theta, normalized, delta):
    scores = _scores(normalized, theta)
    informative = delta != 0.0
    design = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
    with np.errstate(over='raise', invalid='raise'):
        try:
            signed = delta[informative] * scores[informative]
            data_loss = float(np.logaddexp(0.0, -signed).sum(dtype=np.float64) / len(delta))
            residual = -delta[informative] * np.exp(-np.logaddexp(0.0, signed))
            loss = data_loss + (L2 / 2.0) * float(theta @ theta)
            gradient = (design[informative].T @ residual) / len(delta) + L2 * theta
        except FloatingPointError as exc:
            raise ValueError('action objective or gradient overflowed FP64') from exc
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite action objective or gradient')
    return float(loss), np.asarray(gradient, dtype=np.float64)


def objective_gradient(theta, features, delta, scales):
    """Return regularized loss and three-coordinate analytic gradient.

    Scales are supplied explicitly so heldout scoring never fits new RMS
    statistics.  All N rows remain in the denominator, including delta=0.
    """
    values = _features(features)
    labels = _delta(delta, len(values))
    normalized = _normalized_features(values, _scales(scales))
    return _evaluate(_theta(theta), normalized, labels)


def fit_action(features, delta):
    """Fit one fixed action head and return JSON-serializable diagnostics.

    All-neutral inputs return zero parameters, hence deterministic HOLD.
    A single informative class still uses the prescribed regularized loss.
    Optimizer failure raises rather than silently substituting a new recipe.
    """
    values = _features(features)
    labels = _delta(delta, len(values))
    scales = fit_rms(values)
    normalized = _normalized_features(values, scales)
    initial = np.zeros(PARAMETERS, dtype=np.float64)
    initial_loss, initial_gradient = _evaluate(initial, normalized, labels)
    informative = int(np.count_nonzero(labels))
    if informative == 0:
        theta = initial
        optimizer = dict(method='L-BFGS-B', success=True, status=0,
                         message='All calibration deltas are zero; deterministic HOLD fallback without optimization',
                         iterations=0, function_evaluations=0, invoked=False)
    else:
        result = minimize(lambda t: _evaluate(t, normalized, labels), initial,
                          method='L-BFGS-B', jac=True, bounds=None,
                          options=dict(maxiter=MAXITER, ftol=FTOL, gtol=GTOL))
        if not result.success:
            raise RuntimeError('Action L-BFGS-B did not converge: ' + str(result.message))
        theta = _theta(result.x)
        optimizer = dict(method='L-BFGS-B', success=bool(result.success), status=int(result.status),
                         message=str(result.message), iterations=int(result.nit),
                         function_evaluations=int(result.nfev), invoked=True)
    fitted_loss, gradient = _evaluate(theta, normalized, labels)
    if fitted_loss > initial_loss + 1e-10:
        raise RuntimeError('Fitted action objective exceeds its zero-initialized objective')
    optimizer.update(maxiter=MAXITER, ftol=FTOL, gtol=GTOL, bounds=None, initialization='all zero')
    # Recompute raw RMS zero flags with the same FP64 square/mean recipe.
    rms_raw = np.sqrt(np.mean(values * values, axis=0, dtype=np.float64))
    diagnostics = dict(calibration_rows=len(values), informative_rows=informative,
                       positive_rows=int(np.count_nonzero(labels == 1)),
                       negative_rows=int(np.count_nonzero(labels == -1)),
                       neutral_rows=int(np.count_nonzero(labels == 0)),
                       denominator=len(values), rms_rows=len(values), rms_includes_neutral=True,
                       rms_zero_fallback=[bool(v == 0.0) for v in rms_raw],
                       initial_loss=initial_loss, fitted_loss=fitted_loss,
                       initial_gradient_hex=[float(v).hex() for v in initial_gradient],
                       gradient_hex=[float(v).hex() for v in gradient],
                       gradient_inf_norm=float(np.max(np.abs(gradient))),
                       optimizer=optimizer, optimizer_success=optimizer['success'],
                       optimizer_status=optimizer['status'], optimizer_message=optimizer['message'],
                       iterations=optimizer['iterations'], function_evaluations=optimizer['function_evaluations'],
                       informative_zero_fallback=informative == 0,
                       centered=False, clipping=False, l2=L2, bias_regularized=True,
                       exact_net_bias_sweep=False, hyperparameter_search=False)
    return dict(status='RANK_ACTION_FIT_COMPLETE', theta=[float(v) for v in theta],
                theta_hex=[float(v).hex() for v in theta], scale=[float(v) for v in scales],
                scale_hex=[float(v).hex() for v in scales], diagnostics=diagnostics,
                feature_names=['top1_minus_top2_gap', 'selected_challenger_signed_RAW_gap'],
                feature_formula='((d/sd)*theta0+(r/sr)*theta1)+theta2',
                switch_threshold=0.0, switch_rule='strictly greater than zero',
                original_switches_locked=False)


def action_scores(features, head):
    """Score rows with frozen RMS/parameters; a zero score means HOLD."""
    values = _features(features)
    if not isinstance(head, dict) or not isinstance(head.get('theta_hex'), (list, tuple)) or not isinstance(head.get('scale_hex'), (list, tuple)):
        raise ValueError('head must contain theta_hex[3] and scale_hex[2]')
    try:
        theta = _theta([float.fromhex(v) for v in head['theta_hex']])
        scales = _scales([float.fromhex(v) for v in head['scale_hex']])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid hexadecimal action parameters or scales') from exc
    if 'theta' in head and [float(v).hex() for v in head['theta']] != list(head['theta_hex']):
        raise ValueError('decimal theta and hexadecimal theta disagree')
    if 'scale' in head and [float(v).hex() for v in head['scale']] != list(head['scale_hex']):
        raise ValueError('decimal scales and hexadecimal scales disagree')
    return _scores(_normalized_features(values, scales), theta)


def self_test():
    """Synthetic checks only, including two tiny synthetic action fits."""
    checks = 0
    x = np.zeros((127, 6), dtype=np.float64)
    x[:, 0] = -np.arange(127, dtype=np.float64)-1.0
    logits = np.zeros(127, dtype=np.float64)
    logits[3] = logits[9] = 2.0
    top, features = action_features(logits, x)
    assert top == 3 and features.tolist() == [0.0, -4.0]
    assert action_features(logits+8., x)[0] == top
    assert np.array_equal(action_features(logits+8., x)[1], features)
    checks += 2
    inputs = np.asarray([[1.,-2.], [3.,-4.], [100.,-200.]], dtype=np.float64)
    delta = np.array([1, -1, 0])
    scales = fit_rms(inputs)
    assert np.array_equal(scales, np.sqrt(np.mean(inputs*inputs, axis=0)))
    assert not np.array_equal(scales, fit_rms(inputs[:2]))
    assert np.array_equal(fit_rms(np.zeros((3,2))), np.ones(2))
    checks += 2
    theta = np.asarray([.2, -.3, .4], dtype=np.float64)
    loss, gradient = objective_gradient(theta, inputs, delta, scales)
    reduced, reduced_gradient = objective_gradient(theta, inputs[:2], delta[:2], scales)
    regularizer = .0005*float(theta @ theta)
    assert abs((loss-regularizer)-(reduced-regularizer)*2/3) < 1e-14
    assert np.max(abs((gradient-.001*theta)-(reduced_gradient-.001*theta)*2/3)) < 1e-14
    step = 1e-5
    numeric = np.zeros(3)
    for i in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step; minus[i] -= step
        numeric[i] = (objective_gradient(plus,inputs,delta,scales)[0]-
                      objective_gradient(minus,inputs,delta,scales)[0])/(2*step)
    assert np.max(abs(numeric-gradient)) < 1e-9
    checks += 2
    # A neutral row must not add log(2)/N to the objective.
    zero_loss, zero_grad = objective_gradient(np.zeros(3), inputs, np.zeros(3), scales)
    assert zero_loss == 0 and np.array_equal(zero_grad, np.zeros(3))
    bias_only = np.array([0., 0., .25])
    l, g = objective_gradient(bias_only, inputs, np.zeros(3), scales)
    assert l == .0005*.25*.25 and np.array_equal(g, .001*bias_only)
    checks += 2
    extreme_theta = np.array([1000.,-1000.,1000.])
    extreme_loss, extreme_gradient = objective_gradient(extreme_theta, inputs, delta, scales)
    assert np.isfinite(extreme_loss) and np.isfinite(extreme_gradient).all()
    checks += 1
    # All-neutral fallback includes its rows in RMS but creates exact HOLD.
    hold = fit_action(inputs, np.zeros(3))
    assert hold['theta_hex'] == [0.0.hex()]*3 and hold['diagnostics']['informative_zero_fallback']
    assert np.array_equal(action_scores(inputs, hold), np.zeros(3))
    assert not np.any(action_scores(inputs, hold)>0)
    checks += 1
    learn = np.array([[2.,-1.], [3.,-1.], [0.,-1.], [.1,-1.]], dtype=np.float64)
    learn_delta = np.array([1,1,-1,-1])
    learned = fit_action(learn, learn_delta)
    prediction = action_scores(learn, learned)
    assert np.array_equal(prediction>0, learn_delta>0)
    assert learned['diagnostics']['fitted_loss'] < learned['diagnostics']['initial_loss']
    assert learned['diagnostics']['gradient_inf_norm'] < 1e-5
    # Replay scores in the exact declared FP64 operation order.
    t = np.array([float.fromhex(v) for v in learned['theta_hex']])
    s = np.array([float.fromhex(v) for v in learned['scale_hex']])
    manual = ((learn[:,0]/s[0])*t[0] + (learn[:,1]/s[1])*t[1]) + t[2]
    assert np.array_equal(prediction, manual)
    checks += 2
    invalid = [(np.array([[-1., 0.]]), np.array([1])),
               (np.array([[1., np.nan]]), np.array([1])),
               (np.array([[1., 0.]]), np.array([2]))]
    for bad_features, bad_delta in invalid:
        try:
            fit_action(bad_features, bad_delta)
        except ValueError:
            checks += 1
        else:
            raise AssertionError('invalid action fit input accepted')
    return dict(status='RANK_ACTION_SYNTHETIC_PASS', checks=checks, natural_data_reads=0,
                neutral_denominator_verified=True, all_row_rms_verified=True,
                first_argmax_ties_verified=True, no_informative_fallback_HOLD=True,
                bias_in_regularizer_verified=True)


if __name__ == '__main__':
    print(self_test())
