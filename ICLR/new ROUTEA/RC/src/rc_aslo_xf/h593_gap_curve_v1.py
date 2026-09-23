"""Three-parameter challenger-gap curve with the RAW coefficient locked to zero.

The nonlinear feature is d squared divided by its RMS over all original
training HOLD rows. The public h_raw input is deliberately ignored: every
delegated numerical operation receives canonical positive zero. This prevents
even signed-zero RAW values from influencing the fitted or applied model.

Only the conditional bias calibration is an exact empirical net optimum.
The three free coefficients are fitted to a regularized convex surrogate.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from . import h593_conditional_gap_v1 as frozen

MODE = "GAP_CURVE3"
MODES = {MODE: ("d", "phi")}
L2 = frozen.L2
surrogate_loss_gradient = frozen.surrogate_loss_gradient
projected_gradient = frozen.projected_gradient
first_crossing_bias = frozen.first_crossing_bias


def _mode(mode: str) -> str:
    if mode != MODE:
        raise ValueError(f"unknown mode {mode!r}")
    return "GAP_CURVE4"


def _rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Do not coerce or inspect h_raw: it is absent from this model's inputs.
    return frozen._rows({**r, "h_raw": 0.0} for r in records)


def feature_value(h_raw: float, d: float, mode: str, scale: float) -> float:
    return frozen.feature_value(0.0, d, _mode(mode), scale)


def fit_scale(records: Iterable[Mapping[str, Any]], mode: str = MODE) -> dict[str, Any]:
    result = frozen.fit_scale(_rows(records), _mode(mode))
    result["mode"] = mode
    return result


def _internal_head(head: Mapping[str, Any]) -> dict[str, Any]:
    internal = dict(head)
    internal["mode"] = _mode(str(head["mode"]))
    gamma = (float.fromhex(head["gamma_hex"]) if "gamma_hex" in head
             else float(head["gamma"]))
    if gamma != 0.0:
        raise ValueError("GAP_CURVE3 gamma must remain zero")
    # Canonicalization is essential when gamma or the unused input is -0.0.
    internal.update(gamma=0.0, gamma_hex=0.0.hex())
    return internal


def gate_score(m: float, h_raw: float, d: float, head: Mapping[str, Any]) -> float:
    return frozen.gate_score(m, 0.0, d, _internal_head(head))


def apply_head(original_logits: Sequence[float], h_raw: float, d: float,
               head: Mapping[str, Any]) -> tuple[float, ...]:
    return frozen.apply_head(original_logits, 0.0, d, _internal_head(head))


def calibrate_bias(records: Iterable[Mapping[str, Any]], beta: float,
                   eta: float, mode: str = MODE, scale: float = 1.0) -> dict[str, Any]:
    result = frozen.calibrate_bias(_rows(records), 0.0, beta, eta,
                                   _mode(mode), scale)
    result.update(mode=mode, frozen_gamma=0.0, ignored_raw_input=True)
    return result


def fit_head(records: Iterable[Mapping[str, Any]], mode: str = MODE) -> dict[str, Any]:
    """Fit beta>=0, free eta/bias, then calibrate bias with slopes frozen."""
    rows = _rows(records)
    scaling = fit_scale(rows, mode)
    scale = scaling["scale"]
    informative = [r for r in rows if r["m"] <= 0.0 and r["delta"] != 0]
    offsets = np.asarray([r["m"] for r in informative], dtype=np.float64)
    design = np.asarray([[r["d"], feature_value(0.0, r["d"], mode, scale), 1.0]
                         for r in informative], dtype=np.float64).reshape(len(informative), 3)
    targets = np.asarray([r["delta"] for r in informative], dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    initial_loss, _ = surrogate_loss_gradient(zero, offsets, design, targets)
    if informative:
        opt = minimize(surrogate_loss_gradient, zero, args=(offsets, design, targets),
                       method="L-BFGS-B", jac=True,
                       bounds=[(0.0, None), (None, None), (None, None)],
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        theta = np.asarray(opt.x, dtype=np.float64)
        success, message, status, nit, nfev = (bool(opt.success), str(opt.message),
                                             int(opt.status), int(opt.nit), int(opt.nfev))
    else:
        theta = zero
        success, message, status, nit, nfev = True, "no informative HOLD rows", 0, 0, 0
    if not np.isfinite(theta).all() or theta[0] < 0.0:
        raise ArithmeticError("optimizer returned invalid parameters")
    final_loss, gradient = surrogate_loss_gradient(theta, offsets, design, targets)
    pg = projected_gradient(theta, gradient, 1)
    beta, eta = map(float, theta[:2])
    head = calibrate_bias(rows, beta, eta, mode, scale)
    calibrated_theta = theta.copy()
    calibrated_theta[-1] = head["bias"]
    calibrated_loss, _ = surrogate_loss_gradient(calibrated_theta, offsets, design, targets)
    head.update(training_scale=scaling, training_scale_count=scaling["count"],
                training_scale_rule=scaling["rule"], optimization={
        "method": "L-BFGS-B", "success": success, "status": status,
        "message": message, "iterations": nit, "function_evaluations": nfev,
        "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "l2": L2,
        "l2_definition": "lambda/2 * sum(all three free coefficients squared), including bias; gamma fixed zero",
        "free_features": ["d", "phi", "bias"],
        "design_features": ["d", "phi", "bias"],
        "coefficient_order": ["beta", "eta", "bias"],
        "bounds": [[0.0, None], [None, None], [None, None]],
        "initialization": "all zero", "theta_hex": [float(x).hex() for x in theta],
        "surrogate_bias_hex": float(theta[-1]).hex(),
        "informative_hold_count": len(informative),
        "neutral_holds_omitted_from_surrogate": sum(r["m"] <= 0 and r["delta"] == 0 for r in rows),
        "initial_surrogate_loss": initial_loss, "fitted_surrogate_loss": final_loss,
        "calibrated_bias_surrogate_loss": calibrated_loss,
        "calibrated_loss_is_applied_when_disabled": False,
        "gradient_hex": [float(x).hex() for x in gradient],
        "projected_gradient_inf_norm": float(np.abs(pg).max(initial=0.0)),
        "joint_empirical_net_optimum_claimed": False,
    })
    return head
