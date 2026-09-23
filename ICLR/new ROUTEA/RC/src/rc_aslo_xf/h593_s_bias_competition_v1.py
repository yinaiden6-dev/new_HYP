"""Nested HOLD calibration with nonnegative S/competition slopes.

The convex logistic fit learns a direction, not the empirical net-gain optimum.
Only the subsequent one-dimensional bias fit enumerates all finite binary64
bias decision patterns exactly. Neither step reads an outer evaluation label.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MODES = {"BIAS1": (), "S_BIAS2": ("h_s",), "GAP_BIAS2": ("d",),
         "S_GAP3": ("h_s", "d")}
L2 = 0.001
MAX_FINITE = sys.float_info.max


def _finite(x: Any, name: str) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _nonnegative(x: Any, name: str) -> float:
    x = _finite(x, name)
    if x < 0:
        raise ValueError(f"{name} must be nonnegative")
    return x


def _rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(records):
        m = _finite(r["m"], "m")
        h = _nonnegative(r["h_s"], "h_s")
        d = _nonnegative(r["d"], "d")
        if r["delta"] not in (-1, 0, 1):
            raise ValueError("delta must be -1, 0 or 1")
        out.append({"index": i, "m": m, "h_s": h, "d": d,
                    "delta": int(r["delta"])})
    return out


def _u(m: float, h_s: float, d: float, alpha: float, beta: float) -> float:
    # Explicit scalar binary64 operations, no dot product/FMA/reassociation.
    ah = _finite(float(alpha * h_s), "alpha*h_s")
    bd = _finite(float(beta * d), "beta*d")
    mh = _finite(float(m + ah), "m+alpha*h_s")
    return _finite(float(mh + bd), "m+alpha*h_s+beta*d")


def first_crossing_bias(u: float) -> float | None:
    """Smallest finite bias with fl(u+b)>0; predecessor gives exact zero."""
    u = _finite(u, "u")
    zero = -u
    crossing = math.nextafter(zero, math.inf)
    if not math.isfinite(crossing):
        return None
    if not float(u + crossing) > 0.0:
        raise ArithmeticError("binary64 crossing failed")
    if float(u + math.nextafter(crossing, -math.inf)) > 0.0:
        raise ArithmeticError("crossing predecessor is already positive")
    return crossing


def _parameters(head: Mapping[str, Any]) -> tuple[float, float, float]:
    def read(name: str) -> float:
        return (float.fromhex(head[name + "_hex"]) if name + "_hex" in head
                else float(head[name]))
    return (_nonnegative(read("alpha"), "alpha"),
            _nonnegative(read("beta"), "beta"), _finite(read("bias"), "bias"))


def gate_score(m: float, h_s: float, d: float, head: Mapping[str, Any]) -> float:
    """Gate for one row; existing SWITCH and disabled heads return original m."""
    m = _finite(m, "m")
    h_s, d = _nonnegative(h_s, "h_s"), _nonnegative(d, "d")
    alpha, beta, bias = _parameters(head)
    if m > 0.0 or head.get("disabled", False):
        return m
    return _finite(float(_u(m, h_s, d, alpha, beta) + bias), "gate")


def apply_head(original_logits: Sequence[float], h_s: float, d: float,
               head: Mapping[str, Any]) -> tuple[float, ...]:
    """Change only frozen top challenger, and only when a HOLD becomes SWITCH.

    First maximum resolves ties. All 127 values of existing SWITCH rows,
    disabled heads and retained HOLD rows retain their exact original bits.
    """
    values = tuple(_finite(v, "logit") for v in original_logits)
    if not values:
        raise ValueError("nonempty challenger axis required")
    pos = max(range(len(values)), key=values.__getitem__)
    score = gate_score(values[pos], h_s, d, head)
    if values[pos] > 0.0 or score <= 0.0 or head.get("disabled", False):
        return values
    changed = list(values)
    changed[pos] = score
    return tuple(changed)


def calibrate_bias(records: Iterable[Mapping[str, Any]], alpha: float,
                   beta: float) -> dict[str, Any]:
    """Exactly maximize net gain over bias with fixed nonnegative slopes.

    Include neutral rows in decision counts/ties. Enumerate grouped crossings
    and select the closest-to-zero binary64 bias in each constant-action
    interval. Overflowing intervals are excluded; their closest-to-zero value
    overflows, hence every value in that same-sign interval does too. The
    explicit disabled head is the unchanged baseline, even if no finite bias
    can represent it. Disable unless the best empirical gain is positive.
    """
    rows = _rows(records)
    alpha, beta = _nonnegative(alpha, "alpha"), _nonnegative(beta, "beta")
    holds = [r for r in rows if r["m"] <= 0]
    groups: dict[float, list[dict[str, Any]]] = {}
    certs = []
    for r in holds:
        u = _u(r["m"], r["h_s"], r["d"], alpha, beta)
        crossing = first_crossing_bias(u)
        entry = {"index": r["index"], "u_hex": u.hex(), "delta": r["delta"],
                 "crossing_hex": None if crossing is None else crossing.hex(),
                 "predecessor_hex": None if crossing is None else
                     math.nextafter(crossing, -math.inf).hex()}
        certs.append(entry)
        if crossing is not None:
            groups.setdefault(crossing, []).append(r)
    crossings = sorted(groups)
    intervals = []
    rescues = breaks = neutral = changed = 0
    best_key = (0, 0, 0.0, 0.0)
    best_bias = 0.0
    best_counts = {"net_gain": 0, "changed_holds": 0, "rescues": 0,
                   "breaks": 0, "both_wrong": 0}
    us = [float.fromhex(c["u_hex"]) for c in certs]
    for j in range(len(crossings) + 1):
        lower = -MAX_FINITE if j == 0 else crossings[j - 1]
        upper = (MAX_FINITE if j == len(crossings) else
                 math.nextafter(crossings[j], -math.inf))
        if j:
            deltas = [r["delta"] for r in groups[lower]]
            rescues += deltas.count(1)
            breaks += deltas.count(-1)
            neutral += deltas.count(0)
            changed += len(deltas)
        # -0 and +0 are a single numeric bias; canonical +0 is deterministic.
        bias = 0.0 if lower <= 0.0 <= upper else (lower if lower > 0 else upper)
        scores = [float(u + bias) for u in us]
        feasible = all(math.isfinite(s) for s in scores)
        counts = {"net_gain": rescues - breaks, "changed_holds": changed,
                  "rescues": rescues, "breaks": breaks, "both_wrong": neutral}
        if feasible:
            observed = [c["delta"] for c, score in zip(certs, scores) if score > 0]
            if len(observed) != changed or sum(observed) != rescues - breaks:
                raise ArithmeticError("crossing sweep disagrees with literal gate")
        intervals.append({"lower_hex": lower.hex(), "upper_hex": upper.hex(),
                          "bias_hex": bias.hex(), "feasible": feasible, **counts})
        key = (counts["net_gain"], -changed, -abs(bias), -bias)
        if feasible and counts["net_gain"] > 0 and key > best_key:
            best_key, best_bias, best_counts = key, bias, counts
    disabled = best_counts["net_gain"] <= 0
    digest = hashlib.sha256(json.dumps(certs, sort_keys=True,
                           separators=(",", ":")).encode()).hexdigest()
    return {
        "alpha": alpha, "beta": beta, "bias": best_bias,
        "alpha_hex": alpha.hex(), "beta_hex": beta.hex(),
        "bias_hex": best_bias.hex(), "disabled": disabled,
        "training_net_gain": best_counts["net_gain"],
        "training_changed_holds": best_counts["changed_holds"],
        "training_rescues": best_counts["rescues"],
        "training_breaks": best_counts["breaks"],
        "training_both_wrong": best_counts["both_wrong"],
        "calibration_certificate": {
            "status": "EXACT_FINITE_BINARY64_BIAS_NET_GAIN_OPTIMUM_FIXED_SLOPES",
            "query_count": len(rows), "original_hold_count": len(holds),
            "original_switch_count": len(rows) - len(holds),
            "distinct_crossings": len(crossings),
            "unreachable_hold_count": sum(c["crossing_hex"] is None for c in certs),
            "breakpoints_sha256": digest, "certificates": certs,
            "intervals": intervals, "selected_counts": best_counts,
            "tie_break": "max net gain, min changed HOLDs, min abs(bias), min bias; disable unless gain>0",
            "scope": "exact bias calibration at frozen slopes; not joint net-gain optimality",
        },
    }


def surrogate_loss_gradient(theta: Sequence[float], offsets: np.ndarray,
                            design: np.ndarray, targets: np.ndarray,
                            l2: float = L2) -> tuple[float, np.ndarray]:
    """Mean logistic loss + l2/2 * squared norm, including free bias."""
    theta = np.asarray(theta, dtype=np.float64)
    offsets, design, targets = (np.asarray(x, dtype=np.float64)
                                for x in (offsets, design, targets))
    if len(offsets) == 0:
        return float(l2 * np.dot(theta, theta) / 2), l2 * theta
    z = offsets + design @ theta
    yz = targets * z
    loss = float(np.logaddexp(0.0, -yz).mean() + l2 * np.dot(theta, theta) / 2)
    gradient = design.T @ (-targets * expit(-yz)) / len(offsets) + l2 * theta
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite surrogate objective or gradient")
    return loss, gradient


def projected_gradient(theta: Sequence[float], gradient: Sequence[float],
                       constrained_count: int) -> np.ndarray:
    """KKT projected gradient for nonnegative slopes and unconstrained bias."""
    x = np.asarray(theta, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64).copy()
    for i in range(constrained_count):
        if x[i] <= 0.0 and g[i] > 0.0:
            g[i] = 0.0
    return g


def fit_head(records: Iterable[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    """Convex fit on informative HOLD rows, then exact net-gain bias refit."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rows = _rows(records)
    features = MODES[mode]
    informative = [r for r in rows if r["m"] <= 0.0 and r["delta"] != 0]
    offsets = np.asarray([r["m"] for r in informative], dtype=np.float64)
    design = np.asarray([[r[f] for f in features] + [1.0] for r in informative],
                        dtype=np.float64).reshape(len(informative), len(features) + 1)
    targets = np.asarray([r["delta"] for r in informative], dtype=np.float64)
    zero = np.zeros(len(features) + 1, dtype=np.float64)
    initial_loss, _ = surrogate_loss_gradient(zero, offsets, design, targets)
    if informative:
        opt = minimize(surrogate_loss_gradient, zero, args=(offsets, design, targets),
                       method="L-BFGS-B", jac=True,
                       bounds=[(0.0, None)] * len(features) + [(None, None)],
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        theta = np.asarray(opt.x, dtype=np.float64)
        success, message, status, nit, nfev = (bool(opt.success), str(opt.message),
                                             int(opt.status), int(opt.nit), int(opt.nfev))
    else:
        theta = zero
        success, message, status, nit, nfev = True, "no informative HOLD rows", 0, 0, 0
    if not np.isfinite(theta).all() or (theta[:len(features)] < 0.0).any():
        raise ArithmeticError("optimizer returned invalid parameters")
    final_loss, gradient = surrogate_loss_gradient(theta, offsets, design, targets)
    pg = projected_gradient(theta, gradient, len(features))
    weights = dict(zip(features, map(float, theta[:-1])))
    alpha, beta = weights.get("h_s", 0.0), weights.get("d", 0.0)
    head = calibrate_bias(rows, alpha, beta)
    calibrated_theta = theta.copy()
    calibrated_theta[-1] = head["bias"]
    calibrated_loss, _ = surrogate_loss_gradient(calibrated_theta, offsets, design, targets)
    head.update(mode=mode, optimization={
        "method": "L-BFGS-B", "success": success, "status": status,
        "message": message, "iterations": nit, "function_evaluations": nfev,
        "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "l2": L2,
        "l2_definition": "lambda/2 * sum(all free coefficients squared), including bias",
        "free_features": list(features) + ["bias"],
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
