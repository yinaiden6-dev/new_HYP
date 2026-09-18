#!/usr/bin/env python3
"""Bounded convex-loss diagnostics with independently replayable rational bounds.

This module does not read data, train Adam, evaluate retrieval, or submit jobs.
Every claim is restricted to the supplied real feature endpoints and theta box.
SciPy supplies search points and dual suggestions, never a trusted certificate.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import time

import mpmath
from mpmath import iv
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit


INTERVAL_DPS = 50
SCOPE = "SUPPLIED_REAL_ENDPOINT_OBJECTIVE_ON_FIXED_BOX"
CERTIFICATE_SCHEMA = "RC_CONVEX_CAUSE_EXACT_BOX_CERTIFICATE_V1"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _objective_sha(terms):
    return _sha([{"weight": str(t["weight"]),
                  "vectors": [[str(v) for v in row] for row in t["vectors"]]}
                 for t in terms])


def _cone_prepare(cone_exact, dimension):
    if cone_exact is None:
        return None
    if not len(cone_exact) or any(len(row) != dimension for row in cone_exact):
        raise ValueError("cone dimensions")
    return [[rat(v) for v in row] for row in cone_exact]


def _cone_sha(cone_exact):
    return _sha(None if cone_exact is None else [[str(v) for v in row] for row in cone_exact])


def rat(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite endpoint")
    return Fraction.from_float(value)


def _mpf_rat(value):
    sign, man, exponent, _ = value
    if exponent >= 0:
        ans = Fraction(man << exponent)
    else:
        ans = Fraction(man, 1 << -exponent)
    return -ans if sign else ans


def _interval(value):
    value = rat(value)
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def _ends(value):
    return tuple(_mpf_rat(x) for x in value._mpi_)


def _down(value):
    value = rat(value)
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("nonfinite rounded lower bound")
    while rat(out) > value:
        out = float(np.nextafter(out, -np.inf))
    return out


def _up(value):
    value = rat(value)
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("nonfinite rounded upper bound")
    while rat(out) < value:
        out = float(np.nextafter(out, np.inf))
    return out


def build_terms(pair_x, pair_y=None, full_x=None, targets=None):
    """Return c*softplus(max(v dot theta)) terms for the original SIGN loss.

    pair_x is Nx7 including intercept; full_x is Qx127x7. targets[q] is -1
    for a correct RAW winner, otherwise the target's challenger-row index.
    A dictionary containing these four arguments may be passed instead.
    """
    if isinstance(pair_x, dict):
        data = pair_x
        pair_x, pair_y, full_x, targets = (
            data[k] for k in ("pair_x", "pair_y", "full_x", "targets")
        )
    px = np.asarray(pair_x, dtype=np.float64)
    py = np.asarray(pair_y)
    fx = np.asarray(full_x, dtype=np.float64)
    if px.ndim != 2 or fx.ndim != 3 or px.shape[1] != fx.shape[2]:
        raise ValueError("feature shape mismatch")
    if py.ndim != 1 or len(px) != len(py) or len(fx) != len(targets) or not len(px) or not len(fx):
        raise ValueError("label population mismatch")
    if not np.isfinite(px).all() or not np.isfinite(fx).all():
        raise ValueError("nonfinite features")
    terms = []
    for x, y in zip(px, py):
        if y not in (0, 1, False, True):
            raise ValueError("PAIR label must be binary")
        sign = -1 if bool(y) else 1
        terms.append({"weight": Fraction(1 if bool(y) else 4, len(px)),
                      "vectors": [[sign * rat(v) for v in x]]})
    for x, target in zip(fx, targets):
        if rat(target).denominator != 1:
            raise ValueError("target challenger position must be an integer")
        target = int(target)
        if target < -1 or target >= len(x):
            raise ValueError("invalid target challenger position")
        if target >= 0:
            terms.append({"weight": Fraction(1, len(fx)),
                          "vectors": [[-rat(v) for v in x[target]]]})
        wrong = [row for i, row in enumerate(x) if i != target]
        if not wrong:
            raise ValueError("no negative challenger")
        terms.append({"weight": Fraction(4, len(fx)),
                      "vectors": [[rat(v) for v in row] for row in wrong]})
    return terms


def decode_terms(payload):
    """Decode a JSON terms list whose feature endpoints are float.hex strings."""
    return [{"weight": rat(term["weight"]),
             "vectors": [[rat(float.fromhex(v)) if isinstance(v, str) and
                          ("0x" in v.lower()) else rat(v) for v in row]
                         for row in term["vectors"]]} for term in payload]


def _prepare(terms):
    exact = []
    numeric = []
    dimension = None
    for term in terms:
        weight = rat(term["weight"])
        vectors = [[rat(v) for v in row] for row in term["vectors"]]
        if weight <= 0 or not vectors:
            raise ValueError("terms must have positive weights and nonempty vectors")
        dimension = len(vectors[0]) if dimension is None else dimension
        if dimension < 1:
            raise ValueError("empty feature dimension")
        if any(len(v) != dimension for v in vectors):
            raise ValueError("inconsistent vector dimensions")
        exact.append({"weight": weight, "vectors": vectors})
        vf = np.asarray(vectors, dtype=np.float64)
        if not math.isfinite(float(weight)) or not np.isfinite(vf).all():
            raise ValueError("nonfinite numeric objective")
        numeric.append((float(weight), vf))
    if not exact:
        raise ValueError("empty objective")
    return exact, numeric, dimension


def _entropy_lower(p):
    if p == 0 or p == 1:
        return Fraction(0)
    x = _interval(p)
    return _ends(-x * iv.log(x) - (1 - x) * iv.log(1 - x))[0]


def _cut_from_choices(terms, probabilities, choices, bound):
    """Fenchel cut, corrected for rounded coefficients over the exact box."""
    iv.dps = INTERVAL_DPS
    d = len(terms[0]["vectors"][0])
    a = [Fraction(0)] * d
    b = Fraction(0)
    if len(probabilities) != len(terms) or len(choices) != len(terms):
        raise ValueError("cut choices count")
    for term, p, choice in zip(terms, probabilities, choices):
        p = rat(p)
        if not 0 <= p <= 1 or not 0 <= choice < len(term["vectors"]):
            raise ValueError("invalid Fenchel support")
        c = term["weight"]
        vector = term["vectors"][choice]
        for j in range(d):
            a[j] += c * p * vector[j]
        b += c * _entropy_lower(p)
    af = [float(x) for x in a]
    correction = rat(bound) * sum((abs(x - rat(y)) for x, y in zip(a, af)), Fraction(0))
    bf = _down(b - correction)
    return {"a_hex": [x.hex() for x in af], "b_hex": bf.hex(),
            "probability_hex": [float(p).hex() for p in probabilities],
            "vector_index": [int(x) for x in choices]}


def _make_cut(terms, numeric, theta, bound):
    choices, probabilities = [], []
    for _, vectors in numeric:
        logits = vectors @ theta
        index = int(np.argmax(logits))
        choices.append(index)
        probabilities.append(rat(float(expit(float(logits[index])))))
    return _cut_from_choices(terms, probabilities, choices, bound)


def _numeric_objective(numeric, theta):
    return math.fsum(c * float(np.logaddexp(0., np.max(v @ theta))) for c, v in numeric)


def objective_interval(terms, theta):
    """Outward rational interval, evaluating all max candidates exactly."""
    iv.dps = INTERVAL_DPS
    theta = [rat(x) for x in theta]
    if not terms or not theta or any(len(row) != len(theta) for term in terms for row in term["vectors"]):
        raise ValueError("objective interval dimensions")
    low = high = Fraction(0)
    for term in terms:
        z = max(sum((rat(v) * t for v, t in zip(row, theta)), Fraction(0))
                for row in term["vectors"])
        iz = _interval(z)
        # Stable on both signs, with no truncation or clipping approximation.
        sp = iz + iv.log(1 + iv.exp(-iz)) if z >= 0 else iv.log(1 + iv.exp(iz))
        lo, hi = _ends(sp)
        low += rat(term["weight"]) * lo
        high += rat(term["weight"]) * hi
    return {"lower_rational": str(low), "upper_rational": str(high),
            "lower": _down(low), "upper": _up(high)}


def _dual_certificate(cuts, cut_marginals, cone_marginals, cone_exact, bound):
    raw_mu = [max(Fraction(0), rat(-x)) for x in cut_marginals]
    total = sum(raw_mu, Fraction(0))
    if total <= 0:
        return None
    mu = [(i, x / total) for i, x in enumerate(raw_mu) if x]
    nu = [(i, max(Fraction(0), rat(-x)) / total)
          for i, x in enumerate(cone_marginals) if x < 0]
    cert = {"cut_weights": [[i, str(x)] for i, x in mu],
            "cone_weights": [[i, str(x)] for i, x in nu]}
    cert["lower_bound_rational"] = str(_certificate_lower(cuts, cert, cone_exact, bound))
    cert["lower_bound"] = _down(rat(cert["lower_bound_rational"]))
    return cert


def _certificate_lower(cuts, certificate, cone_exact, bound):
    d = len(cuts[0]["a_hex"])
    a = [Fraction(0)] * d
    b = Fraction(0)
    mu = [(int(i), rat(v)) for i, v in certificate["cut_weights"]]
    nu = [(int(i), rat(v)) for i, v in certificate["cone_weights"]]
    if any(v < 0 for _, v in mu + nu) or sum((v for _, v in mu), Fraction(0)) != 1:
        raise ValueError("invalid nonnegative normalized dual mixture")
    for i, weight in mu:
        if not 0 <= i < len(cuts):
            raise ValueError("invalid cut index")
        cut = cuts[i]
        b += weight * rat(float.fromhex(cut["b_hex"]))
        for j, value in enumerate(cut["a_hex"]):
            a[j] += weight * rat(float.fromhex(value))
    for i, weight in nu:
        if cone_exact is None or not 0 <= i < len(cone_exact):
            raise ValueError("invalid cone index")
        for j, value in enumerate(cone_exact[i]):
            a[j] -= weight * rat(value)
    return b - rat(bound) * sum((abs(v) for v in a), Fraction(0))


def _primal_candidate(theta, cone_exact, strict_witness, bound):
    """Return exact box/cone feasible coefficients, or no upper-bound witness."""
    b = rat(bound)
    candidate = [min(b, max(-b, rat(x))) for x in theta]
    if cone_exact is None:
        return candidate
    margins = [sum((rat(a) * x for a, x in zip(row, candidate)), Fraction(0))
               for row in cone_exact]
    if min(margins, default=Fraction(0)) >= 0:
        return candidate
    if strict_witness is None:
        # The exact origin always belongs to a homogeneous closed cone.
        return [Fraction(0)] * len(candidate)
    witness = [rat(x) for x in strict_witness]
    if len(witness) != len(candidate):
        raise ValueError("strict witness dimension")
    size = max(abs(x) for x in witness)
    if size == 0:
        raise ValueError("zero strict witness")
    scale = min(Fraction(1), b / (2 * size))
    witness = [x * scale for x in witness]
    wm = [sum((rat(a) * x for a, x in zip(row, witness)), Fraction(0))
          for row in cone_exact]
    if min(wm) <= 0:
        raise ValueError("provided witness is not exact strict cone feasible")
    alpha = max((-m / (w - m) for m, w in zip(margins, wm) if m < 0), default=Fraction(0))
    # Add an exact small interior margin, without using these coefficients as a model.
    alpha = min(Fraction(1), alpha + Fraction(1, 1 << 40))
    return [(1 - alpha) * x + alpha * w for x, w in zip(candidate, witness)]


def solve(terms, starttheta, cone_float=None, cone_exact=None, strict_witness=None,
          max_cuts=256, max_seconds=420., bound=64., tolerance=1e-5,
          conflict_threshold=None, progress=None):
    """Kelley LP search; bounds remain valid despite inexact LP duals or cones.

    conflict_threshold is a previously certified objective UPPER bound. A stop
    on lower > threshold certifies conflict on this box only. Cone coefficients
    returned here are mathematical diagnostics, never deployable parameters.
    progress, if supplied, receives a JSON-safe event after each LP iteration.
    max_seconds bounds search admission and the LP time limit, not final exact
    witness evaluation or an already executing rational/interval operation.
    """
    started = time.monotonic()
    terms, numeric, dimension = _prepare(terms)
    bound = rat(bound)
    if (bound <= 0 or not math.isfinite(float(bound)) or
            isinstance(max_cuts, bool) or int(max_cuts) != max_cuts or max_cuts < 1 or
            not math.isfinite(float(max_seconds)) or max_seconds <= 0 or
            not math.isfinite(float(tolerance)) or tolerance <= 0):
        raise ValueError("invalid resource or accuracy bounds")
    if (cone_float is None) != (cone_exact is None):
        raise ValueError("both exact and numeric cone matrices are required")
    if cone_exact is not None:
        cone_exact = _cone_prepare(cone_exact, dimension)
        cone_float = np.asarray(cone_float, dtype=np.float64)
        if cone_float.shape != (len(cone_exact), dimension) or not np.isfinite(cone_float).all():
            raise ValueError("invalid numeric cone")
    theta = np.asarray(starttheta, dtype=np.float64)
    if theta.shape != (dimension,) or not np.isfinite(theta).all():
        raise ValueError("invalid start theta")
    theta = np.clip(theta, -float(bound), float(bound))
    best_theta = theta.copy() if cone_exact is None else np.zeros(dimension)
    best_value = _numeric_objective(numeric, best_theta)
    cuts, history = [], []
    best_cert = None
    stop_reason = "MAX_CUTS"
    last_result = None
    for _ in range(int(max_cuts)):
        remaining = float(max_seconds) - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "WALL_TIME_BUDGET"
            break
        cuts.append(_make_cut(terms, numeric, theta, bound))
        a = np.asarray([[float.fromhex(v) for v in c["a_hex"]] + [-1.] for c in cuts])
        rhs = np.asarray([-float.fromhex(c["b_hex"]) for c in cuts])
        if cone_float is not None:
            a = np.vstack((a, np.column_stack((-cone_float, np.zeros(len(cone_float))))))
            rhs = np.r_[rhs, np.zeros(len(cone_float))]
        remaining = float(max_seconds) - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "WALL_TIME_BUDGET"
            break
        result = linprog(np.r_[np.zeros(dimension), 1.], A_ub=a, b_ub=rhs,
                         bounds=[(-float(bound), float(bound))] * dimension + [(None, None)],
                         method="highs", options={"time_limit": max(.001, remaining),
                         "primal_feasibility_tolerance": 1e-9,
                         "dual_feasibility_tolerance": 1e-9})
        last_result = {"status": int(result.status), "message": str(result.message)}
        if not result.success:
            stop_reason = "LP_SEARCH_UNRESOLVED"
            break
        marginals = result.ineqlin.marginals
        cert = _dual_certificate(cuts, marginals[:len(cuts)], marginals[len(cuts):], cone_exact, bound)
        if cert is not None and (best_cert is None or rat(cert["lower_bound_rational"]) > rat(best_cert["lower_bound_rational"])):
            best_cert = cert
        theta = np.clip(result.x[:dimension], -float(bound), float(bound))
        value = _numeric_objective(numeric, theta)
        if value < best_value:
            best_value, best_theta = value, theta.copy()
        history.append({"cuts": len(cuts), "numeric_upper_suggestion": best_value,
                        "certified_lower": best_cert["lower_bound"] if best_cert else None})
        if progress is not None:
            progress({**history[-1], "event": "LP_ITERATION",
                      "elapsed_seconds": time.monotonic() - started})
        if best_cert is not None:
            lower = rat(best_cert["lower_bound_rational"])
            if conflict_threshold is not None and lower > rat(conflict_threshold):
                stop_reason = "CERTIFIED_BOX_CONFLICT"
                break
            if best_value - float(lower) <= tolerance:
                primal = _primal_candidate(best_theta, cone_exact, strict_witness, bound)
                if primal is not None:
                    interval = objective_interval(terms, primal)
                    if rat(interval["upper_rational"]) - lower <= rat(tolerance):
                        stop_reason = "CERTIFIED_BOX_OPTIMALITY_GAP"
                        break
    primal = _primal_candidate(best_theta, cone_exact, strict_witness, bound)
    interval = objective_interval(terms, primal) if primal is not None else None
    if best_cert is None:
        # Softplus terms with positive weights are nonnegative everywhere.
        lower = Fraction(0)
    else:
        lower = rat(best_cert["lower_bound_rational"])
    upper = rat(interval["upper_rational"]) if interval else None
    if upper is not None and lower > upper:
        raise RuntimeError("CERTIFICATE_LOWER_EXCEEDS_UPPER")
    return {"status": stop_reason, "scope": SCOPE,
            "certificate_schema": CERTIFICATE_SCHEMA,
            "objective_sha256": _objective_sha(terms), "cone_sha256": _cone_sha(cone_exact),
            "unbounded_global_optimality_claimed": False,
            "bound_rational": str(bound), "dimension": dimension, "cuts": cuts,
            "certificate": best_cert, "lower_bound_rational": str(lower),
            "lower_bound": _down(lower), "upper_interval": interval,
            "upper_bound": interval["upper"] if interval else None,
            "certified_gap_upper": _up(upper - lower) if upper is not None else None,
            "best_theta": [float(x) for x in primal] if primal is not None else None,
            "theta_binary64": [float(x).hex() for x in primal] if primal is not None else None,
            "best_theta_exact": [str(x) for x in primal] if primal is not None else None,
            "theta_float_is_exact_primal": primal is not None and all(rat(float(x)) == x for x in primal),
            "diagnostic_only": cone_exact is not None,
            "box_active_coordinates": [i for i, x in enumerate(primal or []) if abs(x) == bound],
            "max_cuts": int(max_cuts), "max_seconds": float(max_seconds),
            "time_budget_scope": "SEARCH_ADMISSION_AND_LP_ONLY_EXCLUDES_FINAL_EXACT_REPLAY",
            "tolerance_rational": str(rat(tolerance)),
            "conflict_threshold_rational": str(rat(conflict_threshold)) if conflict_threshold is not None else None,
            "elapsed_seconds": time.monotonic() - started, "cut_count": len(cuts),
            "history": history, "last_lp": last_result, "interval_dps": INTERVAL_DPS,
            "mpmath_version": mpmath.__version__}


def validate_certificate(terms, result, cone_exact=None, bound=None):
    """Reconstruct every supporting cut and bound without invoking a solver."""
    terms, _, dimension = _prepare(terms)
    cone_exact = _cone_prepare(cone_exact, dimension)
    if (result.get("certificate_schema") != CERTIFICATE_SCHEMA or
            result.get("scope") != SCOPE or result.get("unbounded_global_optimality_claimed") is not False or
            result.get("diagnostic_only") != (cone_exact is not None) or
            result.get("objective_sha256") != _objective_sha(terms) or
            result.get("cone_sha256") != _cone_sha(cone_exact)):
        raise ValueError("objective, cone, or claim scope binding mismatch")
    if bound is not None and rat(bound) != rat(result["bound_rational"]):
        raise ValueError("caller box does not match certificate")
    bound = rat(result["bound_rational"])
    if result["dimension"] != dimension or bound <= 0:
        raise ValueError("result scope mismatch")
    if result.get("cut_count") != len(result["cuts"]) or len(result["cuts"]) > result["max_cuts"]:
        raise ValueError("cut accounting mismatch")
    for cut in result["cuts"]:
        ps = [rat(float.fromhex(x)) for x in cut["probability_hex"]]
        replay = _cut_from_choices(terms, ps, cut["vector_index"], bound)
        if replay != cut:
            raise ValueError("Fenchel cut reconstruction mismatch")
    cert = result["certificate"]
    lower = _certificate_lower(result["cuts"], cert, cone_exact, bound) if cert else Fraction(0)
    if lower != rat(result["lower_bound_rational"]):
        raise ValueError("exact lower bound mismatch")
    if result.get("lower_bound") != _down(lower):
        raise ValueError("lower display rounding mismatch")
    if cert is not None and (rat(cert["lower_bound_rational"]) != lower or cert["lower_bound"] != _down(lower)):
        raise ValueError("certificate lower bound mismatch")
    upper = None
    if result["best_theta_exact"] is not None:
        theta = [rat(x) for x in result["best_theta_exact"]]
        if len(theta) != dimension or any(abs(x) > bound for x in theta):
            raise ValueError("primal outside box")
        if cone_exact is not None and any(sum((rat(a) * x for a, x in zip(row, theta)), Fraction(0)) < 0 for row in cone_exact):
            raise ValueError("primal outside exact cone")
        interval = objective_interval(terms, theta)
        if interval != result["upper_interval"]:
            raise ValueError("outward upper interval mismatch")
        if (result.get("upper_bound") != interval["upper"] or
                result.get("certified_gap_upper") != _up(rat(interval["upper_rational"]) - lower) or
                result.get("best_theta") != [float(x) for x in theta] or
                result.get("theta_binary64") != [float(x).hex() for x in theta] or
                result.get("theta_float_is_exact_primal") != all(rat(float(x)) == x for x in theta) or
                result.get("box_active_coordinates") != [i for i, x in enumerate(theta) if abs(x) == bound]):
            raise ValueError("primal display or gap mismatch")
        upper = rat(interval["upper_rational"])
        if lower > upper:
            raise ValueError("lower exceeds upper")
    elif (any(result.get(key) is not None for key in
              ("upper_interval", "upper_bound", "certified_gap_upper", "best_theta", "theta_binary64")) or
          result.get("theta_float_is_exact_primal") is not False or result.get("box_active_coordinates") != []):
        raise ValueError("unsupported upper bound fields")
    allowed_statuses = {"MAX_CUTS", "WALL_TIME_BUDGET", "LP_SEARCH_UNRESOLVED",
                        "CERTIFIED_BOX_OPTIMALITY_GAP", "CERTIFIED_BOX_CONFLICT"}
    if result["status"] not in allowed_statuses or rat(result["tolerance_rational"]) <= 0:
        raise ValueError("invalid termination metadata")
    if result["status"] == "CERTIFIED_BOX_OPTIMALITY_GAP" and (
            upper is None or upper - lower > rat(result["tolerance_rational"])):
        raise ValueError("unsupported optimality stop")
    if result["status"] == "CERTIFIED_BOX_CONFLICT" and (
            result["conflict_threshold_rational"] is None or
            lower <= rat(result["conflict_threshold_rational"])):
        raise ValueError("unsupported conflict stop")
    return {"status": "EXACT_RATIONAL_CUT_AND_INTERVAL_BOUND_REPLAY_PASS",
            "cuts_checked": len(result["cuts"]), "lower_bound_rational": str(lower),
            "upper_bound_rational": str(upper) if upper is not None else None,
            "solver_calls": 0, "scope": result["scope"]}
