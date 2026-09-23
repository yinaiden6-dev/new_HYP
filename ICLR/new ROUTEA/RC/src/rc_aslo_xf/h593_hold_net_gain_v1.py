"""Exact one-parameter calibration of decisions originally assigned HOLD.

The frozen best challenger's logit is lifted by ``alpha * h``.  The product
and sum are separate binary64 operations; alpha is finite and nonnegative,
and h lies in [0, 1].  Existing SWITCH decisions and challenger ordering
are protected.  Training uses retrieval correctness differences only.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


MAX_FINITE_BITS = 0x7FEFFFFFFFFFFFFF


def _from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _to_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _weight(value: Any) -> float:
    value = _finite(value, "h")
    if not 0.0 <= value <= 1.0:
        raise ValueError("h must lie in [0, 1]")
    return value


def _alpha(value: Any) -> float:
    value = _finite(value, "alpha")
    if value < 0.0:
        raise ValueError("alpha must be nonnegative")
    return value


def lifted_logit(m: float, h: float, alpha: float) -> float:
    """Specified separate binary64 product and addition, without FMA."""
    m, h, alpha = _finite(m, "m"), _weight(h), _alpha(alpha)
    if m > 0.0 or alpha == 0.0:
        return m
    return float(m + float(alpha * h))


def first_crossing_alpha(m: float, h: float) -> float | None:
    """Smallest finite binary64 alpha making an original HOLD strictly > 0.

    Positive finite IEEE-754 values are ordered by their unsigned bits.
    Binary search therefore handles rounding, subnormals and exact ties
    without approximating the crossing by -m/h.
    """
    m, h = _finite(m, "m"), _weight(h)
    if m > 0.0:
        raise ValueError("first_crossing_alpha requires an original HOLD (m <= 0)")
    if h == 0.0:
        return None

    def crosses(bits: int) -> bool:
        return float(m + float(_from_bits(bits) * h)) > 0.0

    if not crosses(MAX_FINITE_BITS):
        return None
    lo, hi = 0, MAX_FINITE_BITS  # lo does not cross; hi crosses.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if crosses(mid):
            hi = mid
        else:
            lo = mid
    return _from_bits(hi)


def top_position(logits: Sequence[float]) -> int:
    """First maximum, matching the frozen challenger-axis tie convention."""
    values = tuple(_finite(x, "logit") for x in logits)
    if not values:
        raise ValueError("at least one challenger logit is required")
    return max(range(len(values)), key=values.__getitem__)


def apply_hold_lift(logits: Sequence[float], h: float, alpha: float) -> tuple[float, ...]:
    """Lift only the original best challenger of an original HOLD.

    All other binary64 values, including signed zero, remain unchanged.
    Alpha zero and all original SWITCH rows are complete identity maps.
    """
    values = tuple(_finite(x, "logit") for x in logits)
    h, alpha = _weight(h), _alpha(alpha)
    if not values:
        raise ValueError("at least one challenger logit is required")
    pos = max(range(len(values)), key=values.__getitem__)
    if alpha == 0.0 or values[pos] > 0.0:
        return values
    changed = list(values)
    changed[pos] = lifted_logit(values[pos], h, alpha)
    return tuple(changed)


def fit_hold_net_gain(
    rows: Iterable[Mapping[str, Any]], *, zero_break: bool = False
) -> dict[str, Any]:
    """Globally maximize training net gain over finite nonnegative alpha.

    Each row has m (frozen top logit), h (label-free lift weight), and delta
    (challenger correctness minus original winner correctness). Delta is
    -1, 0 or +1. Both-wrong rows count as changed decisions, not as breaks.
    Equal gains prefer fewer changed HOLDs, then the smallest alpha. If no
    strictly positive net gain exists, alpha is exactly zero. ``zero_break``
    optionally restricts the fit to settings with no training breaks.
    """
    grouped: dict[int, list[int]] = {}
    certificates: list[dict[str, Any]] = []
    original_switches = 0
    unreachable_holds = 0
    hold_count = 0
    for index, row in enumerate(rows):
        m, h = _finite(row["m"], "m"), _weight(row["h"])
        if row["delta"] not in (-1, 0, 1):
            raise ValueError("delta must be -1, 0 or +1")
        delta = int(row["delta"])
        cert: dict[str, Any] = {
            "index": index, "m_hex": m.hex(), "h_hex": h.hex(), "delta": delta
        }
        if m > 0.0:
            original_switches += 1
            cert.update(status="existing_switch_locked", alpha_hex=None)
        else:
            hold_count += 1
            crossing = first_crossing_alpha(m, h)
            if crossing is None:
                unreachable_holds += 1
                cert.update(status="unreachable_with_finite_alpha", alpha_hex=None)
            else:
                bits = _to_bits(crossing)
                grouped.setdefault(bits, []).append(delta)
                predecessor = _from_bits(bits - 1)
                cert.update(
                    status="crossing", alpha_hex=crossing.hex(),
                    predecessor_hex=predecessor.hex(),
                    predecessor_logit_hex=lifted_logit(m, h, predecessor).hex(),
                    crossing_logit_hex=lifted_logit(m, h, crossing).hex(),
                )
        certificates.append(cert)

    best_bits = 0
    best_gain = best_changed = best_rescues = best_breaks = best_neutral = 0
    rescues = breaks = neutral = changed = 0
    intervals: list[dict[str, Any]] = [{
        "alpha_hex": 0.0.hex(), "net_gain": 0, "changed_holds": 0,
        "rescues": 0, "breaks": 0, "both_wrong": 0, "feasible": True,
    }]
    for bits, deltas in sorted(grouped.items()):
        rescues += deltas.count(1)
        breaks += deltas.count(-1)
        neutral += deltas.count(0)
        changed += len(deltas)
        gain = rescues - breaks
        feasible = not zero_break or breaks == 0
        intervals.append({
            "alpha_hex": _from_bits(bits).hex(), "net_gain": gain,
            "changed_holds": changed, "rescues": rescues, "breaks": breaks,
            "both_wrong": neutral, "feasible": feasible,
        })
        candidate_key = (gain, -changed, -bits)
        best_key = (best_gain, -best_changed, -best_bits)
        if feasible and gain > 0 and candidate_key > best_key:
            best_bits = bits
            best_gain, best_changed = gain, changed
            best_rescues, best_breaks, best_neutral = rescues, breaks, neutral

    encoded = json.dumps(certificates, sort_keys=True, separators=(",", ":")).encode()
    alpha = _from_bits(best_bits)
    return {
        "status": "EXACT_FINITE_BINARY64_HOLD_NET_GAIN_OPTIMUM",
        "alpha": alpha, "alpha_hex": alpha.hex(), "zero_break": bool(zero_break),
        "training_net_gain": best_gain, "training_changed_holds": best_changed,
        "training_rescues": best_rescues, "training_breaks": best_breaks,
        "training_both_wrong": best_neutral,
        "query_count": len(certificates), "original_hold_count": hold_count,
        "original_switch_count": original_switches,
        "unreachable_hold_count": unreachable_holds,
        "distinct_crossings": len(grouped),
        "breakpoints_sha256": hashlib.sha256(encoded).hexdigest(),
        "certificates": certificates, "intervals": intervals,
        "optimality": "All realizable decision patterns enumerated at their smallest finite alpha; existing SWITCH decisions locked.",
        "tie_break": "max net gain, min changed HOLDs, min alpha; alpha=0 unless gain>0",
    }
