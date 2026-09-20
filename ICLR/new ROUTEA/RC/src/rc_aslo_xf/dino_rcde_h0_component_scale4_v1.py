"""Binary64 candidate-score primitive for the frozen component Scale-4 arm.

The input is one PJ1 candidate: INIT and TRACK_H each carry exactly the two
registered directional raw logits and their big-endian binary64 receipts.
Direction means use the same operation order as the frozen PJ2 reducer.  The
single registered successor is then computed at candidate-score level as

    INIT + 4.0 * (TRACK_H - INIT)

This module deliberately exposes no scale-selection or decision-cutoff API.
It performs no identity reduction, label join, I/O, or scientific inference.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import Any


SCALE = 4.0
SOURCE_ARMS = ("INIT", "TRACK_H")
SCALE4_ARM = "SCALE4"
SCORE_ARMS = (*SOURCE_ARMS, SCALE4_ARM)
DIRECTIONS = ("a_to_b", "b_to_a")


class ComponentScale4Error(ValueError):
    """A PJ1 candidate or one of its binary64 receipts is malformed."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ComponentScale4Error(message)


def _binary64(value: object, *, name: str) -> float:
    _require(
        type(value) in (int, float),
        f"{name} must be a JSON number",
    )
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ComponentScale4Error(f"{name} is not finite binary64") from error
    _require(math.isfinite(result), f"{name} is not finite binary64")
    return result


def float_bits(value: object) -> str:
    """Return the canonical big-endian binary64 receipt for a finite number."""

    number = _binary64(value, name="value")
    return struct.pack(">d", number).hex()


def candidate_score(a_to_b: object, b_to_a: object) -> float:
    """Reproduce the registered PJ2 binary64 direction-mean operation order."""

    first = _binary64(a_to_b, name=DIRECTIONS[0])
    second = _binary64(b_to_a, name=DIRECTIONS[1])
    score = 0.5 * (first + second)
    _require(math.isfinite(score), "nonfinite binary64 direction mean")
    return float(score)


def scale4_score(init_score: object, track_h_score: object) -> float:
    """Apply the one registered candidate-level Scale-4 formula exactly."""

    init = _binary64(init_score, name="INIT score")
    track_h = _binary64(track_h_score, name="TRACK_H score")
    delta = track_h - init
    _require(math.isfinite(delta), "nonfinite TRACK_H-minus-INIT delta")
    scaled = init + SCALE * delta
    _require(math.isfinite(scaled), "nonfinite SCALE4 score")
    return float(scaled)


def _raw_logit(
    raw_directional_logits: Mapping[str, Any], arm: str, direction: str
) -> float:
    arm_payload = raw_directional_logits.get(arm)
    _require(
        isinstance(arm_payload, Mapping)
        and tuple(arm_payload) == DIRECTIONS
        and set(arm_payload) == set(DIRECTIONS),
        f"{arm} direction axis drift",
    )
    item = arm_payload.get(direction)
    _require(
        isinstance(item, Mapping)
        and set(item) == {"raw_logit", "raw_logit_bits"},
        f"{arm}/{direction} raw logit schema drift",
    )
    value = _binary64(item.get("raw_logit"), name=f"{arm}/{direction} raw logit")
    bits = item.get("raw_logit_bits")
    _require(
        isinstance(bits, str)
        and len(bits) == 16
        and bits == float_bits(value),
        f"{arm}/{direction} raw_logit_bits drift",
    )
    return value


def build_scale4_candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build one JSON-ready INIT/TRACK_H/SCALE4 candidate-score payload.

    The output retains the PJ1 address and corrected identity needed by the
    frozen PJ2 duplicate-identity and stable-tie reducer.  A reference-source
    receipt is retained when the PJ1 candidate supplies it.
    """

    _require(isinstance(candidate, Mapping), "PJ1 candidate must be a mapping")
    candidate_key = candidate.get("candidate_key")
    physical_row = candidate.get("candidate_physical_row")
    corrected_identity = candidate.get("candidate_corrected_identity")
    _require(
        isinstance(candidate_key, str) and bool(candidate_key),
        "candidate key is empty",
    )
    _require(
        isinstance(physical_row, int)
        and not isinstance(physical_row, bool)
        and physical_row >= 0,
        "candidate physical row drift",
    )
    _require(
        isinstance(corrected_identity, str) and bool(corrected_identity),
        "candidate corrected identity is empty",
    )

    raw = candidate.get("raw_directional_logits")
    _require(
        isinstance(raw, Mapping)
        and tuple(raw) == SOURCE_ARMS
        and set(raw) == set(SOURCE_ARMS),
        "PJ1 source arm axis drift",
    )
    source_scores = {
        arm: candidate_score(
            _raw_logit(raw, arm, DIRECTIONS[0]),
            _raw_logit(raw, arm, DIRECTIONS[1]),
        )
        for arm in SOURCE_ARMS
    }
    scale4 = scale4_score(source_scores["INIT"], source_scores["TRACK_H"])
    score_values = {
        "INIT": source_scores["INIT"],
        "TRACK_H": source_scores["TRACK_H"],
        SCALE4_ARM: scale4,
    }
    scores = {
        arm: {"score": score_values[arm], "score_bits": float_bits(score_values[arm])}
        for arm in SCORE_ARMS
    }

    payload: dict[str, Any] = {
        "candidate_key": candidate_key,
        "candidate_physical_row": physical_row,
        "candidate_corrected_identity": corrected_identity,
        "scores": scores,
    }
    if "candidate_reference_source_sha256" in candidate:
        reference_sha = candidate.get("candidate_reference_source_sha256")
        _require(
            isinstance(reference_sha, str)
            and len(reference_sha) == 64
            and all(character in "0123456789abcdef" for character in reference_sha),
            "candidate reference source SHA256 drift",
        )
        payload["candidate_reference_source_sha256"] = reference_sha
    return payload


def pair_transition(
    baseline_strict_top1: bool, comparison_strict_top1: bool
) -> str:
    """Classify one fixed-arm strict-top-1 pair without recomputing scores."""

    _require(type(baseline_strict_top1) is bool, "baseline top1 flag drift")
    _require(type(comparison_strict_top1) is bool, "comparison top1 flag drift")
    if not baseline_strict_top1 and comparison_strict_top1:
        return "rescue"
    if baseline_strict_top1 and not comparison_strict_top1:
        return "break"
    return "both_correct" if baseline_strict_top1 else "both_wrong"


__all__ = [
    "ComponentScale4Error",
    "DIRECTIONS",
    "SCALE",
    "SCALE4_ARM",
    "SCORE_ARMS",
    "SOURCE_ARMS",
    "build_scale4_candidate_payload",
    "candidate_score",
    "float_bits",
    "pair_transition",
    "scale4_score",
]
