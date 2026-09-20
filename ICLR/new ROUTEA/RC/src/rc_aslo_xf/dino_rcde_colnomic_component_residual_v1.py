"""Binary64 primitive for frozen ColNomic/component residual fusion.

The ColNomic score is supplied only through its immutable I0 big-endian
binary64 receipt.  One PJ1 candidate supplies the two registered directional
raw logits for INIT and TRACK_H.  This module applies exactly

    INIT = 0.5 * (INIT_a_to_b + INIT_b_to_a)
    TRACK_H = 0.5 * (TRACK_H_a_to_b + TRACK_H_b_to_a)
    delta = TRACK_H - INIT
    FUSED = RAW + delta

in that operation order.  The residual weight is the single frozen value
1.0; there is no weight-selection, normalization, cutoff, or decision API.
The module performs no identity reduction, label join, I/O, or inference.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import Any


RESIDUAL_WEIGHT = 1.0
SOURCE_ARMS = ("INIT", "TRACK_H")
SCORE_ARMS = ("RAW", "FUSED")
DIRECTIONS = ("a_to_b", "b_to_a")


class ComponentResidualError(ValueError):
    """An I0 score receipt or PJ1 candidate is malformed."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ComponentResidualError(message)


def _binary64(value: object, *, name: str) -> float:
    _require(type(value) in (int, float), f"{name} must be a JSON number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ComponentResidualError(f"{name} is not finite binary64") from error
    _require(math.isfinite(number), f"{name} is not finite binary64")
    return number


def float_bits(value: object) -> str:
    """Return the canonical big-endian binary64 receipt for a finite number."""

    number = _binary64(value, name="value")
    return struct.pack(">d", number).hex()


def raw_score_from_bits(raw_score_bits: str) -> float:
    """Decode and validate one immutable I0 ColNomic raw-score receipt."""

    _require(
        isinstance(raw_score_bits, str)
        and len(raw_score_bits) == 16
        and all(character in "0123456789abcdef" for character in raw_score_bits),
        "I0 candidate_raw_score_bits drift",
    )
    try:
        score = struct.unpack(">d", bytes.fromhex(raw_score_bits))[0]
    except (ValueError, struct.error) as error:
        raise ComponentResidualError("I0 candidate_raw_score_bits drift") from error
    _require(math.isfinite(score), "I0 raw score is not finite binary64")
    _require(
        float_bits(score) == raw_score_bits,
        "I0 candidate_raw_score_bits round-trip drift",
    )
    return float(score)


def direction_mean(a_to_b: object, b_to_a: object) -> float:
    """Apply the frozen candidate-level binary64 direction-mean order."""

    first = _binary64(a_to_b, name=DIRECTIONS[0])
    second = _binary64(b_to_a, name=DIRECTIONS[1])
    score = 0.5 * (first + second)
    _require(math.isfinite(score), "nonfinite binary64 direction mean")
    return float(score)


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


def candidate_component_scores(
    pj1_candidate: Mapping[str, Any],
) -> dict[str, float]:
    """Return the frozen INIT and TRACK_H direction means for one candidate."""

    _require(isinstance(pj1_candidate, Mapping), "PJ1 candidate must be a mapping")
    raw = pj1_candidate.get("raw_directional_logits")
    _require(
        isinstance(raw, Mapping)
        and tuple(raw) == SOURCE_ARMS
        and set(raw) == set(SOURCE_ARMS),
        "PJ1 source arm axis drift",
    )

    init_a_to_b = _raw_logit(raw, "INIT", "a_to_b")
    init_b_to_a = _raw_logit(raw, "INIT", "b_to_a")
    init_score = direction_mean(init_a_to_b, init_b_to_a)

    track_h_a_to_b = _raw_logit(raw, "TRACK_H", "a_to_b")
    track_h_b_to_a = _raw_logit(raw, "TRACK_H", "b_to_a")
    track_h_score = direction_mean(track_h_a_to_b, track_h_b_to_a)

    return {"INIT": init_score, "TRACK_H": track_h_score}


def fused_score(
    raw_score: object, init_component_score: object, track_h_component_score: object
) -> float:
    """Apply the single frozen unit-residual formula in its fixed order."""

    raw = _binary64(raw_score, name="RAW score")
    init = _binary64(init_component_score, name="INIT component score")
    track_h = _binary64(track_h_component_score, name="TRACK_H component score")
    delta = track_h - init
    _require(math.isfinite(delta), "nonfinite TRACK_H-minus-INIT residual")
    fused = raw + delta
    _require(math.isfinite(fused), "nonfinite FUSED score")
    return float(fused)


def _candidate_address(pj1_candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_key = pj1_candidate.get("candidate_key")
    physical_row = pj1_candidate.get("candidate_physical_row")
    corrected_identity = pj1_candidate.get("candidate_corrected_identity")
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

    address: dict[str, Any] = {
        "candidate_key": candidate_key,
        "candidate_physical_row": physical_row,
        "candidate_corrected_identity": corrected_identity,
    }
    if "candidate_reference_source_sha256" in pj1_candidate:
        reference_sha = pj1_candidate.get("candidate_reference_source_sha256")
        _require(
            isinstance(reference_sha, str)
            and len(reference_sha) == 64
            and all(character in "0123456789abcdef" for character in reference_sha),
            "candidate reference source SHA256 drift",
        )
        address["candidate_reference_source_sha256"] = reference_sha
    return address


def build_fused_candidate_payload(
    raw_score_bits: str, pj1_candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one JSON-ready RAW/FUSED payload while retaining PJ1 identity."""

    _require(isinstance(pj1_candidate, Mapping), "PJ1 candidate must be a mapping")
    address = _candidate_address(pj1_candidate)
    raw_score = raw_score_from_bits(raw_score_bits)
    component_scores = candidate_component_scores(pj1_candidate)
    fused = fused_score(
        raw_score,
        component_scores["INIT"],
        component_scores["TRACK_H"],
    )
    scores = {
        "RAW": {"score": raw_score, "score_bits": raw_score_bits},
        "FUSED": {"score": fused, "score_bits": float_bits(fused)},
    }
    return {**address, "scores": scores}


__all__ = [
    "ComponentResidualError",
    "DIRECTIONS",
    "RESIDUAL_WEIGHT",
    "SCORE_ARMS",
    "SOURCE_ARMS",
    "build_fused_candidate_payload",
    "candidate_component_scores",
    "direction_mean",
    "float_bits",
    "fused_score",
    "raw_score_from_bits",
]
