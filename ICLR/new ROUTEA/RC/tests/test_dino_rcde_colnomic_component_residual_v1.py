from __future__ import annotations

import copy
import inspect
import struct

import pytest

from rc_aslo_xf import dino_rcde_colnomic_component_residual_v1 as residual


def _bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _raw_logit(value: float) -> dict[str, object]:
    return {"raw_logit": float(value), "raw_logit_bits": _bits(value)}


def _candidate(
    *,
    init: tuple[float, float] = (0.1, 0.2),
    track_h: tuple[float, float] = (0.3, 0.4),
) -> dict[str, object]:
    return {
        "source_candidate_record_sha256": "1" * 64,
        "candidate_key": "candidate-7",
        "candidate_physical_row": 7,
        "candidate_reference_source_sha256": "2" * 64,
        "candidate_corrected_identity": "identity-7",
        "raw_directional_logits": {
            "INIT": {
                "a_to_b": _raw_logit(init[0]),
                "b_to_a": _raw_logit(init[1]),
            },
            "TRACK_H": {
                "a_to_b": _raw_logit(track_h[0]),
                "b_to_a": _raw_logit(track_h[1]),
            },
        },
        "record_sha256": "3" * 64,
    }


def test_i0_raw_score_bits_decode_exactly_and_fail_closed() -> None:
    assert residual.raw_score_from_bits("3ff0000000000000") == 1.0
    negative_zero = residual.raw_score_from_bits("8000000000000000")
    assert residual.float_bits(negative_zero) == "8000000000000000"

    for bad in (
        "3FF0000000000000",
        "3ff000000000000",
        "3ff00000000000000",
        "not-binary64-hex",
        "7ff0000000000000",
        "7ff8000000000000",
    ):
        with pytest.raises(residual.ComponentResidualError):
            residual.raw_score_from_bits(bad)


def test_fixed_binary64_candidate_means_and_unit_residual_order() -> None:
    init_pair = (-7.312715117751976, 6.9486747387446535)
    track_h_pair = (5.275492379532281, -4.898619485211566)
    raw_score = 0.123456789
    candidate = _candidate(init=init_pair, track_h=track_h_pair)

    component = residual.candidate_component_scores(candidate)
    expected_init = 0.5 * (init_pair[0] + init_pair[1])
    expected_track_h = 0.5 * (track_h_pair[0] + track_h_pair[1])
    expected_fused = raw_score + (expected_track_h - expected_init)
    reordered_per_direction = raw_score + 0.5 * (
        (track_h_pair[0] - init_pair[0])
        + (track_h_pair[1] - init_pair[1])
    )
    assert expected_fused != reordered_per_direction
    assert component == {"INIT": expected_init, "TRACK_H": expected_track_h}

    payload = residual.build_fused_candidate_payload(_bits(raw_score), candidate)
    assert tuple(payload["scores"]) == ("RAW", "FUSED")
    assert payload["scores"]["RAW"] == {
        "score": raw_score,
        "score_bits": _bits(raw_score),
    }
    assert payload["scores"]["FUSED"] == {
        "score": expected_fused,
        "score_bits": _bits(expected_fused),
    }


def test_payload_preserves_candidate_address_and_corrected_identity() -> None:
    candidate = _candidate()
    payload = residual.build_fused_candidate_payload(_bits(2.5), candidate)
    for key in (
        "candidate_key",
        "candidate_physical_row",
        "candidate_reference_source_sha256",
        "candidate_corrected_identity",
    ):
        assert payload[key] == candidate[key]

    absent_reference = _candidate()
    del absent_reference["candidate_reference_source_sha256"]
    payload_without_reference = residual.build_fused_candidate_payload(
        _bits(2.5), absent_reference
    )
    assert "candidate_reference_source_sha256" not in payload_without_reference


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_key", "", "candidate key"),
        ("candidate_physical_row", True, "physical row"),
        ("candidate_physical_row", -1, "physical row"),
        ("candidate_corrected_identity", "", "corrected identity"),
        ("candidate_reference_source_sha256", "bad-sha", "reference source"),
    ],
)
def test_candidate_address_drift_fails_closed(
    field: str, value: object, message: str
) -> None:
    candidate = _candidate()
    candidate[field] = value
    with pytest.raises(residual.ComponentResidualError, match=message):
        residual.build_fused_candidate_payload(_bits(1.0), candidate)


def test_every_pj1_directional_bit_receipt_and_axis_is_checked() -> None:
    for arm in residual.SOURCE_ARMS:
        for direction in residual.DIRECTIONS:
            candidate = copy.deepcopy(_candidate())
            candidate["raw_directional_logits"][arm][direction][
                "raw_logit_bits"
            ] = "0" * 16
            with pytest.raises(
                residual.ComponentResidualError, match="raw_logit_bits"
            ):
                residual.build_fused_candidate_payload(_bits(1.0), candidate)

    extra_arm = copy.deepcopy(_candidate())
    extra_arm["raw_directional_logits"]["OTHER"] = copy.deepcopy(
        extra_arm["raw_directional_logits"]["INIT"]
    )
    with pytest.raises(residual.ComponentResidualError, match="arm axis"):
        residual.build_fused_candidate_payload(_bits(1.0), extra_arm)

    extra_direction = copy.deepcopy(_candidate())
    extra_direction["raw_directional_logits"]["TRACK_H"]["other"] = (
        _raw_logit(0.0)
    )
    with pytest.raises(residual.ComponentResidualError, match="direction axis"):
        residual.build_fused_candidate_payload(_bits(1.0), extra_direction)


def test_only_the_frozen_unit_residual_api_is_exposed() -> None:
    assert residual.RESIDUAL_WEIGHT == 1.0
    assert tuple(
        inspect.signature(residual.build_fused_candidate_payload).parameters
    ) == ("raw_score_bits", "pj1_candidate")
    assert tuple(inspect.signature(residual.fused_score).parameters) == (
        "raw_score",
        "init_component_score",
        "track_h_component_score",
    )
    with pytest.raises(TypeError):
        residual.build_fused_candidate_payload(
            _bits(1.0), _candidate(), weight=2.0
        )
    with pytest.raises(TypeError):
        residual.fused_score(1.0, 0.0, 0.5, cutoff=0.25)


def test_bool_and_nonfinite_component_values_fail_closed() -> None:
    with pytest.raises(residual.ComponentResidualError, match="JSON number"):
        residual.direction_mean(True, 0.0)
    with pytest.raises(residual.ComponentResidualError, match="finite"):
        residual.fused_score(float("inf"), 0.0, 1.0)
