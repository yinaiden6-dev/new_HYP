from __future__ import annotations

import copy
import inspect
import struct

import pytest

from rc_aslo_xf import dino_rcde_h0_component_scale4_v1 as scale4


def _raw(value: float) -> dict[str, object]:
    number = float(value)
    return {
        "raw_logit": number,
        "raw_logit_bits": struct.pack(">d", number).hex(),
    }


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
            "INIT": {"a_to_b": _raw(init[0]), "b_to_a": _raw(init[1])},
            "TRACK_H": {
                "a_to_b": _raw(track_h[0]),
                "b_to_a": _raw(track_h[1]),
            },
        },
        "record_sha256": "3" * 64,
    }


def test_single_scale_and_candidate_level_binary64_operation_order() -> None:
    assert scale4.SCALE == 4.0
    init_pair = (-7.312715117751976, 6.9486747387446535)
    track_h_pair = (5.275492379532281, -4.898619485211566)
    payload = scale4.build_scale4_candidate_payload(
        _candidate(init=init_pair, track_h=track_h_pair)
    )
    init = 0.5 * (init_pair[0] + init_pair[1])
    track_h = 0.5 * (track_h_pair[0] + track_h_pair[1])
    expected = init + 4.0 * (track_h - init)
    per_direction_then_mean = 0.5 * (
        (init_pair[0] + 4.0 * (track_h_pair[0] - init_pair[0]))
        + (init_pair[1] + 4.0 * (track_h_pair[1] - init_pair[1]))
    )
    assert expected != per_direction_then_mean
    assert payload["scores"]["SCALE4"]["score"] == expected
    assert payload["scores"]["SCALE4"]["score_bits"] == struct.pack(
        ">d", expected
    ).hex()


def test_payload_is_pj2_score_compatible_and_preserves_candidate_address() -> None:
    candidate = _candidate(init=(0.1, 0.2), track_h=(0.2, 0.6))
    payload = scale4.build_scale4_candidate_payload(candidate)
    assert payload["candidate_key"] == candidate["candidate_key"]
    assert payload["candidate_physical_row"] == candidate["candidate_physical_row"]
    assert (
        payload["candidate_reference_source_sha256"]
        == candidate["candidate_reference_source_sha256"]
    )
    assert payload["candidate_corrected_identity"] == candidate[
        "candidate_corrected_identity"
    ]
    assert tuple(payload["scores"]) == ("INIT", "TRACK_H", "SCALE4")
    assert payload["scores"]["INIT"]["score"] == scale4.candidate_score(0.1, 0.2)
    assert payload["scores"]["TRACK_H"]["score"] == scale4.candidate_score(
        0.2, 0.6
    )
    for score_payload in payload["scores"].values():
        assert set(score_payload) == {"score", "score_bits"}
        assert score_payload["score_bits"] == struct.pack(
            ">d", score_payload["score"]
        ).hex()


@pytest.mark.parametrize(
    ("baseline", "comparison", "expected"),
    [
        (False, True, "rescue"),
        (True, False, "break"),
        (True, True, "both_correct"),
        (False, False, "both_wrong"),
    ],
)
def test_pair_transition_helper(
    baseline: bool, comparison: bool, expected: str
) -> None:
    assert scale4.pair_transition(baseline, comparison) == expected


def test_every_raw_logit_bit_receipt_is_checked_fail_closed() -> None:
    for arm in scale4.SOURCE_ARMS:
        for direction in scale4.DIRECTIONS:
            candidate = copy.deepcopy(_candidate())
            candidate["raw_directional_logits"][arm][direction][
                "raw_logit_bits"
            ] = "0" * 16
            with pytest.raises(scale4.ComponentScale4Error, match="raw_logit_bits"):
                scale4.build_scale4_candidate_payload(candidate)


def test_source_axes_extra_arm_and_extra_direction_fail_closed() -> None:
    extra_arm = copy.deepcopy(_candidate())
    extra_arm["raw_directional_logits"]["SCALE3"] = copy.deepcopy(
        extra_arm["raw_directional_logits"]["INIT"]
    )
    with pytest.raises(scale4.ComponentScale4Error, match="arm axis"):
        scale4.build_scale4_candidate_payload(extra_arm)

    extra_direction = copy.deepcopy(_candidate())
    extra_direction["raw_directional_logits"]["TRACK_H"]["third"] = _raw(0.0)
    with pytest.raises(scale4.ComponentScale4Error, match="direction axis"):
        scale4.build_scale4_candidate_payload(extra_direction)


def test_no_alternative_scale_or_cutoff_api_is_exposed() -> None:
    assert tuple(inspect.signature(scale4.scale4_score).parameters) == (
        "init_score",
        "track_h_score",
    )
    assert tuple(
        inspect.signature(scale4.build_scale4_candidate_payload).parameters
    ) == ("candidate",)
    with pytest.raises(TypeError):
        scale4.scale4_score(0.0, 1.0, scale=3.0)
    with pytest.raises(TypeError):
        scale4.build_scale4_candidate_payload(_candidate(), scale=3.0)
    with pytest.raises(TypeError):
        scale4.pair_transition(False, True, cutoff=0.5)


def test_nonfinite_bool_and_bad_optional_reference_receipt_fail_closed() -> None:
    with pytest.raises(scale4.ComponentScale4Error, match="JSON number"):
        scale4.candidate_score(True, 0.0)
    with pytest.raises(scale4.ComponentScale4Error, match="finite"):
        scale4.scale4_score(float("inf"), 0.0)
    bad_reference = _candidate()
    bad_reference["candidate_reference_source_sha256"] = "not-a-sha"
    with pytest.raises(scale4.ComponentScale4Error, match="reference source"):
        scale4.build_scale4_candidate_payload(bad_reference)

    absent_reference = _candidate()
    del absent_reference["candidate_reference_source_sha256"]
    payload = scale4.build_scale4_candidate_payload(absent_reference)
    assert "candidate_reference_source_sha256" not in payload
