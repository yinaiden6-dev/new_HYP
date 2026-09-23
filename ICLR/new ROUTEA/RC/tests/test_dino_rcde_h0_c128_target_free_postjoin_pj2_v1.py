from __future__ import annotations

import copy
import struct

import pytest

import reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as reducer
import validate_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as validator


def _raw(value: float) -> dict[str, object]:
    number = float(value)
    return {"raw_logit": number, "raw_logit_bits": struct.pack(">d", number).hex()}


def _candidate(
    row: int,
    label: str,
    init_score: float,
    h_score: float,
    *,
    init_pair: tuple[float, float] | None = None,
    h_pair: tuple[float, float] | None = None,
) -> dict[str, object]:
    init = init_pair or (init_score, init_score)
    track_h = h_pair or (h_score, h_score)
    value: dict[str, object] = {
        "source_candidate_record_sha256": f"{row + 1:064x}",
        "candidate_key": f"candidate-{row}",
        "candidate_physical_row": row,
        "candidate_reference_source_sha256": f"{10_000 + row:064x}",
        "candidate_corrected_identity": label,
        "raw_directional_logits": {
            "INIT": {
                "a_to_b": _raw(init[0]),
                "b_to_a": _raw(init[1]),
            },
            "TRACK_H": {
                "a_to_b": _raw(track_h[0]),
                "b_to_a": _raw(track_h[1]),
            },
        },
    }
    value["record_sha256"] = reducer.record_sha(value)
    return value


def _source(
    ordinal: int,
    fold: int,
    group: str,
    candidates: list[dict[str, object]],
    target: str = "target",
) -> dict[str, object]:
    value: dict[str, object] = {
        "source_prejoin_record_sha256": f"{20_000 + ordinal:064x}",
        "execution_ordinal": ordinal,
        "query_id": f"Q-{ordinal:04d}",
        "query_source_image_sha256": f"{30_000 + ordinal:064x}",
        "outer_fold": fold,
        "supergroup": group,
        "track": "fixture",
        "target_corrected_identity": target,
        "candidate_axis_sha256": f"{40_000 + ordinal:064x}",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_population_sha256": reducer.canonical(candidates),
    }
    value["record_sha256"] = reducer.record_sha(value)
    return value


def _go_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal in range(8):
        fold = ordinal // 2 + 1
        rows.append(
            _source(
                ordinal,
                fold,
                f"group-{ordinal}",
                [
                    _candidate(0, "target", 2.0, 4.0),
                    _candidate(1, "wrong-a", 3.0, 1.0),
                    _candidate(2, "wrong-b", 1.0, 3.0),
                ],
            )
        )
    return rows


def test_registered_direction_mean_and_big_endian_bits_are_exact() -> None:
    score = reducer.candidate_score(0.1, 0.2)
    assert score == 0.5 * (0.1 + 0.2)
    assert reducer.float_bits(score) == struct.pack(">d", score).hex()
    candidate = _candidate(
        0,
        "target",
        0.0,
        0.0,
        init_pair=(0.1, 0.2),
    )
    arm = reducer.reduce_arm([candidate, _candidate(1, "wrong", 0.0, 0.0)], "INIT", "target")
    assert arm["target"]["score"] == score
    assert arm["target"]["score_bits"] == struct.pack(">d", score).hex()


def test_same_identity_max_and_exact_tie_choose_smaller_physical_row() -> None:
    candidates = [
        _candidate(9, "target", 1.0, 1.0),
        _candidate(2, "target", 1.0, 1.0),
        _candidate(7, "wrong", 0.0, 0.0),
    ]
    arm = reducer.reduce_arm(candidates, "INIT", "target")
    assert arm["reduced_identity_count"] == 2
    assert arm["target"]["physical_row"] == 2
    assert arm["target"]["score"] == 1.0
    independent = validator.independently_reduce_arm(candidates, "INIT", "target")
    assert independent == arm


def test_strongest_wrong_order_and_strict_tie_rank() -> None:
    candidates = [
        _candidate(10, "target", 1.0, 1.0),
        _candidate(4, "wrong-z", 1.0, 1.0),
        _candidate(3, "wrong-a", 1.0, 1.0),
        _candidate(8, "lower", 0.0, 0.0),
    ]
    arm = reducer.reduce_arm(candidates, "INIT", "target")
    assert arm["strongest_wrong"]["corrected_identity"] == "wrong-a"
    assert arm["margin"] == 0.0
    assert arm["strict_top1"] is False
    assert arm["rank"] == 3
    assert arm["reciprocal_rank"] == 1.0 / 3.0


def test_each_arm_owns_its_strongest_wrong_and_no_margin_difference_is_emitted() -> None:
    reduced = reducer.reduce_rows(_go_rows())
    first = reduced["rows"][0]
    assert first["arms"]["INIT"]["strongest_wrong"]["corrected_identity"] == "wrong-a"
    assert first["arms"]["TRACK_H"]["strongest_wrong"]["corrected_identity"] == "wrong-b"
    assert first["transition"] == "init_wrong_h_correct"
    assert first["paired_top1_difference"] == 1
    assert first["paired_reciprocal_rank_difference"] == 0.5
    assert "paired_margin_difference" not in first
    assert "margin_difference" not in first


def test_paired_group_statistics_and_all_registered_go_gates() -> None:
    reduced = reducer.reduce_rows(_go_rows())
    assert reduced["supergroup_paired_differences"]["top1"] == {
        f"group-{index}": 1.0 for index in range(8)
    }
    assert reduced["top1_inference"]["group_balanced_gain"] == 1.0
    assert reduced["top1_inference"]["bootstrap_95_ci"] == [1.0, 1.0]
    assert reduced["top1_inference"]["one_sided_group_signflip_p"] < 0.05
    assert all(
        summary["group_balanced_top1_gain"] == 1.0
        and summary["group_balanced_mrr_gain"] == 0.5
        for summary in reduced["fold_summaries"].values()
    )
    assert reduced["transition_counts"] == {
        "init_wrong_h_correct": 8,
        "init_correct_h_wrong": 0,
        "both": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    assert all(reduced["gates"].values())
    assert reduced["all_gates_pass"] is True


def test_group_balance_is_not_query_balance() -> None:
    rows = [
        {"supergroup": "large", "d": 1.0},
        {"supergroup": "large", "d": 1.0},
        {"supergroup": "large", "d": 1.0},
        {"supergroup": "small", "d": -1.0},
    ]
    groups = reducer.group_means(rows, "d")
    assert groups == {"large": 1.0, "small": -1.0}
    assert sum(groups.values()) / len(groups) == 0.0


def test_independent_statistics_replay_matches_registered_reducer() -> None:
    reduced = reducer.reduce_rows(_go_rows())
    top1 = reduced["supergroup_paired_differences"]["top1"]
    mrr = reduced["supergroup_paired_differences"]["reciprocal_rank"]
    assert validator.independently_infer(top1) == reduced["top1_inference"]
    assert validator.independently_infer(mrr) == reduced["mrr_inference"]
    for source, output in zip(_go_rows(), reduced["rows"], strict=True):
        for arm in reducer.ARMS:
            assert validator.independently_reduce_arm(
                source["candidates"], arm, source["target_corrected_identity"]
            ) == output["arms"][arm]


def test_tampered_raw_bit_pattern_fails_closed() -> None:
    candidates = [_candidate(0, "target", 1.0, 1.0), _candidate(1, "wrong", 0.0, 0.0)]
    bad = copy.deepcopy(candidates)
    bad[0]["raw_directional_logits"]["INIT"]["a_to_b"]["raw_logit_bits"] = "0" * 16
    with pytest.raises(reducer.PJ2Error, match="bit-pattern"):
        reducer.reduce_arm(bad, "INIT", "target")
    with pytest.raises(validator.PJ2ValidationError, match="bits"):
        validator.independently_reduce_arm(bad, "INIT", "target")


def test_no_go_is_terminal_and_never_auto_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, object]] = []
    for ordinal in range(8):
        rows.append(
            _source(
                ordinal,
                ordinal // 2 + 1,
                f"group-{ordinal}",
                [
                    _candidate(0, "target", 2.0, 2.0),
                    _candidate(1, "wrong-a", 3.0, 3.0),
                    _candidate(2, "wrong-b", 1.0, 1.0),
                ],
            )
        )
    reduced = reducer.reduce_rows(rows)
    assert reduced["all_gates_pass"] is False
    authority = {"logical_sha256": "a" * 64}
    pj1: dict[str, object] = {
        "namespace": reducer.NAMESPACE,
        "status": reducer.PJ1_STATUS,
        "query_count": 594,
        "candidate_count": 76_032,
        "rows": [],
        "row_population_sha256": reducer.canonical([]),
    }
    pj1["logical_sha256"] = reducer.logical(pj1)
    monkeypatch.setattr(
        reducer,
        "reduce_rows",
        lambda source_rows, exact_population: reduced,
    )
    result = reducer.build_result(authority, "b" * 64, pj1, "c" * 64)
    assert result["status"] == reducer.NO_GO_STATUS
    assert result["scientific_GO_or_NO_GO"] == "NO_GO"
    assert result["automatic_stage_advance"] is False
    assert result["next_authorized_stage"] is None
    assert result["forbidden_cross_arm_margin_difference_computed"] is False
