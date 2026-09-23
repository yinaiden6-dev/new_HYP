from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "programs"))
sys.path.insert(0, str(RC_ROOT / "src"))

import fit_dino_rcde_h0_bias_calibrator_v1 as producer  # noqa: E402
import validate_dino_rcde_h0_bias_calibrator_v1 as validator  # noqa: E402
from rc_aslo_xf.dino_rcde_h0_bias_calibrator_v1 import (  # noqa: E402
    apply_bias_calibrator,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _row(fold: int, ordinal: int, group: str) -> dict[str, object]:
    # Fold-specific offsets make LOFO genuinely cross-fitted while retaining a
    # stable shared intercept.  The +0.20 target/donor margin is never fitted.
    offset = -0.01 * fold
    arms = {
        "INIT": {
            "checkpoint_sha256": _sha(f"init:{fold}"),
            "target": {"a_to_b": 0.0, "b_to_a": 0.0},
            "donor": {"a_to_b": 0.0, "b_to_a": 0.0},
        },
        "TRACK_R": {
            "checkpoint_sha256": _sha(f"r:{fold}"),
            "target": {"a_to_b": -0.05, "b_to_a": -0.05},
            "donor": {"a_to_b": -0.15, "b_to_a": -0.15},
        },
        "TRACK_H": {
            "checkpoint_sha256": _sha(f"h:{fold}"),
            "target": {"a_to_b": -0.30 + offset, "b_to_a": -0.30 + offset},
            "donor": {"a_to_b": -0.50 + offset, "b_to_a": -0.50 + offset},
        },
    }
    row: dict[str, object] = {
        "arms": arms,
        "directional_receipts": [
            {"direction": "a_to_b", "structural_ready": True},
            {"direction": "b_to_a", "structural_ready": True},
        ],
        "outer_fold": fold,
        "outer_ledger_record_sha256": _sha(f"outer:{ordinal}"),
        "recipient_supergroup_sha256": group,
        "query_id": f"fixture-{ordinal}",
    }
    row["record_sha256"] = producer.canonical_sha256(row)
    return row


def _aggregate() -> dict[str, object]:
    rows = []
    ordinal = 0
    # Unequal row counts prove that observations, rather than groups, are not
    # accidentally given equal mass.
    for fold, count in enumerate((1, 2, 3, 4), start=1):
        group = _sha(f"group:{fold}")
        for _ in range(count):
            rows.append(_row(fold, ordinal, group))
            ordinal += 1
    value: dict[str, object] = {
        "schema_version": producer.AGGREGATE_SCHEMA,
        "status": producer.AGGREGATE_STATUS,
        "scientific_metric_count": 0,
        "scientific_reducer_authorized": False,
        "quarantine_released": False,
        "matched_count": len(rows),
        "unmatched_count": 0,
        "query_count": len(rows),
        "rows": rows,
        "unmatched_rows": [],
        "row_population_sha256": producer.canonical_sha256(rows),
    }
    value["logical_sha256"] = producer.logical_sha256(value)
    return value


def test_group_balanced_weights_give_each_supergroup_equal_mass() -> None:
    aggregate = _aggregate()
    rows = producer.validate_aggregate(aggregate)
    weights = producer.group_balanced_observation_weights(rows)
    assert weights.dtype == torch.float64
    offset = 0
    group_mass: dict[str, float] = {}
    for row in rows:
        group = str(row["recipient_supergroup_sha256"])
        group_mass[group] = group_mass.get(group, 0.0) + float(weights[offset : offset + 4].sum())
        offset += 4
    assert set(group_mass.values()) == {1.0}
    assert float(weights.sum()) == 4.0


def test_producer_and_independent_validator_close_global_lofo_and_gates() -> None:
    aggregate = _aggregate()
    source_sha = _sha("fixture aggregate file")
    result = producer.build_result(
        aggregate,
        aggregate_path="fixture/aggregate.json",
        aggregate_file_sha256=source_sha,
    )
    assert result["status"] == producer.RESULT_STATUS
    assert result["U4_role"] == "OPTIMIZATION_ONLY_NOT_CONFIRMATION"
    assert result["all_qualification_gates_pass"] is True
    assert all(result["qualification_gates"].values())
    assert len(result["lofo_fits"]) == 4
    assert result["combined_lofo_oof_metrics"]["improvement_vs_init"] >= 0.01
    assert all(
        value > 0.0
        for value in result["heldout_fold_improvement_vs_raw_track_h"].values()
    )
    assert result["bias_range"]["lofo_max_bias"] - result["bias_range"]["lofo_min_bias"] <= 0.15
    assert result["margin_invariance"]["global_max_abs_directional_margin_change"] <= 1e-12
    assert result["margin_invariance"]["lofo_heldout_max_abs_directional_margin_change"] <= 1e-12
    assert result["structural_h0_exact_zero_contract"] is True
    assert result["structural_h0_probe_output"] == 0.0
    assert result["model_forward_count"] == 0
    assert result["scientific_GO_or_NO_GO"] is None

    validation = validator.validate_result(
        aggregate,
        result,
        aggregate_file_sha256=source_sha,
    )
    assert validation["status"] == validator.VALIDATION_STATUS
    assert validation["validation_pass"] is True
    assert validation["all_qualification_gates_recomputed_pass"] is True
    assert validation["model_forward_count"] == 0
    assert validation["scientific_GO_or_NO_GO"] is None


def test_independent_validator_rejects_tampered_bias_even_with_rehashed_envelope() -> None:
    aggregate = _aggregate()
    source_sha = _sha("fixture aggregate file")
    result = producer.build_result(
        aggregate,
        aggregate_path="fixture/aggregate.json",
        aggregate_file_sha256=source_sha,
    )
    tampered = copy.deepcopy(result)
    tampered["global_fit"]["fit"]["bias"] += 0.01
    tampered["logical_sha256"] = producer.logical_sha256(tampered)
    with pytest.raises(validator.BiasFitValidationError, match="bias optimum drift"):
        validator.validate_result(
            aggregate,
            tampered,
            aggregate_file_sha256=source_sha,
        )


def test_failed_qualification_is_published_as_abort_and_recomputed() -> None:
    aggregate = _aggregate()
    # Make fold 4 require a very different intercept.  This is still a valid
    # optimization input, but it must not receive READY status.
    for row in aggregate["rows"]:
        if row["outer_fold"] == 4:
            for role in ("target", "donor"):
                for direction in producer.DIRECTIONS:
                    row["arms"]["TRACK_H"][role][direction] -= 1.0
            row["record_sha256"] = producer.canonical_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
    aggregate["row_population_sha256"] = producer.canonical_sha256(aggregate["rows"])
    aggregate["logical_sha256"] = producer.logical_sha256(aggregate)
    source_sha = _sha("unstable fixture aggregate")
    result = producer.build_result(
        aggregate,
        aggregate_path="fixture/unstable.json",
        aggregate_file_sha256=source_sha,
    )
    assert result["status"] == producer.RESULT_ABORT_STATUS
    assert result["all_qualification_gates_pass"] is False
    validation = validator.validate_result(
        aggregate,
        result,
        aggregate_file_sha256=source_sha,
    )
    assert validation["validation_pass"] is True
    assert validation["all_qualification_gates_recomputed_pass"] is False


def test_structural_h0_is_exact_zero_while_ready_score_gets_common_bias() -> None:
    score = torch.tensor([-0.4, -0.2], dtype=torch.float64)
    bias = 0.37
    assert torch.equal(
        apply_bias_calibrator(score, structural_ready=False, bias=bias),
        torch.zeros_like(score),
    )
    calibrated = apply_bias_calibrator(score, structural_ready=True, bias=bias)
    assert torch.allclose(calibrated, score + bias, rtol=0.0, atol=0.0)
    assert float((calibrated[0] - calibrated[1]) - (score[0] - score[1])) == pytest.approx(0.0, abs=1e-15)


def test_current_u4_aggregate_meets_registered_optimization_qualification() -> None:
    path = RC_ROOT / "results/dino_rcde_h0_u4_raw_aggregate_v2/result.json"
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    result = producer.build_result(
        aggregate,
        aggregate_path=str(path.relative_to(RC_ROOT)),
        aggregate_file_sha256=producer.file_sha256(path),
    )
    assert result["status"] == producer.RESULT_STATUS
    assert result["all_qualification_gates_pass"] is True
    assert result["global_fit"]["fit"]["bias"] == pytest.approx(0.384608865331, abs=2e-11)
    assert result["global_fit"]["metrics"]["raw_track_h_loss"] == pytest.approx(0.690322827249, abs=2e-12)
    assert result["global_fit"]["metrics"]["calibrated_track_h_loss"] == pytest.approx(0.672813396151, abs=2e-12)
    assert result["combined_lofo_oof_metrics"]["improvement_vs_init"] >= 0.01
    validation = validator.validate_result(
        aggregate,
        result,
        aggregate_file_sha256=producer.file_sha256(path),
    )
    assert validation["status"] == validator.VALIDATION_STATUS
    assert validation["all_qualification_gates_recomputed_pass"] is True
