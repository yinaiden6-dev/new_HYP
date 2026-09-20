#!/usr/bin/env python3
"""Fit a scalar additive bias to the optimization-only U4 Track-H logits.

The U4 aggregate is reused strictly as optimization data.  This program does
not run a model, does not alter target-minus-donor margins, and does not make
or confirm a U4 scientific decision.  It emits a whole-population diagnostic
fit plus four leave-one-outer-fold-out (LOFO) fits and held-out metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_h0_bias_calibrator_v1 import (  # noqa: E402
    apply_bias_calibrator,
    fit_bias_calibrator,
    weighted_log_loss,
)


AGGREGATE_SCHEMA = "rc_dino_rcde_h0_u4_raw_score_aggregate_v2_20260823"
AGGREGATE_STATUS = "H0_U4_V2_RAW_SCORE_AGGREGATE_QUARANTINED"
RESULT_SCHEMA = "rc_dino_rcde_h0_bias_calibrator_fit_v1_20260824"
RESULT_STATUS = "H0_BIAS_CALIBRATOR_OPTIMIZATION_READY"
RESULT_ABORT_STATUS = "H0_BIAS_CALIBRATOR_OPTIMIZATION_ABORT"
RESULT_CLAIM = "OPTIMIZATION_ONLY_BIAS_CALIBRATOR_FIT_NOT_CONFIRMATION"
U4_ROLE = "OPTIMIZATION_ONLY_NOT_CONFIRMATION"
ARMS = ("INIT", "TRACK_R", "TRACK_H")
FIT_ARM = "TRACK_H"
DIRECTIONS = ("a_to_b", "b_to_a")
FOLDS = (1, 2, 3, 4)


class BiasFitPipelineError(RuntimeError):
    """The raw aggregate or calibration result violates the fixed contract."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise BiasFitPipelineError(message)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_aggregate(aggregate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Fail closed on anything other than a pristine raw U4 aggregate."""

    require(aggregate.get("schema_version") == AGGREGATE_SCHEMA, "aggregate schema drift")
    require(aggregate.get("status") == AGGREGATE_STATUS, "aggregate status drift")
    require(
        aggregate.get("logical_sha256") == logical_sha256(aggregate),
        "aggregate logical hash drift",
    )
    require(aggregate.get("scientific_metric_count") == 0, "aggregate was reduced")
    require(
        aggregate.get("scientific_reducer_authorized") is False,
        "aggregate reducer flag drift",
    )
    require(aggregate.get("quarantine_released") is False, "aggregate quarantine drift")
    rows = aggregate.get("rows")
    unmatched = aggregate.get("unmatched_rows")
    require(isinstance(rows, list) and rows, "empty/non-list matched rows")
    require(isinstance(unmatched, list), "non-list unmatched rows")
    require(aggregate.get("matched_count") == len(rows), "matched count drift")
    require(aggregate.get("unmatched_count") == len(unmatched), "unmatched count drift")
    require(
        aggregate.get("query_count") == len(rows) + len(unmatched),
        "query count drift",
    )
    require(
        aggregate.get("row_population_sha256") == canonical_sha256(rows),
        "row population hash drift",
    )

    seen: set[str] = set()
    observed_folds: set[int] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"row {index} is not an object")
        record_sha = str(row.get("record_sha256", ""))
        require(
            len(record_sha) == 64
            and record_sha
            == canonical_sha256(
                {key: item for key, item in row.items() if key != "record_sha256"}
            ),
            f"row {index} record hash drift",
        )
        outer_record = str(row.get("outer_ledger_record_sha256", ""))
        require(len(outer_record) == 64 and outer_record not in seen, "row identity drift")
        seen.add(outer_record)
        fold = int(row.get("outer_fold", -1))
        require(fold in FOLDS, f"row {index} outer fold drift")
        observed_folds.add(fold)
        group = row.get("recipient_supergroup_sha256")
        require(isinstance(group, str) and len(group) == 64, f"row {index} group drift")
        arms = row.get("arms")
        require(isinstance(arms, Mapping) and set(arms) == set(ARMS), "arm population drift")
        for arm in ARMS:
            arm_value = arms[arm]
            require(isinstance(arm_value, Mapping), f"row {index} arm drift")
            for role in ("target", "donor"):
                values = arm_value.get(role)
                require(
                    isinstance(values, Mapping) and set(values) == set(DIRECTIONS),
                    f"row {index} {arm}/{role} direction drift",
                )
                require(
                    all(math.isfinite(float(values[direction])) for direction in DIRECTIONS),
                    f"row {index} has non-finite logits",
                )
        receipts = row.get("directional_receipts")
        require(isinstance(receipts, list) and len(receipts) == 2, "receipt drift")
        require(
            {item.get("direction") for item in receipts} == set(DIRECTIONS)
            and all(item.get("structural_ready") is True for item in receipts),
            "matched row contains structural H0 direction",
        )
    require(observed_folds == set(FOLDS), "four-fold population drift")
    return rows


def group_balanced_observation_weights(
    rows: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Give every recipient supergroup equal total mass.

    Each row supplies four binary observations (two positive target directions
    and two negative donor directions).  An observation in a group with n rows
    receives weight 1/(4n), so each represented group has total weight one.
    """

    require(bool(rows), "cannot weight an empty population")
    counts = Counter(str(row["recipient_supergroup_sha256"]) for row in rows)
    weights: list[float] = []
    for row in rows:
        weight = 1.0 / (4.0 * counts[str(row["recipient_supergroup_sha256"])])
        weights.extend((weight, weight, weight, weight))
    return torch.tensor(weights, dtype=torch.float64)


def arm_tensors(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(arm in ARMS, f"unknown arm {arm}")
    logits: list[float] = []
    labels: list[float] = []
    for row in rows:
        value = row["arms"][arm]
        logits.extend(float(value["target"][direction]) for direction in DIRECTIONS)
        labels.extend((1.0, 1.0))
        logits.extend(float(value["donor"][direction]) for direction in DIRECTIONS)
        labels.extend((0.0, 0.0))
    return (
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(labels, dtype=torch.float64),
        group_balanced_observation_weights(rows),
    )


def _population(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "recipient_supergroup_count": len(
            {str(row["recipient_supergroup_sha256"]) for row in rows}
        ),
        "binary_observation_count": 4 * len(rows),
        "positive_observation_count": 2 * len(rows),
        "negative_observation_count": 2 * len(rows),
    }


def _fit_payload(fit: Any) -> dict[str, Any]:
    value = {
        "bias": float(fit.bias),
        "initial_loss": float(fit.initial_loss),
        "final_loss": float(fit.final_loss),
        "gradient": float(fit.gradient),
        "hessian": float(fit.hessian),
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
    }
    require(all(math.isfinite(value[key]) for key in ("bias", "initial_loss", "final_loss", "gradient", "hessian")), "non-finite fit")
    require(value["converged"], "bias optimizer did not converge")
    require(value["final_loss"] <= value["initial_loss"], "optimizer increased loss")
    require(value["hessian"] > 0.0, "non-positive fitted Hessian")
    return value


def _row_biases(
    rows: Sequence[Mapping[str, Any]], bias: float | Mapping[int, float]
) -> list[float]:
    if isinstance(bias, Mapping):
        return [float(bias[int(row["outer_fold"])]) for row in rows]
    return [float(bias)] * len(rows)


def evaluate(
    rows: Sequence[Mapping[str, Any]], bias: float | Mapping[int, float]
) -> dict[str, Any]:
    """Evaluate group-balanced BCE and additive-margin invariance."""

    h_logits, labels, weights = arm_tensors(rows, FIT_ARM)
    init_logits, init_labels, init_weights = arm_tensors(rows, "INIT")
    require(torch.equal(labels, init_labels) and torch.equal(weights, init_weights), "metric axis drift")
    biases = _row_biases(rows, bias)
    observation_bias = torch.tensor(
        [item for row_bias in biases for item in (row_bias,) * 4],
        dtype=torch.float64,
    )
    raw_h_loss = float(weighted_log_loss(h_logits, labels, weights, bias=0.0))
    calibrated_h_loss = float(
        weighted_log_loss(h_logits + observation_bias, labels, weights, bias=0.0)
    )
    raw_init_loss = float(
        weighted_log_loss(init_logits, init_labels, init_weights, bias=0.0)
    )

    raw_margins: list[float] = []
    calibrated_margins: list[float] = []
    margin_weights: list[float] = []
    counts = Counter(str(row["recipient_supergroup_sha256"]) for row in rows)
    for row, row_bias in zip(rows, biases, strict=True):
        group = str(row["recipient_supergroup_sha256"])
        direction_weight = 1.0 / (2.0 * counts[group])
        arm_value = row["arms"][FIT_ARM]
        for direction in DIRECTIONS:
            target = float(arm_value["target"][direction])
            donor = float(arm_value["donor"][direction])
            raw_margins.append(target - donor)
            calibrated_margins.append((target + row_bias) - (donor + row_bias))
            margin_weights.append(direction_weight)
    margin_weight = math.fsum(margin_weights)
    raw_mean_margin = math.fsum(
        weight * margin for weight, margin in zip(margin_weights, raw_margins, strict=True)
    ) / margin_weight
    calibrated_mean_margin = math.fsum(
        weight * margin
        for weight, margin in zip(margin_weights, calibrated_margins, strict=True)
    ) / margin_weight
    max_abs_margin_change = max(
        abs(after - before)
        for before, after in zip(raw_margins, calibrated_margins, strict=True)
    )
    return {
        **_population(rows),
        "raw_track_h_loss": raw_h_loss,
        "calibrated_track_h_loss": calibrated_h_loss,
        "raw_init_loss": raw_init_loss,
        "improvement_vs_raw_track_h": raw_h_loss - calibrated_h_loss,
        "improvement_vs_init": raw_init_loss - calibrated_h_loss,
        "raw_group_balanced_mean_margin": raw_mean_margin,
        "calibrated_group_balanced_mean_margin": calibrated_mean_margin,
        "max_abs_directional_margin_change": max_abs_margin_change,
    }


def build_result(
    aggregate: Mapping[str, Any],
    *,
    aggregate_path: str,
    aggregate_file_sha256: str,
) -> dict[str, Any]:
    rows = validate_aggregate(aggregate)
    h_logits, labels, weights = arm_tensors(rows, FIT_ARM)
    global_fit = _fit_payload(fit_bias_calibrator(h_logits, labels, weights))
    global_metrics = evaluate(rows, global_fit["bias"])

    lofo_fits: dict[str, Any] = {}
    lofo_biases: dict[int, float] = {}
    fold_improvements: dict[str, float] = {}
    for fold in FOLDS:
        training = [row for row in rows if int(row["outer_fold"]) != fold]
        heldout = [row for row in rows if int(row["outer_fold"]) == fold]
        train_logits, train_labels, train_weights = arm_tensors(training, FIT_ARM)
        fit = _fit_payload(
            fit_bias_calibrator(train_logits, train_labels, train_weights)
        )
        metrics = evaluate(heldout, fit["bias"])
        lofo_biases[fold] = fit["bias"]
        fold_improvements[str(fold)] = metrics["improvement_vs_raw_track_h"]
        lofo_fits[str(fold)] = {
            "excluded_outer_fold": fold,
            "training_population": _population(training),
            "heldout_population": _population(heldout),
            "fit": fit,
            "heldout_metrics": metrics,
        }

    combined_oof = evaluate(rows, lofo_biases)
    all_biases = [global_fit["bias"], *lofo_biases.values()]
    structural_probe = apply_bias_calibrator(
        torch.tensor(1.23456789, dtype=torch.float64),
        structural_ready=False,
        bias=global_fit["bias"],
    )
    require(float(structural_probe) == 0.0, "structural H0 calibration drift")
    all_fold_improvements_positive = all(value > 0.0 for value in fold_improvements.values())
    lofo_margin_max_abs = max(
        float(lofo_fits[str(fold)]["heldout_metrics"]["max_abs_directional_margin_change"])
        for fold in FOLDS
    )
    qualification_gates = {
        "combined_lofo_improvement_vs_init_at_least_0p01": combined_oof[
            "improvement_vs_init"
        ]
        >= 0.01,
        "all_four_heldout_folds_improve_raw_track_h": all_fold_improvements_positive,
        "lofo_bias_range_at_most_0p15": max(lofo_biases.values())
        - min(lofo_biases.values())
        <= 0.15,
        "global_margin_max_abs_at_most_1e_12": global_metrics[
            "max_abs_directional_margin_change"
        ]
        <= 1e-12,
        "lofo_margin_max_abs_at_most_1e_12": lofo_margin_max_abs <= 1e-12,
        "global_final_loss_not_above_initial": global_fit["final_loss"]
        <= global_fit["initial_loss"],
    }
    all_qualification_gates_pass = all(qualification_gates.values())
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": RESULT_STATUS if all_qualification_gates_pass else RESULT_ABORT_STATUS,
        "claim_level": RESULT_CLAIM,
        "U4_role": U4_ROLE,
        "source_aggregate": {
            "path": aggregate_path,
            "file_sha256": aggregate_file_sha256,
            "logical_sha256": aggregate["logical_sha256"],
            "schema_version": aggregate["schema_version"],
            "status": aggregate["status"],
        },
        "fit_arm": FIT_ARM,
        "calibration_transform": "z_calibrated = z_raw + b",
        "objective": "recipient-supergroup-balanced binary log loss",
        "weighting": {
            "scheme": "EQUAL_RECIPIENT_SUPERGROUP_MASS",
            "observation_weight": "1 / (4 * rows_in_recipient_supergroup)",
            "target_label": 1,
            "donor_label": 0,
        },
        "population": {
            **_population(rows),
            "outer_fold_count": 4,
            "unmatched_row_count": len(aggregate["unmatched_rows"]),
        },
        "global_fit": {"fit": global_fit, "metrics": global_metrics},
        "lofo_fits": lofo_fits,
        "combined_lofo_oof_metrics": combined_oof,
        "heldout_fold_improvement_vs_raw_track_h": fold_improvements,
        "all_heldout_folds_improve_raw_track_h": all_fold_improvements_positive,
        "qualification_gates": qualification_gates,
        "all_qualification_gates_pass": all_qualification_gates_pass,
        "bias_range": {
            "global_bias": global_fit["bias"],
            "lofo_min_bias": min(lofo_biases.values()),
            "lofo_max_bias": max(lofo_biases.values()),
            "all_fit_min_bias": min(all_biases),
            "all_fit_max_bias": max(all_biases),
        },
        "margin_invariance": {
            "theoretical_identity": "(target+b)-(donor+b)=target-donor",
            "common_additive_bias_cancels": True,
            "global_max_abs_directional_margin_change": global_metrics[
                "max_abs_directional_margin_change"
            ],
            "lofo_oof_max_abs_directional_margin_change": combined_oof[
                "max_abs_directional_margin_change"
            ],
            "lofo_heldout_max_abs_directional_margin_change": lofo_margin_max_abs,
        },
        "structural_h0_exact_zero_contract": True,
        "structural_h0_probe_output": float(structural_probe),
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
        "scientific_metric_count": 0,
        "calibrator_consumption_authorized": False,
        "retrieval_or_ownership_claim_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(
                json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        require(not path.exists(), f"output appeared during publication: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = read_json(args.aggregate)
    try:
        aggregate_path = str(args.aggregate.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        aggregate_path = str(args.aggregate.resolve())
    result = build_result(
        aggregate,
        aggregate_path=aggregate_path,
        aggregate_file_sha256=file_sha256(args.aggregate),
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "global_bias": result["global_fit"]["fit"]["bias"],
                "all_heldout_folds_improve_raw_track_h": result[
                    "all_heldout_folds_improve_raw_track_h"
                ],
                "scientific_GO_or_NO_GO": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
