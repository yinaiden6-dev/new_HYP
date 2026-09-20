#!/usr/bin/env python3
"""Independent validation for the optimization-only H0 bias fit.

The validator deliberately does not import the producer or calibration core.
It reconstructs recipient-supergroup weights, solves the one-dimensional
convex optimum by bracketed bisection, and recomputes all global/LOFO losses
and margin invariants from raw JSON logits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SCHEMA = "rc_dino_rcde_h0_u4_raw_score_aggregate_v2_20260823"
AGGREGATE_STATUS = "H0_U4_V2_RAW_SCORE_AGGREGATE_QUARANTINED"
RESULT_SCHEMA = "rc_dino_rcde_h0_bias_calibrator_fit_v1_20260824"
RESULT_STATUS = "H0_BIAS_CALIBRATOR_OPTIMIZATION_READY"
RESULT_ABORT_STATUS = "H0_BIAS_CALIBRATOR_OPTIMIZATION_ABORT"
RESULT_CLAIM = "OPTIMIZATION_ONLY_BIAS_CALIBRATOR_FIT_NOT_CONFIRMATION"
U4_ROLE = "OPTIMIZATION_ONLY_NOT_CONFIRMATION"
VALIDATION_SCHEMA = "rc_dino_rcde_h0_bias_calibrator_fit_validation_v1_20260824"
VALIDATION_STATUS = "H0_BIAS_CALIBRATOR_OPTIMIZATION_INDEPENDENT_VALIDATION_PASS"
VALIDATION_CLAIM = "INDEPENDENT_OPTIMIZATION_ONLY_BIAS_FIT_VALIDATION"
ARMS = ("INIT", "TRACK_R", "TRACK_H")
DIRECTIONS = ("a_to_b", "b_to_a")
FOLDS = (1, 2, 3, 4)


class BiasFitValidationError(RuntimeError):
    """Independent recomputation or envelope validation failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise BiasFitValidationError(message)


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


def _validate_aggregate(aggregate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    require(aggregate.get("schema_version") == AGGREGATE_SCHEMA, "aggregate schema drift")
    require(aggregate.get("status") == AGGREGATE_STATUS, "aggregate status drift")
    require(aggregate.get("logical_sha256") == logical_sha256(aggregate), "aggregate hash drift")
    require(aggregate.get("scientific_metric_count") == 0, "aggregate metric drift")
    require(aggregate.get("scientific_reducer_authorized") is False, "aggregate reducer drift")
    require(aggregate.get("quarantine_released") is False, "aggregate quarantine drift")
    rows = aggregate.get("rows")
    unmatched = aggregate.get("unmatched_rows")
    require(isinstance(rows, list) and rows, "empty/non-list aggregate rows")
    require(isinstance(unmatched, list), "unmatched row drift")
    require(aggregate.get("matched_count") == len(rows), "matched count drift")
    require(aggregate.get("unmatched_count") == len(unmatched), "unmatched count drift")
    require(aggregate.get("query_count") == len(rows) + len(unmatched), "query count drift")
    require(aggregate.get("row_population_sha256") == canonical_sha256(rows), "population hash drift")
    folds: set[int] = set()
    groups: set[str] = set()
    records: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"row {index} type drift")
        require(
            row.get("record_sha256")
            == canonical_sha256(
                {key: item for key, item in row.items() if key != "record_sha256"}
            ),
            f"row {index} hash drift",
        )
        record = str(row.get("outer_ledger_record_sha256", ""))
        require(len(record) == 64 and record not in records, "record identity drift")
        records.add(record)
        fold = int(row.get("outer_fold", -1))
        require(fold in FOLDS, "fold drift")
        folds.add(fold)
        group = str(row.get("recipient_supergroup_sha256", ""))
        require(len(group) == 64, "group drift")
        groups.add(group)
        require(set(row.get("arms", {})) == set(ARMS), "arm drift")
        for arm in ARMS:
            for role in ("target", "donor"):
                logits = row["arms"][arm].get(role)
                require(isinstance(logits, Mapping) and set(logits) == set(DIRECTIONS), "direction drift")
                require(all(math.isfinite(float(logits[d])) for d in DIRECTIONS), "non-finite logit")
        receipts = row.get("directional_receipts")
        require(
            isinstance(receipts, list)
            and len(receipts) == 2
            and {item.get("direction") for item in receipts} == set(DIRECTIONS)
            and all(item.get("structural_ready") is True for item in receipts),
            "structural-ready receipt drift",
        )
    require(folds == set(FOLDS) and groups, "four-fold/group population drift")
    return rows


def _observations(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> list[tuple[float, float, float]]:
    counts = Counter(str(row["recipient_supergroup_sha256"]) for row in rows)
    output: list[tuple[float, float, float]] = []
    for row in rows:
        weight = 1.0 / (4.0 * counts[str(row["recipient_supergroup_sha256"])])
        values = row["arms"][arm]
        output.extend((float(values["target"][d]), 1.0, weight) for d in DIRECTIONS)
        output.extend((float(values["donor"][d]), 0.0, weight) for d in DIRECTIONS)
    return output


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_negative = math.exp(-value)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(value)
    return exp_positive / (1.0 + exp_positive)


def _loss(observations: Sequence[tuple[float, float, float]], bias: float) -> float:
    denominator = math.fsum(weight for _, _, weight in observations)
    return math.fsum(
        weight * (_softplus(logit + bias) - label * (logit + bias))
        for logit, label, weight in observations
    ) / denominator


def _gradient_hessian(
    observations: Sequence[tuple[float, float, float]], bias: float
) -> tuple[float, float]:
    denominator = math.fsum(weight for _, _, weight in observations)
    probabilities = [(_sigmoid(logit + bias), label, weight) for logit, label, weight in observations]
    gradient = math.fsum(weight * (probability - label) for probability, label, weight in probabilities) / denominator
    hessian = math.fsum(weight * probability * (1.0 - probability) for probability, _, weight in probabilities) / denominator
    return gradient, hessian


def _independent_optimum(
    observations: Sequence[tuple[float, float, float]],
) -> float:
    """Solve the strictly convex intercept problem without the producer solver."""

    lower, upper = -1.0, 1.0
    lower_gradient, _ = _gradient_hessian(observations, lower)
    upper_gradient, _ = _gradient_hessian(observations, upper)
    while lower_gradient >= 0.0:
        lower *= 2.0
        require(lower > -1024.0, "failed to bracket optimum below")
        lower_gradient, _ = _gradient_hessian(observations, lower)
    while upper_gradient <= 0.0:
        upper *= 2.0
        require(upper < 1024.0, "failed to bracket optimum above")
        upper_gradient, _ = _gradient_hessian(observations, upper)
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        gradient, _ = _gradient_hessian(observations, midpoint)
        if gradient < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= 2e-14:
            break
    return 0.5 * (lower + upper)


def _population(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "recipient_supergroup_count": len({str(row["recipient_supergroup_sha256"]) for row in rows}),
        "binary_observation_count": 4 * len(rows),
        "positive_observation_count": 2 * len(rows),
        "negative_observation_count": 2 * len(rows),
    }


def _biases(rows: Sequence[Mapping[str, Any]], bias: float | Mapping[int, float]) -> list[float]:
    if isinstance(bias, Mapping):
        return [float(bias[int(row["outer_fold"])]) for row in rows]
    return [float(bias)] * len(rows)


def _metrics(
    rows: Sequence[Mapping[str, Any]], bias: float | Mapping[int, float]
) -> dict[str, Any]:
    row_biases = _biases(rows, bias)
    raw_h = _observations(rows, "TRACK_H")
    raw_init = _observations(rows, "INIT")
    calibrated_h: list[tuple[float, float, float]] = []
    for row_bias, offset in zip(row_biases, range(0, len(raw_h), 4), strict=True):
        calibrated_h.extend(
            (logit + row_bias, label, weight)
            for logit, label, weight in raw_h[offset : offset + 4]
        )
    raw_h_loss = _loss(raw_h, 0.0)
    calibrated_h_loss = _loss(calibrated_h, 0.0)
    raw_init_loss = _loss(raw_init, 0.0)

    counts = Counter(str(row["recipient_supergroup_sha256"]) for row in rows)
    raw_margins: list[float] = []
    calibrated_margins: list[float] = []
    weights: list[float] = []
    for row, row_bias in zip(rows, row_biases, strict=True):
        weight = 1.0 / (2.0 * counts[str(row["recipient_supergroup_sha256"])])
        value = row["arms"]["TRACK_H"]
        for direction in DIRECTIONS:
            target = float(value["target"][direction])
            donor = float(value["donor"][direction])
            raw_margins.append(target - donor)
            calibrated_margins.append((target + row_bias) - (donor + row_bias))
            weights.append(weight)
    denominator = math.fsum(weights)
    raw_mean = math.fsum(w * x for w, x in zip(weights, raw_margins, strict=True)) / denominator
    calibrated_mean = math.fsum(w * x for w, x in zip(weights, calibrated_margins, strict=True)) / denominator
    return {
        **_population(rows),
        "raw_track_h_loss": raw_h_loss,
        "calibrated_track_h_loss": calibrated_h_loss,
        "raw_init_loss": raw_init_loss,
        "improvement_vs_raw_track_h": raw_h_loss - calibrated_h_loss,
        "improvement_vs_init": raw_init_loss - calibrated_h_loss,
        "raw_group_balanced_mean_margin": raw_mean,
        "calibrated_group_balanced_mean_margin": calibrated_mean,
        "max_abs_directional_margin_change": max(
            abs(after - before)
            for before, after in zip(raw_margins, calibrated_margins, strict=True)
        ),
    }


def _close(actual: Any, expected: Any, path: str, tolerance: float = 3e-13) -> None:
    if isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance),
            f"numeric drift at {path}: {actual!r} != {expected!r}",
        )
    elif isinstance(expected, dict):
        require(isinstance(actual, Mapping) and set(actual) == set(expected), f"key drift at {path}")
        for key in expected:
            _close(actual[key], expected[key], f"{path}.{key}", tolerance)
    else:
        require(type(actual) is type(expected) and actual == expected, f"value drift at {path}")


def _validate_fit(
    actual: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], path: str
) -> float:
    expected_keys = {
        "bias", "initial_loss", "final_loss", "gradient", "hessian", "iterations", "converged"
    }
    require(set(actual) == expected_keys, f"fit key drift at {path}")
    observations = _observations(rows, "TRACK_H")
    optimum = _independent_optimum(observations)
    bias = float(actual["bias"])
    require(math.isclose(bias, optimum, rel_tol=0.0, abs_tol=5e-11), f"bias optimum drift at {path}")
    gradient, hessian = _gradient_hessian(observations, bias)
    expected = {
        "bias": bias,
        "initial_loss": _loss(observations, 0.0),
        "final_loss": _loss(observations, bias),
        "gradient": gradient,
        "hessian": hessian,
    }
    for key, value in expected.items():
        _close(actual[key], value, f"{path}.{key}", 5e-12)
    require(actual["converged"] is True, f"non-converged fit at {path}")
    require(isinstance(actual["iterations"], int) and 0 <= actual["iterations"] <= 64, f"iteration drift at {path}")
    require(abs(gradient) <= 2e-11 and hessian > 0.0, f"stationarity drift at {path}")
    require(float(actual["final_loss"]) <= float(actual["initial_loss"]), f"loss increase at {path}")
    return bias


def validate_result(
    aggregate: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    aggregate_file_sha256: str,
    result_file_sha256: str | None = None,
) -> dict[str, Any]:
    rows = _validate_aggregate(aggregate)
    require(result.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    require(result.get("status") in {RESULT_STATUS, RESULT_ABORT_STATUS}, "result status drift")
    require(result.get("claim_level") == RESULT_CLAIM, "result claim drift")
    require(result.get("U4_role") == U4_ROLE, "U4 role drift")
    require(result.get("logical_sha256") == logical_sha256(result), "result logical hash drift")
    source = result.get("source_aggregate")
    require(
        isinstance(source, Mapping)
        and source.get("file_sha256") == aggregate_file_sha256
        and source.get("logical_sha256") == aggregate["logical_sha256"]
        and source.get("schema_version") == AGGREGATE_SCHEMA
        and source.get("status") == AGGREGATE_STATUS,
        "source aggregate binding drift",
    )
    require(result.get("fit_arm") == "TRACK_H", "fit arm drift")
    require(result.get("calibration_transform") == "z_calibrated = z_raw + b", "transform drift")
    require(result.get("objective") == "recipient-supergroup-balanced binary log loss", "objective drift")
    _close(
        result.get("weighting"),
        {
            "scheme": "EQUAL_RECIPIENT_SUPERGROUP_MASS",
            "observation_weight": "1 / (4 * rows_in_recipient_supergroup)",
            "target_label": 1,
            "donor_label": 0,
        },
        "weighting",
    )
    _close(
        result.get("population"),
        {**_population(rows), "outer_fold_count": 4, "unmatched_row_count": len(aggregate["unmatched_rows"])},
        "population",
    )

    global_record = result.get("global_fit")
    require(isinstance(global_record, Mapping) and set(global_record) == {"fit", "metrics"}, "global fit drift")
    global_bias = _validate_fit(global_record["fit"], rows, "global_fit.fit")
    _close(global_record["metrics"], _metrics(rows, global_bias), "global_fit.metrics")

    lofo = result.get("lofo_fits")
    require(isinstance(lofo, Mapping) and set(lofo) == {str(fold) for fold in FOLDS}, "LOFO key drift")
    lofo_biases: dict[int, float] = {}
    improvements: dict[str, float] = {}
    for fold in FOLDS:
        record = lofo[str(fold)]
        require(
            isinstance(record, Mapping)
            and set(record) == {"excluded_outer_fold", "training_population", "heldout_population", "fit", "heldout_metrics"}
            and record["excluded_outer_fold"] == fold,
            f"LOFO envelope drift fold {fold}",
        )
        training = [row for row in rows if int(row["outer_fold"]) != fold]
        heldout = [row for row in rows if int(row["outer_fold"]) == fold]
        _close(record["training_population"], _population(training), f"lofo.{fold}.training_population")
        _close(record["heldout_population"], _population(heldout), f"lofo.{fold}.heldout_population")
        bias = _validate_fit(record["fit"], training, f"lofo.{fold}.fit")
        expected_metrics = _metrics(heldout, bias)
        _close(record["heldout_metrics"], expected_metrics, f"lofo.{fold}.heldout_metrics")
        lofo_biases[fold] = bias
        improvements[str(fold)] = expected_metrics["improvement_vs_raw_track_h"]

    combined = _metrics(rows, lofo_biases)
    _close(result.get("combined_lofo_oof_metrics"), combined, "combined_lofo_oof_metrics")
    _close(result.get("heldout_fold_improvement_vs_raw_track_h"), improvements, "fold_improvements")
    all_folds_improve = all(value > 0.0 for value in improvements.values())
    require(
        result.get("all_heldout_folds_improve_raw_track_h") is all_folds_improve,
        "held-out improvement flag drift",
    )

    all_biases = [global_bias, *lofo_biases.values()]
    expected_range = {
        "global_bias": global_bias,
        "lofo_min_bias": min(lofo_biases.values()),
        "lofo_max_bias": max(lofo_biases.values()),
        "all_fit_min_bias": min(all_biases),
        "all_fit_max_bias": max(all_biases),
    }
    _close(result.get("bias_range"), expected_range, "bias_range")
    expected_margin = {
        "theoretical_identity": "(target+b)-(donor+b)=target-donor",
        "common_additive_bias_cancels": True,
        "global_max_abs_directional_margin_change": global_record["metrics"]["max_abs_directional_margin_change"],
        "lofo_oof_max_abs_directional_margin_change": combined["max_abs_directional_margin_change"],
        "lofo_heldout_max_abs_directional_margin_change": max(
            float(lofo[str(fold)]["heldout_metrics"]["max_abs_directional_margin_change"])
            for fold in FOLDS
        ),
    }
    _close(result.get("margin_invariance"), expected_margin, "margin_invariance")
    expected_gates = {
        "combined_lofo_improvement_vs_init_at_least_0p01": combined[
            "improvement_vs_init"
        ]
        >= 0.01,
        "all_four_heldout_folds_improve_raw_track_h": all_folds_improve,
        "lofo_bias_range_at_most_0p15": max(lofo_biases.values())
        - min(lofo_biases.values())
        <= 0.15,
        "global_margin_max_abs_at_most_1e_12": expected_margin[
            "global_max_abs_directional_margin_change"
        ]
        <= 1e-12,
        "lofo_margin_max_abs_at_most_1e_12": expected_margin[
            "lofo_heldout_max_abs_directional_margin_change"
        ]
        <= 1e-12,
        "global_final_loss_not_above_initial": float(
            global_record["fit"]["final_loss"]
        )
        <= float(global_record["fit"]["initial_loss"]),
    }
    _close(result.get("qualification_gates"), expected_gates, "qualification_gates")
    gates_pass = all(expected_gates.values())
    require(result.get("all_qualification_gates_pass") is gates_pass, "qualification summary drift")
    require(
        result.get("status") == (RESULT_STATUS if gates_pass else RESULT_ABORT_STATUS),
        "READY/ABORT decision drift",
    )
    require(result.get("structural_h0_exact_zero_contract") is True, "structural H0 contract drift")
    require(result.get("structural_h0_probe_output") == 0.0, "structural H0 probe drift")
    for key in ("model_forward_count", "model_backward_count", "model_update_count", "scientific_metric_count"):
        require(result.get(key) == 0, f"nonzero {key}")
    require(
        result.get("calibrator_consumption_authorized") is False
        and result.get("retrieval_or_ownership_claim_authorized") is False
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False
        and result.get("next_authorized_stage") is None,
        "claim/advance boundary drift",
    )

    validation: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "claim_level": VALIDATION_CLAIM,
        "validation_pass": True,
        "U4_role": U4_ROLE,
        "aggregate_logical_sha256": aggregate["logical_sha256"],
        "aggregate_file_sha256": aggregate_file_sha256,
        "result_logical_sha256": result["logical_sha256"],
        "result_file_sha256": result_file_sha256,
        "group_balanced_weights_recomputed": True,
        "global_and_four_lofo_optima_recomputed": True,
        "heldout_metrics_recomputed": True,
        "combined_lofo_vs_init_improvement": combined["improvement_vs_init"],
        "all_four_heldout_folds_improve_raw_track_h": all_folds_improve,
        "all_qualification_gates_recomputed_pass": gates_pass,
        "margin_invariance_recomputed": True,
        "structural_h0_exact_zero_contract": True,
        "model_forward_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical_sha256(validation)
    return validation


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = read_json(args.aggregate)
    result = read_json(args.result)
    validation = validate_result(
        aggregate,
        result,
        aggregate_file_sha256=file_sha256(args.aggregate),
        result_file_sha256=file_sha256(args.result),
    )
    atomic_json(args.output, validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "combined_lofo_vs_init_improvement": validation[
                    "combined_lofo_vs_init_improvement"
                ],
                "scientific_GO_or_NO_GO": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
