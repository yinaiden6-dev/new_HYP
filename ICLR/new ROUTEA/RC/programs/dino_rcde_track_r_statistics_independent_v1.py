#!/usr/bin/env python3
"""Independent Track-R statistics replay; never imports the production engine."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ARMS = (
    "ALL_PATCH_SAME_MODEL",
    "CW1_QUERY_MULTITILE_FULL_REFERENCE",
    "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET",
)
REGIONAL_ARMS = ARMS[1:]
PRIMARY_PERMUTATIONS = 9999
SEED = 17


class IndependentStatisticsError(RuntimeError):
    pass


def check(condition: Any, message: str) -> None:
    if not condition:
        raise IndependentStatisticsError(message)


def _population(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    values = tuple(rows)
    seen = set()
    for row in values:
        address = (row.get("query_id"), row.get("outer_fold"))
        check(
            isinstance(address[0], str)
            and address[1] in (1, 2, 3, 4)
            and isinstance(row.get("group_sha256"), str)
            and set(row.get("arms", {})) == set(ARMS)
            and address not in seen,
            "independent row schema/address drift",
        )
        seen.add(address)
        check(math.isfinite(float(row["raw_margin"])), "nonfinite RAW margin")
        for arm in ARMS:
            control = row["arms"][arm]
            check(
                set(control) >= {"REAL", "C_DINO_V", "P_QUERY", "P_REFERENCE"}
                and all(math.isfinite(float(control[key])) for key in ("REAL", "C_DINO_V", "P_QUERY", "P_REFERENCE")),
                "independent control schema drift",
            )
    check(bool(values), "empty population")
    return values


def _group_array(rows: Sequence[Mapping[str, Any]], fn: Callable[[Mapping[str, Any]], float]) -> tuple[np.ndarray, tuple[str, ...]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row["group_sha256"])].append(float(fn(row)))
    names = tuple(sorted(buckets))
    values = np.asarray([np.mean(buckets[name]) for name in names], dtype=np.float64)
    check(np.isfinite(values).all(), "nonfinite group values")
    return values, names


def _summary(rows: Sequence[Mapping[str, Any]], fn: Callable[[Mapping[str, Any]], float]) -> dict[str, Any]:
    margins, groups = _group_array(rows, fn)
    directions, same = _group_array(rows, lambda row: float(fn(row) > 0.0))
    check(groups == same, "summary group order drift")
    return {
        "query_count": len(rows),
        "group_count": len(groups),
        "group_balanced_direction": float(directions.mean()),
        "group_balanced_mean_margin": float(margins.mean()),
    }


def _bootstrap_lower(values: np.ndarray, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(repetitions, values.size))
    return float(np.quantile(values[indexes].mean(axis=1), 0.025, method="linear"))


def _signflip(values: np.ndarray, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, values.size))
    null = (signs * values[None, :]).mean(axis=1)
    return float((1 + np.count_nonzero(null >= values.mean())) / (repetitions + 1))


def _t(values: np.ndarray) -> float:
    standard = float(values.std(ddof=1))
    if standard == 0.0:
        return float("inf") if float(values.mean()) > 0.0 else float("-inf")
    return float(values.mean() / (standard / math.sqrt(values.size)))


def _max_t(arrays: Mapping[str, np.ndarray], seed: int, repetitions: int) -> dict[str, float]:
    observed = {arm: _t(arrays[arm]) for arm in ARMS}
    count = arrays[ARMS[0]].size
    rng = np.random.default_rng(seed)
    exceed = {arm: 0 for arm in ARMS}
    for _ in range(repetitions):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=count)
        maximum = max(_t(arrays[arm] * signs) for arm in ARMS)
        for arm in ARMS:
            exceed[arm] += int(maximum >= observed[arm])
    return {arm: float((1 + exceed[arm]) / (repetitions + 1)) for arm in ARMS}


def _fold_direction(rows: Sequence[Mapping[str, Any]], fn: Callable[[Mapping[str, Any]], float]) -> dict[str, float]:
    result = {}
    for fold in (1, 2, 3, 4):
        selected = [row for row in rows if row["outer_fold"] == fold]
        directions, groups = _group_array(selected, lambda row: float(fn(row) > 0.0))
        check(len(groups) >= 6, "independent fold group floor failed")
        result[str(fold)] = float(directions.mean())
    return result


def _destruction(rows: Sequence[Mapping[str, Any]], arm: str, control: str, seed: int, repetitions: int) -> dict[str, Any]:
    delta = lambda row: float(row["arms"][arm]["REAL"] - row["arms"][arm][control])
    drops, groups = _group_array(rows, delta)
    real_direction, _ = _group_array(rows, lambda row: float(row["arms"][arm]["REAL"] > 0.0))
    control_direction, _ = _group_array(rows, lambda row: float(row["arms"][arm][control] > 0.0))
    wins, _ = _group_array(rows, lambda row: float(delta(row) > 0.0))
    fold_drop = {}
    for fold in (1, 2, 3, 4):
        values, fold_groups = _group_array([row for row in rows if row["outer_fold"] == fold], delta)
        check(len(fold_groups) >= 6, "independent destruction fold floor failed")
        fold_drop[str(fold)] = float(values.mean())
    lower = _bootstrap_lower(drops, seed, repetitions)
    p_value = _signflip(drops, seed, repetitions)
    direction_drop = float(real_direction.mean() - control_direction.mean())
    gate = (
        direction_drop >= 0.05
        and float(drops.mean()) > 0.0
        and lower > 0.0
        and p_value < 0.05
        and float(wins.mean()) >= 0.60
        and sum(value > 0.0 for value in fold_drop.values()) >= 3
    )
    return {
        "group_count": len(groups),
        "direction_drop": direction_drop,
        "mean_margin_drop": float(drops.mean()),
        "cluster_bootstrap_lower": lower,
        "signflip_p": p_value,
        "real_greater_control_case_fraction": float(wins.mean()),
        "fold_mean_drop": fold_drop,
        "gate": gate,
    }


def independent_evaluate_track_r(rows: Sequence[Mapping[str, Any]], *, repetitions: int = PRIMARY_PERMUTATIONS, seed: int = SEED) -> dict[str, Any]:
    population = _population(rows)
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in population:
        group_folds[str(row["group_sha256"])].add(int(row["outer_fold"]))
    check(len(group_folds) >= 32 and all(len(value) == 1 for value in group_folds.values()), "independent group/fold isolation drift")
    fold_groups = {fold: {row["group_sha256"] for row in population if row["outer_fold"] == fold} for fold in (1, 2, 3, 4)}
    check(all(len(value) >= 6 for value in fold_groups.values()), "independent fold floor drift")
    wrong = tuple(row for row in population if float(row["raw_margin"]) <= 0.0)
    correct = tuple(row for row in population if float(row["raw_margin"]) > 0.0)
    check(bool(wrong) and bool(correct), "independent RAW strata empty")
    direction_arrays = {}
    arms = {}
    for arm in ARMS:
        margin = lambda row, arm=arm: float(row["arms"][arm]["REAL"])
        overall = _summary(population, margin)
        wrong_summary = _summary(wrong, margin)
        correct_summary = _summary(correct, margin)
        directions, _ = _group_array(population, lambda row, margin=margin: float(margin(row) > 0.0) - 0.5)
        direction_arrays[arm] = directions
        folds = _fold_direction(population, margin)
        c_gate = _destruction(population, arm, "C_DINO_V", seed + 101, repetitions)
        p_query = _destruction(population, arm, "P_QUERY", seed + 211, repetitions)
        p_reference = _destruction(population, arm, "P_REFERENCE", seed + 307, repetitions)
        base = (
            overall["group_balanced_direction"] >= 0.60
            and overall["group_balanced_mean_margin"] > 0.0
            and wrong_summary["group_balanced_direction"] >= 0.60
            and wrong_summary["group_balanced_mean_margin"] > 0.0
            and correct_summary["group_balanced_direction"] >= 0.70
            and sum(value > 0.50 for value in folds.values()) >= 3
            and c_gate["gate"]
        )
        arms[arm] = {
            "overall": overall,
            "RAW_wrong": wrong_summary,
            "RAW_correct": correct_summary,
            "fold_direction": folds,
            "C_DINO_V": c_gate,
            "P_QUERY": p_query,
            "P_REFERENCE": p_reference,
            "base_gate_before_maxT": base,
            "spatial_gate_before_maxT": base and p_query["gate"] and p_reference["gate"],
        }
    adjusted = _max_t(direction_arrays, seed + 401, repetitions)
    any_bound = any_spatial = False
    for arm in ARMS:
        arms[arm]["maxT_adjusted_p"] = adjusted[arm]
        arms[arm]["candidate_bound_gate"] = arms[arm]["base_gate_before_maxT"] and adjusted[arm] < 0.05
        arms[arm]["spatial_gate"] = arms[arm]["spatial_gate_before_maxT"] and adjusted[arm] < 0.05
        any_bound |= arms[arm]["candidate_bound_gate"]
        any_spatial |= arms[arm]["spatial_gate"]
    regional = {}
    for index, arm in enumerate(REGIONAL_ARMS):
        difference = lambda row, arm=arm: float(row["arms"][arm]["REAL"] - row["arms"][ARMS[0]]["REAL"])
        values, _ = _group_array(population, difference)
        regional_direction, _ = _group_array(population, lambda row, arm=arm: float(row["arms"][arm]["REAL"] > 0.0))
        all_direction, _ = _group_array(population, lambda row: float(row["arms"][ARMS[0]]["REAL"] > 0.0))
        rescues = sum(row["arms"][ARMS[0]]["REAL"] <= 0.0 < row["arms"][arm]["REAL"] for row in population)
        breaks = sum(row["arms"][ARMS[0]]["REAL"] > 0.0 >= row["arms"][arm]["REAL"] for row in population)
        fold_gain = {}
        for fold in (1, 2, 3, 4):
            fold_values, _ = _group_array([row for row in population if row["outer_fold"] == fold], difference)
            fold_gain[str(fold)] = float(fold_values.mean())
        lower = _bootstrap_lower(values, seed + 503 + index, repetitions)
        p_value = _signflip(values, seed + 607 + index, repetitions)
        direction_gain = float(regional_direction.mean() - all_direction.mean())
        gate = arms[arm]["candidate_bound_gate"] and direction_gain >= 0.03 and float(values.mean()) > 0.0 and lower > 0.0 and p_value < 0.05 and sum(value > 0.0 for value in fold_gain.values()) >= 3 and rescues > breaks
        regional[arm] = {
            "direction_gain_over_allpatch": direction_gain,
            "mean_margin_gain": float(values.mean()),
            "cluster_bootstrap_lower": lower,
            "signflip_p": p_value,
            "fold_mean_gain": fold_gain,
            "rescues": int(rescues),
            "breaks": int(breaks),
            "gate": gate,
        }
    if any_spatial:
        decision_a, boundary = "DINO_SPECIFIC_REFERENCE_EVIDENCE_GO", "DINO_SPATIAL_CORRESPONDENCE_GO"
    elif any_bound:
        decision_a, boundary = "DINO_SPECIFIC_REFERENCE_EVIDENCE_GO", "CANDIDATE_BOUND_NONSPATIAL_ONLY"
    else:
        decision_a, boundary = "DINO_SPECIFIC_REFERENCE_EVIDENCE_NO_GO", "NO_SPATIAL_CLAIM"
    decision_b = "P_LOCK_REGIONAL_INCREMENT_GO" if any(item["gate"] for item in regional.values()) else "P_LOCK_REGIONAL_INCREMENT_NO_INCREMENT"
    return {
        "schema_version": "rc_dino_rcde_track_r_statistics_v1_20260821",
        "query_count": len(population),
        "group_count": len(group_folds),
        "fold_group_count": {str(key): len(value) for key, value in fold_groups.items()},
        "RAW_wrong_query_count": len(wrong),
        "RAW_correct_query_count": len(correct),
        "permutations": repetitions,
        "seed": seed,
        "arms": arms,
        "regional_increment": regional,
        "decision_A": decision_a,
        "spatial_boundary": boundary,
        "decision_B": decision_b,
    }


__all__ = ["independent_evaluate_track_r"]
