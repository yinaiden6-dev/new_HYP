"""Frozen group-balanced statistics for Track-R decisions A and B."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "rc_dino_rcde_track_r_statistics_v1_20260821"
ARMS = (
    "ALL_PATCH_SAME_MODEL",
    "CW1_QUERY_MULTITILE_FULL_REFERENCE",
    "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET",
)
REGIONAL_ARMS = ARMS[1:]
MIN_GROUPS = 32
MIN_GROUPS_PER_FOLD = 6
PRIMARY_PERMUTATIONS = 9999
SEED = 17


class TrackRStatisticsError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRStatisticsError(message)


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    output = tuple(rows)
    require(bool(output), "Track-R statistical population is empty")
    addresses = set()
    for row in output:
        require(
            isinstance(row, Mapping)
            and isinstance(row.get("query_id"), str)
            and isinstance(row.get("group_sha256"), str)
            and row.get("outer_fold") in (1, 2, 3, 4)
            and isinstance(row.get("raw_margin"), (int, float))
            and math.isfinite(float(row["raw_margin"]))
            and set(row.get("arms", {})) == set(ARMS),
            "Track-R statistical row schema drift",
        )
        address = (row["query_id"], row["outer_fold"])
        require(address not in addresses, "Track-R statistical query address duplicate")
        addresses.add(address)
        for arm in ARMS:
            values = row["arms"][arm]
            require(
                isinstance(values, Mapping)
                and set(values) >= {"REAL", "C_DINO_V", "P_QUERY", "P_REFERENCE"}
                and all(
                    isinstance(values[key], (int, float))
                    and math.isfinite(float(values[key]))
                    for key in ("REAL", "C_DINO_V", "P_QUERY", "P_REFERENCE")
                ),
                f"Track-R {arm} control row drift",
            )
    return output


def _group_values(
    rows: Sequence[Mapping[str, Any]],
    value,
) -> tuple[np.ndarray, tuple[str, ...]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_sha256"])].append(float(value(row)))
    groups = tuple(sorted(grouped))
    values = np.asarray(
        [float(np.mean(grouped[group])) for group in groups], dtype=np.float64
    )
    require(np.isfinite(values).all(), "group statistic is nonfinite")
    return values, groups


def _summary(rows: Sequence[Mapping[str, Any]], margin) -> dict[str, float | int]:
    margins, groups = _group_values(rows, margin)
    directions, direction_groups = _group_values(
        rows, lambda row: float(margin(row) > 0.0)
    )
    require(groups == direction_groups, "group summary axis drift")
    return {
        "query_count": len(rows),
        "group_count": len(groups),
        "group_balanced_direction": float(directions.mean()),
        "group_balanced_mean_margin": float(margins.mean()),
    }


def _bootstrap_lower(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    require(values.ndim == 1 and values.size >= 2, "bootstrap needs at least two groups")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(repetitions, values.size))
    samples = values[indexes].mean(axis=1)
    return float(np.quantile(samples, 0.025, method="linear"))


def _signflip_p(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    require(values.ndim == 1 and values.size >= 2, "sign flip needs at least two groups")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, values.size))
    permuted = (signs * values[None, :]).mean(axis=1)
    return float((1 + int(np.count_nonzero(permuted >= observed))) / (repetitions + 1))


def _studentized(values: np.ndarray) -> float:
    if values.size < 2:
        return float("-inf")
    standard = float(values.std(ddof=1))
    if standard == 0.0:
        return float("inf") if float(values.mean()) > 0 else float("-inf")
    return float(values.mean() / (standard / math.sqrt(values.size)))


def _max_t_adjusted_p(
    values_by_arm: Mapping[str, np.ndarray],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    require(set(values_by_arm) == set(ARMS), "maxT arm set drift")
    sizes = {value.size for value in values_by_arm.values()}
    require(len(sizes) == 1 and next(iter(sizes)) >= 2, "maxT group axis drift")
    count = next(iter(sizes))
    observed = {arm: _studentized(value) for arm, value in values_by_arm.items()}
    rng = np.random.default_rng(seed)
    exceed = {arm: 0 for arm in ARMS}
    for _ in range(repetitions):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=count)
        maximum = max(_studentized(values_by_arm[arm] * signs) for arm in ARMS)
        for arm in ARMS:
            if maximum >= observed[arm]:
                exceed[arm] += 1
    return {
        arm: float((1 + exceed[arm]) / (repetitions + 1)) for arm in ARMS
    }


def _fold_directions(rows: Sequence[Mapping[str, Any]], margin) -> dict[str, float]:
    output = {}
    for fold in (1, 2, 3, 4):
        selected = [row for row in rows if row["outer_fold"] == fold]
        directions, groups = _group_values(
            selected, lambda row: float(margin(row) > 0.0)
        )
        require(len(groups) >= MIN_GROUPS_PER_FOLD, f"fold{fold} group floor failed")
        output[str(fold)] = float(directions.mean())
    return output


def _control_gate(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    control: str,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    difference = lambda row: float(
        row["arms"][arm]["REAL"] - row["arms"][arm][control]
    )
    margin_diff, groups = _group_values(rows, difference)
    direction_real, _ = _group_values(
        rows, lambda row: float(row["arms"][arm]["REAL"] > 0.0)
    )
    direction_control, _ = _group_values(
        rows, lambda row: float(row["arms"][arm][control] > 0.0)
    )
    real_wins, _ = _group_values(rows, lambda row: float(difference(row) > 0.0))
    fold_mean = {}
    for fold in (1, 2, 3, 4):
        fold_rows = [row for row in rows if row["outer_fold"] == fold]
        values, fold_groups = _group_values(fold_rows, difference)
        require(len(fold_groups) >= MIN_GROUPS_PER_FOLD, "control fold group floor failed")
        fold_mean[str(fold)] = float(values.mean())
    lower = _bootstrap_lower(margin_diff, seed=seed, repetitions=repetitions)
    p_value = _signflip_p(margin_diff, seed=seed, repetitions=repetitions)
    direction_drop = float(direction_real.mean() - direction_control.mean())
    gate = (
        direction_drop >= 0.05
        and float(margin_diff.mean()) > 0.0
        and lower > 0.0
        and p_value < 0.05
        and float(real_wins.mean()) >= 0.60
        and sum(value > 0.0 for value in fold_mean.values()) >= 3
    )
    return {
        "group_count": len(groups),
        "direction_drop": direction_drop,
        "mean_margin_drop": float(margin_diff.mean()),
        "cluster_bootstrap_lower": lower,
        "signflip_p": p_value,
        "real_greater_control_case_fraction": float(real_wins.mean()),
        "fold_mean_drop": fold_mean,
        "gate": gate,
    }


def evaluate_track_r(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = PRIMARY_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    population = _validate_rows(rows)
    groups = {str(row["group_sha256"]) for row in population}
    fold_groups = {
        fold: {str(row["group_sha256"]) for row in population if row["outer_fold"] == fold}
        for fold in (1, 2, 3, 4)
    }
    require(len(groups) >= MIN_GROUPS, "Track-R statistical group floor failed")
    require(all(len(value) >= MIN_GROUPS_PER_FOLD for value in fold_groups.values()), "Track-R fold group floor failed")
    raw_wrong = tuple(row for row in population if float(row["raw_margin"]) <= 0.0)
    raw_correct = tuple(row for row in population if float(row["raw_margin"]) > 0.0)
    require(bool(raw_wrong) and bool(raw_correct), "RAW strata are empty")
    direction_values: dict[str, np.ndarray] = {}
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        margin = lambda row, arm=arm: float(row["arms"][arm]["REAL"])
        overall = _summary(population, margin)
        wrong = _summary(raw_wrong, margin)
        correct = _summary(raw_correct, margin)
        directions, _ = _group_values(
            population, lambda row, margin=margin: float(margin(row) > 0.0) - 0.5
        )
        direction_values[arm] = directions
        folds = _fold_directions(population, margin)
        c_gate = _control_gate(
            population, arm, "C_DINO_V", seed=seed + 101, repetitions=repetitions
        )
        p_query = _control_gate(
            population, arm, "P_QUERY", seed=seed + 211, repetitions=repetitions
        )
        p_reference = _control_gate(
            population, arm, "P_REFERENCE", seed=seed + 307, repetitions=repetitions
        )
        base_gate = (
            overall["group_balanced_direction"] >= 0.60
            and overall["group_balanced_mean_margin"] > 0.0
            and wrong["group_balanced_direction"] >= 0.60
            and wrong["group_balanced_mean_margin"] > 0.0
            and correct["group_balanced_direction"] >= 0.70
            and sum(value > 0.50 for value in folds.values()) >= 3
            and c_gate["gate"]
        )
        spatial = base_gate and p_query["gate"] and p_reference["gate"]
        arm_results[arm] = {
            "overall": overall,
            "RAW_wrong": wrong,
            "RAW_correct": correct,
            "fold_direction": folds,
            "C_DINO_V": c_gate,
            "P_QUERY": p_query,
            "P_REFERENCE": p_reference,
            "base_gate_before_maxT": base_gate,
            "spatial_gate_before_maxT": spatial,
        }
    adjusted = _max_t_adjusted_p(
        direction_values, seed=seed + 401, repetitions=repetitions
    )
    any_candidate_bound = False
    any_spatial = False
    for arm in ARMS:
        arm_results[arm]["maxT_adjusted_p"] = adjusted[arm]
        arm_results[arm]["candidate_bound_gate"] = (
            arm_results[arm]["base_gate_before_maxT"] and adjusted[arm] < 0.05
        )
        arm_results[arm]["spatial_gate"] = (
            arm_results[arm]["spatial_gate_before_maxT"] and adjusted[arm] < 0.05
        )
        any_candidate_bound |= arm_results[arm]["candidate_bound_gate"]
        any_spatial |= arm_results[arm]["spatial_gate"]

    regional_results = {}
    for arm_index, arm in enumerate(REGIONAL_ARMS):
        diff_margin = lambda row, arm=arm: float(
            row["arms"][arm]["REAL"]
            - row["arms"]["ALL_PATCH_SAME_MODEL"]["REAL"]
        )
        diffs, _ = _group_values(population, diff_margin)
        regional_dir, _ = _group_values(
            population, lambda row, arm=arm: float(row["arms"][arm]["REAL"] > 0.0)
        )
        all_dir, _ = _group_values(
            population,
            lambda row: float(row["arms"]["ALL_PATCH_SAME_MODEL"]["REAL"] > 0.0),
        )
        rescues = sum(
            row["arms"]["ALL_PATCH_SAME_MODEL"]["REAL"] <= 0.0
            and row["arms"][arm]["REAL"] > 0.0
            for row in population
        )
        breaks = sum(
            row["arms"]["ALL_PATCH_SAME_MODEL"]["REAL"] > 0.0
            and row["arms"][arm]["REAL"] <= 0.0
            for row in population
        )
        fold_diff = {}
        for fold in (1, 2, 3, 4):
            values, _ = _group_values(
                [row for row in population if row["outer_fold"] == fold], diff_margin
            )
            fold_diff[str(fold)] = float(values.mean())
        lower = _bootstrap_lower(
            diffs, seed=seed + 503 + arm_index, repetitions=repetitions
        )
        p_value = _signflip_p(
            diffs, seed=seed + 607 + arm_index, repetitions=repetitions
        )
        direction_gain = float(regional_dir.mean() - all_dir.mean())
        gate = (
            arm_results[arm]["candidate_bound_gate"]
            and direction_gain >= 0.03
            and float(diffs.mean()) > 0.0
            and lower > 0.0
            and p_value < 0.05
            and sum(value > 0.0 for value in fold_diff.values()) >= 3
            and rescues > breaks
        )
        regional_results[arm] = {
            "direction_gain_over_allpatch": direction_gain,
            "mean_margin_gain": float(diffs.mean()),
            "cluster_bootstrap_lower": lower,
            "signflip_p": p_value,
            "fold_mean_gain": fold_diff,
            "rescues": int(rescues),
            "breaks": int(breaks),
            "gate": gate,
        }

    if any_spatial:
        decision_a = "DINO_SPECIFIC_REFERENCE_EVIDENCE_GO"
        spatial_boundary = "DINO_SPATIAL_CORRESPONDENCE_GO"
    elif any_candidate_bound:
        decision_a = "DINO_SPECIFIC_REFERENCE_EVIDENCE_GO"
        spatial_boundary = "CANDIDATE_BOUND_NONSPATIAL_ONLY"
    else:
        decision_a = "DINO_SPECIFIC_REFERENCE_EVIDENCE_NO_GO"
        spatial_boundary = "NO_SPATIAL_CLAIM"
    decision_b = (
        "P_LOCK_REGIONAL_INCREMENT_GO"
        if any(item["gate"] for item in regional_results.values())
        else "P_LOCK_REGIONAL_INCREMENT_NO_INCREMENT"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "query_count": len(population),
        "group_count": len(groups),
        "fold_group_count": {str(key): len(value) for key, value in fold_groups.items()},
        "RAW_wrong_query_count": len(raw_wrong),
        "RAW_correct_query_count": len(raw_correct),
        "permutations": repetitions,
        "seed": seed,
        "arms": arm_results,
        "regional_increment": regional_results,
        "decision_A": decision_a,
        "spatial_boundary": spatial_boundary,
        "decision_B": decision_b,
    }


__all__ = [
    "SCHEMA_VERSION",
    "ARMS",
    "REGIONAL_ARMS",
    "MIN_GROUPS",
    "MIN_GROUPS_PER_FOLD",
    "PRIMARY_PERMUTATIONS",
    "SEED",
    "TrackRStatisticsError",
    "evaluate_track_r",
]
