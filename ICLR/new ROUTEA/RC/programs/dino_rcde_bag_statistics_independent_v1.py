#!/usr/bin/env python3
"""Independent implementation of the frozen BAG-completion statistics."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np


VALID_QUERY_COUNT = 594
FOLDS = (1, 2, 3, 4)
PERMUTATIONS = 9_999
SEED = 17


class IndependentBagStatisticsError(RuntimeError):
    pass


def check(condition: Any, message: str) -> None:
    if not condition:
        raise IndependentBagStatisticsError(message)


def _population(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    values = tuple(rows)
    seen: set[tuple[str, int]] = set()
    for row in values:
        address = (str(row.get("query_id")), int(row.get("outer_fold", -1)))
        check(address[0] and address[1] in FOLDS and address not in seen, "independent row address drift")
        seen.add(address)
        check(
            all(math.isfinite(float(row.get(name))) for name in ("raw_margin", "real_margin", "c_bind_margin")),
            "independent nonfinite margin",
        )
        check(isinstance(row.get("group_sha256"), str), "independent group absent")
    check(len(values) == VALID_QUERY_COUNT, "independent population is not 594")
    return values


def _groups(rows: Sequence[Mapping[str, Any]], fn) -> tuple[np.ndarray, tuple[str, ...]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row["group_sha256"])].append(float(fn(row)))
    names = tuple(sorted(buckets))
    values = np.asarray([np.mean(buckets[name]) for name in names], dtype=np.float64)
    check(bool(names) and bool(np.isfinite(values).all()), "independent group reduction drift")
    return values, names


def _summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    margins, names = _groups(rows, lambda row: float(row[key]))
    directions, same = _groups(rows, lambda row: float(float(row[key]) > 0.0))
    check(names == same, "independent summary order drift")
    return {
        "query_count": len(rows),
        "group_count": len(names),
        "group_balanced_direction": float(directions.mean()),
        "group_balanced_mean_margin": float(margins.mean()),
    }


def _bootstrap_lower(values: np.ndarray, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(repetitions, values.size))
    return float(np.quantile(values[indexes].mean(axis=1), 0.025, method="linear"))


def _t(values: np.ndarray) -> float:
    standard = float(values.std(ddof=1))
    if standard == 0.0:
        return float("inf") if float(values.mean()) > 0.0 else float("-inf")
    return float(values.mean() / (standard / math.sqrt(values.size)))


def _signflip(values: np.ndarray, seed: int, repetitions: int, studentized: bool = False) -> float:
    observed = _t(values) if studentized else float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, values.size))
    flipped = values[None, :] * signs
    if studentized:
        means = flipped.mean(axis=1)
        standard = flipped.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            null = means / (standard / math.sqrt(values.size))
        null = np.where(standard == 0.0, np.where(means > 0.0, np.inf, -np.inf), null)
    else:
        null = flipped.mean(axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (repetitions + 1))


def _tail(n: int, threshold: int, probability: float) -> float:
    return float(sum(math.comb(n, k) * probability**k * (1.0 - probability) ** (n - k) for k in range(threshold, n + 1)))


def _power(n: int) -> dict[str, Any]:
    threshold = next(k for k in range(n + 1) if _tail(n, k, 0.5) < 0.05)
    return {
        "group_count": n,
        "alternative_direction": 0.70,
        "alpha": 0.05,
        "minimum_positive_groups": threshold,
        "power": _tail(n, threshold, 0.70),
    }


def independent_evaluate_bag_completion(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    population = _population(rows)
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in population:
        group_folds[str(row["group_sha256"])].add(int(row["outer_fold"]))
    fold_groups = {
        fold: {str(row["group_sha256"]) for row in population if int(row["outer_fold"]) == fold}
        for fold in FOLDS
    }
    wrong = tuple(row for row in population if float(row["raw_margin"]) <= 0.0)
    correct = tuple(row for row in population if float(row["raw_margin"]) > 0.0)
    wrong_groups = {str(row["group_sha256"]) for row in wrong}
    correct_groups = {str(row["group_sha256"]) for row in correct}
    eligible = (
        len(group_folds) >= 40
        and all(len(value) == 1 for value in group_folds.values())
        and all(len(fold_groups[fold]) >= 8 for fold in FOLDS)
        and len(wrong_groups) >= 12
        and len(correct_groups) >= 12
        and all(
            len({str(row["group_sha256"]) for row in part if int(row["outer_fold"]) == fold}) >= 2
            for part in (wrong, correct)
            for fold in FOLDS
        )
    )
    power = _power(len(group_folds))
    if not eligible:
        return {
            "schema_version": "rc_dino_rcde_bag_completion_statistics_v1_20260827",
            "status": "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE",
            "query_count": len(population),
            "group_count": len(group_folds),
            "fold_group_count": {str(key): len(value) for key, value in fold_groups.items()},
            "RAW_wrong_query_count": len(wrong),
            "RAW_correct_query_count": len(correct),
            "RAW_wrong_group_count": len(wrong_groups),
            "RAW_correct_group_count": len(correct_groups),
            "mra_power": power,
            "scientific_GO_or_NO_GO": None,
        }
    overall = _summary(population, "real_margin")
    wrong_summary = _summary(wrong, "real_margin")
    correct_summary = _summary(correct, "real_margin")
    fold_direction: dict[str, float] = {}
    for fold in FOLDS:
        selected = tuple(row for row in population if int(row["outer_fold"]) == fold)
        values, _ = _groups(selected, lambda row: float(float(row["real_margin"]) > 0.0))
        fold_direction[str(fold)] = float(values.mean())
    centered, names = _groups(population, lambda row: float(float(row["real_margin"]) > 0.0) - 0.5)
    directions, same = _groups(population, lambda row: float(float(row["real_margin"]) > 0.0))
    check(names == same, "independent primary group order drift")
    primary_p = _signflip(centered, seed + 401, repetitions, True)
    direction_lower = _bootstrap_lower(directions, seed + 409, repetitions)
    drops, drop_names = _groups(population, lambda row: float(row["real_margin"]) - float(row["c_bind_margin"]))
    real_dir, real_names = _groups(population, lambda row: float(float(row["real_margin"]) > 0.0))
    control_dir, control_names = _groups(population, lambda row: float(float(row["c_bind_margin"]) > 0.0))
    wins, win_names = _groups(population, lambda row: float(float(row["real_margin"]) > float(row["c_bind_margin"])))
    check(drop_names == real_names == control_names == win_names, "independent C_BIND order drift")
    fold_drop: dict[str, float] = {}
    for fold in FOLDS:
        selected = tuple(row for row in population if int(row["outer_fold"]) == fold)
        values, _ = _groups(selected, lambda row: float(row["real_margin"]) - float(row["c_bind_margin"]))
        fold_drop[str(fold)] = float(values.mean())
    c_bind: dict[str, Any] = {
        "group_count": len(drop_names),
        "direction_drop": float(real_dir.mean() - control_dir.mean()),
        "mean_margin_drop": float(drops.mean()),
        "cluster_bootstrap_lower": _bootstrap_lower(drops, seed + 101, repetitions),
        "signflip_p": _signflip(drops, seed + 211, repetitions),
        "real_greater_control_case_fraction": float(wins.mean()),
        "fold_mean_drop": fold_drop,
    }
    c_bind["gate"] = (
        c_bind["direction_drop"] >= 0.05
        and c_bind["mean_margin_drop"] > 0.0
        and c_bind["cluster_bootstrap_lower"] > 0.0
        and c_bind["signflip_p"] < 0.05
        and c_bind["real_greater_control_case_fraction"] >= 0.60
        and sum(value > 0.0 for value in fold_drop.values()) >= 3
    )
    go = bool(
        overall["group_balanced_direction"] >= 0.60
        and overall["group_balanced_mean_margin"] > 0.0
        and direction_lower > 0.50
        and primary_p < 0.05
        and wrong_summary["group_balanced_direction"] >= 0.60
        and wrong_summary["group_balanced_mean_margin"] > 0.0
        and correct_summary["group_balanced_direction"] >= 0.70
        and sum(value > 0.50 for value in fold_direction.values()) >= 3
        and c_bind["gate"]
    )
    return {
        "schema_version": "rc_dino_rcde_bag_completion_statistics_v1_20260827",
        "status": "DINO_RCDE_BAG_COMPLETION_SCIENTIFIC_REDUCTION_COMPLETE",
        "query_count": len(population),
        "group_count": len(group_folds),
        "fold_group_count": {str(key): len(value) for key, value in fold_groups.items()},
        "RAW_wrong_query_count": len(wrong),
        "RAW_correct_query_count": len(correct),
        "RAW_wrong_group_count": len(wrong_groups),
        "RAW_correct_group_count": len(correct_groups),
        "permutations": repetitions,
        "seed": seed,
        "mra_power": power,
        "overall": overall,
        "RAW_wrong": wrong_summary,
        "RAW_correct": correct_summary,
        "fold_direction": fold_direction,
        "direction_bootstrap_lower": direction_lower,
        "single_arm_maxT_adjusted_p": primary_p,
        "C_BIND": c_bind,
        "candidate_reorder_gate": True,
        "bag_fixed_decoder_gate": go,
        "decision": (
            "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_NONSPATIAL_EVIDENCE_GO"
            if go
            else "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_EVIDENCE_NO_GO"
        ),
        "scientific_GO_or_NO_GO": "GO" if go else "NO_GO",
    }


__all__ = ["IndependentBagStatisticsError", "independent_evaluate_bag_completion"]
