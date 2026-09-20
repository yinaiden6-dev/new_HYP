"""Pure contracts and statistics for the BAG-only RCDE completion.

The module deliberately contains no filesystem, model, label-join, or Slurm
code.  Target-free producers use the permutation helpers; the primary reducer
uses :func:`evaluate_bag_completion`.  The independent validator reimplements
the statistics instead of importing that function.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


QUERY_COUNT = 600
VALID_QUERY_COUNT = 594
CANDIDATE_COUNT = 128
EXCLUDED_EXECUTIONS = (25, 26, 101, 346, 354, 470)
FOLDS = (1, 2, 3, 4)
SHARD_SIZE = 12
SHARD_COUNT = 50
PERMUTATIONS = 9_999
SEED = 17
C_BIND_NAMESPACE = "RCDE_BAG_FULL594_C_BIND_V1_SEED17"
REORDER_NAMESPACE = "RCDE_BAG_FULL594_CANDIDATE_REORDER_V1_SEED17"


class BagCompletionError(RuntimeError):
    """A frozen BAG-completion invariant failed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise BagCompletionError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes(order="C")).hexdigest()


def hash_parts(namespace: str, *parts: object) -> str:
    return canonical_sha256([namespace, *parts])


def deterministic_identity_derangement(
    physical_rows: Sequence[int],
    corrected_identities: Sequence[str],
    *,
    query_id: str,
    namespace: str = C_BIND_NAMESPACE,
) -> tuple[int, ...]:
    """Return destination->source positions for a complete C128 C_BIND.

    The matching is deterministic, has no fixed point, and never assigns a
    donor with the same corrected identity.  It is computed without target or
    rival roles.
    """

    rows = tuple(int(value) for value in physical_rows)
    identities = tuple(str(value) for value in corrected_identities)
    require(len(rows) == CANDIDATE_COUNT, "C_BIND axis is not C128")
    require(rows == tuple(sorted(rows)) and len(set(rows)) == len(rows), "C_BIND physical-row axis drift")
    require(len(identities) == len(rows) and all(identities), "C_BIND identity axis drift")
    keys = tuple(range(len(rows)))
    donor_for: dict[int, int] = {}
    destination_for_donor: dict[int, int] = {}

    def assign(destination: int, seen: set[int]) -> bool:
        donors = sorted(
            (
                donor
                for donor in keys
                if donor != destination
                and identities[donor] != identities[destination]
            ),
            key=lambda donor: hash_parts(
                namespace,
                query_id,
                "DONOR",
                rows[destination],
                rows[donor],
            ),
        )
        for donor in donors:
            if donor in seen:
                continue
            seen.add(donor)
            prior = destination_for_donor.get(donor)
            if prior is None or assign(prior, seen):
                destination_for_donor[donor] = destination
                donor_for[destination] = donor
                return True
        return False

    destinations = sorted(
        keys,
        key=lambda position: hash_parts(
            namespace, query_id, "DEST", rows[position]
        ),
    )
    require(
        all(assign(destination, set()) for destination in destinations),
        "C_BIND identity-disjoint perfect matching does not exist",
    )
    order = tuple(donor_for[position] for position in keys)
    require(sorted(order) == list(keys), "C_BIND donor map is not a permutation")
    require(all(index != donor for index, donor in enumerate(order)), "C_BIND fixed point")
    require(
        all(identities[index] != identities[donor] for index, donor in enumerate(order)),
        "C_BIND corrected-identity collision",
    )
    return order


def deterministic_candidate_reorder(
    physical_rows: Sequence[int],
    *,
    query_id: str,
    namespace: str = REORDER_NAMESPACE,
) -> tuple[int, ...]:
    """Return a result-blind source-position order for an invariance replay."""

    rows = tuple(int(value) for value in physical_rows)
    require(len(rows) == CANDIDATE_COUNT, "candidate reorder axis is not C128")
    require(rows == tuple(sorted(rows)) and len(set(rows)) == len(rows), "candidate reorder physical-row axis drift")
    order = tuple(
        sorted(
            range(len(rows)),
            key=lambda position: hash_parts(
                namespace, query_id, rows[position], position
            ),
        )
    )
    require(order != tuple(range(len(rows))), "candidate reorder is identity")
    return order


def permuted_delta(delta: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    matrix = torch.as_tensor(delta, dtype=torch.float32).detach().cpu().contiguous()
    indexes = torch.as_tensor(tuple(map(int, order)), dtype=torch.long)
    require(tuple(matrix.shape) == (CANDIDATE_COUNT, CANDIDATE_COUNT), "Delta is not C128xC128")
    require(sorted(indexes.tolist()) == list(range(CANDIDATE_COUNT)), "Delta permutation drift")
    output = matrix.index_select(0, indexes).index_select(1, indexes).contiguous()
    require(torch.equal(output, -output.T), "permuted Delta antisymmetry drift")
    require(torch.equal(torch.diagonal(output), torch.zeros(CANDIDATE_COUNT)), "permuted Delta diagonal drift")
    return output


def _population(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    values = tuple(rows)
    seen: set[tuple[str, int]] = set()
    for row in values:
        address = (str(row.get("query_id")), int(row.get("outer_fold", -1)))
        require(
            address[0] and address[1] in FOLDS and address not in seen,
            "scientific row address drift",
        )
        seen.add(address)
        require(
            all(
                math.isfinite(float(row.get(name)))
                for name in ("raw_margin", "real_margin", "c_bind_margin")
            ),
            "nonfinite scientific margin",
        )
        require(isinstance(row.get("group_sha256"), str), "group hash absent")
    require(len(values) == VALID_QUERY_COUNT, "scientific population is not 594")
    return values


def _group_values(rows: Sequence[Mapping[str, Any]], fn) -> tuple[np.ndarray, tuple[str, ...]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row["group_sha256"])].append(float(fn(row)))
    names = tuple(sorted(buckets))
    values = np.asarray([np.mean(buckets[name]) for name in names], dtype=np.float64)
    require(bool(names) and bool(np.isfinite(values).all()), "group reduction drift")
    return values, names


def _summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    margins, groups = _group_values(rows, lambda row: float(row[key]))
    directions, same = _group_values(rows, lambda row: float(float(row[key]) > 0.0))
    require(groups == same, "summary group order drift")
    return {
        "query_count": len(rows),
        "group_count": len(groups),
        "group_balanced_direction": float(directions.mean()),
        "group_balanced_mean_margin": float(margins.mean()),
    }


def _bootstrap_lower(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(repetitions, values.size))
    return float(np.quantile(values[indexes].mean(axis=1), 0.025, method="linear"))


def _studentized(values: np.ndarray) -> float:
    standard = float(values.std(ddof=1))
    if standard == 0.0:
        return float("inf") if float(values.mean()) > 0.0 else float("-inf")
    return float(values.mean() / (standard / math.sqrt(values.size)))


def _signflip_p(values: np.ndarray, *, seed: int, repetitions: int, studentized: bool = False) -> float:
    observed = _studentized(values) if studentized else float(values.mean())
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


def _binomial_tail(n: int, threshold: int, probability: float) -> float:
    return float(
        sum(
            math.comb(n, value)
            * probability**value
            * (1.0 - probability) ** (n - value)
            for value in range(threshold, n + 1)
        )
    )


def exact_sign_mra_power(group_count: int, *, alternative: float = 0.70, alpha: float = 0.05) -> dict[str, Any]:
    threshold = next(
        value
        for value in range(group_count + 1)
        if _binomial_tail(group_count, value, 0.5) < alpha
    )
    return {
        "group_count": group_count,
        "alternative_direction": alternative,
        "alpha": alpha,
        "minimum_positive_groups": threshold,
        "power": _binomial_tail(group_count, threshold, alternative),
    }


def evaluate_bag_completion(
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
    statistical_eligibility = (
        len(group_folds) >= 40
        and all(len(value) == 1 for value in group_folds.values())
        and all(len(fold_groups[fold]) >= 8 for fold in FOLDS)
        and len(wrong_groups) >= 12
        and len(correct_groups) >= 12
        and all(
            len({str(row["group_sha256"]) for row in slice_rows if int(row["outer_fold"]) == fold}) >= 2
            for slice_rows in (wrong, correct)
            for fold in FOLDS
        )
    )
    power = exact_sign_mra_power(len(group_folds))
    if not statistical_eligibility:
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
        values, _ = _group_values(selected, lambda row: float(float(row["real_margin"]) > 0.0))
        fold_direction[str(fold)] = float(values.mean())

    direction_centered, direction_groups = _group_values(
        population, lambda row: float(float(row["real_margin"]) > 0.0) - 0.5
    )
    direction_values, same_groups = _group_values(
        population, lambda row: float(float(row["real_margin"]) > 0.0)
    )
    require(direction_groups == same_groups, "direction group order drift")
    primary_p = _signflip_p(
        direction_centered,
        seed=seed + 401,
        repetitions=repetitions,
        studentized=True,
    )
    direction_lower = _bootstrap_lower(
        direction_values, seed=seed + 409, repetitions=repetitions
    )

    margin_drops, drop_groups = _group_values(
        population, lambda row: float(row["real_margin"]) - float(row["c_bind_margin"])
    )
    real_direction, real_groups = _group_values(
        population, lambda row: float(float(row["real_margin"]) > 0.0)
    )
    control_direction, control_groups = _group_values(
        population, lambda row: float(float(row["c_bind_margin"]) > 0.0)
    )
    wins, win_groups = _group_values(
        population, lambda row: float(float(row["real_margin"]) > float(row["c_bind_margin"]))
    )
    require(drop_groups == real_groups == control_groups == win_groups, "C_BIND group order drift")
    fold_drop: dict[str, float] = {}
    for fold in FOLDS:
        selected = tuple(row for row in population if int(row["outer_fold"]) == fold)
        values, _ = _group_values(
            selected, lambda row: float(row["real_margin"]) - float(row["c_bind_margin"])
        )
        fold_drop[str(fold)] = float(values.mean())
    c_bind = {
        "group_count": len(drop_groups),
        "direction_drop": float(real_direction.mean() - control_direction.mean()),
        "mean_margin_drop": float(margin_drops.mean()),
        "cluster_bootstrap_lower": _bootstrap_lower(
            margin_drops, seed=seed + 101, repetitions=repetitions
        ),
        "signflip_p": _signflip_p(
            margin_drops, seed=seed + 211, repetitions=repetitions
        ),
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
    base_gate = (
        overall["group_balanced_direction"] >= 0.60
        and overall["group_balanced_mean_margin"] > 0.0
        and direction_lower > 0.50
        and primary_p < 0.05
        and wrong_summary["group_balanced_direction"] >= 0.60
        and wrong_summary["group_balanced_mean_margin"] > 0.0
        and correct_summary["group_balanced_direction"] >= 0.70
        and sum(value > 0.50 for value in fold_direction.values()) >= 3
        and bool(c_bind["gate"])
    )
    go = bool(base_gate)
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


__all__ = [
    "BagCompletionError",
    "CANDIDATE_COUNT",
    "C_BIND_NAMESPACE",
    "EXCLUDED_EXECUTIONS",
    "FOLDS",
    "PERMUTATIONS",
    "QUERY_COUNT",
    "REORDER_NAMESPACE",
    "SEED",
    "SHARD_COUNT",
    "SHARD_SIZE",
    "VALID_QUERY_COUNT",
    "canonical_sha256",
    "deterministic_candidate_reorder",
    "deterministic_identity_derangement",
    "evaluate_bag_completion",
    "exact_sign_mra_power",
    "hash_parts",
    "permuted_delta",
    "require",
    "tensor_sha256",
]
