#!/usr/bin/env python3
"""Registered PJ2 reduction for the immutable target-free natural-C128 ledger."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "registry/h0/h0_c128_target_free_postjoin_pj2_authority_v1_20260824.json"
NAMESPACE = "h0_c128_target_free_postjoin"
AUTHORITY_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ2_REDUCTION_AUTHORIZED"
PJ1_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
PJ1_VALIDATION_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS"
GO_STATUS = "RELATIVE_C128_RANK_INCREMENT_GO"
NO_GO_STATUS = "RELATIVE_C128_RANK_INCREMENT_NO_GO"
ARMS = ("INIT", "TRACK_H")
DIRECTIONS = ("a_to_b", "b_to_a")
EXPECTED_FOLD_COUNTS = Counter({1: 149, 3: 149, 2: 148, 4: 148})


class PJ2Error(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise PJ2Error(message)


def canonical(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def record_sha(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "record_sha256"})


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent/unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def float_bits(value: float) -> str:
    value = float(value)
    require(math.isfinite(value), "nonfinite binary64 value")
    return struct.pack(">d", value).hex()


def candidate_score(a_to_b: float, b_to_a: float) -> float:
    """The registered binary64 direction mean, in the registered op order."""
    a = float(a_to_b)
    b = float(b_to_a)
    require(math.isfinite(a) and math.isfinite(b), "nonfinite directional logit")
    score = 0.5 * (a + b)
    require(math.isfinite(score), "nonfinite direction mean")
    return float(score)


def _raw_logit(candidate: Mapping[str, Any], arm: str, direction: str) -> float:
    arm_payload = candidate.get("raw_directional_logits", {}).get(arm)
    require(isinstance(arm_payload, Mapping) and set(arm_payload) == set(DIRECTIONS), "direction axis drift")
    item = arm_payload.get(direction)
    require(isinstance(item, Mapping) and set(item) == {"raw_logit", "raw_logit_bits"}, "raw logit schema drift")
    value = float(item["raw_logit"])
    require(item["raw_logit_bits"] == float_bits(value), "raw logit bit-pattern drift")
    return value


def reduce_arm(
    candidates: Sequence[Mapping[str, Any]], arm: str, target_identity: str
) -> dict[str, Any]:
    """Reduce physical candidates to corrected identities for one arm only."""
    require(arm in ARMS, "unknown arm")
    physical_scores: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    for candidate in candidates:
        physical_row = int(candidate["candidate_physical_row"])
        require(physical_row >= 0 and physical_row not in seen_rows, "candidate physical-row collision")
        seen_rows.add(physical_row)
        identity = str(candidate["candidate_corrected_identity"])
        require(identity != "", "empty corrected identity")
        score = candidate_score(
            _raw_logit(candidate, arm, DIRECTIONS[0]),
            _raw_logit(candidate, arm, DIRECTIONS[1]),
        )
        physical_scores.append(
            {
                "candidate_physical_row": physical_row,
                "candidate_corrected_identity": identity,
                "score": score,
                "score_bits": float_bits(score),
            }
        )

    # An identity is represented by its maximum physical-row score.  Exact
    # ties are resolved only by the smaller physical row.
    reduced: dict[str, dict[str, Any]] = {}
    for item in physical_scores:
        identity = str(item["candidate_corrected_identity"])
        incumbent = reduced.get(identity)
        if (
            incumbent is None
            or float(item["score"]) > float(incumbent["score"])
            or (
                float(item["score"]) == float(incumbent["score"])
                and int(item["candidate_physical_row"])
                < int(incumbent["candidate_physical_row"])
            )
        ):
            reduced[identity] = item

    require(target_identity in reduced, "target corrected identity absent")
    require(any(identity != target_identity for identity in reduced), "wrong identity absent")
    target = reduced[target_identity]
    wrong = min(
        (item for identity, item in reduced.items() if identity != target_identity),
        key=lambda item: (
            -float(item["score"]),
            int(item["candidate_physical_row"]),
            str(item["candidate_corrected_identity"]).encode("utf-8"),
        ),
    )
    target_score = float(target["score"])
    wrong_score = float(wrong["score"])
    margin = target_score - wrong_score
    require(math.isfinite(margin), "nonfinite target-own-rival margin")
    rank = 1 + sum(
        float(item["score"]) >= target_score
        for identity, item in reduced.items()
        if identity != target_identity
    )
    return {
        "physical_candidate_scores": physical_scores,
        "physical_candidate_score_population_sha256": canonical(physical_scores),
        "reduced_identity_count": len(reduced),
        "target": {
            "corrected_identity": target_identity,
            "physical_row": int(target["candidate_physical_row"]),
            "score": target_score,
            "score_bits": float_bits(target_score),
        },
        "strongest_wrong": {
            "corrected_identity": str(wrong["candidate_corrected_identity"]),
            "physical_row": int(wrong["candidate_physical_row"]),
            "score": wrong_score,
            "score_bits": float_bits(wrong_score),
        },
        "margin": margin,
        "margin_bits": float_bits(margin),
        "strict_top1": margin > 0.0,
        "rank": int(rank),
        "reciprocal_rank": float(1.0 / rank),
    }


def group_means(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["supergroup"])].append(float(row[field]))
    require(grouped, "empty supergroup population")
    return {
        group: float(np.asarray(values, dtype=np.float64).mean())
        for group, values in sorted(grouped.items())
    }


def registered_inference(
    group_values: Mapping[str, float], seed: int = 17, repetitions: int = 9_999
) -> dict[str, Any]:
    require(seed == 17 and repetitions == 9_999, "registered inference parameters drift")
    values = np.asarray(
        [float(group_values[group]) for group in sorted(group_values)],
        dtype=np.float64,
    )
    require(values.size > 1 and np.isfinite(values).all(), "invalid supergroup statistic population")
    observed = float(values.mean())

    bootstrap_rng = np.random.Generator(np.random.PCG64(seed))
    indices = bootstrap_rng.integers(
        0, values.size, size=(repetitions, values.size)
    )
    bootstrap = values[indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap, [0.025, 0.975], method="linear"
    )

    # This is a separate PCG64(seed=17) stream, not a continuation of the
    # bootstrap stream.
    signflip_rng = np.random.Generator(np.random.PCG64(seed))
    signs = np.where(
        signflip_rng.integers(0, 2, size=(repetitions, values.size)) == 0,
        -1.0,
        1.0,
    )
    permuted = (signs * values).mean(axis=1)
    p_value = float(
        (1 + np.count_nonzero(permuted >= observed)) / 10_000
    )
    return {
        "group_count": int(values.size),
        "group_balanced_gain": observed,
        "bootstrap_replicates": repetitions,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "signflip_replicates": repetitions,
        "signflip_seed": seed,
        "one_sided_group_signflip_p": p_value,
    }


def _validate_source_rows(rows: Sequence[Mapping[str, Any]], exact_population: bool) -> None:
    require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)), "PJ1 row sequence absent")
    executions: list[int] = []
    query_ids: set[str] = set()
    group_folds: dict[str, int] = {}
    fold_counts: Counter[int] = Counter()
    for row in rows:
        require(isinstance(row, Mapping) and row.get("record_sha256") == record_sha(row), "PJ1 row SHA drift")
        execution = int(row["execution_ordinal"])
        query_id = str(row["query_id"])
        fold = int(row["outer_fold"])
        group = str(row["supergroup"])
        require(fold in (1, 2, 3, 4) and query_id not in query_ids and group != "", "query/fold/group drift")
        previous_fold = group_folds.setdefault(group, fold)
        require(previous_fold == fold, "supergroup crosses outer folds")
        query_ids.add(query_id)
        executions.append(execution)
        fold_counts[fold] += 1
        candidates = row.get("candidates")
        require(isinstance(candidates, list), "candidate sequence absent")
        if exact_population:
            require(len(candidates) == 128, "natural-C128 candidate count drift")
        require(row.get("candidate_population_sha256") == canonical(candidates), "candidate population SHA drift")
        require(str(row.get("target_corrected_identity", "")) in {str(item["candidate_corrected_identity"]) for item in candidates}, "natural target absent")
        for candidate in candidates:
            require(candidate.get("record_sha256") == record_sha(candidate), "PJ1 candidate SHA drift")
    require(len(executions) == len(set(executions)), "execution ordinal collision")
    if exact_population:
        require(len(rows) == 594 and fold_counts == EXPECTED_FOLD_COUNTS, "PJ1 full query/fold population drift")


def reduce_rows(
    rows: Sequence[Mapping[str, Any]], *, exact_population: bool = False
) -> dict[str, Any]:
    _validate_source_rows(rows, exact_population)
    output_rows: list[dict[str, Any]] = []
    for source in rows:
        target_identity = str(source["target_corrected_identity"])
        arms = {
            arm: reduce_arm(source["candidates"], arm, target_identity)
            for arm in ARMS
        }
        init = arms["INIT"]
        track_h = arms["TRACK_H"]
        if not init["strict_top1"] and track_h["strict_top1"]:
            transition = "init_wrong_h_correct"
            both_subcategory = None
        elif init["strict_top1"] and not track_h["strict_top1"]:
            transition = "init_correct_h_wrong"
            both_subcategory = None
        else:
            transition = "both"
            both_subcategory = (
                "both_correct" if init["strict_top1"] else "both_wrong"
            )
        row = {
            "source_pj1_record_sha256": source["record_sha256"],
            "execution_ordinal": int(source["execution_ordinal"]),
            "query_id": str(source["query_id"]),
            "outer_fold": int(source["outer_fold"]),
            "supergroup": str(source["supergroup"]),
            "track": str(source["track"]),
            "target_corrected_identity": target_identity,
            "arms": arms,
            "paired_top1_difference": int(track_h["strict_top1"])
            - int(init["strict_top1"]),
            "paired_reciprocal_rank_difference": float(
                track_h["reciprocal_rank"] - init["reciprocal_rank"]
            ),
            "transition": transition,
            "both_subcategory": both_subcategory,
        }
        row["record_sha256"] = record_sha(row)
        output_rows.append(row)

    top1_groups = group_means(output_rows, "paired_top1_difference")
    mrr_groups = group_means(output_rows, "paired_reciprocal_rank_difference")
    top1_inference = registered_inference(top1_groups)
    mrr_inference = registered_inference(mrr_groups)

    fold_summaries: dict[str, dict[str, Any]] = {}
    for fold in range(1, 5):
        selected = [row for row in output_rows if int(row["outer_fold"]) == fold]
        require(selected, f"outer fold absent: {fold}")
        fold_top1 = group_means(selected, "paired_top1_difference")
        fold_mrr = group_means(selected, "paired_reciprocal_rank_difference")
        fold_summaries[str(fold)] = {
            "query_count": len(selected),
            "supergroup_count": len(fold_top1),
            "group_balanced_top1_gain": float(
                np.asarray(list(fold_top1.values()), dtype=np.float64).mean()
            ),
            "group_balanced_mrr_gain": float(
                np.asarray(list(fold_mrr.values()), dtype=np.float64).mean()
            ),
        }

    transition_counts = {
        category: sum(row["transition"] == category for row in output_rows)
        for category in ("init_wrong_h_correct", "init_correct_h_wrong", "both")
    }
    transition_counts.update(
        {
            "both_correct": sum(
                row["both_subcategory"] == "both_correct" for row in output_rows
            ),
            "both_wrong": sum(
                row["both_subcategory"] == "both_wrong" for row in output_rows
            ),
        }
    )
    gates = {
        "group_balanced_top1_gain_at_least_0p03": top1_inference["group_balanced_gain"] >= 0.03,
        "top1_bootstrap_95_lower_strictly_positive": top1_inference["bootstrap_95_ci"][0] > 0.0,
        "top1_one_sided_group_signflip_p_below_0p05": top1_inference["one_sided_group_signflip_p"] < 0.05,
        "group_balanced_mrr_gain_strictly_positive": mrr_inference["group_balanced_gain"] > 0.0,
        "mrr_bootstrap_95_lower_strictly_positive": mrr_inference["bootstrap_95_ci"][0] > 0.0,
        "mrr_one_sided_group_signflip_p_below_0p05": mrr_inference["one_sided_group_signflip_p"] < 0.05,
        "positive_top1_folds_at_least_3_of_4": sum(
            item["group_balanced_top1_gain"] > 0.0
            for item in fold_summaries.values()
        ) >= 3,
        "positive_mrr_folds_at_least_3_of_4": sum(
            item["group_balanced_mrr_gain"] > 0.0
            for item in fold_summaries.values()
        ) >= 3,
        "zero_to_one_strictly_exceeds_one_to_zero": transition_counts["init_wrong_h_correct"]
        > transition_counts["init_correct_h_wrong"],
    }
    return {
        "rows": output_rows,
        "row_population_sha256": canonical(output_rows),
        "transition_counts": transition_counts,
        "supergroup_paired_differences": {
            "top1": top1_groups,
            "reciprocal_rank": mrr_groups,
        },
        "top1_inference": top1_inference,
        "mrr_inference": mrr_inference,
        "fold_summaries": fold_summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def resolve_binding(authority: Mapping[str, Any], name: str) -> Path:
    binding = authority.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"binding absent: {name}")
    raw = Path(str(binding.get("path", "")))
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"binding path absent/unsafe: {name}")
    require(
        path.stat().st_size == int(binding.get("bytes", -1))
        and file_sha(path) == binding.get("sha256"),
        f"binding bytes/SHA drift: {name}",
    )
    if "logical_sha256" in binding:
        payload = read_json(path)
        require(
            payload.get("logical_sha256") == logical(payload) == binding["logical_sha256"],
            f"binding logical SHA drift: {name}",
        )
    return path


def validate_authority(path: Path, output: Path) -> tuple[dict[str, Any], str]:
    require(
        path.resolve() == AUTHORITY.resolve()
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_mode & 0o777 == 0o444,
        "PJ2 authority path/mode drift",
    )
    authority = read_json(path)
    require(
        authority.get("namespace") == NAMESPACE
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("scientific_reduction_authorized") is True
        and authority.get("training_authorized") is False
        and authority.get("model_forward_authorized") is False
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") is None
        and output.resolve(strict=False)
        == (ROOT / str(authority.get("output", ""))).resolve(strict=False),
        "PJ2 authority envelope drift",
    )
    for name in authority.get("bindings", {}):
        resolve_binding(authority, str(name))
    return authority, file_sha(path)


def build_result(
    authority: Mapping[str, Any], authority_sha: str, pj1: Mapping[str, Any], pj1_sha: str
) -> dict[str, Any]:
    source_rows = pj1.get("rows")
    require(
        pj1.get("namespace") == NAMESPACE
        and pj1.get("status") == PJ1_STATUS
        and pj1.get("logical_sha256") == logical(pj1)
        and pj1.get("query_count") == 594
        and pj1.get("candidate_count") == 76_032
        and isinstance(source_rows, list)
        and pj1.get("row_population_sha256") == canonical(source_rows),
        "PJ1 result envelope drift",
    )
    reduction = reduce_rows(source_rows, exact_population=True)
    go = bool(reduction["all_gates_pass"])
    result: dict[str, Any] = {
        "schema_version": "rc_h0_c128_target_free_postjoin_pj2_result_v1_20260824",
        "namespace": NAMESPACE,
        "status": GO_STATUS if go else NO_GO_STATUS,
        "claim_level": "RELATIVE_C128_RANK_INCREMENT_ONLY",
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "pj1_result_sha256": pj1_sha,
        "pj1_result_logical_sha256": pj1["logical_sha256"],
        "query_count": 594,
        "candidate_count": 76_032,
        **reduction,
        "forbidden_cross_arm_margin_difference_computed": False,
        "i0_raw_rival_read_count": 0,
        "rescue_break_read_count": 0,
        "c_p_control_count": 0,
        "scientific_GO_or_NO_GO": "GO" if go else "NO_GO",
        "full_gallery_retrieval_or_ownership_claim_authorized": False,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical(result)
    return result


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable PJ2 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority, authority_sha = validate_authority(args.authority, args.output)
    pj1_path = resolve_binding(authority, "pj1_result")
    pj1_validation_path = resolve_binding(authority, "pj1_validation")
    pj1_validation = read_json(pj1_validation_path)
    require(
        pj1_validation.get("status") == PJ1_VALIDATION_STATUS
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("logical_sha256") == logical(pj1_validation)
        and pj1_validation.get("result_sha256") == file_sha(pj1_path),
        "PJ1 independent validation drift",
    )
    pj1 = read_json(pj1_path)
    result = build_result(authority, authority_sha, pj1, file_sha(pj1_path))
    atomic_write(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_GO_or_NO_GO": result["scientific_GO_or_NO_GO"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
