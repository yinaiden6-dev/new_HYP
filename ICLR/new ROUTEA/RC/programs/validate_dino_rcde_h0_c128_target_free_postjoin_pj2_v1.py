#!/usr/bin/env python3
"""Independent, full-population replay validator for target-free C128 PJ2."""

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
AUTHORITY_PATH = ROOT / "registry/h0/h0_c128_target_free_postjoin_pj2_authority_v1_20260824.json"
NAMESPACE = "h0_c128_target_free_postjoin"
AUTHORITY_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ2_REDUCTION_AUTHORIZED"
PJ1_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
PJ1_VALIDATION_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS"
VALIDATION_STATUS = "H0_C128_TARGET_FREE_POSTJOIN_PJ2_INDEPENDENT_VALIDATION_PASS"
GO_STATUS = "RELATIVE_C128_RANK_INCREMENT_GO"
NO_GO_STATUS = "RELATIVE_C128_RANK_INCREMENT_NO_GO"
ARMS = ("INIT", "TRACK_H")
DIRECTIONS = ("a_to_b", "b_to_a")
EXPECTED_FOLDS = Counter({1: 149, 3: 149, 2: 148, 4: 148})

EXPECTED_FORMULA_CONTRACT = {
    "candidate_score": "binary64(0.5*(a_to_b+b_to_a))",
    "same_identity_reduce": "max_then_smaller_physical_row",
    "strongest_wrong_order": "(-score,physical_row,UTF8_label)",
    "margin": "target_score-own_strongest_wrong_score",
    "strict_top1": "margin>0",
    "rank": "1+count(wrong_score>=target_score)",
    "reciprocal_rank": "1/rank",
    "forbidden_cross_arm_margin_difference": True,
}
EXPECTED_STATISTICAL_CONTRACT = {
    "aggregation": "supergroup_mean_then_equal_supergroup_mean",
    "bootstrap_replicates": 9_999,
    "bootstrap_seed": 17,
    "bootstrap_rng": "numpy.PCG64",
    "bootstrap_quantile_method": "linear",
    "signflip_replicates": 9_999,
    "signflip_seed": 17,
    "signflip_rng": "numpy.PCG64",
    "signflip_p": "(1+count(null>=observed))/10000",
}
EXPECTED_GATE_CONTRACT = {
    "minimum_group_balanced_top1_gain": 0.03,
    "top1_ci_lower_strictly_positive": True,
    "top1_one_sided_p_below": 0.05,
    "mrr_gain_strictly_positive": True,
    "mrr_ci_lower_strictly_positive": True,
    "mrr_one_sided_p_below": 0.05,
    "minimum_positive_top1_folds": 3,
    "minimum_positive_mrr_folds": 3,
    "zero_to_one_strictly_exceeds_one_to_zero": True,
}


class PJ2ValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise PJ2ValidationError(message)


def canonical(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


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


def binary64_bits(value: float) -> str:
    number = float(value)
    require(math.isfinite(number), "nonfinite binary64 value")
    return struct.pack(">d", number).hex()


def bound_path(authority: Mapping[str, Any], name: str) -> Path:
    binding = authority.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"binding absent: {name}")
    raw = Path(str(binding.get("path", "")))
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"binding absent/unsafe: {name}")
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


def _direction_value(candidate: Mapping[str, Any], arm: str, direction: str) -> float:
    arm_value = candidate.get("raw_directional_logits", {}).get(arm)
    require(isinstance(arm_value, Mapping) and set(arm_value) == set(DIRECTIONS), "direction axis drift")
    payload = arm_value.get(direction)
    require(isinstance(payload, Mapping) and set(payload) == {"raw_logit", "raw_logit_bits"}, "raw logit schema drift")
    value = float(payload["raw_logit"])
    require(payload["raw_logit_bits"] == binary64_bits(value), "raw logit bits drift")
    return value


def independently_reduce_arm(
    candidates: Sequence[Mapping[str, Any]], arm: str, target_identity: str
) -> dict[str, Any]:
    physical: list[dict[str, Any]] = []
    physical_rows: set[int] = set()
    for candidate in candidates:
        row = int(candidate["candidate_physical_row"])
        require(row >= 0 and row not in physical_rows, "candidate physical-row collision")
        physical_rows.add(row)
        label = str(candidate["candidate_corrected_identity"])
        require(label, "empty candidate identity")
        left = _direction_value(candidate, arm, DIRECTIONS[0])
        right = _direction_value(candidate, arm, DIRECTIONS[1])
        require(math.isfinite(left) and math.isfinite(right), "nonfinite raw logit")
        score = float(0.5 * (left + right))
        require(math.isfinite(score), "nonfinite candidate score")
        physical.append(
            {
                "candidate_physical_row": row,
                "candidate_corrected_identity": label,
                "score": score,
                "score_bits": binary64_bits(score),
            }
        )

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in physical:
        members[str(item["candidate_corrected_identity"])].append(item)
    reduced = {
        label: sorted(
            items,
            key=lambda item: (-float(item["score"]), int(item["candidate_physical_row"])),
        )[0]
        for label, items in members.items()
    }
    require(target_identity in reduced and len(reduced) >= 2, "target/wrong identity absent")
    target = reduced[target_identity]
    wrong_items = [item for label, item in reduced.items() if label != target_identity]
    wrong = sorted(
        wrong_items,
        key=lambda item: (
            -float(item["score"]),
            int(item["candidate_physical_row"]),
            str(item["candidate_corrected_identity"]).encode("utf-8"),
        ),
    )[0]
    target_score = float(target["score"])
    wrong_score = float(wrong["score"])
    margin = float(target_score - wrong_score)
    require(math.isfinite(margin), "nonfinite margin")
    rank = 1 + len(
        [
            item
            for label, item in reduced.items()
            if label != target_identity and float(item["score"]) >= target_score
        ]
    )
    return {
        "physical_candidate_scores": physical,
        "physical_candidate_score_population_sha256": canonical(physical),
        "reduced_identity_count": len(reduced),
        "target": {
            "corrected_identity": target_identity,
            "physical_row": int(target["candidate_physical_row"]),
            "score": target_score,
            "score_bits": binary64_bits(target_score),
        },
        "strongest_wrong": {
            "corrected_identity": str(wrong["candidate_corrected_identity"]),
            "physical_row": int(wrong["candidate_physical_row"]),
            "score": wrong_score,
            "score_bits": binary64_bits(wrong_score),
        },
        "margin": margin,
        "margin_bits": binary64_bits(margin),
        "strict_top1": margin > 0.0,
        "rank": rank,
        "reciprocal_rank": float(1.0 / rank),
    }


def independently_group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row["supergroup"])].append(float(row[field]))
    require(buckets, "empty group population")
    answer: dict[str, float] = {}
    for group in sorted(buckets):
        answer[group] = float(np.asarray(buckets[group], dtype=np.float64).mean())
    return answer


def independently_infer(group_values: Mapping[str, float]) -> dict[str, Any]:
    values = np.asarray(
        [float(group_values[key]) for key in sorted(group_values)], dtype=np.float64
    )
    require(values.size > 1 and np.isfinite(values).all(), "invalid group values")
    observed = float(values.mean())
    bootstrap_generator = np.random.Generator(np.random.PCG64(17))
    sampled = bootstrap_generator.integers(
        low=0, high=values.size, size=(9_999, values.size)
    )
    bootstrap_means = np.take(values, sampled).mean(axis=1)
    interval = np.quantile(
        bootstrap_means, np.asarray([0.025, 0.975]), method="linear"
    )
    sign_generator = np.random.Generator(np.random.PCG64(17))
    sign_bits = sign_generator.integers(
        low=0, high=2, size=(9_999, values.size)
    )
    signs = sign_bits.astype(np.float64) * 2.0 - 1.0
    null_means = np.multiply(signs, values).mean(axis=1)
    exceedances = int(np.count_nonzero(null_means >= observed))
    return {
        "group_count": int(values.size),
        "group_balanced_gain": observed,
        "bootstrap_replicates": 9_999,
        "bootstrap_seed": 17,
        "bootstrap_95_ci": [float(interval[0]), float(interval[1])],
        "signflip_replicates": 9_999,
        "signflip_seed": 17,
        "one_sided_group_signflip_p": float((1 + exceedances) / 10_000),
    }


def replay_reduction(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(isinstance(source_rows, list) and len(source_rows) == 594, "PJ1 query population drift")
    output: list[dict[str, Any]] = []
    folds: Counter[int] = Counter()
    executions: set[int] = set()
    queries: set[str] = set()
    group_fold: dict[str, int] = {}
    for source in source_rows:
        require(source.get("record_sha256") == record_sha(source), "source row SHA drift")
        execution = int(source["execution_ordinal"])
        query = str(source["query_id"])
        fold = int(source["outer_fold"])
        group = str(source["supergroup"])
        require(execution not in executions and query not in queries, "source key collision")
        require(fold in (1, 2, 3, 4) and group, "source fold/group drift")
        executions.add(execution)
        queries.add(query)
        folds[fold] += 1
        require(group_fold.setdefault(group, fold) == fold, "supergroup crosses folds")
        candidates = source.get("candidates")
        require(
            isinstance(candidates, list)
            and len(candidates) == 128
            and source.get("candidate_population_sha256") == canonical(candidates),
            "source candidate population drift",
        )
        for candidate in candidates:
            require(candidate.get("record_sha256") == record_sha(candidate), "source candidate SHA drift")
        target_identity = str(source["target_corrected_identity"])
        require(target_identity in {str(candidate["candidate_corrected_identity"]) for candidate in candidates}, "natural target absent")
        arm_results = {
            arm: independently_reduce_arm(candidates, arm, target_identity)
            for arm in ARMS
        }
        init = arm_results["INIT"]
        h_arm = arm_results["TRACK_H"]
        if not init["strict_top1"] and h_arm["strict_top1"]:
            transition, both_subcategory = "init_wrong_h_correct", None
        elif init["strict_top1"] and not h_arm["strict_top1"]:
            transition, both_subcategory = "init_correct_h_wrong", None
        else:
            transition = "both"
            both_subcategory = "both_correct" if init["strict_top1"] else "both_wrong"
        row = {
            "source_pj1_record_sha256": source["record_sha256"],
            "execution_ordinal": execution,
            "query_id": query,
            "outer_fold": fold,
            "supergroup": group,
            "track": str(source["track"]),
            "target_corrected_identity": target_identity,
            "arms": arm_results,
            "paired_top1_difference": int(h_arm["strict_top1"])
            - int(init["strict_top1"]),
            "paired_reciprocal_rank_difference": float(
                h_arm["reciprocal_rank"] - init["reciprocal_rank"]
            ),
            "transition": transition,
            "both_subcategory": both_subcategory,
        }
        row["record_sha256"] = record_sha(row)
        output.append(row)
    require(folds == EXPECTED_FOLDS, "source fold population drift")

    top1_groups = independently_group(output, "paired_top1_difference")
    reciprocal_groups = independently_group(output, "paired_reciprocal_rank_difference")
    top1_inference = independently_infer(top1_groups)
    mrr_inference = independently_infer(reciprocal_groups)
    fold_summaries: dict[str, dict[str, Any]] = {}
    for fold in (1, 2, 3, 4):
        selected = [row for row in output if int(row["outer_fold"]) == fold]
        top = independently_group(selected, "paired_top1_difference")
        rr = independently_group(selected, "paired_reciprocal_rank_difference")
        fold_summaries[str(fold)] = {
            "query_count": len(selected),
            "supergroup_count": len(top),
            "group_balanced_top1_gain": float(
                np.asarray(list(top.values()), dtype=np.float64).mean()
            ),
            "group_balanced_mrr_gain": float(
                np.asarray(list(rr.values()), dtype=np.float64).mean()
            ),
        }
    transitions = {
        key: len([row for row in output if row["transition"] == key])
        for key in ("init_wrong_h_correct", "init_correct_h_wrong", "both")
    }
    transitions["both_correct"] = len(
        [row for row in output if row["both_subcategory"] == "both_correct"]
    )
    transitions["both_wrong"] = len(
        [row for row in output if row["both_subcategory"] == "both_wrong"]
    )
    gates = {
        "group_balanced_top1_gain_at_least_0p03": top1_inference["group_balanced_gain"] >= 0.03,
        "top1_bootstrap_95_lower_strictly_positive": top1_inference["bootstrap_95_ci"][0] > 0.0,
        "top1_one_sided_group_signflip_p_below_0p05": top1_inference["one_sided_group_signflip_p"] < 0.05,
        "group_balanced_mrr_gain_strictly_positive": mrr_inference["group_balanced_gain"] > 0.0,
        "mrr_bootstrap_95_lower_strictly_positive": mrr_inference["bootstrap_95_ci"][0] > 0.0,
        "mrr_one_sided_group_signflip_p_below_0p05": mrr_inference["one_sided_group_signflip_p"] < 0.05,
        "positive_top1_folds_at_least_3_of_4": len(
            [x for x in fold_summaries.values() if x["group_balanced_top1_gain"] > 0.0]
        ) >= 3,
        "positive_mrr_folds_at_least_3_of_4": len(
            [x for x in fold_summaries.values() if x["group_balanced_mrr_gain"] > 0.0]
        ) >= 3,
        "zero_to_one_strictly_exceeds_one_to_zero": transitions["init_wrong_h_correct"]
        > transitions["init_correct_h_wrong"],
    }
    return {
        "rows": output,
        "row_population_sha256": canonical(output),
        "transition_counts": transitions,
        "supergroup_paired_differences": {
            "top1": top1_groups,
            "reciprocal_rank": reciprocal_groups,
        },
        "top1_inference": top1_inference,
        "mrr_inference": mrr_inference,
        "fold_summaries": fold_summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def expected_result(
    authority: Mapping[str, Any], authority_sha: str, pj1: Mapping[str, Any], pj1_sha: str
) -> dict[str, Any]:
    require(
        pj1.get("namespace") == NAMESPACE
        and pj1.get("status") == PJ1_STATUS
        and pj1.get("logical_sha256") == logical(pj1)
        and pj1.get("query_count") == 594
        and pj1.get("candidate_count") == 76_032
        and pj1.get("row_population_sha256") == canonical(pj1.get("rows")),
        "PJ1 envelope drift",
    )
    replay = replay_reduction(pj1["rows"])
    go = bool(replay["all_gates_pass"])
    value: dict[str, Any] = {
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
        **replay,
        "forbidden_cross_arm_margin_difference_computed": False,
        "i0_raw_rival_read_count": 0,
        "rescue_break_read_count": 0,
        "c_p_control_count": 0,
        "scientific_GO_or_NO_GO": "GO" if go else "NO_GO",
        "full_gallery_retrieval_or_ownership_claim_authorized": False,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical(value)
    return value


def validate_payload(
    authority: Mapping[str, Any], result: Mapping[str, Any], pj1: Mapping[str, Any],
    authority_sha: str, pj1_sha: str
) -> dict[str, Any]:
    expected = expected_result(authority, authority_sha, pj1, pj1_sha)
    require(result == expected, "PJ2 full independent replay mismatch")
    require(
        result["status"] in (GO_STATUS, NO_GO_STATUS)
        and result["next_authorized_stage"] is None
        and result["automatic_stage_advance"] is False
        and result["forbidden_cross_arm_margin_difference_computed"] is False,
        "PJ2 terminal/forbidden-analysis drift",
    )
    for row in result["rows"]:
        require("paired_margin_difference" not in row, "forbidden paired margin stored")
        require(
            row["transition"] in ("init_wrong_h_correct", "init_correct_h_wrong", "both"),
            "transition category drift",
        )
        require(
            row["transition"] == "both"
            if row["both_subcategory"] in ("both_correct", "both_wrong")
            else row["both_subcategory"] is None,
            "both subcategory closure drift",
        )
    return expected


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable PJ2 validation already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
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
    parser.add_argument("--authority", type=Path, default=AUTHORITY_PATH)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.authority.resolve() == AUTHORITY_PATH.resolve()
        and args.authority.is_file()
        and not args.authority.is_symlink()
        and args.authority.stat().st_mode & 0o777 == 0o444,
        "PJ2 authority path/mode drift",
    )
    authority = read_json(args.authority)
    require(
        authority.get("namespace") == NAMESPACE
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("formula_contract") == EXPECTED_FORMULA_CONTRACT
        and authority.get("statistical_contract") == EXPECTED_STATISTICAL_CONTRACT
        and authority.get("gate_contract") == EXPECTED_GATE_CONTRACT
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") is None,
        "PJ2 authority contract drift",
    )
    for name in authority.get("bindings", {}):
        bound_path(authority, str(name))
    require(
        args.result.resolve() == (ROOT / str(authority["output"])).resolve()
        and args.result.is_file()
        and not args.result.is_symlink()
        and args.result.stat().st_mode & 0o777 == 0o444
        and args.output.resolve(strict=False)
        == (ROOT / str(authority["validation_output"])).resolve(strict=False),
        "PJ2 result/output path or mode drift",
    )
    pj1_path = bound_path(authority, "pj1_result")
    pj1_validation = read_json(bound_path(authority, "pj1_validation"))
    require(
        pj1_validation.get("status") == PJ1_VALIDATION_STATUS
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("logical_sha256") == logical(pj1_validation)
        and pj1_validation.get("result_sha256") == file_sha(pj1_path),
        "PJ1 independent validation drift",
    )
    pj1 = read_json(pj1_path)
    result = read_json(args.result)
    validate_payload(
        authority,
        result,
        pj1,
        file_sha(args.authority),
        file_sha(pj1_path),
    )
    validation: dict[str, Any] = {
        "schema_version": "rc_h0_c128_target_free_postjoin_pj2_validation_v1_20260824",
        "namespace": NAMESPACE,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "authority_sha256": file_sha(args.authority),
        "authority_logical_sha256": authority["logical_sha256"],
        "result_sha256": file_sha(args.result),
        "result_logical_sha256": result["logical_sha256"],
        "pj1_result_sha256": file_sha(pj1_path),
        "query_count": 594,
        "candidate_count": 76_032,
        "independent_binary64_direction_replay": True,
        "independent_identity_max_tie_replay": True,
        "independent_arm_specific_strongest_wrong_replay": True,
        "independent_strict_rank_replay": True,
        "independent_paired_group_statistics_replay": True,
        "forbidden_cross_arm_margin_difference_computed": False,
        "scientific_GO_or_NO_GO": result["scientific_GO_or_NO_GO"],
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical(validation)
    atomic_write(args.output, validation)
    print(
        json.dumps(
            {"status": VALIDATION_STATUS, "validation_pass": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
