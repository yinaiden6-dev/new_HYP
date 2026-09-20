#!/usr/bin/env python3
"""Confirm ColNomic RAW plus unit candidate-component residual on 582 queries."""

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

import numpy as np
import torch

import reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as PJ2
import reduce_dino_rcde_h0_component_scale4_confirmation_v1 as S4
from rc_aslo_xf.dino_rcde_colnomic_component_residual_v1 import (
    RESIDUAL_WEIGHT,
    build_fused_candidate_payload,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/colnomic_component_residual_confirmation_authority_v38_20260824.json"
)
AUTHORITY_STATUS = "COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_CONFIRMATION_EXECUTION_AUTHORIZED"
GO = "COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_GO"
NO_GO = "COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_NO_GO"
ARMS = ("RAW", "FUSED")
EXCLUDED = tuple(range(12))


class ResidualConfirmationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ResidualConfirmationError(message)


def canonical(value: Any) -> str:
    return PJ2.canonical(value)


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
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def bound_path(authority: Mapping[str, Any], name: str) -> Path:
    binding = authority.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"binding absent: {name}")
    path = (ROOT / str(binding.get("path", ""))).resolve()
    require(
        ROOT in path.parents
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(binding.get("bytes", -1))
        and file_sha(path) == binding.get("sha256"),
        f"binding drift: {name}",
    )
    return path


def reduce_arm(
    candidates: Sequence[Mapping[str, Any]], arm: str, target_identity: str
) -> dict[str, Any]:
    require(arm in ARMS, "unknown residual arm")
    reduced: dict[str, Mapping[str, Any]] = {}
    seen_rows: set[int] = set()
    for candidate in candidates:
        row = int(candidate["candidate_physical_row"])
        identity = str(candidate["candidate_corrected_identity"])
        require(row not in seen_rows and identity != "", "candidate row/identity drift")
        seen_rows.add(row)
        item_score = candidate["scores"][arm]
        score = float(item_score["score"])
        require(
            math.isfinite(score) and item_score["score_bits"] == PJ2.float_bits(score),
            "residual score/bit drift",
        )
        item = {
            "candidate_physical_row": row,
            "candidate_corrected_identity": identity,
            "score": score,
            "score_bits": item_score["score_bits"],
        }
        incumbent = reduced.get(identity)
        if (
            incumbent is None
            or score > float(incumbent["score"])
            or (
                score == float(incumbent["score"])
                and row < int(incumbent["candidate_physical_row"])
            )
        ):
            reduced[identity] = item
    require(target_identity in reduced and len(reduced) > 1, "target/wrong absent")
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
    rank = 1 + sum(
        float(item["score"]) >= target_score
        for identity, item in reduced.items()
        if identity != target_identity
    )
    return {
        "reduced_identity_count": len(reduced),
        "target": {
            "corrected_identity": target_identity,
            "physical_row": int(target["candidate_physical_row"]),
            "score": target_score,
            "score_bits": PJ2.float_bits(target_score),
        },
        "strongest_wrong": {
            "corrected_identity": str(wrong["candidate_corrected_identity"]),
            "physical_row": int(wrong["candidate_physical_row"]),
            "score": wrong_score,
            "score_bits": PJ2.float_bits(wrong_score),
        },
        "margin": margin,
        "margin_bits": PJ2.float_bits(margin),
        "strict_top1": margin > 0.0,
        "rank": int(rank),
        "reciprocal_rank": float(1.0 / rank),
    }


def transition(base: Mapping[str, Any], fused: Mapping[str, Any]) -> str:
    left, right = bool(base["strict_top1"]), bool(fused["strict_top1"])
    if not left and right:
        return "rescue"
    if left and not right:
        return "break"
    return "both_correct" if left else "both_wrong"


def validate_raw_prejoin(raw: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = raw.get("records")
    require(
        raw.get("status") == "RCDE_SR0_MT_I0_PREJOIN_LEDGER_SEALED"
        and raw.get("query_count") == 600
        and raw.get("candidate_count_per_query") == 128
        and raw.get("target_or_label_join_count") == 0
        and raw.get("label_read_count") == 0
        and isinstance(records, list)
        and len(records) == 600
        and raw.get("record_sequence_sha256") == canonical(records),
        "RAW prejoin envelope drift",
    )
    result: dict[int, Mapping[str, Any]] = {}
    for execution, row in enumerate(records):
        require(
            int(row["execution_ordinal"]) == execution
            and row["record_sha256"] == record_sha(row)
            and len(row["candidate_physical_rows"]) == 128
            and len(row["candidate_raw_score_bits"]) == 128,
            "RAW prejoin row drift",
        )
        result[execution] = row
    return result


def reduce_rows(
    pj1_rows: Sequence[Mapping[str, Any]], raw_by_execution: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    PJ2._validate_source_rows(pj1_rows, exact_population=True)
    sources = [row for row in pj1_rows if int(row["execution_ordinal"]) not in EXCLUDED]
    require(
        tuple(int(row["execution_ordinal"]) for row in sources) == S4.EXPECTED_EXECUTIONS
        and Counter(int(row["outer_fold"]) for row in sources) == S4.EXPECTED_FOLDS
        and Counter(str(row["track"]) for row in sources) == S4.EXPECTED_TRACKS,
        "residual confirmation population drift",
    )
    output_rows: list[dict[str, Any]] = []
    for source in sources:
        execution = int(source["execution_ordinal"])
        raw = raw_by_execution[execution]
        require(
            raw["query_id"] == source["query_id"]
            and int(raw["outer_fold"]) == int(source["outer_fold"])
            and list(map(int, raw["candidate_physical_rows"]))
            == [int(candidate["candidate_physical_row"]) for candidate in source["candidates"]],
            "RAW/PJ1 query-fold-candidate axis drift",
        )
        fused_candidates = [
            build_fused_candidate_payload(score_bits, candidate)
            for score_bits, candidate in zip(
                raw["candidate_raw_score_bits"], source["candidates"], strict=True
            )
        ]
        target_identity = str(source["target_corrected_identity"])
        arms = {arm: reduce_arm(fused_candidates, arm, target_identity) for arm in ARMS}
        base, fused = arms["RAW"], arms["FUSED"]
        row: dict[str, Any] = {
            "source_pj1_record_sha256": source["record_sha256"],
            "source_raw_record_sha256": raw["record_sha256"],
            "execution_ordinal": execution,
            "query_id": str(source["query_id"]),
            "outer_fold": int(source["outer_fold"]),
            "supergroup": str(source["supergroup"]),
            "track": str(source["track"]),
            "target_corrected_identity": target_identity,
            "arms": arms,
            "RAW_strict_top1": int(base["strict_top1"]),
            "RAW_reciprocal_rank": float(base["reciprocal_rank"]),
            "FUSED_strict_top1": int(fused["strict_top1"]),
            "FUSED_reciprocal_rank": float(fused["reciprocal_rank"]),
            "fused_vs_raw_top1_difference": int(fused["strict_top1"])
            - int(base["strict_top1"]),
            "fused_vs_raw_mrr_difference": float(
                fused["reciprocal_rank"] - base["reciprocal_rank"]
            ),
            "fused_vs_raw_transition": transition(base, fused),
        }
        row["record_sha256"] = record_sha(row)
        output_rows.append(row)
    comparison = S4.comparison_statistics(output_rows, prefix="fused_vs_raw")
    arm_summaries: dict[str, Any] = {}
    for arm in ARMS:
        top1_groups = PJ2.group_means(output_rows, f"{arm}_strict_top1")
        mrr_groups = PJ2.group_means(output_rows, f"{arm}_reciprocal_rank")
        arm_summaries[arm] = {
            "correct_count": sum(int(row[f"{arm}_strict_top1"]) for row in output_rows),
            "query_top1_rate": float(np.mean([row[f"{arm}_strict_top1"] for row in output_rows])),
            "query_mean_reciprocal_rank": float(
                np.mean([row[f"{arm}_reciprocal_rank"] for row in output_rows])
            ),
            "group_balanced_top1": float(np.mean(list(top1_groups.values()))),
            "group_balanced_mrr": float(np.mean(list(mrr_groups.values()))),
        }
    top1 = comparison["top1_inference"]
    mrr = comparison["mrr_inference"]
    transitions = comparison["transition_counts"]
    gates = {
        "top1_gain_vs_raw_at_least_0p03": top1["group_balanced_gain"] >= 0.03,
        "top1_bootstrap_lower_positive": top1["bootstrap_95_ci"][0] > 0.0,
        "top1_signflip_p_below_0p05": top1["one_sided_group_signflip_p"] < 0.05,
        "mrr_gain_vs_raw_positive": mrr["group_balanced_gain"] > 0.0,
        "mrr_bootstrap_lower_positive": mrr["bootstrap_95_ci"][0] > 0.0,
        "mrr_signflip_p_below_0p05": mrr["one_sided_group_signflip_p"] < 0.05,
        "positive_top1_folds_at_least_3": sum(
            item["group_balanced_top1_gain"] > 0.0
            for item in comparison["fold_summaries"].values()
        ) >= 3,
        "positive_mrr_folds_at_least_3": sum(
            item["group_balanced_mrr_gain"] > 0.0
            for item in comparison["fold_summaries"].values()
        ) >= 3,
        "rescue_strictly_above_break_vs_raw": transitions["rescue"] > transitions["break"],
        "difficult_rescue_at_least_break_vs_raw": comparison["track_transition_counts"][
            "difficult"
        ]["rescue"]
        >= comparison["track_transition_counts"]["difficult"]["break"],
        "absolute_top1_strictly_above_raw": arm_summaries["FUSED"]["group_balanced_top1"]
        > arm_summaries["RAW"]["group_balanced_top1"],
        "absolute_mrr_strictly_above_raw": arm_summaries["FUSED"]["group_balanced_mrr"]
        > arm_summaries["RAW"]["group_balanced_mrr"],
    }
    return {
        "rows": output_rows,
        "row_population_sha256": canonical(output_rows),
        "arm_summaries": arm_summaries,
        "fused_vs_raw": comparison,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable residual output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=ROOT / AUTHORITY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exact = (ROOT / AUTHORITY).resolve(strict=True)
    require(args.authority.resolve(strict=True) == exact, "residual authority path drift")
    authority = read_json(exact)
    require(
        authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("residual_reduction_authorized") is True
        and authority.get("residual_weight") == 1.0
        and authority.get("model_forward_authorized") is False
        and args.output.resolve(strict=False)
        == (ROOT / str(authority["output"])).resolve(strict=False),
        "residual authority envelope drift",
    )
    raw_path = bound_path(authority, "raw_prejoin_ledger")
    raw_validation_path = bound_path(authority, "raw_prejoin_validation")
    pj1_path = bound_path(authority, "pj1_result")
    pj1_validation_path = bound_path(authority, "pj1_validation")
    raw_validation = read_json(raw_validation_path)
    pj1_validation = read_json(pj1_validation_path)
    require(
        raw_validation.get("status") == "RCDE_SR0_MT_I0_PREJOIN_INDEPENDENT_VALIDATION_PASS"
        and raw_validation.get("prejoin_ledger_sha256") == file_sha(raw_path)
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("result_sha256") == file_sha(pj1_path),
        "raw/PJ1 validation drift",
    )
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    pj1 = read_json(pj1_path)
    reduction = reduce_rows(pj1["rows"], validate_raw_prejoin(raw))
    go = bool(reduction["all_gates_pass"])
    value: dict[str, Any] = {
        "schema_version": "rc_colnomic_component_residual_confirmation_result_v1_20260824",
        "status": GO if go else NO_GO,
        "claim_level": "N582_INTERNAL_NATURAL_C128_INCREMENT_ONLY",
        "authority_sha256": file_sha(exact),
        "authority_logical_sha256": authority["logical_sha256"],
        "residual_weight": RESIDUAL_WEIGHT,
        "development_execution_ordinals_excluded": list(EXCLUDED),
        "query_count": 582,
        "candidate_count": 74496,
        "raw_prejoin_ledger_sha256": file_sha(raw_path),
        "pj1_result_sha256": file_sha(pj1_path),
        **reduction,
        "alternative_weight_count": 0,
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
        "full_gallery_retrieval_or_ownership_claim_authorized": False,
        "scientific_GO_or_NO_GO": "GO" if go else "NO_GO",
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical(value)
    atomic_write(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "raw_correct": reduction["arm_summaries"]["RAW"]["correct_count"],
                "fused_correct": reduction["arm_summaries"]["FUSED"]["correct_count"],
                "rescue": reduction["fused_vs_raw"]["transition_counts"]["rescue"],
                "break": reduction["fused_vs_raw"]["transition_counts"]["break"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
