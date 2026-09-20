#!/usr/bin/env python3
"""Evaluate the preregistered Scale-4 component successor on 582 queries."""

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

import reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1 as PJ2
from rc_aslo_xf.dino_rcde_h0_component_scale4_v1 import (
    SCALE,
    build_scale4_candidate_payload,
    pair_transition,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/h0_component_scale4_confirmation_authority_v36_20260824.json"
)
AUTHORITY_STATUS = "H0_COMPONENT_SCALE4_INTERNAL_CONFIRMATION_EXECUTION_AUTHORIZED"
GO = "COMPONENT_SCALE4_INTERNAL_CONFIRMATION_GO"
NO_GO = "COMPONENT_SCALE4_INTERNAL_CONFIRMATION_NO_GO"
ARMS = ("INIT", "TRACK_H", "SCALE4")
EXCLUDED = tuple(range(12))
EXPECTED_FOLDS = Counter({1: 149, 2: 140, 3: 145, 4: 148})
TARGET_MISSES = {25, 26, 101, 346, 354, 470}
EXPECTED_EXECUTIONS = tuple(
    execution for execution in range(12, 600) if execution not in TARGET_MISSES
)
EXPECTED_TRACKS = Counter({"difficult": 58, "new_difficult_train": 24, "outcome": 500})


class Scale4ConfirmationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise Scale4ConfirmationError(message)


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
    if "logical_sha256" in binding:
        payload = read_json(path)
        require(
            payload.get("logical_sha256") == logical(payload) == binding["logical_sha256"],
            f"binding logical drift: {name}",
        )
    return path


def reduce_scored_arm(
    candidates: Sequence[Mapping[str, Any]], arm: str, target_identity: str
) -> dict[str, Any]:
    require(arm in ARMS, "unknown Scale-4 arm")
    reduced: dict[str, Mapping[str, Any]] = {}
    seen_rows: set[int] = set()
    for candidate in candidates:
        physical_row = int(candidate["candidate_physical_row"])
        identity = str(candidate["candidate_corrected_identity"])
        require(physical_row not in seen_rows and identity != "", "candidate identity/row drift")
        seen_rows.add(physical_row)
        score_item = candidate["scores"][arm]
        score = float(score_item["score"])
        require(
            math.isfinite(score) and score_item["score_bits"] == PJ2.float_bits(score),
            "candidate score/bit drift",
        )
        item = {
            "candidate_physical_row": physical_row,
            "candidate_corrected_identity": identity,
            "score": score,
            "score_bits": score_item["score_bits"],
        }
        incumbent = reduced.get(identity)
        if (
            incumbent is None
            or score > float(incumbent["score"])
            or (
                score == float(incumbent["score"])
                and physical_row < int(incumbent["candidate_physical_row"])
            )
        ):
            reduced[identity] = item
    require(target_identity in reduced and len(reduced) > 1, "target/wrong identity absent")
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


def comparison_transition(base: Mapping[str, Any], successor: Mapping[str, Any]) -> str:
    return pair_transition(bool(base["strict_top1"]), bool(successor["strict_top1"]))


def comparison_statistics(
    rows: Sequence[Mapping[str, Any]], *, prefix: str
) -> dict[str, Any]:
    top1_field = f"{prefix}_top1_difference"
    mrr_field = f"{prefix}_mrr_difference"
    transition_field = f"{prefix}_transition"
    top1_groups = PJ2.group_means(rows, top1_field)
    mrr_groups = PJ2.group_means(rows, mrr_field)
    top1_inference = PJ2.registered_inference(top1_groups)
    mrr_inference = PJ2.registered_inference(mrr_groups)
    fold_summaries: dict[str, Any] = {}
    for fold in range(1, 5):
        selected = [row for row in rows if int(row["outer_fold"]) == fold]
        require(len(selected) == EXPECTED_FOLDS[fold], f"fold population drift: {fold}")
        fold_top1 = PJ2.group_means(selected, top1_field)
        fold_mrr = PJ2.group_means(selected, mrr_field)
        fold_summaries[str(fold)] = {
            "query_count": len(selected),
            "supergroup_count": len(fold_top1),
            "group_balanced_top1_gain": float(np.mean(list(fold_top1.values()))),
            "group_balanced_mrr_gain": float(np.mean(list(fold_mrr.values()))),
        }
    transitions = Counter(str(row[transition_field]) for row in rows)
    track_transitions: dict[str, dict[str, int]] = {}
    for track in sorted({str(row["track"]) for row in rows}):
        selected = [row for row in rows if str(row["track"]) == track]
        counts = Counter(str(row[transition_field]) for row in selected)
        track_transitions[track] = {
            "query_count": len(selected),
            "rescue": counts["rescue"],
            "break": counts["break"],
            "both_correct": counts["both_correct"],
            "both_wrong": counts["both_wrong"],
            "net_rescue": counts["rescue"] - counts["break"],
        }
    return {
        "top1_inference": top1_inference,
        "mrr_inference": mrr_inference,
        "fold_summaries": fold_summaries,
        "transition_counts": {
            category: transitions[category]
            for category in ("rescue", "break", "both_correct", "both_wrong")
        },
        "track_transition_counts": track_transitions,
        "supergroup_paired_differences": {
            "top1": top1_groups,
            "reciprocal_rank": mrr_groups,
        },
    }


def reduce_rows(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    PJ2._validate_source_rows(source_rows, exact_population=True)
    selected_sources = [
        row for row in source_rows if int(row["execution_ordinal"]) not in EXCLUDED
    ]
    require(
        len(selected_sources) == 582
        and tuple(int(row["execution_ordinal"]) for row in selected_sources)
        == EXPECTED_EXECUTIONS
        and Counter(int(row["outer_fold"]) for row in selected_sources) == EXPECTED_FOLDS,
        "Scale-4 confirmation population drift",
    )
    require(
        Counter(str(row["track"]) for row in selected_sources) == EXPECTED_TRACKS,
        "Scale-4 track population drift",
    )
    group_fold: dict[str, int] = {}
    group_track: dict[str, str] = {}
    for source in selected_sources:
        group = str(source["supergroup"])
        fold = int(source["outer_fold"])
        track = str(source["track"])
        require(
            group_fold.setdefault(group, fold) == fold
            and group_track.setdefault(group, track) == track,
            "supergroup crosses fold/track",
        )
    require(
        len(group_fold) == 48
        and Counter(group_fold.values()) == Counter({1: 13, 2: 11, 3: 12, 4: 12})
        and Counter(group_track.values())
        == Counter({"difficult": 6, "new_difficult_train": 5, "outcome": 37}),
        "supergroup population drift",
    )
    rows: list[dict[str, Any]] = []
    for source in selected_sources:
        scored_candidates = [
            build_scale4_candidate_payload(candidate) for candidate in source["candidates"]
        ]
        target_identity = str(source["target_corrected_identity"])
        arms = {
            arm: reduce_scored_arm(scored_candidates, arm, target_identity)
            for arm in ARMS
        }
        scale = arms["SCALE4"]
        init = arms["INIT"]
        track_h = arms["TRACK_H"]
        row: dict[str, Any] = {
            "source_pj1_record_sha256": source["record_sha256"],
            "execution_ordinal": int(source["execution_ordinal"]),
            "query_id": str(source["query_id"]),
            "outer_fold": int(source["outer_fold"]),
            "supergroup": str(source["supergroup"]),
            "track": str(source["track"]),
            "target_corrected_identity": target_identity,
            "arms": arms,
            "INIT_strict_top1": int(init["strict_top1"]),
            "INIT_reciprocal_rank": float(init["reciprocal_rank"]),
            "TRACK_H_strict_top1": int(track_h["strict_top1"]),
            "TRACK_H_reciprocal_rank": float(track_h["reciprocal_rank"]),
            "SCALE4_strict_top1": int(scale["strict_top1"]),
            "SCALE4_reciprocal_rank": float(scale["reciprocal_rank"]),
            "scale4_vs_init_top1_difference": int(scale["strict_top1"])
            - int(init["strict_top1"]),
            "scale4_vs_init_mrr_difference": float(
                scale["reciprocal_rank"] - init["reciprocal_rank"]
            ),
            "scale4_vs_init_transition": comparison_transition(init, scale),
            "scale4_vs_track_h_top1_difference": int(scale["strict_top1"])
            - int(track_h["strict_top1"]),
            "scale4_vs_track_h_mrr_difference": float(
                scale["reciprocal_rank"] - track_h["reciprocal_rank"]
            ),
            "scale4_vs_track_h_transition": comparison_transition(track_h, scale),
        }
        row["record_sha256"] = record_sha(row)
        rows.append(row)
    vs_init = comparison_statistics(rows, prefix="scale4_vs_init")
    vs_h = comparison_statistics(rows, prefix="scale4_vs_track_h")
    arm_summaries: dict[str, Any] = {}
    for arm in ARMS:
        top1_values = PJ2.group_means(rows, f"{arm}_strict_top1")
        mrr_values = PJ2.group_means(rows, f"{arm}_reciprocal_rank")
        arm_summaries[arm] = {
            "correct_count": sum(int(row[f"{arm}_strict_top1"]) for row in rows),
            "query_top1_rate": float(
                np.mean([float(row[f"{arm}_strict_top1"]) for row in rows])
            ),
            "query_mean_reciprocal_rank": float(
                np.mean([float(row[f"{arm}_reciprocal_rank"]) for row in rows])
            ),
            "group_balanced_top1": float(np.mean(list(top1_values.values()))),
            "group_balanced_mrr": float(np.mean(list(mrr_values.values()))),
        }
    init_top1 = vs_init["top1_inference"]
    init_mrr = vs_init["mrr_inference"]
    h_top1 = vs_h["top1_inference"]
    h_mrr = vs_h["mrr_inference"]
    gates = {
        "top1_gain_vs_init_at_least_0p03": init_top1["group_balanced_gain"] >= 0.03,
        "top1_vs_init_bootstrap_lower_positive": init_top1["bootstrap_95_ci"][0] > 0.0,
        "top1_vs_init_signflip_p_below_0p05": init_top1["one_sided_group_signflip_p"] < 0.05,
        "mrr_gain_vs_init_positive": init_mrr["group_balanced_gain"] > 0.0,
        "mrr_vs_init_bootstrap_lower_positive": init_mrr["bootstrap_95_ci"][0] > 0.0,
        "mrr_vs_init_signflip_p_below_0p05": init_mrr["one_sided_group_signflip_p"] < 0.05,
        "positive_top1_folds_vs_init_at_least_3": sum(
            item["group_balanced_top1_gain"] > 0.0
            for item in vs_init["fold_summaries"].values()
        ) >= 3,
        "positive_mrr_folds_vs_init_at_least_3": sum(
            item["group_balanced_mrr_gain"] > 0.0
            for item in vs_init["fold_summaries"].values()
        ) >= 3,
        "rescue_strictly_above_break_vs_init": vs_init["transition_counts"]["rescue"]
        > vs_init["transition_counts"]["break"],
        "difficult_rescue_at_least_break_vs_init": vs_init["track_transition_counts"].get(
            "difficult", {"rescue": 0, "break": 1}
        )["rescue"]
        >= vs_init["track_transition_counts"].get("difficult", {"break": 1})["break"],
        "top1_gain_vs_track_h_positive": h_top1["group_balanced_gain"] > 0.0,
        "mrr_gain_vs_track_h_positive": h_mrr["group_balanced_gain"] > 0.0,
        "rescue_strictly_above_break_vs_track_h": vs_h["transition_counts"]["rescue"]
        > vs_h["transition_counts"]["break"],
    }
    return {
        "rows": rows,
        "row_population_sha256": canonical(rows),
        "scale4_vs_init": vs_init,
        "scale4_vs_track_h": vs_h,
        "arm_summaries": arm_summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable Scale-4 output already exists")
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
    require(args.authority.resolve(strict=True) == exact, "Scale-4 authority path drift")
    authority = read_json(exact)
    require(
        authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("scale4_reduction_authorized") is True
        and authority.get("model_forward_authorized") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and args.output.resolve(strict=False)
        == (ROOT / str(authority["output"])).resolve(strict=False),
        "Scale-4 authority envelope drift",
    )
    pj1_path = bound_path(authority, "pj1_result")
    pj1_validation_path = bound_path(authority, "pj1_validation")
    pj1_validation = read_json(pj1_validation_path)
    pj1 = read_json(pj1_path)
    require(
        pj1.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
        and pj1.get("logical_sha256") == logical(pj1)
        and pj1_validation.get("status")
        == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS"
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("result_sha256") == file_sha(pj1_path),
        "PJ1 qualification drift",
    )
    reduction = reduce_rows(pj1["rows"])
    go = bool(reduction["all_gates_pass"])
    value: dict[str, Any] = {
        "schema_version": "rc_h0_component_scale4_confirmation_result_v1_20260824",
        "status": GO if go else NO_GO,
        "claim_level": "N582_INTERNAL_C128_CONFIRMATION_ONLY",
        "authority_sha256": file_sha(exact),
        "authority_logical_sha256": authority["logical_sha256"],
        "preregistered_scale": SCALE,
        "development_execution_ordinals_excluded": list(EXCLUDED),
        "query_count": 582,
        "candidate_count": 74_496,
        "pj1_result_sha256": file_sha(pj1_path),
        "pj1_result_logical_sha256": pj1["logical_sha256"],
        **reduction,
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
        "alternative_scale_count": 0,
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
                "query_count": 582,
                "scale4_vs_init_rescue": reduction["scale4_vs_init"]["transition_counts"]["rescue"],
                "scale4_vs_init_break": reduction["scale4_vs_init"]["transition_counts"]["break"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
