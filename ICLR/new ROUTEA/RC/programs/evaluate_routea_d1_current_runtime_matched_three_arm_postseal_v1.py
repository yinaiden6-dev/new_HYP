#!/usr/bin/env python3
"""Post-seal local-evidence diagnostic for matched A/B/C on RAW and D1."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT.parent / "route_a_core"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402
from route_a.o1_c6direct_m1_d1 import (  # noqa: E402
    aggregate_exact_label_scores,
    exact_label_diagnostic,
)


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_V1_20260902.md"
ARM_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
ARM_VALIDATION = ARM_ROOT / "validation.json"
BASE_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
BASE_VALIDATION = BASE_ROOT / "validation.json"
ROLE_ROOT = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
ROLE_MANIFEST = ROLE_ROOT / "role_manifest.json"
ROLE_VALIDATION = ROLE_ROOT / "independent_validation.json"
OUT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_postseal_v1/result.json"
BASES = ("RAW", "D1")
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arm_diagnostic(
    axis: list[int],
    values: list[float],
    labels: tuple[str, ...],
    target: str,
    base_rival: str,
) -> dict:
    by_identity: dict[str, float] = {}
    for row, score in zip(axis, values, strict=True):
        identity = labels[row]
        prior = by_identity.get(identity)
        if prior is None or score > prior:
            by_identity[identity] = score
    if len(by_identity) != 128 or target not in by_identity or base_rival not in by_identity:
        raise RuntimeError("local exact-identity axis drift")
    target_score = by_identity[target]
    wrong = sorted(
        ((identity, score) for identity, score in by_identity.items() if identity != target),
        key=lambda item: item[0],
    )
    best_wrong_score = max(score for _, score in wrong)
    best_wrong_identity = next(
        identity for identity, score in wrong if score == best_wrong_score
    )
    return {
        "rank": 1 + sum(score >= target_score for _, score in wrong),
        "target_score": target_score,
        "strongest_wrong_identity": best_wrong_identity,
        "strongest_wrong_score": best_wrong_score,
        "own_margin": target_score - best_wrong_score,
        "base_rival_identity": base_rival,
        "base_rival_score": by_identity[base_rival],
        "base_rival_margin": target_score - by_identity[base_rival],
    }


def arm_summary(cases: list[dict], base: str, arm: str) -> dict:
    metrics = [case["bases"][base]["arms"][arm] for case in cases]
    count = len(metrics)
    real_top1_count = sum(item["real"]["rank"] == 1 for item in metrics)
    real_top1_retained = sum(
        item["real"]["rank"] == 1 and item["cbind"]["rank"] == 1
        for item in metrics
    )
    return {
        "query_count": count,
        "top1": sum(item["real"]["rank"] == 1 for item in metrics),
        "R@1": sum(item["real"]["rank"] == 1 for item in metrics) / count,
        "MRR": sum(1.0 / item["real"]["rank"] for item in metrics) / count,
        "mean_own_margin": sum(item["real"]["own_margin"] for item in metrics) / count,
        "target_gt_base_rival": sum(
            item["real"]["base_rival_margin"] > 0 for item in metrics
        ),
        "mean_base_rival_margin": sum(
            item["real"]["base_rival_margin"] for item in metrics
        )
        / count,
        "cbind_top1": sum(item["cbind"]["rank"] == 1 for item in metrics),
        "cbind_MRR": sum(1.0 / item["cbind"]["rank"] for item in metrics) / count,
        "real_top1_count": real_top1_count,
        "real_top1_retained_by_cbind": real_top1_retained,
        "real_top1_retention": (
            real_top1_retained / real_top1_count if real_top1_count else 1.0
        ),
        "mean_cbind_own_margin": sum(
            item["cbind"]["own_margin"] for item in metrics
        )
        / count,
        "real_margin_gt_cbind": sum(
            item["real"]["own_margin"] > item["cbind"]["own_margin"]
            for item in metrics
        ),
        "mean_real_minus_cbind_margin": sum(
            item["real"]["own_margin"] - item["cbind"]["own_margin"]
            for item in metrics
        )
        / count,
    }


def cohort_summary(cases: list[dict]) -> dict:
    return {
        base: {arm: arm_summary(cases, base, arm) for arm in ARMS}
        for base in BASES
    }


def comparison_summary(cases: list[dict], base: str, other_arm: str) -> dict:
    paired = [case["bases"][base]["arms"]["C_PAIRED"]["real"] for case in cases]
    other = [case["bases"][base]["arms"][other_arm]["real"] for case in cases]
    return {
        "query_count": len(cases),
        "C_rank_better": sum(left["rank"] < right["rank"] for left, right in zip(paired, other, strict=True)),
        "C_rank_equal": sum(left["rank"] == right["rank"] for left, right in zip(paired, other, strict=True)),
        "C_rank_worse": sum(left["rank"] > right["rank"] for left, right in zip(paired, other, strict=True)),
        "C_margin_greater": sum(left["own_margin"] > right["own_margin"] for left, right in zip(paired, other, strict=True)),
        "C_margin_equal": sum(left["own_margin"] == right["own_margin"] for left, right in zip(paired, other, strict=True)),
        "C_margin_lower": sum(left["own_margin"] < right["own_margin"] for left, right in zip(paired, other, strict=True)),
    }


def comparison_cohort(cases: list[dict]) -> dict:
    return {
        base: {
            "C_vs_A": comparison_summary(cases, base, "A_ALL"),
            "C_vs_B": comparison_summary(cases, base, "B_QUERY"),
        }
        for base in BASES
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable matched-three-arm postseal result exists: {OUT}")
    arm_validation = json.loads(ARM_VALIDATION.read_text())
    base_validation = json.loads(BASE_VALIDATION.read_text())
    role_manifest = json.loads(ROLE_MANIFEST.read_text())
    role_validation = json.loads(ROLE_VALIDATION.read_text())
    if (
        arm_validation.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        or arm_validation.get("logical_sha256") != e0.logical_sha256(arm_validation)
        or not all(arm_validation.get("checks", {}).values())
        or arm_validation.get("next_authorized_stage")
        != "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_DIAGNOSTIC"
        or base_validation.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        or base_validation.get("logical_sha256") != e0.logical_sha256(base_validation)
        or role_manifest.get("status") != "RGH_P0_A0_ROLE_MANIFEST_READY"
        or role_manifest.get("logical_sha256") != e0.logical_sha256(role_manifest)
        or role_validation.get("status")
        != "RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS"
        or role_validation.get("logical_sha256") != e0.logical_sha256(role_validation)
    ):
        raise RuntimeError("postseal authority drift")
    arm_seals = {int(item["shard"]): item for item in arm_validation["shards"]}
    base_seals = {int(item["shard"]): item for item in base_validation["shards"]}
    role_entries = {
        int(item["execution_ordinal"]): item for item in role_manifest["shards"]
    }
    gallery = build_gallery_source(verify_cache_file_sha256=True)
    labels = gallery.corrected_identities
    if len(labels) != 5413 or len(set(labels)) != 5412:
        raise RuntimeError("corrected gallery population drift")
    cases: list[dict] = []
    shard_bindings: list[dict] = []
    for shard in range(8):
        arm_payload_path = ARM_ROOT / f"shard{shard:02d}/payload.pt"
        arm_receipt_path = ARM_ROOT / f"shard{shard:02d}/receipt.json"
        arm_shard_validation_path = ARM_ROOT / f"shard{shard:02d}/validation.json"
        base_payload_path = BASE_ROOT / f"shard{shard:02d}/payload.pt"
        base_receipt_path = BASE_ROOT / f"shard{shard:02d}/receipt.json"
        base_shard_validation_path = BASE_ROOT / f"shard{shard:02d}/validation.json"
        arm_seal = arm_seals[shard]
        base_seal = base_seals[shard]
        if not (
            arm_seal["payload_sha256"] == e0.sha256_file(arm_payload_path)
            and arm_seal["receipt_sha256"] == e0.sha256_file(arm_receipt_path)
            and arm_seal["validation_sha256"]
            == e0.sha256_file(arm_shard_validation_path)
            and base_seal["payload_sha256"] == e0.sha256_file(base_payload_path)
            and base_seal["receipt_sha256"] == e0.sha256_file(base_receipt_path)
            and base_seal["validation_sha256"]
            == e0.sha256_file(base_shard_validation_path)
        ):
            raise RuntimeError("postseal shard seal drift")
        arm_payload = torch.load(
            arm_payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        base_payload = torch.load(
            base_payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        base_by_execution = {
            int(record["execution_ordinal"]): record for record in base_payload["records"]
        }
        for arm_record in arm_payload["records"]:
            execution = int(arm_record["execution_ordinal"])
            base_record = base_by_execution[execution]
            role_entry = role_entries[execution]
            role_path = Path(role_entry["path"])
            role = json.loads(role_path.read_text())
            if not (
                role_entry["sha256"] == e0.sha256_file(role_path)
                and role.get("logical_sha256") == e0.logical_sha256(role)
                and role.get("status") == "RGH_P0_A0_ROLE_SHARD_READY"
                and role.get("execution_ordinal") == execution
                and role.get("query_id") == arm_record["query_id"]
                and role.get("track") == arm_record["track"]
                and role.get("target_insertion_count") == 0
                and role.get("target_spatial_supervision_count") == 0
                and role.get("raw_d1_field_count") == 0
            ):
                raise RuntimeError("postseal role shard drift")
            target = str(role["identity"])
            if target not in set(labels):
                raise RuntimeError("target identity absent from corrected gallery")
            candidate_by_row = {
                int(candidate["physical_row"]): candidate
                for candidate in arm_record["candidates"]
            }
            base_outputs: dict[str, dict] = {}
            for base, axis_key, full_score_key, arm_score_key, cbind_key in (
                (
                    "RAW",
                    "raw_candidate_physical_rows",
                    "raw_full_gallery_scores",
                    "raw_arm_scores",
                    "raw_cbind_destination_to_source_physical_rows",
                ),
                (
                    "D1",
                    "d1_candidate_physical_rows",
                    "d1_full_gallery_scores",
                    "d1_arm_scores",
                    "d1_cbind_destination_to_source_physical_rows",
                ),
            ):
                axis = list(map(int, arm_record[axis_key]))
                reduced = aggregate_exact_label_scores(
                    torch.as_tensor(base_record[full_score_key], dtype=torch.float64),
                    labels,
                )
                base_diag = exact_label_diagnostic(reduced, target)
                base_summary = {
                    "rank": int(base_diag.rank),
                    "target_score": base_diag.target_score,
                    "strongest_wrong_identity": str(base_diag.best_rival_label),
                    "strongest_wrong_score": base_diag.best_rival_score,
                    "margin": base_diag.margin,
                }
                arms: dict[str, dict] = {}
                cbind_rows = list(map(int, arm_record[cbind_key]))
                for arm in ARMS:
                    real_values = [
                        float(candidate_by_row[row][arm_score_key][arm]) for row in axis
                    ]
                    cbind_values = [
                        float(candidate_by_row[row][arm_score_key][arm])
                        for row in cbind_rows
                    ]
                    arms[arm] = {
                        "real": arm_diagnostic(
                            axis,
                            real_values,
                            labels,
                            target,
                            base_summary["strongest_wrong_identity"],
                        ),
                        "cbind": arm_diagnostic(
                            axis,
                            cbind_values,
                            labels,
                            target,
                            base_summary["strongest_wrong_identity"],
                        ),
                    }
                base_outputs[base] = {"base": base_summary, "arms": arms}
            cases.append(
                {
                    "execution_ordinal": execution,
                    "query_id": arm_record["query_id"],
                    "role": arm_record["role"],
                    "track": arm_record["track"],
                    "heldout_fold": int(arm_record["heldout_fold"]),
                    "target_identity": target,
                    "supergroup": str(role["supergroup"]),
                    "bases": base_outputs,
                    "role_shard_sha256": role_entry["sha256"],
                }
            )
        shard_bindings.append(
            {
                "shard": shard,
                "arm_payload_sha256": arm_seal["payload_sha256"],
                "arm_receipt_sha256": arm_seal["receipt_sha256"],
                "arm_validation_sha256": arm_seal["validation_sha256"],
                "base_payload_sha256": base_seal["payload_sha256"],
                "base_receipt_sha256": base_seal["receipt_sha256"],
                "base_validation_sha256": base_seal["validation_sha256"],
            }
        )
    cases.sort(key=lambda case: case["execution_ordinal"])
    if not (len(cases) == len({case["query_id"] for case in cases}) == 64):
        raise RuntimeError("postseal case population drift")
    summary_all = cohort_summary(cases)
    summary_by_role = {
        role: cohort_summary([case for case in cases if case["role"] == role])
        for role in ("TRAIN", "EVAL")
    }
    comparison_by_role = {
        role: comparison_cohort([case for case in cases if case["role"] == role])
        for role in ("TRAIN", "EVAL")
    }
    comparison_all = comparison_cohort(cases)
    summary_by_track = {
        track: cohort_summary([case for case in cases if case["track"] == track])
        for track in sorted({case["track"] for case in cases})
    }
    summary_eval_by_fold = {
        str(fold): cohort_summary(
            [
                case
                for case in cases
                if case["role"] == "EVAL" and case["heldout_fold"] == fold
            ]
        )
        for fold in sorted(
            {
                case["heldout_fold"]
                for case in cases
                if case["role"] == "EVAL"
            }
        )
    }
    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_postseal_v1_20260902",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_DIAGNOSTIC_COMPLETE",
        "claim_level": "INTERNAL_LOCAL_EVIDENCE_DIAGNOSTIC_NO_MODEL_SELECTION",
        "primary_role": "EVAL",
        "primary_base": "RAW",
        "d1_checkpoint_scope": "HISTORICAL_RUNTIME_CHECKPOINT_PORTABILITY_ONLY",
        "summary_all": summary_all,
        "summary_by_role": summary_by_role,
        "comparison_by_role": comparison_by_role,
        "comparison_all": comparison_all,
        "summary_by_track": summary_by_track,
        "summary_eval_by_heldout_fold": summary_eval_by_fold,
        "cases": cases,
        "bindings": {
            "contract_sha256": e0.sha256_file(CONTRACT),
            "arm_validation_sha256": e0.sha256_file(ARM_VALIDATION),
            "base_validation_sha256": e0.sha256_file(BASE_VALIDATION),
            "role_manifest_sha256": e0.sha256_file(ROLE_MANIFEST),
            "role_validation_sha256": e0.sha256_file(ROLE_VALIDATION),
            "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
            "shards": shard_bindings,
        },
        "target_join_after_all_prejoin_shards": True,
        "target_role_read_count": 64,
        "model_update_count": 0,
        "sealed_read_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "eval": summary_by_role["EVAL"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
