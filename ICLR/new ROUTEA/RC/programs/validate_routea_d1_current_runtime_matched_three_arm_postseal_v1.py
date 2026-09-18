#!/usr/bin/env python3
"""Independent source-to-metric replay of the matched A/B/C post-seal result."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_V1_20260902.md"
ARM_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
ARM_VALIDATION = ARM_ROOT / "validation.json"
BASE_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
BASE_VALIDATION = BASE_ROOT / "validation.json"
ROLE_ROOT = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
ROLE_MANIFEST = ROLE_ROOT / "role_manifest.json"
ROLE_VALIDATION = ROLE_ROOT / "independent_validation.json"
RESULT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_postseal_v1/result.json"
OUT = RESULT.parent / "independent_validation.json"
BASES = ("RAW", "D1")
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
RESULT_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "primary_role",
    "primary_base",
    "d1_checkpoint_scope",
    "summary_all",
    "summary_by_role",
    "comparison_by_role",
    "comparison_all",
    "summary_by_track",
    "summary_eval_by_heldout_fold",
    "cases",
    "bindings",
    "target_join_after_all_prejoin_shards",
    "target_role_read_count",
    "model_update_count",
    "sealed_read_count",
    "scientific_GO_or_NO_GO",
    "next_authorized_stage",
    "logical_sha256",
}


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


def diagnostic(values_by_identity: dict[str, float], target: str) -> dict:
    target_score = values_by_identity[target]
    wrong = sorted(
        ((identity, score) for identity, score in values_by_identity.items() if identity != target),
        key=lambda item: item[0],
    )
    best = max(score for _, score in wrong)
    rival = next(identity for identity, score in wrong if score == best)
    return {
        "rank": 1 + sum(score >= target_score for _, score in wrong),
        "target_score": target_score,
        "strongest_wrong_identity": rival,
        "strongest_wrong_score": best,
        "margin": target_score - best,
    }


def full_diagnostic(scores: torch.Tensor, labels: tuple[str, ...], target: str) -> dict:
    reduced: dict[str, float] = {}
    for label, value in zip(labels, scores.tolist(), strict=True):
        score = float(value)
        prior = reduced.get(label)
        if prior is None or score > prior:
            reduced[label] = score
    if len(reduced) != 5412 or target not in reduced:
        raise RuntimeError("independent full-gallery identity reduction drift")
    return diagnostic(reduced, target)


def local_diagnostic(
    axis: list[int],
    values: list[float],
    labels: tuple[str, ...],
    target: str,
    base_rival: str,
) -> dict:
    reduced = {labels[row]: float(score) for row, score in zip(axis, values, strict=True)}
    if len(reduced) != 128 or target not in reduced or base_rival not in reduced:
        raise RuntimeError("independent local identity reduction drift")
    own = diagnostic(reduced, target)
    return {
        "rank": own["rank"],
        "target_score": own["target_score"],
        "strongest_wrong_identity": own["strongest_wrong_identity"],
        "strongest_wrong_score": own["strongest_wrong_score"],
        "own_margin": own["margin"],
        "base_rival_identity": base_rival,
        "base_rival_score": reduced[base_rival],
        "base_rival_margin": reduced[target] - reduced[base_rival],
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
        "target_gt_base_rival": sum(item["real"]["base_rival_margin"] > 0 for item in metrics),
        "mean_base_rival_margin": sum(item["real"]["base_rival_margin"] for item in metrics) / count,
        "cbind_top1": sum(item["cbind"]["rank"] == 1 for item in metrics),
        "cbind_MRR": sum(1.0 / item["cbind"]["rank"] for item in metrics) / count,
        "real_top1_count": real_top1_count,
        "real_top1_retained_by_cbind": real_top1_retained,
        "real_top1_retention": (
            real_top1_retained / real_top1_count if real_top1_count else 1.0
        ),
        "mean_cbind_own_margin": sum(item["cbind"]["own_margin"] for item in metrics) / count,
        "real_margin_gt_cbind": sum(item["real"]["own_margin"] > item["cbind"]["own_margin"] for item in metrics),
        "mean_real_minus_cbind_margin": sum(item["real"]["own_margin"] - item["cbind"]["own_margin"] for item in metrics) / count,
    }


def cohort(cases: list[dict]) -> dict:
    return {base: {arm: arm_summary(cases, base, arm) for arm in ARMS} for base in BASES}


def comparison(cases: list[dict], base: str, other: str) -> dict:
    left = [case["bases"][base]["arms"]["C_PAIRED"]["real"] for case in cases]
    right = [case["bases"][base]["arms"][other]["real"] for case in cases]
    return {
        "query_count": len(cases),
        "C_rank_better": sum(a["rank"] < b["rank"] for a, b in zip(left, right, strict=True)),
        "C_rank_equal": sum(a["rank"] == b["rank"] for a, b in zip(left, right, strict=True)),
        "C_rank_worse": sum(a["rank"] > b["rank"] for a, b in zip(left, right, strict=True)),
        "C_margin_greater": sum(a["own_margin"] > b["own_margin"] for a, b in zip(left, right, strict=True)),
        "C_margin_equal": sum(a["own_margin"] == b["own_margin"] for a, b in zip(left, right, strict=True)),
        "C_margin_lower": sum(a["own_margin"] < b["own_margin"] for a, b in zip(left, right, strict=True)),
    }


def comparison_cohort(cases: list[dict]) -> dict:
    return {
        base: {
            "C_vs_A": comparison(cases, base, "A_ALL"),
            "C_vs_B": comparison(cases, base, "B_QUERY"),
        }
        for base in BASES
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable postseal independent validation exists: {OUT}")
    result = json.loads(RESULT.read_text())
    arm_validation = json.loads(ARM_VALIDATION.read_text())
    base_validation = json.loads(BASE_VALIDATION.read_text())
    role_manifest = json.loads(ROLE_MANIFEST.read_text())
    role_validation = json.loads(ROLE_VALIDATION.read_text())
    arm_seals = {int(item["shard"]): item for item in arm_validation["shards"]}
    base_seals = {int(item["shard"]): item for item in base_validation["shards"]}
    role_entries = {int(item["execution_ordinal"]): item for item in role_manifest["shards"]}
    gallery = build_gallery_source(verify_cache_file_sha256=True)
    labels = gallery.corrected_identities
    expected_cases: list[dict] = []
    expected_shards: list[dict] = []
    for shard in range(8):
        arm_payload_path = ARM_ROOT / f"shard{shard:02d}/payload.pt"
        arm_receipt_path = ARM_ROOT / f"shard{shard:02d}/receipt.json"
        arm_shard_validation_path = ARM_ROOT / f"shard{shard:02d}/validation.json"
        base_payload_path = BASE_ROOT / f"shard{shard:02d}/payload.pt"
        base_receipt_path = BASE_ROOT / f"shard{shard:02d}/receipt.json"
        base_shard_validation_path = BASE_ROOT / f"shard{shard:02d}/validation.json"
        arm_seal, base_seal = arm_seals[shard], base_seals[shard]
        if not (
            arm_seal["payload_sha256"] == e0.sha256_file(arm_payload_path)
            and arm_seal["receipt_sha256"] == e0.sha256_file(arm_receipt_path)
            and arm_seal["validation_sha256"] == e0.sha256_file(arm_shard_validation_path)
            and base_seal["payload_sha256"] == e0.sha256_file(base_payload_path)
            and base_seal["receipt_sha256"] == e0.sha256_file(base_receipt_path)
            and base_seal["validation_sha256"] == e0.sha256_file(base_shard_validation_path)
        ):
            raise RuntimeError("independent postseal shard drift")
        arms = torch.load(arm_payload_path, map_location="cpu", weights_only=False, mmap=True)
        bases = torch.load(base_payload_path, map_location="cpu", weights_only=False, mmap=True)
        base_by_execution = {int(item["execution_ordinal"]): item for item in bases["records"]}
        for arm_record in arms["records"]:
            execution = int(arm_record["execution_ordinal"])
            base_record = base_by_execution[execution]
            role_entry = role_entries[execution]
            role_path = Path(role_entry["path"])
            role = json.loads(role_path.read_text())
            if not (
                e0.sha256_file(role_path) == role_entry["sha256"]
                and role.get("logical_sha256") == e0.logical_sha256(role)
                and role.get("status") == "RGH_P0_A0_ROLE_SHARD_READY"
                and role.get("query_id") == arm_record["query_id"]
                and role.get("track") == arm_record["track"]
                and role.get("target_insertion_count") == 0
                and role.get("target_spatial_supervision_count") == 0
                and role.get("raw_d1_field_count") == 0
            ):
                raise RuntimeError("independent role drift")
            target = str(role["identity"])
            candidate_by_row = {int(item["physical_row"]): item for item in arm_record["candidates"]}
            output_bases = {}
            for base, axis_key, full_key, score_key, cbind_key in (
                ("RAW", "raw_candidate_physical_rows", "raw_full_gallery_scores", "raw_arm_scores", "raw_cbind_destination_to_source_physical_rows"),
                ("D1", "d1_candidate_physical_rows", "d1_full_gallery_scores", "d1_arm_scores", "d1_cbind_destination_to_source_physical_rows"),
            ):
                axis = list(map(int, arm_record[axis_key]))
                base_diag = full_diagnostic(torch.as_tensor(base_record[full_key], dtype=torch.float64), labels, target)
                cbind = list(map(int, arm_record[cbind_key]))
                output_arms = {}
                for arm in ARMS:
                    real = [float(candidate_by_row[row][score_key][arm]) for row in axis]
                    control = [float(candidate_by_row[row][score_key][arm]) for row in cbind]
                    output_arms[arm] = {
                        "real": local_diagnostic(axis, real, labels, target, base_diag["strongest_wrong_identity"]),
                        "cbind": local_diagnostic(axis, control, labels, target, base_diag["strongest_wrong_identity"]),
                    }
                output_bases[base] = {"base": base_diag, "arms": output_arms}
            expected_cases.append({
                "execution_ordinal": execution,
                "query_id": arm_record["query_id"],
                "role": arm_record["role"],
                "track": arm_record["track"],
                "heldout_fold": int(arm_record["heldout_fold"]),
                "target_identity": target,
                "supergroup": str(role["supergroup"]),
                "bases": output_bases,
                "role_shard_sha256": role_entry["sha256"],
            })
        expected_shards.append({
            "shard": shard,
            "arm_payload_sha256": arm_seal["payload_sha256"],
            "arm_receipt_sha256": arm_seal["receipt_sha256"],
            "arm_validation_sha256": arm_seal["validation_sha256"],
            "base_payload_sha256": base_seal["payload_sha256"],
            "base_receipt_sha256": base_seal["receipt_sha256"],
            "base_validation_sha256": base_seal["validation_sha256"],
        })
    expected_cases.sort(key=lambda item: item["execution_ordinal"])
    by_role = {role: cohort([case for case in expected_cases if case["role"] == role]) for role in ("TRAIN", "EVAL")}
    by_track = {track: cohort([case for case in expected_cases if case["track"] == track]) for track in sorted({case["track"] for case in expected_cases})}
    by_fold = {str(fold): cohort([case for case in expected_cases if case["role"] == "EVAL" and case["heldout_fold"] == fold]) for fold in sorted({case["heldout_fold"] for case in expected_cases if case["role"] == "EVAL"})}
    comparisons = {role: comparison_cohort([case for case in expected_cases if case["role"] == role]) for role in ("TRAIN", "EVAL")}
    expected_bindings = {
        "contract_sha256": e0.sha256_file(CONTRACT),
        "arm_validation_sha256": e0.sha256_file(ARM_VALIDATION),
        "base_validation_sha256": e0.sha256_file(BASE_VALIDATION),
        "role_manifest_sha256": e0.sha256_file(ROLE_MANIFEST),
        "role_validation_sha256": e0.sha256_file(ROLE_VALIDATION),
        "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
        "shards": expected_shards,
    }
    checks = {
        "authorities": arm_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and arm_validation.get("logical_sha256") == e0.logical_sha256(arm_validation)
        and all(arm_validation.get("checks", {}).values())
        and arm_validation.get("next_authorized_stage")
        == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_DIAGNOSTIC"
        and base_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and base_validation.get("logical_sha256") == e0.logical_sha256(base_validation)
        and all(base_validation.get("checks", {}).values())
        and role_manifest.get("status") == "RGH_P0_A0_ROLE_MANIFEST_READY"
        and role_manifest.get("role_shard_count") == 600
        and role_manifest.get("logical_sha256") == e0.logical_sha256(role_manifest)
        and role_validation.get("status")
        == "RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS"
        and role_validation.get("role_receipt_count") == 600
        and role_validation.get("logical_sha256") == e0.logical_sha256(role_validation),
        "result_schema": set(result) == RESULT_KEYS
        and result.get("schema_version")
        == "routea_d1_current_runtime_matched_three_arm_postseal_v1_20260902"
        and result.get("claim_level")
        == "INTERNAL_LOCAL_EVIDENCE_DIAGNOSTIC_NO_MODEL_SELECTION",
        "producer_logical": result.get("logical_sha256") == e0.logical_sha256(result),
        "case_population": len(expected_cases) == 64 and len({item["query_id"] for item in expected_cases}) == 64,
        "cases_exact": result.get("cases") == expected_cases,
        "summary_all": result.get("summary_all") == cohort(expected_cases),
        "summary_by_role": result.get("summary_by_role") == by_role,
        "summary_by_track": result.get("summary_by_track") == by_track,
        "summary_by_fold": result.get("summary_eval_by_heldout_fold") == by_fold,
        "comparisons": result.get("comparison_by_role") == comparisons,
        "comparisons_all": result.get("comparison_all")
        == comparison_cohort(expected_cases),
        "bindings": result.get("bindings") == expected_bindings,
        "boundary": result.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_DIAGNOSTIC_COMPLETE"
        and result.get("primary_role") == "EVAL"
        and result.get("primary_base") == "RAW"
        and result.get("d1_checkpoint_scope") == "HISTORICAL_RUNTIME_CHECKPOINT_PORTABILITY_ONLY"
        and result.get("target_join_after_all_prejoin_shards") is True
        and result.get("target_role_read_count") == 64
        and result.get("model_update_count") == 0
        and result.get("sealed_read_count") == 0
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("next_authorized_stage") == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT",
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_postseal_independent_validation_v1_20260902",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_INDEPENDENT_VALIDATION_PASS" if passed else "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_INDEPENDENT_VALIDATION_ABORT",
        "checks": checks,
        "query_count": len(expected_cases),
        "producer_result_sha256": e0.sha256_file(RESULT),
        "target_role_read_count": 64,
        "model_update_count": 0,
        "next_authorized_stage": "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT" if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
