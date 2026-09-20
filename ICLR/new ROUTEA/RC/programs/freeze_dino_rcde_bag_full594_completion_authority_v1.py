#!/usr/bin/env python3
"""Freeze the non-superseding BAG-only full-594 completion authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RC_ROOT.parents[2]
OUTPUT_REL = "registry/dino_rcde_bag_full594_completion_authority_v1_20260827.json"
GALLERY_CACHE = (
    WORKSPACE / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
).resolve()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(relative: str) -> dict[str, Any]:
    path = (RC_ROOT / relative).resolve()
    if not path.is_file() or path.is_symlink() or RC_ROOT not in path.parents:
        raise RuntimeError(f"unsafe/missing authority binding: {relative}")
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RC_ROOT / OUTPUT_REL)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != (RC_ROOT / OUTPUT_REL).resolve() or output.exists():
        raise RuntimeError("authority output path exists or drifts")
    foldset_path = RC_ROOT / "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_manifest.json"
    foldset = json.loads(foldset_path.read_text(encoding="utf-8"))
    foldset_validation = json.loads(
        (RC_ROOT / "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_validation.json").read_text(encoding="utf-8")
    )
    prejoin_e0 = json.loads(
        (RC_ROOT / "results/dino_rcde_r1_oof_prejoin_e0_v4_0/aggregate_result.json").read_text(encoding="utf-8")
    )
    i0 = json.loads(
        (RC_ROOT / "results/dino_rcde_track_r_i0_science_metadata_v1/result.json").read_text(encoding="utf-8")
    )
    if not (
        foldset.get("status") == "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_FROZEN"
        and foldset_validation.get("status") == "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS"
        and foldset_validation.get("heldout_forward_count") == 0
        and prejoin_e0.get("unique_query_count") == 8
        and prejoin_e0.get("heldout_label_join") is False
        and prejoin_e0.get("scientific_GO_or_NO_GO") is None
        and i0.get("status") == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
        and i0.get("query_count") == 594
    ):
        raise RuntimeError("BAG completion predecessor disposition drift")
    checkpoints = {
        f"bag_checkpoint_fold{int(row['outer_fold'])}": bind(row["checkpoint"]["path"])
        for row in foldset["folds"]
    }
    relative_bindings = {
        "freezer": "programs/freeze_dino_rcde_bag_full594_completion_authority_v1.py",
        "contract": "plan/DINO_RCDE_BAG_FULL594_COMPLETION_CONTRACT_V1_20260827.md",
        "historical_scientific_contract": "plan/DINO_RCDE_LEARNED_DENSE_EVIDENCE_DECODER_V1_1_20260812.md",
        "authority_v22_historical_mainline_switch": "registry/current_authority_v22_20260814.json",
        "current_authority_v123": "registry/current_authority_v123_20260822.json",
        "bag_foldset_manifest": "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_manifest.json",
        "bag_foldset_validation": "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_validation.json",
        "eight_query_prejoin_aggregate": "results/dino_rcde_r1_oof_prejoin_e0_v4_0/aggregate_result.json",
        "redacted_schedule": "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json",
        "redacted_cache_index": "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_cache_index.json",
        "redacted_inputs_validation": "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_inputs_validation.json",
        "i0_science_metadata": "results/dino_rcde_track_r_i0_science_metadata_v1/result.json",
        "i0_science_metadata_validation": "results/dino_rcde_track_r_i0_science_metadata_validation_v1/result.json",
        "core": "src/rc_aslo_xf/dino_rcde_bag_completion_v1.py",
        "model_core": "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py",
        "gallery_identity_repair": "src/rc_aslo_xf/gallery_identity_repair.py",
        "io": "programs/dino_rcde_bag_completion_io_v1.py",
        "legacy_materializer": "programs/materialize_dino_rcde_r1_oof_prejoin_v3.py",
        "legacy_validator": "programs/validate_dino_rcde_r1_oof_prejoin_v1.py",
        "producer": "programs/materialize_dino_rcde_bag_full594_shard_v1.py",
        "shard_validator": "programs/validate_dino_rcde_bag_full594_shard_v1.py",
        "prejoin_finalizer": "programs/finalize_dino_rcde_bag_full594_prejoin_v1.py",
        "label_joiner": "programs/join_dino_rcde_bag_full594_labels_v1.py",
        "scientific_reducer": "programs/reduce_dino_rcde_bag_full594_science_v1.py",
        "independent_statistics": "programs/dino_rcde_bag_statistics_independent_v1.py",
        "scientific_validator": "programs/validate_dino_rcde_bag_full594_science_v1.py",
        "gpu_launcher": "slurm/dino_rcde_bag_full594_shards_v1.sbatch",
        "finalize_launcher": "slurm/dino_rcde_bag_full594_finalize_v1.sbatch",
        "tests": "tests/test_dino_rcde_bag_completion_v1.py",
    }
    bindings = {name: bind(path) for name, path in relative_bindings.items()}
    bindings.update(checkpoints)
    if not GALLERY_CACHE.is_file() or GALLERY_CACHE.is_symlink():
        raise RuntimeError("external gallery cache unavailable")
    bindings["gallery_cache"] = {
        "path": str(GALLERY_CACHE),
        "sha256": sha(GALLERY_CACHE),
        "bytes": GALLERY_CACHE.stat().st_size,
    }
    authority: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_bag_full594_completion_authority_v1_20260827",
        "date": "2026-08-27",
        "status": "DINO_RCDE_BAG_FULL594_COMPLETION_AUTHORIZED",
        "claim_level": "GIVEN_C128_FIXED_BAG_DECODER_NONSPATIAL_REPRESENTATION_QUALIFICATION_ONLY",
        "non_superseding_sidecar": True,
        "supersedes_current_mainline": False,
        "historical_disposition": "RCDE_BAG_TRAINING_COMPLETE_SCIENTIFIC_EVALUATION_UNFINISHED_NOT_NO_GO",
        "population": {
            "schedule_query_count": 600,
            "eligible_query_count": 594,
            "candidate_count_per_query": 128,
            "excluded_execution_ordinals": [25, 26, 101, 346, 354, 470],
            "outer_folds": [1, 2, 3, 4],
            "shard_size": 12,
            "shard_count": 50,
        },
        "controls": {
            "C_BIND_namespace": "RCDE_BAG_FULL594_C_BIND_V1_SEED17",
            "C_BIND_scope": "COMPLETE_C128_FIXED_POINT_FREE_CORRECTED_IDENTITY_DISJOINT_REFERENCE_REBIND",
            "candidate_reorder_namespace": "RCDE_BAG_FULL594_CANDIDATE_REORDER_V1_SEED17",
            "candidate_reorder_role": "INVARIANCE_GATE_NOT_C_BIND",
        },
        "statistics": {
            "independent_unit": "SUPERGROUP",
            "seed": 17,
            "permutations": 9999,
            "single_arm_family": "RCDE_BAG_ONLY_NO_CONTEXT_SELECTION",
            "gates_frozen_in_contract": True,
        },
        "allowed_decisions": [
            "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_NONSPATIAL_EVIDENCE_GO",
            "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_EVIDENCE_NO_GO",
            "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE",
        ],
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "training_authorized": False,
        "checkpoint_update_authorized": False,
        "target_free_control_materialization_authorized": True,
        "target_rival_join_before_prejoin_aggregate_authorized": False,
        "label_join_conditionally_authorized": True,
        "scientific_reduction_before_label_join_authorized": False,
        "scientific_reduction_after_label_join_authorized": True,
        "model_forward_after_label_join_authorized": False,
        "protected_access_authorized": False,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "resource_contract": {
            "producer_partition": "accelerated",
            "producer_gpu_count": 1,
            "producer_cpus_per_task": 8,
            "producer_memory_megabytes": 128000,
            "producer_walltime_seconds": 3600,
            "array": "0-49%8",
            "staged_submission": "SHARD0_CANARY_THEN_SHARDS1_49_PERCENT8",
            "finalizer_partition": "cpuonly",
            "finalizer_cpus_per_task": 8,
            "finalizer_memory_megabytes": 128000,
            "finalizer_walltime_seconds": 7200,
        },
        "runtime_paths": {
            "stable_cache_root": "runtime/dino_rcde_p0_v1_2/stable_fp16_cache",
            "result_root": "results/dino_rcde_bag_full594_completion_v1",
            "producer_root": "results/dino_rcde_bag_full594_completion_v1/prejoin",
            "validation_root": "results/dino_rcde_bag_full594_completion_v1/prejoin_validation",
        },
        "forbidden_claims": [
            "DEPLOYABLE_DISPERSED_MULTI_COMPONENT_MODEL",
            "SPATIAL_CORRESPONDENCE",
            "CONNECTED_TARGET_REGION",
            "FULL_GALLERY_RETRIEVAL_GAIN",
            "TARGET_ABSENCE",
            "HOLD_SWITCH",
            "OWNERSHIP",
            "OPENED_OR_SEALED_GENERALIZATION",
            "ALL_DISPERSED_PATCH_MECHANISMS_NO_GO",
        ],
        "bindings": bindings,
        "scientific_GO_or_NO_GO": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    output.chmod(0o444)
    print(json.dumps({"status": authority["status"], "output": str(output), "sha256": sha(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
