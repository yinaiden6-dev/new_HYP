#!/usr/bin/env python3
"""Freeze the bounded four-fold GX-CBNR R0 scientific-reduction authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H
import freeze_dino_rcde_track_r_relative_v_fit_authority_v119 as CLOSURE


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path("registry/h0/gx_cbnr_formal_r0_science_authority_v68_20260825.json")
PARENT = Path("registry/h0/gx_cbnr_formal_training_authority_v54_20260824.json")
CACHE_AUTHORITY = Path("registry/h0/gx_cbnr_relational_cache_authority_v52_20260824.json")
CACHE_AGGREGATE_AUTHORITY = Path("registry/h0/gx_cbnr_relational_cache_aggregate_authority_v53_20260824.json")
MANIFEST_AUTHORITY = Path("registry/h0/gx_cbnr_r0_manifest_v2_authority_v50_20260824.json")
ADDRESS = Path("results/dino_rcde_gx_cbnr_r0_manifest_v2/address_manifest.json")
ROLES = Path("results/dino_rcde_gx_cbnr_r0_manifest_v2/loss_role_manifest.json")
MANIFEST_VALIDATION = Path("results/dino_rcde_gx_cbnr_r0_manifest_validation_v2/result.json")
CACHE_INDEX = Path("results/dino_rcde_gx_relational_cache_aggregate_v1/cache_index.json")
CACHE_VALIDATION = Path("results/dino_rcde_gx_relational_cache_aggregate_v1/validation.json")
CONTRACT = Path("plan/DINO_RCDE_GX_CBNR_R0_CONTRACT_V1_20260824.md")
REPAIR_ADDENDUM = Path("plan/DINO_RCDE_GX_CBNR_R0_MECHANISM_REPAIR_ADDENDUM_V2_20260824.md")
REDUCER = Path("programs/reduce_dino_rcde_gx_formal_r0_science_v1.py")
VALIDATOR = Path("programs/validate_dino_rcde_gx_formal_r0_science_v1.py")
FREEZER = Path("programs/freeze_dino_rcde_gx_formal_r0_science_authority_v68.py")
LAUNCHER = Path("slurm/dino_rcde_gx_formal_r0_science_v68.sbatch")
TEST_REDUCER = Path("tests/test_reduce_dino_rcde_gx_formal_r0_science_v1.py")
TEST_VALIDATOR = Path("tests/test_validate_dino_rcde_gx_formal_r0_science_v1.py")
TEST_AUTHORITY = Path("tests/test_freeze_dino_rcde_gx_formal_r0_science_authority_v68.py")
STATUS = "GX_CBNR_R0_SCIENTIFIC_REDUCTION_EXECUTION_AUTHORIZED"
SCIENCE_RESULT = "results/dino_rcde_gx_formal_r0_science_v1/result.json"
SCIENCE_VALIDATION = "results/dino_rcde_gx_formal_r0_science_validation_v1/result.json"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: str | Path) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def build() -> dict[str, Any]:
    parent = read(PARENT)
    cache_authority = read(CACHE_AUTHORITY)
    cache_aggregate_authority = read(CACHE_AGGREGATE_AUTHORITY)
    manifest_authority = read(MANIFEST_AUTHORITY)
    address = read(ADDRESS)
    roles = read(ROLES)
    manifest_validation = read(MANIFEST_VALIDATION)
    cache_index = read(CACHE_INDEX)
    cache_validation = read(CACHE_VALIDATION)
    require(
        parent.get("status") == "GX_CBNR_FORMAL_FOURFOLD_TRAINING_EXECUTION_AUTHORIZED"
        and parent.get("logical_sha256") == H.logical(parent)
        and parent.get("formal_r0_authorized") is True
        and parent.get("scientific_reduction_authorized") is False
        and cache_authority.get("logical_sha256") == H.logical(cache_authority)
        and cache_aggregate_authority.get("logical_sha256") == H.logical(cache_aggregate_authority)
        and manifest_authority.get("logical_sha256") == H.logical(manifest_authority)
        and address.get("status") == "GX_CBNR_R0_ELIGIBILITY_AWARE_MANIFEST_READY"
        and roles.get("status") == "GX_CBNR_R0_ELIGIBILITY_AWARE_MANIFEST_READY"
        and address.get("logical_sha256") == H.logical(address)
        and roles.get("logical_sha256") == H.logical(roles)
        and manifest_validation.get("validation_pass") is True
        and cache_index.get("status") == "GX_CBNR_RELATIONAL_CACHE_INDEX_READY"
        and cache_index.get("context_count") == 224
        and cache_index.get("all_shard_validations_pass") is True
        and cache_index.get("all_direct_cache_init_parity_pass") is True
        and cache_index.get("logical_sha256") == H.logical(cache_index)
        and cache_validation.get("validation_pass") is True,
        "formal science prerequisite drift",
    )
    bindings: dict[str, Any] = {
        "parent_training_authority_v54": H.bind(str(PARENT), immutable=True),
        "cache_authority_v52": H.bind(str(CACHE_AUTHORITY), immutable=True),
        "cache_aggregate_authority_v53": H.bind(str(CACHE_AGGREGATE_AUTHORITY), immutable=True),
        "manifest_authority_v50": H.bind(str(MANIFEST_AUTHORITY), immutable=True),
        "address_manifest": H.bind(str(ADDRESS), immutable=True),
        "loss_role_manifest": H.bind(str(ROLES), immutable=True),
        "manifest_validation": H.bind(str(MANIFEST_VALIDATION), immutable=True),
        "cache_index": H.bind(str(CACHE_INDEX), immutable=True),
        "cache_index_validation": H.bind(str(CACHE_VALIDATION), immutable=True),
        "contract": H.bind(str(CONTRACT)),
        "mechanism_repair_addendum": H.bind(str(REPAIR_ADDENDUM)),
        "reducer": H.bind(str(REDUCER)),
        "independent_validator": H.bind(str(VALIDATOR)),
        "freezer": H.bind(str(FREEZER)),
        "launcher": H.bind(str(LAUNCHER)),
        "test_reducer": H.bind(str(TEST_REDUCER)),
        "test_validator": H.bind(str(TEST_VALIDATOR)),
        "test_authority": H.bind(str(TEST_AUTHORITY)),
    }
    for fold in (1, 2, 3, 4):
        outputs = parent["fold_outputs"][str(fold)]
        result_path = Path(outputs["result"])
        validation_path = Path(outputs["validation"])
        checkpoint_path = Path(outputs["checkpoint"])
        trace_path = Path(outputs["trace"])
        result = read(result_path)
        validation = read(validation_path)
        require(
            result.get("status") == "GX_CBNR_FORMAL_FOLD_TRAIN_COMPLETE"
            and result.get("completed_updates") == 128
            and result.get("outer_fold") == fold
            and result.get("evaluation_record_count") == 8
            and result.get("logical_sha256") == H.logical(result)
            and validation.get("status") == "GX_CBNR_FORMAL_FOLD_TRAIN_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and validation.get("outer_fold") == fold
            and validation.get("fresh_resume_byte_parity") is True
            and result.get("checkpoint_sha256") == H.file_sha(ROOT / checkpoint_path),
            f"fold {fold} validated training output drift",
        )
        bindings[f"fold{fold}_result"] = H.bind(str(result_path), immutable=True)
        bindings[f"fold{fold}_validation"] = H.bind(str(validation_path), immutable=True)
        bindings[f"fold{fold}_checkpoint"] = H.bind(str(checkpoint_path), immutable=True)
        bindings[f"fold{fold}_trace"] = H.bind(str(trace_path), immutable=True)
        bindings[f"fold{fold}_init_checkpoint"] = cache_authority["bindings"][f"init_checkpoint_fold{fold}"]
    paths, edges = CLOSURE.discover_runtime_closure((str(REDUCER), str(VALIDATOR)))
    rows = [H.bind(path) for path in paths]
    bindings["runtime_import_closure"] = {
        "count": len(rows),
        "rows": rows,
        "logical_sha256": H.logical({"rows": rows}),
        "edge_count": len(edges),
        "edges_sha256": H.logical({"edges": edges}),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_gx_formal_r0_science_authority_v68_20260825",
        "status": STATUS,
        "claim_level": "INTERNAL_BALANCED32_MATCHED_NULL_CALIBRATED_CONNECTED_DINO_RELATIONAL_ENERGY_SCREEN",
        "formal_r0_authorized": True,
        "scientific_reduction_authorized": True,
        "scientific_validation_authorized": True,
        "postjoin_cached_head_replay_authorized": True,
        "postjoin_DINO_forward_authorized": False,
        "model_backward_or_update_authorized": False,
        "opened_or_sealed_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "training_root": "results/dino_rcde_gx_formal_training_v1",
        "training_validation_root": "results/dino_rcde_gx_formal_training_validation_v1",
        "output": SCIENCE_RESULT,
        "validation_output": SCIENCE_VALIDATION,
        "science_result_output": SCIENCE_RESULT,
        "science_validation_output": SCIENCE_VALIDATION,
        "science_contract": {
            "folds": [1, 2, 3, 4],
            "evaluation_records_per_fold": 8,
            "query_count": 32,
            "raw_correct_count": 16,
            "raw_wrong_count": 16,
            "mandatory_nulls": ["C_BIND", "P_COORD", "N_REGION_RESAMPLE"],
            "T_ROOT_ASSIGNMENT_POPULATION_role": "AUXILIARY_NON_GATING_DIAGNOSTIC_ONLY",
            "bootstrap_cluster_key": "group_sha256",
            "bootstrap_rng": "numpy.PCG64",
            "bootstrap_seed": 17,
            "bootstrap_repetitions": 10000,
            "bootstrap_quantile_method": "linear",
            "bootstrap_endpoints": [
                "correctness_increment",
                "target_z_minus_rival_z",
                "target_real_minus_C_BIND",
                "target_real_minus_P_COORD",
                "target_real_minus_N_REGION_RESAMPLE",
            ],
        },
        "resource_contract": {
            "partition": "accelerated",
            "gpus_per_task": 1,
            "GPU_compute_authorized": False,
            "CUDA_must_be_hidden_after_allocation_check": True,
            "cpus_per_task": 16,
            "memory_megabytes": 64000,
            "walltime_seconds": 7200,
        },
        "bindings": bindings,
    }
    value["logical_sha256"] = H.logical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTHORITY)
    args = parser.parse_args()
    require(
        args.output.resolve(strict=False) == (ROOT / AUTHORITY).resolve(strict=False),
        "V68 output drift",
    )
    value = build()
    if args.output.exists():
        require(read(AUTHORITY) == value, "existing V68 authority drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
