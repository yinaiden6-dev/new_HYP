#!/usr/bin/env python3
"""Stdlib-only validator for the current64 rematerialization authority."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_STAGING_EXECUTION_AUTHORITY_V1_20260904.json"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_STAGING_EXECUTION_AUTHORIZED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SETUP_EXECUTION"

INPUT_PATHS = {
    "rematerialization_contract": "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md",
    "exact_map_cache_reuse_addendum": "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_EXACT_ROMA_MAP_CACHE_REUSE_ADDENDUM_V1_20260904.md",
    "j0_result": "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1/result.json",
    "j0_independent_validation": "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1/independent_validation.json",
    "v3_oof_execution_authority": "plan/ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_SCORING_EXECUTION_AUTHORITY_V3_20260903.json",
    "fresh_d1_oof_aggregate": "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json",
}
PROGRAM_PATHS = {
    "source_manifest_producer": "programs/materialize_routea_n2_fresh_d1_matched_three_arm_source_manifest_v1.py",
    "source_manifest_validator": "programs/validate_routea_n2_fresh_d1_matched_three_arm_source_manifest_v1.py",
    "reference_cache_producer": "programs/materialize_routea_n2_fresh_d1_matched_three_arm_reference_cache_v1.py",
    "reference_cache_validator": "programs/validate_routea_n2_fresh_d1_matched_three_arm_reference_cache_v1.py",
    "current64_prejoin_shard_producer": "programs/materialize_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py",
    "current64_prejoin_shard_validator": "programs/validate_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py",
    "current64_prejoin_aggregate_validator": "programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_prejoin_aggregate_v1.py",
}
LAUNCHER_PATHS = {
    "setup": "slurm/routea_n2_fresh_d1_matched_three_arm_current64_setup_v1_30m.sbatch",
    "prejoin_array": "slurm/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_array_v1_1h.sbatch",
    "aggregate": "slurm/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_aggregate_v1_30m.sbatch",
}
OUTPUT_PATHS = {
    "source_manifest": "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json",
    "source_manifest_validation": "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/independent_validation.json",
    "reference_cache_payload": "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1/payload.pt",
    "reference_cache_result": "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1/result.json",
    "reference_cache_validation": "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1/independent_validation.json",
    "current64_prejoin_shard_pattern": "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/shard{shard:02d}/{payload.pt,receipt.json,validation.json}",
    "current64_prejoin_aggregate_validation": "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json",
}
LAUNCHER_STAGE_PROGRAM = {
    "setup": PROGRAM_PATHS["source_manifest_producer"],
    "prejoin_array": PROGRAM_PATHS["current64_prejoin_shard_producer"],
    "aggregate": PROGRAM_PATHS["current64_prejoin_aggregate_validator"],
}


class AuthorityValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"bound file absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def verify_binding(binding: Any, relative: str, *, logical: bool) -> bool:
    if not isinstance(binding, Mapping) or set(binding) != (
        {"path", "sha256", "logical_sha256"} if logical else {"path", "sha256"}
    ):
        return False
    path = ROOT / relative
    valid = binding.get("path") == relative and binding.get("sha256") == sha256_file(path)
    if logical:
        value = read_json(path)
        valid = valid and value.get("logical_sha256") == logical_sha256(value)
        valid = valid and binding.get("logical_sha256") == value.get("logical_sha256")
    return bool(valid)


def main() -> None:
    authority = read_json(AUTHORITY)
    expected_keys = {
        "version",
        "status",
        "claim_level",
        "authorized_scope",
        "population",
        "stage_topology",
        "inputs",
        "programs",
        "authority_validator",
        "launchers",
        "outputs",
        "resource_contract",
        "hardware_identity_is_authority",
        "access_boundary",
        "scientific_GO_or_NO_GO",
        "ownership_GO_or_NO_GO",
        "automatic_stage_advance",
        "next_authorized_stage",
        "logical_sha256",
    }
    require(set(authority) == expected_keys, "authority top-level schema drift")
    require(
        authority.get("status") == STATUS
        and authority.get("claim_level")
        == "ENGINEERING_TARGET_FREE_CURRENT64_STAGING_EXECUTION_ONLY_PAIR64_PENDING"
        and authority.get("logical_sha256") == logical_sha256(authority),
        "authority envelope or logical hash drift",
    )
    scope = authority.get("authorized_scope", {})
    require(
        scope
        == {
            "source_manifest_materialization": True,
            "source_manifest_independent_validation": True,
            "current64_reference_cache_materialization": True,
            "current64_reference_cache_independent_validation": True,
            "current64_prejoin_shard_materialization": True,
            "current64_prejoin_shard_independent_validation": True,
            "current64_prejoin_aggregate_validation": True,
            "pair64_prejoin": False,
            "postseal_label_join": False,
            "head_training": False,
        },
        "authorized scope drift",
    )
    require(
        authority.get("population")
        == {
            "current64_query_count": 64,
            "pair64_pending_query_count": 64,
            "contract_population_complete": False,
            "candidate_count_per_query": 128,
            "current64_candidate_pair_count": 8192,
            "old_map_exact_reuse_pair_count": 8168,
            "frozen_roma_completion_pair_count": 24,
            "reference_union_count": 2805,
            "old_reference_exact_reuse_count": 2801,
            "reference_completion_rows": [467, 1488, 2435, 5159],
        },
        "authority population drift",
    )
    require(
        authority.get("stage_topology")
        == [
            "CURRENT64_SOURCE_MANIFEST_AND_REFERENCE_CACHE_SETUP",
            "CURRENT64_PREJOIN_SHARDS_0_TO_7",
            "CURRENT64_PREJOIN_AGGREGATE_VALIDATION",
        ],
        "stage topology drift",
    )

    inputs = authority.get("inputs", {})
    require(set(inputs) == set(INPUT_PATHS), "authority input set drift")
    logical_inputs = {
        "j0_result",
        "j0_independent_validation",
        "v3_oof_execution_authority",
        "fresh_d1_oof_aggregate",
    }
    require(
        all(
            verify_binding(inputs[key], relative, logical=key in logical_inputs)
            for key, relative in INPUT_PATHS.items()
        ),
        "authority input binding drift",
    )
    require(
        inputs["rematerialization_contract"]["sha256"]
        == "6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3"
        and inputs["exact_map_cache_reuse_addendum"]["sha256"]
        == "4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6",
        "contract/addendum immutable hash drift",
    )
    j0 = read_json(ROOT / INPUT_PATHS["j0_result"])
    j0_validation = read_json(ROOT / INPUT_PATHS["j0_independent_validation"])
    v3 = read_json(ROOT / INPUT_PATHS["v3_oof_execution_authority"])
    oof = read_json(ROOT / INPUT_PATHS["fresh_d1_oof_aggregate"])
    require(
        j0.get("status") == "N2_CANDIDATE_RECALL_NO_GO"
        and j0_validation.get("status")
        == "N2_CANDIDATE_RECALL_NO_GO_INDEPENDENTLY_VALIDATED"
        and j0_validation.get("result_sha256")
        == sha256_file(ROOT / INPUT_PATHS["j0_result"])
        and j0.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT"
        and j0_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT",
        "J0 NO-GO claim boundary drift",
    )
    require(
        v3.get("status") == "FROZEN_BEFORE_N2_FRESH_D1_OOF_SCORING"
        and oof.get("status")
        == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED"
        and all(oof.get("checks", {}).values())
        and oof.get("bindings", {}).get("execution_authority_sha256")
        == sha256_file(ROOT / INPUT_PATHS["v3_oof_execution_authority"])
        and oof.get("bindings", {}).get("execution_authority_logical_sha256")
        == v3.get("logical_sha256"),
        "V3/OOF authority chain drift",
    )

    programs = authority.get("programs", {})
    require(set(programs) == set(PROGRAM_PATHS), "authority program set drift")
    require(
        all(
            verify_binding(programs[key], relative, logical=False)
            for key, relative in PROGRAM_PATHS.items()
        ),
        "program hash binding drift",
    )
    require(
        verify_binding(
            authority.get("authority_validator"),
            "programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_staging_execution_authority_v1.py",
            logical=False,
        ),
        "authority validator self-binding drift",
    )
    launchers = authority.get("launchers", {})
    require(set(launchers) == set(LAUNCHER_PATHS), "authority launcher set drift")
    require(
        all(
            verify_binding(launchers[key], relative, logical=False)
            for key, relative in LAUNCHER_PATHS.items()
        ),
        "launcher hash binding drift",
    )
    authority_validator_name = Path(__file__).name
    for key, relative in LAUNCHER_PATHS.items():
        text = (ROOT / relative).read_text()
        validator_offset = text.find(authority_validator_name)
        stage_offset = text.find(LAUNCHER_STAGE_PROGRAM[key])
        require(
            "#SBATCH --export=NIL" in text
            and validator_offset >= 0
            and stage_offset >= 0
            and validator_offset < stage_offset,
            f"launcher {key} does not validate authority before stage execution",
        )

    require(authority.get("outputs") == OUTPUT_PATHS, "authority outputs drift")
    require(
        authority.get("resource_contract")
        == {
            "setup": {
                "default_partition": "cpuonly",
                "allowed_partitions": ["cpuonly", "dev_cpuonly"],
                "cpus_per_task": 8,
                "memory_gib": 64,
                "walltime_seconds": 1800,
            },
            "prejoin_array": {
                "default_partition": "accelerated",
                "allowed_partitions": ["accelerated", "dev_accelerated"],
                "canonical_array_tasks": "0-7",
                "default_array": "0-7%8",
                "dev_max_concurrency": 4,
                "gpu_count": 1,
                "cpus_per_task": 8,
                "memory_gib": 96,
                "walltime_seconds": 3600,
            },
            "aggregate": {
                "default_partition": "cpuonly",
                "allowed_partitions": ["cpuonly", "dev_cpuonly"],
                "cpus_per_task": 4,
                "memory_gib": 64,
                "walltime_seconds": 1800,
            },
        }
        and authority.get("hardware_identity_is_authority") is False,
        "resource or hardware-identity boundary drift",
    )
    if os.environ.get("SLURM_JOB_ID"):
        stage = os.environ.get("ROUTEA_CURRENT64_AUTHORITY_STAGE")
        require(stage in authority["resource_contract"], "runtime stage is not declared")
        require(
            os.environ.get("SLURM_JOB_PARTITION")
            in authority["resource_contract"][stage]["allowed_partitions"],
            "runtime partition is outside execution-only alternatives",
        )
        if stage == "prejoin_array":
            require(
                os.environ.get("SLURM_ARRAY_TASK_MIN") == "0"
                and os.environ.get("SLURM_ARRAY_TASK_MAX") == "7"
                and os.environ.get("SLURM_ARRAY_TASK_COUNT") == "8",
                "runtime array does not cover canonical tasks 0-7",
            )
    require(
        authority.get("access_boundary")
        == {
            "target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "retrieval_outcome_read_count": 0,
            "action_read_count": 0,
            "target_insertion_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        }
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("ownership_GO_or_NO_GO") is None
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") == NEXT,
        "authority claim/access boundary drift",
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "authority_sha256": sha256_file(AUTHORITY),
                "authority_logical_sha256": authority["logical_sha256"],
                "hardware_identity_is_authority": False,
                "current64_query_count": 64,
                "pair64_pending_query_count": 64,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
