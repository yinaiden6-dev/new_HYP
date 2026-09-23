#!/usr/bin/env python3
"""Freeze V122 only after V121 target-free OOF prejoin completion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H
from freeze_dino_rcde_track_r_relative_v_fit_authority_v119 import (
    discover_runtime_closure,
)


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v122_20260821.json"
PARENT = "registry/current_authority_v121_20260821.json"
COMPLETION = "results/dino_rcde_track_r_oof_prejoin_completion_v120/result.json"
STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
PRODUCER_ROOT = "results/dino_rcde_track_r_p_v2_utility_shards_v1/producer"
VALIDATION_ROOT = "results/dino_rcde_track_r_p_v2_utility_shards_v1/validation"
COMPARATOR = "results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json"
COMPARATOR_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json"
REPAIR_ADDENDUM = "plan/DINO_RCDE_TRACK_R_V122_SINGLE_OUTER_SCORER_REPAIR_ADDENDUM_V1_20260823.md"
FAILED_AUTHORITY = "registry/archive/current_authority_v122_failed_job5102194_20260823.json"
FAILED_AUTHORITY_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_before_job5102194.json"
RESOURCE_ADDENDUM = "plan/DINO_RCDE_TRACK_R_V122_4H_SCHEDULER_RESOURCE_ADDENDUM_V1_20260823.md"
PRIOR_AUTHORITY = "registry/archive/current_authority_v122_revision2_before_4h_20260823.json"
PRIOR_AUTHORITY_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision2_before_4h.json"
FULL_SHARD0_CANARY_VALIDATION = "results/dino_rcde_track_r_p_v2_full_shard0_canary_v1/validation_shard_000_012.json"
RESOURCE_VALIDATION_REPAIR_ADDENDUM = "plan/DINO_RCDE_TRACK_R_V122_4H_INDEPENDENT_VALIDATION_REPAIR_ADDENDUM_V1_20260823.md"
RESOURCE_AUTHORITY_REVISION3 = "registry/archive/current_authority_v122_revision3_before_independent_4h_validation_20260823.json"
RESOURCE_AUTHORITY_REVISION3_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision3_before_independent_4h_validation.json"
RESOURCE_EVIDENCE_HARDENING_ADDENDUM = "plan/DINO_RCDE_TRACK_R_V122_4H_FULL_EVIDENCE_REPLAY_ADDENDUM_V1_20260823.md"
RESOURCE_AUTHORITY_REVISION4 = "registry/archive/current_authority_v122_revision4_before_full_resource_evidence_replay_20260823.json"
RESOURCE_AUTHORITY_REVISION4_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision4_before_full_resource_evidence_replay.json"
FULL_SHARD0_CANARY_SCHEDULER_RECEIPT = "results/dino_rcde_track_r_p_v2_full_shard0_canary_v1/scheduler_receipt_job5102372.json"
FULL_SHARD0_CANARY_STDOUT = "logs/rcde_tr_puf-5102372.out"
RUNTIME_ROOTS = (
    "programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    "programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py",
    "programs/dino_rcde_track_r_p_v2_utility_runtime_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_authority_v122.py",
    "src/rc_aslo_xf/__init__.py",
)


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def latest() -> tuple[int, Path]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [
        (int(match.group(1)), path.resolve())
        for path in (ROOT / "registry").glob("current_authority_v*_*.json")
        if (match := pattern.fullmatch(path.name))
        and path.is_file()
        and not path.is_symlink()
    ]
    version = max(item[0] for item in rows)
    paths = [path for item_version, path in rows if item_version == version]
    req(len(paths) == 1, "latest authority alias")
    return version, paths[0]


def build() -> dict[str, Any]:
    version, current = latest()
    parent_path = (ROOT / PARENT).resolve(strict=True)
    self_path = (ROOT / AUTH).resolve(strict=False)
    req(
        (version == 121 and current == parent_path)
        or (version == 122 and current == self_path and self_path.is_file()),
        "V121/V122 authority state drift",
    )
    parent = json.loads(parent_path.read_text())
    completion = json.loads((ROOT / COMPLETION).read_text())
    failed_authority_path = (ROOT / FAILED_AUTHORITY).resolve(strict=True)
    failed_validation_path = (ROOT / FAILED_AUTHORITY_VALIDATION).resolve(strict=True)
    failed_authority = json.loads(failed_authority_path.read_text())
    failed_validation = json.loads(failed_validation_path.read_text())
    prior_authority_path = (ROOT / PRIOR_AUTHORITY).resolve(strict=True)
    prior_validation_path = (ROOT / PRIOR_AUTHORITY_VALIDATION).resolve(strict=True)
    canary_validation_path = (ROOT / FULL_SHARD0_CANARY_VALIDATION).resolve(strict=True)
    prior_authority = json.loads(prior_authority_path.read_text())
    prior_validation = json.loads(prior_validation_path.read_text())
    canary_validation = json.loads(canary_validation_path.read_text())
    revision3_path = (ROOT / RESOURCE_AUTHORITY_REVISION3).resolve(strict=True)
    revision3_validation_path = (
        ROOT / RESOURCE_AUTHORITY_REVISION3_VALIDATION
    ).resolve(strict=True)
    revision3 = json.loads(revision3_path.read_text())
    revision3_validation = json.loads(revision3_validation_path.read_text())
    revision4_path = (ROOT / RESOURCE_AUTHORITY_REVISION4).resolve(strict=True)
    revision4_validation_path = (
        ROOT / RESOURCE_AUTHORITY_REVISION4_VALIDATION
    ).resolve(strict=True)
    scheduler_receipt_path = (
        ROOT / FULL_SHARD0_CANARY_SCHEDULER_RECEIPT
    ).resolve(strict=True)
    canary_stdout_path = (ROOT / FULL_SHARD0_CANARY_STDOUT).resolve(strict=True)
    revision4 = json.loads(revision4_path.read_text())
    revision4_validation = json.loads(revision4_validation_path.read_text())
    scheduler_receipt = json.loads(scheduler_receipt_path.read_text())
    req(
        parent.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
        and parent.get("scientific_GO_or_NO_GO") is None
        and completion.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_V120_COMPLETE"
        and completion.get("query_count") == 594
        and completion.get("excluded_query_count") == 6
        and completion.get("scientific_GO_or_NO_GO") is None,
        "V121 target-free OOF completion absent",
    )
    req(
        failed_authority.get("schema_version") == "rc_current_authority_v122_20260821"
        and failed_authority.get("status") == STATUS
        and failed_authority.get("logical_sha256") == H.logical(failed_authority)
        and failed_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and failed_validation.get("authority_sha256")
        == H.file_sha(failed_authority_path),
        "V122 failed-authority repair lineage drift",
    )
    req(
        prior_authority.get("status") == STATUS
        and prior_authority.get("authority_revision") == 2
        and prior_authority.get("logical_sha256") == H.logical(prior_authority)
        and prior_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and prior_validation.get("authority_sha256") == H.file_sha(prior_authority_path)
        and canary_validation.get("status")
        == "V122_FULL_SHARD0_CANARY_INDEPENDENT_REPLAY_PASS"
        and canary_validation.get("validation_pass") is True
        and canary_validation.get("authority_sha256") == H.file_sha(prior_authority_path)
        and canary_validation.get("query_count") == 12
        and canary_validation.get("producer_model_forward_count") == 3072
        and canary_validation.get("replay_model_forward_count") == 3072
        and canary_validation.get("formal_write_count") == 0
        and canary_validation.get("logical_sha256") == H.logical(canary_validation),
        "V122 four-hour resource evidence/lineage drift",
    )
    req(
        revision3.get("status") == STATUS
        and revision3.get("authority_revision") == 3
        and revision3.get("resource_contract", {}).get("walltime_seconds") == 14400
        and revision3.get("logical_sha256") == H.logical(revision3)
        and revision3_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and revision3_validation.get("authority_sha256") == H.file_sha(revision3_path),
        "V122 revision-3 validation-repair lineage drift",
    )
    req(
        revision4.get("status") == STATUS
        and revision4.get("authority_revision") == 4
        and revision4.get("resource_contract", {}).get("walltime_seconds") == 14400
        and revision4.get("logical_sha256") == H.logical(revision4)
        and revision4_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and revision4_validation.get("authority_sha256") == H.file_sha(revision4_path)
        and scheduler_receipt.get("status")
        == "V122_FULL_SHARD0_CANARY_SCHEDULER_RECEIPT_CLOSED"
        and scheduler_receipt.get("logical_sha256") == H.logical(scheduler_receipt)
        and scheduler_receipt.get("stdout_sha256") == H.file_sha(canary_stdout_path)
        and scheduler_receipt.get("validation_sha256")
        == H.file_sha(canary_validation_path),
        "V122 revision-4/scheduler receipt lineage drift",
    )
    parent_sha = H.file_sha(parent_path)
    oof_producer_root = ROOT / str(parent["producer_root"])
    oof_validation_root = ROOT / str(parent["validation_root"])
    req(len(completion.get("ordered_shard_receipts", [])) == 50, "V121 OOF completion receipt count drift")
    for ordinal, receipt in enumerate(completion["ordered_shard_receipts"]):
        start, stop = ordinal * 12, ordinal * 12 + 12
        producer_path = oof_producer_root / f"shard_{start:03d}_{stop:03d}.pt"
        validation_path = oof_validation_root / f"shard_{start:03d}_{stop:03d}.validation.json"
        shard = torch.load(producer_path, map_location="cpu", weights_only=True, mmap=True)
        validation = json.loads(validation_path.read_text())
        req(
            receipt.get("shard_ordinal") == ordinal
            and receipt.get("producer_sha256") == H.file_sha(producer_path)
            and receipt.get("validation_sha256") == H.file_sha(validation_path)
            and shard.get("source_bindings", {}).get("authority_sha256") == parent_sha
            and validation.get("producer_shard_sha256") == receipt["producer_sha256"],
            f"V121 OOF completion/authority lineage drift: {ordinal}",
        )
    manifest_index = json.loads(
        (ROOT / "results/dino_rcde_sr0_mt_p_v2_lock_execution_manifests_v1/index.json").read_text()
    )
    manifest_validation = json.loads(
        (ROOT / "results/dino_rcde_sr0_mt_p_v2_lock_execution_manifest_validation_v1/result.json").read_text()
    )
    req(
        manifest_index.get("status") == "RCDE_SR0_MT_P_V2_LOCK_EXECUTION_MANIFEST_INDEX_READY"
        and manifest_index.get("shard_count") == 50
        and manifest_validation.get("status")
        == "RCDE_SR0_MT_P_V2_LOCK_EXECUTION_MANIFESTS_INDEPENDENT_VALIDATION_PASS"
        and manifest_validation.get("validation_pass") is True,
        "P-V2 source manifest qualification absent",
    )
    closure_paths, closure_edges = discover_runtime_closure(RUNTIME_ROOTS)
    closure_rows = [
        {"module": relative.removesuffix(".py").replace("/", "."), **H.bind(relative)}
        for relative in closure_paths
    ]
    source_rows = []
    for index_row in manifest_index["shards"]:
        manifest_relative = str(index_row["path"])
        manifest = json.loads((ROOT / manifest_relative).read_text())
        source_path = (ROOT / str(manifest["source_artifact_path"])).resolve(strict=True)
        req(
            source_path.is_file()
            and not source_path.is_symlink()
            and (source_path.stat().st_mode & 0o777) == 0o444,
            "immutable compact source artifact drift",
        )
        source_rows.append(
            {
                "shard_ordinal": int(index_row["shard_ordinal"]),
                "execution_manifest": H.bind(
                    manifest_relative, with_logical=True, immutable=True
                ),
                "source_descriptor": H.bind(
                    str(manifest["source_manifest_path"]),
                    with_logical=True,
                    immutable=True,
                ),
                "source_artifact": {
                    "path": str(manifest["source_artifact_path"]),
                    "sha256": str(manifest["source_artifact_sha256"]),
                    "bytes": source_path.stat().st_size,
                },
            }
        )
    bindings: dict[str, Any] = {
        "parent_authority_v121": H.bind(PARENT, with_logical=True, immutable=True),
        "oof_prejoin_completion": H.bind(COMPLETION, with_logical=True, immutable=True),
        "lock_manifest_index": H.bind("results/dino_rcde_sr0_mt_p_v2_lock_execution_manifests_v1/index.json", with_logical=True, immutable=True),
        "lock_manifest_validation": H.bind("results/dino_rcde_sr0_mt_p_v2_lock_execution_manifest_validation_v1/result.json", with_logical=True, immutable=True),
        "fold_schedule": H.bind("protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json", with_logical=True),
        "full600_payload": H.bind("cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"),
        "contract": H.bind("plan/DINO_RCDE_TRACK_R_V122_P_V2_UTILITY_CONTRACT_V1_20260822.md"),
        "repair_addendum": H.bind(REPAIR_ADDENDUM),
        "failed_authority": H.bind(FAILED_AUTHORITY, with_logical=True, immutable=True),
        "failed_authority_validation": H.bind(
            FAILED_AUTHORITY_VALIDATION, with_logical=True, immutable=True
        ),
        "resource_addendum": H.bind(RESOURCE_ADDENDUM),
        "resource_parent_authority": H.bind(
            PRIOR_AUTHORITY, with_logical=True, immutable=True
        ),
        "resource_parent_authority_validation": H.bind(
            PRIOR_AUTHORITY_VALIDATION, with_logical=True, immutable=True
        ),
        "full_shard0_canary_validation": H.bind(
            FULL_SHARD0_CANARY_VALIDATION, with_logical=True, immutable=True
        ),
        "resource_validation_repair_addendum": H.bind(
            RESOURCE_VALIDATION_REPAIR_ADDENDUM
        ),
        "resource_authority_revision3": H.bind(
            RESOURCE_AUTHORITY_REVISION3, with_logical=True, immutable=True
        ),
        "resource_authority_revision3_validation": H.bind(
            RESOURCE_AUTHORITY_REVISION3_VALIDATION,
            with_logical=True,
            immutable=True,
        ),
        "resource_evidence_hardening_addendum": H.bind(
            RESOURCE_EVIDENCE_HARDENING_ADDENDUM
        ),
        "resource_authority_revision4": H.bind(
            RESOURCE_AUTHORITY_REVISION4, with_logical=True, immutable=True
        ),
        "resource_authority_revision4_validation": H.bind(
            RESOURCE_AUTHORITY_REVISION4_VALIDATION,
            with_logical=True,
            immutable=True,
        ),
        "full_shard0_canary_scheduler_receipt": H.bind(
            FULL_SHARD0_CANARY_SCHEDULER_RECEIPT,
            with_logical=True,
            immutable=True,
        ),
        "full_shard0_canary_stdout": H.bind(FULL_SHARD0_CANARY_STDOUT),
        "runtime_helper": H.bind("programs/dino_rcde_track_r_p_v2_utility_runtime_v1.py"),
        "producer": H.bind(RUNTIME_ROOTS[0]),
        "validator": H.bind(RUNTIME_ROOTS[1]),
        "reducer": H.bind(RUNTIME_ROOTS[2]),
        "aggregate_validator": H.bind(RUNTIME_ROOTS[3]),
        "authority_validator": H.bind("programs/validate_dino_rcde_track_r_p_v2_utility_authority_v122.py"),
        "package_initializer": H.bind(RUNTIME_ROOTS[6]),
        "runtime_import_closure": {
            "root_programs": list(RUNTIME_ROOTS),
            "module_count": len(closure_rows),
            "modules": closure_rows,
            "module_sequence_sha256": H.logical({"rows": closure_rows}),
            "import_edges": closure_edges,
            "import_edge_sequence_sha256": H.logical({"rows": closure_edges}),
        },
        "source_shard_manifests": {
            "count": 50,
            "rows": source_rows,
            "sequence_sha256": H.logical({"rows": source_rows}),
        },
        "launcher": H.bind("slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch"),
        "post_oof_controller_v2": H.bind("slurm/dino_rcde_track_r_post_oof_controller_v2.sbatch"),
        "post_shard0_controller": H.bind("slurm/dino_rcde_track_r_post_p_utility_shard0_controller_v1.sbatch"),
        "post_p_controller": H.bind("slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch"),
        "science_freezer": H.bind("programs/freeze_dino_rcde_track_r_science_authority_v123.py"),
        "science_launcher": H.bind("slurm/dino_rcde_track_r_science_v1.sbatch"),
        "test": H.bind("tests/test_dino_rcde_track_r_v122_p_utility_chain_v1.py"),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_p_v2_utility_authority_v121.py"),
    }
    for fold in (1, 2, 3, 4):
        root = f"results/dino_rcde_sr0_mt_p_v2_formal_fits_v1/P_OUTER{fold}_OUTER_REFIT"
        bindings[f"p_outer_refit_checkpoint_fold{fold}"] = H.bind(
            f"{root}/primary/checkpoint_update2048.pt", immutable=True
        )
        bindings[f"p_outer_refit_manifest_fold{fold}"] = H.bind(
            f"results/dino_rcde_sr0_mt_p_v2_training_manifests_v1/P_OUTER{fold}_OUTER_REFIT/fit_manifest.json",
            with_logical=True,
            immutable=True,
        )
    authority: dict[str, Any] = {
        "schema_version": "rc_current_authority_v122_20260821",
        "status": STATUS,
        "authority_revision": 5,
        "repair_parent_authority_sha256": H.file_sha(failed_authority_path),
        "failed_job_id": 5102194,
        "repair_scope": "EXECUTION_INTERFACE_ONLY_SINGLE_OUTER_HEAD",
        "resource_contract_revision": 2,
        "resource_parent_authority_sha256": H.file_sha(prior_authority_path),
        "resource_repair_scope": "SCHEDULER_TIME_LIMIT_ONLY_4H",
        "resource_validation_repair_scope": "INDEPENDENT_RESOURCE_EVIDENCE_REPLAY",
        "resource_validation_parent_authority_sha256": H.file_sha(revision3_path),
        "resource_evidence_hardening_scope": "FULL_PATH_AND_SCHEDULER_RECEIPT_REPLAY",
        "resource_evidence_parent_authority_sha256": H.file_sha(revision4_path),
        "full_shard0_canary_job_id": 5102372,
        "stage": "TRACK_R_TARGET_FREE_P_V2_UTILITY_COMPARATOR",
        "claim_level": "TARGET_FREE_P_V2_OFFLINE_COMPARATOR_ONLY",
        "query_count": 600,
        "eligible_query_count": 594,
        "excluded_execution_ordinals": [25, 26, 101, 346, 354, 470],
        "shard_count": 50,
        "shard_size": 12,
        "p_v2_utility_materialization_authorized": True,
        "p_v2_utility_validation_authorized": True,
        "p_v2_utility_reduction_authorized": True,
        "p_v2_utility_aggregate_validation_authorized": True,
        "p_model_load_forward_authorized": True,
        "colnomic_token_read_authorized": True,
        "target_rival_read_authorized": False,
        "label_identity_supergroup_read_authorized": False,
        "dino_token_read_authorized": False,
        "p_model_backward_update_authorized": False,
        "scientific_reduction_authorized": False,
        "protected_access_authorized": False,
        "scheduler_continuation_authorized": True,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "producer_root": PRODUCER_ROOT,
        "validation_root": VALIDATION_ROOT,
        "comparator_output": COMPARATOR,
        "comparator_validation_output": COMPARATOR_VALIDATION,
        "authority_validation_output": "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/result.json",
        "resource_contract": {
            "partition": "cpuonly",
            "cpus_per_task": 4,
            "memory_megabytes": 65536,
            "walltime_seconds": 14400,
            "launcher_header_walltime_seconds": 21600,
            "scheduler_time_limit_override_required": True,
            "scheduler_override_job_ids": [5102240, 5102486],
            "staged_arrays": ["0", "1-49%8"],
            "reducer_cpus": 2,
            "reducer_memory_megabytes": 16384,
            "reducer_walltime_seconds": 1800,
        },
        "bindings": bindings,
    }
    authority["logical_sha256"] = H.logical(authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False), "output drift")
    H.atomic(args.output, build())


if __name__ == "__main__":
    main()
