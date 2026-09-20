#!/usr/bin/env python3
"""Freeze V123 only after V121 evidence and the V122 comparator validate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H
from freeze_dino_rcde_track_r_oof_prejoin_authority_v120 import (
    discover_runtime_import_closure,
)
from seal_dino_rcde_track_r_i0_science_metadata_v1 import build_sealed_metadata


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v123_20260822.json"
PARENT = "registry/current_authority_v122_20260821.json"
PARENT_SCHEMA = "rc_current_authority_v122_20260821"
PARENT_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
V121_AUTHORITY = "registry/current_authority_v121_20260821.json"
PREJOIN_ROOT = "results/dino_rcde_track_r_oof_prejoin_v1/producer"
PREJOIN_VALIDATION_ROOT = "results/dino_rcde_track_r_oof_prejoin_v1/validation"
PREJOIN_COMPLETION = "results/dino_rcde_track_r_oof_prejoin_completion_v120/result.json"
P_COMPARATOR = "results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json"
P_VALIDATION = "results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json"
LOSS_JOIN = "results/dino_rcde_sr0_mt_loss_join_v1/loss_join.json"
LOSS_VALIDATION = "results/dino_rcde_sr0_mt_loss_join_validation_v1/result.json"
I0_POSTJOIN = "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/postjoin_ledger.json"
I0_VALIDATION = "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/postjoin_validation.json"
ROLE_FREE = "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json"
FOLD_SCHEDULE = "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json"
CONTRACT = "plan/DINO_RCDE_TRACK_R_V123_SCIENTIFIC_REDUCTION_CONTRACT_V1_20260822.md"
AUTOMATIC_CONTINUATION_ADDENDUM = "plan/DINO_RCDE_TRACK_R_V123_AUTOMATIC_CONTINUATION_ADDENDUM_V1_20260824.md"
SEALER = "programs/seal_dino_rcde_track_r_i0_science_metadata_v1.py"
SEAL_VALIDATOR = "programs/validate_dino_rcde_track_r_i0_science_metadata_v1.py"
STATISTICS = "src/rc_aslo_xf/dino_rcde_track_r_statistics_v1.py"
INDEPENDENT_STATISTICS = "programs/dino_rcde_track_r_statistics_independent_v1.py"
REDUCER = "programs/reduce_dino_rcde_track_r_science_v1.py"
VALIDATOR = "programs/validate_dino_rcde_track_r_science_v1.py"
RUNTIME_ENTRY_VALIDATOR = "programs/validate_dino_rcde_track_r_v123_runtime_binding_v1.py"
LAUNCHER = "slurm/dino_rcde_track_r_science_v1.sbatch"
POST_P_CONTROLLER = "slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch"
PROMOTION_CONTROLLER = "slurm/dino_rcde_track_r_v123_promotion_controller_v1.sbatch"
FREEZER = "programs/freeze_dino_rcde_track_r_science_authority_v123.py"
AUTOCHAIN_REGISTRAR = "programs/register_dino_rcde_track_r_v123_autochain_v1.py"
AUTOCHAIN_REGISTRATION_VALIDATOR = "programs/validate_dino_rcde_track_r_v123_autochain_registration_v1.py"
AUTOCHAIN_REGISTRATION = "results/dino_rcde_track_r_autochain_v1/v123_registration.json"
AUTOCHAIN_REGISTRATION_VALIDATION = "results/dino_rcde_track_r_autochain_v1/v123_registration.validation.json"
LINEAGE = "programs/dino_rcde_track_r_v121_lineage_v1.py"
PACKAGE_INITIALIZER = "src/rc_aslo_xf/__init__.py"
I0_SEAL_OUTPUT = "results/dino_rcde_track_r_i0_science_metadata_v1/result.json"
I0_SEAL_VALIDATION_OUTPUT = "results/dino_rcde_track_r_i0_science_metadata_validation_v1/result.json"
SCIENCE_OUTPUT = "results/dino_rcde_track_r_scientific_result_v1/result.json"
SCIENCE_VALIDATION_OUTPUT = "results/dino_rcde_track_r_scientific_validation_v1/result.json"
RUNTIME_ROOTS = (
    SEALER,
    SEAL_VALIDATOR,
    INDEPENDENT_STATISTICS,
    REDUCER,
    VALIDATOR,
    RUNTIME_ENTRY_VALIDATOR,
    LINEAGE,
    PACKAGE_INITIALIZER,
)


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    req(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


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
        (version == 122 and current == parent_path)
        or (version == 123 and current == self_path and self_path.is_file()),
        "V122/V123 authority state drift",
    )
    parent = read(PARENT)
    req(
        parent.get("schema_version") == PARENT_SCHEMA
        and parent.get("status") == PARENT_STATUS
        and parent.get("logical_sha256") == H.logical(parent)
        and (parent_path.stat().st_mode & 0o777) == 0o444,
        "V122 parent drift",
    )
    parent_sha256 = H.file_sha(parent_path)
    v121_path = (ROOT / V121_AUTHORITY).resolve(strict=True)
    v121 = read(V121_AUTHORITY)
    req(
        v121.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
        and v121.get("logical_sha256") == H.logical(v121)
        and (v121_path.stat().st_mode & 0o777) == 0o444,
        "V121 authority drift",
    )
    v121_sha256 = H.file_sha(v121_path)
    completion = read(PREJOIN_COMPLETION)
    req(
        completion.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_V120_COMPLETE"
        and completion.get("query_count") == 594
        and completion.get("source_v121_authority_sha256") == v121_sha256
        and completion.get("logical_sha256") == H.logical(completion),
        "V121 prejoin completion drift",
    )
    comparator = read(P_COMPARATOR)
    comparator_validation = read(P_VALIDATION)
    req(
        comparator.get("status") == "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY"
        and comparator.get("query_count") == 594
        and comparator.get("candidate_count_per_query") == 128
        and comparator.get("target_free_prejoin_score_ledger") is True
        and comparator.get("logical_sha256") == H.logical(comparator)
        and comparator_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS"
        and comparator_validation.get("validation_pass") is True
        and comparator_validation.get("comparator_sha256")
        == H.file_sha(ROOT / P_COMPARATOR)
        and comparator_validation.get("comparator_logical_sha256")
        == comparator["logical_sha256"],
        "V122 comparator independent PASS absent",
    )
    loss = read(LOSS_JOIN)
    i0 = read(I0_POSTJOIN)
    role = read(ROLE_FREE)
    folds = read(FOLD_SCHEDULE)
    raw_bindings = {
        "loss_join_sha256": H.file_sha(ROOT / LOSS_JOIN),
        "i0_postjoin_sha256": H.file_sha(ROOT / I0_POSTJOIN),
        "role_free_pair_sha256": H.file_sha(ROOT / ROLE_FREE),
        "fold_schedule_sha256": H.file_sha(ROOT / FOLD_SCHEDULE),
    }
    expected_seal = build_sealed_metadata(
        loss_join=loss,
        i0_postjoin=i0,
        role_free=role,
        fold_schedule=folds,
        source_bindings=raw_bindings,
    )
    shard_rows = []
    for ordinal in range(50):
        start, stop = ordinal * 12, ordinal * 12 + 12
        producer = ROOT / PREJOIN_ROOT / f"shard_{start:03d}_{stop:03d}.pt"
        validation_path = ROOT / PREJOIN_VALIDATION_ROOT / f"shard_{start:03d}_{stop:03d}.validation.json"
        shard = torch.load(producer, map_location="cpu", weights_only=True, mmap=True)
        validation = json.loads(validation_path.read_text())
        req(
            shard.get("source_bindings", {}).get("authority_sha256") == v121_sha256
            and validation.get("source_v121_authority_sha256") == v121_sha256
            and validation.get("producer_shard_sha256") == H.file_sha(producer)
            and validation.get("producer_shard_logical_sha256") == shard.get("logical_sha256"),
            f"V121 prejoin shard{ordinal} lineage drift",
        )
        shard_rows.append(
            {
                "shard_ordinal": ordinal,
                "producer": H.bind(producer.relative_to(ROOT).as_posix(), immutable=True),
                "validation": H.bind(validation_path.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
            }
        )
    closure_paths, closure_edges = discover_runtime_import_closure(RUNTIME_ROOTS)
    closure_rows = [H.bind(path) for path in closure_paths]
    bindings = {
        "parent_authority_v122": H.bind(PARENT, with_logical=True, immutable=True),
        "authority_v121": H.bind(V121_AUTHORITY, with_logical=True, immutable=True),
        "prejoin_completion": H.bind(PREJOIN_COMPLETION, with_logical=True, immutable=True),
        "prejoin_shards": {"count": 50, "rows": shard_rows, "logical_sha256": H.logical({"rows": shard_rows})},
        "p_comparator": H.bind(P_COMPARATOR, with_logical=True, immutable=True),
        "p_comparator_validation": H.bind(P_VALIDATION, with_logical=True, immutable=True),
        "loss_join": H.bind(LOSS_JOIN, with_logical=True, immutable=True),
        "loss_join_validation": H.bind(LOSS_VALIDATION, with_logical=True, immutable=True),
        "i0_postjoin": H.bind(I0_POSTJOIN, with_logical=True),
        "i0_postjoin_validation": H.bind(I0_VALIDATION, with_logical=True),
        "role_free_pair": H.bind(ROLE_FREE, with_logical=True, immutable=True),
        "fold_schedule": H.bind(FOLD_SCHEDULE, with_logical=True),
        "contract": H.bind(CONTRACT),
        "automatic_continuation_addendum": H.bind(AUTOMATIC_CONTINUATION_ADDENDUM),
        "test": H.bind("tests/test_dino_rcde_track_r_v123_science_chain_v1.py"),
        "metadata_sealer": H.bind(SEALER),
        "metadata_validator": H.bind(SEAL_VALIDATOR),
        "statistics": H.bind(STATISTICS),
        "independent_statistics": H.bind(INDEPENDENT_STATISTICS),
        "reducer": H.bind(REDUCER),
        "validator": H.bind(VALIDATOR),
        "runtime_entry_validator": H.bind(RUNTIME_ENTRY_VALIDATOR),
        "launcher": H.bind(LAUNCHER),
        "post_p_controller": H.bind(POST_P_CONTROLLER),
        "v123_promotion_controller": H.bind(PROMOTION_CONTROLLER),
        "v123_autochain_registrar": H.bind(AUTOCHAIN_REGISTRAR),
        "v123_autochain_registration_validator": H.bind(AUTOCHAIN_REGISTRATION_VALIDATOR),
        "v123_autochain_registration": H.bind(AUTOCHAIN_REGISTRATION, with_logical=True, immutable=True),
        "v123_autochain_registration_validation": H.bind(AUTOCHAIN_REGISTRATION_VALIDATION, with_logical=True, immutable=True),
        "freezer": H.bind(FREEZER),
        "lineage_runtime": H.bind(LINEAGE),
        "package_initializer": H.bind(PACKAGE_INITIALIZER),
        "closure_discovery": H.bind(
            "programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"
        ),
        "runtime_import_closure": {
            "count": len(closure_rows),
            "rows": closure_rows,
            "logical_sha256": H.logical({"rows": closure_rows}),
            "edge_count": len(closure_edges),
            "edges_sha256": H.logical({"edges": closure_edges}),
        },
    }
    authority: dict[str, Any] = {
        "schema_version": "rc_current_authority_v123_20260822",
        "status": STATUS,
        "stage": "TRACK_R_C128_CONDITIONAL_SCIENTIFIC_REDUCTION",
        "claim_level": "GIVEN_C128_SPECIFIC_REFERENCE_EVIDENCE_AND_P_LOCK_INCREMENT_ONLY",
        "parent_v122_authority_sha256": parent_sha256,
        "parent_v122_authority_logical_sha256": parent["logical_sha256"],
        "source_v121_authority_sha256": v121_sha256,
        "source_v121_authority_logical_sha256": v121["logical_sha256"],
        "expected_i0_seal_logical_sha256": expected_seal["logical_sha256"],
        "metadata_seal_authorized": True,
        "metadata_seal_validation_authorized": True,
        "scientific_reduction_authorized": True,
        "scientific_validation_authorized": True,
        "postjoin_model_forward_authorized": False,
        "opened_sealed_access_authorized": False,
        "protected_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "prejoin_root": PREJOIN_ROOT,
        "prejoin_validation_root": PREJOIN_VALIDATION_ROOT,
        "i0_seal_output": I0_SEAL_OUTPUT,
        "i0_seal_validation_output": I0_SEAL_VALIDATION_OUTPUT,
        "science_result_output": SCIENCE_OUTPUT,
        "science_validation_output": SCIENCE_VALIDATION_OUTPUT,
        "resource_contract": {"partition": "cpuonly", "cpus_per_task": 4, "memory_megabytes": 65536, "walltime_seconds": 3600},
        "bindings": bindings,
    }
    authority["logical_sha256"] = H.logical(authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False), "output drift")
    value = build()
    H.atomic(args.output, value)
    print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
