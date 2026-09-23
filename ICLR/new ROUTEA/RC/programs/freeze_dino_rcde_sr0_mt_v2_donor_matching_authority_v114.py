#!/usr/bin/env python3
"""Freeze V114 scheduler adoption and exact-geometry donor matching reducer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v114_20260820.json"
PARENT = "registry/current_authority_v113_20260820.json"
STATUS = "RCDE_SR0_MT_V2_DONOR_MATCHING_REDUCER_AUTHORIZED"


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
        (version == 113 and current == parent_path)
        or (version == 114 and current == self_path and self_path.is_file()),
        "V113/V114 authority state drift",
    )
    parent = json.loads(parent_path.read_text())
    req(
        parent.get("status")
        == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_RESCOPE_REPAIR_AUTHORIZED",
        "V113 parent drift",
    )
    migration_path = (
        "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/"
        "scheduler_migration_closure_input_v1.json"
    )
    migration = json.loads((ROOT / migration_path).read_text())
    moved_launcher = (
        "slurm/dino_rcde_sr0_mt_v2_donor_geometry_projection_cpu_moved11_19_v1.sbatch"
    )
    override_addendum = (
        "plan/DINO_RCDE_SR0_MT_V2_DONOR_GEOMETRY_V113_SCHEDULER_OVERRIDE_20260820.md"
    )
    migration_stop = (
        "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/"
        "continuation/dev_submit_11.claim.json"
    )
    req(
        migration.get("status")
        == "RCDE_SR0_MT_V2_PROJECTION_SCHEDULER_MIGRATION_INPUT_READY"
        and migration.get("shard_count") == 50
        and migration.get("query_count") == 594
        and migration.get("scope_projection_count") == 2376
        and migration.get("direction_record_decode_count") == 4752
        and migration.get("moved_launcher_sha256") == H.file_sha(ROOT / moved_launcher)
        and migration.get("scheduler_override_addendum_sha256")
        == H.file_sha(ROOT / override_addendum)
        and migration.get("migration_stop_sha256") == H.file_sha(ROOT / migration_stop),
        "scheduler migration receipt drift",
    )
    v92_authority = json.loads((ROOT / "registry/current_authority_v92_20260818.json").read_text())
    bindings = {
        "parent_authority_v113": H.bind(PARENT, with_logical=True, immutable=True),
        "scheduler_migration_receipt": H.bind(
            migration_path, with_logical=True, immutable=True
        ),
        "scheduler_migration_receipt_producer": H.bind(
            "programs/materialize_dino_rcde_sr0_mt_v2_projection_scheduler_migration_receipt_v1.py"
        ),
        "scheduler_override_addendum": H.bind(override_addendum),
        "moved_launcher": H.bind(moved_launcher),
        "migration_stop": H.bind(migration_stop, with_logical=True, immutable=True),
        "v87_query_manifest": H.bind(
            "results/dino_rcde_sr0_mt_training_consumer_e0_full594_structure_sidecar_validation_v1/manifest.json",
            with_logical=True,
            immutable=True,
        ),
        "prejoin_folds": H.bind(
            "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json",
            with_logical=True,
        ),
        "training_roles": H.bind(
            "protocols/dino_rcde_training_roles_600_v1_2_20260812.json",
            with_logical=True,
        ),
        "v92_authority": H.bind(
            "registry/current_authority_v92_20260818.json",
            with_logical=True,
            immutable=True,
        ),
        "v92_feasibility": H.bind(
            "results/dino_rcde_sr0_mt_v2_donor_metadata_feasibility_v1/feasibility.json",
            with_logical=True,
            immutable=True,
        ),
        "v92_feasibility_validation": H.bind(
            "results/dino_rcde_sr0_mt_v2_donor_metadata_feasibility_validation_v1/result.json",
            with_logical=True,
            immutable=True,
        ),
        "donor_contract": H.bind(
            "plan/DINO_RCDE_SR0_MT_V2_GEOMETRY_MATCHED_DONOR_CONTRACT_V1_20260820.md"
        ),
        "reducer_contract": H.bind(
            "plan/DINO_RCDE_SR0_MT_V2_DONOR_MATCHING_REDUCER_CONTRACT_V1_20260820.md"
        ),
        "producer": H.bind(
            "programs/materialize_dino_rcde_sr0_mt_v2_donor_matching_v1.py"
        ),
        "validator": H.bind(
            "programs/validate_dino_rcde_sr0_mt_v2_donor_matching_v1.py"
        ),
        "freezer": H.bind(
            "programs/freeze_dino_rcde_sr0_mt_v2_donor_matching_authority_v114.py"
        ),
        "launcher": H.bind(
            "slurm/dino_rcde_sr0_mt_v2_donor_matching_dev_v1.sbatch"
        ),
        "test": H.bind("tests/test_dino_rcde_sr0_mt_v2_donor_matching_v114.py"),
    }
    # Preserve the exact 16 membership bindings already independently frozen by V92.
    bindings["membership_manifests"] = v92_authority["bindings"][
        "membership_manifests"
    ]
    authority = {
        "schema_version": "rc_current_authority_v114_20260820",
        "status": STATUS,
        "stage": "SR0_MT_V2_SCHEDULER_ADOPTION_AND_DONOR_MATCHING_REDUCER",
        "claim_level": "ENGINEERING_SCHEDULER_ADOPTION_AND_FROZEN_EXACT_GEOMETRY_DONOR_MATCHING_ONLY",
        "authorized_scope": {
            "scheduler_migration_adoption": True,
            "projection_receipt_aggregate": True,
            "optimization_metadata_edge_join": True,
            "canonical_partial_donor_matching": True,
            "independent_empty_graph_validation": True,
        },
        "query_count": 594,
        "scope_count": 16,
        "scope_node_count": 2376,
        "direction_record_count": 4752,
        "projection_receipt_read_authorized": True,
        "scheduler_migration_adoption_authorized": True,
        "optimization_identity_supergroup_read_authorized": True,
        "final_donor_matching_authorized": True,
        "lock_payload_deserialization_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "training_authorized": False,
        "V_training_authorized": False,
        "heldout_scoring_authorized": False,
        "protected_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "output": "results/dino_rcde_sr0_mt_v2_donor_matching_v1/result.json",
        "validation_output": "results/dino_rcde_sr0_mt_v2_donor_matching_validation_v1/result.json",
        "resource_contract": {
            "partition": "dev_cpuonly",
            "cpus_per_task": 1,
            "memory_megabytes": 4096,
            "walltime_seconds": 1800,
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
    value = build()
    H.atomic(args.output, value)
    print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
