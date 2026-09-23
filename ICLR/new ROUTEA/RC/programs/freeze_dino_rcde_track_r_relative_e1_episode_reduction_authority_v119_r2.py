#!/usr/bin/env python3
"""Freeze same-number V119-r2 after the runtime episode-address failure.

The old V119 must first be moved intact to the fixed archive path.  This
freezer preserves V118 and every E1 shard, binds job 5097842's failure, and
authorizes only a lossless ledger-runtime address normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v119_20260821.json"
ARCHIVED_V119_R1 = "registry/archive/authority_v119_r1_20260822.json"
PARENT = "registry/current_authority_v118_20260821.json"
STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_REDUCTION_AUTHORIZED"
SCHEMA = "rc_current_authority_v119_20260821"
OLD_V119_SHA256 = "d54b40a969a81d5706600bbb76c7cd6ea04198397ed8504e626e596b481cef51"
OLD_V119_LOGICAL_SHA256 = (
    "e8775882e7fea0afb630dd7de75599c0d904778fb835f50510a786beaf110938"
)
FAILURE_OUT = "logs/rcde_tr_c1v3-5097842.out"
FAILURE_ERR = "logs/rcde_tr_c1v3-5097842.err"
FAILURE_OUT_SHA256 = "1e0a27d0d2d52f2b1a2bdd663f1d90432b8d07ff2f58ad541dd19366ce48e728"
FAILURE_ERR_SHA256 = "4c12ae82cd11eb51d21e74c7626a1dd0582b6d2eac29dc3944d9292d02c70306"
RUNTIME_EPISODE_ID_NAMESPACE = (
    "DINO_RCDE_TRACK_R_LEDGER_RUNTIME_EPISODE_ID_V2"
)
PRODUCER_ROOT = "results/dino_rcde_track_r_relative_e1_shards_v1/producer"
VALIDATION_ROOT = "results/dino_rcde_track_r_relative_e1_shards_v1/validation"
LEDGER_ROOT = "results/dino_rcde_track_r_relative_e1_ledgers_v1"
LEDGER_VALIDATION = (
    "results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json"
)
V120 = "registry/current_authority_v120_20260821.json"
SHARD_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_v1_20260821"
SHARD_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_READY"
SHARD_VALIDATION_STATUS = (
    "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
)
EXPECTED_PER_OUTER = {1: 445, 2: 446, 3: 445, 4: 446}
EXPECTED_SOURCE_COLLISIONS = {1: 2, 2: 2, 3: 3, 4: 2}


class V119RepairError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise V119RepairError(message)


def require_shard_source_authority(
    shard: Mapping[str, Any], expected_sha256: str, shard_ordinal: int
) -> None:
    source_bindings = shard.get("source_bindings")
    require(
        isinstance(source_bindings, Mapping)
        and source_bindings.get("authority_sha256") == expected_sha256,
        f"E1 shard{shard_ordinal} V118 source-authority SHA drift",
    )


def read_json(relative: str) -> dict[str, Any]:
    path = (ROOT / relative).resolve(strict=True)
    require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), f"unsafe JSON path: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def validate_archive_and_failure() -> dict[str, Any]:
    current = ROOT / AUTH
    require(
        not current.exists() and not current.is_symlink(),
        "current V119 must be absent after explicit archive",
    )
    archive = (ROOT / ARCHIVED_V119_R1).resolve(strict=True)
    require(
        archive.is_file()
        and not archive.is_symlink()
        and (archive.stat().st_mode & 0o777) == 0o444
        and H.file_sha(archive) == OLD_V119_SHA256,
        "archived V119-r1 physical drift",
    )
    old = read_json(ARCHIVED_V119_R1)
    require(
        old.get("schema_version") == SCHEMA
        and old.get("status") == STATUS
        and old.get("logical_sha256")
        == H.logical(old)
        == OLD_V119_LOGICAL_SHA256
        and old.get("parent_v118_authority_revision") == 2
        and old.get("training_authorized") is False
        and old.get("scientific_GO_or_NO_GO") is None,
        "archived V119-r1 logical/boundary drift",
    )
    out_path = (ROOT / FAILURE_OUT).resolve(strict=True)
    err_path = (ROOT / FAILURE_ERR).resolve(strict=True)
    require(
        H.file_sha(out_path) == FAILURE_OUT_SHA256
        and H.file_sha(err_path) == FAILURE_ERR_SHA256
        and "outer fold1 episode ID collision" in err_path.read_text(encoding="utf-8")
        and "TrackRE1ReductionError" in err_path.read_text(encoding="utf-8"),
        "job5097842 failure evidence drift",
    )
    return old


def runtime_id(source_episode: Mapping[str, Any]) -> str:
    query = source_episode.get("query")
    require(isinstance(query, Mapping), "source query address absent")
    value = {
        "namespace": RUNTIME_EPISODE_ID_NAMESPACE,
        "source_episode_id": source_episode.get("episode_id"),
        "source_record_sha256": source_episode.get("record_sha256"),
        "execution_ordinal": query.get("execution_ordinal"),
        "outer_fold": query.get("outer_fold"),
    }
    require(
        isinstance(value["source_episode_id"], str)
        and len(value["source_episode_id"]) == 64
        and isinstance(value["source_record_sha256"], str)
        and len(value["source_record_sha256"]) == 64
        and type(value["execution_ordinal"]) is int
        and value["outer_fold"] in EXPECTED_PER_OUTER,
        "source episode normalization input drift",
    )
    return H.logical(value)


def build() -> dict[str, Any]:
    old = validate_archive_and_failure()
    for relative in (LEDGER_ROOT, LEDGER_VALIDATION, V120):
        path = ROOT / relative
        require(not path.exists() and not path.is_symlink(), f"failed V119 created downstream output: {relative}")

    parent_binding = H.bind(PARENT, with_logical=True, immutable=True)
    require(
        parent_binding == old.get("bindings", {}).get("parent_authority_v118"),
        "V118 scientific parent differs from archived V119-r1",
    )
    parent = read_json(PARENT)
    require(
        parent.get("schema_version") == "rc_current_authority_v118_20260821"
        and parent.get("status")
        == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED"
        and parent.get("authority_revision") == 2,
        "V118-r2 parent drift",
    )
    parent_sha256 = parent_binding["sha256"]

    shard_rows: list[dict[str, Any]] = []
    source_ids: dict[int, list[str]] = {fold: [] for fold in EXPECTED_PER_OUTER}
    runtime_ids: dict[int, list[str]] = {fold: [] for fold in EXPECTED_PER_OUTER}
    total_queries = total_excluded = total_episodes = 0
    producer_root = ROOT / PRODUCER_ROOT
    validation_root = ROOT / VALIDATION_ROOT
    for ordinal in range(50):
        start = ordinal * 12
        stop = start + 12
        producer = producer_root / f"shard_{start:03d}_{stop:03d}.pt"
        validation_path = validation_root / f"shard_{start:03d}_{stop:03d}.validation.json"
        require(producer.is_file() and validation_path.is_file(), f"E1 shard{ordinal} absent")
        shard = torch.load(producer, map_location="cpu", weights_only=True, mmap=True)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("shard_ordinal") == ordinal
            and shard.get("logical_sha256") == H.logical(shard)
            and validation.get("status") == SHARD_VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("producer_shard_sha256") == H.file_sha(producer)
            and validation.get("producer_shard_logical_sha256")
            == shard["logical_sha256"],
            f"E1 shard{ordinal} validation/source drift",
        )
        require_shard_source_authority(shard, parent_sha256, ordinal)
        for episode in shard["episodes"]:
            fold = int(episode["query"]["outer_fold"])
            source_ids[fold].append(str(episode["episode_id"]))
            runtime_ids[fold].append(runtime_id(episode))
        total_queries += int(shard["eligible_query_count"])
        total_excluded += int(shard["excluded_query_count"])
        total_episodes += int(shard["episode_count"])
        shard_rows.append(
            {
                "shard_ordinal": ordinal,
                "producer": H.bind(producer.relative_to(ROOT).as_posix(), immutable=True),
                "validation": H.bind(validation_path.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
            }
        )
    require((total_queries, total_excluded, total_episodes) == (594, 6, 1782), "E1 594/6/1782 closure drift")
    diagnostic_rows = []
    total_source_collisions = 0
    for fold in EXPECTED_PER_OUTER:
        source = source_ids[fold]
        runtime = runtime_ids[fold]
        source_collisions = len(source) - len(set(source))
        runtime_collisions = len(runtime) - len(set(runtime))
        require(
            len(source) == EXPECTED_PER_OUTER[fold]
            and source_collisions == EXPECTED_SOURCE_COLLISIONS[fold]
            and runtime_collisions == 0,
            f"outer fold{fold} runtime address diagnostic drift",
        )
        total_source_collisions += source_collisions
        diagnostic_rows.append(
            {
                "outer_fold": fold,
                "episode_count": len(source),
                "source_unique_episode_id_count": len(set(source)),
                "source_episode_id_collision_count": source_collisions,
                "runtime_unique_episode_id_count": len(set(runtime)),
                "runtime_episode_id_collision_count": runtime_collisions,
                "source_episode_id_sequence_sha256": H.logical({"rows": source}),
                "runtime_episode_id_sequence_sha256": H.logical({"rows": runtime}),
            }
        )
    require(
        total_source_collisions == 9,
        "job5097842 exact nine-collision diagnostic not reproduced",
    )
    all_runtime = [item for fold in EXPECTED_PER_OUTER for item in runtime_ids[fold]]
    require(len(set(all_runtime)) == 1782, "global runtime episode address collision")

    bindings = {
        "parent_authority_v118": parent_binding,
        "superseded_authority_v119_revision1_archive": H.bind(ARCHIVED_V119_R1, with_logical=True, immutable=True),
        "failure_job5097842_stdout": H.bind(FAILURE_OUT),
        "failure_job5097842_stderr": H.bind(FAILURE_ERR),
        "reducer": H.bind("programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"),
        "validator": H.bind("programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"),
        "episode_core_unchanged": H.bind("src/rc_aslo_xf/dino_rcde_track_r_episode_v1.py"),
        "test": H.bind("tests/test_dino_rcde_track_r_episode_v1.py"),
        "address_repair_test": H.bind("tests/test_dino_rcde_track_r_v119_r2_episode_address_repair_v1.py"),
        "orchestration_test": H.bind(
            "tests/test_dino_rcde_track_r_post_e1_controller_v3.py"
        ),
        "cross_revision_test": H.bind("tests/test_dino_rcde_track_r_v119_cross_revision_guard_v1.py"),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2.py"),
        "controller": H.bind("slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch"),
        "shards": {
            "count": 50,
            "rows": shard_rows,
            "logical_sha256": H.logical({"rows": shard_rows}),
        },
    }
    authority: dict[str, Any] = {
        "schema_version": SCHEMA,
        "authority_version": 119,
        "authority_revision": 2,
        "status": STATUS,
        "stage": "TRACK_R_RELATIVE_E1_EPISODE_LEDGER_REDUCTION",
        "claim_level": "ENGINEERING_LOSSLESS_LEDGER_RUNTIME_EPISODE_ADDRESS_REPAIR_ONLY",
        "supersedes": {
            "authority_version": 119,
            "authority_revision": 1,
            "physical_sha256": OLD_V119_SHA256,
            "logical_sha256": OLD_V119_LOGICAL_SHA256,
            "status": STATUS,
            "failure_job_id": 5097842,
        },
        "parent_v118_authority_revision": 2,
        "shard_source_authority_sha256": parent_sha256,
        "all_shard_source_authority_match": True,
        "runtime_episode_address_repair": {
            "namespace": RUNTIME_EPISODE_ID_NAMESPACE,
            "scope": "LEDGER_RUNTIME_EPISODE_ID_AND_DERIVED_RECORD_SHA256_ONLY",
            "id_input_fields": [
                "namespace",
                "source_episode_id",
                "source_record_sha256",
                "execution_ordinal",
                "outer_fold",
            ],
            "changed_episode_fields": ["episode_id", "record_sha256"],
            "source_shard_or_episode_mutation_authorized": False,
            "source_to_runtime_address_map_required": True,
            "source_episode_count": 1782,
            "source_episode_id_collision_count": total_source_collisions,
            "runtime_episode_id_collision_count": 0,
            "global_runtime_unique_episode_id_count": 1782,
            "per_outer_fold": diagnostic_rows,
            "per_outer_fold_logical_sha256": H.logical({"rows": diagnostic_rows}),
        },
        "query_count": 594,
        "excluded_query_count": 6,
        "total_episode_count": 1782,
        "expected_per_outer_fold": {str(key): value for key, value in EXPECTED_PER_OUTER.items()},
        "episode_reduction_authorized": True,
        "episode_reduction_validation_authorized": True,
        "token_load_authorized": False,
        "model_load_authorized": False,
        "training_authorized": False,
        "heldout_scoring_authorized": False,
        "protected_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "output_root": LEDGER_ROOT,
        "validation_output": LEDGER_VALIDATION,
        "resource_contract": {
            "partition": "cpuonly",
            "cpus_per_task": 4,
            "memory_megabytes": 65536,
            "walltime_seconds": 3600,
        },
        "bindings": bindings,
    }
    authority["logical_sha256"] = H.logical(authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    require(
        args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False),
        "V119-r2 output path drift",
    )
    value = build()
    H.atomic(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "authority_revision": 2,
                "source_episode_id_collision_count": value[
                    "runtime_episode_address_repair"
                ]["source_episode_id_collision_count"],
                "logical_sha256": value["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
