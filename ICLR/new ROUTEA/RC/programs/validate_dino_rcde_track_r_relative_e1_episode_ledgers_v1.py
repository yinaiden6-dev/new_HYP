#!/usr/bin/env python3
"""Independently replay the four reduced Track-R compact episode ledgers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_track_r_episode_v1 import (  # noqa: E402
    SOURCE_FOLDS,
    validate_compact_episode,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402


AUTHORITY_PATH = "registry/current_authority_v119_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v119_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_REDUCTION_AUTHORIZED"
PARENT_AUTHORITY_PATH = "registry/current_authority_v118_20260821.json"
PARENT_AUTHORITY_SCHEMA = "rc_current_authority_v118_20260821"
PARENT_AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED"
SHARD_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_v1_20260821"
SHARD_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_READY"
SHARD_VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
LEDGER_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledger_v1_20260821"
LEDGER_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_READY"
INDEX_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledger_index_v1_20260821"
INDEX_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_INDEX_READY"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledgers_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS"
SHARD_COUNT = 50
SHARD_SIZE = 12
QUERY_COUNT = 600
VALID_QUERY_COUNT = 594
EXPECTED_PER_OUTER = {1: 445, 2: 446, 3: 445, 4: 446}
RUNTIME_EPISODE_ID_NAMESPACE = (
    "DINO_RCDE_TRACK_R_LEDGER_RUNTIME_EPISODE_ID_V2"
)
SUPERSEDED_V119_SHA256 = (
    "d54b40a969a81d5706600bbb76c7cd6ea04198397ed8504e626e596b481cef51"
)
SUPERSEDED_V119_LOGICAL_SHA256 = (
    "e8775882e7fea0afb630dd7de75599c0d904778fb835f50510a786beaf110938"
)


class TrackRE1LedgerValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRE1LedgerValidationError(message)


def independently_normalize_ledger_runtime_episode_address(
    source_episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently reconstruct the lossless ledger-only address repair."""

    validate_compact_episode(source_episode)
    query = source_episode.get("query")
    require(isinstance(query, Mapping), "source episode query absent")
    execution = query.get("execution_ordinal")
    outer_fold = query.get("outer_fold")
    source_episode_id = source_episode.get("episode_id")
    source_record_sha256 = source_episode.get("record_sha256")
    require(
        type(execution) is int
        and outer_fold in SOURCE_FOLDS
        and isinstance(source_episode_id, str)
        and len(source_episode_id) == 64
        and isinstance(source_record_sha256, str)
        and len(source_record_sha256) == 64,
        "source episode address drift",
    )
    inputs = {
        "namespace": RUNTIME_EPISODE_ID_NAMESPACE,
        "source_episode_id": source_episode_id,
        "source_record_sha256": source_record_sha256,
        "execution_ordinal": execution,
        "outer_fold": outer_fold,
    }
    runtime_episode_id = canonical_sha256(inputs)
    runtime_episode = copy.deepcopy(dict(source_episode))
    runtime_episode["episode_id"] = runtime_episode_id
    runtime_episode["record_sha256"] = canonical_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in runtime_episode.items()
            if key != "record_sha256"
        }
    )
    validate_compact_episode(runtime_episode)
    require(
        {
            key: value
            for key, value in runtime_episode.items()
            if key not in {"episode_id", "record_sha256"}
        }
        == {
            key: value
            for key, value in source_episode.items()
            if key not in {"episode_id", "record_sha256"}
        },
        "independent runtime normalization changed source payload",
    )
    address = {
        **inputs,
        "runtime_episode_id": runtime_episode_id,
        "runtime_record_sha256": runtime_episode["record_sha256"],
        "changed_fields": ["episode_id", "record_sha256"],
    }
    address["address_record_sha256"] = canonical_sha256(address)
    return runtime_episode, address


def require_shard_source_authority(
    shard: Mapping[str, Any], expected_sha256: str, shard_ordinal: int
) -> None:
    source_bindings = shard.get("source_bindings")
    require(
        isinstance(source_bindings, Mapping)
        and source_bindings.get("authority_sha256") == expected_sha256,
        f"E1 shard{shard_ordinal} V118 source-authority SHA drift",
    )


def bound_parent_v118_sha256(authority: Mapping[str, Any]) -> str:
    parent_binding = authority.get("bindings", {}).get("parent_authority_v118")
    require(
        isinstance(parent_binding, Mapping)
        and parent_binding.get("path") == PARENT_AUTHORITY_PATH
        and isinstance(parent_binding.get("sha256"), str)
        and authority.get("parent_v118_authority_revision") == 2
        and authority.get("shard_source_authority_sha256")
        == parent_binding.get("sha256")
        and authority.get("all_shard_source_authority_match") is True,
        "V119/V118-r2 source-authority contract drift",
    )
    parent_path = safe_path(RC_ROOT / PARENT_AUTHORITY_PATH)
    parent = read_json(parent_path)
    expected_parent_sha256 = str(parent_binding["sha256"])
    require(
        parent.get("schema_version") == PARENT_AUTHORITY_SCHEMA
        and parent.get("status") == PARENT_AUTHORITY_STATUS
        and parent.get("authority_revision") == 2
        and file_sha256(parent_path) == expected_parent_sha256,
        "independent bound current V118-r2 physical SHA drift",
    )
    return expected_parent_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, must_exist: bool = True, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable validation exists: {output}")
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    output.chmod(0o444)


def validate(
    authority_path: Path,
    producer_root: Path,
    shard_validation_root: Path,
    ledger_root: Path,
    output: Path,
) -> dict[str, Any]:
    authority_path = safe_path(authority_path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "authority path drift")
    authority = read_json(authority_path)
    repair = authority.get("runtime_episode_address_repair")
    supersedes = authority.get("supersedes")
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("authority_version") == 119
        and authority.get("authority_revision") == 2
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("episode_reduction_validation_authorized") is True
        and authority.get("token_load_authorized") is False
        and authority.get("model_load_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V119 validation authority drift",
    )
    require(
        isinstance(supersedes, Mapping)
        and supersedes.get("physical_sha256") == SUPERSEDED_V119_SHA256
        and supersedes.get("logical_sha256")
        == SUPERSEDED_V119_LOGICAL_SHA256
        and supersedes.get("failure_job_id") == 5097842
        and isinstance(repair, Mapping)
        and repair.get("namespace") == RUNTIME_EPISODE_ID_NAMESPACE
        and repair.get("scope")
        == "LEDGER_RUNTIME_EPISODE_ID_AND_DERIVED_RECORD_SHA256_ONLY"
        and repair.get("changed_episode_fields")
        == ["episode_id", "record_sha256"]
        and repair.get("source_shard_or_episode_mutation_authorized") is False
        and repair.get("source_to_runtime_address_map_required") is True
        and repair.get("source_episode_id_collision_count", 0) > 0
        and repair.get("runtime_episode_id_collision_count") == 0,
        "independent V119-r2 runtime address repair contract drift",
    )
    expected_v118_sha256 = bound_parent_v118_sha256(authority)
    producer = safe_path(producer_root, file=False)
    shard_validations = safe_path(shard_validation_root, file=False)
    ledgers_root = safe_path(ledger_root, file=False)
    index_path = safe_path(ledgers_root / "index.json")
    index = read_json(index_path)
    require(
        index.get("schema_version") == INDEX_SCHEMA
        and index.get("status") == INDEX_STATUS
        and index.get("logical_sha256") == logical_sha256(index)
        and index.get("authority_sha256") == file_sha256(authority_path)
        and index.get("query_count") == VALID_QUERY_COUNT
        and index.get("excluded_query_count") == 6
        and index.get("episode_multiplicity_per_query") == 3
        and index.get("total_episode_count") == VALID_QUERY_COUNT * 3,
        "episode ledger index drift",
    )
    by_outer: dict[int, list[dict[str, Any]]] = {fold: [] for fold in SOURCE_FOLDS}
    shard_receipts: list[dict[str, Any]] = []
    source_bindings: Mapping[str, Any] | None = None
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(
            shard_validations / f"shard_{start:03d}_{stop:03d}.validation.json"
        )
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("logical_sha256") == logical_sha256(shard)
            and validation.get("status") == SHARD_VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and validation.get("producer_shard_logical_sha256")
            == shard["logical_sha256"],
            "validated source shard drift",
        )
        require_shard_source_authority(
            shard, expected_v118_sha256, shard_ordinal
        )
        if source_bindings is None:
            source_bindings = shard["source_bindings"]
        else:
            require(shard["source_bindings"] == source_bindings, "source binding drift")
        for episode in shard["episodes"]:
            validate_compact_episode(episode)
            by_outer[int(episode["query"]["outer_fold"])].append(episode)
        shard_receipts.append(
            {
                "shard_ordinal": shard_ordinal,
                "execution_start": start,
                "execution_stop": stop,
                "producer_path": shard_path.relative_to(RC_ROOT).as_posix(),
                "producer_sha256": file_sha256(shard_path),
                "producer_logical_sha256": shard["logical_sha256"],
                "validation_path": validation_path.relative_to(RC_ROOT).as_posix(),
                "validation_sha256": file_sha256(validation_path),
                "validation_logical_sha256": validation["logical_sha256"],
            }
        )
    require(index.get("shard_receipts") == shard_receipts, "index/source shard receipt drift")
    require(index.get("ordered_shards_sha256") == canonical_sha256(shard_receipts), "ordered shard digest drift")
    ledger_receipts: list[dict[str, Any]] = []
    runtime_address_map_receipts: list[dict[str, Any]] = []
    total_source_episode_id_collisions = 0
    query_occurrences: dict[int, int] = {}
    for outer_fold in SOURCE_FOLDS:
        path = safe_path(ledgers_root / f"outer_fold{outer_fold}.episodes.pt")
        ledger = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        source_expected = sorted(
            by_outer[outer_fold],
            key=lambda item: int(item["query"]["execution_ordinal"]),
        )
        source_collision_count = len(source_expected) - len(
            {item["episode_id"] for item in source_expected}
        )
        total_source_episode_id_collisions += source_collision_count
        expected: list[dict[str, Any]] = []
        expected_address_map: list[dict[str, Any]] = []
        for source_episode in source_expected:
            runtime_episode, address = (
                independently_normalize_ledger_runtime_episode_address(
                    source_episode
                )
            )
            expected.append(runtime_episode)
            expected_address_map.append(address)
        address_map_sha256 = canonical_sha256(expected_address_map)
        require(
            isinstance(ledger, Mapping)
            and ledger.get("schema_version") == LEDGER_SCHEMA
            and ledger.get("status") == LEDGER_STATUS
            and ledger.get("logical_sha256") == logical_sha256(ledger)
            and ledger.get("outer_fold") == outer_fold
            and ledger.get("episode_count") == EXPECTED_PER_OUTER[outer_fold]
            and ledger.get("episodes") == expected
            and ledger.get("runtime_episode_id_namespace")
            == RUNTIME_EPISODE_ID_NAMESPACE
            and ledger.get("source_to_runtime_address_map")
            == expected_address_map
            and ledger.get("source_to_runtime_address_map_sha256")
            == address_map_sha256
            and ledger.get("source_episode_id_collision_count")
            == source_collision_count
            and ledger.get("runtime_episode_id_collision_count") == 0
            and len({item["episode_id"] for item in expected})
            == len(expected)
            and ledger.get("episode_id_sequence_sha256")
            == canonical_sha256([item["episode_id"] for item in expected])
            and ledger.get("episode_record_sequence_sha256")
            == canonical_sha256([item["record_sha256"] for item in expected])
            and ledger.get("source_bindings") == dict(source_bindings or {})
            and ledger.get("source_shard_receipts_sha256")
            == canonical_sha256(shard_receipts),
            f"outer fold{outer_fold} ledger replay drift",
        )
        for episode in expected:
            execution = int(episode["query"]["execution_ordinal"])
            query_occurrences[execution] = query_occurrences.get(execution, 0) + 1
        ledger_receipts.append(
            {
                "outer_fold": outer_fold,
                "path": f"outer_fold{outer_fold}.episodes.pt",
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "logical_sha256": ledger["logical_sha256"],
                "episode_count": len(expected),
                "source_to_runtime_address_map_sha256": address_map_sha256,
                "source_episode_id_collision_count": source_collision_count,
                "runtime_episode_id_collision_count": 0,
            }
        )
        runtime_address_map_receipts.append(
            {
                "outer_fold": outer_fold,
                "address_count": len(expected_address_map),
                "source_to_runtime_address_map_sha256": address_map_sha256,
                "source_episode_id_collision_count": source_collision_count,
                "runtime_episode_id_collision_count": 0,
            }
        )
    require(index.get("ledgers") == ledger_receipts, "index/ledger receipt drift")
    require(
        total_source_episode_id_collisions
        == repair["source_episode_id_collision_count"]
        == 9
        and index.get("runtime_episode_id_namespace")
        == RUNTIME_EPISODE_ID_NAMESPACE
        and index.get("runtime_address_map_receipts")
        == runtime_address_map_receipts
        and index.get("runtime_address_map_receipts_sha256")
        == canonical_sha256(runtime_address_map_receipts)
        and index.get("source_episode_id_collision_count")
        == total_source_episode_id_collisions
        and index.get("runtime_episode_id_collision_count") == 0,
        "index runtime address repair receipt drift",
    )
    require(
        len(query_occurrences) == VALID_QUERY_COUNT
        and set(query_occurrences.values()) == {3},
        "594-query three-ledger multiplicity drift",
    )
    result: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "index_sha256": file_sha256(index_path),
        "index_logical_sha256": index["logical_sha256"],
        "query_count": VALID_QUERY_COUNT,
        "excluded_query_count": 6,
        "total_episode_count": VALID_QUERY_COUNT * 3,
        "per_outer_fold_episode_count": {
            str(key): value for key, value in EXPECTED_PER_OUTER.items()
        },
        "independent_shard_read_count": SHARD_COUNT,
        "independent_ledger_read_count": 4,
        "source_v118_authority_sha256": expected_v118_sha256,
        "all_shard_source_authority_match": True,
        "runtime_episode_id_namespace": RUNTIME_EPISODE_ID_NAMESPACE,
        "source_episode_id_collision_count": total_source_episode_id_collisions,
        "runtime_episode_id_collision_count": 0,
        "runtime_address_map_receipts_sha256": canonical_sha256(
            runtime_address_map_receipts
        ),
        "token_load_count": 0,
        "model_load_count": 0,
        "training_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "next_eligible_contract": "TRACK_R_RELATIVE_V_FIT",
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--shard-validation-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.authority,
        args.producer_root,
        args.shard_validation_root,
        args.ledger_root,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_episode_count": result["total_episode_count"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
