#!/usr/bin/env python3
"""Reduce 50 validated compact shards into four Track-R V-training ledgers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
SHARD_VALIDATION_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_validation_v1_20260821"
SHARD_VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
LEDGER_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledger_v1_20260821"
LEDGER_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_READY"
INDEX_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_ledger_index_v1_20260821"
INDEX_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGER_INDEX_READY"
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


class TrackRE1ReductionError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRE1ReductionError(message)


def normalize_ledger_runtime_episode_address(
    source_episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Losslessly replace only the runtime ID and its integrity hash.

    V118 source shards remain byte-for-byte untouched.  The map returned with
    the runtime copy seals the reversible source address and the exact ID
    derivation used only by the reduced training ledger.
    """

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
    address_inputs = {
        "namespace": RUNTIME_EPISODE_ID_NAMESPACE,
        "source_episode_id": source_episode_id,
        "source_record_sha256": source_record_sha256,
        "execution_ordinal": execution,
        "outer_fold": outer_fold,
    }
    runtime_episode_id = canonical_sha256(address_inputs)
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
        "runtime address normalization changed episode payload",
    )
    address = {
        **address_inputs,
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
        "bound current V118-r2 physical SHA drift",
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
    require(not path.exists(), f"immutable JSON exists: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.chmod(0o444)


def atomic_torch(path: Path, value: object) -> None:
    require(not path.exists(), f"immutable tensor ledger exists: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)
    path.chmod(0o444)


def validate_authority(path: Path) -> tuple[dict[str, Any], str, str]:
    authority_path = safe_path(path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "V119 authority path drift")
    value = read_json(authority_path)
    repair = value.get("runtime_episode_address_repair")
    supersedes = value.get("supersedes")
    require(
        value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("authority_version") == 119
        and value.get("authority_revision") == 2
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("episode_reduction_authorized") is True
        and value.get("token_load_authorized") is False
        and value.get("model_load_authorized") is False
        and value.get("training_authorized") is False
        and value.get("protected_access_authorized") is False
        and value.get("automatic_stage_advance") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("next_authorized_stage") is None,
        "V119 reduction authority drift",
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
        "V119-r2 runtime address repair contract drift",
    )
    expected_parent_sha256 = bound_parent_v118_sha256(value)
    return value, file_sha256(authority_path), expected_parent_sha256


def reduce(
    authority_path: Path,
    producer_root: Path,
    validation_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    authority, authority_sha, expected_v118_sha256 = validate_authority(
        authority_path
    )
    producer = safe_path(producer_root, file=False)
    validations = safe_path(validation_root, file=False)
    output = safe_path(output_root, must_exist=False, file=False)
    require(not output.exists(), "immutable E1 ledger output root exists")
    by_outer: dict[int, list[dict[str, Any]]] = {fold: [] for fold in SOURCE_FOLDS}
    query_occurrence: dict[int, set[int]] = {}
    shard_receipts: list[dict[str, Any]] = []
    total_eligible = total_excluded = total_episodes = 0
    common_sources: Mapping[str, Any] | None = None
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(
            validations / f"shard_{start:03d}_{stop:03d}.validation.json"
        )
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("shard_ordinal") == shard_ordinal
            and shard.get("execution_start") == start
            and shard.get("execution_stop") == stop
            and shard.get("logical_sha256") == logical_sha256(shard),
            "episode producer shard drift",
        )
        require_shard_source_authority(
            shard, expected_v118_sha256, shard_ordinal
        )
        require(
            validation.get("schema_version") == SHARD_VALIDATION_SCHEMA
            and validation.get("status") == SHARD_VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("shard_ordinal") == shard_ordinal
            and validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and validation.get("producer_shard_logical_sha256")
            == shard["logical_sha256"]
            and validation.get("episode_count") == shard["episode_count"],
            "episode shard independent validation drift",
        )
        if common_sources is None:
            common_sources = shard["source_bindings"]
        else:
            require(shard["source_bindings"] == common_sources, "episode shard source binding drift")
        for episode in shard["episodes"]:
            validate_compact_episode(episode)
            outer_fold = int(episode["query"]["outer_fold"])
            execution = int(episode["query"]["execution_ordinal"])
            by_outer[outer_fold].append(episode)
            query_occurrence.setdefault(execution, set()).add(outer_fold)
        total_eligible += int(shard["eligible_query_count"])
        total_excluded += int(shard["excluded_query_count"])
        total_episodes += int(shard["episode_count"])
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
    require(total_eligible == VALID_QUERY_COUNT and total_excluded == 6, "594/6 query closure drift")
    require(total_episodes == VALID_QUERY_COUNT * 3, "three-outer-fold episode multiplicity drift")
    require(
        len(query_occurrence) == VALID_QUERY_COUNT
        and all(len(folds) == 3 for folds in query_occurrence.values()),
        "query three-fold occurrence closure drift",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent)).resolve()
    try:
        ledger_receipts: list[dict[str, Any]] = []
        runtime_address_map_receipts: list[dict[str, Any]] = []
        total_source_episode_id_collisions = 0
        for outer_fold in SOURCE_FOLDS:
            source_episodes = sorted(
                by_outer[outer_fold],
                key=lambda item: int(item["query"]["execution_ordinal"]),
            )
            require(len(source_episodes) == EXPECTED_PER_OUTER[outer_fold], f"outer fold{outer_fold} episode count drift")
            source_collision_count = len(source_episodes) - len(
                {item["episode_id"] for item in source_episodes}
            )
            total_source_episode_id_collisions += source_collision_count
            episodes: list[dict[str, Any]] = []
            address_map: list[dict[str, Any]] = []
            for source_episode in source_episodes:
                runtime_episode, address = normalize_ledger_runtime_episode_address(
                    source_episode
                )
                episodes.append(runtime_episode)
                address_map.append(address)
            require(
                len({item["episode_id"] for item in episodes}) == len(episodes),
                f"outer fold{outer_fold} runtime episode ID collision",
            )
            require(
                [item["execution_ordinal"] for item in address_map]
                == [
                    int(item["query"]["execution_ordinal"])
                    for item in source_episodes
                ]
                and all(
                    item["outer_fold"] == outer_fold for item in address_map
                ),
                f"outer fold{outer_fold} runtime address map order drift",
            )
            address_map_sha256 = canonical_sha256(address_map)
            ledger: dict[str, Any] = {
                "schema_version": LEDGER_SCHEMA,
                "status": LEDGER_STATUS,
                "claim_level": "ENGINEERING_COMPACT_INNER_OOF_RELATIVE_LOSS_EPISODE_LEDGER_ONLY",
                "outer_fold": outer_fold,
                "input_role": "INNER_OOF_P_V2_LOCKS_ONLY",
                "episode_count": len(episodes),
                "outer_heldout_record_count": 0,
                "episodes": episodes,
                "runtime_episode_id_namespace": RUNTIME_EPISODE_ID_NAMESPACE,
                "source_to_runtime_address_map": address_map,
                "source_to_runtime_address_map_sha256": address_map_sha256,
                "source_episode_id_collision_count": source_collision_count,
                "runtime_episode_id_collision_count": 0,
                "episode_id_sequence_sha256": canonical_sha256(
                    [item["episode_id"] for item in episodes]
                ),
                "episode_record_sequence_sha256": canonical_sha256(
                    [item["record_sha256"] for item in episodes]
                ),
                "source_bindings": dict(common_sources or {}),
                "source_shard_receipts_sha256": canonical_sha256(shard_receipts),
                "access_audit": {
                    "dino_token_serialized_count": 0,
                    "model_load_count": 0,
                    "model_forward_count": 0,
                    "training_count": 0,
                    "score_rank_winner_gap_outcome_read_count": 0,
                    "identity_supergroup_read_count": 0,
                    "donor_null_h0_hold_switch_read_count": 0,
                    "protected_access_count": 0,
                },
                "training_authorized": False,
                "scientific_GO_or_NO_GO": None,
                "automatic_stage_advance": False,
                "next_authorized_stage": None,
            }
            ledger["logical_sha256"] = logical_sha256(ledger)
            path = staging / f"outer_fold{outer_fold}.episodes.pt"
            atomic_torch(path, ledger)
            ledger_receipts.append(
                {
                    "outer_fold": outer_fold,
                    "path": f"outer_fold{outer_fold}.episodes.pt",
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "logical_sha256": ledger["logical_sha256"],
                    "episode_count": len(episodes),
                    "source_to_runtime_address_map_sha256": address_map_sha256,
                    "source_episode_id_collision_count": source_collision_count,
                    "runtime_episode_id_collision_count": 0,
                }
            )
            runtime_address_map_receipts.append(
                {
                    "outer_fold": outer_fold,
                    "address_count": len(address_map),
                    "source_to_runtime_address_map_sha256": address_map_sha256,
                    "source_episode_id_collision_count": source_collision_count,
                    "runtime_episode_id_collision_count": 0,
                }
            )
        require(
            total_source_episode_id_collisions
            == authority["runtime_episode_address_repair"][
                "source_episode_id_collision_count"
            ]
            == 9,
            "full-source exact nine-collision diagnostic drift",
        )
        index: dict[str, Any] = {
            "schema_version": INDEX_SCHEMA,
            "status": INDEX_STATUS,
            "claim_level": "ENGINEERING_FOUR_COMPACT_INNER_OOF_EPISODE_LEDGERS_ONLY",
            "authority_sha256": authority_sha,
            "query_count": VALID_QUERY_COUNT,
            "excluded_query_count": 6,
            "episode_multiplicity_per_query": 3,
            "total_episode_count": total_episodes,
            "expected_per_outer_fold": {
                str(key): value for key, value in EXPECTED_PER_OUTER.items()
            },
            "ledgers": ledger_receipts,
            "runtime_episode_id_namespace": RUNTIME_EPISODE_ID_NAMESPACE,
            "runtime_address_map_receipts": runtime_address_map_receipts,
            "runtime_address_map_receipts_sha256": canonical_sha256(
                runtime_address_map_receipts
            ),
            "source_episode_id_collision_count": total_source_episode_id_collisions,
            "runtime_episode_id_collision_count": 0,
            "shard_receipts": shard_receipts,
            "ordered_shards_sha256": canonical_sha256(shard_receipts),
            "source_bindings": dict(common_sources or {}),
            "training_authorized": False,
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": None,
            "next_eligible_contract": "TRACK_R_RELATIVE_V_FIT",
        }
        index["logical_sha256"] = logical_sha256(index)
        atomic_json(staging / "index.json", index)
        os.replace(staging, output)
        return index
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(
        args.authority, args.producer_root, args.validation_root, args.output_root
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
