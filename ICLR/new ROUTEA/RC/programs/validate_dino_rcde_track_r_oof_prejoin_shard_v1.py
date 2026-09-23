#!/usr/bin/env python3
"""Independent source/receipt validator for one Track-R OOF prejoin shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402

from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_v121_authority,
    require_shard_source_authority,
    validate_runtime_bindings,
)


AUTHORITY_PATH = "registry/current_authority_v121_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v121_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
OUTPUT_SCHEMA = "rc_dino_rcde_track_r_oof_prejoin_shard_v1_20260821"
OUTPUT_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_READY"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_oof_prejoin_shard_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_INDEPENDENT_VALIDATION_PASS"
SHARD_SIZE = 12
SHARD_COUNT = 50
QUERY_COUNT = 600
EXCLUDED = (25, 26, 101, 346, 354, 470)
ARMS = {
    "ALL_PATCH_SAME_MODEL",
    "CW1_QUERY_MULTITILE_FULL_REFERENCE",
    "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET",
}


class TrackROOFValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackROOFValidationError(message)


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


def safe_path(path: Path, *, must_exist: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require(value.is_file() and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping) and set(item) >= {"path", "sha256"}, f"binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item["sha256"], f"binding hash drift: {name}")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), "immutable validation exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    output.chmod(0o444)


def validate(
    authority_path: Path,
    shard_path: Path,
    shard_ordinal: int,
    output: Path,
) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority_path = safe_path(authority_path)
    authority, authority_sha256 = read_v121_authority(
        authority_path,
        root=RC_ROOT,
        expected_relative=AUTHORITY_PATH,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("oof_prejoin_shard_validation_authorized") is True
        and authority.get("label_target_rival_read_authorized") is False
        and authority.get("model_backward_authorized") is False
        and authority.get("model_update_authorized") is False
        and authority.get("scientific_reduction_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V121 validation authority drift",
    )
    validate_runtime_bindings(
        authority,
        root=RC_ROOT,
        required_names=(
            "producer",
            "validator",
            "finalizer",
            "lineage_runtime",
            "runtime_entry_validator",
            "package_initializer",
            "launcher",
            "post_vfit_controller",
            "post_oof_controller",
        ),
    )
    pair_outer = read_json(binding_path(authority, "role_free_pair_manifest"))
    folds = read_json(binding_path(authority, "fold_schedule"))
    pair_records = {
        int(item["execution_ordinal"]): item
        for item in pair_outer["pair_manifest"]["records"]
    }
    fold_sequence = tuple(folds["records"])
    require(
        len(fold_sequence) == QUERY_COUNT
        and len({int(item["query_ordinal"]) for item in fold_sequence}) == QUERY_COUNT,
        "independent OOF fold sequence drift",
    )
    fold_records = dict(enumerate(fold_sequence))
    require(
        set(pair_records) == set(range(QUERY_COUNT)) - set(EXCLUDED),
        "independent OOF pair execution-axis drift",
    )
    shard_path = safe_path(shard_path)
    value = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version") == OUTPUT_SCHEMA
        and value.get("status") == OUTPUT_STATUS
        and value.get("shard_ordinal") == shard_ordinal
        and value.get("execution_start") == start
        and value.get("execution_stop") == stop
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False,
        "OOF prejoin shard envelope drift",
    )
    require_shard_source_authority(value, authority_sha256, shard_ordinal)
    require(
        value.get("source_bindings", {}).get("parent_v120_authority_sha256")
        == authority.get("parent_v120_authority_sha256"),
        "OOF shard parent V120 lineage drift",
    )
    require(
        value.get("source_bindings", {}).get(
            "parent_v120_authority_logical_sha256"
        )
        == authority.get("parent_v120_authority_logical_sha256"),
        "OOF shard parent V120 logical lineage drift",
    )
    expected_excluded = [
        {"execution_ordinal": execution, "reason": "NATURAL_C128_TARGET_MISS"}
        for execution in range(start, stop)
        if execution in EXCLUDED
    ]
    require(value.get("exclusions") == expected_excluded, "prejoin exclusion drift")
    rows = value.get("rows")
    require(isinstance(rows, list), "prejoin rows absent")
    expected_executions = [
        execution for execution in range(start, stop) if execution not in EXCLUDED
    ]
    require(
        [int(item["execution_ordinal"]) for item in rows] == expected_executions
        and value.get("query_count") == len(rows)
        and value.get("excluded_query_count") == len(expected_excluded)
        and value.get("row_sequence_sha256")
        == canonical_sha256([item["record_sha256"] for item in rows]),
        "prejoin query population/order drift",
    )
    for row in rows:
        execution = int(row["execution_ordinal"])
        pair = pair_records[execution]
        fold = fold_records[execution]
        require(
            row["query_id"] == pair["query_id"] == fold["query_id"]
            and row["query_source_image_sha256"]
            == pair["query_source_image_sha256"]
            == fold["source_image_sha256"]
            and row["outer_fold"] == fold["inner_fold"]
            and row["pair_sha256"] == pair["pair_sha256"]
            and row["candidate_axis_sha256"] == pair["candidate_axis_sha256"]
            and row["member_addresses"] == pair["members"]
            and row["member_order"]
            == [member["candidate_key"] for member in pair["members"]],
            "prejoin row/pair/fold source drift",
        )
        require(
            row["record_sha256"]
            == logical_sha256(
                {key: item for key, item in row.items() if key != "record_sha256"}
            ),
            "prejoin row hash drift",
        )
        for name in ("real", "candidate_reorder", "pair_swap", "P_QUERY", "C_DINO_V"):
            evidence = row.get(name)
            require(isinstance(evidence, Mapping) and set(evidence.get("arms", {})) == ARMS, f"{name} three-arm drift")
        require(
            set(row.get("P_REFERENCE", {})) == ARMS
            and row.get("candidate_reorder_max_abs", 1.0) <= 1.0e-7
            and row.get("pair_swap_max_abs", 1.0) <= 1.0e-7
            and row.get("real_control_direction_inspection_count") == 0,
            "prejoin invariance/control boundary drift",
        )
    access = value.get("access_audit")
    require(
        isinstance(access, Mapping)
        and access.get("model_backward_count") == 0
        and access.get("model_update_count") == 0
        and access.get("label_target_rival_read_count") == 0
        and access.get("raw_score_rank_correctness_outcome_read_count") == 0
        and access.get("role_free_pair_read_count") == len(rows)
        and access.get("scientific_reduction_count") == 0
        and access.get("protected_access_count") == 0,
        "prejoin access audit drift",
    )
    result: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "producer_shard_sha256": file_sha256(shard_path),
        "producer_shard_logical_sha256": value["logical_sha256"],
        "source_v121_authority_sha256": authority_sha256,
        "parent_v120_authority_sha256": authority[
            "parent_v120_authority_sha256"
        ],
        "parent_v120_authority_logical_sha256": authority[
            "parent_v120_authority_logical_sha256"
        ],
        "query_count": len(rows),
        "excluded_query_count": len(expected_excluded),
        "three_arm_query_count": len(rows),
        "control_query_count": len(rows),
        "label_target_rival_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.authority, args.shard, args.shard_ordinal, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "shard_ordinal": result["shard_ordinal"],
                "query_count": result["query_count"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
