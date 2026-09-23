#!/usr/bin/env python3
"""Validate one target-free P-V2 C128 utility comparator shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_p_v2_utility_runtime_v1 import (  # noqa: E402
    source_bindings as runtime_source_bindings,
    validate_authority as validate_v122_authority,
)
from materialize_dino_rcde_track_r_p_v2_utility_shard_v1 import (  # noqa: E402
    materialize as replay_materialize,
)


AUTHORITY_PATH = "registry/current_authority_v122_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v122_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
OUTPUT_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_shard_v1_20260821"
OUTPUT_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_READY"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_shard_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_INDEPENDENT_VALIDATION_PASS"
SHARD_SIZE = 12
QUERY_COUNT = 600
EXCLUDED = (25, 26, 101, 346, 354, 470)


class TrackRPUtilityValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRPUtilityValidationError(message)


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
        not {"c8", "s8", "opened", "sealed"}.intersection(part.lower() for part in value.parts),
        f"protected path requested: {value}",
    )
    if must_exist:
        require(value.is_file() and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object absent")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), "immutable validation exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, output)
    output.chmod(0o444)


def validate(authority_path: Path, shard_path: Path, shard_ordinal: int, output: Path) -> dict[str, Any]:
    authority_path = safe_path(authority_path)
    authority, authority_sha = validate_v122_authority(
        authority_path,
        required_flag="p_v2_utility_validation_authorized",
        runtime_binding="validator",
        runtime_path="programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    )
    shard_path = safe_path(shard_path)
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    require(
        shard_path
        == (RC_ROOT / str(authority["producer_root"]) / f"shard_{start:03d}_{stop:03d}.pt").resolve()
        and output.resolve()
        == (RC_ROOT / str(authority["validation_root"]) / f"shard_{start:03d}_{stop:03d}.validation.json").resolve(),
        "V122 validator canonical path drift",
    )
    value = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version") == OUTPUT_SCHEMA
        and value.get("status") == OUTPUT_STATUS
        and value.get("shard_ordinal") == shard_ordinal
        and value.get("execution_start") == start
        and value.get("execution_stop") == stop
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("target_rival_read_count") == 0
        and value.get("dino_token_read_count") == 0
        and value.get("rank_winner_gap_outcome_read_count") == 0
        and value.get("scientific_GO_or_NO_GO") is None,
        "P utility shard envelope drift",
    )
    require(
        value.get("source_bindings")
        == runtime_source_bindings(authority, authority_sha),
        "P utility shard source lineage drift",
    )
    expected = [execution for execution in range(start, stop) if execution not in EXCLUDED]
    require([row["execution_ordinal"] for row in value["rows"]] == expected, "P utility query order drift")
    for row in value["rows"]:
        candidates = row.get("candidates")
        require(
            row.get("target_free") is True
            and isinstance(candidates, list)
            and len(candidates) == 128
            and [item["candidate_position"] for item in candidates] == list(range(128))
            and len({item["candidate_key"] for item in candidates}) == 128
            and len({item["candidate_physical_row"] for item in candidates}) == 128
            and row.get("candidate_axis_sha256")
            == canonical_sha256([item["candidate_physical_row"] for item in candidates]),
            "P utility C128 axis drift",
        )
        for item in candidates:
            directions = item.get("direction_utility")
            require(
                isinstance(directions, Mapping)
                and set(directions) == {"a_to_b", "b_to_a"}
                and all(math.isfinite(float(number)) for number in directions.values())
                and math.isfinite(float(item["utility"]))
                and abs(
                    float(item["utility"])
                    - 0.5 * (float(directions["a_to_b"]) + float(directions["b_to_a"]))
                )
                <= 1.0e-12,
                "P utility direction fusion drift",
            )
        require(row["record_sha256"] == logical_sha256({key: item for key, item in row.items() if key != "record_sha256"}), "P utility row hash drift")
    require(
        value.get("model_forward_count") == 256 * len(value["rows"]),
        "P utility model-forward closure drift",
    )
    replayed = replay_materialize(
        authority_path,
        shard_ordinal,
        RC_ROOT / str(authority["producer_root"]),
        replay_only=True,
    )
    require(replayed == value, "independent full P-utility replay drift")
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "producer_shard_sha256": file_sha256(shard_path),
        "producer_shard_logical_sha256": value["logical_sha256"],
        "query_count": len(value["rows"]),
        "candidate_utility_count": 128 * len(value["rows"]),
        "independent_full_replay": True,
        "independent_model_forward_count": replayed["model_forward_count"],
        "source_bindings": runtime_source_bindings(authority, authority_sha),
        "target_rival_read_count": 0,
        "dino_token_read_count": 0,
        "scientific_reduction_count": 0,
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
    value = validate(args.authority, args.shard, args.shard_ordinal, args.output)
    print(json.dumps({"status": value["status"], "shard_ordinal": value["shard_ordinal"], "query_count": value["query_count"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
