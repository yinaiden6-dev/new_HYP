#!/usr/bin/env python3
"""Reduce 50 validated target-free P-V2 utility shards."""

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
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_p_v2_utility_runtime_v1 import (  # noqa: E402
    source_bindings as runtime_source_bindings,
    validate_authority as validate_v122_authority,
)


AUTHORITY_PATH = "registry/current_authority_v122_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v122_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
SHARD_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_shard_v1_20260821"
SHARD_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_READY"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_INDEPENDENT_VALIDATION_PASS"
OUTPUT_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_comparator_v1_20260821"
OUTPUT_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY"
SHARD_COUNT = 50
SHARD_SIZE = 12
QUERY_COUNT = 600
VALID_QUERY_COUNT = 594
EXCLUDED = (25, 26, 101, 346, 354, 470)


class TrackRPUtilityReductionError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRPUtilityReductionError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def safe_path(path: Path, *, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(not {"c8", "s8", "opened", "sealed"}.intersection(part.lower() for part in value.parts), f"protected path requested: {value}")
    require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object absent")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable comparator exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(0o444)


def reduce(authority_path: Path, producer_root: Path, validation_root: Path, output: Path) -> dict[str, Any]:
    authority_path = safe_path(authority_path)
    authority, authority_sha = validate_v122_authority(
        authority_path,
        required_flag="p_v2_utility_reduction_authorized",
        runtime_binding="reducer",
        runtime_path="programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py",
    )
    producer = safe_path(producer_root, file=False)
    validations = safe_path(validation_root, file=False)
    require(producer == (RC_ROOT / str(authority["producer_root"])).resolve(), "V122 reducer producer root drift")
    require(validations == (RC_ROOT / str(authority["validation_root"])).resolve(), "V122 reducer validation root drift")
    require(output.resolve() == (RC_ROOT / str(authority["comparator_output"])).resolve(), "V122 comparator output drift")
    records = []
    receipts = []
    exclusions = []
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(validations / f"shard_{start:03d}_{stop:03d}.validation.json")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("logical_sha256") == logical_sha256(shard)
            and validation.get("status") == VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and validation.get("producer_shard_logical_sha256") == shard["logical_sha256"],
            "validated P utility shard drift",
        )
        require(
            validation.get("logical_sha256") == logical_sha256(validation)
            and shard.get("shard_ordinal") == shard_ordinal
            and shard.get("execution_start") == start
            and shard.get("execution_stop") == stop
            and shard.get("source_bindings")
            == runtime_source_bindings(authority, authority_sha)
            and validation.get("source_bindings") == shard["source_bindings"]
            and validation.get("independent_full_replay") is True,
            "V122 validated shard lineage/range drift",
        )
        records.extend(shard["rows"])
        exclusions.extend(shard["exclusions"])
        receipts.append({
            "shard_ordinal": shard_ordinal,
            "producer_sha256": file_sha256(shard_path),
            "producer_logical_sha256": shard["logical_sha256"],
            "validation_sha256": file_sha256(validation_path),
            "validation_logical_sha256": validation["logical_sha256"],
        })
    records.sort(key=lambda item: int(item["execution_ordinal"]))
    require(len(records) == VALID_QUERY_COUNT and len({item["execution_ordinal"] for item in records}) == VALID_QUERY_COUNT, "P comparator 594-query closure drift")
    require([item["execution_ordinal"] for item in exclusions] == list(EXCLUDED), "P comparator exclusion closure drift")
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "claim_level": "TARGET_FREE_P_V2_OUTER_REFIT_C128_UTILITY_COMPARATOR_ONLY",
        "query_count": VALID_QUERY_COUNT,
        "candidate_count_per_query": 128,
        "excluded_execution_ordinals": list(EXCLUDED),
        "target_free_prejoin_score_ledger": True,
        "records": records,
        "record_sequence_sha256": canonical_sha256([item["record_sha256"] for item in records]),
        "shard_receipts": receipts,
        "ordered_shards_sha256": canonical_sha256(receipts),
        "target_rival_read_count": 0,
        "dino_token_read_count": 0,
        "rank_winner_gap_outcome_read_count": 0,
        "source_bindings": runtime_source_bindings(authority, authority_sha),
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
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = reduce(args.authority, args.producer_root, args.validation_root, args.output)
    print(json.dumps({"status": value["status"], "query_count": value["query_count"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
