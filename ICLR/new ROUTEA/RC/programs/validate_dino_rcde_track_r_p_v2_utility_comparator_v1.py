#!/usr/bin/env python3
"""Independently reconstruct and validate the V122 aggregate comparator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_p_v2_utility_runtime_v1 import (  # noqa: E402
    fsha,
    source_bindings,
    validate_authority,
)


STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS"
EXCLUDED = (25, 26, 101, 346, 354, 470)


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def logical(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists(), "immutable V122 aggregate validation exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)
    path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority, authority_sha = validate_authority(
        args.authority,
        required_flag="p_v2_utility_aggregate_validation_authorized",
        runtime_binding="aggregate_validator",
        runtime_path="programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py",
    )
    producer = args.producer_root.resolve(strict=True)
    validations = args.validation_root.resolve(strict=True)
    req(producer == (ROOT / authority["producer_root"]).resolve(), "producer root drift")
    req(validations == (ROOT / authority["validation_root"]).resolve(), "validation root drift")
    comparator_path = args.comparator.resolve(strict=True)
    req(comparator_path == (ROOT / authority["comparator_output"]).resolve(), "comparator path drift")
    comparator = json.loads(comparator_path.read_text())
    records = []
    exclusions = []
    for ordinal in range(50):
        start, stop = ordinal * 12, ordinal * 12 + 12
        shard_path = producer / f"shard_{start:03d}_{stop:03d}.pt"
        validation_path = validations / f"shard_{start:03d}_{stop:03d}.validation.json"
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = json.loads(validation_path.read_text())
        req(
            shard.get("logical_sha256") == logical(shard)
            and shard.get("source_bindings") == source_bindings(authority, authority_sha)
            and validation.get("logical_sha256") == logical(validation)
            and validation.get("source_bindings") == shard["source_bindings"]
            and validation.get("independent_full_replay") is True,
            f"validated V122 shard drift: {ordinal}",
        )
        records.extend(shard["rows"])
        exclusions.extend(shard["exclusions"])
    records.sort(key=lambda item: int(item["execution_ordinal"]))
    req(len(records) == 594 and [item["execution_ordinal"] for item in exclusions] == list(EXCLUDED), "V122 aggregate population drift")
    req(
        comparator.get("status") == "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY"
        and comparator.get("logical_sha256") == logical(comparator)
        and comparator.get("records") == records
        and comparator.get("record_sequence_sha256")
        == canonical_sha256([item["record_sha256"] for item in records])
        and comparator.get("source_bindings") == source_bindings(authority, authority_sha)
        and comparator.get("target_rival_read_count") == 0
        and comparator.get("dino_token_read_count") == 0
        and comparator.get("scientific_GO_or_NO_GO") is None,
        "V122 comparator independent reconstruction drift",
    )
    result = {
        "schema_version": "rc_dino_rcde_track_r_p_v2_utility_comparator_validation_v1_20260822",
        "status": STATUS,
        "validation_pass": True,
        "comparator_sha256": fsha(comparator_path),
        "comparator_logical_sha256": comparator["logical_sha256"],
        "query_count": 594,
        "candidate_utility_count": 594 * 128,
        "shard_count": 50,
        "source_bindings": source_bindings(authority, authority_sha),
        "target_rival_read_count": 0,
        "label_identity_supergroup_read_count": 0,
        "dino_token_read_count": 0,
        "scientific_reduction_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical(result)
    req(args.output.resolve() == (ROOT / authority["comparator_validation_output"]).resolve(), "aggregate validation output drift")
    atomic(args.output.resolve(), result)
    print(json.dumps({"status": STATUS, "logical_sha256": result["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
