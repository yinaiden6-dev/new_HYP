#!/usr/bin/env python3
"""Seal completion of all 50 independently validated V120 prejoin shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_v121_authority,
    require_shard_source_authority,
    validate_runtime_bindings,
)

AUTHORITY = "registry/current_authority_v121_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v121_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists(), "immutable V120 completion exists")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority_path = args.authority.resolve()
    req(authority_path == (ROOT / AUTHORITY).resolve(), "V120 authority path drift")
    authority, authority_sha256 = read_v121_authority(
        authority_path,
        root=ROOT,
        expected_relative=AUTHORITY,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    validate_runtime_bindings(
        authority,
        root=ROOT,
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
    producer_root = args.producer_root.resolve()
    validation_root = args.validation_root.resolve()
    req(producer_root == (ROOT / authority["producer_root"]).resolve(), "producer root drift")
    req(validation_root == (ROOT / authority["validation_root"]).resolve(), "validation root drift")
    receipts = []
    executions = set()
    total_queries = total_excluded = 0
    for ordinal in range(50):
        start = ordinal * 12
        stop = start + 12
        producer = producer_root / f"shard_{start:03d}_{stop:03d}.pt"
        validation_path = validation_root / f"shard_{start:03d}_{stop:03d}.validation.json"
        req(producer.is_file() and validation_path.is_file(), f"V120 shard{ordinal} absent")
        shard = torch.load(producer, map_location="cpu", weights_only=True, mmap=True)
        validation = json.loads(validation_path.read_text())
        req(
            shard.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_READY"
            and shard.get("shard_ordinal") == ordinal
            and shard.get("logical_sha256") == logical(shard)
            and validation.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_INDEPENDENT_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and validation.get("logical_sha256") == logical(validation)
            and validation.get("producer_shard_sha256") == fsha(producer)
            and validation.get("producer_shard_logical_sha256") == shard["logical_sha256"]
            and validation.get("source_v121_authority_sha256")
            == authority_sha256,
            f"V120 shard{ordinal} validation drift",
        )
        require_shard_source_authority(shard, authority_sha256, ordinal)
        req(
            shard.get("source_bindings", {}).get(
                "parent_v120_authority_sha256"
            )
            == validation.get("parent_v120_authority_sha256")
            == authority.get("parent_v120_authority_sha256"),
            f"V120 shard{ordinal} parent-authority lineage drift",
        )
        req(
            shard.get("source_bindings", {}).get(
                "parent_v120_authority_logical_sha256"
            )
            == validation.get("parent_v120_authority_logical_sha256")
            == authority.get("parent_v120_authority_logical_sha256"),
            f"V120 shard{ordinal} parent-authority logical lineage drift",
        )
        for row in shard["rows"]:
            execution = int(row["execution_ordinal"])
            req(execution not in executions, "V120 query duplicate")
            executions.add(execution)
        total_queries += int(shard["query_count"])
        total_excluded += int(shard["excluded_query_count"])
        receipts.append({
            "shard_ordinal": ordinal,
            "producer_sha256": fsha(producer),
            "producer_logical_sha256": shard["logical_sha256"],
            "validation_sha256": fsha(validation_path),
            "validation_logical_sha256": validation["logical_sha256"],
        })
    req(total_queries == 594 and total_excluded == 6 and len(executions) == 594, "V120 594/6 closure drift")
    result = {
        "schema_version": "rc_dino_rcde_track_r_oof_prejoin_completion_v120_20260821",
        "status": "DINO_RCDE_TRACK_R_OOF_PREJOIN_V120_COMPLETE",
        "claim_level": "TARGET_FREE_OOF_PREJOIN_COMPLETE_NOT_SCIENTIFIC_GO_OR_NO_GO",
        "query_count": 594,
        "excluded_query_count": 6,
        "shard_count": 50,
        "source_v121_authority_sha256": authority_sha256,
        "parent_v120_authority_sha256": authority[
            "parent_v120_authority_sha256"
        ],
        "parent_v120_authority_logical_sha256": authority[
            "parent_v120_authority_logical_sha256"
        ],
        "all_shard_source_authority_match": True,
        "ordered_shard_receipts": receipts,
        "ordered_shard_receipts_sha256": canonical_sha256(receipts),
        "label_target_rival_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "next_eligible_contract": "V121_P_V2_UTILITY_COMPARATOR",
    }
    result["logical_sha256"] = logical(result)
    atomic(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "logical_sha256": result["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
