#!/usr/bin/env python3
"""Join frozen target/strongest-rival roles after BAG prejoin sealing."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_bag_completion_v1 import (  # noqa: E402
    EXCLUDED_EXECUTIONS,
    QUERY_COUNT,
    SHARD_COUNT,
    SHARD_SIZE,
    VALID_QUERY_COUNT,
    canonical_sha256,
)
from dino_rcde_bag_completion_io_v1 import (  # noqa: E402
    atomic_json,
    binding_path,
    file_sha256,
    read_authority,
    read_json,
    require,
    safe_path,
)
from finalize_dino_rcde_bag_full594_prejoin_v1 import (  # noqa: E402
    CONTROL_SCHEMA,
    CONTROL_STATUS,
    MANIFEST_SCHEMA,
    MANIFEST_STATUS,
)


SCHEMA = "rc_dino_rcde_bag_full594_label_join_v1_20260827"
STATUS = "DINO_RCDE_BAG_FULL594_TARGET_STRONGEST_RIVAL_LABEL_JOIN_READY"


def _bits(value: object) -> float:
    require(isinstance(value, str) and len(value) == 16, "RAW score bits drift")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(torch.isfinite(torch.tensor(result)), "nonfinite RAW score")
    return float(result)


def join(
    authority_path: Path,
    producer_root: Path,
    controls_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authority, authority_sha = read_authority(authority_path)
    require(
        authority.get("label_join_conditionally_authorized") is True
        and authority.get("model_forward_after_label_join_authorized") is False
        and authority.get("training_authorized") is False,
        "label-join authority drift",
    )
    require(binding_path(authority, "label_joiner") == Path(__file__).resolve(), "label joiner runtime binding drift")
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("status") == MANIFEST_STATUS
        and manifest.get("all_shard_validations_pass") is True
        and manifest.get("candidate_reorder_gate") is True
        and manifest.get("c_bind_complete_c128_identity_disjoint") is True
        and manifest.get("authority_sha256") == authority_sha
        and manifest.get("target_rival_read_count") == 0,
        "label join predecessor aggregate drift",
    )
    controls_path = safe_path(controls_path)
    require(
        manifest["control_ledger"]["sha256"] == file_sha256(controls_path),
        "control ledger/manifest hash drift",
    )
    controls = torch.load(controls_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(controls, Mapping)
        and controls.get("schema_version") == CONTROL_SCHEMA
        and controls.get("status") == CONTROL_STATUS
        and controls.get("query_count") == VALID_QUERY_COUNT
        and controls.get("target_rival_read_count") == 0,
        "control ledger envelope drift",
    )
    control_by_execution = {int(row["execution_ordinal"]): row for row in controls["rows"]}
    i0_path = binding_path(authority, "i0_science_metadata")
    i0 = read_json(i0_path)
    require(
        i0.get("status") == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
        and i0.get("query_count") == VALID_QUERY_COUNT
        and i0.get("excluded_execution_ordinals") == list(EXCLUDED_EXECUTIONS),
        "I0 label metadata seal drift",
    )
    i0_by_execution = {int(row["execution_ordinal"]): row for row in i0["records"]}
    expected_executions = set(range(QUERY_COUNT)) - set(EXCLUDED_EXECUTIONS)
    require(set(i0_by_execution) == set(control_by_execution) == expected_executions, "594-query label/control population drift")
    rows: list[dict[str, Any]] = []
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer_root / f"shard_{start:03d}_{stop:03d}.pt")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        for record in shard["records"]:
            execution = int(record["execution_ordinal"])
            meta = i0_by_execution[execution]
            control = control_by_execution[execution]
            candidate_rows = list(map(int, record["candidate_physical_rows"]))
            target_row = int(meta["target_physical_row"])
            rival_row = int(meta["rival_physical_row"])
            require(
                target_row in candidate_rows
                and rival_row in candidate_rows
                and target_row != rival_row
                and record["query_id"] == meta["query_id"] == control["query_id"]
                and record["source_image_sha256"] == meta["query_source_image_sha256"] == control["query_source_image_sha256"]
                and int(record["heldout_fold"]) == int(meta["outer_fold"]) == int(control["outer_fold"])
                and record["candidate_axis_sha256"] == meta["candidate_axis_sha256"] == control["candidate_axis_sha256"]
                and record["logical_sha256"] == control["source_record_logical_sha256"],
                "postjoin query/role/control binding drift",
            )
            target_position = candidate_rows.index(target_row)
            rival_position = candidate_rows.index(rival_row)
            real_margin = float(record["delta"][target_position, rival_position])
            c_bind_margin = float(control["C_BIND"]["delta"][target_position, rival_position])
            raw_margin = _bits(meta["target_raw_score_bits"]) - _bits(meta["rival_raw_score_bits"])
            row: dict[str, Any] = {
                "query_id": str(record["query_id"]),
                "execution_ordinal": execution,
                "group_sha256": str(meta["group_sha256"]),
                "outer_fold": int(meta["outer_fold"]),
                "candidate_axis_sha256": str(record["candidate_axis_sha256"]),
                "target_physical_row": target_row,
                "rival_physical_row": rival_row,
                "target_position": target_position,
                "rival_position": rival_position,
                "raw_margin": raw_margin,
                "real_margin": real_margin,
                "c_bind_margin": c_bind_margin,
                "source_record_logical_sha256": str(record["logical_sha256"]),
                "control_record_logical_sha256": str(control["logical_sha256"]),
                "i0_record_sha256": str(meta["record_sha256"]),
            }
            row["logical_sha256"] = canonical_sha256(row)
            rows.append(row)
    rows.sort(key=lambda row: int(row["execution_ordinal"]))
    require(len(rows) == VALID_QUERY_COUNT and len({row["execution_ordinal"] for row in rows}) == VALID_QUERY_COUNT, "postjoin row population drift")
    hashes = [str(row["logical_sha256"]) for row in rows]
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "LABEL_JOIN_ONLY_NO_SCIENTIFIC_DECISION",
        "query_count": VALID_QUERY_COUNT,
        "excluded_execution_ordinals": list(EXCLUDED_EXECUTIONS),
        "rows": rows,
        "row_logical_sha256s": hashes,
        "row_sequence_sha256": canonical_sha256(hashes),
        "source_bindings": {
            "authority_sha256": authority_sha,
            "prejoin_manifest_sha256": file_sha256(manifest_path),
            "control_ledger_sha256": file_sha256(controls_path),
            "i0_science_metadata_sha256": file_sha256(i0_path),
        },
        "target_rival_read_count": VALID_QUERY_COUNT,
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
    }
    output["logical_sha256"] = canonical_sha256(output)
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = join(args.authority, args.producer_root, args.controls, args.manifest, args.output)
    print({"status": result["status"], "query_count": result["query_count"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
