#!/usr/bin/env python3
"""Independent raw-input replay of the BAG full-594 scientific result."""

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
    CANDIDATE_COUNT,
    EXCLUDED_EXECUTIONS,
    PERMUTATIONS,
    QUERY_COUNT,
    SEED,
    SHARD_COUNT,
    SHARD_SIZE,
    VALID_QUERY_COUNT,
    canonical_sha256,
    deterministic_identity_derangement,
    permuted_delta,
    tensor_sha256,
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
from dino_rcde_bag_statistics_independent_v1 import independent_evaluate_bag_completion  # noqa: E402
from finalize_dino_rcde_bag_full594_prejoin_v1 import CONTROL_SCHEMA, CONTROL_STATUS, MANIFEST_SCHEMA, MANIFEST_STATUS  # noqa: E402
from join_dino_rcde_bag_full594_labels_v1 import SCHEMA as JOIN_SCHEMA, STATUS as JOIN_STATUS  # noqa: E402
from reduce_dino_rcde_bag_full594_science_v1 import SCHEMA as RESULT_SCHEMA, STATUS as RESULT_STATUS  # noqa: E402


SCHEMA = "rc_dino_rcde_bag_full594_scientific_validation_v1_20260827"
STATUS = "DINO_RCDE_BAG_FULL594_INDEPENDENT_SCIENTIFIC_VALIDATION_PASS"


def _bits(value: object) -> float:
    require(isinstance(value, str) and len(value) == 16, "independent RAW bits drift")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(torch.isfinite(torch.tensor(result)), "independent nonfinite RAW score")
    return float(result)


def validate(
    authority_path: Path,
    producer_root: Path,
    controls_path: Path,
    manifest_path: Path,
    join_path: Path,
    result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authority, authority_sha = read_authority(authority_path)
    require(binding_path(authority, "scientific_validator") == Path(__file__).resolve(), "scientific validator runtime binding drift")
    manifest = read_json(manifest_path)
    join = read_json(join_path)
    result = read_json(result_path)
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("status") == MANIFEST_STATUS
        and manifest.get("authority_sha256") == authority_sha
        and join.get("schema_version") == JOIN_SCHEMA
        and join.get("status") == JOIN_STATUS
        and result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == RESULT_STATUS,
        "independent predecessor envelope drift",
    )
    controls = torch.load(safe_path(controls_path), map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(controls, Mapping)
        and controls.get("schema_version") == CONTROL_SCHEMA
        and controls.get("status") == CONTROL_STATUS
        and manifest["control_ledger"]["sha256"] == file_sha256(controls_path),
        "independent control ledger drift",
    )
    controls_by_execution = {int(row["execution_ordinal"]): row for row in controls["rows"]}
    i0_path = binding_path(authority, "i0_science_metadata")
    i0 = read_json(i0_path)
    i0_by_execution = {int(row["execution_ordinal"]): row for row in i0["records"]}
    expected = set(range(QUERY_COUNT)) - set(EXCLUDED_EXECUTIONS)
    require(set(controls_by_execution) == set(i0_by_execution) == expected, "independent 594 population drift")
    replay_rows: list[dict[str, Any]] = []
    control_count = 0
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer_root / f"shard_{start:03d}_{stop:03d}.pt")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        for record in shard["records"]:
            execution = int(record["execution_ordinal"])
            control = controls_by_execution[execution]
            meta = i0_by_execution[execution]
            physical_rows = list(map(int, record["candidate_physical_rows"]))
            c_bind = control["C_BIND"]
            order = tuple(map(int, c_bind["destination_to_source_positions"]))
            require(
                len(order) == CANDIDATE_COUNT
                and sorted(order) == list(range(CANDIDATE_COUNT))
                and all(index != source for index, source in enumerate(order))
                and c_bind["mapping_sha256"] == canonical_sha256(list(order))
                and c_bind["fixed_point_count"] == 0
                and c_bind["identity_disjoint"] is True,
                "independent C_BIND mapping drift",
            )
            expected_control_delta = permuted_delta(record["delta"], order)
            require(
                torch.equal(expected_control_delta, c_bind["delta"])
                and c_bind["delta_sha256"] == tensor_sha256(expected_control_delta),
                "independent C_BIND score replay drift",
            )
            reorder = control["CANDIDATE_REORDER"]
            reorder_order = tuple(map(int, reorder["source_positions"]))
            inverse = tuple(map(int, reorder["inverse_positions"]))
            require(
                len(reorder_order) == CANDIDATE_COUNT
                and sorted(reorder_order) == list(range(CANDIDATE_COUNT))
                and len(inverse) == CANDIDATE_COUNT
                and sorted(inverse) == list(range(CANDIDATE_COUNT))
                and reorder["mapping_sha256"] == canonical_sha256(list(reorder_order))
                and reorder["inverse_exact"] is True,
                "independent candidate reorder mapping drift",
            )
            reordered_delta = permuted_delta(record["delta"], reorder_order)
            require(
                reorder["delta_sha256"] == tensor_sha256(reordered_delta)
                and torch.equal(permuted_delta(reordered_delta, inverse), record["delta"]),
                "independent candidate reorder inverse replay drift",
            )
            target = int(meta["target_physical_row"])
            rival = int(meta["rival_physical_row"])
            require(
                target in physical_rows
                and rival in physical_rows
                and record["query_id"] == control["query_id"] == meta["query_id"]
                and record["candidate_axis_sha256"] == control["candidate_axis_sha256"] == meta["candidate_axis_sha256"],
                "independent label/prejoin join drift",
            )
            ti, ri = physical_rows.index(target), physical_rows.index(rival)
            replay_rows.append(
                {
                    "query_id": str(record["query_id"]),
                    "execution_ordinal": execution,
                    "group_sha256": str(meta["group_sha256"]),
                    "outer_fold": int(meta["outer_fold"]),
                    "candidate_axis_sha256": str(record["candidate_axis_sha256"]),
                    "target_physical_row": target,
                    "rival_physical_row": rival,
                    "target_position": ti,
                    "rival_position": ri,
                    "raw_margin": _bits(meta["target_raw_score_bits"]) - _bits(meta["rival_raw_score_bits"]),
                    "real_margin": float(record["delta"][ti, ri]),
                    "c_bind_margin": float(expected_control_delta[ti, ri]),
                    "source_record_logical_sha256": str(record["logical_sha256"]),
                    "control_record_logical_sha256": str(control["logical_sha256"]),
                    "i0_record_sha256": str(meta["record_sha256"]),
                }
            )
            replay_rows[-1]["logical_sha256"] = canonical_sha256(replay_rows[-1])
            control_count += 1
    replay_rows.sort(key=lambda row: int(row["execution_ordinal"]))
    require(replay_rows == join["rows"], "independent label-join row replay drift")
    statistics = independent_evaluate_bag_completion(
        replay_rows, repetitions=PERMUTATIONS, seed=SEED
    )
    expected_decision = statistics.get("decision")
    if statistics.get("status") == "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE":
        expected_decision = "DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE"
    require(
        result.get("statistics") == statistics
        and result.get("decision") == expected_decision
        and result.get("scientific_GO_or_NO_GO") == statistics.get("scientific_GO_or_NO_GO")
        and result.get("candidate_reorder_gate") is True,
        "independent scientific reduction replay drift",
    )
    validation: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "validation_pass": True,
        "query_count": VALID_QUERY_COUNT,
        "control_replay_count": control_count,
        "candidate_reorder_replay_count": control_count,
        "label_join_row_replay_exact": True,
        "statistics_replay_exact": True,
        "decision": expected_decision,
        "scientific_GO_or_NO_GO": statistics.get("scientific_GO_or_NO_GO"),
        "source_bindings": {
            "authority_sha256": authority_sha,
            "prejoin_manifest_sha256": file_sha256(manifest_path),
            "control_ledger_sha256": file_sha256(controls_path),
            "label_join_sha256": file_sha256(join_path),
            "scientific_result_sha256": file_sha256(result_path),
            "i0_science_metadata_sha256": file_sha256(i0_path),
        },
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "protected_access_count": 0,
        "automatic_stage_advance": False,
    }
    validation["logical_sha256"] = canonical_sha256(validation)
    atomic_json(output_path, validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.authority, args.producer_root, args.controls, args.manifest, args.join, args.result, args.output)
    print({"status": result["status"], "decision": result["decision"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
