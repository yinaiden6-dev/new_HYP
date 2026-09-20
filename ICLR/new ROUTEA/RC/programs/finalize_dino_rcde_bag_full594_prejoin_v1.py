#!/usr/bin/env python3
"""Seal the 594-query BAG prejoin aggregate and target-free controls."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_bag_completion_v1 import (  # noqa: E402
    CANDIDATE_COUNT,
    C_BIND_NAMESPACE,
    EXCLUDED_EXECUTIONS,
    QUERY_COUNT,
    REORDER_NAMESPACE,
    SHARD_COUNT,
    SHARD_SIZE,
    VALID_QUERY_COUNT,
    canonical_sha256,
    deterministic_candidate_reorder,
    deterministic_identity_derangement,
    permuted_delta,
    tensor_sha256,
)
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402
from dino_rcde_bag_completion_io_v1 import (  # noqa: E402
    atomic_json,
    atomic_torch,
    binding_path,
    external_binding_path,
    file_sha256,
    read_authority,
    read_json,
    require,
    safe_path,
)
from materialize_dino_rcde_bag_full594_shard_v1 import SCHEMA as SHARD_SCHEMA, STATUS as SHARD_STATUS  # noqa: E402
from validate_dino_rcde_bag_full594_shard_v1 import (  # noqa: E402
    VALIDATION_SCHEMA,
    VALIDATION_STATUS,
)


CONTROL_SCHEMA = "rc_dino_rcde_bag_full594_target_free_controls_v1_20260827"
CONTROL_STATUS = "DINO_RCDE_BAG_FULL594_TARGET_FREE_CONTROLS_SEALED"
MANIFEST_SCHEMA = "rc_dino_rcde_bag_full594_prejoin_aggregate_v1_20260827"
MANIFEST_STATUS = "DINO_RCDE_BAG_FULL594_PREJOIN_AGGREGATE_PASS"


def _identity_labels(gallery_path: Path) -> tuple[str, ...]:
    gallery = torch.load(gallery_path, map_location="cpu", weights_only=True, mmap=True)
    legacy = gallery.get("setids") if isinstance(gallery, Mapping) else None
    require(isinstance(legacy, (list, tuple)) and len(legacy) == 5413, "gallery identity axis drift")
    return build_identity_map(tuple(map(str, legacy))).labels


def finalize(
    authority_path: Path,
    producer_root: Path,
    validation_root: Path,
    controls_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    authority, authority_sha = read_authority(authority_path)
    require(
        authority.get("target_free_control_materialization_authorized") is True
        and authority.get("target_rival_join_before_prejoin_aggregate_authorized") is False
        and authority.get("scientific_reduction_before_label_join_authorized") is False,
        "target-free finalizer authority drift",
    )
    require(binding_path(authority, "prejoin_finalizer") == Path(__file__).resolve(), "finalizer runtime binding drift")
    producer_root = safe_path(producer_root, file=False)
    validation_root = safe_path(validation_root, file=False)
    controls_path = safe_path(controls_path, must_exist=False)
    manifest_path = safe_path(manifest_path, must_exist=False)
    require(not controls_path.exists() and not manifest_path.exists(), "prejoin aggregate output already exists")
    identities = _identity_labels(external_binding_path(authority, "gallery_cache"))
    controls: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    executions: list[int] = []
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(producer_root / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(validation_root / f"shard_{start:03d}_{stop:03d}.validation.json")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("shard_ordinal") == shard_ordinal
            and validation.get("schema_version") == VALIDATION_SCHEMA
            and validation.get("status") == VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and validation.get("producer_shard_logical_sha256") == shard.get("logical_sha256")
            and validation.get("authority_sha256") == authority_sha,
            f"validated shard drift: {shard_ordinal}",
        )
        for record in shard["records"]:
            execution = int(record["execution_ordinal"])
            rows = tuple(map(int, record["candidate_physical_rows"]))
            require(len(rows) == CANDIDATE_COUNT and rows == tuple(sorted(rows)), "control candidate axis drift")
            corrected = tuple(identities[row] for row in rows)
            c_bind_order = deterministic_identity_derangement(
                rows, corrected, query_id=str(record["query_id"])
            )
            reorder_order = deterministic_candidate_reorder(
                rows, query_id=str(record["query_id"])
            )
            delta = record["delta"].detach().cpu().to(torch.float32).contiguous()
            c_bind_delta = permuted_delta(delta, c_bind_order)
            reorder_delta = permuted_delta(delta, reorder_order)
            inverse = [0] * CANDIDATE_COUNT
            for new_position, source_position in enumerate(reorder_order):
                inverse[source_position] = new_position
            require(
                torch.equal(permuted_delta(reorder_delta, inverse), delta),
                "candidate reorder inverse closure drift",
            )
            row: dict[str, Any] = {
                "query_id": str(record["query_id"]),
                "query_source_image_sha256": str(record["source_image_sha256"]),
                "execution_ordinal": execution,
                "outer_fold": int(record["heldout_fold"]),
                "candidate_physical_rows": list(rows),
                "candidate_axis_sha256": str(record["candidate_axis_sha256"]),
                "source_record_logical_sha256": str(record["logical_sha256"]),
                "source_delta_sha256": tensor_sha256(delta),
                "C_BIND": {
                    "namespace": C_BIND_NAMESPACE,
                    "destination_to_source_positions": list(c_bind_order),
                    "destination_to_source_physical_rows": [rows[position] for position in c_bind_order],
                    "mapping_sha256": canonical_sha256(list(c_bind_order)),
                    "identity_disjoint": all(corrected[index] != corrected[source] for index, source in enumerate(c_bind_order)),
                    "fixed_point_count": sum(index == source for index, source in enumerate(c_bind_order)),
                    "delta": c_bind_delta,
                    "delta_sha256": tensor_sha256(c_bind_delta),
                },
                "CANDIDATE_REORDER": {
                    "namespace": REORDER_NAMESPACE,
                    "source_positions": list(reorder_order),
                    "inverse_positions": inverse,
                    "mapping_sha256": canonical_sha256(list(reorder_order)),
                    "delta_sha256": tensor_sha256(reorder_delta),
                    "inverse_exact": True,
                },
            }
            row["logical_sha256"] = canonical_sha256(
                {
                    "query_id": row["query_id"],
                    "query_source_image_sha256": row["query_source_image_sha256"],
                    "execution_ordinal": execution,
                    "outer_fold": row["outer_fold"],
                    "candidate_physical_rows": row["candidate_physical_rows"],
                    "candidate_axis_sha256": row["candidate_axis_sha256"],
                    "source_record_logical_sha256": row["source_record_logical_sha256"],
                    "source_delta_sha256": row["source_delta_sha256"],
                    "C_BIND": {key: value for key, value in row["C_BIND"].items() if key != "delta"},
                    "CANDIDATE_REORDER": row["CANDIDATE_REORDER"],
                }
            )
            controls.append(row)
            executions.append(execution)
        shard_receipts.append(
            {
                "shard_ordinal": shard_ordinal,
                "execution_start": start,
                "execution_stop": stop,
                "query_count": int(shard["query_count"]),
                "producer_sha256": file_sha256(shard_path),
                "producer_logical_sha256": str(shard["logical_sha256"]),
                "validation_sha256": file_sha256(validation_path),
                "validation_logical_sha256": str(validation["logical_sha256"]),
            }
        )
        del shard
        gc.collect()
    controls.sort(key=lambda row: int(row["execution_ordinal"]))
    expected = sorted(set(range(QUERY_COUNT)) - set(EXCLUDED_EXECUTIONS))
    require(executions == expected and len(controls) == VALID_QUERY_COUNT, "594-query prejoin aggregate scope drift")
    control_hashes = [str(row["logical_sha256"]) for row in controls]
    control_payload: dict[str, Any] = {
        "schema_version": CONTROL_SCHEMA,
        "status": CONTROL_STATUS,
        "claim_level": "TARGET_FREE_FULL_C128_CONTROLS_ONLY",
        "query_count": VALID_QUERY_COUNT,
        "candidate_count_per_query": CANDIDATE_COUNT,
        "excluded_execution_ordinals": list(EXCLUDED_EXECUTIONS),
        "rows": controls,
        "row_logical_sha256s": control_hashes,
        "row_sequence_sha256": canonical_sha256(control_hashes),
        "authority_sha256": authority_sha,
        "target_rival_read_count": 0,
        "label_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
    }
    atomic_torch(controls_path, control_payload)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "claim_level": "TARGET_FREE_FULL_C128_BAG_PREJOIN_AND_CONTROLS_COMPLETE",
        "query_count": VALID_QUERY_COUNT,
        "candidate_count_per_query": CANDIDATE_COUNT,
        "shard_count": SHARD_COUNT,
        "excluded_execution_ordinals": list(EXCLUDED_EXECUTIONS),
        "execution_ordinals_sha256": canonical_sha256(expected),
        "shard_sequence_sha256": canonical_sha256(
            [item for receipt in shard_receipts for item in [receipt["producer_logical_sha256"]]]
        ),
        "control_ledger": {
            "path": str(controls_path.relative_to(RC_ROOT)),
            "sha256": file_sha256(controls_path),
            "row_sequence_sha256": control_payload["row_sequence_sha256"],
        },
        "shards": shard_receipts,
        "all_shard_validations_pass": True,
        "candidate_reorder_gate": True,
        "c_bind_complete_c128_identity_disjoint": True,
        "authority_sha256": authority_sha,
        "target_rival_read_count": 0,
        "label_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_stage": "BAG_FULL594_LABEL_JOIN_CONDITIONALLY_AUTHORIZED",
        "automatic_stage_advance": False,
    }
    manifest["logical_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.authority, args.producer_root, args.validation_root, args.controls, args.manifest)
    print({"status": result["status"], "query_count": result["query_count"], "manifest": str(args.manifest)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
