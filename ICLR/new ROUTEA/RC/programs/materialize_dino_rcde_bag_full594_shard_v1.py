#!/usr/bin/env python3
"""Materialize one target-free full-C128 BAG outer-OOF shard."""

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
    EXCLUDED_EXECUTIONS,
    FOLDS,
    QUERY_COUNT,
    SHARD_COUNT,
    SHARD_SIZE,
    VALID_QUERY_COUNT,
    canonical_sha256,
)
import materialize_dino_rcde_r1_oof_prejoin_v3 as legacy  # noqa: E402
from dino_rcde_bag_completion_io_v1 import (  # noqa: E402
    atomic_torch,
    binding_path,
    file_sha256,
    read_authority,
    read_json,
    require,
    safe_path,
)


SCHEMA = "rc_dino_rcde_bag_full594_prejoin_shard_v1_20260827"
STATUS = "DINO_RCDE_BAG_FULL594_PREJOIN_SHARD_READY"


def materialize(authority_path: Path, shard_ordinal: int, output_root: Path, *, device: str) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority, authority_sha = read_authority(authority_path)
    require(
        authority.get("model_load_authorized") is True
        and authority.get("model_forward_authorized") is True
        and authority.get("training_authorized") is False
        and authority.get("target_rival_join_before_prejoin_aggregate_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "prejoin authority flags drift",
    )
    require(binding_path(authority, "producer") == Path(__file__).resolve(), "producer runtime binding drift")
    schedule_path = binding_path(authority, "redacted_schedule")
    cache_index_path = binding_path(authority, "redacted_cache_index")
    foldset_path = binding_path(authority, "bag_foldset_manifest")
    foldset_validation_path = binding_path(authority, "bag_foldset_validation")
    foldset = read_json(foldset_path)
    foldset_validation = read_json(foldset_validation_path)
    require(
        foldset.get("arm") == "RCDE_BAG"
        and foldset.get("status") == "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_FROZEN"
        and foldset_validation.get("status") == "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS"
        and foldset_validation.get("heldout_forward_count") == 0
        and foldset_validation.get("heldout_label_join_count") == 0
        and foldset_validation.get("scientific_metric_count") == 0,
        "BAG foldset predecessor drift",
    )
    schedule, schedule_hashes = legacy.load_redacted_schedule(
        redacted_schedule=schedule_path,
        prejoin_folds=None,
        model_visible_c128=None,
    )
    schedule_reference_rows = {
        int(row)
        for schedule_row in schedule
        for row in schedule_row["candidate_physical_rows"]
    }
    schedule_json = read_json(schedule_path)
    cache_index, cache_index_sha = legacy.load_redacted_cache_index(
        cache_index_path,
        expected_reference_rows=schedule_reference_rows,
        expected_schedule_logical_sha256=schedule_json["logical_sha256"],
    )
    legacy.validate_schedule_cache_index_global_join(schedule, cache_index)
    cache_root = safe_path(
        RC_ROOT / str(authority["runtime_paths"]["stable_cache_root"]),
        file=False,
    )
    checkpoint_by_fold: dict[int, Path] = {}
    checkpoint_receipts: list[dict[str, Any]] = []
    foldset_by_fold = {int(row["outer_fold"]): row for row in foldset["folds"]}
    for fold in FOLDS:
        checkpoint = binding_path(authority, f"bag_checkpoint_fold{fold}")
        expected = foldset_by_fold[fold]["checkpoint"]
        require(
            str(checkpoint.relative_to(RC_ROOT)) == expected["path"]
            and file_sha256(checkpoint) == expected["sha256"],
            f"fold {fold} checkpoint/foldset drift",
        )
        checkpoint_by_fold[fold] = checkpoint
        checkpoint_receipts.append(
            {"outer_fold": fold, "path": str(checkpoint.relative_to(RC_ROOT)), "sha256": file_sha256(checkpoint)}
        )

    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    selected = [
        row
        for row in schedule
        if start <= int(row["execution_ordinal"]) < stop
        and int(row["execution_ordinal"]) not in EXCLUDED_EXECUTIONS
    ]
    require(bool(selected), "empty prejoin shard")
    records: list[dict[str, Any]] = []
    fold_query_counts: dict[str, int] = {}
    for fold in FOLDS:
        fold_rows = [row for row in selected if int(row["heldout_fold"]) == fold]
        fold_query_counts[str(fold)] = len(fold_rows)
        if not fold_rows:
            continue
        ordinals = sorted(int(row["historical_query_ordinal"]) for row in fold_rows)
        payload = legacy.materialize_payload(
            arm="RCDE_BAG",
            fold=fold,
            requested_query_ordinals=ordinals,
            schedule=schedule,
            cache_root=cache_root,
            checkpoint_path=checkpoint_by_fold[fold],
            device=torch.device(device),
            source_hashes={
                **schedule_hashes,
                "redacted_cache_index_sha256": cache_index_sha,
                "bag_completion_authority_sha256": authority_sha,
                "bag_foldset_manifest_sha256": file_sha256(foldset_path),
                "bag_foldset_validation_sha256": file_sha256(foldset_validation_path),
            },
            redacted_cache_index=cache_index,
            test_mode=False,
        )
        records.extend(payload["records"])
        del payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records.sort(key=lambda row: int(row["execution_ordinal"]))
    expected_executions = [int(row["execution_ordinal"]) for row in selected]
    require(
        [int(row["execution_ordinal"]) for row in records] == expected_executions,
        "prejoin shard execution order drift",
    )
    record_hashes = [str(row["logical_sha256"]) for row in records]
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "TARGET_FREE_FULL_C128_BAG_PREJOIN_ONLY",
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "eligible_execution_ordinals": expected_executions,
        "excluded_execution_ordinals_in_interval": [
            value for value in EXCLUDED_EXECUTIONS if start <= value < stop
        ],
        "query_count": len(records),
        "candidate_count_per_query": CANDIDATE_COUNT,
        "pair_count_per_query": CANDIDATE_COUNT * (CANDIDATE_COUNT - 1) // 2,
        "fold_query_counts": fold_query_counts,
        "records": records,
        "record_logical_sha256s": record_hashes,
        "record_sequence_sha256": canonical_sha256(record_hashes),
        "checkpoint_receipts": checkpoint_receipts,
        "source_bindings": {
            "authority_sha256": authority_sha,
            "redacted_schedule_sha256": file_sha256(schedule_path),
            "redacted_schedule_logical_sha256": schedule_json["logical_sha256"],
            "redacted_cache_index_sha256": file_sha256(cache_index_path),
            "bag_foldset_manifest_sha256": file_sha256(foldset_path),
            "bag_foldset_validation_sha256": file_sha256(foldset_validation_path),
            "legacy_materializer_sha256": file_sha256(binding_path(authority, "legacy_materializer")),
        },
        "model_load_count": sum(1 for value in fold_query_counts.values() if value),
        "candidate_forward_count": len(records) * CANDIDATE_COUNT,
        "unordered_pair_forward_count": len(records) * CANDIDATE_COUNT * (CANDIDATE_COUNT - 1) // 2,
        "target_rival_read_count": 0,
        "label_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "training_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
    }
    output["logical_sha256"] = canonical_sha256(
        {key: value for key, value in output.items() if key not in {"records", "logical_sha256"}}
    )
    output_path = safe_path(
        output_root / f"shard_{start:03d}_{stop:03d}.pt", must_exist=False
    )
    atomic_torch(output_path, output)
    return {
        "status": STATUS,
        "shard_ordinal": shard_ordinal,
        "query_count": len(records),
        "output": str(output_path),
        "sha256": file_sha256(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(materialize(args.authority, args.shard_ordinal, args.output_root, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
