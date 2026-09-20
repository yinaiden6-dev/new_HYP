#!/usr/bin/env python3
"""Independently validate one BAG full-594 target-free shard."""

from __future__ import annotations

import argparse
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
    canonical_sha256,
    tensor_sha256,
)
import materialize_dino_rcde_r1_oof_prejoin_v3 as legacy  # noqa: E402
import validate_dino_rcde_r1_oof_prejoin_v1 as legacy_validator  # noqa: E402
from dino_rcde_bag_completion_io_v1 import (  # noqa: E402
    atomic_json,
    binding_path,
    file_sha256,
    read_authority,
    read_json,
    require,
    safe_path,
)
from materialize_dino_rcde_bag_full594_shard_v1 import SCHEMA, STATUS  # noqa: E402


VALIDATION_SCHEMA = "rc_dino_rcde_bag_full594_prejoin_shard_validation_v1_20260827"
VALIDATION_STATUS = "DINO_RCDE_BAG_FULL594_PREJOIN_SHARD_VALIDATION_PASS"


def validate(authority_path: Path, shard_path: Path, shard_ordinal: int, output_path: Path, *, device: str) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "validation shard ordinal drift")
    authority, authority_sha = read_authority(authority_path)
    require(binding_path(authority, "shard_validator") == Path(__file__).resolve(), "validator runtime binding drift")
    shard_path = safe_path(shard_path)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(shard, Mapping)
        and shard.get("schema_version") == SCHEMA
        and shard.get("status") == STATUS
        and shard.get("shard_ordinal") == shard_ordinal,
        "prejoin shard envelope drift",
    )
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    require(shard.get("execution_start") == start and shard.get("execution_stop") == stop, "shard interval drift")
    schedule_path = binding_path(authority, "redacted_schedule")
    schedule_json = read_json(schedule_path)
    schedule = schedule_json["records"]
    expected = [
        row
        for row in schedule
        if start <= int(row["execution_ordinal"]) < stop
        and int(row["execution_ordinal"]) not in EXCLUDED_EXECUTIONS
    ]
    records = shard.get("records")
    require(isinstance(records, list) and len(records) == len(expected), "shard query count drift")
    require(
        shard.get("eligible_execution_ordinals") == [int(row["execution_ordinal"]) for row in expected],
        "shard eligible execution scope drift",
    )
    checkpoint_by_fold = {
        fold: binding_path(authority, f"bag_checkpoint_fold{fold}") for fold in FOLDS
    }
    models: dict[int, Any] = {}
    sampled_count = 0
    reorder_count = 0
    logical_hashes: list[str] = []
    target_device = torch.device(device)
    for expected_row, record in zip(expected, records):
        require(isinstance(record, Mapping), "prejoin record is not a mapping")
        for output_key, schedule_key in (
            ("query_id", "query_id"),
            ("source_image_sha256", "source_image_sha256"),
            ("historical_query_ordinal", "historical_query_ordinal"),
            ("execution_ordinal", "execution_ordinal"),
            ("heldout_fold", "heldout_fold"),
            ("candidate_physical_rows", "candidate_physical_rows"),
            ("candidate_axis_sha256", "candidate_axis_sha256"),
        ):
            require(record.get(output_key) == expected_row.get(schedule_key), f"record/schedule mismatch: {output_key}")
        fold = int(record["heldout_fold"])
        rows = list(map(int, record["candidate_physical_rows"]))
        require(rows == sorted(rows) and len(rows) == CANDIDATE_COUNT and len(set(rows)) == CANDIDATE_COUNT, "candidate axis drift")
        require(record["candidate_axis_sha256"] == canonical_sha256(rows), "candidate axis hash drift")
        query_mask = record.get("query_valid_patch_mask")
        require(isinstance(query_mask, torch.Tensor) and query_mask.dtype == torch.bool, "query mask drift")
        candidates = record.get("candidates")
        require(isinstance(candidates, list) and len(candidates) == CANDIDATE_COUNT, "candidate evidence population drift")
        relational_rows = []
        for physical_row, candidate in zip(rows, candidates):
            require(isinstance(candidate, Mapping) and candidate.get("physical_row") == physical_row, "candidate evidence address drift")
            reference_mask = candidate.get("reference_valid_patch_mask")
            require(isinstance(reference_mask, torch.Tensor) and reference_mask.dtype == torch.bool, "reference mask drift")
            legacy_validator.validate_candidate(
                candidate, query_mask=query_mask, reference_mask=reference_mask
            )
            relational_rows.append(candidate["relational"])
        delta = record.get("delta")
        require(
            isinstance(delta, torch.Tensor)
            and delta.dtype == torch.float32
            and tuple(delta.shape) == (CANDIDATE_COUNT, CANDIDATE_COUNT)
            and bool(torch.isfinite(delta).all())
            and record.get("delta_sha256") == tensor_sha256(delta)
            and torch.equal(delta, -delta.T)
            and torch.equal(torch.diagonal(delta), torch.zeros(CANDIDATE_COUNT)),
            "full Delta closure drift",
        )
        if fold not in models:
            model, _, _ = legacy.load_frozen_model(
                checkpoint_by_fold[fold],
                arm="RCDE_BAG",
                fold=fold,
                device=target_device,
            )
            models[fold] = model
        model = models[fold]
        relational = torch.stack(relational_rows, dim=0).to(target_device)
        mask_device = query_mask.to(target_device)
        pairs = legacy_validator.sampled_pairs(CANDIDATE_COUNT)
        with torch.no_grad():
            for left, right in pairs:
                scalar = model.compare_relational(
                    relational[left], relational[right], mask_device
                ).logit.detach().cpu().to(torch.float32)
                require(torch.allclose(scalar, delta[left, right], rtol=1e-6, atol=1e-6), "sampled comparator replay drift")
                sampled_count += 1
            reverse = torch.arange(CANDIDATE_COUNT - 1, -1, -1, device=target_device)
            reversed_relational = relational.index_select(0, reverse)
            for left, right in pairs:
                reverse_left = CANDIDATE_COUNT - 1 - left
                reverse_right = CANDIDATE_COUNT - 1 - right
                value = model.compare_relational_batch(
                    reversed_relational,
                    torch.tensor([reverse_left], device=target_device),
                    torch.tensor([reverse_right], device=target_device),
                    mask_device,
                )[0].detach().cpu()
                require(torch.allclose(value, delta[left, right], rtol=1e-6, atol=1e-6), "candidate reorder replay drift")
                reorder_count += 1
        logical = legacy_validator.record_logical_sha256(record)
        require(record.get("logical_sha256") == logical, "record logical hash drift")
        logical_hashes.append(logical)
    require(
        shard.get("record_logical_sha256s") == logical_hashes
        and shard.get("record_sequence_sha256") == canonical_sha256(logical_hashes)
        and shard.get("target_rival_read_count") == 0
        and shard.get("label_read_count") == 0
        and shard.get("scientific_reduction_count") == 0
        and shard.get("protected_access_count") == 0
        and shard.get("training_count") == 0
        and shard.get("scientific_GO_or_NO_GO") is None,
        "target-free shard boundary drift",
    )
    result: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_count": len(records),
        "candidate_count_per_query": CANDIDATE_COUNT,
        "sampled_comparator_closure_count": sampled_count,
        "candidate_reorder_closure_count": reorder_count,
        "full_delta_strict_antisymmetry": True,
        "target_free_boundary_pass": True,
        "producer_shard_sha256": file_sha256(shard_path),
        "producer_shard_logical_sha256": shard["logical_sha256"],
        "authority_sha256": authority_sha,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
    }
    result["logical_sha256"] = canonical_sha256(result)
    atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(validate(args.authority, args.shard, args.shard_ordinal, args.output, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
