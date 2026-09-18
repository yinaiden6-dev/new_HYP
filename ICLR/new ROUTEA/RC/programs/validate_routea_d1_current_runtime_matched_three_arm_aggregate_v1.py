#!/usr/bin/env python3
"""Aggregate eight target-free matched A/B/C shards."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_V1_20260902.md"
SOURCE_VALIDATION = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1/validation.json"
REFERENCE_VALIDATION = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1/independent_validation.json"
LABEL_JOIN_HANDOFF = ROOT / "results/routea_d1_current_runtime_64_label_join_v1/independent_validation.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
OUT = OUT_ROOT / "validation.json"
SHARD_VALIDATION_KEYS = {
    "schema_version",
    "status",
    "shard",
    "checks",
    "query_count",
    "formula_score_max_abs",
    "payload_sha256",
    "receipt_sha256",
    "target_role_read_count",
    "model_update_count",
    "next_authorized_stage",
    "logical_sha256",
}
SHARD_CHECK_KEYS = {
    "payload_envelope",
    "input_authorities",
    "bindings",
    "checkpoint_direct_hash",
    "access",
    "record_population",
    "record_formula_and_raw_authority_replay",
    "receipt",
    "receipt_next_stage",
}


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable matched-three-arm aggregate exists: {OUT}")
    source_validation = json.loads(SOURCE_VALIDATION.read_text())
    reference_validation = json.loads(REFERENCE_VALIDATION.read_text())
    label_join_handoff = json.loads(LABEL_JOIN_HANDOFF.read_text())
    shard_receipts: list[dict] = []
    all_records: list[dict] = []
    shard_checks: list[bool] = []
    for shard in range(8):
        shard_dir = OUT_ROOT / f"shard{shard:02d}"
        payload_path = shard_dir / "payload.pt"
        receipt_path = shard_dir / "receipt.json"
        validation_path = shard_dir / "validation.json"
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        shard_checks.append(
            set(validation) == SHARD_VALIDATION_KEYS
            and set(validation.get("checks", {})) == SHARD_CHECK_KEYS
            and all(
                type(value) is bool and value
                for value in validation.get("checks", {}).values()
            )
            and validation.get("schema_version")
            == "routea_d1_current_runtime_matched_three_arm_prejoin_shard_validation_v1_20260902"
            and validation.get("status")
            == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_VALIDATED"
            and validation.get("logical_sha256") == e0.logical_sha256(validation)
            and validation.get("shard") == shard
            and validation.get("query_count") == 8
            and validation.get("payload_sha256") == e0.sha256_file(payload_path)
            and validation.get("receipt_sha256") == e0.sha256_file(receipt_path)
            and validation.get("formula_score_max_abs", 1.0) <= 1e-12
            and receipt.get("payload_sha256") == e0.sha256_file(payload_path)
            and payload.get("shard") == shard
            and len(payload.get("records", [])) == 8
            and validation.get("target_role_read_count") == 0
            and validation.get("model_update_count") == 0
            and validation.get("next_authorized_stage")
            == "MATCHED_THREE_ARM_AGGREGATE_VALIDATION"
        )
        all_records.extend(payload["records"])
        shard_receipts.append(
            {
                "shard": shard,
                "payload_sha256": e0.sha256_file(payload_path),
                "receipt_sha256": e0.sha256_file(receipt_path),
                "validation_sha256": e0.sha256_file(validation_path),
            }
        )
    roles = Counter(record["role"] for record in all_records)
    tracks = Counter(record["track"] for record in all_records)
    folds = Counter(int(record["heldout_fold"]) for record in all_records)
    executions = [int(record["execution_ordinal"]) for record in all_records]
    query_ids = [record["query_id"] for record in all_records]
    union_counts = [len(record["union_physical_rows"]) for record in all_records]
    candidate_records = sum(len(record["candidates"]) for record in all_records)
    checks = {
        "input_authorities": source_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and source_validation.get("logical_sha256")
        == e0.logical_sha256(source_validation)
        and reference_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and reference_validation.get("logical_sha256")
        == e0.logical_sha256(reference_validation)
        and label_join_handoff.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_LABEL_JOIN_INDEPENDENT_VALIDATION_PASS"
        and label_join_handoff.get("logical_sha256")
        == e0.logical_sha256(label_join_handoff)
        and all(label_join_handoff.get("checks", {}).values())
        and label_join_handoff.get("next_authorized_stage")
        == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_TARGET_FREE_MATERIALIZATION",
        "all_shards_validated": all(shard_checks),
        "query_population": len(all_records) == 64
        and len(set(executions)) == 64
        and len(set(query_ids)) == 64,
        "role_balance": roles == Counter({"TRAIN": 32, "EVAL": 32}),
        "track_population": tracks
        == Counter({"outcome": 53, "difficult": 9, "new_difficult_train": 2}),
        "fold_population": folds == Counter({1: 16, 2: 19, 3: 12, 4: 17}),
        "candidate_population": candidate_records == sum(union_counts)
        and all(128 <= count <= 256 for count in union_counts),
        "target_free_access": all(
            record.get("target_role_read_count") == 0
            and record.get("target_insertion_count") == 0
            and record.get("model_update_count") == 0
            for record in all_records
        ),
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_prejoin_aggregate_validation_v1_20260902",
        "status": (
            "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
            if passed
            else "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATION_ABORT"
        ),
        "claim_level": "TARGET_FREE_MATCHED_A_B_C_RAW_AND_D1_PORTABILITY_DIAGNOSTIC",
        "checks": checks,
        "query_count": len(all_records),
        "roles": dict(sorted(roles.items())),
        "tracks": dict(sorted(tracks.items())),
        "heldout_folds": {str(key): value for key, value in sorted(folds.items())},
        "candidate_record_count": candidate_records,
        "candidate_union_count_min": min(union_counts),
        "candidate_union_count_max": max(union_counts),
        "candidate_union_count_mean": sum(union_counts) / len(union_counts),
        "shards": shard_receipts,
        "contract_sha256": e0.sha256_file(CONTRACT),
        "source_validation_sha256": e0.sha256_file(SOURCE_VALIDATION),
        "reference_validation_sha256": e0.sha256_file(REFERENCE_VALIDATION),
        "label_join_handoff_sha256": e0.sha256_file(LABEL_JOIN_HANDOFF),
        "target_role_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": (
            "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_POSTSEAL_DIAGNOSTIC"
            if passed
            else None
        ),
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
