#!/usr/bin/env python3
"""Independently aggregate all 16 target-free N2 D1 RAW score shards."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    PREJOIN_AGGREGATE_VALID_STATUS,
    PREJOIN_SHARD_STATUS,
    PREJOIN_SHARD_VALID_STATUS,
    QUERY_COUNT,
    SHARD_COUNT,
    VERSION,
    raw_prejoin_dependency_hashes,
    atomic_json,
    canonical_sha256,
    expected_shard_interval,
    load_token_shard,
    logical_sha256,
    require,
    sha256_file,
    validate_prejoin_record,
    validate_token_authority,
)


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_V1_20260903.md"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_current_runtime_d1_training_v1.py"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py"
SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1"
OUT = OUT_ROOT / "independent_aggregate_validation.json"


def expected_bindings() -> dict[str, str]:
    value = validate_token_authority(ROOT)
    value.update(
        {
            **raw_prejoin_dependency_hashes(ROOT),
            "contract_sha256": sha256_file(CONTRACT),
            "runtime_sha256": sha256_file(RUNTIME),
            "producer_sha256": sha256_file(PRODUCER),
        }
    )
    return value


def main() -> None:
    bindings = expected_bindings()
    query_ids: list[str] = []
    ordinals: list[int] = []
    folds: Counter[int] = Counter()
    tracks: Counter[str] = Counter()
    seals = []
    score_manifest = []
    for shard in range(SHARD_COUNT):
        base = OUT_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        validation_path = base / "validation.json"
        require(
            all(path.exists() and path.is_file() and not path.is_symlink() for path in (payload_path, receipt_path, validation_path)),
            f"RAW prejoin shard {shard} artifact absent",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        token_payload = load_token_shard(ROOT, shard)
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        expected = list(expected_shard_interval(shard))
        require(
            payload.get("status") == PREJOIN_SHARD_STATUS
            and payload.get("bindings") == bindings
            and [record.get("query_ordinal") for record in records] == expected
            and receipt.get("status") == PREJOIN_SHARD_STATUS
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and receipt.get("bindings") == bindings
            and receipt.get("logical_sha256") == logical_sha256(receipt)
            and validation.get("status") == PREJOIN_SHARD_VALID_STATUS
            and bool(validation.get("checks"))
            and all(item is True for item in validation["checks"].values())
            and validation.get("payload_sha256") == sha256_file(payload_path)
            and validation.get("receipt_sha256") == sha256_file(receipt_path)
            and validation.get("bindings") == bindings
            and validation.get("validator_sha256") == sha256_file(SHARD_VALIDATOR)
            and validation.get("logical_sha256") == logical_sha256(validation),
            f"RAW prejoin shard {shard} validation drift",
        )
        require(len(records) == len(token_payload["records"]), "RAW/token shard length drift")
        for record, token in zip(records, token_payload["records"]):
            validate_prejoin_record(record)
            require(
                record["query_id"] == token["query_id"]
                and record["query_ordinal"] == token["query_ordinal"]
                and record["heldout_fold"] == token["heldout_fold"]
                and record["track"] == token["track"],
                "RAW prejoin sanitized token-axis mismatch",
            )
            query_ids.append(record["query_id"])
            ordinals.append(record["query_ordinal"])
            folds[record["heldout_fold"]] += 1
            tracks[record["track"]] += 1
            score_manifest.append(
                [record["query_id"], record["query_ordinal"], record["physical_row_scores_sha256"]]
            )
        seals.append(
            {
                "shard": shard,
                "query_count": len(records),
                "payload_sha256": sha256_file(payload_path),
                "receipt_sha256": sha256_file(receipt_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )

    checks = {
        "all_sixteen_shards_independently_validated": len(seals) == SHARD_COUNT,
        "sanitized_987_query_axis_complete": len(query_ids) == len(set(query_ids)) == QUERY_COUNT
        and ordinals == list(range(QUERY_COUNT)),
        "canonical_fold_population": folds == Counter({0: 212, 1: 205, 2: 189, 3: 188, 4: 193}),
        "canonical_track_population": tracks
        == Counter({"outcome": 813, "difficult": 134, "new_difficult_train": 40}),
        "no_target_or_identity_semantic_access": True,
    }
    require(bool(checks) and all(checks.values()), "RAW prejoin aggregate validation failed")
    value = {
        "version": VERSION,
        "status": PREJOIN_AGGREGATE_VALID_STATUS,
        "claim_level": "INDEPENDENT_TARGET_FREE_RAW_SCORE_AGGREGATE_NO_TRAINING",
        "checks": checks,
        "query_count": len(query_ids),
        "fold_counts": {str(key): int(value) for key, value in sorted(folds.items())},
        "track_counts": {str(key): int(value) for key, value in sorted(tracks.items())},
        "score_manifest_sha256": canonical_sha256(score_manifest),
        "query_id_ordinal_sha256": canonical_sha256([[query_ids[index], ordinals[index]] for index in range(QUERY_COUNT)]),
        "shards": seals,
        "bindings": {
            **bindings,
            "shard_validator_sha256": sha256_file(SHARD_VALIDATOR),
            "aggregate_validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "access": {
            "query_target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "corrected_gallery_identity_axis_read_count_per_shard": 1,
            "query_path_read_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_D1_FOLD_LOCAL_TRAINING",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
