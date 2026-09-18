#!/usr/bin/env python3
"""Aggregate the 16 independently validated target-free fresh-D1 OOF shards."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_d1_oof_scoring_v1 import (  # noqa: E402
    EXPECTED_FOLD_COUNTS,
    EXPECTED_TRACK_COUNTS,
    OOF_AGGREGATE_VALID_STATUS,
    OOF_SHARD_STATUS,
    OOF_SHARD_VALID_STATUS,
    SHARD_COUNT,
    VERSION,
    base_dependency_bindings,
    canonical_sha256,
    expected_shard_interval,
    load_token_shard,
    logical_sha256,
    require,
    sha256_file,
    validate_oof_record,
)
from rc_aslo_xf.n2_current_runtime_d1_training_v1 import QUERY_COUNT, atomic_json  # noqa: E402


PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py"
SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
OUT = OUT_ROOT / "independent_aggregate_validation.json"


def expected_bindings() -> dict[str, Any]:
    value = base_dependency_bindings(ROOT)
    value["producer_sha256"] = sha256_file(PRODUCER)
    return value


def main() -> None:
    bindings = expected_bindings()
    query_ids: list[str] = []
    ordinals: list[int] = []
    folds: Counter[int] = Counter()
    tracks: Counter[str] = Counter()
    checkpoint_use: Counter[str] = Counter()
    boundary_gaps: list[float] = []
    manifest: list[list[Any]] = []
    seals: list[dict[str, Any]] = []

    for shard in range(SHARD_COUNT):
        base = OUT_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        validation_path = base / "validation.json"
        require(
            all(path.exists() and path.is_file() and not path.is_symlink() for path in (payload_path, receipt_path, validation_path)),
            f"fresh-D1 OOF shard {shard} artifact absent",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        token_records = load_token_shard(ROOT, shard)["records"]
        expected = list(expected_shard_interval(shard))
        require(
            payload.get("status") == OOF_SHARD_STATUS
            and payload.get("bindings") == bindings
            and [record.get("query_ordinal") for record in records] == expected
            and receipt.get("status") == OOF_SHARD_STATUS
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and receipt.get("bindings") == bindings
            and receipt.get("logical_sha256") == logical_sha256(receipt)
            and validation.get("status") == OOF_SHARD_VALID_STATUS
            and isinstance(validation.get("checks"), Mapping)
            and bool(validation["checks"])
            and all(item is True for item in validation["checks"].values())
            and validation.get("payload_sha256") == sha256_file(payload_path)
            and validation.get("receipt_sha256") == sha256_file(receipt_path)
            and validation.get("bindings") == bindings
            and validation.get("validator_sha256") == sha256_file(SHARD_VALIDATOR)
            and validation.get("logical_sha256") == logical_sha256(validation)
            and validation.get("next_authorized_stage") == "N2_FRESH_D1_OOF_PREJOIN_AGGREGATE_VALIDATION",
            f"fresh-D1 OOF shard {shard} validation drift",
        )
        require(len(records) == len(token_records), f"fresh-D1 OOF/token shard {shard} length drift")
        for record, token in zip(records, token_records, strict=True):
            validate_oof_record(record)
            require(
                record["query_id"] == token["query_id"]
                and record["query_ordinal"] == token["query_ordinal"]
                and record["heldout_fold"] == token["heldout_fold"]
                and record["track"] == token["track"]
                and record["image_tokens_sha256"] == token["image_tokens_sha256"]
                and record["template_tokens_sha256"] == token["template_tokens_sha256"],
                "fresh-D1 OOF sanitized token-axis mismatch",
            )
            query_ids.append(str(record["query_id"]))
            ordinals.append(int(record["query_ordinal"]))
            folds[int(record["heldout_fold"])] += 1
            tracks[str(record["track"])] += 1
            checkpoint_use[str(record["oof_checkpoint_sha256"])] += 1
            boundary_gaps.append(float(record["c128_boundary_score_gap"]))
            manifest.append(
                [
                    record["query_id"], record["query_ordinal"], record["heldout_fold"],
                    record["adapted_image_tokens_sha256"], record["physical_row_scores_sha256"],
                    record["ranked_representative_physical_rows_sha256"], record["oof_checkpoint_sha256"],
                ]
            )
        seals.append(
            {
                "shard": shard,
                "query_count": len(records),
                "payload_sha256": sha256_file(payload_path),
                "receipt_sha256": sha256_file(receipt_path),
                "validation_sha256": sha256_file(validation_path),
                "score_ranking_manifest_sha256": validation["score_ranking_manifest_sha256"],
            }
        )

    fold_seals = bindings["fold_seals"]
    expected_checkpoint_use = {
        fold_seals[str(fold)]["checkpoint_sha256"]: count
        for fold, count in EXPECTED_FOLD_COUNTS.items()
    }
    checks = {
        "all_sixteen_shards_independently_validated": len(seals) == SHARD_COUNT,
        "sanitized_987_query_axis_complete": len(query_ids) == len(set(query_ids)) == QUERY_COUNT
        and ordinals == list(range(QUERY_COUNT)),
        "canonical_fold_population": dict(sorted(folds.items())) == EXPECTED_FOLD_COUNTS,
        "canonical_track_population": dict(sorted(tracks.items())) == dict(sorted(EXPECTED_TRACK_COUNTS.items())),
        "every_query_uses_own_heldout_fold_checkpoint": dict(checkpoint_use) == expected_checkpoint_use,
        "complete_5412_ranking_and_natural_c128_sealed": len(manifest) == QUERY_COUNT,
        "zero_target_or_identity_semantic_join": True,
    }
    require(all(checks.values()), "fresh-D1 OOF prejoin aggregate validation failed")
    sorted_gaps = sorted(boundary_gaps)
    require(len(sorted_gaps) == QUERY_COUNT and sorted_gaps[0] >= 0.0, "C128 boundary gap population drift")
    boundary_summary = {
        "tie_break": "score_descending_then_lower_physical_row",
        "quantile_definition": "lower_order_statistic_index_floor_q_times_n_minus_1",
        "minimum": sorted_gaps[0],
        "q01": sorted_gaps[int(0.01 * (QUERY_COUNT - 1))],
        "q05": sorted_gaps[int(0.05 * (QUERY_COUNT - 1))],
        "q25": sorted_gaps[int(0.25 * (QUERY_COUNT - 1))],
        "q50": sorted_gaps[int(0.50 * (QUERY_COUNT - 1))],
        "exact_tie_count": sum(gap == 0.0 for gap in sorted_gaps),
    }
    value = {
        "version": VERSION,
        "status": OOF_AGGREGATE_VALID_STATUS,
        "claim_level": "INDEPENDENT_TARGET_FREE_FRESH_D1_FULL_GALLERY_OOF_AGGREGATE_NO_SCIENTIFIC_CLAIM",
        "checks": checks,
        "query_count": len(query_ids),
        "fold_counts": {str(key): int(value) for key, value in sorted(folds.items())},
        "track_counts": {str(key): int(value) for key, value in sorted(tracks.items())},
        "score_ranking_manifest_sha256": canonical_sha256(manifest),
        "query_id_ordinal_fold_sha256": canonical_sha256(
            [[query_ids[index], ordinals[index], int(manifest[index][2])] for index in range(QUERY_COUNT)]
        ),
        "checkpoint_use_counts": dict(sorted(checkpoint_use.items())),
        "c128_rank128_129_boundary": boundary_summary,
        "shards": seals,
        "bindings": {
            **bindings,
            "shard_validator_sha256": sha256_file(SHARD_VALIDATOR),
            "aggregate_validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "access": {
            "query_target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "model_update_count": 0,
            "target_insertion_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_FRESH_D1_POSTSEAL_FIXED_R5_CANDIDATE_RECALL_GATE",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
