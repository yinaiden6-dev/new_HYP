#!/usr/bin/env python3
"""Independently validate the fixed-R5 fresh-D1 candidate-recall gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_d1_oof_scoring_v1 import (  # noqa: E402
    EXPECTED_FOLD_COUNTS,
    EXPECTED_TRACK_COUNTS,
    OOF_AGGREGATE_VALID_STATUS,
    SHARD_COUNT,
    VERSION,
    canonical_sha256,
    logical_sha256,
    require,
    sha256_file,
    validate_oof_record,
    validate_r5_fixed_authority,
)
from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (  # noqa: E402
    QUERY_COUNT,
    atomic_json,
    load_gallery,
    validate_prejoin_record,
)


OOF_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
OOF_AGGREGATE = OOF_ROOT / "independent_aggregate_validation.json"
RAW_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_raw_prejoin_v1"
RAW_AGGREGATE = RAW_ROOT / "independent_aggregate_validation.json"
EVALUATOR = ROOT / "programs/evaluate_routea_matched_three_arm_n2_d1_candidate_recall_gate_v1.py"
RESULT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1"
RESULT = RESULT_ROOT / "result.json"
OUT = RESULT_ROOT / "independent_validation.json"


def direct_rank(
    scores: torch.Tensor,
    ranked_rows: Sequence[int],
    labels: Sequence[str],
    target: str,
) -> tuple[int, bool, float]:
    rows = list(map(int, ranked_rows))
    identities = [str(labels[row]) for row in rows]
    require(len(rows) == len(set(identities)) == 5412 and target in identities, "independent reduced axis drift")
    slot = identities.index(target)
    target_score = float(scores[rows[slot]])
    rank = 1 + sum(
        float(scores[row]) >= target_score
        for index, row in enumerate(rows)
        if index != slot
    )
    return rank, target in set(identities[:128]), target_score


def raw_rows(scores: torch.Tensor, labels: Sequence[str]) -> list[int]:
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    seen: set[str] = set()
    output = []
    for row in order:
        identity = str(labels[row])
        if identity not in seen:
            seen.add(identity)
            output.append(int(row))
    require(len(output) == len(seen) == 5412, "independent RAW reduction drift")
    return output


def summary(cases: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    count = len(cases)
    if count == 0:
        return {
            "query_count": 0,
            "top1_count": 0,
            "R@1": None,
            "MRR": None,
            "c128_target_coverage_count": 0,
            "c128_target_coverage": None,
            "rank_bands": {"1_128": 0, "129_512": 0, "gt_512": 0},
        }
    ranks = [int(row[f"{prefix}_rank"]) for row in cases]
    reach = [bool(row[f"{prefix}_target_in_c128"]) for row in cases]
    return {
        "query_count": count,
        "top1_count": sum(rank == 1 for rank in ranks),
        "R@1": sum(rank == 1 for rank in ranks) / count,
        "MRR": sum(1.0 / rank for rank in ranks) / count,
        "c128_target_coverage_count": sum(reach),
        "c128_target_coverage": sum(reach) / count,
        "rank_bands": {
            "1_128": sum(rank <= 128 for rank in ranks),
            "129_512": sum(128 < rank <= 512 for rank in ranks),
            "gt_512": sum(rank > 512 for rank in ranks),
        },
    }


def grouped(cases: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cases:
        values[str(row[key])].append(row)
    return {
        value: {"RAW": summary(rows, "raw"), "fresh_D1": summary(rows, "fresh_d1")}
        for value, rows in sorted(values.items())
    }


def main() -> None:
    require(RESULT.is_file() and not RESULT.is_symlink(), "candidate-recall result absent")
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        require(
            existing.get("status")
            in {
                "N2_CANDIDATE_RECALL_GO_INDEPENDENTLY_VALIDATED",
                "N2_CANDIDATE_RECALL_NO_GO_INDEPENDENTLY_VALIDATED",
            }
            and existing.get("logical_sha256") == logical_sha256(existing)
            and existing.get("result_sha256") == sha256_file(RESULT)
            and existing.get("validator_sha256") == sha256_file(Path(__file__).resolve())
            and isinstance(existing.get("checks"), Mapping)
            and all(value is True for value in existing["checks"].values()),
            f"immutable candidate-recall validation is stale or corrupt: {OUT}",
        )
        print(json.dumps({"status": existing["status"], "existing": True}, sort_keys=True), flush=True)
        return
    result = json.loads(RESULT.read_text())
    oof_aggregate = json.loads(OOF_AGGREGATE.read_text())
    raw_aggregate = json.loads(RAW_AGGREGATE.read_text())
    require(
        result.get("logical_sha256") == logical_sha256(result)
        and result.get("status") in {"N2_CANDIDATE_RECALL_GO", "N2_CANDIDATE_RECALL_NO_GO"}
        and oof_aggregate.get("status") == OOF_AGGREGATE_VALID_STATUS
        and oof_aggregate.get("logical_sha256") == logical_sha256(oof_aggregate)
        and raw_aggregate.get("status") == "ROUTEA_N2_D1_RAW_PREJOIN_AGGREGATE_VALIDATED"
        and raw_aggregate.get("logical_sha256") == logical_sha256(raw_aggregate),
        "candidate-recall/result aggregate envelope drift",
    )
    _r5_result, r5_records, fixed235_records, r5_seals = validate_r5_fixed_authority(ROOT)
    r5_by_id = {str(row["query_id"]): row for row in r5_records}
    fixed235_by_id = {str(row["query_id"]): row for row in fixed235_records}
    _, labels, _gallery_receipt = load_gallery(ROOT)
    oof_seals = {int(row["shard"]): row for row in oof_aggregate["shards"]}
    raw_seals = {int(row["shard"]): row for row in raw_aggregate["shards"]}
    expected_cases: list[dict[str, Any]] = []
    source_seals: list[dict[str, Any]] = []
    for shard in range(SHARD_COUNT):
        oof_path = OOF_ROOT / f"shard{shard:02d}/payload.pt"
        oof_validation_path = OOF_ROOT / f"shard{shard:02d}/validation.json"
        raw_path = RAW_ROOT / f"shard{shard:02d}/payload.pt"
        require(
            oof_seals[shard]["payload_sha256"] == sha256_file(oof_path)
            and oof_seals[shard]["validation_sha256"] == sha256_file(oof_validation_path)
            and raw_seals[shard]["payload_sha256"] == sha256_file(raw_path),
            "independent candidate-recall source seal drift",
        )
        oof_records = torch.load(oof_path, map_location="cpu", weights_only=False, mmap=True)["records"]
        raw_records = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)["records"]
        require(len(oof_records) == len(raw_records), "independent post-seal shard length drift")
        for oof, raw in zip(oof_records, raw_records, strict=True):
            validate_oof_record(oof)
            validate_prejoin_record(raw)
            query_id = str(oof["query_id"])
            role = r5_by_id.get(query_id)
            require(
                role is not None
                and raw["query_id"] == query_id
                and raw["query_ordinal"] == oof["query_ordinal"]
                and raw["heldout_fold"] == oof["heldout_fold"] == role["fold"]
                and raw["track"] == oof["track"] == role["track"],
                "independent candidate-recall query join drift",
            )
            target = str(role["identity"])
            fixed_role = fixed235_by_id.get(query_id)
            fresh_scores = torch.as_tensor(oof["physical_row_scores"])
            fresh_rank, fresh_in, fresh_score = direct_rank(
                fresh_scores,
                oof["ranked_representative_physical_rows"].tolist(),
                labels,
                target,
            )
            raw_scores_value = torch.as_tensor(raw["physical_row_scores"])
            raw_rank, raw_in, raw_score = direct_rank(
                raw_scores_value, raw_rows(raw_scores_value, labels), labels, target
            )
            expected_cases.append(
                {
                    "query_id": query_id,
                    "query_ordinal": int(oof["query_ordinal"]),
                    "fold": int(oof["heldout_fold"]),
                    "track": str(oof["track"]),
                    "target_identity": target,
                    "supergroup": str(role["supergroup"]),
                    "raw_rank": raw_rank,
                    "raw_target_in_c128": raw_in,
                    "raw_target_score": raw_score,
                    "fresh_d1_rank": fresh_rank,
                    "fresh_d1_target_in_c128": fresh_in,
                    "fresh_d1_target_score": fresh_score,
                    "old_r5_d1_top1_correct": bool(role["D1_top1_correct"]),
                    "old_r5_d1_rank": int(role["D1_target_rank"]),
                    "old_r5_d1_target_in_c128": bool(role["D1_target_in_C128"]),
                    "old_r5_union_target_in_c256": bool(role["union_target_in_C256"]),
                    "in_fixed_r5_old_d1_wrong_235": fixed_role is not None,
                    "fresh_d1_added_over_old_c128": fixed_role is not None
                    and (not bool(fixed_role["old_D1_target_in_C128"]))
                    and fresh_in,
                    "old_deep_tail": bool(fixed_role["fixed_deep_tail22"])
                    if fixed_role is not None
                    else False,
                    "fresh_d1_checkpoint_sha256": str(oof["oof_checkpoint_sha256"]),
                    "target_insertion_count": 0,
                }
            )
        source_seals.append(
            {
                "shard": shard,
                "oof_payload_sha256": sha256_file(oof_path),
                "oof_validation_sha256": sha256_file(oof_validation_path),
                "raw_payload_sha256": sha256_file(raw_path),
            }
        )
    expected_cases.sort(key=lambda row: row["query_ordinal"])
    require(expected_cases == result.get("cases"), "independent per-query candidate-recall replay differs")

    fixed = [row for row in expected_cases if row["in_fixed_r5_old_d1_wrong_235"]]
    deep = [row for row in fixed if row["old_deep_tail"]]
    added = [row for row in fixed if row["fresh_d1_added_over_old_c128"]]
    raw_wrong = [row for row in expected_cases if row["raw_rank"] != 1]
    fresh_wrong = [row for row in expected_cases if row["fresh_d1_rank"] != 1]
    identity_counts = Counter(row["target_identity"] for row in added)
    added_groups = {row["supergroup"] for row in added}
    concentration = max(identity_counts.values()) / len(added) if added else 1.0
    fixed_reach = sum(row["fresh_d1_target_in_c128"] for row in fixed)
    deep_reach = sum(row["fresh_d1_target_in_c128"] for row in deep)
    gate = {
        "fixed235_fresh_d1_c128_coverage_at_least_90_percent": fixed_reach >= math.ceil(0.90 * 235),
        "at_least_five_added_target_identities": len(identity_counts) >= 5,
        "at_least_four_added_supergroups": len(added_groups) >= 4,
        "maximum_added_identity_concentration_at_most_40_percent": concentration <= 0.40,
        "old_deep_tail_recovery_at_least_20_percent": deep_reach >= math.ceil(0.20 * 22),
    }
    expected_status = "N2_CANDIDATE_RECALL_GO" if all(gate.values()) else "N2_CANDIDATE_RECALL_NO_GO"
    metrics = result.get("metrics", {})
    metrics_checks = {
        "all987_summary": metrics.get("all987")
        == {"RAW": summary(expected_cases, "raw"), "fresh_D1": summary(expected_cases, "fresh_d1")},
        "raw_wrong_summary": metrics.get("raw_wrong")
        == {"RAW": summary(raw_wrong, "raw"), "fresh_D1": summary(raw_wrong, "fresh_d1")},
        "fresh_wrong_summary": metrics.get("fresh_d1_wrong")
        == {"RAW": summary(fresh_wrong, "raw"), "fresh_D1": summary(fresh_wrong, "fresh_d1")},
        "fixed235_metrics": metrics.get("fixed_r5_old_d1_wrong_235")
        == {
            "query_count": 235,
            "old_d1_c128_coverage_count": 190,
            "old_d1_c128_coverage": 190 / 235,
            "old_r5_union_coverage_count": 194,
            "old_r5_union_coverage": 194 / 235,
            "fresh_d1": summary(fixed, "fresh_d1"),
            "fresh_d1_added_query_count": len(added),
            "fresh_d1_added_identity_count": len(identity_counts),
            "fresh_d1_added_supergroup_count": len(added_groups),
            "maximum_single_added_identity_share": concentration,
        },
        "deep22_metrics": metrics.get("fixed_r5_old_deep_tail_22")
        == {
            "query_count": 22,
            "fresh_d1_c128_recovered_count": deep_reach,
            "fresh_d1_c128_recovery": deep_reach / 22,
        },
        "raw_to_fresh_top1_transition": metrics.get("raw_to_fresh_d1_top1")
        == {
            "rescue": sum(row["raw_rank"] != 1 and row["fresh_d1_rank"] == 1 for row in expected_cases),
            "break": sum(row["raw_rank"] == 1 and row["fresh_d1_rank"] != 1 for row in expected_cases),
            "both_correct": sum(row["raw_rank"] == row["fresh_d1_rank"] == 1 for row in expected_cases),
            "both_wrong": sum(row["raw_rank"] != 1 and row["fresh_d1_rank"] != 1 for row in expected_cases),
        },
        "group_summaries": result.get("summaries_by_fold") == grouped(expected_cases, "fold")
        and result.get("summaries_by_track") == grouped(expected_cases, "track")
        and result.get("summaries_by_supergroup") == grouped(expected_cases, "supergroup"),
    }
    binding_checks = {
        "oof_aggregate": result.get("bindings", {}).get("oof_aggregate_sha256") == sha256_file(OOF_AGGREGATE),
        "raw_aggregate": result.get("bindings", {}).get("raw_aggregate_sha256") == sha256_file(RAW_AGGREGATE),
        "r5_seals": all(result.get("bindings", {}).get(key) == value for key, value in r5_seals.items()),
        "evaluator": result.get("bindings", {}).get("evaluator_sha256") == sha256_file(EVALUATOR),
        "source_shards": result.get("bindings", {}).get("source_shards_sha256") == canonical_sha256(source_seals),
    }
    checks = {
        "complete_987_case_replay": len(expected_cases) == QUERY_COUNT,
        "canonical_fold_population": Counter(row["fold"] for row in expected_cases) == Counter(EXPECTED_FOLD_COUNTS),
        "canonical_track_population": Counter(row["track"] for row in expected_cases) == Counter(EXPECTED_TRACK_COUNTS),
        "fixed_235_and_deep_22_reproduced": len(fixed) == 235 and len(deep) == 22,
        "gate_recomputed": result.get("candidate_recall_gate_checks") == gate and result.get("status") == expected_status,
        "metric_reduction_recomputed": all(metrics_checks.values()),
        "added_identity_and_group_ledgers_recomputed": result.get("added_target_identities")
        == dict(sorted(identity_counts.items()))
        and result.get("added_supergroups") == sorted(added_groups),
        "c128_boundary_diagnostic_seal": result.get("c128_rank128_129_boundary")
        == oof_aggregate.get("c128_rank128_129_boundary"),
        "source_bindings_exact": all(binding_checks.values()),
        "target_join_after_seal_and_zero_insertion": result.get("access")
        == {
            "query_target_identity_read_count": QUERY_COUNT,
            "query_supergroup_read_count": QUERY_COUNT,
            "target_join_after_oof_aggregate_seal": True,
            "target_insertion_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "claim_boundary": result.get("scientific_GO_or_NO_GO") is None
        and result.get("ownership_GO_or_NO_GO") is None
        and result.get("external_confirmation") is False,
        "result_internal_checks": isinstance(result.get("checks"), Mapping)
        and bool(result["checks"])
        and all(value is True for value in result["checks"].values()),
    }
    require(all(checks.values()), "independent candidate-recall gate validation failed")
    value = {
        "version": VERSION,
        "status": f"{expected_status}_INDEPENDENTLY_VALIDATED",
        "claim_level": "INDEPENDENT_INTERNAL_OPENED_CANDIDATE_RECALL_GATE_VALIDATION_NOT_OWNERSHIP_OR_EXTERNAL_CLAIM",
        "checks": checks,
        "metric_checks": metrics_checks,
        "binding_checks": binding_checks,
        "candidate_recall_gate_checks": gate,
        "fixed235_fresh_d1_c128_coverage_count": fixed_reach,
        "deep22_fresh_d1_c128_recovered_count": deep_reach,
        "result_sha256": sha256_file(RESULT),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "model_update_count": 0,
        "target_insertion_count": 0,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
