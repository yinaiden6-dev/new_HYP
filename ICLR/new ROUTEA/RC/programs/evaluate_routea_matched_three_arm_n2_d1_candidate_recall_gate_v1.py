#!/usr/bin/env python3
"""Post-seal candidate-recall evaluation on the fixed R5 235/22 population."""

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
    OOF_SHARD_STATUS,
    OOF_SHARD_VALID_STATUS,
    SHARD_COUNT,
    VERSION,
    base_dependency_bindings,
    canonical_sha256,
    load_token_shard,
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
OOF_PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py"
OOF_SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py"
OOF_AGGREGATE_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_aggregate_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1"
OUT = OUT_ROOT / "result.json"


def expected_oof_bindings() -> dict[str, Any]:
    value = base_dependency_bindings(ROOT)
    value["producer_sha256"] = sha256_file(OOF_PRODUCER)
    return value


def strict_rank_and_membership(
    scores: torch.Tensor,
    ranked_rows: Sequence[int],
    corrected_labels: Sequence[str],
    target: str,
) -> tuple[int, bool, float]:
    labels = tuple(map(str, corrected_labels))
    require(target in labels, "post-seal target absent from corrected gallery")
    rows = list(map(int, ranked_rows))
    ranked_labels = [labels[row] for row in rows]
    require(len(rows) == len(ranked_labels) == 5412 and len(set(ranked_labels)) == 5412, "post-seal reduced axis drift")
    slot = ranked_labels.index(target)
    target_score = float(scores[rows[slot]])
    strict_rank = 1 + sum(
        float(scores[row]) >= target_score
        for index, row in enumerate(rows)
        if index != slot
    )
    return strict_rank, target in set(ranked_labels[:128]), target_score


def summarize(cases: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    ranks = [int(case[f"{prefix}_rank"]) for case in cases]
    coverage = [bool(case[f"{prefix}_target_in_c128"]) for case in cases]
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
    return {
        "query_count": count,
        "top1_count": sum(rank == 1 for rank in ranks),
        "R@1": sum(rank == 1 for rank in ranks) / count,
        "MRR": sum(1.0 / rank for rank in ranks) / count,
        "c128_target_coverage_count": sum(coverage),
        "c128_target_coverage": sum(coverage) / count,
        "rank_bands": {
            "1_128": sum(rank <= 128 for rank in ranks),
            "129_512": sum(128 < rank <= 512 for rank in ranks),
            "gt_512": sum(rank > 512 for rank in ranks),
        },
    }


def grouped_summary(cases: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case[key])].append(case)
    return {
        value: {"RAW": summarize(rows, "raw"), "fresh_D1": summarize(rows, "fresh_d1")}
        for value, rows in sorted(groups.items())
    }


def main() -> None:
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        require(
            existing.get("status") in {"N2_CANDIDATE_RECALL_GO", "N2_CANDIDATE_RECALL_NO_GO"}
            and existing.get("logical_sha256") == logical_sha256(existing)
            and existing.get("bindings", {}).get("evaluator_sha256")
            == sha256_file(Path(__file__).resolve()),
            f"immutable candidate-recall result is stale or corrupt: {OUT}",
        )
        print(json.dumps({"status": existing["status"], "existing": True}, sort_keys=True), flush=True)
        return
    oof_aggregate = json.loads(OOF_AGGREGATE.read_text())
    raw_aggregate = json.loads(RAW_AGGREGATE.read_text())
    oof_bindings = expected_oof_bindings()
    require(
        oof_aggregate.get("status") == OOF_AGGREGATE_VALID_STATUS
        and isinstance(oof_aggregate.get("checks"), Mapping)
        and bool(oof_aggregate["checks"])
        and all(value is True for value in oof_aggregate["checks"].values())
        and oof_aggregate.get("query_count") == QUERY_COUNT
        and oof_aggregate.get("logical_sha256") == logical_sha256(oof_aggregate)
        and oof_aggregate.get("next_authorized_stage")
        == "N2_FRESH_D1_POSTSEAL_FIXED_R5_CANDIDATE_RECALL_GATE"
        and oof_aggregate.get("bindings")
        == {
            **oof_bindings,
            "shard_validator_sha256": sha256_file(OOF_SHARD_VALIDATOR),
            "aggregate_validator_sha256": sha256_file(OOF_AGGREGATE_VALIDATOR),
        },
        "fresh-D1 OOF aggregate does not authorize target join",
    )
    require(
        raw_aggregate.get("status") == "ROUTEA_N2_D1_RAW_PREJOIN_AGGREGATE_VALIDATED"
        and raw_aggregate.get("query_count") == QUERY_COUNT
        and isinstance(raw_aggregate.get("checks"), Mapping)
        and all(value is True for value in raw_aggregate["checks"].values())
        and raw_aggregate.get("logical_sha256") == logical_sha256(raw_aggregate),
        "RAW prejoin aggregate drift",
    )
    _r5_result, r5_records, fixed235_records, r5_seals = validate_r5_fixed_authority(ROOT)
    r5_by_id = {str(row["query_id"]): row for row in r5_records}
    fixed235_by_id = {str(row["query_id"]): row for row in fixed235_records}
    require(len(r5_by_id) == QUERY_COUNT, "R5 query-ID population drift")
    require(len(fixed235_by_id) == 235, "fixed235 ledger V2 population drift")

    _references, corrected, gallery_receipt = load_gallery(ROOT)
    del _references
    oof_shard_seals = {int(row["shard"]): row for row in oof_aggregate["shards"]}
    raw_shard_seals = {int(row["shard"]): row for row in raw_aggregate["shards"]}
    cases: list[dict[str, Any]] = []
    source_seals: list[dict[str, Any]] = []
    for shard in range(SHARD_COUNT):
        oof_payload_path = OOF_ROOT / f"shard{shard:02d}/payload.pt"
        oof_validation_path = OOF_ROOT / f"shard{shard:02d}/validation.json"
        raw_payload_path = RAW_ROOT / f"shard{shard:02d}/payload.pt"
        require(
            oof_shard_seals[shard]["payload_sha256"] == sha256_file(oof_payload_path)
            and oof_shard_seals[shard]["validation_sha256"] == sha256_file(oof_validation_path)
            and raw_shard_seals[shard]["payload_sha256"] == sha256_file(raw_payload_path),
            f"post-seal source shard {shard} hash drift",
        )
        oof_payload = torch.load(oof_payload_path, map_location="cpu", weights_only=False, mmap=True)
        raw_payload = torch.load(raw_payload_path, map_location="cpu", weights_only=False, mmap=True)
        token_records = load_token_shard(ROOT, shard)["records"]
        oof_records = oof_payload["records"]
        raw_records = raw_payload["records"]
        require(len(oof_records) == len(raw_records) == len(token_records), "post-seal shard length drift")
        for oof, raw, token in zip(oof_records, raw_records, token_records, strict=True):
            validate_oof_record(oof)
            validate_prejoin_record(raw)
            query_id = str(token["query_id"])
            role = r5_by_id.get(query_id)
            require(
                role is not None
                and oof["query_id"] == raw["query_id"] == query_id
                and oof["query_ordinal"] == raw["query_ordinal"] == token["query_ordinal"]
                and oof["heldout_fold"] == raw["heldout_fold"] == token["heldout_fold"] == role["fold"]
                and oof["track"] == raw["track"] == token["track"] == role["track"],
                "post-seal OOF/RAW/token/R5 query join drift",
            )
            target = str(role["identity"])
            fixed_role = fixed235_by_id.get(query_id)
            fresh_scores = torch.as_tensor(oof["physical_row_scores"])
            fresh_rows = oof["ranked_representative_physical_rows"].tolist()
            fresh_rank, fresh_in, fresh_target_score = strict_rank_and_membership(
                fresh_scores, fresh_rows, corrected, target
            )
            raw_scores = torch.as_tensor(raw["physical_row_scores"])
            # Reuse the already target-free fresh ranking helper semantics by
            # stable-sorting RAW physical rows and taking each identity once.
            order = torch.argsort(raw_scores, descending=True, stable=True).tolist()
            seen: set[str] = set()
            raw_rows: list[int] = []
            for physical_row in order:
                identity = corrected[physical_row]
                if identity not in seen:
                    seen.add(identity)
                    raw_rows.append(int(physical_row))
            raw_rank, raw_in, raw_target_score = strict_rank_and_membership(
                raw_scores, raw_rows, corrected, target
            )
            cases.append(
                {
                    "query_id": query_id,
                    "query_ordinal": int(token["query_ordinal"]),
                    "fold": int(token["heldout_fold"]),
                    "track": str(token["track"]),
                    "target_identity": target,
                    "supergroup": str(role["supergroup"]),
                    "raw_rank": raw_rank,
                    "raw_target_in_c128": raw_in,
                    "raw_target_score": raw_target_score,
                    "fresh_d1_rank": fresh_rank,
                    "fresh_d1_target_in_c128": fresh_in,
                    "fresh_d1_target_score": fresh_target_score,
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
                "oof_payload_sha256": sha256_file(oof_payload_path),
                "oof_validation_sha256": sha256_file(oof_validation_path),
                "raw_payload_sha256": sha256_file(raw_payload_path),
            }
        )

    cases.sort(key=lambda row: row["query_ordinal"])
    require(
        len(cases) == len({row["query_id"] for row in cases}) == QUERY_COUNT
        and [row["query_ordinal"] for row in cases] == list(range(QUERY_COUNT))
        and Counter(row["fold"] for row in cases) == Counter(EXPECTED_FOLD_COUNTS)
        and Counter(row["track"] for row in cases) == Counter(EXPECTED_TRACK_COUNTS),
        "post-seal candidate-recall population drift",
    )
    fixed = [row for row in cases if row["in_fixed_r5_old_d1_wrong_235"]]
    deep = [row for row in fixed if row["old_deep_tail"]]
    added = [row for row in fixed if row["fresh_d1_added_over_old_c128"]]
    fresh_wrong = [row for row in cases if row["fresh_d1_rank"] != 1]
    raw_wrong = [row for row in cases if row["raw_rank"] != 1]
    require(
        len(fixed) == 235
        and len(deep) == 22
        and sum(row["old_r5_d1_target_in_c128"] for row in fixed) == 190
        and sum(row["old_r5_union_target_in_c256"] for row in fixed) == 194,
        "fixed R5 235/22 baseline reproduction failed",
    )
    added_identity_counts = Counter(row["target_identity"] for row in added)
    added_groups = {row["supergroup"] for row in added}
    maximum_concentration = (
        max(added_identity_counts.values()) / len(added) if added else 1.0
    )
    fresh_fixed_coverage_count = sum(row["fresh_d1_target_in_c128"] for row in fixed)
    deep_recovered_count = sum(row["fresh_d1_target_in_c128"] for row in deep)
    gate_checks = {
        "fixed235_fresh_d1_c128_coverage_at_least_90_percent": fresh_fixed_coverage_count >= math.ceil(0.90 * 235),
        "at_least_five_added_target_identities": len(added_identity_counts) >= 5,
        "at_least_four_added_supergroups": len(added_groups) >= 4,
        "maximum_added_identity_concentration_at_most_40_percent": maximum_concentration <= 0.40,
        "old_deep_tail_recovery_at_least_20_percent": deep_recovered_count >= math.ceil(0.20 * 22),
    }
    go = all(gate_checks.values())
    value = {
        "version": VERSION,
        "status": "N2_CANDIDATE_RECALL_GO" if go else "N2_CANDIDATE_RECALL_NO_GO",
        "claim_level": (
            "INTERNAL_OPENED_FIXED_R5_CANDIDATE_RECALL_GATE_PASSED_NOT_EXTERNAL_CONFIRMATION"
            if go
            else "INTERNAL_OPENED_FIXED_R5_CANDIDATE_RECALL_GATE_FAILED_NOT_OWNERSHIP_NO_GO"
        ),
        "checks": {
            "full_987_target_free_oof_scores_sealed_before_join": True,
            "fixed_r5_987_235_22_population_reproduced": True,
            "corrected_5413_row_5412_identity_axis": gallery_receipt["corrected_identity_count"] == 5412,
            "target_insertion_zero": all(row["target_insertion_count"] == 0 for row in cases),
        },
        "candidate_recall_gate_checks": gate_checks,
        "c128_rank128_129_boundary": oof_aggregate["c128_rank128_129_boundary"],
        "metrics": {
            "all987": {"RAW": summarize(cases, "raw"), "fresh_D1": summarize(cases, "fresh_d1")},
            "raw_wrong": {"RAW": summarize(raw_wrong, "raw"), "fresh_D1": summarize(raw_wrong, "fresh_d1")},
            "fresh_d1_wrong": {"RAW": summarize(fresh_wrong, "raw"), "fresh_D1": summarize(fresh_wrong, "fresh_d1")},
            "fixed_r5_old_d1_wrong_235": {
                "query_count": len(fixed),
                "old_d1_c128_coverage_count": 190,
                "old_d1_c128_coverage": 190 / 235,
                "old_r5_union_coverage_count": 194,
                "old_r5_union_coverage": 194 / 235,
                "fresh_d1": summarize(fixed, "fresh_d1"),
                "fresh_d1_added_query_count": len(added),
                "fresh_d1_added_identity_count": len(added_identity_counts),
                "fresh_d1_added_supergroup_count": len(added_groups),
                "maximum_single_added_identity_share": maximum_concentration,
            },
            "fixed_r5_old_deep_tail_22": {
                "query_count": len(deep),
                "fresh_d1_c128_recovered_count": deep_recovered_count,
                "fresh_d1_c128_recovery": deep_recovered_count / 22,
            },
            "raw_to_fresh_d1_top1": {
                "rescue": sum(row["raw_rank"] != 1 and row["fresh_d1_rank"] == 1 for row in cases),
                "break": sum(row["raw_rank"] == 1 and row["fresh_d1_rank"] != 1 for row in cases),
                "both_correct": sum(row["raw_rank"] == row["fresh_d1_rank"] == 1 for row in cases),
                "both_wrong": sum(row["raw_rank"] != 1 and row["fresh_d1_rank"] != 1 for row in cases),
            },
        },
        "summaries_by_fold": grouped_summary(cases, "fold"),
        "summaries_by_track": grouped_summary(cases, "track"),
        "summaries_by_supergroup": grouped_summary(cases, "supergroup"),
        "added_target_identities": dict(sorted(added_identity_counts.items())),
        "added_supergroups": sorted(added_groups),
        "cases": cases,
        "bindings": {
            "oof_aggregate_sha256": sha256_file(OOF_AGGREGATE),
            "oof_aggregate_logical_sha256": str(oof_aggregate["logical_sha256"]),
            "raw_aggregate_sha256": sha256_file(RAW_AGGREGATE),
            "raw_aggregate_logical_sha256": str(raw_aggregate["logical_sha256"]),
            **r5_seals,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "source_shards_sha256": canonical_sha256(source_seals),
        },
        "access": {
            "query_target_identity_read_count": QUERY_COUNT,
            "query_supergroup_read_count": QUERY_COUNT,
            "target_join_after_oof_aggregate_seal": True,
            "target_insertion_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "external_confirmation": False,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    # A killed job may have created the directory but not the atomic result;
    # the result file, rather than an empty parent directory, is immutable.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "fixed235_fresh_d1_c128": fresh_fixed_coverage_count,
                "deep22_recovered": deep_recovered_count,
                "gate_checks": gate_checks,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
