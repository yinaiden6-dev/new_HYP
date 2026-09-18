#!/usr/bin/env python3
"""Independently aggregate eight fresh-D1 current64 A/B/C shards.

This validates the 64-query current64 staging population only. Pair64 remains
pending, so success authorizes creation of a Pair64 prejoin contract and never
claims completion of the 128-query rematerialization contract.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md"
EXACT_MAP_REUSE_ADDENDUM = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_EXACT_ROMA_MAP_CACHE_REUSE_ADDENDUM_V1_20260904.md"
V3_AUTHORITY = ROOT / "plan/ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_SCORING_EXECUTION_AUTHORITY_V3_20260903.json"
OOF_AGGREGATE = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json"
SOURCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.json"
SOURCE_VALIDATION = SOURCE_ROOT / "independent_validation.json"
REFERENCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_RESULT = REFERENCE_ROOT / "result.json"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"
OUT = OUT_ROOT / "validation.json"
SHARD_PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py"
SHARD_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py"

VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_prejoin_aggregate_v1_20260904"
SHARD_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_READY"
SHARD_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_VALIDATED"
AGGREGATE_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
SHARD_NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATION"
AGGREGATE_NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_PREJOIN_CONTRACT"
SHARD_COUNT = 8
QUERY_COUNT = 64
QUERY_COUNT_PER_SHARD = 8
CANDIDATE_COUNT = 128
CANDIDATE_PAIR_COUNT = 8192
MAP_REUSE_COUNT = 8168
MAP_COMPLETION_COUNT = 24
REFERENCE_UNION_COUNT = 2805
PAIR64_PENDING = 64
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
EXPECTED_EXACT_MAP_REUSE_ADDENDUM_SHA256 = (
    "4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6"
)

SHARD_VALIDATION_KEYS = {
    "version",
    "status",
    "claim_level",
    "shard",
    "checks",
    "query_count",
    "candidate_count",
    "map_reuse_count",
    "map_completion_count",
    "score_max_abs",
    "feature_max_abs",
    "map_replay_atol",
    "missing_map_replay_max_abs",
    "candidate_reorder_max_abs",
    "cbind_base_gap_max_abs",
    "payload_sha256",
    "receipt_sha256",
    "producer_sha256",
    "validator_sha256",
    "pair64_pending_query_count",
    "contract_population_complete",
    "scientific_GO_or_NO_GO",
    "ownership_GO_or_NO_GO",
    "automatic_stage_advance",
    "next_authorized_stage",
    "logical_sha256",
}
SHARD_CHECK_KEYS = {
    "payload_scope",
    "bindings",
    "access",
    "records_and_formula",
    "map_exact_reuse_or_independent_completion",
    "candidate_reorder_invariance",
    "cbind_base_gap_unchanged_and_complete_bundle_shift",
    "claim_boundary",
    "receipt",
}


class AggregateValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent/non-regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON input absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable aggregate validation exists: {path}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validated_authorities() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    v3 = read_json(V3_AUTHORITY)
    oof = read_json(OOF_AGGREGATE)
    source = read_json(SOURCE_MANIFEST)
    source_validation = read_json(SOURCE_VALIDATION)
    reference_result = read_json(REFERENCE_RESULT)
    reference_validation = read_json(REFERENCE_VALIDATION)
    require(
        v3.get("status") == "FROZEN_BEFORE_N2_FRESH_D1_OOF_SCORING"
        and v3.get("logical_sha256") == logical_sha256(v3)
        and v3.get("target_free_prejoin") is True
        and v3.get("prejoin_query_target_read_count") == 0
        and v3.get("prejoin_target_insertion_count") == 0,
        "V3 OOF execution authority drift",
    )
    require(
        sha256_file(EXACT_MAP_REUSE_ADDENDUM)
        == EXPECTED_EXACT_MAP_REUSE_ADDENDUM_SHA256,
        "exact RoMa-map cache-reuse addendum drift",
    )
    require(
        oof.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED"
        and oof.get("logical_sha256") == logical_sha256(oof)
        and all(oof.get("checks", {}).values())
        and oof.get("query_count") == 987
        and oof.get("bindings", {}).get("execution_authority_sha256")
        == sha256_file(V3_AUTHORITY)
        and oof.get("bindings", {}).get("execution_authority_logical_sha256")
        == v3["logical_sha256"]
        and all(value == 0 for value in oof.get("access", {}).values()),
        "fresh-D1 OOF/V3 binding drift",
    )
    require(
        source.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and source.get("logical_sha256") == logical_sha256(source)
        and source.get("contract_population_complete") is False
        and source.get("population", {}).get("query_count") == QUERY_COUNT
        and source.get("population", {}).get("pair64_pending_count")
        == PAIR64_PENDING
        and source.get("population", {}).get("fresh_candidate_pair_count")
        == CANDIDATE_PAIR_COUNT
        and source.get("population", {}).get("old_map_reuse_pair_count")
        == MAP_REUSE_COUNT
        and source.get("population", {}).get("missing_map_pair_count")
        == MAP_COMPLETION_COUNT
        and source.get("population", {}).get("fresh_reference_union_count")
        == REFERENCE_UNION_COUNT
        and source.get("bindings", {}).get("rematerialization_contract_sha256")
        == sha256_file(CONTRACT)
        and source.get("bindings", {}).get(
            "exact_map_cache_reuse_addendum_sha256"
        )
        == sha256_file(EXACT_MAP_REUSE_ADDENDUM)
        and source.get("bindings", {}).get("oof_aggregate_sha256")
        == sha256_file(OOF_AGGREGATE)
        and source.get("bindings", {}).get("oof_aggregate_logical_sha256")
        == oof["logical_sha256"]
        and all(value == 0 for value in source.get("access", {}).values())
        and source.get("scientific_GO_or_NO_GO") is None
        and source.get("automatic_stage_advance") is False,
        "source manifest authority drift",
    )
    require(
        source_validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and source_validation.get("logical_sha256")
        == logical_sha256(source_validation)
        and all(source_validation.get("checks", {}).values())
        and source_validation.get("manifest_sha256")
        == sha256_file(SOURCE_MANIFEST)
        and source_validation.get("manifest_logical_sha256")
        == source["logical_sha256"]
        and source_validation.get("contract_population_complete") is False
        and source_validation.get("pair64_pending_count") == PAIR64_PENDING
        and source_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION",
        "source-manifest independent validation drift",
    )
    require(
        reference_result.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
        and reference_result.get("logical_sha256")
        == logical_sha256(reference_result)
        and reference_result.get("contract_population_complete") is False
        and reference_result.get("current64_query_count") == QUERY_COUNT
        and reference_result.get("pair64_pending") == PAIR64_PENDING
        and reference_result.get("reference_count") == REFERENCE_UNION_COUNT
        and reference_result.get("payload_sha256")
        == sha256_file(REFERENCE_PAYLOAD)
        and reference_result.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS",
        "current64 reference-cache result drift",
    )
    require(
        reference_validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_VALIDATED"
        and reference_validation.get("logical_sha256")
        == logical_sha256(reference_validation)
        and all(reference_validation.get("checks", {}).values())
        and reference_validation.get("contract_population_complete") is False
        and reference_validation.get("current64_query_count") == QUERY_COUNT
        and reference_validation.get("pair64_pending") == PAIR64_PENDING
        and reference_validation.get("reference_count") == REFERENCE_UNION_COUNT
        and reference_validation.get("payload_sha256")
        == sha256_file(REFERENCE_PAYLOAD)
        and reference_validation.get("producer_result_sha256")
        == sha256_file(REFERENCE_RESULT)
        and reference_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS",
        "current64 reference-cache independent validation drift",
    )
    return v3, oof, source, source_validation, reference_validation


def main() -> None:
    require(not OUT.exists(), f"immutable aggregate validation exists: {OUT}")
    v3, oof, source, source_validation, reference_validation = validated_authorities()
    reference_payload = torch.load(
        REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True
    )
    reference_union = list(map(int, reference_payload.get("reference_union", [])))
    require(
        reference_payload.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
        and reference_payload.get("contract_population_complete") is False
        and reference_payload.get("pair64_pending") == PAIR64_PENDING
        and reference_union == list(map(int, source["fresh_reference_union"]))
        and len(reference_union) == len(set(reference_union))
        == REFERENCE_UNION_COUNT,
        "reference payload/current64 union drift",
    )

    source_records = sorted(
        source["records"], key=lambda item: int(item["execution_ordinal"])
    )
    source_by_query = {str(item["query_id"]): item for item in source_records}
    require(
        len(source_records) == len(source_by_query) == QUERY_COUNT,
        "source current64 record population drift",
    )

    all_records: list[dict[str, Any]] = []
    shard_seals: list[dict[str, Any]] = []
    shard_checks: list[bool] = []
    for shard in range(SHARD_COUNT):
        directory = OUT_ROOT / f"shard{shard:02d}"
        payload_path = directory / "payload.pt"
        receipt_path = directory / "receipt.json"
        validation_path = directory / "validation.json"
        require(
            payload_path.is_file()
            and receipt_path.is_file()
            and validation_path.is_file(),
            f"current64 shard {shard} outputs absent",
        )
        payload = torch.load(
            payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        receipt = read_json(receipt_path)
        validation = read_json(validation_path)
        expected_source = source_records[
            shard * QUERY_COUNT_PER_SHARD : (shard + 1) * QUERY_COUNT_PER_SHARD
        ]
        expected_query_ids = [str(item["query_id"]) for item in expected_source]
        records = payload.get("records", [])
        record_query_ids = [str(item.get("query_id")) for item in records]
        shard_validation_ok = (
            set(validation) == SHARD_VALIDATION_KEYS
            and set(validation.get("checks", {})) == SHARD_CHECK_KEYS
            and all(
                type(value) is bool and value
                for value in validation.get("checks", {}).values()
            )
            and validation.get("version")
            == "routea_n2_fresh_d1_matched_three_arm_current64_prejoin_shard_v1_20260904"
            and validation.get("status") == SHARD_VALIDATED
            and validation.get("logical_sha256") == logical_sha256(validation)
            and validation.get("shard") == shard
            and validation.get("query_count") == QUERY_COUNT_PER_SHARD
            and validation.get("candidate_count")
            == QUERY_COUNT_PER_SHARD * CANDIDATE_COUNT
            and validation.get("map_reuse_count")
            + validation.get("map_completion_count")
            == QUERY_COUNT_PER_SHARD * CANDIDATE_COUNT
            and validation.get("payload_sha256") == sha256_file(payload_path)
            and validation.get("receipt_sha256") == sha256_file(receipt_path)
            and validation.get("producer_sha256") == sha256_file(SHARD_PRODUCER)
            and validation.get("validator_sha256") == sha256_file(SHARD_VALIDATOR)
            and validation.get("pair64_pending_query_count") == PAIR64_PENDING
            and validation.get("contract_population_complete") is False
            and validation.get("scientific_GO_or_NO_GO") is None
            and validation.get("ownership_GO_or_NO_GO") is None
            and validation.get("automatic_stage_advance") is False
            and validation.get("next_authorized_stage") == SHARD_NEXT
            and float(validation.get("score_max_abs", math.inf)) <= 1e-12
            and float(validation.get("feature_max_abs", math.inf)) <= 1e-12
            and float(validation.get("map_replay_atol", math.inf)) == 1e-6
            and float(validation.get("missing_map_replay_max_abs", math.inf))
            <= 1e-6
            and float(validation.get("candidate_reorder_max_abs", math.inf))
            <= 1e-12
            and float(validation.get("cbind_base_gap_max_abs", math.inf)) == 0.0
        )
        payload_scope_ok = (
            payload.get("status") == SHARD_READY
            and payload.get("shard") == shard
            and payload.get("shard_count") == SHARD_COUNT
            and payload.get("current64_query_count") == QUERY_COUNT_PER_SHARD
            and payload.get("pair64_pending_query_count") == PAIR64_PENDING
            and payload.get("contract_population_complete") is False
            and tuple(payload.get("arms", ())) == ARMS
            and payload.get("map_replay_atol") == 1e-6
            and payload.get("scientific_GO_or_NO_GO") is None
            and payload.get("ownership_GO_or_NO_GO") is None
            and payload.get("automatic_stage_advance") is False
            and payload.get("next_authorized_stage")
            == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION"
            and record_query_ids == expected_query_ids
        )
        bindings = payload.get("bindings", {})
        bindings_ok = (
            bindings.get("contract_sha256") == sha256_file(CONTRACT)
            and bindings.get("source_manifest_sha256")
            == sha256_file(SOURCE_MANIFEST)
            and bindings.get("source_validation_sha256")
            == sha256_file(SOURCE_VALIDATION)
            and bindings.get("reference_payload_sha256")
            == sha256_file(REFERENCE_PAYLOAD)
            and bindings.get("reference_result_sha256")
            == sha256_file(REFERENCE_RESULT)
            and bindings.get("reference_validation_sha256")
            == sha256_file(REFERENCE_VALIDATION)
            and bindings.get("producer_sha256") == sha256_file(SHARD_PRODUCER)
        )
        access = payload.get("access", {})
        access_ok = (
            access.get("target_identity_read_count") == 0
            and access.get("query_supergroup_read_count") == 0
            and access.get("retrieval_outcome_read_count") == 0
            and access.get("action_read_count") == 0
            and access.get("target_insertion_count") == 0
            and access.get("model_update_count") == 0
            and access.get("external_read_count") == 0
            and access.get("sealed_read_count") == 0
            and access.get("roma_model_forward_count")
            == validation.get("map_completion_count")
        )
        receipt_ok = (
            receipt.get("status") == SHARD_READY
            and receipt.get("logical_sha256") == logical_sha256(receipt)
            and receipt.get("shard") == shard
            and receipt.get("query_count") == QUERY_COUNT_PER_SHARD
            and receipt.get("candidate_count")
            == QUERY_COUNT_PER_SHARD * CANDIDATE_COUNT
            and receipt.get("map_reuse_count")
            + receipt.get("map_completion_count")
            == QUERY_COUNT_PER_SHARD * CANDIDATE_COUNT
            and receipt.get("map_replay_atol") == 1e-6
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and receipt.get("pair64_pending_query_count") == PAIR64_PENDING
            and receipt.get("contract_population_complete") is False
            and receipt.get("access") == access
            and receipt.get("scientific_GO_or_NO_GO") is None
            and receipt.get("automatic_stage_advance") is False
            and receipt.get("next_authorized_stage")
            == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION"
        )
        records_ok = len(records) == QUERY_COUNT_PER_SHARD
        for record, expected in zip(records, expected_source, strict=True):
            axis = list(map(int, record.get("candidate_physical_rows", [])))
            candidates = record.get("candidates", [])
            candidate_positions = [
                int(item.get("candidate_position", -1)) for item in candidates
            ]
            physical_rows = [int(item.get("physical_row", -1)) for item in candidates]
            feature_ok = True
            for family, width in (
                ("real_common_features", 2),
                ("cbind_common_features", 2),
                ("real_native_features", 6),
                ("cbind_native_features", 6),
            ):
                feature_map = record.get(family, {})
                feature_ok = feature_ok and set(feature_map) == set(ARMS)
                for arm in ARMS:
                    tensor = torch.as_tensor(feature_map.get(arm))
                    feature_ok = feature_ok and (
                        tensor.shape == (CANDIDATE_COUNT - 1, width)
                        and bool(torch.isfinite(tensor).all())
                    )
            records_ok = records_ok and (
                str(record.get("query_id")) == str(expected["query_id"])
                and int(record.get("execution_ordinal"))
                == int(expected["execution_ordinal"])
                and int(record.get("heldout_fold")) == int(expected["heldout_fold"])
                and axis == list(map(int, expected["fresh_c128_physical_rows"]))
                and len(axis) == len(set(axis)) == CANDIDATE_COUNT
                and record.get("base_winner_position") == 0
                and record.get("challenger_positions") == list(range(1, 128))
                and record.get("cbind_source_positions")
                == [(position + 64) % 128 for position in range(128)]
                and len(candidates) == CANDIDATE_COUNT
                and candidate_positions == list(range(CANDIDATE_COUNT))
                and physical_rows == axis
                and int(record.get("map_reuse_count", -1))
                == int(expected["old_map_reuse_count"])
                and int(record.get("map_completion_count", -1))
                == int(expected["missing_map_pair_count"])
                and record.get("target_identity_read_count") == 0
                and record.get("query_supergroup_read_count") == 0
                and record.get("action_read_count") == 0
                and record.get("target_insertion_count") == 0
                and record.get("model_update_count") == 0
                and feature_ok
            )
        shard_checks.append(
            shard_validation_ok
            and payload_scope_ok
            and bindings_ok
            and access_ok
            and receipt_ok
            and records_ok
        )
        all_records.extend(records)
        shard_seals.append(
            {
                "shard": shard,
                "query_count": len(records),
                "candidate_count": sum(len(item["candidates"]) for item in records),
                "map_reuse_count": sum(int(item["map_reuse_count"]) for item in records),
                "map_completion_count": sum(
                    int(item["map_completion_count"]) for item in records
                ),
                "payload_sha256": sha256_file(payload_path),
                "receipt_sha256": sha256_file(receipt_path),
                "validation_sha256": sha256_file(validation_path),
                "validation_logical_sha256": validation["logical_sha256"],
            }
        )

    ordered_query_ids = [str(item["query_id"]) for item in all_records]
    expected_query_ids = [str(item["query_id"]) for item in source_records]
    roles = Counter(str(item["role_membership"]) for item in all_records)
    folds = Counter(int(item["heldout_fold"]) for item in all_records)
    candidate_pairs = sum(len(item["candidates"]) for item in all_records)
    map_reuse = sum(int(item["map_reuse_count"]) for item in all_records)
    map_completion = sum(int(item["map_completion_count"]) for item in all_records)
    observed_union = sorted(
        {
            int(row)
            for record in all_records
            for row in record["candidate_physical_rows"]
        }
    )
    target_free_records = all(
        record.get("target_identity_read_count") == 0
        and record.get("query_supergroup_read_count") == 0
        and record.get("action_read_count") == 0
        and record.get("target_insertion_count") == 0
        and record.get("model_update_count") == 0
        for record in all_records
    )
    checks = {
        "v3_oof_authority_bound": oof["bindings"]["execution_authority_sha256"]
        == sha256_file(V3_AUTHORITY)
        and oof["bindings"]["execution_authority_logical_sha256"]
        == v3["logical_sha256"],
        "source_manifest_and_independent_validation_bound": source_validation[
            "manifest_sha256"
        ]
        == sha256_file(SOURCE_MANIFEST),
        "current64_reference_cache_and_validation_bound": reference_validation[
            "payload_sha256"
        ]
        == sha256_file(REFERENCE_PAYLOAD),
        "all_eight_shards_independently_validated": len(shard_checks)
        == SHARD_COUNT
        and all(shard_checks),
        "current64_64_query_population": len(all_records)
        == len(set(ordered_query_ids))
        == QUERY_COUNT
        and ordered_query_ids == expected_query_ids,
        "train_eval_role_balance": roles == Counter({"TRAIN": 32, "EVAL": 32})
        and dict(sorted(roles.items()))
        == source.get("population", {}).get("role_counts"),
        "heldout_fold_population": {
            str(key): value for key, value in sorted(folds.items())
        }
        == source.get("population", {}).get("fold_counts"),
        "complete_8192_candidate_pairs": candidate_pairs == CANDIDATE_PAIR_COUNT,
        "map_partition_8168_plus_24": map_reuse == MAP_REUSE_COUNT
        and map_completion == MAP_COMPLETION_COUNT,
        "fresh_reference_union_2805": observed_union
        == list(map(int, source["fresh_reference_union"]))
        == reference_union,
        "zero_target_group_outcome_action_or_update": target_free_records,
        "current64_staging_not_full_contract": len(shard_seals) == SHARD_COUNT
        and all(seal["query_count"] == QUERY_COUNT_PER_SHARD for seal in shard_seals)
        and source.get("contract_population_complete") is False
        and source.get("population", {}).get("pair64_pending_count")
        == PAIR64_PENDING,
        "no_scientific_or_ownership_claim": True,
    }
    passed = all(checks.values())
    value: dict[str, Any] = {
        "version": VERSION,
        "status": (
            AGGREGATE_VALIDATED
            if passed
            else "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATION_ABORT"
        ),
        "claim_level": "INDEPENDENT_TARGET_FREE_CURRENT64_MATCHED_A_B_C_STAGING_ONLY_PAIR64_PENDING",
        "checks": checks,
        "contract_population_complete": False,
        "current64_query_count": len(all_records),
        "pair64_pending_query_count": PAIR64_PENDING,
        "candidate_pair_count": candidate_pairs,
        "map_reuse_count": map_reuse,
        "map_completion_count": map_completion,
        "reference_union_count": len(observed_union),
        "role_counts": dict(sorted(roles.items())),
        "heldout_fold_counts": {
            str(key): value for key, value in sorted(folds.items())
        },
        "query_id_order_sha256": canonical_sha256(ordered_query_ids),
        "shards": shard_seals,
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "exact_map_cache_reuse_addendum_sha256": sha256_file(
                EXACT_MAP_REUSE_ADDENDUM
            ),
            "v3_authority_sha256": sha256_file(V3_AUTHORITY),
            "v3_authority_logical_sha256": v3["logical_sha256"],
            "oof_aggregate_sha256": sha256_file(OOF_AGGREGATE),
            "oof_aggregate_logical_sha256": oof["logical_sha256"],
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "source_manifest_logical_sha256": source["logical_sha256"],
            "source_validation_sha256": sha256_file(SOURCE_VALIDATION),
            "source_validation_logical_sha256": source_validation[
                "logical_sha256"
            ],
            "reference_payload_sha256": sha256_file(REFERENCE_PAYLOAD),
            "reference_result_sha256": sha256_file(REFERENCE_RESULT),
            "reference_validation_sha256": sha256_file(REFERENCE_VALIDATION),
            "reference_validation_logical_sha256": reference_validation[
                "logical_sha256"
            ],
            "shard_producer_sha256": sha256_file(SHARD_PRODUCER),
            "shard_validator_sha256": sha256_file(SHARD_VALIDATOR),
            "aggregate_validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "access": {
            "target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "retrieval_outcome_read_count": 0,
            "action_read_count": 0,
            "target_insertion_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": AGGREGATE_NEXT if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(
        json.dumps(
            {"status": value["status"], "checks": checks}, sort_keys=True
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
