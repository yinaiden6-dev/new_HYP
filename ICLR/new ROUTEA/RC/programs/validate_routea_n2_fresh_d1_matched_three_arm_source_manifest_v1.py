#!/usr/bin/env python3
"""Independent validator for the fresh-D1 matched-three-arm source manifest.

The producer is never imported.  All query, candidate-pair, reference-row and
source-seal ledgers are reconstructed directly from immutable predecessors.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
REMATERIALIZATION_CONTRACT = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md"
EXPECTED_REMATERIALIZATION_CONTRACT_SHA256 = "6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3"
EXACT_MAP_CACHE_REUSE_ADDENDUM = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_EXACT_ROMA_MAP_CACHE_REUSE_ADDENDUM_V1_20260904.md"
EXPECTED_EXACT_MAP_CACHE_REUSE_ADDENDUM_SHA256 = "4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6"
OOF_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
OOF_AGGREGATE = OOF_ROOT / "independent_aggregate_validation.json"
ROSTER_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
ROSTER_VALIDATION = ROSTER_ROOT / "validation.json"
MAP_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
MAP_VALIDATION = MAP_ROOT / "validation.json"
REFERENCE_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_source_manifest_v1.py"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
MANIFEST = OUT_ROOT / "manifest.json"
OUT = OUT_ROOT / "independent_validation.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_source_manifest_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"


class SourceManifestValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceManifestValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"input absent/non-regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"JSON input absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_text() == encoded, "immutable validation drift")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reconstruct() -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        sha256_file(REMATERIALIZATION_CONTRACT) == EXPECTED_REMATERIALIZATION_CONTRACT_SHA256,
        "independent rematerialization contract drift",
    )
    require(
        sha256_file(EXACT_MAP_CACHE_REUSE_ADDENDUM)
        == EXPECTED_EXACT_MAP_CACHE_REUSE_ADDENDUM_SHA256,
        "independent exact RoMa-map cache reuse addendum drift",
    )
    oof_aggregate = read_json(OOF_AGGREGATE)
    roster_validation = read_json(ROSTER_VALIDATION)
    map_validation = read_json(MAP_VALIDATION)
    reference_validation = read_json(REFERENCE_VALIDATION)
    require(
        oof_aggregate.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED"
        and oof_aggregate.get("query_count") == 987
        and all(oof_aggregate.get("checks", {}).values())
        and oof_aggregate.get("logical_sha256") == logical_sha256(oof_aggregate),
        "independent OOF aggregate validation failed",
    )
    require(
        roster_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and roster_validation.get("query_count") == 64
        and roster_validation.get("roles") == {"EVAL": 32, "TRAIN": 32}
        and roster_validation.get("target_label_read_count") == 0
        and roster_validation.get("model_update_count") == 0
        and all(roster_validation.get("checks", {}).values())
        and roster_validation.get("logical_sha256") == logical_sha256(roster_validation),
        "independent current64 roster validation failed",
    )
    require(
        map_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and map_validation.get("query_count") == 64
        and map_validation.get("target_role_read_count") == 0
        and map_validation.get("model_update_count") == 0
        and all(map_validation.get("checks", {}).values())
        and map_validation.get("logical_sha256") == logical_sha256(map_validation),
        "independent old-map authority validation failed",
    )
    require(
        reference_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and reference_validation.get("reference_count") == 3004
        and reference_validation.get("target_identity_read_count") == 0
        and reference_validation.get("model_update_count") == 0
        and all(reference_validation.get("checks", {}).values())
        and reference_validation.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and reference_validation.get("logical_sha256") == logical_sha256(reference_validation),
        "independent reference-cache authority validation failed",
    )

    oof_sealed = {int(item["shard"]): item for item in oof_aggregate["shards"]}
    roster_sealed = {int(item["shard"]): item for item in roster_validation["shards"]}
    map_sealed = {int(item["shard"]): item for item in map_validation["shards"]}
    require(set(oof_sealed) == set(range(16)), "OOF seal population drift")
    require(set(roster_sealed) == set(map_sealed) == set(range(8)), "current64/map seal population drift")

    oof: dict[str, dict[str, Any]] = {}
    oof_seals: list[dict[str, Any]] = []
    for shard in range(16):
        base = OOF_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        validation_path = base / "validation.json"
        seal = oof_sealed[shard]
        require(
            seal["payload_sha256"] == sha256_file(payload_path)
            and seal["receipt_sha256"] == sha256_file(receipt_path)
            and seal["validation_sha256"] == sha256_file(validation_path),
            f"OOF shard {shard} source seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        require(payload.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_SHARD_READY", "OOF payload drift")
        for record in payload["records"]:
            query_id = str(record["query_id"])
            require(query_id not in oof, "duplicate OOF query_id")
            oof[query_id] = {"record": record, "source_shard": shard, "source_payload_sha256": seal["payload_sha256"]}
        oof_seals.append(
            {"shard": shard, "payload_sha256": seal["payload_sha256"], "receipt_sha256": seal["receipt_sha256"], "validation_sha256": seal["validation_sha256"]}
        )
    require(len(oof) == 987, "independent OOF query population drift")

    roster: list[dict[str, Any]] = []
    roster_seals: list[dict[str, Any]] = []
    map_bank: dict[str, dict[str, Any]] = {}
    map_seals: list[dict[str, Any]] = []
    for shard in range(8):
        roster_base = ROSTER_ROOT / f"shard{shard:02d}"
        roster_payload_path = roster_base / "payload.pt"
        roster_receipt_path = roster_base / "receipt.json"
        roster_shard_validation_path = roster_base / "validation.json"
        roster_seal = roster_sealed[shard]
        require(
            roster_seal["payload_sha256"] == sha256_file(roster_payload_path)
            and roster_seal["receipt_sha256"] == sha256_file(roster_receipt_path)
            and roster_seal["validation_sha256"] == sha256_file(roster_shard_validation_path),
            f"roster shard {shard} source seal drift",
        )
        roster_payload = torch.load(roster_payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in roster_payload["records"]:
            require(record.get("target_or_label_read_count") == 0, "roster target access drift")
            roster.append(
                {
                    "role": str(record["role"]),
                    "execution_ordinal": int(record["execution_ordinal"]),
                    "query_id": str(record["query_id"]),
                    "heldout_fold": int(record["heldout_fold"]),
                    "query_source_sha256": str(record["query_source_sha256"]),
                    "query_grid_shape": list(map(int, record["query_grid_shape"])),
                    "source_shard": shard,
                    "source_payload_sha256": roster_seal["payload_sha256"],
                }
            )
        roster_seals.append({"shard": shard, **roster_seal})

        map_payload_path = MAP_ROOT / f"shard{shard:02d}/payload.pt"
        map_seal = map_sealed[shard]
        require(map_seal["payload_sha256"] == sha256_file(map_payload_path), f"map shard {shard} source seal drift")
        map_payload = torch.load(map_payload_path, map_location="cpu", weights_only=False, mmap=True)
        map_payload_bindings = map_payload.get("bindings", {})
        require(
            isinstance(map_payload_bindings, Mapping)
            and map_payload_bindings.get("source_validation_sha256")
            == sha256_file(ROSTER_VALIDATION)
            and map_payload_bindings.get("source_payload_sha256")
            == roster_seal.get("payload_sha256"),
            f"independent old-map shard {shard} roster binding absent",
        )
        for record in map_payload["records"]:
            query_id = str(record["query_id"])
            require(query_id not in map_bank and record.get("target_role_read_count") == 0, "map record drift")
            candidates = {int(item["physical_row"]): item for item in record["candidates"]}
            require(set(candidates) == set(map(int, record["union_physical_rows"])), "map candidate union drift")
            map_bank[query_id] = {
                "execution_ordinal": int(record["execution_ordinal"]),
                "heldout_fold": int(record["heldout_fold"]),
                "source_shard": shard,
                "source_payload_sha256": map_seal["payload_sha256"],
                "old_map_bound_roster_payload_sha256": str(
                    map_payload_bindings["source_payload_sha256"]
                ),
                "candidates": candidates,
            }
        map_seals.append({"shard": shard, **map_seal})

    roster.sort(key=lambda item: item["execution_ordinal"])
    require(
        len(roster) == len({item["query_id"] for item in roster}) == len(map_bank) == 64
        and set(item["query_id"] for item in roster) == set(map_bank)
        and Counter(item["role"] for item in roster) == Counter({"TRAIN": 32, "EVAL": 32}),
        "independent roster/map query population drift",
    )

    reference_payload = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    references = {int(row): value for row, value in reference_payload["references"].items()}
    require(
        reference_payload.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and len(references) == 3004
        and sorted(references) == list(map(int, reference_payload["reference_union"]))
        and reference_payload.get("access", {}).get("target_identity_read_count") == 0
        and reference_payload.get("access", {}).get("target_role_read_count") == 0,
        "independent reference payload validation failed",
    )

    records: list[dict[str, Any]] = []
    reuse_pairs: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    reference_union_set: set[int] = set()
    for roster_record in roster:
        query_id = roster_record["query_id"]
        fresh_source = oof.get(query_id)
        old_source = map_bank.get(query_id)
        require(fresh_source is not None and old_source is not None, f"independent query_id join failed: {query_id}")
        fresh = fresh_source["record"]
        require(
            int(fresh["heldout_fold"]) == roster_record["heldout_fold"] == old_source["heldout_fold"]
            and old_source["execution_ordinal"] == roster_record["execution_ordinal"]
            and old_source["old_map_bound_roster_payload_sha256"]
            == roster_record["source_payload_sha256"]
            and tuple(fresh["grid_shape"]) == tuple(roster_record["query_grid_shape"]),
            "independent execution/fold/grid join drift",
        )
        axis = list(map(int, fresh["natural_c128_representative_physical_rows"].tolist()))
        require(len(axis) == len(set(axis)) == 128, "independent fresh C128 drift")
        reference_union_set.update(axis)
        reused = missing = 0
        for position, row in enumerate(axis):
            candidate = old_source["candidates"].get(row)
            if candidate is None:
                missing += 1
                missing_pairs.append(
                    {
                        "query_id": query_id,
                        "execution_ordinal": roster_record["execution_ordinal"],
                        "heldout_fold": roster_record["heldout_fold"],
                        "fresh_candidate_position": position,
                        "physical_row": row,
                        "reference_in_old_cache": row in references,
                    }
                )
            else:
                reused += 1
                reference = references.get(row)
                require(candidate["query_map"].numel() == fresh["adapted_image_tokens"].shape[0], "query-map/token length drift")
                require(
                    reference is not None
                    and candidate["reference_map"].numel() == reference["tokens"].shape[0]
                    and str(candidate["reference_tokens_sha256"]) == str(reference["tokens_sha256"]),
                    "reference-map/token length or hash drift",
                )
                require(
                    str(candidate["reference_source_image_sha256"])
                    == str(reference["source_image_sha256"])
                    and str(candidate["reference_source_logical_sha256"])
                    == str(reference["source_logical_sha256"]),
                    "reference source-image receipt drift",
                )
                reuse_pairs.append(
                    {
                        "query_id": query_id,
                        "execution_ordinal": roster_record["execution_ordinal"],
                        "heldout_fold": roster_record["heldout_fold"],
                        "fresh_candidate_position": position,
                        "physical_row": row,
                        "old_map_source_shard": old_source["source_shard"],
                        "old_map_source_payload_sha256": old_source["source_payload_sha256"],
                        "old_map_bound_roster_payload_sha256": old_source[
                            "old_map_bound_roster_payload_sha256"
                        ],
                        "query_source_sha256": roster_record["query_source_sha256"],
                        "query_map_sha256": str(candidate["query_map_sha256"]),
                        "reference_map_sha256": str(candidate["reference_map_sha256"]),
                        "reference_tokens_sha256": str(candidate["reference_tokens_sha256"]),
                        "reference_source_image_sha256": str(
                            candidate["reference_source_image_sha256"]
                        ),
                        "reference_source_logical_sha256": str(
                            candidate["reference_source_logical_sha256"]
                        ),
                    }
                )
        records.append(
            {
                "role": roster_record["role"],
                "execution_ordinal": roster_record["execution_ordinal"],
                "query_id": query_id,
                "heldout_fold": roster_record["heldout_fold"],
                "query_source_sha256": roster_record["query_source_sha256"],
                "query_grid_shape": roster_record["query_grid_shape"],
                "oof_query_ordinal": int(fresh["query_ordinal"]),
                "oof_source_shard": fresh_source["source_shard"],
                "oof_source_payload_sha256": fresh_source["source_payload_sha256"],
                "oof_checkpoint_sha256": str(fresh["oof_checkpoint_sha256"]),
                "adapted_image_tokens_sha256": str(fresh["adapted_image_tokens_sha256"]),
                "physical_row_scores_sha256": str(fresh["physical_row_scores_sha256"]),
                "fresh_c128_physical_rows": axis,
                "fresh_c128_physical_rows_sha256": canonical_sha256(axis),
                "old_map_reuse_count": reused,
                "missing_map_pair_count": missing,
            }
        )

    records.sort(key=lambda item: item["execution_ordinal"])
    reuse_pairs.sort(key=lambda item: (item["execution_ordinal"], item["fresh_candidate_position"]))
    missing_pairs.sort(key=lambda item: (item["execution_ordinal"], item["fresh_candidate_position"]))
    reference_union = sorted(reference_union_set)
    reference_reuse_rows = sorted(reference_union_set & set(references))
    reference_missing_rows = sorted(reference_union_set - set(references))
    reference_reuse_manifest = [
        {
            "physical_row": row,
            "tokens_sha256": str(references[row]["tokens_sha256"]),
            "source_image_sha256": str(references[row]["source_image_sha256"]),
            "grid_shape": list(map(int, references[row]["grid_shape"])),
            "source_kind": str(references[row]["source_kind"]),
            "source_logical_sha256": str(references[row]["source_logical_sha256"]),
        }
        for row in reference_reuse_rows
    ]
    require(
        len(records) == 64
        and len(reuse_pairs) == 8168
        and len(missing_pairs) == 24
        and len({item["query_id"] for item in missing_pairs}) == 17
        and len(reference_union) == 2805
        and len(reference_reuse_rows) == 2801
        and reference_missing_rows == [467, 1488, 2435, 5159],
        "independent frozen source population failed",
    )
    expected = {
        "version": VERSION,
        "status": READY,
        "claim_level": "TARGET_FREE_CURRENT64_SOURCE_MANIFEST_STAGING_NO_MODEL_OR_SCIENTIFIC_CLAIM",
        "population": {
            "query_count": 64,
            "current64_query_count": 64,
            "pair64_pending_count": 64,
            "role_counts": dict(sorted(Counter(item["role"] for item in records).items())),
            "fold_counts": {str(key): count for key, count in sorted(Counter(item["heldout_fold"] for item in records).items())},
            "fresh_candidate_pair_count": 8192,
            "fresh_reference_union_count": 2805,
            "old_map_reuse_pair_count": 8168,
            "missing_map_pair_count": 24,
            "missing_map_query_count": 17,
            "old_reference_cache_count": 3004,
            "reference_reuse_count": 2801,
            "reference_missing_count": 4,
        },
        "records": records,
        "fresh_reference_union": reference_union,
        "fresh_reference_union_sha256": canonical_sha256(reference_union),
        "old_map_reuse_pairs": reuse_pairs,
        "old_map_reuse_pairs_sha256": canonical_sha256(reuse_pairs),
        "missing_map_pairs": missing_pairs,
        "missing_map_pairs_sha256": canonical_sha256(missing_pairs),
        "reference_reuse": {
            "rows": reference_reuse_rows,
            "rows_sha256": canonical_sha256(reference_reuse_rows),
            "manifest_sha256": canonical_sha256(reference_reuse_manifest),
            "missing_rows": reference_missing_rows,
            "missing_rows_sha256": canonical_sha256(reference_missing_rows),
        },
        "bindings": {
            "rematerialization_contract_sha256": sha256_file(REMATERIALIZATION_CONTRACT),
            "exact_map_cache_reuse_addendum_sha256": sha256_file(EXACT_MAP_CACHE_REUSE_ADDENDUM),
            "oof_aggregate_sha256": sha256_file(OOF_AGGREGATE),
            "oof_aggregate_logical_sha256": str(oof_aggregate["logical_sha256"]),
            "oof_shards_sha256": canonical_sha256(oof_seals),
            "current64_validation_sha256": sha256_file(ROSTER_VALIDATION),
            "current64_validation_logical_sha256": str(roster_validation["logical_sha256"]),
            "current64_shards_sha256": canonical_sha256(roster_seals),
            "old_map_validation_sha256": sha256_file(MAP_VALIDATION),
            "old_map_validation_logical_sha256": str(map_validation["logical_sha256"]),
            "old_map_shards_sha256": canonical_sha256(map_seals),
            "reference_payload_sha256": sha256_file(REFERENCE_PAYLOAD),
            "reference_validation_sha256": sha256_file(REFERENCE_VALIDATION),
            "reference_validation_logical_sha256": str(reference_validation["logical_sha256"]),
            "reference_payload_binding_sha256": str(reference_validation["payload_sha256"]),
            "producer_sha256": sha256_file(PRODUCER),
        },
        "access": {
            "target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "retrieval_outcome_read_count": 0,
            "action_read_count": 0,
            "model_forward_count": 0,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "contract_population_complete": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION",
        "logical_sha256": "",
    }
    expected["logical_sha256"] = logical_sha256(expected)
    diagnostics = {
        "query_id_join_count": len(records),
        "map_reuse_pair_count": len(reuse_pairs),
        "missing_map_pair_count": len(missing_pairs),
        "missing_map_query_count": len({item["query_id"] for item in missing_pairs}),
        "reference_union_count": len(reference_union),
        "reference_reuse_count": len(reference_reuse_rows),
        "reference_missing_rows": reference_missing_rows,
    }
    return expected, diagnostics


def main() -> None:
    manifest = read_json(MANIFEST)
    expected, diagnostics = reconstruct()
    require(manifest == expected, "independent source-manifest reconstruction differs")
    checks = {
        "producer_not_imported": True,
        "v3_oof_aggregate_and_all_shards_sealed": True,
        "current64_target_free_roster_reconstructed": True,
        "query_id_only_cross_axis_join": True,
        "fresh_c128_64x128_exact": True,
        "fresh_reference_union_2805": True,
        "old_map_reuse_pairs_8168": True,
        "missing_map_pairs_24_across_17_queries": True,
        "old_reference_rows_reused_2801": True,
        "missing_reference_rows_exact_four": True,
        "map_lengths_match_fresh_query_tokens": True,
        "source_bindings_and_hashes_exact": True,
        "zero_target_group_outcome_action_access": True,
        "zero_model_forward_or_update": True,
        "no_external_or_sealed_access": True,
        "current64_staging_and_pair64_pending_explicit": expected.get("contract_population_complete") is False
        and expected.get("population", {}).get("pair64_pending_count") == 64,
        "no_scientific_claim_or_automatic_transition": True,
    }
    require(all(checks.values()), "one or more source-manifest checks failed")
    value: dict[str, Any] = {
        "version": VERSION,
        "status": VALIDATED,
        "claim_level": "INDEPENDENT_TARGET_FREE_SOURCE_MANIFEST_VALIDATION_NO_MODEL_OR_SCIENTIFIC_CLAIM",
        "checks": checks,
        "diagnostics": diagnostics,
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_logical_sha256": str(manifest["logical_sha256"]),
        "producer_sha256": sha256_file(PRODUCER),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "model_forward_count": 0,
        "model_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "contract_population_complete": False,
        "pair64_pending_count": 64,
        "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "diagnostics": diagnostics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
