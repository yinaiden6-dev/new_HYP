#!/usr/bin/env python3
"""Freeze the target-free source manifest for fresh-D1 matched A/B/C.

This stage performs no model forward and reads no target, supergroup,
retrieval outcome, or action.  It only joins already sealed target-free
sources by ``query_id`` and records which historical RoMa maps/reference
tokens can be reused and which exact pairs/rows must be materialized later.
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
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
OUT = OUT_ROOT / "manifest.json"

VERSION = "routea_n2_fresh_d1_matched_three_arm_source_manifest_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
EXPECTED_QUERY_COUNT = 64
EXPECTED_FRESH_PAIR_COUNT = 8192
EXPECTED_FRESH_REFERENCE_UNION = 2805
EXPECTED_MAP_REUSE_COUNT = 8168
EXPECTED_MAP_MISSING_COUNT = 24
EXPECTED_MAP_MISSING_QUERY_COUNT = 17
EXPECTED_REFERENCE_REUSE_COUNT = 2801
EXPECTED_REFERENCE_MISSING_ROWS = [467, 1488, 2435, 5159]


class SourceManifestError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceManifestError(message)


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
        require(path.is_file() and not path.is_symlink() and path.read_text() == encoded, "immutable manifest drift")
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


def validated_oof_records() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregate = read_json(OOF_AGGREGATE)
    require(
        aggregate.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED"
        and aggregate.get("query_count") == 987
        and isinstance(aggregate.get("checks"), dict)
        and all(value is True for value in aggregate["checks"].values())
        and aggregate.get("logical_sha256") == logical_sha256(aggregate),
        "fresh-D1 OOF aggregate authority failed",
    )
    seals = {int(item["shard"]): item for item in aggregate.get("shards", [])}
    require(set(seals) == set(range(16)), "fresh-D1 OOF shard seal population drift")
    records: dict[str, dict[str, Any]] = {}
    source_seals: list[dict[str, Any]] = []
    for shard in range(16):
        base = OOF_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        validation_path = base / "validation.json"
        seal = seals[shard]
        require(
            seal.get("payload_sha256") == sha256_file(payload_path)
            and seal.get("receipt_sha256") == sha256_file(receipt_path)
            and seal.get("validation_sha256") == sha256_file(validation_path),
            f"fresh-D1 OOF shard {shard} seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        require(payload.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_SHARD_READY", "OOF payload status drift")
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            require(query_id not in records, f"duplicate OOF query_id: {query_id}")
            records[query_id] = {
                "record": record,
                "source_shard": shard,
                "source_payload_sha256": seal["payload_sha256"],
            }
        source_seals.append(
            {
                "shard": shard,
                "payload_sha256": seal["payload_sha256"],
                "receipt_sha256": seal["receipt_sha256"],
                "validation_sha256": seal["validation_sha256"],
            }
        )
    require(len(records) == 987, "fresh-D1 OOF query population drift")
    return records, source_seals, aggregate


def validated_roster() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation = read_json(ROSTER_VALIDATION)
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("query_count") == EXPECTED_QUERY_COUNT
        and validation.get("roles") == {"EVAL": 32, "TRAIN": 32}
        and validation.get("target_label_read_count") == 0
        and validation.get("model_update_count") == 0
        and isinstance(validation.get("checks"), dict)
        and all(value is True for value in validation["checks"].values())
        and validation.get("logical_sha256") == logical_sha256(validation),
        "current64 target-free roster validation failed",
    )
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(set(seals) == set(range(8)), "current64 roster shard seal population drift")
    records: list[dict[str, Any]] = []
    source_seals: list[dict[str, Any]] = []
    for shard in range(8):
        base = ROSTER_ROOT / f"shard{shard:02d}"
        payload_path = base / "payload.pt"
        receipt_path = base / "receipt.json"
        shard_validation_path = base / "validation.json"
        seal = seals[shard]
        require(
            seal.get("payload_sha256") == sha256_file(payload_path)
            and seal.get("receipt_sha256") == sha256_file(receipt_path)
            and seal.get("validation_sha256") == sha256_file(shard_validation_path),
            f"current64 roster shard {shard} seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload.get("records", []):
            require(record.get("target_or_label_read_count") == 0, "roster record is not target-free")
            records.append(
                {
                    "role": str(record["role"]),
                    "execution_ordinal": int(record["execution_ordinal"]),
                    "query_id": str(record["query_id"]),
                    "heldout_fold": int(record["heldout_fold"]),
                    "query_source_sha256": str(record["query_source_sha256"]),
                    "query_grid_shape": list(map(int, record["query_grid_shape"])),
                    "source_shard": shard,
                    "source_payload_sha256": seal["payload_sha256"],
                }
            )
        source_seals.append({"shard": shard, **seal})
    records.sort(key=lambda item: item["execution_ordinal"])
    require(
        len(records) == len({item["query_id"] for item in records}) == EXPECTED_QUERY_COUNT
        and len({item["execution_ordinal"] for item in records}) == EXPECTED_QUERY_COUNT
        and Counter(item["role"] for item in records) == Counter({"TRAIN": 32, "EVAL": 32}),
        "current64 target-free roster population drift",
    )
    return records, source_seals, validation


def validated_old_map_bank() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validation = read_json(MAP_VALIDATION)
    roster_validation = read_json(ROSTER_VALIDATION)
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and validation.get("query_count") == EXPECTED_QUERY_COUNT
        and validation.get("target_role_read_count") == 0
        and validation.get("model_update_count") == 0
        and isinstance(validation.get("checks"), dict)
        and all(value is True for value in validation["checks"].values())
        and validation.get("logical_sha256") == logical_sha256(validation),
        "old target-free RoMa-map bank validation failed",
    )
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    roster_seals = {
        int(item["shard"]): item for item in roster_validation.get("shards", [])
    }
    require(set(seals) == set(range(8)), "old RoMa-map shard population drift")
    require(set(roster_seals) == set(range(8)), "old map bound-roster shard population drift")
    records: dict[str, dict[str, Any]] = {}
    source_seals: list[dict[str, Any]] = []
    for shard in range(8):
        payload_path = MAP_ROOT / f"shard{shard:02d}/payload.pt"
        seal = seals[shard]
        require(seal.get("payload_sha256") == sha256_file(payload_path), f"old map shard {shard} payload drift")
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        payload_bindings = payload.get("bindings", {})
        require(
            isinstance(payload_bindings, Mapping)
            and payload_bindings.get("source_validation_sha256")
            == sha256_file(ROSTER_VALIDATION)
            and payload_bindings.get("source_payload_sha256")
            == roster_seals[shard].get("payload_sha256"),
            f"old map shard {shard} roster-source binding absent",
        )
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            require(query_id not in records and record.get("target_role_read_count") == 0, "old map record drift")
            candidates = {int(item["physical_row"]): item for item in record["candidates"]}
            require(set(candidates) == set(map(int, record["union_physical_rows"])), "old map pair population drift")
            records[query_id] = {
                "execution_ordinal": int(record["execution_ordinal"]),
                "heldout_fold": int(record["heldout_fold"]),
                "source_shard": shard,
                "source_payload_sha256": seal["payload_sha256"],
                "old_map_bound_roster_payload_sha256": str(
                    payload_bindings["source_payload_sha256"]
                ),
                "candidates": candidates,
            }
        source_seals.append({"shard": shard, **seal})
    require(len(records) == EXPECTED_QUERY_COUNT, "old map-bank query population drift")
    return records, source_seals, validation


def validated_reference_cache() -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    validation = read_json(REFERENCE_VALIDATION)
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and validation.get("reference_count") == 3004
        and validation.get("target_identity_read_count") == 0
        and validation.get("model_update_count") == 0
        and isinstance(validation.get("checks"), dict)
        and all(value is True for value in validation["checks"].values())
        and validation.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and validation.get("logical_sha256") == logical_sha256(validation),
        "old reference cache validation failed",
    )
    payload = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    references = {int(row): item for row, item in payload.get("references", {}).items()}
    require(
        payload.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_READY"
        and len(references) == 3004
        and sorted(references) == list(map(int, payload.get("reference_union", [])))
        and payload.get("access", {}).get("target_identity_read_count") == 0
        and payload.get("access", {}).get("target_role_read_count") == 0,
        "old reference cache payload drift",
    )
    return references, payload, validation


def build_manifest() -> dict[str, Any]:
    require(
        sha256_file(REMATERIALIZATION_CONTRACT) == EXPECTED_REMATERIALIZATION_CONTRACT_SHA256,
        "fresh-D1 matched-three-arm rematerialization contract drift",
    )
    require(
        sha256_file(EXACT_MAP_CACHE_REUSE_ADDENDUM)
        == EXPECTED_EXACT_MAP_CACHE_REUSE_ADDENDUM_SHA256,
        "exact RoMa-map cache reuse addendum drift",
    )
    oof, oof_seals, oof_aggregate = validated_oof_records()
    roster, roster_seals, roster_validation = validated_roster()
    old_maps, map_seals, map_validation = validated_old_map_bank()
    references, reference_payload, reference_validation = validated_reference_cache()

    require(set(item["query_id"] for item in roster) == set(old_maps), "roster/map query_id set drift")
    output_records: list[dict[str, Any]] = []
    reuse_pairs: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    fresh_reference_rows: set[int] = set()
    for roster_record in roster:
        query_id = roster_record["query_id"]
        fresh_source = oof.get(query_id)
        old_source = old_maps.get(query_id)
        require(fresh_source is not None and old_source is not None, f"query_id source join failed: {query_id}")
        fresh = fresh_source["record"]
        require(
            int(fresh["heldout_fold"]) == roster_record["heldout_fold"] == old_source["heldout_fold"]
            and old_source["execution_ordinal"] == roster_record["execution_ordinal"]
            and old_source["old_map_bound_roster_payload_sha256"]
            == roster_record["source_payload_sha256"]
            and tuple(fresh["grid_shape"]) == tuple(roster_record["query_grid_shape"]),
            f"fresh/roster/map execution, fold or grid drift: {query_id}",
        )
        axis = list(map(int, fresh["natural_c128_representative_physical_rows"].tolist()))
        require(len(axis) == len(set(axis)) == 128 and all(0 <= row < 5413 for row in axis), "fresh C128 axis drift")
        fresh_reference_rows.update(axis)
        query_reuse = 0
        query_missing = 0
        for position, physical_row in enumerate(axis):
            candidate = old_source["candidates"].get(physical_row)
            if candidate is None:
                query_missing += 1
                missing_pairs.append(
                    {
                        "query_id": query_id,
                        "execution_ordinal": roster_record["execution_ordinal"],
                        "heldout_fold": roster_record["heldout_fold"],
                        "fresh_candidate_position": position,
                        "physical_row": physical_row,
                        "reference_in_old_cache": physical_row in references,
                    }
                )
            else:
                query_reuse += 1
                reference = references.get(physical_row)
                require(
                    int(candidate["query_map"].numel()) == int(fresh["adapted_image_tokens"].shape[0]),
                    f"reused query-map length drift: {query_id}/{physical_row}",
                )
                require(
                    reference is not None
                    and int(candidate["reference_map"].numel()) == int(reference["tokens"].shape[0])
                    and str(candidate["reference_tokens_sha256"]) == str(reference["tokens_sha256"]),
                    f"reused reference-map/token drift: {query_id}/{physical_row}",
                )
                require(
                    str(candidate["reference_source_image_sha256"])
                    == str(reference["source_image_sha256"])
                    and str(candidate["reference_source_logical_sha256"])
                    == str(reference["source_logical_sha256"]),
                    f"reused reference source-image receipt drift: {query_id}/{physical_row}",
                )
                reuse_pairs.append(
                    {
                        "query_id": query_id,
                        "execution_ordinal": roster_record["execution_ordinal"],
                        "heldout_fold": roster_record["heldout_fold"],
                        "fresh_candidate_position": position,
                        "physical_row": physical_row,
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
        output_records.append(
            {
                "role": roster_record["role"],
                "execution_ordinal": roster_record["execution_ordinal"],
                "query_id": query_id,
                "heldout_fold": roster_record["heldout_fold"],
                "query_source_sha256": roster_record["query_source_sha256"],
                "query_grid_shape": roster_record["query_grid_shape"],
                "oof_query_ordinal": int(fresh["query_ordinal"]),
                "oof_source_shard": int(fresh_source["source_shard"]),
                "oof_source_payload_sha256": str(fresh_source["source_payload_sha256"]),
                "oof_checkpoint_sha256": str(fresh["oof_checkpoint_sha256"]),
                "adapted_image_tokens_sha256": str(fresh["adapted_image_tokens_sha256"]),
                "physical_row_scores_sha256": str(fresh["physical_row_scores_sha256"]),
                "fresh_c128_physical_rows": axis,
                "fresh_c128_physical_rows_sha256": canonical_sha256(axis),
                "old_map_reuse_count": query_reuse,
                "missing_map_pair_count": query_missing,
            }
        )

    output_records.sort(key=lambda item: item["execution_ordinal"])
    reuse_pairs.sort(key=lambda item: (item["execution_ordinal"], item["fresh_candidate_position"]))
    missing_pairs.sort(key=lambda item: (item["execution_ordinal"], item["fresh_candidate_position"]))
    reference_union = sorted(fresh_reference_rows)
    reference_reuse_rows = sorted(fresh_reference_rows & set(references))
    reference_missing_rows = sorted(fresh_reference_rows - set(references))
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
        len(output_records) == EXPECTED_QUERY_COUNT
        and sum(len(item["fresh_c128_physical_rows"]) for item in output_records) == EXPECTED_FRESH_PAIR_COUNT
        and len(reference_union) == EXPECTED_FRESH_REFERENCE_UNION
        and len(reuse_pairs) == EXPECTED_MAP_REUSE_COUNT
        and len(missing_pairs) == EXPECTED_MAP_MISSING_COUNT
        and len({item["query_id"] for item in missing_pairs}) == EXPECTED_MAP_MISSING_QUERY_COUNT
        and len(reference_reuse_rows) == EXPECTED_REFERENCE_REUSE_COUNT
        and reference_missing_rows == EXPECTED_REFERENCE_MISSING_ROWS,
        "fresh-D1 source-manifest frozen population failed",
    )
    value: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "claim_level": "TARGET_FREE_CURRENT64_SOURCE_MANIFEST_STAGING_NO_MODEL_OR_SCIENTIFIC_CLAIM",
        "population": {
            "query_count": len(output_records),
            "current64_query_count": len(output_records),
            "pair64_pending_count": 64,
            "role_counts": dict(sorted(Counter(item["role"] for item in output_records).items())),
            "fold_counts": {str(key): count for key, count in sorted(Counter(item["heldout_fold"] for item in output_records).items())},
            "fresh_candidate_pair_count": EXPECTED_FRESH_PAIR_COUNT,
            "fresh_reference_union_count": len(reference_union),
            "old_map_reuse_pair_count": len(reuse_pairs),
            "missing_map_pair_count": len(missing_pairs),
            "missing_map_query_count": len({item["query_id"] for item in missing_pairs}),
            "old_reference_cache_count": len(references),
            "reference_reuse_count": len(reference_reuse_rows),
            "reference_missing_count": len(reference_missing_rows),
        },
        "records": output_records,
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
            "producer_sha256": sha256_file(Path(__file__).resolve()),
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
    value["logical_sha256"] = logical_sha256(value)
    return value


def main() -> None:
    value = build_manifest()
    atomic_json(OUT, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "query_count": value["population"]["query_count"],
                "map_reuse": value["population"]["old_map_reuse_pair_count"],
                "map_missing": value["population"]["missing_map_pair_count"],
                "reference_missing_rows": value["reference_reuse"]["missing_rows"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
