#!/usr/bin/env python3
"""Aggregate 16 independently validated N2 token-cache shards."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
# Keep the historical D1 package root ahead of the separate ``ROUTEA/route_a``
# package.  Repeated ``insert(0, ...)`` used to reverse the written order and
# made this depend on the ambient PYTHONPATH: ``ROUTEA/route_a`` then shadowed
# ``route_a_core/route_a``.  Slice insertion gives one explicit, stable import
# order while the historical runner itself is still loaded from its absolute
# file path by ``run_a0_fold_d1.import_runner``.
_IMPORT_ROOTS = (
    ROOT / "src",
    ROOT / "programs",
    ROUTEA / "route_a_core",
)
for _path in _IMPORT_ROOTS:
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
sys.path[:0] = [str(_path) for _path in _IMPORT_ROOTS]

import run_a0_fold_d1 as d1_wrapper  # noqa: E402
from rc_aslo_xf.n2_current_runtime_token_cache_v1 import (  # noqa: E402
    AGGREGATE_STATUS,
    CURRENT64_COUNT,
    EXPECTED_D1_PARAMETER_COUNT,
    EXPECTED_D1_PARAMETER_SCHEMA_SHA256,
    FRESH_COUNT,
    QUERY_COUNT,
    SHARD_COUNT,
    VALIDATED_STATUS,
    assert_parameter_names_reference_defined,
    canonical_sha256,
    expected_query_ordinals,
    logical_sha256,
    parameter_schema,
    parameter_schema_sha256,
    validate_prejoin_record_schema,
)


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_V1_20260902.md"
E0_RESULT = ROOT / "results/routea_matched_three_arm_n2_e0_v1/result.json"
E0_VALIDATION = ROOT / "results/routea_matched_three_arm_n2_e0_v1/independent_validation.json"
SCOPE_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_ADDENDUM_V1_20260902.md"
SCOPE_VALIDATION = ROOT / "results/routea_matched_three_arm_n2_e0_scope_correction_v1/independent_validation.json"
QUERY_LEDGER = ROOT / "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json"
CURRENT64_VALIDATION = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1/validation.json"
MODEL_LINEAGE = ROOT / "results/romav2_colnomic_new_difficult_model_lineage_v1/result.json"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_token_cache_shard_v1.py"
SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_shard_v1.py"
FINAL_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_aggregate_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_token_cache_v1"
OUT = OUT_ROOT / "aggregate.json"
EXPECTED_ENCODER_FINGERPRINT = "af1f3f83297ac2ab737a50704455fea1da10635b7b14cf101ba0dc02ec30191f"
EXPECTED_MODEL_LINEAGE_SHA256 = "6d5670561a2188deaea01f9c4373f2d4435ba45114307d95bcde555e5e7f2c82"
EXPECTED_SCOPE_VALIDATION_SHA256 = "c9ecdb33d94b68e8bcecf99384830b23e22b3c4eb0c593f828fdcb0b0bf8aa59"
EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256 = "936cf64caa5ce47f808a3a981112deb7967583a690f78c631dc9e762da5cbe99"
PAYLOAD_KEYS = {
    "schema_version", "status", "claim_level", "shard", "shard_count",
    "query_ordinal_begin", "query_ordinal_end_exclusive", "records", "model",
    "bindings", "access",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_shard_access() -> dict[str, int]:
    return {
        "protected_metadata_read_count": 0,
        "current64_artifact_deserialization_count": 8,
        "current64_candidate_score_axis_semantic_consumption_count": 0,
        "model_lineage_byte_binding_count": 1,
        "model_lineage_gallery_semantic_consumption_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
        "encoder_update_count": 0,
        "d1_checkpoint_load_count": 0,
        "d1_update_count": 0,
        "action_head_update_count": 0,
    }


def expected_shard_bindings() -> dict[str, str]:
    return {
        "contract_sha256": sha256_file(CONTRACT),
        "e0_result_sha256": sha256_file(E0_RESULT),
        "e0_validation_sha256": sha256_file(E0_VALIDATION),
        "scope_addendum_sha256": sha256_file(SCOPE_ADDENDUM),
        "scope_validation_sha256": sha256_file(SCOPE_VALIDATION),
        "query_ledger_sha256": sha256_file(QUERY_LEDGER),
        "current64_validation_sha256": sha256_file(CURRENT64_VALIDATION),
        "model_lineage_sha256": EXPECTED_MODEL_LINEAGE_SHA256,
        "producer_sha256": sha256_file(PRODUCER),
        "validator_sha256": sha256_file(SHARD_VALIDATOR),
    }


def main() -> None:
    existing = json.loads(OUT.read_text()) if OUT.exists() else None

    ledger_document = json.loads(QUERY_LEDGER.read_text())
    ledger = ledger_document.get("queries", [])
    require(len(ledger) == QUERY_COUNT, "canonical ledger population drift")
    records: list[dict[str, Any]] = []
    shard_seals: list[dict[str, Any]] = []
    encoder_schema_hashes: set[str] = set()
    encoder_parameter_counts: set[int] = set()
    production_current64_deserializations = 0
    production_current64_candidate_semantic_consumption = 0
    expected_payload_bindings = expected_shard_bindings()
    expected_payload_access = expected_shard_access()
    for shard in range(SHARD_COUNT):
        shard_dir = OUT_ROOT / f"shard{shard:02d}"
        payload_path = shard_dir / "payload.pt"
        receipt_path = shard_dir / "receipt.json"
        validation_path = shard_dir / "validation.json"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        require(
            validation.get("status") == VALIDATED_STATUS
            and all(validation.get("checks", {}).values())
            and validation.get("logical_sha256") == logical_sha256(validation)
            and validation.get("payload_sha256") == sha256_file(payload_path)
            and validation.get("receipt_sha256") == sha256_file(receipt_path)
            and receipt.get("payload_sha256") == sha256_file(payload_path),
            f"shard {shard} validation/seal drift",
        )
        expected = expected_query_ordinals(shard)
        shard_records = payload.get("records", [])
        observed_mode_counts = dict(
            sorted(Counter(str(record.get("materialization_mode")) for record in shard_records).items())
        )
        require(
            [int(record["query_ordinal"]) for record in shard_records] == list(expected),
            f"shard {shard} ordinal interval drift",
        )
        require(
            set(payload) == PAYLOAD_KEYS
            and payload.get("schema_version")
            == "routea_n2_current_runtime_987_token_cache_shard_v1_20260902"
            and payload.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_READY"
            and payload.get("claim_level") == "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY"
            and payload.get("shard") == shard
            and payload.get("shard_count") == SHARD_COUNT
            and payload.get("query_ordinal_begin") == expected[0]
            and payload.get("query_ordinal_end_exclusive") == expected[-1] + 1
            and payload.get("bindings") == expected_payload_bindings
            and payload.get("access") == expected_payload_access
            and payload.get("model", {}).get("stable_encoder_fingerprint_sha256")
            == EXPECTED_ENCODER_FINGERPRINT
            and payload.get("model", {}).get("encoder_trainable_parameter_count") == 0
            and payload.get("model", {}).get("d1_checkpoint_load_count") == 0
            and payload.get("model", {}).get("d1_model_update_count") == 0
            and receipt.get("query_ordinal_begin") == expected[0]
            and receipt.get("query_ordinal_end_exclusive") == expected[-1] + 1
            and receipt.get("mode_counts") == observed_mode_counts,
            f"shard {shard} current envelope/binding/access drift",
        )
        for record in shard_records:
            validate_prejoin_record_schema(record)
        records.extend(shard_records)
        encoder_schema_hashes.add(str(payload["model"]["encoder_parameter_schema_sha256"]))
        encoder_parameter_counts.add(int(payload["model"]["encoder_parameter_count"]))
        production_current64_deserializations += int(
            payload["access"]["current64_artifact_deserialization_count"]
        )
        production_current64_candidate_semantic_consumption += int(
            payload["access"]["current64_candidate_score_axis_semantic_consumption_count"]
        )
        shard_seals.append(
            {
                "shard": shard,
                "query_count": len(shard_records),
                "payload_sha256": sha256_file(payload_path),
                "receipt_sha256": sha256_file(receipt_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )

    module = d1_wrapper.import_runner()
    d1_wrapper.configure_fold(module, 0)
    d1 = module.d1.build_fresh_d1_adapter()
    d1_schema = parameter_schema(d1)
    d1_count = sum(parameter.numel() for parameter in d1.parameters())
    assert_parameter_names_reference_defined(item["name"] for item in d1_schema)
    require(
        d1_count == EXPECTED_D1_PARAMETER_COUNT
        and parameter_schema_sha256(d1) == EXPECTED_D1_PARAMETER_SCHEMA_SHA256,
        "shared D1 parameter schema/count drift",
    )
    scope = json.loads(SCOPE_VALIDATION.read_text())

    query_ids = [str(record["query_id"]) for record in records]
    ordinals = [int(record["query_ordinal"]) for record in records]
    tracks = Counter(str(record["track"]) for record in records)
    folds = Counter(int(record["heldout_fold"]) for record in records)
    grids = Counter(tuple(map(int, record["grid_shape"])) for record in records)
    modes = Counter(str(record["materialization_mode"]) for record in records)
    manifest = [
        {
            "query_id": record["query_id"],
            "query_ordinal": record["query_ordinal"],
            "shard": next(
                shard for shard in range(SHARD_COUNT) if record["query_ordinal"] in expected_query_ordinals(shard)
            ),
            "source_image_sha256": record["source_image_sha256"],
            "decode_frame": record["decode_frame"],
            "grid_shape": list(record["grid_shape"]),
            "image_tokens_sha256": record["image_tokens_sha256"],
            "template_tokens_sha256": record["template_tokens_sha256"],
            "materialization_mode": record["materialization_mode"],
        }
        for record in records
    ]
    checks = {
        "all_sixteen_shards_independently_validated": len(shard_seals) == SHARD_COUNT,
        "canonical_987_exact_coverage": len(records) == QUERY_COUNT
        and ordinals == list(range(QUERY_COUNT))
        and len(set(query_ids)) == QUERY_COUNT
        and all(record["query_id"] == ledger[index]["query_id"] for index, record in enumerate(records)),
        "current64_64_and_fresh923": modes
        == Counter({"CURRENT64_EXACT_REUSE": CURRENT64_COUNT, "FRESH_CURRENT_RUNTIME_FORWARD": FRESH_COUNT}),
        "track_population": tracks == Counter({"outcome": 813, "difficult": 134, "new_difficult_train": 40}),
        "fold_population": folds == Counter({0: 212, 1: 205, 2: 189, 3: 188, 4: 193}),
        "grid_population": grids == Counter({(36, 20): 813, (24, 32): 162, (32, 24): 11, (25, 29): 1}),
        "shared_encoder_schema_all_shards": len(encoder_schema_hashes) == 1 and len(encoder_parameter_counts) == 1,
        "shared_d1_schema_49792": d1_count == EXPECTED_D1_PARAMETER_COUNT
        and parameter_schema_sha256(d1) == EXPECTED_D1_PARAMETER_SCHEMA_SHA256,
        "protected_record_fields_absent": True,
        "production_access_accounting": production_current64_deserializations
        == SHARD_COUNT * 8
        and production_current64_candidate_semantic_consumption == 0,
        "scope_correction_physical_and_logical": sha256_file(SCOPE_VALIDATION)
        == EXPECTED_SCOPE_VALIDATION_SHA256
        and scope.get("logical_sha256") == EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256
        and scope.get("logical_sha256") == logical_sha256(scope)
        and scope.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED"
        and scope.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT"
        and sha256_file(MODEL_LINEAGE) == EXPECTED_MODEL_LINEAGE_SHA256,
    }
    require(all(checks.values()), "N2 token-cache aggregate hard check failed")
    value = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_aggregate_v1_20260902",
        "status": AGGREGATE_STATUS,
        "claim_level": "TARGET_FREE_TOKEN_CACHE_AGGREGATE_ENGINEERING_ONLY",
        "checks": checks,
        "query_count": len(records),
        "mode_counts": dict(sorted(modes.items())),
        "track_counts": dict(sorted(tracks.items())),
        "fold_counts": {str(key): value for key, value in sorted(folds.items())},
        "grid_counts": {f"{key[0]}x{key[1]}": value for key, value in sorted(grids.items())},
        "tensor_manifest_sha256": canonical_sha256(manifest),
        "query_id_ordinal_sha256": canonical_sha256([[record["query_id"], record["query_ordinal"]] for record in records]),
        "shared_encoder_parameter_schema_sha256": next(iter(encoder_schema_hashes)),
        "shared_encoder_parameter_count": next(iter(encoder_parameter_counts)),
        "shared_d1_parameter_schema_sha256": parameter_schema_sha256(d1),
        "shared_d1_parameter_count": d1_count,
        "shards": shard_seals,
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "query_ledger_sha256": sha256_file(QUERY_LEDGER),
            "producer_sha256": sha256_file(PRODUCER),
            "shard_validator_sha256": sha256_file(SHARD_VALIDATOR),
            "aggregate_sha256": sha256_file(Path(__file__).resolve()),
            "final_validator_sha256": sha256_file(FINAL_VALIDATOR),
        },
        "access": {
            "token_shard_artifact_deserialization_count": SHARD_COUNT,
            "production_current64_artifact_deserialization_count": production_current64_deserializations,
            "production_current64_candidate_score_axis_semantic_consumption_count": production_current64_candidate_semantic_consumption,
            "protected_metadata_read_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
            "encoder_load_count": 0,
            "encoder_update_count": 0,
            "d1_schema_construction_count": 1,
            "d1_checkpoint_load_count": 0,
            "d1_update_count": 0,
            "action_head_update_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "n2_d1_training_authorized": False,
        "external_execution_authorized": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_INDEPENDENT_VALIDATION",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    if existing is None:
        atomic_json(OUT, value)
    else:
        require(existing == value, "stale or corrupt immutable aggregate fails closed")
    print(json.dumps({"status": value["status"], "checks": checks, "query_count": len(records)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
