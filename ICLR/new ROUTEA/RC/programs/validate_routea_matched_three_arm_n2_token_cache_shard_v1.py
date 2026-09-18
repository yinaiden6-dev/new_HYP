#!/usr/bin/env python3
"""Independent CPU integrity validation for one N2 token-cache shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_token_cache_v1 import (  # noqa: E402
    CURRENT64_COUNT,
    READY_STATUS,
    SHARD_COUNT,
    VALIDATED_STATUS,
    expected_query_ordinals,
    logical_sha256,
    tensor_sha256,
    validate_prejoin_record_schema,
)


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_V1_20260902.md"
E0_RESULT = ROOT / "results/routea_matched_three_arm_n2_e0_v1/result.json"
E0_VALIDATION = ROOT / "results/routea_matched_three_arm_n2_e0_v1/independent_validation.json"
SCOPE_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_ADDENDUM_V1_20260902.md"
SCOPE_VALIDATION = ROOT / "results/routea_matched_three_arm_n2_e0_scope_correction_v1/independent_validation.json"
QUERY_LEDGER = ROOT / "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json"
CURRENT64_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
CURRENT64_VALIDATION = CURRENT64_ROOT / "validation.json"
MODEL_LINEAGE = ROOT / "results/romav2_colnomic_new_difficult_model_lineage_v1/result.json"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_token_cache_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_token_cache_v1"
EXPECTED_ENCODER_FINGERPRINT = "af1f3f83297ac2ab737a50704455fea1da10635b7b14cf101ba0dc02ec30191f"
EXPECTED_SCOPE_VALIDATION_SHA256 = "c9ecdb33d94b68e8bcecf99384830b23e22b3c4eb0c593f828fdcb0b0bf8aa59"
EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256 = "936cf64caa5ce47f808a3a981112deb7967583a690f78c631dc9e762da5cbe99"
PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "shard",
    "shard_count",
    "query_ordinal_begin",
    "query_ordinal_end_exclusive",
    "records",
    "model",
    "bindings",
    "access",
}
MODEL_KEYS = {
    "stable_encoder_fingerprint_sha256",
    "encoder_parameter_schema_sha256",
    "encoder_parameter_count",
    "encoder_trainable_parameter_count",
    "d1_checkpoint_load_count",
    "d1_model_update_count",
    "hardware_identity_is_authority",
}
RECEIPT_KEYS = {
    "schema_version",
    "status",
    "shard",
    "query_count",
    "query_ordinal_begin",
    "query_ordinal_end_exclusive",
    "mode_counts",
    "payload_sha256",
    "model_update_count",
    "next_authorized_stage",
    "logical_sha256",
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


def ledger_rows() -> list[dict[str, Any]]:
    document = json.loads(QUERY_LEDGER.read_text())
    rows = document.get("queries", [])
    require(
        document.get("version") == "l0_natural_hardneg_v2_targetfree_query_ledger_v1"
        and document.get("query_count") == 987
        and document.get("target_label_read") is False
        and len(rows) == 987
        and all(int(row["query_ordinal"]) == ordinal for ordinal, row in enumerate(rows))
        and len({str(row["query_id"]) for row in rows}) == 987,
        "independent canonical ledger drift",
    )
    return rows


def current64_by_query() -> dict[str, tuple[dict[str, Any], int, str]]:
    validation = json.loads(CURRENT64_VALIDATION.read_text())
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("query_count") == CURRENT64_COUNT
        and set(seals) == set(range(8))
        and all(validation.get("checks", {}).values()),
        "independent current64 aggregate drift",
    )
    output: dict[str, tuple[dict[str, Any], int, str]] = {}
    for shard in range(8):
        payload_path = CURRENT64_ROOT / f"shard{shard:02d}/payload.pt"
        seal = seals[shard]
        require(
            seal.get("payload_sha256") == sha256_file(payload_path)
            and seal.get("receipt_sha256") == sha256_file(CURRENT64_ROOT / f"shard{shard:02d}/receipt.json")
            and seal.get("validation_sha256") == sha256_file(CURRENT64_ROOT / f"shard{shard:02d}/validation.json"),
            "independent current64 shard seal drift",
        )
        payload_sha = sha256_file(payload_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            require(query_id not in output, "independent current64 duplicate")
            output[query_id] = (record, shard, payload_sha)
    require(len(output) == CURRENT64_COUNT, "independent current64 population drift")
    return output


def expected_geometry(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"]).resolve()
    require(sha256_file(path) == row["source_image_sha256"], "live source hash drift")
    with Image.open(path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        raw = (int(opened.height), int(opened.width))
        if row["track"] == "new_difficult_train":
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            frame = "EXIF_ORIENTED_BEFORE_RESIZE"
        else:
            oriented = opened.convert("RGB")
            frame = "DECODED_RAW_BEFORE_EXIF"
    return {
        "source_exif_orientation": orientation,
        "decode_frame": frame,
        "raw_size_hw": raw,
        "oriented_size_hw": (int(oriented.height), int(oriented.width)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard must be in 0..15")
    shard_dir = OUT_ROOT / f"shard{args.shard:02d}"
    payload_path = shard_dir / "payload.pt"
    receipt_path = shard_dir / "receipt.json"
    out = shard_dir / "validation.json"
    existing = json.loads(out.read_text()) if out.exists() else None

    rows = ledger_rows()
    old = current64_by_query()
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads(receipt_path.read_text())
    ordinals = expected_query_ordinals(args.shard)
    records = payload.get("records", [])
    record_checks: list[bool] = []
    reuse_count = 0
    fresh_count = 0
    for expected_ordinal, record in zip(ordinals, records):
        validate_prejoin_record_schema(record)
        row = rows[expected_ordinal]
        geometry = expected_geometry(row)
        image_tokens = torch.as_tensor(record["image_tokens"])
        template_tokens = torch.as_tensor(record["template_tokens"])
        grid = (int(row["grid_h"]), int(row["grid_w"]))
        current = old.get(str(row["query_id"]))
        if current is None:
            mode_ok = (
                record["materialization_mode"] == "FRESH_CURRENT_RUNTIME_FORWARD"
                and record["current64_source_shard"] is None
                and record["current64_source_payload_sha256"] is None
            )
            fresh_count += 1
        else:
            authority, source_shard, source_payload_sha = current
            mode_ok = (
                record["materialization_mode"] == "CURRENT64_EXACT_REUSE"
                and record["current64_source_shard"] == source_shard
                and record["current64_source_payload_sha256"] == source_payload_sha
                and torch.equal(image_tokens, torch.as_tensor(authority["raw_image_tokens"]))
                and torch.equal(template_tokens, torch.as_tensor(authority["template_tokens"]))
                and tuple(authority["query_grid_shape"]) == grid
                and authority["query_source_sha256"] == row["source_image_sha256"]
            )
            reuse_count += 1
        record_checks.append(
            int(record["query_ordinal"]) == expected_ordinal
            and record["query_id"] == row["query_id"]
            and int(record["heldout_fold"]) == int(row["heldout_fold"])
            and record["track"] == row["track"]
            and record["source_image_sha256"] == row["source_image_sha256"]
            and all(record[key] == value for key, value in geometry.items())
            and tuple(record["grid_shape"]) == grid
            and image_tokens.dtype == template_tokens.dtype == torch.float16
            and image_tokens.is_contiguous()
            and template_tokens.is_contiguous()
            and image_tokens.shape == (grid[0] * grid[1], 128)
            and template_tokens.ndim == 2
            and template_tokens.shape[0] > 0
            and template_tokens.shape[1] == 128
            and bool(torch.isfinite(image_tokens).all())
            and bool(torch.isfinite(template_tokens).all())
            and record["image_tokens_sha256"] == tensor_sha256(image_tokens)
            and record["template_tokens_sha256"] == tensor_sha256(template_tokens)
            and mode_ok
            and record["model_update_count"] == 0
        )

    expected_access = {
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
    expected_bindings = {
        "contract_sha256": sha256_file(CONTRACT),
        "e0_result_sha256": sha256_file(E0_RESULT),
        "e0_validation_sha256": sha256_file(E0_VALIDATION),
        "scope_addendum_sha256": sha256_file(SCOPE_ADDENDUM),
        "scope_validation_sha256": sha256_file(SCOPE_VALIDATION),
        "query_ledger_sha256": sha256_file(QUERY_LEDGER),
        "current64_validation_sha256": sha256_file(CURRENT64_VALIDATION),
        "model_lineage_sha256": sha256_file(MODEL_LINEAGE),
        "producer_sha256": sha256_file(PRODUCER),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
    }
    scope = json.loads(SCOPE_VALIDATION.read_text())
    checks = {
        "e0_scope_correction_authority": sha256_file(SCOPE_VALIDATION)
        == EXPECTED_SCOPE_VALIDATION_SHA256
        and scope.get("logical_sha256") == EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256
        and scope.get("logical_sha256") == logical_sha256(scope)
        and scope.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED"
        and scope.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT",
        "payload_envelope": set(payload) == PAYLOAD_KEYS
        and set(payload.get("model", {})) == MODEL_KEYS
        and set(payload.get("bindings", {})) == set(expected_bindings)
        and set(payload.get("access", {})) == set(expected_access)
        and payload.get("schema_version")
        == "routea_n2_current_runtime_987_token_cache_shard_v1_20260902"
        and payload.get("status") == READY_STATUS
        and payload.get("claim_level") == "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY"
        and payload.get("shard") == args.shard
        and payload.get("shard_count") == SHARD_COUNT
        and payload.get("query_ordinal_begin") == ordinals[0]
        and payload.get("query_ordinal_end_exclusive") == ordinals[-1] + 1,
        "receipt_and_physical_seal": set(receipt) == RECEIPT_KEYS
        and receipt.get("schema_version")
        == "routea_n2_current_runtime_987_token_cache_receipt_v1_20260902"
        and receipt.get("status") == READY_STATUS
        and receipt.get("shard") == args.shard
        and receipt.get("query_count") == len(ordinals)
        and receipt.get("query_ordinal_begin") == ordinals[0]
        and receipt.get("query_ordinal_end_exclusive") == ordinals[-1] + 1
        and receipt.get("mode_counts")
        == {
            key: value
            for key, value in sorted(
                {
                    "CURRENT64_EXACT_REUSE": reuse_count,
                    "FRESH_CURRENT_RUNTIME_FORWARD": fresh_count,
                }.items()
            )
            if value > 0
        }
        and receipt.get("payload_sha256") == sha256_file(payload_path)
        and receipt.get("model_update_count") == 0
        and receipt.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_SHARD_VALIDATION"
        and receipt.get("logical_sha256") == logical_sha256(receipt),
        "exact_canonical_interval": len(records) == len(ordinals)
        and len({record["query_id"] for record in records}) == len(records),
        "record_schema_geometry_tensor_hashes": all(record_checks),
        "current64_query_id_reuse": reuse_count == sum(row["query_id"] in old for row in rows[ordinals[0] : ordinals[-1] + 1]),
        "bindings": payload.get("bindings") == expected_bindings
        and sha256_file(MODEL_LINEAGE)
        == "6d5670561a2188deaea01f9c4373f2d4435ba45114307d95bcde555e5e7f2c82",
        "shared_frozen_encoder": payload.get("model", {}).get("stable_encoder_fingerprint_sha256")
        == EXPECTED_ENCODER_FINGERPRINT
        and isinstance(payload.get("model", {}).get("encoder_parameter_schema_sha256"), str)
        and len(payload["model"]["encoder_parameter_schema_sha256"]) == 64
        and payload["model"].get("encoder_parameter_count", 0) > 0
        and payload["model"].get("encoder_trainable_parameter_count") == 0
        and payload["model"].get("d1_checkpoint_load_count") == 0
        and payload["model"].get("d1_model_update_count") == 0
        and payload["model"].get("hardware_identity_is_authority") is False,
        "protected_access_zero": payload.get("access") == expected_access,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_shard_validation_v1_20260902",
        "status": VALIDATED_STATUS if passed else "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_VALIDATION_ABORT",
        "claim_level": "INDEPENDENT_CPU_SHARD_INTEGRITY_VALIDATION_ONLY",
        "shard": args.shard,
        "checks": checks,
        "query_count": len(records),
        "reuse_count": reuse_count,
        "fresh_count": fresh_count,
        "payload_sha256": sha256_file(payload_path),
        "receipt_sha256": sha256_file(receipt_path),
        "model_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_AGGREGATE" if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    if existing is None:
        atomic_json(out, value)
    else:
        require(existing == value, "stale or corrupt immutable shard validation fails closed")
    print(json.dumps({"status": value["status"], "shard": args.shard, "checks": checks}, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
