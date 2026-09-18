#!/usr/bin/env python3
"""Independent aggregate and fresh-forward validation of the N2 token cache.

This validator intentionally does not import either the shard producer or the
aggregate producer.
"""

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
from torch.nn import functional as F
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
WORKSPACE = ROOT.parents[2]
# Use the same explicit precedence as the aggregate producer.  In particular,
# ``route_a_core/route_a`` must win over the unrelated ``ROUTEA/route_a``
# package even when the batch environment already placed ROUTEA on PYTHONPATH.
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
    FINAL_STATUS,
    FRESH_COUNT,
    FRESH_FORWARD_AUDIT_QUERY_IDS,
    QUERY_COUNT,
    SHARD_COUNT,
    VALIDATED_STATUS,
    assert_parameter_names_reference_defined,
    canonical_sha256,
    expected_query_ordinals,
    logical_sha256,
    parameter_schema,
    parameter_schema_sha256,
    tensor_sha256,
    validate_audit_population,
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
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_n2_token_cache_shard_v1.py"
SHARD_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_shard_v1.py"
AGGREGATOR = ROOT / "programs/aggregate_routea_matched_three_arm_n2_token_cache_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_token_cache_v1"
AGGREGATE = OUT_ROOT / "aggregate.json"
OUT = OUT_ROOT / "independent_validation.json"
EXPECTED_ENCODER_FINGERPRINT = "af1f3f83297ac2ab737a50704455fea1da10635b7b14cf101ba0dc02ec30191f"
EXPECTED_MODEL_LINEAGE_SHA256 = "6d5670561a2188deaea01f9c4373f2d4435ba45114307d95bcde555e5e7f2c82"
EXPECTED_SCOPE_VALIDATION_SHA256 = "c9ecdb33d94b68e8bcecf99384830b23e22b3c4eb0c593f828fdcb0b0bf8aa59"
EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256 = "936cf64caa5ce47f808a3a981112deb7967583a690f78c631dc9e762da5cbe99"
MAX_ABS_TOLERANCE = 0.00390625
MIN_COSINE = 0.99999
PAYLOAD_KEYS = {
    "schema_version", "status", "claim_level", "shard", "shard_count",
    "query_ordinal_begin", "query_ordinal_end_exclusive", "records", "model",
    "bindings", "access",
}
MODEL_KEYS = {
    "stable_encoder_fingerprint_sha256", "encoder_parameter_schema_sha256",
    "encoder_parameter_count", "encoder_trainable_parameter_count",
    "d1_checkpoint_load_count", "d1_model_update_count",
    "hardware_identity_is_authority",
}
RECEIPT_KEYS = {
    "schema_version", "status", "shard", "query_count", "query_ordinal_begin",
    "query_ordinal_end_exclusive", "mode_counts", "payload_sha256",
    "model_update_count", "next_authorized_stage", "logical_sha256",
}
AGGREGATE_KEYS = {
    "schema_version", "status", "claim_level", "checks", "query_count",
    "mode_counts", "track_counts", "fold_counts", "grid_counts",
    "tensor_manifest_sha256", "query_id_ordinal_sha256",
    "shared_encoder_parameter_schema_sha256", "shared_encoder_parameter_count",
    "shared_d1_parameter_schema_sha256", "shared_d1_parameter_count", "shards",
    "bindings", "access", "scientific_GO_or_NO_GO", "n2_d1_training_authorized",
    "external_execution_authorized", "next_authorized_stage", "logical_sha256",
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


def independent_source_geometry(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"]).resolve()
    require(sha256_file(path) == row["source_image_sha256"], "independent live source hash drift")
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


def load_ledger() -> list[dict[str, Any]]:
    document = json.loads(QUERY_LEDGER.read_text())
    rows = document.get("queries", [])
    require(
        document.get("version") == "l0_natural_hardneg_v2_targetfree_query_ledger_v1"
        and document.get("query_count") == QUERY_COUNT
        and document.get("target_label_read") is False
        and len(rows) == QUERY_COUNT
        and all(int(row["query_ordinal"]) == ordinal for ordinal, row in enumerate(rows))
        and len({str(row["query_id"]) for row in rows}) == QUERY_COUNT,
        "independent canonical ledger drift",
    )
    validate_audit_population(FRESH_FORWARD_AUDIT_QUERY_IDS, rows)
    return rows


def load_current64() -> dict[str, tuple[dict[str, Any], int, str]]:
    validation = json.loads(CURRENT64_VALIDATION.read_text())
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("query_count") == CURRENT64_COUNT
        and set(seals) == set(range(8))
        and all(validation.get("checks", {}).values()),
        "independent current64 authority drift",
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


def load_all_shards(ledger: list[dict[str, Any]], current64: dict[str, tuple[dict[str, Any], int, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    records: list[dict[str, Any]] = []
    seals: list[dict[str, Any]] = []
    schema_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    expected_bindings = expected_shard_bindings()
    expected_access = expected_shard_access()
    for shard in range(SHARD_COUNT):
        shard_dir = OUT_ROOT / f"shard{shard:02d}"
        payload_path = shard_dir / "payload.pt"
        receipt_path = shard_dir / "receipt.json"
        validation_path = shard_dir / "validation.json"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        expected = expected_query_ordinals(shard)
        shard_records = payload.get("records", [])
        observed_mode_counts = dict(
            sorted(Counter(str(record.get("materialization_mode")) for record in shard_records).items())
        )
        require(
            validation.get("status") == VALIDATED_STATUS
            and all(validation.get("checks", {}).values())
            and validation.get("logical_sha256") == logical_sha256(validation)
            and validation.get("payload_sha256") == sha256_file(payload_path)
            and validation.get("receipt_sha256") == sha256_file(receipt_path)
            and receipt.get("payload_sha256") == sha256_file(payload_path)
            and [int(record["query_ordinal"]) for record in shard_records] == list(expected),
            f"independent shard {shard} seal/interval drift",
        )
        require(
            set(payload) == PAYLOAD_KEYS
            and set(payload.get("model", {})) == MODEL_KEYS
            and payload.get("schema_version")
            == "routea_n2_current_runtime_987_token_cache_shard_v1_20260902"
            and payload.get("status") == "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_READY"
            and payload.get("claim_level") == "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY"
            and payload.get("shard") == shard
            and payload.get("shard_count") == SHARD_COUNT
            and payload.get("query_ordinal_begin") == expected[0]
            and payload.get("query_ordinal_end_exclusive") == expected[-1] + 1
            and payload.get("bindings") == expected_bindings
            and payload.get("access") == expected_access
            and payload.get("model", {}).get("stable_encoder_fingerprint_sha256")
            == EXPECTED_ENCODER_FINGERPRINT
            and payload.get("model", {}).get("encoder_trainable_parameter_count") == 0
            and payload.get("model", {}).get("d1_checkpoint_load_count") == 0
            and payload.get("model", {}).get("d1_model_update_count") == 0,
            f"independent shard {shard} outer envelope/binding/access drift",
        )
        require(
            set(receipt) == RECEIPT_KEYS
            and receipt.get("schema_version")
            == "routea_n2_current_runtime_987_token_cache_receipt_v1_20260902"
            and receipt.get("status")
            == "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_READY"
            and receipt.get("shard") == shard
            and receipt.get("query_count") == len(expected)
            and receipt.get("query_ordinal_begin") == expected[0]
            and receipt.get("query_ordinal_end_exclusive") == expected[-1] + 1
            and receipt.get("mode_counts") == observed_mode_counts
            and receipt.get("model_update_count") == 0
            and receipt.get("next_authorized_stage")
            == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_SHARD_VALIDATION"
            and receipt.get("logical_sha256") == logical_sha256(receipt),
            f"independent shard {shard} receipt bound/mode drift",
        )
        for record in shard_records:
            validate_prejoin_record_schema(record)
            row = ledger[int(record["query_ordinal"])]
            image = torch.as_tensor(record["image_tokens"])
            template = torch.as_tensor(record["template_tokens"])
            grid = (int(row["grid_h"]), int(row["grid_w"]))
            geometry = independent_source_geometry(row)
            require(
                record["query_id"] == row["query_id"]
                and record["track"] == row["track"]
                and int(record["heldout_fold"]) == int(row["heldout_fold"])
                and record["source_image_sha256"] == row["source_image_sha256"]
                and all(record[key] == value for key, value in geometry.items())
                and tuple(record["grid_shape"]) == grid
                and image.dtype == template.dtype == torch.float16
                and image.is_contiguous()
                and template.is_contiguous()
                and image.shape == (grid[0] * grid[1], 128)
                and template.ndim == 2
                and template.shape[0] > 0
                and template.shape[1] == 128
                and bool(torch.isfinite(image).all())
                and bool(torch.isfinite(template).all())
                and record["image_tokens_sha256"] == tensor_sha256(image)
                and record["template_tokens_sha256"] == tensor_sha256(template)
                and record["model_update_count"] == 0,
                "independent record tensor/ledger drift",
            )
            old = current64.get(record["query_id"])
            if old is None:
                require(
                    record["materialization_mode"] == "FRESH_CURRENT_RUNTIME_FORWARD"
                    and record["current64_source_shard"] is None
                    and record["current64_source_payload_sha256"] is None,
                    "independent fresh-mode drift",
                )
            else:
                authority, old_shard, old_sha = old
                require(
                    record["materialization_mode"] == "CURRENT64_EXACT_REUSE"
                    and record["current64_source_shard"] == old_shard
                    and record["current64_source_payload_sha256"] == old_sha
                    and torch.equal(image, torch.as_tensor(authority["raw_image_tokens"]))
                    and torch.equal(template, torch.as_tensor(authority["template_tokens"])),
                    "independent current64 exact reuse drift",
                )
        records.extend(shard_records)
        schema_hashes.add(str(payload["model"]["encoder_parameter_schema_sha256"]))
        parameter_counts.add(int(payload["model"]["encoder_parameter_count"]))
        seals.append(
            {
                "shard": shard,
                "query_count": len(shard_records),
                "payload_sha256": sha256_file(payload_path),
                "receipt_sha256": sha256_file(receipt_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )
    require(len(schema_hashes) == len(parameter_counts) == 1, "shared encoder schema differs across shards")
    return records, seals, next(iter(schema_hashes)), next(iter(parameter_counts))


def decode_and_encode(row: dict[str, Any], *, encoder: torch.nn.Module, processor: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], dict[str, Any]]:
    path = Path(row["path"]).resolve()
    require(sha256_file(path) == row["source_image_sha256"], "fresh-audit source hash drift")
    with Image.open(path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        raw = (int(opened.height), int(opened.width))
        if row["track"] == "new_difficult_train":
            image = ImageOps.exif_transpose(opened).convert("RGB")
            frame = "EXIF_ORIENTED_BEFORE_RESIZE"
        else:
            image = opened.convert("RGB")
            frame = "DECODED_RAW_BEFORE_EXIF"
    oriented = (int(image.height), int(image.width))
    inputs = processor.process_images([image]).to(device)
    with torch.inference_mode():
        encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    require(image_mask.shape == encoded.shape[:1] and bool(image_mask.any()), "fresh-audit image mask drift")
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
    temporal, height, width = [int(item) for item in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    require(temporal == 1, "fresh-audit temporal grid drift")
    return image_tokens, template_tokens, (height // merge, width // merge), {
        "source_exif_orientation": orientation,
        "decode_frame": frame,
        "raw_size_hw": raw,
        "oriented_size_hw": oriented,
    }


def numeric_replay(fresh: torch.Tensor, stored: torch.Tensor) -> tuple[float, float]:
    require(fresh.shape == stored.shape and fresh.dtype == stored.dtype, "fresh-audit tensor shape/dtype drift")
    left = fresh.float().flatten()
    right = stored.float().flatten()
    maximum = float((left - right).abs().max())
    cosine = float(F.cosine_similarity(left[None], right[None], dim=1)[0])
    require(maximum <= MAX_ABS_TOLERANCE and cosine >= MIN_COSINE, "fresh-audit numeric tolerance failed")
    return maximum, cosine


def existing_final_is_current(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    old_audit = existing.get("fresh_forward_audit", {})
    current_audit = current.get("fresh_forward_audit", {})
    old_rows = old_audit.get("rows", [])
    return (
        existing.get("logical_sha256") == logical_sha256(existing)
        and existing.get("schema_version") == current.get("schema_version")
        and existing.get("status") == current.get("status") == FINAL_STATUS
        and existing.get("claim_level") == current.get("claim_level")
        and set(existing.get("checks", {})) == set(current.get("checks", {}))
        and all(existing.get("checks", {}).values())
        and existing.get("query_count") == current.get("query_count") == QUERY_COUNT
        and existing.get("mode_counts") == current.get("mode_counts")
        and existing.get("aggregate_sha256") == current.get("aggregate_sha256")
        and existing.get("shared_encoder_parameter_schema_sha256")
        == current.get("shared_encoder_parameter_schema_sha256")
        and existing.get("shared_encoder_parameter_count")
        == current.get("shared_encoder_parameter_count")
        and existing.get("shared_d1_parameter_schema_sha256")
        == current.get("shared_d1_parameter_schema_sha256")
        and existing.get("shared_d1_parameter_count") == current.get("shared_d1_parameter_count")
        and existing.get("reference_defined_model_boundary")
        == current.get("reference_defined_model_boundary")
        and existing.get("access") == current.get("access")
        and existing.get("scientific_GO_or_NO_GO") is None
        and existing.get("n2_d1_training_authorized") is False
        and existing.get("external_execution_authorized") is False
        and existing.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT"
        and old_audit.get("query_count") == current_audit.get("query_count")
        and old_audit.get("query_ids") == current_audit.get("query_ids")
        and old_audit.get("max_abs_tolerance") == current_audit.get("max_abs_tolerance")
        and old_audit.get("min_cosine") == current_audit.get("min_cosine")
        and len(old_rows) == len(FRESH_FORWARD_AUDIT_QUERY_IDS)
        and [row.get("query_id") for row in old_rows] == list(FRESH_FORWARD_AUDIT_QUERY_IDS)
        and max(max(float(row["image_max_abs"]), float(row["template_max_abs"])) for row in old_rows)
        <= MAX_ABS_TOLERANCE
        and min(min(float(row["image_cosine"]), float(row["template_cosine"])) for row in old_rows)
        >= MIN_COSINE
    )


def main() -> None:
    existing = json.loads(OUT.read_text()) if OUT.exists() else None
    scope = json.loads(SCOPE_VALIDATION.read_text())
    require(
        sha256_file(SCOPE_VALIDATION) == EXPECTED_SCOPE_VALIDATION_SHA256
        and scope.get("logical_sha256") == EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256
        and scope.get("logical_sha256") == logical_sha256(scope)
        and scope.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED"
        and scope.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT",
        "scope-correction authority drift",
    )
    aggregate = json.loads(AGGREGATE.read_text())
    require(
        aggregate.get("status") == AGGREGATE_STATUS
        and all(aggregate.get("checks", {}).values())
        and aggregate.get("logical_sha256") == logical_sha256(aggregate),
        "aggregate authority drift",
    )
    ledger = load_ledger()
    current64 = load_current64()
    records, shard_seals, encoder_schema_sha, encoder_parameter_count = load_all_shards(ledger, current64)
    by_id = {str(record["query_id"]): record for record in records}
    require(len(records) == len(by_id) == QUERY_COUNT, "independent 987 record population drift")

    modes = Counter(str(record["materialization_mode"]) for record in records)
    tracks = Counter(str(record["track"]) for record in records)
    folds = Counter(int(record["heldout_fold"]) for record in records)
    grids = Counter(tuple(map(int, record["grid_shape"])) for record in records)
    manifest = [
        {
            "query_id": record["query_id"],
            "query_ordinal": record["query_ordinal"],
            "shard": next(shard for shard in range(SHARD_COUNT) if record["query_ordinal"] in expected_query_ordinals(shard)),
            "source_image_sha256": record["source_image_sha256"],
            "decode_frame": record["decode_frame"],
            "grid_shape": list(record["grid_shape"]),
            "image_tokens_sha256": record["image_tokens_sha256"],
            "template_tokens_sha256": record["template_tokens_sha256"],
            "materialization_mode": record["materialization_mode"],
        }
        for record in records
    ]

    module = d1_wrapper.import_runner()
    d1_wrapper.configure_fold(module, 0)
    d1 = module.d1.build_fresh_d1_adapter()
    d1_schema = parameter_schema(d1)
    d1_count = sum(parameter.numel() for parameter in d1.parameters())
    assert_parameter_names_reference_defined(item["name"] for item in d1_schema)
    require(
        d1_count == EXPECTED_D1_PARAMETER_COUNT
        and parameter_schema_sha256(d1) == EXPECTED_D1_PARAMETER_SCHEMA_SHA256,
        "independent shared D1 schema drift",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("final fresh-forward audit requires CUDA")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(MODEL), local_files_only=True)
    observed_encoder_schema = parameter_schema_sha256(encoder)
    observed_encoder_count = sum(parameter.numel() for parameter in encoder.parameters())
    assert_parameter_names_reference_defined(name for name, _ in encoder.named_parameters())
    require(
        observed_encoder_schema == encoder_schema_sha
        and observed_encoder_count == encoder_parameter_count
        and all(not parameter.requires_grad for parameter in encoder.parameters()),
        "independent shared encoder parameter schema drift",
    )

    audit_rows: list[dict[str, Any]] = []
    ledger_by_id = {str(row["query_id"]): row for row in ledger}
    for query_id in FRESH_FORWARD_AUDIT_QUERY_IDS:
        row = ledger_by_id[query_id]
        stored = by_id[query_id]
        image, template, grid, geometry = decode_and_encode(row, encoder=encoder, processor=processor, device=device)
        image_max, image_cosine = numeric_replay(image, torch.as_tensor(stored["image_tokens"]))
        template_max, template_cosine = numeric_replay(template, torch.as_tensor(stored["template_tokens"]))
        require(
            grid == tuple(stored["grid_shape"])
            and all(stored[key] == value for key, value in geometry.items()),
            "fresh-audit geometry/decode drift",
        )
        audit_rows.append(
            {
                "query_id": query_id,
                "query_ordinal": int(row["query_ordinal"]),
                "materialization_mode": stored["materialization_mode"],
                "image_max_abs": image_max,
                "image_cosine": image_cosine,
                "template_max_abs": template_max,
                "template_cosine": template_cosine,
            }
        )
        print(json.dumps({"event": "n2_token_cache_fresh_audit", **audit_rows[-1]}, sort_keys=True), flush=True)

    checks = {
        "aggregate_envelope_and_bindings": set(aggregate) == AGGREGATE_KEYS
        and aggregate.get("schema_version")
        == "routea_n2_current_runtime_987_token_cache_aggregate_v1_20260902"
        and aggregate.get("status") == AGGREGATE_STATUS
        and aggregate.get("claim_level")
        == "TARGET_FREE_TOKEN_CACHE_AGGREGATE_ENGINEERING_ONLY"
        and aggregate.get("query_count") == QUERY_COUNT
        and aggregate.get("mode_counts")
        == {"CURRENT64_EXACT_REUSE": CURRENT64_COUNT, "FRESH_CURRENT_RUNTIME_FORWARD": FRESH_COUNT}
        and aggregate.get("track_counts")
        == {"difficult": 134, "new_difficult_train": 40, "outcome": 813}
        and aggregate.get("fold_counts")
        == {"0": 212, "1": 205, "2": 189, "3": 188, "4": 193}
        and aggregate.get("grid_counts")
        == {"24x32": 162, "25x29": 1, "32x24": 11, "36x20": 813}
        and aggregate.get("query_id_ordinal_sha256")
        == canonical_sha256([[record["query_id"], record["query_ordinal"]] for record in records])
        and aggregate.get("access")
        == {
            "token_shard_artifact_deserialization_count": SHARD_COUNT,
            "production_current64_artifact_deserialization_count": SHARD_COUNT * 8,
            "production_current64_candidate_score_axis_semantic_consumption_count": 0,
            "protected_metadata_read_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
            "encoder_load_count": 0,
            "encoder_update_count": 0,
            "d1_schema_construction_count": 1,
            "d1_checkpoint_load_count": 0,
            "d1_update_count": 0,
            "action_head_update_count": 0,
        }
        and aggregate.get("scientific_GO_or_NO_GO") is None
        and aggregate.get("n2_d1_training_authorized") is False
        and aggregate.get("external_execution_authorized") is False
        and aggregate.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_INDEPENDENT_VALIDATION"
        and aggregate.get("bindings")
        == {
            "contract_sha256": sha256_file(CONTRACT),
            "query_ledger_sha256": sha256_file(QUERY_LEDGER),
            "producer_sha256": sha256_file(PRODUCER),
            "shard_validator_sha256": sha256_file(SHARD_VALIDATOR),
            "aggregate_sha256": sha256_file(AGGREGATOR),
            "final_validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope_correction_physical_and_logical": True,
        "all_shard_seals_independent": aggregate.get("shards") == shard_seals,
        "canonical_987_exact_coverage": [record["query_ordinal"] for record in records] == list(range(QUERY_COUNT))
        and [record["query_id"] for record in records] == [row["query_id"] for row in ledger],
        "current64_64_and_fresh923": modes
        == Counter({"CURRENT64_EXACT_REUSE": CURRENT64_COUNT, "FRESH_CURRENT_RUNTIME_FORWARD": FRESH_COUNT}),
        "track_fold_grid_populations": tracks == Counter({"outcome": 813, "difficult": 134, "new_difficult_train": 40})
        and folds == Counter({0: 212, 1: 205, 2: 189, 3: 188, 4: 193})
        and grids == Counter({(36, 20): 813, (24, 32): 162, (32, 24): 11, (25, 29): 1}),
        "tensor_manifest_independent": aggregate.get("tensor_manifest_sha256") == canonical_sha256(manifest),
        "shared_encoder_schema_independent": aggregate.get("shared_encoder_parameter_schema_sha256")
        == encoder_schema_sha == observed_encoder_schema
        and aggregate.get("shared_encoder_parameter_count") == encoder_parameter_count == observed_encoder_count
        and sha256_file(MODEL_LINEAGE) == EXPECTED_MODEL_LINEAGE_SHA256,
        "shared_d1_schema_independent": aggregate.get("shared_d1_parameter_schema_sha256")
        == parameter_schema_sha256(d1)
        == EXPECTED_D1_PARAMETER_SCHEMA_SHA256
        and aggregate.get("shared_d1_parameter_count")
        == d1_count
        == EXPECTED_D1_PARAMETER_COUNT,
        "fresh_forward_twenty_query_audit": len(audit_rows) == len(FRESH_FORWARD_AUDIT_QUERY_IDS)
        and max(max(row["image_max_abs"], row["template_max_abs"]) for row in audit_rows) <= MAX_ABS_TOLERANCE
        and min(min(row["image_cosine"], row["template_cosine"]) for row in audit_rows) >= MIN_COSINE,
        "reference_defined_zero_update_boundary": True,
    }
    require(all(checks.values()), "N2 token-cache final independent validation failed")
    value = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_independent_validation_v1_20260902",
        "status": FINAL_STATUS,
        "claim_level": "INDEPENDENT_ENGINEERING_TOKEN_CACHE_VALIDATION_NO_TRAINING_NO_SCIENTIFIC_CLAIM",
        "checks": checks,
        "query_count": len(records),
        "mode_counts": dict(sorted(modes.items())),
        "fresh_forward_audit": {
            "query_count": len(audit_rows),
            "query_ids": list(FRESH_FORWARD_AUDIT_QUERY_IDS),
            "max_abs_tolerance": MAX_ABS_TOLERANCE,
            "min_cosine": MIN_COSINE,
            "maximum_observed_abs": max(max(row["image_max_abs"], row["template_max_abs"]) for row in audit_rows),
            "minimum_observed_cosine": min(min(row["image_cosine"], row["template_cosine"]) for row in audit_rows),
            "rows": audit_rows,
        },
        "aggregate_sha256": sha256_file(AGGREGATE),
        "shared_encoder_parameter_schema_sha256": observed_encoder_schema,
        "shared_encoder_parameter_count": observed_encoder_count,
        "shared_d1_parameter_schema_sha256": parameter_schema_sha256(d1),
        "shared_d1_parameter_count": d1_count,
        "reference_defined_model_boundary": {
            "encoder_shared_across_all_references": True,
            "d1_shared_across_all_references": True,
            "per_identity_or_gallery_row_or_slot_parameter_count": 0,
            "new_reference_requires_parameter_update": False,
            "new_reference_allowed_mutation": "REFERENCE_TOKEN_CACHE_AND_INDEX_ONLY",
        },
        "access": {
            "token_shard_artifact_deserialization_count": SHARD_COUNT,
            "validation_current64_artifact_deserialization_count": 8,
            "production_current64_artifact_deserialization_count": SHARD_COUNT * 8,
            "current64_candidate_score_axis_semantic_consumption_count": 0,
            "protected_metadata_read_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
            "encoder_load_count": 1,
            "encoder_forward_count": len(audit_rows),
            "encoder_update_count": 0,
            "d1_schema_construction_count": 1,
            "d1_checkpoint_load_count": 0,
            "d1_update_count": 0,
            "action_head_update_count": 0,
        },
        "runtime_diagnostics_not_in_model_identity": {
            "cuda_device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "scientific_GO_or_NO_GO": None,
        "n2_d1_training_authorized": False,
        "external_execution_authorized": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    if existing is None:
        atomic_json(OUT, value)
    else:
        require(existing_final_is_current(existing, value), "stale or corrupt immutable final validation fails closed")
    print(json.dumps({"status": value["status"], "checks": checks, "next_authorized_stage": value["next_authorized_stage"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
