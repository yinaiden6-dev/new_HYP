#!/usr/bin/env python3
"""Materialize one immutable target-free N2 current-runtime token shard."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

from rc_aslo_xf.l0_targetfree_source import load_query_ledger  # noqa: E402
from rc_aslo_xf.n2_current_runtime_token_cache_v1 import (  # noqa: E402
    CURRENT64_COUNT,
    FRESH_COUNT,
    QUERY_COUNT,
    READY_STATUS,
    SHARD_COUNT,
    expected_query_ordinals,
    logical_sha256,
    parameter_schema_sha256,
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
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_token_cache_v1"
VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_shard_v1.py"

EXPECTED = {
    E0_RESULT: "91b5685e2d3906dd86fac87705cefcb134d863248b2f9c989086feff6d1c4797",
    E0_VALIDATION: "6e2975fa61642d07654b876d44d24bb803d46d085b7f1cd236acc08aaa00f1b1",
    SCOPE_ADDENDUM: "dd3b667907ecebce34aec4d37cbfb5520c4e432ef71a630b74b9fcffd8b00264",
    SCOPE_VALIDATION: "c9ecdb33d94b68e8bcecf99384830b23e22b3c4eb0c593f828fdcb0b0bf8aa59",
    QUERY_LEDGER: "df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec",
    CURRENT64_VALIDATION: "6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb",
    MODEL_LINEAGE: "6d5670561a2188deaea01f9c4373f2d4435ba45114307d95bcde555e5e7f2c82",
}
EXPECTED_ENCODER_FINGERPRINT = "af1f3f83297ac2ab737a50704455fea1da10635b7b14cf101ba0dc02ec30191f"
EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256 = "936cf64caa5ce47f808a3a981112deb7967583a690f78c631dc9e762da5cbe99"
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


def expected_access() -> dict[str, int]:
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


def expected_bindings() -> dict[str, str]:
    return {
        "contract_sha256": sha256_file(CONTRACT),
        "e0_result_sha256": sha256_file(E0_RESULT),
        "e0_validation_sha256": sha256_file(E0_VALIDATION),
        "scope_addendum_sha256": sha256_file(SCOPE_ADDENDUM),
        "scope_validation_sha256": sha256_file(SCOPE_VALIDATION),
        "query_ledger_sha256": sha256_file(QUERY_LEDGER),
        "current64_validation_sha256": sha256_file(CURRENT64_VALIDATION),
        "model_lineage_sha256": EXPECTED[MODEL_LINEAGE],
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "validator_sha256": sha256_file(VALIDATOR),
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


def validate_authorities() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), f"required authority absent: {path}")
        require(sha256_file(path) == expected, f"required authority hash drift: {path}")
    e0 = json.loads(E0_RESULT.read_text())
    e0_validation = json.loads(E0_VALIDATION.read_text())
    scope = json.loads(SCOPE_VALIDATION.read_text())
    # The lineage file is byte-bound only.  Its unrelated gallery subtree is
    # deliberately not deserialized or semantically consumed here.
    lineage_sha256 = EXPECTED[MODEL_LINEAGE]
    require(
        e0.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_READY"
        and e0_validation.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED"
        and scope.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED"
        and scope.get("logical_sha256") == EXPECTED_SCOPE_VALIDATION_LOGICAL_SHA256
        and scope.get("logical_sha256") == logical_sha256(scope)
        and all(scope.get("checks", {}).values())
        and scope.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT"
        and scope.get("bindings", {}).get("addendum_sha256") == sha256_file(SCOPE_ADDENDUM)
        and scope.get("bindings", {}).get("e0_result_sha256") == sha256_file(E0_RESULT)
        and scope.get("bindings", {}).get("e0_validation_sha256") == sha256_file(E0_VALIDATION),
        "N2 E0 plus scope-correction authority is not closed",
    )
    require(
        lineage_sha256 == EXPECTED[MODEL_LINEAGE],
        "stable current-runtime encoder lineage drift",
    )
    return scope, {"physical_sha256": lineage_sha256}


def current64_by_query() -> dict[str, tuple[dict[str, Any], int, str]]:
    validation = json.loads(CURRENT64_VALIDATION.read_text())
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("query_count") == CURRENT64_COUNT
        and set(seals) == set(range(8))
        and all(validation.get("checks", {}).values()),
        "current64 aggregate authority drift",
    )
    records: dict[str, tuple[dict[str, Any], int, str]] = {}
    for shard in range(8):
        payload_path = CURRENT64_ROOT / f"shard{shard:02d}/payload.pt"
        receipt_path = CURRENT64_ROOT / f"shard{shard:02d}/receipt.json"
        validation_path = CURRENT64_ROOT / f"shard{shard:02d}/validation.json"
        seal = seals[shard]
        require(
            seal.get("payload_sha256") == sha256_file(payload_path)
            and seal.get("receipt_sha256") == sha256_file(receipt_path)
            and seal.get("validation_sha256") == sha256_file(validation_path),
            f"current64 shard {shard} seal drift",
        )
        payload_sha = sha256_file(payload_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            require(query_id not in records, "duplicate current64 query_id")
            records[query_id] = (record, shard, payload_sha)
    require(len(records) == CURRENT64_COUNT, "current64 query population drift")
    return records


def image_receipt(spec: Any) -> tuple[Image.Image, dict[str, Any]]:
    source_path = Path(spec.path).resolve()
    require(sha256_file(source_path) == spec.source_image_sha256, "source image hash drift")
    with Image.open(source_path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        raw_size_hw = (int(opened.height), int(opened.width))
        if spec.track == "new_difficult_train":
            image = ImageOps.exif_transpose(opened).convert("RGB")
            decode_frame = "EXIF_ORIENTED_BEFORE_RESIZE"
        else:
            image = opened.convert("RGB")
            decode_frame = "DECODED_RAW_BEFORE_EXIF"
    return image, {
        "source_exif_orientation": orientation,
        "decode_frame": decode_frame,
        "raw_size_hw": raw_size_hw,
        "oriented_size_hw": (int(image.height), int(image.width)),
    }


def encode(image: Image.Image, *, encoder: torch.nn.Module, processor: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    inputs = processor.process_images([image]).to(device)
    with torch.inference_mode():
        encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    require(image_mask.shape == encoded.shape[:1] and bool(image_mask.any()), "image-token mask drift")
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
    temporal, height, width = [int(item) for item in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    require(temporal == 1, "temporal grid drift")
    return image_tokens, template_tokens, (height // merge, width // merge)


def existing_complete(out_dir: Path, shard: int) -> bool:
    payload_path, receipt_path = out_dir / "payload.pt", out_dir / "receipt.json"
    if not (payload_path.is_file() and receipt_path.is_file()):
        return False
    receipt = json.loads(receipt_path.read_text())
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    ordinals = expected_query_ordinals(shard)
    mode_counts = dict(
        sorted(Counter(str(record.get("materialization_mode")) for record in payload.get("records", [])).items())
    )
    return (
        set(payload) == PAYLOAD_KEYS
        and set(payload.get("model", {})) == MODEL_KEYS
        and payload.get("schema_version")
        == "routea_n2_current_runtime_987_token_cache_shard_v1_20260902"
        and payload.get("status") == READY_STATUS
        and payload.get("claim_level") == "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY"
        and payload.get("shard") == shard
        and payload.get("shard_count") == SHARD_COUNT
        and payload.get("query_ordinal_begin") == ordinals[0]
        and payload.get("query_ordinal_end_exclusive") == ordinals[-1] + 1
        and len(payload.get("records", [])) == len(ordinals)
        and payload.get("bindings") == expected_bindings()
        and payload.get("access") == expected_access()
        and payload.get("model", {}).get("stable_encoder_fingerprint_sha256")
        == EXPECTED_ENCODER_FINGERPRINT
        and payload.get("model", {}).get("encoder_trainable_parameter_count") == 0
        and payload.get("model", {}).get("d1_checkpoint_load_count") == 0
        and payload.get("model", {}).get("d1_model_update_count") == 0
        and receipt.get("schema_version")
        == "routea_n2_current_runtime_987_token_cache_receipt_v1_20260902"
        and receipt.get("status") == READY_STATUS
        and receipt.get("shard") == shard
        and receipt.get("query_count") == len(ordinals)
        and receipt.get("query_ordinal_begin") == ordinals[0]
        and receipt.get("query_ordinal_end_exclusive") == ordinals[-1] + 1
        and receipt.get("mode_counts") == mode_counts
        and receipt.get("payload_sha256") == sha256_file(payload_path)
        and receipt.get("model_update_count") == 0
        and receipt.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_SHARD_VALIDATION"
        and receipt.get("logical_sha256") == logical_sha256(receipt)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard must be in 0..15")
    out_dir = OUT_ROOT / f"shard{args.shard:02d}"
    scope, lineage = validate_authorities()
    if out_dir.exists():
        if existing_complete(out_dir, args.shard):
            print(json.dumps({"event": "n2_token_cache_shard_already_committed", "shard": args.shard}), flush=True)
            return
        raise RuntimeError(f"corrupt/incomplete committed shard fails closed: {out_dir}")

    ledger = load_query_ledger()
    require(len(ledger) == QUERY_COUNT, "canonical ledger population drift")
    ordinals = expected_query_ordinals(args.shard)
    specs = [ledger[ordinal] for ordinal in ordinals]
    require(all(spec.query_ordinal == ordinal for spec, ordinal in zip(specs, ordinals)), "ledger ordinal drift")
    old = current64_by_query()

    if not torch.cuda.is_available():
        raise RuntimeError("N2 token materialization requires CUDA")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(MODEL), local_files_only=True)
    encoder_schema_sha = parameter_schema_sha256(encoder)
    encoder_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    require(all(not parameter.requires_grad for parameter in encoder.parameters()), "encoder is not frozen")

    records: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()
    for local_ordinal, spec in enumerate(specs):
        image, geometry = image_receipt(spec)
        authority_tuple = old.get(spec.query_id)
        if authority_tuple is None:
            image_tokens, template_tokens, grid = encode(
                image, encoder=encoder, processor=processor, device=device
            )
            mode = "FRESH_CURRENT_RUNTIME_FORWARD"
            source_shard = None
            source_payload_sha = None
        else:
            authority, source_shard, source_payload_sha = authority_tuple
            image_tokens = torch.as_tensor(authority["raw_image_tokens"]).detach().cpu().contiguous().clone()
            template_tokens = torch.as_tensor(authority["template_tokens"]).detach().cpu().contiguous().clone()
            grid = tuple(map(int, authority["query_grid_shape"]))
            require(
                authority.get("query_id") == spec.query_id
                and authority.get("track") == spec.track
                and int(authority.get("heldout_fold")) == int(spec.heldout_fold)
                and authority.get("query_source_sha256") == spec.source_image_sha256
                and authority.get("decode_frame") == geometry["decode_frame"]
                and int(authority.get("source_exif_orientation")) == geometry["source_exif_orientation"],
                "current64 canonical/source replay drift",
            )
            mode = "CURRENT64_EXACT_REUSE"
        expected_grid = (int(spec.grid_h), int(spec.grid_w))
        require(
            grid == expected_grid
            and image_tokens.dtype == template_tokens.dtype == torch.float16
            and image_tokens.shape == (grid[0] * grid[1], 128)
            and template_tokens.ndim == 2
            and template_tokens.shape[0] > 0
            and template_tokens.shape[1] == 128
            and bool(torch.isfinite(image_tokens).all())
            and bool(torch.isfinite(template_tokens).all()),
            "current-runtime tensor/grid schema drift",
        )
        record = {
            "query_id": str(spec.query_id),
            "query_ordinal": int(spec.query_ordinal),
            "heldout_fold": int(spec.heldout_fold),
            "track": str(spec.track),
            "source_image_sha256": str(spec.source_image_sha256),
            **geometry,
            "grid_shape": grid,
            "image_tokens": image_tokens,
            "image_tokens_sha256": tensor_sha256(image_tokens),
            "template_tokens": template_tokens,
            "template_tokens_sha256": tensor_sha256(template_tokens),
            "materialization_mode": mode,
            "current64_source_shard": source_shard,
            "current64_source_payload_sha256": source_payload_sha,
            "model_update_count": 0,
        }
        validate_prejoin_record_schema(record)
        records.append(record)
        modes[mode] += 1
        print(
            json.dumps(
                {
                    "event": "n2_token_cache_query_ready",
                    "shard": args.shard,
                    "local": local_ordinal,
                    "query_ordinal": spec.query_ordinal,
                    "query_id": spec.query_id,
                    "mode": mode,
                    "grid": list(grid),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    access = expected_access()
    payload = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_shard_v1_20260902",
        "status": READY_STATUS,
        "claim_level": "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "query_ordinal_begin": ordinals[0],
        "query_ordinal_end_exclusive": ordinals[-1] + 1,
        "records": records,
        "model": {
            "stable_encoder_fingerprint_sha256": EXPECTED_ENCODER_FINGERPRINT,
            "encoder_parameter_schema_sha256": encoder_schema_sha,
            "encoder_parameter_count": encoder_parameter_count,
            "encoder_trainable_parameter_count": 0,
            "d1_checkpoint_load_count": 0,
            "d1_model_update_count": 0,
            "hardware_identity_is_authority": False,
        },
        "bindings": expected_bindings(),
        "access": access,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".n2-token-shard{args.shard:02d}-", dir=OUT_ROOT))
    try:
        payload_path = staging / "payload.pt"
        torch.save(payload, payload_path)
        with payload_path.open("rb") as handle:
            os.fsync(handle.fileno())
        receipt = {
            "schema_version": "routea_n2_current_runtime_987_token_cache_receipt_v1_20260902",
            "status": READY_STATUS,
            "shard": args.shard,
            "query_count": len(records),
            "query_ordinal_begin": ordinals[0],
            "query_ordinal_end_exclusive": ordinals[-1] + 1,
            "mode_counts": dict(sorted(modes.items())),
            "payload_sha256": sha256_file(payload_path),
            "model_update_count": 0,
            "next_authorized_stage": "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_SHARD_VALIDATION",
            "logical_sha256": "",
        }
        receipt["logical_sha256"] = logical_sha256(receipt)
        receipt_path = staging / "receipt.json"
        with receipt_path.open("w") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
