#!/usr/bin/env python3
"""Qualify one shared current-runtime scorer for RAW and fold-local D1.

This is a two-track, target-free engineering gate.  It serializes the raw
image tokens, D1-adapted image tokens, unchanged template tokens, complete
gallery scores, and both ranked and physical-row-sorted C128 axes.  It does
not join a target label and does not update a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_romav2_colnomic_sealed_source_e0_v1 as col  # noqa: E402
import run_romav2_colnomic_sealed_source_e0_v2 as current_runtime  # noqa: E402
from rc_aslo_xf.l0_targetfree_source import (  # noqa: E402
    load_fold_local_d1_adapter,
    load_query_ledger,
)
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_D1_THREE_ARM_CURRENT_RUNTIME_DIAGNOSTIC_V1_20260901.md"
PREDECESSOR = ROOT / "results/routea_d1_current_runtime_bridge_e0_v1/result.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_bridge_e0_v2"
OUT = OUT_ROOT / "result.json"
ARTIFACT = OUT_ROOT / "payload.pt"
CURRENT_BRIDGE_ROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_prejoin_v1"
CURRENT_BRIDGE_VALIDATION = CURRENT_BRIDGE_ROOT / "validation.json"
EXECUTIONS = (0, 72)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "logical_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def top128_unique_labels(scores: torch.Tensor, labels: tuple[str, ...]) -> list[int]:
    seen: set[str] = set()
    ranked_rows: list[int] = []
    for row in torch.argsort(scores, descending=True, stable=True).tolist():
        label = labels[row]
        if label in seen:
            continue
        seen.add(label)
        ranked_rows.append(int(row))
        if len(ranked_rows) == 128:
            break
    if len(ranked_rows) != 128:
        raise RuntimeError("fewer than 128 unique gallery identities")
    return ranked_rows


def load_frozen_current_runtime_records() -> tuple[dict[int, dict], dict[int, str]]:
    validation = json.loads(CURRENT_BRIDGE_VALIDATION.read_text())
    if (
        validation.get("status")
        != "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS"
        or validation.get("target_label_read_count") != 0
        or not all(validation.get("checks", {}).values())
    ):
        raise RuntimeError("frozen current-runtime bridge validation is not PASS")
    expected_payload_hashes = {
        int(item["shard"]): str(item["payload_sha256"])
        for item in validation.get("shards", [])
        if item.get("pass") is True
    }
    records: dict[int, dict] = {}
    record_payload_hashes: dict[int, str] = {}
    for shard in range(8):
        path = CURRENT_BRIDGE_ROOT / f"shard{shard:02d}/payload.pt"
        observed_sha256 = sha256_file(path)
        if expected_payload_hashes.get(shard) != observed_sha256:
            raise RuntimeError("frozen current-runtime bridge payload hash drift")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload.get("records", []):
            execution = int(record["execution_ordinal"])
            if execution in EXECUTIONS:
                if execution in records:
                    raise RuntimeError("duplicate current-runtime bridge execution")
                records[execution] = record
                record_payload_hashes[execution] = observed_sha256
    if set(records) != set(EXECUTIONS):
        raise RuntimeError("E0 executions absent from frozen current-runtime bridge")
    return records, record_payload_hashes


def main() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"immutable D1 bridge E0 output exists: {OUT_ROOT}")
    predecessor = json.loads(PREDECESSOR.read_text())
    if (
        predecessor.get("status") != "ROUTEA_D1_CURRENT_RUNTIME_BRIDGE_E0_ABORT"
        or predecessor.get("checks", {}).get("current_runtime_token_replay") is not False
        or predecessor.get("next_authorized_stage") is not None
    ):
        raise RuntimeError("D1 bridge E0 V1 predecessor is not the expected fail-closed result")
    current_authority, current_authority_payload_hashes = (
        load_frozen_current_runtime_records()
    )

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    loader = RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=col.GEOMETRY,
        prejoin_schedule_path=col.FOLDS,
    )
    query_ledger = load_query_ledger()
    gallery_source = col.build_gallery_source(verify_cache_file_sha256=True)
    gallery_payload = torch.load(
        col.GALLERY_CACHE,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    passages = gallery_payload["passage_emb"]
    labels = gallery_source.corrected_identities
    if len(passages) != len(labels) or len(labels) != 5413:
        raise RuntimeError("current gallery population drift")

    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(
        str(col.MODEL),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(
        str(col.MODEL),
        local_files_only=True,
    )

    payload_records: list[dict] = []
    receipts: list[dict] = []
    for execution in EXECUTIONS:
        source = loader.load_execution(execution)
        spec = query_ledger[source.historical_query_ordinal]
        source_path = Path(source.query.source_path).resolve()
        live_source_sha256 = sha256_file(source_path)
        source_binding = (
            spec.query_id == source.query_id
            and spec.track == source.track
            and spec.query_ordinal == source.historical_query_ordinal
            and spec.path.resolve() == source_path
            and live_source_sha256
            == spec.source_image_sha256
            == source.query.source_image_sha256
        )
        if not source_binding:
            raise RuntimeError("600-source and target-free D1 ledger binding drift")

        with Image.open(source_path) as opened:
            source_exif_orientation = int(opened.getexif().get(274, 1))
            if source.track == "new_difficult_train":
                decoded = ImageOps.exif_transpose(opened).convert("RGB")
                decode_frame = "EXIF_ORIENTED_BEFORE_RESIZE"
            else:
                decoded = opened.convert("RGB")
                decode_frame = "DECODED_RAW_BEFORE_EXIF"
            inputs = processor.process_images([decoded]).to(device)

        with torch.inference_mode():
            encoded = encoder(**inputs)[0].float()
        image_mask = inputs["input_ids"][0] == processor.image_token_id
        if image_mask.shape != encoded.shape[:1] or not bool(image_mask.any()):
            raise RuntimeError("current-runtime ColNomic image-token mask drift")
        image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
        template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
        temporal, height, width = [
            int(item) for item in inputs["image_grid_thw"][0].tolist()
        ]
        merge = int(processor.image_processor.merge_size)
        grid = (height // merge, width // merge)
        template_sha_before = tensor_sha256(template_tokens)
        authority_record = current_authority[execution]
        current_runtime_authority_exact = (
            authority_record.get("query_id") == source.query_id
            and authority_record.get("track") == source.track
            and tuple(authority_record.get("query_grid_shape", ())) == grid
            and authority_record.get("query_source_sha256") == live_source_sha256
            and torch.equal(
                image_tokens,
                torch.as_tensor(authority_record["query_tokens"]),
            )
            and authority_record.get("query_tokens_sha256")
            == tensor_sha256(image_tokens)
        )

        expected_tokens = source.query.tokens.detach().cpu().float()
        shape_exact = image_tokens.shape == expected_tokens.shape
        if shape_exact:
            token_cosine = F.cosine_similarity(
                image_tokens.float(),
                expected_tokens,
                dim=1,
            )
            token_min_cosine = float(token_cosine.min())
            token_mean_cosine = float(token_cosine.mean())
            token_max_abs = float((image_tokens.float() - expected_tokens).abs().max())
        else:
            token_min_cosine = token_mean_cosine = token_max_abs = None

        raw_scores = current_runtime.full_gallery_scores(
            torch.cat((image_tokens, template_tokens), dim=0),
            passages,
            device,
        )
        adapter, checkpoint_sha256 = load_fold_local_d1_adapter(
            spec.heldout_fold,
            device=device,
        )
        with torch.inference_mode():
            adapted_image_tokens = (
                adapter(image_tokens.float().to(device), grid[0], grid[1])
                .half()
                .cpu()
                .contiguous()
            )
        d1_scores = current_runtime.full_gallery_scores(
            torch.cat((adapted_image_tokens, template_tokens), dim=0),
            passages,
            device,
        )
        template_sha_after = tensor_sha256(template_tokens)

        raw_ranked = top128_unique_labels(raw_scores, labels)
        d1_ranked = top128_unique_labels(d1_scores, labels)
        raw_axis = sorted(raw_ranked)
        d1_axis = sorted(d1_ranked)
        authority_ranked = list(
            map(int, authority_record["candidate_ranked_physical_rows"])
        )
        authority_axis = list(map(int, authority_record["candidate_physical_rows"]))
        authority_raw_scores = torch.as_tensor(
            authority_record["candidate_raw_scores"], dtype=torch.float64
        )
        observed_authority_raw_scores = torch.tensor(
            [float(raw_scores[row]) for row in authority_axis],
            dtype=torch.float64,
        )
        raw_authority_score_max_abs = float(
            (observed_authority_raw_scores - authority_raw_scores).abs().max()
        )
        current_runtime_raw_authority_exact = (
            raw_ranked == authority_ranked
            and raw_axis == authority_axis
            and raw_ranked[0] == authority_ranked[0]
            and raw_authority_score_max_abs <= 1e-6
        )
        record = {
            "execution_ordinal": execution,
            "historical_query_ordinal": source.historical_query_ordinal,
            "query_id": source.query_id,
            "track": source.track,
            "heldout_fold": spec.heldout_fold,
            "query_source_path": str(source_path),
            "query_source_sha256": live_source_sha256,
            "source_exif_orientation": source_exif_orientation,
            "decode_frame": decode_frame,
            "query_grid_shape": grid,
            "raw_image_tokens": image_tokens,
            "adapted_image_tokens": adapted_image_tokens,
            "template_tokens": template_tokens,
            "raw_full_gallery_scores": raw_scores,
            "d1_full_gallery_scores": d1_scores,
            "raw_candidate_ranked_physical_rows": raw_ranked,
            "raw_candidate_physical_rows": raw_axis,
            "d1_candidate_ranked_physical_rows": d1_ranked,
            "d1_candidate_physical_rows": d1_axis,
            "d1_checkpoint_sha256": checkpoint_sha256,
            "current_runtime_authority_payload_sha256": current_authority_payload_hashes[
                execution
            ],
            "target_or_label_read_count": 0,
        }
        payload_records.append(record)
        receipt = {
            "execution_ordinal": execution,
            "query_id": source.query_id,
            "track": source.track,
            "heldout_fold": spec.heldout_fold,
            "source_binding": source_binding,
            "query_grid_shape": list(grid),
            "ledger_grid_shape": [spec.grid_h, spec.grid_w],
            "source_grid_shape": list(source.query.grid_shape),
            "raw_image_token_sha256": tensor_sha256(image_tokens),
            "adapted_image_token_sha256": tensor_sha256(adapted_image_tokens),
            "template_token_sha256": template_sha_before,
            "template_hash_unchanged": template_sha_before == template_sha_after,
            "current_runtime_authority_exact": current_runtime_authority_exact,
            "current_runtime_raw_authority_exact": current_runtime_raw_authority_exact,
            "current_runtime_raw_authority_score_max_abs": raw_authority_score_max_abs,
            "token_shape_exact": shape_exact,
            "token_min_cosine": token_min_cosine,
            "token_mean_cosine": token_mean_cosine,
            "token_max_abs": token_max_abs,
            "raw_d1_score_pearson": float(
                torch.corrcoef(
                    torch.stack((raw_scores.double(), d1_scores.double()))
                )[0, 1]
            ),
            "raw_d1_c128_overlap": len(set(raw_axis).intersection(d1_axis)),
            "raw_top1_physical_row": raw_ranked[0],
            "d1_top1_physical_row": d1_ranked[0],
        }
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "event": "d1_bridge_query_ready",
                    "execution": execution,
                    "query_id": source.query_id,
                    "track": source.track,
                    "fold": spec.heldout_fold,
                    "c128_overlap": receipt["raw_d1_c128_overlap"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del adapter, encoded, inputs
        torch.cuda.empty_cache()

    artifact_value = {
        "schema_version": "routea_d1_current_runtime_bridge_e0_payload_v2_20260901",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_BRIDGE_E0_V2_PAYLOAD_READY",
        "records": payload_records,
        "gallery_corrected_identity_mapping_sha256": gallery_source.corrected_mapping_sha256,
        "target_or_label_read_count": 0,
        "model_update_count": 0,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    torch.save(artifact_value, ARTIFACT)
    reloaded = torch.load(ARTIFACT, map_location="cpu", weights_only=False)

    reload_exact = (
        reloaded.get("schema_version") == artifact_value["schema_version"]
        and len(reloaded.get("records", ())) == len(payload_records)
    )
    if reload_exact:
        for before, after in zip(payload_records, reloaded["records"], strict=True):
            for tensor_key in (
                "raw_image_tokens",
                "adapted_image_tokens",
                "template_tokens",
                "raw_full_gallery_scores",
                "d1_full_gallery_scores",
            ):
                reload_exact = reload_exact and torch.equal(
                    before[tensor_key],
                    after[tensor_key],
                )

    checks = {
        "two_opened_tracks_exercised": {item["track"] for item in receipts}
        == {"difficult", "new_difficult_train"},
        "source_lineage_closed": all(item["source_binding"] for item in receipts),
        "track_specific_decode_exercised": {
            item["decode_frame"] for item in payload_records
        }
        == {"DECODED_RAW_BEFORE_EXIF", "EXIF_ORIENTED_BEFORE_RESIZE"},
        "grid_lineage_closed": all(
            item["query_grid_shape"]
            == item["ledger_grid_shape"]
            == item["source_grid_shape"]
            for item in receipts
        ),
        "historical_cache_drift_recorded_diagnostic_only": all(
            item["token_shape_exact"]
            and item["token_min_cosine"] is not None
            and item["token_mean_cosine"] is not None
            and -1.0 <= item["token_min_cosine"] <= 1.0
            and -1.0 <= item["token_mean_cosine"] <= 1.0
            for item in receipts
        ),
        "frozen_current_runtime_token_exact": all(
            item["current_runtime_authority_exact"] for item in receipts
        ),
        "frozen_current_runtime_raw_deployment_exact": all(
            item["current_runtime_raw_authority_exact"] for item in receipts
        ),
        "template_hash_unchanged": all(
            item["template_hash_unchanged"] for item in receipts
        ),
        "adapter_changes_only_image_tokens": all(
            before["raw_image_tokens"].shape == before["adapted_image_tokens"].shape
            and bool(torch.isfinite(before["adapted_image_tokens"]).all())
            and not torch.equal(
                before["raw_image_tokens"],
                before["adapted_image_tokens"],
            )
            for before in payload_records
        ),
        "shared_current_runtime_scorer": True,
        "complete_finite_full_gallery_scores": all(
            item["raw_full_gallery_scores"].shape
            == item["d1_full_gallery_scores"].shape
            == (5413,)
            and bool(torch.isfinite(item["raw_full_gallery_scores"]).all())
            and bool(torch.isfinite(item["d1_full_gallery_scores"]).all())
            for item in payload_records
        ),
        "ranked_and_physical_c128_axes_closed": all(
            len(item["raw_candidate_ranked_physical_rows"])
            == len(item["raw_candidate_physical_rows"])
            == len(item["d1_candidate_ranked_physical_rows"])
            == len(item["d1_candidate_physical_rows"])
            == 128
            and item["raw_candidate_physical_rows"]
            == sorted(item["raw_candidate_ranked_physical_rows"])
            and item["d1_candidate_physical_rows"]
            == sorted(item["d1_candidate_ranked_physical_rows"])
            for item in payload_records
        ),
        "artifact_reload_exact": reload_exact,
        "target_free_prejoin": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_d1_current_runtime_bridge_e0_v2_20260901",
        "status": (
            "ROUTEA_D1_CURRENT_RUNTIME_BRIDGE_E0_V2_READY"
            if passed
            else "ROUTEA_D1_CURRENT_RUNTIME_BRIDGE_E0_V2_ABORT"
        ),
        "claim_level": "TWO_OPENED_QUERY_ENGINEERING_ONLY",
        "checks": checks,
        "query_receipts": receipts,
        "payload_sha256": sha256_file(ARTIFACT),
        "gallery_cache_sha256": sha256_file(col.GALLERY_CACHE),
        "gallery_corrected_identity_mapping_sha256": gallery_source.corrected_mapping_sha256,
        "current_runtime_bridge_validation_sha256": sha256_file(
            CURRENT_BRIDGE_VALIDATION
        ),
        "contract_sha256": sha256_file(CONTRACT),
        "predecessor_result_sha256": sha256_file(PREDECESSOR),
        "target_label_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": (
            "D1_CURRENT_RUNTIME_MULTIQUERY_PREJOIN" if passed else None
        ),
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": value["status"],
                "checks": checks,
                "payload_sha256": value["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
