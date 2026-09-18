#!/usr/bin/env python3
"""Materialize target-free matched A/B/C scores on RAW and D1 axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from torch.nn import functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa: E402
import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import (  # noqa: E402
    DECODED_RAW_BEFORE_EXIF,
    EXIF_ORIENTED_BEFORE_RESIZE,
    build_colnomic_canonical_geometry_v2,
)
from romav2 import RoMaV2  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_V1_20260902.md"
SOURCE_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
SOURCE_VALIDATION = SOURCE_ROOT / "validation.json"
REFERENCE_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
LABEL_JOIN_HANDOFF = ROOT / "results/routea_d1_current_runtime_64_label_join_v1/independent_validation.json"
RAW_C_ROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
RAW_C_VALIDATION = RAW_C_ROOT / "validation.json"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
EXPECTED_ROMA_CHECKPOINT_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
PROCESSOR_CONFIG = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
SHARD_COUNT = 8
ARM_NAMES = ("A_ALL", "B_QUERY", "C_PAIRED")


def map_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def geometry(
    path: Path,
    source_sha256: str,
    source_key: str,
    grid_shape: tuple[int, int],
    frame: str,
):
    with Image.open(path) as image:
        raw_hw = (int(image.height), int(image.width))
        orientation = int(image.getexif().get(274, 1))
    return build_colnomic_canonical_geometry_v2(
        source_image_sha256=source_sha256,
        source_key=source_key,
        processor_config_sha256=e0.sha256_file(PROCESSOR_CONFIG),
        raw_size_hw=raw_hw,
        exif_orientation=orientation,
        merged_grid_shape=grid_shape,
        processor_input_frame=frame,
    )


def matched_arm_scores(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_map: torch.Tensor,
    reference_map: torch.Tensor,
) -> dict[str, float]:
    query_unit = F.normalize(query_tokens.to(torch.float64), dim=1)
    reference_unit = F.normalize(reference_tokens.to(torch.float64), dim=1)
    similarity = query_unit @ reference_unit.T
    query_weight = query_map.to(torch.float64)
    reference_weight = reference_map.to(torch.float64)
    if (
        query_weight.ndim != 1
        or reference_weight.ndim != 1
        or similarity.shape != (query_weight.numel(), reference_weight.numel())
        or not bool(torch.isfinite(similarity).all())
        or not bool(torch.isfinite(query_weight).all())
        or not bool(torch.isfinite(reference_weight).all())
        or float(query_weight.min()) < 0.0
        or float(query_weight.max()) > 1.0
        or float(reference_weight.min()) < 0.0
        or float(reference_weight.max()) > 1.0
    ):
        raise RuntimeError("matched-arm tensor shape or finiteness drift")
    local_all = similarity.max(dim=1).values
    score_a = local_all.mean()
    query_denominator = query_weight.sum().clamp_min(1e-12)
    score_b = torch.sqrt(query_weight.mean()) * (
        query_weight * local_all
    ).sum() / query_denominator
    local_paired = (similarity * reference_weight[None]).max(dim=1).values
    score_c = torch.sqrt(query_weight.mean() * reference_weight.mean()) * (
        query_weight * local_paired
    ).sum() / query_denominator
    scores = {
        "A_ALL": float(score_a),
        "B_QUERY": float(score_b),
        "C_PAIRED": float(score_c),
    }
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError("matched-arm score became non-finite")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard must be in 0..7")
    out_dir = OUT_ROOT / f"shard{args.shard:02d}"
    if out_dir.exists():
        raise RuntimeError(f"immutable matched-three-arm shard exists: {out_dir}")

    source_validation = json.loads(SOURCE_VALIDATION.read_text())
    reference_validation = json.loads(REFERENCE_VALIDATION.read_text())
    label_join_handoff = json.loads(LABEL_JOIN_HANDOFF.read_text())
    raw_c_validation = json.loads(RAW_C_VALIDATION.read_text())
    if (
        source_validation.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        or source_validation.get("logical_sha256")
        != e0.logical_sha256(source_validation)
        or not all(source_validation.get("checks", {}).values())
        or reference_validation.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        or reference_validation.get("logical_sha256")
        != e0.logical_sha256(reference_validation)
        or not all(reference_validation.get("checks", {}).values())
        or reference_validation.get("next_authorized_stage")
        != "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARDS"
        or label_join_handoff.get("status")
        != "ROUTEA_D1_CURRENT_RUNTIME_64_LABEL_JOIN_INDEPENDENT_VALIDATION_PASS"
        or label_join_handoff.get("logical_sha256")
        != e0.logical_sha256(label_join_handoff)
        or not all(label_join_handoff.get("checks", {}).values())
        or label_join_handoff.get("next_authorized_stage")
        != "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_TARGET_FREE_MATERIALIZATION"
        or raw_c_validation.get("status")
        != "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS"
        or not all(raw_c_validation.get("checks", {}).values())
        or raw_c_validation.get("target_label_read_count") != 0
        or e0.sha256_file(ROMA_WEIGHTS) != EXPECTED_ROMA_CHECKPOINT_SHA256
    ):
        raise RuntimeError("matched-three-arm input authority drift")

    sealed_source = {
        int(item["shard"]): item for item in source_validation["shards"]
    }[args.shard]
    source_payload_path = SOURCE_ROOT / f"shard{args.shard:02d}/payload.pt"
    source_receipt_path = SOURCE_ROOT / f"shard{args.shard:02d}/receipt.json"
    source_shard_validation_path = SOURCE_ROOT / f"shard{args.shard:02d}/validation.json"
    if not (
        sealed_source["payload_sha256"] == e0.sha256_file(source_payload_path)
        and sealed_source["receipt_sha256"] == e0.sha256_file(source_receipt_path)
        and sealed_source["validation_sha256"]
        == e0.sha256_file(source_shard_validation_path)
    ):
        raise RuntimeError("D1 source shard aggregate seal drift")
    source_payload = torch.load(
        source_payload_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    reference_payload = torch.load(
        REFERENCE_PAYLOAD,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if reference_validation.get("payload_sha256") != e0.sha256_file(REFERENCE_PAYLOAD):
        raise RuntimeError("reference-cache payload binding drift")
    references = reference_payload["references"]

    raw_c_path = RAW_C_ROOT / f"shard{args.shard:02d}/result.json"
    raw_c_seal = {
        int(item["shard"]): item for item in raw_c_validation["shards"]
    }[args.shard]
    if raw_c_seal.get("sha256") != e0.sha256_file(raw_c_path):
        raise RuntimeError("RAW+C authority shard hash drift")
    raw_c_payload = json.loads(raw_c_path.read_text())
    if raw_c_payload.get("logical_sha256") != core.logical(raw_c_payload):
        raise RuntimeError("RAW+C authority logical drift")
    raw_c_by_execution = {
        int(record["execution_ordinal"]): record for record in raw_c_payload["rows"]
    }

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    model = RoMaV2()
    output_records: list[dict] = []
    live_reference_sha256: dict[int, str] = {}
    for source in source_payload["records"]:
        execution = int(source["execution_ordinal"])
        query_path = Path(source["query_source_path"])
        if e0.sha256_file(query_path) != source["query_source_sha256"]:
            raise RuntimeError("live query bytes drifted after source seal")
        query_image = core.oriented(query_path)
        query_frame = (
            EXIF_ORIENTED_BEFORE_RESIZE
            if source["track"] == "new_difficult_train"
            else DECODED_RAW_BEFORE_EXIF
        )
        query_geometry = geometry(
            query_path,
            source["query_source_sha256"],
            f"matched-three-arm-query:{execution}",
            tuple(source["query_grid_shape"]),
            query_frame,
        )
        raw_axis = list(map(int, source["raw_candidate_physical_rows"]))
        d1_axis = list(map(int, source["d1_candidate_physical_rows"]))
        union_axis = sorted(set(raw_axis).union(d1_axis))
        if not (len(raw_axis) == len(d1_axis) == 128):
            raise RuntimeError("base C128 width drift")
        raw_position = {row: index for index, row in enumerate(raw_axis)}
        d1_position = {row: index for index, row in enumerate(d1_axis)}
        raw_authority = raw_c_by_execution.get(execution)
        if raw_authority is None or raw_authority.get("query_id") != source["query_id"]:
            raise RuntimeError("RAW+C query authority drift")
        raw_authority_candidates = {
            int(item["physical_row"]): item for item in raw_authority["candidates"]
        }
        if list(raw_authority_candidates) != raw_axis:
            raise RuntimeError("RAW+C authority axis drift")

        candidates: list[dict] = []
        for ordinal, physical_row in enumerate(union_axis):
            reference = references.get(physical_row)
            if reference is None:
                raise RuntimeError("union reference absent from validated cache")
            reference_path = Path(reference["source_path"])
            observed_reference_sha256 = live_reference_sha256.setdefault(
                physical_row, e0.sha256_file(reference_path)
            )
            if observed_reference_sha256 != reference["source_image_sha256"]:
                raise RuntimeError("live reference bytes drifted after cache seal")
            reference_geometry = geometry(
                reference_path,
                reference["source_image_sha256"],
                f"gallery-row:{physical_row}",
                tuple(reference["grid_shape"]),
                DECODED_RAW_BEFORE_EXIF,
            )
            prediction = model.match(query_image, core.oriented(reference_path))
            query_map = core.cell_means(
                prediction["overlap_AB"][0, ..., 0].detach().cpu(),
                query_geometry,
            ).contiguous()
            reference_map = core.cell_means(
                prediction["overlap_BA"][0, ..., 0].detach().cpu(),
                reference_geometry,
            ).contiguous()
            raw_scores = matched_arm_scores(
                source["raw_image_tokens"],
                reference["tokens"],
                query_map,
                reference_map,
            )
            d1_scores = matched_arm_scores(
                source["adapted_image_tokens"],
                reference["tokens"],
                query_map,
                reference_map,
            )
            query_hash = map_sha256(query_map)
            reference_hash = map_sha256(reference_map)
            if physical_row in raw_position:
                authority = raw_authority_candidates[physical_row]
                if not (
                    query_hash == authority["query_map_sha256"]
                    and reference_hash == authority["reference_map_sha256"]
                    and abs(raw_scores["C_PAIRED"] - float(authority["real_score"]))
                    <= 1e-12
                ):
                    raise RuntimeError("frozen RAW+C map or score replay drift")
            candidates.append(
                {
                    "union_ordinal": ordinal,
                    "physical_row": physical_row,
                    "raw_position": raw_position.get(physical_row),
                    "d1_position": d1_position.get(physical_row),
                    "reference_tokens_sha256": reference["tokens_sha256"],
                    "reference_source_image_sha256": observed_reference_sha256,
                    "reference_source_logical_sha256": reference[
                        "source_logical_sha256"
                    ],
                    "query_map": query_map,
                    "reference_map": reference_map,
                    "query_map_sha256": query_hash,
                    "reference_map_sha256": reference_hash,
                    "raw_arm_scores": raw_scores,
                    "d1_arm_scores": d1_scores,
                }
            )
            if (ordinal + 1) % 32 == 0 or ordinal + 1 == len(union_axis):
                print(
                    json.dumps(
                        {
                            "event": "matched_three_arm_candidate_progress",
                            "shard": args.shard,
                            "execution": execution,
                            "done": ordinal + 1,
                            "total": len(union_axis),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        output_records.append(
            {
                "role": source["role"],
                "execution_ordinal": execution,
                "query_id": source["query_id"],
                "track": source["track"],
                "heldout_fold": int(source["heldout_fold"]),
                "raw_candidate_physical_rows": raw_axis,
                "d1_candidate_physical_rows": d1_axis,
                "union_physical_rows": union_axis,
                "raw_cbind_destination_to_source_physical_rows": raw_axis[64:]
                + raw_axis[:64],
                "d1_cbind_destination_to_source_physical_rows": d1_axis[64:]
                + d1_axis[:64],
                "candidates": candidates,
                "target_role_read_count": 0,
                "target_insertion_count": 0,
                "model_update_count": 0,
            }
        )
        print(
            json.dumps(
                {
                    "event": "matched_three_arm_query_ready",
                    "shard": args.shard,
                    "execution": execution,
                    "query_id": source["query_id"],
                    "union_count": len(union_axis),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_prejoin_shard_v1_20260902",
        "status": "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_READY",
        "claim_level": "TARGET_FREE_MATCHED_A_B_C_RAW_AND_D1_PORTABILITY_DIAGNOSTIC",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "arm_names": list(ARM_NAMES),
        "records": output_records,
        "bindings": {
            "contract_sha256": e0.sha256_file(CONTRACT),
            "source_payload_sha256": e0.sha256_file(source_payload_path),
            "source_validation_sha256": e0.sha256_file(SOURCE_VALIDATION),
            "reference_payload_sha256": e0.sha256_file(REFERENCE_PAYLOAD),
            "reference_validation_sha256": e0.sha256_file(REFERENCE_VALIDATION),
            "label_join_handoff_sha256": e0.sha256_file(LABEL_JOIN_HANDOFF),
            "raw_c_authority_sha256": e0.sha256_file(raw_c_path),
            "raw_c_validation_sha256": e0.sha256_file(RAW_C_VALIDATION),
            "roma_checkpoint_sha256": e0.sha256_file(ROMA_WEIGHTS),
            "processor_config_sha256": e0.sha256_file(PROCESSOR_CONFIG),
        },
        "access": {
            "target_identity_read_count": 0,
            "target_role_read_count": 0,
            "supergroup_read_count": 0,
            "sealed_read_count": 0,
            "target_insertion_count": 0,
            "model_update_count": 0,
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".matched-three-arm-shard{args.shard:02d}-",
            dir=OUT_ROOT,
        )
    )
    stage_payload = staging / "payload.pt"
    stage_receipt = staging / "receipt.json"
    torch.save(value, stage_payload)
    receipt = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_prejoin_receipt_v1_20260902",
        "status": value["status"],
        "shard": args.shard,
        "query_count": len(output_records),
        "candidate_union_counts": [
            len(record["union_physical_rows"]) for record in output_records
        ],
        "payload_sha256": e0.sha256_file(stage_payload),
        "target_role_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": "MATCHED_THREE_ARM_SHARD_VALIDATION",
        "logical_sha256": "",
    }
    receipt["logical_sha256"] = e0.logical_sha256(receipt)
    try:
        stage_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.rename(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
