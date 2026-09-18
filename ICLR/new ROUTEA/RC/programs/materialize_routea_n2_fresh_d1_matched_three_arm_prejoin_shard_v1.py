#!/usr/bin/env python3
"""Materialize one target-free fresh-D1 current64 matched-three-arm shard.

This is the current64-only third S0 substage.  It reuses exact historical
RoMa maps by ``(query_id, physical_row)`` and runs RoMa only for pairs named
missing by the independently sealed source manifest.  It is deliberately not
the final Pair64+current64 prejoin status and cannot authorize head training.
"""

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
from typing import Any, Mapping

import torch
from torch.nn import functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "programs")]

from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import (  # noqa: E402
    DECODED_RAW_BEFORE_EXIF,
    EXIF_ORIENTED_BEFORE_RESIZE,
    build_colnomic_canonical_geometry_v2,
)
from romav2 import RoMaV2  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md"
MAP_REUSE_ADDENDUM = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_EXACT_ROMA_MAP_CACHE_REUSE_ADDENDUM_V1_20260904.md"
SOURCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.json"
SOURCE_VALIDATION = SOURCE_ROOT / "independent_validation.json"
REFERENCE_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_RESULT = REFERENCE_ROOT / "result.json"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
OOF_ROOT = ROOT / "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1"
ROSTER_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
ROSTER_VALIDATION = ROSTER_ROOT / "validation.json"
OLD_MAP_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
OLD_MAP_VALIDATION = OLD_MAP_ROOT / "validation.json"
OLD_MAP_PRODUCER = ROOT / "programs/materialize_routea_d1_current_runtime_matched_three_arm_shard_v1.py"
OLD_MAP_VALIDATOR = ROOT / "programs/validate_routea_d1_current_runtime_matched_three_arm_shard_v1.py"
GEOMETRY_SOURCE = ROOT / "src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
PROCESSOR_CONFIG = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"

VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_prejoin_shard_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_READY"
SHARD_COUNT = 8
QUERY_COUNT_PER_SHARD = 8
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
EXPECTED_ROMA_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
EXPECTED_PROCESSOR_CONFIG_SHA256 = "1a427e12a15406a5c4701273c91a0b319528de5c78523ae5ba5329d86e5d5557"
EXPECTED_MAP_REUSE_ADDENDUM_SHA256 = "4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6"
EXPECTED_OLD_MAP_VALIDATION_SHA256 = "60bd07480bf4963503993ecb926b696e3fc568c588189c557c0bc9975f32c66f"
EXPECTED_OLD_MAP_PRODUCER_SHA256 = "ca10470f271da2d6fef5cd59d11ad930c5d9045774774d2caac7df8d1b339383"
EXPECTED_OLD_MAP_VALIDATOR_SHA256 = "bd4cda9d350796cb6a7a7f2ba3c26047254a9566ea481e20338ad49ea0e9008d"
EXPECTED_GEOMETRY_SOURCE_SHA256 = "c732dc933c502f17fd5cb8cc80ae45d116716fe801cc8f3034a1e6a49cef6508"
MAP_REPLAY_ATOL = 1e-6
SOURCE_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
SOURCE_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
REFERENCE_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
REFERENCE_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_VALIDATED"


class MaterializationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


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


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def n2_tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def map_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        torch.as_tensor(value).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"JSON input absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def oriented(path: Path) -> Image.Image:
    with Image.open(path) as source:
        try:
            orientation = int(source.getexif().get(274, 1))
        except (TypeError, ValueError):
            orientation = 1
        image = source.copy()
    operation = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if operation is not None:
        image = image.transpose(operation)
    return image.convert("RGB")


def geometry(path: Path, source_sha256: str, source_key: str, grid: tuple[int, int], frame: str):
    with Image.open(path) as image:
        raw_hw = (int(image.height), int(image.width))
        orientation = int(image.getexif().get(274, 1))
    return build_colnomic_canonical_geometry_v2(
        source_image_sha256=source_sha256,
        source_key=source_key,
        processor_config_sha256=sha256_file(PROCESSOR_CONFIG),
        raw_size_hw=raw_hw,
        exif_orientation=orientation,
        merged_grid_shape=grid,
        processor_input_frame=frame,
    )


def geometry_payload(value: Any, frame: str) -> dict[str, Any]:
    affine = torch.as_tensor(value.raw_to_oriented_affine).detach().cpu().contiguous()
    valid = torch.as_tensor(value.valid_patch_mask).detach().cpu().contiguous()
    boxes = torch.as_tensor(value.cell_boxes_xyxy).detach().cpu().contiguous()
    return {
        "source_image_sha256": str(value.source_image_sha256),
        "source_key": str(value.source_key),
        "processor_config_sha256": str(value.processor_config_sha256),
        "raw_size_hw": tuple(map(int, value.raw_size_hw)),
        "oriented_size_hw": tuple(map(int, value.oriented_size_hw)),
        "exif_orientation": int(value.exif_orientation),
        "raw_to_oriented_affine": affine,
        "grid_shape": tuple(map(int, value.grid_shape)),
        "valid_patch_mask": valid,
        "cell_boxes_xyxy": boxes,
        "coordinate_frame": str(value.coordinate_frame),
        "processor_input_frame": frame,
        "raw_to_oriented_affine_sha256": tensor_sha256(affine),
        "valid_patch_mask_sha256": tensor_sha256(valid),
        "cell_boxes_xyxy_sha256": tensor_sha256(boxes),
    }


def cell_means(overlap: torch.Tensor, patch_geometry: Any) -> torch.Tensor:
    values = torch.as_tensor(overlap, dtype=torch.float64)
    height, width = values.shape
    output = []
    for box, valid in zip(patch_geometry.cell_boxes_xyxy, patch_geometry.valid_patch_mask, strict=True):
        if not bool(valid):
            output.append(values.new_zeros(()))
            continue
        x0, y0, x1, y1 = map(float, box)
        ix0 = max(0, min(width - 1, int(math.floor(x0 * width))))
        iy0 = max(0, min(height - 1, int(math.floor(y0 * height))))
        ix1 = max(ix0 + 1, min(width, int(math.ceil(x1 * width))))
        iy1 = max(iy0 + 1, min(height, int(math.ceil(y1 * height))))
        output.append(values[iy0:iy1, ix0:ix1].mean())
    return torch.stack(output).contiguous()


def score(query: torch.Tensor, reference: torch.Tensor, query_weight: torch.Tensor, reference_weight: torch.Tensor) -> tuple[float, float]:
    q = F.normalize(query.to(torch.float64), dim=1)
    r = F.normalize(reference.to(torch.float64), dim=1)
    wq = query_weight.to(torch.float64)
    wr = reference_weight.to(torch.float64)
    require(
        q.ndim == r.ndim == 2
        and q.shape[1] == r.shape[1] == 128
        and wq.shape == (q.shape[0],)
        and wr.shape == (r.shape[0],)
        and bool(torch.isfinite(q).all() and torch.isfinite(r).all() and torch.isfinite(wq).all() and torch.isfinite(wr).all())
        and float(wq.min()) >= 0.0
        and float(wq.max()) <= 1.0
        and float(wr.min()) >= 0.0
        and float(wr.max()) <= 1.0,
        "score tensor contract failed",
    )
    similarity = q @ r.T
    local = (similarity * wr[None]).max(dim=1).values
    mass = torch.sqrt(wq.mean() * wr.mean())
    value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
    require(bool(torch.isfinite(value) and torch.isfinite(mass)), "non-finite score")
    return float(value), float(mass)


def arm_evidence(query: torch.Tensor, reference: torch.Tensor, query_map: torch.Tensor, reference_map: torch.Tensor) -> dict[str, dict[str, float]]:
    query_unit = F.normalize(query.to(torch.float64), dim=1)
    reference_unit = F.normalize(reference.to(torch.float64), dim=1)
    similarity = query_unit @ reference_unit.T
    ones_q = torch.ones_like(query_map, dtype=torch.float64)
    ones_r = torch.ones_like(reference_map, dtype=torch.float64)
    half_q = max(1, query_map.numel() // 2)
    half_r = max(1, reference_map.numel() // 2)

    def reduce(qweight: torch.Tensor, rweight: torch.Tensor) -> tuple[float, float]:
        wq, wr = qweight.to(torch.float64), rweight.to(torch.float64)
        local = (similarity * wr[None]).max(dim=1).values
        mass = torch.sqrt(wq.mean() * wr.mean())
        value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
        require(bool(torch.isfinite(value) and torch.isfinite(mass)), "non-finite arm evidence")
        return float(value), float(mass)

    a, a_mass = reduce(ones_q, ones_r)
    b, b_mass = reduce(query_map, ones_r)
    b_q, _ = reduce(query_map.roll(half_q), ones_r)
    c, c_mass = reduce(query_map, reference_map)
    c_q, _ = reduce(query_map.roll(half_q), reference_map)
    c_r, _ = reduce(query_map, reference_map.roll(half_r))
    return {
        "A_ALL": {"real_score": a, "visibility_mass": a_mass, "query_control_score": a, "reference_control_score": a},
        "B_QUERY": {"real_score": b, "visibility_mass": b_mass, "query_control_score": b_q, "reference_control_score": b},
        "C_PAIRED": {"real_score": c, "visibility_mass": c_mass, "query_control_score": c_q, "reference_control_score": c_r},
    }


def symmetric(a: float, b: float) -> float:
    return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)


def candidate_feature(base_scores: list[float], evidence: Mapping[int, Mapping[str, float]], challenger: int, winner: int) -> torch.Tensor:
    mean = sum(base_scores) / len(base_scores)
    std = (sum((value - mean) ** 2 for value in base_scores) / len(base_scores)) ** 0.5
    challenger_evidence = evidence[challenger]
    winner_evidence = evidence[winner]
    score_c = float(challenger_evidence["real_score"]) / max(float(challenger_evidence["visibility_mass"]), 1e-12)
    score_w = float(winner_evidence["real_score"]) / max(float(winner_evidence["visibility_mass"]), 1e-12)
    query_c = float(challenger_evidence["real_score"]) - float(challenger_evidence["query_control_score"])
    query_w = float(winner_evidence["real_score"]) - float(winner_evidence["query_control_score"])
    reference_c = float(challenger_evidence["real_score"]) - float(challenger_evidence["reference_control_score"])
    reference_w = float(winner_evidence["real_score"]) - float(winner_evidence["reference_control_score"])
    return torch.tensor(
        [
            (base_scores[challenger] - base_scores[winner]) / max(std, 1e-12),
            symmetric(challenger_evidence["real_score"], winner_evidence["real_score"]),
            symmetric(challenger_evidence["visibility_mass"], winner_evidence["visibility_mass"]),
            symmetric(score_c, score_w),
            symmetric(query_c, query_w),
            symmetric(reference_c, reference_w),
        ],
        dtype=torch.float64,
    )


def validated_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(SOURCE_MANIFEST)
    source_validation = read_json(SOURCE_VALIDATION)
    reference_result = read_json(REFERENCE_RESULT)
    reference_validation = read_json(REFERENCE_VALIDATION)
    require(
        manifest.get("status") == SOURCE_READY
        and manifest.get("population", {}).get("query_count") == 64
        and manifest.get("population", {}).get("fresh_candidate_pair_count") == 8192
        and manifest.get("population", {}).get("old_map_reuse_pair_count") == 8168
        and manifest.get("population", {}).get("missing_map_pair_count") == 24
        and manifest.get("population", {}).get("pair64_pending_count") == 64
        and manifest.get("contract_population_complete") is False
        and manifest.get("bindings", {}).get("rematerialization_contract_sha256")
        == sha256_file(CONTRACT)
        and manifest.get("bindings", {}).get("exact_map_cache_reuse_addendum_sha256")
        == sha256_file(MAP_REUSE_ADDENDUM)
        and manifest.get("logical_sha256") == logical_sha256(manifest)
        and manifest.get("access", {}).get("target_identity_read_count") == 0
        and manifest.get("access", {}).get("query_supergroup_read_count") == 0
        and manifest.get("access", {}).get("retrieval_outcome_read_count") == 0
        and manifest.get("access", {}).get("action_read_count") == 0,
        "source manifest authority failed",
    )
    require(
        source_validation.get("status") == SOURCE_VALIDATED
        and source_validation.get("manifest_sha256") == sha256_file(SOURCE_MANIFEST)
        and source_validation.get("manifest_logical_sha256") == manifest["logical_sha256"]
        and all(source_validation.get("checks", {}).values())
        and source_validation.get("logical_sha256") == logical_sha256(source_validation)
        and source_validation.get("contract_population_complete") is False
        and source_validation.get("pair64_pending_count") == 64
        and source_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION",
        "source manifest independent validation failed",
    )
    require(
        reference_result.get("status") == REFERENCE_READY
        and reference_result.get("contract_population_complete") is False
        and reference_result.get("current64_query_count") == 64
        and reference_result.get("pair64_pending") == 64
        and reference_result.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and reference_result.get("logical_sha256") == logical_sha256(reference_result)
        and reference_result.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS",
        "current64 reference-cache result failed",
    )
    require(
        reference_validation.get("status") == REFERENCE_VALIDATED
        and reference_validation.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and reference_validation.get("producer_result_sha256") == sha256_file(REFERENCE_RESULT)
        and all(reference_validation.get("checks", {}).values())
        and reference_validation.get("logical_sha256") == logical_sha256(reference_validation)
        and reference_validation.get("contract_population_complete") is False
        and reference_validation.get("current64_query_count") == 64
        and reference_validation.get("pair64_pending") == 64
        and reference_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS",
        "current64 reference-cache validation failed",
    )
    reference_payload = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    references = {int(row): value for row, value in reference_payload.get("references", {}).items()}
    union = list(map(int, manifest["fresh_reference_union"]))
    require(
        reference_payload.get("status") == REFERENCE_READY
        and reference_payload.get("contract_population_complete") is False
        and reference_payload.get("current64_query_count") == 64
        and reference_payload.get("pair64_pending") == 64
        and list(map(int, reference_payload.get("reference_union", []))) == union
        and sorted(references) == union
        and len(references) == 2805,
        "current64 reference-cache payload interface failed",
    )
    old_map_validation = read_json(OLD_MAP_VALIDATION)
    require(
        sha256_file(MAP_REUSE_ADDENDUM) == EXPECTED_MAP_REUSE_ADDENDUM_SHA256
        and sha256_file(OLD_MAP_VALIDATION) == EXPECTED_OLD_MAP_VALIDATION_SHA256
        and sha256_file(OLD_MAP_PRODUCER) == EXPECTED_OLD_MAP_PRODUCER_SHA256
        and sha256_file(OLD_MAP_VALIDATOR) == EXPECTED_OLD_MAP_VALIDATOR_SHA256
        and sha256_file(GEOMETRY_SOURCE) == EXPECTED_GEOMETRY_SOURCE_SHA256
        and sha256_file(PROCESSOR_CONFIG) == EXPECTED_PROCESSOR_CONFIG_SHA256
        and
        old_map_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and old_map_validation.get("query_count") == 64
        and old_map_validation.get("target_role_read_count") == 0
        and old_map_validation.get("model_update_count") == 0
        and all(old_map_validation.get("checks", {}).values())
        and old_map_validation.get("logical_sha256") == logical_sha256(old_map_validation)
        and manifest.get("bindings", {}).get("old_map_validation_sha256") == sha256_file(OLD_MAP_VALIDATION),
        "old target-free map authority failed",
    )
    return manifest, reference_payload, references, old_map_validation


def projected_roster(records: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validation = read_json(ROSTER_VALIDATION)
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("target_label_read_count") == 0
        and validation.get("model_update_count") == 0
        and all(validation.get("checks", {}).values())
        and validation.get("logical_sha256") == logical_sha256(validation)
        and manifest.get("bindings", {}).get("current64_validation_sha256") == sha256_file(ROSTER_VALIDATION),
        "target-free current64 path resolver authority failed",
    )
    wanted = {record["query_id"] for record in records}
    projected: dict[str, dict[str, Any]] = {}
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(set(seals) == set(range(8)), "target-free roster shard seals absent")
    for shard in range(8):
        shard_root = ROSTER_ROOT / f"shard{shard:02d}"
        payload_path = shard_root / "payload.pt"
        require(
            seals[shard]["payload_sha256"] == sha256_file(payload_path)
            and seals[shard]["receipt_sha256"] == sha256_file(shard_root / "receipt.json")
            and seals[shard]["validation_sha256"] == sha256_file(shard_root / "validation.json"),
            f"target-free roster shard {shard} seal drift",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            if query_id not in wanted:
                continue
            require(query_id not in projected and record.get("target_or_label_read_count") == 0, "query resolver duplicate/non-target-free")
            projected[query_id] = {
                "role": str(record["role"]),
                "execution_ordinal": int(record["execution_ordinal"]),
                "heldout_fold": int(record["heldout_fold"]),
                "query_source_path": str(record["query_source_path"]),
                "query_source_sha256": str(record["query_source_sha256"]),
                "query_grid_shape": tuple(map(int, record["query_grid_shape"])),
                "source_exif_orientation": int(record["source_exif_orientation"]),
                "decode_frame": str(record["decode_frame"]),
                "roster_source_payload_sha256": str(seals[shard]["payload_sha256"]),
            }
    require(set(projected) == wanted, "query path resolver did not cover shard")
    return projected


def oof_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    by_shard: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_shard.setdefault(int(record["oof_source_shard"]), []).append(record)
    for shard, expected in by_shard.items():
        path = OOF_ROOT / f"shard{shard:02d}/payload.pt"
        require(all(item["oof_source_payload_sha256"] == sha256_file(path) for item in expected), "OOF source payload hash drift")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        candidates = {str(item["query_id"]): item for item in payload.get("records", [])}
        for source in expected:
            query_id = source["query_id"]
            value = candidates.get(query_id)
            require(value is not None and query_id not in output, f"fresh OOF query missing/duplicate: {query_id}")
            axis = list(map(int, value["natural_c128_representative_physical_rows"].tolist()))
            require(
                int(value["query_ordinal"]) == int(source["oof_query_ordinal"])
                and int(value["heldout_fold"]) == int(source["heldout_fold"])
                and tuple(value["grid_shape"]) == tuple(source["query_grid_shape"])
                and value["oof_checkpoint_sha256"] == source["oof_checkpoint_sha256"]
                and value["adapted_image_tokens_sha256"] == source["adapted_image_tokens_sha256"]
                and n2_tensor_sha256(value["adapted_image_tokens"]) == source["adapted_image_tokens_sha256"]
                and value["physical_row_scores_sha256"] == source["physical_row_scores_sha256"]
                and n2_tensor_sha256(value["physical_row_scores"]) == source["physical_row_scores_sha256"]
                and axis == source["fresh_c128_physical_rows"]
                and canonical_sha256(axis) == source["fresh_c128_physical_rows_sha256"],
                f"fresh OOF record binding failed: {query_id}",
            )
            output[query_id] = value
    return output


def old_maps(records: list[dict[str, Any]], reuse_pairs: Mapping[tuple[str, int], Mapping[str, Any]], validation: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    wanted_queries = {record["query_id"] for record in records}
    seals = {int(item["shard"]): item for item in validation["shards"]}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for shard in sorted({int(item["old_map_source_shard"]) for item in reuse_pairs.values() if item["query_id"] in wanted_queries}):
        path = OLD_MAP_ROOT / f"shard{shard:02d}/payload.pt"
        require(seals[shard]["payload_sha256"] == sha256_file(path), f"old map shard {shard} seal drift")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        payload_bindings = payload.get("bindings", {})
        require(
            isinstance(payload_bindings, Mapping)
            and isinstance(payload_bindings.get("source_payload_sha256"), str),
            f"old map shard {shard} roster-payload binding absent",
        )
        for record in payload.get("records", []):
            query_id = str(record["query_id"])
            if query_id not in wanted_queries:
                continue
            for candidate in record["candidates"]:
                key = (query_id, int(candidate["physical_row"]))
                if key not in reuse_pairs:
                    continue
                authority = reuse_pairs[key]
                require(
                    int(authority["old_map_source_shard"]) == shard
                    and authority["old_map_source_payload_sha256"] == seals[shard]["payload_sha256"]
                    and authority["query_map_sha256"] == candidate["query_map_sha256"]
                    and authority["reference_map_sha256"] == candidate["reference_map_sha256"]
                    and authority["reference_tokens_sha256"] == candidate["reference_tokens_sha256"],
                    f"old map pair binding failed: {key}",
                )
                require(
                    map_sha256(torch.as_tensor(candidate["query_map"]))
                    == candidate["query_map_sha256"]
                    == authority["query_map_sha256"]
                    and map_sha256(torch.as_tensor(candidate["reference_map"]))
                    == candidate["reference_map_sha256"]
                    == authority["reference_map_sha256"],
                    f"old map tensor bytes differ from manifest authority: {key}",
                )
                stored = dict(candidate)
                stored["_old_map_source_payload_sha256"] = seals[shard]["payload_sha256"]
                stored["_old_map_bound_roster_payload_sha256"] = payload_bindings["source_payload_sha256"]
                output[key] = stored
    expected = {key for key in reuse_pairs if key[0] in wanted_queries}
    require(set(output) == expected, "old map bank did not exactly cover shard reuse pairs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    require(args.shard in range(SHARD_COUNT), "shard must be in 0..7")
    out_dir = OUT_ROOT / f"shard{args.shard:02d}"
    require(not out_dir.exists(), f"immutable current64 prejoin shard exists: {out_dir}")

    manifest, reference_payload, references, old_validation = validated_inputs()
    all_records = sorted(manifest["records"], key=lambda item: int(item["execution_ordinal"]))
    require(len(all_records) == 64, "source manifest record population drift")
    records = [dict(item) for item in all_records[args.shard * 8 : (args.shard + 1) * 8]]
    require(len(records) == QUERY_COUNT_PER_SHARD, "current64 shard is not eight queries")
    query_ids = {record["query_id"] for record in records}
    reuse_pairs = {(item["query_id"], int(item["physical_row"])): item for item in manifest["old_map_reuse_pairs"]}
    missing_pairs = {(item["query_id"], int(item["physical_row"])): item for item in manifest["missing_map_pairs"]}
    require(len(reuse_pairs) == 8168 and len(missing_pairs) == 24 and not (set(reuse_pairs) & set(missing_pairs)), "reuse/missing map partition drift")
    roster = projected_roster(records, manifest)
    fresh = oof_records(records)
    reused = old_maps(records, reuse_pairs, old_validation)
    shard_missing = {key for key in missing_pairs if key[0] in query_ids}
    require(torch.cuda.is_available() or not shard_missing, "missing RoMa pairs require CUDA")
    require(sha256_file(ROMA_WEIGHTS) == EXPECTED_ROMA_SHA256, "RoMa checkpoint drift")
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    roma = RoMaV2() if shard_missing else None
    processor_hash = sha256_file(PROCESSOR_CONFIG)

    output_records: list[dict[str, Any]] = []
    roma_forward_count = 0
    for source in records:
        query_id = str(source["query_id"])
        path_info = roster[query_id]
        oof = fresh[query_id]
        require(
            source["role"] == path_info["role"]
            and int(source["execution_ordinal"]) == path_info["execution_ordinal"]
            and int(source["heldout_fold"]) == path_info["heldout_fold"]
            and source["query_source_sha256"] == path_info["query_source_sha256"]
            and tuple(source["query_grid_shape"]) == path_info["query_grid_shape"],
            f"source/path projection disagreement: {query_id}",
        )
        query_path = Path(path_info["query_source_path"])
        require(sha256_file(query_path) == path_info["query_source_sha256"], f"live query bytes drift: {query_id}")
        query_geometry = geometry(
            query_path,
            path_info["query_source_sha256"],
            f"matched-three-arm-query:{source['execution_ordinal']}",
            path_info["query_grid_shape"],
            path_info["decode_frame"],
        )
        query_geometry_saved = geometry_payload(query_geometry, path_info["decode_frame"])
        query_tokens = torch.as_tensor(oof["adapted_image_tokens"]).detach().cpu().contiguous()
        axis = list(map(int, source["fresh_c128_physical_rows"]))
        base_scores = [float(oof["physical_row_scores"][row]) for row in axis]
        winner = max(range(128), key=lambda position: (base_scores[position], -axis[position]))
        require(winner == 0, f"fresh C128 is not in deployed ranking order: {query_id}")

        candidates: list[dict[str, Any]] = []
        evidence_by_arm: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ARMS}
        for position, physical_row in enumerate(axis):
            reference = references[physical_row]
            require(
                set(reference) == {"physical_row", "source_path", "source_image_sha256", "grid_shape", "tokens", "tokens_sha256", "source_kind", "source_logical_sha256"}
                and int(reference["physical_row"]) == physical_row,
                f"reference interface drift: row {physical_row}",
            )
            reference_path = Path(reference["source_path"])
            require(sha256_file(reference_path) == reference["source_image_sha256"], f"live reference bytes drift: {physical_row}")
            reference_tokens = torch.as_tensor(reference["tokens"]).detach().cpu().contiguous()
            require(
                reference_tokens.dtype == torch.float16
                and reference_tokens.shape == (math.prod(tuple(reference["grid_shape"])), 128)
                and tensor_sha256(reference_tokens) == reference["tokens_sha256"],
                f"reference token replay failed: {physical_row}",
            )
            reference_geometry = geometry(
                reference_path,
                reference["source_image_sha256"],
                f"gallery-row:{physical_row}",
                tuple(map(int, reference["grid_shape"])),
                DECODED_RAW_BEFORE_EXIF,
            )
            key = (query_id, physical_row)
            if key in reused:
                old = reused[key]
                query_map = torch.as_tensor(old["query_map"], dtype=torch.float64).detach().cpu().contiguous().clone()
                reference_map = torch.as_tensor(old["reference_map"], dtype=torch.float64).detach().cpu().contiguous().clone()
                map_source = "HISTORICAL_TARGET_FREE_ROMA_EXACT_REUSE"
                old_map_source_shard = int(reuse_pairs[key]["old_map_source_shard"])
                require(
                    map_sha256(query_map) == reuse_pairs[key]["query_map_sha256"]
                    and map_sha256(reference_map) == reuse_pairs[key]["reference_map_sha256"],
                    f"reused map bytes changed after load: {key}",
                )
                require(
                    old["reference_tokens_sha256"] == reference["tokens_sha256"]
                    and old["reference_source_image_sha256"] == reference["source_image_sha256"]
                    and old["reference_source_logical_sha256"] == reference["source_logical_sha256"],
                    f"reused map/reference provenance drift: {key}",
                )
                require(
                    reuse_pairs[key]["query_source_sha256"] == path_info["query_source_sha256"]
                    and reuse_pairs[key]["old_map_source_payload_sha256"]
                    == old["_old_map_source_payload_sha256"]
                    and reuse_pairs[key]["old_map_bound_roster_payload_sha256"]
                    == old["_old_map_bound_roster_payload_sha256"]
                    == path_info["roster_source_payload_sha256"]
                    and reuse_pairs[key]["reference_source_image_sha256"]
                    == reference["source_image_sha256"]
                    and reuse_pairs[key]["reference_source_logical_sha256"]
                    == reference["source_logical_sha256"],
                    f"source-manifest exact cached-pair provenance drift: {key}",
                )
            else:
                require(key in shard_missing and roma is not None, f"pair absent from frozen reuse/missing partition: {key}")
                prediction = roma.match(oriented(query_path), oriented(reference_path))
                query_map = cell_means(prediction["overlap_AB"][0, ..., 0].detach().cpu(), query_geometry)
                reference_map = cell_means(prediction["overlap_BA"][0, ..., 0].detach().cpu(), reference_geometry)
                map_source = "FRESH_FROZEN_ROMA_COMPLETION"
                old_map_source_shard = None
                roma_forward_count += 1
            require(
                query_map.shape == (query_tokens.shape[0],)
                and reference_map.shape == (reference_tokens.shape[0],)
                and bool(torch.isfinite(query_map).all() and torch.isfinite(reference_map).all())
                and float(query_map.min()) >= 0.0
                and float(query_map.max()) <= 1.0
                and float(reference_map.min()) >= 0.0
                and float(reference_map.max()) <= 1.0,
                f"map shape/range failed: {key}",
            )
            evidence = arm_evidence(query_tokens, reference_tokens, query_map, reference_map)
            for arm in ARMS:
                evidence_by_arm[arm][position] = evidence[arm]
            candidates.append(
                {
                    "candidate_position": position,
                    "physical_row": physical_row,
                    "reference_tokens_sha256": str(reference["tokens_sha256"]),
                    "reference_source_image_sha256": str(reference["source_image_sha256"]),
                    "reference_source_logical_sha256": str(reference["source_logical_sha256"]),
                    "map_source": map_source,
                    "old_map_source_shard": old_map_source_shard,
                    "query_map": query_map,
                    "reference_map": reference_map,
                    "query_map_sha256": map_sha256(query_map),
                    "reference_map_sha256": map_sha256(reference_map),
                    "reference_geometry": geometry_payload(reference_geometry, DECODED_RAW_BEFORE_EXIF),
                    "evidence": evidence,
                }
            )

        challengers = [position for position in range(128) if position != winner]
        cbind_sources = [(position + 64) % 128 for position in range(128)]
        real_native: dict[str, torch.Tensor] = {}
        real_common: dict[str, torch.Tensor] = {}
        cbind_native: dict[str, torch.Tensor] = {}
        cbind_common: dict[str, torch.Tensor] = {}
        for arm in ARMS:
            native = torch.stack([candidate_feature(base_scores, evidence_by_arm[arm], challenger, winner) for challenger in challengers])
            controlled = {destination: evidence_by_arm[arm][source_position] for destination, source_position in enumerate(cbind_sources)}
            cbind = torch.stack([candidate_feature(base_scores, controlled, challenger, winner) for challenger in challengers])
            real_native[arm] = native
            real_common[arm] = native[:, :2].contiguous()
            cbind_native[arm] = cbind
            cbind_common[arm] = cbind[:, :2].contiguous()
        feature_sha = {
            "real_native": {arm: tensor_sha256(real_native[arm]) for arm in ARMS},
            "real_common": {arm: tensor_sha256(real_common[arm]) for arm in ARMS},
            "cbind_native": {arm: tensor_sha256(cbind_native[arm]) for arm in ARMS},
            "cbind_common": {arm: tensor_sha256(cbind_common[arm]) for arm in ARMS},
        }
        map_reuse_count = sum(
            candidate["map_source"] == "HISTORICAL_TARGET_FREE_ROMA_EXACT_REUSE"
            for candidate in candidates
        )
        map_completion_count = sum(
            candidate["map_source"] == "FRESH_FROZEN_ROMA_COMPLETION"
            for candidate in candidates
        )
        require(
            map_reuse_count == int(source["old_map_reuse_count"])
            and map_completion_count == int(source["missing_map_pair_count"])
            and map_reuse_count + map_completion_count == 128,
            f"per-query source map partition drift: {query_id}",
        )
        output_records.append(
            {
                "role_membership": str(source["role"]),
                "execution_ordinal": int(source["execution_ordinal"]),
                "query_id": query_id,
                "heldout_fold": int(source["heldout_fold"]),
                "oof_query_ordinal": int(source["oof_query_ordinal"]),
                "oof_source_shard": int(source["oof_source_shard"]),
                "oof_checkpoint_sha256": str(source["oof_checkpoint_sha256"]),
                "adapted_image_tokens_sha256": str(source["adapted_image_tokens_sha256"]),
                "physical_row_scores_sha256": str(source["physical_row_scores_sha256"]),
                "candidate_physical_rows": axis,
                "candidate_axis_sha256": canonical_sha256(axis),
                "base_scores": torch.tensor(base_scores, dtype=torch.float64),
                "base_winner_position": winner,
                "challenger_positions": challengers,
                "cbind_source_positions": cbind_sources,
                "query_geometry": query_geometry_saved,
                "candidates": candidates,
                "real_common_features": real_common,
                "real_native_features": real_native,
                "cbind_common_features": cbind_common,
                "cbind_native_features": cbind_native,
                "feature_sha256": feature_sha,
                "map_reuse_count": map_reuse_count,
                "map_completion_count": map_completion_count,
                "target_identity_read_count": 0,
                "query_supergroup_read_count": 0,
                "action_read_count": 0,
                "target_insertion_count": 0,
                "model_update_count": 0,
            }
        )
        print(json.dumps({"event": "fresh_d1_current64_query_ready", "shard": args.shard, "query_id": query_id, "roma_completions": output_records[-1]["map_completion_count"]}, sort_keys=True), flush=True)

    payload: dict[str, Any] = {
        "version": VERSION,
        "status": READY,
        "claim_level": "TARGET_FREE_CURRENT64_MATCHED_A_B_C_STAGING_ONLY_NO_PAIR64_OR_SCIENTIFIC_CLAIM",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "current64_query_count": len(output_records),
        "pair64_pending_query_count": 64,
        "contract_population_complete": False,
        "arms": list(ARMS),
        "map_replay_atol": MAP_REPLAY_ATOL,
        "records": output_records,
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "exact_map_cache_reuse_addendum_sha256": sha256_file(MAP_REUSE_ADDENDUM),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "source_validation_sha256": sha256_file(SOURCE_VALIDATION),
            "reference_payload_sha256": sha256_file(REFERENCE_PAYLOAD),
            "reference_result_sha256": sha256_file(REFERENCE_RESULT),
            "reference_validation_sha256": sha256_file(REFERENCE_VALIDATION),
            "old_map_validation_sha256": sha256_file(OLD_MAP_VALIDATION),
            "roster_validation_sha256": sha256_file(ROSTER_VALIDATION),
            "roma_checkpoint_sha256": sha256_file(ROMA_WEIGHTS),
            "processor_config_sha256": processor_hash,
            "producer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "access": {
            "source_manifest_read_count": 1,
            "target_free_roster_artifact_read_count": 1,
            "target_free_old_map_artifact_read_count": 1,
            "target_identity_read_count": 0,
            "query_supergroup_read_count": 0,
            "retrieval_outcome_read_count": 0,
            "action_read_count": 0,
            "target_insertion_count": 0,
            "roma_model_forward_count": roma_forward_count,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".current64-prejoin-{args.shard:02d}-", dir=OUT_ROOT))
    try:
        staged_payload = staging / "payload.pt"
        staged_receipt = staging / "receipt.json"
        torch.save(payload, staged_payload)
        receipt: dict[str, Any] = {
            "version": VERSION,
            "status": READY,
            "shard": args.shard,
            "query_count": len(output_records),
            "candidate_count": sum(len(record["candidates"]) for record in output_records),
            "map_reuse_count": sum(record["map_reuse_count"] for record in output_records),
            "map_completion_count": sum(record["map_completion_count"] for record in output_records),
            "map_replay_atol": MAP_REPLAY_ATOL,
            "pair64_pending_query_count": 64,
            "contract_population_complete": False,
            "payload_sha256": sha256_file(staged_payload),
            "access": payload["access"],
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION",
            "logical_sha256": "",
        }
        receipt["logical_sha256"] = logical_sha256(receipt)
        staged_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.rename(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": READY, "shard": args.shard, "queries": len(output_records), "map_completions": roma_forward_count}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
