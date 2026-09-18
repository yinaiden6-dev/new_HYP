#!/usr/bin/env python3
"""Independent validator for one fresh-D1 current64 A/B/C shard.

The producer is never imported.  Scores, six-feature vectors, shift-64
C_BIND, geometry, and reused/completed RoMa maps are reconstructed through an
independent implementation.  This validates current64 staging only; Pair64
remains pending and no training is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"

VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_prejoin_shard_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_READY"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_VALIDATED"
SHARD_COUNT = 8
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
EXPECTED_ROMA_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
EXPECTED_PROCESSOR_CONFIG_SHA256 = "1a427e12a15406a5c4701273c91a0b319528de5c78523ae5ba5329d86e5d5557"
EXPECTED_MAP_REUSE_ADDENDUM_SHA256 = "4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6"
EXPECTED_OLD_MAP_VALIDATION_SHA256 = "60bd07480bf4963503993ecb926b696e3fc568c588189c557c0bc9975f32c66f"
EXPECTED_OLD_MAP_PRODUCER_SHA256 = "ca10470f271da2d6fef5cd59d11ad930c5d9045774774d2caac7df8d1b339383"
EXPECTED_OLD_MAP_VALIDATOR_SHA256 = "bd4cda9d350796cb6a7a7f2ba3c26047254a9566ea481e20338ad49ea0e9008d"
EXPECTED_GEOMETRY_SOURCE_SHA256 = "c732dc933c502f17fd5cb8cc80ae45d116716fe801cc8f3034a1e6a49cef6508"
MAP_REPLAY_ATOL = 1e-6


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"input absent/non-regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
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
    return hashlib.sha256(torch.as_tensor(value).detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def independent_oriented(path: Path) -> Image.Image:
    with Image.open(path) as source:
        try:
            orientation = int(source.getexif().get(274, 1))
        except (TypeError, ValueError):
            orientation = 1
        image = source.copy()
    operation = {
        2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if operation is not None:
        image = image.transpose(operation)
    return image.convert("RGB")


def independent_geometry(path: Path, image_hash: str, key: str, grid: tuple[int, int], frame: str):
    with Image.open(path) as image:
        raw_hw = (int(image.height), int(image.width))
        orientation = int(image.getexif().get(274, 1))
    return build_colnomic_canonical_geometry_v2(
        source_image_sha256=image_hash,
        source_key=key,
        processor_config_sha256=sha256_file(PROCESSOR_CONFIG),
        raw_size_hw=raw_hw,
        exif_orientation=orientation,
        merged_grid_shape=grid,
        processor_input_frame=frame,
    )


def geometry_dict(value: Any, frame: str) -> dict[str, Any]:
    affine = torch.as_tensor(value.raw_to_oriented_affine).detach().cpu().contiguous()
    valid = torch.as_tensor(value.valid_patch_mask).detach().cpu().contiguous()
    boxes = torch.as_tensor(value.cell_boxes_xyxy).detach().cpu().contiguous()
    return {
        "source_image_sha256": str(value.source_image_sha256), "source_key": str(value.source_key),
        "processor_config_sha256": str(value.processor_config_sha256), "raw_size_hw": tuple(map(int, value.raw_size_hw)),
        "oriented_size_hw": tuple(map(int, value.oriented_size_hw)), "exif_orientation": int(value.exif_orientation),
        "raw_to_oriented_affine": affine, "grid_shape": tuple(map(int, value.grid_shape)),
        "valid_patch_mask": valid, "cell_boxes_xyxy": boxes, "coordinate_frame": str(value.coordinate_frame),
        "processor_input_frame": frame, "raw_to_oriented_affine_sha256": tensor_sha256(affine),
        "valid_patch_mask_sha256": tensor_sha256(valid), "cell_boxes_xyxy_sha256": tensor_sha256(boxes),
    }


def equal_nested(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal_nested(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def independent_cell_means(overlap: torch.Tensor, geometry_value: Any) -> torch.Tensor:
    source = torch.as_tensor(overlap, dtype=torch.float64)
    height, width = source.shape
    cells = []
    for box, valid in zip(geometry_value.cell_boxes_xyxy, geometry_value.valid_patch_mask, strict=True):
        if not bool(valid):
            cells.append(source.new_zeros(())); continue
        x0, y0, x1, y1 = map(float, box)
        left = max(0, min(width - 1, math.floor(x0 * width)))
        top = max(0, min(height - 1, math.floor(y0 * height)))
        right = max(left + 1, min(width, math.ceil(x1 * width)))
        bottom = max(top + 1, min(height, math.ceil(y1 * height)))
        cells.append(source[top:bottom, left:right].mean())
    return torch.stack(cells).contiguous()


def independent_score(query: torch.Tensor, reference: torch.Tensor, qweight: torch.Tensor, rweight: torch.Tensor) -> tuple[float, float]:
    q = F.normalize(query.to(torch.float64), dim=1)
    r = F.normalize(reference.to(torch.float64), dim=1)
    wq, wr = qweight.to(torch.float64), rweight.to(torch.float64)
    similarity = q @ r.T
    local = (similarity * wr[None]).max(dim=1).values
    mass = torch.sqrt(wq.mean() * wr.mean())
    value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
    require(bool(torch.isfinite(value) and torch.isfinite(mass)), "independent score nonfinite")
    return float(value), float(mass)


def independent_evidence(query: torch.Tensor, reference: torch.Tensor, qmap: torch.Tensor, rmap: torch.Tensor) -> dict[str, dict[str, float]]:
    query_unit = F.normalize(query.to(torch.float64), dim=1)
    reference_unit = F.normalize(reference.to(torch.float64), dim=1)
    similarity = query_unit @ reference_unit.T
    oq, or_ = torch.ones_like(qmap), torch.ones_like(rmap)
    hq, hr = max(1, qmap.numel() // 2), max(1, rmap.numel() // 2)

    def reduce(qweight: torch.Tensor, rweight: torch.Tensor) -> tuple[float, float]:
        wq, wr = qweight.to(torch.float64), rweight.to(torch.float64)
        local = (similarity * wr[None]).max(dim=1).values
        mass = torch.sqrt(wq.mean() * wr.mean())
        value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
        require(bool(torch.isfinite(value) and torch.isfinite(mass)), "independent arm evidence nonfinite")
        return float(value), float(mass)

    a, am = reduce(oq, or_)
    b, bm = reduce(qmap, or_)
    bq, _ = reduce(qmap.roll(hq), or_)
    c, cm = reduce(qmap, rmap)
    cq, _ = reduce(qmap.roll(hq), rmap)
    cr, _ = reduce(qmap, rmap.roll(hr))
    return {
        "A_ALL": {"real_score": a, "visibility_mass": am, "query_control_score": a, "reference_control_score": a},
        "B_QUERY": {"real_score": b, "visibility_mass": bm, "query_control_score": bq, "reference_control_score": b},
        "C_PAIRED": {"real_score": c, "visibility_mass": cm, "query_control_score": cq, "reference_control_score": cr},
    }


def sym(a: float, b: float) -> float:
    return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)


def independent_feature(base: list[float], evidence: Mapping[int, Mapping[str, float]], challenger: int, winner: int) -> torch.Tensor:
    mean = sum(base) / len(base)
    std = (sum((value - mean) ** 2 for value in base) / len(base)) ** 0.5
    c, w = evidence[challenger], evidence[winner]
    cn = float(c["real_score"]) / max(float(c["visibility_mass"]), 1e-12)
    wn = float(w["real_score"]) / max(float(w["visibility_mass"]), 1e-12)
    cq, wq = float(c["real_score"]) - float(c["query_control_score"]), float(w["real_score"]) - float(w["query_control_score"])
    cr, wr = float(c["real_score"]) - float(c["reference_control_score"]), float(w["real_score"]) - float(w["reference_control_score"])
    return torch.tensor([(base[challenger]-base[winner])/max(std,1e-12), sym(c["real_score"],w["real_score"]), sym(c["visibility_mass"],w["visibility_mass"]), sym(cn,wn), sym(cq,wq), sym(cr,wr)], dtype=torch.float64)


def load_roster() -> dict[str, dict[str, Any]]:
    validation = read_json(ROSTER_VALIDATION)
    require(validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED" and all(validation.get("checks", {}).values()) and validation.get("target_label_read_count") == 0, "independent roster validation failed")
    output = {}
    seals = {int(item["shard"]): item for item in validation.get("shards", [])}
    require(set(seals) == set(range(8)), "independent roster seals absent")
    for shard in range(8):
        shard_root = ROSTER_ROOT/f"shard{shard:02d}"
        payload_path = shard_root/"payload.pt"
        require(
            seals[shard]["payload_sha256"] == sha256_file(payload_path)
            and seals[shard]["receipt_sha256"] == sha256_file(shard_root/"receipt.json")
            and seals[shard]["validation_sha256"] == sha256_file(shard_root/"validation.json"),
            f"independent roster shard {shard} seal failed",
        )
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for row in payload["records"]:
            require(row.get("target_or_label_read_count") == 0 and row["query_id"] not in output, "independent roster record failed")
            output[str(row["query_id"])] = row
            output[str(row["query_id"])]["_roster_source_payload_sha256"] = seals[shard]["payload_sha256"]
    require(len(output) == 64, "independent roster not 64")
    return output


def load_old_maps(wanted: set[str], validation: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    seals = {int(item["shard"]): item for item in validation["shards"]}
    output = {}
    for shard in range(8):
        path = OLD_MAP_ROOT/f"shard{shard:02d}/payload.pt"
        require(seals[shard]["payload_sha256"] == sha256_file(path), "independent old map seal failed")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        payload_bindings = payload.get("bindings", {})
        require(
            isinstance(payload_bindings, Mapping)
            and isinstance(payload_bindings.get("source_payload_sha256"), str),
            f"independent old map shard {shard} roster binding absent",
        )
        for record in payload["records"]:
            if record["query_id"] not in wanted: continue
            for candidate in record["candidates"]:
                stored = dict(candidate)
                stored["_old_map_source_payload_sha256"] = seals[shard]["payload_sha256"]
                stored["_old_map_bound_roster_payload_sha256"] = payload_bindings["source_payload_sha256"]
                output[(str(record["query_id"]), int(candidate["physical_row"]))] = stored
    return output


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"immutable validation exists: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--shard", type=int, required=True); args = parser.parse_args()
    require(args.shard in range(SHARD_COUNT), "shard must be 0..7")
    shard_dir = OUT_ROOT/f"shard{args.shard:02d}"
    payload_path, receipt_path, out = shard_dir/"payload.pt", shard_dir/"receipt.json", shard_dir/"validation.json"
    require(not out.exists(), f"immutable validation exists: {out}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = read_json(receipt_path)
    manifest, source_validation = read_json(SOURCE_MANIFEST), read_json(SOURCE_VALIDATION)
    ref_result, ref_validation = read_json(REFERENCE_RESULT), read_json(REFERENCE_VALIDATION)
    old_validation = read_json(OLD_MAP_VALIDATION)
    require(
        manifest.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and manifest.get("contract_population_complete") is False
        and manifest.get("population", {}).get("pair64_pending_count") == 64
        and manifest.get("bindings", {}).get("rematerialization_contract_sha256")
        == sha256_file(CONTRACT)
        and manifest.get("bindings", {}).get("exact_map_cache_reuse_addendum_sha256")
        == sha256_file(MAP_REUSE_ADDENDUM)
        and manifest.get("logical_sha256") == logical_sha256(manifest),
        "independent source manifest failed",
    )
    require(
        source_validation.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and source_validation.get("manifest_sha256") == sha256_file(SOURCE_MANIFEST)
        and source_validation.get("manifest_logical_sha256") == manifest["logical_sha256"]
        and source_validation.get("contract_population_complete") is False
        and source_validation.get("pair64_pending_count") == 64
        and source_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_COMPLETION"
        and source_validation.get("logical_sha256") == logical_sha256(source_validation)
        and all(source_validation.get("checks", {}).values()),
        "independent source validation failed",
    )
    require(
        ref_result.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_READY"
        and ref_result.get("contract_population_complete") is False
        and ref_result.get("current64_query_count") == 64
        and ref_result.get("pair64_pending") == 64
        and ref_result.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and ref_result.get("next_authorized_stage") == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS"
        and ref_result.get("logical_sha256") == logical_sha256(ref_result),
        "independent ref result failed",
    )
    require(
        ref_validation.get("status") == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_REFERENCE_CACHE_VALIDATED"
        and ref_validation.get("contract_population_complete") is False
        and ref_validation.get("current64_query_count") == 64
        and ref_validation.get("pair64_pending") == 64
        and ref_validation.get("payload_sha256") == sha256_file(REFERENCE_PAYLOAD)
        and ref_validation.get("producer_result_sha256") == sha256_file(REFERENCE_RESULT)
        and ref_validation.get("next_authorized_stage") == "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARDS"
        and ref_validation.get("logical_sha256") == logical_sha256(ref_validation)
        and all(ref_validation.get("checks", {}).values()),
        "independent ref validation failed",
    )
    references_payload = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    references = {int(key): value for key,value in references_payload["references"].items()}
    require(list(map(int,references_payload["reference_union"])) == list(map(int,manifest["fresh_reference_union"])) and len(references)==2805, "independent reference union failed")
    require(
        sha256_file(MAP_REUSE_ADDENDUM) == EXPECTED_MAP_REUSE_ADDENDUM_SHA256
        and sha256_file(OLD_MAP_VALIDATION) == EXPECTED_OLD_MAP_VALIDATION_SHA256
        and sha256_file(OLD_MAP_PRODUCER) == EXPECTED_OLD_MAP_PRODUCER_SHA256
        and sha256_file(OLD_MAP_VALIDATOR) == EXPECTED_OLD_MAP_VALIDATOR_SHA256
        and sha256_file(GEOMETRY_SOURCE) == EXPECTED_GEOMETRY_SOURCE_SHA256
        and sha256_file(PROCESSOR_CONFIG) == EXPECTED_PROCESSOR_CONFIG_SHA256
        and old_validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and all(old_validation.get("checks",{}).values()),
        "independent old map validation failed",
    )
    roster = load_roster()
    all_source = sorted(manifest["records"], key=lambda x:int(x["execution_ordinal"]))
    source_records = all_source[args.shard*8:(args.shard+1)*8]
    wanted = {str(row["query_id"]) for row in source_records}
    old_maps = load_old_maps(wanted, old_validation)
    reuse = {(x["query_id"],int(x["physical_row"])):x for x in manifest["old_map_reuse_pairs"]}
    missing = {(x["query_id"],int(x["physical_row"])):x for x in manifest["missing_map_pairs"]}
    shard_missing = {key for key in missing if key[0] in wanted}
    require(torch.cuda.is_available() or not shard_missing, "independent missing-map replay requires CUDA")
    require(sha256_file(ROMA_WEIGHTS)==EXPECTED_ROMA_SHA256, "independent RoMa checkpoint failed")
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    roma = RoMaV2() if shard_missing else None

    source_by_query = {str(x["query_id"]):x for x in source_records}
    oof_by_query = {}
    for source_shard in sorted({int(x["oof_source_shard"]) for x in source_records}):
        oof_path=OOF_ROOT/f"shard{source_shard:02d}/payload.pt"
        require(
            all(
                item["oof_source_payload_sha256"] == sha256_file(oof_path)
                for item in source_records
                if int(item["oof_source_shard"]) == source_shard
            ),
            f"independent OOF shard {source_shard} source hash failed",
        )
        oof_payload=torch.load(oof_path,map_location="cpu",weights_only=False,mmap=True)
        for row in oof_payload["records"]:
            if row["query_id"] in wanted:oof_by_query[str(row["query_id"])]=row
    records = payload.get("records",[])
    require(len(records)==8 and {r["query_id"] for r in records}==wanted, "output shard query population failed")
    score_error=feature_error=reorder_error=cbind_base_gap_error=0.0
    missing_map_error=0.0
    reused_maps_exact=True
    missing_replays=0
    cbind_complete_bundle_shift = True
    record_checks=[]
    for record in records:
        query_id=str(record["query_id"]); source=source_by_query[query_id]; roster_row=roster[query_id]; oof=oof_by_query[query_id]
        query_path=Path(roster_row["query_source_path"]); require(sha256_file(query_path)==source["query_source_sha256"], "query bytes failed")
        qgeometry=independent_geometry(query_path,source["query_source_sha256"],f"matched-three-arm-query:{source['execution_ordinal']}",tuple(source["query_grid_shape"]),str(roster_row["decode_frame"]))
        require(equal_nested(record["query_geometry"],geometry_dict(qgeometry,str(roster_row["decode_frame"]))), "query geometry differs")
        qtokens=torch.as_tensor(oof["adapted_image_tokens"])
        axis=list(map(int,source["fresh_c128_physical_rows"]))
        require(
            int(oof["query_ordinal"]) == int(source["oof_query_ordinal"])
            and int(oof["heldout_fold"]) == int(source["heldout_fold"])
            and tuple(oof["grid_shape"]) == tuple(source["query_grid_shape"])
            and oof["oof_checkpoint_sha256"] == source["oof_checkpoint_sha256"]
            and n2_tensor_sha256(qtokens) == source["adapted_image_tokens_sha256"]
            and oof["adapted_image_tokens_sha256"] == source["adapted_image_tokens_sha256"]
            and n2_tensor_sha256(oof["physical_row_scores"]) == source["physical_row_scores_sha256"]
            and oof["physical_row_scores_sha256"] == source["physical_row_scores_sha256"]
            and list(map(int,oof["natural_c128_representative_physical_rows"].tolist())) == axis
            and canonical_sha256(axis) == source["fresh_c128_physical_rows_sha256"],
            "fresh OOF record replay failed",
        )
        base=[float(oof["physical_row_scores"][row]) for row in axis]
        winner=max(range(128),key=lambda p:(base[p],-axis[p])); challengers=[p for p in range(128) if p!=winner]; require(winner==0,"fresh ranking winner failed")
        evidence_by_arm={arm:{} for arm in ARMS}; candidate_checks=[]
        candidate_by_position={int(c["candidate_position"]):c for c in record["candidates"]}
        for position,row in enumerate(axis):
            candidate=candidate_by_position[position]; reference=references[row]; rtokens=torch.as_tensor(reference["tokens"]); rpath=Path(reference["source_path"])
            require(sha256_file(rpath)==reference["source_image_sha256"] and tensor_sha256(rtokens)==reference["tokens_sha256"],"reference replay failed")
            rgeometry=independent_geometry(rpath,reference["source_image_sha256"],f"gallery-row:{row}",tuple(reference["grid_shape"]),DECODED_RAW_BEFORE_EXIF)
            require(equal_nested(candidate["reference_geometry"],geometry_dict(rgeometry,DECODED_RAW_BEFORE_EXIF)),"reference geometry differs")
            qmap=torch.as_tensor(candidate["query_map"]); rmap=torch.as_tensor(candidate["reference_map"]); key=(query_id,row)
            if key in reuse:
                old=old_maps[key]; expected_q=torch.as_tensor(old["query_map"]);expected_r=torch.as_tensor(old["reference_map"])
                reused_maps_exact = reused_maps_exact and torch.equal(qmap, expected_q) and torch.equal(rmap, expected_r)
                source_ok=(
                    candidate["map_source"]=="HISTORICAL_TARGET_FREE_ROMA_EXACT_REUSE"
                    and candidate["old_map_source_shard"]==reuse[key]["old_map_source_shard"]
                    and map_sha256(expected_q)==old["query_map_sha256"]==reuse[key]["query_map_sha256"]
                    and map_sha256(expected_r)==old["reference_map_sha256"]==reuse[key]["reference_map_sha256"]
                    and old["reference_tokens_sha256"]==reuse[key]["reference_tokens_sha256"]==reference["tokens_sha256"]
                    and old["reference_source_image_sha256"]==reference["source_image_sha256"]
                    and old["reference_source_logical_sha256"]==reference["source_logical_sha256"]
                    and reuse[key]["query_source_sha256"]==source["query_source_sha256"]==roster_row["query_source_sha256"]
                    and reuse[key]["old_map_source_payload_sha256"]==old["_old_map_source_payload_sha256"]
                    and reuse[key]["old_map_bound_roster_payload_sha256"]==old["_old_map_bound_roster_payload_sha256"]==roster_row["_roster_source_payload_sha256"]
                    and reuse[key]["reference_source_image_sha256"]==reference["source_image_sha256"]
                    and reuse[key]["reference_source_logical_sha256"]==reference["source_logical_sha256"]
                )
            else:
                require(key in shard_missing and roma is not None,"independent pair partition failed")
                prediction=roma.match(independent_oriented(query_path),independent_oriented(rpath));expected_q=independent_cell_means(prediction["overlap_AB"][0,...,0].detach().cpu(),qgeometry);expected_r=independent_cell_means(prediction["overlap_BA"][0,...,0].detach().cpu(),rgeometry);missing_replays+=1
                missing_map_error=max(missing_map_error,float((qmap-expected_q).abs().max()),float((rmap-expected_r).abs().max()))
                source_ok=candidate["map_source"]=="FRESH_FROZEN_ROMA_COMPLETION" and candidate["old_map_source_shard"] is None
            expected_evidence=independent_evidence(qtokens,rtokens,qmap,rmap)
            for arm in ARMS:
                evidence_by_arm[arm][position]=expected_evidence[arm]
                for name in ("real_score","visibility_mass","query_control_score","reference_control_score"):
                    score_error=max(score_error,abs(float(candidate["evidence"][arm][name])-float(expected_evidence[arm][name])))
            candidate_checks.append(candidate["physical_row"]==row and candidate["reference_tokens_sha256"]==reference["tokens_sha256"] and candidate["query_map_sha256"]==map_sha256(qmap) and candidate["reference_map_sha256"]==map_sha256(rmap) and source_ok)
        cbind=[(p+64)%128 for p in range(128)]
        for arm in ARMS:
            real=torch.stack([independent_feature(base,evidence_by_arm[arm],c,winner) for c in challengers]);controlled={d:evidence_by_arm[arm][s] for d,s in enumerate(cbind)}; cb=torch.stack([independent_feature(base,controlled,c,winner) for c in challengers])
            cbind_complete_bundle_shift = cbind_complete_bundle_shift and all(
                controlled[destination] == evidence_by_arm[arm][source_position]
                and set(controlled[destination])
                == {"real_score", "visibility_mass", "query_control_score", "reference_control_score"}
                for destination, source_position in enumerate(cbind)
            )
            cbind_base_gap_error=max(cbind_base_gap_error,float((real[:,0]-cb[:,0]).abs().max()))
            feature_error=max(feature_error,float((real-record["real_native_features"][arm]).abs().max()),float((cb-record["cbind_native_features"][arm]).abs().max()),float((real[:,:2]-record["real_common_features"][arm]).abs().max()),float((cb[:,:2]-record["cbind_common_features"][arm]).abs().max()))
            require(record["feature_sha256"]["real_native"][arm]==tensor_sha256(real) and record["feature_sha256"]["cbind_native"][arm]==tensor_sha256(cb),"feature hash failed")
            permutation=list(reversed(range(128))); inverse={old:new for new,old in enumerate(permutation)}; pbase=[base[old] for old in permutation];pevidence={new:evidence_by_arm[arm][old] for new,old in enumerate(permutation)};pwin=inverse[winner]
            for old_challenger, original in zip(challengers,real,strict=True):
                replay=independent_feature(pbase,pevidence,inverse[old_challenger],pwin); reorder_error=max(reorder_error,float((original-replay).abs().max()))
        record_checks.append(record["role_membership"]==source["role"] and record["execution_ordinal"]==source["execution_ordinal"] and record["heldout_fold"]==source["heldout_fold"] and record["candidate_physical_rows"]==axis and torch.equal(record["base_scores"],torch.tensor(base,dtype=torch.float64)) and record["base_winner_position"]==winner and record["challenger_positions"]==challengers and record["cbind_source_positions"]==cbind and all(candidate_checks) and record["map_reuse_count"]==source["old_map_reuse_count"] and record["map_completion_count"]==source["missing_map_pair_count"] and record["map_reuse_count"]+record["map_completion_count"]==128 and record["target_identity_read_count"]==record["query_supergroup_read_count"]==record["action_read_count"]==record["target_insertion_count"]==record["model_update_count"]==0)

    expected_access={"source_manifest_read_count":1,"target_free_roster_artifact_read_count":1,"target_free_old_map_artifact_read_count":1,"target_identity_read_count":0,"query_supergroup_read_count":0,"retrieval_outcome_read_count":0,"action_read_count":0,"target_insertion_count":0,"roma_model_forward_count":len(shard_missing),"model_update_count":0,"external_read_count":0,"sealed_read_count":0}
    expected_bindings={"contract_sha256":sha256_file(CONTRACT),"exact_map_cache_reuse_addendum_sha256":sha256_file(MAP_REUSE_ADDENDUM),"source_manifest_sha256":sha256_file(SOURCE_MANIFEST),"source_validation_sha256":sha256_file(SOURCE_VALIDATION),"reference_payload_sha256":sha256_file(REFERENCE_PAYLOAD),"reference_result_sha256":sha256_file(REFERENCE_RESULT),"reference_validation_sha256":sha256_file(REFERENCE_VALIDATION),"old_map_validation_sha256":sha256_file(OLD_MAP_VALIDATION),"roster_validation_sha256":sha256_file(ROSTER_VALIDATION),"roma_checkpoint_sha256":sha256_file(ROMA_WEIGHTS),"processor_config_sha256":sha256_file(PROCESSOR_CONFIG),"producer_sha256":sha256_file(PRODUCER)}
    checks={"payload_scope":payload.get("version")==VERSION and payload.get("status")==READY and payload.get("shard")==args.shard and payload.get("shard_count")==8 and payload.get("current64_query_count")==8 and payload.get("pair64_pending_query_count")==64 and payload.get("contract_population_complete") is False and tuple(payload.get("arms",()))==ARMS and payload.get("map_replay_atol")==MAP_REPLAY_ATOL,"bindings":payload.get("bindings")==expected_bindings,"access":payload.get("access")==expected_access,"records_and_formula":all(record_checks) and score_error<=1e-12 and feature_error<=1e-12,"map_exact_reuse_or_independent_completion":reused_maps_exact and missing_map_error<=MAP_REPLAY_ATOL and missing_replays==len(shard_missing),"candidate_reorder_invariance":reorder_error<=1e-12,"cbind_base_gap_unchanged_and_complete_bundle_shift":cbind_base_gap_error==0.0 and cbind_complete_bundle_shift,"claim_boundary":payload.get("scientific_GO_or_NO_GO") is None and payload.get("ownership_GO_or_NO_GO") is None and payload.get("automatic_stage_advance") is False and payload.get("next_authorized_stage")=="N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION","receipt":receipt.get("status")==READY and receipt.get("payload_sha256")==sha256_file(payload_path) and receipt.get("query_count")==8 and receipt.get("candidate_count")==1024 and receipt.get("pair64_pending_query_count")==64 and receipt.get("contract_population_complete") is False and receipt.get("map_replay_atol")==MAP_REPLAY_ATOL and receipt.get("next_authorized_stage")=="N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_INDEPENDENT_VALIDATION" and receipt.get("logical_sha256")==logical_sha256(receipt)}
    passed=all(checks.values())
    value={"version":VERSION,"status":VALIDATED if passed else "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_SHARD_VALIDATION_ABORT","claim_level":"INDEPENDENT_TARGET_FREE_CURRENT64_MATCHED_A_B_C_STAGING_VALIDATION_ONLY","shard":args.shard,"checks":checks,"query_count":len(records),"candidate_count":sum(len(r["candidates"]) for r in records),"map_reuse_count":sum(r["map_reuse_count"] for r in records),"map_completion_count":sum(r["map_completion_count"] for r in records),"score_max_abs":score_error,"feature_max_abs":feature_error,"map_replay_atol":MAP_REPLAY_ATOL,"missing_map_replay_max_abs":missing_map_error,"candidate_reorder_max_abs":reorder_error,"cbind_base_gap_max_abs":cbind_base_gap_error,"payload_sha256":sha256_file(payload_path),"receipt_sha256":sha256_file(receipt_path),"producer_sha256":sha256_file(PRODUCER),"validator_sha256":sha256_file(Path(__file__).resolve()),"pair64_pending_query_count":64,"contract_population_complete":False,"scientific_GO_or_NO_GO":None,"ownership_GO_or_NO_GO":None,"automatic_stage_advance":False,"next_authorized_stage":"N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATION" if passed else None,"logical_sha256":""}
    value["logical_sha256"]=logical_sha256(value);atomic_json(out,value);print(json.dumps({"status":value["status"],"checks":checks},sort_keys=True));raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
