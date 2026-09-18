#!/usr/bin/env python3
"""Materialize one Pair64 training-only fresh-RoMa A/B/C feature shard."""

from __future__ import annotations

import argparse
import copy
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
    build_colnomic_canonical_geometry_v2,
)
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import (  # noqa: E402
    HybridSpatialReferenceResolver,
)
from romav2 import RoMaV2  # noqa: E402


ACTIVE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md"
SCOPE = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md"
LABEL = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md"
DISPOSITION = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_ACTIVE_CONTRACT_DISPOSITION_AND_EXECUTION_CLARIFICATION_V1_20260904.md"
P0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1"
P0 = P0_ROOT / "preseal.json"
P0_VALIDATION = P0_ROOT / "independent_validation.json"
P0_PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
P0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py"
J0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1"
J0 = J0_ROOT / "fixed_pairs.json"
J0_VALIDATION = J0_ROOT / "independent_validation.json"
J0_PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py"
J0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
PROCESSOR_CONFIG = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json"
RESOLVER_SOURCE = ROOT / "src/rc_aslo_xf/cw1_sr0_s8_feature_runtime_v1.py"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1"

VERSION = "routea_n2_fresh_d1_matched_three_arm_pair64_training_feature_shard_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_READY"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_INDEPENDENT_VALIDATION"
SHARD_COUNT = 8
QUERIES_PER_SHARD = 8
ENDPOINTS_PER_QUERY = 2
MAPS_PER_SHARD = 16
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
FEATURE_NAMES = (
    "fresh_d1_base_gap_over_c128_std",
    "symmetric_real_score",
    "symmetric_visibility_mass",
    "symmetric_score_over_mass",
    "symmetric_query_roll_response",
    "symmetric_reference_roll_response",
)
EXPECTED_HASHES = {
    ACTIVE: "bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade",
    SCOPE: "19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124",
    LABEL: "9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035",
    DISPOSITION: "502de768d26584613bb89f329e214e431d959e821fe6caa3ab5cd1cb5ad538b8",
    J0_PRODUCER: "5ef394b3cbc1688b8315b7ee7c2ad625a29f255b82d1018e48c4d4a3f5267fd4",
    J0_VALIDATOR: "a536c5fdcc37834dd78a6cc8b18a7f13d76c14091db625533059291e88ad8c16",
    P0_PRODUCER: "068c8297a0b773c4efefa093f2887393588213dce7ca347cb5d6517cc22e13b5",
    P0_VALIDATOR: "239f67385076b72b3f28c4223a5c4208f7ee64ba90c22008d618aaa627adf6db",
    ROMA_WEIGHTS: "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
}
J0_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY"
J0_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_VALIDATED"
P0_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY"
P0_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED"
J0_ALLOWED_KEYS = {
    "pair64_ordinal",
    "pair_cohort",
    "query_id",
    "canonical_heldout_fold",
    "canonical_track",
    "winner",
    "counterpart",
    "switch_label",
    "preseal_record_sha256",
    "preseal_natural_c128_sha256",
}
J0_PROTECTED_KEYS = {
    "target_identity",
    "target_supergroup",
    "target_physical_row_provenance",
    "n2_label_source_folds",
    "target_state",
    "target_naturally_in_fresh_c128",
    "target_candidate_position",
}


class M0Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M0Error(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"input absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


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
    return hashlib.sha256(
        torch.as_tensor(value).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def atomic_directory(target: Path, payload: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    require(not target.exists(), f"immutable shard exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        torch.save(payload, staging / "payload.pt")
        value = dict(receipt)
        value["payload_sha256"] = sha256_file(staging / "payload.pt")
        value["logical_sha256"] = logical_sha256(value)
        (staging / "receipt.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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


def geometry(path: Path, image_sha: str, key: str, grid: tuple[int, int], frame: str):
    with Image.open(path) as image:
        raw_hw = (int(image.height), int(image.width))
        orientation = int(image.getexif().get(274, 1))
    return build_colnomic_canonical_geometry_v2(
        source_image_sha256=image_sha,
        source_key=key,
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
    for box, valid in zip(
        patch_geometry.cell_boxes_xyxy,
        patch_geometry.valid_patch_mask,
        strict=True,
    ):
        if not bool(valid):
            output.append(values.new_zeros(()))
            continue
        x0, y0, x1, y1 = map(float, box)
        ix0 = max(0, min(width - 1, math.floor(x0 * width)))
        iy0 = max(0, min(height - 1, math.floor(y0 * height)))
        ix1 = max(ix0 + 1, min(width, math.ceil(x1 * width)))
        iy1 = max(iy0 + 1, min(height, math.ceil(y1 * height)))
        output.append(values[iy0:iy1, ix0:ix1].mean())
    return torch.stack(output).contiguous()


def reduce_score(
    similarity: torch.Tensor, query_weight: torch.Tensor, reference_weight: torch.Tensor
) -> tuple[float, float]:
    wq = query_weight.to(torch.float64)
    wr = reference_weight.to(torch.float64)
    require(
        wq.shape == (similarity.shape[0],)
        and wr.shape == (similarity.shape[1],)
        and bool(torch.isfinite(wq).all() and torch.isfinite(wr).all())
        and float(wq.min()) >= 0.0
        and float(wq.max()) <= 1.0
        and float(wr.min()) >= 0.0
        and float(wr.max()) <= 1.0,
        "weight tensor contract failed",
    )
    local = (similarity * wr[None]).max(dim=1).values
    mass = torch.sqrt(wq.mean() * wr.mean())
    value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
    require(bool(torch.isfinite(value) and torch.isfinite(mass)), "non-finite evidence")
    return float(value), float(mass)


def endpoint_evidence(
    query: torch.Tensor,
    reference: torch.Tensor,
    query_map: torch.Tensor,
    reference_map: torch.Tensor,
    query_shift: int,
    reference_shift: int,
) -> dict[str, dict[str, float]]:
    query_unit = F.normalize(query.to(torch.float64), dim=1)
    reference_unit = F.normalize(reference.to(torch.float64), dim=1)
    similarity = query_unit @ reference_unit.T
    ones_q = torch.ones_like(query_map, dtype=torch.float64)
    ones_r = torch.ones_like(reference_map, dtype=torch.float64)
    a, am = reduce_score(similarity, ones_q, ones_r)
    b, bm = reduce_score(similarity, query_map, ones_r)
    bq, _ = reduce_score(similarity, query_map.roll(query_shift), ones_r)
    c, cm = reduce_score(similarity, query_map, reference_map)
    cq, _ = reduce_score(
        similarity, query_map.roll(query_shift), reference_map
    )
    cr, _ = reduce_score(
        similarity, query_map, reference_map.roll(reference_shift)
    )
    return {
        "A_ALL": {
            "real_score": a,
            "visibility_mass": am,
            "query_control_score": a,
            "reference_control_score": a,
        },
        "B_QUERY": {
            "real_score": b,
            "visibility_mass": bm,
            "query_control_score": bq,
            "reference_control_score": b,
        },
        "C_PAIRED": {
            "real_score": c,
            "visibility_mass": cm,
            "query_control_score": cq,
            "reference_control_score": cr,
        },
    }


def symmetric(left: float, right: float) -> float:
    return (float(left) - float(right)) / (
        abs(float(left)) + abs(float(right)) + 1e-12
    )


def native6_feature(
    base_scores: list[float],
    winner_evidence: Mapping[str, float],
    counterpart_evidence: Mapping[str, float],
    counterpart_position: int,
) -> torch.Tensor:
    mean = sum(base_scores) / len(base_scores)
    std = (
        sum((value - mean) ** 2 for value in base_scores) / len(base_scores)
    ) ** 0.5
    winner_normalized = float(winner_evidence["real_score"]) / max(
        float(winner_evidence["visibility_mass"]), 1e-12
    )
    counterpart_normalized = float(counterpart_evidence["real_score"]) / max(
        float(counterpart_evidence["visibility_mass"]), 1e-12
    )
    winner_query = float(winner_evidence["real_score"]) - float(
        winner_evidence["query_control_score"]
    )
    counterpart_query = float(counterpart_evidence["real_score"]) - float(
        counterpart_evidence["query_control_score"]
    )
    winner_reference = float(winner_evidence["real_score"]) - float(
        winner_evidence["reference_control_score"]
    )
    counterpart_reference = float(counterpart_evidence["real_score"]) - float(
        counterpart_evidence["reference_control_score"]
    )
    return torch.tensor(
        [
            (base_scores[counterpart_position] - base_scores[0]) / max(std, 1e-12),
            symmetric(counterpart_evidence["real_score"], winner_evidence["real_score"]),
            symmetric(
                counterpart_evidence["visibility_mass"],
                winner_evidence["visibility_mass"],
            ),
            symmetric(counterpart_normalized, winner_normalized),
            symmetric(counterpart_query, winner_query),
            symmetric(counterpart_reference, winner_reference),
        ],
        dtype=torch.float64,
    ).unsqueeze(0)


def control_shifts(cohort: str, query_length: int, reference_length: int) -> tuple[int, int]:
    if cohort == "BALANCED32_V1":
        return 1, 1
    if cohort == "BALANCED32_V2":
        return max(1, query_length // 2), max(1, reference_length // 2)
    raise M0Error("unknown Pair64 cohort")


def geometry_projection(j0_record: Mapping[str, Any], p0_record: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(J0_ALLOWED_KEYS).issubset(j0_record)
        and int(j0_record["pair64_ordinal"]) == int(p0_record["pair64_ordinal"])
        and str(j0_record["pair_cohort"]) == str(p0_record["pair_cohort"])
        and str(j0_record["query_id"]) == str(p0_record["query_id"])
        and int(j0_record["canonical_heldout_fold"])
        == int(p0_record["canonical_heldout_fold"])
        and str(j0_record["canonical_track"]) == str(p0_record["canonical_track"])
        and str(j0_record["preseal_natural_c128_sha256"])
        == str(p0_record["natural_c128"]["sha256"]),
        "J0/P0 geometry projection drift",
    )
    require(
        str(j0_record["preseal_record_sha256"])
        == canonical_sha256(p0_record),
        "J0 preseal-record binding drift",
    )
    winner = j0_record["winner"]
    counterpart = j0_record["counterpart"]
    return {
        "pair64_ordinal": int(j0_record["pair64_ordinal"]),
        "pair_cohort": str(j0_record["pair_cohort"]),
        "query_id": str(j0_record["query_id"]),
        "canonical_heldout_fold": int(j0_record["canonical_heldout_fold"]),
        "canonical_track": str(j0_record["canonical_track"]),
        "query_source_path": str(p0_record["query_source_path"]),
        "query_source_image_sha256": str(p0_record["query_source_image_sha256"]),
        "query_grid_shape": list(map(int, p0_record["query_grid_shape"])),
        "preprocessing_frame": str(p0_record["preprocessing_frame"]),
        "oof_pointer": dict(p0_record["oof_pointer"]),
        "adapted_image_tokens": dict(p0_record["adapted_image_tokens"]),
        "physical_row_scores": dict(p0_record["physical_row_scores"]),
        "natural_c128_representative_physical_rows": list(
            map(int, p0_record["natural_c128"]["representative_physical_rows"])
        ),
        "preseal_record_sha256": str(j0_record["preseal_record_sha256"]),
        "preseal_natural_c128_sha256": str(
            j0_record["preseal_natural_c128_sha256"]
        ),
        "winner": {
            "candidate_position": int(winner["candidate_position"]),
            "representative_physical_row": int(winner["representative_physical_row"]),
        },
        "counterpart": {
            "candidate_position": int(counterpart["candidate_position"]),
            "representative_physical_row": int(
                counterpart["representative_physical_row"]
            ),
        },
    }


def training_projection(j0_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair64_ordinal": int(j0_record["pair64_ordinal"]),
        "pair_cohort": str(j0_record["pair_cohort"]),
        "query_id": str(j0_record["query_id"]),
        "switch_label": bool(j0_record["switch_label"]),
    }


def validated_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"fixed input hash drift: {path.name}")
    p0 = read_json(P0)
    p0_validation = read_json(P0_VALIDATION)
    j0 = read_json(J0)
    j0_validation = read_json(J0_VALIDATION)
    require(
        p0.get("status") == P0_READY
        and p0.get("logical_sha256") == logical_sha256(p0)
        and p0_validation.get("status") == P0_VALIDATED
        and p0_validation.get("preseal_sha256") == sha256_file(P0)
        and p0_validation.get("preseal_logical_sha256") == p0["logical_sha256"]
        and p0_validation.get("logical_sha256") == logical_sha256(p0_validation)
        and p0_validation.get("producer_sha256") == EXPECTED_HASHES[P0_PRODUCER]
        and p0_validation.get("validator_sha256") == EXPECTED_HASHES[P0_VALIDATOR]
        and all(p0_validation.get("checks", {}).values()),
        "P0 base preseal authority drift",
    )
    require(
        j0.get("status") == J0_READY
        and j0.get("logical_sha256") == logical_sha256(j0)
        and j0.get("population", {}).get("query_count") == 64
        and j0.get("population", {}).get("fixed_pair_count") == 64
        and j0.get("population", {}).get("selected_endpoint_count") == 128
        and j0.get("p0_unchanged_after_target_join") is True
        and j0.get("p0_hash_before_target_join") == sha256_file(P0)
        and j0.get("p0_hash_after_target_join") == sha256_file(P0)
        and j0.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_FRESH_ROMA_PAIR_FEATURES",
        "J0 fixed-pair authority drift",
    )
    require(
        j0_validation.get("status") == J0_VALIDATED
        and j0_validation.get("fixed_pairs_sha256") == sha256_file(J0)
        and j0_validation.get("fixed_pairs_logical_sha256") == j0["logical_sha256"]
        and j0_validation.get("logical_sha256")
        == logical_sha256(j0_validation)
        and j0_validation.get("p0_preseal_sha256") == sha256_file(P0)
        and j0_validation.get("p0_validation_sha256") == sha256_file(P0_VALIDATION)
        and j0_validation.get("producer_sha256") == EXPECTED_HASHES[J0_PRODUCER]
        and j0_validation.get("validator_sha256") == EXPECTED_HASHES[J0_VALIDATOR]
        and all(j0_validation.get("checks", {}).values())
        and j0_validation.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_FRESH_ROMA_PAIR_FEATURES",
        "J0 independent validation drift",
    )
    return p0, j0


def synthetic_self_test() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(6, 128, generator=generator)
    reference = torch.randn(7, 128, generator=generator)
    qmap = torch.sigmoid(torch.randn(6, generator=generator)).to(torch.float64)
    rmap = torch.sigmoid(torch.randn(7, generator=generator)).to(torch.float64)
    for ordinal in (0, 32):
        qshift = 1 if ordinal < 32 else max(1, qmap.numel() // 2)
        rshift = 1 if ordinal < 32 else max(1, rmap.numel() // 2)
        evidence = endpoint_evidence(query, reference, qmap, rmap, qshift, rshift)
        for arm in ARMS:
            feature = native6_feature([0.5, 0.4], evidence[arm], evidence[arm], 1)
            require(feature.shape == (1, 6) and bool(torch.isfinite(feature).all()), "synthetic feature failed")
    j0 = {
        "pair64_ordinal": 0,
        "pair_cohort": "BALANCED32_V1",
        "query_id": "fixture",
        "canonical_heldout_fold": 1,
        "canonical_track": "outcome",
        "winner": {"candidate_position": 0, "representative_physical_row": 3, "corrected_identity": "winner"},
        "counterpart": {"candidate_position": 1, "representative_physical_row": 4, "corrected_identity": "counterpart"},
        "switch_label": True,
        "preseal_record_sha256": "a" * 64,
        "preseal_natural_c128_sha256": "b" * 64,
        **{key: "SECRET" for key in J0_PROTECTED_KEYS},
    }
    p0 = {
        "pair64_ordinal": 0,
        "pair_cohort": "BALANCED32_V1",
        "query_id": "fixture",
        "canonical_heldout_fold": 1,
        "canonical_track": "outcome",
        "query_source_path": "/tmp/fixture.png",
        "query_source_image_sha256": "c" * 64,
        "query_grid_shape": [2, 3],
        "preprocessing_frame": "DECODED_RAW_BEFORE_EXIF",
        "oof_pointer": {},
        "adapted_image_tokens": {},
        "physical_row_scores": {},
        "natural_c128": {
            "representative_physical_rows": list(range(128)),
            "sha256": "b" * 64,
        },
    }
    j0["preseal_record_sha256"] = canonical_sha256(p0)
    before = geometry_projection(j0, p0)
    poisoned = copy.deepcopy(j0)
    poisoned.update({key: "POISON" for key in J0_PROTECTED_KEYS})
    poisoned["winner"]["corrected_identity"] = "POISON_WINNER"
    poisoned["counterpart"]["corrected_identity"] = "POISON_COUNTERPART"
    require(geometry_projection(poisoned, p0) == before, "protected-field poison changed geometry")
    flipped = copy.deepcopy(j0)
    flipped["switch_label"] = False
    require(geometry_projection(flipped, p0) == before, "label changed geometry")
    require(
        training_projection(flipped)["switch_label"] is False
        and training_projection(j0)["switch_label"] is True,
        "training projection label isolation failed",
    )
    print(json.dumps({"status": "M0_SYNTHETIC_SELF_TEST_PASS"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_self_test()
        return
    require(args.shard in range(SHARD_COUNT), "shard must be 0..7")
    require(torch.cuda.is_available(), "formal M0 fresh RoMa requires CUDA")
    out_dir = OUT_ROOT / f"shard{args.shard:02d}"
    p0, j0 = validated_inputs()
    p0_records = p0.get("records", [])
    j0_records = j0.get("records", [])
    start = args.shard * QUERIES_PER_SHARD
    stop = start + QUERIES_PER_SHARD
    selected_p0 = p0_records[start:stop]
    selected_j0 = j0_records[start:stop]
    require(
        len(p0_records) == len(j0_records) == 64
        and len(selected_p0) == len(selected_j0) == QUERIES_PER_SHARD,
        "P0/J0 shard population drift",
    )

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    resolver = HybridSpatialReferenceResolver(gallery_source=gallery)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    roma = RoMaV2()
    output_records = []
    for p0_record, j0_record in zip(selected_p0, selected_j0, strict=True):
        projection = geometry_projection(j0_record, p0_record)
        training = training_projection(j0_record)
        ordinal = projection["pair64_ordinal"]
        query_path = Path(projection["query_source_path"])
        require(
            sha256_file(query_path) == projection["query_source_image_sha256"],
            "query image bytes drift",
        )
        pointer = projection["oof_pointer"]
        oof_path = ROOT / pointer["relative_payload_path"]
        require(sha256_file(oof_path) == pointer["payload_sha256"], "OOF pointer drift")
        oof_payload = torch.load(oof_path, map_location="cpu", weights_only=False, mmap=True)
        oof_record = oof_payload["records"][int(pointer["record_index"])]
        require(
            str(oof_record["query_id"]) == projection["query_id"]
            and int(oof_record["query_ordinal"]) == int(pointer["record_query_ordinal"]),
            "OOF pointed record drift",
        )
        query_tokens = torch.as_tensor(oof_record["adapted_image_tokens"]).detach().cpu().contiguous()
        full_scores = torch.as_tensor(oof_record["physical_row_scores"]).detach().cpu().contiguous()
        require(
            n2_tensor_sha256(query_tokens) == projection["adapted_image_tokens"]["sha256"]
            and n2_tensor_sha256(full_scores) == projection["physical_row_scores"]["sha256"],
            "OOF tensor reopen hash drift",
        )
        require(
            query_tokens.dtype == torch.float32
            and query_tokens.ndim == 2
            and query_tokens.shape[1] == 128
            and full_scores.dtype == torch.float32
            and full_scores.shape == (5413,)
            and bool(torch.isfinite(query_tokens).all())
            and bool(torch.isfinite(full_scores).all()),
            "OOF tensor dtype/shape/finiteness drift",
        )
        c128_rows = projection["natural_c128_representative_physical_rows"]
        base_scores = [float(full_scores[row]) for row in c128_rows]
        qgeometry = geometry(
            query_path,
            projection["query_source_image_sha256"],
            f"pair64-training-query:{ordinal}",
            tuple(projection["query_grid_shape"]),
            projection["preprocessing_frame"],
        )
        qgeometry_payload = geometry_payload(qgeometry, projection["preprocessing_frame"])
        query_image = oriented(query_path)
        endpoint_outputs: dict[str, dict[str, Any]] = {}
        for endpoint_name in ("winner", "counterpart"):
            endpoint = projection[endpoint_name]
            position = int(endpoint["candidate_position"])
            row = int(endpoint["representative_physical_row"])
            require(c128_rows[position] == row, "endpoint row/position drift")
            resolved = resolver.resolve(row)
            require(resolved.physical_row == row, "resolver physical-row drift")
            reference_tokens = resolved.tokens.detach().cpu().contiguous()
            reference_path = gallery.raw_paths[row]
            reference_sha = sha256_file(reference_path)
            rgeometry = geometry(
                reference_path,
                reference_sha,
                f"gallery-row:{row}",
                tuple(resolved.grid_shape),
                DECODED_RAW_BEFORE_EXIF,
            )
            prediction = roma.match(query_image.copy(), oriented(reference_path))
            query_map = cell_means(
                prediction["overlap_AB"][0, ..., 0].detach().cpu(), qgeometry
            )
            reference_map = cell_means(
                prediction["overlap_BA"][0, ..., 0].detach().cpu(), rgeometry
            )
            require(
                query_map.shape == (query_tokens.shape[0],)
                and reference_map.shape == (reference_tokens.shape[0],)
                and reference_tokens.dtype == torch.float16
                and reference_tokens.ndim == 2
                and reference_tokens.shape[1] == 128
                and tensor_sha256(reference_tokens) == resolved.tokens_sha256
                and bool(torch.isfinite(query_map).all() and torch.isfinite(reference_map).all())
                and float(query_map.min()) >= 0.0
                and float(query_map.max()) <= 1.0
                and float(reference_map.min()) >= 0.0
                and float(reference_map.max()) <= 1.0,
                "fresh RoMa map contract failed",
            )
            qshift, rshift = control_shifts(
                projection["pair_cohort"], query_tokens.shape[0], reference_tokens.shape[0]
            )
            evidence = endpoint_evidence(
                query_tokens,
                reference_tokens,
                query_map,
                reference_map,
                qshift,
                rshift,
            )
            endpoint_outputs[endpoint_name] = {
                "candidate_position": position,
                "representative_physical_row": row,
                "reference_source_path": str(reference_path),
                "reference_source_image_sha256": reference_sha,
                "reference_grid_shape": tuple(map(int, resolved.grid_shape)),
                "reference_tokens_sha256": str(resolved.tokens_sha256),
                "reference_source_kind": str(resolved.source_kind),
                "reference_source_logical_sha256": str(resolved.source_logical_sha256),
                "query_map": query_map,
                "reference_map": reference_map,
                "query_map_sha256": map_sha256(query_map),
                "reference_map_sha256": map_sha256(reference_map),
                "reference_geometry": geometry_payload(
                    rgeometry, DECODED_RAW_BEFORE_EXIF
                ),
                "query_roll_shift": qshift,
                "reference_roll_shift": rshift,
                "evidence": evidence,
                "historical_map_reuse_count": 0,
            }
        native_features = {}
        counterpart_position = projection["counterpart"]["candidate_position"]
        for arm in ARMS:
            native_features[arm] = native6_feature(
                base_scores,
                endpoint_outputs["winner"]["evidence"][arm],
                endpoint_outputs["counterpart"]["evidence"][arm],
                counterpart_position,
            )
        output_records.append(
            {
                "pair64_ordinal": ordinal,
                "pair_cohort": projection["pair_cohort"],
                "query_id": projection["query_id"],
                "canonical_heldout_fold": projection["canonical_heldout_fold"],
                "canonical_track": projection["canonical_track"],
                "geometry_projection_sha256": canonical_sha256(projection),
                "preseal_record_sha256": projection["preseal_record_sha256"],
                "preseal_natural_c128_sha256": projection[
                    "preseal_natural_c128_sha256"
                ],
                "query_source_path": projection["query_source_path"],
                "query_source_image_sha256": projection[
                    "query_source_image_sha256"
                ],
                "query_grid_shape": projection["query_grid_shape"],
                "preprocessing_frame": projection["preprocessing_frame"],
                "query_geometry": qgeometry_payload,
                "oof_pointer": projection["oof_pointer"],
                "adapted_image_tokens_sha256": projection[
                    "adapted_image_tokens"
                ]["sha256"],
                "physical_row_scores_sha256": projection[
                    "physical_row_scores"
                ]["sha256"],
                "natural_c128_representative_physical_rows": c128_rows,
                "base_c128_scores": torch.tensor(base_scores, dtype=torch.float64),
                "base_c128_scores_sha256": tensor_sha256(
                    torch.tensor(base_scores, dtype=torch.float64)
                ),
                "endpoint_order": ["winner", "counterpart"],
                "endpoints": endpoint_outputs,
                "native6_features": native_features,
                "native6_feature_sha256": {
                    arm: tensor_sha256(native_features[arm]) for arm in ARMS
                },
                "training_projection": training,
                "training_projection_sha256": canonical_sha256(training),
                "roll_convention": "SHIFT_ONE" if projection["pair_cohort"] == "BALANCED32_V1" else "HALF_LENGTH",
                "roma_pair_evaluation_count": 2,
                "similarity_compute_count": 2,
                "historical_postjoin_map_reuse_count": 0,
                "cbind_feature_count": 0,
                "fullnegative_feature_count": 0,
                "j0_protected_field_semantic_read_count": 0,
                "model_update_count": 0,
            }
        )
        print(
            json.dumps(
                {
                    "event": "pair64_training_feature_ready",
                    "shard": args.shard,
                    "query_id": projection["query_id"],
                    "maps": 2,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    bindings = {
        "active_contract_sha256": sha256_file(ACTIVE),
        "scope_correction_sha256": sha256_file(SCOPE),
        "label_correction_sha256": sha256_file(LABEL),
        "active_disposition_sha256": sha256_file(DISPOSITION),
        "p0_preseal_sha256": sha256_file(P0),
        "p0_validation_sha256": sha256_file(P0_VALIDATION),
        "p0_producer_sha256": sha256_file(P0_PRODUCER),
        "p0_validator_sha256": sha256_file(P0_VALIDATOR),
        "j0_fixed_pairs_sha256": sha256_file(J0),
        "j0_validation_sha256": sha256_file(J0_VALIDATION),
        "j0_producer_sha256": sha256_file(J0_PRODUCER),
        "j0_validator_sha256": sha256_file(J0_VALIDATOR),
        "roma_checkpoint_sha256": sha256_file(ROMA_WEIGHTS),
        "processor_config_sha256": sha256_file(PROCESSOR_CONFIG),
        "resolver_source_sha256": sha256_file(RESOLVER_SOURCE),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    access = {
        "p0_preseal_read_count": 1,
        "j0_fixed_pair_artifact_read_count": 1,
        "j0_allowed_record_projection_count": QUERIES_PER_SHARD,
        "j0_protected_field_semantic_read_count": 0,
        "target_identity_read_count": 0,
        "target_supergroup_read_count": 0,
        "target_state_read_count": 0,
        "target_candidate_position_read_count": 0,
        "reference_resolver_call_count": MAPS_PER_SHARD,
        "encoder_model_load_count": 0,
        "encoder_model_forward_count": 0,
        "roma_model_load_count": 1,
        "roma_pair_evaluation_count": MAPS_PER_SHARD,
        "historical_postjoin_map_read_count": 0,
        "cbind_feature_count": 0,
        "fullnegative_feature_count": 0,
        "model_update_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    payload = {
        "version": VERSION,
        "status": READY,
        "claim_level": "POSTSEAL_PAIR64_TRAINING_ONLY_FRESH_ROMA_A_B_C_NATIVE6_FEATURE_SHARD",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "query_count": len(output_records),
        "fixed_pair_count": len(output_records),
        "selected_endpoint_count": len(output_records) * ENDPOINTS_PER_QUERY,
        "roma_pair_evaluation_count": len(output_records) * ENDPOINTS_PER_QUERY,
        "arms": list(ARMS),
        "feature_names": list(FEATURE_NAMES),
        "records": output_records,
        "bindings": bindings,
        "access": access,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
    }
    receipt = {
        "version": VERSION,
        "status": READY,
        "shard": args.shard,
        "query_count": len(output_records),
        "fixed_pair_count": len(output_records),
        "selected_endpoint_count": len(output_records) * ENDPOINTS_PER_QUERY,
        "roma_pair_evaluation_count": len(output_records) * ENDPOINTS_PER_QUERY,
        "switch_label_count": sum(
            record["training_projection"]["switch_label"] for record in output_records
        ),
        "hold_label_count": sum(
            not record["training_projection"]["switch_label"]
            for record in output_records
        ),
        "historical_postjoin_map_reuse_count": 0,
        "cbind_feature_count": 0,
        "fullnegative_feature_count": 0,
        "bindings": bindings,
        "access": access,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "payload_sha256": "",
        "logical_sha256": "",
    }
    atomic_directory(out_dir, payload, receipt)
    print(
        json.dumps(
            {
                "status": READY,
                "shard": args.shard,
                "queries": len(output_records),
                "roma_pairs": MAPS_PER_SHARD,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
