#!/usr/bin/env python3
"""Independent validator for one Pair64 fresh-RoMa training feature shard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch.nn import functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
import sys

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
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
PROCESSOR_CONFIG = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1"

VERSION = "routea_n2_fresh_d1_matched_three_arm_pair64_training_feature_shard_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_READY"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_VALIDATED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_AGGREGATE_VALIDATION"
SHARD_COUNT = 8
QUERIES_PER_SHARD = 8
MAPS_PER_SHARD = 16
MAP_REPLAY_ATOL = 1e-6
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
P0_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY"
P0_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED"
J0_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY"
J0_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_VALIDATED"
PROTECTED_KEYS = {
    "target_identity",
    "target_supergroup",
    "target_physical_row_provenance",
    "n2_label_source_folds",
    "target_state",
    "target_naturally_in_fresh_c128",
    "target_candidate_position",
}
FORBIDDEN_OUTPUT_KEYS = PROTECTED_KEYS | {"corrected_identity"}
RECORD_KEYS = {
    "pair64_ordinal",
    "pair_cohort",
    "query_id",
    "canonical_heldout_fold",
    "canonical_track",
    "geometry_projection_sha256",
    "preseal_record_sha256",
    "preseal_natural_c128_sha256",
    "query_source_path",
    "query_source_image_sha256",
    "query_grid_shape",
    "preprocessing_frame",
    "query_geometry",
    "oof_pointer",
    "adapted_image_tokens_sha256",
    "physical_row_scores_sha256",
    "natural_c128_representative_physical_rows",
    "base_c128_scores",
    "base_c128_scores_sha256",
    "endpoint_order",
    "endpoints",
    "native6_features",
    "native6_feature_sha256",
    "training_projection",
    "training_projection_sha256",
    "roll_convention",
    "roma_pair_evaluation_count",
    "similarity_compute_count",
    "historical_postjoin_map_reuse_count",
    "cbind_feature_count",
    "fullnegative_feature_count",
    "j0_protected_field_semantic_read_count",
    "model_update_count",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


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
    require(isinstance(value, dict), "JSON root is not object")
    return value


def equal_nested(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(equal_nested(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                equal_nested(a, b) for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def walk_keys(value: Any) -> set[str]:
    output = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            output.add(str(key))
            output |= walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            output |= walk_keys(child)
    return output


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


def independent_geometry(
    path: Path, image_sha: str, key: str, grid: tuple[int, int], frame: str
):
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


def geometry_dict(value: Any, frame: str) -> dict[str, Any]:
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


def independent_cell_means(overlap: torch.Tensor, patch_geometry: Any) -> torch.Tensor:
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
        left = max(0, min(width - 1, math.floor(x0 * width)))
        top = max(0, min(height - 1, math.floor(y0 * height)))
        right = max(left + 1, min(width, math.ceil(x1 * width)))
        bottom = max(top + 1, min(height, math.ceil(y1 * height)))
        output.append(values[top:bottom, left:right].mean())
    return torch.stack(output).contiguous()


def reduce_score(
    similarity: torch.Tensor, query_weight: torch.Tensor, reference_weight: torch.Tensor
) -> tuple[float, float]:
    wq = query_weight.to(torch.float64)
    wr = reference_weight.to(torch.float64)
    local = (similarity * wr[None]).max(dim=1).values
    mass = torch.sqrt(wq.mean() * wr.mean())
    value = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
    require(bool(torch.isfinite(value) and torch.isfinite(mass)), "formula nonfinite")
    return float(value), float(mass)


def independent_evidence(
    query: torch.Tensor,
    reference: torch.Tensor,
    query_map: torch.Tensor,
    reference_map: torch.Tensor,
    query_shift: int,
    reference_shift: int,
) -> dict[str, dict[str, float]]:
    similarity = F.normalize(query.to(torch.float64), dim=1) @ F.normalize(
        reference.to(torch.float64), dim=1
    ).T
    oq = torch.ones_like(query_map, dtype=torch.float64)
    or_ = torch.ones_like(reference_map, dtype=torch.float64)
    a, am = reduce_score(similarity, oq, or_)
    b, bm = reduce_score(similarity, query_map, or_)
    bq, _ = reduce_score(similarity, query_map.roll(query_shift), or_)
    c, cm = reduce_score(similarity, query_map, reference_map)
    cq, _ = reduce_score(
        similarity, query_map.roll(query_shift), reference_map
    )
    cr, _ = reduce_score(
        similarity, query_map, reference_map.roll(reference_shift)
    )
    return {
        "A_ALL": {"real_score": a, "visibility_mass": am, "query_control_score": a, "reference_control_score": a},
        "B_QUERY": {"real_score": b, "visibility_mass": bm, "query_control_score": bq, "reference_control_score": b},
        "C_PAIRED": {"real_score": c, "visibility_mass": cm, "query_control_score": cq, "reference_control_score": cr},
    }


def sym(left: float, right: float) -> float:
    return (float(left) - float(right)) / (abs(float(left)) + abs(float(right)) + 1e-12)


def independent_feature(
    base: list[float], winner: Mapping[str, float], counterpart: Mapping[str, float], position: int
) -> torch.Tensor:
    mean = sum(base) / len(base)
    std = (sum((value - mean) ** 2 for value in base) / len(base)) ** 0.5
    wn = float(winner["real_score"]) / max(float(winner["visibility_mass"]), 1e-12)
    cn = float(counterpart["real_score"]) / max(float(counterpart["visibility_mass"]), 1e-12)
    wq = float(winner["real_score"]) - float(winner["query_control_score"])
    cq = float(counterpart["real_score"]) - float(counterpart["query_control_score"])
    wr = float(winner["real_score"]) - float(winner["reference_control_score"])
    cr = float(counterpart["real_score"]) - float(counterpart["reference_control_score"])
    return torch.tensor(
        [
            (base[position] - base[0]) / max(std, 1e-12),
            sym(counterpart["real_score"], winner["real_score"]),
            sym(counterpart["visibility_mass"], winner["visibility_mass"]),
            sym(cn, wn),
            sym(cq, wq),
            sym(cr, wr),
        ],
        dtype=torch.float64,
    ).unsqueeze(0)


def control_shifts(cohort: str, query_length: int, reference_length: int) -> tuple[int, int]:
    if cohort == "BALANCED32_V1":
        return 1, 1
    if cohort == "BALANCED32_V2":
        return max(1, query_length // 2), max(1, reference_length // 2)
    raise ValidationError("unknown Pair64 cohort")


def geometry_projection(j0: Mapping[str, Any], p0: Mapping[str, Any]) -> dict[str, Any]:
    require(
        int(j0["pair64_ordinal"]) == int(p0["pair64_ordinal"])
        and str(j0["pair_cohort"]) == str(p0["pair_cohort"])
        and str(j0["query_id"]) == str(p0["query_id"])
        and int(j0["canonical_heldout_fold"])
        == int(p0["canonical_heldout_fold"])
        and str(j0["canonical_track"]) == str(p0["canonical_track"])
        and str(j0["preseal_record_sha256"]) == canonical_sha256(p0)
        and str(j0["preseal_natural_c128_sha256"])
        == str(p0["natural_c128"]["sha256"]),
        "independent J0/P0 projection drift",
    )
    return {
        "pair64_ordinal": int(j0["pair64_ordinal"]),
        "pair_cohort": str(j0["pair_cohort"]),
        "query_id": str(j0["query_id"]),
        "canonical_heldout_fold": int(j0["canonical_heldout_fold"]),
        "canonical_track": str(j0["canonical_track"]),
        "query_source_path": str(p0["query_source_path"]),
        "query_source_image_sha256": str(p0["query_source_image_sha256"]),
        "query_grid_shape": list(map(int, p0["query_grid_shape"])),
        "preprocessing_frame": str(p0["preprocessing_frame"]),
        "oof_pointer": dict(p0["oof_pointer"]),
        "adapted_image_tokens": dict(p0["adapted_image_tokens"]),
        "physical_row_scores": dict(p0["physical_row_scores"]),
        "natural_c128_representative_physical_rows": list(
            map(int, p0["natural_c128"]["representative_physical_rows"])
        ),
        "preseal_record_sha256": str(j0["preseal_record_sha256"]),
        "preseal_natural_c128_sha256": str(j0["preseal_natural_c128_sha256"]),
        "winner": {
            "candidate_position": int(j0["winner"]["candidate_position"]),
            "representative_physical_row": int(j0["winner"]["representative_physical_row"]),
        },
        "counterpart": {
            "candidate_position": int(j0["counterpart"]["candidate_position"]),
            "representative_physical_row": int(j0["counterpart"]["representative_physical_row"]),
        },
    }


def training_projection(j0: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair64_ordinal": int(j0["pair64_ordinal"]),
        "pair_cohort": str(j0["pair_cohort"]),
        "query_id": str(j0["query_id"]),
        "switch_label": bool(j0["switch_label"]),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable validation exists: {path}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_self_test() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(6, 128, generator=generator)
    reference = torch.randn(7, 128, generator=generator)
    qmap = torch.sigmoid(torch.randn(6, generator=generator)).to(torch.float64)
    rmap = torch.sigmoid(torch.randn(7, generator=generator)).to(torch.float64)
    evidence = independent_evidence(query, reference, qmap, rmap, 1, 1)
    for arm in ARMS:
        feature = independent_feature([0.5, 0.4], evidence[arm], evidence[arm], 1)
        require(feature.shape == (1, 6) and bool(torch.isfinite(feature).all()), "fixture failed")
    p0 = {
        "pair64_ordinal": 0,
        "pair_cohort": "BALANCED32_V1",
        "query_id": "fixture",
        "canonical_heldout_fold": 1,
        "canonical_track": "outcome",
        "query_source_path": "/tmp/fixture.png",
        "query_source_image_sha256": "a" * 64,
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
    j0 = {
        "pair64_ordinal": 0,
        "pair_cohort": "BALANCED32_V1",
        "query_id": "fixture",
        "canonical_heldout_fold": 1,
        "canonical_track": "outcome",
        "winner": {
            "candidate_position": 0,
            "representative_physical_row": 0,
            "corrected_identity": "winner",
        },
        "counterpart": {
            "candidate_position": 1,
            "representative_physical_row": 1,
            "corrected_identity": "counterpart",
        },
        "switch_label": True,
        "preseal_record_sha256": canonical_sha256(p0),
        "preseal_natural_c128_sha256": "b" * 64,
        **{key: "SECRET" for key in PROTECTED_KEYS},
    }
    base_projection = geometry_projection(j0, p0)
    base_training = training_projection(j0)
    poison = copy.deepcopy(j0)
    poison.update({key: "POISON" for key in PROTECTED_KEYS})
    poison["winner"]["corrected_identity"] = "POISON_WINNER"
    poison["counterpart"]["corrected_identity"] = "POISON_COUNTERPART"
    require(
        geometry_projection(poison, p0) == base_projection
        and training_projection(poison) == base_training,
        "protected-field poison fixture failed",
    )
    flipped = copy.deepcopy(j0)
    flipped["switch_label"] = False
    require(
        geometry_projection(flipped, p0) == base_projection
        and canonical_sha256(training_projection(flipped))
        != canonical_sha256(base_training),
        "label-isolation fixture failed",
    )
    print(json.dumps({"status": "M0_VALIDATOR_SYNTHETIC_SELF_TEST_PASS"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_self_test()
        return
    require(args.shard in range(SHARD_COUNT), "shard must be 0..7")
    require(torch.cuda.is_available(), "formal independent RoMa replay requires CUDA")
    shard_dir = OUT_ROOT / f"shard{args.shard:02d}"
    payload_path = shard_dir / "payload.pt"
    receipt_path = shard_dir / "receipt.json"
    out = shard_dir / "validation.json"
    require(not out.exists(), "immutable shard validation exists")
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"fixed input hash drift: {path.name}")
    require(sha256_file(PRODUCER) == read_json(receipt_path)["bindings"]["producer_sha256"], "producer hash drift")

    p0 = read_json(P0)
    p0_validation = read_json(P0_VALIDATION)
    j0 = read_json(J0)
    j0_validation = read_json(J0_VALIDATION)
    require(
        p0.get("status") == P0_READY
        and p0_validation.get("status") == P0_VALIDATED
        and p0_validation.get("preseal_sha256") == sha256_file(P0)
        and p0_validation.get("preseal_logical_sha256")
        == p0.get("logical_sha256")
        and p0_validation.get("logical_sha256")
        == logical_sha256(p0_validation)
        and p0_validation.get("producer_sha256") == EXPECTED_HASHES[P0_PRODUCER]
        and p0_validation.get("validator_sha256") == EXPECTED_HASHES[P0_VALIDATOR]
        and all(p0_validation.get("checks", {}).values()),
        "P0 validation drift",
    )
    require(
        j0.get("status") == J0_READY
        and j0.get("logical_sha256") == logical_sha256(j0)
        and j0_validation.get("status") == J0_VALIDATED
        and j0_validation.get("fixed_pairs_sha256") == sha256_file(J0)
        and j0_validation.get("fixed_pairs_logical_sha256")
        == j0.get("logical_sha256")
        and j0_validation.get("logical_sha256")
        == logical_sha256(j0_validation)
        and j0_validation.get("producer_sha256") == EXPECTED_HASHES[J0_PRODUCER]
        and j0_validation.get("validator_sha256") == EXPECTED_HASHES[J0_VALIDATOR]
        and all(j0_validation.get("checks", {}).values()),
        "J0 validation drift",
    )
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = read_json(receipt_path)
    start = args.shard * QUERIES_PER_SHARD
    stop = start + QUERIES_PER_SHARD
    p0_records = p0["records"][start:stop]
    j0_records = j0["records"][start:stop]
    records = payload.get("records", [])
    require(len(records) == len(p0_records) == len(j0_records) == QUERIES_PER_SHARD, "shard population drift")

    gallery = build_gallery_source(verify_cache_file_sha256=True)
    resolver = HybridSpatialReferenceResolver(gallery_source=gallery)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    roma = RoMaV2()
    map_error = score_error = feature_error = 0.0
    protected_poison_ok = label_geometry_invariance = True
    record_checks = []
    switch_count = 0
    for output, p0_record, j0_record in zip(records, p0_records, j0_records, strict=True):
        require(set(output) == RECORD_KEYS, "record schema/forbidden output drift")
        require(not (walk_keys(output) & FORBIDDEN_OUTPUT_KEYS), "protected J0 field leaked to output")
        projection = geometry_projection(j0_record, p0_record)
        training = training_projection(j0_record)
        poisoned = copy.deepcopy(j0_record)
        poisoned.update({key: "POISON" for key in PROTECTED_KEYS})
        poisoned["winner"]["corrected_identity"] = "POISON_WINNER"
        poisoned["counterpart"]["corrected_identity"] = "POISON_COUNTERPART"
        protected_poison_ok = protected_poison_ok and geometry_projection(poisoned, p0_record) == projection and training_projection(poisoned) == training
        flipped = copy.deepcopy(j0_record)
        flipped["switch_label"] = not bool(j0_record["switch_label"])
        flipped_training = training_projection(flipped)
        label_geometry_invariance = (
            label_geometry_invariance
            and geometry_projection(flipped, p0_record) == projection
            and flipped_training["switch_label"] != training["switch_label"]
            and canonical_sha256(flipped_training) != canonical_sha256(training)
        )

        pointer = projection["oof_pointer"]
        oof_path = ROOT / pointer["relative_payload_path"]
        require(sha256_file(oof_path) == pointer["payload_sha256"], "OOF pointer drift")
        oof_payload = torch.load(oof_path, map_location="cpu", weights_only=False, mmap=True)
        oof = oof_payload["records"][int(pointer["record_index"])]
        query_tokens = torch.as_tensor(oof["adapted_image_tokens"]).detach().cpu().contiguous()
        full_scores = torch.as_tensor(oof["physical_row_scores"]).detach().cpu().contiguous()
        require(
            oof["query_id"] == projection["query_id"]
            and n2_tensor_sha256(query_tokens) == projection["adapted_image_tokens"]["sha256"]
            and n2_tensor_sha256(full_scores) == projection["physical_row_scores"]["sha256"],
            "OOF tensor replay drift",
        )
        require(
            query_tokens.dtype == torch.float32
            and query_tokens.ndim == 2
            and query_tokens.shape[1] == 128
            and full_scores.dtype == torch.float32
            and full_scores.shape == (5413,)
            and bool(torch.isfinite(query_tokens).all())
            and bool(torch.isfinite(full_scores).all()),
            "OOF tensor contract drift",
        )
        rows = projection["natural_c128_representative_physical_rows"]
        base = [float(full_scores[row]) for row in rows]
        query_path = Path(projection["query_source_path"])
        require(sha256_file(query_path) == projection["query_source_image_sha256"], "query source drift")
        qgeometry = independent_geometry(
            query_path,
            projection["query_source_image_sha256"],
            f"pair64-training-query:{projection['pair64_ordinal']}",
            tuple(projection["query_grid_shape"]),
            projection["preprocessing_frame"],
        )
        require(
            equal_nested(
                output["query_geometry"],
                geometry_dict(qgeometry, projection["preprocessing_frame"]),
            ),
            "query geometry replay drift",
        )
        query_image = oriented(query_path)
        evidence_by_endpoint = {}
        endpoint_checks = []
        for name in ("winner", "counterpart"):
            stored = output["endpoints"][name]
            expected_endpoint = projection[name]
            row = expected_endpoint["representative_physical_row"]
            position = expected_endpoint["candidate_position"]
            require(rows[position] == row, "endpoint position drift")
            resolved = resolver.resolve(row)
            require(resolved.physical_row == row, "resolver physical-row drift")
            reference_tokens = resolved.tokens.detach().cpu().contiguous()
            require(
                reference_tokens.dtype == torch.float16
                and reference_tokens.ndim == 2
                and reference_tokens.shape[1] == 128
                and tensor_sha256(reference_tokens) == resolved.tokens_sha256
                and bool(torch.isfinite(reference_tokens).all()),
                "reference token contract drift",
            )
            reference_path = gallery.raw_paths[row]
            reference_sha = sha256_file(reference_path)
            rgeometry = independent_geometry(
                reference_path,
                reference_sha,
                f"gallery-row:{row}",
                tuple(resolved.grid_shape),
                DECODED_RAW_BEFORE_EXIF,
            )
            prediction = roma.match(query_image.copy(), oriented(reference_path))
            replay_q = independent_cell_means(
                prediction["overlap_AB"][0, ..., 0].detach().cpu(), qgeometry
            )
            replay_r = independent_cell_means(
                prediction["overlap_BA"][0, ..., 0].detach().cpu(), rgeometry
            )
            stored_q = torch.as_tensor(stored["query_map"]).detach().cpu().contiguous()
            stored_r = torch.as_tensor(stored["reference_map"]).detach().cpu().contiguous()
            map_error = max(
                map_error,
                float((stored_q - replay_q).abs().max()),
                float((stored_r - replay_r).abs().max()),
            )
            qshift, rshift = control_shifts(
                projection["pair_cohort"], query_tokens.shape[0], reference_tokens.shape[0]
            )
            evidence = independent_evidence(
                query_tokens, reference_tokens, stored_q, stored_r, qshift, rshift
            )
            for arm in ARMS:
                for field in (
                    "real_score",
                    "visibility_mass",
                    "query_control_score",
                    "reference_control_score",
                ):
                    score_error = max(
                        score_error,
                        abs(float(stored["evidence"][arm][field]) - float(evidence[arm][field])),
                    )
            evidence_by_endpoint[name] = evidence
            endpoint_checks.append(
                stored["candidate_position"] == position
                and stored["representative_physical_row"] == row
                and stored["reference_source_path"] == str(reference_path)
                and stored["reference_source_image_sha256"] == reference_sha
                and tuple(stored["reference_grid_shape"]) == tuple(resolved.grid_shape)
                and stored["reference_tokens_sha256"] == resolved.tokens_sha256
                and stored["reference_source_kind"] == resolved.source_kind
                and stored["reference_source_logical_sha256"] == resolved.source_logical_sha256
                and stored["query_map_sha256"] == map_sha256(stored_q)
                and stored["reference_map_sha256"] == map_sha256(stored_r)
                and equal_nested(
                    stored["reference_geometry"],
                    geometry_dict(rgeometry, DECODED_RAW_BEFORE_EXIF),
                )
                and stored["query_roll_shift"] == qshift
                and stored["reference_roll_shift"] == rshift
                and stored["historical_map_reuse_count"] == 0
            )
        expected_features = {}
        counterpart_position = projection["counterpart"]["candidate_position"]
        for arm in ARMS:
            feature = independent_feature(
                base,
                evidence_by_endpoint["winner"][arm],
                evidence_by_endpoint["counterpart"][arm],
                counterpart_position,
            )
            expected_features[arm] = feature
            observed = torch.as_tensor(output["native6_features"][arm])
            require(observed.shape == (1, 6), "native6 stored shape drift")
            feature_error = max(feature_error, float((observed - feature).abs().max()))
            require(output["native6_feature_sha256"][arm] == tensor_sha256(feature), "feature hash drift")
        base_tensor = torch.tensor(base, dtype=torch.float64)
        record_checks.append(
            output["pair64_ordinal"] == projection["pair64_ordinal"]
            and output["pair_cohort"] == projection["pair_cohort"]
            and output["query_id"] == projection["query_id"]
            and output["geometry_projection_sha256"] == canonical_sha256(projection)
            and output["training_projection"] == training
            and output["training_projection_sha256"]
            == canonical_sha256(training)
            and output["natural_c128_representative_physical_rows"] == rows
            and torch.equal(output["base_c128_scores"], base_tensor)
            and output["base_c128_scores_sha256"] == tensor_sha256(base_tensor)
            and output["endpoint_order"] == ["winner", "counterpart"]
            and set(output["endpoints"]) == {"winner", "counterpart"}
            and all(endpoint_checks)
            and set(output["native6_features"]) == set(ARMS)
            and output["roll_convention"]
            == ("SHIFT_ONE" if projection["pair_cohort"] == "BALANCED32_V1" else "HALF_LENGTH")
            and output["roma_pair_evaluation_count"] == 2
            and output["similarity_compute_count"] == 2
            and output["historical_postjoin_map_reuse_count"] == 0
            and output["cbind_feature_count"] == 0
            and output["fullnegative_feature_count"] == 0
            and output["j0_protected_field_semantic_read_count"] == 0
            and output["model_update_count"] == 0
        )
        switch_count += int(training["switch_label"])

    expected_bindings = {
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
        "resolver_source_sha256": sha256_file(ROOT / "src/rc_aslo_xf/cw1_sr0_s8_feature_runtime_v1.py"),
        "producer_sha256": sha256_file(PRODUCER),
    }
    expected_access = {
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
    checks = {
        "payload_scope_and_schema": payload.get("version") == VERSION
        and payload.get("status") == READY
        and payload.get("shard") == args.shard
        and payload.get("shard_count") == SHARD_COUNT
        and payload.get("query_count") == QUERIES_PER_SHARD
        and payload.get("fixed_pair_count") == QUERIES_PER_SHARD
        and payload.get("selected_endpoint_count") == MAPS_PER_SHARD
        and payload.get("roma_pair_evaluation_count") == MAPS_PER_SHARD
        and tuple(payload.get("arms", ())) == ARMS
        and tuple(payload.get("feature_names", ())) == FEATURE_NAMES,
        "input_bindings": payload.get("bindings") == expected_bindings,
        "access_boundary": payload.get("access") == expected_access,
        "record_population_and_formula": all(record_checks)
        and score_error <= 1e-12
        and feature_error <= 1e-12,
        "fresh_roma_replay": map_error <= MAP_REPLAY_ATOL,
        "protected_field_poison_invariance": protected_poison_ok,
        "switch_label_geometry_and_feature_invariance": label_geometry_invariance,
        "no_target_group_state_output": not (walk_keys(records) & FORBIDDEN_OUTPUT_KEYS),
        "no_cbind_fullnegative_or_historical_maps": all(
            record["cbind_feature_count"]
            == record["fullnegative_feature_count"]
            == record["historical_postjoin_map_reuse_count"]
            == 0
            for record in records
        ),
        "claim_boundary": payload.get("scientific_GO_or_NO_GO") is None
        and payload.get("ownership_GO_or_NO_GO") is None
        and payload.get("automatic_stage_advance") is False
        and payload.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_INDEPENDENT_VALIDATION",
        "receipt": receipt.get("status") == READY
        and receipt.get("logical_sha256") == logical_sha256(receipt)
        and receipt.get("payload_sha256") == sha256_file(payload_path)
        and receipt.get("shard") == args.shard
        and receipt.get("query_count") == QUERIES_PER_SHARD
        and receipt.get("selected_endpoint_count") == MAPS_PER_SHARD
        and receipt.get("roma_pair_evaluation_count") == MAPS_PER_SHARD
        and receipt.get("switch_label_count") == switch_count
        and receipt.get("hold_label_count") == QUERIES_PER_SHARD - switch_count
        and receipt.get("historical_postjoin_map_reuse_count") == 0
        and receipt.get("cbind_feature_count") == 0
        and receipt.get("fullnegative_feature_count") == 0
        and receipt.get("bindings") == expected_bindings
        and receipt.get("access") == expected_access,
    }
    passed = all(checks.values())
    value = {
        "version": VERSION,
        "status": VALIDATED if passed else "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_VALIDATION_ABORT",
        "claim_level": "INDEPENDENT_POSTSEAL_PAIR64_TRAINING_ONLY_FRESH_ROMA_NATIVE6_VALIDATION",
        "shard": args.shard,
        "checks": checks,
        "query_count": len(records),
        "fixed_pair_count": len(records),
        "selected_endpoint_count": len(records) * 2,
        "roma_pair_evaluation_count": len(records) * 2,
        "switch_label_count": switch_count,
        "hold_label_count": len(records) - switch_count,
        "stored_map_formula_max_abs": score_error,
        "native6_feature_max_abs": feature_error,
        "fresh_roma_replay_atol": MAP_REPLAY_ATOL,
        "fresh_roma_replay_max_abs": map_error,
        "historical_postjoin_map_reuse_count": 0,
        "cbind_feature_count": 0,
        "fullnegative_feature_count": 0,
        "payload_sha256": sha256_file(payload_path),
        "receipt_sha256": sha256_file(receipt_path),
        "producer_sha256": sha256_file(PRODUCER),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "access": expected_access,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(out, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
