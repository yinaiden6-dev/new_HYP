"""Connected-superregion bridge from ColNomic P to DINO-RCDE V.

This module is deliberately additive.  It does not change the hash-bound
DINO-RCDE V1.2 core.  ColNomic produces anonymous multi-support hypotheses;
this module converts them into immutable paired connected footprints in a
canonical oriented-image coordinate frame.  DINO-RCDE may then be decoded and
compared only inside those footprints.

No API accepts a target label, retrieval rank, winner flag, D1 score, or action.
The only P->V payload is geometry, boolean membership, anonymous candidate/source
keys, and hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol, Sequence

import torch

from .dino_rcde_v1_2_resource_core import PairEvidence
from .geometry_hypothesis_v1 import (
    MODEL_AFFINE,
    build_seed_local_connected_footprints,
    connected_components_4,
)
from .lt_hyp_pvlock_v4 import MAX_HYPOTHESES, SealedMultiPatchHypotheses


SCHEMA_VERSION = "rc_dino_rcde_colnomic_connected_superregion_v1"
COORDINATE_FRAME = "EXIF_ORIENTED_FULL_IMAGE_NORMALIZED_XYXY_V1"
STATUS_READY = "CONNECTED_SUPERREGION_READY"
STATUS_H0 = "H0_NO_LEGAL_CONNECTED_REGION"
STATUS_AMBIGUOUS = "MULTI_COMPONENT_AMBIGUITY_HOLD"
STATUS_SINGLE = "SINGLE_CONNECTED_COMPONENT"
MIN_REGION_PATCHES = 4
P_DIRECTIONS = frozenset({"a_to_b", "b_to_a"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SuperregionContractError(ValueError):
    """Fail-closed contract violation in the P->superregion->V bridge."""


class FloatReductionClosureError(ValueError):
    """A finite scalar reconstruction exceeds the frozen bounded-ULP policy."""


FLOAT_REDUCTION_MAX_ULPS = 8
FLOAT_REDUCTION_ATOL_FLOOR = {
    torch.float32: 0.0,
    torch.float64: 0.0,
}


def _scalar_float_contract(
    observed: torch.Tensor, reconstructed: torch.Tensor, *, name: str
) -> tuple[torch.Tensor, torch.Tensor, float]:
    left = torch.as_tensor(observed)
    right = torch.as_tensor(reconstructed)
    if (
        left.ndim != 0
        or right.ndim != 0
        or left.device != right.device
        or left.dtype != right.dtype
        or left.dtype not in FLOAT_REDUCTION_ATOL_FLOOR
    ):
        raise FloatReductionClosureError(
            f"{name} requires same-device float32/float64 scalars"
        )
    if not bool(torch.isfinite(left) & torch.isfinite(right)):
        raise FloatReductionClosureError(f"{name} encountered nonfinite evidence")
    left_ulp = (
        torch.nextafter(left, torch.full_like(left, float("inf"))) - left
    ).abs()
    right_ulp = (
        torch.nextafter(right, torch.full_like(right, float("inf"))) - right
    ).abs()
    ulp_tolerance = FLOAT_REDUCTION_MAX_ULPS * max(
        float(left_ulp.detach().cpu()), float(right_ulp.detach().cpu())
    )
    return left, right, max(FLOAT_REDUCTION_ATOL_FLOOR[left.dtype], ulp_tolerance)


def validate_bounded_float_reduction(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    name: str,
) -> tuple[float, float]:
    """Validate one scalar floating reduction without changing either value.

    Structural identities, masks, addresses and inactive zeros remain exact.
    This helper is only for two finite scalar results whose floating reduction
    order may differ across CUDA kernels.
    """

    left, right, tolerance = _scalar_float_contract(
        observed, reconstructed, name=name
    )
    difference = float((left - right).abs().detach().cpu())
    if difference > tolerance:
        raise FloatReductionClosureError(
            f"{name} failed: abs={difference:.9g}, tolerance={tolerance:.9g}, "
            f"dtype={left.dtype}"
        )
    return difference, tolerance


def validate_conditioned_float_decomposition(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    absolute_mass: torch.Tensor | float,
    operation_count: int,
    name: str,
) -> tuple[float, float]:
    """Validate two algebraically equal, differently associated reductions.

    The bound is condition-aware: it scales with the L1 mass before cancellation
    and the registered number of floating operations.  It therefore does not
    confuse severe cancellation with a missing term, while remaining far tighter
    than an arbitrary relative tolerance.
    """

    left, right, ulp_floor = _scalar_float_contract(
        observed, reconstructed, name=name
    )
    if type(operation_count) is not int or operation_count <= 0:
        raise FloatReductionClosureError(f"{name} operation count is invalid")
    mass_tensor = torch.as_tensor(absolute_mass, dtype=torch.float64).detach().cpu()
    if mass_tensor.ndim != 0 or not bool(torch.isfinite(mass_tensor)) or bool(
        mass_tensor.lt(0)
    ):
        raise FloatReductionClosureError(f"{name} absolute mass is invalid")
    if float(mass_tensor) == 0.0:
        if not torch.equal(left, right):
            difference = float((left - right).abs().detach().cpu())
            raise FloatReductionClosureError(
                f"{name} failed exact zero-mass closure: abs={difference:.9g}"
            )
        return 0.0, 0.0
    epsilon = float(torch.finfo(left.dtype).eps)
    product = operation_count * epsilon
    if product >= 0.5:
        raise FloatReductionClosureError(
            f"{name} operation count exceeds stable error-bound range"
        )
    gamma = product / (1.0 - product)
    backward_bound = 4.0 * gamma * float(mass_tensor)
    tolerance = max(ulp_floor, backward_bound)
    difference = float((left - right).abs().detach().cpu())
    if difference > tolerance:
        raise FloatReductionClosureError(
            f"{name} failed: abs={difference:.9g}, tolerance={tolerance:.9g}, "
            f"dtype={left.dtype}, mass={float(mass_tensor):.9g}, "
            f"operations={operation_count}"
        )
    return difference, tolerance


def validate_float32_forward_equivalence(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    operation_count: int,
    safety_factor: float,
    scale_floor: float,
    name: str,
) -> tuple[float, float, float]:
    """Bound two float32 execution paths using a forward-error envelope."""

    left = torch.as_tensor(observed)
    right = torch.as_tensor(reconstructed)
    if (
        left.shape != right.shape
        or left.device != right.device
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
    ):
        raise FloatReductionClosureError(
            f"{name} requires same-shape finite float32 tensors"
        )
    if type(operation_count) is not int or operation_count <= 0:
        raise FloatReductionClosureError(f"{name} operation count is invalid")
    if not isinstance(safety_factor, (int, float)) or not math.isfinite(
        float(safety_factor)
    ) or float(safety_factor) <= 0:
        raise FloatReductionClosureError(f"{name} safety factor is invalid")
    if not isinstance(scale_floor, (int, float)) or not math.isfinite(
        float(scale_floor)
    ) or float(scale_floor) < 0:
        raise FloatReductionClosureError(f"{name} scale floor is invalid")
    epsilon = float(torch.finfo(torch.float32).eps)
    product = operation_count * epsilon
    if product >= 0.5:
        raise FloatReductionClosureError(
            f"{name} operation count exceeds stable error-bound range"
        )
    gamma = product / (1.0 - product)
    maximum_left = 0.0 if left.numel() == 0 else float(left.abs().max().detach().cpu())
    maximum_right = 0.0 if right.numel() == 0 else float(right.abs().max().detach().cpu())
    scale = max(float(scale_floor), maximum_left, maximum_right)
    tolerance = float(safety_factor) * gamma * scale
    difference = (
        0.0
        if left.numel() == 0
        else float((left - right).abs().max().detach().cpu())
    )
    if difference > tolerance:
        raise FloatReductionClosureError(
            f"{name} failed: abs={difference:.9g}, tolerance={tolerance:.9g}, "
            f"scale={scale:.9g}, operations={operation_count}, "
            f"safety={float(safety_factor):.9g}"
        )
    return difference, tolerance, scale


class SuperregionIneligibleError(SuperregionContractError):
    """A correctly bound hypothesis that cannot form a legal destination H1."""


def _shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise SuperregionContractError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _sha256_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SuperregionContractError(f"{name} must be a lowercase SHA256")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _mask_sha256(value: torch.Tensor, grid_shape: tuple[int, int]) -> str:
    shape = _shape(grid_shape, name="mask grid")
    mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    if mask.shape != (math.prod(shape),):
        raise SuperregionContractError("mask/grid shape mismatch")
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    digest.update(json.dumps(list(shape), separators=(",", ":")).encode("utf-8"))
    digest.update(mask.to(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalPatchGeometry:
    """Per-image patch geometry in the common EXIF-oriented image frame.

    ``cell_boxes_xyxy`` contains one half-open normalized oriented-image rectangle
    per patch.  Padding patches may carry any finite nonnegative rectangle but are
    excluded by ``valid_patch_mask``.  Natural materializers must derive these
    boxes from the frozen processor's exact resize/crop/pad receipt.
    """

    source_image_sha256: str
    source_key: str
    processor_config_sha256: str
    raw_size_hw: tuple[int, int]
    oriented_size_hw: tuple[int, int]
    exif_orientation: int
    raw_to_oriented_affine: torch.Tensor
    grid_shape: tuple[int, int]
    valid_patch_mask: torch.Tensor
    cell_boxes_xyxy: torch.Tensor
    coordinate_frame: str = COORDINATE_FRAME

    def __post_init__(self) -> None:
        image_sha = _sha256_text(self.source_image_sha256, name="source image hash")
        processor_sha = _sha256_text(
            self.processor_config_sha256, name="processor config hash"
        )
        if not isinstance(self.source_key, str) or not self.source_key:
            raise SuperregionContractError("source key must be a nonempty anonymous key")
        raw_shape = _shape(self.raw_size_hw, name="raw image size")
        oriented_shape = _shape(self.oriented_size_hw, name="oriented image size")
        if (
            isinstance(self.exif_orientation, bool)
            or not isinstance(self.exif_orientation, int)
            or self.exif_orientation not in range(1, 9)
        ):
            raise SuperregionContractError("EXIF orientation must be an integer in [1,8]")
        affine = torch.as_tensor(
            self.raw_to_oriented_affine, dtype=torch.float64
        ).detach().cpu().contiguous()
        if affine.shape != (3, 3) or not bool(torch.isfinite(affine).all()):
            raise SuperregionContractError("raw-to-oriented affine must be finite [3,3]")
        shape = _shape(self.grid_shape, name="patch grid")
        count = math.prod(shape)
        valid = torch.as_tensor(
            self.valid_patch_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        boxes = torch.as_tensor(
            self.cell_boxes_xyxy, dtype=torch.float64
        ).detach().cpu().contiguous()
        if valid.shape != (count,) or not bool(valid.any()):
            raise SuperregionContractError("valid patch mask must select at least one grid cell")
        if boxes.shape != (count, 4) or not bool(torch.isfinite(boxes).all()):
            raise SuperregionContractError("cell boxes must be finite [H*W,4]")
        left, top, right, bottom = boxes.unbind(dim=1)
        valid_boxes = boxes[valid]
        if (
            bool((valid_boxes < 0.0).any())
            or bool((valid_boxes > 1.0).any())
            or bool((right[valid] <= left[valid]).any())
            or bool((bottom[valid] <= top[valid]).any())
        ):
            raise SuperregionContractError(
                "valid patch boxes must have positive area inside normalized frame"
            )
        if self.coordinate_frame != COORDINATE_FRAME:
            raise SuperregionContractError("patch geometry coordinate-frame drift")
        object.__setattr__(self, "source_image_sha256", image_sha)
        object.__setattr__(self, "processor_config_sha256", processor_sha)
        object.__setattr__(self, "raw_size_hw", raw_shape)
        object.__setattr__(self, "oriented_size_hw", oriented_shape)
        object.__setattr__(self, "raw_to_oriented_affine", affine)
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "valid_patch_mask", valid)
        object.__setattr__(self, "cell_boxes_xyxy", boxes)

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "coordinate_frame": self.coordinate_frame,
            "source_image_sha256": self.source_image_sha256,
            "source_key": self.source_key,
            "processor_config_sha256": self.processor_config_sha256,
            "raw_size_hw": list(self.raw_size_hw),
            "oriented_size_hw": list(self.oriented_size_hw),
            "exif_orientation": self.exif_orientation,
            "raw_to_oriented_affine_sha256": _tensor_sha256(
                self.raw_to_oriented_affine
            ),
            "grid_shape": list(self.grid_shape),
            "valid_patch_mask_sha256": _tensor_sha256(self.valid_patch_mask),
            "cell_boxes_xyxy_sha256": _tensor_sha256(self.cell_boxes_xyxy),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def regular_grid_geometry(
    *,
    source_image_sha256: str,
    source_key: str,
    processor_config_sha256: str,
    grid_shape: tuple[int, int],
    valid_patch_mask: torch.Tensor | None = None,
    raw_size_hw: tuple[int, int] = (1, 1),
    oriented_size_hw: tuple[int, int] = (1, 1),
    exif_orientation: int = 1,
) -> CanonicalPatchGeometry:
    """Construct an exact no-crop regular-grid receipt for E0 fixtures only."""

    height, width = _shape(grid_shape, name="regular grid")
    rows = torch.arange(height, dtype=torch.float64)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.float64)[None, :].expand(height, width)
    boxes = torch.stack(
        (
            columns / width,
            rows / height,
            (columns + 1.0) / width,
            (rows + 1.0) / height,
        ),
        dim=-1,
    ).reshape(-1, 4)
    valid = (
        torch.ones(height * width, dtype=torch.bool)
        if valid_patch_mask is None
        else torch.as_tensor(valid_patch_mask, dtype=torch.bool)
    )
    return CanonicalPatchGeometry(
        source_image_sha256=source_image_sha256,
        source_key=source_key,
        processor_config_sha256=processor_config_sha256,
        raw_size_hw=raw_size_hw,
        oriented_size_hw=oriented_size_hw,
        exif_orientation=exif_orientation,
        raw_to_oriented_affine=torch.eye(3, dtype=torch.float64),
        grid_shape=(height, width),
        valid_patch_mask=valid,
        cell_boxes_xyxy=boxes,
    )


def rasterize_by_canonical_overlap(
    source_mask: torch.Tensor,
    source_geometry: CanonicalPatchGeometry,
    destination_geometry: CanonicalPatchGeometry,
) -> torch.Tensor:
    """Map patch *areas* through the common oriented-image coordinate frame.

    Every active source cell must overlap at least one valid destination cell.
    The function never fills holes or chooses a connected component; structural
    validation is a separate fail-closed step.
    """

    if source_geometry.source_image_sha256 != destination_geometry.source_image_sha256:
        raise SuperregionContractError("cross-backbone geometry source-image mismatch")
    if source_geometry.source_key != destination_geometry.source_key:
        raise SuperregionContractError("cross-backbone geometry source-key mismatch")
    if source_geometry.coordinate_frame != destination_geometry.coordinate_frame:
        raise SuperregionContractError("cross-backbone coordinate-frame mismatch")
    source = torch.as_tensor(source_mask, dtype=torch.bool).detach().cpu().contiguous()
    if source.shape != (math.prod(source_geometry.grid_shape),):
        raise SuperregionContractError("source footprint/grid mismatch")
    if bool((source & ~source_geometry.valid_patch_mask).any()):
        raise SuperregionContractError("source footprint includes processor-invalid cells")
    output = torch.zeros(
        math.prod(destination_geometry.grid_shape), dtype=torch.bool
    )
    active = torch.nonzero(source, as_tuple=False).flatten()
    if active.numel() == 0:
        return output
    source_boxes = source_geometry.cell_boxes_xyxy[active]
    destination_valid = torch.nonzero(
        destination_geometry.valid_patch_mask, as_tuple=False
    ).flatten()
    destination_boxes = destination_geometry.cell_boxes_xyxy[destination_valid]
    left = torch.maximum(source_boxes[:, None, 0], destination_boxes[None, :, 0])
    top = torch.maximum(source_boxes[:, None, 1], destination_boxes[None, :, 1])
    right = torch.minimum(source_boxes[:, None, 2], destination_boxes[None, :, 2])
    bottom = torch.minimum(source_boxes[:, None, 3], destination_boxes[None, :, 3])
    overlap = right.gt(left) & bottom.gt(top)
    if not bool(overlap.any(dim=1).all()):
        raise SuperregionIneligibleError(
            "an active source patch has no valid destination-area overlap"
        )
    output[destination_valid[overlap.any(dim=0)]] = True
    return output


def _region_is_legal(mask: torch.Tensor, grid_shape: tuple[int, int]) -> bool:
    shape = _shape(grid_shape, name="region grid")
    value = torch.as_tensor(mask, dtype=torch.bool).detach().cpu().contiguous()
    if value.shape != (math.prod(shape),) or int(value.sum()) < MIN_REGION_PATCHES:
        return False
    _, diagnostics = connected_components_4(value, shape)
    return bool(
        diagnostics.connected_valid
        and diagnostics.component_count == 1
        and diagnostics.has_2d_span
    )


def _dilate4(mask: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    height, width = _shape(grid_shape, name="dilation grid")
    value = torch.as_tensor(mask, dtype=torch.bool).reshape(height, width)
    output = value.clone()
    output[1:] |= value[:-1]
    output[:-1] |= value[1:]
    output[:, 1:] |= value[:, :-1]
    output[:, :-1] |= value[:, 1:]
    return output.reshape(-1)


def _touch_or_overlap(
    first: torch.Tensor, second: torch.Tensor, grid_shape: tuple[int, int]
) -> bool:
    return bool((_dilate4(first, grid_shape) & torch.as_tensor(second, dtype=torch.bool)).any())


@dataclass(frozen=True)
class SealedConnectedSuperregion:
    """One anonymous, candidate-conditioned paired P->V geometry payload."""

    candidate_key: str
    direction: str
    source_slots: tuple[int, ...]
    query_source_image_sha256: str
    reference_source_image_sha256: str
    query_geometry_sha256: str
    reference_geometry_sha256: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    joint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise SuperregionContractError("candidate key must be nonempty")
        if self.direction not in P_DIRECTIONS:
            raise SuperregionContractError("unknown P direction")
        slots = tuple(int(item) for item in self.source_slots)
        if not slots or tuple(sorted(set(slots))) != slots or not all(
            0 <= item < MAX_HYPOTHESES for item in slots
        ):
            raise SuperregionContractError("source slots must be sorted unique H1 indices")
        query_shape = _shape(self.query_grid_shape, name="query region grid")
        reference_shape = _shape(self.reference_grid_shape, name="reference region grid")
        query = torch.as_tensor(
            self.query_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        reference = torch.as_tensor(
            self.reference_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        if not _region_is_legal(query, query_shape) or not _region_is_legal(
            reference, reference_shape
        ):
            raise SuperregionContractError("sealed region must be a legal paired 4CC")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "candidate_key": self.candidate_key,
            "direction": self.direction,
            "source_slots": list(slots),
            "query_source_image_sha256": _sha256_text(
                self.query_source_image_sha256, name="query source hash"
            ),
            "reference_source_image_sha256": _sha256_text(
                self.reference_source_image_sha256, name="reference source hash"
            ),
            "query_geometry_sha256": _sha256_text(
                self.query_geometry_sha256, name="query geometry hash"
            ),
            "reference_geometry_sha256": _sha256_text(
                self.reference_geometry_sha256, name="reference geometry hash"
            ),
            "query_grid_shape": list(query_shape),
            "reference_grid_shape": list(reference_shape),
            "query_mask_sha256": _mask_sha256(query, query_shape),
            "reference_mask_sha256": _mask_sha256(reference, reference_shape),
        }
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.joint_sha256 != expected:
            raise SuperregionContractError("joint superregion hash drift")
        object.__setattr__(self, "source_slots", slots)
        object.__setattr__(self, "query_grid_shape", query_shape)
        object.__setattr__(self, "reference_grid_shape", reference_shape)
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "reference_mask", reference)

    @property
    def query_mask_sha256(self) -> str:
        return _mask_sha256(self.query_mask, self.query_grid_shape)

    @property
    def reference_mask_sha256(self) -> str:
        return _mask_sha256(self.reference_mask, self.reference_grid_shape)


def _make_region(
    *,
    candidate_key: str,
    direction: str,
    source_slots: tuple[int, ...],
    query_geometry: CanonicalPatchGeometry,
    reference_geometry: CanonicalPatchGeometry,
    query_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> SealedConnectedSuperregion:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_key": candidate_key,
        "direction": direction,
        "source_slots": list(source_slots),
        "query_source_image_sha256": query_geometry.source_image_sha256,
        "reference_source_image_sha256": reference_geometry.source_image_sha256,
        "query_geometry_sha256": query_geometry.sha256,
        "reference_geometry_sha256": reference_geometry.sha256,
        "query_grid_shape": list(query_geometry.grid_shape),
        "reference_grid_shape": list(reference_geometry.grid_shape),
        "query_mask_sha256": _mask_sha256(query_mask, query_geometry.grid_shape),
        "reference_mask_sha256": _mask_sha256(
            reference_mask, reference_geometry.grid_shape
        ),
    }
    joint_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SealedConnectedSuperregion(
        candidate_key=candidate_key,
        direction=direction,
        source_slots=source_slots,
        query_source_image_sha256=query_geometry.source_image_sha256,
        reference_source_image_sha256=reference_geometry.source_image_sha256,
        query_geometry_sha256=query_geometry.sha256,
        reference_geometry_sha256=reference_geometry.sha256,
        query_grid_shape=query_geometry.grid_shape,
        reference_grid_shape=reference_geometry.grid_shape,
        query_mask=query_mask,
        reference_mask=reference_mask,
        joint_sha256=joint_sha,
    )


@dataclass(frozen=True)
class CandidateSuperregionSeal:
    candidate_key: str
    direction: str
    regions: tuple[SealedConnectedSuperregion, ...]
    status: str

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if self.direction not in P_DIRECTIONS:
            raise SuperregionContractError("candidate seal direction drift")
        if any(
            item.candidate_key != self.candidate_key or item.direction != self.direction
            for item in regions
        ):
            raise SuperregionContractError("candidate seal contains a foreign region")
        if len({item.joint_sha256 for item in regions}) != len(regions):
            raise SuperregionContractError("candidate seal contains duplicate regions")
        expected = STATUS_READY if regions else STATUS_H0
        if self.status != expected:
            raise SuperregionContractError("candidate seal status/region mismatch")
        object.__setattr__(self, "regions", regions)


@dataclass(frozen=True)
class _DraftRegion:
    source_slots: tuple[int, ...]
    query_mask: torch.Tensor
    reference_mask: torch.Tensor


def _merge_touching_drafts(
    drafts: Sequence[_DraftRegion],
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[_DraftRegion, ...]:
    """Merge only components that touch in both query and reference grids."""

    items = tuple(drafts)
    if not items:
        return ()
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = root(first), root(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for first in range(len(items)):
        for second in range(first + 1, len(items)):
            if _touch_or_overlap(
                items[first].query_mask,
                items[second].query_mask,
                query_grid_shape,
            ) and _touch_or_overlap(
                items[first].reference_mask,
                items[second].reference_mask,
                reference_grid_shape,
            ):
                union(first, second)

    grouped: dict[int, list[int]] = {}
    for index in range(len(items)):
        grouped.setdefault(root(index), []).append(index)
    merged: list[_DraftRegion] = []
    for indices in grouped.values():
        query = torch.stack([items[index].query_mask for index in indices]).any(dim=0)
        reference = torch.stack(
            [items[index].reference_mask for index in indices]
        ).any(dim=0)
        # Pairwise touching in a graph component guarantees connected unions, but
        # retain an independent validator so a future change fails closed.
        if not _region_is_legal(query, query_grid_shape) or not _region_is_legal(
            reference, reference_grid_shape
        ):
            continue
        slots = tuple(
            sorted({slot for index in indices for slot in items[index].source_slots})
        )
        merged.append(_DraftRegion(slots, query, reference))
    merged.sort(
        key=lambda item: (
            item.source_slots,
            _mask_sha256(item.query_mask, query_grid_shape),
            _mask_sha256(item.reference_mask, reference_grid_shape),
        )
    )
    return tuple(merged)


def map_colnomic_hypotheses_to_dino_superregions(
    hypotheses: SealedMultiPatchHypotheses,
    *,
    candidate_key: str,
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
) -> CandidateSuperregionSeal:
    """Build the immutable, target-free ColNomic-P -> DINO-V region seal."""

    if hypotheses.direction not in P_DIRECTIONS:
        raise SuperregionContractError("proposal direction drift")
    if hypotheses.query_grid_shape != colnomic_query_geometry.grid_shape or (
        hypotheses.reference_grid_shape != colnomic_reference_geometry.grid_shape
    ):
        raise SuperregionContractError("ColNomic hypothesis/geometry grid mismatch")
    if (
        colnomic_query_geometry.source_image_sha256
        != dino_query_geometry.source_image_sha256
        or colnomic_query_geometry.source_key != dino_query_geometry.source_key
        or colnomic_reference_geometry.source_image_sha256
        != dino_reference_geometry.source_image_sha256
        or colnomic_reference_geometry.source_key
        != dino_reference_geometry.source_key
    ):
        raise SuperregionContractError("ColNomic/DINO source binding mismatch")

    drafts: list[_DraftRegion] = []
    for slot in torch.nonzero(hypotheses.legal, as_tuple=False).flatten().tolist():
        generated = build_seed_local_connected_footprints(
            hypotheses.support_query_indices[slot],
            hypotheses.support_reference_indices[slot],
            hypotheses.support_mask[slot],
            seed_query_index=int(hypotheses.seed_query_indices[slot]),
            seed_reference_index=int(hypotheses.seed_reference_indices[slot]),
            query_grid_shape=hypotheses.query_grid_shape,
            reference_grid_shape=hypotheses.reference_grid_shape,
            maximum_seed_radius=6,
            minimum_anchors=4,
            maximum_bbox_cells=64,
            maximum_aspect_ratio=4.0,
            padding=0,
            geometry_model=MODEL_AFFINE,
            maximum_reprojection_rmse=0.04,
            maximum_reprojection_error=0.08,
            reference_admissible_mask=colnomic_reference_geometry.valid_patch_mask,
        )
        if not generated.legal:
            continue
        source_query = generated.query_footprint & colnomic_query_geometry.valid_patch_mask
        source_reference = (
            generated.reference_footprint
            & colnomic_reference_geometry.valid_patch_mask
        )
        if not _region_is_legal(
            source_query, colnomic_query_geometry.grid_shape
        ) or not _region_is_legal(
            source_reference, colnomic_reference_geometry.grid_shape
        ):
            continue
        try:
            mapped_query = rasterize_by_canonical_overlap(
                source_query, colnomic_query_geometry, dino_query_geometry
            )
            mapped_reference = rasterize_by_canonical_overlap(
                source_reference,
                colnomic_reference_geometry,
                dino_reference_geometry,
            )
        except SuperregionIneligibleError:
            # Geometry/source drift remains a hard error.  Only a correctly
            # bound footprint cut away by destination validity becomes H0.
            continue
        if not _region_is_legal(
            mapped_query, dino_query_geometry.grid_shape
        ) or not _region_is_legal(
            mapped_reference, dino_reference_geometry.grid_shape
        ):
            continue
        drafts.append(_DraftRegion((int(slot),), mapped_query, mapped_reference))

    merged = _merge_touching_drafts(
        drafts,
        query_grid_shape=dino_query_geometry.grid_shape,
        reference_grid_shape=dino_reference_geometry.grid_shape,
    )
    seen: set[tuple[str, str]] = set()
    regions: list[SealedConnectedSuperregion] = []
    for item in merged:
        signature = (
            _mask_sha256(item.query_mask, dino_query_geometry.grid_shape),
            _mask_sha256(item.reference_mask, dino_reference_geometry.grid_shape),
        )
        if signature in seen:
            continue
        seen.add(signature)
        regions.append(
            _make_region(
                candidate_key=candidate_key,
                direction=hypotheses.direction,
                source_slots=item.source_slots,
                query_geometry=dino_query_geometry,
                reference_geometry=dino_reference_geometry,
                query_mask=item.query_mask,
                reference_mask=item.reference_mask,
            )
        )
    regions.sort(key=lambda item: bytes.fromhex(item.joint_sha256))
    return CandidateSuperregionSeal(
        candidate_key=candidate_key,
        direction=hypotheses.direction,
        regions=tuple(regions),
        status=STATUS_READY if regions else STATUS_H0,
    )


class _RegionalDecoder(Protocol):
    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **kwargs: Any,
    ) -> Any: ...


class _PairComparator(Protocol):
    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> PairEvidence: ...


@dataclass(frozen=True)
class RegionalRelationalField:
    candidate_key: str
    query_region_sha256: str
    reference_region_sha256: str
    query_grid_shape: tuple[int, int]
    query_mask: torch.Tensor
    relational: torch.Tensor

    def __post_init__(self) -> None:
        shape = _shape(self.query_grid_shape, name="regional field grid")
        mask = torch.as_tensor(self.query_mask, dtype=torch.bool).detach().contiguous()
        relational = torch.as_tensor(self.relational).contiguous()
        if (
            not isinstance(self.candidate_key, str)
            or not self.candidate_key
            or mask.shape != shape
            or relational.ndim != 2
            or relational.shape[0] != math.prod(shape)
            or not relational.is_floating_point()
            or not bool(torch.isfinite(relational).all())
        ):
            raise SuperregionContractError("regional relational-field schema drift")
        flattened = mask.flatten().to(relational.device)
        if bool(relational[~flattened].ne(0).any()):
            raise SuperregionContractError("regional field is nonzero outside sealed query mask")
        _sha256_text(self.query_region_sha256, name="query region hash")
        _sha256_text(self.reference_region_sha256, name="reference region hash")
        object.__setattr__(self, "query_grid_shape", shape)
        object.__setattr__(self, "query_mask", mask)
        object.__setattr__(self, "relational", relational)


def decode_candidate_in_superregion(
    decoder: _RegionalDecoder,
    query_layers: torch.Tensor,
    reference_layers: torch.Tensor,
    *,
    candidate_key: str,
    query_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_region_sha256: str,
    reference_region_sha256: str,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> RegionalRelationalField:
    """Run the existing decoder with both query and reference hard masks."""

    query_shape = _shape(query_grid_shape, name="decode query grid")
    reference_shape = _shape(reference_grid_shape, name="decode reference grid")
    qmask = torch.as_tensor(query_mask, dtype=torch.bool).reshape(query_shape)
    rmask = torch.as_tensor(reference_mask, dtype=torch.bool).reshape(reference_shape)
    if not _region_is_legal(qmask.flatten(), query_shape) or not _region_is_legal(
        rmask.flatten(), reference_shape
    ):
        raise SuperregionContractError("decoder received an illegal regional mask")
    evidence = decoder.decode_candidate(
        query_layers,
        reference_layers,
        qmask,
        rmask,
        query_shape,
        reference_shape,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    relational = torch.as_tensor(evidence.relational).clone()
    relational[~qmask.flatten().to(relational.device)] = 0
    return RegionalRelationalField(
        candidate_key=candidate_key,
        query_region_sha256=query_region_sha256,
        reference_region_sha256=reference_region_sha256,
        query_grid_shape=query_shape,
        query_mask=qmask,
        relational=relational,
    )


def compare_regional_fields(
    comparator: _PairComparator,
    left: RegionalRelationalField,
    right: RegionalRelationalField,
) -> PairEvidence:
    """Compare two candidates on one byte-identical sealed query footprint."""

    if (
        left.query_grid_shape != right.query_grid_shape
        or left.query_region_sha256 != right.query_region_sha256
        or not torch.equal(left.query_mask, right.query_mask)
    ):
        raise SuperregionContractError("regional pair does not share one query footprint")
    result = comparator.compare_relational(
        left.relational, right.relational, left.query_mask
    )
    outside = ~left.query_mask.flatten().to(result.contributions.device)
    if bool(result.contributions[outside].ne(0).any()):
        raise SuperregionContractError("pair comparator leaked outside-region contribution")
    expected = result.contributions.sum() / left.query_mask.sum().to(
        result.contributions.dtype
    )
    if not bool(torch.isfinite(result.contributions).all()):
        raise SuperregionContractError(
            "regional contribution-sum closure encountered nonfinite evidence"
        )
    try:
        validate_conditioned_float_decomposition(
            result.logit,
            expected,
            absolute_mass=(
                result.contributions.abs().to(torch.float64).sum()
                / left.query_mask.sum().to(torch.float64)
            ),
            operation_count=result.contributions.numel() + 2,
            name="regional contribution-sum closure",
        )
    except FloatReductionClosureError as error:
        raise SuperregionContractError(str(error)) from error
    return result


@dataclass(frozen=True)
class CrossfitRegionalPairDecision:
    logit: torch.Tensor
    status: str
    directional_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def __post_init__(self) -> None:
        value = torch.as_tensor(self.logit)
        terms = tuple(torch.as_tensor(item) for item in self.directional_logits)
        if value.ndim != 0 or len(terms) != 4 or any(item.ndim != 0 for item in terms):
            raise SuperregionContractError("regional pair decision schema drift")
        if self.status not in {STATUS_SINGLE, STATUS_H0, STATUS_AMBIGUOUS}:
            raise SuperregionContractError("unknown regional pair decision status")
        if self.status in {STATUS_H0, STATUS_AMBIGUOUS} and bool(value.ne(0)):
            raise SuperregionContractError("HOLD/H0 regional decision must be exact zero")


def reduce_crossfit_regional_pair(
    forward_by_direction: Sequence[PairEvidence | None],
    reverse_by_direction: Sequence[PairEvidence | None],
    *,
    ambiguity: bool = False,
) -> CrossfitRegionalPairDecision:
    """Fixed-denominator symmetric reducer for two checkerboard directions."""

    forward = tuple(forward_by_direction)
    reverse = tuple(reverse_by_direction)
    if len(forward) != 2 or len(reverse) != 2:
        raise SuperregionContractError("cross-fit reducer requires exactly two directions")
    exemplar = next(
        (item.logit for item in (*forward, *reverse) if item is not None), None
    )
    zero = torch.zeros((), dtype=torch.float32) if exemplar is None else exemplar.new_zeros(())
    terms = (
        zero if forward[0] is None else forward[0].logit,
        zero if reverse[0] is None else -reverse[0].logit,
        zero if forward[1] is None else forward[1].logit,
        zero if reverse[1] is None else -reverse[1].logit,
    )
    if ambiguity:
        return CrossfitRegionalPairDecision(
            logit=zero,
            status=STATUS_AMBIGUOUS,
            directional_logits=terms,
        )
    if all(item is None for item in (*forward, *reverse)):
        return CrossfitRegionalPairDecision(
            logit=zero,
            status=STATUS_H0,
            directional_logits=terms,
        )
    logit = sum(terms, start=zero) / 4.0
    return CrossfitRegionalPairDecision(
        logit=logit,
        status=STATUS_SINGLE,
        directional_logits=terms,
    )


def resolve_actionable_component_state(region_hashes: Sequence[str]) -> str:
    """Turn externally calibrated actionable components into HOLD/single state.

    Thresholding is deliberately outside this structural module.  Once a frozen
    calibrator declares components actionable, zero means H0, one means a single
    explanation, and more than one means ambiguity/HOLD.
    """

    hashes = tuple(_sha256_text(item, name="actionable region hash") for item in region_hashes)
    if len(set(hashes)) != len(hashes):
        raise SuperregionContractError("actionable component list contains duplicates")
    if not hashes:
        return STATUS_H0
    if len(hashes) == 1:
        return STATUS_SINGLE
    return STATUS_AMBIGUOUS


def reorder_candidate_seals(
    seals: Sequence[CandidateSuperregionSeal], order: torch.Tensor
) -> tuple[CandidateSuperregionSeal, ...]:
    items = tuple(seals)
    permutation = torch.as_tensor(order, dtype=torch.long).detach().cpu().contiguous()
    if (
        permutation.shape != (len(items),)
        or permutation.unique().numel() != len(items)
        or (len(items) and (int(permutation.min()) != 0 or int(permutation.max()) != len(items) - 1))
    ):
        raise SuperregionContractError("candidate order must be a complete permutation")
    return tuple(items[int(index)] for index in permutation.tolist())
