"""Canonical cross-backbone geometry receipts for ColNomic P and DINO V.

The frozen ColNomic gallery pipeline passes ``image.convert("RGB")`` to the
Qwen2-VL processor, so its spatial tokens live in the *decoded raw-pixel*
orientation.  DINO explicitly applies EXIF orientation before resize/pad.  This
module maps both grids to normalized pixel-boundary coordinates of the complete
EXIF-oriented image.  It performs no model forward and reads no target label.
"""

from __future__ import annotations

import math
from typing import Mapping, Any

import torch

from .dino_rcde_colnomic_superregion_v1 import (
    COORDINATE_FRAME,
    CanonicalPatchGeometry,
    SuperregionContractError,
)


SCHEMA_VERSION = "rc_colnomic_dino_canonical_geometry_v1"
COLNOMIC_PATCH_SIZE = 14
COLNOMIC_MERGE_SIZE = 2
COLNOMIC_FACTOR = COLNOMIC_PATCH_SIZE * COLNOMIC_MERGE_SIZE
COLNOMIC_MIN_PIXELS = 3136
COLNOMIC_MAX_PIXELS = 602112
DINO_PATCH_SIZE = 14


def exif_boundary_affine(orientation: int) -> torch.Tensor:
    """Return raw-normalized-boundary -> EXIF-oriented-boundary affine."""

    matrices = {
        1: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        2: ((-1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        3: ((-1.0, 0.0, 1.0), (0.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
        4: ((1.0, 0.0, 0.0), (0.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
        5: ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        6: ((0.0, -1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        7: ((0.0, -1.0, 1.0), (-1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        8: ((0.0, 1.0, 0.0), (-1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    }
    if isinstance(orientation, bool) or orientation not in matrices:
        raise SuperregionContractError("EXIF orientation must be in [1,8]")
    return torch.tensor(matrices[int(orientation)], dtype=torch.float64)


def expected_oriented_size(
    raw_size_hw: tuple[int, int], orientation: int
) -> tuple[int, int]:
    raw_h, raw_w = (int(raw_size_hw[0]), int(raw_size_hw[1]))
    if raw_h <= 0 or raw_w <= 0:
        raise SuperregionContractError("raw image size must be positive")
    exif_boundary_affine(orientation)
    return (raw_w, raw_h) if int(orientation) in {5, 6, 7, 8} else (raw_h, raw_w)


def qwen_smart_resize(
    height: int,
    width: int,
    *,
    factor: int = COLNOMIC_FACTOR,
    minimum_pixels: int = COLNOMIC_MIN_PIXELS,
    maximum_pixels: int = COLNOMIC_MAX_PIXELS,
) -> tuple[int, int]:
    """Frozen Qwen2-VL smart-resize rule used by the ColNomic processor."""

    if min(height, width, factor, minimum_pixels, maximum_pixels) <= 0:
        raise SuperregionContractError("invalid Qwen smart-resize arguments")
    if max(height, width) / min(height, width) > 200:
        raise SuperregionContractError("Qwen smart-resize aspect ratio exceeds 200")
    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > maximum_pixels:
        beta = math.sqrt((height * width) / maximum_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < minimum_pixels:
        beta = math.sqrt(minimum_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return int(resized_h), int(resized_w)


def _regular_boxes(grid_shape: tuple[int, int]) -> torch.Tensor:
    height, width = (int(grid_shape[0]), int(grid_shape[1]))
    if height <= 0 or width <= 0:
        raise SuperregionContractError("grid shape must be positive")
    rows = torch.arange(height, dtype=torch.float64)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.float64)[None, :].expand(height, width)
    return torch.stack(
        (
            columns / width,
            rows / height,
            (columns + 1.0) / width,
            (rows + 1.0) / height,
        ),
        dim=-1,
    ).reshape(-1, 4)


def transform_boxes_by_exif(
    boxes_xyxy: torch.Tensor, orientation: int
) -> torch.Tensor:
    boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not bool(torch.isfinite(boxes).all()):
        raise SuperregionContractError("EXIF box input must be finite [N,4]")
    left, top, right, bottom = boxes.unbind(dim=1)
    corners = torch.stack(
        (
            torch.stack((left, top), dim=1),
            torch.stack((right, top), dim=1),
            torch.stack((right, bottom), dim=1),
            torch.stack((left, bottom), dim=1),
        ),
        dim=1,
    )
    homogeneous = torch.cat(
        (corners, torch.ones((*corners.shape[:2], 1), dtype=torch.float64)), dim=2
    )
    mapped = homogeneous @ exif_boundary_affine(orientation).T
    xy = mapped[..., :2] / mapped[..., 2:3]
    output = torch.stack(
        (
            xy[..., 0].min(dim=1).values,
            xy[..., 1].min(dim=1).values,
            xy[..., 0].max(dim=1).values,
            xy[..., 1].max(dim=1).values,
        ),
        dim=1,
    )
    output = output.clamp(0.0, 1.0)
    if bool((output[:, 2] <= output[:, 0]).any()) or bool(
        (output[:, 3] <= output[:, 1]).any()
    ):
        raise SuperregionContractError("EXIF transform collapsed a patch rectangle")
    return output.contiguous()


def build_colnomic_canonical_geometry(
    *,
    source_image_sha256: str,
    source_key: str,
    processor_config_sha256: str,
    raw_size_hw: tuple[int, int],
    exif_orientation: int,
    merged_grid_shape: tuple[int, int],
) -> CanonicalPatchGeometry:
    """Build ColNomic patch boxes after exact processor-grid replay.

    ColNomic saw decoded RGB pixels without EXIF transpose.  Its merged grid tiles
    the complete smart-resized raw frame with no crop or padding.  The cell boxes
    are therefore first built in raw normalized coordinates, then transformed to
    the common EXIF-oriented frame.
    """

    raw_h, raw_w = (int(raw_size_hw[0]), int(raw_size_hw[1]))
    grid_h, grid_w = (int(merged_grid_shape[0]), int(merged_grid_shape[1]))
    expected_resize = qwen_smart_resize(raw_h, raw_w)
    observed_resize = (grid_h * COLNOMIC_FACTOR, grid_w * COLNOMIC_FACTOR)
    if observed_resize != expected_resize:
        raise SuperregionContractError(
            "ColNomic processor grid does not close to frozen Qwen smart-resize"
        )
    oriented_hw = expected_oriented_size((raw_h, raw_w), exif_orientation)
    raw_boxes = _regular_boxes((grid_h, grid_w))
    oriented_boxes = transform_boxes_by_exif(raw_boxes, exif_orientation)
    return CanonicalPatchGeometry(
        source_image_sha256=source_image_sha256,
        source_key=source_key,
        processor_config_sha256=processor_config_sha256,
        raw_size_hw=(raw_h, raw_w),
        oriented_size_hw=oriented_hw,
        exif_orientation=exif_orientation,
        raw_to_oriented_affine=exif_boundary_affine(exif_orientation),
        grid_shape=(grid_h, grid_w),
        valid_patch_mask=torch.ones(grid_h * grid_w, dtype=torch.bool),
        cell_boxes_xyxy=oriented_boxes,
        coordinate_frame=COORDINATE_FRAME,
    )


def build_dino_canonical_geometry_from_receipt(
    *,
    source_image_sha256: str,
    source_key: str,
    processor_config_sha256: str,
    exif_orientation: int,
    receipt: Mapping[str, Any],
    valid_patch_mask: torch.Tensor,
) -> CanonicalPatchGeometry:
    """Convert the frozen DINO full-frame receipt into canonical patch boxes."""

    required = {
        "decoded_hw_before_exif",
        "oriented_hw_after_exif",
        "resized_hw",
        "padded_hw",
        "grid_hw",
        "padding_tblr",
        "resize_rule",
        "validity_rule",
    }
    if not isinstance(receipt, Mapping) or not required.issubset(receipt):
        raise SuperregionContractError("DINO canonical geometry receipt is incomplete")
    raw_hw = tuple(map(int, receipt["decoded_hw_before_exif"]))
    oriented_hw = tuple(map(int, receipt["oriented_hw_after_exif"]))
    resized_hw = tuple(map(int, receipt["resized_hw"]))
    padded_hw = tuple(map(int, receipt["padded_hw"]))
    grid_hw = tuple(map(int, receipt["grid_hw"]))
    padding = tuple(map(int, receipt["padding_tblr"]))
    if any(len(item) != 2 or min(item) <= 0 for item in (raw_hw, oriented_hw, resized_hw, padded_hw, grid_hw)):
        raise SuperregionContractError("DINO geometry dimensions are invalid")
    if padded_hw[0] < resized_hw[0] or padded_hw[1] < resized_hw[1]:
        raise SuperregionContractError("DINO padded dimensions are smaller than resized image")
    if oriented_hw != expected_oriented_size(raw_hw, exif_orientation):
        raise SuperregionContractError("DINO EXIF-oriented dimensions do not close")
    if (
        padded_hw != (grid_hw[0] * DINO_PATCH_SIZE, grid_hw[1] * DINO_PATCH_SIZE)
        or padding
        != (
            0,
            padded_hw[0] - resized_hw[0],
            0,
            padded_hw[1] - resized_hw[1],
        )
        or receipt["resize_rule"]
        != "long_side_518_round_half_up_aspect_preserving_no_crop"
        or receipt["validity_rule"]
        != "complete_14x14_footprint_inside_resized_rectangle"
    ):
        raise SuperregionContractError("DINO resize/pad/validity semantics drift")
    valid = torch.as_tensor(valid_patch_mask, dtype=torch.bool).reshape(grid_hw)
    expected_valid = torch.zeros(grid_hw, dtype=torch.bool)
    expected_valid[
        : resized_hw[0] // DINO_PATCH_SIZE,
        : resized_hw[1] // DINO_PATCH_SIZE,
    ] = True
    if not torch.equal(valid.cpu(), expected_valid):
        raise SuperregionContractError("DINO valid mask does not match its receipt")
    rows = torch.arange(grid_hw[0], dtype=torch.float64)[:, None].expand(grid_hw)
    columns = torch.arange(grid_hw[1], dtype=torch.float64)[None, :].expand(grid_hw)
    boxes = torch.stack(
        (
            columns * DINO_PATCH_SIZE / resized_hw[1],
            rows * DINO_PATCH_SIZE / resized_hw[0],
            (columns + 1.0) * DINO_PATCH_SIZE / resized_hw[1],
            (rows + 1.0) * DINO_PATCH_SIZE / resized_hw[0],
        ),
        dim=-1,
    ).reshape(-1, 4).clamp(0.0, 1.0)
    return CanonicalPatchGeometry(
        source_image_sha256=source_image_sha256,
        source_key=source_key,
        processor_config_sha256=processor_config_sha256,
        raw_size_hw=raw_hw,
        oriented_size_hw=oriented_hw,
        exif_orientation=exif_orientation,
        raw_to_oriented_affine=exif_boundary_affine(exif_orientation),
        grid_shape=grid_hw,
        valid_patch_mask=valid.flatten(),
        cell_boxes_xyxy=boxes,
        coordinate_frame=COORDINATE_FRAME,
    )
