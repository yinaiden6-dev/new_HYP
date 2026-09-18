"""Explicit-input-frame ColNomic canonical geometry.

This additive successor leaves :mod:`colnomic_dino_canonical_geometry_v1`
unchanged.  V1 correctly replays ColNomic caches whose processor consumed the
decoded raw frame before EXIF orientation.  Some independently materialized
cache lineages instead applied EXIF orientation before Qwen smart-resize.  V2
therefore makes the processor input frame an explicit, caller-provenance-bound
argument; it never infers that frame from a token-grid shape.

Both modes return patch rectangles in the same complete EXIF-oriented,
normalized coordinate frame used by the connected-superregion bridge.  The
input-frame value is intentionally not added to ``CanonicalPatchGeometry``:
formal materializers must preserve it as a separate record field and include
it in their record hash.
"""

from __future__ import annotations

from typing import FrozenSet, Literal

import torch

from .colnomic_dino_canonical_geometry_v1 import (
    COLNOMIC_FACTOR,
    build_colnomic_canonical_geometry,
    exif_boundary_affine,
    expected_oriented_size,
    qwen_smart_resize,
)
from .dino_rcde_colnomic_superregion_v1 import (
    COORDINATE_FRAME,
    CanonicalPatchGeometry,
    SuperregionContractError,
)


SCHEMA_VERSION = "rc_colnomic_dino_canonical_geometry_v2"
DECODED_RAW_BEFORE_EXIF = "DECODED_RAW_BEFORE_EXIF"
EXIF_ORIENTED_BEFORE_RESIZE = "EXIF_ORIENTED_BEFORE_RESIZE"
ProcessorInputFrame = Literal[
    "DECODED_RAW_BEFORE_EXIF",
    "EXIF_ORIENTED_BEFORE_RESIZE",
]
PROCESSOR_INPUT_FRAMES: FrozenSet[str] = frozenset(
    {DECODED_RAW_BEFORE_EXIF, EXIF_ORIENTED_BEFORE_RESIZE}
)


def validate_processor_input_frame(value: object) -> ProcessorInputFrame:
    """Validate an explicit, provenance-derived ColNomic processor frame."""

    if not isinstance(value, str) or value not in PROCESSOR_INPUT_FRAMES:
        raise SuperregionContractError(
            "ColNomic processor input frame must be explicitly one of "
            f"{sorted(PROCESSOR_INPUT_FRAMES)}"
        )
    return value  # type: ignore[return-value]


def _regular_oriented_boxes(grid_shape: tuple[int, int]) -> torch.Tensor:
    grid_h, grid_w = (int(grid_shape[0]), int(grid_shape[1]))
    if grid_h <= 0 or grid_w <= 0:
        raise SuperregionContractError("grid shape must be positive")
    rows = torch.arange(grid_h, dtype=torch.float64)[:, None].expand(
        grid_h, grid_w
    )
    columns = torch.arange(grid_w, dtype=torch.float64)[None, :].expand(
        grid_h, grid_w
    )
    return torch.stack(
        (
            columns / grid_w,
            rows / grid_h,
            (columns + 1.0) / grid_w,
            (rows + 1.0) / grid_h,
        ),
        dim=-1,
    ).reshape(-1, 4).contiguous()


def build_colnomic_canonical_geometry_v2(
    *,
    source_image_sha256: str,
    source_key: str,
    processor_config_sha256: str,
    raw_size_hw: tuple[int, int],
    exif_orientation: int,
    merged_grid_shape: tuple[int, int],
    processor_input_frame: ProcessorInputFrame,
) -> CanonicalPatchGeometry:
    """Build ColNomic geometry under an explicit processor-input-frame contract.

    ``DECODED_RAW_BEFORE_EXIF`` delegates directly to the hash-bound V1
    implementation, preserving its numerical output exactly.

    ``EXIF_ORIENTED_BEFORE_RESIZE`` closes the observed grid against Qwen
    smart-resize of the EXIF-oriented dimensions.  Since that grid already
    tiles the oriented full image, its regular cells are canonical boxes
    directly; applying the EXIF affine to them again would be a double rotate.
    """

    frame = validate_processor_input_frame(processor_input_frame)
    if frame == DECODED_RAW_BEFORE_EXIF:
        return build_colnomic_canonical_geometry(
            source_image_sha256=source_image_sha256,
            source_key=source_key,
            processor_config_sha256=processor_config_sha256,
            raw_size_hw=raw_size_hw,
            exif_orientation=exif_orientation,
            merged_grid_shape=merged_grid_shape,
        )

    raw_h, raw_w = (int(raw_size_hw[0]), int(raw_size_hw[1]))
    grid_h, grid_w = (int(merged_grid_shape[0]), int(merged_grid_shape[1]))
    oriented_hw = expected_oriented_size((raw_h, raw_w), exif_orientation)
    expected_resize = qwen_smart_resize(*oriented_hw)
    observed_resize = (grid_h * COLNOMIC_FACTOR, grid_w * COLNOMIC_FACTOR)
    if observed_resize != expected_resize:
        raise SuperregionContractError(
            "ColNomic processor grid does not close to frozen Qwen smart-resize "
            f"for explicit input frame {frame}"
        )

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
        cell_boxes_xyxy=_regular_oriented_boxes((grid_h, grid_w)),
        coordinate_frame=COORDINATE_FRAME,
    )

