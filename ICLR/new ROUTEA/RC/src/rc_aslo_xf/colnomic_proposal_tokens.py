"""Spatial ColNomic proposal-token recovery without a second 7B forward.

The frozen raw-gallery cache stores every valid output token but omitted the
``input_ids`` image-token mask and ``image_grid_thw``.  Those two values are
processor outputs, not model outputs.  They can therefore be reconstructed
from the original image with the frozen processor and used to split the
already-cached embedding exactly.  This module never infers a grid merely by
factoring a token count and never assumes the observed ``[4:-7]`` span.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


COLNOMIC_LOCAL_DIM = 128
EXPECTED_MERGE_SIZE = 2
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})


class ColNomicProposalTokenError(ValueError):
    pass


@dataclass(frozen=True)
class ColNomicLocalGrid:
    tokens: torch.Tensor
    grid_shape: tuple[int, int]
    image_token_indices: torch.Tensor

    def __post_init__(self) -> None:
        tokens = torch.as_tensor(self.tokens).detach().cpu().contiguous()
        indices = torch.as_tensor(
            self.image_token_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        if (
            not isinstance(self.grid_shape, tuple)
            or len(self.grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.grid_shape
            )
            or tokens.ndim != 2
            or tuple(tokens.shape) != (self.grid_shape[0] * self.grid_shape[1], COLNOMIC_LOCAL_DIM)
            or not tokens.is_floating_point()
            or not bool(torch.isfinite(tokens).all())
            or indices.shape != (tokens.shape[0],)
            or indices.unique().numel() != indices.numel()
            or (indices.numel() and int(indices.min()) < 0)
        ):
            raise ColNomicProposalTokenError("invalid spatial ColNomic token grid")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "image_token_indices", indices)


def reconstruct_gallery_paths(root: Path) -> tuple[Path, ...]:
    """Reproduce the original sorted recursive raw-gallery traversal."""

    if not root.is_dir():
        raise ColNomicProposalTokenError("raw gallery root is missing")
    paths = tuple(
        sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ColNomicProposalTokenError("raw gallery contains no images")
    return paths


def validate_gallery_path_label_order(
    paths: Sequence[Path], legacy_setids: Sequence[str]
) -> None:
    labels = tuple(path.stem.strip() for path in paths)
    expected = tuple(map(str, legacy_setids))
    if len(paths) != len(expected) or labels != expected:
        raise ColNomicProposalTokenError(
            "raw-gallery path order does not reproduce frozen ColNomic setids"
        )


def split_cached_spatial_tokens(
    cached_valid_embedding: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    merge_size: int,
) -> ColNomicLocalGrid:
    """Recover image tokens by replaying the processor mask exactly."""

    cached = torch.as_tensor(cached_valid_embedding).detach().cpu().contiguous()
    ids = torch.as_tensor(input_ids).detach().cpu()
    attention = torch.as_tensor(attention_mask, dtype=torch.bool).detach().cpu()
    grid = torch.as_tensor(image_grid_thw, dtype=torch.long).detach().cpu()
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if attention.ndim == 2 and attention.shape[0] == 1:
        attention = attention[0]
    if grid.ndim == 2 and grid.shape[0] == 1:
        grid = grid[0]
    if (
        cached.ndim != 2
        or cached.shape[1] != COLNOMIC_LOCAL_DIM
        or not cached.is_floating_point()
        or not bool(torch.isfinite(cached).all())
        or ids.ndim != 1
        or attention.shape != ids.shape
        or grid.shape != (3,)
        or int(merge_size) != EXPECTED_MERGE_SIZE
    ):
        raise ColNomicProposalTokenError("ColNomic cache/processor schema drift")
    valid_positions = torch.nonzero(attention, as_tuple=False).flatten()
    if cached.shape[0] != valid_positions.numel():
        raise ColNomicProposalTokenError(
            "cached valid-token count disagrees with replayed attention mask"
        )
    image_in_full = torch.nonzero(
        attention & ids.eq(int(image_token_id)), as_tuple=False
    ).flatten()
    if not image_in_full.numel():
        raise ColNomicProposalTokenError("processor emitted no image tokens")
    # Convert full padded sequence positions to positions in the saved compact
    # valid-token sequence.  This remains correct if padding policy changes.
    full_to_valid = torch.full((ids.numel(),), -1, dtype=torch.long)
    full_to_valid[valid_positions] = torch.arange(valid_positions.numel())
    image_in_cached = full_to_valid[image_in_full]
    temporal, raw_h, raw_w = (int(value) for value in grid.tolist())
    if (
        temporal != 1
        or raw_h <= 0
        or raw_w <= 0
        or raw_h % merge_size
        or raw_w % merge_size
    ):
        raise ColNomicProposalTokenError("unsupported ColNomic image grid")
    grid_shape = (raw_h // merge_size, raw_w // merge_size)
    if image_in_cached.numel() != grid_shape[0] * grid_shape[1]:
        raise ColNomicProposalTokenError(
            "processor grid disagrees with replayed image-token mask"
        )
    return ColNomicLocalGrid(
        tokens=cached[image_in_cached],
        grid_shape=grid_shape,
        image_token_indices=image_in_cached,
    )


def proposal_grid_sha256(grid: ColNomicLocalGrid) -> str:
    value = grid.tokens.detach().cpu().contiguous()
    indices = grid.image_token_indices.detach().cpu().contiguous()
    header = {
        "dtype": str(value.dtype),
        "grid_shape": list(grid.grid_shape),
        "indices": indices.tolist(),
        "shape": list(value.shape),
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

