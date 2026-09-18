"""Target-free controls and receipts for natural-superregion S2.

The helpers in this module do not know labels, ranks, retrieval outcomes or
candidate identities.  They only construct deterministic permutations and a
shape-preserving representation of misbound reference content.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .geometry_hypothesis_v1 import fixed_macro_bank_split


SCHEMA_VERSION = "rc_cw1_sr0_natural_s2_controls_v1"
DESCRIPTOR_DIMENSION = 128


class NaturalS2ControlError(RuntimeError):
    pass


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _hash_order(indices: Sequence[int], *, namespace: str, key: str) -> list[int]:
    return sorted(
        (int(item) for item in indices),
        key=lambda item: hashlib.sha256(
            f"{namespace}\0{key}\0{item}".encode("utf-8")
        ).digest(),
    )


def deterministic_derangement(
    count: int, *, namespace: str, key: str
) -> torch.Tensor:
    """Return destination-to-source indices with no fixed point."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise NaturalS2ControlError("derangement population must contain at least two rows")
    order = _hash_order(range(count), namespace=namespace, key=key)
    output = torch.arange(count, dtype=torch.long)
    for ordinal, destination in enumerate(order):
        output[destination] = order[(ordinal + 1) % count]
    if (
        output.unique().numel() != count
        or bool(output.eq(torch.arange(count)).any())
    ):
        raise NaturalS2ControlError("deterministic derangement failed")
    return output.contiguous()


def role_preserving_query_derangement(
    grid_shape: tuple[int, int], *, namespace: str, key: str
) -> torch.Tensor:
    """Derange query contents without allowing rows to cross P/V banks."""

    if (
        not isinstance(grid_shape, tuple)
        or len(grid_shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in grid_shape)
        or min(grid_shape) <= 0
    ):
        raise NaturalS2ControlError("invalid query grid")
    count = math.prod(grid_shape)
    split = fixed_macro_bank_split(torch.ones(count, dtype=torch.bool), grid_shape)
    output = torch.arange(count, dtype=torch.long)
    for bank_name, mask in (("A", split.a), ("B", split.b)):
        members = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if len(members) < 2:
            raise NaturalS2ControlError("a fixed P/V bank cannot be deranged")
        order = _hash_order(
            members,
            namespace=f"{namespace}:{bank_name}",
            key=key,
        )
        for ordinal, destination in enumerate(order):
            output[destination] = order[(ordinal + 1) % len(order)]
    if (
        output.unique().numel() != count
        or bool(output.eq(torch.arange(count)).any())
        or not torch.equal(split.a, split.a[output])
        or not torch.equal(split.b, split.b[output])
    ):
        raise NaturalS2ControlError("role-preserving query derangement failed")
    return output.contiguous()


def resample_reference_content(
    tokens: torch.Tensor,
    source_grid_shape: tuple[int, int],
    destination_grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Resample foreign reference content to a P-seal's native grid.

    This is used only for V_C_BIND.  The P proposal stays bound to the
    destination reference; another candidate's token field supplies V content.
    """

    source = torch.as_tensor(tokens)
    if (
        source.shape != (math.prod(source_grid_shape), DESCRIPTOR_DIMENSION)
        or source.dtype not in {torch.float16, torch.float32, torch.float64}
        or not bool(torch.isfinite(source).all())
    ):
        raise NaturalS2ControlError("foreign reference token field is invalid")
    height, width = map(int, source_grid_shape)
    dst_height, dst_width = map(int, destination_grid_shape)
    if min(height, width, dst_height, dst_width) <= 0:
        raise NaturalS2ControlError("reference grids must be positive")
    field = source.to(torch.float32).reshape(height, width, DESCRIPTOR_DIMENSION)
    field = field.permute(2, 0, 1).unsqueeze(0)
    if (height, width) != (dst_height, dst_width):
        field = F.interpolate(
            field,
            size=(dst_height, dst_width),
            mode="bilinear",
            align_corners=False,
        )
    output = field[0].permute(1, 2, 0).reshape(-1, DESCRIPTOR_DIMENSION)
    return F.normalize(output, p=2, dim=1, eps=1.0e-12).contiguous()


def audit_positions(
    physical_rows: Sequence[int], *, count: int, namespace: str, key: str
) -> tuple[int, ...]:
    if count <= 0 or count > len(physical_rows):
        raise NaturalS2ControlError("invalid audit population")
    positions = sorted(
        range(len(physical_rows)),
        key=lambda position: hashlib.sha256(
            f"{namespace}\0{key}\0{int(physical_rows[position])}".encode("utf-8")
        ).digest(),
    )[:count]
    return tuple(positions)


__all__ = [
    "SCHEMA_VERSION",
    "NaturalS2ControlError",
    "tensor_sha256",
    "deterministic_derangement",
    "role_preserving_query_derangement",
    "resample_reference_content",
    "audit_positions",
]
