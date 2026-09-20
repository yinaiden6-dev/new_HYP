"""CW1 multi-tile target layer for the DINO-RCDE mainline.

This additive E0 core restores the two levels that must not be collapsed:

``fixed CW1 query tile -> candidate-bound local reference component``
``fixed connected union of several query tiles -> target hypothesis``

CW1 is enumerated on the ColNomic raster.  Every constituent tile/component is
then mapped independently into the DINO raster by canonical rectangle overlap.
The DINO overlap counts are recomputed after mapping.  Reference components stay
an ordered ``root -> component`` set; this module never replaces that binding by
one union mask.

The complete r1--r4 bank is retained for every candidate.  An ineligible row is
an explicit H0 row and is never filtered or renumbered.  The module also emits
the mandatory same-model arm family:

* ALL_PATCH_SAME_MODEL;
* CW1_QUERY_MULTITILE_FULL_REFERENCE; and
* CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET.

It contains no target/label join, training, D1/rank input, retrieval action, or
protected endpoint access.  It is an engineering primitive, not a scientific
GO/NO-GO result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import torch

from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .cw1_sr0_structure_v1 import (
    CW1SR0SuperRegion,
    enumerate_superregion_bank,
    superregion_bank_sha256,
    superregion_sha256,
)
from .dino_rcde_colnomic_superregion_v1 import (
    CanonicalPatchGeometry,
    P_DIRECTIONS,
    SuperregionIneligibleError,
    rasterize_by_canonical_overlap,
)
from .geometry_hypothesis_v1 import connected_components_4


SCHEMA_VERSION = "rc_dino_rcde_cw1_multitile_superregion_v2"

STATUS_READY = "CW1_MULTITILE_TARGET_READY"
STATUS_H0 = "H0_NO_CW1_MULTITILE_TARGET"
ROOT_READY = "ROOT_LOCAL_COMPONENT_READY"
ROOT_MISSING = "ROOT_LOCAL_COMPONENT_MISSING"

ARM_ALL_PATCH = "ALL_PATCH_SAME_MODEL"
ARM_QUERY_FULL_REFERENCE = "CW1_QUERY_MULTITILE_FULL_REFERENCE"
ARM_QUERY_LOCAL_COMPONENTS = "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET"
MANDATORY_ARMS = (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)

MIN_COMPONENT_PATCHES = 4
MIN_READY_ROOTS = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CW1MultiTileContractError(ValueError):
    """A structural, geometry, binding, or matched-arm contract failed."""


@lru_cache(maxsize=64)
def _cached_query_macro_seeds(
    grid_shape: tuple[int, int],
):
    """Immutable grid-only CW1 roots; never keyed by runtime/model state."""

    return enumerate_query_macro_seeds(grid_shape)


@lru_cache(maxsize=64)
def _cached_target_bank(grid_shape: tuple[int, int]):
    """Immutable complete r1--r4 structure bank for one exact query grid."""

    return enumerate_superregion_bank(grid_shape, include_r0_control=False)


def _fresh_target_bank(grid_shape: tuple[int, int]):
    """Clone cached structural rows so mutable tensors never escape the cache."""

    return copy.deepcopy(_cached_target_bank(grid_shape))


def _clear_structural_caches_for_tests() -> None:
    """Testing hook; production never adapts or clears structural caches."""

    _cached_query_macro_seeds.cache_clear()
    _cached_target_bank.cache_clear()
    _cached_component_contract.cache_clear()
    _cached_mapped_query_union.cache_clear()
    _cached_overlap_identity.cache_clear()


def _structural_cache_info_for_tests() -> dict[str, object]:
    return {
        "query_roots": _cached_query_macro_seeds.cache_info(),
        "target_bank": _cached_target_bank.cache_info(),
        "components": _cached_component_contract.cache_info(),
        "query_unions": _cached_mapped_query_union.cache_info(),
        "overlap": _cached_overlap_identity.cache_info(),
    }


def _mask_bytes(value: torch.Tensor) -> bytes:
    return (
        torch.as_tensor(value, dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
        .to(torch.uint8)
        .numpy()
        .tobytes()
    )


def _bool_from_bytes(value: bytes) -> torch.Tensor:
    return torch.tensor(tuple(value), dtype=torch.uint8).to(torch.bool)


@lru_cache(maxsize=16384)
def _cached_component_contract(
    grid_shape: tuple[int, int], mask_bytes: bytes
) -> bool:
    """Replay the full 4CC contract once per exact grid/membership pair."""

    value = _bool_from_bytes(mask_bytes)
    if value.shape != (math.prod(grid_shape),):
        return False
    _, diagnostic = connected_components_4(value, grid_shape)
    return bool(
        diagnostic.component_count == 1
        and diagnostic.connected_valid
        and diagnostic.has_2d_span
        and diagnostic.active_count >= MIN_COMPONENT_PATCHES
    )


@lru_cache(maxsize=4096)
def _cached_mapped_query_union(
    grid_shape: tuple[int, int], mask_bytes: tuple[bytes, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = tuple(_bool_from_bytes(item) for item in mask_bytes)
    stacked = torch.stack(items).to(torch.int64)
    coverage = stacked.sum(dim=0)
    union = coverage.gt(0)
    reciprocal = torch.zeros_like(coverage, dtype=torch.float64)
    reciprocal[union] = coverage[union].to(torch.float64).reciprocal()
    return union.contiguous(), coverage.contiguous(), reciprocal.contiguous()


@lru_cache(maxsize=4096)
def _cached_overlap_identity(
    masks: tuple[bytes, ...], coverage_bytes: bytes
) -> bool:
    items = tuple(_bool_from_bytes(item) for item in masks)
    coverage = torch.frombuffer(bytearray(coverage_bytes), dtype=torch.int64).clone()
    active = coverage.gt(0)
    total = Fraction(0, 1)
    for item in items:
        for count in coverage[item].tolist():
            total += Fraction(1, int(count))
    return total == Fraction(int(active.sum()), 1)


def _shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or min(value) <= 0
    ):
        raise CW1MultiTileContractError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _sha(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CW1MultiTileContractError(f"{name} must be a lowercase SHA256")
    return value


def _mask(
    value: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    name: str,
    allow_empty: bool = False,
) -> torch.Tensor:
    shape = _shape(grid_shape, name=f"{name} grid")
    result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    if result.shape != (math.prod(shape),):
        raise CW1MultiTileContractError(f"{name} shape/grid mismatch")
    if not allow_empty and not bool(result.any()):
        raise CW1MultiTileContractError(f"{name} must be nonempty")
    return result


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _payload_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _one_component(
    value: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    result = _mask(value, grid_shape, name=name)
    if not _cached_component_contract(grid_shape, _mask_bytes(result)):
        raise CW1MultiTileContractError(
            f"{name} must be one >=4-patch connected component with 2-D span"
        )
    return result


def _mapped_query_union(
    masks: Sequence[torch.Tensor],
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = tuple(_mask(item, grid_shape, name="mapped query tile") for item in masks)
    if not items:
        raise CW1MultiTileContractError("a target row needs mapped query tiles")
    union, coverage, reciprocal = _cached_mapped_query_union(
        grid_shape, tuple(_mask_bytes(item) for item in items)
    )
    # Cached tensors never escape: callers may retain output masks and torch
    # tensors are mutable even inside frozen dataclasses.
    return union.clone(), coverage.clone(), reciprocal.clone()


def _exact_mapped_overlap_identity(
    masks: Sequence[torch.Tensor], coverage: torch.Tensor
) -> bool:
    items = tuple(torch.as_tensor(item, dtype=torch.bool) for item in masks)
    count = torch.as_tensor(coverage, dtype=torch.int64).detach().cpu().contiguous()
    return _cached_overlap_identity(
        tuple(_mask_bytes(item) for item in items), count.numpy().tobytes()
    )


def _same_source(first: CanonicalPatchGeometry, second: CanonicalPatchGeometry) -> bool:
    return (
        first.source_image_sha256 == second.source_image_sha256
        and first.source_key == second.source_key
        and first.coordinate_frame == second.coordinate_frame
        and first.raw_size_hw == second.raw_size_hw
        and first.oriented_size_hw == second.oriented_size_hw
        and first.exif_orientation == second.exif_orientation
        and torch.equal(first.raw_to_oriented_affine, second.raw_to_oriented_affine)
    )


@dataclass(frozen=True)
class LocalRootBindingV2:
    """One fixed CW1 query root and one candidate-bound local component or H0."""

    candidate_key: str
    direction: str
    root_ordinal: int
    colnomic_query_grid_shape: tuple[int, int]
    colnomic_reference_grid_shape: tuple[int, int]
    dino_query_grid_shape: tuple[int, int]
    dino_reference_grid_shape: tuple[int, int]
    colnomic_query_tile_mask: torch.Tensor
    dino_query_tile_mask: torch.Tensor
    colnomic_reference_component_mask: torch.Tensor
    dino_reference_component_mask: torch.Tensor
    colnomic_query_geometry_sha256: str
    colnomic_reference_geometry_sha256: str
    dino_query_geometry_sha256: str
    dino_reference_geometry_sha256: str
    status: str
    reason: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise CW1MultiTileContractError("candidate key must be nonempty")
        if self.direction not in P_DIRECTIONS:
            raise CW1MultiTileContractError("proposal direction drift")
        cq_shape = _shape(self.colnomic_query_grid_shape, name="ColNomic query grid")
        cr_shape = _shape(
            self.colnomic_reference_grid_shape, name="ColNomic reference grid"
        )
        dq_shape = _shape(self.dino_query_grid_shape, name="DINO query grid")
        dr_shape = _shape(self.dino_reference_grid_shape, name="DINO reference grid")
        roots = _cached_query_macro_seeds(cq_shape)
        if (
            isinstance(self.root_ordinal, bool)
            or not isinstance(self.root_ordinal, int)
            or not 0 <= self.root_ordinal < len(roots)
        ):
            raise CW1MultiTileContractError("root ordinal is outside the frozen CW1 bank")
        cq = _mask(self.colnomic_query_tile_mask, cq_shape, name="ColNomic query tile")
        if not torch.equal(cq, roots[self.root_ordinal].window.mask):
            raise CW1MultiTileContractError("query tile is not the fixed CW1 root window")
        dq = _mask(
            self.dino_query_tile_mask,
            dq_shape,
            name="DINO query tile",
            allow_empty=self.status == ROOT_MISSING,
        )
        cr = _mask(
            self.colnomic_reference_component_mask,
            cr_shape,
            name="ColNomic reference component",
            allow_empty=self.status == ROOT_MISSING,
        )
        dr = _mask(
            self.dino_reference_component_mask,
            dr_shape,
            name="DINO reference component",
            allow_empty=self.status == ROOT_MISSING,
        )
        if self.status == ROOT_READY:
            _one_component(dq, dq_shape, name="mapped DINO query tile")
            _one_component(cr, cr_shape, name="ColNomic reference component")
            _one_component(dr, dr_shape, name="mapped DINO reference component")
            if self.reason != "READY":
                raise CW1MultiTileContractError("READY root must use the canonical reason")
        elif self.status == ROOT_MISSING:
            if bool(cr.any()) or bool(dr.any()) or not isinstance(self.reason, str) or not self.reason:
                raise CW1MultiTileContractError("missing root must have zero reference masks/reason")
        else:
            raise CW1MultiTileContractError("unknown root-binding status")
        geometry_hashes = (
            self.colnomic_query_geometry_sha256,
            self.colnomic_reference_geometry_sha256,
            self.dino_query_geometry_sha256,
            self.dino_reference_geometry_sha256,
        )
        for index, value in enumerate(geometry_hashes):
            _sha(value, name=f"geometry hash {index}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "candidate_key": self.candidate_key,
            "direction": self.direction,
            "root_ordinal": self.root_ordinal,
            "grid_shapes": [list(cq_shape), list(cr_shape), list(dq_shape), list(dr_shape)],
            "mask_sha256": [_tensor_sha(item) for item in (cq, dq, cr, dr)],
            "geometry_sha256": list(geometry_hashes),
            "status": self.status,
            "reason": self.reason,
        }
        if self.binding_sha256 != _payload_sha(payload):
            raise CW1MultiTileContractError("root binding hash drift")
        object.__setattr__(self, "colnomic_query_grid_shape", cq_shape)
        object.__setattr__(self, "colnomic_reference_grid_shape", cr_shape)
        object.__setattr__(self, "dino_query_grid_shape", dq_shape)
        object.__setattr__(self, "dino_reference_grid_shape", dr_shape)
        object.__setattr__(self, "colnomic_query_tile_mask", cq)
        object.__setattr__(self, "dino_query_tile_mask", dq)
        object.__setattr__(self, "colnomic_reference_component_mask", cr)
        object.__setattr__(self, "dino_reference_component_mask", dr)


def _binding_payload(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    cq_shape: tuple[int, int],
    cr_shape: tuple[int, int],
    dq_shape: tuple[int, int],
    dr_shape: tuple[int, int],
    cq: torch.Tensor,
    dq: torch.Tensor,
    cr: torch.Tensor,
    dr: torch.Tensor,
    geometries: tuple[CanonicalPatchGeometry, ...],
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_key": candidate_key,
        "direction": direction,
        "root_ordinal": root_ordinal,
        "grid_shapes": [list(cq_shape), list(cr_shape), list(dq_shape), list(dr_shape)],
        "mask_sha256": [_tensor_sha(item) for item in (cq, dq, cr, dr)],
        "geometry_sha256": [item.sha256 for item in geometries],
        "status": status,
        "reason": reason,
    }


def _make_root_binding(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    source_reference_component: torch.Tensor | None,
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
) -> LocalRootBindingV2:
    cq_shape = colnomic_query_geometry.grid_shape
    cr_shape = colnomic_reference_geometry.grid_shape
    dq_shape = dino_query_geometry.grid_shape
    dr_shape = dino_reference_geometry.grid_shape
    roots = _cached_query_macro_seeds(cq_shape)
    cq = roots[root_ordinal].window.mask.detach().cpu().contiguous().clone()
    zero_cr = torch.zeros(math.prod(cr_shape), dtype=torch.bool)
    zero_dr = torch.zeros(math.prod(dr_shape), dtype=torch.bool)
    try:
        if bool((cq & ~colnomic_query_geometry.valid_patch_mask).any()):
            raise SuperregionIneligibleError("CW1 query tile intersects invalid source patches")
        dq = rasterize_by_canonical_overlap(
            cq,
            source_geometry=colnomic_query_geometry,
            destination_geometry=dino_query_geometry,
        )
        _one_component(dq, dq_shape, name="mapped DINO query tile")
    except (SuperregionIneligibleError, CW1MultiTileContractError) as exc:
        dq = torch.zeros(math.prod(dq_shape), dtype=torch.bool)
        status, reason, cr, dr = ROOT_MISSING, f"QUERY_MAPPING_INELIGIBLE:{exc}", zero_cr, zero_dr
    else:
        if source_reference_component is None:
            status, reason, cr, dr = ROOT_MISSING, "NO_CANDIDATE_BOUND_COMPONENT", zero_cr, zero_dr
        else:
            try:
                cr = _one_component(
                    source_reference_component,
                    cr_shape,
                    name="ColNomic reference component",
                )
                if bool((cr & ~colnomic_reference_geometry.valid_patch_mask).any()):
                    raise SuperregionIneligibleError(
                        "reference component intersects invalid source patches"
                    )
                dr = rasterize_by_canonical_overlap(
                    cr,
                    source_geometry=colnomic_reference_geometry,
                    destination_geometry=dino_reference_geometry,
                )
                _one_component(dr, dr_shape, name="mapped DINO reference component")
            except (SuperregionIneligibleError, CW1MultiTileContractError) as exc:
                status, reason, cr, dr = ROOT_MISSING, f"REFERENCE_MAPPING_INELIGIBLE:{exc}", zero_cr, zero_dr
            else:
                status, reason = ROOT_READY, "READY"
    geometries = (
        colnomic_query_geometry,
        colnomic_reference_geometry,
        dino_query_geometry,
        dino_reference_geometry,
    )
    payload = _binding_payload(
        candidate_key=candidate_key,
        direction=direction,
        root_ordinal=root_ordinal,
        cq_shape=cq_shape,
        cr_shape=cr_shape,
        dq_shape=dq_shape,
        dr_shape=dr_shape,
        cq=cq,
        dq=dq,
        cr=cr,
        dr=dr,
        geometries=geometries,
        status=status,
        reason=reason,
    )
    return LocalRootBindingV2(
        candidate_key=candidate_key,
        direction=direction,
        root_ordinal=root_ordinal,
        colnomic_query_grid_shape=cq_shape,
        colnomic_reference_grid_shape=cr_shape,
        dino_query_grid_shape=dq_shape,
        dino_reference_grid_shape=dr_shape,
        colnomic_query_tile_mask=cq,
        dino_query_tile_mask=dq,
        colnomic_reference_component_mask=cr,
        dino_reference_component_mask=dr,
        colnomic_query_geometry_sha256=colnomic_query_geometry.sha256,
        colnomic_reference_geometry_sha256=colnomic_reference_geometry.sha256,
        dino_query_geometry_sha256=dino_query_geometry.sha256,
        dino_reference_geometry_sha256=dino_reference_geometry.sha256,
        status=status,
        reason=reason,
        binding_sha256=_payload_sha(payload),
    )


@dataclass(frozen=True)
class CW1MultiTileRowV2:
    bank_ordinal: int
    structural_region: CW1SR0SuperRegion
    root_bindings: tuple[LocalRootBindingV2, ...]
    dino_query_grid_shape: tuple[int, int]
    dino_query_union_mask: torch.Tensor
    dino_query_coverage_count: torch.Tensor
    dino_query_reciprocal_count: torch.Tensor
    status: str
    reason: str
    row_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.bank_ordinal, bool) or self.bank_ordinal < 0:
            raise CW1MultiTileContractError("bank ordinal must be non-negative")
        region = self.structural_region
        if not isinstance(region, CW1SR0SuperRegion) or not region.reportable_target:
            raise CW1MultiTileContractError("row must bind a reportable CW1 r1--r4 region")
        bindings = tuple(self.root_bindings)
        expected_roots = region.contributing_root_ordinals
        if tuple(item.root_ordinal for item in bindings) != expected_roots:
            raise CW1MultiTileContractError("row bindings drifted from fixed CW1 root order")
        if len({item.binding_sha256 for item in bindings}) != len(bindings):
            raise CW1MultiTileContractError("row contains duplicate root-binding receipts")
        dq_shape = _shape(self.dino_query_grid_shape, name="row DINO query grid")
        masks = tuple(item.dino_query_tile_mask for item in bindings)
        union = _mask(
            self.dino_query_union_mask,
            dq_shape,
            name="row DINO query union",
            allow_empty=True,
        )
        coverage = torch.as_tensor(
            self.dino_query_coverage_count, dtype=torch.int64
        ).detach().cpu().contiguous()
        reciprocal = torch.as_tensor(
            self.dino_query_reciprocal_count, dtype=torch.float64
        ).detach().cpu().contiguous()
        expected_union, expected_coverage, expected_reciprocal = _mapped_query_union(
            [item for item in masks if bool(item.any())], dq_shape
        ) if any(bool(item.any()) for item in masks) else (
            torch.zeros(math.prod(dq_shape), dtype=torch.bool),
            torch.zeros(math.prod(dq_shape), dtype=torch.int64),
            torch.zeros(math.prod(dq_shape), dtype=torch.float64),
        )
        if (
            not torch.equal(union, expected_union)
            or not torch.equal(coverage, expected_coverage)
            or not torch.equal(reciprocal, expected_reciprocal)
        ):
            raise CW1MultiTileContractError("mapped DINO query union/coverage drift")
        nonempty_masks = tuple(item for item in masks if bool(item.any()))
        if nonempty_masks and not _exact_mapped_overlap_identity(nonempty_masks, coverage):
            raise CW1MultiTileContractError("mapped DINO overlap identity failed")
        ready_roots = sum(item.status == ROOT_READY for item in bindings)
        mapped_target_legal = False
        if ready_roots >= MIN_READY_ROOTS and nonempty_masks:
            try:
                _one_component(union, dq_shape, name="mapped DINO query target")
                mapped_target_legal = int(union.sum()) > max(
                    int(item.sum()) for item in nonempty_masks
                )
            except CW1MultiTileContractError:
                mapped_target_legal = False
        expected_status = STATUS_READY if mapped_target_legal else STATUS_H0
        if self.status != expected_status:
            raise CW1MultiTileContractError("row READY/H0 state drift")
        if self.status == STATUS_READY:
            if self.reason != "READY":
                raise CW1MultiTileContractError("READY row must use canonical reason")
        elif not isinstance(self.reason, str) or not self.reason:
            raise CW1MultiTileContractError("H0 row must retain a rejection reason")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "bank_ordinal": self.bank_ordinal,
            "structural_region_sha256": superregion_sha256(region),
            "root_binding_sha256": [item.binding_sha256 for item in bindings],
            "dino_query_grid_shape": list(dq_shape),
            "dino_query_union_sha256": _tensor_sha(union),
            "dino_query_coverage_sha256": _tensor_sha(coverage),
            "dino_query_reciprocal_sha256": _tensor_sha(reciprocal),
            "status": self.status,
            "reason": self.reason,
        }
        if self.row_sha256 != _payload_sha(payload):
            raise CW1MultiTileContractError("row hash drift")
        object.__setattr__(self, "root_bindings", bindings)
        object.__setattr__(self, "dino_query_grid_shape", dq_shape)
        object.__setattr__(self, "dino_query_union_mask", union)
        object.__setattr__(self, "dino_query_coverage_count", coverage)
        object.__setattr__(self, "dino_query_reciprocal_count", reciprocal)


@dataclass(frozen=True)
class CW1MultiTilePopulationV2:
    candidate_key: str
    direction: str
    cw1_bank_sha256: str
    root_bindings: tuple[LocalRootBindingV2, ...]
    rows: tuple[CW1MultiTileRowV2, ...]
    population_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise CW1MultiTileContractError("population candidate key must be nonempty")
        if self.direction not in P_DIRECTIONS:
            raise CW1MultiTileContractError("population direction drift")
        _sha(self.cw1_bank_sha256, name="CW1 bank hash")
        bindings = tuple(self.root_bindings)
        rows = tuple(self.rows)
        if tuple(item.root_ordinal for item in bindings) != tuple(range(len(bindings))):
            raise CW1MultiTileContractError("population must seal every root exactly once")
        if tuple(item.bank_ordinal for item in rows) != tuple(range(len(rows))):
            raise CW1MultiTileContractError("population must retain every bank row/ordinal")
        if not rows:
            raise CW1MultiTileContractError("population cannot omit the CW1 bank")
        expected_root_count = len(
            _cached_query_macro_seeds(rows[0].structural_region.grid_shape)
        )
        if len(bindings) != expected_root_count:
            raise CW1MultiTileContractError(
                "population must seal every canonical CW1 root exactly once"
            )
        canonical_bank = _cached_target_bank(rows[0].structural_region.grid_shape)
        if (
            len(rows) != len(canonical_bank)
            or any(
                superregion_sha256(observed.structural_region)
                != superregion_sha256(expected)
                for observed, expected in zip(rows, canonical_bank, strict=True)
            )
            or superregion_bank_sha256(canonical_bank) != self.cw1_bank_sha256
        ):
            raise CW1MultiTileContractError("population rows drifted from the frozen CW1 bank")
        if any(
            item.candidate_key != self.candidate_key or item.direction != self.direction
            for item in bindings
        ):
            raise CW1MultiTileContractError("population contains a foreign root binding")
        for row in rows:
            if any(
                item.binding_sha256
                != bindings[item.root_ordinal].binding_sha256
                for item in row.root_bindings
            ):
                raise CW1MultiTileContractError(
                    "population row does not reference the canonical root ledger"
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "candidate_key": self.candidate_key,
            "direction": self.direction,
            "cw1_bank_sha256": self.cw1_bank_sha256,
            "root_binding_sha256": [item.binding_sha256 for item in bindings],
            "row_sha256": [item.row_sha256 for item in rows],
        }
        if self.population_sha256 != _payload_sha(payload):
            raise CW1MultiTileContractError("population hash drift")
        object.__setattr__(self, "root_bindings", bindings)
        object.__setattr__(self, "rows", rows)


def seal_cw1_multitile_population(
    *,
    candidate_key: str,
    direction: str,
    reference_component_by_root: Mapping[int, torch.Tensor | None],
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
) -> CW1MultiTilePopulationV2:
    """Seal a complete target-free r1--r4 population without selecting a row."""

    if direction not in P_DIRECTIONS:
        raise CW1MultiTileContractError("proposal direction drift")
    if not _same_source(colnomic_query_geometry, dino_query_geometry):
        raise CW1MultiTileContractError("ColNomic/DINO query source binding mismatch")
    if not _same_source(colnomic_reference_geometry, dino_reference_geometry):
        raise CW1MultiTileContractError("ColNomic/DINO reference source binding mismatch")
    roots = _cached_query_macro_seeds(colnomic_query_geometry.grid_shape)
    supplied = set(reference_component_by_root)
    expected = set(range(len(roots)))
    if supplied != expected:
        raise CW1MultiTileContractError(
            "component ledger must contain exactly one ready-or-missing entry per CW1 root"
        )
    bindings = tuple(
        _make_root_binding(
            candidate_key=candidate_key,
            direction=direction,
            root_ordinal=root,
            source_reference_component=reference_component_by_root[root],
            colnomic_query_geometry=colnomic_query_geometry,
            colnomic_reference_geometry=colnomic_reference_geometry,
            dino_query_geometry=dino_query_geometry,
            dino_reference_geometry=dino_reference_geometry,
        )
        for root in range(len(roots))
    )
    bank = _fresh_target_bank(colnomic_query_geometry.grid_shape)
    rows: list[CW1MultiTileRowV2] = []
    for ordinal, region in enumerate(bank):
        selected = tuple(bindings[root] for root in region.contributing_root_ordinals)
        nonempty_query = tuple(
            item.dino_query_tile_mask for item in selected if bool(item.dino_query_tile_mask.any())
        )
        if nonempty_query:
            union, coverage, reciprocal = _mapped_query_union(
                nonempty_query, dino_query_geometry.grid_shape
            )
        else:
            count = math.prod(dino_query_geometry.grid_shape)
            union = torch.zeros(count, dtype=torch.bool)
            coverage = torch.zeros(count, dtype=torch.int64)
            reciprocal = torch.zeros(count, dtype=torch.float64)
        ready_count = sum(item.status == ROOT_READY for item in selected)
        if ready_count < MIN_READY_ROOTS:
            status, reason = STATUS_H0, "FEWER_THAN_TWO_READY_ROOT_BINDINGS"
        else:
            try:
                _one_component(union, dino_query_geometry.grid_shape, name="mapped target")
                if int(union.sum()) <= max(int(item.sum()) for item in nonempty_query):
                    raise CW1MultiTileContractError("mapped target did not exceed local tile")
            except CW1MultiTileContractError as exc:
                status, reason = STATUS_H0, f"MAPPED_QUERY_TARGET_INELIGIBLE:{exc}"
            else:
                status, reason = STATUS_READY, "READY"
        row_payload = {
            "schema_version": SCHEMA_VERSION,
            "bank_ordinal": ordinal,
            "structural_region_sha256": superregion_sha256(region),
            "root_binding_sha256": [item.binding_sha256 for item in selected],
            "dino_query_grid_shape": list(dino_query_geometry.grid_shape),
            "dino_query_union_sha256": _tensor_sha(union),
            "dino_query_coverage_sha256": _tensor_sha(coverage),
            "dino_query_reciprocal_sha256": _tensor_sha(reciprocal),
            "status": status,
            "reason": reason,
        }
        rows.append(
            CW1MultiTileRowV2(
                bank_ordinal=ordinal,
                structural_region=region,
                root_bindings=selected,
                dino_query_grid_shape=dino_query_geometry.grid_shape,
                dino_query_union_mask=union,
                dino_query_coverage_count=coverage,
                dino_query_reciprocal_count=reciprocal,
                status=status,
                reason=reason,
                row_sha256=_payload_sha(row_payload),
            )
        )
    bank_hash = superregion_bank_sha256(bank)
    population_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_key": candidate_key,
        "direction": direction,
        "cw1_bank_sha256": bank_hash,
        "root_binding_sha256": [item.binding_sha256 for item in bindings],
        "row_sha256": [item.row_sha256 for item in rows],
    }
    return CW1MultiTilePopulationV2(
        candidate_key=candidate_key,
        direction=direction,
        cw1_bank_sha256=bank_hash,
        root_bindings=bindings,
        rows=tuple(rows),
        population_sha256=_payload_sha(population_payload),
    )


@dataclass(frozen=True)
class ArmScopeV2:
    name: str
    active: bool
    query_mask: torch.Tensor
    full_reference_mask: torch.Tensor
    root_ordinals: tuple[int, ...]
    query_tile_masks: tuple[torch.Tensor, ...]
    reference_masks_by_root: tuple[torch.Tensor, ...]
    scope_sha256: str

    def __post_init__(self) -> None:
        if self.name not in MANDATORY_ARMS:
            raise CW1MultiTileContractError("unknown matched arm")
        query = torch.as_tensor(self.query_mask, dtype=torch.bool).detach().cpu().contiguous()
        full_reference = torch.as_tensor(
            self.full_reference_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        roots = tuple(int(item) for item in self.root_ordinals)
        qtiles = tuple(torch.as_tensor(item, dtype=torch.bool).detach().cpu().contiguous() for item in self.query_tile_masks)
        refs = tuple(torch.as_tensor(item, dtype=torch.bool).detach().cpu().contiguous() for item in self.reference_masks_by_root)
        if len(roots) != len(qtiles) or len(roots) != len(refs):
            raise CW1MultiTileContractError("arm root/tile/reference arity drift")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "active": bool(self.active),
            "query_mask_sha256": _tensor_sha(query),
            "full_reference_mask_sha256": _tensor_sha(full_reference),
            "root_ordinals": list(roots),
            "query_tile_sha256": [_tensor_sha(item) for item in qtiles],
            "reference_mask_sha256": [_tensor_sha(item) for item in refs],
        }
        if self.scope_sha256 != _payload_sha(payload):
            raise CW1MultiTileContractError("arm-scope hash drift")
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "full_reference_mask", full_reference)
        object.__setattr__(self, "root_ordinals", roots)
        object.__setattr__(self, "query_tile_masks", qtiles)
        object.__setattr__(self, "reference_masks_by_root", refs)


@dataclass(frozen=True)
class ThreeArmFamilyV2:
    candidate_key: str
    direction: str
    bank_ordinal: int
    population_sha256: str
    model_checkpoint_sha256: str
    pair_comparator_sha256: str
    reducer_sha256: str
    arms: tuple[ArmScopeV2, ArmScopeV2, ArmScopeV2]
    family_sha256: str

    def __post_init__(self) -> None:
        _sha(self.population_sha256, name="population hash")
        for name, value in (
            ("checkpoint", self.model_checkpoint_sha256),
            ("pair comparator", self.pair_comparator_sha256),
            ("reducer", self.reducer_sha256),
        ):
            _sha(value, name=f"{name} hash")
        arms = tuple(self.arms)
        if tuple(item.name for item in arms) != MANDATORY_ARMS:
            raise CW1MultiTileContractError("all three mandatory arms must be present in order")
        all_patch, query_full, query_local = arms
        if not all_patch.active:
            raise CW1MultiTileContractError("geometry-valid all-patch control must remain active")
        if (
            query_full.active != query_local.active
            or not torch.equal(query_full.query_mask, query_local.query_mask)
            or query_full.root_ordinals != query_local.root_ordinals
            or len(query_full.query_tile_masks) != len(query_local.query_tile_masks)
            or any(
                not torch.equal(left, right)
                for left, right in zip(
                    query_full.query_tile_masks,
                    query_local.query_tile_masks,
                    strict=True,
                )
            )
        ):
            raise CW1MultiTileContractError("full/local arms do not share byte-identical Q")
        if (
            not bool(all_patch.full_reference_mask.any())
            or not torch.equal(all_patch.full_reference_mask, query_full.full_reference_mask)
            or bool(query_local.full_reference_mask.any())
        ):
            raise CW1MultiTileContractError(
                "all-patch/full-reference scopes or local-component separation drifted"
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "candidate_key": self.candidate_key,
            "direction": self.direction,
            "bank_ordinal": self.bank_ordinal,
            "population_sha256": self.population_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "pair_comparator_sha256": self.pair_comparator_sha256,
            "reducer_sha256": self.reducer_sha256,
            "arm_scope_sha256": [item.scope_sha256 for item in arms],
        }
        if self.family_sha256 != _payload_sha(payload):
            raise CW1MultiTileContractError("three-arm family hash drift")
        object.__setattr__(self, "arms", arms)


def _arm_scope(
    name: str,
    *,
    active: bool,
    query_mask: torch.Tensor,
    full_reference_mask: torch.Tensor,
    roots: tuple[int, ...],
    query_tiles: tuple[torch.Tensor, ...],
    reference_masks: tuple[torch.Tensor, ...],
) -> ArmScopeV2:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "active": bool(active),
        "query_mask_sha256": _tensor_sha(query_mask),
        "full_reference_mask_sha256": _tensor_sha(full_reference_mask),
        "root_ordinals": list(roots),
        "query_tile_sha256": [_tensor_sha(item) for item in query_tiles],
        "reference_mask_sha256": [_tensor_sha(item) for item in reference_masks],
    }
    return ArmScopeV2(
        name=name,
        active=active,
        query_mask=query_mask,
        full_reference_mask=full_reference_mask,
        root_ordinals=roots,
        query_tile_masks=query_tiles,
        reference_masks_by_root=reference_masks,
        scope_sha256=_payload_sha(payload),
    )


def build_three_arm_family(
    population: CW1MultiTilePopulationV2,
    *,
    bank_ordinal: int,
    dino_query_valid_patch_mask: torch.Tensor,
    dino_reference_valid_patch_mask: torch.Tensor,
    model_checkpoint_sha256: str,
    pair_comparator_sha256: str,
    reducer_sha256: str,
) -> ThreeArmFamilyV2:
    """Emit all three matched scopes from one immutable population row."""

    if not 0 <= bank_ordinal < len(population.rows):
        raise CW1MultiTileContractError("bank ordinal is outside the complete population")
    row = population.rows[bank_ordinal]
    q_valid = torch.as_tensor(
        dino_query_valid_patch_mask, dtype=torch.bool
    ).detach().cpu().contiguous()
    r_valid = torch.as_tensor(
        dino_reference_valid_patch_mask, dtype=torch.bool
    ).detach().cpu().contiguous()
    if q_valid.shape != row.dino_query_union_mask.shape or not bool(q_valid.any()):
        raise CW1MultiTileContractError("DINO query valid mask drift")
    if not bool(r_valid.any()):
        raise CW1MultiTileContractError("DINO reference valid mask must be nonempty")
    expected_reference_count = math.prod(
        population.root_bindings[0].dino_reference_grid_shape
    )
    if r_valid.shape != (expected_reference_count,):
        raise CW1MultiTileContractError("DINO reference valid mask/grid drift")
    roots = row.structural_region.contributing_root_ordinals
    qtiles = tuple(item.dino_query_tile_mask for item in row.root_bindings)
    full_refs = tuple(r_valid.clone() for _ in roots)
    local_refs = tuple(item.dino_reference_component_mask for item in row.root_bindings)
    active = row.status == STATUS_READY
    arms = (
        _arm_scope(
            ARM_ALL_PATCH,
            active=True,
            query_mask=q_valid,
            full_reference_mask=r_valid,
            roots=(),
            query_tiles=(),
            reference_masks=(),
        ),
        _arm_scope(
            ARM_QUERY_FULL_REFERENCE,
            active=active,
            query_mask=row.dino_query_union_mask,
            full_reference_mask=r_valid,
            roots=roots,
            query_tiles=qtiles,
            reference_masks=full_refs,
        ),
        _arm_scope(
            ARM_QUERY_LOCAL_COMPONENTS,
            active=active,
            query_mask=row.dino_query_union_mask,
            full_reference_mask=torch.zeros_like(r_valid),
            roots=roots,
            query_tiles=qtiles,
            reference_masks=local_refs,
        ),
    )
    family_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_key": population.candidate_key,
        "direction": population.direction,
        "bank_ordinal": bank_ordinal,
        "population_sha256": population.population_sha256,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "pair_comparator_sha256": pair_comparator_sha256,
        "reducer_sha256": reducer_sha256,
        "arm_scope_sha256": [item.scope_sha256 for item in arms],
    }
    return ThreeArmFamilyV2(
        candidate_key=population.candidate_key,
        direction=population.direction,
        bank_ordinal=bank_ordinal,
        population_sha256=population.population_sha256,
        model_checkpoint_sha256=model_checkpoint_sha256,
        pair_comparator_sha256=pair_comparator_sha256,
        reducer_sha256=reducer_sha256,
        arms=arms,
        family_sha256=_payload_sha(family_payload),
    )


def aggregate_mapped_root_contributions(
    row: CW1MultiTileRowV2,
    contribution_by_root: Mapping[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlap-correct root evidence on unique mapped DINO query patches.

    Missing roots are exact zero.  The denominator is the complete unique query
    union, never the number of available components.
    """

    allowed = set(row.structural_region.contributing_root_ordinals)
    if not set(contribution_by_root).issubset(allowed):
        raise CW1MultiTileContractError("contribution ledger contains a foreign root")
    exemplar = next(iter(contribution_by_root.values()), None)
    dtype = torch.float32 if exemplar is None else torch.as_tensor(exemplar).dtype
    device = torch.device("cpu") if exemplar is None else torch.as_tensor(exemplar).device
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise CW1MultiTileContractError("root contributions must use one floating dtype")
    total = torch.zeros(
        row.dino_query_union_mask.numel(), dtype=dtype, device=device
    )
    for binding in row.root_bindings:
        if binding.status != ROOT_READY:
            if binding.root_ordinal in contribution_by_root and bool(
                torch.as_tensor(contribution_by_root[binding.root_ordinal]).ne(0).any()
            ):
                raise CW1MultiTileContractError(
                    "missing root attempted to inject nonzero evidence"
                )
            continue
        if binding.root_ordinal not in contribution_by_root:
            continue
        value = torch.as_tensor(
            contribution_by_root[binding.root_ordinal], device=device
        )
        if (
            value.dtype != dtype
            or not value.is_floating_point()
            or value.shape != total.shape
            or not bool(torch.isfinite(value).all())
        ):
            raise CW1MultiTileContractError("root contribution must be finite [DINO query patches]")
        tile = binding.dino_query_tile_mask.to(device)
        if bool(value[~tile].ne(0).any()):
            raise CW1MultiTileContractError("root contribution leaked outside its mapped query tile")
        # Multiplication is required even after the forward-value assertion: it
        # also makes the reverse-mode derivative outside the tile exactly zero.
        total = total + value * tile.to(dtype)
    reciprocal = row.dino_query_reciprocal_count.to(device=device, dtype=dtype)
    combined = total * reciprocal
    outside = ~row.dino_query_union_mask.to(device)
    if bool(combined[outside].ne(0).any()):
        raise CW1MultiTileContractError("regional aggregation leaked outside Q")
    if row.status == STATUS_H0 or not bool(row.dino_query_union_mask.any()):
        return torch.zeros_like(combined), torch.zeros((), dtype=dtype, device=device)
    scalar = combined.sum() / row.dino_query_union_mask.sum().to(device=device, dtype=dtype)
    return combined, scalar


def derange_local_component_scope(
    scope: ArmScopeV2, permutation: Sequence[int]
) -> ArmScopeV2:
    """C_LOCAL_COMPONENT_BIND: keep component multiset, scramble root binding."""

    if scope.name != ARM_QUERY_LOCAL_COMPONENTS:
        raise CW1MultiTileContractError("component-binding destruction needs the local arm")
    order = tuple(int(item) for item in permutation)
    if tuple(sorted(order)) != tuple(range(len(scope.reference_masks_by_root))):
        raise CW1MultiTileContractError("component permutation must be complete")
    if len(order) < 2 or any(index == value for index, value in enumerate(order)):
        raise CW1MultiTileContractError(
            "binding destruction requires a fixed-point-free derangement"
        )
    return _arm_scope(
        scope.name,
        active=scope.active,
        query_mask=scope.query_mask,
        full_reference_mask=scope.full_reference_mask,
        roots=scope.root_ordinals,
        query_tiles=scope.query_tile_masks,
        reference_masks=tuple(scope.reference_masks_by_root[index] for index in order),
    )


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_READY",
    "STATUS_H0",
    "ROOT_READY",
    "ROOT_MISSING",
    "ARM_ALL_PATCH",
    "ARM_QUERY_FULL_REFERENCE",
    "ARM_QUERY_LOCAL_COMPONENTS",
    "MANDATORY_ARMS",
    "CW1MultiTileContractError",
    "LocalRootBindingV2",
    "CW1MultiTileRowV2",
    "CW1MultiTilePopulationV2",
    "ArmScopeV2",
    "ThreeArmFamilyV2",
    "seal_cw1_multitile_population",
    "build_three_arm_family",
    "aggregate_mapped_root_contributions",
    "derange_local_component_scope",
]
