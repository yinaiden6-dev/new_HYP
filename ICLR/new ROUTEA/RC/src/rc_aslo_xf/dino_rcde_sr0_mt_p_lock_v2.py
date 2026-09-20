"""Successor P-lock schema with reference-independent query structure.

This module is additive.  Its claim level is deliberately limited to
``CORE_E0_ONLY_NOT_NATURAL_P_LOCK``: natural address/cross-fit/bank provenance
must be supplied by a later full adapter and authority.  It deliberately does
not import, decode, or upgrade
the frozen V1 lock classes.  A V2 lock separates two facts that V1 collapsed:

* a canonical query root exists structurally; and
* that root has a legal component in a particular candidate reference.

Consequently ``ROOT_REFERENCE_MISSING`` retains its legal query mask while its
reference mask and relational evidence are exact zero.  The fixed-denominator
reducer counts that structural query mask in coverage but never renormalizes
over only the roots whose reference component is available.

The module contains no target, label, rank, winner, rival, D1 score, training
loss, or fallback.  It is only a schema and an exact aggregation primitive for
the canonical ``CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET`` arm.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_lock_v2_20260818"
CANONICAL_ARM = "CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET"
CLAIM_LEVEL = "CORE_E0_ONLY_NOT_NATURAL_P_LOCK"

LOCK_PROPOSAL_READY = "P_LOCK_PROPOSAL_READY"
LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE = "P_LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE"

ROOT_READY = "ROOT_READY"
ROOT_REFERENCE_MISSING = "ROOT_REFERENCE_MISSING"
ROOT_QUERY_UNMAPPABLE = "ROOT_QUERY_UNMAPPABLE"
ROOT_STATUSES = (
    ROOT_READY,
    ROOT_REFERENCE_MISSING,
    ROOT_QUERY_UNMAPPABLE,
)

RELATIONAL_ACTIVE = "RELATIONAL_ACTIVE"
RELATIONAL_EXACT_ZERO = "RELATIONAL_EXACT_ZERO"
MIN_READY_ROOTS = 2


class PLockV2ContractError(ValueError):
    """A fail-closed V2 P-lock contract violation."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PLockV2ContractError(f"{name} must be a lowercase SHA-256")
    return value


def _grid(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if (
        len(value) != 2
        or any(isinstance(item, bool) or int(item) != item or int(item) <= 0 for item in value)
    ):
        raise PLockV2ContractError(f"{name} must be two positive integers")
    return (int(value[0]), int(value[1]))


def _mask(
    value: torch.Tensor | Sequence[bool],
    shape: tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    if result.shape != (math.prod(shape),):
        raise PLockV2ContractError(f"{name} does not match its grid")
    return result.clone()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    payload = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes_hex": tensor.numpy().tobytes(order="C").hex(),
    }
    return canonical_sha256(payload)


def _border_touch(mask: torch.Tensor, shape: tuple[int, int]) -> bool:
    image = mask.reshape(shape)
    return bool(
        image[0].any()
        or image[-1].any()
        or image[:, 0].any()
        or image[:, -1].any()
    )


def _component_count(mask: torch.Tensor, shape: tuple[int, int]) -> int:
    coordinates = torch.nonzero(mask.reshape(shape), as_tuple=False).tolist()
    if not coordinates:
        return 0
    active_pairs = {(int(row), int(col)) for row, col in coordinates}
    visited: set[tuple[int, int]] = set()
    count = 0
    for seed in sorted(active_pairs):
        if seed in visited:
            continue
        count += 1
        stack = [seed]
        while stack:
            row, col = stack.pop()
            if (row, col) in visited:
                continue
            visited.add((row, col))
            for neighbour in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbour in active_pairs and neighbour not in visited:
                    stack.append(neighbour)
    return count


def _is_four_connected(mask: torch.Tensor, shape: tuple[int, int]) -> bool:
    return _component_count(mask, shape) == 1


@dataclass(frozen=True)
class MaskGeometryV2:
    """Canonical geometry receipt for one flattened patch mask."""

    mask_sha256: str
    area: int
    row_span: int
    col_span: int
    component_count: int
    border_touch: bool

    def __post_init__(self) -> None:
        _require_sha256(self.mask_sha256, name="mask geometry hash")
        for name, value in (
            ("area", self.area),
            ("row_span", self.row_span),
            ("col_span", self.col_span),
            ("component_count", self.component_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PLockV2ContractError(f"mask geometry {name} is invalid")
        if not isinstance(self.border_touch, bool):
            raise PLockV2ContractError("mask geometry border_touch is not boolean")
        if (self.area == 0) != (
            self.row_span == 0 and self.col_span == 0 and self.component_count == 0
        ):
            raise PLockV2ContractError("empty mask geometry span drift")
        if self.area > 0 and self.component_count <= 0:
            raise PLockV2ContractError("nonempty mask has no component")

    def as_payload(self) -> dict[str, object]:
        return {
            "mask_sha256": self.mask_sha256,
            "area": self.area,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "component_count": self.component_count,
            "border_touch": self.border_touch,
        }


def mask_geometry_v2(mask: torch.Tensor, shape: tuple[int, int]) -> MaskGeometryV2:
    mask = _mask(mask, shape, name="geometry mask")
    coordinates = torch.nonzero(mask.reshape(shape), as_tuple=False)
    if coordinates.numel() == 0:
        row_span = col_span = 0
    else:
        row_span = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
        col_span = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
    return MaskGeometryV2(
        mask_sha256=tensor_sha256(mask),
        area=int(mask.sum()),
        row_span=row_span,
        col_span=col_span,
        component_count=_component_count(mask, shape),
        border_touch=_border_touch(mask, shape) if bool(mask.any()) else False,
    )


def _component_is_legal(mask: torch.Tensor, shape: tuple[int, int]) -> bool:
    stats = mask_geometry_v2(mask, shape)
    return (
        stats.area >= 4
        and stats.row_span >= 2
        and stats.col_span >= 2
        and _is_four_connected(mask, shape)
    )


@dataclass(frozen=True)
class StructuralRootLockV2:
    """One canonical query root and its candidate-specific reference binding."""

    root_ordinal: int
    root_topology_sha256: str
    status: str
    state_reason: str
    action_key_sha256: str | None
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    relational_evidence_status: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_ordinal, bool)
            or not isinstance(self.root_ordinal, int)
            or self.root_ordinal < 0
        ):
            raise PLockV2ContractError("root ordinal is invalid")
        _require_sha256(self.root_topology_sha256, name="root topology")
        if not isinstance(self.state_reason, str) or not self.state_reason:
            raise PLockV2ContractError("root state reason is empty")
        query_grid = _grid(self.query_grid_shape, name="root query grid")
        reference_grid = _grid(self.reference_grid_shape, name="root reference grid")
        query = _mask(self.query_mask, query_grid, name="root query mask")
        reference = _mask(
            self.reference_mask, reference_grid, name="root reference mask"
        )
        if self.status == ROOT_READY:
            _require_sha256(self.action_key_sha256, name="READY action key")
            if self.relational_evidence_status != RELATIONAL_ACTIVE:
                raise PLockV2ContractError("READY root must enable relational evidence")
            if not _component_is_legal(query, query_grid) or not _component_is_legal(
                reference, reference_grid
            ):
                raise PLockV2ContractError(
                    "READY root requires legal query and reference components"
                )
        elif self.status == ROOT_REFERENCE_MISSING:
            if self.action_key_sha256 is not None:
                raise PLockV2ContractError("reference-missing root cannot expose an action")
            if self.relational_evidence_status != RELATIONAL_EXACT_ZERO:
                raise PLockV2ContractError(
                    "reference-missing root must seal exact-zero relational evidence"
                )
            if not _component_is_legal(query, query_grid) or bool(reference.any()):
                raise PLockV2ContractError(
                    "reference-missing root must retain a legal query mask and empty reference"
                )
        elif self.status == ROOT_QUERY_UNMAPPABLE:
            if self.action_key_sha256 is not None:
                raise PLockV2ContractError("query-unmappable root cannot expose an action")
            if self.relational_evidence_status != RELATIONAL_EXACT_ZERO:
                raise PLockV2ContractError(
                    "query-unmappable root must seal exact-zero relational evidence"
                )
            if bool(query.any()) or bool(reference.any()):
                raise PLockV2ContractError(
                    "query-unmappable root must have exact-empty masks"
                )
        else:
            raise PLockV2ContractError("unknown V2 root status")
        object.__setattr__(self, "query_grid_shape", query_grid)
        object.__setattr__(self, "reference_grid_shape", reference_grid)
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "reference_mask", reference)

    @property
    def query_geometry(self) -> MaskGeometryV2:
        return mask_geometry_v2(self.query_mask, self.query_grid_shape)

    @property
    def reference_geometry(self) -> MaskGeometryV2:
        return mask_geometry_v2(self.reference_mask, self.reference_grid_shape)

    @property
    def structurally_mapped(self) -> bool:
        return self.status in (ROOT_READY, ROOT_REFERENCE_MISSING)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "root_ordinal": self.root_ordinal,
            "root_topology_sha256": self.root_topology_sha256,
            "status": self.status,
            "state_reason": self.state_reason,
            "query_component": self.query_geometry.as_payload(),
            "reference_component": self.reference_geometry.as_payload(),
            "relational_evidence_status": self.relational_evidence_status,
        }


def selected_row_signature_v2(roots: Sequence[StructuralRootLockV2]) -> str:
    """Hash candidate-independent membership of the selected CW1 row.

    Candidate-specific binding status and reference-component geometry belong
    to the enclosing geometry signature, not to this row identity.  Thus the
    same selected structural row keeps the same signature when one candidate
    has a reference component and another records REFERENCE_MISSING.
    """

    ordered = tuple(sorted(roots, key=lambda item: item.root_ordinal))
    if len({item.root_ordinal for item in ordered}) != len(ordered):
        raise PLockV2ContractError("selected row contains duplicate root ordinals")
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "ordered_root_topology": [
                {
                    "root_ordinal": item.root_ordinal,
                    "root_topology_sha256": item.root_topology_sha256,
                    "query_component": item.query_geometry.as_payload(),
                }
                for item in ordered
            ],
        }
    )


@dataclass(frozen=True)
class PLockGeometrySignatureV2:
    """Complete geometry/provenance signature required by every V2 lock."""

    canonical_arm: str
    track: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_valid_geometry: MaskGeometryV2
    reference_valid_geometry: MaskGeometryV2
    ordered_root_topology: tuple[dict[str, object], ...]
    selected_root_ordinals: tuple[int, ...]
    lock_state: str
    selected_row_sha256: str | None
    candidate_row_signature_sha256: str
    p_checkpoint_sha256: str
    query_source_image_sha256: str
    reference_source_image_sha256: str
    query_geometry_sha256: str
    reference_geometry_sha256: str
    signature_sha256: str

    def __post_init__(self) -> None:
        if self.canonical_arm != CANONICAL_ARM:
            raise PLockV2ContractError("V2 lock canonical arm drift")
        if not isinstance(self.track, str) or not self.track:
            raise PLockV2ContractError("V2 lock track is empty")
        _grid(self.query_grid_shape, name="signature query grid")
        _grid(self.reference_grid_shape, name="signature reference grid")
        for name, digest in (
            ("candidate row", self.candidate_row_signature_sha256),
            ("P checkpoint", self.p_checkpoint_sha256),
            ("query source", self.query_source_image_sha256),
            ("reference source", self.reference_source_image_sha256),
            ("query geometry", self.query_geometry_sha256),
            ("reference geometry", self.reference_geometry_sha256),
            ("geometry signature", self.signature_sha256),
        ):
            _require_sha256(digest, name=name)
        if self.lock_state == LOCK_PROPOSAL_READY:
            _require_sha256(self.selected_row_sha256, name="selected row")
            if not self.selected_root_ordinals:
                raise PLockV2ContractError("READY signature has no selected roots")
        elif self.lock_state == LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE:
            if self.selected_row_sha256 is not None or self.selected_root_ordinals:
                raise PLockV2ContractError(
                    "structural-unavailable signature exposes a selected row"
                )
        else:
            raise PLockV2ContractError("geometry signature lock state drift")
        if self.signature_sha256 != canonical_sha256(self.payload(include_hash=False)):
            raise PLockV2ContractError("V2 lock geometry signature drift")

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "canonical_arm": self.canonical_arm,
            "track": self.track,
            "query_grid_shape": list(self.query_grid_shape),
            "reference_grid_shape": list(self.reference_grid_shape),
            "query_valid_geometry": self.query_valid_geometry.as_payload(),
            "reference_valid_geometry": self.reference_valid_geometry.as_payload(),
            "ordered_root_topology": list(self.ordered_root_topology),
            "selected_root_ordinals": list(self.selected_root_ordinals),
            "lock_state": self.lock_state,
            "selected_row_sha256": self.selected_row_sha256,
            "candidate_row_signature_sha256": self.candidate_row_signature_sha256,
            "p_checkpoint_sha256": self.p_checkpoint_sha256,
            "query_source_image_sha256": self.query_source_image_sha256,
            "reference_source_image_sha256": self.reference_source_image_sha256,
            "query_geometry_sha256": self.query_geometry_sha256,
            "reference_geometry_sha256": self.reference_geometry_sha256,
        }
        if include_hash:
            value["signature_sha256"] = self.signature_sha256
        return value


def _candidate_row_signature(
    *, candidate_key: str, candidate_physical_row: int, reference_source_sha256: str
) -> str:
    return canonical_sha256(
        {
            "candidate_key": candidate_key,
            "candidate_physical_row": candidate_physical_row,
            "reference_source_image_sha256": reference_source_sha256,
        }
    )


@dataclass(frozen=True)
class CandidatePLockV2:
    """One immutable candidate P lock under the successor schema."""

    candidate_key: str
    candidate_physical_row: int
    track: str
    query_valid_mask: torch.Tensor
    reference_valid_mask: torch.Tensor
    roots: tuple[StructuralRootLockV2, ...]
    selected_root_ordinals: tuple[int, ...]
    lock_state: str
    geometry_signature: PLockGeometrySignatureV2
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PLockV2ContractError(
                "V1/unknown P-lock consumption is forbidden; an explicit V2 lock is required"
            )
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise PLockV2ContractError("candidate key is empty")
        if (
            isinstance(self.candidate_physical_row, bool)
            or not isinstance(self.candidate_physical_row, int)
            or self.candidate_physical_row < 0
        ):
            raise PLockV2ContractError("candidate physical row is invalid")
        signature = self.geometry_signature
        query_valid = _mask(
            self.query_valid_mask, signature.query_grid_shape, name="query valid mask"
        )
        reference_valid = _mask(
            self.reference_valid_mask,
            signature.reference_grid_shape,
            name="reference valid mask",
        )
        roots = tuple(self.roots)
        if not roots:
            raise PLockV2ContractError("V2 P lock has no structural roots")
        ordinals = tuple(item.root_ordinal for item in roots)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise PLockV2ContractError("V2 P lock roots are not canonical and unique")
        selected = tuple(int(item) for item in self.selected_root_ordinals)
        if (
            selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or not set(selected).issubset(ordinals)
        ):
            raise PLockV2ContractError("selected root membership is not canonical")
        selected_roots = tuple(root for root in roots if root.root_ordinal in selected)
        if self.lock_state == LOCK_PROPOSAL_READY:
            if not selected_roots or any(
                root.status == ROOT_QUERY_UNMAPPABLE for root in selected_roots
            ):
                raise PLockV2ContractError(
                    "READY proposal needs selected mapped structural roots"
                )
        elif self.lock_state == LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE:
            if selected:
                raise PLockV2ContractError(
                    "structural-unavailable proposal cannot select roots"
                )
        else:
            raise PLockV2ContractError("unknown V2 lock state")
        for root in roots:
            if (
                root.query_grid_shape != signature.query_grid_shape
                or root.reference_grid_shape != signature.reference_grid_shape
            ):
                raise PLockV2ContractError("root/lock grid drift")
            if bool((root.query_mask & ~query_valid).any()) or bool(
                (root.reference_mask & ~reference_valid).any()
            ):
                raise PLockV2ContractError("root mask escapes its valid patch mask")
        expected_row = (
            selected_row_signature_v2(selected_roots)
            if self.lock_state == LOCK_PROPOSAL_READY
            else None
        )
        expected_candidate = _candidate_row_signature(
            candidate_key=self.candidate_key,
            candidate_physical_row=self.candidate_physical_row,
            reference_source_sha256=signature.reference_source_image_sha256,
        )
        if (
            self.track != signature.track
            or mask_geometry_v2(query_valid, signature.query_grid_shape)
            != signature.query_valid_geometry
            or mask_geometry_v2(reference_valid, signature.reference_grid_shape)
            != signature.reference_valid_geometry
            or tuple(item.geometry_payload() for item in roots)
            != signature.ordered_root_topology
            or selected != signature.selected_root_ordinals
            or self.lock_state != signature.lock_state
            or signature.selected_row_sha256 != expected_row
            or signature.candidate_row_signature_sha256 != expected_candidate
        ):
            raise PLockV2ContractError("V2 P-lock geometry/provenance closure drift")
        object.__setattr__(self, "query_valid_mask", query_valid)
        object.__setattr__(self, "reference_valid_mask", reference_valid)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "selected_root_ordinals", selected)

    @property
    def selected_roots(self) -> tuple[StructuralRootLockV2, ...]:
        membership = set(self.selected_root_ordinals)
        return tuple(root for root in self.roots if root.root_ordinal in membership)

    @property
    def query_union_mask(self) -> torch.Tensor:
        mapped = [
            root.query_mask for root in self.selected_roots if root.structurally_mapped
        ]
        if not mapped:
            return torch.zeros_like(self.query_valid_mask)
        return torch.stack(mapped).any(dim=0)

    @property
    def ready_root_count(self) -> int:
        return sum(root.status == ROOT_READY for root in self.selected_roots)

    @property
    def relationally_usable(self) -> bool:
        return self.ready_root_count >= MIN_READY_ROOTS

    @property
    def query_structure_sha256(self) -> str:
        """Candidate-independent receipt used to close a candidate population."""

        signature = self.geometry_signature
        return canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "canonical_arm": signature.canonical_arm,
                "track": signature.track,
                "query_source_image_sha256": signature.query_source_image_sha256,
                "query_geometry_sha256": signature.query_geometry_sha256,
                "query_grid_shape": list(signature.query_grid_shape),
                "query_valid_geometry": signature.query_valid_geometry.as_payload(),
                "ordered_query_root_topology": [
                    {
                        "root_ordinal": root.root_ordinal,
                        "root_topology_sha256": root.root_topology_sha256,
                        "query_component": root.query_geometry.as_payload(),
                    }
                    for root in self.roots
                ],
                "selected_row_sha256": signature.selected_row_sha256,
            }
        )


def make_candidate_p_lock_v2(
    *,
    candidate_key: str,
    candidate_physical_row: int,
    track: str,
    canonical_arm: str,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_valid_mask: torch.Tensor,
    reference_valid_mask: torch.Tensor,
    roots: Sequence[StructuralRootLockV2],
    selected_root_ordinals: Sequence[int],
    lock_state: str,
    p_checkpoint_sha256: str,
    query_source_image_sha256: str,
    reference_source_image_sha256: str,
    query_geometry_sha256: str,
    reference_geometry_sha256: str,
) -> CandidatePLockV2:
    """Canonicalize root order and seal a complete V2 geometry signature."""

    query_grid = _grid(query_grid_shape, name="lock query grid")
    reference_grid = _grid(reference_grid_shape, name="lock reference grid")
    query_valid = _mask(query_valid_mask, query_grid, name="query valid mask")
    reference_valid = _mask(
        reference_valid_mask, reference_grid, name="reference valid mask"
    )
    ordered = tuple(sorted(roots, key=lambda item: item.root_ordinal))
    selected = tuple(sorted(int(item) for item in selected_root_ordinals))
    membership = set(selected)
    selected_roots = tuple(
        root for root in ordered if root.root_ordinal in membership
    )
    selected_row_sha256 = (
        selected_row_signature_v2(selected_roots)
        if lock_state == LOCK_PROPOSAL_READY
        else None
    )
    candidate_row_sha256 = _candidate_row_signature(
        candidate_key=candidate_key,
        candidate_physical_row=candidate_physical_row,
        reference_source_sha256=reference_source_image_sha256,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "canonical_arm": canonical_arm,
        "track": track,
        "query_grid_shape": list(query_grid),
        "reference_grid_shape": list(reference_grid),
        "query_valid_geometry": mask_geometry_v2(query_valid, query_grid).as_payload(),
        "reference_valid_geometry": mask_geometry_v2(
            reference_valid, reference_grid
        ).as_payload(),
        "ordered_root_topology": [item.geometry_payload() for item in ordered],
        "selected_root_ordinals": list(selected),
        "lock_state": lock_state,
        "selected_row_sha256": selected_row_sha256,
        "candidate_row_signature_sha256": candidate_row_sha256,
        "p_checkpoint_sha256": p_checkpoint_sha256,
        "query_source_image_sha256": query_source_image_sha256,
        "reference_source_image_sha256": reference_source_image_sha256,
        "query_geometry_sha256": query_geometry_sha256,
        "reference_geometry_sha256": reference_geometry_sha256,
    }
    signature = PLockGeometrySignatureV2(
        canonical_arm=canonical_arm,
        track=track,
        query_grid_shape=query_grid,
        reference_grid_shape=reference_grid,
        query_valid_geometry=mask_geometry_v2(query_valid, query_grid),
        reference_valid_geometry=mask_geometry_v2(reference_valid, reference_grid),
        ordered_root_topology=tuple(item.geometry_payload() for item in ordered),
        selected_root_ordinals=selected,
        lock_state=lock_state,
        selected_row_sha256=selected_row_sha256,
        candidate_row_signature_sha256=candidate_row_sha256,
        p_checkpoint_sha256=p_checkpoint_sha256,
        query_source_image_sha256=query_source_image_sha256,
        reference_source_image_sha256=reference_source_image_sha256,
        query_geometry_sha256=query_geometry_sha256,
        reference_geometry_sha256=reference_geometry_sha256,
        signature_sha256=canonical_sha256(payload),
    )
    return CandidatePLockV2(
        candidate_key=candidate_key,
        candidate_physical_row=candidate_physical_row,
        track=track,
        query_valid_mask=query_valid,
        reference_valid_mask=reference_valid,
        roots=ordered,
        selected_root_ordinals=selected,
        lock_state=lock_state,
        geometry_signature=signature,
    )


@dataclass(frozen=True)
class FixedDenominatorAggregationV2:
    candidate_key: str
    candidate_physical_row: int
    query_union_mask: torch.Tensor
    structural_coverage: torch.Tensor
    patch_evidence: torch.Tensor
    root_contributions: Mapping[int, torch.Tensor]
    scalar_score: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "root_contributions", MappingProxyType(dict(self.root_contributions))
        )


def aggregate_fixed_denominator_v2(
    lock: CandidatePLockV2,
    relational_evidence_by_root: Mapping[int, torch.Tensor],
) -> FixedDenominatorAggregationV2:
    """Aggregate signed root evidence with a structural fixed denominator.

    Coverage counts READY and REFERENCE_MISSING query masks.  Only READY roots
    may contribute evidence.  Therefore a missing reference has exact-zero
    contribution and gradient while still reducing the share of overlapping
    READY evidence.  No denominator depends on how many references happen to
    be available.
    """

    if not isinstance(lock, CandidatePLockV2):
        raise PLockV2ContractError("fixed reducer requires a V2 P lock")
    evidence = dict(relational_evidence_by_root)
    expected = {root.root_ordinal for root in lock.roots}
    if set(evidence) != expected:
        raise PLockV2ContractError("relational evidence root population drift")
    coverage = torch.zeros_like(lock.query_valid_mask, dtype=torch.int64)
    for root in lock.selected_roots:
        if root.structurally_mapped:
            coverage = coverage + root.query_mask.to(torch.int64)
    union = coverage > 0
    first = next(iter(evidence.values()))
    exemplar = torch.as_tensor(first)
    if exemplar.ndim != 1 or exemplar.numel() != lock.query_valid_mask.numel():
        raise PLockV2ContractError("relational evidence/query grid drift")
    total = exemplar.new_zeros(exemplar.shape)
    contributions: dict[int, torch.Tensor] = {}
    reciprocal = exemplar.new_zeros(exemplar.shape)
    covered = union.to(device=exemplar.device)
    reciprocal[covered] = coverage[union].to(
        device=exemplar.device, dtype=exemplar.dtype
    ).reciprocal()
    selected_membership = set(lock.selected_root_ordinals)
    for root in lock.roots:
        value = torch.as_tensor(evidence[root.root_ordinal])
        if (
            value.shape != exemplar.shape
            or value.device != exemplar.device
            or value.dtype != exemplar.dtype
            or not bool(torch.isfinite(value.detach()).all())
        ):
            raise PLockV2ContractError("relational evidence tensor drift")
        selected = root.root_ordinal in selected_membership
        active = selected and root.status == ROOT_READY
        if (root.status != ROOT_READY) and bool(value.detach().ne(0).any()):
            raise PLockV2ContractError(
                "missing/unmappable root relational evidence is not exact zero"
            )
        query_mask = root.query_mask.to(device=value.device, dtype=value.dtype)
        if not selected:
            query_mask = torch.zeros_like(query_mask)
        active_gate = value.new_tensor(1.0 if active else 0.0)
        contribution = value * query_mask * reciprocal * active_gate
        contributions[root.root_ordinal] = contribution
        total = total + contribution
    union_on_device = union.to(device=total.device)
    patch = total * union_on_device.to(total.dtype)
    if bool(patch.detach()[~union_on_device].ne(0).any()):
        raise PLockV2ContractError("fixed reducer leaked evidence outside query union")
    if bool(union.any()):
        scalar = patch.sum() / union.sum().to(device=patch.device, dtype=patch.dtype)
    else:
        scalar = patch.sum() * 0.0
    return FixedDenominatorAggregationV2(
        candidate_key=lock.candidate_key,
        candidate_physical_row=lock.candidate_physical_row,
        query_union_mask=union,
        structural_coverage=coverage,
        patch_evidence=patch,
        root_contributions=contributions,
        scalar_score=scalar,
    )


def aggregate_candidate_batch_v2(
    items: Sequence[tuple[CandidatePLockV2, Mapping[int, torch.Tensor]]],
) -> tuple[FixedDenominatorAggregationV2, ...]:
    """Aggregate a candidate population in canonical physical-row order."""

    keys: set[str] = set()
    rows: set[int] = set()
    ordered = sorted(
        items,
        key=lambda item: (
            item[0].candidate_physical_row,
            item[0].geometry_signature.reference_source_image_sha256,
            item[0].candidate_key,
        ),
    )
    outputs: list[FixedDenominatorAggregationV2] = []
    query_structure_sha256: str | None = None
    for lock, evidence in ordered:
        if lock.candidate_key in keys or lock.candidate_physical_row in rows:
            raise PLockV2ContractError("candidate axis is not one-to-one")
        if query_structure_sha256 is None:
            query_structure_sha256 = lock.query_structure_sha256
        elif lock.query_structure_sha256 != query_structure_sha256:
            raise PLockV2ContractError(
                "candidate population disagrees on candidate-independent query structure"
            )
        keys.add(lock.candidate_key)
        rows.add(lock.candidate_physical_row)
        outputs.append(aggregate_fixed_denominator_v2(lock, evidence))
    return tuple(outputs)


_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "claim_level",
        "canonical_arm",
        "candidate_key",
        "candidate_physical_row",
        "track",
        "query_valid_mask",
        "reference_valid_mask",
        "roots",
        "selected_root_ordinals",
        "lock_state",
        "geometry_signature",
        "record_sha256",
    }
)
_ROOT_RECORD_FIELDS = frozenset(
    {
        "root_ordinal",
        "root_topology_sha256",
        "status",
        "state_reason",
        "action_key_sha256",
        "query_mask",
        "reference_mask",
        "query_grid_shape",
        "reference_grid_shape",
        "relational_evidence_status",
    }
)
_SIGNATURE_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_arm",
        "track",
        "query_grid_shape",
        "reference_grid_shape",
        "query_valid_geometry",
        "reference_valid_geometry",
        "ordered_root_topology",
        "selected_root_ordinals",
        "lock_state",
        "selected_row_sha256",
        "candidate_row_signature_sha256",
        "p_checkpoint_sha256",
        "query_source_image_sha256",
        "reference_source_image_sha256",
        "query_geometry_sha256",
        "reference_geometry_sha256",
        "signature_sha256",
    }
)
_MASK_GEOMETRY_FIELDS = frozenset(
    {
        "mask_sha256",
        "area",
        "row_span",
        "col_span",
        "component_count",
        "border_touch",
    }
)
_TOPOLOGY_FIELDS = frozenset(
    {
        "root_ordinal",
        "root_topology_sha256",
        "status",
        "state_reason",
        "query_component",
        "reference_component",
        "relational_evidence_status",
    }
)


def _exact_fields(value: object, expected: frozenset[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PLockV2ContractError(f"{name} field set drift")
    return value


def _bool_list(value: object, *, name: str) -> list[bool]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise PLockV2ContractError(f"{name} must be an exact boolean list")
    return value


def _mask_geometry_from_record(value: object, *, name: str) -> MaskGeometryV2:
    item = _exact_fields(value, _MASK_GEOMETRY_FIELDS, name=name)
    return MaskGeometryV2(
        mask_sha256=item["mask_sha256"],
        area=item["area"],
        row_span=item["row_span"],
        col_span=item["col_span"],
        component_count=item["component_count"],
        border_touch=item["border_touch"],
    )


def _topology_from_record(value: object, *, name: str) -> dict[str, object]:
    item = _exact_fields(value, _TOPOLOGY_FIELDS, name=name)
    query = _mask_geometry_from_record(
        item["query_component"], name=f"{name}.query_component"
    )
    reference = _mask_geometry_from_record(
        item["reference_component"], name=f"{name}.reference_component"
    )
    if (
        isinstance(item["root_ordinal"], bool)
        or not isinstance(item["root_ordinal"], int)
        or item["root_ordinal"] < 0
    ):
        raise PLockV2ContractError(f"{name} root ordinal is invalid")
    _require_sha256(item["root_topology_sha256"], name=f"{name} topology")
    if item["status"] not in ROOT_STATUSES:
        raise PLockV2ContractError(f"{name} status drift")
    if item["relational_evidence_status"] not in (
        RELATIONAL_ACTIVE,
        RELATIONAL_EXACT_ZERO,
    ):
        raise PLockV2ContractError(f"{name} evidence status drift")
    return {
        "root_ordinal": item["root_ordinal"],
        "root_topology_sha256": item["root_topology_sha256"],
        "status": item["status"],
        "state_reason": item["state_reason"],
        "query_component": query.as_payload(),
        "reference_component": reference.as_payload(),
        "relational_evidence_status": item["relational_evidence_status"],
    }


def candidate_p_lock_v2_to_record(lock: CandidatePLockV2) -> dict[str, object]:
    """Serialize a V2 lock with an exact, self-hashed JSON field schema."""

    if not isinstance(lock, CandidatePLockV2):
        raise PLockV2ContractError("V2 serializer received a non-V2 lock")
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_level": CLAIM_LEVEL,
        "canonical_arm": lock.geometry_signature.canonical_arm,
        "candidate_key": lock.candidate_key,
        "candidate_physical_row": lock.candidate_physical_row,
        "track": lock.track,
        "query_valid_mask": lock.query_valid_mask.tolist(),
        "reference_valid_mask": lock.reference_valid_mask.tolist(),
        "roots": [
            {
                "root_ordinal": root.root_ordinal,
                "root_topology_sha256": root.root_topology_sha256,
                "status": root.status,
                "state_reason": root.state_reason,
                "action_key_sha256": root.action_key_sha256,
                "query_mask": root.query_mask.tolist(),
                "reference_mask": root.reference_mask.tolist(),
                "query_grid_shape": list(root.query_grid_shape),
                "reference_grid_shape": list(root.reference_grid_shape),
                "relational_evidence_status": root.relational_evidence_status,
            }
            for root in lock.roots
        ],
        "selected_root_ordinals": list(lock.selected_root_ordinals),
        "lock_state": lock.lock_state,
        "geometry_signature": lock.geometry_signature.payload(),
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def candidate_p_lock_v2_from_record(record: Mapping[str, object]) -> CandidatePLockV2:
    """Decode only the exact V2 schema; V1, missing, and extra fields fail closed."""

    if not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA_VERSION:
        raise PLockV2ContractError(
            "V1/unknown P-lock consumption is forbidden; rematerialize a V2 lock"
        )
    item = _exact_fields(record, _RECORD_FIELDS, name="V2 P-lock record")
    if item["claim_level"] != CLAIM_LEVEL:
        raise PLockV2ContractError("V2 core claim level drift")
    if item["canonical_arm"] != CANONICAL_ARM:
        raise PLockV2ContractError("V2 record canonical arm drift")
    unsigned = {key: item[key] for key in sorted(_RECORD_FIELDS - {"record_sha256"})}
    _require_sha256(item["record_sha256"], name="V2 P-lock record")
    if item["record_sha256"] != canonical_sha256(unsigned):
        raise PLockV2ContractError("V2 P-lock record hash drift")
    signature_record = _exact_fields(
        item["geometry_signature"], _SIGNATURE_FIELDS, name="V2 geometry signature"
    )
    if signature_record["schema_version"] != SCHEMA_VERSION:
        raise PLockV2ContractError("nested V2 schema version drift")
    topology_raw = signature_record["ordered_root_topology"]
    if not isinstance(topology_raw, list):
        raise PLockV2ContractError("ordered root topology must be a list")
    topology = tuple(
        _topology_from_record(value, name=f"ordered_root_topology[{index}]")
        for index, value in enumerate(topology_raw)
    )
    signature = PLockGeometrySignatureV2(
        canonical_arm=signature_record["canonical_arm"],
        track=signature_record["track"],
        query_grid_shape=_grid(signature_record["query_grid_shape"], name="signature query grid"),
        reference_grid_shape=_grid(
            signature_record["reference_grid_shape"], name="signature reference grid"
        ),
        query_valid_geometry=_mask_geometry_from_record(
            signature_record["query_valid_geometry"], name="query valid geometry"
        ),
        reference_valid_geometry=_mask_geometry_from_record(
            signature_record["reference_valid_geometry"], name="reference valid geometry"
        ),
        ordered_root_topology=topology,
        selected_root_ordinals=tuple(signature_record["selected_root_ordinals"]),
        lock_state=signature_record["lock_state"],
        selected_row_sha256=signature_record["selected_row_sha256"],
        candidate_row_signature_sha256=signature_record[
            "candidate_row_signature_sha256"
        ],
        p_checkpoint_sha256=signature_record["p_checkpoint_sha256"],
        query_source_image_sha256=signature_record["query_source_image_sha256"],
        reference_source_image_sha256=signature_record[
            "reference_source_image_sha256"
        ],
        query_geometry_sha256=signature_record["query_geometry_sha256"],
        reference_geometry_sha256=signature_record["reference_geometry_sha256"],
        signature_sha256=signature_record["signature_sha256"],
    )
    roots_raw = item["roots"]
    if not isinstance(roots_raw, list):
        raise PLockV2ContractError("V2 roots must be a list")
    roots: list[StructuralRootLockV2] = []
    for index, value in enumerate(roots_raw):
        root = _exact_fields(value, _ROOT_RECORD_FIELDS, name=f"V2 root[{index}]")
        roots.append(
            StructuralRootLockV2(
                root_ordinal=root["root_ordinal"],
                root_topology_sha256=root["root_topology_sha256"],
                status=root["status"],
                state_reason=root["state_reason"],
                action_key_sha256=root["action_key_sha256"],
                query_mask=_bool_list(root["query_mask"], name=f"root[{index}] query mask"),
                reference_mask=_bool_list(
                    root["reference_mask"], name=f"root[{index}] reference mask"
                ),
                query_grid_shape=_grid(root["query_grid_shape"], name=f"root[{index}] query grid"),
                reference_grid_shape=_grid(
                    root["reference_grid_shape"], name=f"root[{index}] reference grid"
                ),
                relational_evidence_status=root["relational_evidence_status"],
            )
        )
    return CandidatePLockV2(
        candidate_key=item["candidate_key"],
        candidate_physical_row=item["candidate_physical_row"],
        track=item["track"],
        query_valid_mask=_bool_list(item["query_valid_mask"], name="query valid mask"),
        reference_valid_mask=_bool_list(
            item["reference_valid_mask"], name="reference valid mask"
        ),
        roots=tuple(roots),
        selected_root_ordinals=tuple(item["selected_root_ordinals"]),
        lock_state=item["lock_state"],
        geometry_signature=signature,
        schema_version=item["schema_version"],
    )


def require_v2_record(record: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the exact serialized schema and return the original mapping."""

    candidate_p_lock_v2_from_record(record)
    return record


__all__ = [
    "CANONICAL_ARM",
    "CLAIM_LEVEL",
    "LOCK_PROPOSAL_READY",
    "LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE",
    "MIN_READY_ROOTS",
    "PLockV2ContractError",
    "RELATIONAL_ACTIVE",
    "RELATIONAL_EXACT_ZERO",
    "ROOT_QUERY_UNMAPPABLE",
    "ROOT_READY",
    "ROOT_REFERENCE_MISSING",
    "SCHEMA_VERSION",
    "CandidatePLockV2",
    "FixedDenominatorAggregationV2",
    "MaskGeometryV2",
    "PLockGeometrySignatureV2",
    "StructuralRootLockV2",
    "aggregate_candidate_batch_v2",
    "aggregate_fixed_denominator_v2",
    "candidate_p_lock_v2_from_record",
    "candidate_p_lock_v2_to_record",
    "canonical_sha256",
    "make_candidate_p_lock_v2",
    "mask_geometry_v2",
    "require_v2_record",
    "selected_row_signature_v2",
    "tensor_sha256",
]
