"""Pure RoMa reference-first connected hypotheses and RAW-ColNomic unary V.

This module is a new lineage.  Historical RGH used ColNomic tokens inside its
assignment; the proposal objects below never accept tokens or retrieval scores.
Each valid reference cell is one atom.  A complete upper-level component tree
of a fixed RoMa reliability field supplies a fixed-width H/H0 family.  Only the
sealed binary supports cross the P -> V boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from .current_d1_roma_rawlocal_hyp_m0b1_natural_atom_adapter_v1 import (
    NULL_SHA256,
    build_natural_roma_atom_payload,
)
from .r0_natural_binding_v1 import spatial_permutation_is_affine


SCHEMA = "roma_rgh_reference_first_v4_20260910"
FAMILY_SCHEMA = "roma_rgh_reference_first_family_v4_20260910"
RAW_COLNOMIC = "RAW_COLNOMIC"
FEATURE_NAMES = (
    "overlap_q",
    "overlap_r",
    "sqrt_overlap_product",
    "cycle_quality",
    "precision_quality",
)
MIL_TEMPERATURE = 0.10
H0_PRIOR = 0.50
DEFAULT_HYPOTHESIS_SLOT_CAPACITY = 768


def tensor_sha256(value: torch.Tensor) -> str:
    item = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(json.dumps(list(item.shape), separators=(",", ":")).encode("ascii"))
    digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _grid(value: Sequence[int], name: str) -> tuple[int, int]:
    shape = tuple(int(x) for x in value)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} must be positive HxW")
    return shape


def _axis_coordinates(shape: tuple[int, int]) -> torch.Tensor:
    index = torch.arange(math.prod(shape), dtype=torch.int64)
    return torch.stack((index // shape[1], index % shape[1]), dim=1)


def _complete_permutation(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
    order = torch.as_tensor(value, dtype=torch.int64).detach().cpu().contiguous()
    if order.shape != (count,) or not torch.equal(torch.sort(order).values, torch.arange(count)):
        raise ValueError(f"{name} must be a complete permutation")
    return order


def _four_connected(indices: torch.Tensor, shape: tuple[int, int]) -> bool:
    cells = {(int(i) // shape[1], int(i) % shape[1]) for i in torch.unique(indices).tolist()}
    if not cells:
        return False
    unseen = set(cells)
    stack = [unseen.pop()]
    while stack:
        row, col = stack.pop()
        for other in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if other in unseen:
                unseen.remove(other)
                stack.append(other)
    return not unseen


def _two_axis(indices: torch.Tensor, shape: tuple[int, int]) -> bool:
    value = torch.unique(torch.as_tensor(indices, dtype=torch.int64))
    if value.numel() == 0:
        return False
    rows, cols = value // shape[1], value % shape[1]
    return bool(rows.min() < rows.max() and cols.min() < cols.max())


def _canonical_reciprocal_overlap(overlap_q: torch.Tensor, overlap_r: torch.Tensor) -> torch.Tensor:
    """Replay reciprocal overlap in the bank's canonical FP64 arithmetic.

    Natural RoMa tensors may be FP32.  The upstream adapter deliberately
    preserves that native arithmetic, including its FP32 square root, before
    widening the result for its own sealed payload.  A reference-first bank,
    however, seals FP64 overlap primitives.  Reusing the widened native square
    root would mix two arithmetic contracts: its inputs replay in FP64 while
    its result was rounded in FP32.  Recompute once from the sealed FP64
    primitives so construction and validation have one exact canonical value.
    """
    left = torch.as_tensor(overlap_q, dtype=torch.float64).detach().cpu().contiguous()
    right = torch.as_tensor(overlap_r, dtype=torch.float64).detach().cpu().contiguous()
    if left.shape != right.shape:
        raise ValueError("reciprocal overlap shape drift")
    return torch.sqrt((left * right).clamp_min(0.0)).contiguous()


@dataclass(frozen=True)
class ReferenceFirstRoMaAtomBank:
    query_resource_key: str
    destination_candidate_key: str
    source_reference_resource_key: str
    verification_reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    roma_features: torch.Tensor  # [R,5], RoMa only
    valid_mask: torch.Tensor  # [R]
    query_valid_axis: torch.Tensor  # [Q]
    reference_valid_axis: torch.Tensor  # [R]
    query_content_indices: torch.Tensor  # nearest original query content row
    reference_content_indices: torch.Tensor  # arange(R)
    query_coordinate_indices: torch.Tensor  # active geometry endpoint
    reference_coordinate_indices: torch.Tensor  # active geometry endpoint
    mapped_query_xy: torch.Tensor
    cycled_reference_xy: torch.Tensor
    control_namespace: str
    source_binding_sha256: str
    parent_real_sha256: str
    control_plan_sha256: str
    logical_sha256: str

    @property
    def atom_count(self) -> int:
        return math.prod(self.reference_grid_shape)

    @property
    def reliability(self) -> torch.Tensor:
        feature = self.roma_features
        return (feature[:, 2] * feature[:, 3] * feature[:, 4]).contiguous()


def _bank_hash(value: ReferenceFirstRoMaAtomBank) -> str:
    return logical_sha256(
        {
            "schema": SCHEMA,
            "query_resource_key": value.query_resource_key,
            "destination_candidate_key": value.destination_candidate_key,
            "source_reference_resource_key": value.source_reference_resource_key,
            "verification_reference_resource_key": value.verification_reference_resource_key,
            "query_grid_shape": list(value.query_grid_shape),
            "reference_grid_shape": list(value.reference_grid_shape),
            "roma_features": tensor_sha256(value.roma_features),
            "valid_mask": tensor_sha256(value.valid_mask),
            "query_valid_axis": tensor_sha256(value.query_valid_axis),
            "reference_valid_axis": tensor_sha256(value.reference_valid_axis),
            "query_content_indices": tensor_sha256(value.query_content_indices),
            "reference_content_indices": tensor_sha256(value.reference_content_indices),
            "query_coordinate_indices": tensor_sha256(value.query_coordinate_indices),
            "reference_coordinate_indices": tensor_sha256(value.reference_coordinate_indices),
            "mapped_query_xy": tensor_sha256(value.mapped_query_xy),
            "cycled_reference_xy": tensor_sha256(value.cycled_reference_xy),
            "control_namespace": value.control_namespace,
            "source_binding_sha256": value.source_binding_sha256,
            "parent_real_sha256": value.parent_real_sha256,
            "control_plan_sha256": value.control_plan_sha256,
        }
    )


def validate_reference_first_bank(value: ReferenceFirstRoMaAtomBank) -> ReferenceFirstRoMaAtomBank:
    qshape = _grid(value.query_grid_shape, "query grid")
    rshape = _grid(value.reference_grid_shape, "reference grid")
    qcount, rcount = math.prod(qshape), math.prod(rshape)
    if not all(isinstance(x, str) and x for x in (
        value.query_resource_key, value.destination_candidate_key,
        value.source_reference_resource_key, value.verification_reference_resource_key,
        value.control_namespace,
    )):
        raise ValueError("empty resource/control key")
    _sha(value.source_binding_sha256, "source binding")
    _sha(value.parent_real_sha256, "parent REAL")
    _sha(value.control_plan_sha256, "control plan")
    expected = {
        "roma_features": (torch.float64, (rcount, 5)),
        "valid_mask": (torch.bool, (rcount,)),
        "query_valid_axis": (torch.bool, (qcount,)),
        "reference_valid_axis": (torch.bool, (rcount,)),
        "query_content_indices": (torch.int64, (rcount,)),
        "reference_content_indices": (torch.int64, (rcount,)),
        "query_coordinate_indices": (torch.int64, (rcount,)),
        "reference_coordinate_indices": (torch.int64, (rcount,)),
        "mapped_query_xy": (torch.float64, (rcount, 2)),
        "cycled_reference_xy": (torch.float64, (rcount, 2)),
    }
    for name, (dtype, shape) in expected.items():
        item = torch.as_tensor(getattr(value, name))
        if item.dtype != dtype or tuple(item.shape) != shape or not item.is_contiguous():
            raise ValueError(f"{name} dtype/shape/contiguity drift")
        if item.is_floating_point() and not bool(torch.isfinite(item).all()):
            raise ValueError(f"{name} nonfinite")
    if value.roma_features.numel() and bool(((value.roma_features < 0) | (value.roma_features > 1)).any()):
        raise ValueError("RoMa feature outside [0,1]")
    if not bool(value.query_valid_axis.any()) or not bool(value.reference_valid_axis.any()):
        raise ValueError("canonical valid axis is empty")
    if not torch.equal(
        value.roma_features[:, 2],
        _canonical_reciprocal_overlap(value.roma_features[:, 0], value.roma_features[:, 1]),
    ):
        raise ValueError("reciprocal overlap does not replay")
    if not torch.equal(value.reference_content_indices, torch.arange(rcount)):
        raise ValueError("reference-first content axis must be arange(R)")
    if bool(((value.query_content_indices < 0) | (value.query_content_indices >= qcount)).any()):
        raise ValueError("query content index outside grid")
    if bool((
        value.valid_mask
        & (
            ~value.query_valid_axis[value.query_content_indices]
            | ~value.reference_valid_axis[value.reference_content_indices]
        )
    ).any()):
        raise ValueError("valid atom points into canonical padding")
    _complete_permutation(value.reference_coordinate_indices, rcount, "reference coordinates")
    if bool(((value.query_coordinate_indices < 0) | (value.query_coordinate_indices >= qcount)).any()):
        raise ValueError("query coordinate outside grid")
    if value.control_namespace == "REAL":
        if (
            value.destination_candidate_key != value.source_reference_resource_key
            or value.destination_candidate_key != value.verification_reference_resource_key
            or value.parent_real_sha256 != NULL_SHA256
            or value.control_plan_sha256 != NULL_SHA256
            or not torch.equal(value.query_coordinate_indices, value.query_content_indices)
            or not torch.equal(value.reference_coordinate_indices, value.reference_content_indices)
        ):
            raise ValueError("REAL candidate/reference binding drift")
    elif value.parent_real_sha256 == NULL_SHA256 or value.control_plan_sha256 == NULL_SHA256:
        raise ValueError("control missing parent REAL")
    expected_hash = _bank_hash(replace(value, logical_sha256="PENDING"))
    if value.logical_sha256 != expected_hash:
        raise ValueError("reference-first bank hash drift")
    return value


def seal_reference_first_bank(value: ReferenceFirstRoMaAtomBank) -> ReferenceFirstRoMaAtomBank:
    staged = replace(value, logical_sha256="PENDING")
    return validate_reference_first_bank(replace(staged, logical_sha256=_bank_hash(staged)))


def build_reference_first_bank_from_dense_roma(
    *,
    query_resource_key: str,
    candidate_resource_key: str,
    reference_resource_key: str,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_cell_boxes_xyxy: torch.Tensor,
    reference_cell_boxes_xyxy: torch.Tensor,
    query_valid_mask: torch.Tensor,
    reference_valid_mask: torch.Tensor,
    warp_ab: torch.Tensor,
    overlap_ab: torch.Tensor,
    precision_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    overlap_ba: torch.Tensor,
    precision_ba: torch.Tensor,
    source_binding_sha256: str,
) -> ReferenceFirstRoMaAtomBank:
    """Sample the reverse RoMa direction on the complete reference-cell axis."""

    # The already-validated pure-RoMa adapter is mathematically symmetric.  We
    # swap A/B solely to sample one atom per original reference cell, then copy
    # only its five RoMa fields and geometry into the narrower public object.
    swapped = build_natural_roma_atom_payload(
        query_resource_key=reference_resource_key,
        candidate_resource_key=query_resource_key,
        reference_resource_key=query_resource_key,
        query_grid_shape=reference_grid_shape,
        reference_grid_shape=query_grid_shape,
        query_cell_boxes_xyxy=reference_cell_boxes_xyxy,
        reference_cell_boxes_xyxy=query_cell_boxes_xyxy,
        query_valid_mask=reference_valid_mask,
        reference_valid_mask=query_valid_mask,
        warp_ab=warp_ba,
        overlap_ab=overlap_ba,
        precision_ab=precision_ba,
        warp_ba=warp_ab,
        overlap_ba=overlap_ab,
        precision_ba=precision_ab,
        source_binding_sha256=source_binding_sha256,
    )
    rcount = math.prod(reference_grid_shape)
    overlap_q = swapped.features[:, 1].clone().contiguous()
    overlap_r = swapped.features[:, 0].clone().contiguous()
    # Do not carry the adapter's widened FP32 square root into this FP64 bank.
    # The canonical bank equation is rebuilt from its own sealed primitives.
    reciprocal = _canonical_reciprocal_overlap(overlap_q, overlap_r)
    features = torch.stack(
        (overlap_q, overlap_r, reciprocal, swapped.features[:, 3], swapped.features[:, 4]),
        dim=1,
    ).contiguous()
    provisional = ReferenceFirstRoMaAtomBank(
        query_resource_key=query_resource_key,
        destination_candidate_key=candidate_resource_key,
        source_reference_resource_key=reference_resource_key,
        verification_reference_resource_key=reference_resource_key,
        query_grid_shape=tuple(query_grid_shape),
        reference_grid_shape=tuple(reference_grid_shape),
        roma_features=features,
        valid_mask=swapped.valid_mask.clone().contiguous(),
        query_valid_axis=swapped.reference_valid_axis.clone().contiguous(),
        reference_valid_axis=swapped.query_valid_axis.clone().contiguous(),
        query_content_indices=swapped.reference_indices.clone().contiguous(),
        reference_content_indices=torch.arange(rcount, dtype=torch.int64),
        query_coordinate_indices=swapped.reference_indices.clone().contiguous(),
        reference_coordinate_indices=torch.arange(rcount, dtype=torch.int64),
        mapped_query_xy=swapped.forward_reference_xy.clone().contiguous(),
        cycled_reference_xy=swapped.reverse_query_xy.clone().contiguous(),
        control_namespace="REAL",
        source_binding_sha256=source_binding_sha256,
        parent_real_sha256=NULL_SHA256,
        control_plan_sha256=NULL_SHA256,
        logical_sha256="PENDING",
    )
    return seal_reference_first_bank(provisional)


def coordinate_destroy_bank(
    bank: ReferenceFirstRoMaAtomBank,
    *,
    query_permutation: torch.Tensor,
    reference_permutation: torch.Tensor,
    namespace: str,
) -> ReferenceFirstRoMaAtomBank:
    source = validate_reference_first_bank(bank)
    if source.control_namespace != "REAL" or not namespace or namespace == "REAL":
        raise ValueError("coordinate control must derive from REAL")
    qcount, rcount = math.prod(source.query_grid_shape), math.prod(source.reference_grid_shape)
    qperm = _complete_permutation(query_permutation, qcount, "query control")
    rperm = _complete_permutation(reference_permutation, rcount, "reference control")
    qidentity = torch.equal(qperm, torch.arange(qcount))
    ridentity = torch.equal(rperm, torch.arange(rcount))
    if qidentity and ridentity:
        raise ValueError("coordinate control must destroy at least one endpoint")
    for order, count, shape, identity, name in (
        (qperm, qcount, source.query_grid_shape, qidentity, "query"),
        (rperm, rcount, source.reference_grid_shape, ridentity, "reference"),
    ):
        if not identity and (
            bool(order.eq(torch.arange(count)).any())
            or spatial_permutation_is_affine(order, shape)
        ):
            raise ValueError(f"changed {name} coordinate control must be fixed-point-free and non-affine")
    if not torch.equal(source.query_valid_axis, source.query_valid_axis[qperm]) or not torch.equal(
        source.reference_valid_axis, source.reference_valid_axis[rperm]
    ):
        raise ValueError("coordinate control crosses valid/padding strata")
    controlled = replace(
        source,
        query_coordinate_indices=qperm[source.query_content_indices].contiguous(),
        reference_coordinate_indices=rperm[source.reference_content_indices].contiguous(),
        control_namespace=namespace,
        parent_real_sha256=source.logical_sha256,
        control_plan_sha256=logical_sha256({
            "namespace": namespace,
            "query_permutation": qperm.tolist(),
            "reference_permutation": rperm.tolist(),
        }),
        logical_sha256="PENDING",
    )
    return seal_reference_first_bank(controlled)


def candidate_binding_destroy_population(
    banks: Sequence[ReferenceFirstRoMaAtomBank],
    *,
    donor_for_destination: Sequence[int],
    candidate_identity_keys: Sequence[str],
    namespace: str,
    proposal_only: bool,
) -> tuple[ReferenceFirstRoMaAtomBank, ...]:
    """Move a complete donor RoMa proposal bundle before P is materialized.

    ``proposal_only=True`` keeps destination ColNomic content for the primary
    C_P_BIND control.  False moves donor proposal and donor content together
    for the separately reported C_WHOLE control.
    """
    source = tuple(validate_reference_first_bank(item) for item in banks)
    count = len(source)
    plan = tuple(int(x) for x in donor_for_destination)
    identities = tuple(str(x) for x in candidate_identity_keys)
    if (
        count < 2
        or sorted(plan) != list(range(count))
        or any(index == donor for index, donor in enumerate(plan))
        or not namespace
        or namespace == "REAL"
        or any(item.control_namespace != "REAL" for item in source)
        or len({item.query_resource_key for item in source}) != 1
        or len({item.destination_candidate_key for item in source}) != count
        or len(identities) != count
        or any(not item for item in identities)
        or any(identities[index] == identities[donor] for index, donor in enumerate(plan))
    ):
        raise ValueError("candidate-binding donor population invalid")
    output = []
    plan_sha256 = logical_sha256({
        "namespace": namespace,
        "proposal_only": proposal_only,
        "destination_keys": [item.destination_candidate_key for item in source],
        "donor_for_destination": list(plan),
        "candidate_identity_keys": list(identities),
        "donor_source_reference_keys": [source[index].source_reference_resource_key for index in plan],
    })
    for destination, donor in enumerate(plan):
        destination_item, donor_item = source[destination], source[donor]
        controlled = replace(
            donor_item,
            destination_candidate_key=destination_item.destination_candidate_key,
            verification_reference_resource_key=(
                destination_item.verification_reference_resource_key
                if proposal_only
                else donor_item.verification_reference_resource_key
            ),
            control_namespace=namespace,
            parent_real_sha256=destination_item.logical_sha256,
            control_plan_sha256=plan_sha256,
            logical_sha256="PENDING",
        )
        output.append(seal_reference_first_bank(controlled))
    return tuple(output)


@dataclass(frozen=True)
class FrozenHypothesisSlot:
    slot: int
    state: str
    query_coordinate_indices: torch.Tensor
    reference_coordinate_indices: torch.Tensor
    query_content_indices: torch.Tensor
    reference_content_indices: torch.Tensor
    provenance_query_content_indices: torch.Tensor
    provenance_reference_content_indices: torch.Tensor
    logical_sha256: str


@dataclass(frozen=True)
class FrozenRoMaRGHFamily:
    schema_version: str
    query_resource_key: str
    candidate_resource_key: str
    source_reference_resource_key: str
    verification_reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    control_namespace: str
    source_bank_sha256: str
    slot_capacity: int
    slots: tuple[FrozenHypothesisSlot, ...]
    h1_count: int
    logical_sha256: str


def _slot_hash(slot: FrozenHypothesisSlot) -> str:
    return logical_sha256({
        "slot": slot.slot, "state": slot.state,
        "query_coordinate_indices": tensor_sha256(slot.query_coordinate_indices),
        "reference_coordinate_indices": tensor_sha256(slot.reference_coordinate_indices),
        "query_content_indices": tensor_sha256(slot.query_content_indices),
        "reference_content_indices": tensor_sha256(slot.reference_content_indices),
        "provenance_query_content_indices": tensor_sha256(slot.provenance_query_content_indices),
        "provenance_reference_content_indices": tensor_sha256(slot.provenance_reference_content_indices),
    })


def _family_hash(value: FrozenRoMaRGHFamily) -> str:
    return logical_sha256({
        "schema": value.schema_version,
        "query_resource_key": value.query_resource_key,
        "candidate_resource_key": value.candidate_resource_key,
        "source_reference_resource_key": value.source_reference_resource_key,
        "verification_reference_resource_key": value.verification_reference_resource_key,
        "query_grid_shape": list(value.query_grid_shape),
        "reference_grid_shape": list(value.reference_grid_shape),
        "control_namespace": value.control_namespace,
        "source_bank_sha256": value.source_bank_sha256,
        "slot_capacity": value.slot_capacity,
        "slots": [slot.logical_sha256 for slot in value.slots],
        "h1_count": value.h1_count,
    })


def validate_family(value: FrozenRoMaRGHFamily) -> FrozenRoMaRGHFamily:
    qshape, rshape = _grid(value.query_grid_shape, "family query"), _grid(value.reference_grid_shape, "family reference")
    rcount = math.prod(rshape)
    if isinstance(value.slot_capacity, bool) or value.slot_capacity <= 0:
        raise ValueError("family slot capacity invalid")
    if rcount > value.slot_capacity:
        raise ValueError("reference grid exceeds frozen hypothesis capacity")
    if value.schema_version != FAMILY_SCHEMA or len(value.slots) != value.slot_capacity:
        raise ValueError("family schema/fixed-width drift")
    if tuple(slot.slot for slot in value.slots) != tuple(range(value.slot_capacity)):
        raise ValueError("family slot axis drift")
    h1 = 0
    for slot in value.slots:
        for tensor in (
            slot.query_coordinate_indices, slot.reference_coordinate_indices,
            slot.query_content_indices, slot.reference_content_indices,
            slot.provenance_query_content_indices,
            slot.provenance_reference_content_indices,
        ):
            if tensor.dtype != torch.int64 or tensor.ndim != 1 or not tensor.is_contiguous():
                raise ValueError("slot index envelope drift")
        if slot.logical_sha256 != _slot_hash(replace(slot, logical_sha256="PENDING")):
            raise ValueError("slot hash drift")
        if slot.state == "H0":
            if any(t.numel() for t in (
                slot.query_coordinate_indices, slot.reference_coordinate_indices,
                slot.query_content_indices, slot.reference_content_indices,
                slot.provenance_query_content_indices,
                slot.provenance_reference_content_indices,
            )):
                raise ValueError("H0 slot must be empty")
            continue
        if slot.state != "H1":
            raise ValueError("unknown slot state")
        h1 += 1
        if (
            slot.slot >= rcount
            or not torch.equal(slot.query_coordinate_indices, torch.unique(slot.query_coordinate_indices, sorted=True))
            or not torch.equal(slot.reference_coordinate_indices, torch.unique(slot.reference_coordinate_indices, sorted=True))
            or not torch.equal(slot.query_content_indices, torch.unique(slot.query_content_indices, sorted=True))
            or not torch.equal(slot.reference_content_indices, torch.unique(slot.reference_content_indices, sorted=True))
            or slot.provenance_query_content_indices.numel() == 0
            or not _four_connected(slot.query_coordinate_indices, qshape)
            or not _four_connected(slot.reference_coordinate_indices, rshape)
            or not _two_axis(slot.query_coordinate_indices, qshape)
            or not _two_axis(slot.reference_coordinate_indices, rshape)
            or slot.provenance_query_content_indices.shape != slot.provenance_reference_content_indices.shape
            or set(slot.query_content_indices.tolist()) != set(slot.provenance_query_content_indices.tolist())
            or set(slot.reference_content_indices.tolist()) != set(slot.provenance_reference_content_indices.tolist())
            or bool(((slot.query_coordinate_indices < 0) | (slot.query_coordinate_indices >= math.prod(qshape))).any())
            or bool(((slot.query_content_indices < 0) | (slot.query_content_indices >= math.prod(qshape))).any())
            or bool(((slot.reference_coordinate_indices < 0) | (slot.reference_coordinate_indices >= rcount)).any())
            or bool(((slot.reference_content_indices < 0) | (slot.reference_content_indices >= rcount)).any())
            or bool(((slot.provenance_query_content_indices < 0) | (slot.provenance_query_content_indices >= math.prod(qshape))).any())
            or bool(((slot.provenance_reference_content_indices < 0) | (slot.provenance_reference_content_indices >= rcount)).any())
        ):
            raise ValueError("H1 dual-end geometry/provenance drift")
    if h1 != value.h1_count or value.logical_sha256 != _family_hash(replace(value, logical_sha256="PENDING")):
        raise ValueError("family count/hash drift")
    return value


def _neighbours(index: int, shape: tuple[int, int]) -> tuple[int, ...]:
    row, col = divmod(index, shape[1])
    result = []
    if row: result.append(index - shape[1])
    if row + 1 < shape[0]: result.append(index + shape[1])
    if col: result.append(index - 1)
    if col + 1 < shape[1]: result.append(index + 1)
    return tuple(result)


def _joint_component_tree(bank: ReferenceFirstRoMaAtomBank) -> list[dict[str, object]]:
    count = bank.atom_count
    values = bank.reliability
    valid = bank.valid_mask & values.gt(0)
    parents = list(range(count)); active = [False] * count
    frontier = [set() for _ in range(count)]; nodes: list[dict[str, object]] = []
    by_reference_coordinate = {int(bank.reference_coordinate_indices[i]): i for i in range(count)}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]; index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right: return
        if left > right: left, right = right, left
        parents[right] = left; frontier[left].update(frontier[right]); frontier[right].clear()

    ordered = sorted(torch.nonzero(valid, as_tuple=False).flatten().tolist(), key=lambda i: (-float(values[i]), i))
    start = 0
    while start < len(ordered):
        level = float(values[ordered[start]]); stop = start + 1
        while stop < len(ordered) and float(values[ordered[stop]]) == level: stop += 1
        added = ordered[start:stop]
        for atom in added: active[atom] = True
        for atom in added:
            rcoord = int(bank.reference_coordinate_indices[atom])
            for neighbour_coord in _neighbours(rcoord, bank.reference_grid_shape):
                other = by_reference_coordinate[neighbour_coord]
                # The complete max-tree is defined only by the reference
                # scalar field and native reference 4-neighbour graph.  Query
                # geometry is a downstream H1 legality test, not an edge that
                # may silently delete reference upper-level components.
                if active[other]:
                    union(atom, other)
        additions: dict[int, list[int]] = {}
        for atom in added: additions.setdefault(find(atom), []).append(atom)
        for root in sorted(additions):
            children = sorted(frontier[root]); new_atoms = sorted(additions[root]); node_id = len(nodes)
            nodes.append({"node_id": node_id, "level": level, "new_atom_indices": new_atoms,
                          "child_node_ids": children, "parent_node_id": None})
            for child in children: nodes[child]["parent_node_id"] = node_id
            frontier[root] = {node_id}
        start = stop
    return nodes


def reference_upper_level_supports(
    bank: ReferenceFirstRoMaAtomBank,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return all canonical max-tree nodes as ``(slot, atom_support)``."""
    source = validate_reference_first_bank(bank)
    nodes = _joint_component_tree(source)
    supports: list[tuple[int, ...]] = []
    output: list[tuple[int, tuple[int, ...]]] = []
    for node in nodes:
        support = list(node["new_atom_indices"])
        for child in node["child_node_ids"]:
            support.extend(supports[int(child)])
        atoms = tuple(sorted(set(int(x) for x in support)))
        supports.append(atoms)
        output.append((min(int(x) for x in node["new_atom_indices"]), atoms))
    return tuple(output)


def freeze_hypothesis_family(
    bank: ReferenceFirstRoMaAtomBank,
    *,
    slot_capacity: int = DEFAULT_HYPOTHESIS_SLOT_CAPACITY,
) -> FrozenRoMaRGHFamily:
    source = validate_reference_first_bank(bank)
    rcount = source.atom_count
    if isinstance(slot_capacity, bool) or slot_capacity <= 0 or rcount > slot_capacity:
        raise ValueError("reference atom count exceeds global slot capacity")
    slots: list[FrozenHypothesisSlot | None] = [None] * slot_capacity
    for slot_index, atoms in reference_upper_level_supports(source):
        rows = torch.tensor(atoms, dtype=torch.int64)
        qcoords = torch.unique(source.query_coordinate_indices[rows], sorted=True)
        rcoords = torch.unique(source.reference_coordinate_indices[rows], sorted=True)
        qcontent = torch.unique(source.query_content_indices[rows], sorted=True)
        rcontent = torch.unique(source.reference_content_indices[rows], sorted=True)
        legal = bool(
            _four_connected(qcoords, source.query_grid_shape)
            and _four_connected(rcoords, source.reference_grid_shape)
            and _two_axis(qcoords, source.query_grid_shape)
            and _two_axis(rcoords, source.reference_grid_shape)
        )
        empty = torch.empty(0, dtype=torch.int64)
        provisional = FrozenHypothesisSlot(
            slot=slot_index, state="H1" if legal else "H0",
            query_coordinate_indices=qcoords.contiguous() if legal else empty,
            reference_coordinate_indices=rcoords.contiguous() if legal else empty,
            query_content_indices=qcontent.contiguous() if legal else empty,
            reference_content_indices=rcontent.contiguous() if legal else empty,
            provenance_query_content_indices=source.query_content_indices[rows].clone().contiguous() if legal else empty,
            provenance_reference_content_indices=source.reference_content_indices[rows].clone().contiguous() if legal else empty,
            logical_sha256="PENDING",
        )
        slots[slot_index] = replace(provisional, logical_sha256=_slot_hash(provisional))
    for index in range(slot_capacity):
        if slots[index] is None:
            empty = torch.empty(0, dtype=torch.int64)
            provisional = FrozenHypothesisSlot(index, "H0", empty, empty, empty, empty, empty, empty, "PENDING")
            slots[index] = replace(provisional, logical_sha256=_slot_hash(provisional))
    family = FrozenRoMaRGHFamily(
        schema_version=FAMILY_SCHEMA,
        query_resource_key=source.query_resource_key,
        candidate_resource_key=source.destination_candidate_key,
        source_reference_resource_key=source.source_reference_resource_key,
        verification_reference_resource_key=source.verification_reference_resource_key,
        query_grid_shape=source.query_grid_shape,
        reference_grid_shape=source.reference_grid_shape,
        control_namespace=source.control_namespace,
        source_bank_sha256=source.logical_sha256,
        slot_capacity=slot_capacity,
        slots=tuple(slots),  # type: ignore[arg-type]
        h1_count=sum(item.state == "H1" for item in slots if item is not None),
        logical_sha256="PENDING",
    )
    return validate_family(replace(family, logical_sha256=_family_hash(family)))


@dataclass(frozen=True)
class RawColNomicGrid:
    resource_key: str
    grid_shape: tuple[int, int]
    tokens: torch.Tensor
    valid_mask: torch.Tensor | None = None
    provenance: str = RAW_COLNOMIC


@dataclass(frozen=True)
class UnaryVerification:
    all_patch_full_reference: torch.Tensor
    rgh_query_full_reference: torch.Tensor
    rgh_paired_region: torch.Tensor
    paired_eligible: bool
    qfull_slot_scores: torch.Tensor
    paired_slot_scores: torch.Tensor


def _validate_raw(value: RawColNomicGrid) -> RawColNomicGrid:
    shape = _grid(value.grid_shape, "RAW grid")
    tokens = torch.as_tensor(value.tokens)
    if value.provenance != RAW_COLNOMIC or not value.resource_key or tokens.ndim != 2 or tokens.shape[0] != math.prod(shape):
        raise ValueError("RAW-ColNomic grid envelope drift")
    if not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all()):
        raise ValueError("RAW-ColNomic tokens invalid")
    valid = torch.ones(tokens.shape[0], dtype=torch.bool) if value.valid_mask is None else torch.as_tensor(
        value.valid_mask, dtype=torch.bool
    ).detach().cpu().contiguous()
    if valid.shape != (tokens.shape[0],) or not bool(valid.any()):
        raise ValueError("RAW-ColNomic valid mask drift")
    return RawColNomicGrid(value.resource_key, shape, tokens.detach().cpu().contiguous(), valid, value.provenance)


def h0_marginal(scores: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("fixed hypothesis slot vector required")
    if bool(values.eq(0).all()):
        return values.abs().sum() * 0.0
    return MIL_TEMPERATURE * torch.logaddexp(
        values.new_tensor(math.log(H0_PRIOR)),
        values.new_tensor(math.log(1.0 - H0_PRIOR))
        + torch.logsumexp(values / MIL_TEMPERATURE, dim=0) - math.log(values.numel()),
    )


def verify_family_unary(
    family: FrozenRoMaRGHFamily,
    query: RawColNomicGrid,
    reference: RawColNomicGrid,
) -> UnaryVerification:
    item = validate_family(family); q = _validate_raw(query); r = _validate_raw(reference)
    if (
        q.resource_key != item.query_resource_key
        or r.resource_key != item.verification_reference_resource_key
        or q.grid_shape != item.query_grid_shape
    ):
        raise ValueError("P/V resource binding drift")
    if (
        item.source_reference_resource_key == item.verification_reference_resource_key
        and r.grid_shape != item.reference_grid_shape
    ):
        raise ValueError("P/V reference grid binding drift")
    sim = F.normalize(q.tokens.to(torch.float64), dim=1) @ F.normalize(r.tokens.to(torch.float64), dim=1).T
    sim[:, ~r.valid_mask] = -torch.inf
    all_score = sim[q.valid_mask].max(dim=1).values.mean()
    qslots = torch.zeros(len(item.slots), dtype=torch.float64)
    pslots = torch.zeros(len(item.slots), dtype=torch.float64)
    paired_eligible = bool(
        r.resource_key == item.source_reference_resource_key
        and r.grid_shape == item.reference_grid_shape
    )
    for slot in item.slots:
        if slot.state == "H0": continue
        qindex = slot.query_content_indices
        if not bool(q.valid_mask[qindex].all()):
            raise ValueError("H1 reads invalid query content")
        qslots[slot.slot] = sim[qindex].max(dim=1).values.mean()
        # Paired-region V reads only the two binary component supports.  The
        # atom-level RoMa mapping is audit provenance and must not hard-bind a
        # query token to its predicted reference token.
        if paired_eligible:
            refs = slot.reference_content_indices[r.valid_mask[slot.reference_content_indices]]
            if refs.numel() == 0:
                raise RuntimeError("sealed paired component has no valid reference content")
            pslots[slot.slot] = sim[qindex][:, refs].max(dim=1).values.mean()
    return UnaryVerification(
        all_score,
        h0_marginal(qslots),
        h0_marginal(pslots) if paired_eligible else pslots.abs().sum() * 0.0,
        paired_eligible,
        qslots,
        pslots,
    )


@dataclass(frozen=True)
class FrozenQueryOnlyFamily:
    query_resource_key: str
    candidate_keys: tuple[str, ...]
    source_map_sha256: tuple[str, ...]
    candidate_visibility_sha256: str
    mean_visibility_sha256: str
    family: FrozenRoMaRGHFamily
    logical_sha256: str


def _query_only_hash(value: FrozenQueryOnlyFamily) -> str:
    return logical_sha256({
        "query_resource_key": value.query_resource_key,
        "candidate_keys": list(value.candidate_keys),
        "source_map_sha256": list(value.source_map_sha256),
        "candidate_visibility_sha256": value.candidate_visibility_sha256,
        "mean_visibility_sha256": value.mean_visibility_sha256,
        "family_sha256": value.family.logical_sha256,
    })


def build_query_only_family(
    query_resource_key: str,
    query_grid_shape: tuple[int, int],
    candidate_visibilities: torch.Tensor,
    candidate_keys: Sequence[str],
    source_map_sha256: Sequence[str],
    *,
    slot_capacity: int = DEFAULT_HYPOTHESIS_SLOT_CAPACITY,
) -> FrozenQueryOnlyFamily:
    """Seal the exact mean of a complete anonymous C128 visibility population."""
    shape = _grid(query_grid_shape, "query-only grid"); count = math.prod(shape)
    raw_population = torch.as_tensor(candidate_visibilities)
    if raw_population.dtype != torch.float64:
        raise ValueError("query-only source maps must be canonical FP64 before hashing")
    population = raw_population.detach().cpu().contiguous()
    keys = tuple(str(item) for item in candidate_keys)
    map_hashes = tuple(str(item) for item in source_map_sha256)
    if (
        population.shape != (128, count)
        or not bool(torch.isfinite(population).all())
        or bool(((population < 0) | (population > 1)).any())
        or len(keys) != 128
        or len(set(keys)) != 128
        or any(not key for key in keys)
        or len(map_hashes) != 128
        or any(_sha(item, "query visibility map") != item for item in map_hashes)
    ):
        raise ValueError("query-only requires one complete anonymous C128 visibility tensor")
    if any(map_hashes[index] != tensor_sha256(population[index]) for index in range(128)):
        raise ValueError("query-only per-candidate visibility hash drift")
    values = population.mean(dim=0).contiguous()
    population_hash = tensor_sha256(population)
    mean_hash = tensor_sha256(values)
    features = torch.stack((values, values, values, torch.ones_like(values), torch.ones_like(values)), dim=1)
    source_hash = logical_sha256({
        "query_resource_key": query_resource_key,
        "candidate_keys": list(keys),
        "source_map_sha256": list(map_hashes),
        "population_sha256": population_hash,
        "mean_sha256": mean_hash,
    })
    dummy = f"QUERY_ONLY:{query_resource_key}"
    bank = ReferenceFirstRoMaAtomBank(
        query_resource_key=query_resource_key,
        destination_candidate_key=dummy,
        source_reference_resource_key=dummy,
        verification_reference_resource_key=dummy,
        query_grid_shape=shape,
        reference_grid_shape=shape,
        roma_features=features,
        valid_mask=values.gt(0),
        query_valid_axis=torch.ones(count, dtype=torch.bool),
        reference_valid_axis=torch.ones(count, dtype=torch.bool),
        query_content_indices=torch.arange(count),
        reference_content_indices=torch.arange(count),
        query_coordinate_indices=torch.arange(count),
        reference_coordinate_indices=torch.arange(count),
        mapped_query_xy=torch.zeros((count, 2), dtype=torch.float64),
        cycled_reference_xy=torch.zeros((count, 2), dtype=torch.float64),
        control_namespace="QUERY_ONLY",
        source_binding_sha256=source_hash,
        parent_real_sha256=source_hash,
        control_plan_sha256=source_hash,
        logical_sha256="PENDING",
    )
    family = freeze_hypothesis_family(seal_reference_first_bank(bank), slot_capacity=slot_capacity)
    provisional = FrozenQueryOnlyFamily(
        query_resource_key, keys, map_hashes, population_hash, mean_hash, family, "PENDING"
    )
    return replace(provisional, logical_sha256=_query_only_hash(provisional))


def verify_query_only_unary(
    family: FrozenQueryOnlyFamily,
    query: RawColNomicGrid,
    reference: RawColNomicGrid,
) -> torch.Tensor:
    if (
        not isinstance(family, FrozenQueryOnlyFamily)
        or family.logical_sha256 != _query_only_hash(replace(family, logical_sha256="PENDING"))
        or family.query_resource_key != query.resource_key
        or reference.resource_key not in family.candidate_keys
    ):
        raise ValueError("query-only family binding drift")
    q, r = _validate_raw(query), _validate_raw(reference)
    base = validate_family(family.family)
    if q.grid_shape != base.query_grid_shape:
        raise ValueError("query-only grid drift")
    sim = F.normalize(q.tokens.to(torch.float64), dim=1) @ F.normalize(r.tokens.to(torch.float64), dim=1).T
    sim[:, ~r.valid_mask] = -torch.inf
    scores = torch.zeros(base.slot_capacity, dtype=torch.float64)
    for slot in base.slots:
        if slot.state == "H1":
            indices = slot.query_content_indices
            if not bool(q.valid_mask[indices].all()):
                raise ValueError("query-only family reads invalid content")
            scores[slot.slot] = sim[indices].max(dim=1).values.mean()
    return h0_marginal(scores)


__all__ = [
    "DEFAULT_HYPOTHESIS_SLOT_CAPACITY", "FEATURE_NAMES", "FAMILY_SCHEMA", "H0_PRIOR", "MIL_TEMPERATURE",
    "ReferenceFirstRoMaAtomBank", "FrozenHypothesisSlot", "FrozenRoMaRGHFamily",
    "RawColNomicGrid", "UnaryVerification", "build_reference_first_bank_from_dense_roma",
    "build_query_only_family", "candidate_binding_destroy_population", "coordinate_destroy_bank", "freeze_hypothesis_family",
    "h0_marginal", "logical_sha256", "reference_upper_level_supports", "seal_reference_first_bank", "tensor_sha256",
    "validate_family", "validate_reference_first_bank", "verify_family_unary",
    "verify_query_only_unary",
]
