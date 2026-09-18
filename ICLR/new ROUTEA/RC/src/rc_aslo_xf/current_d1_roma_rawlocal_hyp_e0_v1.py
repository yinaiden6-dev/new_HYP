"""Isolated synthetic core for RoMa-only P and RAW-ColNomic-only V.

The proposal API deliberately has no retrieval-base or local-token input.  It
turns five RoMa facts into one sealed connected hypothesis or exact H0.  The
verifier is a separate function and rejects every token provenance other than
RAW_COLNOMIC.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ROMA_FEATURE_NAMES = (
    "overlap_q",
    "overlap_r",
    "sqrt_overlap_product",
    "cycle_quality",
    "precision_quality",
)
ROMA_FEATURE_DIM = 5
SELECTOR_PARAMETER_COUNT = 6
MIN_QUERY_PATCHES = 4
MIN_REFERENCE_CELLS = 2
REFERENCE_LOCAL_RADIUS = 2
RAW_TOKEN_PROVENANCE = "RAW_COLNOMIC"
FORBIDDEN_PROPOSAL_TERMS = ("colnomic", "d1", "rank", "slot", "winner", "target", "gap")
NULL_SHA256 = "0" * 64


def tensor_sha256(value: torch.Tensor) -> str:
    item = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(json.dumps(list(item.shape), separators=(",", ":")).encode("ascii"))
    # ``reshape`` is required for scalar optimizer state (for example AdamW's
    # zero-dimensional step tensor); it changes no bytes or hash dialect.
    digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sparsemax(logits: torch.Tensor) -> torch.Tensor:
    """One-dimensional deterministic sparsemax."""
    value = torch.as_tensor(logits)
    if value.ndim != 1 or value.numel() < 2 or not value.is_floating_point():
        raise ValueError("sparsemax expects a floating vector with at least two entries")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("sparsemax input must be finite")
    ordered = torch.sort(value, descending=True).values
    steps = torch.arange(1, value.numel() + 1, dtype=value.dtype, device=value.device)
    cumulative = ordered.cumsum(0)
    support = 1.0 + steps * ordered > cumulative
    count = int(support.sum().detach().cpu())
    threshold = (cumulative[count - 1] - 1.0) / float(count)
    weights = (value - threshold).clamp_min(0.0)
    return weights / weights.sum()


@dataclass(frozen=True)
class RoMaProposalInput:
    """The complete and exclusive Proposal input."""

    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    features: torch.Tensor
    valid_mask: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_coordinates: torch.Tensor
    reference_coordinates: torch.Tensor
    roma_payload_sha256: str
    coordinate_binding_sha256: str
    control_namespace: str = "REAL"
    parent_real_binding_sha256: str = NULL_SHA256
    control_plan_sha256: str = NULL_SHA256
    control_realization_sha256: str = NULL_SHA256


def _positive_grid(shape: tuple[int, int], name: str) -> tuple[int, int]:
    result = tuple(int(item) for item in shape)
    if len(result) != 2 or min(result) <= 0:
        raise ValueError(f"{name} must be a positive two-dimensional grid")
    return result


def _binding_hash(value: RoMaProposalInput) -> str:
    return logical_sha256(
        {
            "query_resource_key": value.query_resource_key,
            "candidate_resource_key": value.candidate_resource_key,
            "reference_resource_key": value.reference_resource_key,
            "query_grid_shape": list(value.query_grid_shape),
            "reference_grid_shape": list(value.reference_grid_shape),
            "features": tensor_sha256(value.features),
            "valid_mask": tensor_sha256(value.valid_mask),
            "query_indices": tensor_sha256(value.query_indices),
            "reference_indices": tensor_sha256(value.reference_indices),
            "query_coordinates": tensor_sha256(value.query_coordinates),
            "reference_coordinates": tensor_sha256(value.reference_coordinates),
            "control_namespace": value.control_namespace,
            "parent_real_binding_sha256": value.parent_real_binding_sha256,
            "control_plan_sha256": value.control_plan_sha256,
            "control_realization_sha256": value.control_realization_sha256,
        }
    )


def seal_proposal_input(value: RoMaProposalInput) -> RoMaProposalInput:
    provisional = replace(value, roma_payload_sha256="PENDING", coordinate_binding_sha256="PENDING")
    payload = logical_sha256(
        {
            "feature_names": list(ROMA_FEATURE_NAMES),
            "features": tensor_sha256(provisional.features),
            "valid_mask": tensor_sha256(provisional.valid_mask),
        }
    )
    provisional = replace(provisional, roma_payload_sha256=payload)
    return replace(provisional, coordinate_binding_sha256=_binding_hash(provisional))


def validate_proposal_input(value: RoMaProposalInput) -> RoMaProposalInput:
    qshape = _positive_grid(value.query_grid_shape, "query grid")
    rshape = _positive_grid(value.reference_grid_shape, "reference grid")
    feature = torch.as_tensor(value.features)
    if feature.ndim != 2 or feature.shape[1] != ROMA_FEATURE_DIM or feature.dtype != torch.float64:
        raise ValueError("RoMa features must be FP64 [N,5]")
    count = feature.shape[0]
    expected = {
        "valid_mask": (torch.bool, (count,)),
        "query_indices": (torch.int64, (count,)),
        "reference_indices": (torch.int64, (count,)),
        "query_coordinates": (torch.int64, (count, 2)),
        "reference_coordinates": (torch.int64, (count, 2)),
    }
    for name, (dtype, shape) in expected.items():
        item = torch.as_tensor(getattr(value, name))
        if item.dtype != dtype or tuple(item.shape) != shape:
            raise ValueError(f"{name} dtype/shape drift")
    if not bool(torch.isfinite(feature).all()):
        raise ValueError("RoMa features contain non-finite values")
    if bool(((feature < 0.0) | (feature > 1.0)).any()):
        raise ValueError("RoMa feature outside [0,1]")
    if not torch.allclose(feature[:, 2], torch.sqrt(feature[:, 0] * feature[:, 1]), atol=1e-12, rtol=1e-12):
        raise ValueError("sqrt overlap product does not replay")
    qidx = value.query_indices
    ridx = value.reference_indices
    if not bool(((qidx >= 0) & (qidx < qshape[0] * qshape[1])).all()):
        raise ValueError("query index outside grid")
    if not bool(((ridx >= 0) & (ridx < rshape[0] * rshape[1])).all()):
        raise ValueError("reference index outside grid")
    expected_q = torch.stack((qidx // qshape[1], qidx % qshape[1]), dim=1)
    expected_r = torch.stack((ridx // rshape[1], ridx % rshape[1]), dim=1)
    if not torch.equal(value.query_coordinates, expected_q) or not torch.equal(value.reference_coordinates, expected_r):
        raise ValueError("index/coordinate binding drift")
    if len(set(value.query_resource_key for _ in (0,))) != 1 or any(
        not isinstance(item, str) or not item
        for item in (value.query_resource_key, value.candidate_resource_key, value.reference_resource_key,
                     value.roma_payload_sha256, value.coordinate_binding_sha256, value.control_namespace)
    ):
        raise ValueError("empty resource/provenance key")
    if not all(_is_sha256(item) for item in (
        value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
    )):
        raise ValueError("Proposal control provenance hash is invalid")
    if value.control_namespace == "REAL":
        if value.candidate_resource_key != value.reference_resource_key:
            raise ValueError("REAL Proposal must bind candidate to its own reference")
        if any(item != NULL_SHA256 for item in (
            value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
        )):
            raise ValueError("REAL Proposal may not carry control-parent provenance")
    elif any(item == NULL_SHA256 for item in (
        value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
    )):
        raise ValueError("controlled Proposal lacks parent/plan provenance")
    payload = logical_sha256(
        {"feature_names": list(ROMA_FEATURE_NAMES), "features": tensor_sha256(feature),
         "valid_mask": tensor_sha256(value.valid_mask)}
    )
    if payload != value.roma_payload_sha256 or _binding_hash(replace(value, coordinate_binding_sha256="PENDING")) != value.coordinate_binding_sha256:
        raise ValueError("Proposal input seal does not replay")
    return value


@dataclass(frozen=True)
class HypothesisSeal:
    state: str
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    atom_indices: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_coordinates: torch.Tensor
    reference_coordinates: torch.Tensor
    source_roma_payload_sha256: str
    source_coordinate_binding_sha256: str
    control_namespace: str
    parent_real_binding_sha256: str
    control_plan_sha256: str
    control_realization_sha256: str
    logical_sha256: str


def _seal_hash(value: HypothesisSeal) -> str:
    return logical_sha256(
        {
            "state": value.state,
            "query_resource_key": value.query_resource_key,
            "candidate_resource_key": value.candidate_resource_key,
            "reference_resource_key": value.reference_resource_key,
            "query_grid_shape": list(value.query_grid_shape),
            "reference_grid_shape": list(value.reference_grid_shape),
            "atom_indices": tensor_sha256(value.atom_indices),
            "query_indices": tensor_sha256(value.query_indices),
            "reference_indices": tensor_sha256(value.reference_indices),
            "query_coordinates": tensor_sha256(value.query_coordinates),
            "reference_coordinates": tensor_sha256(value.reference_coordinates),
            "source_roma_payload_sha256": value.source_roma_payload_sha256,
            "source_coordinate_binding_sha256": value.source_coordinate_binding_sha256,
            "control_namespace": value.control_namespace,
            "parent_real_binding_sha256": value.parent_real_binding_sha256,
            "control_plan_sha256": value.control_plan_sha256,
            "control_realization_sha256": value.control_realization_sha256,
        }
    )


def validate_hypothesis_seal(value: HypothesisSeal) -> HypothesisSeal:
    qshape = _positive_grid(value.query_grid_shape, "sealed query grid")
    rshape = _positive_grid(value.reference_grid_shape, "sealed reference grid")
    count = value.atom_indices.numel()
    if value.state not in ("H0", "H1"):
        raise ValueError("unknown hypothesis state")
    for item in (value.atom_indices, value.query_indices, value.reference_indices):
        if item.dtype != torch.int64 or tuple(item.shape) != (count,):
            raise ValueError("hypothesis index shape/dtype drift")
    for item in (value.query_coordinates, value.reference_coordinates):
        if item.dtype != torch.int64 or tuple(item.shape) != (count, 2):
            raise ValueError("hypothesis coordinate shape/dtype drift")
    if not all(_is_sha256(item) for item in (
        value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
    )):
        raise ValueError("hypothesis control provenance hash is invalid")
    if value.control_namespace == "REAL":
        if value.candidate_resource_key != value.reference_resource_key:
            raise ValueError("REAL hypothesis candidate/reference binding drift")
        if any(item != NULL_SHA256 for item in (
            value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
        )):
            raise ValueError("REAL hypothesis contains control provenance")
    elif any(item == NULL_SHA256 for item in (
        value.parent_real_binding_sha256, value.control_plan_sha256, value.control_realization_sha256
    )):
        raise ValueError("controlled hypothesis lacks parent/plan provenance")
    if count:
        if not bool((value.atom_indices >= 0).all()) or torch.unique(value.atom_indices).numel() != count:
            raise ValueError("hypothesis atom indices must be unique and nonnegative")
        if not bool(((value.query_indices >= 0) & (value.query_indices < qshape[0] * qshape[1])).all()):
            raise ValueError("sealed query index outside grid")
        if not bool(((value.reference_indices >= 0) & (value.reference_indices < rshape[0] * rshape[1])).all()):
            raise ValueError("sealed reference index outside grid")
        expected_q = torch.stack((value.query_indices // qshape[1], value.query_indices % qshape[1]), dim=1)
        expected_r = torch.stack((value.reference_indices // rshape[1], value.reference_indices % rshape[1]), dim=1)
        if not torch.equal(value.query_coordinates, expected_q) or not torch.equal(value.reference_coordinates, expected_r):
            raise ValueError("sealed index/coordinate binding drift")
    if value.state == "H0" and count != 0:
        raise ValueError("H0 must have empty geometry")
    if value.state == "H1":
        if torch.unique(value.query_indices).numel() < MIN_QUERY_PATCHES:
            raise ValueError("H1 has too few query patches")
        if torch.unique(value.reference_indices).numel() < MIN_REFERENCE_CELLS:
            raise ValueError("H1 has too few reference cells")
        if not _is_four_connected(value.query_coordinates):
            raise ValueError("H1 query footprint is not four-connected")
        if len(_paired_components(value.query_coordinates, value.reference_coordinates)) != 1:
            raise ValueError("H1 query-reference geometry is not locally coherent")
    if _seal_hash(replace(value, logical_sha256="PENDING")) != value.logical_sha256:
        raise ValueError("hypothesis seal hash does not replay")
    return value


@dataclass(frozen=True)
class ProposalDecision:
    candidate_resource_key: str
    state: str
    candidate_score: torch.Tensor
    training_score: torch.Tensor
    h0_score: torch.Tensor
    sparse_atom_count: int
    legal_component_count: int
    seal: HypothesisSeal


def _four_components(coordinates: torch.Tensor) -> list[list[int]]:
    rows = [tuple(int(v) for v in row) for row in torch.as_tensor(coordinates).detach().cpu().tolist()]
    cells: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(rows):
        cells.setdefault(cell, []).append(index)
    unseen = set(cells)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        visited = [start]
        while stack:
            row, col = stack.pop()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
                    visited.append(neighbour)
        members: list[int] = []
        for cell in sorted(visited):
            members.extend(cells[cell])
        components.append(sorted(members))
    return components


def _is_four_connected(coordinates: torch.Tensor) -> bool:
    unique = torch.unique(torch.as_tensor(coordinates, dtype=torch.int64), dim=0)
    return unique.numel() > 0 and len(_four_components(unique)) == 1


def _paired_components(query_coordinates: torch.Tensor, reference_coordinates: torch.Tensor) -> list[list[int]]:
    """Components whose query neighbours also agree locally in reference space."""
    query = torch.as_tensor(query_coordinates, dtype=torch.int64).detach().cpu()
    reference = torch.as_tensor(reference_coordinates, dtype=torch.int64).detach().cpu()
    if query.shape != reference.shape or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("paired coordinates must both be [N,2]")
    count = query.shape[0]
    neighbours: list[set[int]] = [set() for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            query_l1 = int((query[left] - query[right]).abs().sum())
            reference_linf = int((reference[left] - reference[right]).abs().max())
            if query_l1 <= 1 and reference_linf <= REFERENCE_LOCAL_RADIUS:
                neighbours[left].add(right)
                neighbours[right].add(left)
    unseen = set(range(count))
    output: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbour in sorted(neighbours[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        output.append(sorted(component))
    return output


def _training_surrogate(
    bank: RoMaProposalInput, logits: torch.Tensor, valid_indices: torch.Tensor
) -> torch.Tensor:
    """Differentiable readiness over legal four-patch paired-geometry seeds.

    The deployed seal remains discrete and exact-H0.  This separate score keeps
    structurally possible H0 examples learnable without adding parameters or
    leaking a retrieval/base feature into P.
    """
    if valid_indices.numel() == 0:
        return bank.features.new_zeros(())
    by_query: dict[tuple[int, int], list[int]] = {}
    for local_index, atom_index in enumerate(valid_indices.detach().cpu().tolist()):
        cell = tuple(int(item) for item in bank.query_coordinates[atom_index].tolist())
        by_query.setdefault(cell, []).append(local_index)
    cells = set(by_query)
    connected_sets: set[frozenset[tuple[int, int]]] = set()
    for start in sorted(cells):
        frontier = {frozenset((start,))}
        for _ in range(MIN_QUERY_PATCHES - 1):
            grown: set[frozenset[tuple[int, int]]] = set()
            for current in frontier:
                candidates = set()
                for row, col in current:
                    candidates.update(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
                for candidate in candidates & cells:
                    if candidate not in current:
                        grown.add(frozenset((*current, candidate)))
            frontier = grown
        connected_sets.update(item for item in frontier if len(item) == MIN_QUERY_PATCHES)
    scores: list[torch.Tensor] = []
    for cell_set in sorted(connected_sets, key=lambda item: sorted(item)):
        chosen_local = []
        for cell in sorted(cell_set):
            candidates = torch.tensor(by_query[cell], dtype=torch.int64, device=logits.device)
            chosen_local.append(candidates[torch.argmax(logits[candidates])])
        local = torch.stack(chosen_local)
        atoms = valid_indices[local]
        if torch.unique(bank.reference_indices[atoms]).numel() < MIN_REFERENCE_CELLS:
            continue
        if len(_paired_components(bank.query_coordinates[atoms], bank.reference_coordinates[atoms])) != 1:
            continue
        scores.append(logits[local].mean())
    if not scores:
        # Geometry that cannot possibly form H1 is not made learnable by a fake
        # scalar target; exact H0 remains the only valid output.
        return logits.sum() * 0.0
    stacked = torch.stack(scores)
    return torch.logsumexp(stacked, dim=0) - math.log(len(scores))


class RoMaConnectedProposal(nn.Module):
    """Six-parameter RoMa-only sparse connected Proposal."""

    def __init__(self) -> None:
        super().__init__()
        self.selector = nn.Linear(ROMA_FEATURE_DIM, 1, bias=True, dtype=torch.float64)

    def forward(self, proposal_input: RoMaProposalInput) -> ProposalDecision:
        bank = validate_proposal_input(proposal_input)
        valid_indices = torch.nonzero(bank.valid_mask, as_tuple=False).flatten()
        zero = bank.features.new_zeros(())
        if valid_indices.numel() == 0:
            return self._h0(bank, zero, zero, 0)
        logits = self.selector(bank.features[valid_indices]).flatten()
        training_score = _training_surrogate(bank, logits, valid_indices)
        masses = sparsemax(torch.cat((logits, zero.reshape(1))))
        selected_local = torch.nonzero(masses[:-1] > 0.0, as_tuple=False).flatten()
        selected = valid_indices[selected_local]
        if selected.numel() == 0:
            return self._h0(bank, zero, training_score, 0)
        components = _paired_components(bank.query_coordinates[selected], bank.reference_coordinates[selected])
        legal: list[tuple[torch.Tensor, torch.Tensor]] = []
        for members in components:
            local = torch.tensor(members, dtype=torch.int64, device=selected.device)
            atoms = selected[local]
            if torch.unique(bank.query_indices[atoms]).numel() < MIN_QUERY_PATCHES:
                continue
            if torch.unique(bank.reference_indices[atoms]).numel() < MIN_REFERENCE_CELLS:
                continue
            component_logits = logits[selected_local[local]]
            component_mass = masses[:-1][selected_local[local]]
            score = (component_mass * component_logits).sum() / component_mass.sum()
            legal.append((score, atoms))
        if not legal:
            return self._h0(bank, zero, training_score, int(selected.numel()))
        best_index = max(
            range(len(legal)),
            key=lambda idx: (float(legal[idx][0].detach().cpu()), -int(legal[idx][1].min().detach().cpu())),
        )
        best_score, atoms = legal[best_index]
        if not bool(best_score > zero):
            return self._h0(bank, zero, training_score, int(selected.numel()), len(legal))
        seal = self._make_seal(bank, "H1", atoms)
        return ProposalDecision(bank.candidate_resource_key, "H1", best_score, training_score, zero,
                                int(selected.numel()), len(legal), seal)

    @staticmethod
    def _make_seal(bank: RoMaProposalInput, state: str, atoms: torch.Tensor) -> HypothesisSeal:
        atoms = atoms.detach().cpu().to(torch.int64).sort().values
        provisional = HypothesisSeal(
            state=state,
            query_resource_key=bank.query_resource_key,
            candidate_resource_key=bank.candidate_resource_key,
            reference_resource_key=bank.reference_resource_key,
            query_grid_shape=bank.query_grid_shape,
            reference_grid_shape=bank.reference_grid_shape,
            atom_indices=atoms,
            query_indices=bank.query_indices[atoms].detach().cpu(),
            reference_indices=bank.reference_indices[atoms].detach().cpu(),
            query_coordinates=bank.query_coordinates[atoms].detach().cpu(),
            reference_coordinates=bank.reference_coordinates[atoms].detach().cpu(),
            source_roma_payload_sha256=bank.roma_payload_sha256,
            source_coordinate_binding_sha256=bank.coordinate_binding_sha256,
            control_namespace=bank.control_namespace,
            parent_real_binding_sha256=bank.parent_real_binding_sha256,
            control_plan_sha256=bank.control_plan_sha256,
            control_realization_sha256=bank.control_realization_sha256,
            logical_sha256="PENDING",
        )
        return replace(provisional, logical_sha256=_seal_hash(provisional))

    def _h0(self, bank: RoMaProposalInput, zero: torch.Tensor, training_score: torch.Tensor,
            sparse_count: int,
            legal_count: int = 0) -> ProposalDecision:
        empty = torch.empty(0, dtype=torch.int64)
        seal = self._make_seal(bank, "H0", empty)
        return ProposalDecision(bank.candidate_resource_key, "H0", zero, training_score, zero,
                                sparse_count, legal_count, seal)


def proposal_api_is_isolated() -> bool:
    field_names = tuple(item.name.lower() for item in fields(RoMaProposalInput))
    forward_names = tuple(RoMaConnectedProposal.forward.__code__.co_varnames[:2])
    return not any(term in name for name in field_names + forward_names for term in FORBIDDEN_PROPOSAL_TERMS)


def score_candidate_axis(model: RoMaConnectedProposal, banks: Sequence[RoMaProposalInput]) -> tuple[ProposalDecision, ...]:
    keys = [item.candidate_resource_key for item in banks]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate axis contains duplicate opaque keys")
    query_keys = {item.query_resource_key for item in banks}
    if len(query_keys) != 1:
        raise ValueError("candidate axis spans multiple queries")
    return tuple(model(item) for item in banks)


def _derangement(count: int, namespace: str) -> torch.Tensor:
    if count < 2:
        raise ValueError("a derangement requires at least two entries")
    keys = [hashlib.sha256(f"{namespace}:{index}".encode()).digest() for index in range(count)]
    ordered = sorted(range(count), key=lambda index: keys[index])
    for shift in range(1, count):
        permutation = torch.empty(count, dtype=torch.int64)
        for position, source in enumerate(ordered):
            permutation[source] = ordered[(position + shift) % count]
        if not bool((permutation == torch.arange(count)).any()):
            return permutation
    raise RuntimeError("failed to construct fixed-point-free permutation")


def _key_derangement(keys: Sequence[str], namespace: str) -> torch.Tensor:
    """Destination-ordinal to donor-ordinal map stable under axis reorder."""
    if len(keys) < 2 or len(keys) != len(set(keys)) or any(not key for key in keys):
        raise ValueError("key derangement requires at least two unique nonempty keys")
    canonical = sorted(keys, key=lambda key: (hashlib.sha256(f"{namespace}:{key}".encode()).digest(), key))
    donor_by_key = {key: canonical[(index + 1) % len(canonical)] for index, key in enumerate(canonical)}
    ordinal_by_key = {key: index for index, key in enumerate(keys)}
    result = torch.tensor([ordinal_by_key[donor_by_key[key]] for key in keys], dtype=torch.int64)
    if bool((result == torch.arange(len(keys))).any()):
        raise RuntimeError("key derangement contains fixed point")
    return result


def candidate_binding_before_proposal(
    banks: Sequence[RoMaProposalInput], namespace: str = "C_BIND_E0_V1"
) -> tuple[tuple[RoMaProposalInput, ...], torch.Tensor]:
    if len({item.query_resource_key for item in banks}) != 1:
        raise ValueError("C_BIND banks must share one query")
    if any(validate_proposal_input(item).control_namespace != "REAL" for item in banks):
        raise ValueError("C_BIND must start from REAL Proposal inputs")
    keys = tuple(item.candidate_resource_key for item in banks)
    permutation = _key_derangement(keys, namespace)
    plan_sha256 = logical_sha256(
        {
            "control": "C_BIND_BEFORE_P",
            "namespace": namespace,
            "destination_to_donor": {
                keys[destination]: keys[donor] for destination, donor in enumerate(permutation.tolist())
            },
        }
    )
    output = []
    for destination, donor_index in enumerate(permutation.tolist()):
        donor = banks[donor_index]
        realization_sha256 = logical_sha256(
            {
                "control_plan_sha256": plan_sha256,
                "destination_candidate_key": banks[destination].candidate_resource_key,
                "donor_reference_key": donor.reference_resource_key,
                "parent_real_binding_sha256": donor.coordinate_binding_sha256,
            }
        )
        output.append(
            seal_proposal_input(
                replace(
                    donor,
                    candidate_resource_key=banks[destination].candidate_resource_key,
                    control_namespace=namespace,
                    parent_real_binding_sha256=donor.coordinate_binding_sha256,
                    control_plan_sha256=plan_sha256,
                    control_realization_sha256=realization_sha256,
                )
            )
        )
    return tuple(output), permutation


def destroy_coordinate_endpoint(
    bank: RoMaProposalInput, side: str, namespace: str | None = None
) -> tuple[RoMaProposalInput, torch.Tensor]:
    validate_proposal_input(bank)
    if side not in ("query", "reference"):
        raise ValueError("side must be query or reference")
    shape = bank.query_grid_shape if side == "query" else bank.reference_grid_shape
    count = shape[0] * shape[1]
    control = namespace or f"{side.upper()}_COORD_E0_V1"
    if bank.control_namespace != "REAL":
        raise ValueError("coordinate destruction must start from a REAL Proposal input")
    permutation = _derangement(count, control)
    source = bank.query_indices if side == "query" else bank.reference_indices
    active = permutation[source]
    coordinates = torch.stack((active // shape[1], active % shape[1]), dim=1)
    plan_sha256 = logical_sha256({"control": f"{side.upper()}_COORD", "namespace": control})
    realization_sha256 = logical_sha256(
        {"control_plan_sha256": plan_sha256, "grid_shape": list(shape),
         "permutation": permutation.tolist(), "parent_real_binding_sha256": bank.coordinate_binding_sha256}
    )
    kwargs = {
        f"{side}_indices": active,
        f"{side}_coordinates": coordinates,
        "control_namespace": control,
        "parent_real_binding_sha256": bank.coordinate_binding_sha256,
        "control_plan_sha256": plan_sha256,
        "control_realization_sha256": realization_sha256,
    }
    return seal_proposal_input(replace(bank, **kwargs)), permutation


@dataclass(frozen=True)
class RawTokenGrid:
    resource_key: str
    grid_shape: tuple[int, int]
    tokens: torch.Tensor
    provenance: str
    payload_sha256: str
    source_receipt_sha256: str
    source_contract_sha256: str
    token_space_sha256: str


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _raw_payload_hash(value: RawTokenGrid) -> str:
    return logical_sha256(
        {
            "resource_key": value.resource_key,
            "grid_shape": list(value.grid_shape),
            "tokens": tensor_sha256(value.tokens),
            "provenance": value.provenance,
            "source_receipt_sha256": value.source_receipt_sha256,
            "source_contract_sha256": value.source_contract_sha256,
            "token_space_sha256": value.token_space_sha256,
        }
    )


def seal_raw_token_grid(value: RawTokenGrid) -> RawTokenGrid:
    provisional = replace(value, payload_sha256="PENDING")
    return replace(provisional, payload_sha256=_raw_payload_hash(provisional))


def validate_raw_token_grid(value: RawTokenGrid) -> RawTokenGrid:
    shape = _positive_grid(value.grid_shape, "token grid")
    token = torch.as_tensor(value.tokens)
    if value.provenance != RAW_TOKEN_PROVENANCE:
        raise ValueError("verifier accepts RAW_COLNOMIC provenance only")
    if not all(_is_sha256(item) for item in (
        value.source_receipt_sha256, value.source_contract_sha256, value.token_space_sha256
    )):
        raise ValueError("RAW token source receipt/contract hash is invalid")
    if token.ndim != 2 or token.shape[0] != shape[0] * shape[1] or not token.is_floating_point():
        raise ValueError("RAW token/grid shape drift")
    if not bool(torch.isfinite(token).all()):
        raise ValueError("RAW token contains non-finite values")
    if value.payload_sha256 != _raw_payload_hash(replace(value, payload_sha256="PENDING")):
        raise ValueError("RAW token payload hash does not replay")
    return value


@dataclass(frozen=True)
class CandidateProposalBundle:
    proposal_input: RoMaProposalInput
    reference_tokens: RawTokenGrid


def candidate_binding_bundle_before_proposal(
    bundles: Sequence[CandidateProposalBundle], namespace: str = "C_BIND_BUNDLE_E0_V1"
) -> tuple[tuple[CandidateProposalBundle, ...], torch.Tensor]:
    for bundle in bundles:
        validate_proposal_input(bundle.proposal_input)
        validate_raw_token_grid(bundle.reference_tokens)
        if bundle.proposal_input.reference_resource_key != bundle.reference_tokens.resource_key:
            raise ValueError("bundle reference payload binding mismatch")
    banks, permutation = candidate_binding_before_proposal(
        tuple(bundle.proposal_input for bundle in bundles), namespace
    )
    output = tuple(
        CandidateProposalBundle(banks[destination], bundles[donor].reference_tokens)
        for destination, donor in enumerate(permutation.tolist())
    )
    return output, permutation


@dataclass(frozen=True)
class PairVerification:
    candidate_g: str
    candidate_w: str
    union_query_indices: torch.Tensor
    score_g: torch.Tensor
    score_w: torch.Tensor
    margin_g_minus_w: torch.Tensor
    support_g: torch.Tensor
    support_w: torch.Tensor
    contributions_g: torch.Tensor
    contributions_w: torch.Tensor


def _candidate_on_union(
    seal: HypothesisSeal, union: torch.Tensor, query: RawTokenGrid, reference: RawTokenGrid
) -> tuple[torch.Tensor, torch.Tensor]:
    token_q = query.tokens.to(dtype=torch.float64)
    token_r = reference.tokens.to(dtype=torch.float64)
    values = token_q.new_zeros(union.numel())
    support = torch.zeros(union.numel(), dtype=torch.bool)
    if seal.state == "H0":
        return values, support
    for position, query_index in enumerate(union.tolist()):
        atoms = torch.nonzero(seal.query_indices == query_index, as_tuple=False).flatten()
        if atoms.numel() == 0:
            continue
        reference_indices = seal.reference_indices[atoms]
        query_vector = token_q[query_index].expand(reference_indices.numel(), -1)
        values[position] = F.cosine_similarity(query_vector, token_r[reference_indices], dim=1).mean()
        support[position] = True
    return values, support


def verify_pair_on_shared_query_union(
    hypothesis_g: HypothesisSeal,
    hypothesis_w: HypothesisSeal,
    query_tokens: RawTokenGrid,
    reference_tokens_g: RawTokenGrid,
    reference_tokens_w: RawTokenGrid,
) -> PairVerification:
    g = validate_hypothesis_seal(hypothesis_g)
    w = validate_hypothesis_seal(hypothesis_w)
    query = validate_raw_token_grid(query_tokens)
    ref_g = validate_raw_token_grid(reference_tokens_g)
    ref_w = validate_raw_token_grid(reference_tokens_w)
    if g.query_resource_key != w.query_resource_key or query.resource_key != g.query_resource_key:
        raise ValueError("verifier query binding mismatch")
    if g.query_grid_shape != w.query_grid_shape or query.grid_shape != g.query_grid_shape:
        raise ValueError("verifier query grid binding mismatch")
    if ref_g.resource_key != g.reference_resource_key or ref_w.resource_key != w.reference_resource_key:
        raise ValueError("verifier reference binding mismatch")
    if ref_g.grid_shape != g.reference_grid_shape or ref_w.grid_shape != w.reference_grid_shape:
        raise ValueError("verifier reference grid binding mismatch")
    if g.candidate_resource_key == w.candidate_resource_key:
        raise ValueError("verifier pair must contain distinct candidates")
    if g.control_namespace != w.control_namespace or g.control_plan_sha256 != w.control_plan_sha256:
        raise ValueError("verifier may not mix REAL/control namespaces or plans")
    if len({query.tokens.shape[1], ref_g.tokens.shape[1], ref_w.tokens.shape[1]}) != 1:
        raise ValueError("verifier RAW token feature dimension mismatch")
    if len({query.token_space_sha256, ref_g.token_space_sha256, ref_w.token_space_sha256}) != 1:
        raise ValueError("verifier RAW token-space mismatch")
    union = torch.unique(torch.cat((g.query_indices, w.query_indices))).sort().values
    if union.numel() == 0:
        zero = query.tokens.to(dtype=torch.float64).new_zeros(())
        empty = torch.empty(0, dtype=torch.bool)
        empty_values = query.tokens.to(dtype=torch.float64).new_empty(0)
        return PairVerification(g.candidate_resource_key, w.candidate_resource_key, union, zero, zero, zero,
                                empty, empty, empty_values, empty_values)
    values_g, support_g = _candidate_on_union(g, union, query, ref_g)
    values_w, support_w = _candidate_on_union(w, union, query, ref_w)
    score_g, score_w = values_g.mean(), values_w.mean()
    return PairVerification(g.candidate_resource_key, w.candidate_resource_key, union,
                            score_g, score_w, score_g - score_w, support_g, support_w,
                            values_g, values_w)


def fixture_bank(
    candidate_key: str,
    high_cells: Sequence[int] = (7, 8, 13, 14),
    *,
    query_key: str = "synthetic-query",
    reference_key: str | None = None,
    high: bool = True,
) -> RoMaProposalInput:
    shape = (6, 6)
    count = 36
    query_indices = torch.arange(count, dtype=torch.int64)
    reference_indices = query_indices.clone()
    query_coordinates = torch.stack((query_indices // 6, query_indices % 6), dim=1)
    reference_coordinates = query_coordinates.clone()
    features = torch.empty((count, 5), dtype=torch.float64)
    features[:, 0] = 0.08
    features[:, 1] = 0.07
    features[:, 2] = math.sqrt(0.08 * 0.07)
    features[:, 3] = math.exp(-4.0 * 0.45)
    features[:, 4] = 0.06
    if high:
        for offset, index in enumerate(high_cells):
            overlap_q = 0.96 - 0.01 * offset
            overlap_r = 0.93 - 0.01 * offset
            features[index] = torch.tensor(
                [overlap_q, overlap_r, math.sqrt(overlap_q * overlap_r),
                 math.exp(-4.0 * (0.006 + 0.001 * offset)), 0.95 - 0.01 * offset],
                dtype=torch.float64,
            )
    provisional = RoMaProposalInput(
        query_resource_key=query_key,
        candidate_resource_key=candidate_key,
        reference_resource_key=reference_key or candidate_key,
        query_grid_shape=shape,
        reference_grid_shape=shape,
        features=features,
        valid_mask=torch.ones(count, dtype=torch.bool),
        query_indices=query_indices,
        reference_indices=reference_indices,
        query_coordinates=query_coordinates,
        reference_coordinates=reference_coordinates,
        roma_payload_sha256="PENDING",
        coordinate_binding_sha256="PENDING",
    )
    return seal_proposal_input(provisional)


def configured_fixture_model() -> RoMaConnectedProposal:
    """Behavior fixture only; never an authorized natural initialization."""
    model = RoMaConnectedProposal()
    with torch.no_grad():
        model.selector.weight.copy_(torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.6]], dtype=torch.float64))
        model.selector.bias.fill_(-2.7)
    return model


def zero_initialized_model() -> RoMaConnectedProposal:
    """Frozen initialization candidate for a separately authorized successor."""
    model = RoMaConnectedProposal()
    with torch.no_grad():
        model.selector.weight.zero_()
        model.selector.bias.zero_()
    return model


def fixture_raw_tokens(resource_key: str, shift: int = 0, dimension: int = 36) -> RawTokenGrid:
    tokens = torch.eye(36, dimension, dtype=torch.float64)
    if shift:
        tokens = torch.roll(tokens, shifts=shift, dims=0)
    receipt = hashlib.sha256(f"synthetic-receipt:{resource_key}".encode()).hexdigest()
    contract = hashlib.sha256(b"synthetic-raw-colnomic-contract-v1").hexdigest()
    token_space = hashlib.sha256(b"synthetic-raw-colnomic-token-space-v1").hexdigest()
    return seal_raw_token_grid(
        RawTokenGrid(resource_key, (6, 6), tokens, RAW_TOKEN_PROVENANCE,
                     "PENDING", receipt, contract, token_space)
    )


def model_state_sha256(model: nn.Module) -> str:
    return logical_sha256({name: tensor_sha256(value) for name, value in sorted(model.state_dict().items())})


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    serial: dict[str, object] = {"param_groups": state["param_groups"]}
    serial["state"] = {
        str(key): {name: tensor_sha256(value) if isinstance(value, torch.Tensor) else value
                   for name, value in sorted(payload.items())}
        for key, payload in sorted(state["state"].items())
    }
    return logical_sha256(serial)


def training_update(model: RoMaConnectedProposal, optimizer: torch.optim.Optimizer,
                    bank: RoMaProposalInput, target: float = 1.0) -> float:
    optimizer.zero_grad(set_to_none=True)
    score = model(bank).training_score
    loss = F.binary_cross_entropy_with_logits(score.reshape(1), score.new_tensor([target]))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def pairwise_training_update(
    model: RoMaConnectedProposal,
    optimizer: torch.optim.Optimizer,
    positive: RoMaProposalInput,
    negative: RoMaProposalInput,
) -> float:
    """Pairwise selectivity plus symmetric absolute H0 anchors.

    The anchors identify the fixed zero point that a relative loss alone
    cannot identify.  Only legal-seed training scores enter this loss.
    """
    optimizer.zero_grad(set_to_none=True)
    positive_score = model(positive).training_score
    negative_score = model(negative).training_score
    loss = (
        F.softplus(-(positive_score - negative_score))
        + 0.5 * F.softplus(-positive_score)
        + 0.5 * F.softplus(negative_score)
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


__all__ = [
    "FORBIDDEN_PROPOSAL_TERMS", "MIN_QUERY_PATCHES", "MIN_REFERENCE_CELLS",
    "RAW_TOKEN_PROVENANCE", "REFERENCE_LOCAL_RADIUS", "ROMA_FEATURE_DIM", "ROMA_FEATURE_NAMES",
    "SELECTOR_PARAMETER_COUNT", "HypothesisSeal", "PairVerification", "ProposalDecision",
    "CandidateProposalBundle", "RawTokenGrid", "RoMaConnectedProposal", "RoMaProposalInput",
    "candidate_binding_before_proposal", "candidate_binding_bundle_before_proposal",
    "configured_fixture_model", "destroy_coordinate_endpoint",
    "fixture_bank", "fixture_raw_tokens", "logical_sha256", "model_state_sha256",
    "optimizer_state_sha256", "pairwise_training_update", "proposal_api_is_isolated", "score_candidate_axis",
    "seal_proposal_input", "seal_raw_token_grid", "sparsemax", "tensor_sha256", "training_update",
    "validate_hypothesis_seal", "validate_proposal_input", "validate_raw_token_grid",
    "verify_pair_on_shared_query_union",
    "zero_initialized_model",
]
