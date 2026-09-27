"""P-immutable RAW-ColNomic three-arm Verification core.

This module never creates, scores, ranks, expands, or moves a Proposal.  It
consumes an already frozen connected P view and measures the same candidate
pair with three fixed RAW-ColNomic scopes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .current_d1_roma_rawlocal_hyp_e0_v1 import (
    HypothesisSeal,
    RawTokenGrid,
    validate_hypothesis_seal,
    validate_raw_token_grid,
)


ARM_ALL = "ALL_PATCH_RAW_COLNOMIC"
ARM_QUERY_FULL = "FROZEN_QUERY_REGION_FULL_REFERENCE_RAW_COLNOMIC"
ARM_PAIRED = "FROZEN_PAIRED_REGION_RAW_COLNOMIC"
ARMS = (ARM_ALL, ARM_QUERY_FULL, ARM_PAIRED)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _logical(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != "logical_sha256"})
    ).hexdigest()


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")
    return value


def _tensor_sha(value: torch.Tensor) -> str:
    item = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(_canonical(list(item.shape)))
    digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _connected(indices: torch.Tensor, shape: tuple[int, int]) -> bool:
    cells = {(int(index) // shape[1], int(index) % shape[1]) for index in indices.tolist()}
    if not cells:
        return False
    unseen = set(cells)
    stack = [unseen.pop()]
    seen = set(stack)
    while stack:
        row, col = stack.pop()
        for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbour in unseen:
                unseen.remove(neighbour)
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == len(cells)


@dataclass(frozen=True)
class FrozenPView:
    state: str
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    source_p_logical_sha256: str
    source_p_authority_sha256: str
    control_namespace: str
    logical_sha256: str


def _view_hash(value: FrozenPView) -> str:
    return _logical(
        {
            "state": value.state,
            "query_resource_key": value.query_resource_key,
            "candidate_resource_key": value.candidate_resource_key,
            "reference_resource_key": value.reference_resource_key,
            "query_grid_shape": list(value.query_grid_shape),
            "reference_grid_shape": list(value.reference_grid_shape),
            "query_indices": _tensor_sha(value.query_indices),
            "reference_indices": _tensor_sha(value.reference_indices),
            "source_p_logical_sha256": value.source_p_logical_sha256,
            "source_p_authority_sha256": value.source_p_authority_sha256,
            "control_namespace": value.control_namespace,
        }
    )


def seal_frozen_p_view(value: FrozenPView) -> FrozenPView:
    return replace(value, logical_sha256=_view_hash(replace(value, logical_sha256="PENDING")))


def validate_frozen_p_view(value: FrozenPView) -> FrozenPView:
    q = torch.as_tensor(value.query_indices).detach().cpu().to(torch.int64).contiguous()
    r = torch.as_tensor(value.reference_indices).detach().cpu().to(torch.int64).contiguous()
    qshape = tuple(int(x) for x in value.query_grid_shape)
    rshape = tuple(int(x) for x in value.reference_grid_shape)
    if (
        value.state not in ("H0", "H1")
        or len(qshape) != 2
        or len(rshape) != 2
        or min((*qshape, *rshape)) <= 0
        or q.ndim != 1
        or r.ndim != 1
        or q.shape != r.shape
        or not all(
            isinstance(item, str) and item
            for item in (
                value.query_resource_key,
                value.candidate_resource_key,
                value.reference_resource_key,
                value.control_namespace,
            )
        )
    ):
        raise ValueError("frozen P view envelope invalid")
    _sha(value.source_p_logical_sha256)
    _sha(value.source_p_authority_sha256)
    if bool(((q < 0) | (q >= math.prod(qshape))).any()) or bool(
        ((r < 0) | (r >= math.prod(rshape))).any()
    ):
        raise ValueError("frozen P indices outside grid")
    if value.state == "H0" and q.numel() != 0:
        raise ValueError("H0 view must be empty")
    if value.state == "H1" and (
        torch.unique(q).numel() < 4
        or torch.unique(r).numel() < 2
        or not _connected(torch.unique(q), qshape)
    ):
        raise ValueError("H1 view is not a connected multi-patch hypothesis")
    normalized = replace(value, query_indices=q, reference_indices=r)
    if value.logical_sha256 != _view_hash(replace(normalized, logical_sha256="PENDING")):
        raise ValueError("frozen P view hash drift")
    return normalized


def view_from_hypothesis_seal(
    value: HypothesisSeal, *, source_p_authority_sha256: str
) -> FrozenPView:
    item = validate_hypothesis_seal(value)
    provisional = FrozenPView(
        state=item.state,
        query_resource_key=item.query_resource_key,
        candidate_resource_key=item.candidate_resource_key,
        reference_resource_key=item.reference_resource_key,
        query_grid_shape=item.query_grid_shape,
        reference_grid_shape=item.reference_grid_shape,
        query_indices=item.query_indices,
        reference_indices=item.reference_indices,
        source_p_logical_sha256=item.logical_sha256,
        source_p_authority_sha256=_sha(source_p_authority_sha256),
        control_namespace=item.control_namespace,
        logical_sha256="PENDING",
    )
    return validate_frozen_p_view(seal_frozen_p_view(provisional))


@dataclass(frozen=True)
class ArmPairScore:
    arm_name: str
    candidate_a: str
    candidate_b: str
    query_indices: torch.Tensor
    score_a: torch.Tensor
    score_b: torch.Tensor
    margin_a_minus_b: torch.Tensor
    contributions_a: torch.Tensor
    contributions_b: torch.Tensor
    support_a: torch.Tensor
    support_b: torch.Tensor


@dataclass(frozen=True)
class ThreeArmPairScore:
    candidate_a: str
    candidate_b: str
    p_view_a_sha256: str
    p_view_b_sha256: str
    arms: tuple[ArmPairScore, ArmPairScore, ArmPairScore]

    def by_name(self) -> dict[str, ArmPairScore]:
        return {item.arm_name: item for item in self.arms}


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.to(torch.float64), dim=1, eps=1.0e-12)


def _all_patch(query: RawTokenGrid, reference: RawTokenGrid) -> torch.Tensor:
    return (_normalized(query.tokens) @ _normalized(reference.tokens).T).max(dim=1).values


def _region_full(
    view: FrozenPView, union: torch.Tensor, query: RawTokenGrid, reference: RawTokenGrid
) -> tuple[torch.Tensor, torch.Tensor]:
    values = query.tokens.new_zeros(union.numel(), dtype=torch.float64)
    support = torch.zeros(union.numel(), dtype=torch.bool)
    if view.state == "H0":
        return values, support
    qn, rn = _normalized(query.tokens), _normalized(reference.tokens)
    active = set(int(x) for x in view.query_indices.tolist())
    for position, index in enumerate(union.tolist()):
        if index in active:
            values[position] = (qn[index] @ rn.T).max()
            support[position] = True
    return values, support


def _region_paired(
    view: FrozenPView, union: torch.Tensor, query: RawTokenGrid, reference: RawTokenGrid
) -> tuple[torch.Tensor, torch.Tensor]:
    values = query.tokens.new_zeros(union.numel(), dtype=torch.float64)
    support = torch.zeros(union.numel(), dtype=torch.bool)
    if view.state == "H0":
        return values, support
    qn, rn = _normalized(query.tokens), _normalized(reference.tokens)
    for position, index in enumerate(union.tolist()):
        atoms = torch.nonzero(view.query_indices == index, as_tuple=False).flatten()
        if atoms.numel():
            values[position] = (qn[index].expand(atoms.numel(), -1) * rn[view.reference_indices[atoms]]).sum(1).mean()
            support[position] = True
    return values, support


def _arm(
    name: str,
    candidate_a: str,
    candidate_b: str,
    indices: torch.Tensor,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    support_a: torch.Tensor,
    support_b: torch.Tensor,
) -> ArmPairScore:
    if indices.numel() == 0:
        score_a = values_a.new_zeros(())
        score_b = values_b.new_zeros(())
    else:
        score_a, score_b = values_a.mean(), values_b.mean()
    return ArmPairScore(
        name,
        candidate_a,
        candidate_b,
        indices,
        score_a,
        score_b,
        score_a - score_b,
        values_a,
        values_b,
        support_a,
        support_b,
    )


def verify_frozen_p_three_arm_pair(
    view_a: FrozenPView,
    view_b: FrozenPView,
    query_tokens: RawTokenGrid,
    reference_tokens_a: RawTokenGrid,
    reference_tokens_b: RawTokenGrid,
) -> ThreeArmPairScore:
    a, b = validate_frozen_p_view(view_a), validate_frozen_p_view(view_b)
    q = validate_raw_token_grid(query_tokens)
    ra, rb = validate_raw_token_grid(reference_tokens_a), validate_raw_token_grid(reference_tokens_b)
    if (
        a.query_resource_key != b.query_resource_key
        or q.resource_key != a.query_resource_key
        or a.query_grid_shape != b.query_grid_shape
        or q.grid_shape != a.query_grid_shape
        or ra.resource_key != a.reference_resource_key
        or rb.resource_key != b.reference_resource_key
        or ra.grid_shape != a.reference_grid_shape
        or rb.grid_shape != b.reference_grid_shape
        or a.candidate_resource_key == b.candidate_resource_key
        or a.control_namespace != b.control_namespace
        or a.source_p_authority_sha256 != b.source_p_authority_sha256
        or len({q.token_space_sha256, ra.token_space_sha256, rb.token_space_sha256}) != 1
    ):
        raise ValueError("three-arm pair binding drift")
    all_indices = torch.arange(q.tokens.shape[0], dtype=torch.int64)
    all_a, all_b = _all_patch(q, ra), _all_patch(q, rb)
    all_support = torch.ones(all_indices.numel(), dtype=torch.bool)
    union = torch.unique(torch.cat((a.query_indices, b.query_indices))).sort().values
    qf_a, qf_sa = _region_full(a, union, q, ra)
    qf_b, qf_sb = _region_full(b, union, q, rb)
    pr_a, pr_sa = _region_paired(a, union, q, ra)
    pr_b, pr_sb = _region_paired(b, union, q, rb)
    arms = (
        _arm(ARM_ALL, a.candidate_resource_key, b.candidate_resource_key, all_indices, all_a, all_b, all_support, all_support.clone()),
        _arm(ARM_QUERY_FULL, a.candidate_resource_key, b.candidate_resource_key, union, qf_a, qf_b, qf_sa, qf_sb),
        _arm(ARM_PAIRED, a.candidate_resource_key, b.candidate_resource_key, union, pr_a, pr_b, pr_sa, pr_sb),
    )
    if tuple(item.arm_name for item in arms) != ARMS or any(
        not bool(torch.isfinite(torch.stack((item.score_a, item.score_b, item.margin_a_minus_b))).all())
        or not torch.equal(item.margin_a_minus_b, item.score_a - item.score_b)
        for item in arms
    ):
        raise RuntimeError("three-arm numeric closure failed")
    return ThreeArmPairScore(
        a.candidate_resource_key,
        b.candidate_resource_key,
        a.logical_sha256,
        b.logical_sha256,
        arms,
    )


__all__ = [
    "ARM_ALL",
    "ARM_PAIRED",
    "ARM_QUERY_FULL",
    "ARMS",
    "ArmPairScore",
    "FrozenPView",
    "ThreeArmPairScore",
    "seal_frozen_p_view",
    "validate_frozen_p_view",
    "verify_frozen_p_three_arm_pair",
    "view_from_hypothesis_seal",
]
