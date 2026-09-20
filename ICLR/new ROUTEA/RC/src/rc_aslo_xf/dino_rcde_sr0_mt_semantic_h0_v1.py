"""Additive semantic-H0 core for SR0-MT.

P structural availability, P proposal ranking, V candidate-unary evidence and
query-level HOLD are separate contracts.  The unary evidence reuses the
existing bias-free V ``signed_head`` over candidate-specific
``relational = raw_summary - null_summary``; no new identity parameter or free
gate is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


SEMANTIC_H0_SCORE = 0.0
HARD_MAX_TIE_ULPS = 16
TRAIN_NULL_NAMESPACE = "RCDE_SR0_MT_SEMANTIC_H0_TRAIN_NULL_V1"
FORMAL_CONTROL_NAMESPACE = "RCDE_SR0_MT_SEMANTIC_H0_FORMAL_CONTROL_V1"
ALLOWED_NULL_CONTROL_KINDS = frozenset(
    {
        "TRAIN_CANDIDATE_BINDING_DERANGEMENT",
        "TRAIN_REFERENCE_BINDING_DERANGEMENT",
    }
)
STRUCTURAL_H0 = "STRUCTURAL_PROPOSAL_UNAVAILABLE"
PROPOSAL_READY = "STRUCTURAL_PROPOSAL_READY"
SEMANTIC_H0 = "ABSENT_OR_UNVERIFIED"
SEMANTIC_H1 = "VERIFIED_PRESENT"
HOLD_NO_MATCH = "HOLD_ABSENT_OR_UNVERIFIED"
HOLD_MULTI_H1 = "PRIMARY_UNDETERMINED_HOLD"
SINGLE_H1 = "SINGLE_VERIFIED_PRESENT_REQUIRES_DEPLOYMENT_GATE"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SemanticH0Error(RuntimeError):
    pass


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SemanticH0Error(message)


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _scalar(value: torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    _require(
        result.ndim == 0
        and result.is_floating_point()
        and bool(torch.isfinite(result)),
        f"{name} must be one finite floating scalar",
    )
    return result


def _positive_tolerance(value: torch.Tensor) -> float:
    detached = _scalar(value, name="semantic score").detach()
    return (
        HARD_MAX_TIE_ULPS
        * torch.finfo(detached.dtype).eps
        * max(1.0, abs(float(detached)))
    )


@dataclass(frozen=True)
class StructuralProposal:
    status: str
    row_index: int | None
    row_sha256: str | None
    row_score: torch.Tensor


def select_structural_proposal(
    row_scores: torch.Tensor,
    row_ready: torch.Tensor,
    row_sha256s: Sequence[str],
) -> StructuralProposal:
    """Select the best legal P row without interpreting score sign as H0."""

    scores = torch.as_tensor(row_scores)
    ready = torch.as_tensor(row_ready, dtype=torch.bool, device=scores.device)
    hashes = tuple(row_sha256s)
    _require(
        scores.ndim == 1
        and scores.is_floating_point()
        and bool(torch.isfinite(scores).all())
        and ready.shape == scores.shape
        and len(hashes) == scores.numel()
        and all(_valid_sha(item) for item in hashes),
        "structural proposal population drift",
    )
    legal = torch.nonzero(ready, as_tuple=False).flatten().tolist()
    if not legal:
        return StructuralProposal(STRUCTURAL_H0, None, None, scores.square().sum() * 0.0)
    detached = scores.detach()
    maximum = detached[torch.tensor(legal, dtype=torch.long, device=scores.device)].max()
    tolerance = (
        HARD_MAX_TIE_ULPS
        * torch.finfo(detached.dtype).eps
        * max(1.0, abs(float(maximum)))
    )
    tied = [
        index
        for index in legal
        if abs(float(detached[index]) - float(maximum)) <= tolerance
    ]
    index = min(tied, key=lambda item: hashes[item])
    return StructuralProposal(PROPOSAL_READY, index, hashes[index], scores[index])


def candidate_unary_evidence(
    signed_head: torch.nn.Module,
    relational: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    structural_ready: bool,
) -> torch.Tensor:
    """Expose a candidate-alone V score from the existing relational field."""

    _require(isinstance(signed_head, torch.nn.Linear), "V signed_head must be Linear")
    _require(
        signed_head.out_features == 1 and signed_head.bias is None,
        "V candidate unary requires one bias-free signed head",
    )
    value = torch.as_tensor(relational)
    mask = torch.as_tensor(query_mask, dtype=torch.bool, device=value.device).flatten()
    _require(
        value.ndim == 2
        and value.shape == (mask.numel(), signed_head.in_features)
        and value.is_floating_point()
        and bool(torch.isfinite(value).all()),
        "candidate unary relational population drift",
    )
    if not structural_ready:
        return signed_head.weight.square().sum() * 0.0
    _require(bool(mask.any()), "candidate unary query mask is empty")
    selected = value[mask]
    if bool(selected.detach().eq(0.0).all()):
        return signed_head.weight.square().sum() * 0.0
    patch_scores = signed_head(value).squeeze(-1)
    return patch_scores[mask].mean()


@dataclass(frozen=True)
class SemanticCandidateDecision:
    status: str
    evidence: torch.Tensor


def decide_candidate(evidence: torch.Tensor) -> SemanticCandidateDecision:
    score = _scalar(evidence, name="candidate evidence")
    status = SEMANTIC_H1 if float(score.detach()) > _positive_tolerance(score) else SEMANTIC_H0
    return SemanticCandidateDecision(status, score)


def decide_bidirectional_candidate(
    a_to_b_evidence: torch.Tensor,
    b_to_a_evidence: torch.Tensor,
) -> SemanticCandidateDecision:
    """Both independently decoded directions must beat semantic H0."""

    first = _scalar(a_to_b_evidence, name="a-to-b candidate evidence")
    second = _scalar(b_to_a_evidence, name="b-to-a candidate evidence")
    _require(first.device == second.device, "direction device drift")
    fused = 0.5 * (first + second.to(dtype=first.dtype))
    verified = (
        decide_candidate(first).status == SEMANTIC_H1
        and decide_candidate(second).status == SEMANTIC_H1
    )
    return SemanticCandidateDecision(SEMANTIC_H1 if verified else SEMANTIC_H0, fused)


@dataclass(frozen=True)
class NullControlReceipt:
    query_id: str
    query_source_image_sha256: str
    candidate_key: str
    candidate_reference_source_sha256: str
    direction: str
    control_kind: str
    namespace: str
    seed: int
    donor_or_permutation_sha256: str
    clean_record_sha256: str
    control_record_sha256: str
    prejoin_ledger_sha256: str
    fixed_point_free: bool
    changed_binding: bool

    def __post_init__(self) -> None:
        _require(isinstance(self.query_id, str) and bool(self.query_id), "null query id")
        _require(isinstance(self.candidate_key, str) and bool(self.candidate_key), "null candidate key")
        _require(
            all(
                _valid_sha(value)
                for value in (
                    self.query_source_image_sha256,
                    self.candidate_reference_source_sha256,
                    self.donor_or_permutation_sha256,
                    self.clean_record_sha256,
                    self.control_record_sha256,
                    self.prejoin_ledger_sha256,
                )
            ),
            "null control SHA binding",
        )
        _require(self.direction in ("a_to_b", "b_to_a"), "null direction")
        _require(self.namespace == TRAIN_NULL_NAMESPACE, "train/formal-control namespace collision")
        _require(self.namespace != FORMAL_CONTROL_NAMESPACE, "formal control entered training")
        _require(self.control_kind in ALLOWED_NULL_CONTROL_KINDS, "unqualified null control kind")
        _require(isinstance(self.seed, int) and self.seed >= 0, "null seed")
        _require(self.clean_record_sha256 != self.control_record_sha256, "null control is byte-identical to clean")
        _require(self.fixed_point_free is True and self.changed_binding is True, "null fixed point/address alias")


@dataclass(frozen=True)
class SemanticNullControl:
    evidence: torch.Tensor
    receipt: NullControlReceipt

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _scalar(self.evidence, name="null-control evidence"))
        _require(isinstance(self.receipt, NullControlReceipt), "null-control receipt absent")


@dataclass(frozen=True)
class SemanticH0Loss:
    pairwise_ranking: torch.Tensor
    exact_match_vs_h0: torch.Tensor
    null_control_vs_h0: torch.Tensor
    total: torch.Tensor


def semantic_h0_bidirectional_training_loss(
    *,
    target_vs_natural_rival_pair_logit: torch.Tensor,
    exact_match_a_to_b: torch.Tensor,
    exact_match_b_to_a: torch.Tensor,
    null_controls_a_to_b: Sequence[SemanticNullControl],
    null_controls_b_to_a: Sequence[SemanticNullControl],
) -> SemanticH0Loss:
    """Pair ranking plus per-direction unary match/null calibration."""

    pair_logit = _scalar(target_vs_natural_rival_pair_logit, name="natural pair logit")
    target_a = _scalar(exact_match_a_to_b, name="exact match a-to-b")
    target_b = _scalar(exact_match_b_to_a, name="exact match b-to-a")
    _require(pair_logit.device == target_a.device == target_b.device, "loss device drift")
    controls_a = tuple(null_controls_a_to_b)
    controls_b = tuple(null_controls_b_to_a)
    _require(bool(controls_a) and bool(controls_b), "both directions require null controls")
    _require(all(item.receipt.direction == "a_to_b" for item in controls_a), "a-to-b null direction drift")
    _require(all(item.receipt.direction == "b_to_a" for item in controls_b), "b-to-a null direction drift")
    null_scores = torch.stack(
        [
            item.evidence.to(device=target_a.device, dtype=target_a.dtype)
            for item in controls_a + controls_b
        ]
    )
    pair = F.softplus(-pair_logit)
    positive = 0.5 * (F.softplus(-target_a) + F.softplus(-target_b))
    negative = F.softplus(null_scores).mean()
    total = pair + positive + negative
    return SemanticH0Loss(pair, positive, negative, total)


@dataclass(frozen=True)
class QueryH0Decision:
    status: str
    candidate_key: str | None
    semantic_h1_candidates: tuple[str, ...]


def decide_query(
    candidate_directional_evidence: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> QueryH0Decision:
    """Target-free conservative decision before no-regret deployment gates."""

    _require(bool(candidate_directional_evidence), "candidate population is empty")
    positives = []
    for key, directions in candidate_directional_evidence.items():
        _require(isinstance(key, str) and bool(key), "candidate key is invalid")
        _require(isinstance(directions, tuple) and len(directions) == 2, "candidate directions absent")
        if decide_bidirectional_candidate(directions[0], directions[1]).status == SEMANTIC_H1:
            positives.append(key)
    ordered = tuple(sorted(positives))
    if not ordered:
        return QueryH0Decision(HOLD_NO_MATCH, None, ordered)
    if len(ordered) > 1:
        return QueryH0Decision(HOLD_MULTI_H1, None, ordered)
    return QueryH0Decision(SINGLE_H1, ordered[0], ordered)


__all__ = [
    "SEMANTIC_H0_SCORE",
    "TRAIN_NULL_NAMESPACE",
    "FORMAL_CONTROL_NAMESPACE",
    "ALLOWED_NULL_CONTROL_KINDS",
    "STRUCTURAL_H0",
    "PROPOSAL_READY",
    "SEMANTIC_H0",
    "SEMANTIC_H1",
    "HOLD_NO_MATCH",
    "HOLD_MULTI_H1",
    "SINGLE_H1",
    "SemanticH0Error",
    "StructuralProposal",
    "select_structural_proposal",
    "candidate_unary_evidence",
    "SemanticCandidateDecision",
    "decide_candidate",
    "decide_bidirectional_candidate",
    "NullControlReceipt",
    "SemanticNullControl",
    "SemanticH0Loss",
    "semantic_h0_bidirectional_training_loss",
    "QueryH0Decision",
    "decide_query",
]

