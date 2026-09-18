"""Wrong-direction-limited P-only utility over the frozen legal H1 family."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


BASE_PATH = Path(__file__).with_name(
    "reference_conditioned_latent_proposal_legal_family_v2.py"
)
BASE_SHA256 = "ff36b44b9115f3b38d7fca4ca3f92708f78de456bf375992586975e29109e983"
FEATURE_DIM = 9
PARAMETER_COUNT = 10
IDENTITY_CHANNEL_COUNT = 3
GEOMETRY_DIM = 5


def _load_base() -> ModuleType:
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError("frozen legal-family V2 hash drift")
    name = "rc_lth_p_only_gisc_legal_family_base_v2"
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen legal-family V2 import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
ProposalAtomBank = _base.ProposalAtomBank
validate_atom_bank = _base.validate_atom_bank
seal_atom_bank = _base.seal_atom_bank
coordinate_destroy_atom_bank = _base.coordinate_destroy_atom_bank
query_only_features = _base.query_only_features
tensor_sha256 = _base.tensor_sha256
logical_sha256 = _base.logical_sha256
LegalHypothesisSeed = _base.LegalHypothesisSeed
ScoredLegalHypothesis = _base.ScoredLegalHypothesis
LegalFamilyDecision = _base.LegalFamilyDecision
LegalFamilyCache = _base.LegalFamilyCache
enumerate_legal_hypotheses = _base.enumerate_legal_hypotheses
build_legal_family_cache = _base.build_legal_family_cache
install_legal_family_cache = _base.install_legal_family_cache
AllPatchNoHypComparator = _base.AllPatchNoHypComparator
QueryOnlyRegionComparator = _base.QueryOnlyRegionComparator
STRUCTURAL_H0_REASON = _base.STRUCTURAL_H0_REASON


class GeometryGatedIdentitySignedHead(nn.Module):
    """Ten FP64 parameters whose output cannot oppose centered identity contrast."""

    def __init__(self) -> None:
        super().__init__()
        self.identity_logits = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.geometry_weight = nn.Parameter(torch.zeros(5, dtype=torch.float64))
        self.geometry_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.scale_raw = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(
            atom_features, dtype=torch.float64, device=self.identity_logits.device
        )
        if features.ndim < 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError("GISC atom feature axis must end in nine columns")
        centred = features[..., 1]
        reciprocal = features[..., 4]
        cycle_quality = features[..., 5]
        channels = torch.stack(
            (
                centred,
                (1.0 + reciprocal) * centred,
                (1.0 + reciprocal * cycle_quality) * centred,
            ),
            dim=-1,
        )
        identity = (torch.softmax(self.identity_logits, dim=0) * channels).sum(dim=-1)
        geometry = features[..., (2, 3, 4, 5, 6)]
        gate = 1.0 + torch.sigmoid(
            geometry @ self.geometry_weight + self.geometry_bias
        )
        scale = 1.0 + F.softplus(self.scale_raw)
        return scale * gate * identity


def _validate_head(head: nn.Module) -> None:
    parameters = tuple(head.parameters())
    if not isinstance(head, GeometryGatedIdentitySignedHead):
        raise TypeError("REAL GISC head type drift")
    if sum(value.numel() for value in parameters) != PARAMETER_COUNT:
        raise ValueError("GISC head must have exactly ten parameters")
    if any(value.dtype != torch.float64 for value in parameters):
        raise ValueError("GISC head parameters must be FP64")


def _cache(bank: ProposalAtomBank) -> LegalFamilyCache:
    cached = _base._FAMILY_CACHE.get(bank.assignment_sha256)
    if cached is None:
        cached = build_legal_family_cache(validate_atom_bank(bank))
        install_legal_family_cache((cached,))
    return cached


def score_legal_hypothesis_family(
    bank: ProposalAtomBank, head: GeometryGatedIdentitySignedHead
) -> LegalFamilyDecision:
    """Use mean atom utility per H and logmeanexp over the complete legal family."""
    _validate_head(head)
    cached = _cache(bank)
    device = next(head.parameters()).device
    if cached.legal_seed_atom_indices.shape[0]:
        atom_scores = head(torch.as_tensor(bank.features, dtype=torch.float64, device=device))
        atom_axis = cached.legal_seed_atom_indices.to(device=device)
        scores = atom_scores[atom_axis].mean(dim=1)
        family = torch.logsumexp(scores, dim=0) - math.log(scores.numel())
        maximum = scores.detach().max()
        tied = [index for index, value in enumerate(scores.detach()) if bool(value == maximum)]
        chosen = min(
            tied,
            key=lambda index: tuple(
                int(value) for value in cached.legal_seed_atom_indices[index].tolist()
            ),
        )
        axis = cached.legal_seed_atom_indices[chosen]
        seed_hash = _base._seed_hash(bank.assignment_sha256, axis)
        seed = LegalHypothesisSeed(
            chosen,
            axis,
            bank.query_indices[axis],
            bank.reference_indices[axis],
            bank.query_rc[axis],
            bank.reference_rc[axis],
            seed_hash,
        )
        mapped = ScoredLegalHypothesis(chosen + 1, seed, scores[chosen])
        hypotheses = (mapped,)
        reason = None
    else:
        zero = sum(
            (parameter.sum() * 0.0 for parameter in head.parameters()),
            torch.zeros((), dtype=torch.float64, device=device),
        )
        scores = torch.empty(0, dtype=torch.float64, device=device)
        family = zero
        hypotheses = ()
        mapped = None
        chosen = None
        reason = STRUCTURAL_H0_REASON
    return LegalFamilyDecision(
        bank.query_resource_key,
        bank.candidate_resource_key,
        bank.reference_resource_key,
        hypotheses,
        family,
        bool(cached.legal_seed_atom_indices.shape[0]),
        reason,
        int(cached.legal_seed_atom_indices.shape[0]),
        cached.legal_seed_atom_indices,
        cached.legal_seed_axis_sha256,
        scores,
        tensor_sha256(scores.detach().cpu().contiguous()),
        chosen,
        None if mapped is None else mapped.seed.logical_sha256,
        mapped,
        bank.assignment_sha256,
        bank.source_assignment_sha256,
        bank.coordinate_binding_sha256,
    )


def deploy_legal_hypothesis_family(
    bank: ProposalAtomBank, head: GeometryGatedIdentitySignedHead
) -> LegalFamilyDecision:
    return score_legal_hypothesis_family(bank, head)


class ConnectedProposalSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = GeometryGatedIdentitySignedHead()

    def forward(self, bank: ProposalAtomBank) -> LegalFamilyDecision:
        return score_legal_hypothesis_family(bank, self.head)


def score_reference_axis(
    selector: ConnectedProposalSelector, banks: Sequence[ProposalAtomBank]
) -> tuple[torch.Tensor, tuple[LegalFamilyDecision, ...]]:
    query_keys = {bank.query_resource_key for bank in banks}
    candidate_keys = [bank.candidate_resource_key for bank in banks]
    if len(query_keys) != 1 or len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate axis invalid")
    proposals = tuple(selector(bank) for bank in banks)
    return torch.stack([proposal.candidate_score for proposal in proposals]), proposals


def serialize_family_decision(value: LegalFamilyDecision) -> dict[str, Any]:
    return _base.serialize_family_decision(value)


def validate_serialized_family_decision(
    value: Mapping[str, Any],
    bank: ProposalAtomBank,
    head: GeometryGatedIdentitySignedHead,
) -> dict[str, Any]:
    replay = serialize_family_decision(deploy_legal_hypothesis_family(bank, head))
    if dict(value) != replay:
        raise ValueError("serialized GISC legal-family parity mismatch")
    return replay


__all__ = [
    "ProposalAtomBank",
    "validate_atom_bank",
    "seal_atom_bank",
    "coordinate_destroy_atom_bank",
    "query_only_features",
    "AllPatchNoHypComparator",
    "QueryOnlyRegionComparator",
    "tensor_sha256",
    "LegalHypothesisSeed",
    "ScoredLegalHypothesis",
    "LegalFamilyDecision",
    "LegalFamilyCache",
    "enumerate_legal_hypotheses",
    "build_legal_family_cache",
    "install_legal_family_cache",
    "GeometryGatedIdentitySignedHead",
    "score_legal_hypothesis_family",
    "deploy_legal_hypothesis_family",
    "serialize_family_decision",
    "validate_serialized_family_decision",
    "ConnectedProposalSelector",
    "score_reference_axis",
    "STRUCTURAL_H0_REASON",
    "FEATURE_DIM",
    "PARAMETER_COUNT",
]
