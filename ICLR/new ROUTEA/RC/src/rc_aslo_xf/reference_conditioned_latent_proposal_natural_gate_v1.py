"""Pure training/statistics core for the RC-LTH P-only natural gate.

The module deliberately has no package-relative imports and performs no file,
image, scheduler, retrieval-base, verification, action, or ownership I/O.  The
P-only core is supplied as an object or loaded directly from its source file so
importing this module never executes :mod:`rc_aslo_xf.__init__`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ARM_REAL = "REAL"
ARM_ALLPATCH = "ALL_PATCH_NO_HYP"
ARM_QUERY_ONLY = "QUERY_ONLY_REGION"
ARM_NAMES = (ARM_REAL, ARM_ALLPATCH, ARM_QUERY_ONLY)

HEAD_PARAMETER_COUNT = 10
TOTAL_PARAMETER_COUNT = 30
TRAIN_QUERY_COUNT = 32
CANDIDATE_COUNT = 128
TRAIN_UPDATES = 512
TRAIN_EPOCHS = 16
TRAIN_SEED = 17
LEARNING_RATE = 0.03
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
C_BIND_NAMESPACE = "RC_LTH_P_ONLY_TRAIN32_C_BIND_BEFORE_ROMA_V1"
P_COORD_NAMESPACE_PREFIX = "RC_LTH_P_ONLY_TRAIN32_P_COORD_V1"

FAILURE_PRECEDENCE = (
    "P_GENERATION_COVERAGE_NO_GO",
    "P_H0_COLLAPSE_ALL_H0",
    "P_H0_COLLAPSE_ALWAYS_H1",
    "P_NONSELECTIVE_RANK_NO_GO",
    "P_RERANKER_SHORTCUT_NO_GO",
    "P_CANDIDATE_BINDING_NO_GO",
    "P_CANDIDATE_BOUND_NONSPATIAL",
    "P_OOF_GENERALIZATION_NO_GO",
)


def _jcs(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _logical_sha256(value: object) -> str:
    return hashlib.sha256(_jcs(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    item = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(_jcs(list(item.shape)))
    digest.update(item.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_standalone_p_core(source_path: str | Path | None = None) -> ModuleType:
    """Load the P core by file path without importing ``rc_aslo_xf``."""
    path = (
        Path(source_path)
        if source_path is not None
        else Path(__file__).with_name("reference_conditioned_latent_proposal_p_only_v1.py")
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_rc_lth_p_only_standalone_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load P-only core from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    required = (
        "ConnectedProposalSelector",
        "AllPatchNoHypComparator",
        "QueryOnlyRegionComparator",
        "query_only_features",
        "score_reference_axis",
        "tensor_sha256",
        "validate_atom_bank",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        raise ImportError(f"P-only core API is incomplete: {missing}")
    return module


@dataclass(frozen=True)
class AnonymousNaturalEpisode:
    """Role-free tensors and opaque destination bindings for one query."""

    query_resource_key: str
    fold: int
    candidate_resource_keys: tuple[str, ...]
    real_banks: tuple[Any, ...]
    c_bind_banks: tuple[Any, ...]
    p_coord_banks: tuple[Any, ...]
    c_bind_source_keys: tuple[str, ...]
    c_bind_plan_sha256: str
    c_bind_operator_stage: str
    c_bind_sealed_atom_or_score_shuffle_count: int
    atom_bank_receipt_sha256: str
    atom_bank_validation_sha256: str
    p_coord_namespace: str
    p_coord_derivation_sha256: str
    query_tokens: torch.Tensor
    query_grid_shape: tuple[int, int]
    query_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        count = len(self.candidate_resource_keys)
        if not self.query_resource_key or count < 2 or len(set(self.candidate_resource_keys)) != count:
            raise ValueError("anonymous episode requires unique opaque candidate keys")
        if any(len(branch) != count for branch in (self.real_banks, self.c_bind_banks, self.p_coord_banks)):
            raise ValueError("anonymous episode control axes drift")
        if (
            len(self.c_bind_source_keys) != count
            or any(not isinstance(item, str) or not item for item in self.c_bind_source_keys)
            or len(self.c_bind_plan_sha256) != 64
            or self.c_bind_operator_stage != "BEFORE_ROMA_AND_ASSIGNMENT"
            or self.c_bind_sealed_atom_or_score_shuffle_count != 0
            or len(self.atom_bank_receipt_sha256) != 64
            or len(self.atom_bank_validation_sha256) != 64
            or not self.p_coord_namespace.startswith(P_COORD_NAMESPACE_PREFIX + ":")
            or len(self.p_coord_derivation_sha256) != 64
        ):
            raise ValueError("anonymous episode control receipt drift")
        shape = tuple(int(item) for item in self.query_grid_shape)
        tokens = torch.as_tensor(self.query_tokens)
        valid = torch.as_tensor(self.query_valid_mask)
        if len(shape) != 2 or min(shape) <= 0 or tokens.ndim != 2 or tokens.shape[0] != math.prod(shape):
            raise ValueError("anonymous query token/grid axis drift")
        if valid.shape != (tokens.shape[0],) or valid.dtype != torch.bool:
            raise ValueError("anonymous query valid mask drift")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ValueError("anonymous fold must be a non-negative integer")


@dataclass(frozen=True)
class PostjoinProposalRole:
    """The only joined fields accepted by this core."""

    target_destination_indices: tuple[int, ...]
    supergroup: str

    def validate(self, candidate_count: int) -> "PostjoinProposalRole":
        if any(isinstance(item, bool) for item in self.target_destination_indices):
            raise ValueError("target destinations must be integer indices, not booleans")
        indices = tuple(int(item) for item in self.target_destination_indices)
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(not 0 <= item < candidate_count for item in indices)
            or not self.supergroup
        ):
            raise ValueError("invalid postjoin target destinations or supergroup")
        return self


class ThreeArmProposalHeads(nn.Module):
    """Three disjoint, zero-initialized, FP64 ten-scalar heads."""

    def __init__(self, p_core: ModuleType) -> None:
        super().__init__()
        self.real = p_core.ConnectedProposalSelector()
        self.allpatch = p_core.AllPatchNoHypComparator()
        self.query_only = p_core.QueryOnlyRegionComparator()
        self._validate()

    def arms(self) -> Mapping[str, nn.Module]:
        return {
            ARM_REAL: self.real,
            ARM_ALLPATCH: self.allpatch,
            ARM_QUERY_ONLY: self.query_only,
        }

    def _validate(self) -> None:
        for name, module in self.arms().items():
            parameters = tuple(module.parameters())
            if sum(item.numel() for item in parameters) != HEAD_PARAMETER_COUNT:
                raise ValueError(f"{name} head is not exactly ten parameters")
            if any(item.dtype != torch.float64 for item in parameters):
                raise ValueError(f"{name} head is not FP64")
            if any(int(torch.count_nonzero(item.detach())) != 0 for item in parameters):
                raise ValueError(f"{name} head is not exact-zero initialized")
        if sum(item.numel() for item in self.parameters()) != TOTAL_PARAMETER_COUNT:
            raise ValueError("three-arm parameter total drift")


def make_optimizers(heads: ThreeArmProposalHeads) -> Mapping[str, torch.optim.AdamW]:
    optimizers = {
        name: torch.optim.AdamW(
            module.parameters(),
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False,
        )
        for name, module in heads.arms().items()
    }
    for optimizer in optimizers.values():
        defaults = optimizer.defaults
        if not (
            defaults["lr"] == LEARNING_RATE
            and defaults["betas"] == ADAM_BETAS
            and defaults["eps"] == ADAM_EPS
            and defaults["weight_decay"] == 0.0
            and defaults["amsgrad"] is False
            and defaults["maximize"] is False
        ):
            raise RuntimeError("AdamW contract drift")
    return optimizers


def deterministic_training_order(
    query_resource_keys: Sequence[str], contract_v2_sha256: str
) -> tuple[str, ...]:
    keys = tuple(str(item) for item in query_resource_keys)
    if len(keys) != TRAIN_QUERY_COUNT or len(set(keys)) != TRAIN_QUERY_COUNT:
        raise ValueError("training order requires exactly 32 unique query resource keys")
    if len(contract_v2_sha256) != 64:
        raise ValueError("contract SHA-256 is malformed")
    output: list[str] = []
    for epoch in range(TRAIN_EPOCHS):
        output.extend(
            sorted(
                keys,
                key=lambda key: (
                    hashlib.sha256(
                        _jcs([contract_v2_sha256, "TRAIN_ORDER", TRAIN_SEED, epoch, key])
                    ).digest(),
                    key,
                ),
            )
        )
    if len(output) != TRAIN_UPDATES:
        raise RuntimeError("training-order length drift")
    return tuple(output)


def _chosen_score(
    scores: torch.Tensor, indices: Sequence[int], opaque_keys: Sequence[str]
) -> tuple[torch.Tensor, int]:
    values = torch.as_tensor(scores)
    allowed = tuple(int(item) for item in indices)
    if values.ndim != 1 or not allowed:
        raise ValueError("strongest-score selection requires a nonempty one-dimensional axis")
    chosen = min(allowed, key=lambda item: (-float(values[item].detach().cpu()), opaque_keys[item]))
    return values[chosen], chosen


@dataclass(frozen=True)
class ArmForward:
    name: str
    scores: torch.Tensor
    c_bind_scores: torch.Tensor
    p_coord_scores: torch.Tensor
    target_score: torch.Tensor
    c_bind_target_score: torch.Tensor
    p_coord_target_score: torch.Tensor
    strongest_wrong_score: torch.Tensor
    strongest_wrong_index: int
    target_witness_score: torch.Tensor
    target_witness_missing: bool
    margin: torch.Tensor
    c_bind_margin: torch.Tensor
    p_coord_margin: torch.Tensor
    h1_decisions: tuple[bool, ...] | None
    allpatch_p_coord_exact: bool


def arm_forward_from_scores(
    *,
    name: str,
    scores: torch.Tensor,
    c_bind_scores: torch.Tensor,
    p_coord_scores: torch.Tensor,
    target_indices: Sequence[int],
    opaque_keys: Sequence[str],
    target_witness_score: torch.Tensor,
    target_witness_missing: bool,
    h1_decisions: Sequence[bool] | None = None,
) -> ArmForward:
    clean = torch.as_tensor(scores, dtype=torch.float64)
    c_bind = torch.as_tensor(c_bind_scores, dtype=torch.float64, device=clean.device)
    p_coord = torch.as_tensor(p_coord_scores, dtype=torch.float64, device=clean.device)
    if clean.ndim != 1 or c_bind.shape != clean.shape or p_coord.shape != clean.shape:
        raise ValueError("arm score axes drift")
    if len(opaque_keys) != clean.numel() or len(set(opaque_keys)) != clean.numel():
        raise ValueError("arm opaque-key axis drift")
    target = tuple(int(item) for item in target_indices)
    wrong = tuple(index for index in range(clean.numel()) if index not in set(target))
    if not target or not wrong:
        raise ValueError("arm needs target and all non-target destinations")
    target_score, _ = _chosen_score(clean, target, opaque_keys)
    wrong_score, wrong_index = _chosen_score(clean, wrong, opaque_keys)
    c_target, _ = _chosen_score(c_bind, target, opaque_keys)
    c_wrong, _ = _chosen_score(c_bind, wrong, opaque_keys)
    p_target, _ = _chosen_score(p_coord, target, opaque_keys)
    p_wrong, _ = _chosen_score(p_coord, wrong, opaque_keys)
    decisions = None if h1_decisions is None else tuple(bool(item) for item in h1_decisions)
    if decisions is not None and len(decisions) != clean.numel():
        raise ValueError("H1-decision axis drift")
    allpatch_exact = name != ARM_ALLPATCH or torch.equal(clean.detach(), p_coord.detach())
    if name == ARM_ALLPATCH and not allpatch_exact:
        raise ValueError("coordinate-free all-patch arm changed under P_COORD")
    return ArmForward(
        name=name,
        scores=clean,
        c_bind_scores=c_bind,
        p_coord_scores=p_coord,
        target_score=target_score,
        c_bind_target_score=c_target,
        p_coord_target_score=p_target,
        strongest_wrong_score=wrong_score,
        strongest_wrong_index=wrong_index,
        target_witness_score=torch.as_tensor(target_witness_score, dtype=torch.float64, device=clean.device),
        target_witness_missing=bool(target_witness_missing),
        margin=target_score - wrong_score,
        c_bind_margin=c_target - c_wrong,
        p_coord_margin=p_target - p_wrong,
        h1_decisions=decisions,
        allpatch_p_coord_exact=allpatch_exact,
    )


def _target_witness_from_proposals(
    proposals: Sequence[Any], target_indices: Sequence[int], reference: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    values: list[tuple[torch.Tensor, int, int]] = []
    for candidate in target_indices:
        for hypothesis in proposals[int(candidate)].hypotheses:
            values.append((hypothesis.score, int(candidate), int(hypothesis.slot)))
    if not values:
        return torch.zeros((), dtype=torch.float64, device=reference.device).detach(), True
    chosen = min(values, key=lambda item: (-float(item[0].detach().cpu()), item[1], item[2]))
    return chosen[0], False


def _query_component_scores(bank: Any, selection: Any, module: nn.Module) -> tuple[torch.Tensor, ...]:
    device = next(module.parameters()).device
    evidence = torch.as_tensor(bank.signed_evidence, dtype=torch.float64, device=device)
    inverse = torch.empty(bank.query_indices.numel(), dtype=torch.long)
    inverse[torch.as_tensor(bank.query_indices, dtype=torch.long)] = torch.arange(bank.query_indices.numel())
    output: list[torch.Tensor] = []
    for group in selection.groups:
        cells = torch.tensor(group, dtype=torch.long)
        rows = inverse[cells].to(device)
        weights = selection.weights[cells.to(device)]
        output.append((weights * evidence[rows]).sum() / weights.sum().clamp_min(1.0e-12))
    return tuple(output)


def _fixed_point_free_non_affine(coordinates: torch.Tensor, permutation: torch.Tensor) -> bool:
    xy = torch.as_tensor(coordinates, dtype=torch.float64)
    perm = torch.as_tensor(permutation, dtype=torch.long)
    count = perm.numel()
    if (
        xy.shape != (count, 2)
        or torch.unique(xy, dim=0).shape[0] < 4
        or not torch.equal(torch.sort(perm).values, torch.arange(count))
        or bool((perm == torch.arange(count)).any())
    ):
        return False
    design = torch.cat((xy, torch.ones((count, 1), dtype=torch.float64)), dim=1)
    solution = torch.linalg.lstsq(design, xy[perm]).solution
    residual = torch.linalg.vector_norm(design @ solution - xy[perm], dim=1).amax()
    return bool(residual > 1.0e-10)


def _same_source_content(left: Any, right: Any, p_core: ModuleType) -> bool:
    tensor_fields = (
        "features",
        "signed_evidence",
        "valid_mask",
        "query_valid_axis",
        "reference_valid_axis",
        "source_query_indices",
        "source_reference_indices",
        "query_coordinate_axis",
        "reference_coordinate_axis",
        "query_centres_xy_axis",
        "reference_centres_xy_axis",
        "source_query_rc",
        "source_reference_rc",
        "source_forward_reference_xy",
        "source_reverse_query_xy",
        "overlap_pair",
        "precision_quality",
        "cycle_error",
    )
    return (
        left.reference_resource_key == right.reference_resource_key
        and left.source_binding_sha256 == right.source_binding_sha256
        and left.query_grid_shape == right.query_grid_shape
        and left.reference_grid_shape == right.reference_grid_shape
        and all(
            p_core.tensor_sha256(getattr(left, field)) == p_core.tensor_sha256(getattr(right, field))
            for field in tensor_fields
        )
    )


def _validate_control_axes(episode: AnonymousNaturalEpisode, p_core: ModuleType) -> None:
    real_references: list[str] = []
    bound_references: list[str] = []
    for destination, key in enumerate(episode.candidate_resource_keys):
        real = p_core.validate_atom_bank(episode.real_banks[destination])
        c_bind = p_core.validate_atom_bank(episode.c_bind_banks[destination])
        p_coord = p_core.validate_atom_bank(episode.p_coord_banks[destination])
        if any(bank.query_resource_key != episode.query_resource_key for bank in (real, c_bind, p_coord)):
            raise ValueError("query resource escaped its episode")
        if any(bank.candidate_resource_key != key for bank in (real, c_bind, p_coord)):
            raise ValueError("destination candidate binding drift")
        if p_coord.reference_resource_key != real.reference_resource_key:
            raise ValueError("P_COORD changed reference content")
        if not (
            p_core.tensor_sha256(p_coord.features) == p_core.tensor_sha256(real.features)
            and p_core.tensor_sha256(p_coord.signed_evidence)
            == p_core.tensor_sha256(real.signed_evidence)
            and p_coord.source_assignment_sha256 == real.source_assignment_sha256
        ):
            raise ValueError("P_COORD changed source atom values or provenance")
        if (
            p_coord.control_namespace != episode.p_coord_namespace
            or p_coord.control_namespace == "REAL"
            or p_coord.assignment_sha256 == real.assignment_sha256
            or p_coord.coordinate_binding_sha256 == real.coordinate_binding_sha256
            or not _fixed_point_free_non_affine(
                p_coord.query_coordinate_axis, p_coord.query_coordinate_permutation
            )
            or not _fixed_point_free_non_affine(
                p_coord.reference_coordinate_axis, p_coord.reference_coordinate_permutation
            )
            or not torch.equal(
                p_coord.query_indices,
                p_coord.query_coordinate_permutation[p_coord.source_query_indices],
            )
            or not torch.equal(
                p_coord.reference_indices,
                p_coord.reference_coordinate_permutation[p_coord.source_reference_indices],
            )
        ):
            raise ValueError("P_COORD destruction receipt or active association is invalid")
        real_references.append(real.reference_resource_key)
        bound_references.append(c_bind.reference_resource_key)
    if sorted(real_references) != sorted(bound_references) or any(
        left == right for left, right in zip(real_references, bound_references, strict=True)
    ):
        raise ValueError("C_BIND is not a complete fixed-point-free reference permutation")
    expected_sources = tuple(episode.c_bind_source_keys)
    if tuple(bound_references) != expected_sources:
        raise ValueError("C_BIND donor source-key receipt drift")
    source_index = {key: index for index, key in enumerate(real_references)}
    if set(source_index) != set(expected_sources):
        raise ValueError("C_BIND donor source population drift")
    for destination, source_key in enumerate(expected_sources):
        if not _same_source_content(episode.c_bind_banks[destination], episode.real_banks[source_index[source_key]], p_core):
            raise ValueError("C_BIND complete donor content does not replay before assignment")
    expected_plan_sha256 = _logical_sha256(
        {
            "namespace": C_BIND_NAMESPACE,
            "query_resource_key": episode.query_resource_key,
            "candidate_keys": list(episode.candidate_resource_keys),
            "source_for_destination": [source_index[key] for key in expected_sources],
        }
    )
    if expected_plan_sha256 != episode.c_bind_plan_sha256:
        raise ValueError("C_BIND pre-assignment donor-plan hash drift")
    expected_p_coord_sha256 = _logical_sha256(
        {
            "algorithm": "coordinate_destroy_atom_bank",
            "namespace": episode.p_coord_namespace,
            "real_source_assignment_sha256": [bank.source_assignment_sha256 for bank in episode.real_banks],
        }
    )
    if expected_p_coord_sha256 != episode.p_coord_derivation_sha256:
        raise ValueError("P_COORD derivation hash drift")


@dataclass(frozen=True)
class EpisodeForward:
    query_resource_key: str
    fold: int
    supergroup: str
    target_indices: tuple[int, ...]
    candidate_resource_keys: tuple[str, ...]
    arms: Mapping[str, ArmForward]
    raw_candidate_atom_present: tuple[bool, ...]
    selected_connected_h1_present: tuple[bool, ...]
    selected_positive_h1_present: tuple[bool, ...]


def score_episode(
    episode: AnonymousNaturalEpisode,
    role: PostjoinProposalRole,
    heads: ThreeArmProposalHeads,
    p_core: ModuleType,
) -> EpisodeForward:
    """Run all three arms from explicit REAL/C_BIND/P_COORD atom banks."""
    count = len(episode.candidate_resource_keys)
    role.validate(count)
    _validate_control_axes(episode, p_core)
    target = role.target_destination_indices

    real_scores, real_proposals = p_core.score_reference_axis(heads.real, episode.real_banks)
    c_scores, _ = p_core.score_reference_axis(heads.real, episode.c_bind_banks)
    p_scores, _ = p_core.score_reference_axis(heads.real, episode.p_coord_banks)
    real_witness, real_missing = _target_witness_from_proposals(real_proposals, target, real_scores)

    all_scores = torch.stack([heads.allpatch(bank) for bank in episode.real_banks])
    all_c_scores = torch.stack([heads.allpatch(bank) for bank in episode.c_bind_banks])
    all_p_scores = torch.stack([heads.allpatch(bank) for bank in episode.p_coord_banks])
    all_witness = torch.zeros((), dtype=torch.float64, device=all_scores.device).detach()

    query_features, query_rc = p_core.query_only_features(
        episode.query_tokens, episode.query_grid_shape, episode.query_valid_mask
    )
    selection = heads.query_only.select(
        episode.query_resource_key, query_features, query_rc, episode.query_valid_mask
    )
    query_scores = torch.stack([heads.query_only.score(bank, selection) for bank in episode.real_banks])
    query_c_scores = torch.stack([heads.query_only.score(bank, selection) for bank in episode.c_bind_banks])
    query_p_scores = torch.stack([heads.query_only.score(bank, selection) for bank in episode.p_coord_banks])
    query_witness_values = [
        (value, int(candidate), ordinal)
        for candidate in target
        for ordinal, value in enumerate(_query_component_scores(episode.real_banks[candidate], selection, heads.query_only))
    ]
    if query_witness_values:
        query_witness = min(
            query_witness_values,
            key=lambda item: (-float(item[0].detach().cpu()), item[1], item[2]),
        )[0]
        query_missing = False
    else:
        query_witness = torch.zeros((), dtype=torch.float64, device=query_scores.device).detach()
        query_missing = True

    arms = {
        ARM_REAL: arm_forward_from_scores(
            name=ARM_REAL,
            scores=real_scores,
            c_bind_scores=c_scores,
            p_coord_scores=p_scores,
            target_indices=target,
            opaque_keys=episode.candidate_resource_keys,
            target_witness_score=real_witness,
            target_witness_missing=real_missing,
            h1_decisions=tuple(item.h1_decision for item in real_proposals),
        ),
        ARM_ALLPATCH: arm_forward_from_scores(
            name=ARM_ALLPATCH,
            scores=all_scores,
            c_bind_scores=all_c_scores,
            p_coord_scores=all_p_scores,
            target_indices=target,
            opaque_keys=episode.candidate_resource_keys,
            target_witness_score=all_witness,
            target_witness_missing=True,
        ),
        ARM_QUERY_ONLY: arm_forward_from_scores(
            name=ARM_QUERY_ONLY,
            scores=query_scores,
            c_bind_scores=query_c_scores,
            p_coord_scores=query_p_scores,
            target_indices=target,
            opaque_keys=episode.candidate_resource_keys,
            target_witness_score=query_witness,
            target_witness_missing=query_missing,
        ),
    }
    raw = []
    for bank in episode.real_banks:
        valid = torch.as_tensor(bank.valid_mask, dtype=torch.bool)
        references = torch.as_tensor(bank.source_reference_rc, dtype=torch.long)[valid]
        raw.append(int(valid.sum()) >= 4 and torch.unique(references, dim=0).shape[0] >= 2)
    connected = tuple(bool(item.hypotheses) for item in real_proposals)
    positive = tuple(bool(item.h1_decision) for item in real_proposals)
    return EpisodeForward(
        query_resource_key=episode.query_resource_key,
        fold=episode.fold,
        supergroup=role.supergroup,
        target_indices=tuple(target),
        candidate_resource_keys=episode.candidate_resource_keys,
        arms=arms,
        raw_candidate_atom_present=tuple(raw),
        selected_connected_h1_present=connected,
        selected_positive_h1_present=positive,
    )


@dataclass(frozen=True)
class FiveTermLoss:
    total: torch.Tensor
    candidate_ce: torch.Tensor
    target_witness: torch.Tensor
    strongest_wrong: torch.Tensor
    c_bind: torch.Tensor
    p_coord: torch.Tensor
    missing_witness: bool


def five_term_loss(arm: ArmForward, target_indices: Sequence[int]) -> FiveTermLoss:
    target = torch.tensor(tuple(target_indices), dtype=torch.long, device=arm.scores.device)
    if target.numel() == 0:
        raise ValueError("loss requires at least one target destination")
    candidate_ce = torch.logsumexp(arm.scores, dim=0) - torch.logsumexp(arm.scores[target], dim=0)
    witness = F.softplus(-arm.target_witness_score)
    wrong = 0.25 * F.softplus(arm.strongest_wrong_score)
    binding = F.softplus(arm.c_bind_target_score - arm.target_score)
    if arm.name == ARM_ALLPATCH:
        if not arm.allpatch_p_coord_exact:
            raise ValueError("all-patch P_COORD invariant failed")
        coordinate = F.softplus(arm.target_score - arm.target_score)
    else:
        coordinate = F.softplus(arm.p_coord_target_score - arm.target_score)
    total = candidate_ce + witness + wrong + binding + coordinate
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite five-term loss")
    return FiveTermLoss(total, candidate_ce, witness, wrong, binding, coordinate, arm.target_witness_missing)


@dataclass(frozen=True)
class TrainingReceipt:
    completed_updates: int
    query_order: tuple[str, ...]
    query_order_sha256: str
    loss_history: Mapping[str, tuple[float, ...]]
    initial_head_sha256: Mapping[str, str]
    final_head_sha256: Mapping[str, str]
    finite_gradient_updates: Mapping[str, int]
    optimizer: Mapping[str, object]


def _head_sha256(module: nn.Module) -> str:
    return _logical_sha256(
        {name: _tensor_sha256(value) for name, value in sorted(module.state_dict().items())}
    )


EpisodeScorer = Callable[
    [AnonymousNaturalEpisode, PostjoinProposalRole, ThreeArmProposalHeads, ModuleType],
    EpisodeForward,
]


def train_optimization_heads(
    episodes: Sequence[AnonymousNaturalEpisode],
    roles: Mapping[str, PostjoinProposalRole],
    *,
    contract_v2_sha256: str,
    p_core: ModuleType,
    forward_fn: EpisodeScorer = score_episode,
) -> tuple[ThreeArmProposalHeads, TrainingReceipt]:
    """Run the frozen 512-update, three-independent-optimizer schedule."""
    items = tuple(episodes)
    if len(items) != TRAIN_QUERY_COUNT or len({item.query_resource_key for item in items}) != TRAIN_QUERY_COUNT:
        raise ValueError("optimization fit requires exactly 32 unique episodes")
    if any(item.fold != 1 or len(item.candidate_resource_keys) != CANDIDATE_COUNT for item in items):
        raise ValueError("optimization episodes must be fold-1 natural C128")
    by_key = {item.query_resource_key: item for item in items}
    if set(roles) != set(by_key):
        raise ValueError("postjoin role axis does not match anonymous episodes")
    for key, item in by_key.items():
        roles[key].validate(len(item.candidate_resource_keys))

    torch.manual_seed(TRAIN_SEED)
    heads = ThreeArmProposalHeads(p_core)
    initial = {name: _head_sha256(module) for name, module in heads.arms().items()}
    optimizers = make_optimizers(heads)
    order = deterministic_training_order(tuple(by_key), contract_v2_sha256)
    history: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    finite_gradients = {name: 0 for name in ARM_NAMES}
    for key in order:
        episode = by_key[key]
        role = roles[key]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=False)
        forward = forward_fn(episode, role, heads, p_core)
        losses = {name: five_term_loss(forward.arms[name], role.target_destination_indices) for name in ARM_NAMES}
        sum(item.total for item in losses.values()).backward()
        for name, module in heads.arms().items():
            gradients = tuple(parameter.grad for parameter in module.parameters())
            if any(item is None or not bool(torch.isfinite(item).all()) for item in gradients):
                raise FloatingPointError(f"{name} has a missing or non-finite gradient")
            finite_gradients[name] += 1
            optimizers[name].step()
            history[name].append(float(losses[name].total.detach().cpu()))
    final = {name: _head_sha256(module) for name, module in heads.arms().items()}
    receipt = TrainingReceipt(
        completed_updates=TRAIN_UPDATES,
        query_order=order,
        query_order_sha256=_logical_sha256(list(order)),
        loss_history={name: tuple(values) for name, values in history.items()},
        initial_head_sha256=initial,
        final_head_sha256=final,
        finite_gradient_updates=finite_gradients,
        optimizer={
            "class": "AdamW",
            "lr": LEARNING_RATE,
            "betas": ADAM_BETAS,
            "eps": ADAM_EPS,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
        },
    )
    return heads, receipt


@dataclass(frozen=True)
class EpisodeStatistic:
    query_resource_key: str
    fold: int
    supergroup: str
    target_present: bool
    raw_target_atom_present: bool
    selected_target_connected_h1_present: bool
    selected_target_positive_h1_present: bool
    h1_candidate_count: int
    candidate_count: int
    real_margin: float
    allpatch_margin: float
    query_only_margin: float
    c_bind_margin: float
    p_coord_margin: float


def detach_episode_statistic(forward: EpisodeForward) -> EpisodeStatistic:
    real = forward.arms[ARM_REAL]
    targets = forward.target_indices
    return EpisodeStatistic(
        query_resource_key=forward.query_resource_key,
        fold=forward.fold,
        supergroup=forward.supergroup,
        target_present=True,
        raw_target_atom_present=any(forward.raw_candidate_atom_present[index] for index in targets),
        selected_target_connected_h1_present=any(
            forward.selected_connected_h1_present[index] for index in targets
        ),
        selected_target_positive_h1_present=any(
            forward.selected_positive_h1_present[index] for index in targets
        ),
        h1_candidate_count=sum(forward.selected_positive_h1_present),
        candidate_count=len(forward.candidate_resource_keys),
        real_margin=float(real.margin.detach().cpu()),
        allpatch_margin=float(forward.arms[ARM_ALLPATCH].margin.detach().cpu()),
        query_only_margin=float(forward.arms[ARM_QUERY_ONLY].margin.detach().cpu()),
        c_bind_margin=float(real.c_bind_margin.detach().cpu()),
        p_coord_margin=float(real.p_coord_margin.detach().cpu()),
    )


@dataclass(frozen=True)
class PairedComparison:
    rescue: int
    break_count: int
    paired_net: int
    direction_mean: float


@dataclass(frozen=True)
class GateReduction:
    stage: str
    go: bool
    failures: tuple[str, ...]
    target_absent_resource_keys: tuple[str, ...]
    eligible_count: int
    raw_target_atom_coverage_count: int
    selected_target_connected_h1_coverage_count: int
    selected_target_positive_h1_coverage_count: int
    real_success_count: int
    real_failure_count: int
    group_balanced_real_margin: float
    c_bind_positive_drop_count: int
    group_balanced_c_bind_drop: float
    p_coord_positive_drop_count: int
    group_balanced_p_coord_drop: float
    global_h1_fraction: float
    allpatch: PairedComparison
    query_only: PairedComparison
    fold_direction_means: Mapping[int, Mapping[str, float]]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot reduce an empty value population")
    return math.fsum(float(item) for item in values) / len(values)


def _group_balanced(rows: Sequence[EpisodeStatistic], getter: Callable[[EpisodeStatistic], float]) -> float:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row.supergroup, []).append(float(getter(row)))
    return _mean([_mean(values) for _, values in sorted(groups.items())])


def _paired(rows: Sequence[EpisodeStatistic], comparator: str) -> PairedComparison:
    getter = (
        (lambda row: row.allpatch_margin)
        if comparator == ARM_ALLPATCH
        else (lambda row: row.query_only_margin)
    )
    rescue = sum(row.real_margin > 0.0 and getter(row) <= 0.0 for row in rows)
    breaks = sum(row.real_margin <= 0.0 and getter(row) > 0.0 for row in rows)
    return PairedComparison(
        rescue=rescue,
        break_count=breaks,
        paired_net=rescue - breaks,
        direction_mean=_mean([row.real_margin - getter(row) for row in rows]),
    )


def reduce_natural_gate(
    statistics: Sequence[EpisodeStatistic], *, stage: str
) -> GateReduction:
    """Apply the complete frozen gates and ordered simultaneous taxonomy."""
    if stage not in {"optimization", "p0"}:
        raise ValueError("stage must be optimization or p0")
    rows = tuple(statistics)
    if len({row.query_resource_key for row in rows}) != len(rows):
        raise ValueError("duplicate query resource key in gate population")
    if any(type(row.target_present) is not bool or not row.query_resource_key for row in rows):
        raise ValueError("gate population target-presence ledger is malformed")
    absent = tuple(sorted(row.query_resource_key for row in rows if not row.target_present))
    eligible = tuple(row for row in rows if row.target_present)
    if len(eligible) != TRAIN_QUERY_COUNT:
        raise ValueError("fixed target-present denominator must be exactly 32")
    if any(
        row.candidate_count != CANDIDATE_COUNT
        or not row.supergroup
        or not 0 <= row.h1_candidate_count <= CANDIDATE_COUNT
        for row in eligible
    ):
        raise ValueError("gate requires complete natural C128 and joined supergroups")
    for row in eligible:
        booleans = (
            row.target_present,
            row.raw_target_atom_present,
            row.selected_target_connected_h1_present,
            row.selected_target_positive_h1_present,
        )
        if any(type(value) is not bool for value in booleans):
            raise ValueError("gate boolean fields require exact bool types")
        if type(row.fold) is not int or type(row.h1_candidate_count) is not int or type(row.candidate_count) is not int:
            raise ValueError("gate count/fold fields require exact integer types")
        if row.selected_target_positive_h1_present and not row.selected_target_connected_h1_present:
            raise ValueError("positive target H1 does not imply a connected target H1")
        if row.selected_target_connected_h1_present and not row.raw_target_atom_present:
            raise ValueError("connected target H1 does not imply raw target atom coverage")
        if row.selected_target_positive_h1_present and row.h1_candidate_count < 1:
            raise ValueError("positive target H1 is absent from the positive-candidate count")
    if stage == "optimization" and any(row.fold != 1 for row in eligible):
        raise ValueError("optimization gate requires the frozen fold-1 population")
    numeric = (
        value
        for row in eligible
        for value in (
            row.real_margin,
            row.allpatch_margin,
            row.query_only_margin,
            row.c_bind_margin,
            row.p_coord_margin,
        )
    )
    if not all(math.isfinite(value) for value in numeric):
        raise FloatingPointError("non-finite gate statistic")

    raw_coverage = sum(row.raw_target_atom_present for row in eligible)
    connected_coverage = sum(row.selected_target_connected_h1_present for row in eligible)
    positive_coverage = sum(row.selected_target_positive_h1_present for row in eligible)
    real_success = sum(row.real_margin > 0.0 for row in eligible)
    real_failure = len(eligible) - real_success
    group_real = _group_balanced(eligible, lambda row: row.real_margin)
    c_drops = [row.real_margin - row.c_bind_margin for row in eligible]
    p_drops = [row.real_margin - row.p_coord_margin for row in eligible]
    c_count = sum(value > 0.0 for value in c_drops)
    p_count = sum(value > 0.0 for value in p_drops)
    group_c = _group_balanced(eligible, lambda row: row.real_margin - row.c_bind_margin)
    group_p = _group_balanced(eligible, lambda row: row.real_margin - row.p_coord_margin)
    total_h1 = sum(row.h1_candidate_count for row in eligible)
    global_fraction = total_h1 / float(len(eligible) * CANDIDATE_COUNT)
    any_all_h0 = any(row.h1_candidate_count == 0 for row in eligible) or global_fraction == 0.0
    any_always_h1 = any(row.h1_candidate_count == CANDIDATE_COUNT for row in eligible) or global_fraction == 1.0
    allpatch = _paired(eligible, ARM_ALLPATCH)
    query_only = _paired(eligible, ARM_QUERY_ONLY)
    binding_pass = c_count >= 21 and group_c > 0.0
    coordinate_pass = p_count >= 21 and group_p > 0.0

    flags: dict[str, bool] = {name: False for name in FAILURE_PRECEDENCE}
    flags["P_GENERATION_COVERAGE_NO_GO"] = connected_coverage < 26 or positive_coverage < 26
    flags["P_H0_COLLAPSE_ALL_H0"] = any_all_h0
    flags["P_H0_COLLAPSE_ALWAYS_H1"] = any_always_h1
    flags["P_NONSELECTIVE_RANK_NO_GO"] = real_success < 21 or group_real <= 0.0 or real_success <= real_failure
    flags["P_RERANKER_SHORTCUT_NO_GO"] = any(
        comparison.paired_net < 4 or comparison.rescue <= comparison.break_count
        for comparison in (allpatch, query_only)
    )
    flags["P_CANDIDATE_BINDING_NO_GO"] = not binding_pass
    flags["P_CANDIDATE_BOUND_NONSPATIAL"] = binding_pass and not coordinate_pass

    fold_means: dict[int, dict[str, float]] = {}
    fold_failure = False
    if stage == "p0":
        fold_counts = {fold: sum(row.fold == fold for row in eligible) for fold in (2, 3, 4)}
        if fold_counts != {2: 11, 3: 11, 4: 10} or any(
            row.fold not in fold_counts for row in eligible
        ):
            raise ValueError("formal P0 fold quotas must be exactly 11/11/10")
        group_folds: dict[str, set[int]] = {}
        for row in eligible:
            group_folds.setdefault(row.supergroup, set()).add(row.fold)
        if any(len(folds) != 1 for folds in group_folds.values()):
            raise ValueError("a joined supergroup crosses formal P0 folds")
        for fold in (2, 3, 4):
            fold_rows = tuple(row for row in eligible if row.fold == fold)
            if not fold_rows:
                raise ValueError(f"formal P0 fold {fold} is empty")
            fold_means[fold] = {
                ARM_ALLPATCH: _mean([row.real_margin - row.allpatch_margin for row in fold_rows]),
                ARM_QUERY_ONLY: _mean([row.real_margin - row.query_only_margin for row in fold_rows]),
            }
            fold_failure |= any(value < 0.0 for value in fold_means[fold].values())
        flags["P_OOF_GENERALIZATION_NO_GO"] = fold_failure or any(
            flags[name] for name in FAILURE_PRECEDENCE[:-1]
        )
    failures = tuple(name for name in FAILURE_PRECEDENCE if flags[name])
    return GateReduction(
        stage=stage,
        go=not failures,
        failures=failures,
        target_absent_resource_keys=absent,
        eligible_count=len(eligible),
        raw_target_atom_coverage_count=raw_coverage,
        selected_target_connected_h1_coverage_count=connected_coverage,
        selected_target_positive_h1_coverage_count=positive_coverage,
        real_success_count=real_success,
        real_failure_count=real_failure,
        group_balanced_real_margin=group_real,
        c_bind_positive_drop_count=c_count,
        group_balanced_c_bind_drop=group_c,
        p_coord_positive_drop_count=p_count,
        group_balanced_p_coord_drop=group_p,
        global_h1_fraction=global_fraction,
        allpatch=allpatch,
        query_only=query_only,
        fold_direction_means=fold_means,
    )


__all__ = [
    "ADAM_BETAS",
    "ADAM_EPS",
    "ARM_ALLPATCH",
    "ARM_NAMES",
    "ARM_QUERY_ONLY",
    "ARM_REAL",
    "CANDIDATE_COUNT",
    "FAILURE_PRECEDENCE",
    "HEAD_PARAMETER_COUNT",
    "LEARNING_RATE",
    "TRAIN_EPOCHS",
    "TRAIN_QUERY_COUNT",
    "TRAIN_SEED",
    "TRAIN_UPDATES",
    "TOTAL_PARAMETER_COUNT",
    "AnonymousNaturalEpisode",
    "ArmForward",
    "EpisodeForward",
    "EpisodeStatistic",
    "FiveTermLoss",
    "GateReduction",
    "PairedComparison",
    "PostjoinProposalRole",
    "ThreeArmProposalHeads",
    "TrainingReceipt",
    "arm_forward_from_scores",
    "detach_episode_statistic",
    "deterministic_training_order",
    "five_term_loss",
    "load_standalone_p_core",
    "make_optimizers",
    "reduce_natural_gate",
    "score_episode",
    "train_optimization_heads",
]
