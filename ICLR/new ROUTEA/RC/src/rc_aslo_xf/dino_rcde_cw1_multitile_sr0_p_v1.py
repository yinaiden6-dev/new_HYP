"""Minimal shared ColNomic P core for formal CW1 multi-tile SR0.

This module is additive.  It does not modify or import the deterministic E1
P-seal resolver.  Its inference graph has one purpose: score an anonymous
candidate's fixed candidate/action-local ColNomic population and seal one
canonical connected multi-root query hypothesis.

The only learned object is a shared, bias-free ``Linear(16, 1)`` head.  The
first feature is the historical raw query-to-reference log-mean-exp statistic;
initialization gives that coordinate weight one and every other coordinate
weight zero.  Labels may select two already-scored candidates for the training
loss, but no target, identity, D1 quantity, DINO value, or retrieval outcome is
accepted by the inference API.

Scientific scope is deliberately narrow: this is a pure core and synthetic
qualification primitive.  It does not implement fold construction, natural
source joins, OOF materialization, a P seal, V training, or a scientific gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .cw1_sr0_structure_v1 import (
    CW1SR0SuperRegion,
    enumerate_superregion_bank,
    superregion_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_cw1_multitile_sr0_shared_p_v1"

# These columns are the complete and ordered P input contract.  Every value is
# candidate/action-local and is derived from the same frozen ColNomic token
# pair.  There is no candidate rank, slot, winner, D1 gap, identity, or DINO
# coordinate hidden in the schema.
FEATURE_SCHEMA = (
    "raw_lme_q_to_r",
    "q_to_r_best_mean",
    "q_to_r_best_std",
    "q_to_r_top1_top2_gap_mean",
    "q_to_r_normalized_entropy",
    "r_to_q_logmeanexp",
    "r_to_q_best_mean",
    "r_to_q_best_std",
    "r_to_q_top1_top2_gap_mean",
    "r_to_q_normalized_entropy",
    "mutual_nn_query_fraction",
    "mutual_nn_reference_fraction",
    "unique_reference_utilization",
    "unique_query_utilization",
    "mutual_pair_mean_similarity",
    "local_vs_full_reference_lme_gain",
)
FEATURE_DIM = 16
RAW_LME_INDEX = 0
COLNOMIC_LME_TEMPERATURE = 0.07
SIMILARITY_DEFINITION = "L2_NORMALIZED_COSINE_FLOAT32"
NORMALIZED_ENTROPY_EPSILON = 1.0e-12
MISSING_TOP2_GAP = 0.0
EMPTY_ACTION_H0 = 0.0
HIERARCHICAL_TEMPERATURE = 1.0
HARD_MAX_TIE_ULPS = 16
TARGET_RADII = (1, 2, 3, 4)
MIN_READY_ROOTS = 2
PARAMETER_COUNT = FEATURE_DIM
P_SEED = 17
P_OPTIMIZER = "AdamW"
P_LEARNING_RATE = 3.0e-4
P_WEIGHT_DECAY = 1.0e-4
P_GRADIENT_CLIP_L2 = 1.0
P_UPDATES_TOTAL = 2048
P_EPISODES_PER_UPDATE = 4
P_WARMUP_UPDATES = 128
P_FINAL_LEARNING_RATE = 3.0e-5
P_CHECKPOINT_INTERVAL_UPDATES = 64
P_SAMPLER_NAMESPACE = "RCDE_SR0_MT_P_QUERY_ORDER_V1"
P_LOCK_REQUIRED_FIELDS = (
    "query_id",
    "historical_query_ordinal",
    "execution_ordinal",
    "query_source_image_sha256",
    "outer_fold",
    "crossfit_role",
    "p_checkpoint_sha256",
    "p_training_manifest_sha256",
    "candidate_physical_row",
    "candidate_reference_source_sha256",
    "direction",
    "complete_bank_sha256",
    "selected_bank_ordinal",
    "selected_row_sha256",
    "status",
    "query_union_mask_rle",
    "query_union_mask_sha256",
    "ordered_root_ordinals",
    "root_query_tile_mask_rle",
    "root_query_tile_mask_sha256",
    "root_reference_component_mask_rle",
    "root_reference_component_mask_sha256",
    "root_binding_status",
    # The selected row alone is not sufficient for candidate-relative V:
    # two candidates may select different CW1 rows, while V must still read
    # the opponent component bound to each *owner* root.  This erased ledger
    # seals the deployable per-root action chosen by P for every canonical
    # root, without retaining scores, ranks, or labels.
    "all_root_ordinals",
    "all_root_action_key_sha256",
    "all_root_query_tile_mask_rle",
    "all_root_query_tile_mask_sha256",
    "all_root_reference_component_mask_rle",
    "all_root_reference_component_mask_sha256",
    "all_root_binding_status",
    "query_geometry_sha256",
    "reference_geometry_sha256",
    "fixed_denominator",
    "erased_fields",
    "record_sha256",
)
P_LOCK_ERASED_FIELDS = (
    "score",
    "confidence",
    "posterior",
    "target",
    "identity",
    "rank",
    "slot",
    "winner",
    "D1_gap",
    "correctness",
)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_SCHEMA, separators=(",", ":")).encode("utf-8")
).hexdigest()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SharedPContractError(ValueError):
    """The shared P feature, population, or scoring contract was violated."""


@dataclass(frozen=True)
class PEpisodeKey:
    """Only target-free addressing fields used by the frozen P sampler."""

    query_id: str
    source_image_sha256: str
    execution_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise SharedPContractError("P sampler query ID must be nonempty")
        if not isinstance(self.source_image_sha256, str) or _SHA256.fullmatch(
            self.source_image_sha256
        ) is None:
            raise SharedPContractError("P sampler source hash must be lowercase SHA256")
        if (
            isinstance(self.execution_ordinal, bool)
            or not isinstance(self.execution_ordinal, int)
            or self.execution_ordinal < 0
        ):
            raise SharedPContractError("P sampler execution ordinal is invalid")


def deterministic_p_epoch_order(
    episodes: Sequence[PEpisodeKey], *, fit_id: str, epoch: int
) -> tuple[int, ...]:
    """Frozen result-blind, without-replacement order for one P fit epoch."""

    items = tuple(episodes)
    if not items or not isinstance(fit_id, str) or not fit_id:
        raise SharedPContractError("P sampler needs a nonempty fit/population")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise SharedPContractError("P sampler epoch is invalid")
    if len({item.query_id for item in items}) != len(items) or len(
        {item.execution_ordinal for item in items}
    ) != len(items):
        raise SharedPContractError("P sampler episode keys are not unique")

    def key(ordinal: int) -> tuple[bytes, int]:
        item = items[ordinal]
        payload = "\x1f".join(
            (
                P_SAMPLER_NAMESPACE,
                str(P_SEED),
                fit_id,
                str(epoch),
                item.source_image_sha256,
                item.query_id,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).digest(), item.execution_ordinal

    return tuple(sorted(range(len(items)), key=key))


def p_learning_rate(completed_update: int) -> float:
    """Learning rate applied at the named one-based completed update."""

    if (
        isinstance(completed_update, bool)
        or not isinstance(completed_update, int)
        or not 1 <= completed_update <= P_UPDATES_TOTAL
    ):
        raise SharedPContractError("P learning-rate update is outside 1..2048")
    if completed_update <= P_WARMUP_UPDATES:
        return P_LEARNING_RATE * completed_update / P_WARMUP_UPDATES
    progress = (completed_update - P_WARMUP_UPDATES) / (
        P_UPDATES_TOTAL - P_WARMUP_UPDATES
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return P_FINAL_LEARNING_RATE + (
        P_LEARNING_RATE - P_FINAL_LEARNING_RATE
    ) * cosine


def make_frozen_p_optimizer(model: "SharedMultitilePHead") -> torch.optim.AdamW:
    if not isinstance(model, SharedMultitilePHead):
        raise SharedPContractError("P optimizer requires the frozen shared head")
    return torch.optim.AdamW(
        model.parameters(),
        lr=P_LEARNING_RATE,
        weight_decay=P_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )


def _finite_floating(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise SharedPContractError(f"{name} must be finite floating point")
    return tensor


def _valid_mask(value: torch.Tensor, count: int, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().contiguous()
    if result.shape != (count,):
        raise SharedPContractError(f"{name} must be a one-dimensional patch mask")
    return result


def _unit_tokens(
    value: torch.Tensor,
    valid_patch_mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    source = _finite_floating(value, name=name)
    if source.ndim != 2 or source.shape[0] != valid_patch_mask.numel() or source.shape[1] <= 0:
        raise SharedPContractError(
            f"{name} must have shape [patch,positive-dimension]"
        )
    # Match the historical ColNomic proposal path: normalize and compute the
    # local cost matrix in float32, then store the sixteen frozen summaries in
    # float64 for the tiny trainable head.
    source = source.detach().to(torch.float32).contiguous()
    valid = valid_patch_mask.to(device=source.device)
    norms = torch.linalg.vector_norm(source, dim=1)
    if bool(norms[valid].le(0.0).any()):
        raise SharedPContractError(f"{name} valid patches must have nonzero norm")
    output = torch.zeros_like(source)
    output[valid] = source[valid] / norms[valid, None]
    return output


def candidate_action_colnomic_features(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    query_action_mask: torch.Tensor,
    reference_action_mask: torch.Tensor,
    query_valid_patch_mask: torch.Tensor,
    reference_valid_patch_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen 16-D feature row for one candidate/action pair.

    This function has no target or candidate-competition argument.  Both masks
    must be nonempty subsets of their immutable geometry-valid masks.  Patch
    tokens are detached: P learns only the shared linear readout, never the
    ColNomic encoder or an action-specific parameter.
    """

    query_source = torch.as_tensor(query_tokens)
    reference_source = torch.as_tensor(reference_tokens)
    if query_source.device != reference_source.device:
        raise SharedPContractError("query/reference ColNomic devices differ")
    if query_source.ndim != 2 or reference_source.ndim != 2:
        raise SharedPContractError("query/reference ColNomic tokens must be matrices")
    if query_source.shape[1] != reference_source.shape[1]:
        raise SharedPContractError("query/reference ColNomic dimensions differ")

    q_valid = _valid_mask(
        query_valid_patch_mask, query_source.shape[0], name="query valid mask"
    ).to(query_source.device)
    r_valid = _valid_mask(
        reference_valid_patch_mask,
        reference_source.shape[0],
        name="reference valid mask",
    ).to(reference_source.device)
    q_mask = _valid_mask(
        query_action_mask, query_source.shape[0], name="query action mask"
    ).to(query_source.device)
    r_mask = _valid_mask(
        reference_action_mask,
        reference_source.shape[0],
        name="reference action mask",
    ).to(reference_source.device)
    if not bool(q_valid.any()) or not bool(r_valid.any()):
        raise SharedPContractError("valid ColNomic patch populations must be nonempty")
    if bool((q_mask & ~q_valid).any()) or bool((r_mask & ~r_valid).any()):
        raise SharedPContractError("candidate/action mask escaped geometry-valid patches")
    if not bool(q_mask.any()) or not bool(r_mask.any()):
        # Empty actions are structural H0.  Returning the exact all-zero row
        # keeps them representable without allowing any learned coordinate to
        # turn an absent action into evidence.
        return torch.full(
            (FEATURE_DIM,),
            EMPTY_ACTION_H0,
            dtype=torch.float64,
            device=query_source.device,
        )

    query_all = _unit_tokens(query_source, q_valid, name="query ColNomic tokens")
    reference_all = _unit_tokens(
        reference_source, r_valid, name="reference ColNomic tokens"
    )
    query = query_all[q_mask]
    reference = reference_all[r_mask]
    pair = query @ reference.T
    tau = COLNOMIC_LME_TEMPERATURE
    q_lme = tau * (
        torch.logsumexp(pair / tau, dim=1) - math.log(reference.shape[0])
    )
    r_lme = tau * (
        torch.logsumexp(pair / tau, dim=0) - math.log(query.shape[0])
    )
    q_best, q_argmax = pair.max(dim=1)
    r_best, r_argmax = pair.max(dim=0)

    def top1_top2_gap(value: torch.Tensor, *, dim: int) -> torch.Tensor:
        if value.shape[dim] < 2:
            output_shape = value.shape[1 - dim]
            return torch.full(
                (output_shape,),
                MISSING_TOP2_GAP,
                dtype=value.dtype,
                device=value.device,
            )
        best_two = torch.topk(value, k=2, dim=dim, largest=True, sorted=True).values
        return best_two.select(dim, 0) - best_two.select(dim, 1)

    def normalized_entropy(value: torch.Tensor, *, dim: int) -> torch.Tensor:
        population = value.shape[dim]
        output_shape = value.shape[1 - dim]
        if population < 2:
            return torch.zeros(output_shape, dtype=value.dtype, device=value.device)
        probability = torch.softmax(value / tau, dim=dim)
        entropy = -(
            probability
            * torch.log(probability.clamp_min(NORMALIZED_ENTROPY_EPSILON))
        ).sum(dim=dim)
        return entropy / math.log(population)

    q_gap = top1_top2_gap(pair, dim=1)
    r_gap = top1_top2_gap(pair, dim=0)
    q_entropy = normalized_entropy(pair, dim=1)
    r_entropy = normalized_entropy(pair, dim=0)
    q_ordinals = torch.arange(query.shape[0], device=pair.device)
    r_ordinals = torch.arange(reference.shape[0], device=pair.device)
    mutual_q = r_argmax[q_argmax].eq(q_ordinals)
    mutual_r = q_argmax[r_argmax].eq(r_ordinals)
    mutual_similarity = q_best[mutual_q]
    if mutual_similarity.numel() == 0:
        mutual_mean = pair.square().sum() * 0.0
    else:
        mutual_mean = mutual_similarity.mean()
    unique_reference_utilization = (
        torch.unique(q_argmax).numel() / float(reference.shape[0])
    )
    unique_query_utilization = torch.unique(r_argmax).numel() / float(query.shape[0])
    full_reference = reference_all[r_valid]
    full_pair = query @ full_reference.T
    full_q_lme = tau * (
        torch.logsumexp(full_pair / tau, dim=1) - math.log(full_reference.shape[0])
    )

    features = torch.stack(
        (
            q_lme.mean(),
            q_best.mean(),
            q_best.std(unbiased=False),
            q_gap.mean(),
            q_entropy.mean(),
            r_lme.mean(),
            r_best.mean(),
            r_best.std(unbiased=False),
            r_gap.mean(),
            r_entropy.mean(),
            mutual_q.to(pair.dtype).mean(),
            mutual_r.to(pair.dtype).mean(),
            pair.new_tensor(unique_reference_utilization),
            pair.new_tensor(unique_query_utilization),
            mutual_mean,
            q_lme.mean() - full_q_lme.mean(),
        )
    ).detach().to(torch.float64).contiguous()
    if features.shape != (FEATURE_DIM,) or not bool(torch.isfinite(features).all()):
        raise RuntimeError("candidate/action ColNomic feature schema drift")
    return features


@dataclass(frozen=True)
class CandidateActionFeatureTable:
    """Complete anonymous per-root action population for one candidate."""

    candidate_key: str
    root_ordinals: tuple[int, ...]
    action_keys_by_root: tuple[tuple[str, ...], ...]
    features_by_root: tuple[torch.Tensor, ...]
    eligible_by_root: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise SharedPContractError("candidate key must be nonempty")
        roots = tuple(self.root_ordinals)
        if roots != tuple(range(len(roots))) or not roots:
            raise SharedPContractError(
                "root ordinals must be the complete canonical zero-based population"
            )
        keys_by_root = tuple(tuple(item) for item in self.action_keys_by_root)
        features_by_root = tuple(self.features_by_root)
        eligible_by_root = tuple(self.eligible_by_root)
        if not (
            len(keys_by_root) == len(features_by_root) == len(eligible_by_root) == len(roots)
        ):
            raise SharedPContractError("candidate/root action table length drift")

        canonical_features: list[torch.Tensor] = []
        canonical_eligible: list[torch.Tensor] = []
        for root, (keys, features, eligible) in enumerate(
            zip(keys_by_root, features_by_root, eligible_by_root, strict=True)
        ):
            values = _finite_floating(features, name=f"root {root} action features")
            flags = torch.as_tensor(
                eligible, dtype=torch.bool, device=values.device
            ).detach().contiguous()
            if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
                raise SharedPContractError(
                    f"root {root} features must have shape [action,{FEATURE_DIM}]"
                )
            if flags.shape != (values.shape[0],) or len(keys) != values.shape[0]:
                raise SharedPContractError(f"root {root} action population drift")
            if len(set(keys)) != len(keys) or any(
                not isinstance(key, str) or _SHA256.fullmatch(key) is None for key in keys
            ):
                raise SharedPContractError(
                    f"root {root} action keys must be unique lowercase SHA256 values"
                )
            values = values.detach().to(torch.float64).contiguous()
            if bool(values[~flags].ne(0.0).any()):
                raise SharedPContractError(
                    f"root {root} ineligible action rows must be exact zero/H0"
                )
            canonical_features.append(values)
            canonical_eligible.append(flags)

        object.__setattr__(self, "root_ordinals", roots)
        object.__setattr__(self, "action_keys_by_root", keys_by_root)
        object.__setattr__(self, "features_by_root", tuple(canonical_features))
        object.__setattr__(self, "eligible_by_root", tuple(canonical_eligible))


def canonical_action_argmax(
    scores: torch.Tensor,
    action_keys: Sequence[str],
    eligible: torch.Tensor,
) -> int | None:
    """Hard max with exact-score ties broken by the canonical action hash."""

    value = _finite_floating(scores, name="action scores")
    keys = tuple(action_keys)
    flags = torch.as_tensor(eligible, dtype=torch.bool, device=value.device)
    if value.ndim != 1 or flags.shape != value.shape or len(keys) != value.numel():
        raise SharedPContractError("canonical action max population drift")
    legal = torch.nonzero(flags, as_tuple=False).flatten().tolist()
    if not legal:
        return None
    detached = value.detach()
    maximum = detached[torch.tensor(legal, device=value.device)].max()
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
    return min(tied, key=lambda index: keys[index])


def _normalized_logmeanexp(values: torch.Tensor) -> torch.Tensor:
    value = _finite_floating(values, name="logmeanexp values")
    if value.ndim != 1 or value.numel() == 0:
        raise SharedPContractError("logmeanexp requires a nonempty vector")
    if bool(value.detach().eq(0.0).all()):
        return value.square().sum() * 0.0
    tau = HIERARCHICAL_TEMPERATURE
    return tau * (
        torch.logsumexp(value / tau, dim=0) - math.log(value.numel())
    )


def scale_balanced_hierarchical_logmeanexp(
    row_scores: torch.Tensor,
    row_radii: Sequence[int],
) -> torch.Tensor:
    """Equal-weight r1--r4 evidence, independent of rows per radius.

    The input is the *complete* canonical bank.  Ineligible rows are already
    exact-zero H0 rows, so they stay in their fixed within-radius denominator.
    Every radius contributes exactly once at the second level.  The explicit
    all-zero path prevents floating reduction noise from changing H0.
    """

    scores = _finite_floating(row_scores, name="row scores")
    radii = tuple(int(item) for item in row_radii)
    if scores.ndim != 1 or len(radii) != scores.numel():
        raise SharedPContractError("hierarchical row/radius population drift")
    if set(radii) != set(TARGET_RADII):
        raise SharedPContractError("hierarchical reducer requires all r1--r4 scales")
    scale_scores = []
    for radius in TARGET_RADII:
        index = [ordinal for ordinal, value in enumerate(radii) if value == radius]
        if not index:
            raise SharedPContractError(f"hierarchical reducer has no r{radius} rows")
        selected = scores[
            torch.tensor(index, dtype=torch.long, device=scores.device)
        ]
        scale_scores.append(_normalized_logmeanexp(selected))
    stacked = torch.stack(scale_scores)
    if bool(stacked.detach().eq(0.0).all()):
        return stacked.square().sum() * 0.0
    return _normalized_logmeanexp(stacked)


@dataclass(frozen=True)
class CandidatePOutput:
    """Target-free shared-P output for one anonymous candidate."""

    candidate_key: str
    selected_action_indices: tuple[int | None, ...]
    selected_action_keys: tuple[str | None, ...]
    root_scores: torch.Tensor
    root_ready: torch.Tensor
    row_hashes: tuple[str, ...]
    row_radii: tuple[int, ...]
    row_scores: torch.Tensor
    row_ready: torch.Tensor
    candidate_evidence: torch.Tensor
    map_row_index: int | None
    map_row_hash: str | None
    map_score: torch.Tensor

    def __post_init__(self) -> None:
        roots = len(self.selected_action_indices)
        rows = len(self.row_hashes)
        if (
            not self.candidate_key
            or len(self.selected_action_keys) != roots
            or self.root_scores.shape != (roots,)
            or self.root_ready.shape != (roots,)
            or self.root_ready.dtype != torch.bool
            or len(self.row_radii) != rows
            or self.row_scores.shape != (rows,)
            or self.row_ready.shape != (rows,)
            or self.row_ready.dtype != torch.bool
            or self.candidate_evidence.ndim != 0
            or self.map_score.ndim != 0
        ):
            raise SharedPContractError("target-free P output shape drift")
        if not all(
            bool(torch.isfinite(item).all())
            for item in (self.root_scores, self.row_scores, self.candidate_evidence, self.map_score)
        ):
            raise SharedPContractError("target-free P output is non-finite")
        if bool(self.root_scores[~self.root_ready].ne(0.0).any()):
            raise SharedPContractError("missing P root is not exact H0")
        if bool(self.row_scores[~self.row_ready].ne(0.0).any()):
            raise SharedPContractError("ineligible P row is not exact H0")
        for index, ready in enumerate(self.root_ready.tolist()):
            has_selection = (
                self.selected_action_indices[index] is not None
                and self.selected_action_keys[index] is not None
            )
            if ready != has_selection:
                raise SharedPContractError("root READY/H0 selection drift")
        if self.map_row_index is None:
            if self.map_row_hash is not None or bool(self.map_score.ne(0.0)):
                raise SharedPContractError("P H0 MAP must have exact zero score")
        elif (
            self.map_row_index not in range(rows)
            or self.map_row_hash != self.row_hashes[self.map_row_index]
            or not bool(self.row_ready[self.map_row_index])
            or not bool(self.map_score.gt(0.0))
        ):
            raise SharedPContractError("P H1 MAP binding drift")


class SharedMultitilePHead(nn.Module):
    """One shared 16-parameter P head; no candidate-specific parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIM, 1, bias=False, dtype=torch.float64)
        self.reset_parameters_frozen()
        if sum(parameter.numel() for parameter in self.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("shared P parameter count drift")

    def reset_parameters_frozen(self) -> None:
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, RAW_LME_INDEX] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = _finite_floating(features, name="shared P features")
        if values.shape[-1:] != (FEATURE_DIM,):
            raise SharedPContractError(
                f"shared P features must end in dimension {FEATURE_DIM}"
            )
        values = values.to(
            device=self.linear.weight.device, dtype=self.linear.weight.dtype
        )
        return self.linear(values).squeeze(-1)

    def _exact_zero(self) -> torch.Tensor:
        # square() ensures the forward value is positive zero even if learned
        # weights later have a negative sum; the reverse-mode gradient is zero.
        return self.linear.weight.square().sum() * 0.0

    def score_candidate(
        self,
        action_table: CandidateActionFeatureTable,
        regions: Sequence[CW1SR0SuperRegion],
    ) -> CandidatePOutput:
        """Score one anonymous candidate without labels or retrieval priors."""

        if not isinstance(action_table, CandidateActionFeatureTable):
            raise SharedPContractError("shared P requires a candidate action table")
        rows = tuple(regions)
        if not rows or any(not isinstance(item, CW1SR0SuperRegion) for item in rows):
            raise SharedPContractError("shared P requires the canonical CW1 row bank")
        grid_shape = rows[0].grid_shape
        canonical = tuple(
            enumerate_superregion_bank(grid_shape, include_r0_control=False)
        )
        if tuple(superregion_sha256(item) for item in rows) != tuple(
            superregion_sha256(item) for item in canonical
        ):
            raise SharedPContractError("shared P rows drifted from the canonical r1--r4 bank")
        expected_roots = len(enumerate_query_macro_seeds(grid_shape))
        if len(action_table.root_ordinals) != expected_roots:
            raise SharedPContractError("candidate action table root count/grid drift")

        selected_indices: list[int | None] = []
        selected_keys: list[str | None] = []
        root_scores: list[torch.Tensor] = []
        root_ready: list[bool] = []
        for keys, features, eligible in zip(
            action_table.action_keys_by_root,
            action_table.features_by_root,
            action_table.eligible_by_root,
            strict=True,
        ):
            logits = self(features)
            index = canonical_action_argmax(logits, keys, eligible.to(logits.device))
            selected_indices.append(index)
            if index is None:
                selected_keys.append(None)
                root_scores.append(self._exact_zero())
                root_ready.append(False)
            else:
                selected_keys.append(keys[index])
                root_scores.append(logits[index])
                root_ready.append(True)

        root_score_tensor = torch.stack(root_scores)
        root_ready_tensor = torch.tensor(
            root_ready, dtype=torch.bool, device=root_score_tensor.device
        )
        row_scores: list[torch.Tensor] = []
        row_ready: list[bool] = []
        row_hashes: list[str] = []
        row_radii: list[int] = []
        for region in rows:
            roots = region.contributing_root_ordinals
            ready_count = sum(root_ready[root] for root in roots)
            if ready_count >= MIN_READY_ROOTS:
                weights = region.aggregation_weights.to(
                    device=root_score_tensor.device, dtype=root_score_tensor.dtype
                )
                # The frozen weights sum root incidences as 1 / overlap-count
                # and divide by the complete unique query union.  Missing roots
                # stay zero; no available-root renormalization is permitted.
                row_score = torch.dot(weights, root_score_tensor)
                ready = True
            else:
                row_score = self._exact_zero()
                ready = False
            row_scores.append(row_score)
            row_ready.append(ready)
            row_hashes.append(superregion_sha256(region))
            row_radii.append(region.radius)

        row_score_tensor = torch.stack(row_scores)
        row_ready_tensor = torch.tensor(
            row_ready, dtype=torch.bool, device=row_score_tensor.device
        )
        evidence = scale_balanced_hierarchical_logmeanexp(
            row_score_tensor, row_radii
        )

        legal = torch.nonzero(row_ready_tensor, as_tuple=False).flatten().tolist()
        map_index: int | None = None
        if legal:
            detached = row_score_tensor.detach()
            maximum = detached[
                torch.tensor(legal, dtype=torch.long, device=detached.device)
            ].max()
            # H0 has exact score zero and wins an exact tie.  A deployable H1
            # therefore requires strictly positive evidence.
            h0_tolerance = (
                HARD_MAX_TIE_ULPS
                * torch.finfo(detached.dtype).eps
                * max(1.0, abs(float(maximum)))
            )
            if float(maximum) > h0_tolerance:
                tied = [
                    index
                    for index in legal
                    if abs(float(detached[index]) - float(maximum)) <= h0_tolerance
                ]
                map_index = min(tied, key=lambda index: row_hashes[index])
        map_hash = None if map_index is None else row_hashes[map_index]
        map_score = self._exact_zero() if map_index is None else row_score_tensor[map_index]
        return CandidatePOutput(
            candidate_key=action_table.candidate_key,
            selected_action_indices=tuple(selected_indices),
            selected_action_keys=tuple(selected_keys),
            root_scores=root_score_tensor,
            root_ready=root_ready_tensor,
            row_hashes=tuple(row_hashes),
            row_radii=tuple(row_radii),
            row_scores=row_score_tensor,
            row_ready=row_ready_tensor,
            candidate_evidence=evidence,
            map_row_index=map_index,
            map_row_hash=map_hash,
            map_score=map_score,
        )


@dataclass(frozen=True)
class PairwisePNoRegretLoss:
    margin: torch.Tensor
    loss: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.margin.ndim != 0
            or self.loss.ndim != 0
            or not bool(torch.isfinite(self.margin))
            or not bool(torch.isfinite(self.loss))
        ):
            raise SharedPContractError("pairwise P loss is not finite scalar")


def pairwise_softplus_no_regret_loss(
    target_evidence: torch.Tensor,
    rival_evidence: torch.Tensor,
) -> PairwisePNoRegretLoss:
    """Post-join natural pair loss; labels never enter candidate inference."""

    target = _finite_floating(target_evidence, name="target P evidence")
    rival = _finite_floating(rival_evidence, name="rival P evidence")
    if target.ndim != 0 or rival.ndim != 0 or target.device != rival.device:
        raise SharedPContractError("target/rival P evidence must be scalar on one device")
    margin = target - rival.to(dtype=target.dtype)
    return PairwisePNoRegretLoss(margin=margin, loss=F.softplus(-margin))


def _append_blob(output: bytearray, tag: bytes, payload: bytes) -> None:
    output.extend(len(tag).to_bytes(4, "big"))
    output.extend(tag)
    output.extend(len(payload).to_bytes(8, "big"))
    output.extend(payload)


def _canonical_state_value(value: object) -> bytes:
    """Deterministic byte encoding for model/optimizer resume tests."""

    output = bytearray()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _append_blob(output, b"tensor-meta", metadata)
        _append_blob(output, b"tensor-data", raw)
    elif isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            encoded_key = _canonical_state_value(key)
            entries.append((encoded_key, _canonical_state_value(item)))
        for encoded_key, encoded_value in sorted(entries, key=lambda item: item[0]):
            _append_blob(output, b"mapping-key", encoded_key)
            _append_blob(output, b"mapping-value", encoded_value)
    elif isinstance(value, (list, tuple)):
        tag = b"list" if isinstance(value, list) else b"tuple"
        for item in value:
            _append_blob(output, tag, _canonical_state_value(item))
    elif value is None:
        _append_blob(output, b"none", b"")
    elif isinstance(value, bool):
        _append_blob(output, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _append_blob(output, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _append_blob(output, b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        _append_blob(output, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _append_blob(output, b"bytes", value)
    else:
        raise SharedPContractError(
            f"unsupported checkpoint value type: {type(value).__name__}"
        )
    return bytes(output)


def canonical_training_state_bytes(
    model: SharedMultitilePHead,
    optimizer: torch.optim.Optimizer,
    *,
    completed_updates: int,
    current_learning_rate: float,
    scheduler_cursor: int,
    sampler_state: Mapping[str, object],
    rng_state: Mapping[str, object],
    eligible_population_sha256: str,
    execution_protocol_sha256: str,
    p_implementation_sha256: str,
) -> bytes:
    """Canonical byte receipt used to prove exact fresh/resume continuation."""

    if not isinstance(model, SharedMultitilePHead):
        raise SharedPContractError("training-state model is not shared P")
    if isinstance(completed_updates, bool) or completed_updates < 0:
        raise SharedPContractError("completed update count must be non-negative")
    if (
        not math.isfinite(current_learning_rate)
        or current_learning_rate < 0.0
        or isinstance(scheduler_cursor, bool)
        or scheduler_cursor != completed_updates
    ):
        raise SharedPContractError("P scheduler/checkpoint cursor drift")
    for name, value in (
        ("eligible population", eligible_population_sha256),
        ("execution protocol", execution_protocol_sha256),
        ("P implementation", p_implementation_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise SharedPContractError(f"{name} hash must be lowercase SHA256")
    required_sampler = {
        "namespace",
        "fit_id",
        "epoch",
        "position",
        "order_sha256",
    }
    required_rng = {
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    if set(sampler_state) != required_sampler or set(rng_state) != required_rng:
        raise SharedPContractError("P sampler/RNG checkpoint schema drift")
    if sampler_state.get("namespace") != P_SAMPLER_NAMESPACE:
        raise SharedPContractError("P sampler namespace drift")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "completed_updates": int(completed_updates),
        "current_learning_rate": float(current_learning_rate),
        "scheduler_cursor": int(scheduler_cursor),
        "sampler_state": dict(sampler_state),
        "rng_state": dict(rng_state),
        "eligible_population_sha256": eligible_population_sha256,
        "execution_protocol_sha256": execution_protocol_sha256,
        "p_implementation_sha256": p_implementation_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return _canonical_state_value(payload)


__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_SHA256",
    "FEATURE_DIM",
    "RAW_LME_INDEX",
    "COLNOMIC_LME_TEMPERATURE",
    "SIMILARITY_DEFINITION",
    "NORMALIZED_ENTROPY_EPSILON",
    "MISSING_TOP2_GAP",
    "EMPTY_ACTION_H0",
    "HIERARCHICAL_TEMPERATURE",
    "HARD_MAX_TIE_ULPS",
    "TARGET_RADII",
    "MIN_READY_ROOTS",
    "PARAMETER_COUNT",
    "P_SEED",
    "P_OPTIMIZER",
    "P_LEARNING_RATE",
    "P_WEIGHT_DECAY",
    "P_GRADIENT_CLIP_L2",
    "P_UPDATES_TOTAL",
    "P_EPISODES_PER_UPDATE",
    "P_WARMUP_UPDATES",
    "P_FINAL_LEARNING_RATE",
    "P_CHECKPOINT_INTERVAL_UPDATES",
    "P_SAMPLER_NAMESPACE",
    "P_LOCK_REQUIRED_FIELDS",
    "P_LOCK_ERASED_FIELDS",
    "SharedPContractError",
    "PEpisodeKey",
    "CandidateActionFeatureTable",
    "CandidatePOutput",
    "PairwisePNoRegretLoss",
    "SharedMultitilePHead",
    "candidate_action_colnomic_features",
    "canonical_action_argmax",
    "scale_balanced_hierarchical_logmeanexp",
    "pairwise_softplus_no_regret_loss",
    "canonical_training_state_bytes",
    "deterministic_p_epoch_order",
    "p_learning_rate",
    "make_frozen_p_optimizer",
]
