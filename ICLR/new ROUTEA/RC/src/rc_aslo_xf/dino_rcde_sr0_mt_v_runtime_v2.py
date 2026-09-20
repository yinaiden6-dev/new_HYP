"""Append-only SR0-MT V runtime with a rootwise full-reference arm.

The authority-bound :mod:`dino_rcde_sr0_mt_v_runtime_v1` decoded the complete
CW1 query union against each full reference in one call.  That changed both
query granularity and reference scope relative to the local-component arm.

This successor keeps the V1 data model, model, comparator, four-term reducer,
training recipe, and serialization surface.  Both regional arms now use the
same ordered-root loop, overlap correction, query-union denominator, H0 path,
and missing-root convention.  The sole branch inside that loop selects either
the complete valid reference mask or the sealed root-bound component mask.

No authority, historical runtime, result, or package initializer is modified
by this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

import torch

# Re-export the unchanged public V1 runtime surface.  Corrected functions are
# defined below and therefore replace their V1 names in this module only.
from .dino_rcde_sr0_mt_v_runtime_v1 import *  # noqa: F401,F403
from . import dino_rcde_sr0_mt_v_runtime_v1 as _v1
from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_REAL,
    FIXED_DIRECTIONS,
    OrderedRootArmTermV1,
    RootPairDecodeReceiptV1,
    TERM_H0,
    TERM_NO_COMPARABLE_ROOT,
    TERM_READY,
    ThreeArmFixedDenominatorEvidenceV1,
    _decode_pair_masks,
    _reduce_arm,
    _zero_term,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import LOCK_H0, ROOT_READY


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_v_runtime_v2"
PARENT_RUNTIME_SCHEMA_VERSION = _v1.SCHEMA_VERSION
REGIONAL_REFERENCE_SCOPE_FULL = "FULL_VALID_REFERENCE"
REGIONAL_REFERENCE_SCOPE_LOCAL = "ROOT_BOUND_LOCAL_COMPONENT"


def _rootwise_regional_term(
    adapter: _v1.DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    owner_lock: SealedDirectionPLockV1,
    opponent_lock: SealedDirectionPLockV1,
    *,
    arm_name: str,
    reference_scope: str,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> OrderedRootArmTermV1:
    """Decode one regional term with one shared ordered-root reducer.

    ``reference_scope`` is deliberately the only branch that changes model
    inputs between FULL and LOCAL.  Root eligibility, order, query masks,
    overlap counts, denominator, and scalar reconstruction are shared code.
    """

    expected_arm = {
        REGIONAL_REFERENCE_SCOPE_FULL: ARM_QUERY_FULL_REFERENCE,
        REGIONAL_REFERENCE_SCOPE_LOCAL: ARM_QUERY_LOCAL_COMPONENTS,
    }.get(reference_scope)
    if expected_arm is None or arm_name != expected_arm:
        raise SR0MTVContractError("regional arm/reference-scope binding drift")

    union = owner_lock.query_union_mask
    if owner_lock.status == LOCK_H0:
        return _zero_term(
            arm_name=arm_name,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_lock.direction,
            status=TERM_H0,
            query_mask=union,
            exemplar=query.layers,
        )

    # Coverage is frozen over the complete owner root list.  A missing
    # comparable opponent root is skipped without changing these weights.
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
    for owner_root in owner_lock.selected_roots:
        coverage += owner_root.query_mask.to(torch.int64)

    receipts: list[RootPairDecodeReceiptV1] = []
    decoded: list[int] = []
    for root in owner_lock.ordered_root_ordinals:
        owner_root = owner_lock.all_roots[root]
        opponent_root = opponent_lock.all_roots[root]
        if opponent_root.binding_status != ROOT_READY:
            continue
        if not torch.equal(owner_root.query_mask, opponent_root.query_mask):
            raise SR0MTVContractError(
                "candidate-specific direct locks disagree on a fixed query root"
            )

        if reference_scope == REGIONAL_REFERENCE_SCOPE_FULL:
            owner_reference_mask = owner.valid_patch_mask
            opponent_reference_mask = opponent.valid_patch_mask
            owner_component_source_root_ordinal = None
            opponent_component_source_root_ordinal = None
        else:
            owner_reference_mask = owner_root.reference_mask
            opponent_reference_mask = opponent_root.reference_mask
            owner_component_source_root_ordinal = root
            opponent_component_source_root_ordinal = root

        pair, receipt = _decode_pair_masks(
            adapter,
            adapter,
            query,
            owner,
            opponent,
            query_mask=owner_root.query_mask,
            owner_reference_mask=owner_reference_mask,
            opponent_reference_mask=opponent_reference_mask,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        qmask = owner_root.query_mask.to(pair.contributions.device)
        total = total + pair.contributions * qmask.to(pair.contributions.dtype)
        receipts.append(
            RootPairDecodeReceiptV1(
                root_ordinal=root,
                owner_component_source_root_ordinal=(
                    owner_component_source_root_ordinal
                ),
                opponent_component_source_root_ordinal=(
                    opponent_component_source_root_ordinal
                ),
                binding_mode=BINDING_REAL,
                query_mask_sha256=receipt.query_mask_sha256,
                owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
                opponent_reference_mask_sha256=(
                    receipt.opponent_reference_mask_sha256
                ),
            )
        )
        decoded.append(root)

    if not decoded:
        return _zero_term(
            arm_name=arm_name,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_lock.direction,
            status=TERM_NO_COMPARABLE_ROOT,
            query_mask=union,
            exemplar=query.layers,
        )

    reciprocal = torch.zeros_like(total)
    covered = coverage > 0
    reciprocal[covered.to(reciprocal.device)] = 1.0 / coverage[covered].to(
        device=reciprocal.device, dtype=reciprocal.dtype
    )
    patch = total * reciprocal
    qmask = union.to(device=patch.device)
    patch = patch * qmask.to(patch.dtype)
    scalar = patch / qmask.sum().to(patch.dtype)
    return OrderedRootArmTermV1(
        arm_name=arm_name,
        owner_candidate_key=owner.candidate_key,
        opponent_candidate_key=opponent.candidate_key,
        direction=owner_lock.direction,
        status=TERM_READY,
        query_mask=union,
        patch_evidence=patch,
        scalar_contributions=scalar,
        logit=scalar.sum(),
        decoded_root_ordinals=tuple(decoded),
        decode_receipts=tuple(receipts),
    )


def decode_direct_arm_term(
    adapter: _v1.DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner_lock: CandidatePLockV1,
    opponent_lock: CandidatePLockV1,
    *,
    direction: str,
    arm_name: str,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> OrderedRootArmTermV1:
    """Decode one direct term under the corrected V2 regional contract."""

    owner = owner_lock.candidate
    opponent = opponent_lock.candidate
    if arm_name == ARM_ALL_PATCH:
        return _v1._all_patch_term(
            adapter,
            query,
            owner,
            opponent,
            direction=direction,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )

    sealed_owner = owner_lock.direction_locks[direction]
    sealed_opponent = opponent_lock.direction_locks[direction]
    if arm_name == ARM_QUERY_FULL_REFERENCE:
        scope = REGIONAL_REFERENCE_SCOPE_FULL
    elif arm_name == ARM_QUERY_LOCAL_COMPONENTS:
        scope = REGIONAL_REFERENCE_SCOPE_LOCAL
    else:
        raise SR0MTVContractError("unknown direct V arm")
    return _rootwise_regional_term(
        adapter,
        query,
        owner,
        opponent,
        sealed_owner,
        sealed_opponent,
        arm_name=arm_name,
        reference_scope=scope,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )


def _assert_matched_regional_structure(
    evidence: ThreeArmFixedDenominatorEvidenceV1,
) -> None:
    """Fail closed unless FULL and LOCAL differ only in reference scope."""

    arms = evidence.by_name()
    full = arms[ARM_QUERY_FULL_REFERENCE]
    local = arms[ARM_QUERY_LOCAL_COMPONENTS]
    term_pairs = (
        *zip(full.forward_by_direction, local.forward_by_direction, strict=True),
        *zip(full.reverse_by_direction, local.reverse_by_direction, strict=True),
    )
    for full_term, local_term in term_pairs:
        if (
            full_term.status != local_term.status
            or full_term.direction != local_term.direction
            or full_term.owner_candidate_key != local_term.owner_candidate_key
            or full_term.opponent_candidate_key != local_term.opponent_candidate_key
            or not torch.equal(full_term.query_mask, local_term.query_mask)
            or full_term.decoded_root_ordinals != local_term.decoded_root_ordinals
            or tuple(item.root_ordinal for item in full_term.decode_receipts)
            != tuple(item.root_ordinal for item in local_term.decode_receipts)
            or tuple(item.query_mask_sha256 for item in full_term.decode_receipts)
            != tuple(item.query_mask_sha256 for item in local_term.decode_receipts)
        ):
            raise SR0MTVContractError(
                "FULL/LOCAL root order, query mask, or fixed-denominator path drift"
            )
        if any(
            item.owner_component_source_root_ordinal is not None
            or item.opponent_component_source_root_ordinal is not None
            for item in full_term.decode_receipts
        ):
            raise SR0MTVContractError(
                "full-reference receipt retained a local component address"
            )


def _decode_pair_impl(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
    selective_autograd: bool,
) -> ThreeArmFixedDenominatorEvidenceV1:
    left_lock, right_lock = _v1._validate_decode_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, reducer_sha = _v1.bind_runtime_lineage(model)
    adapter = _v1.DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    arms = []
    for arm_name in MANDATORY_ARMS:
        if not selective_autograd:
            context = nullcontext()
        elif arm_name == ARM_QUERY_LOCAL_COMPONENTS:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()
        with context:
            forward = tuple(
                decode_direct_arm_term(
                    adapter,
                    query,
                    left_lock,
                    right_lock,
                    direction=direction,
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
                for direction in FIXED_DIRECTIONS
            )
            reverse = tuple(
                decode_direct_arm_term(
                    adapter,
                    query,
                    right_lock,
                    left_lock,
                    direction=direction,
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
                for direction in FIXED_DIRECTIONS
            )
            arms.append(
                _reduce_arm(
                    arm_name,
                    left_key,
                    right_key,
                    forward,  # type: ignore[arg-type]
                    reverse,  # type: ignore[arg-type]
                )
            )
    evidence = ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=tuple(arms),  # type: ignore[arg-type]
    )
    _assert_matched_regional_structure(evidence)
    return evidence


def decode_pair(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    """Decode all three arms with the corrected rootwise FULL arm."""

    return _decode_pair_impl(
        model,
        query,
        locks,
        left_key,
        right_key,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
        selective_autograd=False,
    )


def decode_pair_training_selective_autograd(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    """Use V2 decodes while retaining autograd only for the LOCAL loss arm."""

    return _decode_pair_impl(
        model,
        query,
        locks,
        left_key,
        right_key,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
        selective_autograd=True,
    )


def target_signed_paired_logit(
    model: torch.nn.Module, episode: VPairEpisode, *, target_first: bool
) -> torch.Tensor:
    """V1 pair objective evaluated through the corrected V2 runtime."""

    left, right = (
        (episode.target_key, episode.strongest_rival_key)
        if target_first
        else (episode.strongest_rival_key, episode.target_key)
    )
    evidence = decode_pair_training_selective_autograd(
        model, episode.query, episode.locks, left, right
    )
    logit = evidence.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit
    return logit if target_first else -logit


def train_updates(
    model: torch.nn.Module,
    episodes: Sequence[VPairEpisode],
    *,
    outer_fold: int,
    initialization_checkpoint_sha256: str,
    stop_after_updates: int,
    resume_state: Mapping[str, Any] | None = None,
    loss_fn: Callable[[torch.nn.Module, VPairEpisode, bool], torch.Tensor]
    | None = None,
    update_observer: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    """Delegate the frozen optimizer loop while routing natural loss via V2."""

    if loss_fn is not None:
        active_loss = loss_fn
    else:

        def active_loss(
            active_model: torch.nn.Module,
            item: VPairEpisode,
            order: bool,
        ) -> torch.Tensor:
            device_item = _v1.move_episode_to_device(
                item, next(active_model.parameters()).device
            )
            return _v1.pair_loss(
                target_signed_paired_logit(
                    active_model, device_item, target_first=order
                )
            )

    return _v1.train_updates(
        model,
        episodes,
        outer_fold=outer_fold,
        initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        stop_after_updates=stop_after_updates,
        resume_state=resume_state,
        loss_fn=active_loss,
        update_observer=update_observer,
    )
