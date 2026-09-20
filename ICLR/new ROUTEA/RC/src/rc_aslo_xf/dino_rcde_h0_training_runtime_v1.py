"""Training-only H0 runtime: exact local pair plus full-reference unary."""

from __future__ import annotations

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import CandidateReferenceFieldV1
from .dino_rcde_cw1_multitile_superregion_v2 import ARM_QUERY_LOCAL_COMPONENTS
from .dino_rcde_cw1_multitile_vdecode_v1 import FIXED_DIRECTIONS, _reduce_arm
from .dino_rcde_h0_full_reference_unary_v1 import (
    decode_full_reference_unary,
    matched_absolute_h0_loss,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    DeviceMaskRuntimeAdapter,
    VPairEpisode,
    _validate_decode_inputs,
    bind_runtime_lineage,
    decode_direct_arm_term,
)


class H0TrainingRuntimeError(ValueError):
    pass


class MatchedH0LossV1(tuple):
    __slots__ = ()

    @property
    def total(self) -> torch.Tensor:
        return self[0]

    @property
    def pair_logit(self) -> torch.Tensor:
        return self[1]

    @property
    def positive_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self[2]

    @property
    def donor_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self[3]


def target_signed_local_pair_logit(
    model: torch.nn.Module,
    episode: VPairEpisode,
    *,
    target_first: bool,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Decode only the frozen local-component arm used by pairwise loss.

    The mandatory three-arm decoder remains unchanged for evaluation. Training
    does not construct unrelated arm presentation objects whose logits never
    enter the objective.
    """

    left, right = (
        (episode.target_key, episode.strongest_rival_key)
        if target_first
        else (episode.strongest_rival_key, episode.target_key)
    )
    left_lock, right_lock = _validate_decode_inputs(
        episode.query, episode.locks, left, right
    )
    model_sha, comparator_sha, _ = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    forward = tuple(
        decode_direct_arm_term(
            adapter,
            episode.query,
            left_lock,
            right_lock,
            direction=direction,
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        for direction in FIXED_DIRECTIONS
    )
    reverse = tuple(
        decode_direct_arm_term(
            adapter,
            episode.query,
            right_lock,
            left_lock,
            direction=direction,
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        for direction in FIXED_DIRECTIONS
    )
    arm = _reduce_arm(
        ARM_QUERY_LOCAL_COMPONENTS,
        left,
        right,
        forward,  # type: ignore[arg-type]
        reverse,  # type: ignore[arg-type]
    )
    return arm.logit if target_first else -arm.logit


def matched_episode_loss(
    model: torch.nn.Module,
    episode: VPairEpisode,
    donor: CandidateReferenceFieldV1,
    *,
    target_first: bool,
    streaming_chunk_size: int | None = 64,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> MatchedH0LossV1:
    if donor.candidate_key in episode.locks:
        raise H0TrainingRuntimeError("donor collides with target/rival locks")
    pair = target_signed_local_pair_logit(
        model,
        episode,
        target_first=target_first,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    target = episode.locks[episode.target_key]
    positive_scores = []
    donor_scores = []
    for direction in FIXED_DIRECTIONS:
        sealed = target.direction_locks[direction]
        positive = decode_full_reference_unary(
            model,
            episode.query,
            target.candidate,
            sealed.query_union_mask,
            structural_ready=True,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        negative = decode_full_reference_unary(
            model,
            episode.query,
            donor,
            sealed.query_union_mask,
            structural_ready=True,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        if not torch.equal(positive.query_mask, negative.query_mask):
            raise H0TrainingRuntimeError("positive/donor query masks differ")
        positive_scores.append(positive.score)
        donor_scores.append(negative.score)
    total = matched_absolute_h0_loss(
        pair_logit=pair,
        positive_scores=(positive_scores[0], positive_scores[1]),
        donor_scores=(donor_scores[0], donor_scores[1]),
    )
    return MatchedH0LossV1(
        (
            total,
            pair,
            (positive_scores[0], positive_scores[1]),
            (donor_scores[0], donor_scores[1]),
        )
    )


__all__ = [
    "H0TrainingRuntimeError",
    "MatchedH0LossV1",
    "target_signed_local_pair_logit",
    "matched_episode_loss",
]
