"""Exact-compute acceleration for the frozen Track-R V120-r5 recipe.

This module changes no arm, decode, loss, schedule, gradient accumulation or
optimizer arithmetic.  It removes repeated provenance work from the hot CUDA
loop after validating the canonical CPU episode pool once, binds model lineage
once per optimizer update (the model is unchanged across its four episodes),
and caches immutable masks within each episode decode.  Optional periodic
resume snapshots are engineering recovery artifacts only.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
)
from . import dino_rcde_sr0_mt_v_runtime_v1 as frozen


FAST_SCHEMA = "rc_dino_rcde_sr0_mt_v_runtime_fast_v1_20260822"


def configure_deterministic_backend() -> None:
    """Restore process-global numerical flags on fresh and resumed workers."""

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def validate_cpu_episode_pool_once(
    episodes: Sequence[frozen.VPairEpisode],
) -> None:
    """Replay token/provenance checks once before entering the CUDA loop."""

    for episode in episodes:
        if episode.query.layers.device.type != "cpu":
            raise frozen.SR0MTVContractError("fast V input query pool must be CPU")
        frozen._validate_decode_inputs(
            episode.query,
            episode.locks,
            episode.target_key,
            episode.strongest_rival_key,
        )


class CachedMaskRuntimeAdapter(frozen.DeviceMaskRuntimeAdapter):
    """Reuse exact boolean device masks within one three-arm episode decode."""

    def __init__(self, model: torch.nn.Module, model_sha: str, comparator_sha: str):
        super().__init__(model, model_sha, comparator_sha)
        self._mask_cache: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...], int, str, str],
            tuple[torch.Tensor, int, torch.Tensor, int],
        ] = {}

    def _mask(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        source = torch.as_tensor(value, dtype=torch.bool)
        if source.device == device:
            return source
        key = (
            source.untyped_storage().data_ptr(),
            source.storage_offset(),
            tuple(source.shape),
            tuple(source.stride()),
            source._version,
            str(device),
            str(source.dtype),
        )
        entry = self._mask_cache.get(key)
        if entry is None:
            cached = source.to(device=device, dtype=torch.bool)
            self._mask_cache[key] = (
                source,
                source._version,
                cached,
                cached._version,
            )
            return cached
        _sealed_source, source_version, cached, cached_version = entry
        if (
            source._version != source_version
            or cached._version != cached_version
        ):
            raise frozen.SR0MTVContractError("fast V cached mask mutated")
        return cached

    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **kwargs: Any,
    ) -> Any:
        return self.model.decode_candidate(
            query_layers,
            reference_layers,
            self._mask(query_mask, query_layers.device),
            self._mask(reference_mask, reference_layers.device),
            query_grid,
            reference_grid,
            **kwargs,
        )

    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Any:
        return self.model.compare_relational(
            relational_g,
            relational_c,
            self._mask(query_mask, relational_g.device),
        )


def _prevalidated_pair_locks(
    query: frozen.QueryTokenFieldV1,
    locks: Mapping[str, frozen.CandidatePLockV1],
    left_key: str,
    right_key: str,
) -> tuple[frozen.CandidatePLockV1, frozen.CandidatePLockV1]:
    if left_key == right_key or left_key not in locks or right_key not in locks:
        raise frozen.SR0MTVContractError("fast V pair axis drift")
    left, right = locks[left_key], locks[right_key]
    if (
        left.candidate.candidate_key != left_key
        or right.candidate.candidate_key != right_key
        or left.candidate.layers.device != query.layers.device
        or right.candidate.layers.device != query.layers.device
        or left.candidate.layers.dtype != query.layers.dtype
        or right.candidate.layers.dtype != query.layers.dtype
    ):
        raise frozen.SR0MTVContractError("fast V candidate/device/dtype drift")
    return left, right


def decode_pair_training_prevalidated(
    model: torch.nn.Module,
    query: frozen.QueryTokenFieldV1,
    locks: Mapping[str, frozen.CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    lineage: tuple[str, str, str],
) -> frozen.ThreeArmFixedDenominatorEvidenceV1:
    """Execute the exact three-arm graph using prevalidated immutable tokens."""

    left_lock, right_lock = _prevalidated_pair_locks(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, reducer_sha = lineage
    adapter = CachedMaskRuntimeAdapter(model, model_sha, comparator_sha)
    arms = []
    for arm_name in MANDATORY_ARMS:
        context = (
            torch.enable_grad()
            if arm_name == ARM_QUERY_LOCAL_COMPONENTS
            else torch.no_grad()
        )
        with context:
            forward = tuple(
                frozen.decode_direct_arm_term(
                    adapter,
                    query,
                    left_lock,
                    right_lock,
                    direction=direction,
                    arm_name=arm_name,
                )
                for direction in frozen.FIXED_DIRECTIONS
            )
            reverse = tuple(
                frozen.decode_direct_arm_term(
                    adapter,
                    query,
                    right_lock,
                    left_lock,
                    direction=direction,
                    arm_name=arm_name,
                )
                for direction in frozen.FIXED_DIRECTIONS
            )
            arms.append(
                frozen._reduce_arm(
                    arm_name,
                    left_key,
                    right_key,
                    forward,
                    reverse,
                )
            )
    return frozen.ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=frozen.BINDING_REAL,
        arms=tuple(arms),
    )


def target_signed_paired_logit_prevalidated(
    model: torch.nn.Module,
    episode: frozen.VPairEpisode,
    *,
    target_first: bool,
    lineage: tuple[str, str, str],
) -> torch.Tensor:
    left, right = (
        (episode.target_key, episode.strongest_rival_key)
        if target_first
        else (episode.strongest_rival_key, episode.target_key)
    )
    evidence = decode_pair_training_prevalidated(
        model,
        episode.query,
        episode.locks,
        left,
        right,
        lineage=lineage,
    )
    logit = evidence.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit
    return logit if target_first else -logit


def _atomic_periodic_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_updates_fast(
    model: torch.nn.Module,
    episodes: Sequence[frozen.VPairEpisode],
    *,
    outer_fold: int,
    initialization_checkpoint_sha256: str,
    stop_after_updates: int,
    resume_state: Mapping[str, Any] | None = None,
    loss_fn: Callable[[torch.nn.Module, frozen.VPairEpisode, bool], torch.Tensor]
    | None = None,
    update_observer: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    """Bit-exact recipe with hot-loop provenance and mask-transfer deduplication."""

    if loss_fn is not None:
        return frozen.train_updates(
            model,
            episodes,
            outer_fold=outer_fold,
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
            stop_after_updates=stop_after_updates,
            resume_state=resume_state,
            loss_fn=loss_fn,
            update_observer=update_observer,
        )
    if not 0 <= stop_after_updates <= frozen.UPDATES:
        raise frozen.SR0MTVContractError("fast V stop is outside 0..2048")
    validate_cpu_episode_pool_once(episodes)
    ordered = frozen.ordered_training_pool(episodes, outer_fold)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=frozen.MAX_LR, weight_decay=frozen.WEIGHT_DECAY
    )
    loss_trace: list[Mapping[str, object]] = []
    schedule_events: list[Mapping[str, object]] = []
    completed = 0
    configure_deterministic_backend()
    if resume_state is None:
        frozen.seed_everything()
    else:
        frozen.validate_resume_state(
            resume_state,
            outer_fold=outer_fold,
            ordered=ordered,
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        )
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        loss_trace = list(resume_state["loss_trace"])
        schedule_events = list(resume_state["schedule_events"])
        completed = int(resume_state["completed_updates"])
        frozen.restore_rng_state(resume_state["rng_state"])
    if stop_after_updates < completed:
        raise frozen.SR0MTVContractError("fast V resume moved backwards")

    periodic_raw = os.environ.get("RC_VFIT_PERIODIC_STATE")
    periodic = Path(periodic_raw) if periodic_raw else None
    checkpoint_interval = int(os.environ.get("RC_VFIT_CHECKPOINT_INTERVAL", "16"))
    progress_interval = int(os.environ.get("RC_VFIT_PROGRESS_INTERVAL", "8"))
    started = time.monotonic()
    model.train()
    while completed < stop_after_updates:
        update_one = completed + 1
        selected = frozen.update_episodes(ordered, completed)
        directions = tuple(
            frozen.target_first(
                outer_fold, update_one, frozen.episode_schedule_key(item)
            )
            for item in selected
        )
        event = {
            "update": update_one,
            "query_schedule_keys": [
                frozen.episode_schedule_key(item) for item in selected
            ],
            "target_first": list(directions),
        }
        for group in optimizer.param_groups:
            group["lr"] = frozen.learning_rate(update_one)
        optimizer.zero_grad(set_to_none=True)
        lineage = frozen.bind_runtime_lineage(model)
        detached_losses: list[torch.Tensor] = []
        device = next(model.parameters()).device
        for item, direction in zip(selected, directions, strict=True):
            # Retain the first complete CUDA token-byte verification performed
            # by the frozen constructors.  The accelerated decoder removes
            # only the second redundant SHA pass over these same tensors.
            active = frozen.move_episode_to_device(item, device)
            value = frozen.pair_loss(
                target_signed_paired_logit_prevalidated(
                    model,
                    active,
                    target_first=direction,
                    lineage=lineage,
                )
            )
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise frozen.SR0MTVContractError("fast V loss is nonfinite")
            detached_losses.append(value.detach().cpu())
            (value / frozen.QUERIES_PER_UPDATE).backward()
            del value, active
        loss = torch.stack(detached_losses).mean()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), frozen.GRADIENT_CLIP_L2
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise frozen.SR0MTVContractError("fast V gradient norm is nonfinite")
        optimizer.step()
        loss_trace.append(
            {
                "update": update_one,
                "mean_pair_loss": float(loss.detach().cpu()),
                "learning_rate": frozen.learning_rate(update_one),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "query_schedule_keys": event["query_schedule_keys"],
            }
        )
        schedule_events.append(event)
        completed = update_one
        row = {
            "update": update_one,
            "mean_pair_loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "query_schedule_keys": tuple(event["query_schedule_keys"]),
        }
        if update_observer is not None:
            update_observer(row)
        if progress_interval > 0 and (
            completed % progress_interval == 0 or completed == stop_after_updates
        ):
            print(
                f"FAST_V_PROGRESS fold={outer_fold} update={completed}/{stop_after_updates} "
                f"elapsed_seconds={time.monotonic()-started:.3f} "
                f"loss={row['mean_pair_loss']:.9g}",
                flush=True,
            )
        if periodic is not None and checkpoint_interval > 0 and (
            completed % checkpoint_interval == 0
            or completed == stop_after_updates
        ):
            snapshot = frozen.make_resume_state(
                model,
                optimizer,
                outer_fold=outer_fold,
                completed_updates=completed,
                loss_trace=loss_trace,
                schedule_events=schedule_events,
                ordered=ordered,
                initialization_checkpoint_sha256=initialization_checkpoint_sha256,
            )
            _atomic_periodic_state(periodic, snapshot)
            print(
                f"FAST_V_CHECKPOINT fold={outer_fold} update={completed} path={periodic}",
                flush=True,
            )
    return (
        frozen.make_resume_state(
            model,
            optimizer,
            outer_fold=outer_fold,
            completed_updates=completed,
            loss_trace=loss_trace,
            schedule_events=schedule_events,
            ordered=ordered,
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        ),
        optimizer,
    )


__all__ = [
    "FAST_SCHEMA",
    "CachedMaskRuntimeAdapter",
    "configure_deterministic_backend",
    "decode_pair_training_prevalidated",
    "target_signed_paired_logit_prevalidated",
    "train_updates_fast",
    "validate_cpu_episode_pool_once",
]
