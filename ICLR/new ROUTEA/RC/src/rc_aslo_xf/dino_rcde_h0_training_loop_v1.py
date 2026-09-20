"""Deterministic resumable training loop for H0 relative+unary episodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
)
from .dino_rcde_h0_full_reference_unary_v1 import (
    decode_full_reference_unary,
    matched_absolute_h0_loss,
)
from .dino_rcde_h0_training_runtime_v1 import target_signed_local_pair_logit
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    FINAL_LR,
    GRADIENT_CLIP_L2,
    MAX_LR,
    UPDATES,
    WEIGHT_DECAY,
    VPairEpisode,
    episode_schedule_key,
    learning_rate,
    make_resume_state,
    ordered_training_pool,
    pair_loss,
    restore_rng_state,
    seed_everything,
    target_first,
    update_episodes,
    validate_resume_state,
)


MATCHED = "H0_FULL_REFERENCE_DONOR_MATCHED"
PAIR_ONLY = "H0_Q0_UNMATCHED_PAIR_ONLY"
H0_RESUME_SCHEMA = "rc_dino_rcde_h0_full_reference_unary_resume_v1_20260821"


class H0TrainingLoopError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise H0TrainingLoopError(message)


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _tree_hash(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(_canonical(list(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(_canonical(list(array.shape)).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                visit(key)
                visit(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for value in item:
                visit(value)
        elif isinstance(item, list):
            digest.update(b"list\0")
            for value in item:
                visit(value)
        else:
            digest.update(b"scalar\0")
            digest.update(
                json.dumps(item, sort_keys=True, ensure_ascii=True).encode("ascii")
            )
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def make_h0_resume_envelope(
    core_state: Mapping[str, Any],
    *,
    authority_sha256: str,
    protocol_sha256: str,
    ledger_sha256: str,
    ledger_logical_sha256: str,
    ledger_validation_sha256: str,
    augmentation_population_sha256: str,
) -> dict[str, Any]:
    bindings = {
        "authority_sha256": authority_sha256,
        "protocol_sha256": protocol_sha256,
        "ledger_sha256": ledger_sha256,
        "ledger_logical_sha256": ledger_logical_sha256,
        "ledger_validation_sha256": ledger_validation_sha256,
        "augmentation_population_sha256": augmentation_population_sha256,
        "objective": "DENSE_LOCAL_PAIR_PLUS_MATCHED_FULL_REFERENCE_UNARY_V1",
    }
    value = {
        "schema_version": H0_RESUME_SCHEMA,
        "bindings": bindings,
        "core_state": dict(core_state),
        "core_state_sha256": _tree_hash(core_state),
        "core_model_state_sha256": core_state["model_state_sha256"],
        "core_schedule_sha256": core_state["schedule_sha256"],
        "completed_updates": int(core_state["completed_updates"]),
    }
    value["logical_sha256"] = _canonical(
        {key: item for key, item in value.items() if key != "core_state"}
    )
    return value


def validate_h0_resume_envelope(
    value: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> Mapping[str, Any]:
    require(
        set(value)
        == {
            "schema_version",
            "bindings",
            "core_state",
            "core_state_sha256",
            "core_model_state_sha256",
            "core_schedule_sha256",
            "completed_updates",
            "logical_sha256",
        }
        and value.get("schema_version") == H0_RESUME_SCHEMA
        and value.get("logical_sha256")
        == _canonical(
            {
                key: item
                for key, item in value.items()
                if key not in {"logical_sha256", "core_state"}
            }
        )
        and value.get("bindings") == dict(expected_bindings)
        and value.get("core_state_sha256") == _tree_hash(value["core_state"])
        and value.get("core_model_state_sha256")
        == value.get("core_state", {}).get("model_state_sha256")
        and value.get("core_schedule_sha256")
        == value.get("core_state", {}).get("schedule_sha256")
        and value.get("completed_updates")
        == value.get("core_state", {}).get("completed_updates"),
        "H0 resume envelope/binding drift",
    )
    return value["core_state"]


class _MatchedLoss(tuple):
    __slots__ = ()

    total = property(lambda self: self[0])
    pair_logit = property(lambda self: self[1])
    positive_scores = property(lambda self: self[2])
    donor_scores = property(lambda self: self[3])


@dataclass(frozen=True)
class H0TrainEpisodeV1:
    pair: VPairEpisode
    donor: CandidateReferenceFieldV1 | None
    status: str
    augmentation_record_sha256: str

    def __post_init__(self) -> None:
        require(self.status in {MATCHED, PAIR_ONLY}, "H0 train status drift")
        require(
            (self.status == MATCHED and self.donor is not None)
            or (self.status == PAIR_ONLY and self.donor is None),
            "H0 train donor/status drift",
        )


def u3_matched_loss(
    model: torch.nn.Module,
    item: H0TrainEpisodeV1,
    *,
    target_first_value: bool,
) -> _MatchedLoss:
    assert item.donor is not None
    pair = target_signed_local_pair_logit(
        model,
        item.pair,
        target_first=target_first_value,
        streaming_chunk_size=None,
    )
    target = item.pair.locks[item.pair.target_key]
    positive_scores = []
    donor_scores = []
    for direction in FIXED_DIRECTIONS:
        query_mask = target.direction_locks[direction].query_union_mask
        positive = decode_full_reference_unary(
            model,
            item.pair.query,
            target.candidate,
            query_mask,
            structural_ready=True,
            streaming_chunk_size=64,
        )
        donor = decode_full_reference_unary(
            model,
            item.pair.query,
            item.donor,
            query_mask,
            structural_ready=True,
            streaming_chunk_size=64,
        )
        require(
            torch.equal(positive.query_mask, donor.query_mask),
            "U3 positive/donor query mask drift",
        )
        positive_scores.append(positive.score)
        donor_scores.append(donor.score)
    total = matched_absolute_h0_loss(
        pair_logit=pair,
        positive_scores=(positive_scores[0], positive_scores[1]),
        donor_scores=(donor_scores[0], donor_scores[1]),
    )
    return _MatchedLoss(
        (
            total,
            pair,
            (positive_scores[0], positive_scores[1]),
            (donor_scores[0], donor_scores[1]),
        )
    )


def train_h0_until_deadline(
    model: torch.nn.Module,
    episodes: Sequence[H0TrainEpisodeV1],
    *,
    outer_fold: int,
    initialization_checkpoint_sha256: str,
    resume_state: Mapping[str, Any] | None,
    deadline_monotonic: float,
    stop_after_updates: int = UPDATES,
    episode_to_device: Callable[[H0TrainEpisodeV1], H0TrainEpisodeV1]
    | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer, dict[str, Any]]:
    require(bool(episodes), "H0 training episode pool empty")
    require(0 <= stop_after_updates <= UPDATES, "H0 stop update drift")
    pairs = tuple(item.pair for item in episodes)
    ordered = ordered_training_pool(pairs, outer_fold)
    by_schedule = {episode_schedule_key(item.pair): item for item in episodes}
    require(len(by_schedule) == len(episodes), "H0 episode schedule alias")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY
    )
    loss_trace: list[Mapping[str, object]] = []
    schedule_events: list[Mapping[str, object]] = []
    completed = 0
    if resume_state is None:
        seed_everything()
    else:
        validate_resume_state(
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
        restore_rng_state(resume_state["rng_state"])
    start_update = completed
    segment_matched = 0
    segment_pair_only = 0
    model.train()
    while completed < stop_after_updates:
        if completed > start_update and time.monotonic() >= deadline_monotonic:
            break
        update_one = completed + 1
        selected_pairs = update_episodes(ordered, completed)
        directions = tuple(
            target_first(outer_fold, update_one, episode_schedule_key(item))
            for item in selected_pairs
        )
        selected = tuple(by_schedule[episode_schedule_key(item)] for item in selected_pairs)
        active_selected = (
            tuple(episode_to_device(item) for item in selected)
            if episode_to_device is not None
            else selected
        )
        event = {
            "update": update_one,
            "query_schedule_keys": [episode_schedule_key(item) for item in selected_pairs],
            "target_first": list(directions),
        }
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update_one)
        optimizer.zero_grad(set_to_none=True)
        total_values = []
        pair_values = []
        positive_values = []
        donor_values = []
        matched_count = 0
        for item, direction in zip(active_selected, directions, strict=True):
            if item.status == MATCHED:
                assert item.donor is not None
                result = u3_matched_loss(
                    model,
                    item,
                    target_first_value=direction,
                )
                pair_component = pair_loss(result.pair_logit)
                positive_component = 0.5 * sum(
                    F.softplus(-score) for score in result.positive_scores
                )
                donor_component = 0.5 * sum(
                    F.softplus(score) for score in result.donor_scores
                )
                total = result.total
                matched_count += 1
                segment_matched += 1
            else:
                pair_logit = target_signed_local_pair_logit(
                    model,
                    item.pair,
                    target_first=direction,
                    streaming_chunk_size=None,
                )
                pair_component = pair_loss(pair_logit)
                positive_component = pair_component * 0.0
                donor_component = pair_component * 0.0
                total = pair_component
                segment_pair_only += 1
            require(
                total.ndim == 0 and bool(torch.isfinite(total)),
                "H0 training loss nonfinite/nonscalar",
            )
            total_values.append(float(total.detach().cpu()))
            pair_values.append(float(pair_component.detach().cpu()))
            positive_values.append(float(positive_component.detach().cpu()))
            donor_values.append(float(donor_component.detach().cpu()))
            (total / len(active_selected)).backward()
            del total, pair_component, positive_component, donor_component
            if item.status == MATCHED:
                del result
            else:
                del pair_logit
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_L2
        )
        require(bool(torch.isfinite(gradient_norm)), "H0 gradient norm nonfinite")
        optimizer.step()
        loss_trace.append(
            {
                "update": update_one,
                "mean_total_loss": sum(total_values) / len(total_values),
                "mean_pair_loss": sum(pair_values) / len(pair_values),
                "mean_positive_unary_loss": sum(positive_values)
                / len(positive_values),
                "mean_donor_unary_loss": sum(donor_values)
                / len(donor_values),
                "matched_episode_count": matched_count,
                "pair_only_episode_count": len(active_selected) - matched_count,
                "learning_rate": learning_rate(update_one),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "query_schedule_keys": event["query_schedule_keys"],
            }
        )
        schedule_events.append(event)
        completed = update_one
    state = make_resume_state(
        model,
        optimizer,
        outer_fold=outer_fold,
        completed_updates=completed,
        loss_trace=loss_trace,
        schedule_events=schedule_events,
        ordered=ordered,
        initialization_checkpoint_sha256=initialization_checkpoint_sha256,
    )
    receipt = {
        "start_update": start_update,
        "stop_update": completed,
        "updates_completed_this_segment": completed - start_update,
        "matched_episode_evaluations_this_segment": segment_matched,
        "pair_only_episode_evaluations_this_segment": segment_pair_only,
        "deadline_reached": completed < stop_after_updates,
        "max_lr": MAX_LR,
        "final_lr": FINAL_LR,
    }
    return state, optimizer, receipt


__all__ = [
    "MATCHED",
    "PAIR_ONLY",
    "H0_RESUME_SCHEMA",
    "H0TrainingLoopError",
    "H0TrainEpisodeV1",
    "u3_matched_loss",
    "make_h0_resume_envelope",
    "validate_h0_resume_envelope",
    "train_h0_until_deadline",
]
