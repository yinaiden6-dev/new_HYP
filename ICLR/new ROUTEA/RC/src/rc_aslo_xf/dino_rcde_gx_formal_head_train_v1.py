"""Pure formal trainer for GX cached relational comparator heads.

No file I/O is performed here.  The core consumes sealed pre-comparator
relational caches, trains exactly the live ``ell/rho/F`` heads, and exposes a
deterministic tensor state suitable for external serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_gx_cbnr_v3 import (
    NULL_NAMES,
    geometric_exclusivity,
    gx_cbnr_loss,
)
from .dino_rcde_gx_relational_head_cache_v1 import (
    AUXILIARY_BRANCHES,
    TRAINABLE_HEAD_GROUPS,
    TRAINING_BRANCHES,
    CandidateBranchRelationalCacheV1,
    replay_candidate_energy,
)


SCHEMA_VERSION = "rc_dino_rcde_gx_formal_head_train_v1_20260824"
STATE_SCHEMA_VERSION = "rc_dino_rcde_gx_formal_head_train_state_v1_20260824"
TOTAL_UPDATES = 128
BASE_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
SEED = 17
TRAINABLE_PARAMETER_COUNT = 848
SELECTED_PER_STRATUM = 24
SELECTED_EPISODE_COUNT = 48
RAW_CORRECT = "RAW_CORRECT"
RAW_WRONG = "RAW_WRONG"
STRATA = (RAW_CORRECT, RAW_WRONG)
LOSS_WEIGHT_BY_STRATUM = {RAW_CORRECT: 0.5, RAW_WRONG: 0.5}
EXPECTED_BRANCHES = ("REAL", *NULL_NAMES)


class GXFormalHeadTrainError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GXFormalHeadTrainError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes_hex": tensor.numpy().tobytes(order="C").hex(),
        }
    )


def clone_tree_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: clone_tree_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_tree_cpu(item) for item in value]
    return value


def tree_hash_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_sha256": tensor_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): tree_hash_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"tuple": [tree_hash_payload(item) for item in value]}
    if isinstance(value, list):
        return [tree_hash_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        require(not isinstance(value, float) or math.isfinite(value), "nonfinite state scalar")
        return value
    raise GXFormalHeadTrainError(f"unsupported state type: {type(value).__name__}")


def tree_sha256(value: Any) -> str:
    return canonical_sha256(tree_hash_payload(value))


def learning_rate(update_one_based: int) -> float:
    require(
        type(update_one_based) is int and 1 <= update_one_based <= TOTAL_UPDATES,
        "learning-rate update must be one of 1..128",
    )
    return BASE_LR * float(update_one_based) / float(TOTAL_UPDATES)


@dataclass(frozen=True)
class FormalHeadEpisodeV1:
    episode_id: str
    stratum: str
    raw_target_margin: torch.Tensor
    target_candidate_key: str
    rival_candidate_key: str
    target_branches: Mapping[str, CandidateBranchRelationalCacheV1]
    rival_branches: Mapping[str, CandidateBranchRelationalCacheV1]

    def __post_init__(self) -> None:
        require(
            isinstance(self.episode_id, str)
            and bool(self.episode_id)
            and self.stratum in STRATA
            and isinstance(self.target_candidate_key, str)
            and isinstance(self.rival_candidate_key, str)
            and self.target_candidate_key != self.rival_candidate_key,
            "formal episode address/stratum drift",
        )
        raw = torch.as_tensor(self.raw_target_margin)
        require(
            raw.device.type == "cpu"
            and raw.dtype == torch.float32
            and raw.ndim == 0
            and not raw.requires_grad
            and bool(torch.isfinite(raw)),
            "RAW margin must be a detached finite CPU float32 scalar",
        )
        require(
            (self.stratum == RAW_CORRECT and bool(raw > 0.0))
            or (self.stratum == RAW_WRONG and bool(raw <= 0.0)),
            "RAW margin/stratum mismatch",
        )
        target = dict(self.target_branches)
        rival = dict(self.rival_branches)
        require(
            set(target) == set(rival) == set(EXPECTED_BRANCHES)
            and not set(AUXILIARY_BRANCHES).intersection(target),
            "formal episode requires only REAL+C/P/N and rejects T",
        )
        reference_address = None
        for role, candidate_key, branches in (
            ("target", self.target_candidate_key, target),
            ("rival", self.rival_candidate_key, rival),
        ):
            for branch in EXPECTED_BRANCHES:
                cache = branches[branch]
                require(
                    isinstance(cache, CandidateBranchRelationalCacheV1)
                    and cache.branch == branch
                    and cache.candidate_key == candidate_key,
                    f"{role} cache branch/candidate binding drift",
                )
                cache.validate_bytes()
                address = (cache.outer_fold, cache.execution_ordinal, cache.query_id)
                if reference_address is None:
                    reference_address = address
                require(address == reference_address, "episode cache query/fold address drift")
        object.__setattr__(self, "raw_target_margin", raw.detach().clone())
        object.__setattr__(self, "target_branches", MappingProxyType(target))
        object.__setattr__(self, "rival_branches", MappingProxyType(rival))

    @property
    def logical_sha256(self) -> str:
        return canonical_sha256(
            {
                "episode_id": self.episode_id,
                "stratum": self.stratum,
                "raw_target_margin_sha256": tensor_sha256(self.raw_target_margin),
                "target_candidate_key": self.target_candidate_key,
                "rival_candidate_key": self.rival_candidate_key,
                "target_candidate_cache_sha256": {
                    branch: self.target_branches[branch].candidate_cache_sha256
                    for branch in EXPECTED_BRANCHES
                },
                "rival_candidate_cache_sha256": {
                    branch: self.rival_branches[branch].candidate_cache_sha256
                    for branch in EXPECTED_BRANCHES
                },
            }
        )


@dataclass(frozen=True)
class UpdateScheduleRecordV1:
    update: int
    raw_correct_episode_id: str
    raw_wrong_episode_id: str
    raw_correct_episode_sha256: str
    raw_wrong_episode_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        payload = {
            "update": self.update,
            "raw_correct_episode_id": self.raw_correct_episode_id,
            "raw_wrong_episode_id": self.raw_wrong_episode_id,
            "raw_correct_episode_sha256": self.raw_correct_episode_sha256,
            "raw_wrong_episode_sha256": self.raw_wrong_episode_sha256,
        }
        require(
            1 <= self.update <= TOTAL_UPDATES
            and self.record_sha256 == canonical_sha256(payload),
            "formal update schedule record drift",
        )


def schedule_order_key(episode: FormalHeadEpisodeV1) -> str:
    return canonical_sha256(
        [SCHEMA_VERSION, SEED, episode.stratum, episode.episode_id, episode.logical_sha256]
    )


def build_update_schedule(
    raw_correct_episodes: Sequence[FormalHeadEpisodeV1],
    raw_wrong_episodes: Sequence[FormalHeadEpisodeV1],
    *,
    seed: int = SEED,
) -> tuple[UpdateScheduleRecordV1, ...]:
    require(seed == SEED, "formal schedule seed must be exactly 17")
    correct = tuple(raw_correct_episodes)
    wrong = tuple(raw_wrong_episodes)
    require(correct and wrong, "both RAW strata must be nonempty")
    require(
        all(item.stratum == RAW_CORRECT for item in correct)
        and all(item.stratum == RAW_WRONG for item in wrong),
        "formal schedule pool stratum drift",
    )
    all_items = (*correct, *wrong)
    require(
        len({item.episode_id for item in all_items}) == len(all_items),
        "formal schedule episode ID collision",
    )
    correct = tuple(sorted(correct, key=schedule_order_key))
    wrong = tuple(sorted(wrong, key=schedule_order_key))
    records = []
    for update in range(1, TOTAL_UPDATES + 1):
        first = correct[(update - 1) % len(correct)]
        second = wrong[(update - 1) % len(wrong)]
        payload = {
            "update": update,
            "raw_correct_episode_id": first.episode_id,
            "raw_wrong_episode_id": second.episode_id,
            "raw_correct_episode_sha256": first.logical_sha256,
            "raw_wrong_episode_sha256": second.logical_sha256,
        }
        records.append(UpdateScheduleRecordV1(**payload, record_sha256=canonical_sha256(payload)))
    return tuple(records)


def validate_explicit_schedule(
    raw_correct_episodes: Sequence[FormalHeadEpisodeV1],
    raw_wrong_episodes: Sequence[FormalHeadEpisodeV1],
    schedule: Sequence[UpdateScheduleRecordV1],
) -> tuple[UpdateScheduleRecordV1, ...]:
    """Validate the frozen loss-role schedule without reordering or pairing it."""

    correct = tuple(raw_correct_episodes)
    wrong = tuple(raw_wrong_episodes)
    records = tuple(schedule)
    require(
        len(correct) == len(wrong) == SELECTED_PER_STRATUM
        and len(records) == TOTAL_UPDATES,
        "formal manifest requires 24+24 selected episodes and 128 updates",
    )
    all_episodes = (*correct, *wrong)
    require(
        len({item.episode_id for item in all_episodes})
        == SELECTED_EPISODE_COUNT
        and all(item.stratum == RAW_CORRECT for item in correct)
        and all(item.stratum == RAW_WRONG for item in wrong),
        "formal selected episode population/strata drift",
    )
    by_id = {item.episode_id: item for item in all_episodes}
    consumed_correct = set()
    consumed_wrong = set()
    for expected_update, record in enumerate(records, start=1):
        require(
            isinstance(record, UpdateScheduleRecordV1)
            and record.update == expected_update
            and record.raw_correct_episode_id in by_id
            and record.raw_wrong_episode_id in by_id,
            "explicit formal schedule update/address drift",
        )
        first = by_id[record.raw_correct_episode_id]
        second = by_id[record.raw_wrong_episode_id]
        require(
            first.stratum == RAW_CORRECT
            and second.stratum == RAW_WRONG
            and record.raw_correct_episode_sha256 == first.logical_sha256
            and record.raw_wrong_episode_sha256 == second.logical_sha256,
            "explicit formal schedule episode SHA/stratum drift",
        )
        consumed_correct.add(first.episode_id)
        consumed_wrong.add(second.episode_id)
    require(
        consumed_correct == {item.episode_id for item in correct}
        and consumed_wrong == {item.episode_id for item in wrong},
        "explicit formal schedule does not consume all 48 selected episodes",
    )
    return records


def parameter_contract(model: torch.nn.Module) -> dict[str, Any]:
    groups_method = getattr(model, "functional_parameter_groups", None)
    require(callable(groups_method), "model functional parameter groups absent")
    groups = groups_method()
    require(
        isinstance(groups, Mapping)
        and set(TRAINABLE_HEAD_GROUPS).issubset(groups),
        "ell/rho/F parameter groups absent",
    )
    named = dict(model.named_parameters())
    head_names = tuple(
        name for group in TRAINABLE_HEAD_GROUPS for name in groups[group]
    )
    require(
        len(set(head_names)) == len(head_names)
        and set(head_names).issubset(named),
        "ell/rho/F parameter-name population drift",
    )
    trainable_count = sum(named[name].numel() for name in head_names)
    require(
        trainable_count == TRAINABLE_PARAMETER_COUNT,
        "formal trainable parameter count is not 848",
    )
    for name, parameter in named.items():
        parameter.requires_grad_(name in head_names)
    frozen_names = tuple(sorted(set(named) - set(head_names)))
    return {
        "trainable_group_names": list(TRAINABLE_HEAD_GROUPS),
        "trainable_parameter_names": list(head_names),
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_names": list(frozen_names),
        "frozen_parameter_count": sum(named[name].numel() for name in frozen_names),
        "frozen_parameter_sha256_by_name": {
            name: tensor_sha256(named[name]) for name in frozen_names
        },
    }


def verify_frozen_parameters(
    model: torch.nn.Module, expected: Mapping[str, str]
) -> None:
    named = dict(model.named_parameters())
    require(
        set(expected).issubset(named)
        and all(tensor_sha256(named[name]) == digest for name, digest in expected.items()),
        "frozen parameter bytes changed",
    )


def make_optimizer(model: torch.nn.Module, contract: Mapping[str, Any]) -> torch.optim.AdamW:
    named = dict(model.named_parameters())
    parameters = [named[name] for name in contract["trainable_parameter_names"]]
    require(
        sum(parameter.numel() for parameter in parameters) == TRAINABLE_PARAMETER_COUNT
        and all(parameter.requires_grad for parameter in parameters),
        "optimizer parameter population drift",
    )
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate(1),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def episode_loss(model: torch.nn.Module, episode: FormalHeadEpisodeV1):
    target_energies = {
        branch: replay_candidate_energy(
            model, episode.target_branches[branch], for_training=True
        ).energy
        for branch in EXPECTED_BRANCHES
    }
    rival_energies = {
        branch: replay_candidate_energy(
            model, episode.rival_branches[branch], for_training=True
        ).energy
        for branch in EXPECTED_BRANCHES
    }
    target = geometric_exclusivity(
        target_energies["REAL"],
        tuple(target_energies[name] for name in NULL_NAMES),
    )
    rival = geometric_exclusivity(
        rival_energies["REAL"],
        tuple(rival_energies[name] for name in NULL_NAMES),
    )
    raw = episode.raw_target_margin.to(
        device=target.z.device, dtype=target.z.dtype
    )
    return gx_cbnr_loss(raw_target_margin=raw, target=target, rival=rival)


@dataclass
class FormalHeadTrainerV1:
    model: torch.nn.Module
    optimizer: torch.optim.AdamW
    parameter_contract: Mapping[str, Any]
    schedule: tuple[UpdateScheduleRecordV1, ...]
    episodes_by_id: Mapping[str, FormalHeadEpisodeV1]
    completed_updates: int
    trace: list[dict[str, Any]]
    deterministic_generator_state: torch.Tensor

    def __post_init__(self) -> None:
        require(
            len(self.schedule) == TOTAL_UPDATES
            and 0 <= self.completed_updates <= TOTAL_UPDATES
            and len(self.trace) == self.completed_updates,
            "formal trainer state cardinality drift",
        )
        require(
            isinstance(self.deterministic_generator_state, torch.Tensor)
            and self.deterministic_generator_state.device.type == "cpu"
            and self.deterministic_generator_state.dtype == torch.uint8,
            "formal trainer seed17 generator state drift",
        )
        object.__setattr__(self, "episodes_by_id", MappingProxyType(dict(self.episodes_by_id)))


def start_training(
    model: torch.nn.Module,
    raw_correct_episodes: Sequence[FormalHeadEpisodeV1],
    raw_wrong_episodes: Sequence[FormalHeadEpisodeV1],
    schedule: Sequence[UpdateScheduleRecordV1],
) -> FormalHeadTrainerV1:
    contract = parameter_contract(model)
    frozen_schedule = validate_explicit_schedule(
        raw_correct_episodes, raw_wrong_episodes, schedule
    )
    episodes = {
        item.episode_id: item
        for item in (*tuple(raw_correct_episodes), *tuple(raw_wrong_episodes))
    }
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return FormalHeadTrainerV1(
        model=model,
        optimizer=make_optimizer(model, contract),
        parameter_contract=MappingProxyType(contract),
        schedule=frozen_schedule,
        episodes_by_id=episodes,
        completed_updates=0,
        trace=[],
        deterministic_generator_state=generator.get_state().clone(),
    )


def gradient_receipt(
    model: torch.nn.Module, contract: Mapping[str, Any]
) -> dict[str, Any]:
    named = dict(model.named_parameters())
    groups = model.functional_parameter_groups()
    rows = {}
    for group in TRAINABLE_HEAD_GROUPS:
        parameter_names = tuple(groups[group])
        values = [named[name].grad for name in parameter_names]
        require(
            all(value is not None for value in values)
            and all(bool(torch.isfinite(value).all()) for value in values if value is not None),
            f"{group} gradient absent/nonfinite",
        )
        parameters = {
            name: {
                "finite": bool(torch.isfinite(value).all()),
                "nonzero": bool(value.detach().ne(0).any()),
                "maximum_absolute_gradient": float(value.detach().abs().max()),
            }
            for name, value in zip(parameter_names, values, strict=True)
            if value is not None
        }
        rows[group] = {
            "parameter_names": list(parameter_names),
            "all_finite": True,
            "all_nonzero": all(row["nonzero"] for row in parameters.values()),
            "maximum_absolute_gradient": max(
                float(value.detach().abs().max()) for value in values if value is not None
            ),
            "parameters": parameters,
        }
    return rows


def gradient_coverage_receipt(
    trace: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    names = tuple(contract["trainable_parameter_names"])
    rows = {}
    for name in names:
        observations = []
        for update in trace:
            for group in TRAINABLE_HEAD_GROUPS:
                parameter = update["gradient_groups"][group]["parameters"].get(name)
                if parameter is not None:
                    observations.append(parameter)
        require(len(observations) == len(trace), "gradient trace parameter coverage drift")
        rows[name] = {
            "update_count": len(observations),
            "all_finite": all(item["finite"] for item in observations),
            "ever_nonzero": any(item["nonzero"] for item in observations),
            "maximum_absolute_gradient": max(
                (float(item["maximum_absolute_gradient"]) for item in observations),
                default=0.0,
            ),
        }
    value = {
        "parameter_count": len(rows),
        "parameters": rows,
        "all_tensors_finite": all(row["all_finite"] for row in rows.values()),
        "all_tensors_ever_nonzero": all(
            row["ever_nonzero"] for row in rows.values()
        ),
    }
    value["logical_sha256"] = canonical_sha256(value)
    return value


def run_updates(
    trainer: FormalHeadTrainerV1, *, stop_after_updates: int = TOTAL_UPDATES
) -> FormalHeadTrainerV1:
    require(
        type(stop_after_updates) is int
        and trainer.completed_updates <= stop_after_updates <= TOTAL_UPDATES,
        "formal stop update drift",
    )
    named = dict(trainer.model.named_parameters())
    trainable = [
        named[name] for name in trainer.parameter_contract["trainable_parameter_names"]
    ]
    while trainer.completed_updates < stop_after_updates:
        update = trainer.completed_updates + 1
        schedule = trainer.schedule[update - 1]
        correct = trainer.episodes_by_id[schedule.raw_correct_episode_id]
        wrong = trainer.episodes_by_id[schedule.raw_wrong_episode_id]
        require(
            correct.stratum == RAW_CORRECT and wrong.stratum == RAW_WRONG,
            "update is not exactly one RAW-correct plus one RAW-wrong",
        )
        lr = learning_rate(update)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
            require(
                group["weight_decay"] == WEIGHT_DECAY,
                "AdamW weight-decay drift",
            )
        trainer.optimizer.zero_grad(set_to_none=True)
        correct_loss = episode_loss(trainer.model, correct)
        wrong_loss = episode_loss(trainer.model, wrong)
        total = (
            LOSS_WEIGHT_BY_STRATUM[RAW_CORRECT] * correct_loss.total
            + LOSS_WEIGHT_BY_STRATUM[RAW_WRONG] * wrong_loss.total
        )
        require(total.ndim == 0 and bool(torch.isfinite(total)), "formal loss nonfinite")
        total.backward()
        gradients = gradient_receipt(trainer.model, trainer.parameter_contract)
        norm = torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
        require(bool(torch.isfinite(norm)), "formal gradient norm nonfinite")
        trainer.optimizer.step()
        verify_frozen_parameters(
            trainer.model,
            trainer.parameter_contract["frozen_parameter_sha256_by_name"],
        )
        record = {
            "update": update,
            "learning_rate": lr,
            "raw_correct_episode_id": correct.episode_id,
            "raw_wrong_episode_id": wrong.episode_id,
            "raw_correct_loss_weight": 0.5,
            "raw_wrong_loss_weight": 0.5,
            "raw_correct_loss": float(correct_loss.total.detach().cpu()),
            "raw_wrong_loss": float(wrong_loss.total.detach().cpu()),
            "combined_loss": float(total.detach().cpu()),
            "gradient_norm_before_clip": float(norm.detach().cpu()),
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "gradient_groups": gradients,
            "optimizer_step_count": update,
            "frozen_parameters_byte_identical": True,
        }
        record["record_sha256"] = canonical_sha256(record)
        trainer.trace.append(record)
        trainer.completed_updates = update
    return trainer


@dataclass(frozen=True)
class FormalHeadTrainingStateV1:
    schema_version: str
    completed_updates: int
    schedule_sha256: str
    parameter_contract: Mapping[str, Any]
    model_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: Mapping[str, Any]
    trace: tuple[Mapping[str, Any], ...]
    torch_cpu_rng_state: torch.Tensor
    model_state_sha256: str
    optimizer_state_sha256: str
    trace_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        require(
            self.schema_version == STATE_SCHEMA_VERSION
            and 0 <= self.completed_updates <= TOTAL_UPDATES
            and len(self.trace) == self.completed_updates
            and self.model_state_sha256 == tree_sha256(self.model_state_dict)
            and self.optimizer_state_sha256
            == tree_sha256(self.optimizer_state_dict)
            and self.trace_sha256 == canonical_sha256(list(self.trace)),
            "formal serialized state payload drift",
        )
        payload = {
            "schema_version": self.schema_version,
            "completed_updates": self.completed_updates,
            "schedule_sha256": self.schedule_sha256,
            "parameter_contract_sha256": canonical_sha256(
                dict(self.parameter_contract)
            ),
            "model_state_sha256": self.model_state_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "trace_sha256": self.trace_sha256,
            "torch_cpu_rng_state_sha256": tensor_sha256(
                self.torch_cpu_rng_state
            ),
        }
        require(
            self.logical_sha256 == canonical_sha256(payload),
            "formal serialized state logical SHA drift",
        )


def export_training_state(
    trainer: FormalHeadTrainerV1,
) -> FormalHeadTrainingStateV1:
    verify_frozen_parameters(
        trainer.model,
        trainer.parameter_contract["frozen_parameter_sha256_by_name"],
    )
    model_state = clone_tree_cpu(trainer.model.state_dict())
    optimizer_state = clone_tree_cpu(trainer.optimizer.state_dict())
    trace = tuple(dict(row) for row in trainer.trace)
    coverage = gradient_coverage_receipt(trace, trainer.parameter_contract)
    if trainer.completed_updates == TOTAL_UPDATES:
        require(
            coverage["all_tensors_finite"] is True
            and coverage["all_tensors_ever_nonzero"] is True,
            "formal training completed without finite nonzero gradient coverage for every head tensor",
        )
    schedule_sha = canonical_sha256(
        [row.record_sha256 for row in trainer.schedule]
    )
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "completed_updates": trainer.completed_updates,
        "schedule_sha256": schedule_sha,
        "parameter_contract_sha256": canonical_sha256(
            dict(trainer.parameter_contract)
        ),
        "model_state_sha256": tree_sha256(model_state),
        "optimizer_state_sha256": tree_sha256(optimizer_state),
        "trace_sha256": canonical_sha256(list(trace)),
        "torch_cpu_rng_state_sha256": tensor_sha256(
            trainer.deterministic_generator_state
        ),
    }
    return FormalHeadTrainingStateV1(
        schema_version=STATE_SCHEMA_VERSION,
        completed_updates=trainer.completed_updates,
        schedule_sha256=schedule_sha,
        parameter_contract=clone_tree_cpu(trainer.parameter_contract),
        model_state_dict=model_state,
        optimizer_state_dict=optimizer_state,
        trace=trace,
        torch_cpu_rng_state=trainer.deterministic_generator_state.clone(),
        model_state_sha256=str(payload["model_state_sha256"]),
        optimizer_state_sha256=str(payload["optimizer_state_sha256"]),
        trace_sha256=str(payload["trace_sha256"]),
        logical_sha256=canonical_sha256(payload),
    )


def resume_training(
    model: torch.nn.Module,
    raw_correct_episodes: Sequence[FormalHeadEpisodeV1],
    raw_wrong_episodes: Sequence[FormalHeadEpisodeV1],
    schedule: Sequence[UpdateScheduleRecordV1],
    state: FormalHeadTrainingStateV1,
) -> FormalHeadTrainerV1:
    require(
        isinstance(state, FormalHeadTrainingStateV1),
        "resume requires a validated formal state",
    )
    trainer = start_training(
        model, raw_correct_episodes, raw_wrong_episodes, schedule
    )
    schedule_sha = canonical_sha256(
        [row.record_sha256 for row in trainer.schedule]
    )
    require(
        schedule_sha == state.schedule_sha256
        and trainer.parameter_contract == state.parameter_contract,
        "resume schedule/parameter contract drift",
    )
    trainer.model.load_state_dict(state.model_state_dict, strict=True)
    trainer.optimizer.load_state_dict(state.optimizer_state_dict)
    trainer.completed_updates = state.completed_updates
    trainer.trace = [dict(row) for row in state.trace]
    trainer.deterministic_generator_state = state.torch_cpu_rng_state.clone()
    verify_frozen_parameters(
        trainer.model,
        trainer.parameter_contract["frozen_parameter_sha256_by_name"],
    )
    require(
        tree_sha256(trainer.model.state_dict()) == state.model_state_sha256
        and tree_sha256(trainer.optimizer.state_dict())
        == state.optimizer_state_sha256,
        "resume tensor state reconstruction drift",
    )
    return trainer


__all__ = [
    "BASE_LR",
    "EXPECTED_BRANCHES",
    "FormalHeadEpisodeV1",
    "FormalHeadTrainerV1",
    "FormalHeadTrainingStateV1",
    "GRADIENT_CLIP_NORM",
    "GXFormalHeadTrainError",
    "LOSS_WEIGHT_BY_STRATUM",
    "RAW_CORRECT",
    "RAW_WRONG",
    "SCHEMA_VERSION",
    "SEED",
    "STATE_SCHEMA_VERSION",
    "TOTAL_UPDATES",
    "TRAINABLE_PARAMETER_COUNT",
    "UpdateScheduleRecordV1",
    "SELECTED_EPISODE_COUNT",
    "SELECTED_PER_STRATUM",
    "WEIGHT_DECAY",
    "build_update_schedule",
    "canonical_sha256",
    "episode_loss",
    "export_training_state",
    "gradient_coverage_receipt",
    "learning_rate",
    "parameter_contract",
    "resume_training",
    "run_updates",
    "start_training",
    "tensor_sha256",
    "tree_sha256",
    "validate_explicit_schedule",
]
