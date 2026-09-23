from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "programs"))

import run_dino_rcde_sr0_mt_p_v2_fit_v1 as runner  # noqa: E402
import validate_dino_rcde_sr0_mt_p_v2_fit_v1 as validator  # noqa: E402
from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import PEpisodeKey  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    DeterministicEpisodeSampler,
    initialize_training,
)


def _keys(count: int = 8) -> tuple[PEpisodeKey, ...]:
    return tuple(
        PEpisodeKey(
            query_id=f"fixture-{index}",
            source_image_sha256=hashlib.sha256(
                f"fixture-source-{index}".encode("ascii")
            ).hexdigest(),
            execution_ordinal=index,
        )
        for index in range(count)
    )


def _bindings() -> dict[str, str]:
    return {
        key: hashlib.sha256(f"binding:{key}".encode("ascii")).hexdigest()
        for key in runner.BINDING_FIELDS
    }


def _fake_episode_loss(model, episode):
    scale = model.linear.weight.new_tensor(float(episode + 1) / 17.0)
    target = model.linear.weight.sum() * scale
    rival = -0.25 * target
    margin = target - rival
    return F.softplus(-margin), target, rival, margin


def _checkpoint(
    *,
    kind: str,
    completed: int,
    bindings,
    model,
    optimizer,
    sampler,
    trace,
    restored_file=None,
    restored_state=None,
):
    return runner.make_checkpoint(
        fit_id="P_OUTER1_INNER2_FIT",
        fit_index=0,
        path_kind=kind,
        completed_updates=completed,
        bindings=bindings,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        rolling_trace_state=trace,
        restored_from_checkpoint_file_sha256=restored_file,
        restored_from_checkpoint_state_sha256=restored_state,
    )


def test_small_update64_checkpoint_restore_and_continuation_are_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "episode_loss", _fake_episode_loss)
    torch.set_num_threads(1)
    keys = _keys()
    episodes = tuple(range(len(keys)))
    bindings = _bindings()

    model, optimizer = initialize_training(17)
    sampler = DeterministicEpisodeSampler(keys, fit_id="P_OUTER1_INNER2_FIT")
    trace64 = runner.run_updates(
        model=model,
        optimizer=optimizer,
        episodes=episodes,
        sampler=sampler,
        start_update=0,
        stop_update=64,
        rolling_trace_state=runner.initial_trace("P_OUTER1_INNER2_FIT"),
        checkpoint_callback=None,
    )
    checkpoint64 = _checkpoint(
        kind="PRIMARY",
        completed=64,
        bindings=bindings,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        trace=trace64,
    )
    path64 = tmp_path / "checkpoint_update0064.pt"
    runner.exclusive_torch(path64, checkpoint64)
    assert (path64.stat().st_mode & 0o777) == 0o444
    loaded64 = torch.load(
        path64, map_location="cpu", weights_only=True, mmap=True
    )
    runner.validate_checkpoint(
        loaded64,
        fit_id="P_OUTER1_INNER2_FIT",
        fit_index=0,
        path_kind="PRIMARY",
        completed_updates=64,
        bindings=bindings,
    )

    primary_trace = runner.run_updates(
        model=model,
        optimizer=optimizer,
        episodes=episodes,
        sampler=sampler,
        start_update=64,
        stop_update=128,
        rolling_trace_state=trace64,
        checkpoint_callback=None,
    )
    primary128 = _checkpoint(
        kind="PRIMARY",
        completed=128,
        bindings=bindings,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        trace=primary_trace,
    )

    replay_model, replay_optimizer, replay_sampler, replay_trace = (
        runner.restore_from_checkpoint(
            loaded64,
            episode_keys=keys,
            fit_id="P_OUTER1_INNER2_FIT",
        )
    )
    replay_trace = runner.run_updates(
        model=replay_model,
        optimizer=replay_optimizer,
        episodes=episodes,
        sampler=replay_sampler,
        start_update=64,
        stop_update=128,
        rolling_trace_state=replay_trace,
        checkpoint_callback=None,
    )
    file_sha = runner.file_sha(path64)
    replay128 = _checkpoint(
        kind="REPLAY",
        completed=128,
        bindings=bindings,
        model=replay_model,
        optimizer=replay_optimizer,
        sampler=replay_sampler,
        trace=replay_trace,
        restored_file=file_sha,
        restored_state=loaded64["state_sha256"],
    )

    assert primary128["state_sha256"] == replay128["state_sha256"]
    assert runner.terminal_summary(primary128) == runner.terminal_summary(replay128)
    assert primary128["rolling_trace_state"] == replay128["rolling_trace_state"]
    assert primary128["logical_sha256"] != replay128["logical_sha256"]


def test_constructed_2048_trace_has_32_independently_validated_segments() -> None:
    trace = runner.initial_trace("P_OUTER1_INNER2_FIT")
    checkpoints = []
    for update in range(1, 2049):
        trace = runner.append_trace(trace, {"completed_update": update})
        if update % 64 == 0:
            runner.validate_trace(trace, update)
            checkpoints.append(
                {
                    "completed_updates": update,
                    "rolling_trace_state": dict(trace),
                }
            )
    assert len(checkpoints) == 32
    assert checkpoints[15]["completed_updates"] == 1024
    assert checkpoints[-1]["completed_updates"] == 2048
    assert validator._validate_trace_chain(checkpoints) == trace["chain_sha256"]


def test_checkpoint_and_result_schemas_are_exactly_validator_aligned() -> None:
    assert runner.CHECKPOINT_SCHEMA == validator.CHECKPOINT_SCHEMA
    assert runner.CHECKPOINT_STATUS == validator.CHECKPOINT_STATUS
    assert runner.CHECKPOINT_FIELDS == validator.CHECKPOINT_FIELDS
    assert runner.RESULT_SCHEMA == validator.RESULT_SCHEMA
    assert runner.RESULT_STATUS == validator.RESULT_STATUS
    assert runner.RESULT_CLAIM == validator.RESULT_CLAIM
    assert runner.RESULT_FIELDS == validator.RESULT_FIELDS
    assert set(
        runner.terminal_summary(
            {
                "completed_updates": 2048,
                "state_sha256": "a" * 64,
                "model_state": {},
                "optimizer_state": {},
                "sampler_state": {},
                "rng_state": {},
                "rolling_trace_state": {},
                "scheduler_cursor": 2048,
            }
        )
    ) == validator.TERMINAL_SUMMARY_FIELDS


def test_protected_access_schema_and_exact_counts_match_validator_formula() -> None:
    route_count = 36
    access = {
        "pair_feature_payload_read_count": route_count,
        "sidecar_shard_payload_read_count": route_count,
        "loss_role_read_count": 1,
        "model_load_count": 2,
        "model_forward_count": (2048 + 1024) * 4 * 4 + 2 * route_count,
        "model_backward_count": 2048 + 1024,
        "model_update_count": 2048 + 1024,
        "d1_score_rank_winner_read_count": 0,
        "retrieval_outcome_read_count": 0,
        "opened_read_count": 0,
        "sealed_read_count": 0,
        "C8_read_count": 0,
        "S8_read_count": 0,
        "target_insertion_count": 0,
        "p_lock_materialization_count": 0,
        "identity_specific_parameter_count": 0,
    }
    assert set(access) == {
        "pair_feature_payload_read_count",
        "sidecar_shard_payload_read_count",
        "loss_role_read_count",
        "model_load_count",
        "model_forward_count",
        "model_backward_count",
        "model_update_count",
        "d1_score_rank_winner_read_count",
        "retrieval_outcome_read_count",
        "opened_read_count",
        "sealed_read_count",
        "C8_read_count",
        "S8_read_count",
        "target_insertion_count",
        "p_lock_materialization_count",
        "identity_specific_parameter_count",
    }
    assert all(access[key] > 0 for key in tuple(access)[:7])
    assert all(access[key] == 0 for key in tuple(access)[7:])
