from __future__ import annotations

import ast
import copy
import inspect
import random

import numpy as np
import pytest
import torch

import validate_dino_rcde_sr0_mt_p_v2_fit_v1 as V


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (
            str(numpy_state[0]),
            torch.from_numpy(numpy_state[1].copy()),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
        "torch_cuda_rng_states": (),
    }


def _checkpoint(
    update: int,
    *,
    path_kind: str = "PRIMARY",
    bindings: dict[str, str] | None = None,
    restored_file: str | None = None,
    restored_state: str | None = None,
) -> dict[str, object]:
    model = V.SharedMultitilePHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=0.0001, foreach=False, fused=False
    )
    window_start = max(1, update - 63)
    window = [
        {"completed_update": value, "fixture": f"update-{value}"}
        for value in range(window_start, update + 1)
    ]
    window_sha = V.canonical(window)
    trace = {
        "count": update,
        "chain_sha256": V.canonical(
            {
                "previous_chain_sha256": None,
                "window_start": window_start,
                "window_sha256": window_sha,
            }
        ),
        "window_start": window_start,
        "window": window,
        "window_sha256": window_sha,
    }
    value: dict[str, object] = {
        "schema_version": V.CHECKPOINT_SCHEMA,
        "status": V.CHECKPOINT_STATUS,
        "claim_level": V.CHECKPOINT_CLAIM,
        "fit_id": "P_OUTER1_INNER2_FIT",
        "fit_index": 0,
        "path_kind": path_kind,
        "completed_updates": update,
        "consumable_checkpoint_authorized": False,
        "bindings": {"fixture_sha256": "a" * 64} if bindings is None else bindings,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_state": {
            "namespace": "RCDE_SR0_MT_P_QUERY_ORDER_V1",
            "fit_id": "P_OUTER1_INNER2_FIT",
            "epoch": 1,
            "position": 3,
            "order_sha256": "b" * 64,
        },
        "rng_state": _rng_state(),
        "scheduler_cursor": update,
        "current_learning_rate": V.learning_rate_at(update),
        "rolling_trace_state": trace,
        "state_sha256": None,
        "logical_sha256": None,
        "restored_from_checkpoint_file_sha256": restored_file,
        "restored_from_checkpoint_state_sha256": restored_state,
    }
    state_payload = {name: value[name] for name in V.STATE_PAYLOAD_FIELDS}
    value["state_sha256"] = V.state_sha256(state_payload)
    value["logical_sha256"] = V.checkpoint_logical_sha256(value)
    return value


def test_checkpoint_reconstruction_and_fixed_probe_are_independent_and_exact() -> None:
    bindings = {"fixture_sha256": "a" * 64}
    checkpoint = _checkpoint(1024, bindings=bindings)
    summary = V.validate_checkpoint(
        checkpoint,
        fit_id="P_OUTER1_INNER2_FIT",
        fit_index=0,
        path_kind="PRIMARY",
        completed_updates=1024,
        expected_bindings=bindings,
        restored_file_sha256=None,
        restored_state_sha256=None,
    )
    assert summary["state_sha256"] == checkpoint["state_sha256"]
    assert summary["fixed_probe_sha256"] == V.fixed_probe(checkpoint["model_state"])


def test_checkpoint_tampering_fails_closed() -> None:
    bindings = {"fixture_sha256": "a" * 64}
    base = _checkpoint(1024, bindings=bindings)

    changed_weight = copy.deepcopy(base)
    changed_weight["model_state"]["linear.weight"][0, 0] += 1.0
    with pytest.raises(V.ValidationError, match="state hash"):
        V.validate_checkpoint(
            changed_weight,
            fit_id="P_OUTER1_INNER2_FIT",
            fit_index=0,
            path_kind="PRIMARY",
            completed_updates=1024,
            expected_bindings=bindings,
            restored_file_sha256=None,
            restored_state_sha256=None,
        )

    extra = copy.deepcopy(base)
    extra["legacy_h0"] = True
    with pytest.raises(V.ValidationError, match="field set"):
        V.validate_checkpoint(
            extra,
            fit_id="P_OUTER1_INNER2_FIT",
            fit_index=0,
            path_kind="PRIMARY",
            completed_updates=1024,
            expected_bindings=bindings,
            restored_file_sha256=None,
            restored_state_sha256=None,
        )


def test_replay_must_bind_the_named_primary_split() -> None:
    bindings = {"fixture_sha256": "a" * 64}
    replay = _checkpoint(
        2048,
        path_kind="REPLAY",
        bindings=bindings,
        restored_file="c" * 64,
        restored_state="d" * 64,
    )
    with pytest.raises(V.ValidationError, match="restore lineage"):
        V.validate_checkpoint(
            replay,
            fit_id="P_OUTER1_INNER2_FIT",
            fit_index=0,
            path_kind="REPLAY",
            completed_updates=2048,
            expected_bindings=bindings,
            restored_file_sha256="e" * 64,
            restored_state_sha256="d" * 64,
        )


def test_trace_chain_is_independently_recomputed() -> None:
    first = _checkpoint(64)
    second = _checkpoint(128)
    first_chain = first["rolling_trace_state"]["chain_sha256"]
    second_trace = second["rolling_trace_state"]
    second_trace["chain_sha256"] = V.canonical(
        {
            "previous_chain_sha256": first_chain,
            "window_start": second_trace["window_start"],
            "window_sha256": second_trace["window_sha256"],
        }
    )
    assert V._validate_trace_chain((first, second)) == second_trace["chain_sha256"]
    second_trace["chain_sha256"] = "f" * 64
    with pytest.raises(V.ValidationError, match="trace chain"):
        V._validate_trace_chain((first, second))


def test_validator_does_not_import_new_fit_runtime_or_runner() -> None:
    imports: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(V))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("dino_rcde_sr0_mt_p_v2_fit_runtime_v1" in name for name in imports)
    assert not any("run_dino_rcde_sr0_mt_p_v2_fit_v1" in name for name in imports)
