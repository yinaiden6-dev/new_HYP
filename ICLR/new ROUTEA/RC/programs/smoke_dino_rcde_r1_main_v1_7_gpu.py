#!/usr/bin/env python3
"""Real-GPU qualification for the repaired R1 BAG fold-1 training path.

This smoke deliberately uses the frozen P0 V1.2 ledgers and canonical R1
cache.  It executes real decoder forward/backward/optimizer steps and checks
that a process-boundary save/reload is exactly equivalent to continuous
execution.  It never reads heldout evaluation outputs and never advances the
scientific stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_r1_main_train_v1_5 as base  # noqa: E402
import run_dino_rcde_r1_main_train_v1_7_contract_repair as repaired  # noqa: E402
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402


SCHEMA = "rc_dino_rcde_r1_main_real_gpu_smoke_v1_7_20260813"
PASS = "DINO_RCDE_R1_MAIN_V17_REAL_GPU_SMOKE_PASS"
ABORT = "DINO_RCDE_R1_MAIN_V17_REAL_GPU_SMOKE_ABORT"
ARM = "RCDE_BAG"
FOLD = 1
MODEL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
CACHE_ROOT = ROOT / "runtime/dino_rcde_p0_v1_2/stable_fp16_cache"
TRAINING_ROLES = ROOT / "protocols/dino_rcde_training_roles_600_v1_2_20260812.json"
EPISODE_LEDGER = (
    ROOT
    / "results/dino_rcde_p0_v1_2/finalize_repair_job5069473/data/fold_access_and_episode_receipt.json"
)
RESOURCE_SHAPE_LEDGER = (
    ROOT
    / "results/dino_rcde_p0_v1_2/finalize_repair_job5069473/data/resource_shape_ledger.json"
)
EXPECTED_INPUT_SHA256 = {
    "training_roles": "beac89d8caecd09b3e4815684cde207a363a0fa784e1e793f5694cc5fb7ee672",
    "episode_ledger": "7a35e0acff8032453f064c77cc40ee6a822e227b5aed2852016c0a93ba32b0e7",
    "resource_shape_ledger": "35ae0fc826a794f591bdaa6801ea499448a50a747f265ef9039e81a12b6636fa",
}


class SmokeAbort(RuntimeError):
    pass


def _protocol_view() -> dict[str, Any]:
    """Small protocol view containing exactly the fields input_closure reads."""

    return {
        "bindings": {
            "training_roles": {
                "path": str(TRAINING_ROLES.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256["training_roles"],
            },
            "episode_ledger": {
                "path": str(EPISODE_LEDGER.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256["episode_ledger"],
            },
            "resource_shape_ledger": {
                "path": str(RESOURCE_SHAPE_LEDGER.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256["resource_shape_ledger"],
            },
        },
        "cache": {"model_checkpoint_logical_sha256": MODEL_SHA256},
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _tensor_mapping_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_recursive_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    return left == right


def _semantic_hash(value: Any) -> str:
    """Deterministic recursive hash for model/optimizer/RNG state."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(base.json_bytes(list(tensor.shape)) + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(base.json_bytes(list(array.shape)) + b"\0")
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: (type(value).__name__, repr(value))):
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
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")

    visit(value)
    return digest.hexdigest()


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _changed_scalar_count(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> int:
    if set(before) != set(after):
        raise SmokeAbort("model state keys changed during first optimizer update")
    return sum(
        int(torch.count_nonzero(before[name] != after[name]).item()) for name in before
    )


def _make_tracker(frozen: Mapping[str, Any]) -> repaired.CacheAccessTracker:
    manifest = frozen["manifest"]
    return repaired.CacheAccessTracker(
        index=frozen["index"],
        query_by_execution=frozen["query_by_execution"],
        reference_by_row=frozen["reference_by_row"],
        allowed_queries=manifest["eligible_query_execution_ordinals"],
        allowed_references=manifest["allowed_reference_physical_rows"],
        model_sha256=MODEL_SHA256,
    )


def _run_one(
    *,
    model: DINO_RCDE_V1_2,
    optimizer: torch.optim.Optimizer,
    tracker: repaired.CacheAccessTracker,
    frozen: Mapping[str, Any],
    update_zero: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return repaired.run_update(
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        correct=frozen["correct"],
        wrong=frozen["wrong"],
        fold=FOLD,
        arm=ARM,
        update_zero=update_zero,
        device=device,
    )


def _assert_finite_record(record: Mapping[str, Any]) -> None:
    for key in ("mean_pair_loss", "gradient_norm_before_clip"):
        if not math.isfinite(float(record[key])):
            raise SmokeAbort(f"nonfinite real update field: {key}")
    if float(record["gradient_norm_before_clip"]) <= 0.0:
        raise SmokeAbort("first real full backward produced no positive gradient norm")
    if int(record["pair_count"]) <= 0:
        raise SmokeAbort("first real update contained no target-rival pairs")


def _run_resume_worker(
    *, resume_path: Path, output_path: Path, cache_root: Path
) -> int:
    """Resume update two in a newly exec'd Python process."""

    for path, name in (
        (resume_path, "resume state"),
        (output_path, "worker output"),
        (cache_root, "cache root"),
    ):
        if ROOT not in path.parents:
            raise SmokeAbort(f"resume-worker {name} escapes RC root")
    if not resume_path.is_file() or not cache_root.is_dir():
        raise SmokeAbort("resume-worker input is absent")
    if not torch.cuda.is_available():
        raise SmokeAbort("resume worker requires CUDA")
    if output_path.exists():
        raise SmokeAbort("immutable resume-worker output already exists")
    # Deterministic backend switches (cuDNN, TF32, deterministic algorithms)
    # are process-local and are not part of an RNG checkpoint.  Reapply the
    # exact formal setup in this newly exec'd interpreter before any compute.
    base.seed_everything()
    device = torch.device("cuda")
    frozen = repaired.input_closure(_protocol_view(), cache_root, FOLD)
    loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
    base.seed_everything()
    model = DINO_RCDE_V1_2().to(device)
    frozen_initial_sha = base.state_dict_sha256(model)
    repaired.validate_resume_state(
        loaded,
        arm=ARM,
        fold=FOLD,
        protocol_sha256="real_gpu_smoke_protocol",
        authority_sha256="real_gpu_smoke_authority",
        initial_state_sha256=frozen_initial_sha,
    )
    if int(loaded["completed_updates"]) != 1:
        raise SmokeAbort("resume worker did not receive update-one state")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-4, weight_decay=1.0e-4
    )
    model.load_state_dict(loaded["model_state_dict"], strict=True)
    optimizer.load_state_dict(loaded["optimizer_state_dict"])
    repaired.optimizer_to_device(optimizer, device)
    random.setstate(loaded["python_random_state"])
    np.random.set_state(loaded["numpy_random_state"])
    torch.set_rng_state(loaded["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(loaded["torch_cuda_rng_state_all"])
    tracker = _make_tracker(frozen)
    event, record = _run_one(
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        frozen=frozen,
        update_zero=1,
        device=device,
    )
    _assert_finite_record(record)
    torch.cuda.synchronize(device)
    repaired.atomic_torch_no_clobber(
        output_path,
        {
            "schema_version": "rc_dino_rcde_r1_main_resume_worker_v1_7",
            "worker_pid": os.getpid(),
            "model_state_dict": _cpu_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "event": event,
            "record": record,
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--resume-worker-state", type=Path)
    parser.add_argument("--resume-worker-output", type=Path)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    if args.resume_worker_state is not None or args.resume_worker_output is not None:
        if (
            args.output_dir is not None
            or args.resume_worker_state is None
            or args.resume_worker_output is None
        ):
            raise SmokeAbort("resume-worker arguments are incomplete")
        return _run_resume_worker(
            resume_path=args.resume_worker_state.resolve(),
            output_path=args.resume_worker_output.resolve(),
            cache_root=cache_root,
        )
    if args.output_dir is None:
        raise SmokeAbort("main smoke requires --output-dir")
    output = args.output_dir.resolve()
    result_path = output / "result.json"
    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": ABORT,
        "claim_level": "ENGINEERING_REAL_FORWARD_BACKWARD_AND_EXACT_RESUME_ONLY",
        "arm": ARM,
        "outer_fold": FOLD,
        "natural_training_data_role": "optimization_only",
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "protected_access_counts": repaired.protected_zeros(),
    }
    if ROOT not in output.parents or ROOT not in cache_root.parents:
        raise SmokeAbort("smoke cache/output path escapes RC root")
    if output.exists():
        raise SmokeAbort(f"immutable smoke namespace already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        if not torch.cuda.is_available():
            raise SmokeAbort("real R1 training smoke requires CUDA")
        for name, path in (
            ("training_roles", TRAINING_ROLES),
            ("episode_ledger", EPISODE_LEDGER),
            ("resource_shape_ledger", RESOURCE_SHAPE_LEDGER),
        ):
            if base.file_sha256(path) != EXPECTED_INPUT_SHA256[name]:
                raise SmokeAbort(f"frozen smoke input hash drift: {name}")

        base.seed_everything()
        device = torch.device("cuda")
        frozen = repaired.input_closure(_protocol_view(), cache_root, FOLD)
        first_episodes = base.update_episodes(frozen["correct"], frozen["wrong"], 0)
        # The repaired source-hash ordering intentionally changes update one.
        # It still contains a historical ordinal outside 0..599, so running it
        # exercises the exact namespace bug that killed Job 5070402.  The
        # original OUTCOME-0504 mapping is independently live-loaded below.
        regression = [
            row for row in first_episodes if str(row["query_id"]) == "OUTCOME-0614"
        ]
        if len(regression) != 1 or (
            int(regression[0]["query_ordinal"]),
            int(regression[0]["execution_ordinal"]),
        ) != (779, 445):
            raise SmokeAbort("repaired update one lacks the cache-namespace regression")
        original = [
            row
            for row in frozen["ledger"]["episodes"]
            if str(row["query_id"]) == "OUTCOME-0504"
        ]
        if len(original) != 3 or {
            (int(row["query_ordinal"]), int(row["execution_ordinal"]))
            for row in original
        } != {(678, 374)}:
            raise SmokeAbort("Job 5070402 source/execution mapping drift")
        probe_tracker = _make_tracker(frozen)
        probe_tracker.load("query", 374)
        try:
            probe_tracker.load("query", 678)
        except repaired.ContractRepairAbort:
            pass
        else:
            raise SmokeAbort("historical ordinal 678 was accepted as a cache key")
        if probe_tracker.file_open_counts[("query", 374)] != 1:
            raise SmokeAbort("repaired OUTCOME-0504 payload was not live-loaded once")
        del probe_tracker

        template = DINO_RCDE_V1_2().to(device)
        if sum(parameter.numel() for parameter in template.parameters()) != base.EXPECTED_PARAMETER_COUNT:
            raise SmokeAbort("smoke model parameter count drift")
        initial_model = _cpu_state(template)
        initial_model_sha = base.state_dict_sha256(template)
        initial_rng = _rng_state()

        # Branch A: two uninterrupted real full updates.
        continuous = template
        continuous_optimizer = torch.optim.AdamW(
            continuous.parameters(), lr=3.0e-4, weight_decay=1.0e-4
        )
        _restore_rng(initial_rng)
        continuous_tracker = _make_tracker(frozen)
        event_a1, record_a1 = _run_one(
            model=continuous,
            optimizer=continuous_optimizer,
            tracker=continuous_tracker,
            frozen=frozen,
            update_zero=0,
            device=device,
        )
        _assert_finite_record(record_a1)
        first_query_cache_keys = sorted(
            ordinal
            for (kind, ordinal), count in continuous_tracker.semantic_counts.items()
            if kind == "query" and count > 0
        )
        expected_first_query_keys = sorted(
            int(row["execution_ordinal"]) for row in first_episodes
        )
        if first_query_cache_keys != expected_first_query_keys:
            raise SmokeAbort("first real update did not use repaired execution-ordinal cache key")
        if 445 not in first_query_cache_keys or 779 in first_query_cache_keys:
            raise SmokeAbort("first real update used the historical namespace")
        state_after_first = _cpu_state(continuous)
        changed_count = _changed_scalar_count(initial_model, state_after_first)
        if changed_count <= 0 or base.state_dict_sha256(continuous) == initial_model_sha:
            raise SmokeAbort("first real forward/backward did not change model state")
        event_a2, record_a2 = _run_one(
            model=continuous,
            optimizer=continuous_optimizer,
            tracker=continuous_tracker,
            frozen=frozen,
            update_zero=1,
            device=device,
        )
        _assert_finite_record(record_a2)
        continuous_model = _cpu_state(continuous)
        continuous_optimizer_state = copy.deepcopy(continuous_optimizer.state_dict())
        continuous_rng = _rng_state()
        continuous_model_sha = base.state_dict_sha256(continuous)
        continuous_optimizer_sha = _semantic_hash(continuous_optimizer_state)
        del continuous, continuous_optimizer, continuous_tracker
        torch.cuda.empty_cache()

        # Branch B: update one, serialize process state, rebuild, restore, update two.
        base.seed_everything()
        staged = DINO_RCDE_V1_2().to(device)
        staged.load_state_dict(initial_model, strict=True)
        staged_optimizer = torch.optim.AdamW(
            staged.parameters(), lr=3.0e-4, weight_decay=1.0e-4
        )
        _restore_rng(initial_rng)
        staged_tracker = _make_tracker(frozen)
        event_b1, record_b1 = _run_one(
            model=staged,
            optimizer=staged_optimizer,
            tracker=staged_tracker,
            frozen=frozen,
            update_zero=0,
            device=device,
        )
        _assert_finite_record(record_b1)
        resume_payload = repaired.state_payload(
            model=staged,
            optimizer=staged_optimizer,
            arm=ARM,
            fold=FOLD,
            completed=1,
            loss_trace=[record_b1],
            pair_count=int(record_b1["pair_count"]),
            initial_state_sha256=initial_model_sha,
            protocol_sha256="real_gpu_smoke_protocol",
            authority_sha256="real_gpu_smoke_authority",
        )
        resume_path = output / "resume_after_update1.pt"
        if resume_path.exists():
            raise SmokeAbort("immutable smoke resume artifact already exists")
        torch.save(resume_payload, resume_path)
        resume_file_sha = base.file_sha256(resume_path)
        del staged, staged_optimizer, staged_tracker, resume_payload
        torch.cuda.empty_cache()

        worker_path = output / "resume_worker_after_update2.pt"
        worker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cache-root",
                str(cache_root),
                "--resume-worker-state",
                str(resume_path),
                "--resume-worker-output",
                str(worker_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if worker.returncode != 0:
            raise SmokeAbort(
                "resume worker failed: "
                f"returncode={worker.returncode}; stdout={worker.stdout[-2000:]}; "
                f"stderr={worker.stderr[-2000:]}"
            )
        worker_payload = torch.load(worker_path, map_location="cpu", weights_only=False)
        if (
            worker_payload.get("schema_version")
            != "rc_dino_rcde_r1_main_resume_worker_v1_7"
            or int(worker_payload.get("worker_pid", -1)) == os.getpid()
        ):
            raise SmokeAbort("resume step was not produced by a distinct Python process")
        resumed_model = worker_payload["model_state_dict"]
        resumed_optimizer_state = worker_payload["optimizer_state_dict"]
        resumed_rng = worker_payload["rng_state"]
        event_b2 = worker_payload["event"]
        record_b2 = worker_payload["record"]
        model_exact = _tensor_mapping_equal(continuous_model, resumed_model)
        optimizer_exact = _recursive_equal(
            continuous_optimizer_state, resumed_optimizer_state
        )
        rng_exact = _recursive_equal(continuous_rng, resumed_rng)
        events_exact = event_a1 == event_b1 and event_a2 == event_b2
        records_exact = record_a1 == record_b1 and record_a2 == record_b2
        if not all((model_exact, optimizer_exact, rng_exact, events_exact, records_exact)):
            raise SmokeAbort("continuous and save/reload execution are not exact")
        resumed_model_sha = base.state_dict_sha256(_StateView(resumed_model))
        if resumed_model_sha != continuous_model_sha:
            raise SmokeAbort("final model state SHA differs after exact resume")
        if _semantic_hash(resumed_optimizer_state) != continuous_optimizer_sha:
            raise SmokeAbort("final optimizer semantic SHA differs after exact resume")

        receipt.update(
            {
                "status": PASS,
                "cuda_device_name_audit_only": torch.cuda.get_device_name(device),
                "torch_version_audit_only": torch.__version__,
                "runner_sha256": base.file_sha256(
                    ROOT
                    / "programs/run_dino_rcde_r1_main_train_v1_7_contract_repair.py"
                ),
                "smoke_program_sha256": base.file_sha256(Path(__file__)),
                "frozen_input_sha256": EXPECTED_INPUT_SHA256,
                "cache_model_checkpoint_logical_sha256": MODEL_SHA256,
                "input_preflight_status": frozen["preflight"]["status"],
                "eligible_query_count": frozen["manifest"]["eligible_query_count"],
                "allowed_reference_count": frozen["manifest"]["allowed_reference_count"],
                "job5070402_regression_mapping": {
                    "query_id": "OUTCOME-0504",
                    "query_ordinal": 678,
                    "execution_ordinal": 374,
                    "cache_file": "query_0374.pt",
                    "payload_live_loaded": True,
                    "historical_ordinal_rejected_by_runtime_allowlist": True,
                },
                "first_real_update": {
                    "event": event_a1,
                    "record": record_a1,
                    "initial_model_state_sha256": initial_model_sha,
                    "model_state_sha256_after_update": base.state_dict_sha256(
                        _StateView(state_after_first)
                    ),
                    "changed_parameter_scalar_count": changed_count,
                    "query_cache_execution_ordinals_read": first_query_cache_keys,
                    "cache_namespace_regression_query_id": "OUTCOME-0614",
                    "cache_namespace_regression_query_ordinal": 779,
                    "cache_namespace_regression_execution_ordinal": 445,
                    "correct_cache_key_read": 445 in first_query_cache_keys,
                    "wrong_historical_cache_key_read": 779 in first_query_cache_keys,
                    "full_forward_backward_executed": True,
                    "optimizer_step_executed": True,
                },
                "second_real_update": {"event": event_a2, "record": record_a2},
                "exact_resume": {
                    "resume_artifact": resume_path.name,
                    "resume_artifact_sha256": resume_file_sha,
                    "resume_worker_artifact": worker_path.name,
                    "resume_worker_artifact_sha256": base.file_sha256(worker_path),
                    "parent_pid": os.getpid(),
                    "worker_pid": int(worker_payload["worker_pid"]),
                    "separate_exec_process": True,
                    "continuous_model_state_sha256": continuous_model_sha,
                    "resumed_model_state_sha256": resumed_model_sha,
                    "continuous_optimizer_semantic_sha256": continuous_optimizer_sha,
                    "resumed_optimizer_semantic_sha256": _semantic_hash(
                        resumed_optimizer_state
                    ),
                    "model_tensor_exact": model_exact,
                    "optimizer_state_exact": optimizer_exact,
                    "rng_state_exact": rng_exact,
                    "schedule_events_exact": events_exact,
                    "loss_and_gradient_records_exact": records_exact,
                    "continuous_two_updates_equals_one_save_reload_one": True,
                },
                "wall_seconds": time.time() - started,
            }
        )
    except Exception as error:
        receipt.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_seconds": time.time() - started,
            }
        )
    repaired.atomic_json_no_clobber(result_path, receipt)
    return 0 if receipt["status"] == PASS else 2


class _StateView(torch.nn.Module):
    """Expose an already-captured state mapping to base.state_dict_sha256."""

    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self._captured = state

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del args, kwargs
        return dict(self._captured)


if __name__ == "__main__":
    raise SystemExit(main())
