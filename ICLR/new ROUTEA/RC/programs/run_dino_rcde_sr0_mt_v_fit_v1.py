#!/usr/bin/env python3
"""Run one authorized outer-fold SR0-MT V fit with exact continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RC_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    RESUME_SCHEMA,
    UPDATES,
    VPairEpisode,
    execution_manifest_arguments,
    execution_manifest_argv,
    move_episode_to_device,
    state_dict_sha256,
    train_updates,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    validate_natural_authority,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402


RESULT_SCHEMA = "rc_dino_rcde_sr0_mt_v_fit_result_v1"
EPISODE_LEDGER_SCHEMA = "rc_dino_rcde_sr0_mt_v_training_ledger_v1"
TRAINING_STAGE = "SR0_MT_FORMAL_TRAINING"


class VFitRunnerError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VFitRunnerError(f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VFitRunnerError(f"JSON input is not an object: {path}")
    return value


def safe_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve()
    if RC_ROOT != resolved and RC_ROOT not in resolved.parents:
        raise VFitRunnerError(f"path escapes RC root: {resolved}")
    protected = {"c8", "s8", "opened", "sealed"}
    if protected.intersection(part.lower() for part in resolved.parts):
        raise VFitRunnerError(f"protected path requested: {resolved}")
    if must_exist and not resolved.exists():
        raise VFitRunnerError(f"missing input: {resolved}")
    return resolved


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_execution_authority(
    protocol_path: Path, authority_path: Path, authority_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    validate_natural_authority(
        authority_path,
        expected_file_sha256=authority_sha256,
        required_scope="natural_v_training",
        execution_protocol_sha256=file_sha256(protocol_path),
    )
    if (
        authority.get("natural_training_authorized") is not True
        or authority.get("model_load_authorized") is not True
        or authority.get("model_forward_authorized") is not True
        or authority.get("model_backward_authorized") is not True
        or authority.get("model_update_authorized") is not True
        or authority.get("checkpoint_deserialization_authorized") is not True
        or authority.get("next_authorized_stage") != TRAINING_STAGE
        or authority.get("automatic_stage_advance") is not False
    ):
        raise VFitRunnerError("authority does not authorize formal SR0-MT training")
    if protocol.get("automatic_stage_advance") is not False:
        raise VFitRunnerError("training protocol attempted automatic stage advancement")
    return protocol, authority


def load_training_episodes(
    path: Path,
    validation_path: Path,
    *,
    expected_ledger_sha256: str,
    expected_validation_sha256: str,
    outer_fold: int,
) -> tuple[VPairEpisode, ...]:
    if (
        file_sha256(path) != expected_ledger_sha256
        or file_sha256(validation_path) != expected_validation_sha256
    ):
        raise VFitRunnerError("V episode ledger/validation physical SHA drift")
    validation = read_json(validation_path)
    if (
        validation.get("status")
        != "RCDE_SR0_MT_V_EPISODE_LEDGER_INDEPENDENT_VALIDATION_PASS"
        or validation.get("validation_pass") is not True
        or validation.get("episode_ledger_file_sha256")
        != expected_ledger_sha256
        or int(validation.get("outer_fold", -1)) != outer_fold
        or validation.get("outer_heldout_record_count") != 0
        or validation.get("input_role") != "INNER_OOF_P_LOCKS_ONLY"
        or validation.get("automatic_stage_advance") is not False
    ):
        raise VFitRunnerError("V episode independent-validation closure drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != EPISODE_LEDGER_SCHEMA:
        raise VFitRunnerError("V episode-ledger schema drift")
    if int(payload.get("outer_fold", -1)) != outer_fold:
        raise VFitRunnerError("V episode-ledger outer-fold drift")
    episodes = tuple(payload.get("episodes", ()))
    if not episodes or any(not isinstance(item, VPairEpisode) for item in episodes):
        raise VFitRunnerError("V episode ledger contains noncanonical episodes")
    if payload.get("input_role") != "INNER_OOF_P_LOCKS_ONLY":
        raise VFitRunnerError("V training did not receive inner-OOF P locks only")
    if payload.get("outer_heldout_record_count") != 0:
        raise VFitRunnerError("outer-heldout rows leaked into V training")
    return episodes


def load_context_initialization(
    path: Path, outer_fold: int, device: torch.device
) -> tuple[DINO_RCDE_V1_2, str]:
    file_digest = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise VFitRunnerError("CONTEXT checkpoint is not a mapping")
    if (
        payload.get("arm") not in {"RCDE_CONTEXT", "RCDE_CONTEXT_INITIALIZATION"}
        or int(payload.get("outer_fold", -1)) != outer_fold
        or not isinstance(payload.get("model_state_dict"), dict)
    ):
        raise VFitRunnerError("matched outer-fold CONTEXT initialization drift")
    model = DINO_RCDE_V1_2().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("final_state_sha256") not in {None, state_dict_sha256(model)}:
        raise VFitRunnerError("CONTEXT checkpoint internal state hash drift")
    return model, file_digest


def validate_execution_manifest(
    path: Path,
    *,
    protocol_path: Path,
    authority_path: Path,
    authority_sha256: str,
    episode_ledger: Path,
    episode_ledger_sha256: str,
    episode_ledger_validation: Path,
    episode_ledger_validation_sha256: str,
    context_checkpoint: Path,
    context_checkpoint_sha256: str,
    output_dir: Path,
    outer_fold: int,
    stop_after_updates: int,
    resume: bool,
    device: str,
) -> str:
    expected = {
        "protocol": str(protocol_path),
        "authority": str(authority_path),
        "authority_sha256": authority_sha256,
        "episode_ledger": str(episode_ledger),
        "episode_ledger_sha256": episode_ledger_sha256,
        "episode_ledger_validation": str(episode_ledger_validation),
        "episode_ledger_validation_sha256": episode_ledger_validation_sha256,
        "context_checkpoint": str(context_checkpoint),
        "context_checkpoint_sha256": context_checkpoint_sha256,
        "output_dir": str(output_dir),
        "outer_fold": outer_fold,
        "stop_after_updates": stop_after_updates,
        "resume": resume,
        "device": device,
    }
    observed = dict(
        execution_manifest_arguments(path, expected_program=Path(__file__).name)
    )
    if observed != expected:
        raise VFitRunnerError("V execution-manifest binding drift")
    return file_sha256(path)


def main() -> None:
    expanded_argv, execution_manifest = execution_manifest_argv(
        sys.argv[1:], expected_program=Path(__file__).name
    )
    if execution_manifest is None:
        raise VFitRunnerError("formal V fit requires one JSON execution manifest")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--episode-ledger", type=Path, required=True)
    parser.add_argument("--episode-ledger-sha256", required=True)
    parser.add_argument("--episode-ledger-validation", type=Path, required=True)
    parser.add_argument("--episode-ledger-validation-sha256", required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--context-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--stop-after-updates", type=int, default=UPDATES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(expanded_argv)

    protocol_path = safe_path(args.protocol, must_exist=True)
    authority_path = safe_path(args.authority, must_exist=True)
    ledger_path = safe_path(args.episode_ledger, must_exist=True)
    ledger_validation_path = safe_path(
        args.episode_ledger_validation, must_exist=True
    )
    checkpoint_path = safe_path(args.context_checkpoint, must_exist=True)
    output_dir = safe_path(args.output_dir, must_exist=False)
    manifest_path = safe_path(execution_manifest, must_exist=True)
    protocol, _ = validate_execution_authority(
        protocol_path, authority_path, args.authority_sha256
    )
    manifest_sha = validate_execution_manifest(
        manifest_path,
        protocol_path=protocol_path,
        authority_path=authority_path,
        authority_sha256=args.authority_sha256,
        episode_ledger=ledger_path,
        episode_ledger_sha256=args.episode_ledger_sha256,
        episode_ledger_validation=ledger_validation_path,
        episode_ledger_validation_sha256=args.episode_ledger_validation_sha256,
        context_checkpoint=checkpoint_path,
        context_checkpoint_sha256=args.context_checkpoint_sha256,
        output_dir=output_dir,
        outer_fold=args.outer_fold,
        stop_after_updates=args.stop_after_updates,
        resume=args.resume,
        device=args.device,
    )

    if not 0 <= args.stop_after_updates <= UPDATES:
        raise VFitRunnerError("stop-after-updates is outside 0..2048")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise VFitRunnerError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    episodes = load_training_episodes(
        ledger_path,
        ledger_validation_path,
        expected_ledger_sha256=args.episode_ledger_sha256,
        expected_validation_sha256=args.episode_ledger_validation_sha256,
        outer_fold=args.outer_fold,
    )
    if file_sha256(checkpoint_path) != args.context_checkpoint_sha256:
        raise VFitRunnerError("CONTEXT initialization physical SHA drift")
    model, initialization_sha = load_context_initialization(
        checkpoint_path, args.outer_fold, device
    )
    episodes = tuple(move_episode_to_device(item, device) for item in episodes)

    resume_path = output_dir / "resume_state.pt"
    if args.resume:
        if not output_dir.is_dir() or not resume_path.is_file():
            raise VFitRunnerError("explicit resume requested without a resume state")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(resume_state, dict) or resume_state.get("schema_version") != RESUME_SCHEMA:
            raise VFitRunnerError("resume-state schema drift")
    else:
        if output_dir.exists():
            raise VFitRunnerError("fresh V output namespace already exists")
        output_dir.mkdir(parents=True, exist_ok=False)
        resume_state = None

    state, _ = train_updates(
        model,
        episodes,
        outer_fold=args.outer_fold,
        initialization_checkpoint_sha256=initialization_sha,
        stop_after_updates=args.stop_after_updates,
        resume_state=resume_state,
    )
    atomic_torch(resume_path, state)

    complete = int(state["completed_updates"]) == UPDATES
    final_checkpoint = None
    final_checkpoint_sha = None
    if complete:
        final_checkpoint = output_dir / "checkpoint_update2048.pt"
        if final_checkpoint.exists():
            raise VFitRunnerError("immutable final V checkpoint already exists")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "outer_fold": args.outer_fold,
            "arm": "RCDE_CONTEXT_SR0_MT",
            "update": UPDATES,
            "seed": 17,
            "protocol_sha256": file_sha256(protocol_path),
            "authority_sha256": file_sha256(authority_path),
            "episode_ledger_sha256": file_sha256(ledger_path),
            "episode_ledger_validation_sha256": file_sha256(
                ledger_validation_path
            ),
            "initialization_checkpoint_sha256": initialization_sha,
            "final_state_sha256": state_dict_sha256(model),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "same_checkpoint_all_three_arms_and_controls": True,
        }
        atomic_torch(final_checkpoint, payload)
        final_checkpoint_sha = file_sha256(final_checkpoint)

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "RCDE_SR0_MT_V_FIT_COMPLETE" if complete else "RCDE_SR0_MT_V_FIT_CONTINUATION_READY",
        "claim_level": "ENGINEERING_TRAINING_COMPLETE_NOT_SCIENTIFIC_GO_OR_NO_GO",
        "outer_fold": args.outer_fold,
        "completed_updates": int(state["completed_updates"]),
        "resume_state": resume_path.name,
        "resume_state_sha256": file_sha256(resume_path),
        "final_checkpoint": final_checkpoint.name if final_checkpoint else None,
        "final_checkpoint_sha256": final_checkpoint_sha,
        "model_state_sha256": state["model_state_sha256"],
        "schedule_sha256": state["schedule_sha256"],
        "next_sampler_query_keys": state["next_sampler_query_keys"],
        "same_checkpoint_all_three_arms_and_controls": True,
        "input_role": "INNER_OOF_P_LOCKS_ONLY",
        "outer_heldout_record_count": 0,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "protocol_sha256": file_sha256(protocol_path),
        "authority_sha256": file_sha256(authority_path),
        "episode_ledger_sha256": file_sha256(ledger_path),
        "episode_ledger_validation_sha256": file_sha256(ledger_validation_path),
        "context_initialization_sha256": initialization_sha,
        "execution_manifest_sha256": manifest_sha,
        "access_audit": {
            "C8_access_count": 0,
            "S8_access_count": 0,
            "opened_access_count": 0,
            "sealed_access_count": 0,
            "HOME_file_write_count": 0,
        },
    }
    atomic_json(output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
