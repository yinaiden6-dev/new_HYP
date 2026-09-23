#!/usr/bin/env python3
"""Run one authorized relative-only Track-R V fit from compact V2 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RC_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    RESUME_SCHEMA,
    UPDATES,
    execution_manifest_arguments,
    execution_manifest_argv,
    state_dict_sha256,
    train_updates,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402

import dino_rcde_sr0_mt_v_input_common_v1 as v_input  # noqa: E402
import dino_rcde_track_r_input_common_v1 as track_input  # noqa: E402
from run_dino_rcde_sr0_mt_v_fit_v1 import (  # noqa: E402
    load_context_initialization,
)


AUTHORITY_PATH = "registry/current_authority_v120_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v120_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED"
PROTOCOL_SCHEMA = "rc_dino_rcde_track_r_relative_v_fit_protocol_v1_20260821"
PROTOCOL_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_PROTOCOL_FROZEN"
RESULT_SCHEMA = "rc_dino_rcde_track_r_relative_v_fit_result_v1_20260821"
RESULT_COMPLETE = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_COMPLETE"
RESULT_CONTINUE = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_CONTINUATION_READY"
LEDGER_VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS"
RUNNER_PATH = "programs/run_dino_rcde_track_r_relative_v_fit_v1.py"
VALIDATOR_PATH = "programs/validate_dino_rcde_track_r_relative_v_fit_v1.py"
LAUNCHER_PATH = "slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
PACKAGE_INITIALIZER_PATH = "src/rc_aslo_xf/__init__.py"


class TrackRVFitError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRVFitError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, must_exist: bool, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path, must_exist=True).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def _verify_file_binding(binding: Any, expected_relative: str | None = None) -> Path:
    require(isinstance(binding, Mapping), "V120 runtime binding absent")
    relative = binding.get("path")
    require(
        isinstance(relative, str)
        and (expected_relative is None or relative == expected_relative),
        f"V120 runtime binding path drift: {expected_relative}",
    )
    path = safe_path(RC_ROOT / relative, must_exist=True)
    require(
        binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == file_sha256(path),
        f"V120 runtime binding hash drift: {relative}",
    )
    if "logical_sha256" in binding:
        value = read_json(path)
        require(
            value.get("logical_sha256")
            == logical_sha256(value)
            == binding["logical_sha256"],
            f"V120 runtime logical binding drift: {relative}",
        )
    return path


def validate_runtime_bindings(
    authority: Mapping[str, Any], execution_manifest: Path
) -> None:
    bindings = authority.get("bindings")
    require(isinstance(bindings, Mapping), "V120 runtime bindings absent")
    _verify_file_binding(bindings.get("runner"), RUNNER_PATH)
    _verify_file_binding(bindings.get("validator"), VALIDATOR_PATH)
    _verify_file_binding(bindings.get("launcher"), LAUNCHER_PATH)
    _verify_file_binding(bindings.get("package_initializer"), PACKAGE_INITIALIZER_PATH)
    closure = bindings.get("runtime_import_closure")
    require(
        isinstance(closure, Mapping)
        and closure.get("root_programs")
        == [RUNNER_PATH, VALIDATOR_PATH, PACKAGE_INITIALIZER_PATH]
        and isinstance(closure.get("module_count"), int),
        "V120 runtime import closure envelope drift",
    )
    rows = closure.get("modules")
    require(
        isinstance(rows, list)
        and len(rows) == closure["module_count"]
        and closure.get("module_sequence_sha256")
        == logical_sha256({"rows": rows}),
        "V120 runtime import closure sequence drift",
    )
    for row in rows:
        _verify_file_binding(row)
    edges = closure.get("import_edges")
    require(
        isinstance(edges, list)
        and closure.get("import_edge_sequence_sha256")
        == logical_sha256({"rows": edges}),
        "V120 runtime import edge receipt drift",
    )
    nodes = bindings.get("execution_nodes")
    require(
        isinstance(nodes, Mapping)
        and nodes.get("count") == 12
        and isinstance(nodes.get("rows"), Mapping)
        and nodes.get("logical_sha256")
        == logical_sha256({"rows": nodes["rows"]}),
        "V120 execution-node binding envelope drift",
    )
    for row in nodes["rows"].values():
        _verify_file_binding(row)
    manifest_relative = execution_manifest.relative_to(RC_ROOT).as_posix()
    selected = [
        row for row in nodes["rows"].values() if row.get("path") == manifest_relative
    ]
    require(len(selected) == 1, "active V120 execution node is not authority-bound")
    resource = authority.get("resource_contract")
    require(
        isinstance(resource, Mapping)
        and resource.get("array_spec") == "1-4%4"
        and resource.get("cpus_per_task") == 4
        and resource.get("memory_megabytes") == 64000
        and resource.get("gpus_per_node") == 1
        and resource.get("walltime_seconds") == 10800,
        "V120 formal resource contract drift",
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_protocol_authority(
    protocol_path: Path,
    authority_path: Path,
    outer_fold: int,
    execution_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "authority path drift")
    require(
        (authority_path.stat().st_mode & 0o777) == 0o444
        and authority.get("logical_sha256") == logical_sha256(authority),
        "V120 authority immutable/logical drift",
    )
    require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA
        and protocol.get("status") == PROTOCOL_STATUS
        and protocol.get("training", {}).get("loss") == "PAIRWISE_RELATIVE_ONLY"
        and protocol.get("training", {}).get("updates") == UPDATES
        and protocol.get("training", {}).get("tau") == 1
        and protocol.get("training", {}).get("delta") == 0
        and protocol.get("automatic_stage_advance") is False,
        "Track-R V-fit protocol drift",
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("relative_v_training_authorized") is True
        and authority.get("model_load_authorized") is True
        and authority.get("model_forward_authorized") is True
        and authority.get("model_backward_authorized") is True
        and authority.get("model_update_authorized") is True
        and authority.get("donor_null_absolute_loss_authorized") is False
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False
        and authority.get("automatic_stage_advance") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("next_authorized_stage") is None
        and outer_fold in authority.get("authorized_outer_folds", []),
        "Track-R V-fit authority drift",
    )
    validate_runtime_bindings(authority, execution_manifest)
    return protocol, authority, file_sha256(authority_path)


def validate_manifest(path: Path, expected: Mapping[str, object]) -> str:
    observed = dict(
        execution_manifest_arguments(
            path, expected_program=Path(__file__).name
        )
    )
    require(observed == dict(expected), "V-fit execution manifest drift")
    return file_sha256(path)


def validate_ledger_lineage(
    ledger_path: Path,
    index_path: Path,
    validation_path: Path,
    *,
    outer_fold: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    index = read_json(index_path)
    validation = read_json(validation_path)
    require(
        validation.get("status") == LEDGER_VALIDATION_STATUS
        and validation.get("validation_pass") is True
        and validation.get("index_sha256") == file_sha256(index_path)
        and validation.get("index_logical_sha256") == index.get("logical_sha256")
        and validation.get("scientific_GO_or_NO_GO") is None,
        "episode ledger independent-validation drift",
    )
    rows = [item for item in index.get("ledgers", []) if item.get("outer_fold") == outer_fold]
    require(len(rows) == 1, "episode ledger index fold row drift")
    row = rows[0]
    require(
        file_sha256(ledger_path) == row.get("sha256")
        and ledger_path.name == row.get("path")
        and ledger_path.stat().st_size == row.get("bytes"),
        "episode ledger/index binding drift",
    )
    ledger = track_input.load_compact_ledger(
        ledger_path,
        expected_outer_fold=outer_fold,
        expected_sha256=str(row["sha256"]),
    )
    require(ledger.get("logical_sha256") == row.get("logical_sha256"), "episode ledger logical hash drift")
    return ledger, row


def main() -> int:
    expanded_argv, execution_manifest = execution_manifest_argv(
        sys.argv[1:], expected_program=Path(__file__).name
    )
    require(execution_manifest is not None, "formal Track-R V fit requires an execution manifest")
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--episode-ledger", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-validation", type=Path, required=True)
    parser.add_argument("--redacted-schedule", type=Path, required=True)
    parser.add_argument("--redacted-cache-index", type=Path, required=True)
    parser.add_argument("--redacted-cache-root", type=Path, required=True)
    parser.add_argument("--geometry-payload", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after-updates", type=int, default=UPDATES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(expanded_argv)

    paths = {
        "protocol": safe_path(args.protocol, must_exist=True),
        "authority": safe_path(args.authority, must_exist=True),
        "ledger": safe_path(args.episode_ledger, must_exist=True),
        "index": safe_path(args.episode_index, must_exist=True),
        "validation": safe_path(args.episode_validation, must_exist=True),
        "schedule": safe_path(args.redacted_schedule, must_exist=True),
        "cache_index": safe_path(args.redacted_cache_index, must_exist=True),
        "cache_root": safe_path(args.redacted_cache_root, must_exist=True, file=False),
        "geometry": safe_path(args.geometry_payload, must_exist=True),
        "checkpoint": safe_path(args.context_checkpoint, must_exist=True),
        "output": safe_path(args.output_dir, must_exist=False, file=False),
        "manifest": safe_path(execution_manifest, must_exist=True),
    }
    protocol, authority, authority_sha = validate_protocol_authority(
        paths["protocol"],
        paths["authority"],
        args.outer_fold,
        paths["manifest"],
    )
    authority_logical_sha = str(authority["logical_sha256"])
    expected_manifest = {
        "protocol": str(args.protocol),
        "authority": str(args.authority),
        "episode_ledger": str(args.episode_ledger),
        "episode_index": str(args.episode_index),
        "episode_validation": str(args.episode_validation),
        "redacted_schedule": str(args.redacted_schedule),
        "redacted_cache_index": str(args.redacted_cache_index),
        "redacted_cache_root": str(args.redacted_cache_root),
        "geometry_payload": str(args.geometry_payload),
        "context_checkpoint": str(args.context_checkpoint),
        "outer_fold": args.outer_fold,
        "output_dir": str(args.output_dir),
        "stop_after_updates": args.stop_after_updates,
        "resume": args.resume,
        "device": args.device,
    }
    manifest_sha = validate_manifest(paths["manifest"], expected_manifest)
    require(0 <= args.stop_after_updates <= UPDATES, "stop-after-updates drift")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    ledger, ledger_row = validate_ledger_lineage(
        paths["ledger"], paths["index"], paths["validation"], outer_fold=args.outer_fold
    )
    cache_index, _, cache_index_sha, schedule_sha, schedule_logical = (
        v_input.load_schedule_cache_index(paths["schedule"], paths["cache_index"])
    )
    query_geometry, reference_geometry, geometry_sha = track_input.load_geometry_payload(
        paths["geometry"]
    )
    expected_model_sha = str(protocol["cache_model_checkpoint_logical_sha256"])
    episodes, materialization_receipt = track_input.materialize_vpair_episodes(
        ledger,
        cache_root=paths["cache_root"],
        cache_index=cache_index,
        expected_cache_model_sha256=expected_model_sha,
        query_geometry_by_execution=query_geometry,
        reference_geometry_by_row=reference_geometry,
    )
    device = torch.device(args.device)
    if file_sha256(paths["checkpoint"]) != protocol["context_checkpoint_sha256_by_fold"][str(args.outer_fold)]:
        raise TrackRVFitError("CONTEXT initialization physical hash drift")
    model, initialization_sha = load_context_initialization(
        paths["checkpoint"], args.outer_fold, device
    )
    # Keep the full fold pool on CPU. ``train_updates`` transfers only the four
    # scheduled episodes, one at a time, and releases each activation graph
    # after its scaled backward contribution.

    output_dir = paths["output"]
    resume_path = output_dir / "resume_state.pt"
    if args.resume:
        require(output_dir.is_dir() and resume_path.is_file(), "resume requested without state")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        require(isinstance(resume_state, Mapping) and resume_state.get("schema_version") == RESUME_SCHEMA, "resume state schema drift")
    else:
        require(not output_dir.exists(), "fresh V-fit output exists")
        output_dir.mkdir(parents=True, exist_ok=False)
        resume_state = None
    materialization_receipt.update(
        {
            "episode_ledger_sha256": file_sha256(paths["ledger"]),
            "episode_ledger_logical_sha256": ledger_row["logical_sha256"],
            "episode_index_sha256": file_sha256(paths["index"]),
            "episode_validation_sha256": file_sha256(paths["validation"]),
            "redacted_schedule_sha256": schedule_sha,
            "redacted_schedule_logical_sha256": schedule_logical,
            "redacted_cache_index_sha256": cache_index_sha,
            "geometry_payload_sha256": geometry_sha,
            "authority_sha256": authority_sha,
            "authority_logical_sha256": authority_logical_sha,
        }
    )
    materialization_receipt["logical_sha256"] = canonical_sha256(
        {key: item for key, item in materialization_receipt.items() if key != "logical_sha256"}
    )
    input_receipt_path = output_dir / "input_materialization_receipt.json"
    if input_receipt_path.exists():
        require(
            read_json(input_receipt_path) == materialization_receipt,
            "resume input materialization receipt drift",
        )
    else:
        atomic_json(input_receipt_path, materialization_receipt)

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
    checkpoint_path = None
    checkpoint_sha = None
    if complete:
        checkpoint_path = output_dir / "checkpoint_update2048.pt"
        require(not checkpoint_path.exists(), "final Track-R checkpoint exists")
        checkpoint: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE",
            "outer_fold": args.outer_fold,
            "arm": "RCDE_CONTEXT_TRACK_R_RELATIVE",
            "update": UPDATES,
            "seed": 17,
            "protocol_sha256": file_sha256(paths["protocol"]),
            "authority_sha256": authority_sha,
            "authority_logical_sha256": authority_logical_sha,
            "episode_ledger_sha256": file_sha256(paths["ledger"]),
            "episode_validation_sha256": file_sha256(paths["validation"]),
            "initialization_checkpoint_sha256": initialization_sha,
            "final_state_sha256": state_dict_sha256(model),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "relative_pair_loss_only": True,
            "same_checkpoint_all_three_arms_and_controls": True,
        }
        atomic_torch(checkpoint_path, checkpoint)
        checkpoint_sha = file_sha256(checkpoint_path)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": RESULT_COMPLETE if complete else RESULT_CONTINUE,
        "claim_level": "ENGINEERING_RELATIVE_ONLY_TRAINING_NOT_SCIENTIFIC_GO_OR_NO_GO",
        "outer_fold": args.outer_fold,
        "completed_updates": int(state["completed_updates"]),
        "resume_state": resume_path.name,
        "resume_state_sha256": file_sha256(resume_path),
        "final_checkpoint": None if checkpoint_path is None else checkpoint_path.name,
        "final_checkpoint_sha256": checkpoint_sha,
        "model_state_sha256": state["model_state_sha256"],
        "schedule_sha256": state["schedule_sha256"],
        "same_checkpoint_all_three_arms_and_controls": True,
        "input_role": "INNER_OOF_P_V2_LOCKS_ONLY",
        "outer_heldout_record_count": 0,
        "relative_pair_loss_only": True,
        "donor_null_absolute_loss_count": 0,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "protocol_sha256": file_sha256(paths["protocol"]),
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority_logical_sha,
        "episode_ledger_sha256": file_sha256(paths["ledger"]),
        "episode_validation_sha256": file_sha256(paths["validation"]),
        "context_initialization_sha256": initialization_sha,
        "execution_manifest_sha256": manifest_sha,
        "input_materialization_receipt_sha256": file_sha256(
            input_receipt_path
        ),
        "access_audit": {
            "C8_S8_opened_sealed_access_count": 0,
            "HOME_file_write_count": 0,
            "score_rank_winner_gap_outcome_read_count": 0,
            "donor_null_h0_hold_switch_read_count": 0,
        },
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "outer_fold": args.outer_fold, "completed_updates": result["completed_updates"], "logical_sha256": result["logical_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
