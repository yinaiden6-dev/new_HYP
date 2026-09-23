#!/usr/bin/env python3
"""Validate one Track-R relative V fold by fresh-versus-resume exact replay."""

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
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    RESUME_SCHEMA,
    UPDATES,
    learning_rate,
    state_dict_sha256,
)


AUTHORITY_PATH = "registry/current_authority_v120_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v120_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED"
PROTOCOL_SCHEMA = "rc_dino_rcde_track_r_relative_v_fit_protocol_v1_20260821"
PROTOCOL_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_PROTOCOL_FROZEN"
RESULT_SCHEMA = "rc_dino_rcde_track_r_relative_v_fit_result_v1_20260821"
RESULT_COMPLETE = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_COMPLETE"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_relative_v_fit_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS"
RUNNER_PATH = "programs/run_dino_rcde_track_r_relative_v_fit_v1.py"
VALIDATOR_PATH = "programs/validate_dino_rcde_track_r_relative_v_fit_v1.py"
LAUNCHER_PATH = "slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
PACKAGE_INITIALIZER_PATH = "src/rc_aslo_xf/__init__.py"


class TrackRVFitValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRVFitValidationError(message)


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


def safe_path(path: Path, *, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def _verify_file_binding(binding: Any, expected_relative: str | None = None) -> Path:
    require(isinstance(binding, Mapping), "V120 validation runtime binding absent")
    relative = binding.get("path")
    require(
        isinstance(relative, str)
        and (expected_relative is None or relative == expected_relative),
        f"V120 validation runtime binding path drift: {expected_relative}",
    )
    path = safe_path(RC_ROOT / relative)
    require(
        binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == file_sha256(path),
        f"V120 validation runtime binding hash drift: {relative}",
    )
    if "logical_sha256" in binding:
        value = read_json(path)
        require(
            value.get("logical_sha256")
            == logical_sha256(value)
            == binding["logical_sha256"],
            f"V120 validation logical binding drift: {relative}",
        )
    return path


def validate_runtime_bindings(authority: Mapping[str, Any]) -> None:
    bindings = authority.get("bindings")
    require(isinstance(bindings, Mapping), "V120 validation runtime bindings absent")
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
        "V120 validation import closure envelope drift",
    )
    rows = closure.get("modules")
    require(
        isinstance(rows, list)
        and len(rows) == closure["module_count"]
        and closure.get("module_sequence_sha256")
        == logical_sha256({"rows": rows}),
        "V120 validation import closure sequence drift",
    )
    for row in rows:
        _verify_file_binding(row)
    edges = closure.get("import_edges")
    require(
        isinstance(edges, list)
        and closure.get("import_edge_sequence_sha256")
        == logical_sha256({"rows": edges}),
        "V120 validation import edge receipt drift",
    )
    nodes = bindings.get("execution_nodes")
    require(
        isinstance(nodes, Mapping)
        and nodes.get("count") == 12
        and isinstance(nodes.get("rows"), Mapping)
        and nodes.get("logical_sha256")
        == logical_sha256({"rows": nodes["rows"]}),
        "V120 validation execution-node binding drift",
    )
    for row in nodes["rows"].values():
        _verify_file_binding(row)
    resource = authority.get("resource_contract")
    require(
        isinstance(resource, Mapping)
        and resource.get("array_spec") == "1-4%4"
        and resource.get("cpus_per_task") == 4
        and resource.get("memory_megabytes") == 64000
        and resource.get("gpus_per_node") == 1
        and resource.get("walltime_seconds") == 10800,
        "V120 validation resource contract drift",
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable validation exists: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.chmod(0o444)


def _schedule_sha(events: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(
            json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def _load_run(
    path: Path,
    outer_fold: int,
    expected_authority_sha256: str,
    expected_authority_logical_sha256: str,
) -> dict[str, Any]:
    root = safe_path(path, file=False)
    result = read_json(root / "result.json")
    input_receipt = read_json(root / "input_materialization_receipt.json")
    resume = torch.load(root / "resume_state.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(
        root / "checkpoint_update2048.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == RESULT_COMPLETE
        and result.get("logical_sha256") == logical_sha256(result)
        and result.get("outer_fold") == outer_fold
        and result.get("authority_sha256") == expected_authority_sha256
        and result.get("authority_logical_sha256")
        == expected_authority_logical_sha256
        and result.get("completed_updates") == UPDATES
        and result.get("relative_pair_loss_only") is True
        and result.get("donor_null_absolute_loss_count") == 0
        and result.get("outer_heldout_record_count") == 0
        and result.get("scientific_GO_or_NO_GO") is None,
        "Track-R V-fit result drift",
    )
    require(
        isinstance(resume, Mapping)
        and resume.get("schema_version") == RESUME_SCHEMA
        and resume.get("outer_fold") == outer_fold
        and resume.get("completed_updates") == UPDATES
        and len(resume.get("loss_trace", ())) == UPDATES
        and len(resume.get("schedule_events", ())) == UPDATES
        and resume.get("schedule_sha256")
        == _schedule_sha(list(resume["schedule_events"]))
        and len(resume.get("next_sampler_query_keys", ())) == 4
        and resume.get("next_learning_rate") is None
        and resume.get("model_state_sha256")
        == state_dict_sha256(resume["model_state_dict"]),
        "Track-R resume state drift",
    )
    for update, row in enumerate(resume["loss_trace"], start=1):
        require(
            row.get("update") == update
            and row.get("learning_rate") == learning_rate(update)
            and len(row.get("query_schedule_keys", ())) == 4
            and isinstance(row.get("mean_pair_loss"), float)
            and isinstance(row.get("gradient_norm_before_clip"), float),
            "Track-R loss trace/schedule drift",
        )
    require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("schema_version") == CHECKPOINT_SCHEMA
        and checkpoint.get("status")
        == "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE"
        and checkpoint.get("outer_fold") == outer_fold
        and checkpoint.get("authority_sha256") == expected_authority_sha256
        and checkpoint.get("authority_logical_sha256")
        == expected_authority_logical_sha256
        and checkpoint.get("arm") == "RCDE_CONTEXT_TRACK_R_RELATIVE"
        and checkpoint.get("update") == UPDATES
        and checkpoint.get("relative_pair_loss_only") is True
        and checkpoint.get("same_checkpoint_all_three_arms_and_controls") is True
        and checkpoint.get("final_state_sha256")
        == state_dict_sha256(checkpoint["model_state_dict"])
        == resume["model_state_sha256"]
        == result["model_state_sha256"],
        "Track-R final checkpoint drift",
    )
    require(
        result.get("resume_state_sha256") == file_sha256(root / "resume_state.pt")
        and result.get("final_checkpoint_sha256")
        == file_sha256(root / "checkpoint_update2048.pt")
        and result.get("input_materialization_receipt_sha256")
        == file_sha256(root / "input_materialization_receipt.json")
        and input_receipt.get("outer_fold") == outer_fold
        and input_receipt.get("authority_sha256") == expected_authority_sha256
        and input_receipt.get("authority_logical_sha256")
        == expected_authority_logical_sha256
        and input_receipt.get("episode_count") in (445, 446)
        and len(input_receipt.get("ordered_query_schedule_keys", ()))
        == input_receipt.get("episode_count"),
        "Track-R output/input receipt closure drift",
    )
    return {
        "root": root,
        "result": result,
        "input": input_receipt,
        "resume": resume,
        "checkpoint": checkpoint,
    }


def validate(
    authority_path: Path,
    protocol_path: Path,
    primary_output: Path,
    resume_output: Path,
    outer_fold: int,
    output: Path,
) -> dict[str, Any]:
    authority_path = safe_path(authority_path)
    protocol_path = safe_path(protocol_path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "authority path drift")
    authority = read_json(authority_path)
    protocol = read_json(protocol_path)
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical_sha256(authority)
        and (authority_path.stat().st_mode & 0o777) == 0o444
        and authority.get("relative_v_validation_authorized") is True
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False
        and protocol.get("schema_version") == PROTOCOL_SCHEMA
        and protocol.get("status") == PROTOCOL_STATUS,
        "Track-R V-fit validation authority/protocol drift",
    )
    authority_sha256 = file_sha256(authority_path)
    authority_logical_sha256 = str(authority["logical_sha256"])
    validate_runtime_bindings(authority)
    primary = _load_run(
        primary_output,
        outer_fold,
        authority_sha256,
        authority_logical_sha256,
    )
    resumed = _load_run(
        resume_output,
        outer_fold,
        authority_sha256,
        authority_logical_sha256,
    )
    require(primary["input"] == resumed["input"], "fresh/resume input materialization drift")
    require(
        primary["resume"]["schedule_events"] == resumed["resume"]["schedule_events"]
        and primary["resume"]["loss_trace"] == resumed["resume"]["loss_trace"]
        and primary["resume"]["schedule_sha256"]
        == resumed["resume"]["schedule_sha256"],
        "fresh/resume schedule or loss trace drift",
    )
    primary_state = primary["checkpoint"]["model_state_dict"]
    resumed_state = resumed["checkpoint"]["model_state_dict"]
    require(set(primary_state) == set(resumed_state), "fresh/resume state key drift")
    require(
        all(torch.equal(primary_state[key], resumed_state[key]) for key in primary_state),
        "fresh/resume model tensor drift",
    )
    require(
        primary["result"]["model_state_sha256"]
        == resumed["result"]["model_state_sha256"]
        and primary["result"]["completed_updates"] == resumed["result"]["completed_updates"]
        == UPDATES,
        "fresh/resume terminal state drift",
    )
    result: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "outer_fold": outer_fold,
        "authority_sha256": authority_sha256,
        "authority_logical_sha256": authority_logical_sha256,
        "completed_updates": UPDATES,
        "episode_count": primary["input"]["episode_count"],
        "model_state_sha256": primary["result"]["model_state_sha256"],
        "schedule_sha256": primary["resume"]["schedule_sha256"],
        "fresh_result_sha256": file_sha256(primary["root"] / "result.json"),
        "resume_result_sha256": file_sha256(resumed["root"] / "result.json"),
        "fresh_resume_model_bit_exact": True,
        "fresh_resume_schedule_trace_exact": True,
        "relative_pair_loss_only": True,
        "donor_null_absolute_loss_count": 0,
        "outer_heldout_forward_count": 0,
        "heldout_scoring_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(safe_path(output.parent, file=False) / output.name, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--resume-output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.authority,
        args.protocol,
        args.primary_output,
        args.resume_output,
        args.outer_fold,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "outer_fold": result["outer_fold"],
                "model_state_sha256": result["model_state_sha256"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
