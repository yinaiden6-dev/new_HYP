"""Resource-only two-stage dev_accelerated authority for V124 E1 smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as PARENT


ROOT = PARENT.BASE.RC_ROOT
AUTHORITY_PATH = (
    "registry/dino_rcde_track_r_v124_e1_dev_staged_smoke_authority_v4_20260826.json"
)
PARENT_AUTHORITY_PATH = PARENT.AUTHORITY_PATH
PRODUCER_NAMESPACE = (
    "results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_producer_committed"
)
VALIDATION_NAMESPACE = (
    "results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_validation_committed"
)
FORMAL_PRODUCER_NAMESPACE = PARENT.FORMAL_PRODUCER_NAMESPACE
FORMAL_VALIDATION_NAMESPACE = PARENT.FORMAL_VALIDATION_NAMESPACE

OVERLAY_PATHS = {
    "addendum": "plan/DINO_RCDE_TRACK_R_V124_E1_DEV_ACCELERATED_STAGED_SMOKE_V4_20260826.md",
    "authority_runtime": "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_dev_staged_authority_v4.py",
    "producer": "programs/run_dino_rcde_track_r_v124_e1_dev_staged_producer_v4.py",
    "validator": "programs/validate_dino_rcde_track_r_v124_e1_dev_staged_independent_v4.py",
    "producer_launcher": "slurm/dino_rcde_track_r_v124_e1_dev_staged_producer_v4.sbatch",
    "validator_launcher": "slurm/dino_rcde_track_r_v124_e1_dev_staged_validator_v4.sbatch",
    "freezer": "programs/freeze_dino_rcde_track_r_v124_e1_dev_staged_authority_v4.py",
    "authority_validator": "programs/validate_dino_rcde_track_r_v124_e1_dev_staged_authority_v4.py",
    "tests": "tests/test_dino_rcde_track_r_v124_e1_dev_staged_v4.py",
}


class DevStagedAuthorityV4Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise DevStagedAuthorityV4Error(message)


def _raw(path_value: str | Path) -> Path:
    path = Path(path_value)
    return Path(os.path.abspath(os.fspath(path if path.is_absolute() else ROOT / path)))


def _sha(path: Path) -> str:
    return PARENT.BASE.V2.file_sha256(path)


def _regular(path_value: str | Path, *, mode: int | None = None) -> Path:
    path = _raw(path_value)
    require(os.path.lexists(path), f"bound path absent: {path}")
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"bound path type drift: {path}")
    if mode is not None:
        require(metadata.st_mode & 0o777 == mode, f"bound path mode drift: {path}")
    cursor = ROOT if ROOT in path.parents else Path("/")
    for part in path.relative_to(cursor).parts:
        cursor /= part
        if os.path.lexists(cursor):
            require(not stat.S_ISLNK(os.lstat(cursor).st_mode), f"symlink component: {cursor}")
    require(path.resolve() == path, f"bound path alias drift: {path}")
    return path


def logical(value: Mapping[str, Any]) -> str:
    return PARENT.BASE.V2.logical_sha256(value)


def validate_candidate(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "status", "claim_level", "parent_authority_path",
        "parent_authority_sha256", "execution_scope", "resource_contract",
        "stage_sequence", "output_contract", "bindings", "binding_modes",
        "scientific_GO_or_NO_GO", "eligible_as_e1_result",
        "formal_or_full594_execution_authorized", "target_rival_join_authorized",
        "postjoin_authorized", "scientific_reduction_authorized",
        "automatic_stage_advance", "automatic_submit_authorized",
        "next_authorized_stage", "logical_sha256",
    }
    require(set(value) == required, "V4 authority key set drift")
    require(
        value["schema_version"] == "rc_dino_rcde_track_r_v124_e1_dev_staged_authority_v4_20260826"
        and value["status"] == "DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_EXECUTION_AUTHORIZED"
        and value["claim_level"] == "DEV_ACCELERATED_STAGED_REAL_SMOKE_ENGINEERING_ONLY_NONPROMOTABLE"
        and value["parent_authority_path"] == PARENT_AUTHORITY_PATH
        and value["execution_scope"] == "EXECUTION0_DEV_ACCELERATED_TWO_STAGE_REAL_SMOKE_V4_ONLY_NONPROMOTABLE"
        and value["resource_contract"] == {
            "partition": "dev_accelerated", "allowed_partitions": ["dev_accelerated"],
            "gpu_count": 1, "cpus_per_task": 8, "memory_megabytes": 128000,
            "walltime_seconds_per_stage": 3600,
        }
        and value["stage_sequence"] == [
            {"stage": 0, "role": "PRODUCER", "dependency": None},
            {"stage": 1, "role": "INDEPENDENT_VALIDATOR", "dependency": "afterok:stage0"},
        ]
        and value["output_contract"] == {
            "producer_family": PRODUCER_NAMESPACE,
            "validation_family": VALIDATION_NAMESPACE,
            "append_only": True, "atomic_family_required": True,
            "producer_commit_is_stage_checkpoint": True,
            "exact_committed_reuse": True, "partial_family_consumable": False,
            "eligible_as_e1_result": False,
        }
        and value["eligible_as_e1_result"] is False
        and value["formal_or_full594_execution_authorized"] is False
        and value["target_rival_join_authorized"] is False
        and value["postjoin_authorized"] is False
        and value["scientific_reduction_authorized"] is False
        and value["scientific_GO_or_NO_GO"] is None
        and value["automatic_stage_advance"] is False
        and value["automatic_submit_authorized"] is False
        and value["next_authorized_stage"] is None
        and value["logical_sha256"] == logical(value),
        "V4 authority semantic drift",
    )
    bindings = value["bindings"]
    modes = value["binding_modes"]
    require(set(bindings) == set(OVERLAY_PATHS) and set(modes) == set(OVERLAY_PATHS), "V4 overlay binding set drift")
    for key, expected in OVERLAY_PATHS.items():
        row = bindings[key]
        require(set(row) == {"path", "bytes", "sha256"} and row["path"] == expected, f"V4 binding path drift: {key}")
        path = _regular(expected, mode=modes[key])
        require(path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"], f"V4 binding bytes/hash drift: {key}")


def read_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    authority_path = _regular(path, mode=0o444)
    require(authority_path == _raw(AUTHORITY_PATH) and _sha(authority_path) == expected_sha256, "V4 authority physical/path drift")
    value = json.loads(authority_path.read_text(encoding="ascii"))
    validate_candidate(value)
    parent_path = _regular(PARENT_AUTHORITY_PATH, mode=0o444)
    require(_sha(parent_path) == value["parent_authority_sha256"], "V4 parent authority hash drift")
    parent = PARENT.read_authority(parent_path, value["parent_authority_sha256"])
    for relative in (PRODUCER_NAMESPACE, VALIDATION_NAMESPACE, FORMAL_PRODUCER_NAMESPACE, FORMAL_VALIDATION_NAMESPACE):
        raw = _raw(relative)
        if os.path.lexists(raw):
            require(not stat.S_ISLNK(os.lstat(raw).st_mode), f"V4 output namespace symlink: {relative}")
    merged = dict(parent)
    merged.update({
        "execution_scope": value["execution_scope"],
        "resource_contract": value["resource_contract"],
        "output_contract": value["output_contract"],
        "eligible_as_e1_result": False,
        "formal_or_full594_execution_authorized": False,
        "target_rival_join_authorized": False,
        "postjoin_authorized": False,
        "scientific_reduction_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "automatic_submit_authorized": False,
        "next_authorized_stage": None,
    })
    return merged


__all__ = [
    "AUTHORITY_PATH", "PRODUCER_NAMESPACE", "VALIDATION_NAMESPACE",
    "FORMAL_PRODUCER_NAMESPACE", "FORMAL_VALIDATION_NAMESPACE",
    "OVERLAY_PATHS", "read_authority", "validate_candidate", "logical",
]

