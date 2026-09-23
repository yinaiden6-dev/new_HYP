#!/usr/bin/env python3
"""Build the unfrozen exact-path-map accelerated real-smoke authority V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as S  # noqa: E402


OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.json"
V1_AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.json"
V1_REJECTION = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.terminal_rejection_v1.json"


class SmokeAuthorityV2FreezeError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeAuthorityV2FreezeError(message)


def binding(path_value: str) -> dict[str, object]:
    raw = Path(path_value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"smoke V2 input absent: {path_value}")
    return {
        "path": path_value,
        "bytes": path.stat().st_size,
        "sha256": S.BASE.V2.file_sha256(path),
    }


def build_authority() -> dict[str, Any]:
    for relative in (
        S.PRODUCER_NAMESPACE,
        S.VALIDATION_NAMESPACE,
        S.FORMAL_PRODUCER_NAMESPACE,
        S.FORMAL_VALIDATION_NAMESPACE,
    ):
        raw = S._raw_path(relative)
        require(
            not os.path.lexists(raw),
            f"smoke/formal output exists or is dangling symlink: {relative}",
        )
    v1 = json.loads(V1_AUTHORITY.read_text(encoding="ascii"))
    rejection = json.loads(V1_REJECTION.read_text(encoding="ascii"))
    require(
        S.BASE.V2.file_sha256(V1_AUTHORITY)
        == "0c646a33b76b4dc6b0af6f927a182da58bb39616aca7115db66b3aa6db143ab7"
        and V1_AUTHORITY.stat().st_mode & 0o777 == 0o444
        and rejection.get("terminal_review_decision") == "FAIL"
        and rejection.get("authority_sha256")
        == "0c646a33b76b4dc6b0af6f927a182da58bb39616aca7115db66b3aa6db143ab7"
        and rejection.get("smoke_output_created") is False
        and rejection.get("submission_executed") is False,
        "smoke V1 authority/rejection lineage drift",
    )
    rows = {}
    for key in S.BASE.required_bindings():
        row = v1.get("bindings", {}).get(key)
        require(isinstance(row, Mapping), f"smoke V1 base binding absent: {key}")
        observed = binding(str(row["path"]))
        require(
            observed["bytes"] == row["bytes"]
            and observed["sha256"] == row["sha256"],
            f"smoke V1 base binding drift: {key}",
        )
        rows[key] = observed
    for key, path in S.OVERLAY_PATHS.items():
        rows[key] = binding(path)
    require(set(rows) == S.required_bindings(), "smoke V2 exact binding set drift")
    value: dict[str, Any] = {
        "schema_version": S.BASE.SCHEMA,
        "status": S.BASE.STATUS,
        "bindings": rows,
        "independent_review_status": "PASS",
        "submission_review_status": "PENDING",
        "execution_scope": "EXECUTION0_ACCELERATED_REAL_SMOKE_V2_ONLY_NONPROMOTABLE",
        "real_smoke": True,
        "eligible_as_e1_result": False,
        "formal_or_full594_execution_authorized": False,
        "manual_submission_requires_independent_review": True,
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": S.BASE.CANDIDATE_AXIS_SHA256,
        "pair_sha256": S.BASE.PAIR_SHA256,
        "v_checkpoint_file_sha256": S.BASE.V_CHECKPOINT_FILE_SHA256,
        "v_checkpoint_state_sha256": S.BASE.V_CHECKPOINT_STATE_SHA256,
        "family_sequence": list(S.BASE.FAMILIES),
        "arm_sequence": list(S.BASE.ARMS),
        "expected_pair_arm_record_count": 9,
        "expected_directional_term_count": 36,
        "device_schedule": {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        },
        "resource_contract": {
            "partition": "accelerated",
            "allowed_partitions": ["accelerated"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        },
        "output_contract": {
            "producer_family": S.PRODUCER_NAMESPACE,
            "validation_family": S.VALIDATION_NAMESPACE,
            "append_only": True,
            "atomic_family_required": True,
            "exact_committed_reuse": True,
            "overwrite_or_repair_authorized": False,
            "eligible_as_e1_result": False,
        },
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "p_training_authorized": False,
        "v_training_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "target_rival_join_authorized": False,
        "postjoin_authorized": False,
        "scientific_reduction_authorized": False,
        "automatic_submit_authorized": False,
        "smoke_authorized": True,
        "submission_authorized": False,
        "base_v3_import_closure_count": 65,
        "base_v3_binding_count": 115,
        "smoke_overlay_binding_count": len(S.OVERLAY_PATHS),
        "binding_modes": {
            key: (
                Path(row["path"])
                if Path(row["path"]).is_absolute()
                else ROOT / row["path"]
            ).stat().st_mode
            & 0o777
            for key, row in rows.items()
        },
        "authority_file_mode": 0o444,
        "smoke_overlay_paths": dict(S.OVERLAY_PATHS),
        "smoke_overlay_import_closure": list(
            S.discover_overlay_import_closure()
        ),
        "smoke_overlay_import_closure_sha256": S.BASE.V2.canonical_sha256(
            list(S.discover_overlay_import_closure())
        ),
        "launcher_test_paths": list(S.LAUNCHER_TEST_PATHS),
        "launcher_executable_paths": dict(S.LAUNCHER_EXECUTABLE_PATHS),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "import_closure": list(S.BASE.discover_import_closure()),
        "import_closure_sha256": S.BASE.V2.canonical_sha256(
            list(S.BASE.discover_import_closure())
        ),
    }
    value["logical_sha256"] = S.BASE.V2.logical_sha256(value)
    S.validate_authority_envelope(value)
    return value


def run(output: Path = OUTPUT) -> Mapping[str, Any]:
    require(not output.exists() and not output.is_symlink(), "smoke V2 authority exists")
    value = build_authority()
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii")
    temporary.chmod(0o444)
    os.link(temporary, output)
    temporary.unlink()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.output)["status"]}, sort_keys=True))
