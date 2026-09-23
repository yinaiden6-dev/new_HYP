#!/usr/bin/env python3
"""Build the unfrozen, non-executable E1 review-V3 candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v3 as A  # noqa: E402


OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v3_20260826.json"
SCHEMA = "rc_dino_rcde_track_r_v124_e1_review_authority_v3_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_REVIEW_V3_CANDIDATE_NOT_EXECUTION_AUTHORITY"
V1_REVIEW = "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.json"
V1_REJECTION = "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.terminal_rejection_v1.json"
V2_REJECTION = "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_v2_terminal_rejection_v1_20260826.json"


class E1ReviewV3Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1ReviewV3Error(message)


def binding(path_value: str) -> dict[str, object]:
    raw = Path(path_value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"V3 review binding absent: {path_value}")
    return {
        "path": path_value,
        "bytes": path.stat().st_size,
        "sha256": A.V2.file_sha256(path),
    }


def logical_sha256(value: Mapping[str, Any]) -> str:
    return A.V2.logical_sha256(value)


def build_authority() -> dict[str, Any]:
    v1 = json.loads((ROOT / V1_REVIEW).read_text(encoding="ascii"))
    reject1 = json.loads((ROOT / V1_REJECTION).read_text(encoding="ascii"))
    reject2 = json.loads((ROOT / V2_REJECTION).read_text(encoding="ascii"))
    require(
        v1.get("execution_authorized") is False
        and reject1.get("terminal_review_decision") == "FAIL"
        and reject2.get("terminal_review_decision") == "FAIL"
        and reject2.get("review_v2_authority_created") is False
        and reject2.get("smoke_executed") is False
        and reject2.get("submission_executed") is False,
        "V1/V2 terminal rejection lineage drift",
    )
    rows = {
        "contract_v1": binding(
            "plan/DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_CONTRACT_V1_20260826.md"
        ),
        "contract_v2": binding(
            "plan/DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_CONTRACT_V2_20260826.md"
        ),
        "contract_v3": binding(
            "plan/DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_CONTRACT_V3_20260826.md"
        ),
        "review_v1_authority": binding(V1_REVIEW),
        "review_v1_terminal_rejection": binding(V1_REJECTION),
        "review_v2_terminal_rejection": binding(V2_REJECTION),
        "producer_v3": binding(
            "programs/run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py"
        ),
        "independent_validator_v3": binding(
            "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v3.py"
        ),
        "atomic_family_v3": binding(
            "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_atomic_family_v3.py"
        ),
        "authority_runtime_v3": binding(
            "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_authority_v3.py"
        ),
        "tests_v3": binding(
            "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py"
        ),
        "launcher_test_full_c128_c_dino_v1": binding(
            "tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py"
        ),
        "launcher_test_v_runtime_v2": binding(
            "tests/test_dino_rcde_sr0_mt_v_runtime_v2.py"
        ),
        "launcher_test_phase_b_all_patch_v1": binding(
            "tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py"
        ),
        "launcher_test_e1_v3": binding(
            "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py"
        ),
        "launcher_v3": binding(
            "slurm/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.sbatch"
        ),
        "review_freezer_v3": binding(
            "programs/freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v3.py"
        ),
    }
    for relative in A.discover_import_closure():
        rows[A.closure_key(relative)] = binding(relative)
    for name, row in v1["bindings"].items():
        observed = binding(str(row["path"]))
        require(
            observed["bytes"] == row["bytes"]
            and observed["sha256"] == row["sha256"],
            f"V1 upstream drift: {name}",
        )
        rows[f"upstream_v1__{name}"] = observed
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "REVIEW_ONLY_NO_EXECUTION_NO_OUTPUT_NO_SCIENCE",
        "bindings": rows,
        "terminal_independent_review_status": "PENDING",
        "v2_review_authority_created": False,
        "execution_authorized": False,
        "smoke_authorized": False,
        "submission_authorized": False,
        "raw_symlink_rejection_required": True,
        "family_member_logical_sha_required": True,
        "launcher_test_binding_count": 4,
        "import_closure": list(A.discover_import_closure()),
        "import_closure_sha256": A.V2.canonical_sha256(
            list(A.discover_import_closure())
        ),
        "resource_contract": {
            "partition": "accelerated",
            "allowed_partitions": ["accelerated"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def run(output: Path = OUTPUT) -> Mapping[str, Any]:
    require(not output.exists() and not output.is_symlink(), "V3 review output exists")
    value = build_authority()
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    temporary.chmod(0o444)
    os.link(temporary, output)
    temporary.unlink()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps({"status": run(args.output)["status"]}, sort_keys=True))
