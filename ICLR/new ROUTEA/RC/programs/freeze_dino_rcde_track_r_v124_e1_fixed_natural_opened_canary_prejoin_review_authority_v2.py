#!/usr/bin/env python3
"""Build, but do not auto-promote, the non-executable E1 review-V2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v2 as A  # noqa: E402


OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v2_20260826.json"
SCHEMA = "rc_dino_rcde_track_r_v124_e1_review_authority_v2_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_REVIEW_V2_CANDIDATE_NOT_EXECUTION_AUTHORITY"
V1_REVIEW = "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.json"
V1_REJECTION = "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.terminal_rejection_v1.json"


class E1ReviewV2FreezeError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1ReviewV2FreezeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return A.canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def binding(path_value: str) -> dict[str, object]:
    raw = Path(path_value)
    path = (raw if raw.is_absolute() else ROOT / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"review binding absent: {path_value}")
    return {
        "path": path_value,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_authority() -> dict[str, Any]:
    v1 = json.loads((ROOT / V1_REVIEW).read_text(encoding="ascii"))
    rejection = json.loads((ROOT / V1_REJECTION).read_text(encoding="ascii"))
    require(
        file_sha256(ROOT / V1_REVIEW)
        == "0476e159c039d5bfb685e392984bc61d5607ce09a68ebf7df931b186d8e1f4e7"
        and v1.get("execution_authorized") is False
        and rejection.get("terminal_review_decision") == "FAIL"
        and rejection.get("execution_authorized") is False,
        "review V1 rejection boundary drift",
    )
    rows = {
        "contract_v1": binding(
            "plan/DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_CONTRACT_V1_20260826.md"
        ),
        "contract_v2": binding(
            "plan/DINO_RCDE_TRACK_R_V124_E1_FIXED_NATURAL_OPENED_CANARY_PREJOIN_CONTRACT_V2_20260826.md"
        ),
        "review_v1_authority": binding(V1_REVIEW),
        "review_v1_terminal_rejection": binding(V1_REJECTION),
        "rejection_generator": binding(
            "programs/reject_dino_rcde_track_r_v124_e1_review_authority_v1_terminal_v1.py"
        ),
        "producer_v2": binding(
            "programs/run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.py"
        ),
        "independent_validator_v2": binding(
            "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v2.py"
        ),
        "atomic_family_v2": binding(
            "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_atomic_family_v2.py"
        ),
        "authority_runtime_v2": binding(
            "src/rc_aslo_xf/dino_rcde_track_r_v124_e1_authority_v2.py"
        ),
        "tests_v2": binding(
            "tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.py"
        ),
        "launcher_v2": binding(
            "slurm/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.sbatch"
        ),
        "review_freezer_v2": binding(
            "programs/freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v2.py"
        ),
    }
    for relative in A.discover_import_closure():
        rows[A.closure_key(relative)] = binding(relative)
    # Preserve every upstream prerequisite byte already frozen by review V1.
    for name, row in v1["bindings"].items():
        key = f"upstream_v1__{name}"
        observed = binding(str(row["path"]))
        require(
            observed["bytes"] == row["bytes"]
            and observed["sha256"] == row["sha256"],
            f"review V1 upstream drift: {name}",
        )
        rows[key] = observed
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "REVIEW_ONLY_NO_EXECUTION_NO_OUTPUT_NO_SCIENCE",
        "bindings": rows,
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": A.CANDIDATE_AXIS_SHA256,
        "pair_sha256": A.PAIR_SHA256,
        "v_checkpoint_file_sha256": A.V_CHECKPOINT_FILE_SHA256,
        "v_checkpoint_state_sha256": A.V_CHECKPOINT_STATE_SHA256,
        "family_sequence": list(A.FAMILIES),
        "arm_sequence": list(A.ARMS),
        "c_col_deep_validation_required": {
            "wrapper_count": 128,
            "record_count": 256,
            "pair_runtime_candidate_count": 2,
        },
        "atomic_family_publication_required": True,
        "exact_binding_whitelist_required": True,
        "import_closure": list(A.discover_import_closure()),
        "import_closure_sha256": A.canonical_sha256(
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
        "terminal_independent_review_status": "PENDING",
        "blocking_requirements": [
            "TERMINAL_INDEPENDENT_REVIEW_V2_PASS_MISSING",
            "REVIEW_V2_AUTHORITY_NOT_FROZEN",
            "EXECUTION_AUTHORITY_V2_NOT_CREATED",
            "SMOKE_AND_OUTPUT_FORBIDDEN_UNTIL_TERMINAL_PASS",
        ],
        "execution_authorized": False,
        "smoke_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "submission_authorized": False,
        "target_rival_join_authorized": False,
        "scientific_reduction_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def run(output: Path = OUTPUT) -> Mapping[str, Any]:
    require(Path.cwd().resolve() == ROOT.resolve(), "review V2 cwd drift")
    require(not output.exists() and not output.is_symlink(), "review V2 output exists")
    value = build_authority()
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    temporary.chmod(0o444)
    os.link(temporary, output)
    temporary.unlink()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print(json.dumps({"status": value["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
