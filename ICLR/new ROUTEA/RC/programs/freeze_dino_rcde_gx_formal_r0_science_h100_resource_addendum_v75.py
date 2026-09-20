#!/usr/bin/env python3
"""Freeze resource/provenance addendum for V68 science on accelerated-h100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("registry/h0/gx_cbnr_formal_r0_science_authority_v68_20260825.json")
TRAINING_ADDENDUM = Path("registry/h0/gx_cbnr_formal_training_h100_resource_addendum_v74_20260826.json")
LAUNCHER = Path("slurm/dino_rcde_gx_formal_r0_science_v68_accelerated_h100.sbatch")
OUTPUT = Path("registry/h0/gx_cbnr_formal_r0_science_h100_resource_addendum_v75_20260826.json")
STATUS = "GX_CBNR_R0_SCIENCE_ACCELERATED_H100_RESOURCE_ADDENDUM_AUTHORIZED"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def build() -> dict[str, Any]:
    base = read(BASE)
    training = read(TRAINING_ADDENDUM)
    require(
        base.get("status") == "GX_CBNR_R0_SCIENTIFIC_REDUCTION_EXECUTION_AUTHORIZED"
        and base.get("logical_sha256") == H.logical(base)
        and training.get("status")
        == "GX_CBNR_FORMAL_TRAINING_ACCELERATED_H100_RESOURCE_ADDENDUM_AUTHORIZED"
        and training.get("logical_sha256") == H.logical(training),
        "V68/V74 parent authority drift",
    )
    value: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_gx_formal_r0_science_h100_resource_addendum_v75_20260826",
        "status": STATUS,
        "claim_level": "RESOURCE_AND_TRAINING_PROVENANCE_ADDENDUM_NO_SCIENTIFIC_CHANGE",
        "resource_extension_authorized": True,
        "model_data_metric_threshold_reducer_or_validator_change": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "resource_contract": {
            "partition": "accelerated-h100",
            "cpus_per_task": 16,
            "memory_megabytes": 64000,
            "allocated_gpus": 1,
            "GPU_compute_authorized": False,
            "CUDA_must_be_hidden_after_allocation_check": True,
            "walltime_seconds": 7200,
        },
        "bindings": {
            "base_v68_authority": H.bind(str(BASE), immutable=True),
            "v74_h100_training_addendum": H.bind(str(TRAINING_ADDENDUM), immutable=True),
            "launcher": H.bind(str(LAUNCHER)),
            "freezer": H.bind(
                "programs/freeze_dino_rcde_gx_formal_r0_science_h100_resource_addendum_v75.py"
            ),
        },
    }
    value["logical_sha256"] = H.logical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    require(
        args.output.resolve(strict=False) == (ROOT / OUTPUT).resolve(strict=False),
        "V75 output drift",
    )
    value = build()
    if args.output.exists():
        require(read(OUTPUT) == value, "existing V75 addendum drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
