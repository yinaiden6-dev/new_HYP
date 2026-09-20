#!/usr/bin/env python3
"""Freeze the v2 repair for the BAG foldset-validation status literal."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


RC_ROOT = Path(__file__).resolve().parents[1]
PARENT_REL = "registry/dino_rcde_bag_full594_completion_authority_v1_20260827.json"
OUTPUT_REL = "registry/dino_rcde_bag_full594_predecessor_status_repair_authority_v2_20260827.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(relative: str) -> dict[str, Any]:
    path = (RC_ROOT / relative).resolve()
    if not path.is_file() or path.is_symlink() or RC_ROOT not in path.parents:
        raise RuntimeError(f"unsafe/missing v2 binding: {relative}")
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> int:
    parent_path = (RC_ROOT / PARENT_REL).resolve()
    output = (RC_ROOT / OUTPUT_REL).resolve()
    if output.exists() or sha(parent_path) != "ef9988695ae90cd5f660b5337da59fc0c636d7d264adc83f8b57df312d6b6341":
        raise RuntimeError("v2 parent/output drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    authority = dict(parent)
    authority["schema_version"] = "rc_dino_rcde_bag_full594_predecessor_status_repair_authority_v2_20260827"
    authority["status"] = "DINO_RCDE_BAG_FULL594_PREDECESSOR_STATUS_REPAIR_AUTHORIZED"
    authority["parent_authority"] = {
        "path": PARENT_REL,
        "sha256": sha(parent_path),
        "status": parent["status"],
    }
    authority["repair_receipt"] = {
        "failed_job_id": 5112847,
        "failed_state": "FAILED",
        "failed_exit_code": "1:0",
        "elapsed": "00:00:51",
        "maximum_memory_megabytes": 419,
        "model_load_count": 0,
        "candidate_forward_count": 0,
        "committed_output_count": 0,
        "failure_domain": "PRE_MODEL_PREDECESSOR_STATUS_LITERAL",
        "incorrect_literal": "DINO_RCDE_R1_MAIN_BAG_FOLDSET_VALIDATION_PASS",
        "repaired_literal": "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS",
        "scientific_failure": False,
    }
    bindings = dict(parent["bindings"])
    replacements = {
        "freezer": "programs/freeze_dino_rcde_bag_full594_predecessor_status_repair_authority_v2.py",
        "io": "programs/dino_rcde_bag_completion_io_v1.py",
        "producer": "programs/materialize_dino_rcde_bag_full594_shard_v1.py",
        "tests": "tests/test_dino_rcde_bag_completion_v1.py",
        "gpu_launcher": "slurm/dino_rcde_bag_full594_canary_dev_a100_v2.sbatch",
        "finalize_launcher": "slurm/dino_rcde_bag_full594_serial_completion_v2.sbatch",
    }
    for name, relative in replacements.items():
        bindings[name] = bind(relative)
    # Re-hash every other internal binding so v2 proves that no unlisted source
    # drift is being smuggled through the repair.
    for name, item in tuple(bindings.items()):
        if name == "gallery_cache" or name in replacements:
            continue
        bindings[name] = bind(str(item["path"]))
    gallery = Path(str(bindings["gallery_cache"]["path"])).resolve()
    if not gallery.is_file() or gallery.is_symlink():
        raise RuntimeError("v2 gallery binding drift")
    bindings["gallery_cache"] = {
        "path": str(gallery),
        "sha256": sha(gallery),
        "bytes": gallery.stat().st_size,
    }
    bindings["parent_authority_v1"] = {
        "path": PARENT_REL,
        "sha256": sha(parent_path),
        "bytes": parent_path.stat().st_size,
    }
    authority["bindings"] = bindings
    authority["resource_contract"] = {
        "canary_partition": "dev_accelerated",
        "canary_walltime_seconds": 1800,
        "canary_cpus_per_task": 8,
        "canary_memory_megabytes": 120000,
        "canary_gpu_count": 1,
        "serial_completion_partition": "accelerated",
        "serial_completion_walltime_seconds": 28800,
        "serial_completion_cpus_per_task": 8,
        "serial_completion_memory_megabytes": 128000,
        "serial_completion_gpu_count": 1,
        "serial_order": "SHARDS_0_THROUGH_49_WITH_PER_SHARD_VALIDATION",
    }
    authority["scientific_GO_or_NO_GO"] = None
    authority["automatic_stage_advance"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    output.chmod(0o444)
    print(json.dumps({"status": authority["status"], "output": str(output), "sha256": sha(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
