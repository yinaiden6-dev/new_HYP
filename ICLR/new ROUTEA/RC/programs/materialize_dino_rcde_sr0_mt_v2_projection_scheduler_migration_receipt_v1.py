#!/usr/bin/env python3
"""Retrospectively seal V113 scheduler migration and 50-shard output membership."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "registry/current_authority_v113_20260820.json"
BASE = ROOT / "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1"
OUT = BASE / "scheduler_migration_closure_input_v1.json"


class ReceiptError(RuntimeError):
    pass


def req(condition: Any, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, name: str) -> dict[str, Any]:
    req(path.is_file() and not path.is_symlink(), f"{name} absent")
    value = json.loads(path.read_text())
    req(isinstance(value, dict), f"{name} object drift")
    return value


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists(), "receipt output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.chmod(0o444)
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sacct_rows(job_ids: list[str]) -> dict[str, dict[str, str]]:
    command = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobID,State,ExitCode,Elapsed,Start,End",
        "--parsable2",
        "--noheader",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    req(completed.returncode == 0, f"sacct failed: {completed.stderr.strip()}")
    result = {}
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or "." in fields[0]:
            continue
        job_id, state, exit_code, elapsed, start, end = fields[:6]
        result[job_id] = {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "start": start,
            "end": end,
        }
    return result


def main() -> None:
    authority = read(AUTH, "V113 authority")
    req(
        authority.get("status")
        == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_RESCOPE_REPAIR_AUTHORIZED"
        and authority.get("logical_sha256") == logical(authority),
        "V113 authority drift",
    )
    dev_jobs = {0: "5093425"}
    for index in range(1, 11):
        continuation = read(BASE / "continuation" / f"dev_submit_{index:02d}.json", f"continuation {index}")
        req(
            continuation.get("status") == "DEV_NEXT_SHARD_SUBMISSION_COMMITTED"
            and continuation.get("next_shard_ordinal") == index
            and continuation.get("logical_sha256") == logical(continuation),
            "dev continuation drift",
        )
        dev_jobs[index] = continuation["submitted_job_id"]
    expected_jobs = list(dev_jobs.values()) + ["5093957", "5093426"]
    accounting = sacct_rows(expected_jobs)
    moved_launcher = ROOT / "slurm/dino_rcde_sr0_mt_v2_donor_geometry_projection_cpu_moved11_19_v1.sbatch"
    override_addendum = ROOT / "plan/DINO_RCDE_SR0_MT_V2_DONOR_GEOMETRY_V113_SCHEDULER_OVERRIDE_20260820.md"
    migration_stop = BASE / "continuation/dev_submit_11.claim.json"
    stop_value = read(migration_stop, "migration stop")
    req(
        stop_value.get("schema_version")
        == "rc_dino_rcde_sr0_mt_v2_donor_geometry_dev_migration_stop_v1_20260820"
        and stop_value.get("status") == "DEV_CONTINUATION_STOPPED_FOR_CPUONLY_MIGRATION"
        and stop_value.get("moved_cpuonly_shards") == list(range(11, 20))
        and stop_value.get("logical_sha256") == logical(stop_value),
        "migration stop receipt drift",
    )
    rows = []
    for shard in range(50):
        if shard <= 10:
            job = dev_jobs[shard]
            route = "DEV_SERIAL"
            job_key = job
        elif shard <= 19:
            job = "5093957"
            route = "CPU_MIGRATED_11_19"
            job_key = f"5093957_{shard}"
        else:
            job = "5093426"
            route = "CPU_ORIGINAL_20_49"
            job_key = f"5093426_{shard}"
        req(job_key in accounting, f"accounting row absent {job_key}")
        state = accounting[job_key]
        if shard == 10:
            req(state["state"] == "FAILED" and state["exit_code"] == "1:0", "dev10 expected postvalidation failure drift")
        else:
            req(state["state"] == "COMPLETED" and state["exit_code"] == "0:0", f"shard job failure {shard}")
        start, stop = shard * 12, shard * 12 + 12
        producer_path = BASE / "producer" / f"shard_{start:03d}_{stop:03d}.json"
        validation_path = BASE / "validation" / f"shard_{start:03d}_{stop:03d}.json"
        producer = read(producer_path, f"producer {shard}")
        validation = read(validation_path, f"validation {shard}")
        req(
            producer.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_READY"
            and validation.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_INDEPENDENT_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and validation.get("projection_sha256") == fsha(producer_path)
            and validation.get("projection_logical_sha256") == producer["logical_sha256"],
            "shard output closure drift",
        )
        rows.append(
            {
                "shard_ordinal": shard,
                "route": route,
                "array_job_id": job,
                "accounting": state,
                "producer_path": producer_path.relative_to(ROOT).as_posix(),
                "producer_sha256": fsha(producer_path),
                "producer_logical_sha256": producer["logical_sha256"],
                "validation_path": validation_path.relative_to(ROOT).as_posix(),
                "validation_sha256": fsha(validation_path),
                "validation_logical_sha256": validation["logical_sha256"],
                "query_count": validation["query_count"],
                "scope_projection_count": validation["scope_projection_count"],
                "direction_record_decode_count": validation[
                    "direction_record_decode_count"
                ],
            }
        )
    result = {
        "schema_version": "rc_dino_rcde_sr0_mt_v2_projection_scheduler_migration_receipt_v1_20260820",
        "status": "RCDE_SR0_MT_V2_PROJECTION_SCHEDULER_MIGRATION_INPUT_READY",
        "authority_sha256": fsha(AUTH),
        "authority_logical_sha256": authority["logical_sha256"],
        "superseded_pending_job_id": 5093135,
        "dev_initial_job_id": 5093425,
        "dev_postvalidation_failed_job_id": 5094030,
        "migrated_cpu_array_job_id": 5093957,
        "original_cpu_array_job_id": 5093426,
        "moved_launcher_path": moved_launcher.relative_to(ROOT).as_posix(),
        "moved_launcher_sha256": fsha(moved_launcher),
        "scheduler_override_addendum_path": override_addendum.relative_to(ROOT).as_posix(),
        "scheduler_override_addendum_sha256": fsha(override_addendum),
        "migration_stop_path": migration_stop.relative_to(ROOT).as_posix(),
        "migration_stop_sha256": fsha(migration_stop),
        "migration_stop_logical_sha256": stop_value["logical_sha256"],
        "shard_count": 50,
        "query_count": sum(item["query_count"] for item in rows),
        "scope_projection_count": sum(item["scope_projection_count"] for item in rows),
        "direction_record_decode_count": sum(
            item["direction_record_decode_count"] for item in rows
        ),
        "shard_rows": rows,
        "shard_population_sha256": canonical(rows),
        "projection_logic_changed": False,
        "model_load_count": 0,
        "training_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    req(
        result["query_count"] == 594
        and result["scope_projection_count"] == 2376
        and result["direction_record_decode_count"] == 4752,
        "global migration closure count drift",
    )
    result["logical_sha256"] = logical(result)
    atomic(OUT, result)
    print(json.dumps({key: result[key] for key in ("status", "shard_count", "query_count", "scope_projection_count", "direction_record_decode_count", "logical_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
