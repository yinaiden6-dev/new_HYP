#!/usr/bin/env python3
"""Execution-only four long packs for frozen unified acquisition indices128..592.

Original scientific collectors and validators remain unchanged. The first128 are
owned by their already running chain. This stdlib-only parent writes execution
receipts outside the restricted scientific child process; no labels are opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "registry/rc_unified593_long4_authority_v1_20260927.json"
OUT = ROOT / "results/rc_h593_unified_long4_v1"
CACHE = ROOT / "cache/rc_h593_unified_acquisition_v1"
SCIENCE = ROOT / "registry/rc_h593_unified_acquisition_authority_v1_20260923.json"
RESUME_V2 = ROOT / "registry/rc_unified128_resume_v2_20260927.json"
COLLECTOR = ROOT / "programs/run_rc_unified593_collect_chunk_v3.py"
EXPORTER = ROOT / "programs/run_rc_h593_unified_acquisition_v1.py"
PYTHON = ROOT.parents[2] / ".venv-romav2/bin/python"
INDICES = list(range(128, 593))
PACKS = [INDICES[k::4] for k in range(4)]
FAMILIES = {
    "inside": ("rc_h593_m_inside_v1", "rc_h593_m_inside_authority_v1_20260922.json", "M_VISUAL_ORIGIN_QUERY_PASS"),
    "visual": ("rc_h593_m_visual_origin_v1", "rc_h593_m_visual_origin_authority_v1_20260922.json", "M_VISUAL_ORIGIN_QUERY_PASS"),
    "coordinate": ("rc_h593_roma_coordinate_precision_v2", "rc_h593_roma_coordinate_precision_authority_v2_20260922.json", "ROMA_COORDINATE_QUERY_PASS"),
}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def checked(binding, memo=None):
    path = Path(binding["path"]).resolve()
    key = (str(path), binding["sha256"])
    if memo is None or key not in memo:
        if bind(path)["sha256"] != binding["sha256"]:
            raise ValueError(f"SOURCE_HASH_DRIFT: {path}")
        if memo is not None:
            memo.add(key)
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sources():
    return [Path(__file__).resolve(), COLLECTOR, EXPORTER,
            ROOT / "programs/run_rc_unified128_collect_chunk_v2.py",
            ROOT / "programs/run_rc_unified128_resume_v2.py",
            ROOT / "slurm/rc_unified593_long4_gpu_v1.sbatch",
            ROOT / "slurm/rc_unified593_long4_export_v1.sbatch",
            ROOT / "slurm/rc_unified593_long4_join_v1.sbatch",
            ROOT / "plan/RC_UNIFIED593_LONG4_V1_20260927.md", SCIENCE, RESUME_V2]


def scientific_guard():
    memo = set()
    a = read(SCIENCE)
    if a["status"] != "UNIFIED_GPU_ACQUISITION_AUTHORIZED" or a["queries"] != 593 or a["candidates"] != 128:
        raise ValueError("original scientific authority mismatch")
    for b in [*a["sources"], a["workers"], a["profile"]]:
        checked(b, memo)
    v2 = read(RESUME_V2)
    if v2["status"] != "UNIFIED128_RESUME_V2_AUTHORIZED":
        raise ValueError("original execution repair not qualified")
    for b in [v2["original128_authority"], *v2["sources"]]:
        checked(b, memo)
    old = read(checked(v2["original128_authority"], memo))
    if old["original_authority"] != bind(SCIENCE):
        raise ValueError("old first128 chain has a different scientific source")
    for b in [old["original_authority"], old["pilot_export"], *old["sources"]]:
        checked(b, memo)
    pilot = read(checked(old["pilot_export"], memo))
    if pilot["status"] != "UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS":
        raise ValueError("original exporter qualification missing")
    for b in pilot["validations"].values():
        checked(b, memo)
    workers = read(checked(a["workers"], memo))["records"]
    if len(workers) != 593 or any(w["execution_ordinal"] != i for i, w in enumerate(workers)):
        raise ValueError("original worker index changed")
    return a, workers


def family_records(index, worker):
    """Return validated existing families. A present but bad seal is an error."""
    rows = {}
    for family, (folder, authority_name, status) in FAMILIES.items():
        path = ROOT / "results" / folder / f"query{index:03d}/validation.json"
        if not path.exists():
            continue
        validation = read(path)
        if validation["status"] != status or validation["authority"] != bind(ROOT / "registry" / authority_name):
            raise ValueError(f"invalid scientific validation: {path}")
        if validation.get("label_reads", 0) != 0 or validation.get("model_updates", 0) != 0:
            raise ValueError("acquisition validation contains label reads/model updates")
        payload_path = checked(validation["payload"])
        payload = read(payload_path)
        if payload["query_id"] != worker["query_id"] or payload["execution_ordinal"] != index:
            raise ValueError("scientific validation refers to a different query")
        if payload.get("label_reads", 0) != 0 or payload.get("model_updates", 0) != 0:
            raise ValueError("acquisition payload contains label reads/model updates")
        if len(payload["candidate_physical_rows"]) != 128 or validation.get("candidates", 128) != 128:
            raise ValueError("scientific candidate axis changed")
        rows[family] = {"validation": bind(path), "payload": validation["payload"]}
    return rows


def gpu_ready(index, worker):
    path = CACHE / f"query{index:03d}/ready.json"
    if not path.exists():
        return None
    ready = read(path)
    manifest_path = checked(ready["manifest"])
    m = read(manifest_path)
    if (ready["status"] != "UNIFIED_ALL128_GPU_DATA_READY" or
            m["status"] != "UNIFIED_ALL128_GPU_DATA_READY" or m["authority"] != bind(SCIENCE) or
            m["index"] != index or m["query_id"] != worker["query_id"] or m.get("labels_read", 0) != 0):
        raise ValueError("invalid existing GPU-ready manifest")
    # Original CPU exporter independently checks all capsule/previous-part hashes.
    # Do not reread multi-GB tensor capsules merely to decide whether GPU work exists.
    return {"ready": bind(path), "manifest": ready["manifest"],
            "capsule_hashes_rechecked_by_original_exporter": True}


def guard():
    a = read(AUTH)
    if (a["status"] != "UNIFIED593_LONG4_EXECUTION_AUTHORIZED" or a["indices"] != INDICES or
            a["packs"] != PACKS or a["max_chunks_per_query_per_restart"] != 32 or a["max_job_restarts"] != 3):
        raise ValueError("long4 execution authority drift")
    for b in [*a["sources"], a["prepared_inputs"], a["scientific_authority"], a["resume_v2_authority"]]:
        checked(b)
    scientific_guard()  # code/authority/profile only, not all465 input tensors
    return a


def prepare():
    if AUTH.exists():
        a = guard()
        print(json.dumps({"status": "ALREADY_PREPARED", "authority": str(AUTH), "queries": len(a["indices"])}), flush=True)
        return
    a, workers = scientific_guard()
    memo = set()
    records = []
    for index in INDICES:
        worker = workers[index]
        for kind in ("raw", "roma"):
            for binding in worker[kind].values():
                checked(binding, memo)
            validation = read(checked(worker[kind]["validation"], memo))
            if not str(validation["status"]).endswith("PASS"):
                raise ValueError(f"unqualified {kind} source query{index:03d}")
        op = ROOT / f"results/rc_h593_quality_operator_v1/query{index:03d}/validation.json"
        opv = read(op)
        if opv["status"] != "QUALITY_OPERATOR_QUERY_PASS" or opv.get("label_reads", 0) != 0:
            raise ValueError("quality operator prerequisite not ready")
        payload = read(checked(opv["payload"], memo))
        if payload["query_id"] != worker["query_id"] or payload["execution_ordinal"] != index:
            raise ValueError("quality operator prerequisite query mismatch")
        prior = family_records(index, worker)
        ready = gpu_ready(index, worker)
        records.append({"index": index, "query_id": worker["query_id"], "raw": worker["raw"], "roma": worker["roma"],
                        "operator_validation": bind(op), "operator_payload": opv["payload"],
                        "families_already_validated": prior, "gpu_already_ready": ready})
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = OUT / "prepared_inputs.json"
    write(prepared, {"status": "ALL465_EXISTING_INPUT_BINDINGS_PASS", "indices": INDICES,
                     "records": records, "unique_large_binding_checks": len(memo), "label_reads": 0,
                     "scientific_authority": bind(SCIENCE), "workers": a["workers"]})
    authority = {"status": "UNIFIED593_LONG4_EXECUTION_AUTHORIZED",
                 "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "user_instruction": "Resume existing frozen unified593 collection in four long tasks; reuse all existing caches",
                 "sources": [bind(p) for p in sources()], "scientific_authority": bind(SCIENCE),
                 "resume_v2_authority": bind(RESUME_V2), "prepared_inputs": bind(prepared),
                 "indices": INDICES, "packs": PACKS, "pack_sizes": list(map(len, PACKS)),
                 "excluded_running_first128": list(range(128)), "families": list(FAMILIES),
                 "chunk_seconds": 640, "chunk_timeout_seconds": 850, "export_timeout_seconds": 1800,
                 "max_chunks_per_query_per_restart": 32, "max_chunks": 32,
                 "max_job_restarts": 3, "default_collect_budget_seconds": 169200,
                 "default_export_budget_seconds": 39600, "scientific_changes": 0,
                 "label_reads": 0, "model_updates": 0, "original_validators_unchanged": True}
    tmp = AUTH.with_name(f".{AUTH.name}.{os.getpid()}.tmp")
    write(tmp, authority)
    try:
        os.link(tmp, AUTH)  # fail if another preparer has already frozen authority
    finally:
        tmp.unlink(missing_ok=True)
    guard()
    print(json.dumps({"status": "UNIFIED593_LONG4_PREPARED", "queries": 465,
                      "pack_sizes": list(map(len, PACKS)),
                      "already_complete": sum(len(r["families_already_validated"]) == 3 for r in records),
                      "gpu_already_ready": sum(r["gpu_already_ready"] is not None for r in records)}), flush=True)


def run_child(command, environment, timeout, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return {"returncode": code, "timeout": False, "seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {"returncode": 75, "timeout": True, "seconds": time.monotonic() - started}


def pack_worker(stage, pack, budget):
    started = time.monotonic()
    a = guard()
    if pack not in range(4) or budget <= 0:
        raise ValueError("pack must be0..3 and budget positive")
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not all(c.isdigit() or c == "_" for c in job):
        raise ValueError("collection/export requires a Slurm job ID")
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if not 0 <= restart <= a["max_job_restarts"]:
        raise ValueError("job restart limit exceeded")
    folder = OUT / stage / f"pack{pack}"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "pack.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workers = read(read(SCIENCE)["workers"]["path"])["records"]
    run_binding = bind(AUTH)
    record = {"authority": run_binding, "stage": stage, "pack": pack, "job": job, "restart": restart,
              "indices": PACKS[pack], "done": [], "missing_gpu": [], "budget_seconds": budget}
    receipt = folder / f"job{job}_restart{restart:02d}.json"

    def finish(status, code):
        record.update(status=status, seconds=time.monotonic() - started)
        write(receipt, record)
        write(folder / "progress.json", record)
        print(json.dumps({k: record[k] for k in ("status", "stage", "pack", "job", "restart", "seconds")}
                         | {"done": len(record["done"]), "missing_gpu": len(record["missing_gpu"])}), flush=True)
        return code

    for index in PACKS[pack]:
        worker = workers[index]
        previous = family_records(index, worker)
        if len(previous) == 3:
            record["done"].append({"index": index, "reason": "ALREADY_THREE_FAMILIES_VALIDATED"})
            continue
        if stage == "collect-pack":
            ready = gpu_ready(index, worker)
            counter_path = folder / f"query{index:03d}_attempts.json"
            counter = read(counter_path) if counter_path.exists() else {
                "authority": run_binding, "index": index, "next_attempt": 0, "per_job_restart": {}}
            if counter["authority"] != run_binding or counter["index"] != index:
                raise ValueError("attempt counter binding drift")
            restart_key = f"{job}:{restart}"
            while ready is None:
                if time.monotonic() - started + a["chunk_timeout_seconds"] + 20 > budget:
                    return finish("RESUMABLE_BUDGET", 75)
                if counter["per_job_restart"].get(restart_key, 0) >= a["max_chunks_per_query_per_restart"]:
                    return finish("RESUMABLE_QUERY_CHUNK_LIMIT", 75)
                attempt = counter["next_attempt"]
                counter["next_attempt"] += 1
                counter["per_job_restart"][restart_key] = counter["per_job_restart"].get(restart_key, 0) + 1
                write(counter_path, counter)  # reserve unique attempt before child starts
                env = dict(os.environ, RC_UNIFIED_CHUNK_ATTEMPT=str(attempt))
                log = folder / "logs" / f"query{index:03d}_{job}_restart{restart:02d}_chunk{attempt:03d}.log"
                outcome = run_child([str(PYTHON), str(COLLECTOR), str(index)], env,
                                    a["chunk_timeout_seconds"], log)
                write(folder / "chunks" / f"query{index:03d}_{job}_restart{restart:02d}_chunk{attempt:03d}.json",
                      {"authority": run_binding, "index": index, "job": job, "restart": restart,
                       "attempt": attempt, "log": str(log), **outcome})
                if outcome["timeout"]:
                    return finish("RESUMABLE_CHILD_TIMEOUT", 75)
                if outcome["returncode"] != 0:
                    record["failed_index"] = index
                    record["child_returncode"] = outcome["returncode"]
                    return finish("CHILD_FAILED_CLOSED", 1)
                ready = gpu_ready(index, worker)
            record["done"].append({"index": index, "reason": "GPU_READY_MANIFEST_VALID", **ready})
        else:
            ready = gpu_ready(index, worker)
            if ready is None:
                record["missing_gpu"].append(index)
                continue
            if time.monotonic() - started + a["export_timeout_seconds"] + 20 > budget:
                return finish("RESUMABLE_BUDGET", 75)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            log = folder / "logs" / f"query{index:03d}_{job}_restart{restart:02d}.log"
            outcome = run_child([str(PYTHON), str(EXPORTER), "export", "--index", str(index)],
                                env, a["export_timeout_seconds"], log)
            write(folder / "queries" / f"query{index:03d}_{job}_restart{restart:02d}.json",
                  {"authority": run_binding, "index": index, "job": job, "restart": restart,
                   "log": str(log), **outcome})
            if outcome["timeout"]:
                return finish("RESUMABLE_CHILD_TIMEOUT", 75)
            if outcome["returncode"] != 0:
                record["failed_index"] = index
                record["child_returncode"] = outcome["returncode"]
                return finish("CHILD_FAILED_CLOSED", 1)
            if len(family_records(index, worker)) != 3:
                raise ValueError("export exited zero without three validated scientific families")
            record["done"].append({"index": index, "reason": "CPU_EXPORT_THREE_FAMILIES_PASS"})
        write(folder / "progress.json", record)
        print(json.dumps({"stage": stage, "pack": pack, "index": index, "done": len(record["done"])}), flush=True)
    if record["missing_gpu"]:
        return finish("RESUMABLE_GPU_NOT_READY", 75)
    return finish("PACK_COMPLETE", 0)


def join():
    guard()
    workers = read(read(SCIENCE)["workers"]["path"])["records"]
    records, missing = [], []
    for index in range(593):
        families = family_records(index, workers[index])
        if len(families) != 3:
            missing.append({"index": index, "missing_families": sorted(set(FAMILIES) - set(families))})
        records.append({"index": index, "query_id": workers[index]["query_id"], "families": families})
    result = {"status": "ALL593_THREE_FAMILIES_VALIDATED_PASS" if not missing else "INCOMPLETE_RESUMABLE",
              "authority": bind(AUTH), "queries": 593, "complete_queries": 593 - len(missing),
              "validated_families": sum(len(r["families"]) for r in records),
              "missing": missing, "records": records, "label_reads": 0, "model_updates": 0,
              "scientific_changes": 0, "first128_owner_unchanged": True}
    write(OUT / ("result.json" if not missing else "join_progress.json"), result)
    print(json.dumps({k: result[k] for k in ("status", "queries", "complete_queries", "validated_families")}), flush=True)
    return 0 if not missing else 75


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect-pack", "export-pack", "join"))
    parser.add_argument("--pack", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=169200.)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return 0
    if args.stage == "join":
        return join()
    pack = args.pack if args.pack is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    return pack_worker(args.stage, pack, args.budget_seconds)


if __name__ == "__main__":
    sys.exit(main())
