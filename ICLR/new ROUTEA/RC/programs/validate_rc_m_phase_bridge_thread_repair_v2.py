#!/usr/bin/env python3
"""Reproduce the phase-bridge CPU thread mismatch without modifying inputs.

Writes only results/rc_m_phase_bridge_thread_repair_v2. No training, labels,
RoMa/LLM forwards, protocol edits, scheduler changes, or relaxed tolerances.
The deliberately contrasting torch thread settings are 1, 2, 4, and 8;
the original decomposition used four threads. All tested numeric values are
recomputed, not copied from the prior interactive investigation.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F

import run_rc_m_phase_bridge_v1 as bridge
from analyze_rc_postllm_signal_decomposition_v2 import (
    bind, checked, read, write, tensor_sha, projected,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_m_phase_bridge_thread_repair_v2"
OLD = ROOT / "results/rc_postllm_h593_decomposition_v1"


def old_candidate(item, position):
    path = OLD / "queries" / item["query_id"] / "candidates" / f"{position:04d}.json"
    record = read(path)
    payload = checked(record["patch_evidence"])
    with np.load(payload, allow_pickle=False) as data:
        j = data["token_arms"].tolist().index("NATIVE")
        patches = data["maxsim_by_patch"][j].copy()
        locations = data["best_reference_token"][j].copy()
    return record, patches, locations, bind(path)


def stage_compare(left, right):
    return {
        "exact_equal": bool(torch.equal(left, right)),
        "different_elements": int((left != right).sum()),
        "max_abs_error": float((left.double() - right.double()).abs().max()),
        "left_sha256": tensor_sha(left),
        "right_sha256": tensor_sha(right),
    }


@torch.no_grad()
def run():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    source_protocol = read(bridge.OUT / "protocol.json")
    for source in source_protocol["code_sources"]:
        checked(source)
    sources = {
        "validation_program": bind(Path(__file__)),
        "bridge_protocol": bind(bridge.OUT / "protocol.json"),
        "bridge_inputs": source_protocol["inputs"],
        "original_decomposition_protocol": bind(OLD / "protocol.json"),
        "original_decomposition_program": bind(ROOT / "programs/analyze_rc_postllm_h593_decomposition_v1.py"),
        "original_dispatch_program": bind(ROOT / "programs/dispatch_rc_postllm_h593_decomposition_v1.py"),
        "bridge_dependencies": source_protocol["code_sources"],
        "endpoints": source_protocol["endpoints"],
        "authority": source_protocol["authority"],
        "manifest": source_protocol["manifest"],
    }
    runtime = {
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build": torch.__config__.show(),
        "thread_environment": {k: os.environ[k] for k in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "interop_threads": torch.get_num_interop_threads(),
    }
    write(OUT / "sources.json", sources)
    write(OUT / "runtime.json", runtime)
    row_summaries = []
    all_candidates = []
    worst = None
    for index in (0, 1, 2):
        torch.set_num_threads(4)
        ctx = bridge.load_context(index)
        protocol, protocol_binding, item, adapter, head, cache, refs, projection = ctx
        hidden_binding = item["post_row"].get("hidden_validation")
        hidden_seal_path = (checked(hidden_binding) if hidden_binding else
                            Path(item["post_row"]["hidden_cache_dir"]) / "validation.json")
        hidden_seal = read(hidden_seal_path)
        checked(hidden_seal["payload"])
        hidden = cache["hidden"][cache["image_mask"]]
        adapter.conditioning = "real"
        if index == 0:
            position = 64
            prior, prior_patches, prior_locations, prior_binding = old_candidate(item, position)
            mass = item["masses"]["ORIGINAL_HR1"][position]
            assert mass == prior["M"]
            thread_rows, stages = [], {}
            for threads in (1, 2, 4, 8):
                torch.set_num_threads(threads)
                actual = adapter(hidden, mass)
                full = cache["hidden"].clone()
                full[cache["image_mask"]] = actual
                raw = projection(full)
                norm = raw.norm(dim=-1, keepdim=True)
                tokens_full = raw / norm
                tokens_full = tokens_full * cache["batch"]["attention_mask"].unsqueeze(-1)
                tokens = tokens_full[cache["image_mask"]]
                unit = F.normalize(tokens.double(), dim=-1)
                best, locations = (unit @ refs[position].T).max(1)
                stages[threads] = {"adapter": actual.clone(), "raw_projection": raw.clone(),
                                   "projection_norm": norm.clone(), "normalized_image_tokens": tokens.clone()}
                thread_rows.append({
                    "threads": threads, "adapter_hidden_sha256": tensor_sha(actual),
                    "adapter_matches_original_sha": tensor_sha(actual) == prior["residual_decomposition"]["native_hidden_sha256"],
                    "max_patch_error": float(np.max(np.abs(best.numpy() - prior_patches))),
                    "L_abs_error": abs(float(best.mean()) - prior["scores"]["NATIVE"]),
                    "argmax_changes": int(np.sum(locations.numpy() != prior_locations)),
                    "different_patch_count": int(np.sum(best.numpy() != prior_patches)),
                    "stage_sha256": {name: tensor_sha(value) for name, value in stages[threads].items()},
                })
            worst = {
                "query_id": item["query_id"], "panel_index": 0, "candidate_position": position,
                "candidate_id": prior["candidate_id"], "mass": mass,
                "mass_exact_equal_to_original": True, "old_candidate": prior_binding,
                "old_patch_evidence": prior["patch_evidence"],
                "thread_results": thread_rows,
                "two_vs_four_threads": {name: stage_compare(stages[2][name], stages[4][name])
                                        for name in stages[2]},
            }
            write(OUT / "worst_candidate_thread_sweep.json", worst)
            assert all(r["adapter_matches_original_sha"] for r in thread_rows)
            # A hardware stack without a 2-thread discrepancy may legitimately
            # fail to reproduce that diagnostic; never invent the difference.
            assert thread_rows[2]["max_patch_error"] == 0.0
            assert thread_rows[2]["L_abs_error"] == 0.0
            torch.set_num_threads(4)

        records = []
        for position in range(128):
            prior, prior_patches, prior_locations, prior_binding = old_candidate(item, position)
            mass = item["masses"]["ORIGINAL_HR1"][position]
            assert mass == prior["M"]
            actual = adapter(hidden, mass)
            tokens = projected(projection, cache, actual)
            best, locations = (F.normalize(tokens.double(), dim=-1) @ refs[position].T).max(1)
            record = {
                "panel_index": index, "query_id": item["query_id"], "candidate_position": position,
                "candidate_id": prior["candidate_id"], "mass_equal": True,
                "adapter_hidden_sha256": tensor_sha(actual),
                "old_adapter_hidden_sha256": prior["residual_decomposition"]["native_hidden_sha256"],
                "max_patch_error": float(np.max(np.abs(best.numpy() - prior_patches))),
                "L_abs_error": abs(float(best.mean()) - prior["scores"]["NATIVE"]),
                "argmax_changes": int(np.sum(locations.numpy() != prior_locations)),
                "old_candidate": prior_binding, "old_patch_evidence": prior["patch_evidence"],
                "reference_tokens": item["post_row"]["reference_tokens"][position],
            }
            assert record["adapter_hidden_sha256"] == record["old_adapter_hidden_sha256"]
            assert record["max_patch_error"] == record["L_abs_error"] == 0.0
            assert record["argmax_changes"] == 0
            records.append(record)
        write(OUT / f"row{index:02d}_candidates.json", records)
        row_summaries.append({
            "index": index, "query_id": item["query_id"], "candidates": 128, "threads": 4,
            "all_adapter_hashes_exact": True, "max_patch_error": 0.0, "max_L_error": 0.0,
            "argmax_changes": 0, "candidate_evidence": bind(OUT / f"row{index:02d}_candidates.json"),
            "hidden_validation": bind(hidden_seal_path),
            "hidden_payload": hidden_seal["payload"],
            "projection_source_report": projection.report,
        })
        all_candidates.extend(records)
        print(json.dumps({k: row_summaries[-1][k] for k in
                          ("index", "query_id", "candidates", "max_patch_error", "max_L_error")}), flush=True)
    replicated = (worst["two_vs_four_threads"]["raw_projection"]["different_elements"] > 0
                  and worst["two_vs_four_threads"]["adapter"]["exact_equal"]
                  and worst["two_vs_four_threads"]["projection_norm"]["exact_equal"])
    result = {
        "status": "THREAD4_EXACT_ANCHOR_REPAIR_PASS",
        "diagnostic_thread2_projection_difference_reproduced": replicated,
        "original_bridge_anchor_tolerance_unchanged": 2e-10,
        "four_thread_actual_max_patch_error": 0.0, "four_thread_actual_max_L_error": 0.0,
        "queries": 3, "candidates": len(all_candidates), "rows": row_summaries,
        "worst_candidate_sweep": bind(OUT / "worst_candidate_thread_sweep.json"),
        "sources": bind(OUT / "sources.json"), "runtime": bind(OUT / "runtime.json"),
        "elapsed_seconds": time.monotonic() - start,
        "training": False, "gpu": False, "original_files_modified": False,
        "scientific_scope": "Numerical replay repair only; no new mechanism or retrieval claim.",
        "repair": "Use original four-thread computation in an isolated new bridge version; recompute all worlds. Do not mix two-thread candidate caches or loosen tolerance.",
    }
    write(OUT / "validation.json", result)
    print(json.dumps({k: result[k] for k in ("status", "queries", "candidates",
                                            "diagnostic_thread2_projection_difference_reproduced", "elapsed_seconds")}))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        write(OUT / "validation.json", {"status": "THREAD_REPAIR_VALIDATION_FAILED",
                                        "error_type": type(error).__name__, "error": str(error)})
        raise
