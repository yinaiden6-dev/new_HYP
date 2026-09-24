#!/usr/bin/env python3
"""Reproduce the opened TRAIN16 V3 condition-scale audit without a backbone.

Reads only saved small adapter checkpoints, traces, steps and endpoint JSONs.
The preactivation/residual measurements describe one traced TRAIN image; the
endpoint intervention measurements separately cover all sixteen TRAIN images.
No model, image, cache, reference-token or held-label inputs are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/rc_internal_m_learned_use_v3"
DEFAULT_OUTPUT = ROOT / "results/rc_internal_m_v3_fit_diagnostics/condition_scale_audit.json"
SOURCES: dict[str, dict] = {}


def binding(path: Path) -> dict:
    path = path.resolve()
    key = str(path)
    if key not in SOURCES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        SOURCES[key] = {"path": key, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return SOURCES[key]


def read_json(path: Path) -> dict:
    binding(path)
    return json.loads(path.read_text())


def read_checkpoint(path: Path, torch):
    binding(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def rms(values) -> float:
    return math.sqrt(statistics.mean(float(value) ** 2 for value in values))


def stats(values) -> dict:
    values = list(map(float, values))
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values), "rms": rms(values)}


def read_endpoints(arm: str, intervention: str) -> dict[str, dict]:
    directory = V3 / arm / "endpoints/0128" / intervention
    records = {}
    for path in sorted(directory.glob("*.json")):
        if ".partial." in path.name:
            continue
        record = read_json(path)
        assert record["step"] == 128 and record["arm"] == arm
        assert record["intervention"] == intervention
        assert record["direct_M_in_head"] is False and record["held_label_reads"] == 0
        assert len(record["L"]) == len(record["M"]) == len(record["candidate_ids"]) == 128
        records[record["query_id"]] = record
    assert len(records) == 16
    return records


def endpoint_summary(records: dict[str, dict]) -> dict:
    decisions = [record["decision"] for record in records.values()]
    theta = decisions[0]["theta"]
    assert all(record["theta"] == theta for record in decisions)
    return {"queries": len(records), "correct": sum(record["correct"] for record in decisions),
            "switches": sum(record["switched"] for record in decisions), "head": theta,
            "mean_cost1": statistics.mean(record["cost1"] for record in decisions),
            "target_margin": stats(record["target_vs_strongest_wrong_margin"] for record in decisions),
            "wrong_target_margin": stats(record["target_vs_strongest_wrong_margin"]
                                         for record in decisions if not record["correct"])}


def intervention_comparison(native: dict, alternative: dict) -> dict:
    assert native.keys() == alternative.keys()
    content_deltas, logit_deltas, margin_deltas, details = [], [], [], []
    for query_id, record in native.items():
        other = alternative[query_id]
        assert record["candidate_ids"] == other["candidate_ids"]
        assert record["snapshot"] == other["snapshot"]
        assert record["decision"]["theta"] == other["decision"]["theta"]
        content_deltas.extend(x - y for x, y in zip(record["L"], other["L"]))
        logit_deltas.extend(x - y for x, y in zip(record["decision"]["logits"], other["decision"]["logits"]))
        margin_delta = record["decision"]["target_vs_strongest_wrong_margin"] - other["decision"]["target_vs_strongest_wrong_margin"]
        margin_deltas.append(margin_delta)
        details.append({"query_id": query_id, "native_minus_intervention_target_margin": margin_delta,
                        "prediction_changed": record["decision"]["prediction_position"] != other["decision"]["prediction_position"]})
    return {"difference_direction": "native real M minus intervention at identical saved adapter and head",
            "content_delta": stats(content_deltas), "logit_delta": stats(logit_deltas),
            "target_margin_delta": stats(margin_deltas),
            "queries_with_larger_native_margin": sum(value > 0 for value in margin_deltas),
            "prediction_changes": sum(record["prediction_changed"] for record in details), "queries": details}


def step_audit(arm: str) -> dict:
    steps = [read_json(V3 / arm / "steps" / f"{step:04d}.json") for step in range(1, 129)]
    assert [record["step"] for record in steps] == list(range(1, 129))
    assert all(record["arm"] == arm and record["backbone_trainable"] == 0
               and record["direct_M_in_head"] is False for record in steps)
    order = [record["query_id"] for record in steps[:16]]
    assert len(set(order)) == 16
    assert all(record["query_id"] == order[i % 16] for i, record in enumerate(steps))
    # Every recorded decision is HOLD, so its correctness is RAW correctness.
    assert all(not record["decision_before_update"]["switched"] for record in steps)
    correctness = [record["decision_before_update"]["correct"] for record in steps[:16]]
    return {"steps": len(steps), "complete_ordered_passes": len(steps) // len(order),
            "query_order": order, "raw_correct_in_order": correctness,
            "first_eight_correct_next_eight_incorrect": correctness == [True] * 8 + [False] * 8,
            "head_initial": steps[0]["head_before"], "head_after_first_eight": steps[7]["head_after"],
            "head_after_first_pass": steps[15]["head_after"], "head_final": steps[-1]["head_after"],
            "adapter_gradient_norm": stats(record["adapter_gradient_norm"] for record in steps),
            "head_gradient_norm": stats(record["head_gradient_norm"] for record in steps),
            "adapter_norm_above_one_steps": sum(record["adapter_gradient_norm"] > 1 for record in steps),
            "head_norm_above_one_steps": sum(record["head_gradient_norm"] > 1 for record in steps),
            "gradient_forward_error_max": max(record["gradient_forward_error"] for record in steps),
            "all_adapter_updates_nonzero": all(record["adapter_parameter_change_max"] > 0 for record in steps),
            "all_head_updates_nonzero": all(record["head_parameter_change_max"] > 0 for record in steps),
            "decision_switches_before_update": sum(record["decision_before_update"]["switched"] for record in steps),
            "pass_average_online_loss": [statistics.mean(record["loss_before_update"] for record in steps[i:i + 16])
                                         for i in range(0, 128, 16)]}


def scale_audit(native: dict, torch) -> dict:
    from torch.nn import functional as functional

    result = {"scope": "PRE_REAL saved traces at steps16,64,128; a single TRAIN image, not all TRAIN16",
              "measurements": [], "parameter_norms": {}}
    for arm in ("PRE_REAL", "PRE_CONSTANT"):
        initial = read_checkpoint(V3 / arm / "initial.pt", torch)["adapter"]
        final = read_checkpoint(V3 / arm / "snapshots/0128.pt", torch)["adapter"]
        result["parameter_norms"][arm] = {
            key: {"initial": float(initial[key].norm()), "final": float(final[key].norm()),
                  "change": float((final[key] - initial[key]).norm())}
            for key in ("down.weight", "down.bias", "up.weight", "up.bias")}
        result["parameter_norms"][arm]["down_mass_column"] = {
            "initial": float(initial["down.weight"][:, -1].norm()),
            "final": float(final["down.weight"][:, -1].norm()),
            "change": float((final["down.weight"][:, -1] - initial["down.weight"][:, -1]).norm())}
    query_ids = set()
    for step in (16, 64, 128):
        trace_path = V3 / "PRE_REAL/traces" / f"{step:04d}.pt"
        checkpoint_path = V3 / "PRE_REAL/snapshots" / f"{step:04d}.pt"
        trace = read_checkpoint(trace_path, torch)
        checkpoint = read_checkpoint(checkpoint_path, torch)
        assert checkpoint["step"] == step and checkpoint["arm"] == "PRE_REAL"
        state = checkpoint["adapter"]
        query_id = trace["query_id"]
        query_ids.add(query_id)
        source = trace["source"]
        assert source.ndim == 2 and source.shape[1] == 3584
        normalized = functional.layer_norm(source.float(), (source.shape[1],))
        down = state["down.weight"]
        content = functional.linear(normalized, down[:, :-1], state["down.bias"])
        masses = torch.tensor(native[query_id]["M"], dtype=torch.float32)
        condition = (masses.clamp_min(state["mass_epsilon"]).log() - state["mass_log_mean"]) / state["mass_log_std"]
        mass_branch = condition[:, None] * down[:, -1][None, :]
        chosen_condition = (torch.tensor(float(trace["M"])).clamp_min(state["mass_epsilon"]).log()
                            - state["mass_log_mean"]) / state["mass_log_std"]
        assert abs(float(trace["M"]) - native[query_id]["M"][trace["candidate_position"]]) < 1e-12
        residual_zero = functional.linear(functional.gelu(content), state["up.weight"], state["up.bias"]) * state["residual_scale"]
        residual_real = functional.linear(functional.gelu(content + chosen_condition * down[:, -1]),
                                          state["up.weight"], state["up.bias"]) * state["residual_scale"]
        changed_zero = (source + residual_zero.to(source.dtype)).float()
        changed_real = (source + residual_real.to(source.dtype)).float()
        tensor_rms = lambda tensor: float(tensor.square().mean().sqrt())
        result["measurements"].append({
            "step": step, "query_id": query_id, "patches": len(source), "hidden_size": source.shape[1],
            "source_dtype": str(source.dtype), "trace": binding(trace_path), "checkpoint": binding(checkpoint_path),
            "source_rms": tensor_rms(source.float()), "content_preactivation_including_bias_rms": tensor_rms(content),
            "M_preactivation_over_original_C128_rms": tensor_rms(mass_branch),
            "M_to_content_preactivation_rms_ratio": tensor_rms(mass_branch) / tensor_rms(content),
            "C128_standardized_mass_min": float(condition.min()), "C128_standardized_mass_max": float(condition.max()),
            "mass_column_norm": float(down[:, -1].norm()),
            "trace_candidate_position": trace["candidate_position"], "trace_candidate_mass": trace["M"],
            "trace_candidate_standardized_mass": float(chosen_condition),
            "zero_condition_residual_rms": tensor_rms(residual_zero),
            "trace_real_minus_zero_residual_rms": tensor_rms(residual_real - residual_zero),
            "trace_M_residual_to_zero_condition_residual_rms_ratio": tensor_rms(residual_real - residual_zero) / tensor_rms(residual_zero),
            "trace_bf16_real_minus_zero_injected_tokens_rms": tensor_rms(changed_real - changed_zero),
            "trace_bf16_real_minus_zero_changed_coordinate_fraction": float((changed_real != changed_zero).float().mean()),
        })
    assert len(query_ids) == 1
    result["unique_traced_query_count"] = len(query_ids)
    result["traced_query_ids"] = sorted(query_ids)
    result["formulas"] = {
        "content": "F.linear(F.layer_norm(source.float(), [3584]), down.weight[:,:-1], down.bias)",
        "condition": "(log(clamp(M,epsilon)) - saved_log_mean) / saved_log_std",
        "M_branch": "condition[C128,None] * down.weight[:,-1][None,:]",
        "residual": "saved_residual_scale * F.linear(gelu(content + condition * mass_column), up.weight, up.bias)",
        "precision": "FP32 adapter arithmetic; residual cast to saved source dtype before addition",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    binding(Path(__file__))
    endpoints = {arm: read_endpoints(arm, "native") for arm in ("PRE_REAL", "PRE_CONSTANT")}
    interventions = {name: read_endpoints("PRE_REAL", name) for name in ("constant", "shuffled")}
    output = {
        "status": "V3_CONDITION_SCALE_READONLY_AUDIT_PASS", "evidence_level": "Opened TRAIN16 fit diagnostic only",
        "runtime": {"torch_version": str(torch.__version__), "threads": 1, "device": "cpu",
                    "backbone_forwards": 0, "image_reads": 0, "encoder_cache_reads": 0,
                    "reference_token_reads": 0, "held_label_reads": 0},
        "endpoint_native": {arm: endpoint_summary(value) for arm, value in endpoints.items()},
        "endpoint_real_interventions": {name: endpoint_summary(value) for name, value in interventions.items()},
        "paired_endpoint_interventions": {name: intervention_comparison(endpoints["PRE_REAL"], value)
                                           for name, value in interventions.items()},
        "steps": {arm: step_audit(arm) for arm in endpoints},
        "condition_scale": scale_audit(endpoints["PRE_REAL"], torch),
        "interpretation": {
            "established": [
                "Saved traces show a small M preactivation relative to content preactivation for one TRAIN image.",
                "True-M versus constant/shuffled endpoint interventions change scores but do not change any prediction.",
                "Saved step records report nonzero adapter/head updates and zero no-grad/VJP replay score discrepancy.",
                "Training repeatedly presents eight RAW-correct queries followed by eight RAW-incorrect queries.",
            ],
            "not_established": [
                "The traced image establishes neither all-TRAIN activation statistics nor a unique failure cause.",
                "Nonzero intervention effects exclude complete M removal but do not establish useful M use.",
                "This audit does not identify how much the head schedule, condition scale, or training budget causes failure.",
                "A proposed fixed condition gain is an untested controlled intervention, not a demonstrated repair.",
            ],
            "proposed_minimal_intervention": {"condition_gain": math.sqrt(3584),
                "definition": "Multiply the existing standardized M input by fixed sqrt(hidden_size), preserving architecture and output head.",
                "comparison": "Match initialization, parameters, training schedule and budget across real, constant and fixed within-query shuffled M.",
                "scope": "TRAIN16 fit and causal-use diagnostic; no generalization claim or requirement to exceed the external head."},
        },
    }
    output["sources"] = sorted(SOURCES.values(), key=lambda item: item["path"])
    output["source_count"] = len(output["sources"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": output["status"], "output": str(args.output.resolve()),
                      "source_count": output["source_count"], "traced_queries": output["condition_scale"]["traced_query_ids"]}))


if __name__ == "__main__":
    main()
