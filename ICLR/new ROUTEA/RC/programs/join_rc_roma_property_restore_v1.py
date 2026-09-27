#!/usr/bin/env python3
"""Audit the 18-arm saved-weight fixed-pair factorial before computing effects.

Only CPU arithmetic and file validation run here. Every scientific contrast
uses the newly computed common CPU backend; sealed GPU values are numerical
references, never replacement cells. No POST or external correction head is
evaluated by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from rc_roma_property_restore_v1 import ARMS, PHASE_ARMS, PANEL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_m_property_restoration_factorial_v1"
OLD_ANALYSIS = ROOT / "results/rc_mass_discrimination_source_v1/result.json"
ARITHMETIC_LIMIT = 2e-10
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_DRAWS = 5000
VERIFIED = {}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def checked(source):
    path = Path(source["path"])
    stat = path.stat()
    stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    key = (str(path.resolve()), source["sha256"])
    if key not in VERIFIED:
        assert bind(path) == source, ("SOURCE_SHA_MISMATCH", source["path"])
        VERIFIED[key] = stamp
    assert VERIFIED[key] == stamp, ("SOURCE_CHANGED_DURING_AUDIT", source["path"])
    return path


def check_bindings(value):
    """Verify source trees without interpreting any contained labels."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            checked(value)
        else:
            for child in value.values():
                check_bindings(child)
    elif isinstance(value, list):
        for child in value:
            check_bindings(child)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def independently_computed_mass(sides):
    assert len(sides) == 2
    means = []
    counts = []
    for side in sides:
        weight = side["weights"]
        assert torch.is_tensor(weight) and weight.device.type == "cpu"
        assert weight.dtype == torch.float64 and weight.numel() > 0 and not weight.requires_grad
        values = weight.reshape(-1).numpy()
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        # Deliberately independent of torch/NumPy parallel reduction kernels.
        means.append(math.fsum(map(float, values)) / len(values))
        counts.append(len(values))
    return math.sqrt(means[0] * means[1]), means, counts


def audit_query(index, protocol, protocol_binding):
    folder = OUT / f"query{index:03d}"
    source_path = folder / "validation.json"
    query = read(source_path)
    assert query["status"] == "PROPERTY_RESTORE_CPU_QUERY_PASS"
    assert query["protocol"] == protocol_binding and query["index"] == index
    assert query["arms"] == list(ARMS) and query["all_arms_same_cpu_backend"] is True
    backend = read(checked(query["backend_binding"]))
    assert {k: v for k, v in backend.items() if k != "cpu_capability"} == protocol["runtime"]
    workers = {r["index"]: r for r in protocol["workers"]}
    expected = workers[index]
    historical_axis = read(checked(expected["gpu_validation"]))
    assert query["source_validation"] == expected["gpu_validation"]
    assert query["query_id"] == expected["query_id"]
    axis = query["candidate_physical_rows"]
    assert axis == expected["candidate_physical_rows"] and len(axis) == len(set(axis)) == 128
    assert historical_axis["candidate_physical_rows"] == axis
    positions = query["positions"]
    assert positions == expected["positions"]
    assert len(positions) == len(set(positions)) and all(0 <= p < 128 for p in positions)
    assert len(query["pairs"]) == len(positions)
    audit_path = OUT / "independent_audits" / f"query{index:03d}.json"
    if audit_path.exists():
        prior = read(audit_path)
        assert prior["protocol"] == protocol_binding and prior["index"] == index
        assert prior["source_query"] == bind(source_path) and prior["source_pairs"] == query["pairs"]
        assert prior["audit_program"] == bind(Path(__file__))
        for source in prior["source_pairs"]:
            checked(source)
        for record in prior["mass_audits"]:
            checked(record["payload"])
        return prior, bind(audit_path)
    masses = {arm: {} for arm in ARMS}
    mass_audits = []
    sources = []
    maximum = 0.0
    for position, pair_binding in zip(positions, query["pairs"]):
        pair = read(checked(pair_binding))
        assert pair["status"] == "PROPERTY_RESTORE_CPU_PAIR_PASS"
        assert pair["protocol"] == protocol_binding and pair["index"] == index
        assert pair["query_id"] == query["query_id"] and pair["position"] == position
        assert pair["physical_row"] == axis[position]
        assert pair["backend_binding"] == query["backend_binding"]
        assert [r["arm"] for r in pair["arms"]] == list(ARMS)
        check_bindings(pair.get("sources", {}))
        for record in pair["arms"]:
            arm = record["arm"]
            payload = torch.load(checked(record["payload"]), map_location="cpu", weights_only=True)
            for key, value in (("protocol", protocol_binding), ("index", index),
                               ("query_id", query["query_id"]), ("position", position),
                               ("physical_row", axis[position]), ("arm", arm)):
                assert payload[key] == value, ("PAYLOAD_AXIS_MISMATCH", index, position, arm, key)
            check_bindings(payload.get("sources", {}))
            assert payload["backend_binding"] == query["backend_binding"]
            assert payload["source_pair"] == historical_axis["pairs"][position]
            assert payload["cpu_only_scientific_arm"] is True
            for key in ("source_pair", "source_capture", "backend_binding"):
                if key in payload:
                    check_bindings(payload[key])
            computed, means, counts = independently_computed_mass(payload["sides"])
            errors = [abs(computed - float(payload["M"])), abs(computed - float(record["M"]))]
            error = max(errors)
            assert error < ARITHMETIC_LIMIT, ("MASS_ARITHMETIC", index, position, arm, error)
            assert computed > 0.0, ("LOG_M_UNDEFINED", index, position, arm)
            maximum = max(maximum, error)
            masses[arm][str(position)] = computed
            mass_audits.append({"position": position, "physical_row": axis[position], "arm": arm,
                                "M_independent": computed, "M_payload": float(payload["M"]),
                                "M_receipt": float(record["M"]), "mean_u": means[0], "mean_v": means[1],
                                "token_counts": counts, "max_error": error, "payload": record["payload"]})
        sources.append(pair_binding)
    assert len(mass_audits) == len(positions) * 18
    receipt = {"status": "PROPERTY_FACTORIAL_QUERY_MASS_AUDIT_PASS", "protocol": protocol_binding,
               "index": index, "query_id": query["query_id"], "axis": axis, "positions": positions,
               "arms": list(ARMS), "mass_by_arm": masses, "max_arithmetic_error": maximum,
               "source_query": bind(source_path), "source_pairs": sources,
               "backend_binding": query["backend_binding"], "backend": backend,
               "mass_audits": mass_audits, "no_new_counterpart_selection": True,
               "audit_program": bind(Path(__file__))}
    write(audit_path, receipt)
    return receipt, bind(audit_path)


def group_summary(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["component"], []).append(float(record["value"]))
    components = sorted(grouped)
    values = np.asarray([math.fsum(grouped[k]) / len(grouped[k]) for k in components], np.float64)
    if not len(values):
        return {"groups": 0, "queries": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {"groups": len(values), "queries": len(records), "group_equal_mean": float(values.mean()),
            "exploratory_bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
            "positive_groups": int((values > 0).sum()), "negative_groups": int((values < 0).sum()),
            "zero_groups": int((values == 0).sum()), "group_values": dict(zip(components, values.tolist())),
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "multiplicity_adjusted": False}


def factorial_contrasts(cells):
    a0g, a0l, a1g, a1l = (float(cells[k]) for k in ("A0G", "A0L", "A1G", "A1L"))
    return {
        "phase_restoration_A1G_minus_A1L": a1g - a1l,
        "amplitude_restoration_A0L_minus_A1L": a0l - a1l,
        "interaction_original_minus_permuted_phase_effect": (a0g - a0l) - (a1g - a1l),
        "phase_control_A0G_minus_A0L": a0g - a0l,
        "amplitude_control_A0G_minus_A1G": a0g - a1g,
    }


def analyze_query(audit, historical):
    assert audit["index"] == historical["index"] and audit["query_id"] == historical["query_id"]
    assert audit["axis"] == historical["axis"]
    masses = audit["mass_by_arm"]
    positions = audit["positions"]
    expected_positions = ({int(historical["target_position"]),
                           int(historical["arm_metrics"]["NATIVE"]["strongest_position"])}
                          if historical["target_present"] else set())
    assert set(positions) == expected_positions, "PREDECLARED_FIXED_PAIRS_ONLY"
    shared = [arm for arm in ARMS if arm in historical["mass_by_arm"]]
    drift = {arm: {str(pos): masses[arm][str(pos)] - float(historical["mass_by_arm"][arm][pos])
                   for pos in positions} for arm in shared}
    row = {"index": audit["index"], "query_id": audit["query_id"], "component": historical["component"],
           "axis": audit["axis"], "target_present": historical["target_present"],
           "candidate_positions": positions,
           "mass_by_arm": audit["mass_by_arm"], "CPU_minus_historical_GPU_M_by_arm": drift,
           "arithmetic_max_error": audit["max_arithmetic_error"]}
    if not historical["target_present"]:
        assert positions == []
        row["margin_analysis"] = None
        row["exclusion_reason"] = "Target absent from unchanged natural C128; original group retained, no new pair computed and no identity margin inferred."
        return row
    target = int(historical["target_position"])
    wrong = int(historical["arm_metrics"]["NATIVE"]["strongest_position"])
    assert 0 <= target < 128 and 0 <= wrong < 128 and target != wrong
    assert set(positions) == {target, wrong}, "PREDECLARED_FIXED_PAIRS_ONLY"
    metric_by_arm = {
        arm: {"fixed_target_wrong_logM_gap": math.log(float(v[str(target)])) - math.log(float(v[str(wrong)])),
              "target_logM": math.log(float(v[str(target)])), "fixed_wrong_logM": math.log(float(v[str(wrong)]))}
        for arm, v in masses.items()}
    metric_results = {}
    for metric in ("fixed_target_wrong_logM_gap", "target_logM", "fixed_wrong_logM"):
        cells = {}
        per_axis_sign = []
        for amplitude, prefix in (("A0", ""), ("A1", "AMP_PERMUTE__")):
            for scope, suffix in (("GLOBAL", "G"), ("LOCAL", "L")):
                arms = [prefix + f"{scope}_{axis}_{sign}" for axis in ("X", "Y") for sign in ("PLUS", "MINUS")]
                cells[amplitude + suffix] = math.fsum(metric_by_arm[a][metric] for a in arms) / 4
        for axis in ("X", "Y"):
            for sign in ("PLUS", "MINUS"):
                one = {a + short: metric_by_arm[prefix + f"{scope}_{axis}_{sign}"][metric]
                       for a, prefix in (("A0", ""), ("A1", "AMP_PERMUTE__"))
                       for scope, short in (("GLOBAL", "G"), ("LOCAL", "L"))}
                per_axis_sign.append({"axis": axis, "sign": sign, "cells": one,
                                      "contrasts": factorial_contrasts(one)})
        contrasts = factorial_contrasts(cells)
        for name, value in contrasts.items():
            assert abs(value - math.fsum(r["contrasts"][name] for r in per_axis_sign) / 4) < ARITHMETIC_LIMIT
        metric_results[metric] = {"cells": cells, "contrasts": contrasts, "per_axis_sign": per_axis_sign,
                                  "native": metric_by_arm["NATIVE"][metric],
                                  "amplitude_permute_only": metric_by_arm["AMP_PERMUTE"][metric]}
    row["margin_analysis"] = {"target_position": target, "fixed_wrong_position": wrong,
                              "competitor_source": "historical arm_metrics.NATIVE.strongest_position; never reselected",
                              "metric_by_arm": metric_by_arm, "metrics": metric_results}
    return row


def join(budget):
    started = time.monotonic()
    protocol_path = OUT / "protocol.json"
    protocol = read(protocol_path)
    pb = bind(protocol_path)
    assert protocol["arms"] == list(ARMS) and protocol["indices"] == PANEL
    assert protocol["evaluation_scope"] == "PRIMARY_FIXED_PAIRS"
    assert len(protocol["workers"]) == 8
    assert sum(len(w["positions"]) for w in protocol["workers"]) == 14
    check_bindings(protocol.get("code_sources", []))
    check_bindings(protocol.get("sources", {}))
    historical_binding = protocol["sources"]["historical_analysis"]
    assert Path(historical_binding["path"]).resolve() == OLD_ANALYSIS.resolve()
    missing = [i for i in PANEL if not (OUT / f"query{i:03d}" / "validation.json").exists()]
    if missing:
        print(json.dumps({"status": "WAITING_FOR_ALL_EIGHT_QUERIES", "missing": missing}), flush=True)
        return 75
    audits, receipts = [], []
    for index in PANEL:
        if time.monotonic() - started >= budget - 20:
            print(json.dumps({"status": "PARTIAL_INDEPENDENT_AUDIT", "complete": len(audits)}), flush=True)
            return 75
        audit, receipt = audit_query(index, protocol, pb)
        audits.append(audit)
        receipts.append(receipt)
        print(json.dumps({"independently_audited_index": index, "candidate_arms": len(audit["positions"]) * 18}), flush=True)
    assert sum(len(a["mass_audits"]) for a in audits) == 14 * 18
    assert all(a["backend"] == audits[0]["backend"] for a in audits)
    seal_path = OUT / "independent_prelabel_seal.json"
    write(seal_path, {"status": "ALL_SAVED_WEIGHT_MASSES_VALIDATED", "protocol": pb,
                      "audits": receipts, "queries": 8, "candidate_arms": 252,
                      "fixed_pairs_predeclared_in_protocol": True,
                      "outcome_metrics_computed": False})
    # Labels selected the protocol's frozen pairs before any new output; reopen
    # their original source only after all mass arithmetic passes, never select
    # a new wrong counterpart from any of the intervened outputs.
    old = read(checked(historical_binding))
    assert old["endpoint"] == "COARSE" and old["queries"] == 8
    old_rows = {r["index"]: r for r in old["phase_rows"]}
    assert set(old_rows) == set(PANEL)
    rows = [analyze_query(audit, old_rows[audit["index"]]) for audit in audits]
    eligible = [r for r in rows if r["target_present"]]
    assert len(eligible) == 7
    summaries = {}
    for metric in eligible[0]["margin_analysis"]["metrics"]:
        summaries[metric] = {}
        for name in eligible[0]["margin_analysis"]["metrics"][metric]["contrasts"]:
            summaries[metric][name] = group_summary([
                {"component": r["component"], "value": r["margin_analysis"]["metrics"][metric]["contrasts"][name]}
                for r in eligible])
    drift_summary = {}
    for arm in rows[0]["CPU_minus_historical_GPU_M_by_arm"]:
        values = np.asarray([v for r in rows for v in r["CPU_minus_historical_GPU_M_by_arm"][arm].values()])
        drift_summary[arm] = {"pairs": int(values.size), "max_abs_M_error": float(abs(values).max()),
                              "mean_abs_M_error": float(abs(values).mean()),
                              "mean_signed_M_error": float(values.mean())}
    validation = {
        "status": "PROPERTY_FACTORIAL_FIXED14_18_INDEPENDENT_PASS", "protocol": pb,
        "prelabel_seal": bind(seal_path), "audits": receipts, "original_queries": 8,
        "queries_with_computed_pairs": 7, "original_candidate_axis_size": 128, "computed_pairs": 14,
        "arms": 18, "candidate_arms": 252, "side_weight_arrays": 504,
        "maximum_independent_M_error": max(a["max_arithmetic_error"] for a in audits),
        "arithmetic_limit": ARITHMETIC_LIMIT, "metrics_computed_after_all_mass_audits": True,
        "fixed_pairs_predeclared_using_old_labels": True,
        "historical_CPU_GPU_difference_is_not_arithmetic_failure": True,
    }
    result = {
        "status": "PROPERTY_RESTORATION_FACTORIAL_FIXED14_COMPLETE", "protocol": pb,
        "queries": 8, "target_present": 7, "effective_groups": len({r["component"] for r in eligible}),
        "evaluation_scope": "PRIMARY_FIXED_PAIRS", "computed_pairs": 14,
        "primary_metric": "logM(target)-logM(frozen strongest-content wrong)",
        "cell_definitions": {"A0": "original norm assignment", "A1": "permuted norm assignment",
                             "G": "four GLOBAL phase arms averaged within query",
                             "L": "four LOCAL phase arms averaged within query"},
        "summary": summaries, "CPU_GPU_M_drift": drift_summary, "rows": rows,
        "historical_analysis": historical_binding,
        "scope": "Opened preselected eight-query panel; 14 historically fixed target/wrong pairs from unchanged natural C128; frozen coarse head, all scientific cells on common CPU backend.",
        "no_downstream_bridge_this_round": True,
        "limitations": ["Restoring a computational factor is not recovery of true image geometry.",
                        "Exploratory seven-group bootstrap; no multiplicity correction or unique-cause claim.",
                        "Only 14 fixed pairs were computed: no new C128 ranking, accuracy, or across-C128 common-scale claim.",
                        "Historical external/internal bridge tested phase with original amplitude; it does not supply this joint factorial.",
                        "Current outputs concern upstream M only; POST and external-head propagation remain unexecuted here."],
    }
    write(OUT / "validation.json", validation)
    result["validation"] = bind(OUT / "validation.json")
    write(OUT / "result.json", result)
    lines = ["# 对应分布：幅度分配与相对相位的条件恢复", "",
             "固定粗阶段预测头；8张预定图，保留完整自然C128候选轴。只计算7张有目标图的历史target和冻结strongest-content wrong，共14候选对×18臂＝252候选臂。缺失目标的1张保留原记录，不新增计算、不参与身份分差。",
             "这些候选的粗预测相互独立，因此对同一固定target/wrong的主分差无需计算其他126候选；本实验不产生新C128排名或准确率，也不能检验跨128候选的共同标度分解。",
             "A0保留原范数分配，A1置换范数；G/L分别为同幅度条件下四个GLOBAL/LOCAL相位臂的均值。正值表示指定恢复提高target对冻结错误对手的logM优势。", "",
             "|对照|7组等权均值|探索性95%区间|正/负组|", "|---|---:|---|---:|"]
    for name, summary in summaries["fixed_target_wrong_logM_gap"].items():
        lines.append(f"|{name}|{summary['group_equal_mean']:.8f}|{summary['exploratory_bootstrap95']}|{summary['positive_groups']}/{summary['negative_groups']}|")
    lines += ["", "所有科学对比均使用同一CPU计算；历史GPU输出只用于记录数值漂移，不与新CPU臂拼接。",
              "两侧保存权重经独立fsum均值及平方根重新计算M，数值验收门为2e-10，未改变。",
              "这是条件因素恢复与交互检验，不是准确率实验、真实几何恢复或唯一因果证明。区间为已打开小面板的探索统计，未做多重校正。",
              "本轮未运行POST或外部头；完整上下游桥接仍待后续。",
              "已有[相位外部/内部桥接](../../reports/REPORT_M_PHASE_EXTERNAL_INTERNAL_BRIDGE_V2_20260927.md)只含原幅度条件；已有[H593质量算子](../../reports/REPORT_H593_QUALITY_OPERATOR_V1_20260922.md)研究下游使用方式，两者均不能替代本次上游联合恢复。",
              "", "[逐图、14个固定配对M和漂移](result.json)；[独立验收](validation.json)。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "queries": 8, "target_present": 7,
                      "maximum_independent_M_error": validation["maximum_independent_M_error"]}), flush=True)
    return 0


def self_test():
    values = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float64)
    mass, means, counts = independently_computed_mass([{"weights": values}, {"weights": values * 0.25}])
    assert mass == 0.234375 and means == [0.46875, 0.1171875] and counts == [4, 4]
    contrasts = factorial_contrasts({"A0G": 10, "A0L": 4, "A1G": 3, "A1L": 2})
    assert contrasts["phase_restoration_A1G_minus_A1L"] == 1
    assert contrasts["amplitude_restoration_A0L_minus_A1L"] == 2
    assert contrasts["interaction_original_minus_permuted_phase_effect"] == 5
    summary = group_summary([{"component": "a", "value": 1}, {"component": "a", "value": 3},
                             {"component": "b", "value": 10}])
    assert summary["group_equal_mean"] == 6 and summary["groups"] == 2
    # Sparse position keys must retain original C128 offsets, not become 0/1.
    fixed_cells = {"A0G": 2.0, "A0L": 1.2, "A1G": 0.8, "A1L": 0.5}
    mass = {}
    for arm in ARMS:
        prefix = "A1" if arm.startswith("AMP_PERMUTE") else "A0"
        kind = "L" if "LOCAL" in arm else "G"
        gap = fixed_cells[prefix + kind]
        mass[arm] = {"5": math.exp(-3.0 + gap), "11": math.exp(-3.0)}
    audit = {"index": 0, "query_id": "fixture", "axis": list(range(128)), "positions": [5, 11],
             "mass_by_arm": mass, "max_arithmetic_error": 0.0}
    historical = {"index": 0, "query_id": "fixture", "axis": list(range(128)), "component": "group",
                  "target_present": True, "target_position": 5,
                  "arm_metrics": {"NATIVE": {"strongest_position": 11}},
                  "mass_by_arm": {arm: [0.1] * 128 for arm in ARMS}}
    result = analyze_query(audit, historical)
    actual = result["margin_analysis"]["metrics"]["fixed_target_wrong_logM_gap"]["contrasts"]
    expected = factorial_contrasts(fixed_cells)
    assert all(abs(actual[k] - v) < 1e-12 for k, v in expected.items())
    empty = {**audit, "positions": [], "mass_by_arm": {arm: {} for arm in ARMS}}
    assert analyze_query(empty, {**historical, "target_present": False})["margin_analysis"] is None
    try:
        analyze_query({**audit, "positions": [0, 1]}, historical)
    except AssertionError:
        pass
    else:
        raise AssertionError("fixed-pair axis corruption must fail")
    print(json.dumps({"status": "PROPERTY_FACTORIAL_JOIN_ARITHMETIC_SELF_TEST_PASS"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=480)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        raise SystemExit(join(arguments.budget))
