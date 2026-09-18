#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "programs"))
import run_romav2_colnomic_no_regret_action_gate_v1 as old
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen
SRC = ROOT / "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
HEAD = ROOT / "results/romav2_colnomic_full_negative_action_gate_v1/result.json"
OUT = ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
MODULE = ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise RuntimeError("immutable gate validation exists")
    head = json.load(open(HEAD))
    checks = {
        "feature_names": len(frozen.FEATURE_NAMES) == len(set(frozen.FEATURE_NAMES)) == 6,
        "weights": frozen.WEIGHT.tolist() == head["head"]["weight"] and frozen.BIAS == head["head"]["bias"] and frozen.PARAMETER_COUNT == head["parameter_count"] == 7,
        "threshold": frozen.SWITCH_THRESHOLD == 0.0,
    }
    max_abs, count = 0.0, 0
    for shard in range(8):
        doc = json.load(open(SRC / f"shard{shard:02d}/result.json"))
        for row in doc["rows"]:
            candidates = {int(x["candidate_position"]): x for x in row["candidates"]}
            raw = [float(candidates[i]["raw_score"]) for i in range(128)]
            winner = max(range(128), key=lambda i: (raw[i], -int(candidates[i]["physical_row"])))
            for challenger in range(128):
                if challenger == winner: continue
                a = old.candidate_feature(raw, candidates, challenger, winner)
                b = frozen.candidate_feature(raw, candidates, challenger, winner)
                max_abs = max(max_abs, float((a - b).abs().max())); count += 1
    checks["all_features_bit_exact"] = max_abs == 0.0 and count == 64 * 127
    passed = all(checks.values())
    value = {
        "schema_version": "rc_romav2_colnomic_frozen_gate_definition_validation_v1_20260901",
        "status": "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS" if passed else "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_FAIL",
        "checks": checks, "feature_comparison_count": count, "feature_max_abs": max_abs,
        "module_sha256": sha(MODULE), "head_sha256": sha(HEAD),
        "feature_names": list(frozen.FEATURE_NAMES), "weight": frozen.WEIGHT.tolist(),
        "bias": frozen.BIAS, "parameter_count": frozen.PARAMETER_COUNT,
        "switch_threshold": frozen.SWITCH_THRESHOLD,
    }
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks, "max_abs": max_abs}, sort_keys=True))
    raise SystemExit(0 if passed else 4)
if __name__ == "__main__": main()
