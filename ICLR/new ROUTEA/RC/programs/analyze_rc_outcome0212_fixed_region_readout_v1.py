#!/usr/bin/env python3
"""Fixed GT-region readout of existing four pixel conditions; no fitting.

The existing annotation chooses the support. This is an oracle capacity and
pixel-dependence example, not a target-free proposal or a new EVAL accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/rc_outcome0212_pixel_source_diagnostic_v1"
GEOMETRY = ROOT / "results/rc_outcome0212_exclusive_token_gold_polygon_v1/result.json"
OUT = ROOT / "results/rc_outcome0212_fixed_region_readout_v1"
MODES = ("ORIGINAL", "KEEP_TARGET", "ERASE_TARGET", "ALL_GRAY")


def need(x, message):
    if not bool(x): raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())
def canon(x): return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def bind(path): return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def checked(item):
    path = Path(item["path"])
    need(path.resolve().is_relative_to(ROOT) and sha(path) == item["sha256"], "SOURCE_BINDING")
    return path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canon(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def main(validate=False):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    source = read(SOURCE / "result.json")
    validation = read(SOURCE / "independent_validation.json")
    need(sha(SOURCE / "result.json") == "81bbaaafb37e898524ac42b959e07bc322baf00ce0aaad4ed46e384fdd949388", "PIXEL_RESULT_PIN")
    need(validation["result_sha256"] == sha(SOURCE / "result.json") and all(validation["checks"].values()), "PIXEL_RESULT_NOT_VALIDATED")
    need(sha(GEOMETRY) == "e3253768658172bb3b6d247d69a2f1259400c9f6122eb8d91097480aa1ecc92b", "ORIGINAL_GT_PIN")
    geometry = read(GEOMETRY)
    with np.load(checked(geometry["arrays"]), allow_pickle=False) as arrays:
        inside = arrays["target_polygon_center_mask"].copy()
    need(inside.shape == (720,) and inside.dtype == np.bool_ and int(inside.sum()) == 47, "FIXED_GT_REGION")
    cells = set(map(int, np.flatnonzero(inside)))
    remaining = set(cells); components = []
    while remaining:
        first = min(remaining); remaining.remove(first); reached = [first]
        for p in reached:
            row, col = divmod(p, 20)
            for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                n = rr * 20 + cc
                if 0 <= rr < 36 and 0 <= cc < 20 and n in remaining:
                    remaining.remove(n); reached.append(n)
        components.append(sorted(reached))
    axis = source["candidate_physical_rows"]
    need(len(axis) == len(set(axis)) == 128, "FULL128_AXIS")
    groups = {"ALL720": list(range(720)), "GT47": sorted(cells), "OUTSIDE673": sorted(set(range(720)) - cells)}
    rows = []; largest_error = 0.
    for entry in source["entries"]:
        mode = entry["mode"]
        with np.load(checked(entry["arrays"]), allow_pickle=False) as arrays:
            a = arrays["a"].copy()
        need(a.shape == (128, 720) and a.dtype == np.float64 and np.isfinite(a).all(), "FULL_A_MATRIX")
        at = torch.from_numpy(a)
        for name, ids in groups.items():
            scores = at.mean(dim=1) if name == "ALL720" else at.index_select(1, torch.tensor(ids, dtype=torch.int64)).mean(dim=1)
            oracle = np.array([np.mean(a[c, ids], dtype=np.float64) for c in range(128)])
            error = float(np.max(np.abs(oracle - scores.numpy())))
            largest_error = max(largest_error, error)
            need(error <= 1e-14, "NUMPY_POOL_REPLAY")
            order = sorted(range(128), key=lambda i: (-float(scores[i]), axis[i]))
            numpy_order = sorted(range(128), key=lambda i: (-float(oracle[i]), axis[i]))
            need(order == numpy_order, "INDEPENDENT_FULL_RANK_ORDER")
            if name == "ALL720":
                need([float(x).hex() for x in scores] == source["descriptions"][mode]["mean_MaxSim_scores_binary64"], "ORIGINAL_POOL_BITS")
            target = axis.index(1024); wrong = axis.index(1631)
            strongest = max((i for i in range(128) if i != target), key=lambda i: (float(scores[i]), -axis[i]))
            rows.append({"pixel_condition": mode, "readout_support": name, "token_count": len(ids),
                         "scores_binary64": [float(x).hex() for x in scores],
                         "ranked_physical_rows": [axis[i] for i in order], "target_rank": order.index(target) + 1,
                         "critical_wrong_rank": order.index(wrong) + 1,
                         "target_margin_to_all127": float(scores[target] - scores[strongest]),
                         "strongest_wrong_physical_row": axis[strongest]})
    need([r["pixel_condition"] for r in rows[::3]] == list(MODES), "FOUR_CONDITION_AXIS")
    result = {"status": "FIXED_CONNECTED_GT_REGION_FOUR_PIXEL_CONDITION_READOUT_COMPLETE",
              "sources": {"program": bind(Path(__file__)), "pixel_result": bind(SOURCE / "result.json"),
                          "pixel_validation": bind(SOURCE / "independent_validation.json"), "geometry": bind(GEOMETRY)},
              "query_id": "OUTCOME-0212", "execution_ordinal": 193, "candidate_physical_rows": axis,
              "support_indices": groups, "GT_four_neighbor_components": components,
              "GT_is_one_connected_component": len(components) == 1,
              "readout": "FP64 mean of full-reference unweighted MaxSim over the same fixed native query positions",
              "rows": rows, "independent_numpy_max_abs_score_error": largest_error,
              "new_encoder_forward_count": 0, "training_updates": 0,
              "HYP_GO_claimed": False, "retrieval_accuracy_claimed": False,
              "limits": ["The annotation supplies an oracle target region, not an automatic P.",
                         "This query was already correct under the original Native7/C action.",
                         "This is a single opened query with fixed full C128, not a new 32-query accuracy or stage qualification.",
                         "Native token selection does not imply independence from pixels outside that region."]}
    if validate:
        need(canon(read(OUT / "result.json")) == canon(result), "RESULT_REPLAY")
        save(OUT / "validation.json", {"status": "FIXED_REGION_READOUT_REPLAY_PASS", "result_sha256": sha(OUT / "result.json"),
             "all12_full128_rankings_numpy_independently_agree": True, "all720_baseline_scores_bit_exact": True,
             "new_encoder_forward_count": 0, "training_updates": 0})
    else:
        need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
        save(OUT / "result.json", result)
    print(json.dumps({"connected": result["GT_is_one_connected_component"], "rows": [{k:v for k,v in r.items() if k not in ("scores_binary64", "ranked_physical_rows")} for r in rows]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--validate", action="store_true")
    main(p.parse_args().validate)
