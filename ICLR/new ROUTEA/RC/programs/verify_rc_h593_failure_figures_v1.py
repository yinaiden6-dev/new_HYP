#!/usr/bin/env python3
"""Independently verify the complete H593 COST1 error gallery against sealed data.

This validator does not import the renderer, train a model, or run inference.
Large cache receipts are checked against their declared bindings, while every
exported array is compared directly with the cache and its stored map digest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf.gallery_identity_repair import build_identity_map


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1024)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=4)
def cache(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def gallery_catalogue():
    base = ROOT.parents[2] / "dailymed/data/box_flat_20000_images/data/raw_images"
    paths = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for directory, directories, names in os.walk(base):
        directories.sort()
        paths.extend(Path(directory) / name for name in sorted(names)
                     if Path(name).suffix.lower() in extensions and (Path(directory) / name).is_file())
    labels = build_identity_map([p.stem.strip() for p in paths]).labels
    return paths, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports/figures/new_hyp_h593_failures_en_20260927_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    fit = ROOT / "results/rc_six_cause_isolation_v1/loss_binding"
    population = ROOT / "results/rc_new_hyp593_oof5_v1"
    result = read(fit / "result.json")
    expected = {r["query_id"]: r for r in result["rows"] if not r["correct"]["COST1"]}
    all_rows = {r["query_id"]: r for r in result["rows"]}
    roles = {r["query_id"]: r for r in read(population / "metadata/curator_roles.json")["records"]}
    workers = read(population / "metadata/worker_manifest.json")["records"]
    locations = {}
    for lane in ("reuse", "missing"):
        for i, worker in enumerate(w for w in workers if ("reuse" in w) == (lane == "reuse")):
            locations[worker["query_id"]] = (lane, i // 8)
    workers = {r["query_id"]: r for r in workers}
    folds = {i: read(fit / f"fold{i}/payload.json") for i in range(5)}
    predictions = {p["query_id"]: p["models"]["COST1"]
                   for fold in folds.values() for p in fold["predictions"]}
    gallery, labels = gallery_catalogue()
    manifest = read(out / "cases_manifest.json")
    if isinstance(manifest, dict):
        manifest = manifest["cases"]
    captions = read(out / "data/rendered_english_text.json")
    issues, checks = [], 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not bool(condition):
            issues.append(message)

    check(len(all_rows) == 593, "source population must be H593")
    check(sum(r["correct"]["RAW"] for r in all_rows.values()) == 426, "RAW source total")
    check(sum(r["correct"]["COST1"] for r in all_rows.values()) == 481, "COST1 source total")
    check(len(expected) == len(manifest) == 112, "complete error population must contain 112 cases")
    check(len({c["query_id"] for c in manifest}) == len(manifest), "duplicate query in gallery")
    check({c["query_id"] for c in manifest} == set(expected), "gallery membership differs from all COST1 errors")
    check(set(captions) == {c["stem"] for c in manifest}, "rendered caption membership")
    counts, tracks, present, png_sizes = Counter(), Counter(), Counter(), Counter()
    max_error = 0.0
    for c in manifest:
        qid = c["query_id"]
        row, role, worker = expected[qid], roles[qid], workers[qid]
        prefix = c["display_id"] + ": "
        head = folds[row["fold"]]
        pred = predictions[qid]
        lane, shard = locations[qid]
        fp = population / "features" / lane / f"shard{shard:02d}" / "payload.pt"
        if lane == "reuse":
            rp = Path(worker["reuse"]["roma"]["payload"]["path"])
            source_qid = worker["reuse"]["source_query_id"]
        else:
            rp = population / "roma" / f"shard{shard:02d}" / "payload.pt"
            source_qid = qid
        check(c["source_fold"] == row["fold"] == role["outer_fold"], prefix + "OOF fold")
        check(c["display_id"] == row["original_query_id"], prefix + "display identity")
        check(qid not in head["train_query_ids"], prefix + "query leaked into head training")
        check(c["source_query_id"] == source_qid, prefix + "cache query mapping")
        check(Path(c["head_source"]["path"]).resolve() == (fit / f"fold{row['fold']}/payload.json").resolve(), prefix + "head path")
        check(c["head_source"]["sha256"] == sha(c["head_source"]["path"]), prefix + "head binding")
        check(c["head_parameters_hex"] == head["parameters"]["COST1"], prefix + "frozen head parameters")
        for key, source in (("feature_payload", fp), ("roma_payload", rp)):
            check(Path(c[key]["path"]).resolve() == source.resolve(), prefix + key + " path")
            receipt = read(source.parent / "receipt.json")
            check(c[key]["sha256"] == receipt["payload"]["sha256"], prefix + key + " receipt digest")
        f = next(r for r in cache(str(fp))["records"] if r["query_id"] == qid)
        m = next(r for r in cache(str(rp))["records"] if r["query_id"] == source_qid)
        axis = f["candidate_physical_rows"]
        raw = axis[f["winner"]]
        challengers = [axis[i] for i in f["challenger_positions"]]
        check(axis == m["candidate_physical_rows"], prefix + "candidate alignment")
        check(raw == axis[int(m["candidate_raw_scores"].argmax())], prefix + "RAW selection")
        z = np.array([float.fromhex(x) for x in pred["logits_hex"]], dtype=np.float64)
        chosen = challengers[int(z.argmax())] if z.max() > 0 else raw
        target_rows = [i for i, label in enumerate(labels) if label == role["identity"]]
        check(len(target_rows) == 1, prefix + "unique corrected target reference")
        target = target_rows[0]
        is_present = target in axis
        kind = "BREAK" if labels[raw] == role["identity"] else ("WRONG_HOLD" if chosen == raw else "WRONG_SWITCH")
        check(chosen == pred["selected"] == c["final_selected"], prefix + "sealed action")
        check(c["raw_selected"] == raw and c["target_physical"] == target, prefix + "physical rows")
        check(labels[chosen] != role["identity"] and c["final_correct"] is False, prefix + "final is wrong")
        check(c["raw_correct"] == row["correct"]["RAW"] == (labels[raw] == role["identity"]), prefix + "RAW correctness")
        check(c["failure_kind"] == kind, prefix + "error classification")
        check(c["action"] == ("HOLD" if chosen == raw else "SWITCH"), prefix + "action label")
        check(c["selected_logit"] == (0.0 if chosen == raw else float(z.max())), prefix + "selected utility")
        check(c["max_challenger_logit"] == float(z.max()), prefix + "maximum challenger utility")
        check(c["target_in_C128"] == row["target_in_C128"] == is_present, prefix + "natural recall")
        check(c["challenger_physical_rows"] == challengers, prefix + "challenger order")
        check(c["sealed_COST1_logits_hex"] == pred["logits_hex"], prefix + "sealed exact logit strings")
        check(c["scores128_physical_axis"] == [raw] + challengers, prefix + "128 decision axis")
        check(c["scores128_hex"] == [0.0.hex()] + pred["logits_hex"], prefix + "HOLD plus 127 logits")
        decision_axis = [raw] + challengers
        decision_scores = np.concatenate(([0.0], z))
        expected_target_logit = float(decision_scores[decision_axis.index(target)]) if is_present else None
        check(c["target_logit"] == expected_target_logit, prefix + "target utility")
        check(c["target_raw_full_gallery_rank"] == f["raw_ranked_physical_rows"].index(target) + 1,
              prefix + "full-gallery target rank")
        check(c["candidate_physical_rows"] == axis, prefix + "natural C128 axis")
        check(c["raw_candidate_scores_hex"] == [float(x).hex() for x in m["candidate_raw_scores"]],
              prefix + "RAW candidate scores")
        check(sha(c["query_path"]) == m["query_source_sha256"], prefix + "query bytes")
        check(c["query_geometry"] == m["query_geometry"] and c["processor_input_frame"] == m["processor_input_frame"],
              prefix + "query display geometry")
        check(c["fold_number_displayed"] is False and c["query_excluded_from_head_training"] is True,
              prefix + "declared OOF rendering")
        check(c["training_updates"] == c["encoder_or_matcher_inference_calls"] == 0,
              prefix + "display-only execution")
        values = f["modes"]["REAL"]["X"].numpy()
        theta = np.array([float.fromhex(x) for x in head["parameters"]["COST1"]], dtype=np.float64)
        replay = np.sum(values * theta[:-1], axis=1) + theta[-1]
        error = float(np.max(np.abs(replay - z)))
        max_error = max(max_error, error)
        check(error < 2e-10, prefix + "independent numerical replay")
        with np.load(out / "data" / (c["stem"] + ".npz"), allow_pickle=False) as arrays:
            check(arrays["decision_X"].dtype == values.dtype and arrays["decision_X"].shape == values.shape
                  and arrays["decision_X"].tobytes() == values.tobytes(), prefix + "decision matrix bytes")
            pairs = [("wrong", chosen)] + ([("target", target)] if is_present else [])
            for name, physical in pairs:
                candidate = next(x for x in m["candidates"] if x["physical_row"] == physical)
                evidence = c["evidence"][name]
                check(evidence["physical_row"] == physical, prefix + name + " evidence identity")
                check(evidence["identity"] == labels[physical], prefix + name + " repaired label")
                check(Path(evidence["reference_path"]).resolve() == gallery[physical].resolve(), prefix + name + " reference path")
                check(sha(evidence["reference_path"]) == candidate["reference_image_sha256"], prefix + name + " photo digest")
                check(evidence["reference_image_sha256"] == candidate["reference_image_sha256"], prefix + name + " exported photo digest")
                check(evidence["reference_geometry"] == candidate["reference_geometry"], prefix + name + " reference geometry")
                check(evidence["scores"] == candidate["old_scores"], prefix + name + " cached scores")
                check(evidence["visibility_available"] is True, prefix + name + " map availability")
                mass = math.sqrt(float(candidate["query_visibility"].mean()) * float(candidate["reference_visibility"].mean()))
                check(abs(mass - evidence["scores"]["visibility_mass"]) < 1e-14, prefix + name + " M from native maps")
                for side, grid in (("query", m["query_grid_shape"]), ("reference", candidate["reference_grid_shape"])):
                    original = candidate[side + "_visibility"].numpy().reshape(grid)
                    a = arrays[name + "_" + side + "_visibility"]
                    check(a.shape == original.shape and a.dtype == original.dtype == np.float64
                          and a.tobytes() == original.tobytes(), prefix + name + " " + side + " native array")
                    digest = hashlib.sha256(a.tobytes()).hexdigest()
                    check(digest == candidate[side + "_map_sha256"] == evidence[side + "_map_sha256"], prefix + name + " " + side + " map digest")
                    check(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all(), prefix + name + " " + side + " map range")
            if not is_present:
                check("target_query_visibility" not in arrays and "target_reference_visibility" not in arrays,
                      prefix + "absent target must not have a synthesized heatmap")
                evidence = c["evidence"]["target"]
                check(evidence["physical_row"] == target and evidence["identity"] == labels[target] == role["identity"],
                      prefix + "absent target reference identity")
                check(Path(evidence["reference_path"]).resolve() == gallery[target].resolve(),
                      prefix + "absent target reference path")
                check(sha(evidence["reference_path"]) == evidence["reference_image_sha256"],
                      prefix + "absent target reference digest")
                check(evidence["visibility_available"] is False and evidence["scores"] is None,
                      prefix + "absent target remains unscored")
        with Image.open(out / "medicine" / (c["stem"] + ".png")) as im:
            check(im.size == (3200, 1800), prefix + "presentation dimensions")
            png_sizes[str(im.size)] += 1
        check((out / "medicine" / (c["stem"] + ".pdf")).is_file(), prefix + "single PDF exists")
        check((out / "medicine" / (c["stem"] + ".svg")).is_file(), prefix + "SVG exists")
        text = "\n".join(captions[c["stem"]])
        check(not re.search(r"[\u3400-\u9fff]", text), prefix + "English captions")
        check("fold" not in text.lower(), prefix + "fold hidden in captions")
        check("COST1" in text and c["display_id"] in text, prefix + "case identity and model caption")
        check("Not a segmentation mask" in text and "Shared 0–1 scale" in text, prefix + "map caption")
        check("RAW correct → COST1 incorrect" in text if c["raw_correct"] else "RAW incorrect → COST1 incorrect" in text,
              prefix + "correct transition caption")
        check(f"Decision: {c['action']}" in text, prefix + "decision caption")
        if not is_present:
            check("Target outside C128 (retrieval miss)" in text
                  and "S = not computed    M = not computed" in text and "target not scored" in text,
                  prefix + "absent target caption")
        single = out / (c["stem"] + ".json")
        if not single.exists():
            single = out / "data" / (c["stem"] + ".json")
        check(read(single) == c, prefix + "manifest versus case JSON")
        counts[kind] += 1
        tracks[row["original_query_id"].split("-")[0]] += 1
        present[str(is_present)] += 1
    check(dict(counts) == {"WRONG_HOLD": 92, "WRONG_SWITCH": 17, "BREAK": 3}, "failure totals")
    check(dict(tracks) == {"OUTCOME": 102, "DIFFICULT": 9, "NDV2": 1}, "track totals")
    check(dict(present) == {"True": 89, "False": 23}, "candidate recall totals")
    pdf_pages = {}
    stems = {c["stem"] for c in manifest}
    check((out / "failures112.pdf").is_file(), "combined PDF exists")
    for path in list((out / "medicine").glob("*.pdf")) + list(out.glob("*.pdf")):
        data = path.read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page\b", data))
        pdf_pages[path.name] = pages
        if path.stem in stems:
            check(pages == 1, path.name + ": single-case PDF page count")
        elif "112" in path.stem or "all" in path.stem.lower():
            check(pages == 112, path.name + ": combined PDF page count")
    report = dict(status="H593_COST1_FAILURE_GALLERY_INDEPENDENT_PASS" if not issues else "FAIL",
                  checks=checks, issues=issues, cases=len(manifest), counts=dict(counts), tracks=dict(tracks),
                  target_in_C128=dict(present), png_sizes=dict(png_sizes), maximum_logit_replay_error=max_error,
                  pdf_pages=pdf_pages,
                  source_result=dict(path=str(fit / "result.json"), sha256=sha(fit / "result.json")),
                  cases_manifest_sha256=sha(out / "cases_manifest.json"),
                  large_payload_verification="Receipt bindings and direct exported-array comparisons; no whole-payload rehash",
                  training_updates=0, encoder_or_matcher_inference_calls=0,
                  validator=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    (out / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
