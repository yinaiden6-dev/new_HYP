#!/usr/bin/env python3
"""Independently validate exported rescue90 artifacts against sealed sources."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures/new_hyp_rescue90_20260924_v1"
OLD = ROOT / "reports/figures/new_hyp_visibility_cases_20260915_v1"
CHECKED = {}


def file_sha(path):
    path = Path(path)
    if str(path) not in CHECKED:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        CHECKED[str(path)] = h.hexdigest()
    return CHECKED[str(path)]


def bound_path(value, verify=True):
    if isinstance(value, str):
        return Path(value)
    p = Path(value["path"])
    if verify:
        assert file_sha(p) == value["sha256"], p
    return p


@lru_cache(maxsize=256)
def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=4)
def cache(path):
    p = Path(path)
    receipt = read(p.with_name("receipt.json"))
    val = read(p.with_name("validation.json"))
    assert "PASS" in val["status"]
    # Source collection and rendering already hashed the large payloads. mmap
    # reads only the requested small tensors here; do not scan token bytes again.
    assert bound_path(receipt["payload"], verify=False) == p
    if "payload" in val:
        assert val["payload"] == receipt["payload"]
    return torch.load(p, map_location="cpu", weights_only=True, mmap=True)


def pdf_pages(path):
    """Read Matplotlib's uncompressed page-tree object and cross-check page objects."""
    data = Path(path).read_bytes()
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    counts = re.findall(rb"<<\s*/Type\s*/Pages\s*/Kids\s*\[.*?\]\s*/Count\s+(\d+)\s*>>", data, re.S)
    assert len(counts) == 1, (path, counts)
    count = int(counts[0])
    page_objects = len(re.findall(rb"/Type\s*/Page\b", data))
    assert count == page_objects, (path, count, page_objects)
    return count


def main():
    torch.set_num_threads(2)
    # Final manifests are written only after all figures and PDF writers close.
    # ZIP creation may proceed independently; the parent adds this report later.
    assert (OUT / "cases_manifest.json").is_file(), "Figure rendering must finish first"
    assert read(OUT / "validation.json")["status"] == "COST1_RESCUE90_RENDER_AND_SEALED_DECISION_PASS"
    cases = read(OUT / "cases_manifest.json")
    assert len(cases) == 90
    counts = Counter(c["dataset"] for c in cases)
    assert counts == {"medicine": 30, "grozi": 30, "isic": 30}
    assert len({(c["dataset"], c["query_id"]) for c in cases}) == 90
    assert len({(c["dataset"], c["display_id"]) for c in cases}) == 90
    assert [c["number"] for c in cases] == list(range(1, 91))
    old_medicine = {c["display_id"] for c in read(OLD / "cases_manifest.json") if c["dataset"] == "medicine"}
    assert all(c["display_id"] not in old_medicine for c in cases if c["dataset"] == "medicine")
    assert all(c["query_id"] != "ISIC-Q-0013" for c in cases if c["dataset"] == "isic")

    role_path = ROOT / "results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json"
    roles = {r["query_id"]: r for r in read(role_path)["records"]}
    repair = read(ROOT / "registry/gallery_identity_repair_v1.json")
    overrides = {r["physical_row"]: r["corrected_identity"] for r in repair["filename_collision_overrides"]}
    for r in repair["verified_byte_identical_duplicate_components"]:
        overrides.update({p: r["canonical_identity"] for p in r["physical_rows"]})
    meta_path = ROOT.parents[2] / "data/isic_ima_pp/img_metadata.csv"
    with meta_path.open() as f:
        metadata = {r["isic_id"]: r for r in csv.DictReader(f)}
    isic_input = {c["query_id"]: c for c in read(OUT / "data/isic_candidates.json")["selected"]}
    checked_maps = 0
    checked_provenance = 0
    per_case = []
    max_mass_error = 0.0

    for c in cases:
        dataset, qid, stem = c["dataset"], c["query_id"], c["stem"]
        folder = OUT / dataset
        assert read(folder / "data" / (stem + ".json")) == c
        assert c["model"] == "COST1" and c["raw_correct"] is False and c["final_correct"] is True
        assert c["inference_calls"] == c["training_updates"] == 0
        assert c["raw_selected"] == c["wrong_physical"] != c["target_physical"] == c["final_selected"]
        assert c["action"] == "SWITCH" and c["raw_hold_utility"] == 0.0
        source_result = read(bound_path(c["source_result"]))
        r = next(r for r in source_result["rows"] if r["query_id"] == qid)
        assert r["correct"]["RAW"] is False and r["correct"]["COST1"] is True
        assert r["target_in_C128"]
        for field in ("raw_payload", "roma_payload"):
            p = bound_path(c[field], verify=False)
            if isinstance(c[field], dict):
                assert c[field] == read(p.with_name("receipt.json"))["payload"]
        raw = cache(bound_path(c["raw_payload"], verify=False))
        roma = cache(bound_path(c["roma_payload"], verify=False))
        sqid = c["source_query_id"]
        q = next(r for r in raw["records"] if r["query_id"] == sqid)
        m = next(r for r in roma["records"] if r["query_id"] == sqid)
        assert q["candidate_physical_rows"] == m["candidate_physical_rows"]
        assert q["query_tokens_sha256"] == m["query_tokens_sha256"]
        assert c["query_grid"] == q["query_grid_shape"] == m["query_grid_shape"]
        assert c["query_path"] == q["query_source_path"]
        assert q["query_source_sha256"] == m["query_source_sha256"]
        raw_winner = q["candidate_physical_rows"][int(q["candidate_raw_scores"].argmax())]
        assert raw_winner == c["raw_selected"]
        challengers = [x for x in q["candidate_physical_rows"] if x != raw_winner]
        assert challengers == c["challenger_physical_rows"]
        if dataset == "medicine":
            fitted = read(bound_path(c["head_source"]))
            pred = next(p for p in fitted["predictions"] if p["query_id"] == qid)["models"]["COST1"]
            assert qid not in fitted["train_query_ids"]
            assert c["source_fold"] == fitted["fold"] == roles[qid]["outer_fold"] == r["fold"]
            assert c["head_parameters_hex"] == fitted["parameters"]["COST1"]
            assert c["identity"] == roles[qid]["identity"]
        else:
            pred_data = read(Path(c["prediction_payload"]))
            pred = next(p for p in pred_data["records"] if p["query_id"] == qid)["models"]["COST1"]
            assert r["selected"]["RAW"] == raw_winner
            assert r["selected"]["COST1"] == c["target_physical"]
            assert r["identity"] == c["identity"]
            gallery_name = "gallery_append_manifest.json" if dataset == "grozi" else "gallery_manifest.json"
            gallery = {r["physical_row"]: r for r in read(bound_path(c["source_result"]).parent / gallery_name)["records"]}
            assert gallery[c["target_physical"]]["identity"] == c["identity"]
            if raw_winner in gallery:
                assert gallery[raw_winner]["identity"] != c["identity"]
            else:
                assert dataset == "grozi" and raw_winner < 5413

        assert c["sealed_COST1_logits_hex"] == pred["logits_hex"]
        z = np.array([float.fromhex(v) for v in pred["logits_hex"]], dtype=np.float64)
        assert len(z) == len(challengers) == len(set(challengers)) == 127
        assert np.isfinite(z).all() and float(z.max()) > 0
        selected = challengers[int(z.argmax())] if z.max() > 0 else raw_winner
        assert selected == pred["selected"] == c["final_selected"] == c["target_physical"]
        assert c["selected_logit"] == float(z.max())
        assert c["scores128_physical_axis"] == [raw_winner] + challengers
        assert c["scores128_hex"] == [float(0).hex()] + pred["logits_hex"]
        utilities = list(map(float.fromhex, c["scores128_hex"]))
        assert c["scores128_physical_axis"][int(np.argmax(utilities))] == c["target_physical"]

        with np.load(folder / "data" / (stem + ".npz")) as maps:
            assert set(maps.files) == {f"{role}_{side}_visibility" for role in ("wrong", "target") for side in ("query", "reference")}
            for role in ("wrong", "target"):
                evidence = c["evidence"][role]
                candidate = next(x for x in m["candidates"] if x["physical_row"] == c[role + "_physical"])
                ref = raw["references"][c[role + "_physical"]]
                assert evidence["scores"] == candidate["old_scores"]
                assert evidence["reference_path"] == ref["source_path"]
                assert Path(ref["source_path"]).is_file()
                if "source_image_sha256" in ref:
                    assert ref["source_image_sha256"] == candidate["reference_image_sha256"]
                assert ref["tokens_sha256"] == candidate["reference_tokens_sha256"]
                if dataset == "medicine":
                    ident = overrides.get(c[role + "_physical"], Path(ref["source_path"]).stem.strip())
                    assert (ident == c["identity"]) == (role == "target"), (qid, role, ident, c["identity"])
                for side, shape in [("query", c["query_grid"]), ("reference", ref["grid_shape"])]:
                    a = maps[f"{role}_{side}_visibility"]
                    assert a.dtype == np.float64 and list(a.shape) == shape
                    assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
                    assert hashlib.sha256(a.tobytes()).hexdigest() == evidence[side + "_map_sha256"] == candidate[side + "_map_sha256"]
                    assert a.tobytes() == candidate[side + "_visibility"].numpy().tobytes()
                    checked_maps += 1
                mass = float(np.sqrt(maps[f"{role}_query_visibility"].mean() * maps[f"{role}_reference_visibility"].mean()))
                error = abs(mass - evidence["scores"]["visibility_mass"])
                assert error < 1e-14
                max_mass_error = max(max_mass_error, error)

        with Image.open(folder / (stem + ".png")) as image:
            assert image.size == (3200, 1800)
            image.verify()
        assert (folder / (stem + ".svg")).is_file()
        assert pdf_pages(folder / (stem + ".pdf")) == 1
        for role in ("query", "wrong", "target"):
            with Image.open(folder / "assets" / (stem + "_" + role + ".jpg")) as image:
                image.verify()
        if dataset == "isic":
            assert c["image_provenance"] == isic_input[qid]["image_provenance"]
            assert c["clinical_diagnosis_claimed"] is False
            for role, p in c["image_provenance"].items():
                meta = metadata[p["original_isic_id"]]
                assert p["attribution"] == meta["attribution"]
                assert p["copyright_license"] == meta["copyright_license"]
                assert p["original_lesion_id"] == meta["lesion_id"]
                assert p["metadata_sha256"] == file_sha(meta_path)
                assert Path(p["local_image_path"]).is_file()
                assert Path(p["original_source_path"]).stem == p["original_isic_id"]
                expected_path = c["query_path"] if role == "query" else c["evidence"][role]["reference_path"]
                assert p["local_image_path"] == expected_path
                checked_provenance += 1
            ip = c["image_provenance"]
            assert ip["query"]["original_lesion_id"] == ip["target"]["original_lesion_id"] != ip["wrong"]["original_lesion_id"]
        per_case.append(dict(dataset=dataset, query_id=qid, selected=c["final_selected"],
                             selected_logit=c["selected_logit"], native_maps=4, status="PASS"))

    assert checked_maps == 360 and checked_provenance == 90
    pages = {"rescue90.pdf": pdf_pages(OUT / "rescue90.pdf")}
    assert pages["rescue90.pdf"] == 90
    for group in counts:
        assert len(list((OUT / group).glob("*.png"))) == 30
        key = f"{group}/{group}_30cases.pdf"
        pages[key] = pdf_pages(OUT / key)
        assert pages[key] == 30
    old_hashes = read(OUT / "validation.json")["old18_sha256"]
    for path, expected in old_hashes.items():
        assert file_sha(path) == expected
    report = dict(status="COST1_RESCUE90_INDEPENDENT_SOURCE_ACTION_MAP_FIGURE_VALIDATION_PASS",
                  cases=90, datasets=dict(counts), unique_queries=90, all_sealed_RAW_wrong_COST1_correct=True,
                  all_127_challengers_and_HOLD0_independently_reselected=True, selected_identity_verified=True,
                  medicine_grouped_head_query_exclusion_verified=True, old18_medicine_overlap=0,
                  original18_files_unchanged=len(old_hashes), native_FP64_maps=checked_maps,
                  native_maps_match_source_cache_and_evidence_sha256=True,
                  source_payload_access="CPU mmap; original large-file hashes previously verified by source collection and renderer; not rescanned here",
                  maximum_visibility_mass_recalculation_error=max_mass_error,
                  png_dimensions=[3200, 1800], png_count=90, pdf_pages=pages,
                  all_single_case_pdf_pages=1, isic_image_provenance_roles=checked_provenance,
                  per_image_ISIC_attribution_license_preserved=True,
                  inference_calls=0, training_updates=0,
                  validator=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)),
                  manifest_sha256=file_sha(OUT / "cases_manifest.json"),
                  per_case=per_case)
    report_path = OUT / "independent_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, ensure_ascii=False, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
