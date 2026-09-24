#!/usr/bin/env python3
"""Select 30 real ISIC rescues and verify cached sources; no model inference."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results/rc_new_hyp_isic_transfer_v1"
OUT = ROOT / "reports/figures/new_hyp_rescue90_20260924_v1/data/isic_candidates.json"
METADATA = ROOT.parents[2] / "data/isic_ima_pp/img_metadata.csv"
SOURCES = {}
EXCLUDED = ["ISIC-Q-0013"]
RULE = ("Exclude previously showcased ISIC-Q-0013; among sealed RAW-wrong to COST1-correct "
        "queries, take the lexicographically earliest query per lesion, then append earliest "
        "remaining queries until 30; present by query_id. No visibility brightness, score, "
        "or diagnosis is used to select cases.")


def source(path, expected=None):
    path = Path(path)
    if str(path) not in SOURCES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        SOURCES[str(path)] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
    if expected:
        assert SOURCES[str(path)]["sha256"] == expected, path
    return path


def read(path, expected=None):
    return json.loads(source(path, expected).read_text())


def load(stage, shard):
    folder = WORK / stage / f"shard{shard:02d}"
    receipt = read(folder / "receipt.json")
    validation = read(folder / "validation.json")
    assert "PASS" in validation["status"]
    return torch.load(source(receipt["payload"]["path"], receipt["payload"]["sha256"]),
                      map_location="cpu", weights_only=True, mmap=True)


def main():
    torch.set_num_threads(2)
    result = read(WORK / "result.json", "6af5bfbafd36f9438372b2c5c1fff3e2f154ffa9c9af84000278809b14ea0d46")
    for name in ["result_validation.json", "independent_final_audit_v1.json"]:
        assert "PASS" in read(WORK / name)["status"]
    authority = read(result["authority"]["path"], result["authority"]["sha256"])
    head = authority["heads"]["COST1"]["head"]
    source(head["path"], head["sha256"])
    workers = {r["query_id"]: r for r in read(WORK / "worker_manifest.json")["records"]}
    gallery = {r["physical_row"]: r for r in read(WORK / "gallery_manifest.json")["records"]}
    receipts = {r["asset_id"]: r for r in read(WORK / "image_receipt.json")["files"]}
    curator = read(WORK / "curator_roles.json")
    source(METADATA, curator["source"]["sha256"])
    with METADATA.open() as f:
        metadata = {r["isic_id"]: r for r in csv.DictReader(f)}
    all_rescues = sorted((r for r in result["rows"] if not r["correct"]["RAW"] and r["correct"]["COST1"]),
                        key=lambda r: r["query_id"])
    assert len(all_rescues) == 34
    eligible = [r for r in all_rescues if r["query_id"] not in EXCLUDED]
    selected = []
    seen = set()
    for row in eligible:
        if row["identity"] not in seen:
            seen.add(row["identity"])
            selected.append(row)
    selected_ids = {r["query_id"] for r in selected}
    selected.extend([r for r in eligible if r["query_id"] not in selected_ids][:30-len(selected)])
    selected.sort(key=lambda r: r["query_id"])
    assert len(selected) == len({r["query_id"] for r in selected}) == 30

    def provenance(asset_id):
        receipt = receipts[asset_id]
        original_id = Path(receipt["source_path"]).stem
        meta = metadata[original_id]
        source(receipt["image_path"], receipt["image_sha256"])
        return dict(asset_id=asset_id, original_isic_id=original_id,
                    local_image_path=receipt["image_path"], original_source_path=receipt["source_path"],
                    image_sha256=receipt["image_sha256"], original_lesion_id=meta["lesion_id"],
                    attribution=meta["attribution"], copyright_license=meta["copyright_license"],
                    metadata_source=str(METADATA), metadata_sha256=SOURCES[str(METADATA)]["sha256"],
                    metadata_scope="Existing local metadata copied verbatim; no blanket rights clearance claimed.")

    cases = []
    cache = {}
    for row in selected:
        qid = row["query_id"]
        ordinal = workers[qid]["execution_ordinal"]
        shard = ordinal // 8
        if shard not in cache:
            # Retain just this shard to keep CPU memory use bounded.
            cache.clear()
            predpath = WORK / "predictions" / f"shard{shard:02d}" / "payload.json"
            assert "PASS" in read(predpath.with_name("validation.json"))["status"]
            cache[shard] = (load("raw", shard), load("roma", shard), read(predpath))
        raw, roma, pred = cache[shard]
        q = next(r for r in raw["records"] if r["query_id"] == qid)
        m = next(r for r in roma["records"] if r["query_id"] == qid)
        p = next(r for r in pred["records"] if r["query_id"] == qid)
        wrong, target = row["selected"]["RAW"], row["selected"]["COST1"]
        assert target != wrong and row["target_in_C128"]
        assert gallery[target]["identity"] == row["identity"]
        assert gallery[wrong]["identity"] != row["identity"]
        assert q["query_tokens_sha256"] == m["query_tokens_sha256"]
        assert q["query_grid_shape"] == m["query_grid_shape"]
        assert q["candidate_physical_rows"] == m["candidate_physical_rows"]
        assert p["raw_selected"] == wrong == int(q["candidate_physical_rows"][q["candidate_raw_scores"].argmax()])
        challengers = sorted(set(p["raw_ranked_physical_rows"][:128]) - {wrong})
        logits = list(map(float.fromhex, p["models"]["COST1"]["logits_hex"]))
        assert len(challengers) == len(logits) == 127
        logit, best = sorted(zip(logits, challengers), key=lambda x: (-x[0], x[1]))[0]
        assert logit > 0 and best == target == p["models"]["COST1"]["selected"]
        evidence = {}
        for role, physical in [("wrong", wrong), ("target", target)]:
            c = next(c for c in m["candidates"] if c["physical_row"] == physical)
            ref = raw["references"][physical]
            assert c["reference_tokens_sha256"] == ref["tokens_sha256"]
            assert c["reference_grid_shape"] == ref["grid_shape"]
            assert ref["source_path"] == gallery[physical]["image_path"]
            for side, shape in [("query", m["query_grid_shape"]), ("reference", ref["grid_shape"])]:
                a = c[side + "_visibility"].numpy()
                assert a.dtype == np.float64 and a.size == int(np.prod(shape))
                assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
                assert hashlib.sha256(a.tobytes()).hexdigest() == c[side + "_map_sha256"]
            mass = float(np.sqrt(c["query_visibility"].numpy().mean() * c["reference_visibility"].numpy().mean()))
            assert abs(mass - c["old_scores"]["visibility_mass"]) < 1e-14
            evidence[role] = dict(physical_row=physical, reference_path=ref["source_path"],
                                  reference_grid=ref["grid_shape"], scores=c["old_scores"],
                                  query_map_sha256=c["query_map_sha256"], reference_map_sha256=c["reference_map_sha256"])
        prov = {"query": provenance(qid), "wrong": provenance(gallery[wrong]["reference_id"]),
                "target": provenance(gallery[target]["reference_id"])}
        assert prov["query"]["image_sha256"] == m["query_source_sha256"]
        assert prov["query"]["original_lesion_id"] == prov["target"]["original_lesion_id"]
        assert prov["query"]["original_lesion_id"] != prov["wrong"]["original_lesion_id"]
        cases.append(dict(dataset="isic", dataset_label="ISIC 病灶", panel="ISIC537 探索面板",
                          model="COST1", model_lineage=result["model_lineage"],
                          evidence_scope="已打开 ISIC 队列的跨域探索结果；事后成功案例展示；不声称临床诊断或新外部确认",
                          query_id=qid, source_query_id=qid, display_id=qid, execution_ordinal=ordinal,
                          identity=row["identity"], component=row["component"], query_path=workers[qid]["image_path"],
                          source_result=str(WORK / "result.json"), source_result_sha256=SOURCES[str(WORK / "result.json")]["sha256"],
                          raw_correct=False, final_correct=True, raw_selected=wrong, final_selected=target,
                          target_physical=target, wrong_physical=wrong, wrong_role="RAW 错误 reference",
                          raw_root=str(WORK / "raw"), roma_root=str(WORK / "roma"),
                          raw_payload=str(WORK / "raw" / f"shard{shard:02d}" / "payload.pt"),
                          roma_payload=str(WORK / "roma" / f"shard{shard:02d}" / "payload.pt"),
                          prediction_payload=str(WORK / "predictions" / f"shard{shard:02d}" / "payload.json"),
                          candidate_source=result["candidate_source"], action="SWITCH", max_logit=logit,
                          max_logit_hex=logit.hex(), head=head, selection_rule=RULE,
                          query_grid=m["query_grid_shape"], processor_input_frame=m["processor_input_frame"],
                          evidence=evidence, image_provenance=prov,
                          inference_calls=0, training_updates=0, clinical_diagnosis_claimed=False))
    payload = dict(status="ISIC30_RESCUE_SOURCE_SELECTION_AND_NATIVE_CACHE_VALIDATION_PASS",
                   dataset="isic", selected_count=len(cases), eligible_count_before_exclusion=len(all_rescues),
                   eligible_count=len(eligible), excluded_previously_shown_queries=EXCLUDED,
                   selected_lesion_count=len({c["identity"] for c in cases}),
                   selected_patient_count=len({c["component"] for c in cases}), selection_rule=RULE,
                   population_counts={"RAW": result["counts"]["RAW"], "COST1": result["counts"]["COST1"], "queries": 537},
                   selected_isic_license_values=dict(Counter(p["copyright_license"] for c in cases for p in c["image_provenance"].values())),
                   license_statement="Per-image existing local metadata retained for all three image roles; this is not a blanket licensing or publication clearance.",
                   renderer_load_notes="Same raw/roma shard payload contract as render_rc_new_hyp_visibility_cases_v1.extract. Visibility maps are float64 native token grids; query input frame EXIF_ORIENTED_BEFORE_RESIZE. max_logit is the sealed COST1 challenger logit, not a map statistic.",
                   selected=cases, sources=SOURCES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ["selected", "sources"]}, ensure_ascii=False, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
