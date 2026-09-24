#!/usr/bin/env python3
"""Validate the English edition against the completed Chinese research bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import verify_rc_cost1_rescue90_independent_v1 as BASE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/figures/new_hyp_rescue90_en_20260924_v1"
DEFAULT_SOURCE = ROOT / "reports/figures/new_hyp_rescue90_20260924_v1"
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out=DEFAULT_OUT, source=DEFAULT_SOURCE):
    out, source = Path(out).resolve(), Path(source).resolve()
    assert out != source
    original = json.loads((source / "cases_manifest.json").read_text())
    translated = json.loads((out / "cases_manifest.json").read_text())
    assert len(original) == len(translated) == 90
    assert [(c["dataset"], c["query_id"]) for c in original] == [(c["dataset"], c["query_id"]) for c in translated]
    invariant_fields = [
        "number", "dataset", "model", "query_id", "display_id", "source_query_id",
        "identity", "raw_correct", "final_correct", "raw_selected", "final_selected",
        "target_physical", "wrong_physical", "raw_payload", "roma_payload",
        "sealed_COST1_logits_hex", "challenger_physical_rows", "selected_logit",
        "scores128_physical_axis", "scores128_hex", "raw_hold_utility", "action",
        "query_grid", "processor_input_frame", "evidence", "stem", "relative_stem",
    ]
    arrays_checked, photos_checked = 0, 0
    for old, new in zip(original, translated):
        for field in invariant_fields:
            assert new[field] == old[field], (old["query_id"], field)
        if old["dataset"] == "isic":
            assert new["image_provenance"] == old["image_provenance"]
        for field in ("dataset_label", "panel", "wrong_role"):
            assert not CJK.search(new[field]), (new["query_id"], field)
        relative = Path(old["dataset"]) / "data" / (old["stem"] + ".npz")
        with np.load(source / relative) as a, np.load(out / relative) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                assert a[key].tobytes() == b[key].tobytes(), (old["query_id"], key)
                arrays_checked += 1
        for role in ("query", "wrong", "target"):
            relative = Path(old["dataset"]) / "assets" / (old["stem"] + "_" + role + ".jpg")
            assert sha(source / relative) == sha(out / relative), relative
            photos_checked += 1
        # Matplotlib SVG retains text as XML text or human-readable comments
        # even when glyphs are exported as outlines.
        svg = (out / (new["relative_stem"] + ".svg")).read_text()
        assert not CJK.search(svg), (new["query_id"], "CJK remains in figure SVG")
    assert arrays_checked == 360 and photos_checked == 270
    for filename in ("index.html", "README.md"):
        text = (out / filename).read_text()
        assert not CJK.search(text), (filename, "CJK remains in English presentation")

    BASE.OUT = out
    BASE.main()
    report_path = out / "independent_validation.json"
    report = json.loads(report_path.read_text())
    report.update(
        language="English",
        english_figure_SVG_labels_without_CJK=True,
        english_index_and_readme_without_CJK=True,
        chinese_source_bundle=str(source),
        chinese_source_manifest_sha256=sha(source / "cases_manifest.json"),
        original_case_order_and_decisions_preserved=True,
        original_native_arrays_byte_identical=arrays_checked,
        original_photo_assets_byte_identical=photos_checked,
        english_validator=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "language", "cases", "datasets", "original_case_order_and_decisions_preserved", "original_native_arrays_byte_identical", "original_photo_assets_byte_identical")}, indent=2))
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    main(args.output, args.source)
