#!/usr/bin/env python3
"""Compare actual SVG reference image dimensions in the Chinese and English bundles."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/figures/new_hyp_rescue90_20260924_v1"
OUT = ROOT / "reports/figures/new_hyp_rescue90_en_20260924_v1"
NS = {"svg": "http://www.w3.org/2000/svg"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def images(path):
    tree = ET.parse(path)
    page = tree.getroot()
    result = {}
    for role, axis in [("wrong", "axes_5"), ("target", "axes_6")]:
        groups = tree.findall(f'.//svg:g[@id="{axis}"]', NS)
        assert len(groups) == 1, (path, axis, "unique reference axes")
        found = groups[0].findall(".//svg:image", NS)
        assert len(found) == 1, (path, axis, "unique reference image")
        image = found[0]
        width, height = float(image.attrib["width"]), float(image.attrib["height"])
        assert width > 0 and height > 0
        result[role] = dict(axis_id=axis, width=width, height=height)
    return {k: page.attrib[k] for k in ["width", "height", "viewBox"]}, result


def main():
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["enlarged_reference_panels"] is True
    old = json.loads((SOURCE / "cases_manifest.json").read_text())
    new = json.loads((OUT / "cases_manifest.json").read_text())
    assert len(old) == len(new) == 90
    assert [(c["dataset"], c["query_id"]) for c in old] == [(c["dataset"], c["query_id"]) for c in new]
    rows, widths, heights = [], [], []
    for before, after in zip(old, new):
        assert before["relative_stem"] == after["relative_stem"]
        relative = before["relative_stem"] + ".svg"
        source_page, source_images = images(SOURCE / relative)
        output_page, output_images = images(OUT / relative)
        assert source_page == output_page, (relative, "SVG page geometry changed")
        pairs = {}
        for role in ("wrong", "target"):
            a, b = source_images[role], output_images[role]
            wr, hr = b["width"] / a["width"], b["height"] / a["height"]
            assert wr >= 2.0 and hr >= 2.0, (relative, role, wr, hr)
            widths.append(wr)
            heights.append(hr)
            pairs[role] = dict(source=a, enlarged=b, width_ratio=wr, height_ratio=hr)
        rows.append(dict(dataset=after["dataset"], query_id=after["query_id"], svg=relative,
                         source_svg_sha256=sha(SOURCE / relative), english_svg_sha256=sha(OUT / relative),
                         reference_panels=pairs, status="PASS"))
    assert len(widths) == len(heights) == 180
    report = dict(status="ENGLISH90_REFERENCE_IMAGE_SVG_ENLARGEMENT_INDEPENDENT_PASS",
                  cases=90, reference_images=180,
                  compared_axes={"wrong": "axes_5", "target": "axes_6"},
                  measurement="Actual SVG image element width and height on identical page geometry",
                  all_width_and_height_ratios_at_least=2.0,
                  minimum_width_ratio=min(widths), maximum_width_ratio=max(widths),
                  minimum_height_ratio=min(heights), maximum_height_ratio=max(heights),
                  source_manifest_sha256=sha(SOURCE / "cases_manifest.json"),
                  english_manifest_sha256=sha(OUT / "cases_manifest.json"),
                  validator={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                  per_case=rows)
    path = OUT / "reference_enlargement_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
