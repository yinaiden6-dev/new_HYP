"""CPU-only, resumable all-query frozen-token audit.

Writes only the selected output directory. OCR-derived regions are provisional
measurements, never a claim that an image was visually reviewed. Original cache
vectors, effective-token masking, mean MaxSim, stable row ties and SHA grouping
are preserved. No encoder, GPU, trained model, or generation is used.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.scoring import fuse, tokenize
from scripts.analyze_colnomic_tokens import display_xy, token_hash
from scripts.colnomic_mechanism.common import data, verify_split_records
from scripts.colnomic_mechanism.display_pixels import display_image

SCHEMA = "colnomic.all_query_frozen_token_audit.v1"
VERIFIED_IMAGES = {}
REGIONS = ("key_numbers_specs", "difference_layout", "common_text", "pattern_color",
           "subject_other", "background", "template")
INPUTS = (
    "cache/image_embeddings/colnomic7b/photos.pt",
    "cache/image_embeddings/colnomic7b/references.pt",
    "cache/scores/colnomic7b/image.pt", "cache/scores/bm25/bm25.pt",
    "cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt",
    "cache/ocr/photos.json", "cache/ocr/references.json",
    "dataset/lists/photos.csv", "dataset/lists/references.csv",
    "dataset/lists/positives.csv", "dataset/lists/groups.csv",
    "dataset/lists/train.csv", "dataset/lists/val.csv", "dataset/lists/test.csv",
    "dataset/annotations/subject_boxes.json",
    "results/token_mechanism_analysis/F03/reproduction/token_metadata.json",
    "results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json",
    "results/colnomic7b/image/predictions.jsonl",
    "results/bm25_image_fusion/colnomic7b/alpha_0.5/predictions.jsonl",
    "results/token_mechanism_analysis/B03/summary.json",
    "results/token_mechanism_analysis/F03/summary.json",
    "results/token_mechanism_experiments/20261013_v1/stage2_geometry/alignments.json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n")


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def csv_text(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                         else v for k, v in row.items()})
    return stream.getvalue()


def compressed_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(csv_text(rows).encode())
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def stable_top_two(matrix, raw_reference_indices):
    """First raw-index tie wins for each maximum, including the runner-up."""
    assert matrix.ndim == 2 and matrix.shape[1] >= 2
    top = np.argmax(matrix, axis=1)
    rows = np.arange(len(matrix))
    best = matrix[rows, top].copy()
    second_matrix = matrix.copy()
    second_matrix[rows, top] = -np.inf
    second = np.argmax(second_matrix, axis=1)
    runner = matrix[rows, second].copy()
    return raw_reference_indices[top], best, raw_reference_indices[second], runner


def unique_order(scores, hashes):
    seen = set()
    result = []
    for index in np.argsort(-scores, kind="stable"):
        digest = hashes[int(index)]
        if digest not in seen:
            seen.add(digest)
            result.append(int(index))
    return np.asarray(result, dtype=np.int32)


class TokenRecord:
    """Geometry and OCR metadata without interpreting token cells as words."""
    def __init__(self, bank, spec, ocr, root, subject_box=None):
        self.spec, self.ocr = spec, ocr
        self.id = spec["id"]
        index = spec["bank_index"]
        assert bank["ids"][index] == self.id
        assert bank["sha256"][index] == spec["image_sha256"] == ocr["image_sha256"]
        self.tensor = bank["embeddings"][index]
        assert list(self.tensor.shape) == spec["shape"]
        assert torch.isfinite(self.tensor).all()
        assert token_hash(self.tensor) == spec["all_tokens_sha256"]
        self.valid = np.flatnonzero(self.tensor.float().norm(dim=1).numpy() > 1e-8)
        assert self.valid.tolist() == spec["valid_token_indices"]
        self.path = root / bank["paths"][index]
        self.box = subject_box
        if subject_box is not None:
            assert subject_box["image_sha256"] == spec["image_sha256"]
            assert subject_box["status"] == "confirmed"
            self.box = np.asarray(subject_box["bbox"], np.float32) / np.tile(
                [subject_box["width"], subject_box["height"]], 2)
        layout = spec["layout"]
        k = np.arange(layout["count"])
        xy = np.stack(((k % layout["width"] + .5) / layout["width"],
                       (k // layout["width"] + .5) / layout["height"]), 1)
        self.xy = np.full((len(self.tensor), 2), np.nan)
        self.xy[layout["start"]:layout["start"] + layout["count"]] = display_xy(
            xy, spec["exif_orientation"])
        self.lines = np.full(len(self.tensor), -1, np.int32)
        self.overlap = np.zeros(len(self.tensor))
        self.multiple = np.zeros(len(self.tensor), np.int32)
        self.line_ids = [[] for _ in range(len(self.tensor))]
        size = np.asarray([spec["display_width"], spec["display_height"]], np.float32)
        polys = [np.asarray(line["polygon"], np.float32) / size for line in ocr["lines"]]
        self.line_polygons=polys
        for raw, row, col in zip(range(layout["start"], layout["start"] + layout["count"]),
                                  k // layout["width"], k % layout["width"]):
            corners = np.asarray([[col/layout["width"], row/layout["height"]],
                                  [(col+1)/layout["width"], row/layout["height"]],
                                  [(col+1)/layout["width"], (row+1)/layout["height"]],
                                  [col/layout["width"], (row+1)/layout["height"]]])
            polygon = display_xy(corners, spec["exif_orientation"]).astype(np.float32)
            area = abs(cv2.contourArea(polygon))
            overlaps = np.asarray([cv2.intersectConvexConvex(polygon, p)[0] / area for p in polys])
            detected = np.flatnonzero(overlaps >= .05)
            self.line_ids[raw] = detected.tolist()
            self.multiple[raw] = len(detected)
            if len(detected):
                best = int(overlaps.argmax())
                self.lines[raw], self.overlap[raw] = best, float(overlaps[best])

    def info(self, index):
        index = int(index)
        line = int(self.lines[index])
        xy = self.xy[index]
        visual = bool(np.isfinite(xy).all())
        return dict(index=index, type="visual" if visual else "template",
                    x=float(xy[0]) if visual else None, y=float(xy[1]) if visual else None,
                    ocr_line=line, ocr_text=self.ocr["lines"][line]["text"] if line >= 0 else "",
                    ocr_overlap=float(self.overlap[index]), ocr_candidate_lines=self.line_ids[index],
                    spans_multiple_ocr_regions=bool(self.multiple[index] > 1))

    def subject(self, index):
        return (self.box is not None and np.isfinite(self.xy[index]).all() and
                bool(((self.xy[index] >= self.box[:2]) & (self.xy[index] < self.box[2:])).all()))


def region_for(query, index, positive_text, negative_text):
    info = query.info(index)
    labels = []
    if info["type"] == "template":
        return "template", labels, "Nonzero attended nonvisual/template token; no spatial coordinate."
    if not query.subject(index):
        return "background", labels, "Confirmed subject-box center rule; edge cells may straddle subject."
    tokens = set(tokenize(info["ocr_text"]))
    common = tokens & positive_text & negative_text
    if common:
        labels.append("overlaps_ocr_line_with_shared_candidate_words")
    numerical = bool(re.search(r"\d", info["ocr_text"]))
    if numerical:
        labels.append("overlaps_ocr_line_containing_numbers")
    if info["spans_multiple_ocr_regions"]:
        labels.append("multiple_ocr_line_overlap")
    if numerical:
        region = "key_numbers_specs"
    elif tokens and tokens <= (positive_text & negative_text):
        region = "common_text"
    else:
        region = "subject_other"
    return region, labels, ("Provisional OCR-line overlap, not a verified word/critical region. "
                            "Layout, color/graphics and identity relevance require visual review.")


def region_summary(rows, n):
    result = []
    for region in REGIONS:
        values = np.asarray([r["delta"] for r in rows if r["region"] == region], np.float64)
        total = float(np.abs(values).sum())
        result.append(dict(region=region, tokens=len(values),
                           positive_contribution=float(np.maximum(values, 0).sum()/n),
                           negative_contribution=float(np.minimum(values, 0).sum()/n),
                           net_contribution=float(values.sum()/n),
                           within_region_mean=float(values.mean()) if len(values) else None,
                           largest_absolute_share=float(np.abs(values).max()/total) if total else None,
                           hhi_absolute=float(np.square(np.abs(values)/total).sum()) if total else None,
                           annotation_status="unassigned_requires_visual_review" if region in
                           ["difference_layout", "pattern_color"] else "automatic_provisional"))
    assert sum(r["tokens"] for r in result) == n
    assert abs(sum(r["net_contribution"] for r in result)-np.mean([r["delta"] for r in rows])) < 2e-7
    return result


def correspondence_stats(rows, side, candidate):
    visual = [r for r in rows if r["q_type"] == "visual"]
    winners = [r[side+"_index"] for r in visual]
    counts = Counter(winners)
    candidate_visual = np.isfinite(candidate.xy[candidate.valid]).all(1).sum()
    lines = [r for r in visual if r["q_ocr_line"] >= 0 and r[side+"_ocr_line"] >= 0]
    residuals=[r[side+"_geometry_distance_patches"] for r in rows
               if r.get(side+"_geometry_distance_patches") is not None]
    line_tests=[r[side+"_different_expected_ocr_line"] for r in rows
                if r.get(side+"_different_expected_ocr_line") is not None]
    return dict(query_visual_tokens=len(visual), candidate_valid_tokens=len(candidate.valid),
                candidate_visual_tokens=int(candidate_visual), unique_target_tokens=len(counts),
                target_visual_coverage=(len({i for i in winners if np.isfinite(candidate.xy[i]).all()}) /
                                        int(candidate_visual) if candidate_visual else None),
                max_queries_per_target=max(counts.values()) if counts else 0,
                target_concentration_hhi=float(sum((v/max(1,len(winners)))**2 for v in counts.values())),
                near_ties_within_1e_4=sum(r[side+"_top1_top2_gap"] <= 1e-4 for r in rows),
                exact_ties=sum(r[side+"_top1_top2_gap"] == 0 for r in rows),
                differing_ocr_line_text=sum(r["q_ocr_text"] != r[side+"_ocr_text"] for r in lines),
                both_ocr_line_observed_tokens=len(lines),
                geometric_evaluable_tokens=len(residuals),
                geometry_distance_median_patches=float(np.median(residuals)) if residuals else None,
                geometry_over_3_patch_fraction=float(np.mean(np.asarray(residuals)>3)) if residuals else None,
                differing_expected_ocr_line_tokens=sum(line_tests),expected_ocr_line_evaluable_tokens=len(line_tests),
                packaging_face_relation=None,
                interpretation="Concentration and line differences are descriptive; candidate token count is retained. "
                "Residuals only use existing accepted transforms with explicit provenance; missing transforms remain null. "
                "Packaging face is unreviewed; no anomaly or mechanism asserted.")


def geometric_row(query,candidate,qi,ri,fit):
    result=dict(geometry_distance_patches=None,expected_x=None,expected_y=None,
                expected_ocr_lines=[],different_expected_ocr_line=None)
    if not fit or not fit.get("accepted",False) or not np.isfinite(query.xy[qi]).all() or not np.isfinite(candidate.xy[ri]).all():
        return result
    mapped=np.asarray(fit["homography"])@np.r_[query.xy[qi],1.]
    if abs(mapped[2])<1e-10:
        return result
    expected=mapped[:2]/mapped[2]
    layout=candidate.spec["layout"];width,height=layout["width"],layout["height"]
    if candidate.spec["exif_orientation"]>=5:
        width,height=height,width
    lines=[i for i,poly in enumerate(candidate.line_polygons)
           if cv2.pointPolygonTest(poly,tuple(float(v) for v in expected),False)>=0]
    different=None
    if len(lines)==1 and candidate.lines[ri]>=0 and candidate.multiple[ri]==1:
        different=bool(lines[0]!=candidate.lines[ri])
    result.update(geometry_distance_patches=float(np.linalg.norm((candidate.xy[ri]-expected)*[width,height])),
                  expected_x=float(expected[0]),expected_y=float(expected[1]),expected_ocr_lines=lines,
                  different_expected_ocr_line=different)
    return result


def fit_image(image, width, height):
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(image, ((width-image.width)//2, (height-image.height)//2))
    return panel


def atomic_jpeg(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    image.save(temp, format="JPEG", quality=90)
    temp.replace(path)


def asset_preview(record, out):
    path = out/"assets"/(record.id+".jpg")
    if not record.path.is_file():
        if not path.exists():
            missing=Image.new("RGB",(700,500),"white")
            ImageDraw.Draw(missing).text((20,20),"MISSING ORIGINAL IMAGE: "+record.id,fill="red")
            atomic_jpeg(path,missing)
        VERIFIED_IMAGES[record.id]=dict(path=str(record.path),available=False,expected_sha256=record.spec["image_sha256"])
        return path
    if record.id not in VERIFIED_IMAGES:
        digest=sha(record.path)
        assert digest==record.spec["image_sha256"]
        VERIFIED_IMAGES[record.id]=dict(path=str(record.path),available=True,sha256=digest,
                                       expected_sha256=record.spec["image_sha256"])
    if path.exists():
        return path
    with Image.open(record.path) as original:
        assert list(original.size) == [record.spec["raw_width"], record.spec["raw_height"]]
        assert int(original.getexif().get(274, 1)) == record.spec["exif_orientation"]
        image = display_image(original)
        assert list(image.size) == [record.spec["display_width"], record.spec["display_height"]]
        image.thumbnail((1000,1000), Image.Resampling.LANCZOS)
        atomic_jpeg(path, image)
    return path


def comparison_preview(q, candidates, out, folder):
    selected = [("Query", q)] + candidates
    width, height = 400, 480
    image = Image.new("RGB", (width*len(selected), height+56), "#e5e7eb")
    draw = ImageDraw.Draw(image)
    for i, (role, record) in enumerate(selected):
        p = asset_preview(record, out)
        with Image.open(p) as source:
            image.paste(fit_image(source, width-8, height-4), (i*width+4,52))
        draw.text((i*width+8,8), role, fill="black")
        draw.text((i*width+8,28), record.id, fill="black")
    path = folder/"evidence.jpg"
    atomic_jpeg(path,image)
    return path


def match_svg(query, candidate, selected, side):
    """Actual normalized display coordinates; line overlay is illustrative."""
    parts = ["<svg viewBox='0 0 800 400' width='100%' role='img'>",
             f"<image href='../../assets/{query.id}.jpg' x='0' y='0' width='360' height='400' preserveAspectRatio='none'/>",
             f"<image href='../../assets/{candidate.id}.jpg' x='440' y='0' width='360' height='400' preserveAspectRatio='none'/>"]
    for row in selected:
        qx,qy,rx,ry = row["q_x"],row["q_y"],row[side+"_x"],row[side+"_y"]
        if any(value is None for value in [qx,qy,rx,ry]):
            continue
        color = "#15803d" if row["delta"] > 0 else "#dc2626"
        parts.append(f"<line x1='{qx*360:.2f}' y1='{qy*400:.2f}' x2='{440+rx*360:.2f}' "
                     f"y2='{ry*400:.2f}' stroke='{color}' stroke-opacity='.65' stroke-width='1.2'/>")
        parts.append(f"<circle cx='{qx*360:.2f}' cy='{qy*400:.2f}' r='3' fill='{color}'/>")
    parts.append("</svg>")
    return "".join(parts)


CSS = "body{font:15px system-ui;margin:24px;max-width:1400px;color:#172033}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:5px;font-size:12px}th{background:#eef2ff}img{max-width:100%}a{color:#1d4ed8}.notice{padding:14px;background:#fff7dc}details{margin:20px 0}.positive{color:#15803d}.negative{color:#b91c1c}"


def query_html(folder, dossier, comparisons, query, records, out):
    h = html.escape
    body = ["<!doctype html><meta charset='utf-8'>",f"<title>{query.id}</title><style>{CSS}</style>",
            "<a href='../../index.html'>筛选索引</a>",f"<h1>{query.id}</h1>",
            "<p class='notice'>视觉审核状态：尚未逐图查看。此页仅含冻结分数、确定性贡献和自动 OCR 关联。"
            "数字/公共文字区域是暂定代理；布局、图案、颜色、面与关键身份信息均未确认。不得把本页统计写成已查明原因。</p>",
            f"<p>身份 {h(dossier['identity'])}；来源组 {h(dossier['group_id'])}；split {dossier['split']}；"
            f"客观结果 {h(dossier['outcome'])}；图像正确候选排名 {dossier['image_positive_rank']}，"
            f"融合正确候选排名 {dossier['fusion_positive_rank']}。</p>",
            "<p><a href='dossier.json'>全部正确图库记录、候选和实际分数</a> · "
            "<a href='comparisons.json'>区域守恒与对应统计</a> · "
            "<a href='evidence.jpg'>放大对照图</a></p><img src='evidence.jpg'>"]
    for summary,rows in comparisons:
        p,n = summary["positive"],summary["negative"]
        body += [f"<h2>{p} vs {n}</h2>",
                 f"<p>来源：{h(','.join(summary['negative_roles']))}；原始图像差额 {summary['image_gap']:+.8f}；"
                 f"逐 Token Δ 均值 {summary['delta_mean']:+.8f}；分差复放误差 {summary['delta_conservation_error']:.3g}。"
                 f"{h(summary['contribution_pattern'])}</p>",
                 f"<p><a href='{summary['token_file']}'>全部逐 Token 数据 CSV.gz</a> · "
                 f"<a href='../../{summary['positive_matrix']}'>正候选完整 FP32 矩阵及有效轴</a> · "
                 f"<a href='../../{summary['negative_matrix']}'>负候选完整 FP32 矩阵及有效轴</a></p>",
                 "<table><tr><th>互斥区域（自动暂定）</th><th>Token数</th><th>正贡献</th><th>负贡献</th><th>净贡献</th></tr>"]
        for region in summary["regions"]:
            body.append(f"<tr><td>{region['region']}</td><td>{region['tokens']}</td>"
                        f"<td>{region['positive_contribution']:+.7f}</td><td>{region['negative_contribution']:+.7f}</td>"
                        f"<td>{region['net_contribution']:+.7f}</td></tr>")
        body.append("</table><p>连线分别示例最大正负贡献主体 Token（各最多 8 个）。绿色支持正确、红色支持错误。"
                    "显示图按规范坐标拉伸以对应缓存坐标；连线位移、跨行或集中本身不证明错配。完整对应在 CSV。</p>")
        visual = [row for row in rows if query.subject(row["q_index"])]
        chosen = sorted(visual,key=lambda r:r["delta"])[:8]+sorted(visual,key=lambda r:r["delta"])[-8:]
        chosen = list({row["q_index"]:row for row in chosen}.values())
        body += [match_svg(query,records[p],chosen,"pos"),match_svg(query,records[n],chosen,"neg")]
        selected = sorted(rows,key=lambda r:abs(r["delta"]),reverse=True)[:48]
        body += ["<details open><summary>最大绝对贡献 48 个 Token 的匹配解释</summary><table><tr>"
                 "<th>原始Q索引 / 类型 / 坐标</th><th>区域 / Q OCR</th><th>正确匹配索引 / OCR / sim</th>"
                 "<th>错误匹配索引 / OCR / sim</th><th>正/负 top1−top2</th><th>Δ</th><th>不确定性</th></tr>"]
        for row in selected:
            body.append(f"<tr><td>{row['q_index']} / {row['q_type']} / {row['q_x']},{row['q_y']}</td>"
                        f"<td>{row['region']}<br>{h(row['q_ocr_text'])}</td>"
                        f"<td>{row['pos_index']}<br>{h(row['pos_ocr_text'])}<br>{row['pos_sim']:.6f}</td>"
                        f"<td>{row['neg_index']}<br>{h(row['neg_ocr_text'])}<br>{row['neg_sim']:.6f}</td>"
                        f"<td>{row['pos_top1_top2_gap']:.6f}/{row['neg_top1_top2_gap']:.6f}</td>"
                        f"<td>{row['delta']:+.6f}</td><td>{h(row['region_uncertainty'])}</td></tr>")
        body.append("</table></details>")
    atomic_text(folder/"index.html","\n".join(body))


class Audit:
    def __init__(self, out):
        self.out = out
        self.d = data()
        self.qb,self.rb = self.d["qb"],self.d["rb"]
        self.qindex = {qid:i for i,qid in enumerate(self.qb["ids"])}
        self.rindex = {rid:i for i,rid in enumerate(self.rb["ids"])}
        self.qspec = {r["id"]:r for r in self.d["metadata"]["photos"]}
        self.rspec = {r["id"]:r for r in self.d["metadata"]["references"]}
        self.reference_rows = {r["reference_id"]:r for r in csv_rows(ROOT/"dataset/lists/references.csv")}
        self.positives = defaultdict(list)
        for r in csv_rows(ROOT/"dataset/lists/positives.csv"):
            self.positives[r["package_id"]].append(r["reference_id"])
        self.aliases = defaultdict(list)
        for rid,digest in zip(self.rb["ids"],self.rb["sha256"]):
            self.aliases[digest].append(rid)
        self.ledger = {r["photo_id"]:r for r in read(ROOT/INPUTS[16])}
        assert len(self.qb["ids"]) == 1117 and len(self.qindex) == 1117
        assert len(self.rb["ids"]) == 5405 and len(self.rindex) == 5405
        assert len(self.aliases) == 5330
        self.old_matrices = {}
        self.geometry={}
        for fit in read(ROOT/INPUTS[21]):
            self.geometry[(fit["photo_id"],fit["reference_id"])]=dict(**fit,accepted=True,
                         method="previous full-gallery OCR-text-anchor homography; includes text information")
        for cfg in ["F03","B03"]:
            folder = ROOT/"results/token_mechanism_analysis"/cfg
            for contrast in read(folder/"summary.json")["cases"]:
                for rid,fit in zip([contrast["positive"],contrast["negative"]],contrast["geometry"]):
                    if fit.get("accepted"):
                        self.geometry[(contrast["photo_id"],rid)]=fit
            for row in read(folder/"reproduction/pair_index.json"):
                self.old_matrices[(row["photo_id"],row["reference_id"])] = (folder/row["matrix_file"],row)
        self.rankings = None
        self.uses = Counter()

    def protocol(self):
        return dict(schema=SCHEMA,scope=dict(queries=1117,reference_records=5405,sha_designs=5330),
                    original_score="FP16 frozen vectors; FP32 dot product; valid norm>1e-8; mean MaxSim including nonzero templates",
                    sorting="stable descending full-gallery row scores, SHA dedup after sorting; first row wins exact ties",
                    positives="positives.csv package relation, expand every positive image SHA alias; primary highest original image score then stable bank row",
                    contrasts="every correct candidate record versus unique image/fusion strongest wrong; identical candidate matrices shared",
                    region_rule="mutually exclusive center/confirmed-box; OCR numeric line first, fully common-word line second; remaining subject unknown",
                    unassigned_regions=["difference_layout","pattern_color"],
                    region_warning="automatic OCR-line proxies, not word cells, visual review, critical-region labels or mechanism proof",
                    visual_review_status="unreviewed",ranking_scope="complete gallery",device="CPU",version=1)

    def inputs(self):
        records=[]
        for relative in INPUTS:
            path=ROOT/relative
            records.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha(path)))
        expected=dict(protocol=self.protocol(),inputs=records)
        path=self.out/"input_manifest.json"
        if path.exists():
            original=read(path)
            assert original["protocol"]==expected["protocol"]
            assert original["inputs"]==records[:len(original["inputs"])],"Original source input changed."
            supplement=records[len(original["inputs"]):]
            if supplement:
                extra=self.out/"supplemental_input_manifest.json"
                if extra.exists():
                    assert read(extra)["inputs"]==supplement
                else:
                    write(extra,dict(inputs=supplement,purpose="Existing accepted geometry provenance for additional token diagnostics; frozen primary scores unchanged."))
        else:
            write(path,expected)
        write(self.out/"split_verification.json",self.d["split_verification"])
        write(self.out/"protocol.json",self.protocol())

    def global_outputs(self):
        img=self.d["image"]["scores"].numpy(); bm=self.d["bm25"]["scores"].numpy();fu=self.d["fusion"]["scores"].numpy()
        has_text=[bool(tokenize(self.d["qo"][qid]["text"])) for qid in self.qb["ids"]]
        rebuilt=fuse(self.d["image"]["scores"],self.d["bm25"]["scores"],has_text,.5).numpy()
        error=float(np.max(abs(rebuilt-fu)));assert error<2e-6
        scores_path=self.out/"full_gallery_scores.npz"
        if not scores_path.exists():
            atomic_npz(scores_path,image=img,bm25=bm,fusion=fu,
                       query_ids=np.asarray(self.qb["ids"]),reference_ids=np.asarray(self.rb["ids"]),
                       reference_sha256=np.asarray(self.rb["sha256"]))
        else:
            with np.load(scores_path) as z:
                assert all(np.array_equal(z[key],arr) for key,arr in [("image",img),("bm25",bm),("fusion",fu)])
                assert z["query_ids"].tolist()==self.qb["ids"] and z["reference_ids"].tolist()==self.rb["ids"]
        path=self.out/"full_gallery_rankings.npz"
        if not path.exists():
            orders={key:np.stack([unique_order(row,self.rb["sha256"]) for row in values])
                    for key,values in [("image",img),("bm25",bm),("fusion",fu)]}
            atomic_npz(path,**orders,query_ids=np.asarray(self.qb["ids"]),reference_ids=np.asarray(self.rb["ids"]))
        with np.load(path) as z:
            self.rankings={key:z[key] for key in ["image","bm25","fusion"]}
        predictions={key:{r["photo_id"]:r for r in [json.loads(line) for line in (ROOT/relative).read_text().splitlines()]}
                     for key,relative in [("image",INPUTS[17]),("fusion",INPUTS[18])]}
        self.original_predictions=predictions
        joint=Counter()
        for qi,qid in enumerate(self.qb["ids"]):
            record=self.d["photos"][qid]
            labels=self.positives[record["package_id"]]
            hashes={self.rb["sha256"][self.rindex[rid]] for rid in labels}
            assert hashes==set(self.ledger[qid]["positive_reference_sha256"])
            correct=[]
            for key in ["image","fusion"]:
                order=self.rankings[key][qi]
                assert len(order)==5330 and len({self.rb["sha256"][i] for i in order})==5330
                rank=next(k+1 for k,i in enumerate(order) if self.rb["sha256"][i] in hashes)
                old=predictions[key][qid]
                assert rank==old["rank"] and self.rb["ids"][order[0]]==old["selected_reference_id"]
                correct.append(rank==1)
            joint[tuple(correct)]+=1
        result=dict(query_count=len(self.qb["ids"]),reference_count=len(self.rb["ids"]),sha_design_count=len(self.aliases),
                    joint_counts={f"image_{a}_fusion_{b}":v for (a,b),v in joint.items()},
                    fusion_formula_replay_max_error=error,all_original_image_and_fusion_rankings_reproduced=True,
                    axes_verified=True,full_gallery_scalar_scores_preserved=True,
                    note="These assertions replay cached full-gallery ranks. Token dot score replay below covers selected key candidates, not all 6,037,385 token matrices.")
        write(self.out/"baseline_verification.json",result)

    def record(self,rid,is_query=False):
        return TokenRecord(self.qb if is_query else self.rb,
                           self.qspec[rid] if is_query else self.rspec[rid],
                           self.d["qo"][rid] if is_query else self.d["ro"][rid],ROOT,
                           self.d["boxes"][rid] if is_query else None)

    def matrix(self,q,r):
        path=self.out/"matrices"/q.id/(r.id+".npz")
        if path.exists():
            with np.load(path) as restored:
                matrix=restored["matrix"]
                assert np.array_equal(restored["query_raw_indices"],q.valid)
                assert np.array_equal(restored["reference_raw_indices"],r.valid)
                assert str(restored["query_token_sha256"])==q.spec["all_tokens_sha256"]
                assert str(restored["reference_token_sha256"])==r.spec["all_tokens_sha256"]
                source=str(restored["source"])
            self.uses["resumed_matrix"]+=1
        else:
            previous=self.old_matrices.get((q.id,r.id))
            if previous:
                old,row=previous
                assert sha(old)==row["sha256"],"Existing matrix checksum mismatch: "+str(old)
                raw=np.load(old,mmap_mode="r",allow_pickle=False)
                assert raw.dtype==np.float32 and list(raw.shape)==row["shape"]
                matrix=np.asarray(raw[np.ix_(q.valid,r.valid)],dtype=np.float32)
                source=str(old.relative_to(ROOT))+" sha256="+row["sha256"]
                self.uses["verified_old_matrix_reused"]+=1
            else:
                matrix=(q.tensor[q.valid].float() @ r.tensor[r.valid].float().T).numpy()
                source="computed from original frozen FP16 bank with CPU FP32 dot"
                self.uses["new_cpu_matrix"]+=1
            atomic_npz(path,matrix=matrix,query_raw_indices=q.valid,reference_raw_indices=r.valid,
                       query_token_sha256=np.asarray(q.spec["all_tokens_sha256"]),
                       reference_token_sha256=np.asarray(r.spec["all_tokens_sha256"]),source=np.asarray(source))
        assert matrix.dtype==np.float32 and np.isfinite(matrix).all()
        cached=float(self.d["image"]["scores"][self.qindex[q.id],self.rindex[r.id]])
        score=float(torch.from_numpy(matrix).max(1).values.mean())
        error=abs(score-cached);assert error<2e-6,(q.id,r.id,error)
        return matrix,dict(path=str(path.relative_to(self.out)),sha256=sha(path),
                           query=q.id,reference=r.id,shape=list(matrix.shape),dtype="float32",
                           source=source,cached_image_score=cached,replayed_score=score,replay_error=error)

    def dossier(self,qid):
        qi=self.qindex[qid];photo=self.d["photos"][qid]
        labels=self.positives[photo["package_id"]]
        digests={self.rb["sha256"][self.rindex[rid]] for rid in labels}
        positives=[rid for rid in self.rb["ids"] if self.rb["sha256"][self.rindex[rid]] in digests]
        positives=sorted(positives,key=lambda rid:(-float(self.d["image"]["scores"][qi,self.rindex[rid]]),self.rindex[rid]))
        result=dict(photo_id=qid,**{key:photo[key] for key in ["package_id","identity","design_id","source_sha256"]},
                    group_id=self.d["groups"][photo["package_id"]],split=self.d["splits"][qid],
                    original_path=str((ROOT/self.qb["paths"][qi]).resolve()),image_sha256=self.qb["sha256"][qi],
                    original_image_available=(ROOT/self.qb["paths"][qi]).is_file(),
                    raw_size=[self.qspec[qid]["raw_width"],self.qspec[qid]["raw_height"]],
                    exif_orientation=self.qspec[qid]["exif_orientation"],
                    positive_label_records=labels,positive_reference_ids=positives,positive_image_sha256=sorted(digests),
                    primary_positive=positives[0],primary_positive_rule="highest original image scalar score; stable bank row on ties",
                    visual_review_status="unreviewed",visible_face=None,identity_information_visible=None,image_quality=None,
                    direct_visual_observation=None,observed_error_types=[],supported_mechanisms=[],
                    alternative_explanations=[],counterevidence=[],unresolved=["requires actual image review and mechanism controls"],
                    evidence_level="deterministic_score_and_token_decomposition_only",
                    identity_ambiguity={key:self.ledger[qid][key] for key in
                                        ["image_strongest_wrong_identity_ambiguity","fusion_strongest_wrong_identity_ambiguity"]})
        candidates=set(positives)
        correct=[]
        for key in ["image","fusion","bm25"]:
            order=self.rankings[key][qi]
            rank_by_sha={self.rb["sha256"][int(index)]:rank+1 for rank,index in enumerate(order)}
            positive_rank=min(rank_by_sha[digest] for digest in digests)
            negative=next(int(index) for index in order if self.rb["sha256"][int(index)] not in digests)
            top=int(order[0]);score=self.d[key if key!="image" else "image"]["scores"]
            result[key+"_positive_rank"]=positive_rank
            result[key+"_top1"]=self.rb["ids"][top]
            result[key+"_strongest_wrong"]=self.rb["ids"][negative]
            result[key+"_top1_correct"]=positive_rank==1
            result[key+"_strongest_wrong_rank"]=rank_by_sha[self.rb["sha256"][negative]]
            result[key+"_positive_primary_gap"]=float(score[qi,self.rindex[positives[0]]]-score[qi,negative])
            candidates.update([self.rb["ids"][top],self.rb["ids"][negative]])
            if key in ["image","fusion"]:
                correct.append(positive_rank==1)
                assert result[key+"_top1"]==self.ledger[qid][key+"_top1"]
                assert result[key+"_strongest_wrong"]==self.ledger[qid][key+"_strongest_wrong"]
        result["outcome"]={(True,True):"both_correct",(False,True):"fusion_rescued",
                           (True,False):"fusion_harmed",(False,False):"both_wrong"}[tuple(correct)]
        control_candidates=[r for r in self.ledger.values() if r["package_id"]==photo["package_id"]
                            and r["photo_id"]!=qid and r["image_top1_correct"] and r["fusion_top1_correct"]]
        def control_key(r):
            target=self.d["boxes"][qid];other=self.d["boxes"][r["photo_id"]]
            def area(box):
                return float(np.prod(np.asarray(box["bbox"][2:])-box["bbox"][:2])/(box["width"]*box["height"]))
            return abs(area(target)-area(other)),r["photo_id"]
        result["same_identity_success_control_candidates"]=[dict(photo_id=r["photo_id"],
            image_rank=r["image_positive_rank"],fusion_rank=r["fusion_positive_rank"],face_status="unreviewed",
            confounding="Automatically prioritized by subject area only; face, viewpoint and clarity not verified.")
            for r in sorted(control_candidates,key=control_key)]
        result["verified_same_face_success_control"]=None
        result["same_face_control_status"]="requires_visual_review" if control_candidates else "no_same_identity_joint_success_found"
        result["candidates"]=[]
        for rid in sorted(candidates,key=self.rindex.get):
            ri=self.rindex[rid]
            result["candidates"].append(dict(reference_id=rid,bank_index=ri,
                        correct=self.rb["sha256"][ri] in digests,
                        explicit_positive_label=rid in labels,sha256=self.rb["sha256"][ri],
                        sha_aliases=self.aliases[self.rb["sha256"][ri]],
                        original_path=str((ROOT/self.rb["paths"][ri]).resolve()),
                        original_image_available=(ROOT/self.rb["paths"][ri]).is_file(),
                        identity=self.reference_rows[rid]["identity"],design_id=self.reference_rows[rid]["design_id"],
                        image_score=float(self.d["image"]["scores"][qi,ri]),
                        bm25_score=float(self.d["bm25"]["scores"][qi,ri]),
                        fusion_score=float(self.d["fusion"]["scores"][qi,ri])))
        return result

    def compare(self,q,p,n,pm,nm,negative_roles,pinfo,ninfo):
        pi,ps,pi2,ps2=stable_top_two(pm,p.valid)
        ni,ns,ni2,ns2=stable_top_two(nm,n.valid)
        delta=ps-ns;rows=[]
        ptext=set(tokenize(p.ocr["text"]));ntext=set(tokenize(n.ocr["text"]))
        for i,index in enumerate(q.valid):
            region,labels,uncertainty=region_for(q,int(index),ptext,ntext)
            row=dict(region=region,semantic_overlap_labels=labels,region_uncertainty=uncertainty,
                     delta=float(delta[i]),contribution_to_mean_gap=float(delta[i]/len(delta)))
            for prefix,record,raw in [("q",q,index),("pos",p,pi[i]),("neg",n,ni[i]),
                                       ("pos_second",p,pi2[i]),("neg_second",n,ni2[i])]:
                row.update({prefix+"_"+key:value for key,value in record.info(raw).items()})
            row.update(pos_sim=float(ps[i]),neg_sim=float(ns[i]),pos_second_sim=float(ps2[i]),
                       neg_second_sim=float(ns2[i]),pos_top1_top2_gap=float(ps[i]-ps2[i]),
                       neg_top1_top2_gap=float(ns[i]-ns2[i]))
            for prefix,candidate,match in [("pos",p,pi[i]),("neg",n,ni[i])]:
                residual=geometric_row(q,candidate,int(index),int(match),self.geometry.get((q.id,candidate.id)))
                row.update({prefix+"_"+key:value for key,value in residual.items()})
            rows.append(row)
        regions=region_summary(rows,len(q.valid))
        qi=self.qindex[q.id];pp=self.rindex[p.id];nn=self.rindex[n.id]
        image_gap=float(self.d["image"]["scores"][qi,pp]-self.d["image"]["scores"][qi,nn])
        error=abs(float(delta.mean())-image_gap);assert error<2e-6
        numeric=next(r for r in regions if r["region"]=="key_numbers_specs")
        if numeric["tokens"]==0:
            pattern="无可自动分配的数字规格区域；关键布局/图案仍未核验。"
        elif numeric["net_contribution"]>0 and image_gap<0:
            pattern="自动数字行区域净支持正确候选，但其他区域抵消；该区域是否真正关键尚未视觉确认。"
        elif numeric["net_contribution"]<=0:
            pattern="自动数字行区域没有形成正确优势；它不是已确认的关键字区域。"
        else:
            pattern="自动数字行区域与总分都支持正确候选；不构成布局机制证据。"
        row=dict(query=q.id,positive=p.id,negative=n.id,negative_roles=negative_roles,
                 image_gap=image_gap,bm25_gap=float(self.d["bm25"]["scores"][qi,pp]-self.d["bm25"]["scores"][qi,nn]),
                 fusion_gap=float(self.d["fusion"]["scores"][qi,pp]-self.d["fusion"]["scores"][qi,nn]),
                 delta_mean=float(delta.mean()),delta_conservation_error=error,
                 region_conservation_error=abs(sum(r["net_contribution"] for r in regions)-image_gap),
                 positive_contribution=float(np.maximum(delta,0).sum()/len(delta)),
                 negative_contribution=float(np.minimum(delta,0).sum()/len(delta)),
                 valid_query_tokens=len(q.valid),regions=regions,
                 positive_correspondence=correspondence_stats(rows,"pos",p),
                 negative_correspondence=correspondence_stats(rows,"neg",n),
                 positive_geometry=self.geometry.get((q.id,p.id)),negative_geometry=self.geometry.get((q.id,n.id)),
                 contribution_pattern=pattern,positive_matrix=pinfo["path"],negative_matrix=ninfo["path"],
                 token_file=p.id+"__vs__"+n.id+".csv.gz")
        return row,rows

    def validate_done(self,qid):
        path=self.out/"queries"/qid/"complete.json"
        if not path.exists():
            return False
        commit=read(path)
        assert commit["photo_id"]==qid and commit["schema"]==SCHEMA
        for row in commit["files"]:
            file=self.out/row["path"]
            assert file.exists() and file.stat().st_size==row["bytes"] and sha(file)==row["sha256"],str(file)
        return True

    def run_query(self,qid):
        if self.validate_done(qid):
            self.uses["completed_query_reused"]+=1
            return
        folder=self.out/"queries"/qid;folder.mkdir(parents=True,exist_ok=True)
        dossier=self.dossier(qid);q=self.record(qid,True)
        roles=defaultdict(list)
        for key in ["image","fusion"]:
            roles[dossier[key+"_strongest_wrong"]].append(key+"_strongest_wrong")
        ids=list(dict.fromkeys(dossier["positive_reference_ids"]+list(roles)))
        records={rid:self.record(rid) for rid in ids}
        matrices={};matrix_index=[]
        for rid,record in records.items():
            matrix,index=self.matrix(q,record)
            matrices[rid]=(matrix,index);matrix_index.append(index)
            asset_preview(record,self.out)
        previews=[("Correct primary",records[dossier["primary_positive"]])]
        for rid,kinds in roles.items():
            previews.append(("+".join(k.split("_")[0] for k in kinds)+" strongest wrong",records[rid]))
        comparison_preview(q,previews,self.out,folder)
        comparisons=[]
        for positive in dossier["positive_reference_ids"]:
            for negative,kinds in roles.items():
                pmat,pindex=matrices[positive];nmat,nindex=matrices[negative]
                summary,rows=self.compare(q,records[positive],records[negative],pmat,nmat,kinds,pindex,nindex)
                compressed_csv(folder/summary["token_file"],rows)
                comparisons.append((summary,rows))
        write(folder/"dossier.json",dossier)
        write(folder/"comparisons.json",[r for r,_ in comparisons])
        write(folder/"matrix_index.json",matrix_index)
        query_html(folder,dossier,comparisons,q,records,self.out)
        files=list(folder.glob("*"))
        files += [self.out/index["path"] for index in matrix_index]
        files += [self.out/"assets"/(rid+".jpg") for rid in [qid]+ids]
        manifest=[dict(path=str(path.relative_to(self.out)),bytes=path.stat().st_size,sha256=sha(path))
                  for path in sorted(set(files)) if path.is_file() and path.name!="complete.json" and not path.name.endswith(".tmp")]
        write(folder/"complete.json",dict(schema=SCHEMA,photo_id=qid,completed_utc=now(),
                  contrasts=len(comparisons),matrix_count=len(matrix_index),files=manifest))

    def checkpoint(self,completed):
        write(self.out/"progress.json",dict(schema=SCHEMA,completed_queries=completed,total_queries=1117,
                  last_valid_query=self.qb["ids"][completed-1] if completed else None,
                  complete=completed==1117,updated_utc=now(),reuse_counts=dict(self.uses),
                  visual_reviewed_queries=0,visual_review_status="material_generated_only"))

    def contacts(self,qids,first_index=0):
        index=[]
        labels=["Query original","Query subject crop","Correct primary","Image strongest wrong","Fusion strongest wrong"]
        for start in range(0,len(qids),4):
            group=qids[start:start+4]
            sheet=Image.new("RGB",(2000,4*530),"#e5e7eb")
            draw=ImageDraw.Draw(sheet)
            for i,qid in enumerate(group):
                dossier=read(self.out/"queries"/qid/"dossier.json")
                photo_path=ROOT/self.qb["paths"][self.qindex[qid]]
                if photo_path.is_file():
                    with Image.open(photo_path) as raw:
                        original=display_image(raw)
                        crop=original.crop(tuple(self.d["boxes"][qid]["bbox"]))
                else:
                    original=Image.new("RGB",(500,500),"white");crop=original.copy()
                panels=[original,crop]
                for rid in [dossier["primary_positive"],dossier["image_strongest_wrong"],dossier["fusion_strongest_wrong"]]:
                    asset=self.out/"assets"/(rid+".jpg")
                    assert asset.is_file()
                    with Image.open(asset) as preview:
                        panels.append(preview.copy())
                title=f"{first_index+start+i+1}/1117 {qid} {dossier['identity']} {dossier['split']} image rank={dossier['image_positive_rank']} fusion rank={dossier['fusion_positive_rank']} UNREVIEWED"
                draw.text((8,i*530+5),title,fill="black")
                for col,(label,panel) in enumerate(zip(labels,panels)):
                    draw.text((col*400+8,i*530+27),label,fill="black")
                    sheet.paste(fit_image(panel,392,476),(col*400+4,i*530+49))
            batch=(first_index+start)//4+1
            path=self.out/"visual_material"/f"batch_{batch:03d}.jpg"
            old_index=read(self.out/"contact_sheet_index.json") if (self.out/"contact_sheet_index.json").exists() else []
            old=next((r for r in old_index if r["batch"]==batch),None)
            if path.exists() and old and old["queries"]!=group:
                path=path.with_name(path.stem+"_complete.jpg")
            if not path.exists():
                atomic_jpeg(path,sheet)
            index.append(dict(batch=batch,queries=group,path=str(path.relative_to(self.out)),
                              visual_review_status="unreviewed",sha256=sha(path)))
        previous=read(self.out/"contact_sheet_index.json") if (self.out/"contact_sheet_index.json").exists() else []
        merged={r["batch"]:r for r in previous}
        merged.update({r["batch"]:r for r in index})
        write(self.out/"contact_sheet_index.json",[merged[k] for k in sorted(merged)])

    def finalize(self,qids,verify_only=False):
        dossiers=[];summaries=[];matrices=[];matrix_feature_replay_errors=[]
        for count,qid in enumerate(qids,1):
            assert self.validate_done(qid)
            folder=self.out/"queries"/qid
            dossier=read(folder/"dossier.json");dossiers.append(dossier)
            query_summaries=read(folder/"comparisons.json");summaries.extend(query_summaries)
            for summary in query_summaries:
                with gzip.open(folder/summary["token_file"],"rt") as stream:
                    rows=list(csv.DictReader(stream))
                assert len(rows)==summary["valid_query_tokens"]
                delta=np.asarray([float(r["delta"]) for r in rows],np.float64)
                assert abs(float(delta.mean())-summary["image_gap"])<2e-6
                assert all(abs(float(r["pos_sim"])-float(r["neg_sim"])-float(r["delta"]))<1e-7 for r in rows)
                assert all(float(r[side+"_top1_top2_gap"])>=0 for r in rows for side in ["pos","neg"])
                assert len({int(r["q_index"]) for r in rows})==len(rows)
                assert sum(r["tokens"] for r in summary["regions"])==len(rows)
                assert abs(sum(r["net_contribution"] for r in summary["regions"])-summary["image_gap"])<2e-6
            for index in read(folder/"matrix_index.json"):
                matrices.append(index)
                with np.load(self.out/index["path"]) as z:
                    matrix=z["matrix"];assert matrix.dtype==np.float32 and np.isfinite(matrix).all()
                    assert list(matrix.shape)==index["shape"]
                    assert abs(float(torch.from_numpy(matrix).max(1).values.mean())-index["cached_image_score"])<2e-6
                    if verify_only:
                        q=self.qb["embeddings"][self.qindex[index["query"]]][z["query_raw_indices"]].float()
                        r=self.rb["embeddings"][self.rindex[index["reference"]]][z["reference_raw_indices"]].float()
                        expected=(q@r.T).numpy()
                        error=float(np.max(abs(matrix-expected)))
                        assert error<2e-6
                        matrix_feature_replay_errors.append(error)
            if count%100==0:
                print(f"Validate {count}/{len(qids)}",flush=True)
        assert len({r["photo_id"] for r in dossiers})==len(qids)
        report=dict(schema=SCHEMA,verified_utc=now(),status="passed_for_generated_frozen_analysis_scope",
                    queries=len(qids),expected_queries=1117,all_queries_complete=len(qids)==1117,
                    data_coverage=len(qids)/1117,token_analysis_coverage=len(qids)/1117,
                    visual_reviewed_queries=0,visual_review_coverage=0.,
                    controlled_intervention_identities=0,mechanism_causal_evidence_coverage=0.,
                    total_contrasts=len(summaries),full_matrices=len(matrices),
                    max_matrix_cached_score_error=max(r["replay_error"] for r in matrices),
                    max_token_delta_conservation_error=max(r["delta_conservation_error"] for r in summaries),
                    max_region_conservation_error=max(r["region_conservation_error"] for r in summaries),
                    full_pointwise_matrices_recomputed=len(matrix_feature_replay_errors),
                    max_pointwise_matrix_error=max(matrix_feature_replay_errors) if matrix_feature_replay_errors else None,
                    outcome_counts=dict(Counter(r["outcome"] for r in dossiers)),
                    unreviewed_region_categories=["difference_layout","pattern_color"],
                    limitations=["Automatic OCR regions are not visually audited critical regions.",
                                 "Residuals use existing accepted transforms only; missing transforms and cross-face annotations remain unknown.",
                                 "No causal mechanism inferred from final token statistics.",
                                 "All scalar ranks replay caches; selected candidate dot replay is not full-gallery re-encoding."],
                    input_source_hashes_rechecked=False)
        # Actual post-run source hashes, including read-only originals.
        inputs=read(self.out/"input_manifest.json")["inputs"]
        if (self.out/"supplemental_input_manifest.json").exists():
            inputs+=read(self.out/"supplemental_input_manifest.json")["inputs"]
        verified=[sha(ROOT/r["path"])==r["sha256"] for r in inputs]
        assert all(verified)
        report["input_source_hashes_rechecked"]=all(verified)
        image_sources={}
        for dossier in dossiers:
            image_sources[dossier["photo_id"]]=(Path(dossier["original_path"]),dossier["image_sha256"])
            for candidate in dossier["candidates"]:
                image_sources[candidate["reference_id"]]=(Path(candidate["original_path"]),candidate["sha256"])
        image_checks=[]
        for identifier,(path,expected) in image_sources.items():
            available=path.is_file()
            observed=sha(path) if available else None
            assert not available or observed==expected,str(path)
            image_checks.append(dict(id=identifier,path=str(path),available=available,
                                     expected_sha256=expected,observed_sha256=observed))
        report["original_images_hash_verified"]=sum(r["available"] for r in image_checks)
        report["original_images_missing"]=sum(not r["available"] for r in image_checks)
        write(self.out/"original_images_verified.json",image_checks)
        write(self.out/"validation.json",report)
        if verify_only:
            return report
        write(self.out/"dossiers_all.json",dossiers)
        compressed_csv(self.out/"dossiers_all.csv.gz",dossiers)
        compressed_csv(self.out/"comparisons_all.csv.gz",summaries)
        write(self.out/"matrix_index_all.json",matrices)
        self.contacts(qids)
        self.index_html(dossiers)
        write(self.out/"visual_review_queue.json",[dict(photo_id=r["photo_id"],status="unreviewed",
                          evidence_path=f"queries/{r['photo_id']}/evidence.jpg",case_path=f"queries/{r['photo_id']}/index.html",
                          visible_face=None,difficulties=[],reviewer=None,reviewed_at=None) for r in dossiers])
        return report

    def index_html(self,dossiers):
        body=["<!doctype html><meta charset='utf-8'><title>ColNomic 全量冻结Token审计</title>",
              f"<style>{CSS}</style><h1>ColNomic 全量冻结 Token 审计</h1>",
              "<p class='notice'>此导出已完成客观评分与Token分解；视觉审核覆盖为0。所有图像材料等待实际查看。"
              "不能将自动OCR区域、跨行或匹配集中解释为已证实的失败机制。</p>",
              f"<p>{len(dossiers)}/1117 查询；5405图库记录，5330 SHA设计。"
              "正确候选从positives.csv核验并保留全部SHA别名；完整图库稳定排名可重放。"
              "历史测试已被观察，不是新盲测。</p>",
              "<p><a href='validation.json'>独立数值验证</a> · <a href='baseline_verification.json'>原排名与融合复放</a> · "
              "<a href='full_gallery_scores.npz'>全部标量分数和ID轴</a> · <a href='full_gallery_rankings.npz'>完整SHA排名</a> · "
              "<a href='dossiers_all.csv.gz'>1117主档案</a> · <a href='contact_sheet_index.json'>每4图五列浏览材料</a></p>",
              "<label>筛选：<input id='filter' placeholder='photo_id / identity / split / outcome' style='width:420px'></label>",
              "<table id='cases'><tr><th>案例</th><th>身份</th><th>split</th><th>客观结果</th><th>图像排名</th><th>融合排名</th><th>审核</th></tr>"]
        for row in dossiers:
            qid=row["photo_id"]
            body.append(f"<tr><td><a href='queries/{qid}/index.html'>{qid}</a></td><td>{html.escape(row['identity'])}</td>"
                        f"<td>{row['split']}</td><td>{row['outcome']}</td><td>{row['image_positive_rank']}</td>"
                        f"<td>{row['fusion_positive_rank']}</td><td>尚未视觉审核</td></tr>")
        body.append("</table><script>document.getElementById('filter').oninput=function(){const term=this.value.toLowerCase();"
                    "document.querySelectorAll('#cases tr').forEach((r,i)=>{if(i)r.hidden=!r.innerText.toLowerCase().includes(term)})}</script>")
        atomic_text(self.out/"index.html","\n".join(body))


def self_test():
    matrix=np.array([[1.,1.,.5],[.1,.2,.3]],np.float32)
    one,s,two,t=stable_top_two(matrix,np.array([4,8,12]))
    assert one.tolist()==[4,12] and two.tolist()==[8,8]
    assert np.all(s>=t)
    assert unique_order(np.array([1.,2.,2.,0.]),["a","b","a","c"]).tolist()==[1,2,3]
    rows=[dict(region="subject_other",delta=.2),dict(region="background",delta=-.1)]
    regions=region_summary(rows,2)
    assert abs(sum(r["net_contribution"] for r in regions)-.05)<1e-10
    print("PASS: stable top2 ties, post-sort SHA dedup, mutually exclusive signed-region conservation",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/colnomic_full_audit/20261013_resume/full_tokens")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(args.threads);cv2.setNumThreads(args.threads)
    if args.self_test:
        self_test();return
    out=args.output.resolve()
    assert out.is_relative_to(ROOT/"results") and out!=ROOT/"results", "Output must be a new workspace results subdirectory."
    out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/"runtime.json",dict(pid=os.getpid(),started_utc=now(),python=sys.executable,
                     arguments=sys.argv,script_sha256=sha(Path(__file__)),torch=torch.__version__,
                     numpy=np.__version__,threads=args.threads,device="cpu",status="running"))
        audit=Audit(out);audit.inputs();audit.global_outputs()
        qids=audit.qb["ids"][:args.limit] if args.limit else audit.qb["ids"]
        if args.verify_only:
            result=audit.finalize(qids,verify_only=True)
        else:
            start=time.monotonic()
            for i,qid in enumerate(qids,1):
                audit.run_query(qid);audit.checkpoint(i)
                if i%4==0:
                    audit.contacts(qids[i-4:i],first_index=i-4)
                elif i==len(qids):
                    contact_start=(i//4)*4
                    audit.contacts(qids[contact_start:i],first_index=contact_start)
                if i%10==0 or i==len(qids):
                    print(f"Frozen tokens {i}/{len(qids)}, elapsed {time.monotonic()-start:.1f}s, {dict(audit.uses)}",flush=True)
            result=audit.finalize(qids)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        runtime=read(out/"runtime.json");runtime.update(status="completed",completed_utc=now())
        write(out/"runtime.json",runtime)


if __name__=="__main__":
    main()
