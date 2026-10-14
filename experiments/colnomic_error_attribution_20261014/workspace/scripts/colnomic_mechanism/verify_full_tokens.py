"""Independent read-only replay of the all-query Token export.

This implementation intentionally does not import the analysis exporter: bank
axes, stable SHA rankings, pixel coordinates, top-two correspondence and signed
region accounting are independently derived. Only this verifier's JSON/CSV
outputs are written; original data and per-query exports stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("key_numbers_specs", "difference_layout", "common_text",
           "pattern_color", "subject_other", "background", "template")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")
    temp.replace(path)


def bank(path):
    return torch.load(ROOT / path, map_location="cpu", weights_only=True, mmap=True)


def token_digest(tensor):
    # Match the original archived tensor identity, retaining dtype/shape bytes.
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode())
    h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def normalized_coordinates(spec, indices):
    """Independently apply all eight EXIF orientations to raw grid centers."""
    layout = spec["layout"]
    local = np.asarray(indices) - layout["start"]
    valid = (local >= 0) & (local < layout["count"])
    x = ((local % layout["width"]) + .5) / layout["width"]
    y = ((local // layout["width"]) + .5) / layout["height"]
    orientation = spec["exif_orientation"]
    transforms = {1:(x,y), 2:(1-x,y), 3:(1-x,1-y), 4:(x,1-y),
                  5:(y,x), 6:(1-y,x), 7:(1-y,1-x), 8:(y,1-x)}
    result = np.stack(transforms[orientation], 1).astype(np.float64)
    result[~valid] = np.nan
    return result


def ordered_unique(scores, hashes):
    order = np.argsort(-scores, kind="stable")
    seen = set()
    return np.asarray([i for i in order if not (hashes[i] in seen or seen.add(hashes[i]))],
                      dtype=np.int32)


def top_two(matrix):
    # Stable sorting retains first raw-gallery token on exact equal similarities.
    order = np.argsort(-matrix, axis=1, kind="stable")[:, :2]
    scores = np.take_along_axis(matrix, order, axis=1)
    return order, scores


def coordinate_columns(rows, prefix):
    return np.asarray([[float(r[prefix+"_x"]) if r[prefix+"_x"] else np.nan,
                        float(r[prefix+"_y"]) if r[prefix+"_y"] else np.nan] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/colnomic_full_audit/20261013_resume/full_tokens")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--matrix-replay", choices=["all", "sample"], default="all")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--receipt-dir", type=Path,
                        help="Write fresh verifier receipts here without replacing archived audit receipts")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.output.resolve()
    assert out.is_relative_to(ROOT / "results")
    receipt_dir = args.receipt_dir.resolve() if args.receipt_dir else out
    receipt_dir.mkdir(parents=True,exist_ok=True)
    report_path = receipt_dir / "independent_validation.json"
    try:
        qb = bank("cache/image_embeddings/colnomic7b/photos.pt")
        rb = bank("cache/image_embeddings/colnomic7b/references.pt")
        caches = {name:bank(path) for name,path in [
            ("image", "cache/scores/colnomic7b/image.pt"),
            ("bm25", "cache/scores/bm25/bm25.pt"),
            ("fusion", "cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt")]}
        qids, rids = qb["ids"], rb["ids"]
        assert len(qids) == len(set(qids)) == 1117
        assert len(rids) == len(set(rids)) == 5405
        hashes = np.asarray(rb["sha256"])
        assert len(set(hashes)) == 5330
        qindex = {qid:i for i,qid in enumerate(qids)}
        rindex = {rid:i for i,rid in enumerate(rids)}
        metadata = read(ROOT / "results/token_mechanism_analysis/F03/reproduction/token_metadata.json")
        specs = {r["id"]:r for key in ["photos", "references"] for r in metadata[key]}
        box_version = out / "frozen_inputs/subject_boxes.json"
        box_expected = next(r["sha256"] for r in read(out/"input_manifest.json")["inputs"]
                            if r["path"] == "dataset/annotations/subject_boxes.json")
        assert box_version.is_file() and digest(box_version) == box_expected
        boxes = read(box_version)["records"]
        rocr = read(ROOT / "cache/ocr/references.json")
        dossiers = read(out / "dossiers_all.json")
        assert len(dossiers) == 1117 and [r["photo_id"] for r in dossiers] == qids
        positives = defaultdict(list)
        with (ROOT / "dataset/lists/positives.csv").open() as stream:
            for row in csv.DictReader(stream):
                positives[row["package_id"]].append(row["reference_id"])
        with np.load(out / "full_gallery_scores.npz") as scores:
            assert scores["query_ids"].tolist() == qids
            assert scores["reference_ids"].tolist() == rids
            assert scores["reference_sha256"].tolist() == hashes.tolist()
            for name,cache in caches.items():
                assert cache["query_ids"] == qids and cache["reference_ids"] == rids
                assert np.array_equal(scores[name], cache["scores"].numpy())
        rankings = np.load(out / "full_gallery_rankings.npz")
        assert rankings["query_ids"].tolist() == qids and rankings["reference_ids"].tolist() == rids
        index = read(out / "matrix_index_all.json")
        assert len({(r["query"],r["reference"]) for r in index}) == len(index)
        replay_indices = set(range(len(index))) if args.matrix_replay == "all" else set(
            np.random.default_rng(args.seed).choice(len(index), min(args.sample_size,len(index)), replace=False).tolist())
        replay_pairs = {(index[i]["query"],index[i]["reference"]) for i in replay_indices}
        metrics = Counter()
        maxima = defaultdict(float)
        outcomes = Counter()
        per_query = []
        for qi,dossier in enumerate(dossiers):
            qid = dossier["photo_id"]
            folder = out / "queries" / qid
            commit = read(folder / "complete.json")
            assert commit["photo_id"] == qid
            for item in commit["files"]:
                file = out / item["path"]
                assert file.is_file() and file.stat().st_size == item["bytes"]
                assert digest(file) == item["sha256"], str(file)
                metrics["committed_file_hash_checks"] += 1
            pos_hashes = {hashes[rindex[rid]] for rid in positives[dossier["package_id"]]}
            all_pos = {rid for rid,h in zip(rids,hashes) if h in pos_hashes}
            assert set(dossier["positive_reference_ids"]) == all_pos
            expected_pos = sorted(all_pos, key=lambda rid:(-float(caches["image"]["scores"][qi,rindex[rid]]),rindex[rid]))
            assert dossier["primary_positive"] == expected_pos[0]
            wrong_roles = defaultdict(list)
            for name,cache in caches.items():
                expected = ordered_unique(cache["scores"][qi].numpy(),hashes)
                assert np.array_equal(rankings[name][qi],expected)
                positive_rank = next(k+1 for k,i in enumerate(expected) if hashes[i] in pos_hashes)
                strongest_wrong = next(i for i in expected if hashes[i] not in pos_hashes)
                assert dossier[name+"_positive_rank"] == positive_rank
                assert dossier[name+"_top1"] == rids[expected[0]]
                assert dossier[name+"_strongest_wrong"] == rids[strongest_wrong]
                assert dossier[name+"_top1_correct"] == (positive_rank == 1)
                if name in ["image", "fusion"]:
                    wrong_roles[rids[strongest_wrong]].append(name+"_strongest_wrong")
                metrics["complete_gallery_rank_replays"] += 1
            truth = (dossier["image_top1_correct"], dossier["fusion_top1_correct"])
            outcome = {(True,True):"both_correct",(False,True):"fusion_rescued",
                       (True,False):"fusion_harmed",(False,False):"both_wrong"}[truth]
            assert dossier["outcome"] == outcome
            outcomes[outcome] += 1
            q = qb["embeddings"][qi]
            assert q.dtype == torch.float16 and torch.isfinite(q).all()
            assert token_digest(q) == specs[qid]["all_tokens_sha256"]
            qraw = np.flatnonzero(q.float().norm(dim=1).numpy() > 1e-8)
            qcoords = normalized_coordinates(specs[qid], qraw)
            matrix_records = read(folder / "matrix_index.json")
            expected_candidates = all_pos | set(wrong_roles)
            assert {r["reference"] for r in matrix_records} == expected_candidates
            matrices = {}
            for item in matrix_records:
                rid = item["reference"]
                assert item["query"] == qid and digest(out/item["path"]) == item["sha256"]
                r = rb["embeddings"][rindex[rid]]
                assert r.dtype == torch.float16 and torch.isfinite(r).all()
                assert token_digest(r) == specs[rid]["all_tokens_sha256"]
                rraw = np.flatnonzero(r.float().norm(dim=1).numpy() > 1e-8)
                with np.load(out/item["path"]) as archive:
                    m = archive["matrix"]
                    assert m.dtype == np.float32 and np.isfinite(m).all()
                    assert m.shape == (len(qraw),len(rraw))
                    assert np.array_equal(archive["query_raw_indices"],qraw)
                    assert np.array_equal(archive["reference_raw_indices"],rraw)
                    assert str(archive["query_token_sha256"]) == specs[qid]["all_tokens_sha256"]
                    assert str(archive["reference_token_sha256"]) == specs[rid]["all_tokens_sha256"]
                    if (qid,rid) in replay_pairs:
                        fresh = (q[qraw].float() @ r[rraw].float().T).numpy()
                        point_error = float(np.max(abs(fresh-m)))
                        assert point_error < 2e-6
                        maxima["pointwise_matrix_replay_error"] = max(maxima["pointwise_matrix_replay_error"],point_error)
                        metrics["pointwise_matrices_replayed"] += 1
                    score_error = abs(float(torch.from_numpy(m).max(1).values.mean()) - float(caches["image"]["scores"][qi,rindex[rid]]))
                    assert score_error < 2e-6
                    maxima["selected_candidate_score_error"] = max(maxima["selected_candidate_score_error"],score_error)
                    order,tops = top_two(m)
                    matrices[rid] = (rraw[order],tops,normalized_coordinates(specs[rid],rraw))
                metrics["full_fp32_matrices_checked"] += 1
            comparisons = read(folder / "comparisons.json")
            assert {(r["positive"],r["negative"]) for r in comparisons} == {(p,n) for p in all_pos for n in wrong_roles}
            assert len(comparisons) == len(all_pos)*len(wrong_roles)
            for summary in comparisons:
                p,n = summary["positive"],summary["negative"]
                assert set(summary["negative_roles"]) == set(wrong_roles[n])
                with gzip.open(folder/summary["token_file"],"rt") as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == len(qraw) and np.array_equal([int(r["q_index"]) for r in rows],qraw)
                assert np.allclose(coordinate_columns(rows,"q"),qcoords,atol=1e-10,equal_nan=True)
                side_scores = {}
                for side,rid in [("pos",p),("neg",n)]:
                    raw_indices,tops,_ = matrices[rid]
                    assert np.array_equal([int(r[side+"_index"]) for r in rows],raw_indices[:,0])
                    assert np.array_equal([int(r[side+"_second_index"]) for r in rows],raw_indices[:,1])
                    assert np.array_equal(np.asarray([float(r[side+"_sim"]) for r in rows],np.float32),tops[:,0])
                    assert np.array_equal(np.asarray([float(r[side+"_second_sim"]) for r in rows],np.float32),tops[:,1])
                    assert np.allclose(coordinate_columns(rows,side),normalized_coordinates(specs[rid],raw_indices[:,0]),atol=1e-10,equal_nan=True)
                    assert np.allclose(coordinate_columns(rows,side+"_second"),normalized_coordinates(specs[rid],raw_indices[:,1]),atol=1e-10,equal_nan=True)
                    gap = tops[:,0]-tops[:,1]
                    assert np.array_equal(np.asarray([float(r[side+"_top1_top2_gap"]) for r in rows],np.float32),gap)
                    side_scores[side] = tops[:,0]
                delta = side_scores["pos"]-side_scores["neg"]
                assert np.array_equal(np.asarray([float(r["delta"]) for r in rows],np.float32),delta)
                image_gap = float(caches["image"]["scores"][qi,rindex[p]]-caches["image"]["scores"][qi,rindex[n]])
                conservation = abs(float(delta.astype(np.float64).mean())-image_gap)
                assert conservation < 2e-6
                maxima["token_gap_conservation_error"] = max(maxima["token_gap_conservation_error"],conservation)
                assert set(r["region"] for r in rows).issubset(REGIONS)
                assert len(summary["regions"]) == len(REGIONS)
                box = boxes[qid]
                bbox = np.asarray(box["bbox"])/np.tile([box["width"],box["height"]],2)
                pwords = set(re.findall(r"[^\W_]+",rocr[p]["text"].lower(),re.UNICODE))
                nwords = set(re.findall(r"[^\W_]+",rocr[n]["text"].lower(),re.UNICODE))
                for i,row in enumerate(rows):
                    coords=qcoords[i]
                    if not np.isfinite(coords).all():
                        expected_region="template"
                    elif not ((coords>=bbox[:2]) & (coords<bbox[2:])).all():
                        expected_region="background"
                    elif re.search(r"\d",row["q_ocr_text"]):
                        expected_region="key_numbers_specs"
                    else:
                        words=set(re.findall(r"[^\W_]+",row["q_ocr_text"].lower(),re.UNICODE))
                        expected_region="common_text" if words and words <= (pwords & nwords) else "subject_other"
                    assert row["region"] == expected_region
                total=0
                for region in summary["regions"]:
                    values=delta[[r["region"]==region["region"] for r in rows]].astype(np.float64)
                    assert region["tokens"] == len(values)
                    for key,wanted in [("net_contribution",values.sum()/len(delta)),
                                       ("positive_contribution",np.maximum(values,0).sum()/len(delta)),
                                       ("negative_contribution",np.minimum(values,0).sum()/len(delta))]:
                        assert abs(region[key]-float(wanted)) < 1e-10
                    total += float(region["net_contribution"])
                region_error=abs(total-image_gap)
                assert region_error < 2e-6
                maxima["region_gap_conservation_error"] = max(maxima["region_gap_conservation_error"],region_error)
                metrics["comparison_conservation_checks"] += 1
                metrics["token_rows_checked"] += len(rows)
            assert (folder/"index.html").is_file() and (folder/"evidence.jpg").is_file()
            per_query.append(dict(photo_id=qid,queries_checked=1,contrasts=len(comparisons),
                                  matrices=len(matrix_records),outcome=outcome))
            if (qi+1)%100 == 0:
                print(f"Independent Token verification {qi+1}/1117",flush=True)
        rankings.close()
        sources = read(out/"input_manifest.json")["inputs"]
        if (out/"supplemental_input_manifest.json").exists():
            sources += read(out/"supplemental_input_manifest.json")["inputs"]
        source_revisions=[]
        for item in sources:
            observed=digest(ROOT/item["path"])
            if item["path"] == "dataset/annotations/subject_boxes.json" and observed != item["sha256"]:
                source_revisions.append(dict(path=item["path"],historical_sha256=item["sha256"],
                    current_sha256=observed,scope='Current user-authorized box revision; Token export verified against immutable historical snapshot'))
            else:
                assert observed == item["sha256"], item["path"]
        assert metrics["pointwise_matrices_replayed"] == len(replay_pairs)
        report=dict(status="passed",scope="complete scalar rankings and every exported Token comparison; independent implementation",
                    queries_checked=len(per_query),reference_records=5405,sha_distinct_designs=5330,
                    matrix_replay_mode=args.matrix_replay,replay_seed=args.seed,checks=dict(metrics),
                    maximum_errors=dict(maxima),outcome_counts=dict(outcomes),
                    original_input_hashes_rechecked=len(sources),
                    subject_box_snapshot_sha256=box_expected,current_subject_box_revisions=source_revisions,
                    visual_review_and_causal_mechanisms="outside numerical verifier scope; no completion claimed",
                    unassigned_semantic_regions=["difference_layout","pattern_color"],
                    region_evidence="OCR-line and historical subject-box proxies; one exclusive region per Token. Metadata-confirmed boxes are not guaranteed visually valid; current manual revisions do not apply to these historical regions",
                    verifier_sha256=digest(__file__),verified_utc=datetime.now(timezone.utc).isoformat())
        save(report_path,report)
        with (receipt_dir/"independent_verification_by_query.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(per_query[0]))
            writer.writeheader();writer.writerows(per_query)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception as exc:
        save(report_path,dict(status="failed",error_type=type(exc).__name__,error=str(exc),
                             verifier_sha256=digest(__file__),verified_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()
