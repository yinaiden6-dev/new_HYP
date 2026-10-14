"""Read-only integration of the historical 35-configuration scene evaluation.

Use the historical manifest to select configurations; newer configurations do
not silently change this cohort. Missing predictions remain missing. This code
does not infer a visual observation or causal mechanism from an automatic tag.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset/tools"))
from error_analysis import MAIN, CONFUSION, FUSIONS, controlled_comparisons

DIMENSIONS = {
    "background": [("multi_object", "多物体"), ("single_object", "单体")],
    "placement": [("handheld", "手持"), ("tabletop", "桌面")],
    "occlusion": [("foreground", "有前景遮挡"), ("none", "无前景遮挡")],
}
STATUS = "automatic_rules_before_human_calibration"
EVIDENCE = "deterministic_prediction_statistics_not_mechanism"


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(path.suffix + ".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def ratio(n, d):
    return n / d if d else None


def rank_bucket(rank):
    if rank is None:
        return "missing"
    return "1" if rank == 1 else "2-5" if rank <= 5 else "6-10" if rank <= 10 else ">10"


def automatic_main(cid, pid, predictions):
    """Faithfully preserve E-tag priority, independently of literal rank fields."""
    p = predictions.get(cid, {}).get(pid)
    if p is None or p["correct"]:
        return None
    if p["data_issue"]:
        return "E0"
    if not p["positive_design_ids"]:
        return "E1"
    if cid in FUSIONS:
        a, b = [predictions.get(c, {}).get(pid) for c in FUSIONS[cid]]
        if a is not None and b is not None:
            if a["data_issue"] or b["data_issue"]:
                return "E7"
            if a["correct"] and b["correct"]:
                return "E6"
            if a["correct"] and not b["correct"]:
                return "E4"
            if b["correct"] and not a["correct"]:
                return "E5"
    if p.get("rank") is None or not p.get("selected_design_id"):
        return "E7"
    return "E2" if p["rank"] > 10 else "E3"


def load_labels(root, photos, packages, groups, inputs):
    snapshot=root/'results/scene_statistics/photo_labels.csv'
    if snapshot.exists():
        inputs[str(snapshot.relative_to(root))]=file_hash(snapshot)
        labels={}
        for row in read_csv(snapshot):
            pid=row['photo_id']
            if pid in labels or pid not in photos or row['package_id']!=photos[pid]['package_id']:
                raise ValueError('Invalid historical photo label snapshot')
            labels[pid]=dict(row,scene_reviewed=row['scene_reviewed']=='True',excluded=row['excluded']=='True',
                split=groups[row['package_id']]['split'],metadata_path='dataset/'+packages[row['package_id']]['metadata_path'])
        if set(labels)!=set(photos):raise ValueError('Historical label snapshot does not cover exact queries')
        return labels
    labels = {}
    for package in packages.values():
        path = root / "dataset" / package["metadata_path"]
        inputs[str(path.relative_to(root))] = file_hash(path)
        for photo in read_json(path)["photos"]:
            pid = Path(photo["image"]).stem
            if pid in labels or pid not in photos:
                raise ValueError("Duplicate or unexpected metadata photo: " + pid)
            if photos[pid]["path"] != package["directory"] + "/" + photo["image"]:
                raise ValueError("Metadata path mismatch: " + pid)
            if type(photo["reviewed"]) is not bool or type(photo["excluded"]) is not bool:
                raise ValueError("Non-boolean review status")
            labels[pid] = dict(photo_id=pid, package_id=photos[pid]["package_id"],
                **{d: photo["type"].get(d) or "unlabeled" for d in DIMENSIONS},
                scene_reviewed=photo["reviewed"], excluded=photo["excluded"],
                split=groups[photos[pid]["package_id"]]["split"],
                metadata_path=str(path.relative_to(root)))
    if set(labels) != set(photos):
        raise ValueError("Scene labels do not cover the exact photo inventory")
    return labels


def score_path_for(root, cfg, spec):
    if spec.get("score_path"):
        return root / spec["score_path"]
    if spec.get("region_manifest"):
        return (root / spec["region_manifest"]).parent / "scores.pt"
    mode = cfg["mode"]
    stem = mode + "_alpha_" + f'{spec["alpha"]:g}' if mode in ("fusion", "ocr_fusion") else mode
    return root / "cache/scores" / cfg["model"] / (stem + ".pt")


def configuration_semantics(cid, cfg, spec, root):
    mode = cfg["mode"]
    model_spec = spec.get("model_spec") or spec.get("baselines", {}).get("models", {}).get(cfg["model"], {})
    fusion = FUSIONS.get(cid)
    query = "OCR text" if mode.startswith("ocr_to_") else "RGB full original"
    gallery = "OCR text" if mode == "ocr_to_ocr" else "RGB reference artwork"
    spatial_supervision = None
    if spec.get("route") == "manual":
        query = "human subject-box crop; EXIF-transposed original; 5% maximum-side padding"
        spatial_supervision = "manual confirmed subject localization (extra supervision)"
    elif spec.get("route") == "sam3":
        query = "SAM3 top-confidence mask; white background then padded crop"
        spatial_supervision = "automatic segmentation; prompt and thresholds from settings"
    elif mode == "bm25":
        query, gallery = "PP-OCRv6 photo text", "PP-OCRv6 reference text"
    elif fusion:
        query = "late fusion of cached " + "+".join(fusion)
    if mode.endswith("_crisp"):
        scoring = ("same input and frozen embedding as non-CRISP; weighted forward MaxSim "
            "minus weighted mean similarity plus reverse MaxSim; reference weights="
            "||g_j-mean(g)|| normalized to sum N_g")
    elif mode == "bm25":
        scoring = "BM25 Unicode letters/digits lowercase query-frequency sum; k1=1.2 b=0.75"
    elif fusion:
        scoring = "(1-alpha)*population-Z(image)+alpha*population-Z(text); empty OCR=image fallback"
    elif cfg["model"] in ("Qwen3-VL-Embedding-2B", "dinov3", "siglip2"):
        scoring = "dense dot product of the stored model-normalized vectors"
    else:
        scoring = "FP32 mean MaxSim over nonzero valid query tokens, including attended template"
    return dict(config_id=cid, model=cfg["model"], mode=mode, result=cfg["result"],
        query_input=query, reference_input=gallery, scoring_implementation=scoring,
        fusion_image_config=fusion[0] if fusion else None,
        fusion_text_config=fusion[1] if fusion else None,
        alpha=spec.get("alpha", spec.get("scoring", {}).get("bm25_weight")),
        normalization=spec.get("scoring", {}).get("normalization"),
        crop_or_segmentation_settings=spec.get("settings"),
        additional_supervision=spatial_supervision,
        image_model=model_spec.get("image_model"), text_model=model_spec.get("text_model"),
        image_contract=model_spec.get("image_contract"), text_contract=model_spec.get("text_contract"),
        model_checkpoint=model_spec.get("checkpoint"), region_manifest=spec.get("region_manifest"),
        score_path=str(score_path_for(root, cfg, spec).relative_to(root)),
        configuration_sha256=file_hash(root / cfg["result"] / "config.json"),
        evidence_scope="stored recipe and code verification, not proof of historical encoder reproducibility")


def replay_scores(root, cfg, spec, ordered_photos, ordered_refs, positives, predictions, inputs):
    path = score_path_for(root, cfg, spec)
    if not path.exists():
        return {}, dict(status="missing_score_cache", path=str(path.relative_to(root)))
    inputs[str(path.relative_to(root))] = file_hash(path)
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    scores = value["scores"].float().numpy()
    if scores.shape != (len(ordered_photos), len(ordered_refs)) or not np.isfinite(scores).all():
        raise ValueError("Invalid full-gallery score matrix: " + str(path))
    qids = [p["photo_id"] for p in ordered_photos]
    rids = [r["reference_id"] for r in ordered_refs]
    if "query_ids" in value and value["query_ids"] != qids:
        raise ValueError("Score query-axis mismatch")
    if "reference_ids" in value and value["reference_ids"] != rids:
        raise ValueError("Score reference-axis mismatch")
    declared = spec.get("dataset_sha256", {})
    for name in ("photos.csv", "references.csv", "positives.csv", "groups.csv"):
        if declared.get(name) != inputs["dataset/lists/" + name]:
            raise ValueError("Score configuration inventory hash mismatch: " + name)
    signatures = [s for s in [spec.get("score_signature")] if s is not None]
    if signatures and value.get("signature") not in signatures:
        raise ValueError("Score signature mismatch: " + cfg["id"])
    sha = [r["sha256"] for r in ordered_refs]
    positive_sha = {p: {next(r["sha256"] for r in ordered_refs if r["reference_id"] == rid)
        for rid in rs} for p, rs in positives.items()}
    margins = {}; verified = 0; skipped_empty = 0
    for qi, photo in enumerate(ordered_photos):
        pid = photo["photo_id"]; prediction = predictions.get(pid)
        if prediction is None:
            continue
        if prediction.get("failure") == "empty_ocr":
            if prediction["correct"] or prediction.get("rank") is not None:
                raise ValueError("Invalid empty-OCR prediction")
            skipped_empty += 1
            continue
        row = scores[qi]; order = np.argsort(-row, kind="stable")
        seen = set(); top = []; rank = None; best_positive = None; best_negative = None
        for j in order:
            if sha[j] in seen:
                continue
            seen.add(sha[j])
            if len(top) < 10:
                top.append(rids[j])
            if sha[j] in positive_sha[photo["package_id"]]:
                if best_positive is None:
                    best_positive = int(j); rank = len(seen)
            elif best_negative is None:
                best_negative = int(j)
            if rank is not None and best_negative is not None and len(top) == 10:
                break
        if (prediction["rank"] != rank or prediction["correct"] != (rank == 1)
            or prediction.get("selected_reference_id") != top[0]
            or prediction.get("top_reference_ids") != top):
            raise ValueError("Independent score ranking mismatch: " + cfg["id"] + "/" + pid)
        margins[pid] = dict(top1_score=float(row[order[0]]),
            best_positive_reference_id=rids[best_positive] if best_positive is not None else None,
            best_positive_score=float(row[best_positive]) if best_positive is not None else None,
            best_negative_reference_id=rids[best_negative] if best_negative is not None else None,
            best_negative_score=float(row[best_negative]) if best_negative is not None else None,
            positive_negative_gap=float(row[best_positive]-row[best_negative])
                if best_positive is not None and best_negative is not None else None,
            top1_top2_gap=float(row[order[0]]-row[next(j for j in order if sha[j] != sha[order[0]])]))
        verified += 1
    return margins, dict(status="passed", path=str(path.relative_to(root)),
        shape=list(scores.shape), finite=bool(np.isfinite(scores).all()), verified_predictions=verified,
        empty_ocr_predictions=skipped_empty, axis_scope=("explicit_ids" if "query_ids" in value
            else "configuration_inventory_hashes_and_all_prediction_replay"),
        method="NumPy stable full-row sorting followed by SHA deduplication")


def group_statistics(cid, cfg, dimension, label, label_zh, rows, n, errors):
    successes = sum(r["correct"] for r in rows); failures = len(rows)-successes
    counts = Counter(r["error_type"] for r in rows if not r["correct"])
    if not counts.keys() <= MAIN.keys() or sum(counts.values()) != failures:
        raise ValueError("Failure label partition mismatch")
    result = dict(config_id=cid, model=cfg["model"], mode=cfg["mode"], dimension=dimension,
        label=label, label_zh=label_zh, photos=len(rows), successes=successes, errors=failures,
        success_rate=ratio(successes,len(rows)), error_rate=ratio(failures,len(rows)),
        error_share_of_all_errors=ratio(failures,errors), scene_share_of_all_photos=ratio(len(rows),n),
        all_photos=n,all_errors=errors,failure_label_status=STATUS)
    types = [dict(config_id=cid,model=cfg["model"],mode=cfg["mode"],dimension=dimension,
        label=label,label_zh=label_zh,error_type=e,error_type_zh=title,type_errors=counts[e],
        scene_photos=len(rows),scene_errors=failures,all_errors=errors,
        type_share_of_scene_errors=ratio(counts[e],failures),
        type_rate_among_scene_photos=ratio(counts[e],len(rows)),
        type_share_of_all_errors=ratio(counts[e],errors),failure_label_status=STATUS)
        for e,title in MAIN.items()]
    return result, types


def regenerate_tables(configs, predictions, labels, out):
    summary=[]; types=[]; joint=[]; failures=[]
    scene_counts={d:Counter(l[d] for l in labels.values()) for d in DIMENSIONS}
    for cid,cfg in configs.items():
        rows=[]
        for pid,p in predictions[cid].items():
            code=automatic_main(cid,pid,predictions)
            tags={d:labels[pid][d] for d in DIMENSIONS}
            rows.append(dict(photo_id=pid,correct=p["correct"],error_type=code,**tags))
            if not p["correct"]:
                failures.append(dict(config_id=cid,model=cfg["model"],mode=cfg["mode"],photo_id=pid,
                    package_id=p["package_id"],correct_rank=p.get("rank"),
                    wrong_reference_id=p.get("selected_reference_id"),wrong_design_id=p["selected_design_id"],
                    correct_reference_ids="|".join(p.get("positive_reference_ids",[])),
                    case_id=pid+"__"+(p["selected_design_id"] or "missing_reference"),**tags,
                    error_type=code,error_type_zh=MAIN[code],failure_label_status=STATUS))
        n=len(rows); errors=sum(not r["correct"] for r in rows)
        g,t=group_statistics(cid,cfg,"overall","all","全部",rows,n,errors);summary.append(g);types.extend(t)
        for d,pairs in DIMENSIONS.items():
            names=dict(pairs);names.update({l:l for l in sorted(scene_counts[d].keys()-names.keys())})
            partition=[]
            for label,name in names.items():
                g,t=group_statistics(cid,cfg,d,label,name,[r for r in rows if r[d]==label],n,errors)
                summary.append(g);types.extend(t);partition.append(g)
            if sum(g["photos"] for g in partition)!=n or sum(g["errors"] for g in partition)!=errors:
                raise ValueError("Scene dimension does not partition predictions")
        combinations=itertools.product(*[list(dict(DIMENSIONS[d]))+sorted(scene_counts[d].keys()-dict(DIMENSIONS[d]).keys())
            for d in ["placement","background","occlusion"]])
        for values in combinations:
            selected=[r for r in rows if tuple(r[d] for d in ["placement","background","occlusion"])==values]
            g,_=group_statistics(cid,cfg,"joint","/".join(values),"/".join(values),selected,n,errors);joint.append(g)
    write_csv(out/"summary.csv",summary);write_csv(out/"failure_types.csv",types)
    write_csv(out/"joint_statistics.csv",joint);write_csv(out/"failed_photos.csv",failures)
    label_rows=[{k:labels[pid][k] for k in ["photo_id","package_id",*DIMENSIONS,"scene_reviewed","excluded"]}
        for pid in sorted(labels)]
    write_csv(out/"photo_labels.csv",label_rows)
    index={(g["config_id"],g["dimension"],g["label"]):g for g in summary}
    for d,pairs in DIMENSIONS.items():
        wide=[]
        for cid,cfg in configs.items():
            overall=index[cid,"overall","all"]
            row=dict(config_id=cid,model=cfg["model"],mode=cfg["mode"],all_photos=overall["photos"],all_errors=overall["errors"])
            for label,_ in pairs:
                for key in ["photos","successes","errors","success_rate","error_rate","error_share_of_all_errors"]:
                    row[label+"_"+key]=index[cid,d,label][key]
            wide.append(row)
        write_csv(out/(d+".csv"),wide)
    return summary,types,joint,failures


def compare_csv(old, new, keys):
    a=read_csv(old);b=read_csv(new)
    ai={tuple(r[k] for k in keys):r for r in a};bi={tuple(r[k] for k in keys):r for r in b}
    if len(ai)!=len(a) or len(bi)!=len(b):
        raise ValueError("Nonunique comparison table keys")
    differences=[]
    for key in sorted(ai.keys()|bi.keys()):
        if key not in ai or key not in bi:
            differences.append(dict(key=list(key),reason="missing_row",historical=ai.get(key),recomputed=bi.get(key)))
        elif ai[key]!=bi[key]:
            for column in ai[key].keys()|bi[key].keys():
                if ai[key].get(column)!=bi[key].get(column):
                    differences.append(dict(key=list(key),column=column,historical=ai[key].get(column),recomputed=bi[key].get(column)))
    return dict(historical_rows=len(a),recomputed_rows=len(b),matching=not differences,differences=differences)


def scene_masks(labels, qids):
    result=[("overall","all",np.ones(len(qids),dtype=bool))]
    for d in DIMENSIONS:
        result.extend((d,label,np.array([labels[p][d]==label for p in qids]))
            for label in sorted({labels[p][d] for p in qids}))
    for values in itertools.product(*[sorted({labels[p][d] for p in qids}) for d in ["placement","background","occlusion"]]):
        result.append(("joint","/".join(values),np.array([tuple(labels[p][d] for d in ["placement","background","occlusion"])==values for p in qids])))
    return result


def wilson(successes,n):
    if not n:
        return None,None
    p=successes/n;z=1.959963984540054;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-h),float(c+h)


def paired_statistics(a,b,indices,qids,photos,groups,seed=20261013,repetitions=2000):
    valid=indices & (a>=0) & (b>=0)
    aa=a[valid].astype(bool);bb=b[valid].astype(bool)
    both=int(np.sum(aa & bb));rescue=int(np.sum(~aa & bb));harm=int(np.sum(aa & ~bb));wrong=int(np.sum(~aa & ~bb))
    count=int(valid.sum());selected=[qids[i] for i in np.flatnonzero(valid)]
    pids=[photos[p]["package_id"] for p in selected];gids=[groups[p]["group_id"] for p in pids]
    diff=bb.astype(float)-aa.astype(float);unique_groups=sorted(set(gids));unique_packages=sorted(set(pids))
    low=high=None
    if unique_groups:
        total=np.array([diff[np.array(gids)==g].sum() for g in unique_groups]);ns=np.array([gids.count(g) for g in unique_groups])
        rng=np.random.default_rng(seed);weights=rng.multinomial(len(unique_groups),np.ones(len(unique_groups))/len(unique_groups),size=repetitions)
        samples=(weights@total)/(weights@ns);low,high=np.quantile(samples,[.025,.975]).tolist()
    return dict(available_photos=count,missing_predictions=int(indices.sum())-count,both_correct=both,
        baseline_wrong_variant_correct=rescue,baseline_correct_variant_wrong=harm,both_wrong=wrong,
        rescues=rescue,harms=harm,net=rescue-harm,baseline_correct=both+harm,variant_correct=both+rescue,
        net_rate=ratio(rescue-harm,count),identities=len(unique_packages),source_groups=len(unique_groups),
        package_macro_delta=float(np.mean([diff[np.array(pids)==p].mean() for p in unique_packages])) if unique_packages else None,
        source_group_bootstrap_low=low,source_group_bootstrap_high=high,
        bootstrap_repetitions=repetitions,bootstrap_seed=seed,
        uncertainty="paired source-group percentile bootstrap; small-group strata unstable; descriptive association")


def load_revisions(review_dir, labels, inputs, root):
    revised={p:dict(l) for p,l in labels.items()};events=[];audit={}
    if not review_dir.exists():
        return revised,events,audit
    if review_dir.is_file():
        value=read_json(review_dir)
        rows=value.get('records',[]) if isinstance(value,dict) else value
        if not isinstance(rows,list) or len({r['photo_id'] for r in rows})!=len(rows):
            raise ValueError('Invalid/duplicate consolidated visual revision rows')
        source=str(review_dir.relative_to(root));inputs[source]=file_hash(review_dir)
        for row in rows:
            pid=row['photo_id']
            if pid not in labels:
                raise ValueError('Unknown consolidated visual query: '+pid)
            dimensions=row.get('scene_dimensions',{})
            audited=[];changes={}
            for d in DIMENSIONS:
                details=dimensions.get(d,{})
                if details.get('status') not in ('reviewed_known','reviewed_uncertain'):
                    continue
                label=details.get('value')
                if label not in dict(DIMENSIONS[d]) and label not in ('unclear','other'):
                    raise ValueError('Invalid reviewed scene label: '+pid+' '+d)
                path=details.get('source_path');expected=details.get('source_sha256')
                if not path or not expected or file_hash(root/path)!=expected:
                    raise ValueError('Reviewed scene source hash mismatch: '+pid+' '+d)
                inputs[path]=expected
                if not details.get('evidence'):
                    raise ValueError('No actual image evidence for scene dimension: '+pid+' '+d)
                for evidence in details['evidence']:
                    image=root/evidence['path'];actual=file_hash(image)
                    if actual!=evidence.get('actual_sha256') or (evidence.get('declared_sha256') and actual!=evidence['declared_sha256']):
                        raise ValueError('Scene visual evidence hash mismatch: '+pid)
                    inputs[evidence['path']]=actual
                revised[pid][d]=label;audited.append(d)
                if label!=labels[pid][d]:
                    changes[d]=dict(original=labels[pid][d],verified=label)
            revised[pid]['scene_reviewed']=len(audited)==len(DIMENSIONS) and all(dimensions[d]['status']=='reviewed_known' for d in DIMENSIONS)
            audited_all=len(audited)==len(DIMENSIONS)
            audit[pid]={**row,'_record_path':source,'scene_audit_complete':audited_all}
            events.append(dict(photo_id=pid,changes=changes,source=source,
                scene_dimensions=dimensions,scene_dimensions_reviewed=audited,scene_audit_complete=audited_all,
                scene_all_known=revised[pid]['scene_reviewed'],original_scene_reviewed=labels[pid]['scene_reviewed']))
        return revised,events,audit
    for path in sorted(review_dir.rglob("*.json")):
        if "revisions" in path.parts:
            continue
        value=read_json(path)
        if not isinstance(value,dict):
            continue
        pid=value.get("photo_id")
        if pid not in labels:
            continue
        previous=audit.get(pid)
        if previous is None or value.get("scene_labels") is not None:
            value["_record_path"]=str(path.relative_to(root))
            audit[pid]=value
        scene=value.get("scene_labels")
        if isinstance(scene,dict) and value.get("scene_labels_reviewed") is True:
            if set(scene)!=set(DIMENSIONS):
                raise ValueError("Incomplete independently reviewed scene labels: "+pid)
            inputs[str(path.relative_to(root))]=file_hash(path)
            changes={d:dict(original=labels[pid][d],verified=scene[d]) for d in DIMENSIONS if scene[d]!=labels[pid][d]}
            revised[pid].update(scene);revised[pid]["scene_reviewed"]=True
            events=[event for event in events if event["photo_id"]!=pid]
            events.append(dict(photo_id=pid,changes=changes,source=str(path.relative_to(root)),
                reviewer=value.get("reviewer"),original_scene_reviewed=labels[pid]["scene_reviewed"],
                identity_region_obscured=value.get("identity_region_obscured",value.get("occlusion_detail",{}).get("identity_region_obscured")),
                evidence=value.get("evidence",[value["evidence_image"]] if value.get("evidence_image") else [])))
    return revised,events,audit


def export_preprocessing(root,configs,specs,photos,predictions,labels,inputs,out):
    rows=[]
    ocr=read_json(root/"cache/ocr/photos.json")
    for cid in ("D01","D03"):
        spec=specs[cid];path=root/spec["region_manifest"];inputs[str(path.relative_to(root))]=file_hash(path)
        manifest=read_json(path)
        if [r["photo_id"] for r in manifest["photos"]]!=list(photos):
            raise ValueError("Foreground manifest order mismatch")
        for row in manifest["photos"]:
            pid=row["photo_id"]
            if row["source_sha256"]!=photos[pid]["sha256"]:
                raise ValueError("Foreground source hash mismatch")
            actual=root/row["query_path"]
            if not actual.exists() or file_hash(actual)!=row["query_sha256"]:
                raise ValueError("Foreground input file hash mismatch: "+pid)
            inputs[str(actual.relative_to(root))]=row["query_sha256"]
            if row.get("mask_path"):
                mask_path=root/row["mask_path"]
                if not mask_path.exists() or file_hash(mask_path)!=row["mask_sha256"]:
                    raise ValueError("Foreground mask hash mismatch: "+pid)
                inputs[str(mask_path.relative_to(root))]=row["mask_sha256"]
            bounds=row.get("crop_bounds")
            lines=ocr[pid].get("lines",[]);inside=0
            if bounds:
                x0,y0,x1,y1=bounds
                for line in lines:
                    bb=line.get("bbox")
                    if bb and len(bb)==4:
                        cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2
                        inside+=int(x0<=cx<x1 and y0<=cy<y1)
                crop_area=(x1-x0)*(y1-y0);source_area=row["width"]*row["height"]
            else:
                crop_area=source_area=row["width"]*row["height"];inside=len(lines)
            a=predictions["B03"].get(pid);b=predictions[cid].get(pid)
            rows.append(dict(config_id=cid,photo_id=pid,package_id=photos[pid]["package_id"],
                source_path=row["source_path"],processed_input=row["query_path"],input_sha256=row["query_sha256"],
                original_width=row["width"],original_height=row["height"],
                crop_bounds=json.dumps(bounds),subject_bbox=json.dumps(row.get("bbox")),
                crop_area_fraction=crop_area/source_area,
                subject_bbox_pixels=(row["bbox"][2]-row["bbox"][0])*(row["bbox"][3]-row["bbox"][1]) if row.get("bbox") else None,
                original_OCR_line_count=len(lines),original_OCR_line_centers_within_crop=inside,
                OCR_retention_proxy=ratio(inside,len(lines)),
                proxy_limit="OCR centers are not visibility audit or re-OCR after scaling/crop",
                annotation_source=row.get("annotation_source"),annotation_status=row.get("annotation_status"),
                confidence=row.get("confidence"),mask_path=row.get("mask_path"),fallback=row["fallback"],
                baseline_correct=a["correct"] if a else None,variant_correct=b["correct"] if b else None,
                transition=("rescue" if not a["correct"] and b["correct"] else "harm" if a["correct"] and not b["correct"]
                    else "both_correct" if a["correct"] else "both_wrong") if a and b else "missing"))
    write_csv(out/"preprocessing_inputs.csv",rows)
    return rows


def run(root,out,review_dir):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    history=root/"results/scene_statistics";historical=read_json(history/"manifest.json")
    inputs={};checks={};missing=[]
    for name in ("photos","references","positives","groups","packages"):
        path=root/f"dataset/lists/{name}.csv";inputs[str(path.relative_to(root))]=file_hash(path)
    ordered_photos=read_csv(root/"dataset/lists/photos.csv");photos={r["photo_id"]:r for r in ordered_photos}
    ordered_refs=read_csv(root/"dataset/lists/references.csv");refs={r["reference_id"]:r for r in ordered_refs}
    groups={r["package_id"]:r for r in read_csv(root/"dataset/lists/groups.csv")}
    packages={r["package_id"]:r for r in read_csv(root/"dataset/lists/packages.csv")}
    if len(photos)!=len(ordered_photos) or len(refs)!=len(ordered_refs):
        raise ValueError("Duplicate formal inventory IDs")
    positives=defaultdict(set)
    for row in read_csv(root/"dataset/lists/positives.csv"):
        positives[row["package_id"]].add(row["reference_id"])
    current=read_csv(root/"results/all_results.csv")
    snapshot_dirs={str(Path(name).parent) for name in historical["inputs"] if name.endswith("predictions.jsonl")}
    configs={c["id"]:c for c in current if c["result"] in snapshot_dirs}
    if set(c["result"] for c in configs.values())!=snapshot_dirs or len(configs)!=historical["configurations"]:
        raise ValueError("Cannot recover all historical configuration definitions from project paths")
    specs={cid:read_json(root/c["result"]/"config.json") for cid,c in configs.items()}
    inputs["results/all_results.csv"]=file_hash(root/"results/all_results.csv")
    labels=load_labels(root,photos,packages,groups,inputs)
    revised,revision_events,visual_audit=load_revisions(review_dir,labels,inputs,root)
    predictions={};all_margins={};replays={}
    for cid,cfg in configs.items():
        path=root/cfg["result"]/"predictions.jsonl"
        if not path.exists():
            predictions[cid]={}
            missing.extend(dict(config_id=cid,photo_id=pid,reason="missing_prediction_file",path=str(path.relative_to(root))) for pid in photos)
            continue
        inputs[str(path.relative_to(root))]=file_hash(path)
        inputs[str(path.with_name("config.json").relative_to(root))]=file_hash(path.with_name("config.json"))
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        pred={p["photo_id"]:p for p in rows}
        if len(pred)!=len(rows) or not set(pred)<=set(photos):
            raise ValueError("Duplicate or unknown prediction IDs: "+cid)
        for pid,p in pred.items():
            if type(p["correct"]) is not bool:
                raise ValueError("Nonboolean correctness")
            rid=p.get("selected_reference_id");p["selected_design_id"]=refs.get(rid,{}).get("design_id")
            expected=positives[photos[pid]["package_id"]]
            p["positive_design_ids"]=sorted({refs[r]["design_id"] for r in expected})
            p["data_issue"]=not (set(p.get("positive_reference_ids",[]))==expected
                and p.get("package_id")==photos[pid]["package_id"]
                and p["correct"]==(p["selected_design_id"] in p["positive_design_ids"])
                and ((p.get("rank")==1)==p["correct"]))
            if p.get("rank") is not None and (type(p["rank"]) is not int or p["rank"]<1):
                p["data_issue"]=True
        predictions[cid]=pred
        for pid in photos.keys()-pred.keys():
            missing.append(dict(config_id=cid,photo_id=pid,reason="missing_prediction_row"))
        margins,replay=replay_scores(root,cfg,specs[cid],ordered_photos,ordered_refs,positives,pred,inputs)
        all_margins[cid]=margins;replays[cid]=replay
        print(f"Scene35 replay {cid}: {replay['status']} {replay.get('verified_predictions',0)}",flush=True)
    summary,types,joint,failures=regenerate_tables(configs,predictions,labels,out/"original_labels")
    table_keys={"summary.csv":["config_id","dimension","label"],"failure_types.csv":["config_id","dimension","label","error_type"],
        "joint_statistics.csv":["config_id","dimension","label"],"failed_photos.csv":["config_id","photo_id"],
        "photo_labels.csv":["photo_id"],"background.csv":["config_id"],"placement.csv":["config_id"],"occlusion.csv":["config_id"]}
    checks["historical_table_comparisons"]={name:compare_csv(history/name,out/"original_labels"/name,keys) for name,keys in table_keys.items()}
    checks["historical_input_hash_comparisons"]={name:dict(expected=sha,current=file_hash(root/name) if (root/name).exists() else None,
        unchanged=(root/name).exists() and file_hash(root/name)==sha) for name,sha in historical["inputs"].items()}
    checks["historical_output_hashes"]={name:dict(expected=sha,actual=file_hash(history/name),matching=file_hash(history/name)==sha)
        for name,sha in historical["output_sha256"].items()}
    if revision_events:
        regenerate_tables(configs,predictions,revised,out/"verified_labels")
    reviewed_labels={p:l for p,l in labels.items() if l["scene_reviewed"]}
    reviewed_predictions={cid:{p:v for p,v in pred.items() if p in reviewed_labels} for cid,pred in predictions.items()}
    regenerate_tables(configs,reviewed_predictions,reviewed_labels,out/"original_reviewed_subset")
    qids=list(photos);cids=list(configs);correct=np.full((len(qids),len(cids)),-1,dtype=np.int8)
    ranks=np.full(correct.shape,-1,dtype=np.int32);gaps=np.full(correct.shape,np.nan,dtype=np.float32);top1=np.full(correct.shape,"",dtype="U32")
    matrix_rows=[];photo_interfaces=[]
    for qi,pid in enumerate(qids):
        for ci,cid in enumerate(cids):
            p=predictions[cid].get(pid);margin=all_margins.get(cid,{}).get(pid,{})
            if p:
                correct[qi,ci]=int(p["correct"]);ranks[qi,ci]=p.get("rank") or -1
                top1[qi,ci]=p.get("selected_reference_id") or "";gaps[qi,ci]=margin.get("positive_negative_gap",np.nan)
            channels=FUSIONS.get(cid)
            row=dict(photo_id=pid,config_id=cid,package_id=photos[pid]["package_id"],group_id=groups[photos[pid]["package_id"]]["group_id"],
                split=groups[photos[pid]["package_id"]]["split"],prediction_available=p is not None,
                correct=p["correct"] if p else None,correct_rank=p.get("rank") if p else None,
                true_rank_bucket=rank_bucket(p.get("rank") if p else None),
                top1_reference_id=p.get("selected_reference_id") if p else None,top1_design_id=p.get("selected_design_id") if p else None,
                positive_reference_ids="|".join(sorted(positives[photos[pid]["package_id"]])),
                data_reference_validity="automatic_consistency_issue" if p and p["data_issue"] else "consistent_metadata_not_visual_truth_audit" if p else "missing_prediction",
                automatic_main_label=automatic_main(cid,pid,predictions),
                automatic_main_label_status=STATUS,
                fusion_image_correct=predictions.get(channels[0],{}).get(pid,{}).get("correct") if channels else None,
                fusion_text_correct=predictions.get(channels[1],{}).get(pid,{}).get("correct") if channels else None,
                fusion_correct=p["correct"] if p and channels else None,
                **{d:labels[pid][d] for d in DIMENSIONS},original_scene_reviewed=labels[pid]["scene_reviewed"],
                **margin,evidence_grade=EVIDENCE)
            matrix_rows.append(row)
        photo_interfaces.append(dict(photo_id=pid,package_id=photos[pid]["package_id"],split=labels[pid]["split"],
            original_scene_labels={d:labels[pid][d] for d in DIMENSIONS},original_scene_reviewed=labels[pid]["scene_reviewed"],
            verified_scene_labels={d:revised[pid][d] for d in DIMENSIONS},
            independent_scene_audit_complete=visual_audit.get(pid,{}).get('scene_audit_complete',any(e["photo_id"]==pid for e in revision_events)),
            reviewed_scene_dimension_states=visual_audit.get(pid,{}).get('scene_dimensions',{}),
            candidate_overview_review_path=visual_audit.get(pid,{}).get('primary_review_path'),
            visual_review_path=visual_audit.get(pid,{}).get("_record_path"),
            observable_differences=visual_audit.get(pid,{}).get("observable_difficulties",[]),
            mechanism_evidence=visual_audit.get(pid,{}).get("evidence_grade",visual_audit.get(pid,{}).get("evidence_level","not_visually_audited")),
            token_evidence_directory=f"results/colnomic_full_audit/20261013_resume/full_tokens/queries/{pid}",
            configurations_correct=[cids[j] for j in np.flatnonzero(correct[qi]==1)],
            configurations_wrong=[cids[j] for j in np.flatnonzero(correct[qi]==0)],
            configurations_missing=[cids[j] for j in np.flatnonzero(correct[qi]<0)]))
    write_csv(out/"prediction_matrix.csv",matrix_rows)
    np.savez_compressed(out/"prediction_matrix.npz",correct=correct,rank=ranks,positive_negative_gap=gaps,top1_reference=top1,
        photo_ids=np.array(qids),config_ids=np.array(cids))
    write_json(out/"matrix_schema.json",dict(shape=list(correct.shape),rows="1117 unique photos",columns="35 repeated evaluations",
        correct_missing=-1,rank_missing=-1,gap_missing="NaN",independent_photos=len(qids),evaluation_occurrences=len(matrix_rows)))
    write_json(out/"photo_evidence_interface.json",photo_interfaces)
    write_json(out/"configuration_semantics.json",[configuration_semantics(cid,cfg,specs[cid],root) for cid,cfg in configs.items()])
    write_json(out/"scene_label_revisions.json",revision_events)
    write_json(out/"original_label_snapshot.json",labels)
    for name in ["dataset/review/photos/edits.jsonl","dataset/review/error_analysis/annotations.json"]:
        path=root/name
        if path.exists():
            inputs[name]=file_hash(path);(out/Path(name).name).write_bytes(path.read_bytes())
    bidx=cids.index("B03");fidx=cids.index("F03");pair_rows=[];population=[];config_scene=[]
    for label_set,label_map in [("original_all",labels),("original_reviewed",reviewed_labels),("verified_snapshot",revised)]:
        subset=np.array([p in label_map for p in qids])
        for dim,label,mask in scene_masks(labels if label_set=="original_reviewed" else label_map,qids):
            mask=mask & subset
            for split in ["all","train","val","test"]:
                selected=mask & np.array([split=="all" or labels[p]["split"]==split for p in qids])
                pair_rows.append(dict(label_set=label_set,dimension=dim,label=label,split=split,
                    **paired_statistics(correct[:,bidx],correct[:,fidx],selected,qids,photos,groups)))
                ids=[qids[i] for i in np.flatnonzero(selected)]
                population.append(dict(label_set=label_set,dimension=dim,label=label,split=split,photos=len(ids),
                    identities=len({photos[p]["package_id"] for p in ids}),source_groups=len({groups[photos[p]["package_id"]]["group_id"] for p in ids}),
                    original_reviewed_photos=sum(labels[p]["scene_reviewed"] for p in ids),
                    independently_verified_photos=sum(any(e["photo_id"]==p for e in revision_events) for p in ids),
                    photo_ids="|".join(ids)))
            if label_set in ("original_all","original_reviewed","verified_snapshot"):
                for ci,cid in enumerate(cids):
                    available=mask & (correct[:,ci]>=0);n=int(available.sum());success=int(np.sum(correct[available,ci]))
                    low,high=wilson(success,n)
                    ids=[qids[i] for i in np.flatnonzero(available)];pids=sorted({photos[p]["package_id"] for p in ids})
                    config_scene.append(dict(label_set=label_set,config_id=cid,dimension=dim,label=label,photos=int(mask.sum()),available_predictions=n,
                        missing_predictions=int(mask.sum())-n,successes=success,errors=n-success,success_rate=ratio(success,n),
                        Wilson95_low=low,Wilson95_high=high,identities=len(pids),
                        package_macro_success_rate=float(np.mean([np.mean([correct[qids.index(p),ci] for p in ids if photos[p]["package_id"]==pkg]) for pkg in pids])) if pids else None,
                        uncertainty_scope="Wilson photo approximation descriptive; correlated identities require group bootstrap",
                        evidence_grade=EVIDENCE))
    write_csv(out/"B03_F03_scene_transitions.csv",pair_rows);write_csv(out/"scene_population_splits.csv",population)
    write_csv(out/"configuration_scene_uncertainty.csv",config_scene)
    dimension_denominators=[]
    for d in DIMENSIONS:
        states=Counter(visual_audit.get(p,{}).get('scene_dimensions',{}).get(d,{}).get('status','unreviewed') for p in photos)
        dimension_denominators.append(dict(dimension=d,total_photos=len(photos),
            original_reviewed_subset_photos=len(reviewed_labels),
            reviewed_known=states['reviewed_known'],reviewed_uncertain=states['reviewed_uncertain'],unreviewed=states['unreviewed'],
            reviewed_known_denominator=states['reviewed_known'],reviewed_audited_denominator=states['reviewed_known']+states['reviewed_uncertain'],
            original_counts=json.dumps(dict(Counter(l[d] for l in labels.values()))),
            reviewed_counts=json.dumps(dict(Counter(l[d] for l in revised.values()))),
            unknown_policy='unclear is separate; historical label does not fill reviewed uncertainty'))
    write_csv(out/'reviewed_scene_dimension_denominators.csv',dimension_denominators)
    overlaps=[]
    for ai,a in enumerate(cids):
        for bi,b in enumerate(cids):
            valid=(correct[:,ai]>=0)&(correct[:,bi]>=0);aa=correct[valid,ai].astype(bool);bb=correct[valid,bi].astype(bool)
            union=int(np.sum(~aa|~bb));intersection=int(np.sum(~aa&~bb))
            overlaps.append(dict(baseline=a,variant=b,available_photos=int(valid.sum()),both_correct=int(np.sum(aa&bb)),
                baseline_wrong_variant_correct=int(np.sum(~aa&bb)),baseline_correct_variant_wrong=int(np.sum(aa&~bb)),
                both_wrong=intersection,error_union=union,error_Jaccard=ratio(intersection,union),
                net=int(np.sum(~aa&bb)-np.sum(aa&~bb)),evidence_grade=EVIDENCE))
    write_csv(out/"all_config_pair_intersections.csv",overlaps)
    edges=[]
    for edge in controlled_comparisons():
        a,b=edge["baseline"],edge["variant"]
        if a in cids and b in cids:
            edges.append(dict(**edge,**paired_statistics(correct[:,cids.index(a)],correct[:,cids.index(b)],np.ones(len(qids),dtype=bool),qids,photos,groups)))
    write_csv(out/"controlled_method_transitions.csv",edges)
    overlap_photos=[]
    image_cids=[cid for cid,c in configs.items() if c["mode"]=="image"]
    for qi,pid in enumerate(qids):
        ncorrect=int(np.sum(correct[qi]==1));nwrong=int(np.sum(correct[qi]==0));nmissing=int(np.sum(correct[qi]<0))
        overlap_photos.append(dict(photo_id=pid,package_id=photos[pid]["package_id"],methods_correct=ncorrect,methods_wrong=nwrong,
            methods_missing=nmissing,oracle_any_correct=ncorrect>0,
            all_observed_methods_wrong=ncorrect==0 and nwrong>0,all_35_wrong=nwrong==len(cids),
            B03_wrong_other_image_correct=correct[qi,bidx]==0 and any(correct[qi,cids.index(c)]==1 for c in image_cids if c!="B03"),
            all_image_models_wrong=all(correct[qi,cids.index(c)]==0 for c in image_cids),
            correct_configurations="|".join(cids[j] for j in np.flatnonzero(correct[qi]==1)),
            failed_configurations="|".join(cids[j] for j in np.flatnonzero(correct[qi]==0)),
            **{d:labels[pid][d] for d in DIMENSIONS},original_reviewed=labels[pid]["scene_reviewed"]))
    write_csv(out/"cross_config_photo_evidence.csv",overlap_photos)
    decomp_path=root/"results/colnomic_full_audit/20261013_resume/full_fusion/all_query_decomposition.csv"
    decomp={}
    if decomp_path.exists():
        inputs[str(decomp_path.relative_to(root))]=file_hash(decomp_path)
        decomp={r["photo_id"]:r for r in read_csv(decomp_path)}
    foreground=[]
    for qi,pid in enumerate(qids):
        if labels[pid]["occlusion"]!="foreground":continue
        d=decomp.get(pid,{});audit=visual_audit.get(pid,{})
        foreground.append(dict(photo_id=pid,package_id=photos[pid]["package_id"],split=labels[pid]["split"],
            B03_correct=bool(correct[qi,bidx]==1),F03_correct=bool(correct[qi,fidx]==1),
            fusion_introduced_error=bool(correct[qi,bidx]==1 and correct[qi,fidx]==0),
            original_scene_reviewed=labels[pid]["scene_reviewed"],
            positive_reference=d.get("positive"),fusion_compared_wrong_reference=d.get("negative"),
            image_raw_gap=d.get("image_raw_gap"),BM25_raw_gap=d.get("BM25_raw_gap"),
            normalized_image_contribution=d.get("normalized_image_contribution"),
            normalized_text_contribution=d.get("normalized_text_contribution"),fusion_gap=d.get("fusion_gap"),
            score_conservation_error=d.get("conservation_error"),
            independently_reviewed_scene_labels=json.dumps(audit.get("scene_labels")),
            identity_region_obscured=audit.get("identity_region_obscured",audit.get("occlusion_detail",{}).get("identity_region_obscured","not_visually_verified")),
            occlusion_location=audit.get("occlusion_location",audit.get("occlusion_detail",{}).get("location","not_visually_verified")),
            occlusion_extent=audit.get("occlusion_extent",audit.get("occlusion_detail",{}).get("extent","not_visually_verified")),
            visual_review_path=audit.get("_record_path"),
            mechanism_limit="fusion-score relation does not establish an occlusion cause"))
    write_csv(out/"foreground_32_and_fusion_harms.csv",foreground)
    preprocessing=export_preprocessing(root,configs,specs,photos,predictions,labels,inputs,out)
    literal=[]
    for cid in cids:
        ps=list(predictions[cid].values());err=[p for p in ps if not p["correct"]]
        literal.append(dict(config_id=cid,predictions=len(ps),failures=len(err),
            correct_rank_gt10=sum(p.get("rank") is not None and p["rank"]>10 for p in ps),
            Recall10_unhit=sum(p.get("rank") is None or p["rank"]>10 for p in ps),
            auto_E2=sum(automatic_main(cid,p["photo_id"],predictions)=="E2" for p in err),
            auto_E2_equals_Recall10_unhit=(sum(automatic_main(cid,p["photo_id"],predictions)=="E2" for p in err)
                ==sum(p.get("rank") is None or p["rank"]>10 for p in ps))))
    write_csv(out/"literal_ranks_vs_E2.csv",literal)
    write_json(out/"label_definitions.json",dict(historical_MAIN=MAIN,existing_C0_C7=CONFUSION,
        source="dataset/tools/error_analysis.py",automatic_priority="E0/E1 -> E4/E5/E6 -> E2/E3 -> E7",
        scene_operational_definitions=dict(source="configs/photo_labeling.json and dataset/tools/photo_review.html",
            single_object="only target package, holding hand and supporting surface",
            multi_object="any additional distinct physical object, including keyboard, cable, mouse, cup or loose paper",
            excluded_from_additional_objects="holding hand, supporting table/floor/wall, surface patterns, shadows, target-package printed graphics",
            foreground_occlusion="external hand/finger/object covers target surface; self fold, invisible face and image crop excluded"),
        interpretation=dict(E0="automatic metadata inconsistency; needs data audit",E1="no positive design in reference mapping",
            E2="correct design rank >10; not evidence of candidate generation failure",E3="correct rank 2-10; not evidence of an independent ranking module defect",
            E4="image channel correct, text channel wrong, fusion wrong; inspect score/images for cause",
            E5="text channel correct, image channel wrong, fusion wrong; inspect score/images for cause",
            E6="both constituent channels correct but fusion wrong; construction-consistency flag",
            E7="not determined"),failure_label_status=STATUS,
        observable_difference_fields="independent multi-label visual review",
        mechanism_fields="independent evidence grades, not assigned from E or C code"))
    checks["inventory"]=dict(photos=len(photos),reference_rows=len(refs),unique_reference_sha=len({r["sha256"] for r in refs.values()}),
        configurations=len(configs),matrix_shape=list(correct.shape),evaluation_occurrences=len(matrix_rows),missing_prediction_rows=len(missing),
        original_reviewed_photos=sum(l["scene_reviewed"] for l in labels.values()),
        reviewed_count_matches_historical=sum(l["scene_reviewed"] for l in labels.values())==historical["scene_reviewed_photos"],
        independently_verified_scene_photos=len(revision_events),visual_records_available=len(visual_audit),
        consolidated_scene_dimensions_audited={d:sum(visual_audit.get(p,{}).get('scene_dimensions',{}).get(d,{}).get('status') in ('reviewed_known','reviewed_uncertain') for p in photos) for d in DIMENSIONS},
        new_current_configs_excluded=[c["id"] for c in current if c["id"] not in configs])
    checks["score_replays"]=replays
    checks["unmodified_inputs_after_run"]={name:file_hash(root/name)==sha for name,sha in inputs.items()}
    if not all(checks["unmodified_inputs_after_run"].values()):
        raise ValueError("Input changed during integration; rerun a consistent snapshot")
    checks["matrix_missing_mask_consistent"]=bool(np.sum(correct<0)==len(missing))
    checks["B03_F03_partition"]=next(r for r in pair_rows if r["label_set"]=="original_all" and r["dimension"]=="overall" and r["split"]=="all")
    overall=checks["B03_F03_partition"]
    if sum(overall[k] for k in ["both_correct","rescues","harms","both_wrong"])!=overall["available_photos"]:
        raise ValueError("B03/F03 four-cell count does not conserve photos")
    write_json(out/"validation.json",checks)
    write_json(out/"missing.json",missing)
    manifest=dict(schema="colnomic.scene35.integration.v1",created_utc=datetime.now(timezone.utc).isoformat(),
        historical_source=str(history.relative_to(root)),historical_manifest_sha256=file_hash(history/"manifest.json"),
        script_sha256=file_hash(Path(__file__)),inputs=inputs,inventory=checks["inventory"],
        evidence_grade=EVIDENCE,denominators=dict(class_error_rate="scene_errors/scene_available_predictions",
            error_composition="scene_errors/all_configuration_errors",type_occurrence="type_errors/scene_photos",
            type_within_errors="type_errors/scene_errors"),
        oracle=dict(correct=sum(r["oracle_any_correct"] for r in overlap_photos),photos=len(photos),
            scope="any of 35 configurations correct; posthoc oracle, not deployable retrieval accuracy"))
    write_json(out/"manifest.json",manifest)
    write_report(out,manifest,checks,summary,pair_rows,overlaps,foreground,preprocessing,config_scene)
    print(json.dumps(dict(matrix_shape=correct.shape,reviewed=manifest["inventory"]["original_reviewed_photos"],
        historical_tables_matching=all(c["matching"] for c in checks["historical_table_comparisons"].values()),
        oracle=manifest["oracle"],foreground_harms=sum(r["fusion_introduced_error"] for r in foreground)),ensure_ascii=False),flush=True)


def write_report(out,manifest,checks,summary,pairs,overlaps,foreground,preprocessing,config_scene):
    def table(headers,rows):
        return ["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(str(x) for x in row)+"|" for row in rows]+[""]
    inv=checks["inventory"]
    snapshot=read_json(out/"original_label_snapshot.json")
    reviewed=[l for l in snapshot.values() if l["scene_reviewed"]]
    selection_parts=[]
    for field,value,title in [("occlusion","foreground","前景遮挡"),("placement","tabletop","桌面"),("background","multi_object","多物体")]:
        reviewed_count=sum(l[field]==value for l in reviewed)
        all_count=sum(l[field]==value for l in snapshot.values())
        selection_parts.append(f"{title}：已审核 {reviewed_count}/{len(reviewed)}（{100*reviewed_count/len(reviewed):.1f}%），全量 {all_count}/{len(snapshot)}（{100*all_count/len(snapshot):.1f}%）")
    lines=["# 35 配置统计接续核验", "",
        f"共有 {inv['photos']} 张独立查询、{inv['reference_rows']} 条参考记录、{inv['unique_reference_sha']} 个 SHA 去重设计。"
        f"{inv['evaluation_occurrences']} 条是同一批照片上的 {inv['configurations']} 次重复评测，不是独立图片。缺失预测 {inv['missing_prediction_rows']} 条。", "",
        "当前项目汇总多了 "+", ".join(inv["new_current_configs_excluded"])+"；本报告按历史 manifest 的预测路径恢复原35配置，未混入额外配置。",
        "历史 summary/failure_types/failed_photos/joint_statistics/photo_labels 与三个宽表均逐字段重算；结果见 validation.json。", "",
        "## 原标签下 B03 → F03", "",
        "修复与损害分别报告；净变化是两者之差。以下分母是对应场景可用的逐图预测数。", ""]
    selected=[r for r in pairs if r["label_set"]=="original_all" and r["split"]=="all"]
    lines+=table(["场景维度","场景","N","身份","两者都对","图像错融合对","图像对融合错","两者都错","净变化"],
        [[r["dimension"],r["label"],r["available_photos"],r["identities"],r["both_correct"],r["rescues"],r["harms"],r["both_wrong"],r["net"]] for r in selected])
    lines += ["三个场景维度独立，各自分区完整；不能把维度数量相加。交叉场景、split、独立身份、来源组和配对 bootstrap 已保存。小样本 bootstrap 不稳定，不能把身份组成、可见面、尺度、清晰度差异解释为场景因果。", "",
        "## 审核状态", "",
        f"正式元数据 reviewed=true 经实际读取为 {inv['original_reviewed_photos']}/{inv['photos']}；与历史声明匹配为 {checks['inventory']['reviewed_count_matches_historical']}。"
        f"已读取独立场景修订 {inv['independently_verified_scene_photos']} 张、视觉记录 {inv['visual_records_available']} 张。未核验字段继续保留原标签和未知状态。", "",
        "original_labels/ 保存原标签全量统计，original_reviewed_subset/ 保存原已审核子集统计；verified_labels/ 在存在独立场景修订时保存核验快照。原元数据与 reviewed 标志均未改写，修订见 scene_label_revisions.json。", "",
        "原已审核子集不具有随机抽样保证："+"；".join(selection_parts)+"。这些数字是逐图标签快照的描述，不能代表全量。身份、split、场景构成见 scene_population_splits.csv。前景遮挡与身份关键区域被挡分开记录，后者需要逐图视觉证据。", "",
        ""]
    final_scene=[r for r in pairs if r['label_set']=='verified_snapshot' and r['split']=='all' and r['dimension']!='joint']
    lines+=['## 全量视觉审后重算','',
        '审后场景标签逐维度读取实际图像观察；不确定值单独列为unclear。reviewed_scene_dimension_denominators.csv给出每维已知、不确定、未审分母，configuration_scene_uncertainty.csv分别保存原全量、原reviewed子集和审后快照。','']
    lines+=table(['维度','标签','N','身份','B03正确','F03正确','修复','损害','净变化'],
        [[r['dimension'],r['label'],r['available_photos'],r['identities'],r['baseline_correct'],r['variant_correct'],r['rescues'],r['harms'],r['net']] for r in final_scene])
    harms=[r for r in foreground if r["fusion_introduced_error"]]
    lines+=['## 32 张原前景遮挡照片与8例融合损害实际复核','']
    lines+=table(["照片","遮挡物","图像贡献","文字贡献","融合差额","一般身份区被挡","对比候选区分规格被挡"],
        [[r["photo_id"],r.get('occluder','unknown'),r["normalized_image_contribution"],r["normalized_text_contribution"],r["fusion_gap"],r["identity_region_obscured"],r.get('pair_discriminative_spec_occlusion_status','not_visually_verified')] for r in harms])
    if any(r.get('focused_review_record_sha256') for r in harms):
        lines += ['8例均经原图、局部与正负候选实际审核；区分规格被挡yes3、none4、unknown1。q4920数量前缀缺失不能区分手指遮挡与原始画面截出；q21负参考为蓝色60片猫狗同框，Query数量行不可读；q3657的6片数仍可读；qdaac的144可读而125mg被挡。',
            '[8例细节与逐图图片链接](foreground_harms_reviewed.csv)及[原始人工记录](../foreground_harms_review/manual_reviews.json)保存不确定范围；旧32表与旧scene统计保留于before_final_scene_refresh/。','']
    lines += ["图像项与文字项来自 full_fusion 的正负候选确定性分解，原始 OCR 词贡献表用于定位评分来源。是否遮挡关键区域以独立视觉审核为准；32 张图像全部正确不证明普遍遮挡鲁棒性，这8例也不证明遮挡导致融合失败。", "",
        "## 方法含义与竞争解释", "",
        "B03 是原图 mean MaxSim；B05 是 PP-OCRv6 文本 BM25；B08 是 OCR 文字作为 ColNomic 查询匹配图像参考；B12 融合 B03+B08；F03 融合 B03+B05。两种融合的文字通道不同，虽同用逐 Query population z-score 和 α=0.5。", "",
        "B16 image_crisp 复用 B03 的同一输入、embedding 和相似度，改为参考 Token distinctiveness 加权 forward MaxSim − 加权均值 + reverse MaxSim。它没有裁切或提升像素清晰度，不能把差异归因于清晰度。全部35配置的 checkpoint、revision、输入合同与实际评分代码已记入 configuration_semantics.json。", "",
        "D01 使用人工 confirmed 主体框，在 EXIF 转正原图上加最大边5%边距裁切，属于额外定位监督；D03 使用 SAM3 'product package' 自动最高置信度 mask（阈值0.2，mask阈值0.5），先白底合成再加同样边距裁切。两者原分辨率存储，随后通过模型处理器缩放；背景、尺度与内容完整性同时变化。preprocessing_inputs.csv 保存2234个实际输入、SHA验证、裁切范围、主体像素和原OCR中心保留代理；代理不是人工可见性审核。", "",
        "## 全35配置错误交集", ""]
    b03=[r for r in overlaps if r["baseline"]=="B03" and r["variant"]!="B03"]
    lines+=table(["配置","B03错而该方法对","B03对而该方法错","共同失败","净变化"],
        [[r["variant"],r["baseline_wrong_variant_correct"],r["baseline_correct_variant_wrong"],r["both_wrong"],r["net"]] for r in b03])
    oracle=manifest["oracle"]
    lines += [f"任一配置正确的 oracle 为 {oracle['correct']}/{oracle['photos']}；它需要事后知道哪个方法正确，不是可部署系统的成绩。"
        "总体指标相近不代表错误相同。跨模型成功仅提供比较证据，不能确定 ColNomic 编码器或某个层的失败原因。", "",
        "## E0–E7 与 C0–C7", "",
        "保留历史 E 主标签以及 E0/E1 → E4/E5/E6 → E2/E3 → E7 优先级，全部为未人工校准规则。"
        "E2 的事实仅是正确设计排名>10；E3 仅是正确排名2–10；E4/E5 是图像、文字和融合正确性关系。"
        "literal_ranks_vs_E2.csv 分开保存 Recall@10 未命中与 E2 数量，避免优先级导致的漏计。", "",
        "C0–C7 已读取沿用 dataset/tools/error_analysis.py 原定义，但本脚本不猜测逐图包装关系。数据/参考有效性、实际排名、通道正确性、可见差异和机制证据分别存储。", "",
        "## 数据接口与复现", "",
        "- prediction_matrix.csv/.npz：1117×35 的逐图正确性、排名、第一名、可用完整图库分数差额；缺失使用显式哨兵。",
        "- all_config_pair_intersections.csv、controlled_method_transitions.csv、cross_config_photo_evidence.csv：全部方法两两交集与逐图互补。",
        "- B03_F03_scene_transitions.csv：每个独立场景/交叉场景及 split 下四格、修复、损害、净变化与来源组区间。",
        "- photo_evidence_interface.json：场景—配置—照片—独立视觉审核—Token 证据的连接。",
        "- validation.json：实际轴、全图库稳定排序、SHA去重、历史逐表逐字段、源文件未修改的验证。",
        "- final_scene_statistics/manifest.json：从已核验且未改变的1117×35矩阵重算最终scene定义修订，保留原35重放与新审核的独立来源。", "",
        "```bash", "OMP_NUM_THREADS=2 /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.scene35", "```", "",
        "上述统计属于确定性评分关系和描述性关联；尚未有逐图图像干预或竞争证据的类别不能升级为机制结论。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    page="<!doctype html><html lang='zh'><meta charset='utf-8'><title>35配置统计核验</title><style>body{font:16px sans-serif;margin:2em;max-width:1300px}pre{white-space:pre-wrap}a{color:#167}li{margin:8px 0}</style><h1>35配置统计核验</h1><ul>"
    page+="".join(f"<li><a href='{html.escape(p.name)}'>{html.escape(p.name)}</a></li>" for p in sorted(out.glob("*.csv")))
    page+="</ul><pre>"+html.escape("\n".join(lines))+"</pre></html>"
    (out/"index.html").write_text(page)


def refresh_final_scene(root,out,review_file,foreground_file):
    """Recompute scene summaries from the immutable independently replayed matrix."""
    root=Path(root).resolve();out=Path(out).resolve();final=out/'final_scene_statistics';final.mkdir(exist_ok=True)
    prior=out/'before_final_scene_refresh';prior.mkdir(exist_ok=True)
    source_manifest=read_json(out/'manifest.json');checks=read_json(out/'validation.json')
    matrices={name:file_hash(out/name) for name in ('prediction_matrix.csv','prediction_matrix.npz','matrix_schema.json')}
    if not all(v['status']=='passed' for v in checks['score_replays'].values()):raise ValueError('Initial full-score replay incomplete')
    if file_hash(out/'source_code_at_score_replay.py')!=source_manifest['script_sha256']:raise ValueError('Historical execution code snapshot changed')
    labels=read_json(out/'original_label_snapshot.json');inputs={}
    revised,events,audit=load_revisions(Path(review_file),labels,inputs,root)
    photos={r['photo_id']:r for r in read_csv(root/'dataset/lists/photos.csv')}
    groups={r['package_id']:r for r in read_csv(root/'dataset/lists/groups.csv')}
    matrix=np.load(out/'prediction_matrix.npz');qids=matrix['photo_ids'].tolist();cids=matrix['config_ids'].tolist();correct=matrix['correct']
    if qids!=list(photos) or correct.shape!=(1117,35) or (correct<0).any():raise ValueError('Current immutable matrix/inventory mismatch')
    rows=read_csv(out/'prediction_matrix.csv');refs={r['reference_id']:r for r in read_csv(root/'dataset/lists/references.csv')}
    configs={r['config_id']:dict(model=r['model'],mode=r['mode']) for r in read_csv(out/'original_labels/summary.csv') if r['dimension']=='overall'}
    predictions={cid:{} for cid in cids};axes={(pid,cid):(i,j) for i,pid in enumerate(qids) for j,cid in enumerate(cids)}
    for row in rows:
        pid,cid=row['photo_id'],row['config_id'];i,j=axes[pid,cid];value=row['correct']=='True'
        if value!=bool(correct[i,j]):raise ValueError('CSV/NPZ immutable correctness mismatch')
        positives=row['positive_reference_ids'].split('|')
        predictions[cid][pid]=dict(correct=value,data_issue=row['data_reference_validity']=='automatic_consistency_issue',
            positive_design_ids=sorted({refs[r]['design_id'] for r in positives}),package_id=row['package_id'],
            selected_design_id=row['top1_design_id'],selected_reference_id=row['top1_reference_id'],
            positive_reference_ids=positives,rank=int(row['correct_rank']) if row['correct_rank'] else None)
    if len(rows)!=39095 or len(axes)!=len(rows):raise ValueError('Immutable matrix CSV count mismatch')
    regenerate_tables(configs,predictions,revised,final/'verified_labels')
    for file in (out/'verified_labels').glob('*.csv'):
        target=prior/'verified_labels'/file.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():target.write_bytes(file.read_bytes())
    for file in (final/'verified_labels').glob('*.csv'):
        (out/'verified_labels'/file.name).write_bytes(file.read_bytes())
    label_sets=[('original_all',labels),('original_reviewed',{p:l for p,l in labels.items() if l['scene_reviewed']}),('verified_snapshot',revised)]
    pairs=[];population=[];config_scene=[];bidx=cids.index('B03');fidx=cids.index('F03')
    for label_set,label_map in label_sets:
        subset=np.array([p in label_map for p in qids])
        for dim,label,mask in scene_masks(labels if label_set=='original_reviewed' else label_map,qids):
            mask=mask & subset
            for split in ('all','train','val','test'):
                selected=mask & np.array([split=='all' or labels[p]['split']==split for p in qids])
                pairs.append(dict(label_set=label_set,dimension=dim,label=label,split=split,
                    **paired_statistics(correct[:,bidx],correct[:,fidx],selected,qids,photos,groups)))
                ids=[qids[i] for i in np.flatnonzero(selected)]
                population.append(dict(label_set=label_set,dimension=dim,label=label,split=split,photos=len(ids),
                    identities=len({photos[p]['package_id'] for p in ids}),source_groups=len({groups[photos[p]['package_id']]['group_id'] for p in ids}),
                    original_reviewed_photos=sum(labels[p]['scene_reviewed'] for p in ids),independently_verified_photos=sum(p in audit for p in ids),photo_ids='|'.join(ids)))
            for ci,cid in enumerate(cids):
                available=mask & (correct[:,ci]>=0);n=int(available.sum());success=int(np.sum(correct[available,ci]));low,high=wilson(success,n)
                ids=[qids[i] for i in np.flatnonzero(available)];pids=sorted({photos[p]['package_id'] for p in ids})
                config_scene.append(dict(label_set=label_set,config_id=cid,dimension=dim,label=label,photos=int(mask.sum()),available_predictions=n,
                    missing_predictions=int(mask.sum())-n,successes=success,errors=n-success,success_rate=ratio(success,n),Wilson95_low=low,Wilson95_high=high,
                    identities=len(pids),package_macro_success_rate=float(np.mean([np.mean([correct[qids.index(p),ci] for p in ids if photos[p]['package_id']==pkg]) for pkg in pids])) if pids else None,
                    uncertainty_scope='Wilson photo approximation descriptive; correlated identities require group bootstrap',evidence_grade=EVIDENCE))
    def preserve(name):
        source=out/name;target=prior/name
        if source.exists() and not target.exists():target.write_bytes(source.read_bytes())
    for name,values in [('B03_F03_scene_transitions.csv',pairs),('scene_population_splits.csv',population),('configuration_scene_uncertainty.csv',config_scene)]:
        preserve(name);write_csv(out/name,values);write_csv(final/name,values)
    denominators=[]
    for d in DIMENSIONS:
        states=Counter(audit[p]['scene_dimensions'][d]['status'] for p in photos)
        denominators.append(dict(dimension=d,total_photos=1117,original_reviewed_subset_photos=sum(l['scene_reviewed'] for l in labels.values()),
            reviewed_known=states['reviewed_known'],reviewed_uncertain=states['reviewed_uncertain'],unreviewed=states['unreviewed'],
            reviewed_known_denominator=states['reviewed_known'],reviewed_audited_denominator=states['reviewed_known']+states['reviewed_uncertain'],
            original_counts=json.dumps(dict(Counter(l[d] for l in labels.values()))),reviewed_counts=json.dumps(dict(Counter(l[d] for l in revised.values()))),
            unknown_policy='unclear is separate; historical label does not fill reviewed uncertainty'))
    preserve('reviewed_scene_dimension_denominators.csv');write_csv(out/'reviewed_scene_dimension_denominators.csv',denominators)
    foreground_reviews=read_json(foreground_file);foreground_byid={r['photo_id']:r for r in foreground_reviews}
    foreground=read_csv(out/'foreground_32_and_fusion_harms.csv');harm_ids={r['photo_id'] for r in foreground if r['fusion_introduced_error']=='True'}
    if len(foreground_reviews)!=8 or set(foreground_byid)!=harm_ids:raise ValueError('Focused review is not exact original eight harms')
    focused_path=str(Path(foreground_file).relative_to(root));focused_hash=file_hash(foreground_file)
    visual={r['photo_id']:r for r in read_json(root/'results/colnomic_full_audit/20261013_resume/visual_consolidated/records.json')}
    focused_rows=[];evidence_inputs={};declared={}
    def collect_declared(value):
        if isinstance(value,dict):
            if value.get('path') and value.get('sha256'):declared[value['path']]=value['sha256']
            for item in value.values():collect_declared(item)
        elif isinstance(value,list):
            for item in value:collect_declared(item)
    collect_declared(foreground_reviews)
    for name in ('detail_rois.json','candidate_detail_rois.json'):
        declaration=Path(foreground_file).parent/name;collect_declared(read_json(declaration))
        evidence_inputs[str(declaration.relative_to(root))]=file_hash(declaration)
    for row in foreground:
        pid=row['photo_id'];current=visual[pid];row['fusion_introduced_error']=row['fusion_introduced_error']=='True'
        for key in ('identity_region_obscured','occlusion_location','occlusion_extent'):
            if key in current:row[key]=current[key]
        row['visual_review_path']=current.get('occlusion_detail_source',row['visual_review_path'])
        if pid not in foreground_byid:continue
        raw=foreground_byid[pid]
        if raw['positive_reference_id']!=row['positive_reference'] or raw['fusion_negative_reference_id']!=row['fusion_compared_wrong_reference']:
            raise ValueError('Focused review/reference comparison mismatch')
        if raw.get('review_method')!='actual_view_image' or raw.get('actual_original_crop_and_GT_negative_detail_view_completed') is not True:
            raise ValueError('Focused review lacks explicit actual viewing status')
        for role in ('original','positive','fusion_negative','reviewed_box_snapshot'):
            item=raw[role]
            if file_hash(root/item['path'])!=item['sha256']:raise ValueError('Focused source image/bbox hash changed')
            evidence_inputs[item['path']]=item['sha256']
        review_record=root/raw['visual_review_path']
        if not review_record.exists():raise ValueError('Focused per-query review link missing')
        evidence_inputs[raw['visual_review_path']]=file_hash(review_record)
        for path in raw['actually_viewed_evidence_paths']:
            if path not in declared or file_hash(root/path)!=declared[path]:raise ValueError('Actually viewed focused image SHA mismatch')
            evidence_inputs[path]=declared[path]
        key=raw['identity_region_obscured'];spec=raw['pair_discriminative_spec_occlusion']
        row.update(identity_region_obscured=key['status'],occlusion_location=key['position'],occlusion_extent=key['degree'],
            occluder=raw['occluder'],pair_discriminative_spec_occlusion_status=spec['status'],
            pair_positive_spec=spec['positive_spec'],pair_negative_spec=spec['negative_spec'],pair_query_observation=spec['query_observation'],
            focused_review_record_path=focused_path,focused_review_record_sha256=focused_hash,
            visual_review_path=raw['visual_review_path'],actually_viewed_evidence_paths='|'.join(raw['actually_viewed_evidence_paths']),
            fine_detail_verified=True,fine_detail_scope=raw['fine_detail_scope'],remaining_unknowns=json.dumps(raw['unknowns'],ensure_ascii=False))
        focused_rows.append(row.copy())
    preserve('foreground_32_and_fusion_harms.csv');write_csv(out/'foreground_32_and_fusion_harms.csv',foreground);write_csv(out/'foreground_harms_reviewed.csv',focused_rows)
    interface=read_json(out/'photo_evidence_interface.json');preserve('photo_evidence_interface.json')
    for row in interface:
        pid=row['photo_id'];row['verified_scene_labels']={d:revised[pid][d] for d in DIMENSIONS};row['reviewed_scene_dimension_states']=audit[pid]['scene_dimensions']
        row['visual_review_path']=str(Path(review_file).relative_to(root))
        row['final_visual_record_source']='results/colnomic_full_audit/20261013_resume/visual_consolidated/records.json'
        if pid in foreground_byid:
            row['foreground_harm_actual_review']={k:foreground_byid[pid][k] for k in ('identity_region_obscured','pair_discriminative_spec_occlusion','occluder','actual_visual_observations','unknowns','fine_detail_scope','actually_viewed_evidence_paths')}
            row['foreground_harm_actual_review'].update(record_path=focused_path,record_sha256=focused_hash)
    write_json(out/'photo_evidence_interface.json',interface);write_json(final/'scene_label_revisions.json',events)
    preserve('scene_label_revisions.json');write_json(out/'scene_label_revisions.json',events)
    validation=dict(matrix_files_byte_unchanged=all(file_hash(out/name)==digest for name,digest in matrices.items()),
        initial_score_replay_inputs_unchanged=all(file_hash(root/path)==digest for path,digest in source_manifest['inputs'].items()),
        final_scene_sources_and_evidence_sha_matching=all(file_hash(root/path)==digest for path,digest in inputs.items()),
        focused_actual_evidence_sha_matching=all(file_hash(root/path)==digest for path,digest in evidence_inputs.items()),
        focused_harms_exact8=set(foreground_byid)==harm_ids and len(focused_rows)==8,
        complete1117_scene_join=len(interface)==len(audit)==len(revised)==1117,
        final_dimension_partitions=all(sum(r['available_photos'] for r in pairs if r['label_set']=='verified_snapshot' and r['split']=='all' and r['dimension']==d)==1117 for d in DIMENSIONS))
    if not all(validation.values()):raise ValueError('Final scene refresh failed integrity checks')
    write_json(final/'validation.json',validation)
    write_json(final/'manifest.json',dict(schema='colnomic.scene35.final_scene_refresh.v1',script_sha256=file_hash(Path(__file__)),
        initial_score_replay_manifest_sha256=file_hash(out/'manifest.json'),immutable_matrix_sha256=matrices,
        final_scene_input=str(Path(review_file).relative_to(root)),final_scene_input_sha256=file_hash(review_file),
        foreground_actual_review_path=focused_path,foreground_actual_review_sha256=focused_hash,scene_sources_and_images_sha256=inputs,
        foreground_sources_and_images_sha256=evidence_inputs,final_visual_records_sha256=file_hash(root/'results/colnomic_full_audit/20261013_resume/visual_consolidated/records.json'),
        preserved_prior_artifacts={str((out/name).relative_to(root)):dict(snapshot=str((prior/name).relative_to(root)),sha256=file_hash(prior/name)) for name in ('foreground_32_and_fusion_harms.csv','photo_evidence_interface.json')},
        scope='descriptive scene reaggregation only; original metadata, all35 prediction and score replay matrix unchanged',validation=validation))
    checks['final_scene_refresh']=validation;write_json(out/'validation.json',checks)
    write_report(out,source_manifest,checks,read_csv(out/'original_labels/summary.csv'),pairs,read_csv(out/'all_config_pair_intersections.csv'),foreground,read_csv(out/'preprocessing_inputs.csv'),config_scene)
    print(json.dumps(dict(final_scene_counts={d:dict(Counter(l[d] for l in revised.values())) for d in DIMENSIONS},focused_spec_occlusion=dict(Counter(r['pair_discriminative_spec_occlusion_status'] for r in focused_rows)),validation=validation),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--review-dir",type=Path)
    parser.add_argument('--refresh-final-scene',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    out=args.output or args.root/"results/colnomic_full_audit/20261013_resume/scene35"
    review=args.review_dir or args.root/"results/colnomic_full_audit/20261013_resume/visual_reviews"
    if args.refresh_final_scene:
        review=args.review_dir or args.root/'results/colnomic_full_audit/20261013_resume/visual_consolidated/scene_revisions_final.json'
        refresh_final_scene(args.root,out,review,args.root/'results/colnomic_full_audit/20261013_resume/foreground_harms_review/join_records.json')
    else:run(args.root,out,review)


if __name__=="__main__":
    main()
