# Conservative Qwen3-VL-Reranker rerank for ColNomic top-k candidates.
#
# This uses the official Qwen3-VL-Reranker wrapper downloaded with the model.
# Inputs are query image + query OCR versus gallery image + optional gallery OCR.
# ColNomic rank remains the backbone; Qwen3-VL rerank scores only act as a
# conservative signal and must pass a margin gate before replacing top-1.

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from rerank_monoqwen import (
    DEFAULT_GALLERY_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PREDS_JSONL,
    Metrics,
    build_gallery_setid_to_path,
    clean_query_text,
    compute_metrics,
    eval_gt_from_item,
    load_image,
    load_preds_jsonl,
)
from rerank_monoqwen_deepseek_ocr import get_ocr_text, load_ocr_jsonl
from rerank_monoqwen_hybrid import base_rank_score


DEFAULT_MODEL_DIR = "/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen3-VL-Reranker-2B"
DEFAULT_QUERY_OCR_JSONL = (
    "/hkfs/work/workspace/scratch/ap7811-benchmark/colqwen/result/+ocr/ocr/"
    "query_ocr_deepseek_raw_gallery.jsonl"
)
DEFAULT_GALLERY_OCR_JSONL = "/hkfs/work/workspace/scratch/ap7811-benchmark/ocr_cache/gallery_ocr_deepseek_raw_gallery.jsonl"

INSTRUCTION = (
    "Retrieve the exact same medicine product image. Use visual identity first. "
    "Use OCR text only as supporting evidence. Penalize visually similar but different "
    "products, strengths, package variants, or unrelated products."
)


def _path_keys(path: str) -> List[str]:
    if not path:
        return []
    abs_path = os.path.abspath(path)
    real_path = os.path.realpath(abs_path)
    keys = [abs_path]
    if real_path != abs_path:
        keys.append(real_path)
    return keys


def load_gallery_ocr_by_setid(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            text = item.get("text") or ""
            img_path = item.get("img_path") or item.get("real_img_path") or ""
            if not img_path:
                continue
            setid = os.path.splitext(os.path.basename(img_path))[0].strip()
            if setid and setid not in out:
                out[setid] = text
    return out


def build_final_order(
    base_preds: List[str],
    rerank_scores_by_sid: Dict[str, float],
    rerank_topk: int,
    final_topk: int,
    alpha_base: float,
    beta_rerank: float,
    base_score_mode: str,
    switch_margin: float,
) -> Tuple[List[str], List[dict], bool, float]:
    candidates = base_preds[:rerank_topk]
    table: List[dict] = []
    for rank0, sid in enumerate(candidates):
        bscore = base_rank_score(rank0, rerank_topk, base_score_mode)
        rscore = rerank_scores_by_sid.get(sid)
        hscore = alpha_base * bscore + beta_rerank * (rscore if rscore is not None else 0.0)
        table.append(
            {
                "setid": sid,
                "base_rank": rank0 + 1,
                "base_score": bscore,
                "rerank_score": rscore,
                "hybrid_score": hscore,
            }
        )

    sorted_table = sorted(table, key=lambda x: (x["hybrid_score"], -x["base_rank"]), reverse=True)
    hybrid_order = [x["setid"] for x in sorted_table]
    switched_top1 = False
    switch_advantage = 0.0

    if base_preds and hybrid_order and hybrid_order[0] != base_preds[0]:
        old_top1 = base_preds[0]
        new_top1 = hybrid_order[0]
        old_r = rerank_scores_by_sid.get(old_top1)
        new_r = rerank_scores_by_sid.get(new_top1)
        switch_advantage = (new_r - old_r) if old_r is not None and new_r is not None else 0.0
        if switch_advantage >= switch_margin:
            switched_top1 = True
        else:
            hybrid_order = [old_top1] + [sid for sid in hybrid_order if sid != old_top1]

    seen = set()
    final_preds: List[str] = []
    for sid in hybrid_order + base_preds:
        if sid in seen:
            continue
        seen.add(sid)
        final_preds.append(sid)
        if len(final_preds) >= final_topk:
            break
    return final_preds, table, switched_top1, switch_advantage


def load_qwen3vl_reranker(model_dir: str, torch_dtype: str, attn_implementation: str):
    script_dir = os.path.join(model_dir, "scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from qwen3_vl_reranker import Qwen3VLReranker

    kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    reranker = Qwen3VLReranker(model_name_or_path=model_dir, **kwargs)

    # The downloaded wrapper can leave multimodal token-type fields as Python
    # lists. Newer transformers indexes them with tensor masks, so convert
    # simple numeric/list fields to tensors before model.forward.
    original_tokenize = reranker.tokenize

    def tokenize_with_tensor_fields(pairs):
        inputs = original_tokenize(pairs)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            try:
                if len(value) == 0:
                    continue
                if all(isinstance(x, int) for x in value):
                    inputs[key] = torch.tensor(value, dtype=torch.long)
                elif all(isinstance(x, Sequence) and not isinstance(x, (str, bytes)) for x in value):
                    inputs[key] = torch.tensor(value, dtype=torch.long)
            except Exception:
                # Keep image/video pixel fields untouched if they are not
                # regular numeric tensors.
                pass
        return inputs

    reranker.tokenize = tokenize_with_tensor_fields
    return reranker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=DEFAULT_PREDS_JSONL)
    ap.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--query_ocr_jsonl", default=DEFAULT_QUERY_OCR_JSONL)
    ap.add_argument("--gallery_ocr_jsonl", default=DEFAULT_GALLERY_OCR_JSONL)
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out_jsonl", default=os.path.join(DEFAULT_OUT_DIR, "rerank_preds_qwen3vl_hybrid_image_only_7b.jsonl"))
    ap.add_argument("--out_metrics", default=os.path.join(DEFAULT_OUT_DIR, "rerank_metrics_qwen3vl_hybrid_image_only_7b.json"))

    ap.add_argument("--rerank_topk", type=int, default=10)
    ap.add_argument("--final_topk", type=int, default=10)
    ap.add_argument("--min_chars", type=int, default=0)
    ap.add_argument("--max_query_chars", type=int, default=1200)
    ap.add_argument("--max_doc_chars", type=int, default=800)
    ap.add_argument("--alpha_base", type=float, default=0.80)
    ap.add_argument("--beta_rerank", type=float, default=0.20)
    ap.add_argument("--base_score_mode", choices=["linear", "reciprocal"], default="linear")
    ap.add_argument("--switch_margin", type=float, default=0.15)
    ap.add_argument("--instruction", default=INSTRUCTION)
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    items = load_preds_jsonl(args.preds_jsonl)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[INFO] loaded {len(items)} query preds from {args.preds_jsonl}")

    query_ocr = load_ocr_jsonl(args.query_ocr_jsonl)
    gallery_ocr = load_gallery_ocr_by_setid(args.gallery_ocr_jsonl)
    gallery_map = build_gallery_setid_to_path(args.gallery_dir)
    print(f"[INFO] query OCR keys={len(query_ocr)} from {args.query_ocr_jsonl}")
    print(f"[INFO] gallery OCR setids={len(gallery_ocr)} from {args.gallery_ocr_jsonl}")
    print(f"[INFO] gallery setids={len(gallery_map)} from {args.gallery_dir}")

    done: Dict[int, dict] = {}
    if args.skip_done and os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["query_index"]] = rec
                except Exception:
                    continue
        print(f"[INFO] resume: {len(done)} queries already done")

    reranker = load_qwen3vl_reranker(args.model_dir, args.torch_dtype, args.attn_implementation)

    gt_list: List[str] = []
    base_all: List[List[str]] = []
    hybrid_all: List[List[str]] = []
    missing_ocr = 0
    short_ocr = 0
    missing_gallery_ocr = 0
    switched_top1_count = 0
    changed_order_count = 0
    demoted_correct_top1 = 0
    promoted_correct_top1 = 0

    out_f = open(args.out_jsonl, "a" if args.skip_done else "w", encoding="utf-8")

    for item in tqdm(items, desc="qwen3vl hybrid rerank"):
        qidx = item["query_index"]
        qsid = item["query_setid"]
        eval_qsid = eval_gt_from_item(item)
        qpath = item["query_path"]
        base_preds = item.get("preds") or ["__MISSING__"]

        gt_list.append(eval_qsid)
        base_all.append(base_preds)

        if args.skip_done and qidx in done:
            final_preds = done[qidx]["preds_hybrid"]
            hybrid_all.append(final_preds)
            continue

        t0 = time.time()
        raw_qtext = get_ocr_text(query_ocr, qpath)
        if not raw_qtext:
            missing_ocr += 1
        qtext = clean_query_text(raw_qtext, max_chars=args.max_query_chars)
        qimage = qpath if os.path.exists(qpath) else None

        if len(qtext) < args.min_chars:
            short_ocr += 1
            final_preds = base_preds[: args.final_topk]
            scores_by_sid: Dict[str, float] = {}
            table: List[dict] = []
            switched_top1 = False
            switch_advantage = 0.0
            reranked = False
        else:
            documents = []
            scored_setids: List[str] = []
            for sid in base_preds[: args.rerank_topk]:
                gpath = gallery_map.get(sid)
                if not gpath:
                    continue
                gtext = clean_query_text(gallery_ocr.get(sid, ""), max_chars=args.max_doc_chars)
                if not gtext:
                    missing_gallery_ocr += 1
                documents.append({"text": gtext or None, "image": gpath})
                scored_setids.append(sid)

            scores = reranker.process(
                {
                    "instruction": args.instruction,
                    "query": {"text": qtext or None, "image": qimage},
                    "documents": documents,
                }
            )
            scores_by_sid = dict(zip(scored_setids, [float(x) for x in scores]))
            final_preds, table, switched_top1, switch_advantage = build_final_order(
                base_preds=base_preds,
                rerank_scores_by_sid=scores_by_sid,
                rerank_topk=args.rerank_topk,
                final_topk=args.final_topk,
                alpha_base=args.alpha_base,
                beta_rerank=args.beta_rerank,
                base_score_mode=args.base_score_mode,
                switch_margin=args.switch_margin,
            )
            reranked = True

        if switched_top1:
            switched_top1_count += 1
        if final_preds[: args.final_topk] != base_preds[: args.final_topk]:
            changed_order_count += 1
        if base_preds and base_preds[0] == eval_qsid and (not final_preds or final_preds[0] != eval_qsid):
            demoted_correct_top1 += 1
        if base_preds and base_preds[0] != eval_qsid and final_preds and final_preds[0] == eval_qsid:
            promoted_correct_top1 += 1

        hybrid_all.append(final_preds)
        out_f.write(
            json.dumps(
                {
                    "query_index": qidx,
                    "query_setid": qsid,
                    "eval_query_setid": eval_qsid,
                    "gt_source": "query_path" if eval_qsid != qsid else "query_setid",
                    "query_path": qpath,
                    "query_ocr_jsonl": args.query_ocr_jsonl,
                    "gallery_ocr_jsonl": args.gallery_ocr_jsonl,
                    "query_text_chars": len(qtext),
                    "reranked": reranked,
                    "switched_top1": switched_top1,
                    "switch_advantage": switch_advantage,
                    "alpha_base": args.alpha_base,
                    "beta_rerank": args.beta_rerank,
                    "switch_margin": args.switch_margin,
                    "preds_colnomic": base_preds,
                    "preds_hybrid": final_preds,
                    "candidate_scores": table,
                    "time_ms": (time.time() - t0) * 1000.0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out_f.flush()

    out_f.close()

    metrics_base: Metrics = compute_metrics(gt_list, base_all)
    metrics_hybrid: Metrics = compute_metrics(gt_list, hybrid_all)
    payload = {
        "method": "qwen3vl_reranker_hybrid",
        "model_dir": args.model_dir,
        "preds_jsonl": args.preds_jsonl,
        "gallery_dir": args.gallery_dir,
        "query_ocr_jsonl": args.query_ocr_jsonl,
        "gallery_ocr_jsonl": args.gallery_ocr_jsonl,
        "rerank_topk": args.rerank_topk,
        "final_topk": args.final_topk,
        "alpha_base": args.alpha_base,
        "beta_rerank": args.beta_rerank,
        "base_score_mode": args.base_score_mode,
        "switch_margin": args.switch_margin,
        "instruction": args.instruction,
        "n_queries": metrics_base.n_queries,
        "missing_ocr": missing_ocr,
        "short_ocr": short_ocr,
        "missing_gallery_ocr": missing_gallery_ocr,
        "changed_order_count": changed_order_count,
        "switched_top1_count": switched_top1_count,
        "demoted_correct_top1": demoted_correct_top1,
        "promoted_correct_top1": promoted_correct_top1,
        "colnomic_baseline": {
            "recall@1": metrics_base.recall_at_1,
            "recall@5": metrics_base.recall_at_5,
            "recall@10": metrics_base.recall_at_10,
            "mrr": metrics_base.mrr,
        },
        "qwen3vl_hybrid": {
            "recall@1": metrics_hybrid.recall_at_1,
            "recall@5": metrics_hybrid.recall_at_5,
            "recall@10": metrics_hybrid.recall_at_10,
            "mrr": metrics_hybrid.mrr,
        },
    }
    os.makedirs(os.path.dirname(args.out_metrics) or ".", exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("saved metrics ->", args.out_metrics)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
