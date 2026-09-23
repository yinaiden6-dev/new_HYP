# Rerank ColNomic Top-K candidates with MonoQwen2-VL, using cached DeepSeek-OCR
# text as the query. This keeps the old rerank_monoqwen.py reproducible while
# removing the extra Qwen2-VL query-image OCR step.

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from rerank_monoqwen import (
    DEFAULT_ADAPTER,
    DEFAULT_BASE_MODEL,
    DEFAULT_GALLERY_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PREDS_JSONL,
    Metrics,
    build_gallery_setid_to_path,
    clean_query_text,
    compute_metrics,
    eval_gt_from_item,
    load_image,
    load_reranker,
    score_candidates,
)


DEFAULT_QUERY_OCR_JSONL = (
    "/hkfs/work/workspace/scratch/ap7811-benchmark/colqwen/result/+ocr/ocr/"
    "query_ocr_deepseek_raw_gallery.jsonl"
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


def load_ocr_jsonl(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"query OCR jsonl not found: {path}")
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
            for field in ("img_path", "real_img_path"):
                for key in _path_keys((item.get(field) or "").strip()):
                    out[key] = text
    return out


def get_ocr_text(ocr_map: Dict[str, str], query_path: str) -> str:
    for key in _path_keys(query_path):
        if key in ocr_map:
            return ocr_map[key]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=DEFAULT_PREDS_JSONL)
    ap.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--query_ocr_jsonl", default=DEFAULT_QUERY_OCR_JSONL)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)

    ap.add_argument(
        "--out_jsonl",
        default=os.path.join(DEFAULT_OUT_DIR, "rerank_preds_deepseek_ocr_image_only_7b.jsonl"),
    )
    ap.add_argument(
        "--out_metrics",
        default=os.path.join(DEFAULT_OUT_DIR, "rerank_metrics_deepseek_ocr_image_only_7b.json"),
    )

    ap.add_argument("--rerank_topk", type=int, default=10)
    ap.add_argument("--final_topk", type=int, default=10)
    ap.add_argument("--candidate_batch_size", type=int, default=5)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    device = torch.device(args.device)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    from rerank_monoqwen import load_preds_jsonl

    items = load_preds_jsonl(args.preds_jsonl)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[INFO] loaded {len(items)} query preds from {args.preds_jsonl}")

    query_ocr = load_ocr_jsonl(args.query_ocr_jsonl)
    print(f"[INFO] loaded query DeepSeek-OCR jsonl: {args.query_ocr_jsonl} keys={len(query_ocr)}")

    gallery_map = build_gallery_setid_to_path(args.gallery_dir)
    print(f"[INFO] gallery_dir={args.gallery_dir} unique_setids={len(gallery_map)}")

    done_preds: Dict[int, List[str]] = {}
    if args.skip_done and os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_preds[rec["query_index"]] = rec["preds_rerank"]
                except Exception:
                    continue
        print(f"[INFO] resume: {len(done_preds)} queries already done")

    out_f = open(args.out_jsonl, "a" if args.skip_done else "w", encoding="utf-8")

    model, processor = load_reranker(
        args.base_model, args.adapter, device, torch_dtype, args.min_pixels, args.max_pixels
    )
    true_token_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_token_id = processor.tokenizer.convert_tokens_to_ids("False")
    print(f"[INFO] reranker loaded. true_id={true_token_id} false_id={false_token_id}")

    gt_list: List[str] = []
    colnomic_preds: List[List[str]] = []
    rerank_preds: List[List[str]] = []
    missing_ocr = 0
    short_ocr = 0

    for item in tqdm(items, desc="rerank queries"):
        qidx = item["query_index"]
        qsid = item["query_setid"]
        eval_qsid = eval_gt_from_item(item)
        qpath = item["query_path"]
        base_preds = item.get("preds") or ["__MISSING__"]

        gt_list.append(eval_qsid)
        colnomic_preds.append(base_preds)

        if args.skip_done and qidx in done_preds:
            rerank_preds.append(done_preds[qidx])
            continue

        t0 = time.time()
        raw_ocr_text = get_ocr_text(query_ocr, qpath)
        if not raw_ocr_text:
            missing_ocr += 1
        query_text = clean_query_text(raw_ocr_text, max_chars=args.max_chars)

        if len(query_text) < args.min_chars:
            short_ocr += 1
            final_preds = base_preds[: args.final_topk]
            scores = []
            reranked = False
        else:
            cand_setids = base_preds[: args.rerank_topk]
            scored_setids: List[str] = []
            scored_scores: List[float] = []
            unscored_setids: List[str] = []

            cand_with_path = []
            for sid in cand_setids:
                path = gallery_map.get(sid)
                if path is None:
                    unscored_setids.append(sid)
                else:
                    cand_with_path.append((sid, path))

            for start in range(0, len(cand_with_path), args.candidate_batch_size):
                chunk = cand_with_path[start:start + args.candidate_batch_size]
                images: List[Image.Image] = []
                keep_sids: List[str] = []
                for sid, path in chunk:
                    image = load_image(path)
                    if image is None:
                        unscored_setids.append(sid)
                    else:
                        keep_sids.append(sid)
                        images.append(image)
                if not keep_sids:
                    continue
                scores_chunk = score_candidates(
                    model, processor, device, query_text, images, true_token_id, false_token_id
                )
                scored_setids.extend(keep_sids)
                scored_scores.extend(scores_chunk)

            order = sorted(range(len(scored_setids)), key=lambda i: scored_scores[i], reverse=True)
            final_preds = [scored_setids[i] for i in order] + unscored_setids
            final_preds = final_preds[: args.final_topk]
            scores = [scored_scores[i] for i in order]
            reranked = True

        rerank_preds.append(final_preds)
        elapsed_ms = (time.time() - t0) * 1000.0

        out_f.write(
            json.dumps(
                {
                    "query_index": qidx,
                    "query_setid": qsid,
                    "eval_query_setid": eval_qsid,
                    "gt_source": "query_path" if eval_qsid != qsid else "query_setid",
                    "query_path": qpath,
                    "query_ocr_jsonl": args.query_ocr_jsonl,
                    "query_text_chars": len(query_text),
                    "reranked": reranked,
                    "preds_colnomic": base_preds,
                    "preds_rerank": final_preds,
                    "scores_rerank": scores,
                    "time_ms": elapsed_ms,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out_f.flush()

    out_f.close()

    metrics_colnomic: Metrics = compute_metrics(gt_list, colnomic_preds)
    metrics_rerank: Metrics = compute_metrics(gt_list, rerank_preds)
    payload = {
        "preds_jsonl": args.preds_jsonl,
        "gallery_dir": args.gallery_dir,
        "query_ocr_jsonl": args.query_ocr_jsonl,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "rerank_topk": args.rerank_topk,
        "final_topk": args.final_topk,
        "min_chars": args.min_chars,
        "n_queries": metrics_colnomic.n_queries,
        "missing_ocr": missing_ocr,
        "short_ocr": short_ocr,
        "colnomic_baseline": {
            "recall@1": metrics_colnomic.recall_at_1,
            "recall@5": metrics_colnomic.recall_at_5,
            "recall@10": metrics_colnomic.recall_at_10,
            "mrr": metrics_colnomic.mrr,
        },
        "monoqwen_deepseek_ocr_rerank": {
            "recall@1": metrics_rerank.recall_at_1,
            "recall@5": metrics_rerank.recall_at_5,
            "recall@10": metrics_rerank.recall_at_10,
            "mrr": metrics_rerank.mrr,
        },
    }
    os.makedirs(os.path.dirname(args.out_metrics) or ".", exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("saved metrics ->", args.out_metrics)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
