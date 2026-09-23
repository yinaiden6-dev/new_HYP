# Conservative MonoQwen rerank for ColNomic top-k candidates.
#
# This keeps the original ColNomic rank as the backbone, adds MonoQwen scores as
# a weak signal, and only changes the first result when the reranker has a clear
# margin over the original ColNomic top-1. The reranker prompt can optionally use
# both the query image and cached query OCR.

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

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
    load_preds_jsonl,
    load_reranker,
)
from rerank_monoqwen_deepseek_ocr import get_ocr_text, load_ocr_jsonl


DEFAULT_QUERY_OCR_JSONL = (
    "/hkfs/work/workspace/scratch/ap7811-benchmark/colqwen/result/+ocr/ocr/"
    "query_ocr_deepseek_raw_gallery.jsonl"
)

HYBRID_PROMPT = (
    "You are judging whether a candidate gallery product image matches the query product. "
    "The first image is the query product photo. The second image is the candidate gallery image. "
    "Use visual identity first, and use the OCR text only as supporting evidence. "
    "Answer only True or False. Query OCR text: {query}"
)

TEXT_ONLY_PROMPT = (
    "Assert whether the candidate gallery product image matches the query product. "
    "Use the OCR text as supporting evidence but prefer exact product identity. "
    "Answer only True or False. Query OCR text: {query}"
)


def score_candidates_hybrid(
    model,
    processor,
    device: torch.device,
    query_text: str,
    query_image: Optional[Image.Image],
    candidate_images: List[Image.Image],
    true_token_id: int,
    false_token_id: int,
    include_query_image: bool,
) -> List[float]:
    if include_query_image and query_image is not None:
        prompt = HYBRID_PROMPT.format(query=query_text)
        texts: List[str] = []
        images: List[Image.Image] = []
        for cand_img in candidate_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.extend([query_image, cand_img])
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
    else:
        prompt = TEXT_ONLY_PROMPT.format(query=query_text)
        texts = []
        for _ in candidate_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=candidate_images, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    attn = inputs["attention_mask"]
    last_idx = (attn.cumsum(dim=1) * attn).argmax(dim=1)
    logits_last = outputs.logits[torch.arange(outputs.logits.size(0)), last_idx, :]
    probs = torch.softmax(logits_last[:, [true_token_id, false_token_id]].float(), dim=-1)
    return probs[:, 0].cpu().tolist()


def base_rank_score(rank0: int, topk: int, mode: str) -> float:
    if mode == "reciprocal":
        return 1.0 / float(rank0 + 1)
    if topk <= 1:
        return 1.0
    return 1.0 - (float(rank0) / float(topk - 1))


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
        rscore_for_hybrid = rscore if rscore is not None else 0.0
        hscore = alpha_base * bscore + beta_rerank * rscore_for_hybrid
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
        if old_r is not None and new_r is not None:
            switch_advantage = new_r - old_r
        else:
            switch_advantage = 0.0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=DEFAULT_PREDS_JSONL)
    ap.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--query_ocr_jsonl", default=DEFAULT_QUERY_OCR_JSONL)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument(
        "--out_jsonl",
        default=os.path.join(DEFAULT_OUT_DIR, "rerank_preds_hybrid_queryimg_ocr_image_only_7b.jsonl"),
    )
    ap.add_argument(
        "--out_metrics",
        default=os.path.join(DEFAULT_OUT_DIR, "rerank_metrics_hybrid_queryimg_ocr_image_only_7b.json"),
    )

    ap.add_argument("--rerank_topk", type=int, default=10)
    ap.add_argument("--final_topk", type=int, default=10)
    ap.add_argument("--candidate_batch_size", type=int, default=2)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--min_pixels", type=int, default=192 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--alpha_base", type=float, default=0.80)
    ap.add_argument("--beta_rerank", type=float, default=0.20)
    ap.add_argument("--base_score_mode", choices=["linear", "reciprocal"], default="linear")
    ap.add_argument("--switch_margin", type=float, default=0.15)
    ap.add_argument("--include_query_image", action="store_true")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    device = torch.device(args.device)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    items = load_preds_jsonl(args.preds_jsonl)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[INFO] loaded {len(items)} query preds from {args.preds_jsonl}")

    query_ocr = load_ocr_jsonl(args.query_ocr_jsonl)
    print(f"[INFO] loaded query DeepSeek-OCR jsonl: {args.query_ocr_jsonl} keys={len(query_ocr)}")

    gallery_map = build_gallery_setid_to_path(args.gallery_dir)
    print(f"[INFO] gallery_dir={args.gallery_dir} unique_setids={len(gallery_map)}")

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

    model, processor = load_reranker(
        args.base_model, args.adapter, device, torch_dtype, args.min_pixels, args.max_pixels
    )
    true_token_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_token_id = processor.tokenizer.convert_tokens_to_ids("False")
    print(f"[INFO] reranker loaded. true_id={true_token_id} false_id={false_token_id}")

    gt_list: List[str] = []
    colnomic_preds: List[List[str]] = []
    hybrid_preds: List[List[str]] = []
    missing_ocr = 0
    short_ocr = 0
    missing_query_image = 0
    switched_top1_count = 0
    changed_order_count = 0
    demoted_correct_top1 = 0
    promoted_correct_top1 = 0

    out_f = open(args.out_jsonl, "a" if args.skip_done else "w", encoding="utf-8")

    for item in tqdm(items, desc="hybrid rerank queries"):
        qidx = item["query_index"]
        qsid = item["query_setid"]
        eval_qsid = eval_gt_from_item(item)
        qpath = item["query_path"]
        base_preds = item.get("preds") or ["__MISSING__"]

        gt_list.append(eval_qsid)
        colnomic_preds.append(base_preds)

        if args.skip_done and qidx in done:
            final_preds = done[qidx]["preds_hybrid"]
            hybrid_preds.append(final_preds)
            if final_preds[: args.final_topk] != base_preds[: args.final_topk]:
                changed_order_count += 1
            if base_preds and base_preds[0] == eval_qsid and (not final_preds or final_preds[0] != eval_qsid):
                demoted_correct_top1 += 1
            if base_preds and base_preds[0] != eval_qsid and final_preds and final_preds[0] == eval_qsid:
                promoted_correct_top1 += 1
            continue

        t0 = time.time()
        raw_ocr_text = get_ocr_text(query_ocr, qpath)
        if not raw_ocr_text:
            missing_ocr += 1
        query_text = clean_query_text(raw_ocr_text, max_chars=args.max_chars)
        query_image = load_image(qpath) if args.include_query_image else None
        if args.include_query_image and query_image is None:
            missing_query_image += 1

        scores_by_sid: Dict[str, float] = {}
        table: List[dict] = []
        reranked = False
        switched_top1 = False
        switch_advantage = 0.0

        if len(query_text) < args.min_chars:
            short_ocr += 1
            final_preds = base_preds[: args.final_topk]
        else:
            cand_with_path = []
            unscored_setids: List[str] = []
            for sid in base_preds[: args.rerank_topk]:
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
                scores = score_candidates_hybrid(
                    model,
                    processor,
                    device,
                    query_text,
                    query_image,
                    images,
                    true_token_id,
                    false_token_id,
                    include_query_image=args.include_query_image,
                )
                scores_by_sid.update(dict(zip(keep_sids, scores)))

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

        hybrid_preds.append(final_preds)
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
                    "include_query_image": args.include_query_image,
                    "reranked": reranked,
                    "switched_top1": switched_top1,
                    "switch_advantage": switch_advantage,
                    "alpha_base": args.alpha_base,
                    "beta_rerank": args.beta_rerank,
                    "switch_margin": args.switch_margin,
                    "preds_colnomic": base_preds,
                    "preds_hybrid": final_preds,
                    "candidate_scores": table,
                    "time_ms": elapsed_ms,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out_f.flush()

    out_f.close()

    metrics_colnomic: Metrics = compute_metrics(gt_list, colnomic_preds)
    metrics_hybrid: Metrics = compute_metrics(gt_list, hybrid_preds)
    payload = {
        "method": "monoqwen_hybrid_query_image_ocr",
        "preds_jsonl": args.preds_jsonl,
        "gallery_dir": args.gallery_dir,
        "query_ocr_jsonl": args.query_ocr_jsonl,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "rerank_topk": args.rerank_topk,
        "final_topk": args.final_topk,
        "min_chars": args.min_chars,
        "include_query_image": args.include_query_image,
        "alpha_base": args.alpha_base,
        "beta_rerank": args.beta_rerank,
        "base_score_mode": args.base_score_mode,
        "switch_margin": args.switch_margin,
        "n_queries": metrics_colnomic.n_queries,
        "missing_ocr": missing_ocr,
        "short_ocr": short_ocr,
        "missing_query_image": missing_query_image,
        "changed_order_count": changed_order_count,
        "switched_top1_count": switched_top1_count,
        "demoted_correct_top1": demoted_correct_top1,
        "promoted_correct_top1": promoted_correct_top1,
        "colnomic_baseline": {
            "recall@1": metrics_colnomic.recall_at_1,
            "recall@5": metrics_colnomic.recall_at_5,
            "recall@10": metrics_colnomic.recall_at_10,
            "mrr": metrics_colnomic.mrr,
        },
        "monoqwen_hybrid": {
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
