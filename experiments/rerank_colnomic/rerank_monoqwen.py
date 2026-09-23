# Rerank ColNomic Top-K candidates with the MonoQwen2-VL-v0.1 pointwise reranker.
#
# MonoQwen2-VL was trained on text-query -> document-image relevance (ViDoRe / MonoT5
# objective), not image-query -> image-candidate. Since our queries are photos, we read
# the text off the query image with the reranker's own base model (Qwen2-VL-2B, LoRA
# disabled so it behaves as a plain VLM) and use that text as the reranker's text query.
# DeepSeek-OCR was tried first but its remote modeling code is incompatible with the
# transformers version installed in this venv (missing several LlamaConfig-era fields).

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
QUERY_TRAILING_INDEX_RE = re.compile(r"^(.*)_\d{6,}$")
UUID_FULL_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
UUID_SUFFIX_LABEL_RE = re.compile(
    r"^("
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"-[^_]+)",
    re.I,
)

DEFAULT_PREDS_JSONL = "/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/result/raw_gallery/run_colnomic_retrieval_detail_raw_gallery_name_preds_image_only_7b.jsonl"
DEFAULT_GALLERY_DIR = "/hkfs/work/workspace/scratch/ap7811-benchmark/dailymed/data/box_flat_20000_images/data/raw_images"
DEFAULT_BASE_MODEL = "/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen2-VL-2B-Instruct"
DEFAULT_ADAPTER = "/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/MonoQwen2-VL-v0.1"
DEFAULT_OUT_DIR = "/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result"

RERANK_PROMPT = (
    "Assert the relevance of the previous image document to the following query, "
    "answer True or False. The query is: {query}"
)

READ_TEXT_PROMPT = "Read all the text visible in this image. Output only the raw text you see, with no extra commentary."


@dataclass
class Metrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int


def compute_metrics(query_gt: List[str], topk_pred_setids: List[List[str]]) -> Metrics:
    n = len(query_gt)
    hit1 = hit5 = hit10 = 0
    rr_sum = 0.0
    for gt, preds in zip(query_gt, topk_pred_setids):
        if len(preds) >= 1 and gt == preds[0]:
            hit1 += 1
        if gt in preds[:5]:
            hit5 += 1
        if gt in preds[:10]:
            hit10 += 1
        rr = 0.0
        for rank, pred in enumerate(preds, start=1):
            if pred == gt:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return Metrics(
        recall_at_1=hit1 / n if n else 0.0,
        recall_at_5=hit5 / n if n else 0.0,
        recall_at_10=hit10 / n if n else 0.0,
        mrr=rr_sum / n if n else 0.0,
        n_queries=n,
    )


def query_label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].strip()
    m = QUERY_TRAILING_INDEX_RE.match(stem)
    if m:
        stem = m.group(1)
    return stem.rstrip("_").strip()


def eval_gt_from_item(item: Dict) -> str:
    qsid = str(item.get("query_setid") or "")
    qpath = str(item.get("query_path") or "")
    if qpath and UUID_FULL_RE.fullmatch(qsid):
        stem = os.path.splitext(os.path.basename(qpath))[0].strip()
        match = UUID_SUFFIX_LABEL_RE.match(stem)
        if match and match.group(1).startswith(qsid + "-"):
            return match.group(1)
        path_label = query_label_from_path(qpath)
        if path_label.startswith(qsid + "-"):
            return path_label
    return qsid


def clean_query_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) > 900:
        text = text[:450] + " ... " + text[-450:]
    return text[:max_chars]


def build_gallery_setid_to_path(gallery_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(gallery_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(dirpath, name)
            label = os.path.splitext(name)[0].strip()
            out.setdefault(label, os.path.abspath(path))
    return out


def load_preds_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    items.sort(key=lambda x: x.get("query_index", 0))
    return items


def load_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_reranker(
    base_model_dir: str,
    adapter_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    min_pixels: int,
    max_pixels: int,
):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        base_model_dir, local_files_only=True, min_pixels=min_pixels, max_pixels=max_pixels
    )
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_dir, torch_dtype=torch_dtype, local_files_only=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.to(device).eval()
    return model, processor


def vlm_read_text(model, processor, device: torch.device, image: Image.Image, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": READ_TEXT_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad(), model.disable_adapter():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()


def score_candidates(
    model,
    processor,
    device: torch.device,
    query_text: str,
    candidate_images: List[Image.Image],
    true_token_id: int,
    false_token_id: int,
) -> List[float]:
    prompt = RERANK_PROMPT.format(query=query_text)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=DEFAULT_PREDS_JSONL)
    ap.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)

    ap.add_argument("--out_jsonl", default=os.path.join(DEFAULT_OUT_DIR, "rerank_preds_image_only_7b.jsonl"))
    ap.add_argument("--out_metrics", default=os.path.join(DEFAULT_OUT_DIR, "rerank_metrics_image_only_7b.json"))

    ap.add_argument("--rerank_topk", type=int, default=10, help="how many colnomic candidates to rerank per query")
    ap.add_argument("--final_topk", type=int, default=10)
    ap.add_argument("--candidate_batch_size", type=int, default=5)
    ap.add_argument("--min_chars", type=int, default=40, help="min extracted-text chars to attempt rerank, else keep colnomic order")
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--max_new_tokens", type=int, default=256, help="max tokens when reading text off the query image")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28, help="Qwen2-VL image processor min_pixels (caps vision token count, avoids OOM on high-res photos)")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28, help="Qwen2-VL image processor max_pixels (default is ~16k tokens/image and causes CUDA OOM)")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=0, help="only process first N queries (0 = all)")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    device = torch.device(args.device)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    items = load_preds_jsonl(args.preds_jsonl)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[INFO] loaded {len(items)} query preds from {args.preds_jsonl}")

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
        query_image = load_image(qpath)
        if query_image is None:
            query_text = ""
        else:
            query_text = clean_query_text(
                vlm_read_text(model, processor, device, query_image, args.max_new_tokens),
                max_chars=args.max_chars,
            )

        if len(query_text) < args.min_chars:
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
                p = gallery_map.get(sid)
                if p is None:
                    unscored_setids.append(sid)
                else:
                    cand_with_path.append((sid, p))

            for start in range(0, len(cand_with_path), args.candidate_batch_size):
                chunk = cand_with_path[start:start + args.candidate_batch_size]
                images = [load_image(p) for _, p in chunk]
                keep = [(sid, img) for (sid, _), img in zip(chunk, images) if img is not None]
                if not keep:
                    continue
                keep_sids, keep_imgs = zip(*keep)
                scores_chunk = score_candidates(
                    model, processor, device, query_text, list(keep_imgs), true_token_id, false_token_id
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

    metrics_colnomic = compute_metrics(gt_list, colnomic_preds)
    metrics_rerank = compute_metrics(gt_list, rerank_preds)

    payload = {
        "preds_jsonl": args.preds_jsonl,
        "gallery_dir": args.gallery_dir,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "rerank_topk": args.rerank_topk,
        "final_topk": args.final_topk,
        "min_chars": args.min_chars,
        "n_queries": metrics_colnomic.n_queries,
        "colnomic_baseline": {
            "recall@1": metrics_colnomic.recall_at_1,
            "recall@5": metrics_colnomic.recall_at_5,
            "recall@10": metrics_colnomic.recall_at_10,
            "mrr": metrics_colnomic.mrr,
        },
        "monoqwen_rerank": {
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
