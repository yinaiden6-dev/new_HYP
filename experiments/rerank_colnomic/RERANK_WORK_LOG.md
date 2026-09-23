# ColNomic Rerank Work Log

## 2026-06-25 DeepSeek-OCR Query Text Rerank

### Purpose

This run checks whether a MonoQwen2-VL cross-encoder reranker can improve ColNomic-7B top-k retrieval after replacing the previous query-image OCR step with the existing DeepSeek-OCR cache.

The key fix is methodological:

- Old reranker program: `rerank_monoqwen.py`
- Issue: it first asked the Qwen/MonoQwen pipeline to read text from the query image, then used that generated text to rerank gallery candidates.
- New reranker program: `rerank_monoqwen_deepseek_ocr.py`
- Fix: query text is loaded directly from cached DeepSeek-OCR jsonl. The reranker only scores candidate gallery images against this fixed query OCR text.

This keeps the original reranker reproducible while adding a cleaner control group.

### Main Code

- Program: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/rerank_monoqwen_deepseek_ocr.py`
- Outcome/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_monoqwen_deepseek_ocr_horeka.sbatch`
- Difficult/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_monoqwen_deepseek_ocr_difficult_horeka.sbatch`
- Original reproducible program: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/rerank_monoqwen.py`

### Model

- First-stage retriever: ColNomic 7B image-only results.
- Reranker base model: `/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen2-VL-2B-Instruct`
- Reranker adapter: `/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/MonoQwen2-VL-v0.1`
- Rerank scope: rerank only the first-stage top 10 candidates.
- Final output top-k: 10.
- Short OCR fallback: if cleaned OCR text has fewer than 40 characters, keep the original ColNomic order.

### Inputs

Outcome/raw run:

- Query predictions: `/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/result/raw_gallery/run_colnomic_retrieval_detail_raw_gallery_name_preds_image_only_7b.jsonl`
- Query OCR: `/hkfs/work/workspace/scratch/ap7811-benchmark/colqwen/result/+ocr/ocr/query_ocr_deepseek_raw_gallery.jsonl`
- Gallery: `/hkfs/work/workspace/scratch/ap7811-benchmark/dailymed/data/box_flat_20000_images/data/raw_images`
- Query count: 1132
- OCR coverage: missing 0, short 60

Difficult/raw run:

- Query predictions: `/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/difficult/raw_gallery_7b/run_colnomic_retrieval_detail_name_preds.jsonl`
- Query OCR: `/hkfs/work/workspace/scratch/ap7811-benchmark/ocr_cache/query_ocr_deepseek_difficult.jsonl`
- Gallery: `/hkfs/work/workspace/scratch/ap7811-benchmark/dailymed/data/box_flat_20000_images/data/raw_images`
- Query count: 224
- OCR coverage: missing 0, short 0

### Slurm Jobs

| Job ID | Name | Dataset | State | Runtime | Output |
|---|---|---|---|---:|---|
| 4110904 | `mq_dsocr_rerank` | outcome query vs raw gallery | COMPLETED, exit 0 | 00:28:00 | `rerank_metrics_deepseek_ocr_image_only_7b.json` |
| 4110905 | `mq_dsocr_diff` | difficult query vs raw gallery | COMPLETED, exit 0 | 00:07:29 | `rerank_metrics_deepseek_ocr_difficult_raw_gallery_7b.json` |

Logs:

- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/mq_dsocr_rerank-4110904.out`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/mq_dsocr_rerank-4110904.err`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/mq_dsocr_diff-4110905.out`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/mq_dsocr_diff-4110905.err`

### Outputs

Outcome/raw:

- Predictions: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_deepseek_ocr_image_only_7b.jsonl`
- Metrics: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_deepseek_ocr_image_only_7b.json`
- Prediction rows: 1132

Difficult/raw:

- Predictions: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_deepseek_ocr_difficult_raw_gallery_7b.jsonl`
- Metrics: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_deepseek_ocr_difficult_raw_gallery_7b.json`
- Prediction rows: 224

### Metrics

Outcome/raw:

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| ColNomic-7B baseline | 0.7562 | 0.8728 | 0.9011 | 0.8084 |
| Old MonoQwen rerank | 0.6899 | 0.8622 | 0.9011 | 0.7645 |
| MonoQwen + DeepSeek-OCR text | 0.6714 | 0.8472 | 0.9011 | 0.7450 |

Difficult/raw:

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| ColNomic-7B baseline | 0.5625 | 0.7723 | 0.8036 | 0.6585 |
| Old MonoQwen rerank | 0.5268 | 0.7143 | 0.8036 | 0.6118 |
| MonoQwen + DeepSeek-OCR text | 0.3929 | 0.6607 | 0.8036 | 0.5103 |

### Interpretation

The DeepSeek-OCR version fixed the process problem but did not improve retrieval. It made top-1 worse on both query sets:

- Outcome/raw R@1 drops from 0.7562 to 0.6714.
- Difficult/raw R@1 drops from 0.5625 to 0.3929.
- R@10 is unchanged in both runs because the reranker only reorders the existing top-10 candidates; it cannot recover a missing true item outside the original top-10.

This means the weakness is not only caused by bad query OCR generated by the old reranker. Even with fixed DeepSeek-OCR text, MonoQwen2-VL is not a reliable top-1 discriminator for this instance-level medicine image retrieval setting.

Likely causes:

- The reranker is prompted as a binary relevance scorer, but these candidates often share similar medicine names, packaging text, dosage patterns, or product-family wording.
- Query OCR text can overemphasize incidental visible text and underemphasize instance-level visual identity.
- The reranker sees each candidate independently and does not explicitly compare fine-grained visual differences among the top-10 candidates.
- The task is instance-level retrieval, but OCR text is product-level or label-level evidence. It can help identify a product family, yet still reorder exact instances incorrectly.

### Conclusion

For the current benchmark, reranking ColNomic-7B top-10 with MonoQwen2-VL plus cached DeepSeek-OCR is not beneficial. The clean fixed-control experiment strengthens the conclusion that naive OCR/LLM reranking is not enough for this benchmark. A better reranker would need either:

- pairwise/listwise candidate comparison instead of independent binary scoring;
- a trained reranker with product-level hard negatives;
- explicit visual identity features in addition to OCR text;
- or a TECC-style gating model trained on retrieval loss rather than prompt-only reranking.

This experiment should be treated as a negative but useful control result.

## 2026-06-25 Conservative Hybrid MonoQwen Rerank

### Motivation

The previous MonoQwen rerank runs showed high R@10 but lower R@1. Diagnostics showed that the true target was usually still inside the ColNomic top-10, but pointwise MonoQwen scoring often moved the original correct top-1 downward.

The new run tests three corrections:

- Keep ColNomic rank as the primary score instead of letting MonoQwen fully reorder top-10.
- Only allow top-1 replacement when MonoQwen's rerank-score margin over the original ColNomic top-1 is large enough.
- Give the reranker both query image and query OCR, not query OCR alone.

### Main Code

- Program: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/rerank_monoqwen_hybrid.py`
- Outcome/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_monoqwen_hybrid_horeka.sbatch`
- Difficult/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_monoqwen_hybrid_difficult_horeka.sbatch`

### Scoring Rule

For each candidate in the ColNomic top-10:

```text
base_rank_score = linear rank score from 1.0 at rank 1 to 0.0 at rank 10
hybrid_score = alpha_base * base_rank_score + beta_rerank * monoqwen_score
```

Current submitted parameters:

- `alpha_base = 0.80`
- `beta_rerank = 0.20`
- `switch_margin = 0.15`
- `base_score_mode = linear`
- `include_query_image = true`
- `rerank_topk = 10`
- `final_topk = 10`

Top-1 gate:

```text
If hybrid top-1 differs from ColNomic top-1,
only switch when:
    monoqwen_score(new_top1) - monoqwen_score(original_colnomic_top1) >= switch_margin
otherwise keep the original ColNomic top-1.
```

This makes MonoQwen a conservative tie-breaker rather than the final authority.

### Prompt/Input Form

When `--include_query_image` is enabled, each reranker item receives:

- image 1: query image
- image 2: candidate gallery image
- text: cached DeepSeek-OCR query text

The prompt explicitly says to use visual identity first and OCR only as supporting evidence.

### Submitted Jobs

| Job ID | Name | Dataset | State at submission | Output |
|---|---|---|---|---|
| 4111002 | `mq_hybrid` | outcome query vs raw gallery | PENDING | `rerank_metrics_hybrid_queryimg_ocr_image_only_7b.json` |
| 4111003 | `mq_hyb_diff` | difficult query vs raw gallery | PENDING | `rerank_metrics_hybrid_queryimg_ocr_difficult_raw_gallery_7b.json` |

Expected outputs:

- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_hybrid_queryimg_ocr_image_only_7b.jsonl`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_hybrid_queryimg_ocr_image_only_7b.json`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_hybrid_queryimg_ocr_difficult_raw_gallery_7b.jsonl`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_hybrid_queryimg_ocr_difficult_raw_gallery_7b.json`

### Expected Interpretation

This run is not expected to create new R@10 gains because it still only reranks ColNomic top-10. The target metric is R@1 and MRR stability:

- If R@1 recovers close to ColNomic baseline, the previous failure was mainly over-aggressive reranking.
- If R@1 improves beyond ColNomic baseline, the hybrid gate successfully fixes some top-1 mistakes.
- If R@1 still drops, MonoQwen's score is not reliable even as a weak conservative signal for this task.

## 2026-06-25 Qwen3-VL-Reranker-2B Hybrid Rerank

### Motivation

MonoQwen2-VL is a useful visual-document reranker, but the previous results suggest it is not strong enough for instance-level medicine package reranking. Qwen3-VL-Reranker-2B is a newer dedicated multimodal reranker that natively supports mixed-modality query/document pairs.

This run keeps the same conservative hybrid policy as the MonoQwen hybrid run, but replaces the reranker model with Qwen3-VL-Reranker-2B.

### Model And Dependency Setup

- Downloaded model: `/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen3-VL-Reranker-2B`
- Model size on disk: about 4.0 GB
- Official wrapper script: `/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen3-VL-Reranker-2B/scripts/qwen3_vl_reranker.py`
- Installed dependency in `.venv-colpali`: `qwen-vl-utils==0.0.14`

### Main Code

- Program: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/rerank_qwen3vl_hybrid.py`
- Outcome/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_qwen3vl_hybrid_horeka.sbatch`
- Difficult/raw sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_qwen3vl_hybrid_difficult_horeka.sbatch`

### Input Form

For each query and each ColNomic top-10 gallery candidate:

- Query: `query image + query DeepSeek-OCR text`
- Document: `gallery image + gallery DeepSeek-OCR text`
- Instruction: exact same medicine product image retrieval; visual identity first; OCR only supporting evidence.

### Hybrid Scoring

The same conservative gate is used:

```text
base_rank_score = linear rank score from 1.0 at rank 1 to 0.0 at rank 10
hybrid_score = 0.80 * base_rank_score + 0.20 * qwen3vl_reranker_score
```

Top-1 replacement is allowed only when:

```text
qwen3vl_score(new_top1) - qwen3vl_score(original_colnomic_top1) >= 0.15
```

### Submitted Jobs

| Job ID | Name | Dataset | State at submission | Output |
|---|---|---|---|---|
| 4111034 | `q3vl_hybrid` | outcome query vs raw gallery | PENDING | `rerank_metrics_qwen3vl_hybrid_image_only_7b.json` |
| 4111033 | `q3vl_hybdf` | difficult query vs raw gallery | PENDING | `rerank_metrics_qwen3vl_hybrid_difficult_raw_gallery_7b.json` |

Expected outputs:

- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_qwen3vl_hybrid_image_only_7b.jsonl`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_qwen3vl_hybrid_image_only_7b.json`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_qwen3vl_hybrid_difficult_raw_gallery_7b.jsonl`
- `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_metrics_qwen3vl_hybrid_difficult_raw_gallery_7b.json`

### Checks

- Python syntax check passed for `rerank_qwen3vl_hybrid.py`.
- `bash -n` passed for both Qwen3-VL sbatch scripts.
- Local import of the official wrapper was stopped because it was slow on the login node; actual compatibility will be verified when the GPU job starts.

## 2026-06-25 Difficult GT Fix

### Root Cause

The difficult-query CSV uses `image_path,origin_name,file_name`. Some `origin_name` values contain UUID-like strings with required suffixes, for example:

- Query image: `bef4f237-ccb8-4d28-bfc2-41c511a17d7f-01_00000001.jpg`
- Correct gallery label: `bef4f237-ccb8-4d28-bfc2-41c511a17d7f-01`
- Old parsed label: `bef4f237-ccb8-4d28-bfc2-41c511a17d7f`

The old CSV parser applied UUID extraction before the generic `image_path -> origin_name` case, so it stripped suffixes such as `-01` and `-10`. This caused true top-1 matches to be counted as failures.

### Fix

- Updated active CSV parsers in `colpali/run_colpali_retrieval_detail_raw_gallery.py`, `colnomic/run_colnomic_retrieval_detail_raw_gallery.py`, and `colqwen/run_colqwen_retrieval_detail_raw_gallery.py` to preserve full `origin_name` for 3-column benchmark CSVs.
- Updated rerank evaluation code to use `eval_query_setid` derived from `query_path` for `/1/difficult/` rows while preserving original `query_setid` for traceability.
- Added `tools/recompute_difficult_metrics.py` to recompute existing difficult metrics without rerunning GPU retrieval.
- Updated `tools/update_experiment_summary.py` so workbook NDCG also uses corrected difficult ground truth and all rows with `metrics_path` can be refreshed from metrics JSON.

### Recomputed Results

The recompute updated 60 difficult metrics JSON files and added `gt_fix` metadata to each updated file. Seven TECC metrics were skipped because matching prediction JSONL files were not present.

Key corrected examples:

- ColNomic-7B difficult/raw: R@1 `0.5625 -> 0.598214`, R@10 `0.803571 -> 0.848214`.
- Qwen3-VL-Embedding image-only difficult/raw: R@1 `0.5`, R@10 `0.71875`.
- ColPali difficult/raw: R@1 `0.272321`, R@10 `0.446429`.
- ColQwen image-only difficult/raw: R@1 `0.205357`, R@10 `0.303571`.

The workbook `/hkfs/work/workspace/scratch/ap7811-benchmark/experiment_summary.xlsx` was refreshed after this fix. Backup: `/hkfs/work/workspace/scratch/ap7811-benchmark/experiment_summary.before_difficult_gt_fix_20260625_165256.xlsx`.

## 2026-06-25 CSV Ground-Truth Fix V2

### Correction To The Earlier Difficult-Only Fix

The first fix was too broad: it used the query filename stem as ground truth for every `/1/difficult/` sample. That would incorrectly turn normal numbered filenames such as `cefdinir-fig2_001.JPG` into gt `cefdinir-fig2_001`, while the correct gallery label is `cefdinir-fig2`.

The final rule is narrower and applies to all CSV-query experiments, not only difficult:

```text
if old gt is exactly a bare UUID
and query filename starts with UUID + "-" + suffix
then recover gt as UUID-suffix
else keep the existing CSV/preds gt unchanged
```

This fixes the real parser error without using arbitrary query filename stems as labels.

### Scope

The same UUID-suffix truncation can affect CSV-based query sets:

- `processed`
- `processed_hard`
- `degrade`
- `difficult`

`outcome` name-mode is not affected.

### Code Changes

- `tools/recompute_difficult_metrics.py` was upgraded from difficult-only recomputation to a CSV-query metrics recomputation tool while keeping the filename for compatibility.
- `tools/update_experiment_summary.py` now uses the same UUID-suffix recovery rule for NDCG and supports older `..._metrics_processed_...json` -> `..._preds_processed_...jsonl` naming.
- `rerank/colnomc_rerank/rerank_monoqwen.py` now emits future rerank gt using the same UUID-suffix recovery rule; dependent rerank scripts import this helper.

### Verification

- Recomputed 186 metrics JSON files in the second pass.
- Consistency check over current metrics/preds pairs: `checked=294`, `files_with_recoverable_uuid_suffix=172`, `mismatch=0`.
- Refreshed `/hkfs/work/workspace/scratch/ap7811-benchmark/experiment_summary.xlsx`.
- Workbook backups:
  - `/hkfs/work/workspace/scratch/ap7811-benchmark/experiment_summary.before_csv_gt_fix_20260625_171553.xlsx`
  - `/hkfs/work/workspace/scratch/ap7811-benchmark/experiment_summary.before_csv_gt_fix_v2_20260625_172002.xlsx`

Key corrected examples after the final narrow rule:

- ColNomic-7B difficult/raw: R@1 `0.647321`, R@5 `0.875000`, R@10 `0.906250`, MRR `0.750427`, corrected queries `23`.
- ColNomic-7B processed/raw: R@1 `0.938000`, R@5 `0.979000`, R@10 `0.980500`, MRR `0.955271`, corrected queries `72`.
- ColNomic-7B processed_hard/raw: R@1 `0.937500`, R@5 `0.968000`, R@10 `0.971500`, MRR `0.951115`, corrected queries `84`.
- ColNomic-7B degrade/raw: R@1 `0.678078`, R@5 `0.803556`, R@10 `0.832501`, MRR `0.732157`, corrected queries `174`.

## 2026-06-25 Rerank Re-Run With CSV GT V2

### Reason

Existing rerank `preds.jsonl` files had been corrected at the metrics layer, but for clean reproducibility the final rerank runs should emit `eval_query_setid` from the fixed UUID-suffix gt logic directly.

### Submitted Job

- Slurm array job: `4111716_[0-7]`
- Sbatch: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/run_rerank_csv_gt_v2_all_horeka.sbatch`
- Output directory: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result_csv_gt_v2`
- Logs: `/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/rr_csvgt_v2-4111716_*.out|err`
- Time limit: `06:00:00` per array task

### Array Task Map

| Task | Method | Query | Base retrieval |
|---|---|---|---|
| 0 | MonoQwen original | `outcome` | ColNomic-7B raw-gallery top-10 |
| 1 | MonoQwen original | `difficult` | ColNomic-7B raw-gallery top-10 |
| 2 | MonoQwen + DeepSeek query OCR | `outcome` | ColNomic-7B raw-gallery top-10 |
| 3 | MonoQwen + DeepSeek query OCR | `difficult` | ColNomic-7B raw-gallery top-10 |
| 4 | MonoQwen hybrid query image + OCR | `outcome` | ColNomic-7B raw-gallery top-10 |
| 5 | MonoQwen hybrid query image + OCR | `difficult` | ColNomic-7B raw-gallery top-10 |
| 6 | Qwen3-VL-Reranker hybrid | `outcome` | ColNomic-7B raw-gallery top-10 |
| 7 | Qwen3-VL-Reranker hybrid | `difficult` | ColNomic-7B raw-gallery top-10 |

### Submission Checks

- `bash -n run_rerank_csv_gt_v2_all_horeka.sbatch` passed.
- `python3 -m py_compile` passed for all rerank programs.
- Initial Slurm state: `PENDING` for `4111716_[0-7]`.
