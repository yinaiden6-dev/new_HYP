#!/bin/bash
#SBATCH -J rr_csvgt_v2
#SBATCH -A hk-project-p0025545
#SBATCH -p accelerated
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64000
#SBATCH -t 06:00:00
#SBATCH -o /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%A_%a.out
#SBATCH -e /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%A_%a.err

set -euo pipefail

export SCR=/hkfs/work/workspace/scratch/ap7811-benchmark
export RERANK_DIR="$SCR/rerank/colnomc_rerank"
export RESULT_DIR="$RERANK_DIR/result_csv_gt_v2"
export LOG_DIR="$RERANK_DIR/logs"
export PYTHON_BIN="$SCR/.venv-colpali/bin/python"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SCR:$RERANK_DIR:${PYTHONPATH:-}"

cd "$RERANK_DIR"

OUTCOME_PREDS="$SCR/colnomic/result/raw_gallery/run_colnomic_retrieval_detail_raw_gallery_name_preds_image_only_7b.jsonl"
DIFFICULT_PREDS="$SCR/colnomic/difficult/raw_gallery_7b/run_colnomic_retrieval_detail_name_preds.jsonl"
RAW_GALLERY="$SCR/dailymed/data/box_flat_20000_images/data/raw_images"
OUTCOME_OCR="$SCR/colqwen/result/+ocr/ocr/query_ocr_deepseek_raw_gallery.jsonl"
DIFFICULT_OCR="$SCR/ocr_cache/query_ocr_deepseek_difficult.jsonl"
GALLERY_OCR="$SCR/ocr_cache/gallery_ocr_deepseek_raw_gallery.jsonl"
QWEN2_BASE="$SCR/models/downloaded_models/Qwen2-VL-2B-Instruct"
MONOQWEN_ADAPTER="$SCR/models/downloaded_models/MonoQwen2-VL-v0.1"
QWEN3_RERANKER="$SCR/models/downloaded_models/Qwen3-VL-Reranker-2B"

echo "job=$SLURM_JOB_ID array_task=${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"
echo "workdir=$RERANK_DIR"
echo "result_dir=$RESULT_DIR"
"$PYTHON_BIN" --version
nvidia-smi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_NAME="monoqwen_outcome"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen.py" \
      --preds_jsonl "$OUTCOME_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_image_only_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_image_only_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 5 \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  1)
    RUN_NAME="monoqwen_difficult"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen.py" \
      --preds_jsonl "$DIFFICULT_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_difficult_raw_gallery_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_difficult_raw_gallery_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 5 \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  2)
    RUN_NAME="monoqwen_deepseek_outcome"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen_deepseek_ocr.py" \
      --preds_jsonl "$OUTCOME_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$OUTCOME_OCR" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_deepseek_ocr_image_only_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_deepseek_ocr_image_only_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 5 \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  3)
    RUN_NAME="monoqwen_deepseek_difficult"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen_deepseek_ocr.py" \
      --preds_jsonl "$DIFFICULT_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$DIFFICULT_OCR" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_deepseek_ocr_difficult_raw_gallery_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_deepseek_ocr_difficult_raw_gallery_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 5 \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  4)
    RUN_NAME="monoqwen_hybrid_outcome"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen_hybrid.py" \
      --preds_jsonl "$OUTCOME_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$OUTCOME_OCR" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_hybrid_queryimg_ocr_image_only_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_hybrid_queryimg_ocr_image_only_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 2 \
      --alpha_base 0.80 \
      --beta_rerank 0.20 \
      --switch_margin 0.15 \
      --include_query_image \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  5)
    RUN_NAME="monoqwen_hybrid_difficult"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen_hybrid.py" \
      --preds_jsonl "$DIFFICULT_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$DIFFICULT_OCR" \
      --base_model "$QWEN2_BASE" \
      --adapter "$MONOQWEN_ADAPTER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_hybrid_queryimg_ocr_difficult_raw_gallery_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_hybrid_queryimg_ocr_difficult_raw_gallery_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --candidate_batch_size 2 \
      --alpha_base 0.80 \
      --beta_rerank 0.20 \
      --switch_margin 0.15 \
      --include_query_image \
      --min_chars 40 \
      --device cuda \
      --dtype bfloat16 \
      --skip_done
    ;;
  6)
    RUN_NAME="qwen3vl_hybrid_outcome"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_qwen3vl_hybrid.py" \
      --preds_jsonl "$OUTCOME_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$OUTCOME_OCR" \
      --gallery_ocr_jsonl "$GALLERY_OCR" \
      --model_dir "$QWEN3_RERANKER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_qwen3vl_hybrid_image_only_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_qwen3vl_hybrid_image_only_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --alpha_base 0.80 \
      --beta_rerank 0.20 \
      --switch_margin 0.15 \
      --torch_dtype bfloat16 \
      --skip_done
    ;;
  7)
    RUN_NAME="qwen3vl_hybrid_difficult"
    srun "$PYTHON_BIN" "$RERANK_DIR/rerank_qwen3vl_hybrid.py" \
      --preds_jsonl "$DIFFICULT_PREDS" \
      --gallery_dir "$RAW_GALLERY" \
      --query_ocr_jsonl "$DIFFICULT_OCR" \
      --gallery_ocr_jsonl "$GALLERY_OCR" \
      --model_dir "$QWEN3_RERANKER" \
      --out_jsonl "$RESULT_DIR/rerank_preds_qwen3vl_hybrid_difficult_raw_gallery_7b.jsonl" \
      --out_metrics "$RESULT_DIR/rerank_metrics_qwen3vl_hybrid_difficult_raw_gallery_7b.json" \
      --rerank_topk 10 \
      --final_topk 10 \
      --alpha_base 0.80 \
      --beta_rerank 0.20 \
      --switch_margin 0.15 \
      --torch_dtype bfloat16 \
      --skip_done
    ;;
  *)
    echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "completed run_name=$RUN_NAME"
