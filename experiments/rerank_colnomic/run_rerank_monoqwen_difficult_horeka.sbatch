#!/bin/bash
#SBATCH -J monoqwen_rerank_diff
#SBATCH -A hk-project-p0025545
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64000
#SBATCH -t 06:00:00
#SBATCH -o /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%j.out
#SBATCH -e /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%j.err

set -euo pipefail

export SCR=/hkfs/work/workspace/scratch/ap7811-benchmark
export RERANK_DIR="$SCR/rerank/colnomc_rerank"
export RESULT_DIR="$RERANK_DIR/result"
export LOG_DIR="$RERANK_DIR/logs"
export PYTHON_BIN="$SCR/.venv-colpali/bin/python"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SCR:${PYTHONPATH:-}"

cd "$RERANK_DIR"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "workdir=$RERANK_DIR"
"$PYTHON_BIN" --version
nvidia-smi

srun "$PYTHON_BIN" "$RERANK_DIR/rerank_monoqwen.py" \
  --preds_jsonl "$SCR/colnomic/difficult/raw_gallery_7b/run_colnomic_retrieval_detail_name_preds.jsonl" \
  --gallery_dir "$SCR/dailymed/data/box_flat_20000_images/data/raw_images" \
  --base_model "$SCR/models/downloaded_models/Qwen2-VL-2B-Instruct" \
  --adapter "$SCR/models/downloaded_models/MonoQwen2-VL-v0.1" \
  --out_jsonl "$RESULT_DIR/rerank_preds_difficult_raw_gallery_7b.jsonl" \
  --out_metrics "$RESULT_DIR/rerank_metrics_difficult_raw_gallery_7b.json" \
  --rerank_topk 10 \
  --final_topk 10 \
  --candidate_batch_size 5 \
  --min_chars 40 \
  --device cuda \
  --dtype bfloat16 \
  --skip_done
