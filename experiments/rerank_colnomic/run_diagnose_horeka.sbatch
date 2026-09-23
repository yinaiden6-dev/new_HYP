#!/bin/bash
#SBATCH -J monoqwen_diag
#SBATCH -A hk-project-p0025545
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64000
#SBATCH -t 00:30:00
#SBATCH -o /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%j.out
#SBATCH -e /hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/logs/%x-%j.err

set -euo pipefail

export SCR=/hkfs/work/workspace/scratch/ap7811-benchmark
export RERANK_DIR="$SCR/rerank/colnomc_rerank"
export PYTHON_BIN="$SCR/.venv-colpali/bin/python"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SCR:${PYTHONPATH:-}"

cd "$RERANK_DIR"

echo "job=$SLURM_JOB_ID host=$(hostname)"
"$PYTHON_BIN" --version

srun "$PYTHON_BIN" "$RERANK_DIR/diagnose_failures.py" --n_worse 8 --n_better 4
