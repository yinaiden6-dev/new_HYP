#!/usr/bin/env bash
#SBATCH --job-name=routea_pair64
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/routea_pair64-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/routea_pair64-%j.err
#SBATCH --export=NIL

set -euo pipefail

workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$workspace/ICLR/new ROUTEA/RC"
python_bin="$workspace/.venv-romav2/bin/python"
model_cache="$workspace/third_party/model_cache"
pair64_runtime=${SLURM_TMPDIR:-/tmp}/routea-pair64-${SLURM_JOB_ID}

mkdir -p \
  "$pair64_runtime/tmp" \
  "$pair64_runtime/xdg" \
  "$pair64_runtime/hf" \
  "$pair64_runtime/torchinductor" \
  "$pair64_runtime/triton"

export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$rc_root/src:$rc_root/programs:$workspace"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$model_cache/torch"
export HF_HOME="$pair64_runtime/hf"
export XDG_CACHE_HOME="$pair64_runtime/xdg"
export TMPDIR="$pair64_runtime/tmp"
export TORCHINDUCTOR_CACHE_DIR="$pair64_runtime/torchinductor"
export TRITON_CACHE_DIR="$pair64_runtime/triton"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$rc_root"
echo "job=$SLURM_JOB_ID host=$(hostname) stage=MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$python_bin" programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py
"$python_bin" programs/validate_routea_matched_three_arm_pair64_training_features_v1.py
