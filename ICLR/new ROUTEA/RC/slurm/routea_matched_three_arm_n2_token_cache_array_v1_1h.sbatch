#!/usr/bin/env bash
#SBATCH --job-name=n2_tok987
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_tok987-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_tok987-%A_%a.err
#SBATCH --export=NIL

set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-colpali/bin/python"
shard=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-token-${SLURM_JOB_ID}-${shard}
mkdir -p "$runtime/home" "$runtime/hf" "$runtime/torch" "$runtime/xdg" "$runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$runtime/hf" TORCH_HOME="$runtime/torch" XDG_CACHE_HOME="$runtime/xdg"
export TMPDIR="$runtime/tmp" HOME="$runtime/home"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

cd "$root"
echo "job=$SLURM_JOB_ID array_job=$SLURM_ARRAY_JOB_ID shard=$shard host=$(hostname) stage=N2_TOKEN_CACHE_987"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$shard" == 0 ]]; then
  "$py" -m pytest -q tests/test_n2_current_runtime_token_cache_v1.py
fi
"$py" programs/materialize_routea_matched_three_arm_n2_token_cache_shard_v1.py --shard "$shard"
"$py" programs/validate_routea_matched_three_arm_n2_token_cache_shard_v1.py --shard "$shard"

