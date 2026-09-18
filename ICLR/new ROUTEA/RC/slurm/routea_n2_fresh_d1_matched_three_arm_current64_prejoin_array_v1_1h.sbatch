#!/usr/bin/env bash
#SBATCH --job-name=n2_c64pre
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64pre-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_c64pre-%A_%a.err
#SBATCH --export=NIL

set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-romav2/bin/python"
cache="$workspace/third_party/model_cache"
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-current64-prejoin-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$runtime/home" "$runtime/hf" "$runtime/xdg" "$runtime/tmp" "$runtime/torchinductor" "$runtime/triton"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$runtime/hf" TORCH_HOME="$cache/torch" XDG_CACHE_HOME="$runtime/xdg"
export TMPDIR="$runtime/tmp" HOME="$runtime/home"
export TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" TRITON_CACHE_DIR="$runtime/triton"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

cd "$root"
export ROUTEA_CURRENT64_AUTHORITY_STAGE=prejoin_array
"$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_staging_execution_authority_v1.py
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) stage=N2_FRESH_D1_CURRENT64_PREJOIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shard_dir=$(printf 'results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/shard%02d' "$SLURM_ARRAY_TASK_ID")
if [[ ! -f "$shard_dir/payload.pt" || ! -f "$shard_dir/receipt.json" ]]; then
  "$py" programs/materialize_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
fi
if [[ ! -f "$shard_dir/validation.json" ]]; then
  "$py" programs/validate_routea_n2_fresh_d1_matched_three_arm_prejoin_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
fi
