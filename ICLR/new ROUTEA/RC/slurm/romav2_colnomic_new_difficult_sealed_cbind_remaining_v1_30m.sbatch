#!/usr/bin/env bash
#SBATCH --job-name=roma_ndcr
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --array=1-7%8
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_ndcr-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_ndcr-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-romav2/bin/python";cache="$workspace/third_party/model_cache";runtime=${SLURM_TMPDIR:-/tmp}/roma-ndcr-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID};mkdir -p "$runtime/home" "$runtime/xdg" "$runtime/tmp" "$runtime/torchinductor" "$runtime/triton";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 TORCH_HOME="$cache/torch" XDG_CACHE_HOME="$runtime/xdg" TMPDIR="$runtime/tmp" HOME="$runtime/home" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" TRITON_CACHE_DIR="$runtime/triton" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;cd "$root";echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) stage=NEW_DIFFICULT_SEALED_CBIND";nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;exec "$py" programs/materialize_romav2_colnomic_new_difficult_sealed_cbind_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
