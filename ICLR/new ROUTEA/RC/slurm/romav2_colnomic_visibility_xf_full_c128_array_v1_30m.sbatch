#!/usr/bin/env bash
#SBATCH --job-name=roma_c128
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_c128-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_c128-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-romav2/bin/python";cache="$workspace/third_party/model_cache";tmp=${SLURM_TMPDIR:-/tmp}/roma-c128-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID};mkdir -p "$tmp" "$tmp/torchinductor" "$tmp/triton" "$cache/hf";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_HOME="$cache/torch" HF_HOME="$cache/hf" XDG_CACHE_HOME="$cache" TORCHINDUCTOR_CACHE_DIR="$tmp/torchinductor" TRITON_CACHE_DIR="$tmp/triton" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HOME="$tmp";cd "$root";"$py" programs/materialize_romav2_colnomic_visibility_xf_full_c128_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"

