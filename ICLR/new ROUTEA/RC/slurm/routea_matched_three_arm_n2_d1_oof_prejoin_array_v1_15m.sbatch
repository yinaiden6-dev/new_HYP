#!/bin/bash
#SBATCH --job-name=n2_d1oof
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --array=0-15%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32000
#SBATCH --time=00:15:00
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
python="$workspace/.venv-colpali/bin/python"
test -x "$python"
test -n "${SLURM_ARRAY_TASK_ID:-}"
test "${SLURM_SUBMIT_DIR:-}" = "$root"
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-d1-oof-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$runtime/home" "$runtime/tmp"
export HOME="$runtime/home" TMPDIR="$runtime/tmp" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src:$workspace/ICLR/new ROUTEA/route_a_core"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
"$python" "$root/programs/materialize_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py" --shard "$SLURM_ARRAY_TASK_ID"
exec "$python" "$root/programs/validate_routea_matched_three_arm_n2_d1_oof_prejoin_shard_v1.py" --shard "$SLURM_ARRAY_TASK_ID"
