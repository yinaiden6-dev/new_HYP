#!/bin/bash
#SBATCH --job-name=n2_d1fold
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --array=1-4%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=120000
#SBATCH --time=02:00:00
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
runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-d1-fold-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$runtime/home" "$runtime/tmp"
export HOME="$runtime/home" TMPDIR="$runtime/tmp" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src:$workspace/ICLR/new ROUTEA/route_a_core"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
"$python" "$root/programs/run_routea_matched_three_arm_n2_current_runtime_d1_fold_v1.py" --fold "$SLURM_ARRAY_TASK_ID"
exec "$python" "$root/programs/validate_routea_matched_three_arm_n2_current_runtime_d1_fold_v1.py" --fold "$SLURM_ARRAY_TASK_ID"
