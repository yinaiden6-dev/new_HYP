#!/usr/bin/env bash
#SBATCH --job-name=crisp_hand
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/crisp-hand-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/crisp-hand-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/crisp-hand-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-join}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
case "${1:?stage}" in
 worker) exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_crisp_manual_baseline_v1.py worker --shard "${SLURM_ARRAY_TASK_ID:?}" ;;
 join) export CUDA_VISIBLE_DEVICES=''; exec timeout --signal=TERM --kill-after=10s 540s "$rc_python" programs/run_rc_crisp_manual_baseline_v1.py join ;;
 *) exit 2 ;;
esac
