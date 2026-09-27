#!/usr/bin/env bash
#SBATCH --job-name=rebut_QR2
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-QR2-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-QR2-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
rebut_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rebut_fold="${1:-${SLURM_ARRAY_TASK_ID:-}}"
test -n "$rebut_fold"
set +e
timeout --signal=TERM --kill-after=10s 550s "$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/run_rebut_qr_qrr_frozenbase_fold_v2.py' --fold "$rebut_fold" --wall-seconds 480 --workers 4 --threads 2
rebut_status=$?
set -e
if [[ "$rebut_status" == 75 || "$rebut_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 16 ]]; then exit 124; fi
    rebut_job="$SLURM_JOB_ID"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then rebut_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
    scontrol requeue "$rebut_job"
    exit 0
fi
exit "$rebut_status"
