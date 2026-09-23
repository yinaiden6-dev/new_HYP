#!/usr/bin/env bash
#SBATCH --job-name=cp_nmass_cpu
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cp-nmass-cpu-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cp-nmass-cpu-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
set +e
timeout --signal=TERM --kill-after=10s 260s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u \
    programs/run_rc_colpali_native_mass_cpu_execution_v1.py "${1:-score}" --budget 180 --index "${SLURM_ARRAY_TASK_ID:-0}"
rc_status=$?
set -e
if [[ "$rc_status" == 75 || "$rc_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 16 ]]; then exit 75; fi
    rc_element="$SLURM_JOB_ID"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then rc_element="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
    scontrol requeue "$rc_element"
    exit 0
fi
exit "$rc_status"
