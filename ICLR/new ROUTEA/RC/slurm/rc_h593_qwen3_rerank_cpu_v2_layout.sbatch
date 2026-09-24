#!/usr/bin/env bash
#SBATCH --job-name=h593_q3fcal
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-q3fcal-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-q3fcal-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_stage="${1:-fit}"
rc_options=()
if [[ "$rc_stage" == fit ]]; then rc_options+=(--fold "${SLURM_ARRAY_TASK_ID:?}"); fi
set +e
timeout --signal=TERM --kill-after=10s 565s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u \
    programs/run_rc_h593_qwen3_rerank_v2_layout.py "$rc_stage" --budget 450 "${rc_options[@]}"
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
