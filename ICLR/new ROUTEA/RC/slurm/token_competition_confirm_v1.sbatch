#!/usr/bin/env bash
#SBATCH --job-name=token128c
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/token128c-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/token128c-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
token_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
token_stage="${1:?stage required}"
token_command=("$token_python" -u 'ICLR/new ROUTEA/RC/programs/run_token_competition_confirm_stage_v1.py' "$token_stage" --threads 8 --wall-seconds 480)
case "$token_stage" in
    train)
        token_index="${2:-${SLURM_ARRAY_TASK_ID:-}}"
        test -n "$token_index"
        token_command+=(--index "$token_index")
        ;;
    join) ;;
    *) echo "Unknown token128 confirmation stage: $token_stage" >&2; exit 2 ;;
esac
set +e
timeout --signal=TERM --kill-after=10s 550s "${token_command[@]}"
token_status=$?
set -e
if [[ "$token_status" == 75 || "$token_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 96 ]]; then exit 124; fi
    token_job="$SLURM_JOB_ID"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then token_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
    scontrol requeue "$token_job"
    exit 0
fi
exit "$token_status"
