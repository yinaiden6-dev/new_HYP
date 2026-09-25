#!/usr/bin/env bash
#SBATCH --job-name=h593_q3rr
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --array=0-49%50
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-q3rr-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-q3rr-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_options=()
if [[ "${1:-}" == pilot ]]; then rc_options+=(--pilot); fi
set +e
timeout --signal=TERM --kill-after=25s 835s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u \
    programs/run_rc_h593_qwen3_rerank_v1.py score --budget 735 \
    --shard "$SLURM_ARRAY_TASK_ID" "${rc_options[@]}"
rc_status=$?
set -e
if [[ "$rc_status" == 75 || "$rc_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 16 ]]; then exit 75; fi
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$rc_status"
