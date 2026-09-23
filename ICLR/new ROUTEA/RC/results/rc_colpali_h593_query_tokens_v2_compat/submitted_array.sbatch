#!/usr/bin/env bash
#SBATCH --job-name=colpali_qfix
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --array=0-49%50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-qfix-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-qfix-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_element="${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
rc_requeue() {
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then
        echo "REQUEUE_LIMIT_REACHED $rc_element" >&2
        exit 75
    fi
    echo "CHECKPOINT_REQUEUE $rc_element"
    if ! scontrol requeue "$rc_element"; then
        echo "REQUEUE_REQUEST_FAILED $rc_element" >&2
        exit 76
    fi
    exit 0
}
trap rc_requeue USR1
set +e
timeout --signal=TERM --kill-after=10s 840s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u programs/cache_colpali_h593_queries_v2_compat.py extract --budget 750 --shard "$SLURM_ARRAY_TASK_ID" --shards 50 &
rc_child=$!
wait "$rc_child"
rc_status=$?
set -e
if [[ "$rc_status" == 75 || "$rc_status" == 124 ]]; then
    rc_requeue
fi
exit "$rc_status"
