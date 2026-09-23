#!/usr/bin/env bash
#SBATCH --job-name=colpali_q593
#SBATCH --partition=dev_accelerated,accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-q593-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-q593-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
set +e
timeout --signal=TERM --kill-after=10s 860s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u programs/cache_colpali_h593_queries_v1.py extract --budget 780
rc_status=$?
set -e
if [[ "$rc_status" == 75 && "${SLURM_RESTART_COUNT:-0}" -lt 3 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rc_status"
