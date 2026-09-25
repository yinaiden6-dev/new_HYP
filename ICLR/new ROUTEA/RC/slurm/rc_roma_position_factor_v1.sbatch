#!/usr/bin/env bash
#SBATCH --job-name=roma_phase
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma-phase-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma-phase-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
export PYTHONPATH="$rc_root/programs:$rc_root/src:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_HOME="$rc_workspace/third_party/model_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
set +e
timeout --signal=TERM --kill-after=15s 550s "$rc_workspace/.venv-romav2/bin/python" -u programs/run_rc_roma_position_factor_v1.py "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 16 ]]; then exit 75; fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$code"
