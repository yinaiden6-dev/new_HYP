#!/usr/bin/env bash
#SBATCH --account=hk-project-p0025545
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --requeue
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-expand-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-expand-%A_%a.err
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
ws='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc="$ws/ICLR/new ROUTEA/RC"
export PYTHONPATH="$rc/programs:$rc/src:$ws"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_HOME="$ws/third_party/model_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$rc"
stage="$1"
python="$ws/.venv-colpali/bin/python"
if [[ "$stage" == pilot || "$stage" == collect ]]; then python="$ws/.venv-romav2/bin/python"; fi
set +e
timeout --signal=TERM --kill-after=15s 550s "$python" -u programs/run_rc_m_fine_c128_extension_v1.py "$stage" --ordinal "${SLURM_ARRAY_TASK_ID:-0}"
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 32 ]]; then exit 75; fi
    task="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then task="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
    scontrol requeue "$task"
    exit 0
fi
exit "$code"
