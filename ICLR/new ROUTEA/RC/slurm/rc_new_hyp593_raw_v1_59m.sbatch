#!/usr/bin/env bash
#SBATCH --job-name=hyp593_raw
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --array=0-42%16
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/hyp593_raw-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/hyp593_raw-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/hyp593-raw-${SLURM_ARRAY_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-none}"
mkdir -p "$rc_runtime/tmp" "$rc_runtime/xdg"
export TMPDIR="$rc_runtime/tmp" XDG_CACHE_HOME="$rc_runtime/xdg" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
rc_seconds=$("$rc_python" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
if (( rc_seconds <= 0 )); then
 echo 'USER_RESEARCH_DEADLINE_REACHED'
 exit 1
fi
exec timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" programs/materialize_rc_new_hyp593_inputs_v1.py raw --shard "${SLURM_ARRAY_TASK_ID:?array task required}"
