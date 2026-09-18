#!/usr/bin/env bash
#SBATCH --job-name=eval128_raw_bridge
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_token_raw_bridge-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_token_raw_bridge-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/eval128-raw-bridge-${SLURM_JOB_ID:-manual}"
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
exec timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" programs/materialize_rc_original7_eval128_token_raw_v1.py --phase bridge-existing
