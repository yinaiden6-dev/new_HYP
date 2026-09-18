#!/usr/bin/env bash
#SBATCH --job-name=maxmin_readout
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_reference_support_maxmin_readout_v1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_reference_support_maxmin_readout_v1-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/maxmin-readout-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
for rc_phase in run validate; do
 rc_seconds=$("$rc_python" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
 if (( rc_seconds <= 0 )); then
  echo 'USER_RESEARCH_DEADLINE_REACHED'
  exit 1
 fi
 timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" programs/run_rc_reference_support_maxmin_readout_v1.py --phase "$rc_phase"
done
