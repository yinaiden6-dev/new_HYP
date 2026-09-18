#!/usr/bin/env bash
#SBATCH --job-name=t128_readout
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:59:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_train128_readout-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_train128_readout-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/train128-readout-prepare-${SLURM_JOB_ID:-preflight}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
rc_program=programs/prepare_and_run_rc_original7_train128_readout_v1.py
[[ "$(sha256sum "$rc_program" | cut -d ' ' -f 1)" == 'f80c84f4284444d843d27cef886a41f07e5fa4c15254461c82e1bedd31f09af1' ]]
rc_seconds=$("$rc_python" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
(( rc_seconds > 0 ))
if [[ "${1:-run}" == preflight ]]; then
 exec timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" "$rc_program" --source-preflight
fi
[[ -n "${SLURM_JOB_ID:-}" ]]
exec timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" "$rc_program"
