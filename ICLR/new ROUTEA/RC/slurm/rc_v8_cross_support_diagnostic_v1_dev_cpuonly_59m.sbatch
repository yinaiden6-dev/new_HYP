#!/usr/bin/env bash
#SBATCH --job-name=v8_cross_support
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_v8_cross_support-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_v8_cross_support-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/rc-v8-cross-support-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
"$rc_python" programs/run_rc_v8_cross_support_diagnostic_v1.py --phase run
exec "$rc_python" programs/run_rc_v8_cross_support_diagnostic_v1.py --phase validate
