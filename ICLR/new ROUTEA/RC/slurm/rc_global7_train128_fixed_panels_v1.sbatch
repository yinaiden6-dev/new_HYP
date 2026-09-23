#!/usr/bin/env bash
#SBATCH --job-name=global7_t128
#SBATCH --partition=dev_accelerated,accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/global7-t128-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/global7-t128-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/global7-t128-${SLURM_JOB_ID:?}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
timeout --signal=TERM --kill-after=10s 420s "$rc_python" programs/run_rc_global7_train128_fixed_panels_v1.py fit
"$rc_python" programs/run_rc_global7_train128_fixed_panels_v1.py predict
exec "$rc_python" programs/validate_rc_global7_train128_fixed_panels_v1.py
