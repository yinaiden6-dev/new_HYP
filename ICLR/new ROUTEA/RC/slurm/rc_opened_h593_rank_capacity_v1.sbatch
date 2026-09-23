#!/usr/bin/env bash
#SBATCH --job-name=h593_rankcap
#SBATCH --partition=dev_accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-rankcap-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-rankcap-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/h593-rankcap-${SLURM_JOB_ID:?}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp"
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 550s "$rc_python" programs/run_rc_opened_h593_rank_capacity_v1.py audit
