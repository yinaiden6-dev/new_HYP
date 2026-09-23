#!/usr/bin/env bash
#SBATCH --job-name=query_prior_head
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_shared_query_prior_head_refit_v1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_shared_query_prior_head_refit_v1-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/rc-query-prior-head-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
"$rc_python" programs/run_rc_shared_query_prior_head_refit_v1.py --phase run
exec "$rc_python" programs/run_rc_shared_query_prior_head_refit_v1.py --phase validate
