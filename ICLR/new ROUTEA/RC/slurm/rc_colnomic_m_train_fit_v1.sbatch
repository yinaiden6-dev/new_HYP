#!/usr/bin/env bash
#SBATCH --job-name=col_M_fitdiag
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/col-m-fitdiag-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/col-m-fitdiag-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
timeout --signal=TERM --kill-after=10s 550s "$PY" -u programs/run_rc_colnomic_m_train_fit_v1.py run
"$PY" -u programs/run_rc_colnomic_m_train_fit_v1.py advance
