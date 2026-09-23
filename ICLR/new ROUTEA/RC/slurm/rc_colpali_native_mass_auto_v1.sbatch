#!/usr/bin/env bash
#SBATCH --job-name=cp_nauto
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cp-nauto-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cp-nauto-%j.err
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
'/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u programs/submit_rc_colpali_native_mass_after_quality_v1.py
