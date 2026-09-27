#!/usr/bin/env bash
#SBATCH --job-name=M_restore_probe3
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-restore-probe3-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-restore-probe3-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
restore_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
timeout --signal=TERM --kill-after=10s 540s "$restore_python" -u 'ICLR/new ROUTEA/RC/programs/probe_rc_roma_property_restore_cpu_v3.py' --index 0 --position 0 --threads 4
