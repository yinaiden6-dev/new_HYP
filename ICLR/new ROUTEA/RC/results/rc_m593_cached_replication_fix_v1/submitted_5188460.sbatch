#!/usr/bin/env bash
#SBATCH --job-name=M593_fix
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M593-fix-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M593-fix-%j.err
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
ws='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc="$ws/ICLR/new ROUTEA/RC"
export PYTHONPATH="$rc/programs:$rc/src:$ws"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd "$rc"
exec "$ws/.venv-colpali/bin/python" -u programs/run_rc_m593_cached_replication_fix_v1.py analyze
