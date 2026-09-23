#!/usr/bin/env bash
#SBATCH --job-name=h593_mictl
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-mictl-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-mictl-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
exec "$rc_workspace/.venv-romav2/bin/python" programs/run_rc_h593_m_inside_v1.py "${1:-control}"
