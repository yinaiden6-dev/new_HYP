#!/usr/bin/env bash
#SBATCH --job-name=cr8_develop
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_coherent_residual_v8-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_coherent_residual_v8-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_runtime="${SLURM_TMPDIR:-/tmp}/rc-coherent-residual-${SLURM_JOB_ID:-manual}"
rc_mode="${1:?mode required}"
rc_lane="${2:-fresh}"
case "$rc_mode" in prepare|train|reduce|validate) ;; *) exit 64 ;; esac
case "$rc_lane" in fresh|resume) ;; *) exit 64 ;; esac
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$rc_root"
"$rc_python" programs/run_rc_coherent_residual_development_v8.py --mode sources
exec "$rc_python" programs/run_rc_coherent_residual_development_v8.py --mode "$rc_mode" --lane "$rc_lane" --max-seconds 3200
