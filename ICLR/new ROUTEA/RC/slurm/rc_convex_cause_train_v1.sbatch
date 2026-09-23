#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --no-requeue
#SBATCH --job-name=cause_train
#SBATCH --array=0-1%2
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cause_train-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cause_train-%A_%a.err
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/convex-cause-${SLURM_JOB_ID:?}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
if [[ "${SLURM_ARRAY_TASK_ID:?}" == 0 ]]; then
 timeout --signal=TERM --kill-after=10s 550s "$rc_python" programs/run_rc_convex_cause_isolation_v1.py train
 "$rc_python" programs/run_rc_convex_cause_isolation_v1.py predict
else
 timeout --signal=TERM --kill-after=10s 550s "$rc_python" programs/run_rc_unit_cost96_cause_v1.py fit
 "$rc_python" programs/run_rc_unit_cost96_cause_v1.py predict
fi
