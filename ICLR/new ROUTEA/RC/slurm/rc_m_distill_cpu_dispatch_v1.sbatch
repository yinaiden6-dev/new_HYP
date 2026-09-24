#!/usr/bin/env bash
#SBATCH --job-name=M_gen_repair
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/m-gen-repair-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/m-gen-repair-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
set +e
timeout --signal=TERM --kill-after=10s 550s "$PY" -u programs/run_rc_m_distill_cpu_dispatch_v1.py "$1" "$2"
code=$?
set -e
if [[ "$code" == 124 || "$code" == 75 ]]; then
 if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then exit 124; fi
 job="$SLURM_JOB_ID"
 if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
 scontrol requeue "$job"
 exit 0
fi
if [[ "$code" != 0 ]]; then exit "$code"; fi
"$PY" -u programs/run_rc_m_distill_cpu_dispatch_v1.py "advance-$1" "$2"
