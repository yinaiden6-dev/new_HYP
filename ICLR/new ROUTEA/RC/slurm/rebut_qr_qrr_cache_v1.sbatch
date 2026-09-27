#!/usr/bin/env bash
#SBATCH --job-name=rebut_F71
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-F71-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-F71-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
rebut_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=10s 550s "$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/build_rebut_qr_qrr_evidence_v1.py' --start 0 --stop 71 --threads 8 --wall-seconds 480
rebut_status=$?
set -e
if [[ "$rebut_status" == 75 || "$rebut_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then exit 124; fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rebut_status"
