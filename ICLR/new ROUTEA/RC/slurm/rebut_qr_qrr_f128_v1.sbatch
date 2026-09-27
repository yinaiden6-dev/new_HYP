#!/usr/bin/env bash
#SBATCH --job-name=rebut128
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut128-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut128-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
rebut_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rebut_root='ICLR/new ROUTEA/RC/results/rc_rebut_qr_qrr_f128_v1'
rebut_stage="${1:?stage required}"
case "$rebut_stage" in
    evidence)
        rebut_command=("$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/build_rebut_qr_qrr_evidence_f128_v1.py' --threads 8 --wall-seconds 480)
        ;;
    verify)
        rebut_command=("$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/verify_rebut_qr_qrr_evidence_f128_v1.py' --threads 8)
        ;;
    baseline|residual)
        rebut_fold="${2:-${SLURM_ARRAY_TASK_ID:-}}"
        test -n "$rebut_fold"
        rebut_command=("$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/run_rebut_qr_qrr_f128_fold_v1.py' --phase "$rebut_stage" --fold "$rebut_fold" --workers 4 --threads 2 --wall-seconds 480)
        ;;
    seal)
        rebut_command=("$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/join_rebut_qr_qrr_f128_v1.py' --stage baseline --protocol "$rebut_root/protocol.json" --threads 2 --wall-seconds 480)
        ;;
    join)
        rebut_command=("$rebut_python" -u 'ICLR/new ROUTEA/RC/programs/join_rebut_qr_qrr_f128_v1.py' --stage all --protocol "$rebut_root/protocol.json" --threads 2 --wall-seconds 480)
        ;;
    *) echo "Unknown F128 stage: $rebut_stage" >&2; exit 2 ;;
esac
set +e
timeout --signal=TERM --kill-after=10s 550s "${rebut_command[@]}"
rebut_status=$?
set -e
if [[ "$rebut_status" == 75 || "$rebut_status" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 48 ]]; then exit 124; fi
    rebut_job="$SLURM_JOB_ID"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then rebut_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
    scontrol requeue "$rebut_job"
    exit 0
fi
exit "$rebut_status"
