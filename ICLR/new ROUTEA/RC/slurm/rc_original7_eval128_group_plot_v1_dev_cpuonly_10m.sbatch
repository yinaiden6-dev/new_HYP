#!/usr/bin/env bash
#SBATCH --job-name=original7_groupplot
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterok:5139233
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_group_plot-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rc_original7_eval128_group_plot-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
umask 077
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_plot="$rc_root/programs/plot_rc_original7_eval128_group_results_v1.py"
rc_expected_sha='285de8393168c1fdef52961be7065fac67ff78f7f62d2b3cfa8953d5101bd6b7'
rc_actual_sha=$(sha256sum "$rc_plot" | cut -d ' ' -f 1)
if [[ "$rc_actual_sha" != "$rc_expected_sha" ]]; then
 echo 'PLOT_SCRIPT_SHA_MISMATCH' >&2
 exit 1
fi
rc_runtime="${SLURM_TMPDIR:-/tmp}/original7-group-plot-${SLURM_JOB_ID:-manual}"
mkdir -p "$rc_runtime"
export TMPDIR="$rc_runtime" MPLCONFIGDIR="$rc_runtime/mpl" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
rc_seconds=$("$rc_python" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
if (( rc_seconds <= 0 )); then
 echo 'USER_RESEARCH_DEADLINE_REACHED'
 exit 1
fi
exec timeout --signal=TERM --kill-after=10s "${rc_seconds}s" "$rc_python" "$rc_plot" --execute-reviewed-result
