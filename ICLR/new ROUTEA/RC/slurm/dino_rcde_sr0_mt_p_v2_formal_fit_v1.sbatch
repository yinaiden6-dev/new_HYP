#!/bin/bash
#SBATCH --job-name=rcde_pv2fit
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16384
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/current_authority_v96_20260818.json
fit_root=results/dino_rcde_sr0_mt_p_v2_formal_fits_v1
validation_root=results/dino_rcde_sr0_mt_p_v2_formal_fit_validation_v1
fit_ids=(
  P_OUTER1_INNER2_FIT
  P_OUTER1_INNER3_FIT
  P_OUTER1_INNER4_FIT
  P_OUTER1_OUTER_REFIT
  P_OUTER2_INNER1_FIT
  P_OUTER2_INNER3_FIT
  P_OUTER2_INNER4_FIT
  P_OUTER2_OUTER_REFIT
  P_OUTER3_INNER1_FIT
  P_OUTER3_INNER2_FIT
  P_OUTER3_INNER4_FIT
  P_OUTER3_OUTER_REFIT
  P_OUTER4_INNER1_FIT
  P_OUTER4_INNER2_FIT
  P_OUTER4_INNER3_FIT
  P_OUTER4_OUTER_REFIT
)

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 1
test "${SLURM_MEM_PER_NODE:?}" -eq 16384
test "${SLURM_ARRAY_TASK_COUNT:?}" -eq 16
test "${SLURM_ARRAY_TASK_ID:?}" -ge 0
test "${SLURM_ARRAY_TASK_ID:?}" -le 15
test -x "$python"

cd "$root"
export PYTHONPATH="$root/programs:$root/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

fit_id="${fit_ids[$SLURM_ARRAY_TASK_ID]}"
producer="$fit_root/$fit_id/result.json"
validation="$validation_root/$fit_id/result.json"

test ! -e "$producer"
test ! -e "$validation"

"$python" programs/run_dino_rcde_sr0_mt_p_v2_fit_v1.py \
  --authority "$authority" \
  --fit-index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$fit_root"

test -f "$producer"
test "$(stat -c %a "$producer")" = 444

"$python" programs/validate_dino_rcde_sr0_mt_p_v2_fit_v1.py \
  --authority "$authority" \
  --fit-index "$SLURM_ARRAY_TASK_ID" \
  --fit-root "$fit_root/$fit_id" \
  --output "$validation"

test -f "$validation"
test "$(stat -c %a "$validation")" = 444
