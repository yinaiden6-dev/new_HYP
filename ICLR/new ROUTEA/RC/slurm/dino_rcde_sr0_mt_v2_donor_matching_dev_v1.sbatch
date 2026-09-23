#!/bin/bash
#SBATCH --job-name=rcde_dgmatch
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/current_authority_v114_20260820.json
result=results/dino_rcde_sr0_mt_v2_donor_matching_v1/result.json
validation=results/dino_rcde_sr0_mt_v2_donor_matching_validation_v1/result.json
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
test "${SLURM_JOB_PARTITION:?}" = dev_cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 1
test "${SLURM_MEM_PER_NODE:?}" -eq 4096
test -x "$python"
cmp -s "$0" "$root/slurm/dino_rcde_sr0_mt_v2_donor_matching_dev_v1.sbatch"
cd "$root"
export PYTHONPATH="$root/programs:$root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test ! -e "$result"
test ! -e "$validation"
"$python" programs/materialize_dino_rcde_sr0_mt_v2_donor_matching_v1.py --authority "$authority" --output "$result"
"$python" programs/validate_dino_rcde_sr0_mt_v2_donor_matching_v1.py --authority "$authority" --result "$result" --output "$validation"
test "$(stat -c %a "$result")" = 444
test "$(stat -c %a "$validation")" = 444
