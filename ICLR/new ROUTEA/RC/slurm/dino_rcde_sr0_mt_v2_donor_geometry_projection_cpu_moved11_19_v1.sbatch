#!/bin/bash
#SBATCH --job-name=rcde_dgmove
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=02:00:00
#SBATCH --array=11-19%10
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/current_authority_v113_20260820.json
producer_root=results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/producer
validation_root=results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/validation
index=${SLURM_ARRAY_TASK_ID:?}
test "$index" -ge 11
test "$index" -le 19
start=$((index*12))
stop=$((start+12))
projection=$(printf '%s/shard_%03d_%03d.json' "$producer_root" "$start" "$stop")
validation=$(printf '%s/shard_%03d_%03d.json' "$validation_root" "$start" "$stop")
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 1
test "${SLURM_MEM_PER_NODE:?}" -eq 4096
test -x "$python"
cmp -s "$0" "$root/slurm/dino_rcde_sr0_mt_v2_donor_geometry_projection_cpu_moved11_19_v1.sbatch"
cd "$root"
export PYTHONPATH="$root/programs:$root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ ! -e "$projection" ]; then
  "$python" programs/materialize_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2.py --authority "$authority" --shard-ordinal "$index" --output-root "$producer_root"
fi
test -f "$projection"
test "$(stat -c %a "$projection")" = 444
if [ ! -e "$validation" ]; then
  "$python" programs/validate_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2.py --authority "$authority" --shard-ordinal "$index" --projection "$projection" --output-root "$validation_root"
fi
"$python" programs/check_dino_rcde_sr0_mt_v2_donor_geometry_projection_validation_receipt_v1.py --authority "$authority" --shard-ordinal "$index" --projection "$projection" --validation "$validation"
