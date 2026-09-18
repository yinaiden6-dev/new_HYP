#!/usr/bin/env bash
#SBATCH --job-name=n2_a0seal
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --input=/dev/null
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_a0seal-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/n2_a0seal-%j.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
python_bin="$workspace/.venv-colpali/bin/python"
task_runtime=${SLURM_TMPDIR:-/tmp}/routea-n2-a0-${SLURM_JOB_ID}
umask 077
test -d "$root"; test -x "$python_bin"; test -d "$root/logs"
mkdir -p "$task_runtime/home" "$task_runtime/tmp"
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$root/src:$root/programs:$root/../route_a_core:$root/..:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HOME="$task_runtime/home" TMPDIR="$task_runtime/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root"
authority_validator=programs/validate_routea_n2_fresh_d1_matched_three_arm_target_free_actions_execution_authority_v1.py
producer=programs/materialize_routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1.py
independent_validator=programs/validate_routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1.py
formal_root=results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1
"$python_bin" "$authority_validator"
if [[ ! -e "$formal_root" ]]; then
  "$python_bin" "$producer"
  "$python_bin" "$independent_validator"
fi
"$python_bin" "$authority_validator"
