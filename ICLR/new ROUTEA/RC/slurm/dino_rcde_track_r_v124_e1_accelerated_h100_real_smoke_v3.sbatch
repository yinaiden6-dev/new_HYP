#!/bin/bash
#SBATCH --job-name=rcde_v124_e1h3
#SBATCH --partition=accelerated-h100
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128000M
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_authority="$rc_root/registry/dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_authority_v3_20260826.json"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.sbatch"
rc_producer="$rc_root/programs/run_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_independent_v3.py"
rc_producer_dir="$rc_root/results/dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_v3_committed"
rc_validation_dir="$rc_root/results/dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_validation_v3_committed"
rc_v2_producer_dir="$rc_root/results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_v2_committed"
rc_v2_validation_dir="$rc_root/results/dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_validation_v2_committed"
rc_formal_producer_dir="$rc_root/results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3_committed"
rc_formal_validation_dir="$rc_root/results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v3_committed"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = accelerated-h100
test "${SLURM_CPUS_PER_TASK:?}" -eq 8
test "${SLURM_MEM_PER_NODE:?}" -eq 128000
test -n "${CUDA_VISIBLE_DEVICES:?}"
test -n "${SLURM_JOB_ID:?}"
test -f "$rc_authority"
test ! -L "$rc_authority"
test "$(stat -c %a "$rc_authority")" = 444
test ! -L "$rc_launcher"
test ! -L "$rc_producer"
test ! -L "$rc_validator"
for rc_path in "$rc_producer_dir" "$rc_validation_dir" "$rc_v2_producer_dir" "$rc_v2_validation_dir" "$rc_formal_producer_dir" "$rc_formal_validation_dir"; do
  test ! -e "$rc_path"
  test ! -L "$rc_path"
done
cmp -s "$0" "$rc_launcher"

rc_tmp=$(mktemp -d "/tmp/rcde-v124-e1h3-${SLURM_JOB_ID}.XXXXXX")
trap 'chmod -R u+w "$rc_tmp" 2>/dev/null || true; rm -rf -- "$rc_tmp"' EXIT
export TMPDIR="$rc_tmp"
export PYTHONPATH="$rc_root/src:$rc_root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$rc_root"
rc_authority_sha=$(sha256sum "$rc_authority" | awk '{print $1}')

"$rc_python" -m pytest -q \
  tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py \
  tests/test_dino_rcde_sr0_mt_v_runtime_v2.py \
  tests/test_dino_rcde_track_r_phase_b_all_patch_v1.py \
  tests/test_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3.py \
  tests/test_dino_rcde_track_r_v124_e1_accelerated_h100_real_smoke_v3.py

"$rc_python" "$rc_producer" \
  --authority "$rc_authority" \
  --authority-sha256 "$rc_authority_sha"
test -d "$rc_producer_dir"
for rc_path in "$rc_v2_producer_dir" "$rc_v2_validation_dir" "$rc_formal_producer_dir" "$rc_formal_validation_dir"; do
  test ! -e "$rc_path"
  test ! -L "$rc_path"
done

"$rc_python" "$rc_validator" \
  --authority "$rc_authority" \
  --authority-sha256 "$rc_authority_sha"
test -d "$rc_validation_dir"
for rc_path in "$rc_v2_producer_dir" "$rc_v2_validation_dir" "$rc_formal_producer_dir" "$rc_formal_validation_dir"; do
  test ! -e "$rc_path"
  test ! -L "$rc_path"
done
