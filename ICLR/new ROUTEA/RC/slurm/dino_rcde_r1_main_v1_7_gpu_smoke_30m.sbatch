#!/bin/bash
# Real P0/cache R1 BAG fold-1 forward/backward and exact-resume qualification.
# This job writes only a job-scoped smoke namespace; it cannot start formal R1.
#SBATCH --job-name=rcde_r1sm17
#SBATCH --partition=dev_accelerated-h100
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=00:15:00
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_program="$rc_root/programs/smoke_dino_rcde_r1_main_v1_7_gpu.py"
rc_cache="$rc_root/runtime/dino_rcde_p0_v1_2/stable_fp16_cache"
rc_output="$rc_root/results/dino_rcde_r1_main_v1_7_gpu_smoke/smoke_job${SLURM_JOB_ID:?}"
rc_runtime_cache="${SLURM_TMPDIR:-/tmp}/rcde_r1sm17_${SLURM_JOB_ID:?}"

test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "$#" -eq 0
test -x "$rc_python"
test -f "$rc_program"
test -d "$rc_cache"
test ! -e "$rc_output"
mkdir -p "$rc_runtime_cache"

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$rc_runtime_cache/hf"
export XDG_CACHE_HOME="$rc_runtime_cache/xdg"
export TORCH_HOME="$rc_runtime_cache/torch"
export TRITON_CACHE_DIR="$rc_runtime_cache/triton"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

"$rc_python" "$rc_program" \
  --output-dir "$rc_output" \
  --cache-root "$rc_cache"
"$rc_python" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="DINO_RCDE_R1_MAIN_V17_REAL_GPU_SMOKE_PASS"; assert x["first_real_update"]["changed_parameter_scalar_count"]>0; assert x["exact_resume"]["continuous_two_updates_equals_one_save_reload_one"] is True; print(json.dumps({"status":x["status"],"changed_parameter_scalar_count":x["first_real_update"]["changed_parameter_scalar_count"],"model_tensor_exact":x["exact_resume"]["model_tensor_exact"],"optimizer_state_exact":x["exact_resume"]["optimizer_state_exact"],"rng_state_exact":x["exact_resume"]["rng_state_exact"]},sort_keys=True))' "$rc_output/result.json"
