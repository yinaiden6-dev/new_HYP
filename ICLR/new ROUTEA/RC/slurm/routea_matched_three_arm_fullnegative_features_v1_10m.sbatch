#!/usr/bin/env bash
#SBATCH --job-name=3a_fullfeat
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=0-7%8
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_fullfeat-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/3a_fullfeat-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark;root="$workspace/ICLR/new ROUTEA/RC";py="$workspace/.venv-romav2/bin/python";runtime=${SLURM_TMPDIR:-/tmp}/3a-fullfeat-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID};mkdir -p "$runtime/home" "$runtime/tmp";export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:$workspace" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME="$runtime/home" TMPDIR="$runtime/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;cd "$root";echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) stage=MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES";nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;"$py" programs/materialize_routea_matched_three_arm_fullnegative_features_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID";"$py" programs/validate_routea_matched_three_arm_fullnegative_features_shard_v1.py --shard "$SLURM_ARRAY_TASK_ID"
