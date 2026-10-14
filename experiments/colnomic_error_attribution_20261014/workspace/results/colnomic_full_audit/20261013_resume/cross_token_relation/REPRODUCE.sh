#!/usr/bin/env bash
set -eu
colnomic_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
colnomic_python="${COLNOMIC_PYTHON:-/home/jwei/miniconda3/envs/rebut/bin/python}"
cd -- "$colnomic_root"
export COLNOMIC_WORKSPACE="$colnomic_root"
export PYTHONPATH="$colnomic_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
case "${1:-verify}" in
  verify|train)
    export CUDA_VISIBLE_DEVICES=''
    "$colnomic_python" -B -m scripts.colnomic_mechanism.cross_token_probe --stage "${1:-verify}"
    "$colnomic_python" -B results/colnomic_full_audit/20261013_resume/cross_token_relation/verify_restricted_joins.py
    ;;
  features)
    colnomic_feature_source='results/colnomic_full_audit/20261013_resume/cross_token_relation/source_executed_fb515f5c8425.py'
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 "$colnomic_python" -B "$colnomic_feature_source" --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 0 --cpu-threads 1 &
    colnomic_gpu0_pid=$!
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 "$colnomic_python" -B "$colnomic_feature_source" --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 1 --cpu-threads 1 &
    colnomic_gpu1_pid=$!
    colnomic_status=0
    wait "$colnomic_gpu0_pid" || colnomic_status=1
    wait "$colnomic_gpu1_pid" || colnomic_status=1
    test "$colnomic_status" -eq 0
    CUDA_VISIBLE_DEVICES='' "$colnomic_python" -B -m scripts.colnomic_mechanism.cross_token_probe --stage aggregate
    ;;
  *)
    printf '%s\n' 'usage: REPRODUCE.sh [verify|train|features]' >&2
    exit 2
    ;;
esac
