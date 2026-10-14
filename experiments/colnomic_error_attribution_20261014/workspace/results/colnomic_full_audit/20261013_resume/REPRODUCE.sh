#!/usr/bin/env bash
set -eu
colnomic_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
colnomic_python="${COLNOMIC_PYTHON:-/home/jwei/miniconda3/envs/rebut/bin/python}"
cd -- "$colnomic_root"
export COLNOMIC_WORKSPACE="$colnomic_root" PYTHONPATH="$colnomic_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec "$colnomic_python" -B -m scripts.colnomic_mechanism.reproduce_final_audit "$@"
