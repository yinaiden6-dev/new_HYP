#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
runtime="$DRUGPACK_RUNTIME"
if [[ "${1:-}" == --ocr ]]; then
  runtime="${DRUGPACK_OCR_PYTHON:-${OCR_PYTHON:-$DRUGPACK_ROOT/.venvs/ocr/bin/python}}"
  shift
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
# Older cluster nodes can opt into a compatibility loader explicitly.
if [[ -n "${DRUGPACK_GLIBC_DIR:-}" ]]; then
  exec "$DRUGPACK_GLIBC_DIR/ld-linux-x86-64.so.2" --library-path "$DRUGPACK_GLIBC_DIR" "$runtime" -u -B "$@"
fi
exec "$runtime" -u -B "$@"
