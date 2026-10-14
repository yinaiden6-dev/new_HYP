#!/usr/bin/env bash
# Source from launchers; keep environment selection separate from model recipes.
drugpack_script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DRUGPACK_ROOT="${DRUGPACK_ROOT:-$drugpack_script_root}"
cd "$DRUGPACK_ROOT"
export DRUGPACK_ROOT="$PWD"
DRUGPACK_RUNTIME="${DRUGPACK_PYTHON:-${RELEASE_PYTHON:-${PYTHON:-python3}}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DRUGPACK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
