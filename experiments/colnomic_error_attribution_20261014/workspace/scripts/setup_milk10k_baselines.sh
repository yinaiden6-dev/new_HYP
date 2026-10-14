#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
MILK_PYTHON="${MILK_PYTHON:-$DRUGPACK_RUNTIME}"
MILK_BIOMED_ENV="${MILK_BIOMED_ENV:-$DRUGPACK_ROOT/.venvs/biomed}"
export PYTHONDONTWRITEBYTECODE=1
"$MILK_PYTHON" -B -m evaluation.milk10k --prepare
"$MILK_PYTHON" -B - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
data = Path('other_datasets/MILK10K')
config = json.loads((data / 'baseline_config.json').read_text())
files = {
    'dinov3': ['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE.md'],
    'biomedclip': ['open_clip_config.json', 'open_clip_pytorch_model.bin', 'README.md', 'LICENSE.md'],
}
for model, allow in files.items():
    spec = config['models'][model]['image_model']
    snapshot_download(repo_id=spec['repo'], revision=spec['revision'], allow_patterns=allow,
                      local_dir=Path(config['models'][model]['checkpoint']),
                      cache_dir=Path('weights/.huggingface'))
PY
if [ ! -x "$MILK_BIOMED_ENV/bin/python" ]; then
    "$MILK_PYTHON" -m venv --system-site-packages "$MILK_BIOMED_ENV"
fi
"$MILK_BIOMED_ENV/bin/python" -m pip install --no-compile \
    open_clip_torch==2.23.0 timm==0.9.12 transformers==4.35.2 numpy==1.26.4
"$MILK_BIOMED_ENV/bin/python" -m pip freeze > other_datasets/MILK10K/cache/intermediate/biomed_requirements.txt
