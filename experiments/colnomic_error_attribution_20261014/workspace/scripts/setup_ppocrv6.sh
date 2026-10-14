#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
OCR_ENV="${OCR_ENV:-$DRUGPACK_ROOT/.venvs/ocr}"
OCR_BASE_PYTHON="${OCR_BASE_PYTHON:-python3.11}"
"$OCR_BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "The pinned OCR environment requires Python 3.11"'
if [ ! -x "$OCR_ENV/bin/python" ]; then
    "$OCR_BASE_PYTHON" -m venv "$OCR_ENV"
fi
if [ -f weights/ppocrv6_medium/environment.lock.txt ]; then
    "$OCR_ENV/bin/python" -m pip install --index-url https://pypi.org/simple \
        -r weights/ppocrv6_medium/environment.lock.txt
else
    "$OCR_ENV/bin/python" -m pip install --index-url https://pypi.org/simple \
        'paddleocr==3.7.0' 'paddlex==3.7.2'
    "$OCR_ENV/bin/python" -m pip install --index-url https://pypi.org/simple \
        'https://paddle-whl.cdn.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp311-cp311-linux_x86_64.whl'
fi
"$OCR_ENV/bin/python" - <<'PY'
import hashlib, json, urllib.request, subprocess, sys
from pathlib import Path
root = Path('weights/ppocrv6_medium')
root.mkdir(parents=True, exist_ok=True)
manifest = Path('configs/ppocrv6_models.json')
provenance = json.loads(manifest.read_text())
(root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
for model in provenance['models']:
    for item in model['files']:
        path = root / item['path']
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.download')
        urllib.request.urlretrieve(item['url'], tmp)
        assert hashlib.sha256(tmp.read_bytes()).hexdigest() == item['sha256'], item['path']
        tmp.replace(path)
optimization = provenance.get('runtime_optimization')
if optimization:
    valid = all((root / f['path']).exists() and hashlib.sha256((root / f['path']).read_bytes()).hexdigest() == f['sha256'] for f in optimization['files'])
    if not valid:
        subprocess.check_call([sys.executable, 'scripts/export_ppocrv6_ctc.py'])
        assert all(hashlib.sha256((root / f['path']).read_bytes()).hexdigest() == f['sha256'] for f in optimization['files']), 'Export differs from the pinned inference graph'
else:
    subprocess.check_call([sys.executable, 'scripts/export_ppocrv6_ctc.py'])
print('Pinned model weights verified.')
PY
