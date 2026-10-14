"""Fast, read-only release inventory. Does not claim content or GPU validation."""
import csv
import json
from pathlib import Path
import sys

from .runtime import archived_path, python_executable


def inspect(root, profile='source'):
    root = Path(root)
    checks = []

    def record(name, status, detail):
        checks.append(dict(name=name, status=status, detail=detail))

    def required(name):
        exists = (root / name).exists()
        record(name, 'ok' if exists else 'missing', 'present' if exists else 'not installed in this source tree')

    record('python', 'ok' if sys.version_info >= (3, 12) else 'missing', sys.version.split()[0])
    for name in ('pyproject.toml', 'configs/baselines.json', 'evaluation/evaluate.py', 'drugpack/method/common.py'):
        required(name)
    for role in ('main', 'ocr', 'biomed'):
        try:
            record(f'{role}_python', 'ok', python_executable(root, role))
        except ValueError as error:
            record(f'{role}_python', 'missing' if role == 'main' else 'optional', str(error))
    if profile in ('baseline', 'method'):
        for name in ('photos', 'references', 'packages', 'positives', 'groups'):
            relative = f'dataset/lists/{name}.csv'
            required(relative)
            if (root / relative).is_file():
                with (root / relative).open(newline='') as stream:
                    count = sum(1 for _ in csv.DictReader(stream))
                record(name + '_rows', 'ok', count)
        config = json.loads((root / 'configs/baselines.json').read_text())
        for model in config['models'].values():
            for field in ('checkpoint', 'base_checkpoint'):
                if model.get(field):
                    required(model[field])
        for name in ('weights/ppocrv6_medium', 'cache/ocr', 'cache/image_embeddings',
                     'cache/ocr_embeddings', 'cache/scores', 'results/all_results.csv'):
            required(name)
    if profile == 'method':
        for name in ('weights/romav2/source/src', 'weights/romav2/torchhub',
                     'drugpack/method/legacy', 'cache/method/medicine/manifest.json',
                     'results/method/selection.json'):
            required(name)
        selection = root / 'results/method/selection.json'
        if selection.exists():
            missing = [name for name, run in json.loads(selection.read_text())['runs'].items()
                       if not archived_path(run['checkpoint'], root).is_file()]
            record('selected_checkpoints', 'missing' if missing else 'ok', missing or 'all present (not hashed)')
    return dict(root=str(root), profile=profile, ok=not any(c['status'] == 'missing' for c in checks), checks=checks)
