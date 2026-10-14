"""Resolve source checkouts and Python environments without importing ML libraries."""
import os
from pathlib import Path
import shutil
import sys


def project_root(value=None):
    """An explicit root wins over the environment, then the source checkout."""
    candidate = value or os.environ.get('DRUGPACK_ROOT')
    root = Path(candidate).expanduser().resolve() if candidate else Path(__file__).resolve().parents[1]
    required = ('configs/baselines.json', 'evaluation/evaluate.py', 'drugpack/method/common.py')
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f'Not a DrugPack source tree: {root}. Set DRUGPACK_ROOT or --root; missing {missing}')
    return root


def python_executable(root, role='main', configured=None):
    """Environment overrides, project-local optional environments, then defaults.

    A configured OCR interpreter remains supported for older config files.
    Relative executable paths are always relative to the project, not the caller.
    """
    if role == 'ocr':
        value = os.environ.get('DRUGPACK_OCR_PYTHON') or os.environ.get('OCR_PYTHON')
        value = value or configured or str(Path(root) / '.venvs/ocr/bin/python')
    elif role == 'biomed':
        value = os.environ.get('MILK_BIOMED_PYTHON') or str(Path(root) / '.venvs/biomed/bin/python')
    else:
        value = os.environ.get('DRUGPACK_PYTHON') or os.environ.get('RELEASE_PYTHON') or sys.executable
    value = os.path.expanduser(str(value))
    if '/' in value:
        path = Path(value)
        executable = path if path.is_absolute() else Path(root) / path
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable.absolute())
    else:
        executable = shutil.which(value)
        if executable:
            return executable
    hint = 'DRUGPACK_OCR_PYTHON' if role == 'ocr' else ('MILK_BIOMED_PYTHON' if role == 'biomed' else 'DRUGPACK_PYTHON')
    raise ValueError(f'Python interpreter not executable: {value}. Set {hint} to your environment Python.')


def child_environment(root):
    env = os.environ.copy()
    env['DRUGPACK_ROOT'] = str(root)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return env


# Old roots occur inside signed experiment manifests. Resolve them at read time.
LEGACY_ROOTS = tuple(map(Path, ('/cvhci/temp/jwei/drug_pack', '/cvhci/temp/jwei/release',
                              '/lsdf/users/jwei/DrugPack', '/mnt/lsdf_jwei/users/jwei/DrugPack')))


def archived_path(value, root):
    path, root = Path(value), Path(root)
    for old in (root, *LEGACY_ROOTS):
        try:
            path = path.relative_to(old)
            break
        except ValueError:
            pass
    if not path.is_absolute():
        mappings = (
            ('rebut/results/runs', 'weights/method'),
            ('rebut/results', 'results/method'),
            ('rebut/cache', 'cache/method'),
            ('rebut/核心代码/RC/programs', 'drugpack/method/legacy'),
            ('rebut/train', 'configs/method' if path.suffix == '.json' else 'drugpack/method'),
        )
        for old, new in mappings:
            try:
                return root / new / path.relative_to(old)
            except ValueError:
                pass
    return path if path.is_absolute() else root / path
