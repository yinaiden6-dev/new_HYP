"""Build a deterministic source archive from an explicit allowlist."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

from . import __version__

TREES = ('drugpack', 'evaluation', 'scripts', 'configs', 'tests', 'dataset/tools')
FILES = ('README.md', 'results/README.md', 'pyproject.toml', '.gitignore',
         'requirements.txt',
         'dataset/lists/README.md', 'dataset/review/confusion/README.md',
         'other_datasets/FORB/upstream/metric_helper/metrics.py',
         'other_datasets/FORB/upstream/LICENSE', 'other_datasets/FORB/upstream/README.md')
SUFFIXES = {'.py', '.sh', '.json', '.md', '.html', '.css', '.js'}
OMIT = {'scripts/forb_dynamic_boost.py', 'scripts/forb_live_watch.py',
        'scripts/migrate_milk10k_registry.py'}
EXTERNAL = ('MILK10K', 'GroZi-3.2k', 'FORB')


def source_files(root):
    root = Path(root)
    selected = {root / name for name in FILES}
    for name in TREES:
        for path in (root / name).rglob('*'):
            if path.suffix in SUFFIXES and not any(x.startswith('.') or x == '__pycache__'
                                                for x in path.relative_to(root / name).parts):
                selected.add(path)
    for dataset in EXTERNAL:
        folder = root / 'other_datasets' / dataset
        selected.update(folder.glob('baseline_config.json'))
        selected.update(folder.glob('maxsim_execution.json'))
        selected.update((folder / 'scripts').glob('*.py'))
    files = []
    for path in sorted(selected):
        if path.relative_to(root).as_posix() in OMIT:
            continue
        if not path.exists():
            raise ValueError(f'Missing release source: {path.relative_to(root)}')
        if path.is_symlink():
            raise ValueError(f'Source links must be reviewed before packaging: {path}')
        if path.is_file():
            files.append(path)
    return files


def build(root, output):
    root, output = Path(root).resolve(), Path(output).expanduser().absolute()
    files = source_files(root)
    if output in files or output.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError('Output must be a new .tar.gz artifact outside the source allowlist')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'DrugPack-{__version__}'
    records = {}
    fd, temp = tempfile.mkstemp(prefix='.drugpack-release-', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
                    def add(name, content, mode=0o644):
                        info = tarfile.TarInfo(f'{prefix}/{name}')
                        info.size, info.mode, info.mtime = len(content), mode, 0
                        tar.addfile(info, io.BytesIO(content))
                    for path in files:
                        name = path.relative_to(root).as_posix()
                        content = path.read_bytes()
                        records[name] = hashlib.sha256(content).hexdigest()
                        add(name, content, 0o755 if path.suffix == '.sh' else 0o644)
                    add('SOURCE_MANIFEST.json', (json.dumps(dict(version=__version__, sha256=records),
                        ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode())
            raw.flush()
            os.fsync(raw.fileno())
        # Reread the actual archive before committing it on LSDF.
        with tarfile.open(temp, 'r:gz') as tar:
            for name, expected in records.items():
                with tar.extractfile(f'{prefix}/{name}') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                        raise OSError(f'Archive verification failed: {name}')
        os.replace(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    with output.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(archive=str(output), source_files=len(records), bytes=output.stat().st_size, sha256=checksum)
