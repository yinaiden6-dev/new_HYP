"""Release boundaries: relocation, process forwarding, cache-safe paths and packaging."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from drugpack import cli
from drugpack.release import FILES, build
from drugpack.runtime import archived_path, child_environment, project_root, python_executable

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'source with spaces'
    root.mkdir()
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n' if path.suffix == '.json' else 'release fixture\n')
    for name in ('configs/baselines.json', 'evaluation/evaluate.py', 'drugpack/method/common.py', 'drugpack/__init__.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    for name in ('DRUGPACK_ROOT', 'DRUGPACK_PYTHON', 'RELEASE_PYTHON', 'DRUGPACK_OCR_PYTHON', 'OCR_PYTHON'):
        monkeypatch.delenv(name, raising=False)
    return root


def test_root_is_independent_of_working_directory(source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DRUGPACK_ROOT', str(source))
    assert project_root() == source
    assert project_root(ROOT) == ROOT
    with pytest.raises(ValueError, match='Not a DrugPack'):
        project_root(tmp_path)


def test_archived_paths_rebase_without_mutating_manifest(source):
    old = '/cvhci/temp/jwei/release/rebut/results/runs/example/best.pt'
    manifest = {'checkpoint': old}
    assert archived_path(manifest['checkpoint'], source) == source / 'weights/method/example/best.pt'
    assert manifest['checkpoint'] == old
    assert archived_path('/lsdf/users/jwei/DrugPack/cache/a.pt', source) == source / 'cache/a.pt'
    assert archived_path('/unrelated/archive/a.pt', source) == Path('/unrelated/archive/a.pt')
    assert archived_path('cache/a.pt', source) == source / 'cache/a.pt'


def test_ocr_override_and_relative_config(source, monkeypatch):
    interpreter = source / 'ocr python'
    interpreter.write_text('#!/bin/sh\nexit 0\n')
    interpreter.chmod(0o755)
    assert python_executable(source, 'ocr', './ocr python') == str(interpreter)
    monkeypatch.setenv('DRUGPACK_OCR_PYTHON', sys.executable)
    assert python_executable(source, 'ocr', 'does-not-exist') == sys.executable
    monkeypatch.setenv('DRUGPACK_OCR_PYTHON', str(source / 'missing'))
    with pytest.raises(ValueError, match='DRUGPACK_OCR_PYTHON'):
        python_executable(source, 'ocr')


def test_child_environment_preserves_user_options_and_prioritizes_source(source, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/existing/python/path')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    env = child_environment(source)
    assert env['PYTHONPATH'] == str(source) + os.pathsep + '/existing/python/path'
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_cli_forwards_arguments_exit_status_and_cwd(source, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 7)
    monkeypatch.setattr(cli.subprocess, 'run', run)
    assert cli.main(['--root', str(source), 'baseline', '--model', 'colpali', '--cache-only']) == 7
    command, options = calls[0]
    assert command == [sys.executable, '-B', '-m', 'evaluation.evaluate', '--model', 'colpali', '--cache-only']
    assert options['cwd'] == source
    assert options['env']['DRUGPACK_ROOT'] == str(source)


def test_cli_dry_run_does_not_launch(source, monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: pytest.fail('launched a workflow'))
    assert cli.main(['--root', str(source), '--dry-run', 'method', 'evaluate', '--dataset', 'medicine']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['command'][2:] == ['-m', 'drugpack.method.evaluate', '--dataset', 'medicine']


def test_workflow_help_is_forwarded(source):
    assert cli.workflow('baseline', ['--help'], source)[-1] == '--help'
    assert cli.workflow('method', ['evaluate', '--help'], source)[-1] == '--help'
    assert cli.workflow('verify', ['results', '--read-only'], source)[-1] == '--read-only'


def test_archive_excludes_assets_and_is_reproducible(source, tmp_path):
    for name in ('weights/secret.bin', 'cache/private.pt', 'dataset/packages/raw.jpg',
                 'scripts/forb_dynamic_boost.py', 'drugpack/__pycache__/module.pyc', '.env'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'must not ship')
    first, second = tmp_path / 'first.tar.gz', tmp_path / 'second.tar.gz'
    result = build(source, first)
    assert build(source, second)['sha256'] == result['sha256']
    with tarfile.open(first) as archive:
        names = archive.getnames()
        assert not any('weights/' in name or 'cache/' in name or 'raw.jpg' in name or
                       'dynamic_boost' in name or '__pycache__' in name or name.endswith('/.env') for name in names)
        manifest_name = next(name for name in names if name.endswith('/SOURCE_MANIFEST.json'))
        manifest = json.load(archive.extractfile(manifest_name))
        prefix = manifest_name.rsplit('/', 1)[0]
        for name, expected in manifest['sha256'].items():
            assert hashlib.sha256(archive.extractfile(prefix + '/' + name).read()).hexdigest() == expected
    with pytest.raises(ValueError, match='already exists'):
        build(source, first)


def test_archive_rejects_source_symlink(source, tmp_path):
    (source / 'evaluation/link.py').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='links must be reviewed'):
        build(source, tmp_path / 'bad.tar.gz')


def test_doctor_source_needs_no_model_assets(source):
    from drugpack.doctor import inspect
    report = inspect(source)
    assert report['ok']
    assert not (source / 'weights').exists()
    assert any(c['name'] == 'ocr_python' and c['status'] == 'optional' for c in report['checks'])


@pytest.mark.parametrize('action', ['dispatch', 'report', 'figures'])
def test_legacy_help_never_starts_output_generation(source, monkeypatch, action):
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: pytest.fail('help started a job'))
    with pytest.raises(SystemExit) as result:
        cli.main(['--root', str(source), 'method', action, '--help'])
    assert result.value.code == 0


@pytest.mark.parametrize('old,new', [
    ('rebut/cache/medicine/manifest.json', 'cache/method/medicine/manifest.json'),
    ('rebut/results/selection.json', 'results/method/selection.json'),
    ('rebut/results/runs/model/best.pt', 'weights/method/model/best.pt'),
    ('rebut/train/protocol.json', 'configs/method/protocol.json'),
    ('rebut/train/post.py', 'drugpack/method/post.py'),
])
def test_method_paths_resolve_without_old_directory(source, old, new):
    for path in (old, source / old, '/cvhci/temp/jwei/release/' + old):
        assert archived_path(path, source) == source / new
    assert archived_path(source / new, source) == source / new
    assert not (source / 'rebut').exists()
