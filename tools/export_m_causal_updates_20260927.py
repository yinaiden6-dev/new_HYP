#!/usr/bin/env python3
"""Snapshot current M causal-analysis updates without models or tensor caches."""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

WS = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = WS / 'ICLR/new ROUTEA/RC'
REPO = WS / 'github_exports/new_HYP_20260919'
BACKUP = REPO / 'backup/m_causal_updates_20260927'
SOURCE_BASELINE = 'e2223e0b17f89e3766af83970a75f490a4f65917'
ROOTS = '''rc_m_phase_bridge_thread_repair_v2 rc_m_phase_external_internal_bridge_v1
rc_m_phase_external_internal_bridge_v2 rc_m_property_restoration_factorial_v1
rc_m_property_restoration_factorial_v2 rc_m_property_restoration_numeric_v2
rc_roma_property_restore_cpu_v1 rc_roma_property_restore_cpu_v2 rc_roma_property_restore_cpu_v3
rc_rebut_qr_qrr_v1 rc_rebut_qr_qrr_frozenbase_v2
rc_h593_m_481_492_comparison_20260927_v1 rc_h593_m_scalar_headroom_audit_20260927_v1
rc_h128_unified_resume_v2 rc_h593_unified_long4_v1'''.split()
EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.csv', '.tsv', '.html', '.sh', '.sbatch', '.yaml', '.yml', '.toml'}
PRUNE = {'__pycache__', 'worker_logs', 'logs', 'cache', 'encoder_cache', 'loading', 'locks', '.git'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_read(path):
    assert not path.is_symlink(), str(path)
    for _ in range(4):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino):
            if SECRET.search(data):
                raise RuntimeError('Credential-like content requires review: ' + str(path))
            if path.suffix == '.json':
                json.loads(data)
            return data
        time.sleep(0.05)
    raise RuntimeError('Source changed while copying: ' + str(path))


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    temp = path.with_name(path.name + '.export-tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def export(path):
    data = stable_read(path)
    rel = path.relative_to(WS)
    target = REPO / rel
    encoding = 'identity'
    if len(data) > 60 * 1024**2:
        target = Path(str(target) + '.gz')
        output = gzip.compress(data, compresslevel=6, mtime=0)
        encoding = 'gzip'
        # Never leave an older, uncompressed duplicate silently in place.
        if (REPO / rel).exists():
            raise RuntimeError('Existing plaintext needs explicit migration: ' + str(rel))
    else:
        output = data
    assert len(output) < 95 * 1024**2, str(path)
    put(target, output)
    return {'source': str(rel), 'export': str(target.relative_to(REPO)),
            'source_sha256': digest(data), 'sha256': digest(output),
            'source_bytes': len(data), 'bytes': len(output), 'encoding': encoding}


def main():
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    # A concurrent export commit must not move the source cutoff past our repairs.
    cutoff = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', SOURCE_BASELINE], cwd=REPO))
    files, catalog = set(), {}
    for area in ('reports', 'plan', 'programs', 'registry', 'slurm'):
        for path in (RC / area).iterdir():
            if path.is_file() and path.suffix in EXTENSIONS:
                catalog[path.name] = path
                if path.stat().st_mtime >= cutoff:
                    files.add(path)
    for path in (RC / 'src').rglob('*.py'):
        if '__pycache__' not in path.parts and path.stat().st_mtime >= cutoff:
            files.add(path)
    queue = list(files)
    visited = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        data = stable_read(path).decode('utf-8')
        names = set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh|json|md)', data))
        dependencies = []
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(data)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for module in modules:
                    names.add(module.split('.')[0] + '.py')
                    local = RC / 'src' / (module.replace('.', '/') + '.py')
                    if local.is_file():
                        dependencies.append(local)
        dependencies += [catalog[n] for n in names if n in catalog]
        for dependency in dependencies:
            if dependency not in files:
                files.add(dependency)
                queue.append(dependency)
    exclusions = []
    for name in ROOTS:
        root = RC / 'results' / name
        if not root.is_dir():
            raise RuntimeError('Missing result root: ' + name)
        for directory, dirs, names in os.walk(root, followlinks=False):
            directory = Path(directory)
            for name in list(dirs):
                path = directory / name
                if name in PRUNE or path.is_symlink():
                    dirs.remove(name)
                    exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'runtime_or_cache_directory'})
            for name in names:
                path = directory / name
                if path.suffix in EXTENSIONS and not path.is_symlink():
                    files.add(path)
                else:
                    exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'binary_tensor_model_or_runtime', 'bytes': path.lstat().st_size})
    print(json.dumps({'stage': 'inventory', 'text_files': len(files), 'excluded': len(exclusions)}), flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(export, sorted(files)))
    manifest = {'schema': 'M_CAUSAL_UPDATES_EXPORT_V1', 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
                'base_commit': base, 'source_cutoff_commit': SOURCE_BASELINE,
                'source_workspace': str(WS), 'result_roots': ROOTS,
                'records': records, 'exclusions': exclusions,
                'summary': {'files': len(records), 'source_bytes': sum(r['source_bytes'] for r in records),
                            'export_bytes': sum(r['bytes'] for r in records)},
                'scope_boundary': 'Per-file atomic source snapshots. Running experiments remain partial. Model/feature/patch tensors and runtime logs are excluded; full tensor replay requires the original workspace.'}
    put(BACKUP / 'files.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    # Verify the actual export bytes against the snapshot, even if a live source
    # legitimately advances after its bytes were copied.
    for record in records:
        data = (REPO / record['export']).read_bytes()
        assert digest(data) == record['sha256']
        original = gzip.decompress(data) if record['encoding'] == 'gzip' else data
        assert digest(original) == record['source_sha256']
    validation = {'status': 'EXPORT_SNAPSHOT_BYTES_PASS', 'records': len(records),
                  'manifest_sha256': digest((BACKUP / 'files.json').read_bytes()),
                  'secret_scan': 'PASS', 'excluded_binary_data_included': False,
                  'scientific_validation_implied': False}
    put(BACKUP / 'validation.json', (json.dumps(validation, indent=2) + '\n').encode())
    print(json.dumps({'stage': 'VERIFIED', **manifest['summary']}), flush=True)


if __name__ == '__main__':
    main()
