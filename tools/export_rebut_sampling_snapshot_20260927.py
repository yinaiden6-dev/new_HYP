#!/usr/bin/env python3
"""Copy this research round's text artifacts, preserving their original bytes."""
import ast
import collections
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

import export_mechanism_snapshot_20260925 as base

WS, RC, OUT = base.WS, base.RC, base.OUT
BACKUP = OUT / 'backup/rebut_and_unified_sampling_20260927'
ROOTS = '''rc_rebut_qr_qrr_v1 rc_rebut_qr_qrr_frozenbase_v2
rc_h593_m_481_492_comparison_20260927_v1 rc_h593_m_scalar_headroom_audit_20260927_v1
rc_m_property_restoration_factorial_v1 rc_m_phase_external_internal_bridge_v1
rc_m_phase_external_internal_bridge_v2 rc_m_phase_bridge_thread_repair_v2
rc_roma_property_restore_cpu_v1 rc_roma_property_restore_cpu_v2 rc_roma_property_restore_cpu_v3
rc_h128_unified_resume_v1 rc_h128_unified_resume_v2 rc_h593_unified_long4_v1'''.split()
PRUNE = {'logs', 'cache', '__pycache__', 'locks', 'loading', 'encoder_cache', 'patch_evidence'}
TEXT = base.TEXT | {'.html'}
PATCH_RESULT_ROOTS = {'rc_m_phase_external_internal_bridge_v1', 'rc_m_phase_external_internal_bridge_v2'}


def main():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OUT, text=True).strip()
    cutoff = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', 'HEAD'], cwd=OUT))
    catalog = collections.defaultdict(list)
    files, excluded = set(), []
    for area in ('reports', 'plan', 'programs', 'registry', 'slurm', 'tests', 'src'):
        directory = RC / area
        if not directory.exists():
            continue
        candidates = directory.rglob('*.py') if area == 'src' else directory.iterdir()
        for path in candidates:
            if path.is_file() and not path.is_symlink() and path.suffix in TEXT:
                catalog[path.name].append(path)
                if path.stat().st_mtime >= cutoff:
                    files.add(path)
    for result_name in ROOTS:
        root = RC / 'results' / result_name
        assert root.is_dir(), root
        for parent, dirs, names in os.walk(root, followlinks=False):
            parent = Path(parent)
            for directory in list(dirs):
                path = parent / directory
                if directory in PRUNE or path.is_symlink():
                    dirs.remove(directory)
                    excluded.append({'path': str(path.relative_to(WS)), 'reason': 'cache_or_execution_directory'})
            for name in names:
                path = parent / name
                patch_result = path.suffix == '.npz' and result_name in PATCH_RESULT_ROOTS
                if patch_result:
                    assert path.stat().st_size < 4 * 1024**2, path
                    with zipfile.ZipFile(path) as archive:
                        assert set(archive.namelist()) == {'worlds.npy', 'M.npy', 'L.npy', 'maxsim.npy', 'argmax.npy'}, path
                if ((path.suffix in TEXT or patch_result) and not path.is_symlink()
                        and name not in ('stdout.txt', 'stderr.txt')):
                    files.add(path)
                else:
                    excluded.append({'path': str(path.relative_to(WS)),
                                     'bytes': path.stat().st_size if path.is_file() else None,
                                     'reason': 'binary_checkpoint_cache_or_runtime_file'})
    # Recover text source dependencies mentioned by code, reports or contracts.
    pending, seen = list(files), set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix not in ('.py', '.md', '.sbatch', '.sh', '.json'):
            continue
        content = base.clean_read(path).decode('utf-8')
        names = set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh|json|md)', content))
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(content)):
                if isinstance(node, ast.Import):
                    names.update(item.name.split('.')[-1] + '.py' for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split('.')[-1] + '.py')
        for name in names:
            for dependency in catalog.get(name, []):
                if dependency not in files:
                    files.add(dependency)
                    pending.append(dependency)
    print(json.dumps({'stage': 'inventory', 'text_files': len(files), 'excluded': len(excluded)}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(base.export_file, sorted(files)))
    manifest = {
        'schema': 'new_hyp_rebut_and_sampling_text_snapshot_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'base_commit': head, 'source_workspace': str(WS), 'result_roots': ROOTS,
        'records': records, 'exclusions': excluded,
        'snapshot_policy': 'Original bytes retained. Running jobs are a timestamped snapshot, not completed results.',
        'compact_npz_policy': 'Only the two phase-bridge roots: worlds, M, L, per-patch maxsim and argmax. No image, embedding, model or token arrays.',
        'summary': {'files': len(records), 'bytes': sum(row['bytes'] for row in records),
                    'excluded': len(excluded)},
    }
    base.put(BACKUP / 'files.json', (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode())
    for row in records:
        assert base.digest((OUT / row['export']).read_bytes()) == row['sha256'], row['export']
    validation = {'status': 'ALL_EXPORTED_SOURCE_SNAPSHOT_HASHES_PASS',
                  'files': len(records), 'bytes': manifest['summary']['bytes'],
                  'raw_images_exported': 0, 'binary_model_or_token_cache_exported': 0,
                  'execution_logs_exported': 0, 'remote_push_verified': False}
    base.put(BACKUP / 'validation.json', (json.dumps(validation, indent=2) + '\n').encode())
    print(json.dumps({'stage': 'DONE', **manifest['summary']}), flush=True)


if __name__ == '__main__':
    main()
