#!/usr/bin/env python3
"""Repeatable, scoped attribution snapshot. Does not stage, commit, or push Git."""
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
REPO = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_causal128_attribution_chain_v1'
BACKUP = REPO / 'backup/m_causal128_attribution_20260927'
EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.csv', '.tsv', '.html', '.sh', '.sbatch', '.yaml', '.yml', '.toml'}
PRUNE = {'__pycache__', 'logs', 'worker_logs', '.git', 'locks'}
SENSITIVE_NAMES = {'auth.json', 'hosts.yml', 'credentials.json', 'credentials', '.env', '.netrc', '.git-credentials'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
EXPECTED = {
    'paths/cache_validation.json': 'M_CAUSAL128_EXTRACT_ALL128_PASS',
    'paths/validation.json': 'M_CAUSAL128_COMPENSATION_ALL5_INDEPENDENT_PASS',
    'pair_structure/validation.json': 'M_CAUSAL128_PAIR_STRUCTURE_PASS',
    'bridge/validation.json': 'CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS',
    'bridge/joint_pairs/validation.json': 'JOINT_RESTORATION_FIXED14_BRIDGE_PASS',
    'phase_replication/validation.json': 'PHASE_NEWGROUP_REPLICATION_JOIN_PASS',
    'final_summary/validation.json': 'M_CAUSAL128_ATTRIBUTION_SUMMARY_PASS',
    'property_compensation_fixed_pairs_v1/validation.json': 'PROPERTY_FIXED_PAIR_COMPENSATION_INDEPENDENT_PASS',
    'newgroup_property_bridge_v1/validation.json': 'NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_read(path):
    assert not path.is_symlink(), str(path)
    assert path.name not in SENSITIVE_NAMES and not path.name.startswith('.env.'), str(path)
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
    # The source closure is constrained to this experiment's RC workspace.
    path.resolve().relative_to(RC.resolve())
    data = stable_read(path)
    rel = path.relative_to(WS)
    target = REPO / rel
    encoding = 'identity'
    if len(data) > 60 * 1024**2:
        target = Path(str(target) + '.gz')
        output = gzip.compress(data, compresslevel=6, mtime=0)
        encoding = 'gzip'
        if (REPO / rel).exists():
            raise RuntimeError('Existing plaintext needs explicit migration: ' + str(rel))
    else:
        output = data
    assert len(output) < 95 * 1024**2, str(path)
    target.resolve().relative_to(REPO.resolve())
    put(target, output)
    return {'source': str(rel), 'export': str(target.relative_to(REPO)),
            'source_sha256': digest(data), 'sha256': digest(output),
            'source_bytes': len(data), 'bytes': len(output), 'encoding': encoding}


def source_dependencies(files):
    catalog = {}
    for area in ('reports', 'plan', 'programs', 'registry', 'slurm'):
        for path in (RC / area).iterdir():
            if path.is_file() and path.suffix in EXTENSIONS:
                catalog[path.name] = path
    pending = list(files)
    visited = set()
    while pending:
        path = pending.pop()
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
                pending.append(dependency)


def status_from_export(records):
    statuses = {}
    exported_root = REPO / ROOT.relative_to(WS)
    captured_paths = {record['export'] for record in records}
    for rel, expected in EXPECTED.items():
        path = exported_root / rel
        captured = str(path.relative_to(REPO)) in captured_paths
        status = json.loads(path.read_text()).get('status') if captured and path.exists() else None
        statuses[rel] = {'status': status, 'expected': expected, 'passed': status == expected}
    return statuses


def main():
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    files = set()
    # Only this chain's sources, repair, plan and reports initiate the closure.
    for area in ('programs', 'slurm', 'plan', 'reports'):
        for path in (RC / area).iterdir():
            if not path.is_file() or path.suffix not in EXTENSIONS:
                continue
            low = path.name.lower()
            if ('causal128' in low or 'phase_newgroups' in low):
                files.add(path)
    source_dependencies(files)
    exclusions = []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        directory = Path(directory)
        for name in list(dirs):
            path = directory / name
            if name in PRUNE or path.is_symlink():
                dirs.remove(name)
                exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'runtime_or_symlink_directory'})
        for name in names:
            path = directory / name
            if path.suffix in EXTENSIONS and not path.is_symlink():
                files.add(path)
            else:
                exclusions.append({'path': str(path.relative_to(WS)), 'bytes': path.lstat().st_size,
                                   'reason': 'tensor_checkpoint_image_or_runtime_not_exported'})
    print(json.dumps({'stage': 'inventory', 'files': len(files), 'excluded': len(exclusions)}), flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(export, sorted(files)))
    statuses = status_from_export(records)
    complete = all(x['passed'] for x in statuses.values())
    manifest = {'schema': 'M_CAUSAL128_ATTRIBUTION_EXPORT_V1',
                'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'base_commit': base,
                'source_workspace': str(WS), 'source_result_root': str(ROOT),
                'records': records, 'exclusions': exclusions,
                'scientific_validation_receipts': statuses,
                'all_final_receipts_pass': complete,
                'summary': {'files': len(records), 'source_bytes': sum(r['source_bytes'] for r in records),
                            'export_bytes': sum(r['bytes'] for r in records)},
                'scope_boundary': 'Per-file stable snapshot. No tensors, model checkpoints, original photos or execution logs. Full tensor replay requires the original workspace. Nonseparability is not a new category beyond pairwise information. Whole J/P-input ablation and separate fixed-pair property compensation have different scopes. The latter is bounded scalar-readout retraining, not unrestricted information necessity or complete C128 evaluation.'}
    put(BACKUP / 'files.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    for record in records:
        data = (REPO / record['export']).read_bytes()
        assert digest(data) == record['sha256']
        original = gzip.decompress(data) if record['encoding'] == 'gzip' else data
        assert digest(original) == record['source_sha256']
    validation = {'status': 'EXPORT_SNAPSHOT_BYTES_PASS', 'records': len(records),
                  'manifest_sha256': digest((BACKUP / 'files.json').read_bytes()),
                  'secret_scan': 'PASS', 'excluded_binary_data_included': False,
                  'all_final_receipts_pass': complete,
                  'successful_git_push_implied': False}
    put(BACKUP / 'validation.json', (json.dumps(validation, indent=2) + '\n').encode())
    put(BACKUP / 'EXPERIMENT_STATUS.md', ('# Captured experiment status\n\n'
        + 'Snapshot UTC: ' + manifest['snapshot_utc'] + '\n\n'
        + ('All required final receipts pass.' if complete else '**Incomplete snapshot: final scientific acceptance remains pending.**')
        + '\n\n| Artifact | Captured status | Passed |\n|---|---|---|\n'
        + ''.join('| `' + rel + '` | `' + str(v['status']) + '` | ' + str(v['passed']) + ' |\n' for rel,v in statuses.items())
        + '\nThis table reads the exported bytes, not current scheduler state. Historical failed/cancelled execution records remain in the archive. A final receipt does not prove all four causal steps for every specific property.\n').encode())
    print(json.dumps({'stage': 'VERIFIED', **manifest['summary'], 'all_final_receipts_pass': complete}), flush=True)


if __name__ == '__main__':
    main()
