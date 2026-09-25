#!/usr/bin/env python3
"""Copy the completed GroZi internal-transfer text evidence without changing it."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

WS = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = WS / 'ICLR/new ROUTEA/RC'
OUT = WS / 'github_exports/new_HYP_20260919'
ROOT = RC / 'results/rc_postllm_grozi_external_v1'
BACKUP = OUT / 'backup/grozi_internal_complete_20260925'
TEXT = {'.json', '.md', '.py', '.sbatch', '.csv', '.tsv', '.txt', '.sh', '.yaml', '.yml'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.export-tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def bindings(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
            yield obj
        for value in obj.values():
            yield from bindings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from bindings(value)


def main():
    files = set()
    exclusions = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        relative = p.relative_to(ROOT)
        reason = None
        if p.is_symlink():
            reason = 'symlink'
        elif 'loading' in relative.parts:
            reason = 'runtime_loading_record'
        elif p.suffix not in TEXT:
            reason = 'binary_token_cache_or_runtime_lock'
        elif p.name in ('stdout.txt', 'stderr.txt'):
            reason = 'runtime_log'
        if reason:
            exclusions.append({'path': str(p.relative_to(WS)), 'bytes': p.stat().st_size, 'reason': reason})
        else:
            files.add(p)

    authority_paths = [
        RC / 'registry/rc_postllm_grozi_external_authority_v1_20260925.json',
        RC / 'registry/rc_postllm_grozi_join_repair_v2_20260925.json',
    ]
    bound_files = []
    for ap in authority_paths:
        files.add(ap)
        for b in bindings(json.loads(ap.read_text())):
            p = Path(b['path']).resolve()
            assert p.is_relative_to(WS), p
            if p.suffix in TEXT and p.is_file():
                assert digest(p.read_bytes()) == b['sha256'], ('Frozen source mismatch', p)
                files.add(p)
                bound_files.append({'path': str(p.relative_to(WS)), 'sha256': b['sha256']})
            else:
                exclusions.append({'path': str(p.relative_to(WS)), 'sha256': b['sha256'],
                                   'bytes': p.stat().st_size if p.exists() else None,
                                   'reason': 'bound_binary_model_or_cache_dependency'})
    for area in ['programs', 'tests', 'plan', 'registry', 'slurm', 'reports']:
        for p in (RC / area).iterdir():
            if p.is_file() and 'postllm_grozi' in p.name.lower() and p.suffix in TEXT:
                files.add(p)

    def copy(p):
        assert not p.is_symlink(), p
        data = p.read_bytes()
        assert not SECRET.search(data), ('Secret review required', p)
        assert len(data) < 95 * 1024**2, p
        if p.suffix == '.json':
            json.loads(data)
        rel = p.relative_to(WS)
        target = OUT / rel
        h = digest(data)
        changed = not target.exists() or digest(target.read_bytes()) != h
        if changed:
            write(target, data)
        assert digest(target.read_bytes()) == h
        return {'path': str(rel), 'bytes': len(data), 'sha256': h, 'changed': changed}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(copy, sorted(files)), 1):
            records.append(row)
            if i % 400 == 0:
                print(json.dumps({'stage': 'copied', 'done': i, 'total': len(files)}), flush=True)
    manifest = {
        'schema': 'new_hyp_grozi_complete_text_export_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OUT, text=True).strip(),
        'previous_full_experiment_archive': 'aeeeeb42dcee3635aeebea82866267113387c44d',
        'scope': 'Completed GroZi480 frozen internal transfer; all candidate predictions, saved partial states, shard and parity validation, frozen code, source and join repair.',
        'records': records,
        'frozen_text_bindings': bound_files,
        'exclusions': exclusions,
        'encoder_cache_policy': 'Only JSON parity/hash metadata included; no payload.pt or encoder/token arrays.',
        'scientific_result_policy': 'Original bytes retained. Historical failure and correction records retained. No experiment rerun.',
        'summary': {'files': len(records), 'bytes': sum(r['bytes'] for r in records),
                    'changed_files': sum(r['changed'] for r in records),
                    'changed_bytes': sum(r['bytes'] for r in records if r['changed'])},
    }
    write(BACKUP / 'files.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'stage': 'DONE', **manifest['summary']}), flush=True)


if __name__ == '__main__':
    main()
