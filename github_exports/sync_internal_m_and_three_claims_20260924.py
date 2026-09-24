#!/usr/bin/env python3
"""Publish completed round artifacts without images, weights, caches or logs."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

WS = Path(__file__).resolve().parents[1]
RC = WS / 'ICLR/new ROUTEA/RC'
DEST = WS / 'github_exports/new_HYP_20260919'
TEXT = {'.py', '.md', '.json', '.csv', '.tsv', '.jsonl', '.sbatch', '.sh'}
PRUNE = {'__pycache__', 'cache', 'tokens', 'logs', 'traces'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
FAMILIES = ('rc_simple_external_transfer_v1', 'rc_h593_qwen_quality_joint_v1',
            'rc_h593_simple_explanations_v1', 'rc_prellm_m_adapter_v1',
            'rc_prellm_m_adapter_v2', 'rc_internal_m_loading_diagnostic_v1')
MARKERS = ('three_remaining_claims', 'three_claims', 'simple_external', 'qwen_quality_joint',
           'simple_explanations', 'prellm_m', 'internal_m', 'colnomic_internal_m')
selected, excluded = {}, []


def add(path, reason):
    if not path.is_file() or path.is_symlink() or path.suffix not in TEXT:
        return
    rel = path.relative_to(WS)
    if any(p in PRUNE for p in rel.parts) or path.name == 'cache.json' or '.partial.' in path.name:
        return
    # Do not include an unfinished next-round implementation as a completed experiment.
    if 'learned_use_v3' in path.name.lower() and 'design' not in path.name.lower():
        return
    if path.stat().st_size > 48 * 1024**2:
        excluded.append({'path': str(rel), 'reason': 'larger than 48MiB', 'bytes': path.stat().st_size})
        return
    data = path.read_bytes()
    if SECRET.search(data):
        raise RuntimeError('SECRET_PATTERN:' + str(rel))
    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    changed = not dst.exists() or dst.read_bytes() != data
    if changed:
        dst.write_bytes(data)
        dst.chmod(0o755 if path.suffix in {'.sh', '.sbatch'} else 0o644)
    selected[path] = {'path': str(rel), 'sha256': hashlib.sha256(data).hexdigest(),
                      'bytes': len(data), 'changed': changed, 'reason': reason}


def main():
    for area in ('programs', 'plan', 'slurm', 'registry', 'reports', 'tests'):
        for path in (RC / area).iterdir():
            if path.is_file() and any(m in path.name.lower() for m in MARKERS):
                add(path, 'round_source_protocol_report')
    add(RC / 'results/rc_internal_m_v2_learning_scope_audit_20260924.json', 'v2_fit_audit')
    for family in FAMILIES:
        for path in (RC / 'results' / family).rglob('*'):
            add(path, 'completed_round_result')
    byname = {}
    for area in ('programs', 'src', 'slurm', 'tests'):
        for path in (RC / area).rglob('*'):
            if path.is_file() and path.suffix in TEXT and '__pycache__' not in path.parts:
                byname.setdefault(path.name, []).append(path)
    visited = set()
    while True:
        todo = [p for p in selected if p not in visited]
        if not todo:
            break
        for path in todo:
            visited.add(path)
            text = (DEST / path.relative_to(WS)).read_text()
            names = set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh)', text))
            if path.suffix == '.py':
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, ast.Import):
                        names.update(x.name.split('.')[-1] + '.py' for x in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names.add(node.module.split('.')[-1] + '.py')
            for name in names:
                for dep in byname.get(name, []):
                    if dep not in selected:
                        add(dep, 'source_dependency')
    add(Path(__file__), 'export_program')
    folder = DEST / 'backup/internal_m_and_three_claims_20260924'
    folder.mkdir(parents=True, exist_ok=True)
    record = {'scope': list(FAMILIES), 'files': sorted(selected.values(), key=lambda x: x['path']),
              'oversize_omissions': excluded,
              'exclusions': ['binary weights/checkpoints', 'raw images', 'token/feature caches',
                             'cache.json', 'err/out/log', 'partial predictions', 'unfinished v3 training implementation'],
              'retained': 'Final candidate scores, scalar head parameters, reports, code, audits and provenance.',
              'rebuild_note': 'Original absolute paths and SHA bindings retained; excluded binary inputs must be restored separately.'}
    (folder / 'files.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for item in selected.values():
        assert hashlib.sha256((DEST / item['path']).read_bytes()).hexdigest() == item['sha256']
    print(json.dumps({'files': len(selected), 'changed': sum(x['changed'] for x in selected.values()),
                      'bytes': sum(x['bytes'] for x in selected.values()), 'omitted': excluded,
                      'reasons': dict(Counter(x['reason'] for x in selected.values()))}, indent=2))


if __name__ == '__main__':
    main()
