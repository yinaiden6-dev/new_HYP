#!/usr/bin/env python3
"""Archive final ColQwen/control/distillation text artifacts, no weights or logs."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

WS = Path(__file__).resolve().parents[1]
RC = WS / 'ICLR/new ROUTEA/RC'
DEST = WS / 'github_exports/new_HYP_20260919'
TEXT = {'.py', '.md', '.json', '.jsonl', '.csv', '.tsv', '.sbatch', '.sh'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
FAMILIES = ('rc_colqwen_base_native_v1', 'rc_colqwen_base_native_v2',
            'rc_m_distill_generalization_v1', 'rc_m_distill_full_repair_v1',
            'rc_m_distill_optimizer_repair_v1', 'rc_m_small_scaling_v1',
            'rc_fold0_m_scale_teacher_v2', 'rc_h593_simple_explanations_v1')
MARKERS = ('colqwen_base', 'm_distill', 'm_small_scaling', 'm_scale', 'simple_explanations',
           'rerank_complete', 'final_tasks_check')
EXCLUDED_DIRS = {'__pycache__', 'cache', 'tokens', 'logs'}
selected = {}; skipped = []


def add(p, reason):
    p = Path(p)
    if not p.is_file() or p.is_symlink() or p.suffix not in TEXT:
        return
    rel = p.relative_to(WS)
    if any(x in p.parts for x in EXCLUDED_DIRS) or p.name == 'cache.json':
        return
    if p.stat().st_size > 48 * 1024**2:
        skipped.append(dict(path=str(rel), reason='over48MiB; individual query files retained where available', bytes=p.stat().st_size))
        return
    data = p.read_bytes()
    if SECRET.search(data):
        raise RuntimeError('SECRET_SCAN_BLOCK:' + str(rel))
    dst = DEST / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != data:
        dst.write_bytes(data)
        if p.suffix in ('.sh', '.sbatch'):
            dst.chmod(0o755)
    selected[p] = dict(path=str(rel), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), reason=reason)


def walk(p):
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_symlink():
            continue
        if child.is_dir():
            if child.name not in EXCLUDED_DIRS:
                walk(child)
        else:
            add(child, 'experiment_text_artifact')


def main():
    for area in ('programs', 'plan', 'slurm', 'registry', 'reports'):
        for p in (RC / area).iterdir():
            if p.is_file() and any(m in p.name.lower() for m in MARKERS):
                add(p, 'current_protocol_code_report')
    for family in FAMILIES:
        walk(RC / 'results' / family)
    # Follow local source dependencies and explicitly referenced source filenames.
    byname = {}
    for area in ('programs', 'src', 'registry', 'slurm', 'plan'):
        for p in (RC / area).rglob('*'):
            if p.is_file() and p.suffix in TEXT and '__pycache__' not in p.parts:
                byname.setdefault(p.name, []).append(p)
    visited = set()
    while True:
        pending = [p for p in selected if p not in visited and p.suffix in ('.py', '.md', '.json', '.sbatch', '.sh')]
        if not pending:
            break
        for p in pending:
            visited.add(p)
            text = (DEST / p.relative_to(WS)).read_text()
            names = set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh)', text))
            if p.suffix == '.py':
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names.update(v.name.split('.')[-1] + '.py' for v in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names.add(node.module.split('.')[-1] + '.py')
            for name in names:
                for candidate in byname.get(name, []):
                    if candidate not in selected:
                        add(candidate, 'local_source_dependency')
    add(Path(__file__), 'export_script')
    inventory = DEST / 'backup/final_controls_20260924'
    inventory.mkdir(parents=True, exist_ok=True)
    payload = dict(files=sorted(selected.values(), key=lambda x: x['path']), omitted=skipped,
                   rules='No original images, binary models, checkpoints, token caches, err/out/log. JSON decision parameters and scores retained.',
                   caveat='Absolute paths and hashes preserve original provenance; excluded binary dependencies require workspace/rebuild.')
    (inventory / 'files.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for row in selected.values():
        assert hashlib.sha256((DEST / row['path']).read_bytes()).hexdigest() == row['sha256']
    print(json.dumps(dict(files=len(selected), bytes=sum(v['bytes'] for v in selected.values()),
                         skipped=skipped, family_counts=dict(Counter(v['reason'] for v in selected.values()))), indent=2))


if __name__ == '__main__':
    main()
