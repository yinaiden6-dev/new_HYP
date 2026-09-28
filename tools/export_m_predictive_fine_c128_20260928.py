#!/usr/bin/env python3
"""Archive the submitted predictive/fine-property increment; never operate Git.

Only three new result roots are traversed. Historical dependencies are verified
against the explicit 341-source audit and reused from committed blobs, rather
than walking any prior result tree. Re-run for a final accepted-results snapshot.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import export_m_causal128_attribution_20260927 as common
import export_m_attribution_depth_20260927 as previous

WS, RC = common.WS, common.RC
REPO = Path(__file__).resolve().parents[1]
BACKUP = REPO / 'backup/m_predictive_fine_c128_20260928'
ROOT_NAMES = ('rc_post_m_predictive_response_v1',
              'rc_m_fine_c128_attribution_v1',
              'rc_m_predictive_and_fine_c128_chain_v1')
KEYWORDS = ('post_m_predictive', 'fine_c128', 'predictive_fine')
AUDIT = RC / 'reports/M_ATTRIBUTION_SOURCE_COVERAGE_20260927.json'
REQUIRED = (
    'programs/run_rc_post_m_predictive_response_v1.py',
    'programs/run_rc_m_fine_c128_attribution_v1.py',
    'programs/submit_rc_m_predictive_and_fine_c128_v1.py',
    'programs/refill_rc_m_predictive_fine_dev_v1.py',
    'slurm/rc_m_predictive_and_fine_c128_v1.sbatch',
    'plan/RC_POST_M_PREDICTIVE_RESPONSE_V1_20260927.md',
    'plan/RC_M_PREDICTIVE_AND_FINE_C128_CHAIN_V1_20260927.md',
    'reports/M_ATTRIBUTION_SOURCE_COVERAGE_20260927.json',
    *(f'results/{name}/protocol.json' for name in ROOT_NAMES),
    'results/rc_m_predictive_and_fine_c128_chain_v1/submission.json',
    'results/rc_m_predictive_and_fine_c128_chain_v1/scheduler_validation.json',
)
EXPECTED = {
    'rc_post_m_predictive_response_v1/validation.json': 'POST_M_PREDICTIVE_JOIN_PASS',
    'rc_post_m_predictive_response_v1/independent_validation.json': 'POST_M_PREDICTIVE_SECOND_NUMPY_PASS',
    'rc_m_fine_c128_attribution_v1/phase_validation.json': 'FINE_C128_PHASE_ACCOUNTING_PASS',
    'rc_m_fine_c128_attribution_v1/context_validation.json': 'FINE_C128_CONTEXT_MASSES_PASS',
    'rc_m_fine_c128_attribution_v1/validation.json': 'FINE_C128_THREE_PROPERTY_ACCOUNTING_PASS',
    'rc_m_predictive_and_fine_c128_chain_v1/validation.json': 'M_PREDICTIVE_FINE_BOTH_BRANCHES_PASS',
}
CODE_EXTENSIONS = {'.py', '.sh', '.sbatch'}


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def historical_dependencies(tree, base):
    """Verify a bounded prior audit; no scan of old result or package trees."""
    audit = json.loads(common.stable_read(AUDIT))
    assert len(audit['sources']) == 341 and not audit['missing_published_sources']
    records = []
    source_map = {}
    for source in audit['sources']:
        path = WS / source['path']
        archive = source.get('archive_path', source['path'])
        target = REPO / archive
        target.resolve().relative_to(REPO.resolve())
        data = common.stable_read(path)
        assert common.digest(data) == source['sha256'], ('HISTORICAL_SOURCE_DRIFT', str(path))
        assert tree.get(archive) == blob(data), ('HISTORICAL_SOURCE_NOT_COMMITTED', archive)
        assert target.is_file() and common.stable_read(target) == data, ('LOCAL_ARCHIVE_DRIFT', archive)
        record = dict(source=source['path'], export=archive, source_sha256=source['sha256'],
                      sha256=source['sha256'], source_bytes=len(data), bytes=len(data),
                      encoding='identity', disposition='reused_committed_dependency',
                      commit=base, git_blob=tree[archive])
        records.append(record)
        source_map[path.resolve()] = record
    return source_map, records


def bound_sources(value):
    if isinstance(value, dict):
        path = value.get('path')
        if isinstance(path, str) and Path(path).is_absolute() and Path(path).suffix in CODE_EXTENSIONS:
            yield Path(path), value.get('sha256')
        for item in value.values():
            yield from bound_sources(item)
    elif isinstance(value, list):
        for item in value:
            yield from bound_sources(item)


def new_code_closure(files, historical):
    """Follow local imports of new code; stop at individually verified old code."""
    pending = [p for p in files if p.suffix == '.py']
    seen = set()
    while pending:
        path = pending.pop()
        if path.resolve() in historical or path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(common.stable_read(path).decode('utf-8'))):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                for directory in (RC / 'programs', RC / 'src'):
                    for relative in (module.replace('.', '/') + '.py', module.replace('.', '/') + '/__init__.py'):
                        dependency = directory / relative
                        if dependency.is_file() and dependency.resolve() not in historical and dependency not in files:
                            assert 'qrr' not in dependency.name.lower() and 'rebut' not in dependency.name.lower(), dependency
                            files.add(dependency)
                            pending.append(dependency)


def inventory(historical):
    files = set()
    exclusions = []
    for relative in REQUIRED:
        path = RC / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Missing required archive file: ' + relative)
        files.add(path)
    # Only matching top-level artifacts of this increment; no historical result
    # discovery and no reference-following from the large legacy audit JSON.
    for area in ('programs', 'slurm', 'plan', 'reports'):
        for path in (RC / area).iterdir():
            if path.is_file() and path.suffix in common.EXTENSIONS and any(k in path.name.lower() for k in KEYWORDS):
                files.add(path)
    for name in ROOT_NAMES:
        root = RC / 'results' / name
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError('Missing new result root: ' + name)
        protocol = json.loads(common.stable_read(root / 'protocol.json'))
        for path, expected_sha in bound_sources(protocol):
            resolved = path.resolve()
            if expected_sha:
                assert common.digest(common.stable_read(path)) == expected_sha, ('PROTOCOL_CODE_DRIFT', str(path))
            if resolved not in historical:
                resolved.relative_to(RC.resolve())
                files.add(path)
        for directory, dirs, names in os.walk(root, followlinks=False):
            directory = Path(directory)
            for child in list(dirs):
                path = directory / child
                if child in common.PRUNE or path.is_symlink():
                    dirs.remove(child)
                    exclusions.append(dict(path=str(path.relative_to(WS)), reason='runtime_or_symlink_directory'))
            for child in names:
                path = directory / child
                if path.suffix in common.EXTENSIONS and not path.is_symlink():
                    files.add(path)
                else:
                    exclusions.append(dict(path=str(path.relative_to(WS)), bytes=path.lstat().st_size,
                                           reason='tensor_checkpoint_photo_log_lock_or_temporary_not_exported'))
    new_code_closure(files, historical)
    for path in files:
        rel = path.relative_to(RC)
        assert path.suffix in common.EXTENSIONS and not path.is_symlink(), path
        if rel.parts[0] == 'results':
            assert rel.parts[1] in ROOT_NAMES, ('OUT_OF_SCOPE_RESULT_ROOT', str(path))
        assert 'qrr' not in path.name.lower() and not path.name.lower().startswith(('run_rc_rebut', 'submit_rc_rebut')), path
    return sorted(files), exclusions


def captured_status(records):
    captured = {r['export'] for r in records}
    statuses = {}
    for relative, expected in EXPECTED.items():
        path = REPO / 'ICLR/new ROUTEA/RC/results' / relative
        included = str(path.relative_to(REPO)) in captured
        actual = json.loads(path.read_text()).get('status') if included and path.is_file() else None
        statuses[relative] = dict(status=actual, expected=expected, passed=actual == expected)
    return statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='Verify inventory/dependencies only; write nothing.')
    args = parser.parse_args()
    common.REPO = REPO
    previous.REPO = REPO
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    tree = previous.git_tree()
    historical, reused = historical_dependencies(tree, base)
    files, exclusions = inventory(historical)
    print(json.dumps(dict(stage='inventory', result_roots=ROOT_NAMES, current_files=len(files),
                          verified_prior_code_dependencies=len(reused), excluded=len(exclusions))), flush=True)
    if args.preview:
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda path: previous.capture(path, tree, base), files))
    # Historical code remains at its existing path, including its six explicitly
    # recorded runtime-source relocations. No extra copy of old result data.
    copied = [r for r in records if r['disposition'] == 'copied_current_snapshot']
    statuses = captured_status(records)
    complete = all(x['passed'] for x in statuses.values())
    manifest = dict(schema='M_PREDICTIVE_FINE_C128_EXPORT_V1', snapshot_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=base, source_workspace=str(WS), result_roots=list(ROOT_NAMES), required_files=list(REQUIRED),
        records=records, reused_historical_code=reused, exclusions=exclusions, scientific_validation_receipts=statuses,
        all_final_receipts_pass=complete, source_coverage_audit='ICLR/new ROUTEA/RC/reports/M_ATTRIBUTION_SOURCE_COVERAGE_20260927.json',
        summary=dict(current_files=len(records), copied_files=len(copied), copied_bytes=sum(r['bytes'] for r in copied),
                     reused_historical_code_files=len(reused)),
        export_helpers=[dict(path='tools/'+Path(module.__file__).name, sha256=common.digest(Path(module.__file__).read_bytes()))
                        for module in (common, previous)],
        boundary='Submitted-state or final-result snapshot as identified by the captured receipts. Only three new result roots are traversed. Prior 341 code bindings are individually verified and reused, never rediscovered via old result trees. No QRR expansion, model weights, tensors, patch/token caches, raw photos, err/out logs, Git staging or push. Engineering PASS is not scientific GO. The fine-property branch studies coarse-DPT mechanisms on eleven complete C128 axes, not all128/593 or the RoMa fine-matching stage. Predictive endpoints are new synthetic M values on opened, label-selected pairs, not untouched-query confirmation.')
    common.put(BACKUP / 'files.json', (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode())
    for record in records + reused:
        data = (REPO / record['export']).read_bytes()
        assert common.digest(data) == record['sha256']
        decoded = gzip.decompress(data) if record['encoding'] == 'gzip' else data
        assert common.digest(decoded) == record['source_sha256']
    validation = dict(status='M_PREDICTIVE_FINE_EXPORT_BYTES_PASS', records=len(records),
        reused_historical_code_files=len(reused), manifest_sha256=common.digest((BACKUP/'files.json').read_bytes()),
        secret_scan='PASS', old_result_tree_reexported=False, tensor_model_raw_image_or_log_included=False,
        all_final_receipts_pass=complete, successful_git_push_implied=False)
    common.put(BACKUP/'validation.json', (json.dumps(validation, indent=2)+'\n').encode())
    text = '# Captured predictive/fine-property experiment status\n\nSnapshot UTC: '+manifest['snapshot_utc']+'\n\n'
    text += ('All required numerical and chain receipts pass. Interpret the signed results and scope in the reports.\n\n' if complete
             else '**Submitted/in-progress archive: final scientific results remain incomplete; no scientific GO is claimed.**\n\n')
    text += '| Receipt | Captured status | Passed |\n|---|---|---|\n'
    text += ''.join('| `'+rel+'` | `'+str(value['status'])+'` | '+str(value['passed'])+' |\n' for rel,value in statuses.items())
    text += '\nThe snapshot is not a live scheduler display. Submission, source checks and byte validation do not substitute for completed scientific results.\n'
    common.put(BACKUP/'EXPERIMENT_STATUS.md', text.encode())
    print(json.dumps(dict(stage='VERIFIED', **manifest['summary'], all_final_receipts_pass=complete)), flush=True)


if __name__ == '__main__':
    main()
