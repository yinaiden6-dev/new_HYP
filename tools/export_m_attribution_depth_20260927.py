#!/usr/bin/env python3
"""Export the two attribution branches, orchestration and coarse correction audit.

No model execution, source/protocol changes, Git staging, commit or push.
Earlier causal128 result directories are deliberately not traversed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import export_m_causal128_attribution_20260927 as common

WS, RC, REPO = common.WS, common.RC, Path(__file__).resolve().parents[1]
BACKUP = REPO / 'backup/m_attribution_depth_20260927'
ROOT_NAMES = ('rc_m_context_binding_v1', 'rc_post_m_level_gap_v1', 'rc_m_attribution_depth_v1',
              'rc_m_coarse_path_correction_audit_v1')
KEYWORDS = ('context_binding', 'post_m_level_gap', 'attribution_depth', 'coarse_path_correction')
FIGURES = RC / 'reports/figures/m_attribution_depth_20260927'
FIGURE_MEDIA_EXTENSIONS = {'.png', '.pdf'}
EXCLUDED_SUBTREES = {
    RC / 'results/rc_m_context_binding_v1/independent_interpretation/before_scale_explanation_refresh',
}
REQUIRED_FILES = (
    'plan/RC_M_ATTRIBUTION_DEPTH_INTERPRETATION_RULES_20260927.md',
    'slurm/rc_m_attribution_depth_cpu_v1.sbatch',
    'results/rc_m_attribution_depth_v1/cpu_launcher_validation.json',
    'results/rc_m_attribution_depth_v1/downstream_cpu_replacement.json',
    'results/rc_m_attribution_depth_v1/cpu_postprocess_chain.json',
    'programs/audit_rc_m_coarse_path_corrections_v1.py',
    'reports/REPORT_M_COARSE_PATH_CORRECTION_ACCOUNTING_20260927.md',
    'results/rc_m_coarse_path_correction_audit_v1/result.json',
    'results/rc_m_coarse_path_correction_audit_v1/validation.json',
    'reports/REPORT_POST_M_LEVEL_GAP_INTERPRETATION_20260927.md',
    'reports/REPORT_M_CONTEXT_BINDING_INDEPENDENT_INTERPRETATION_20260927.md',
    'reports/REPORT_M_CONTEXT_BINDING_SCIENTIFIC_READING_20260927.md',
    'results/rc_m_context_binding_v1/independent_interpretation/validation.json',
    'results/rc_m_context_binding_v1/independent_interpretation/scientific_reading_validation.json',
    'reports/REPORT_M_ATTRIBUTION_DEPTH_FINAL_20260927.md',
    'results/rc_m_attribution_depth_v1/final_analysis_v1/validation.json',
    'results/rc_m_attribution_depth_v1/final_analysis_v1/second_review.json',
    'plan/RC_M_ATTRIBUTION_DEPTH_PREDICTIVE_THEORY_REVIEW_20260927.md',
)
EXPECTED = {
    'rc_m_context_binding_v1/upstream_validation.json': 'CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS',
    'rc_m_context_binding_v1/validation.json': 'CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS',
    'rc_post_m_level_gap_v1/validation.json': 'POST_M_LEVEL_GAP_INDEPENDENT_PASS',
    'rc_post_m_level_gap_v1/second_arithmetic_validation.json': 'POST_M_LEVEL_GAP_SECOND_ARITHMETIC_PASS',
    'rc_m_attribution_depth_v1/validation.json': 'M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS',
    'rc_m_coarse_path_correction_audit_v1/validation.json': 'COARSE_PATH_FULL128_CORRECTION_ACCOUNTING_INDEPENDENT_PASS',
    'rc_m_context_binding_v1/independent_interpretation/validation.json': 'CONTEXT_BINDING_SECOND_INTERPRETATION_COMPLETE_PASS',
    'rc_m_context_binding_v1/independent_interpretation/scientific_reading_validation.json': 'CONTEXT_BINDING_SCIENTIFIC_READING_BOUND_PASS',
    'rc_m_attribution_depth_v1/final_analysis_v1/validation.json': 'M_ATTRIBUTION_DEPTH_FINAL_REPORT_VERIFIED',
    'rc_m_attribution_depth_v1/final_analysis_v1/second_review.json': 'M_ATTRIBUTION_DEPTH_FINAL_REPORT_SECOND_REVIEW_PASS',
}


def git_tree():
    data = subprocess.check_output(['git', 'ls-tree', '-r', '-z', 'HEAD'], cwd=REPO)
    tree = {}
    for record in data.split(b'\0'):
        if not record:
            continue
        fields, name = record.split(b'\t', 1)
        mode, kind, oid = fields.split()
        if kind == b'blob':
            tree[name.decode()] = oid.decode()
    return tree


def inventory():
    files = set()
    figure_media = set()
    exclusions = []
    for relative in REQUIRED_FILES:
        path = RC / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Missing required archive file: ' + relative)
        files.add(path)
    for area in ('programs', 'slurm', 'plan', 'reports'):
        for path in (RC / area).iterdir():
            if path.is_file() and path.suffix in common.EXTENSIONS and any(x in path.name.lower() for x in KEYWORDS):
                files.add(path)
    if not FIGURES.is_dir() or FIGURES.is_symlink():
        raise RuntimeError('Missing scientific figure directory: ' + str(FIGURES))
    # Only this explicitly authorized generated-figure directory admits PNG/PDF.
    # Its source, tables and provenance enter the text dependency closure first.
    for directory, dirs, names in os.walk(FIGURES, followlinks=False):
        directory = Path(directory)
        for child in list(dirs):
            path = directory / child
            if child in common.PRUNE or path.is_symlink():
                dirs.remove(child)
                exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'runtime_or_symlink_directory'})
        for child in names:
            path = directory / child
            if path.is_symlink():
                raise RuntimeError('Symlink in figure directory: ' + str(path))
            if path.suffix in common.EXTENSIONS:
                files.add(path)
            elif path.suffix in FIGURE_MEDIA_EXTENSIONS:
                figure_media.add(path)
            else:
                exclusions.append({'path': str(path.relative_to(WS)), 'bytes': path.stat().st_size,
                                   'reason': 'unapproved_scientific_figure_directory_file_type'})
    # Existing text/source dependencies are reused from the committed archive
    # if identical. This does not walk their old result/tensor directories.
    common.source_dependencies(files)
    # source_dependencies decodes UTF-8; binary figures must enter only afterward.
    files.update(figure_media)
    for name in ROOT_NAMES:
        root = RC / 'results' / name
        if not root.is_dir():
            raise RuntimeError('Missing new experiment root: ' + name)
        for directory, dirs, names in os.walk(root, followlinks=False):
            directory = Path(directory)
            for child in list(dirs):
                path = directory / child
                if path in EXCLUDED_SUBTREES:
                    dirs.remove(child)
                    exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'superseded_derived_draft_directory'})
                elif child in common.PRUNE or path.is_symlink():
                    dirs.remove(child)
                    exclusions.append({'path': str(path.relative_to(WS)), 'reason': 'runtime_or_symlink_directory'})
            for child in names:
                path = directory / child
                if path.suffix in common.EXTENSIONS and not path.is_symlink():
                    files.add(path)
                else:
                    exclusions.append({'path': str(path.relative_to(WS)), 'bytes': path.lstat().st_size,
                                       'reason': 'tensor_checkpoint_image_or_runtime_not_exported'})
    # Protect against accidental widening to the previous 17k-file result tree.
    for path in files:
        assert not any(path.is_relative_to(root) for root in EXCLUDED_SUBTREES), ('SUPERSEDED_DRAFT', str(path))
        relative = path.relative_to(RC)
        if relative.parts[0] == 'results':
            assert relative.parts[1] in ROOT_NAMES, ('OUT_OF_SCOPE_RESULT_ROOT', str(path))
        if path.suffix in FIGURE_MEDIA_EXTENSIONS:
            path.resolve().relative_to(FIGURES.resolve())
    return sorted(files), exclusions


def capture(path, tree, base):
    data = common.stable_read(path)
    rel = str(path.relative_to(WS))
    # A dependency identical to a Git blob is a reference to the existing
    # commit, not another copied historical artifact in this increment.
    git_oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if tree.get(rel) == git_oid:
        target = REPO / rel
        assert target.resolve().is_relative_to(REPO.resolve())
        if not target.is_file() or target.read_bytes() != data:
            raise RuntimeError('Existing tracked dependency has local changes: ' + rel)
        return {'disposition': 'reused_committed_file', 'source': rel, 'export': rel,
                'source_sha256': common.digest(data), 'sha256': common.digest(data),
                'source_bytes': len(data), 'bytes': len(data), 'encoding': 'identity',
                'commit': base, 'git_blob': git_oid}
    record = common.export(path)
    record['disposition'] = 'copied_current_snapshot'
    return record


def statuses(records):
    captured = {r['export'] for r in records}
    result = {}
    for relative, expected in EXPECTED.items():
        path = REPO / 'ICLR/new ROUTEA/RC/results' / relative
        included = str(path.relative_to(REPO)) in captured
        actual = json.loads(path.read_text()).get('status') if included and path.is_file() else None
        result[relative] = {'status': actual, 'expected': expected, 'passed': actual == expected}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='Inventory only; do not copy or write a snapshot.')
    args = parser.parse_args()
    common.REPO = REPO
    files, exclusions = inventory()
    print(json.dumps({'stage': 'inventory', 'scoped_result_roots': ROOT_NAMES,
                      'files_with_source_dependencies': len(files), 'excluded': len(exclusions)}), flush=True)
    if args.preview:
        return
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    tree = git_tree()
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda path: capture(path, tree, base), files))
    copied = [r for r in records if r['disposition'] == 'copied_current_snapshot']
    reused = [r for r in records if r['disposition'] == 'reused_committed_file']
    receipt_status = statuses(records)
    complete = all(r['passed'] for r in receipt_status.values())
    manifest = {'schema': 'M_ATTRIBUTION_DEPTH_EXPORT_V1', 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
                'base_commit': base, 'source_workspace': str(WS), 'result_roots': list(ROOT_NAMES),
                'required_archive_files': list(REQUIRED_FILES),
                'allowed_scientific_figure_directory': str(FIGURES.relative_to(WS)),
                'allowed_scientific_figure_media_extensions': sorted(FIGURE_MEDIA_EXTENSIONS),
                'excluded_superseded_subtrees': sorted(str(p.relative_to(WS)) for p in EXCLUDED_SUBTREES),
                'records': records, 'exclusions': exclusions, 'scientific_validation_receipts': receipt_status,
                'all_final_receipts_pass': complete,
                'export_helper': {'path': 'tools/export_m_causal128_attribution_20260927.py',
                                  'sha256': common.digest(Path(common.__file__).read_bytes())},
                'summary': {'copied_files': len(copied), 'copied_bytes': sum(r['bytes'] for r in copied),
                            'reused_committed_files': len(reused), 'reused_bytes': sum(r['bytes'] for r in reused)},
                'boundary': 'Only four explicitly named result roots are traversed: two attribution branches, orchestration and the coarse correction audit. Earlier causal128 results remain in the preceding archive. PNG/PDF are admitted only in the named generated scientific figure directory, with source/data/hash records. Per-file stable snapshots; no weights, patch/feature tensors, raw images or logs. New-branch analysis remains pending until branch and joint acceptance receipts pass. The coarse audit independently recounts existing opened front128 full-C128 decisions, not legacy EVAL128 or new inference. Coordinate-dependent finite effects are not unique causality.'}
    common.put(BACKUP / 'files.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    for record in records:
        data = (REPO / record['export']).read_bytes()
        assert common.digest(data) == record['sha256']
        original = gzip.decompress(data) if record['encoding'] == 'gzip' else data
        assert common.digest(original) == record['source_sha256']
    validation = {'status': 'ATTRIBUTION_DEPTH_EXPORT_BYTES_PASS', 'records': len(records),
                  'manifest_sha256': common.digest((BACKUP / 'files.json').read_bytes()),
                  'secret_scan': 'PASS', 'old_result_tree_reexported': False,
                  'tensor_model_raw_image_or_log_included': False,
                  'scientific_figure_media_files': sum(Path(r['source']).suffix in FIGURE_MEDIA_EXTENSIONS for r in records),
                  'all_final_receipts_pass': complete, 'successful_git_push_implied': False}
    common.put(BACKUP / 'validation.json', (json.dumps(validation, indent=2) + '\n').encode())
    text = '# Captured status of the two new attribution branches\n\n'
    text += 'Snapshot UTC: ' + manifest['snapshot_utc'] + '\n\n'
    text += ('All required branch and closure receipts pass. Interpret scientific support in the reports.\n\n' if complete
             else '**Results remain incomplete; no final scientific conclusion is claimed by this snapshot.**\n\n')
    text += '| Receipt | Captured status | Passed |\n|---|---|---|\n'
    text += ''.join('| `' + rel + '` | `' + str(v['status']) + '` | ' + str(v['passed']) + ' |\n' for rel, v in receipt_status.items())
    text += '\nNumerical-domain checks, scheduler submission and byte verification are not completed scientific results.\n'
    common.put(BACKUP / 'EXPERIMENT_STATUS.md', text.encode())
    print(json.dumps({'stage': 'VERIFIED', **manifest['summary'], 'all_final_receipts_pass': complete}), flush=True)


if __name__ == '__main__':
    main()
