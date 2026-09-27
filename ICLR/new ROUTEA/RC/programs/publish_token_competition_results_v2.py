#!/usr/bin/env python3
"""Prepare or publish the independently validated F128 token experiment.

The default, --prepare-only, writes a reviewable snapshot outside the Git
checkout. Only --push changes that checkout, commits, or accesses the network.
It requires a clean main branch and the exact existing new_HYP remote. There
is no force push, reset, training, checkpoint loading, or credential handling.

Scientific validation, analysis validation, export byte validation and remote
publication are four separate records. A Git clone deliberately lacks images,
model weights and token/trace tensors and is not a full tensor replay package.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import quote


RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
DEFAULT_ROOT = RC / 'results/rc_token_competition_f128_v2'
DEFAULT_REPO = WS / 'github_exports/new_HYP_token128_publication_20260927'
REMOTE = 'https://github.com/yinaiden6-dev/new_HYP.git'
SCIENTIFIC_PASS = 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS'
CONFIRM_PASS = 'TOKEN_COMPETITION_F128_CONFIRM_ALL30_INDEPENDENT_JOIN_PASS'
ARMS = ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI')
TEXT = {'.json', '.csv', '.md', '.py', '.sbatch', '.sh', '.yaml', '.yml', '.toml'}
RESULT_TEXT = {'.json', '.csv', '.md'}
MAX_FILE = 60 * 1024**2
MAX_TOTAL = 350 * 1024**2
PRUNE = {'evidence', 'candidate_parts', 'cache', 'encoder_cache', 'token_cache',
         'feature_cache', 'patch_evidence', '__pycache__', 'logs', 'worker_logs',
         'loading', 'locks', '.git', 'publication'}
RUNTIME = {'evidence_verification_progress.json',
           'pilot_evidence_verification_progress.json', 'publication_receipt.json'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def safe_path(path, parent):
    """Check lexical containment and every symlink component, before resolving."""
    path = Path(os.path.abspath(path))
    parent = Path(os.path.abspath(parent))
    need(inside(path, parent), 'Path is outside the authorized tree: ' + str(path))
    current = path
    while True:
        need(not current.is_symlink(), 'Symlink is not exportable: ' + str(current))
        if current == parent:
            break
        current = current.parent
    return path


def stable_read(path, limit=MAX_FILE):
    path = Path(path)
    need(not path.is_symlink(), 'Symlink is not exportable: ' + str(path))
    for _ in range(4):
        before = path.stat()
        need(before.st_size <= limit, 'File exceeds scoped text limit: ' + str(path))
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino):
            need(not SECRET.search(data), 'Credential-like content requires review: ' + str(path))
            data.decode('utf-8')
            if path.suffix == '.json':
                json.loads(data)
            return data
        time.sleep(0.05)
    raise RuntimeError('Source changed while reading: ' + str(path))


def read_json(path):
    return json.loads(stable_read(path))


def put(path, data):
    path = Path(path)
    need(not path.is_symlink(), 'Refusing to replace a symlink: ' + str(path))
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bound_path(binding):
    need(isinstance(binding, dict) and set(('path', 'sha256')) <= set(binding), 'Missing source binding')
    path = safe_path(binding['path'], RC)
    need(digest(stable_read(path)) == binding['sha256'], 'Source SHA256 mismatch: ' + str(path))
    return path


def git(repo, *args, timeout=120):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    command = ['git', '-C', str(repo), *args]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        # Never echo credential helpers, URLs with credentials, or authentication output.
        raise RuntimeError('Git command failed (' + args[0] + ', exit ' + str(result.returncode) +
                           '); checkout and local commit are preserved for inspection')
    return result.stdout.decode('utf-8').strip()


def check_repository(repo, clean=True):
    need(repo.is_dir(), 'Export checkout does not exist')
    need(Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo.resolve(), 'Not the export repository root')
    need(git(repo, 'branch', '--show-current') == 'main', 'Publication requires branch main')
    for option in ([], ['--push']):
        urls = git(repo, 'remote', 'get-url', '--all', *option, 'origin').splitlines()
        need(urls == [REMOTE], 'Origin fetch/push URL is not the exact authorized new_HYP HTTPS URL')
    if clean:
        need(not git(repo, 'status', '--porcelain=v1', '--untracked-files=all'),
             'Export checkout has staged, unstaged or untracked changes; preserve and resolve them before publication')
    return git(repo, 'rev-parse', 'HEAD')


def load_gates(root):
    """No repository or staging mutation happens before these gates pass."""
    protocol_path = root / 'protocol.json'
    validation_path = root / 'validation.json'
    analysis_path = root / 'analysis/analysis_summary.json'
    need(validation_path.is_file(), 'Final scientific validation is not present yet')
    validation = read_json(validation_path)
    need(validation.get('status') in (SCIENTIFIC_PASS, CONFIRM_PASS),
         'Final scientific validation has not reached an exact supported PASS')
    need(bound_path(validation['protocol']) == protocol_path, 'Scientific receipt belongs to another protocol')
    protocol = read_json(protocol_path)
    confirmation = validation['status'] == CONFIRM_PASS
    need(protocol.get('version') == ('TOKEN_COMPETITION_F128_CONFIRM_V1' if confirmation else 'TOKEN_COMPETITION_F128_V2'),
         'Unsupported protocol version or scientific receipt/version mismatch')
    need(protocol.get('seeds') == ([1, 2] if confirmation else [0]) and protocol.get('arms') == list(ARMS),
         'Unexpected arm/seed registry')
    need(validation.get('fits_complete') == (30 if confirmation else 15) and validation.get('panel_queries') == 128,
         'Scientific receipt does not cover every declared fit and 128 queries')
    if confirmation:
        need(validation.get('new_seeds') == [1, 2] and validation.get('reported_seeds') == [0, 1, 2] and
             validation.get('reference_fits') == 15 and validation.get('seed_selection') is False,
             'Confirmation seed registry or reference-fit boundary mismatch')
        need(validation.get('held_labels_read_only_after_all30_checks') is True and
             validation.get('external_GO') is False and validation.get('formal_H593_replacement') is False,
             'Confirmation scope flags do not match this development publication')
        original = read_json(bound_path(protocol['origin_validation']))
        need(original.get('status') == SCIENTIFIC_PASS and
             original.get('protocol') == protocol.get('origin_protocol'), 'Original seed0 validation is not bound')
    else:
        need(validation.get('held_labels_read_only_after_all15_checks') is True and
             validation.get('external_GO') is False and validation.get('formal_H593_replacement') is False,
             'Scientific scope flags do not match this development publication')
        need(len(protocol.get('code_sources', {})) == 20, 'Expected exactly 20 protocol-bound code sources')
    need(protocol.get('code_sources'), 'Frozen code source bindings are missing')
    for relative, binding in protocol['code_sources'].items():
        path = bound_path(binding)
        if not confirmation:
            need(path == RC / relative, 'Code binding and relative path disagree')
    for name, binding in validation.get('artifacts', {}).items():
        need(bound_path(binding) == root / name, 'Final artifact is outside this experiment')
    required = {'metrics.csv', 'paired_vs_B_CAL.csv', 'paired_structural.csv', 'joined_predictions.json', 'trace_manifest.json'}
    required |= ({'perfold_metrics.csv', 'seed_summary.csv', 'scope.json', 'REPORT_TOKEN_COMPETITION_CONFIRM_V1.md'}
                 if confirmation else {'candidate_predictions.json', 'REPORT_TOKEN_COMPETITION_V2.md'})
    need(required <= set(validation.get('artifacts', {})), 'Final artifact bindings are incomplete')
    evidence_path = Path(protocol['evidence_manifest'])
    evidence = read_json(evidence_path.parent / 'evidence_validation.json')
    need(evidence.get('status') == 'TOKEN_COMPETITION_EVIDENCE128_PASS', 'Evidence validation is incomplete')
    need(bound_path(evidence['manifest']) == evidence_path, 'Evidence manifest binding mismatch')
    if not confirmation:
        need(bound_path(evidence['protocol']) == protocol_path, 'Evidence protocol binding mismatch')
    need(analysis_path.is_file(), 'Post-validation analysis summary is not present yet')
    analysis = read_json(analysis_path)
    # The analysis program supplies these bindings after validating all joins.
    need(analysis.get('status') == 'TOKEN_COMPETITION_ANALYSIS_PASS', 'Analysis has not reached the required PASS')
    need(bound_path(analysis['validation']) == validation_path, 'Analysis is stale or belongs to another scientific receipt')
    need(bound_path(analysis['protocol']) == protocol_path, 'Analysis protocol binding mismatch')
    need(analysis.get('primary_operating_point') == 'zero' and
         analysis.get('secondary_operating_point') == 'inherited_tau' and
         analysis.get('queries') == 128, 'Analysis action/population mismatch')
    for binding in analysis.get('artifacts', {}).values():
        bound_path(binding)
    report_path = bound_path(analysis['artifacts']['report'])
    need(report_path.parent in (root / 'analysis', RC / 'reports') and
         report_path.name.startswith('REPORT_TOKEN_COMPETITION'), 'Unexpected analysis report location')
    return protocol, validation, analysis, report_path


def collect(root, protocol, validation, analysis, report_path):
    """Explicit bindings plus local Python dependencies, never a workspace sweep."""
    selected, exclusions = {}, {}

    def add(path, reason, binding=None):
        path = safe_path(path, RC)
        need(path.suffix in TEXT, 'Disallowed explicit export extension: ' + str(path))
        data = stable_read(path)
        if binding is not None:
            need(digest(data) == binding['sha256'], 'Bound source changed: ' + str(path))
        selected[path] = (data, reason)

    def add_binding(binding, reason):
        add(bound_path(binding), reason, binding)

    for directory, dirs, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in list(dirs):
            path = directory / name
            if name in PRUNE or path.is_symlink():
                dirs.remove(name)
                exclusions[str(path.relative_to(WS))] = 'tensor_cache_runtime_or_symlink_directory'
        for name in sorted(names):
            path = directory / name
            if path.is_symlink() or path.suffix not in RESULT_TEXT or name in RUNTIME:
                exclusions[str(path.relative_to(WS))] = 'binary_tensor_runtime_or_oversized_progress'
            elif path.stat().st_size > MAX_FILE:
                # Required final artifacts are not silently omitted.
                need(name not in validation['artifacts'], 'Required result exceeds text export limit: ' + name)
                exclusions[str(path.relative_to(WS))] = 'text_exceeds_60_MiB_limit'
            else:
                add(path, 'experiment_text')
    for binding in protocol['code_sources'].values():
        add_binding(binding, 'frozen_code_source')
    add_binding(protocol['source_protocol'], 'original_baseline_protocol')
    original_protocol = read_json(protocol['source_protocol']['path'])
    for binding in original_protocol.get('code_sources', {}).values():
        add_binding(binding, 'original_baseline_code_dependency')
    for field in ('source_validation', 'prior_F128_protocol', 'prior_F128_residual_protocol',
                  'prior_F71_protocol', 'prior_F71_validation', 'gallery', 'original593_analysis',
                  'origin_validation', 'origin_protocol', 'origin_evidence_validation'):
        if field in protocol:
            add_binding(protocol[field], 'original_lineage_binding')
    for binding in protocol.get('dependency_bindings', {}).values():
        add_binding(binding, 'original_lineage_binding')
    for binding in validation.get('sources', {}).values():
        if isinstance(binding, dict) and 'path' in binding and Path(binding['path']).suffix in TEXT:
            add_binding(binding, 'scientific_validation_source')
    evidence_path = Path(protocol['evidence_manifest'])
    add(evidence_path, 'native_token_evidence_manifest')
    add(evidence_path.parent / 'evidence_validation.json', 'native_token_evidence_validation')
    for split in protocol['folds'].values():
        add_binding(split['train_roles'], 'original_outer_train_roles')
    baseline_protocols = [protocol]
    if 'origin_protocol' in protocol:
        origin = read_json(bound_path(protocol['origin_protocol']))
        baseline_protocols.append(origin)
        need(len(origin['code_sources']) == 20, 'Original seed0 protocol must retain its exact 20 source bindings')
        for binding in origin['code_sources'].values():
            add_binding(binding, 'original_seed0_frozen_code')
        origin_validation = read_json(bound_path(protocol['origin_validation']))
        for binding in origin_validation['artifacts'].values():
            add_binding(binding, 'original_seed0_validated_artifact')
    for baseline_protocol in baseline_protocols:
        for seeds in baseline_protocol['baseline_bindings'].values():
            for baseline in seeds.values():
                add_binding(baseline['source_result'], 'same_F128_matching_seed_B_CAL')
                add(Path(baseline['source_result']['path']).parent / 'inner_validation.json',
                    'same_F128_matching_seed_B_CAL_inner_predictions')
                for key in ('source_model', 'source_checkpoint'):
                    if key in baseline:
                        exclusions[str(Path(baseline[key]['path']).relative_to(WS))] = 'bound_binary_checkpoint_not_exported'
    old_root = Path(protocol['source_protocol']['path']).parent
    for name in ('metrics.csv', 'paired_vs_B_CAL.csv', 'validation.json', 'baseline_validation.json',
                 'joined_predictions.json', 'REPORT_F128_RESULTS.md'):
        add(old_root / name, 'historical_F128_separate_reference')
    add(report_path, 'post_validation_analysis_report')
    analysis_source = analysis.get('inputs', {}).get('analysis_source', analysis.get('analyzer'))
    need(analysis_source is not None, 'Analysis program source binding is missing')
    add_binding(analysis_source, 'analysis_program')
    for relative in ('programs/decide_token_competition_promotion_v1.py',
                     'programs/complete_token128_pipeline_20260927.py',
                     'plan/RC_TOKEN_COMPETITION_F128_CONFIRMATION_AND_PUBLICATION_20260927.md'):
        path = RC / relative
        if path.is_file():
            add(path, 'authorized_followup_and_publication_workflow')
    if 'origin_protocol' in protocol:
        origin_root = Path(protocol['origin_protocol']['path']).parent
        for relative in ('promotion_policy.json', 'analysis/promotion_decision.json'):
            if (origin_root / relative).is_file():
                add(origin_root / relative, 'original_development_replication_gate')
    add(Path(__file__), 'publication_program')
    submission_report = RC / 'reports/REPORT_TOKEN_COMPETITION_F128_V2_SUBMITTED_20260927.md'
    if submission_report.is_file():
        add(submission_report, 'submission_provenance')
    # Resolve local imports, retaining their exact current bytes. Protocol-bound
    # dependencies were already hash checked above; external packages are not copied.
    visited = set()
    while True:
        pending = [p for p in selected if p.suffix == '.py' and p not in visited]
        if not pending:
            break
        for path in pending:
            visited.add(path)
            tree = ast.parse(selected[path][0].decode('utf-8'), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    relative = Path(*module.split('.'))
                    for base in (path.parent, RC / 'programs', RC / 'src'):
                        for candidate in (base / relative.with_suffix('.py'), base / relative / '__init__.py'):
                            if candidate.is_file() and candidate not in selected:
                                add(candidate, 'local_python_import_dependency')
    need(sum(len(value[0]) for value in selected.values()) <= MAX_TOTAL,
         'Scoped export exceeds 350 MiB; inspect the inventory before expanding it')
    for path, expected in ((root / 'protocol.json', protocol), (root / 'validation.json', validation),
                           (root / 'analysis/analysis_summary.json', analysis)):
        need(json.loads(selected[path][0]) == expected, 'Gate artifact changed during collection: ' + str(path))
    for bindings in (validation['artifacts'], analysis['artifacts']):
        for binding in bindings.values():
            path = Path(binding['path'])
            need(path in selected and digest(selected[path][0]) == binding['sha256'],
                 'Collected artifact differs from its validated binding: ' + str(path))
    return selected, [{'source': path, 'reason': reason} for path, reason in sorted(exclusions.items())]


def markdown_table(rows, columns):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(label for _, label in columns) + ' |',
             '|' + '|'.join('---' for _ in columns) + '|']
    lines += ['| ' + ' | '.join(cell(row.get(key, '')) for key, _ in columns) + ' |' for row in rows]
    return '\n'.join(lines)


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'))))


def describe(root, selected, protocol, analysis, report_path):
    relroot = str(root.relative_to(WS))
    link = '../../' + quote(relroot, safe='/')
    metrics = csv_rows(selected[root / 'metrics.csv'][0])
    pairs = csv_rows(selected[root / 'paired_vs_B_CAL.csv'][0])
    structural = csv_rows(selected[root / 'paired_structural.csv'][0])
    fixed = [r for r in metrics if r['operating_point'] == 'zero']
    secondary = [r for r in metrics if r['operating_point'] == 'inherited_tau']
    confirmation = protocol['version'] == 'TOKEN_COMPETITION_F128_CONFIRM_V1'
    title = 'F128 native token competition: seeds 0/1/2 / 原生 token 竞争三种子确认' if confirmation else 'F128 native token competition V2 / 原生 token 竞争实验'
    completion = ('新增 seeds 1/2 的 30 项 reader 拟合及独立回放，并逐种子保留原 seed0 的 15 项结果。'
                  if confirmation else '已完成 15 项 seed0 reader 拟合及独立回放。')
    seed_scope = 'seeds 0, 1, 2, reported separately without selecting the best seed' if confirmation else 'seed0'
    training = ('30 new fits (seeds 1/2) plus 15 original seed0 reference fits' if confirmation else '15 fits, seed0')
    evidence_link = '../../' + quote(str(Path(protocol['evidence_manifest']).parent.relative_to(WS)), safe='/')
    columns = [('model', 'Model'), ('seed', 'Seed'), ('correct', 'Correct / 128'), ('rescues_vs_RAW', 'RAW rescues'),
               ('breaks_vs_RAW', 'RAW breaks'), ('net_vs_RAW', 'RAW net')]
    paired_columns = [('model', 'Model'), ('seed', 'Seed'), ('operating_point', 'Operating point'), ('rescues', 'Rescues'),
                      ('breaks', 'Breaks'), ('net', 'Net'), ('baseline_correct_loss_rate', 'Baseline-correct loss rate'),
                      ('changed_decisions', 'Changed decisions')]
    aggregate_note = ''
    if confirmation:
        paired_columns += [('query_seed_observations', 'Query-seed observations'), ('mean_net_per_seed', 'Mean net per seed')]
        aggregate_note = ('Rows labelled 1,2 or 0,1,2 sum rescues, breaks, net and changed decisions across repeated seeds; '
                          'they still cover 128 unique queries, not 256 or 384 independent samples. '
                          'Mean net per seed and query-seed observation counts are shown. '
                          'Identity-cluster intervals resample identities jointly across all included seeds.\n\n'
                          '多种子汇总行是同一 128 张图片的重复测量，未把重复 seed 当作新增独立样本，也未选择最优 seed。\n')
    text = f'''# {title}

{completion} 原生 ColNomic token 关系分支与同一 F128、同种子的 B_CAL 对照。
This archive records the completed development experiment and its independently validated predictions.

数据 / Dataset: H593 原自然执行顺序 0–127 的 128 张开发面板，原 identity-grouped 五折交集；不是旧 EVAL128，也不是完整 H593 测试。
Model: frozen ColNomic backbone and RoMa support; native query tokens and full-reference content matching, no new backbone forward or update.
候选 / Candidates: unchanged natural ColNomic C128, all 128 candidates retained.
Head: frozen phase-specific same-F128 same-seed 18-dimensional B_CAL plus D_g − D_RAW; RAW anchor exactly zero.
Action: HOLD/SWITCH over 127 challengers, fixed threshold zero is primary; inherited B_CAL inner threshold is secondary, without new threshold optimization.
Training: three predeclared readers × five original grouped folds; {training}. Inner validation selects epochs, with epoch0 B_CAL fallback.

## 主结果 / Primary: fixed0

{markdown_table(fixed, columns)}

## 次结果 / Secondary: inherited B_CAL threshold

{markdown_table(secondary, columns)}

## 同面板配对变化 / Paired effects against same-F128 B_CAL

救回、误伤、净变化与基线正确样本损失率均完整报告，不仅看困难样本。
Rescues, breaks, net effect, baseline-correct loss rate and changed decisions include easy/baseline-correct queries.

{aggregate_note}
{markdown_table(pairs, paired_columns)}

## 结构对照 / MULTI versus ANCHOR

相同参数，仅改变 rival graph；固定零阈值为主，继承阈值为次。
Identical parameterization; only the rival graph changes. The paired reference below is TOKEN_QRR_ANCHOR.

{markdown_table(structural, paired_columns)}

## 证据边界 / Limits

F128 是已打开的开发面板。不能据此声称完整 H593 提升、空间因果归属、唯一性或独立外部确认；不能用 held 结果重新选择分支、epoch、threshold 或 seed。
This is an opened development panel with {seed_scope}. It does not establish full-H593 improvement, spatial causality, uniqueness, or external confirmation. Historical F71 and old F128 heads remain separate lineages.
The parameter-matched MULTI/ANCHOR comparison is the structural comparison. QR comparisons also change the reader structure. Confidence intervals and easy-query regression analysis are available in the linked report.
The frozen promotion gate is only a trigger for development replication. Passing it does not turn this opened panel into an untouched test or external confirmation. All declared arms and seeds remain reported; the best seed is never substituted for the full result.
预先冻结的晋级门槛只授权开发面板复验，不等于独立统计确认或外部确认；全部声明分支与种子均保留。

## 阅读与复核 / Evidence and validation

- [完整分析 / Full analysis](../../{quote(str(report_path.relative_to(WS)), safe='/')})
- [Analysis summary]({link}/analysis/analysis_summary.json)
- [Final scientific validation]({link}/validation.json) · [Joined predictions]({link}/joined_predictions.json)
- [Frozen protocol]({link}/protocol.json) · [Final result report]({link}/{'REPORT_TOKEN_COMPETITION_CONFIRM_V1.md' if confirmation else 'REPORT_TOKEN_COMPETITION_V2.md'})
- [Evidence manifest]({evidence_link}/evidence_manifest.json) · [Evidence validation]({evidence_link}/evidence_validation.json)
- [File SHA256 inventory](files.json) · [Export byte validation](validation.json)

科学验收证明原工作区预测及独立回放通过；export validation 只验证归档字节和范围，GitHub 推送确认另存工作区 publication receipt。
Scientific validation, export byte validation and remote publication are separate checks. A local commit alone does not prove remote publication.
源码、协议、轻量结果、JSON 参数和原基线逐种子预测保留原相对路径及 SHA。原图、大模型/优化器 checkpoint、token/embedding/trace 张量、重复候选缓存及运行日志均不上传。
No raw images, binary model/optimizer checkpoints, token/embedding/trace tensors, execution logs, or candidate caches are exported. Their original path/SHA bindings remain in provenance. Full tensor replay requires the original workspace; this Git archive alone is not a complete replay package.
'''
    return text.encode()


def prepare_payload(root, repo, protocol, validation, analysis, report_path):
    selected, exclusions = collect(root, protocol, validation, analysis, report_path)
    slug = root.name[3:] if root.name.startswith('rc_') else root.name
    confirmation = validation['status'] == CONFIRM_PASS
    label = ('F128 token competition seeds 0/1/2 / 原生 token 竞争三种子确认' if confirmation
             else 'F128 native token competition V2 / 原生 token 竞争完整结果')
    count_label = '30 项新增拟合，保留 seed0 参考' if confirmation else '15 项独立验收'
    backup = Path('backup') / (slug + '_20260927')
    payload, records = {}, []
    for path, (data, reason) in sorted(selected.items()):
        relative = str(path.relative_to(WS))
        payload[relative] = data
        records.append({'source': relative, 'export': relative, 'bytes': len(data),
                        'source_sha256': digest(data), 'sha256': digest(data), 'reason': reason})
    payload[str(backup / 'README.md')] = describe(root, selected, protocol, analysis, report_path)
    readme_path = repo / 'README.md'
    current = stable_read(readme_path).decode('utf-8')
    start = '<!-- TOKEN_PUBLICATION:' + root.name + ':BEGIN -->'
    end = '<!-- TOKEN_PUBLICATION:' + root.name + ':END -->'
    entry = (start + '\n- **[' + label + ']('
             + str(backup / 'README.md') + ')**：' + count_label + '，fixed0 主结果与继承阈值次结果；'
             '同面板 B_CAL、MULTI/ANCHOR、救回/误伤及开发面板限制。\n' + end)
    need(current.count(start) == current.count(end) and current.count(start) <= 1, 'Malformed publication README marker')
    if start in current:
        updated = current[:current.index(start)] + entry + current[current.index(end) + len(end):]
    else:
        anchor = '## 从这里阅读\n'
        need(anchor in current, 'Top-level README reading index was not found')
        updated = current.replace(anchor, anchor + '\n' + entry + '\n', 1)
    payload['README.md'] = updated.encode()
    generated = [{'export': name, 'bytes': len(data), 'sha256': digest(data)}
                 for name, data in sorted(payload.items()) if name not in {r['export'] for r in records}]
    manifest = {'schema': 'TOKEN_COMPETITION_F128_TEXT_EXPORT_V1', 'experiment_root': str(root.relative_to(WS)),
                'scientific_validation_status': validation['status'],
                'scientific_validation_sha256': digest(selected[root / 'validation.json'][0]),
                'analysis_sha256': digest(selected[root / 'analysis/analysis_summary.json'][0]),
                'frozen_code_sources': protocol['code_sources'],
                'records': records, 'generated_records': generated, 'exclusions': exclusions,
                'summary': {'source_files': len(records), 'source_bytes': sum(r['bytes'] for r in records)},
                'scope': 'Original text bytes; no binary models, images, caches or trace tensors. Not a full tensor replay package.'}
    payload[str(backup / 'files.json')] = encoded(manifest)
    export_validation = {'status': 'TOKEN_COMPETITION_EXPORT_BYTES_PASS', 'manifest_sha256': digest(encoded(manifest)),
                         'source_files': len(records), 'frozen_code_sources_verified': len(protocol['code_sources']),
                         'secret_scan': 'PASS', 'raw_images_exported': 0, 'binary_tensors_exported': 0,
                         'scientific_validation_status': validation['status'],
                         'export_validation_is_scientific_validation': False,
                         'remote_publication_claimed': False}
    payload[str(backup / 'validation.json')] = encoded(export_validation)
    fit_description = ('30 independently replayed seeds 1/2 fits and all original seed0 references'
                       if confirmation else '15 independently replayed seed0 fits')
    body = ('Archive validated F128 native token competition ' + ('seed confirmation' if confirmation else 'V2 results') + '\n\n'
            'Preserve all ' + fit_description + ' on the 128-query development panel, '
            'using frozen ColNomic/RoMa inputs and the unchanged natural C128 candidate set.\n\n'
            'Document fixed-zero primary and inherited-threshold secondary results; compare '
            'same-F128 B_CAL and parameter-matched MULTI/ANCHOR with rescues, breaks, net effects '
            'and baseline-correct regressions. No full-H593 or external-confirmation claim.\n\n'
            'Include frozen code and source-baseline lineage, predictions, analysis and SHA256 '
            'export validation. Exclude images, binary checkpoints, token caches and tensor replay payloads.\n')
    for name, data in payload.items():
        need(not SECRET.search(data), 'Credential-like content in prepared export: ' + name)
        need(len(data) <= MAX_FILE, 'Prepared artifact exceeds file limit: ' + name)
    return payload, body, backup


def verify_payload(folder, payload):
    for relative, data in payload.items():
        path = safe_path(folder / relative, folder)
        need(path.is_file() and digest(path.read_bytes()) == digest(data), 'Export bytes changed: ' + relative)


def write_stage(parent, payload, body):
    """A new isolated snapshot avoids stale staged artifacts being mistaken as current."""
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='prepared-', dir=parent))
    for relative, data in payload.items():
        put(safe_path(folder / relative, folder), data)
    put(folder / 'COMMIT_MESSAGE.txt', body.encode())
    verify_payload(folder, payload)
    return folder


def publish(root, repo, payload, body, base_head):
    """Only called under --push after clean-start fetch/fast-forward."""
    need(check_repository(repo) == base_head, 'Checkout changed after publication preparation')
    for relative, data in payload.items():
        put(safe_path(repo / relative, repo), data)
    verify_payload(repo, payload)
    selected = set(payload)
    changes = git(repo, 'ls-files', '--modified', '--others', '--exclude-standard', '-z').split('\0')
    need({p for p in changes if p} <= selected, 'Unrelated changes appeared during export')
    changed = []
    for relative in sorted(payload):
        # Exact file paths only; no recursive git add of an experiment directory.
        result = subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', '--', relative],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tracked = subprocess.run(['git', '-C', str(repo), 'ls-files', '--error-unmatch', '--', relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if result.returncode or not tracked:
            changed.append(relative)
    for offset in range(0, len(changed), 100):
        git(repo, 'add', '--', *changed[offset:offset + 100])
    staged = {p for p in git(repo, 'diff', '--cached', '--name-only', '-z').split('\0') if p}
    need(staged == set(changed), 'Staged paths differ from the explicit publication selection')
    verify_payload(repo, payload)
    if staged:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', prefix='token-publication-commit-', delete=True) as handle:
            handle.write(body)
            handle.flush()
            git(repo, 'commit', '--file', handle.name)
    commit = check_repository(repo)
    git(repo, 'push', 'origin', 'main', timeout=300)
    remote = git(repo, 'ls-remote', 'origin', 'refs/heads/main').split()
    need(len(remote) == 2 and remote[0] == commit and remote[1] == 'refs/heads/main',
         'Remote main does not match the local publication commit')
    verify_payload(repo, payload)
    return {'status': 'TOKEN_COMPETITION_GITHUB_PUBLICATION_VERIFIED', 'commit': commit,
            'base_commit': base_head, 'remote': REMOTE, 'remote_main': remote[0],
            'remote_verified': True, 'new_commit_created': bool(staged), 'selected_files': len(payload),
            'experiment_root': str(root), 'repository': str(repo),
            'verified_utc': datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='default: create reviewable snapshot outside Git')
    mode.add_argument('--push', action='store_true', help='fetch/ff-only, export, scoped commit, push and verify')
    args = parser.parse_args(argv)
    root = safe_path(args.root.absolute(), RC / 'results')
    repo = args.repo.absolute()
    need(not repo.is_symlink(), 'Export repository must not be a symlink')
    need(not inside(root, repo) and not inside(repo, root), 'Experiment and export checkout must be distinct')
    protocol, validation, analysis, report_path = load_gates(root)
    check_repository(repo)
    publication = WS / 'github_exports/token_publication_records' / root.name
    need(not inside(publication, repo), 'Publication receipts must stay outside the Git checkout')
    publication.mkdir(parents=True, exist_ok=True)
    lock_path = publication.parent / ('repository-' + digest(str(repo.resolve()).encode())[:20] + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base_head = check_repository(repo)
        if args.push:
            git(repo, 'fetch', 'origin', 'main', timeout=300)
            # Reject local unpublished commits instead of accidentally publishing unrelated work.
            ahead = int(git(repo, 'rev-list', '--count', 'origin/main..HEAD'))
            if ahead:
                prior_path = publication / 'pending_push.json'
                need(prior_path.is_file(), 'Unrelated local commits are ahead of origin/main')
                prior = read_json(prior_path)
                ahead_commits = git(repo, 'rev-list', 'origin/main..HEAD').splitlines()
                need(prior.get('commit') == base_head and prior.get('experiment_root') == str(root) and
                     prior.get('repository') == str(repo) and prior.get('owned_commits') == ahead_commits,
                     'Unpublished local commit does not match this helper\'s recorded publication')
                for commit in ahead_commits:
                    own_paths = {p for p in git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit).split('\0') if p}
                    need(own_paths <= set(prior['files']), 'Unpublished commit contains unrelated paths')
                if int(git(repo, 'rev-list', '--count', 'HEAD..origin/main')):
                    # Only our explicitly recorded unpublished commits may be rebased.
                    # A conflict is left intact for inspection; nothing is reset.
                    git(repo, 'rebase', 'origin/main')
                    prior['commit'] = check_repository(repo)
                    prior['base_commit'] = git(repo, 'rev-parse', 'origin/main')
                    prior['owned_commits'] = git(repo, 'rev-list', 'origin/main..HEAD').splitlines()
                    put(prior_path, encoded(prior))
            else:
                git(repo, 'merge', '--ff-only', 'origin/main')
            base_head = check_repository(repo)
        payload, body, backup = prepare_payload(root, repo, protocol, validation, analysis, report_path)
        stage = write_stage(publication, payload, body)
        prepared = {'status': 'TOKEN_COMPETITION_PUBLICATION_PREPARED', 'snapshot': str(stage),
                    'base_commit': base_head, 'experiment_root': str(root), 'backup': str(backup),
                    'repository': str(repo),
                    'files': {p: digest(data) for p, data in sorted(payload.items())},
                    'commit_message_sha256': digest(body.encode()), 'repository_changed': False,
                    'remote_publication_claimed': False}
        put(publication / 'prepared.json', encoded(prepared))
        if not args.push:
            print(json.dumps({'status': prepared['status'], 'snapshot': str(stage),
                              'files': len(payload), 'receipt': str(publication / 'prepared.json'),
                              'repository_changed': False, 'remote_publication_claimed': False}, ensure_ascii=False))
            return 0
        # Record the prepared selection first. If a push fails after commit, retain
        # its identity for an idempotent retry; never reset or discard that commit.
        try:
            receipt = publish(root, repo, payload, body, base_head)
        except Exception:
            commit = git(repo, 'rev-parse', 'HEAD')
            if commit != base_head:
                put(publication / 'pending_push.json', encoded({
                    'commit': commit, 'base_commit': base_head, 'experiment_root': str(root),
                    'repository': str(repo), 'owned_commits': git(repo, 'rev-list', 'origin/main..HEAD').splitlines(),
                    'files': prepared['files'], 'remote_verified': False}))
            raise
        receipt['prepared_snapshot'] = str(stage)
        receipt['scientific_validation_sha256'] = digest(stable_read(root / 'validation.json'))
        put(publication / 'publication_receipt.json', encoded(receipt))
        print(json.dumps(receipt, ensure_ascii=False))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
