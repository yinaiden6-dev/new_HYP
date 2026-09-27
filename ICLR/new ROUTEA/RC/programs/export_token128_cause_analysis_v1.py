#!/usr/bin/env python3
"""Scoped text/code export of the completed F128 cause analysis; no git push."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
ROOT = RC/'results/rc_token_competition_f128_v2'
REMOTE = 'https://github.com/yinaiden6-dev/new_HYP.git'
DENY = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
BACKUP = Path('backup/token_competition_f128_causes_20260927')


def digest(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def encoded(value): return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], text=True, capture_output=True,
                          check=True, timeout=120).stdout.strip()


def stable(path):
    assert not path.is_symlink(), path
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert len(data) < 60*1024*1024 and not DENY.search(data), path
    data.decode('utf-8')
    if path.suffix == '.json': json.loads(data)
    if path.suffix == '.py': ast.parse(data, filename=str(path))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=WS/'github_exports/new_HYP_token128_publication_20260927')
    args = parser.parse_args()
    repo = args.repo.resolve()
    assert not args.repo.is_symlink()
    assert git(repo, 'branch', '--show-current') == 'main'
    assert git(repo, 'remote', 'get-url', 'origin') == REMOTE
    assert git(repo, 'remote', 'get-url', '--push', 'origin') == REMOTE
    assert not git(repo, 'status', '--porcelain', '--untracked-files=all'), 'Preserve unrelated checkout changes'
    assert git(repo, 'rev-parse', 'HEAD') == git(repo, 'rev-parse', 'origin/main'), 'Fetch and ff-only before exporting'
    base_commit = git(repo, 'rev-parse', 'HEAD')

    val = read(ROOT/'validation.json')
    assert val['status'] == 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS' and val['fits_complete'] == 15
    for name, expected in val['artifacts'].items():
        assert expected == binding(ROOT/name)
    protocol = read(ROOT/'protocol.json')
    for expected in protocol['code_sources'].values():
        assert expected == binding(expected['path'])
    decision = read(ROOT/'analysis/promotion_decision.json')
    assert decision['status'] == 'TOKEN_COMPETITION_F128_SEED_CONFIRMATION_GATE_NO_GO'
    cases = read(ROOT/'cause_analysis/cases/case_summary.json')
    assert cases['status'] == 'PRIMARY_FIXED0_CASE_ARITHMETIC_PASS'
    assert [cases['arms'][a]['correct'] for a in protocol['arms']] == [108,108,106]
    training = read(ROOT/'cause_analysis/training/audit_summary.json')
    assert training['checked_fits'] == 15 and training['checked_epoch_records'] == 247
    assert training['selected_equals_global_best_ce_all_fits']
    mechanism = read(ROOT/'cause_analysis/mechanism/saved_edge_probe.json')
    assert mechanism['status'] == 'POSTHOC_SAVED_EDGE_DIAGNOSTICS_COMPLETE'
    assert mechanism['native_reconstruction_max_abs_error'] < 1e-10
    assert [mechanism['summaries'][n]['correct'] for n in ('native_MULTI','same_MULTI_weights_RAW_only','same_MULTI_weights_common_opponents')] == [106,107,107]
    assert mechanism['script_sha256'] == binding(ROOT/'cause_analysis/mechanism/probe_saved_edges.py')['sha256']
    uncertainty = read(ROOT/'cause_analysis/uncertainty/summary.json')
    assert all(v['queries']==128 and v['identity_groups']==48 for v in uncertainty['models'])

    sources = [p for p in (ROOT/'cause_analysis').rglob('*') if p.is_file() and
               p.suffix in {'.py','.json','.csv','.md'} and '__pycache__' not in p.parts and
               p.name != 'probe_run_stdout.json']
    sources += [RC/'reports/REPORT_TOKEN_COMPETITION_F128_V2_CAUSE_ANALYSIS_20260927.md',
                RC/'programs/analyze_token128_group_uncertainty_v1.py', Path(__file__).resolve(),
                RC/'programs/place_token128_chain_dev_20260927.py']
    # Preserve the previously prepared, unsubmitted confirmation implementation.
    from token_competition_confirm_common_v1 import NEW_SOURCES
    sources += [RC/name for name in NEW_SOURCES]
    payload = {str(p.relative_to(WS)): stable(p) for p in sorted(set(sources))}
    inventory = {name: {'sha256': digest(data), 'bytes': len(data), 'source': str(WS/name)}
                 for name, data in payload.items()}
    assert sum(len(x) for x in payload.values()) < 100*1024*1024

    link = '../../ICLR/new%20ROUTEA/RC'
    overview = f'''# F128 correction and rival-aggregation cause analysis

同一F128固定0阈值下，B_CAL103→QR108/ANCHOR108/MULTI106。复算15fits和247个epoch排除了已检查的选模/梯度/计数问题；不同对手的比较分数被直接平均，可在固定MULTI权重下解释一个丢失纠错。另一个样本和新增误伤尚未被同一算子修复。

The positive development signal and failed replication trigger are both retained. Fixed-weight saved-edge diagnostics (106→107) are posthoc mechanism evidence, not a newly trained or validated model result. Native M and candidate axes remain unchanged. The 30-fit seed confirmation was not submitted.

- [Full cause report]({link}/reports/REPORT_TOKEN_COMPETITION_F128_V2_CAUSE_ANALYSIS_20260927.md)
- [Analysis code, data and reproduction commands]({link}/results/rc_token_competition_f128_v2/cause_analysis/README.md)
- [Original completed experiment](../token_competition_f128_v2_20260927/README.md)
- [File inventory](files.json) · [Export checks](validation.json)

This addition includes all four analysis programs, complete saved-score evidence, the original-result source dependencies already in the preceding archive, and prepared seed-confirmation programs for later reuse. No images, weights, token/embedding caches or runtime err/out logs are uploaded. The mechanism input snapshot contains scalar candidate predictions and labels only. The code/data relocation checks establish saved-result replay, not backbone/model training without the excluded assets.
'''.encode()
    payload[str(BACKUP/'README.md')] = overview
    payload[str(BACKUP/'files.json')] = encoded({'base_commit': base_commit, 'files': inventory})
    payload[str(BACKUP/'validation.json')] = encoded({
        'status': 'TOKEN128_CAUSE_EXPORT_SOURCE_CHECK_PASS', 'scientific_validation': binding(ROOT/'validation.json'),
        'original_gate': binding(ROOT/'analysis/promotion_decision.json'),
        'original_results_unchanged': True, 'training_fits_checked': 15, 'epoch_records_checked':247,
        'new_training_runs':0, 'native_saved_edge_error':mechanism['native_reconstruction_max_abs_error'],
        'posthoc_diagnostics_only':True,'prepared_confirmation_submitted':False,
        'code_files':sum(p.endswith(('.py','.sbatch')) for p in inventory),
        'source_files':len(inventory),'bytes':sum(v['bytes'] for v in inventory.values()),
        'excluded':['images','binary_weights','tokens','embedding_tensors','runtime_logs'],
        'remote_publication_not_yet_verified':True})
    readme = (repo/'README.md').read_text()
    label = '### F128 correction and rival-aggregation causes (2026-09-27)'
    if label not in readme:
        readme += '\n\n'+label+'\n\n[Cause analysis, code and saved-score probes](backup/token_competition_f128_causes_20260927/README.md): ' \
                  'QR/ANCHOR net +5, MULTI net +3 on opened F128; fixed-weight operator probes locate one failed rescue, while the original replication gate remains NO_GO.\n'
    payload['README.md'] = readme.encode()
    for name,data in payload.items():
        dest=repo/name
        assert not dest.is_symlink()
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert dest.resolve().is_relative_to(repo)
        dest.write_bytes(data)
        assert dest.read_bytes()==data
    receipt = {'status':'TOKEN128_CAUSE_EXPORT_PREPARED','repository':str(repo),'remote':REMOTE,
        'base_commit':base_commit,'files':{n:digest(v) for n,v in payload.items()},
        'source_files':len(inventory),'code_files':sum(p.endswith(('.py','.sbatch')) for p in inventory),
        'bytes':sum(len(v) for v in payload.values()),'remote_verified':False}
    out=WS/'github_exports/token_publication_records/token128_cause_analysis_20260927'
    out.mkdir(parents=True,exist_ok=True)
    (out/'prepared.json').write_bytes(encoded(receipt))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))


if __name__=='__main__':main()
