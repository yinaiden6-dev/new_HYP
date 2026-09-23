#!/usr/bin/env python3
"""Incremental, nondeleting research publication; excludes binary caches and logs."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess
import datetime
from collections import Counter

WS = Path(__file__).resolve().parents[1]
RC = WS / 'ICLR/new ROUTEA/RC'
DEST = WS / 'github_exports/new_HYP_20260919'
MARKERS = ('new_hyp', 'original7', 'native7', 'romav2_colnomic', 'six_cause', 'h593',
           'h71', 'h70', 'pair_quality', 'mass_teacher', 'fold0_m_teacher', 'fold0_roma_m_teacher',
           'colpali', 'head_training_time', 'crisp_manual', 'matched_three_arm')
TEXT = {'.py', '.md', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.sh', '.sbatch', '.yaml', '.yml', '.toml', '.html', '.css', '.js', '.rst'}
FIG = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.pptx', '.docx', '.xlsx'}
PRUNE = {'__pycache__', '.git', '.pytest_cache', 'logs', 'cache', 'images', 'worker_assets',
         'reference_images', 'query_images', 'gallery_images', 'raw_images', 'node_modules',
         'data_intake', 'features', 'tokens', 'roma', 'raw', 'inputs', 'queries', 'pairs',
         'sam3_crops_topk', 'sam3_mask_cache'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
selected, excluded = {}, []
modules, by_name = {}, {}


def match(p): return any(x in p.name.lower() for x in MARKERS)
def digest(data): return hashlib.sha256(data).hexdigest()
def dump(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def add(p, reason, figure=False):
    if not p.is_file() or p.is_symlink() or p in selected: return
    relative = p.relative_to(WS)
    if p.suffix.lower() not in TEXT | (FIG if figure else set()): return
    if any(x in str(relative).lower() for x in ('/results/rc_opened_', '/results/d1_mi', '/results/d1-mi', '/results/formal392', '/results/rc_lth_p_only_gisc_prerecall_universe')): return
    if p.name.lower() in ('stdout.txt','stderr.txt','auth.json','hosts.yml'):return
    if p.stat().st_size > 50 * 1024**2:
        excluded.append({'path': str(relative), 'reason': 'over_50MiB', 'bytes':p.stat().st_size});return
    data = p.read_bytes()
    if p.suffix.lower() in TEXT and SECRET.search(data): raise RuntimeError('SECRET_SCAN_BLOCK:' + str(relative))
    # Copy exactly the bytes hashed, even if a mutable live status changes later.
    dst = DEST / relative
    old = dst.read_bytes() if dst.exists() else None
    if old != data:
        dst.parent.mkdir(parents=True, exist_ok=True);dst.write_bytes(data)
        dst.chmod(0o755 if p.suffix in ('.sh','.sbatch') else 0o644)
    selected[p] = {'path': str(relative), 'sha256': digest(data), 'bytes': len(data), 'reason': reason, 'changed': old != data}


def tree(folder, reason, figures=False):
    for base, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d not in PRUNE and not (Path(base)/d).is_symlink()]
        for name in files:add(Path(base)/name, reason, figures)


def main():
    areas = ('programs','src','tests','validators','registry','protocols','plan','slurm','scripts')
    for area in areas:
        for p in (RC/area).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            by_name.setdefault(p.name, []).append(p)
            if p.suffix == '.py':modules[p.stem] = p
            if match(p):add(p,'mainline_source_or_protocol')
    # Preserve and refresh existing exported source paths, without deleting remote history.
    for rel in subprocess.check_output(['git','ls-files','-z'],cwd=DEST).decode().split('\0'):
        if not rel.startswith('ICLR/new ROUTEA/RC/'):continue
        p=WS/rel
        if p.is_file() and '/results/' not in rel and not any(x in p.parts for x in PRUNE):
            add(p,'previously_published_source', '/reports/' in rel)
    for p in (RC/'reports').iterdir():
        if match(p):
            if p.name.startswith('sam3_'):continue
            if p.is_dir() and '_build_' in p.name:continue
            if p.is_dir():tree(p,'report_package',not p.name.startswith('sam3_'))
            else:add(p,'report',True)
    for p in (RC/'reports/figures').iterdir():
        if p.is_dir() and match(p):tree(p,'presentation',True)
    inventory=[]
    for folder in sorted((RC/'results').iterdir()):
        if not folder.is_dir() or not match(folder):continue
        final=[]
        for p in folder.iterdir():
            if p.is_file():
                # Candidate/feature intermediate tables are excluded; final result.json is retained.
                if p.name in ('payload.json','predictions.json','prejoin.json','train_inputs.json','held_inputs.json','catalog.json','ready.json'):continue
                if any(x in p.name for x in ('feature_records','raw_aggregate','complete_seed_scores','prejoin_records')):continue
                add(p,'result_or_execution_record')
                if p.name in ('result.json','validation.json','result_validation.json','report_zh.md'):final.append(p.name)
            elif p.is_dir() and (re.fullmatch(r'(?:fit|fold)\d+',p.name) or p.name in ('metadata','target_join','dispatch','train','held','teacher','M_ONLY','COST1','CE','COST4','prediction_seal')):
                tree(p,'fold_result_or_progress')
        status=None
        if (folder/'validation.json').is_file():
            try:status=json.loads((folder/'validation.json').read_text()).get('status')
            except (ValueError,AttributeError):pass
        inventory.append({'family':folder.name,'root_files':final,'validation_status':status,
                          'note':'Validation status is preserved verbatim; absence is not a scientific failure.'})
    # Local Python imports and explicitly named helper programs/authorities.
    scanned=set()
    while True:
        pending=[p for p in selected if p not in scanned and p.suffix in ('.py','.sh','.sbatch','.json','.md')]
        if not pending:break
        for p in pending:
            scanned.add(p)
            text=(DEST/p.relative_to(WS)).read_text(errors='replace')
            names=set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh)',text))
            if p.suffix=='.py':
                try:t=ast.parse(text)
                except SyntaxError:t=None
                if t:
                    for node in ast.walk(t):
                        if isinstance(node,ast.Import):names.update(x.name.split('.')[-1]+'.py' for x in node.names)
                        elif isinstance(node,ast.ImportFrom):
                            if node.module:names.add(node.module.split('.')[-1]+'.py')
            for name in names:
                for dep in by_name.get(name,[]):add(dep,'local_source_dependency')
    copied=sorted(selected.values(),key=lambda d:d['path'])
    dump(DEST/'backup/experiment_sync_20260923/files.json',copied)
    dump(DEST/'backup/experiment_sync_20260923/excluded_oversize.json',excluded)
    dump(DEST/'backup/experiment_sync_20260923/experiment_index.json',inventory)
    summary={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'files':len(copied),'changed_files':sum(x['changed'] for x in copied),
             'changed_bytes':sum(x['bytes'] for x in copied if x['changed']),
             'experiment_families':len(inventory),'oversize_exclusions':len(excluded),
             'source_bytes_unmodified':True,'running_status_is_point_in_time':True,
             'excluded':['binary checkpoints and pretrained weights','feature/token caches','raw images','err/out/log files','protected result populations','duplicate archives'],
             'previous_release_assets_unchanged':True}
    dump(DEST/'backup/experiment_sync_20260923/summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
