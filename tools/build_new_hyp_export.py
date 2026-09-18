#!/usr/bin/env python3
"""Create a reviewable GitHub snapshot without loading models or experiment caches."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict

WS = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = WS / 'ICLR/new ROUTEA/RC'
OUT = WS / 'github_exports/new_HYP_20260919'
PACKAGE = RC / 'reports/new_hyp_complete_results_20260916_v1'
LIMIT = 50 * 1024**2
TEXT = {'.py', '.md', '.json', '.csv', '.tsv', '.txt', '.sh', '.sbatch', '.yaml', '.yml', '.toml', '.html', '.css', '.js', '.rst'}
PRESENTATION = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.pptx', '.docx', '.xlsx'}
BANNED_EXT = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl', '.pickle', '.npy', '.npz', '.h5', '.hdf5', '.onnx', '.gguf', '.pyc', '.out', '.err', '.log', '.zip', '.gz', '.tar', '.mp4', '.avi'}
PRUNE = {'__pycache__', '.git', '.pytest_cache', 'logs', 'cache', 'raw', 'roma', 'features', 'tokens', 'images', 'reference_images', 'query_images', 'gallery_images', 'raw_images', 'node_modules', 'data_intake'}
MARKERS = ('new_hyp', 'original7', 'native7', 'romav2_colnomic', 'six_cause', 'h593', 'head_training_time', 'crisp_manual', 'matched_three_arm')
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')
selected = {}
excluded = {}
dependencies = []
result_roots = set()
queue = []
scanned = set()
all_by_name = defaultdict(list)
program_modules = {}
local_modules = {}
source_sha = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export_relative(path):
    rel=path.relative_to(WS)
    if rel.parts[0].startswith('.venv') and 'site-packages' in rel.parts:
        start=rel.parts.index('site-packages')+1
        return Path('third_party/runtime_source_snapshots').joinpath(*rel.parts[start:])
    return rel


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def protected(path):
    # Preserve public code/contracts, never collect protected scientific payloads.
    s = str(path).lower()
    return '/results/' in s and any(x in s for x in ('d1-mi', 'd1_mi', 'formal392', 'gisc_prerecall_universe', 'rc_opened_'))


def add(path, reason, presentation=False):
    path = Path(os.path.abspath(path))
    if path in selected or path in excluded:
        return
    if not path.is_relative_to(WS) or not path.is_file():
        return
    rel = str(path.relative_to(WS))
    why = None
    if path.is_symlink(): why = 'symlink_not_followed'
    elif protected(path): why = 'protected_result_scope'
    elif path.suffix.lower() in BANNED_EXT:
        why = 'stored_as_release_asset_not_git_blob' if path.is_relative_to(WS/'models/downloaded_models') and path.suffix=='.safetensors' else 'model_cache_log_or_duplicate_archive'
    elif path.name.lower() in ('stdout.txt', 'stderr.txt', 'auth.json', '.env'): why = 'private_runtime'
    elif path.stat().st_size > LIMIT: why = 'above_50_MiB_review_limit'
    elif path.suffix.lower() not in TEXT | (PRESENTATION if presentation else set()) and path.name not in ('LICENSE', 'LICENSE.txt', 'NOTICE', 'Makefile', '.gitignore'): why = 'not_selected_source_or_document_type'
    if why:
        excluded[path] = {'path': rel, 'reason': why, 'bytes': path.stat().st_size}
        return
    data = path.read_bytes()
    if path.suffix.lower() in TEXT and SECRET.search(data):
        raise RuntimeError('SECRET_REVIEW_REQUIRED:' + rel)
    # The user subsequently authorized both pretrained weights and fitted heads.
    # Small JSON heads are tracked here; large binary weights use release assets.
    selected[path] = reason
    source_sha[path] = sha(data)
    if path.suffix in {'.py', '.sh', '.sbatch', '.json', '.md'}:
        queue.append(path)


def tree(path, reason, presentation=False, prune=PRUNE):
    if not path.exists(): return
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in prune and not d.startswith('.venv') and not (Path(root)/d).is_symlink()]
        for name in files:
            add(Path(root)/name, reason, presentation)


def find_import(name):
    if name in local_modules: return local_modules[name]
    if name in program_modules: return program_modules[name]
    return None


def scan(path):
    text = path.read_text(errors='replace')
    if path.suffix == '.py':
        try: syntax = ast.parse(text)
        except SyntaxError: syntax = None
        if syntax:
            for node in ast.walk(syntax):
                names = []
                if isinstance(node, ast.Import): names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ''
                    if node.level and path.is_relative_to(RC/'src'):
                        package = str(path.relative_to(RC/'src').parent).replace('/', '.')
                        parts = package.split('.')
                        base = '.'.join(parts[:len(parts)-node.level+1] + ([base] if base else []))
                    names = [base] + [base+'.'+n.name for n in node.names]
                for name in names:
                    dep = find_import(name)
                    if dep is None and name and '.' not in name:
                        sibling=path.parent/(name+'.py')
                        if sibling.is_file():dep=sibling
                    if dep:
                        add(dep, 'local_import_dependency')
                        dependencies.append({'source':str(path.relative_to(WS)), 'import':name, 'dependency':str(dep.relative_to(WS))})
    if path.suffix=='.json':
        try:record=json.loads(text)
        except ValueError:record=None
        def explicit_sources(obj):
            if isinstance(obj,dict):
                p=obj.get('path')
                if isinstance(p,str):
                    dep=Path(p)
                    if dep.is_absolute() and dep.is_relative_to(WS) and dep.suffix in ('.py','.sh','.toml'):
                        add(dep,'explicit_external_source_dependency')
                for value in obj.values():explicit_sources(value)
            elif isinstance(obj,list):
                for value in obj:explicit_sources(value)
        explicit_sources(record)
    # Handles dynamic imports and runtime code extracted from source files.
    for name in set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh|json|md)', text)):
        for dep in all_by_name.get(name, []):
            if dep.suffix == '.py' or dep.parent.name in ('registry', 'protocols'):
                add(dep, 'referenced_source_or_contract')
    # Explicit current-lineage result directories; only small records are collected later.
    if selected[path] in ('indexed_theory_or_historical_evidence', 'complete_results_publication'):
        for name in re.findall(r'results/(rc_[A-Za-z0-9_]+|romav2_colnomic_[A-Za-z0-9_]+)', text):
            candidate = RC/'results'/name
            if candidate.is_dir() and not protected(candidate): result_roots.add(candidate)


def result_records(folder):
    # No cache traversal; no binary payloads; preserve final output and audit metadata.
    for p in folder.iterdir():
        if not p.is_file(): continue
        name=p.name.lower()
        cache_name = any(t in name for t in ('feature_records', 'raw_aggregate', 'prejoin_records', 'complete_score_vectors', 'complete_seed_scores', 'score_records', 'all_conditions_prejoin', 'all64_prejoin', 'eval_prejoin', 'train_prejoin', 'predictions_prejoin'))
        cache_name = cache_name or name in ('predictions.json', 'prejoin.json', 'payload.json')
        # A large prejoin file containing per-candidate records is an intermediate cache,
        # whereas small SHA/count seals remain useful provenance.
        cache_name = cache_name or ('prejoin' in name and p.stat().st_size > 2*1024**2)
        if cache_name:
            excluded[p]={'path':str(p.relative_to(WS)), 'reason':'historical_intermediate_json_cache', 'bytes':p.stat().st_size}
        else: add(p, 'result_or_validation_record')
    for sub in ('metadata', 'target_join'):
        p=folder/sub
        if p.is_dir():
            for f in p.iterdir():
                if f.is_file(): add(f, 'evaluation_manifest')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for area in ('programs', 'src', 'tests', 'validators', 'registry', 'protocols', 'plan', 'slurm', 'scripts'):
        for p in (RC/area).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts: continue
            all_by_name[p.name].append(p)
            if area=='programs' and p.suffix=='.py': program_modules[p.stem]=p
            if area=='src' and p.suffix=='.py':
                name=str(p.relative_to(RC/'src').with_suffix('')).replace('/','.')
                if name.endswith('.__init__'): name=name[:-9]
                local_modules[name]=p
            if any(m in p.name.lower() for m in MARKERS): add(p,'mainline_entrypoint_or_contract')
    tree(PACKAGE, 'complete_results_publication', True, prune={'__pycache__'})
    for row in csv.DictReader((PACKAGE/'report_index.csv').open(encoding='utf-8-sig')):
        add(RC/row['report'], 'indexed_theory_or_historical_evidence')
    for sub in ('new_hyp_showcase_20260915_v1','new_hyp_visibility_cases_20260915_v1','new_hyp_verified_outcome_matrices_v1','new_hyp_product_contrast_v1','new_hyp_train_support_capacity_v1','briefing_new_hyp_20260912_v1','new_hyp_grozi120_external_v1'):
        tree(RC/'reports/figures'/sub,'presentation_and_mechanism_figures',True,prune={'__pycache__','logs'})
    v=json.loads((PACKAGE/'validation.json').read_text())
    for check in v['checks']:
        if check.get('source'):
            source=RC/check['source']
            add(source,'published_result_source')
            if source.is_relative_to(RC/'results'):
                result_roots.add(RC/'results'/source.relative_to(RC/'results').parts[0])
                result_roots.add(source.parent)
    for p in (RC/'results').iterdir():
        if p.is_dir() and any(m in p.name for m in MARKERS): result_roots.add(p)
    # External fixed heads and their provenance.
    bundle=RC/'results/rc_new_hyp_external_head_freeze_v1/bundle.json'
    add(bundle,'fixed_head_provenance')
    for model in ('COST1','CE','COST4','GROUP_COST4','RAW2_CE'):
        add(bundle.parent/model/'head.json','frozen_fitted_head')
        add(bundle.parent/model/'validation.json','frozen_head_validation')
    frozen=RC/'isolated/romav2_colnomic_candidate_bound_no_regret_v1_20260901/bundle_manifest.json'
    add(frozen,'backbone_fingerprint_without_weights')
    profile=json.loads((RC/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json').read_text())
    for family, record in profile['source_trees'].items():
        for item in record['python_files']:
            p=Path(item['path'])
            if p.is_file() and sha(p.read_bytes())!=item['sha256']: raise RuntimeError('PINNED_THIRD_PARTY_SOURCE_DRIFT:'+str(p))
            add(p,'pinned_third_party_source')
        source_root=Path(record['root'])
        roots=[source_root, WS/'third_party/RoMaV2'] if family=='roma' else [source_root]
        for base in roots:
            for name in ('LICENSE','LICENSE.txt','LICENSE.md','NOTICE','README.md','pyproject.toml','setup.py','requirements.txt','hubconf.py'):
                add(base/name,'third_party_license_and_build_metadata')
    processed_roots=set()
    while queue or result_roots-processed_roots:
        while queue:
            p=queue.pop()
            if p in scanned:continue
            scanned.add(p);scan(p)
        for folder in sorted(result_roots-processed_roots):
            processed_roots.add(folder);result_records(folder)
    # Include package initializers to retain the original import tree.
    for p in list(selected):
        if p.is_relative_to(RC/'src'):
            for parent in p.parents:
                if parent==RC/'src': break
                add(parent/'__init__.py','package_initializer')
    # Small model configuration/tokenizer files travel with the source checkout.
    # .gitattributes from a model download is intentionally omitted: no implicit LFS.
    for name in ('colnomic-embed-multimodal-7b','colqwen2.5-7B-base'):
        model_dir=WS/'models/downloaded_models'/name
        for p in model_dir.iterdir():
            if p.name!='.gitattributes':add(p,'model_configuration_and_tokenizer')
    # Rebuilding may remove only unchanged files owned by our previous export manifest.
    old_manifest=OUT/'backup/copied_files.json'
    if old_manifest.exists():
        for record in json.loads(old_manifest.read_text()):
            if WS/record.get('source_path',record['path']) in selected:continue
            dst=OUT/record['path']
            if dst.is_file():
                if sha(dst.read_bytes())!=record['sha256']:raise RuntimeError('STAGED_FILE_EDITED:'+record['path'])
                dst.unlink()
    copied=[]
    for p,reason in sorted(selected.items()):
        rel=export_relative(p);dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        data=p.read_bytes()
        if sha(data)!=source_sha[p]: raise RuntimeError('SOURCE_CHANGED:'+str(rel))
        dst.write_bytes(data);dst.chmod(0o755 if p.suffix in ('.sh','.sbatch') else 0o644)
        copied.append({'path':str(rel),'source_path':str(p.relative_to(WS)),'sha256':sha(data),'bytes':len(data),'reason':reason})
    write_json(OUT/'backup/copied_files.json',copied)
    write_json(OUT/'backup/excluded_files.json',list(excluded.values()))
    write_json(OUT/'backup/local_import_dependencies.json',dependencies)
    write_json(OUT/'backup/scope.json',{
        'source_workspace':str(WS),'destination_repository':'https://github.com/yinaiden6-dev/new_HYP',
        'snapshot_date':'2026-09-19','copied_files':len(copied),'copied_bytes':sum(x['bytes'] for x in copied),
        'result_families':sorted(str(x.relative_to(WS)) for x in processed_roots),
        'excluded_categories':['feature/token/intermediate caches','err/out/log files','original query/reference images','duplicate zip archives','virtual environments','protected scientific results'],
        'large_model_storage':'same private repository, versioned GitHub Release assets with split/whole SHA256 manifests',
        'source_code_modified':False,'new_training':False,'new_inference':False,
        'execution_note':'Historical absolute paths, source hashes, deadlines and Slurm guards are preserved. This is an archival snapshot, not a tested portable installation.'})
    manifest=json.loads(frozen.read_text())
    head_bundle=json.loads(bundle.read_text())
    model_info={'weights_storage':'GitHub Release; completion is recorded in backup/publication_receipt.json after remote verification','frozen_model_fingerprints':manifest['models'],
                'fixed_heads':head_bundle['heads'],'primary':'COST1','secondary':'CE',
                'fingerprints_source':str(frozen.relative_to(WS)),
                'head_manifest_source':str(bundle.relative_to(WS)),
                'hash_note':'Copied from existing sealed provenance; large weight files were not rehashed during export.'}
    write_json(OUT/'MODEL_DEPENDENCIES.json',model_info)
    with (OUT/'backup/raw_image_manifests.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['manifest','image_count','path_fields','note']);writer.writeheader()
        for p in selected:
            if p.suffix!='.json' or p.stat().st_size>10*1024**2:continue
            try:d=json.loads(p.read_text())
            except ValueError:continue
            rows=d.get('records',[]) if isinstance(d,dict) else []
            if not rows or not isinstance(rows,list):continue
            fields=sorted({k for r in rows if isinstance(r,dict) for k in r if k in ('query_image_path','image_path','source_image_path','reference_image_path')})
            if fields:writer.writerow({'manifest':str(p.relative_to(WS)),'image_count':len(rows),'path_fields':';'.join(fields),'note':'Original pixels omitted; preserve these paths and source hashes in a separate data backup.'})
    print(json.dumps({'stage':str(OUT),'copied_files':len(copied),'MiB':round(sum(x['bytes'] for x in copied)/1024**2,2),'result_families':len(processed_roots),'excluded':len(excluded)},ensure_ascii=False))


if __name__=='__main__':main()
