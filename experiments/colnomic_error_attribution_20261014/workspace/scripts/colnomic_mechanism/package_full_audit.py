"""Inventory, freeze and independently validate the complete ColNomic audit.

Nothing is sealed without --ready. Archives retain workspace-relative paths.
Model weights and credentials are deliberately external; frozen banks/scores are
included. Verification reads actual members, checks CRC32/SHA256, and can
replay the 1117-query/35-configuration axes after extraction. Tar hardlinks are
strictly archive-relative and target an earlier regular member in this archive.
"""
import argparse,binascii,csv,datetime,gzip,hashlib,html.parser,importlib.metadata,io,json,os,re,struct,subprocess,sys,tarfile,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[2]
RESUME=Path('results/colnomic_full_audit/20261013_resume')
SUPPORT=RESUME/'delivery_support'
MANIFEST=SUPPORT/'archive_manifest.json'
ARCHIVE_SUFFIXES=('.zip','.tar.gz','.tar.zst','.tgz','.7z','.rar')
PRIVATE_PARTS={'.git','.aws','.codex','.agents','.ssh','.secrets','credentials','__pycache__'}
PRECOMPRESSED={'.gz','.npz','.jpg','.jpeg','.png','.pdf','.webp','.woff','.woff2'}
# This literal link is already incorrect in the original historical report.
# Preserve that report byte-for-byte and add an explicitly declared, same-SHA
# archive-only compatibility member at the path resolved by the old link.
HISTORICAL_LINK_DEFECTS={
    ('results/method/REPORT.md','../cache/intermediate/audits/evaluation_retry_20260925.json'):
        'cache/method/intermediate/audits/evaluation_retry_20260925.json'}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def csvrows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def ordinary_archive_mode(stat):
    """Readable regular payload, preserving only ordinary source execute bits."""
    return 0o644 | (stat.st_mode & 0o111)


def deduplicate_records(records):
    """A shared inode must never merge different executable permissions."""
    canonical={};aliases={}
    for row in records:
        key=(row['bytes'],row['sha256'],row.get('archive_mode',0o644))
        if key in canonical:aliases[row['path']]=canonical[key]
        else:canonical[key]=row['path']
    return aliases


def exclusion(relative):
    p=PurePosixPath(relative);name=p.name.lower()
    if p.is_absolute() or '..' in p.parts:return 'unsafe_archive_path'
    if any(part.lower() in PRIVATE_PARTS for part in p.parts):return 'credentials_or_repository_private_state'
    if any(part.lower()=='weights' for part in p.parts) or (p.parts and p.parts[0] in ['other_datasets','backup']):return 'external_model_or_unrelated_data'
    if name.startswith('.env') or name in ['credentials.json','credentials.ini','id_rsa','id_ed25519'] or p.suffix.lower() in ['.pem','.key','.p12','.pfx']:
        return 'credential_file'
    if name.endswith(ARCHIVE_SUFFIXES):return 'existing_archive_not_nested'
    if p.suffix.lower() in ['.tmp','.pyc','.lock'] or name.endswith('.part'):return 'transient_file'
    if p.suffix.lower()=='.safetensors' or name.startswith('pytorch_model') or name.startswith('model-000'):
        return 'external_encoder_weight'
    return None


def cohort(root):
    history=read(root/'results/scene_statistics/manifest.json')
    directories={str(Path(path).parent) for path in history['inputs'] if path.endswith('predictions.jsonl')}
    configs=[row for row in csvrows(root/'results/all_results.csv') if row['result'] in directories]
    assert len(configs)==35 and {r['result'] for r in configs}==directories
    return configs


def score_path(config,spec):
    if spec.get('score_path'):return spec['score_path']
    if spec.get('region_manifest'):return str(Path(spec['region_manifest']).parent/'scores.pt')
    mode=config['mode'];stem=mode+'_alpha_'+f"{spec['alpha']:g}" if mode in ['fusion','ocr_fusion'] else mode
    return str(Path('cache/scores')/config['model']/(stem+'.pt'))


def inventory(root,out):
    selected={};excluded=[];missing=[]
    def add(relative,category):
        path=root/relative
        if not path.exists():missing.append(dict(path=str(relative),category=category));return
        if path.is_dir():
            for file in sorted(path.rglob('*')):
                if file.is_file():add(file.relative_to(root),category)
            return
        name=str(Path(relative).as_posix());reason=exclusion(name)
        if reason:excluded.append(dict(path=name,reason=reason,bytes=path.stat().st_size));return
        if not path.resolve().is_relative_to(root):excluded.append(dict(path=name,reason='external_symlink',bytes=path.stat().st_size));return
        resolved_reason=exclusion(str(path.resolve().relative_to(root)))
        if resolved_reason:excluded.append(dict(path=name,reason='resolved_target_'+resolved_reason,bytes=path.stat().st_size));return
        if Path(name).is_relative_to(SUPPORT):return
        stat=path.stat();selected.setdefault(name,dict(path=name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,archive_mode=ordinary_archive_mode(stat),category=category))
    seeds=[(RESUME,'current_full_audit'),
           (Path('results/token_mechanism_experiments/20261013_v1'),'historical_four_stage_v1'),
           (Path('results/token_mechanism_experiments/20261013_v2'),'historical_four_stage_v2'),
           (Path('results/token_mechanism_analysis/B03'),'historical_B03_full_matrices'),
           (Path('results/token_mechanism_analysis/F03'),'historical_F03_full_matrices'),
           (Path('results/scene_statistics'),'historical_scene35'),
           (Path('cache/image_embeddings/colnomic7b'),'frozen_B03_raw_token_banks'),
           (Path('cache/image_embeddings/metadata.json'),'embedding_provenance'),
           (Path('cache/ocr'),'frozen_OCR'),(Path('dataset/lists'),'dataset_inventory'),
           (Path('dataset/annotations'),'current_annotations'),(Path('dataset/tools'),'annotation_code'),
           (Path('dataset/references/metadata.json'),'reference_provenance')]
    for directory in ['scripts','evaluation','drugpack','configs','tests']:seeds.append((Path(directory),'code_and_configuration'))
    for name in ['README.md','requirements.txt','pyproject.toml','LICENSE','CITATION.cff']:
        if (root/name).exists():seeds.append((Path(name),'repository_environment_and_documentation'))
    for path,category in seeds:add(path,category)
    photos=csvrows(root/'dataset/lists/photos.csv');refs=csvrows(root/'dataset/lists/references.csv')
    assert len(photos)==1117 and len(refs)==5405
    for row in photos:add(Path('dataset')/row['path'],'original_1117_query_images')
    for row in refs:add(Path('dataset')/row['path'],'original_5405_gallery_images')
    for path in (root/'dataset/packages').rglob('metadata.json'):add(path.relative_to(root),'original_scene_sources')
    configs=cohort(root);config_paths=[]
    for cfg in configs:
        folder=Path(cfg['result']);add(folder,'original35_predictions_and_recipes')
        spec=read(root/folder/'config.json');path=score_path(cfg,spec);add(Path(path),'original35_actual_scores')
        config_paths.append(dict(config_id=cfg['id'],result=str(folder),score=path,config=str(folder/'config.json'),predictions=str(folder/'predictions.jsonl')))
        if spec.get('region_manifest'):add(Path(spec['region_manifest']),'baseline35_region_provenance')
    for path in ['cache/scores/colnomic7b/image.pt','cache/scores/bm25/bm25.pt','cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt','results/all_results.csv']:
        add(Path(path),'primary_baseline_dependency')
    # Keep linked repository documentation and its small concrete receipts and
    # figures portable. This does not harvest unrelated caches or model weights.
    linked_dependencies=[];compatibility_aliases=[];linked_seen=set();linked_todo=[n for n in selected if n.endswith('.md')]
    while linked_todo:
        name=linked_todo.pop()
        if name in linked_seen:continue
        linked_seen.add(name)
        text=(root/name).read_text(errors='replace')
        targets=[v.strip().split()[0].strip('<>') for v in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',text) if v.strip()]
        for target in targets:
            url=urlsplit(target)
            if url.scheme or url.netloc or not url.path or url.path.startswith('/') or '${' in target or '{{' in target:continue
            candidate=((root/name).parent/unquote(url.path)).resolve()
            if not candidate.is_relative_to(root):
                linked_dependencies.append(dict(source=name,target=target,status='outside_workspace_provenance'));continue
            relative=str(candidate.relative_to(root));reason=exclusion(relative)
            if reason:
                linked_dependencies.append(dict(source=name,target=target,path=relative,status='excluded_provenance',reason=reason));continue
            if candidate.is_dir() or relative in selected:continue
            if not candidate.is_file():
                correct=HISTORICAL_LINK_DEFECTS.get((name,target))
                row=dict(source=name,target=target,path=relative,status='original_source_link_missing',actual_source=correct)
                linked_dependencies.append(row)
                if correct:
                    add(Path(correct),'linked_historical_record_with_original_link_defect')
                    actual=root/correct
                    if actual.is_file():
                        compatibility_aliases.append(dict(path=relative,compatibility_alias_of=correct,bytes=actual.stat().st_size,sha256=digest(actual),
                            original_link_source=name,original_link_target=target,original_workspace_target_missing=True,
                            creation_scope='archive_virtual_member_only; original workspace is unchanged'))
                continue
            if candidate.stat().st_size>50*2**20:
                linked_dependencies.append(dict(source=name,target=target,path=relative,status='large_linked_dependency_requires_review',bytes=candidate.stat().st_size));continue
            add(Path(relative),'linked_repository_documentation_and_protocol')
            if relative in selected and relative.endswith('.md'):linked_todo.append(relative)
    # Exact declared inputs may lie outside those broad trees; include safe files
    # referenced by current/historical manifests, without harvesting all caches.
    declared=[]
    for name in list(selected):
        path=root/name
        if path.suffix!='.json' or path.stat().st_size>25*2**20 or not any(part in path.name for part in ['manifest','protocol','sources','runtime','provenance']):continue
        try:value=read(path)
        except (ValueError,UnicodeError):continue
        def walk(value):
            if isinstance(value,dict):
                for k,v in value.items():yield k;yield from walk(v)
            elif isinstance(value,list):
                for item in value:yield from walk(item)
            elif isinstance(value,str):yield value
        for text in walk(value):
            if text.startswith(str(root)+'/'):text=text[len(str(root))+1:]
            if not text.startswith(('cache/','dataset/','results/','scripts/','evaluation/','configs/','drugpack/','weights/','backup/')):continue
            if '{' in text or '}' in text:
                declared.append(dict(source=name,path=text,status='template_provenance_not_concrete_path'));continue
            candidate=root/text
            if candidate.is_file():
                reason=exclusion(text)
                if reason:declared.append(dict(source=name,path=text,status='excluded_external_or_archive',reason=reason))
                else:add(Path(text),'explicit_manifest_dependency')
            elif not candidate.exists():declared.append(dict(source=name,path=text,status='missing_declared_path'))
    files=sorted(selected.values(),key=lambda r:r['path'])
    classification=Counter()
    for row in declared:
        if row['status']=='missing_declared_path' and '/sam3_auto/' in row['path'] and '/intermediate/regions/' in row['path']:
            row['scope']='historical_SAM3_preprocessing_provenance';row['frozen_score_replay_blocked']=False
        elif row['status']=='excluded_external_or_archive':row['scope']='external_encoding_or_deployment_dependency'
        elif row['status']=='template_provenance_not_concrete_path':row['scope']='configuration_path_template'
        else:row['scope']='declared_dependency_requires_review'
        classification[row['scope']]+=1
    categories=defaultdict(lambda:dict(files=0,bytes=0))
    for row in files:categories[row['category']]['files']+=1;categories[row['category']]['bytes']+=row['bytes']
    pre=sum(r['bytes'] for r in files if Path(r['path']).suffix.lower() in PRECOMPRESSED);raw=sum(r['bytes'] for r in files)
    report=dict(schema='colnomic.full_archive.inventory.v1',created_utc=now(),workspace_root=str(root),
        files=files,file_count=len(files),uncompressed_bytes=raw,category_totals=dict(categories),
        compression_estimate_bytes=[int(pre+.60*(raw-pre)),int(pre+.90*(raw-pre))],compression_estimate_is_unmeasured=True,
        query_images=1117,gallery_images=5405,unique_gallery_SHA256=len({r['sha256'] for r in refs}),
        configurations35=config_paths,missing_required_paths=missing,declared_missing_or_external_paths=declared,
        linked_documentation_dependencies=linked_dependencies,
        archive_compatibility_aliases=compatibility_aliases,
        declared_dependency_scope_counts=dict(classification),
        excluded_files=excluded,scope='All current resume modules are discovered at final inventory, including pending cross-token and natural shared-phrase modules.',
        external_encoding_dependencies='Large encoder/backbone/adapter weights, historical software runtimes and absent original-gallery upstream checkpoints remain external. Frozen B03 token banks and35 actual scalar score caches are included; no claim of portable bitwise historical encoder reproduction.',
        archive_layout='workspace-relative paths; self-contained tar.gz with verified internal hardlinks or independent ZIP64 volumes',
        ready_to_seal=False)
    write(out/'inventory.json',report)
    gallery_note=RESUME/'split_geometry/cache_provenance.md'
    upstream_note=dict(source=str(gallery_note),included=(root/gallery_note).is_file(),scope='historical_gallery_encoding_provenance',
        description='The included provenance note documents the absent original canonical-gallery upstream checkpoint and historical runtime uncertainty. Existing raw gallery Token bank is included for frozen replay.')
    write(out/'dependency_gaps.json',dict(required=missing,declared_missing_or_external=declared,scope_counts=dict(classification),
        linked_documentation_dependencies=linked_dependencies,
        archive_compatibility_aliases=compatibility_aliases,
        historical_gallery_encoder=upstream_note,frozen_35_score_paths_included=len(config_paths),
        external_encoding_dependencies=report['external_encoding_dependencies']))
    print(json.dumps(dict(files=len(files),uncompressed_bytes=raw,compression_estimate=report['compression_estimate_bytes'],missing_required=len(missing),declared_missing_or_external=len(declared))),flush=True)
    return report


def audit_duplicates(root,out,plan=None):
    """Read real bytes for every repeated-size candidate; no digest inference."""
    plan=plan or read(out/'inventory.json');sizes=defaultdict(list)
    stale=[]
    for row in plan['files']:
        path=root/row['path']
        if not path.is_file() or (path.stat().st_size,path.stat().st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or ordinary_archive_mode(path.stat())!=row.get('archive_mode',0o644):stale.append(row['path']);continue
        sizes[row['bytes']].append(row)
    groups=[];hashes={};reads=0;read_bytes=0;candidate_files=0;hashed=[]
    for size,rows in sorted(sizes.items(),reverse=True):
        if len(rows)<2:continue
        by_sha=defaultdict(list)
        for row in rows:
            path=root/row['path'];stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in hashes:
                hashes[key]=digest(path);reads+=1;read_bytes+=size
            assert (path.stat().st_size,path.stat().st_mtime_ns)==(size,row['mtime_ns'])
            mode=ordinary_archive_mode(stat);sha=hashes[key];by_sha[(sha,mode)].append(row['path']);hashed.append(dict(path=row['path'],bytes=size,mtime_ns=row['mtime_ns'],archive_mode=mode,sha256=sha));candidate_files+=1
        for (sha,mode),paths in by_sha.items():
            if len(paths)>1:groups.append(dict(bytes=size,sha256=sha,archive_mode=mode,paths=sorted(paths),copies=len(paths),saved_payload_bytes=size*(len(paths)-1)))
        if candidate_files%500< len(rows):print('Duplicate SHA candidates',candidate_files,'actual bytes read',read_bytes,flush=True)
    groups.sort(key=lambda r:(-r['saved_payload_bytes'],r['paths'][0]))
    saved=sum(r['saved_payload_bytes'] for r in groups);logical=sum(r['bytes'] for r in plan['files'])
    alias_paths={p for row in groups for p in row['paths'][1:]}
    unique=[r for rows in sizes.values() for r in rows if r['path'] not in alias_paths]
    unique_bytes=sum(r['bytes'] for r in unique);pre=sum(r['bytes'] for r in unique if Path(r['path']).suffix.lower() in PRECOMPRESSED)
    report=dict(schema='colnomic.full_archive.dedup.v1',created_utc=now(),inventory_SHA256=digest(out/'inventory.json'),
        logical_file_count=len(plan['files']),logical_bytes=logical,duplicate_payload_bytes=saved,
        unique_payload_bytes=logical-saved,duplicate_groups=len(groups),duplicate_paths=sum(r['copies']-1 for r in groups),
        actual_SHA_candidate_files=candidate_files,actual_physical_files_read=reads,actual_bytes_read=read_bytes,
        single_size_files_provably_unique_without_hash=sum(len(r) for r in sizes.values() if len(r)==1),
        stale_inventory_entries=stale,groups=groups,
        eligible_current_logical_bytes=sum(r['bytes'] for rows in sizes.values() for r in rows),
        eligible_unique_payload_bytes=unique_bytes,
        unmeasured_tar_gzip_size_estimate_bytes=[int(pre+.60*(unique_bytes-pre))+len(plan['files'])*1024,int(pre+.90*(unique_bytes-pre))+len(plan['files'])*2048],
        final_snapshot_note='Preliminary inventory only. Final build independently hashes every current source, guards size/mtime and rechecks SHA after archive creation.',
        tar_safety='Each duplicate is an internal archive-relative hardlink to an earlier regular payload. No original workspace links, no symlinks, no link chains.')
    write(out/'duplicate_content_audit.json',report);write(out/'duplicate_candidate_SHA256.json',hashed)
    print(json.dumps({k:report[k] for k in ['logical_bytes','duplicate_payload_bytes','unique_payload_bytes','duplicate_groups','duplicate_paths','actual_bytes_read','stale_inventory_entries']}),flush=True)
    return report


class Links(html.parser.HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value:self.links.append(value)


def verify_links(root,member_paths,compatibility_aliases=()):
    members=set(member_paths);rows=[];broken=[];historical=[];external=0;absolute=0;dynamic=0
    compatibility={row['path']:row for row in compatibility_aliases}
    for name,row in compatibility.items():
        safe_member(name);safe_member(row['compatibility_alias_of'])
        assert row['compatibility_alias_of'] in members,'Archive compatibility source must be included'
        members.add(name)
    directories={str(parent) for name in members for parent in PurePosixPath(name).parents}
    for name in sorted(members):
        if name.endswith(('.html','.htm')):
            text=(root/name).read_text(errors='replace');parser=Links();parser.feed(text)
            targets=parser.links+re.findall(r"fetch\(\s*['\"]([^'\"]+)['\"]",text)
        elif name.endswith('.md'):
            text=(root/name).read_text(errors='replace')
            targets=[match.strip().split()[0].strip('<>') for match in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',text) if match.strip()]
        elif name.endswith('/dossier.json') and '/delivery/cases/' in name:
            value=read(root/name);targets=list(value.get('links',{}).values())
        else:continue
        for target in targets:
            url=urlsplit(target)
            if url.scheme or url.netloc:external+=1;continue
            if not url.path:continue
            if '${' in target or '{{' in target:dynamic+=1;continue
            if url.path.startswith('/'):
                absolute+=1;rows.append(dict(source=name,target=target,status='legacy_absolute_provenance_not_portable'));continue
            resolved=((root/name).parent/unquote(url.path)).resolve()
            if not resolved.is_relative_to(root):broken.append(dict(source=name,target=target,reason='escapes_archive_root'));continue
            relative=str(resolved.relative_to(root))
            exists=relative in members or relative in directories
            row=dict(source=name,target=target,resolved=relative,status='passed' if exists else 'missing_relative_target')
            if exists and relative in compatibility:
                alias=compatibility[relative]
                row.update(status='passed_via_archive_compatibility_member',compatibility_alias_of=alias['compatibility_alias_of'],
                    original_workspace_target_missing=alias['original_workspace_target_missing'],target_exists_in_checked_root=resolved.is_file())
                historical.append(row)
            rows.append(row)
            if not exists:broken.append(row)
    return dict(status='passed' if not broken else 'relative_links_missing',checked_relative_links=sum(r['status'].startswith('passed') for r in rows),
        broken_relative_links=broken,legacy_absolute_links=absolute,external_links_not_fetched=external,dynamic_links_not_evaluated=dynamic,
        repaired_original_source_links=historical,
        rows=rows,scope='Actual static HTML href/src/fetch, Markdown links, and each delivery/cases/{qid}/dossier.json links, resolved from its directory. Absolute historical filesystem links are explicitly provenance, not portable links. Declared archive-only same-SHA compatibility members repair original historical report links without altering the original files; extracted verification checks their actual bytes and resolved links.')


def environment():
    packages=sorted({(d.metadata.get('Name') or 'unknown',d.version) for d in importlib.metadata.distributions()})
    return dict(captured_utc=now(),python=sys.version,platform=sys.platform,
        distributions=[dict(name=name,version=version) for name,version in packages],
        environment_variables_captured=False,credentials_captured=False,model_weights_included=False)


def build(root,out,ready,volume_bytes,archive_format='tar-hardlinks',extract_validate=True):
    assert ready,'Wait for final producer readiness, then pass --ready; inventory does not authorize sealing unfinished evidence.'
    plan=inventory(root,out);assert not plan['missing_required_paths'],plan['missing_required_paths']
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=out/('archive_'+stamp);folder.mkdir(exist_ok=False)
    records=[];started=time.monotonic();inode_hashes={}
    for i,row in enumerate(plan['files'],1):
        path=root/row['path'];stat=path.stat();assert (stat.st_size,stat.st_mtime_ns)==(row['bytes'],row['mtime_ns'])
        assert ordinary_archive_mode(stat)==row['archive_mode'],row['path']
        key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in inode_hashes:inode_hashes[key]=digest(path)
        records.append({**row,'sha256':inode_hashes[key]})
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(row['bytes'],row['mtime_ns'])
        if i%500==0:print('Archive source SHA',i,'/',len(plan['files']),'elapsed',time.monotonic()-started,flush=True)
    groups=[];current=[];size=0
    for row in records:
        if current and size+row['bytes']>volume_bytes:groups.append(current);current=[];size=0
        current.append(row);size+=row['bytes']
    if current:groups.append(current)
    aliases=deduplicate_records(records)
    if archive_format=='tar-hardlinks':groups=[records]
    env=json.dumps(environment(),ensure_ascii=False,indent=2).encode()
    envpath=str(SUPPORT/'package_environment.json')
    virtual={envpath:env}
    for row in plan['archive_compatibility_aliases']:
        data=(root/row['compatibility_alias_of']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        assert row['path'] not in {r['path'] for r in records}
        virtual[row['path']]=data
    for support_name in ['dependency_gaps.json','duplicate_content_audit.json','axes_entrypoint_preflight.json','self_test_validation.json','package_readme.txt']:
        path=out/support_name
        if path.is_file():virtual[str(SUPPORT/support_name)]=path.read_bytes()
    manifest=dict(schema='colnomic.full_archive.members.v1',created_utc=now(),producer_ready_confirmed=True,
        source_workspace_root=str(root),files=records,virtual_files=[dict(path=name,bytes=len(data),archive_mode=0o644,sha256=hashlib.sha256(data).hexdigest(),
            **({'compatibility_alias_of':next(r['compatibility_alias_of'] for r in plan['archive_compatibility_aliases'] if r['path']==name)}
                if any(r['path']==name for r in plan['archive_compatibility_aliases']) else {})) for name,data in virtual.items()],
        archive_format=archive_format,
        volumes=[dict(name=('colnomic_full_audit.tar.gz' if archive_format=='tar-hardlinks' else f'colnomic_full_audit.part{i:03d}.zip'),
            paths=[r['path'] for r in group],link_targets=aliases if archive_format=='tar-hardlinks' else {}) for i,group in enumerate(groups,1)],
        query_ids=[r['photo_id'] for r in csvrows(root/'dataset/lists/photos.csv')],
        reference_ids=[r['reference_id'] for r in csvrows(root/'dataset/lists/references.csv')],
        configurations35=plan['configurations35'],excluded=plan['excluded_files'],dependency_gaps=plan['declared_missing_or_external_paths'],
        linked_documentation_dependencies=plan['linked_documentation_dependencies'],
        archive_compatibility_aliases=plan['archive_compatibility_aliases'],
        external_encoding_dependencies=plan['external_encoding_dependencies'],
        replay='After extracting every volume into one directory: python scripts/colnomic_mechanism/package_full_audit.py --verify-extracted --root . --axes --links',
        historical_absolute_paths='Preserved provenance; may require explicit rebasing in original analysis scripts. Not evidence of portable historical encoder reproducibility.',
        deduplication=dict(logical_bytes=sum(r['bytes'] for r in records),unique_payload_bytes=sum(r['bytes'] for r in records if r['path'] not in aliases),
            internal_hardlink_members=len(aliases) if archive_format=='tar-hardlinks' else 0,
            safe_link_policy='Only archive-relative links to earlier regular members with identical byte count, SHA256 and ordinary executable permissions. No external targets or symlinks.',
            permissions_policy='0644 plus ordinary source execute bits; setuid/setgid/sticky bits are removed. Differing executable modes use independent payloads.'))
    payload=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode();write(out/'archive_manifest.json',manifest)
    archive_records=[]
    for i,group in enumerate(groups,1):
        name=manifest['volumes'][i-1]['name'];target=folder/name;temporary=target.with_name(name+'.tmp')
        if archive_format=='tar-hardlinks':write_tar(temporary,root,group,aliases,payload,envpath,env,virtual)
        else:
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
                for row in group:
                    path=root/row['path'];stat=path.stat();assert (stat.st_size,stat.st_mtime_ns)==(row['bytes'],row['mtime_ns'])
                    assert ordinary_archive_mode(stat)==row['archive_mode']
                    method=zipfile.ZIP_STORED if path.suffix.lower() in PRECOMPRESSED else zipfile.ZIP_DEFLATED
                    archive.write(path,arcname=row['path'],compress_type=method,compresslevel=1)
                    archive.getinfo(row['path']).external_attr=(0o100000|row['archive_mode'])<<16
                    assert (path.stat().st_size,path.stat().st_mtime_ns)==(row['bytes'],row['mtime_ns'])
                archive.writestr(str(MANIFEST),payload)
                archive.getinfo(str(MANIFEST)).external_attr=(0o100000|0o644)<<16
                for vname,data in virtual.items():
                    archive.writestr(vname,data)
                    archive.getinfo(vname).external_attr=(0o100000|0o644)<<16
        temporary.replace(target)
        verification=verify_archive(target,expected_manifest_sha256=hashlib.sha256(payload).hexdigest())
        sha=digest(target);target.with_name(target.name+'.sha256').write_text(sha+'  '+target.name+'\n')
        archive_records.append(dict(path=str(target.relative_to(root)),bytes=target.stat().st_size,sha256=sha,verification=verification))
        print('Archive verified',i,'/',len(groups),name,target.stat().st_size,flush=True)
    # Check every input again after compression: actual immutable snapshot proof.
    changed=[];rechecked={}
    for row in records:
        path=root/row['path'];stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in rechecked:rechecked[key]=digest(path)
        if (stat.st_size,stat.st_mtime_ns)!=(row['bytes'],row['mtime_ns']) or ordinary_archive_mode(stat)!=row['archive_mode'] or rechecked[key]!=row['sha256']:changed.append(row['path'])
    assert not changed,changed
    links=verify_links(root,[r['path'] for r in records],manifest['archive_compatibility_aliases']);write(out/'archive_link_validation.json',links)
    extracted_receipt=None
    if extract_validate:
        extracted=folder/'extracted_validation_workspace'
        if archive_format=='tar-hardlinks':verify_archive(folder/manifest['volumes'][0]['name'],extract_to=extracted)
        else:
            extracted.mkdir()
            archive_modes={r['path']:r.get('archive_mode',0o644) for r in records+manifest['virtual_files']}
            for volume in manifest['volumes']:
                source=folder/volume['name'];verify_archive(source)
                with zipfile.ZipFile(source) as archive:
                    for info in archive.infolist():
                        safe_member(info.filename);destination=extracted/info.filename
                        if destination.exists():
                            assert digest(destination)==hashlib.sha256(archive.read(info)).hexdigest();continue
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        with archive.open(info) as stream,destination.open('xb') as output:
                            while chunk:=stream.read(2**20):output.write(chunk)
                        os.chmod(destination,archive_modes.get(info.filename,0o644))
        # This root contains only decoded archive members. No source-workspace
        # fallback is consulted for SHA, bank axes, scalar ranks or relative links.
        extracted_receipt=verify_extracted(extracted,extracted/SUPPORT,True,True)
        assert extracted_receipt['axes']['status']=='passed'
        write(out/'final_extracted_validation.json',extracted_receipt)
    report=dict(status='passed' if not links['broken_relative_links'] else 'archive_integrity_passed_links_require_review',
        created_utc=now(),volumes=archive_records,workspace_relative_source_files=len(records),uncompressed_source_bytes=sum(r['bytes'] for r in records),
        archive_bytes=sum(r['bytes'] for r in archive_records),member_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        CRC32_validation='Actual per-member CRC32 for ZIP; actual complete gzip trailer CRC32 and ISIZE for tar.gz.',
        actual_SHA256_all_logical_volume_members=True,source_SHA_rechecked_after_all_volumes=True,
        archive_format=archive_format,deduplication=manifest['deduplication'],
        independent_decoded_workspace_validation=extracted_receipt,
        link_validation='archive_link_validation.json',model_weights_included=False,encoding_reproduction_claimed=False)
    write(out/'archive_set.json',report);return report


def write_tar(path,root,records,aliases,payload,envpath,env,virtual=None):
    def blob(archive,name,data):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    # Explicit GzipFile emits exactly one gzip stream, with a verified trailer.
    with Path(path).open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            blob(archive,str(MANIFEST),payload)
            for name,data in (virtual or {envpath:env}).items():blob(archive,name,data)
            for i,row in enumerate(records,1):
                source=root/row['path'];stat=source.stat();assert (stat.st_size,stat.st_mtime_ns)==(row['bytes'],row['mtime_ns'])
                mode=row.get('archive_mode',0o644)
                assert mode==ordinary_archive_mode(stat) and not mode&0o7000,row['path']
                info=tarfile.TarInfo(row['path']);info.mode=mode
                if row['path'] in aliases:
                    info.type=tarfile.LNKTYPE;info.linkname=aliases[row['path']];archive.addfile(info)
                else:
                    info.size=row['bytes']
                    with source.open('rb') as stream:archive.addfile(info,stream)
                assert (source.stat().st_size,source.stat().st_mtime_ns)==(row['bytes'],row['mtime_ns'])
                assert ordinary_archive_mode(source.stat())==mode,row['path']
                if i%500==0:print('Tar logical members',i,'/',len(records),flush=True)


def safe_member(name):
    p=PurePosixPath(name)
    assert name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,(name,'unsafe archive member')
    assert str(p)==name,(name,'noncanonical archive member')


class GzipCounter:
    def __init__(self,stream):self.stream=stream;self.crc=0;self.bytes=0
    def read(self,size=-1):
        data=self.stream.read(size);self.crc=binascii.crc32(data,self.crc);self.bytes+=len(data);return data


def verify_tar_archive(path,expected_manifest_sha256=None,extract_to=None):
    path=Path(path);seen=set();regular={};manifest=None;raw=None;total=0;links=0
    if extract_to:
        extract_to=Path(extract_to).resolve();assert not extract_to.exists() or not any(extract_to.iterdir()),'Extraction requires an empty destination'
        extract_to.mkdir(parents=True,exist_ok=True)
    with path.open('rb') as source,gzip.GzipFile(fileobj=source,mode='rb') as compressed:
        counted=GzipCounter(compressed)
        with tarfile.open(fileobj=counted,mode='r|') as archive:
            for member in archive:
                safe_member(member.name);assert member.name not in seen,member.name
                assert member.isfile() or member.islnk(),(member.name,'unsupported tar entry type')
                assert not member.mode&~0o777,(member.name,'special permission bits are forbidden')
                if manifest is None:
                    assert member.name==str(MANIFEST) and member.isfile(),'Manifest must be the first member'
                    raw=archive.extractfile(member).read();manifest=json.loads(raw)
                    if expected_manifest_sha256:assert hashlib.sha256(raw).hexdigest()==expected_manifest_sha256
                    volume=next(v for v in manifest['volumes'] if v['name']==path.name)
                    index={r['path']:r for r in manifest['files']+manifest['virtual_files']}
                    expected=set(volume['paths'])|{str(MANIFEST)}|{r['path'] for r in manifest['virtual_files']}
                    declared_links=volume.get('link_targets',{})
                    h=hashlib.sha256(raw);size=len(raw)
                    assert size==member.size
                    assert member.mode==0o644,'Manifest permission must be ordinary 0644'
                    if extract_to:
                        destination=extract_to/member.name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
                        os.chmod(destination,member.mode)
                else:
                    assert member.name in expected,member.name
                    row=index[member.name]
                    assert member.mode==row.get('archive_mode',0o644),(member.name,'archive permission mismatch')
                    if member.islnk():
                        safe_member(member.linkname)
                        assert member.size==0 and member.name in declared_links
                        assert declared_links[member.name]==member.linkname and member.linkname in regular,'Links must target earlier regular members'
                        sha,size,mode=regular[member.linkname];assert sha==row['sha256'] and size==row['bytes'] and mode==member.mode
                        if extract_to:
                            destination=extract_to/member.name;destination.parent.mkdir(parents=True,exist_ok=True);os.link(extract_to/member.linkname,destination)
                        links+=1;total+=size;seen.add(member.name);continue
                    assert member.name not in declared_links
                    h=hashlib.sha256();size=0
                    destination=None
                    if extract_to:
                        destination=extract_to/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
                    output=destination.open('xb') if destination else None
                    try:
                        with archive.extractfile(member) as stream:
                            while chunk:=stream.read(2**20):
                                h.update(chunk);size+=len(chunk)
                                if output:output.write(chunk)
                    finally:
                        if output:output.close()
                    assert size==member.size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
                    if destination:os.chmod(destination,member.mode)
                regular[member.name]=(h.hexdigest(),size,member.mode);total+=size;seen.add(member.name)
        # Read padding through actual gzip EOF, which validates the trailer.
        while counted.read(2**20):pass
        crc=counted.crc&0xffffffff;tar_bytes=counted.bytes
    with path.open('rb') as stream:stream.seek(-8,os.SEEK_END);trailer_crc,trailer_size=struct.unpack('<II',stream.read(8))
    assert crc==trailer_crc and (tar_bytes&0xffffffff)==trailer_size
    assert manifest is not None and seen==expected,(seen-expected,expected-seen)
    assert links==len(declared_links)
    return dict(status='passed',archive_format='tar-hardlinks',logical_members_verified=len(seen),
        regular_payload_SHA256_comparisons=len(regular)-1,internal_hardlink_SHA256_comparisons=links,
        logical_uncompressed_bytes=total,actual_gzip_trailer_CRC32=f'{crc:08x}',actual_gzip_trailer_ISIZE=trailer_size,
        complete_decompressed_tar_bytes=tar_bytes,manifest_sha256=hashlib.sha256(raw).hexdigest(),
        hardlinks_target_earlier_regular_archive_members=True,unsafe_or_duplicate_members=False,
        extracted_to=str(extract_to) if extract_to else None)


def verify_archive(path,expected_manifest_sha256=None,extract_to=None):
    path=Path(path)
    sidecar=path.with_name(path.name+'.sha256');archive_sha=None
    if sidecar.is_file():
        declared=sidecar.read_text().split()[0];assert re.fullmatch('[0-9a-f]{64}',declared)
        archive_sha=digest(path);assert archive_sha==declared,'Archive SHA256 sidecar mismatch'
    if path.name.endswith('.tar.gz'):
        receipt=verify_tar_archive(path,expected_manifest_sha256,extract_to);receipt['external_archive_SHA256_verified']=archive_sha
        return receipt
    assert not extract_to,'Safe extraction currently supports the self-contained tar.gz format'
    path=Path(path);name=path.name;records=0;total=0
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();assert len(names)==len(set(names))
        for member in names:safe_member(member)
        raw=archive.read(str(MANIFEST));manifest=json.loads(raw)
        if expected_manifest_sha256:assert hashlib.sha256(raw).hexdigest()==expected_manifest_sha256
        volume=next(v for v in manifest['volumes'] if v['name']==name)
        index={r['path']:r for r in manifest['files']+manifest['virtual_files']}
        expected=set(volume['paths'])|{str(MANIFEST)}|{r['path'] for r in manifest['virtual_files']}
        assert set(names)==expected,(set(names)-expected,expected-set(names))
        for info in archive.infolist():
            h=hashlib.sha256();crc=0;size=0
            with archive.open(info) as stream:
                while chunk:=stream.read(2**20):h.update(chunk);crc=binascii.crc32(chunk,crc);size+=len(chunk)
            assert size==info.file_size and (crc&0xffffffff)==info.CRC
            zip_mode=info.external_attr>>16
            assert not zip_mode&0o7000,(info.filename,'special ZIP permission bits are forbidden')
            assert zip_mode&0o170000 in (0,0o100000),(info.filename,'non-regular ZIP payload')
            if info.filename!=str(MANIFEST):
                row=index[info.filename];assert size==row['bytes'] and h.hexdigest()==row['sha256']
                if 'archive_mode' in row:assert zip_mode&0o777==row['archive_mode'],(info.filename,'ZIP permission mismatch')
            else:assert h.hexdigest()==hashlib.sha256(raw).hexdigest()
            records+=1;total+=size
    return dict(status='passed',members_read=records,uncompressed_member_bytes=total,
        actual_member_CRC32_comparisons=records,actual_member_SHA256_comparisons=records-1,manifest_bytes_SHA256_reported=True,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),unsafe_or_duplicate_members=False,external_archive_SHA256_verified=archive_sha)


def verify_axes(root,manifest):
    # Reuse the independent stable-rank implementation, not reported pass flags.
    sys.path.insert(0,str(root))
    import torch
    from scripts.colnomic_mechanism.scene35 import replay_scores
    torch.set_num_threads(2)
    photos=csvrows(root/'dataset/lists/photos.csv');refs=csvrows(root/'dataset/lists/references.csv')
    assert [r['photo_id'] for r in photos]==manifest['query_ids'] and len(photos)==1117
    assert [r['reference_id'] for r in refs]==manifest['reference_ids'] and len(refs)==5405
    qb=torch.load(root/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True,mmap=True)
    rb=torch.load(root/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True,mmap=True)
    assert qb['ids']==manifest['query_ids'] and rb['ids']==manifest['reference_ids']
    configs=cohort(root);assert len(configs)==len(manifest['configurations35'])==35
    assert {r['config_id'] for r in manifest['configurations35']}=={r['id'] for r in configs}
    positives=defaultdict(set)
    for r in csvrows(root/'dataset/lists/positives.csv'):positives[r['package_id']].add(r['reference_id'])
    hashes={str(Path('dataset/lists')/(name+'.csv')):digest(root/'dataset/lists'/(name+'.csv')) for name in ['photos','references','positives','groups']}
    records=[]
    for config in configs:
        safe_member(config['result'])
        spec=read(root/config['result']/'config.json')
        scalar_path=Path(score_path(config,spec));safe_member(str(scalar_path))
        assert (root/scalar_path).resolve().is_relative_to(root),'Frozen score replay must use the decoded workspace'
        predictions=[json.loads(line) for line in (root/config['result']/'predictions.jsonl').read_text().splitlines()]
        assert len(predictions)==1117 and {p['photo_id'] for p in predictions}==set(manifest['query_ids'])
        _,receipt=replay_scores(root,config,spec,photos,refs,positives,{p['photo_id']:p for p in predictions},hashes)
        assert receipt['status']=='passed' and receipt['verified_predictions']+receipt['empty_ocr_predictions']==1117
        records.append(dict(config_id=config['id'],**receipt));print('Archive axes/ranks',config['id'],receipt['verified_predictions'],flush=True)
    return dict(status='passed',queries=1117,references=5405,raw_banks_axes_verified=True,configurations=35,
        actual_prediction_rows_replayed=sum(r['verified_predictions'] for r in records),empty_OCR_rows=sum(r['empty_ocr_predictions'] for r in records),receipts=records)


def verify_extracted(root,out,axes,links):
    manifest=read(root/MANIFEST);records=manifest['files']+manifest['virtual_files'];bad=[];inode_hashes={}
    for i,row in enumerate(records,1):
        safe_member(row['path'])
        path=root/row['path']
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size!=row['bytes']:bad.append(row['path']);continue
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in inode_hashes:inode_hashes[key]=digest(path)
        if inode_hashes[key]!=row['sha256'] or path.stat().st_mode&0o7777!=row.get('archive_mode',0o644):bad.append(row['path'])
        if i%500==0:print('Extracted member SHA',i,'/',len(records),flush=True)
    assert not bad,bad
    compatibility=manifest.get('archive_compatibility_aliases',[])
    for row in compatibility:
        assert digest(root/row['path'])==digest(root/row['compatibility_alias_of'])==row['sha256']
    result=dict(status='passed',files_SHA_verified=len(records),physical_file_payloads_read=len(inode_hashes),member_manifest_sha256=digest(root/MANIFEST),
        original_workspace_absolute_paths_not_treated_as_portable=True,archive_compatibility_aliases_SHA_verified=len(compatibility))
    if axes:result['axes']=verify_axes(root,manifest)
    if links:
        validation=verify_links(root,[r['path'] for r in records],compatibility);write(out/'extracted_link_validation.json',validation)
        result['links']=dict(status=validation['status'],broken_relative_links=len(validation['broken_relative_links']),legacy_absolute_links=validation['legacy_absolute_links'],
            repaired_original_source_links=len(validation['repaired_original_source_links']))
    write(out/'extracted_archive_validation.json',result);return result


def self_test(out):
    """Execute positive and corruption cases against real, tiny archives."""
    base=out/('self_test_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    source=base/'source';source.mkdir(parents=True);records=[]
    for name,data in [('data/a.bin',b'actual frozen byte payload\x00'*17),('data/b.bin',b'actual frozen byte payload\x00'*17),('data/c.txt',b'unique original text\n')]:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);stat=path.stat()
        records.append(dict(path=name,bytes=len(data),mtime_ns=stat.st_mtime_ns,sha256=digest(path)))
    envpath=str(SUPPORT/'package_environment.json');env=b'{"synthetic_self_test":true}\n'
    manifest=dict(files=records,virtual_files=[dict(path=envpath,bytes=len(env),sha256=hashlib.sha256(env).hexdigest())],
        volumes=[dict(name='fixture.tar.gz',paths=[r['path'] for r in records],link_targets={'data/b.bin':'data/a.bin'})])
    payload=(json.dumps(manifest)+'\n').encode();tarpath=base/'fixture.tar.gz'
    write_tar(tarpath,source,records,{'data/b.bin':'data/a.bin'},payload,envpath,env)
    receipts=[dict(test='tar_real_SHA_CRC_hardlinks',receipt=verify_archive(tarpath,hashlib.sha256(payload).hexdigest(),base/'extracted'))]
    assert (base/'extracted/data/a.bin').stat().st_ino==(base/'extracted/data/b.bin').stat().st_ino
    for row in records:assert digest(base/'extracted'/row['path'])==row['sha256']
    receipts.append(dict(test='extracted_all_logical_SHA_and_actual_shared_inode',status='passed'))
    receipts.append(dict(test='actual_verify_extracted_entrypoint',receipt=verify_extracted(base/'extracted',base/'extracted'/SUPPORT,False,False)))
    def rejects(name,call):
        try:call()
        except (AssertionError,ValueError,KeyError,StopIteration,tarfile.TarError,zipfile.BadZipFile,gzip.BadGzipFile,EOFError) as error:
            receipts.append(dict(test=name,status='rejected_as_expected',exception_type=type(error).__name__));return
        raise AssertionError('Corruption case incorrectly accepted: '+name)
    rejects('final_sealing_requires_explicit_ready',lambda:build(source,base/'must_not_exist',False,1024))
    assert not (base/'must_not_exist').exists()
    tarpath.with_name(tarpath.name+'.sha256').write_text(digest(tarpath)+'  '+tarpath.name+'\n')
    receipt=verify_archive(tarpath);assert receipt['external_archive_SHA256_verified']==digest(tarpath)
    receipts.append(dict(test='external_archive_SHA_sidecar_matches_actual_bytes',status='passed'))
    tarpath.with_name(tarpath.name+'.sha256').write_text('0'*64+'  '+tarpath.name+'\n')
    rejects('external_archive_SHA_sidecar_mismatch',lambda:verify_archive(tarpath))
    tarpath.with_name(tarpath.name+'.sha256').write_text(digest(tarpath)+'  '+tarpath.name+'\n')
    rejects('externally_expected_manifest_SHA_mismatch',lambda:verify_archive(tarpath,'0'*64))
    corrupt=base/'bad_crc'/'fixture.tar.gz';corrupt.parent.mkdir();content=bytearray(tarpath.read_bytes());content[-8]^=1;corrupt.write_bytes(content)
    rejects('gzip_actual_trailer_CRC_corruption',lambda:verify_archive(corrupt))
    bad=base/'bad_links'/'fixture.tar.gz';bad.parent.mkdir()
    bad_manifest=json.loads(payload);bad_manifest['volumes'][0]['link_targets']['data/b.bin']='../outside.bin'
    write_tar(bad,source,records,{'data/b.bin':'../outside.bin'},json.dumps(bad_manifest).encode(),envpath,env)
    rejects('tar_hardlink_path_escape',lambda:verify_archive(bad))
    badsha=base/'bad_sha_tar'/'fixture.tar.gz';badsha.parent.mkdir()
    bad_manifest=json.loads(payload);bad_manifest['files'][2]['sha256']='0'*64
    write_tar(badsha,source,records,{'data/b.bin':'data/a.bin'},json.dumps(bad_manifest).encode(),envpath,env)
    rejects('tar_actual_payload_SHA_mismatch',lambda:verify_archive(badsha))
    zmanifest=json.loads(payload);zmanifest['volumes'][0]['name']='fixture.zip';zmanifest['volumes'][0]['link_targets']={}
    zraw=json.dumps(zmanifest).encode();zpath=base/'fixture.zip'
    def make_zip(path,data_override=None,extra=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
            for row in records:archive.writestr(row['path'],data_override if data_override is not None and row['path']=='data/c.txt' else (source/row['path']).read_bytes())
            archive.writestr(str(MANIFEST),zraw);archive.writestr(envpath,env)
            if extra:archive.writestr(extra,b'unsafe')
    make_zip(zpath);receipts.append(dict(test='zip_real_SHA_and_all_member_CRC',receipt=verify_archive(zpath)))
    corrupt=base/'bad_crc_zip'/'fixture.zip';corrupt.parent.mkdir()
    data=bytearray(zpath.read_bytes())
    with zipfile.ZipFile(zpath) as archive:offset=archive.getinfo('data/a.bin').header_offset
    nl,el=struct.unpack('<HH',data[offset+26:offset+30]);data[offset+30+nl+el]^=1;corrupt.write_bytes(data)
    rejects('zip_actual_member_CRC_corruption',lambda:verify_archive(corrupt))
    corrupt=base/'bad_sha_zip'/'fixture.zip';make_zip(corrupt,b'changed original txt\n')
    rejects('zip_actual_payload_SHA_mismatch',lambda:verify_archive(corrupt))
    corrupt=base/'bad_path_zip'/'fixture.zip';make_zip(corrupt,extra='../outside.bin')
    rejects('zip_member_path_escape',lambda:verify_archive(corrupt))
    linkroot=base/'links';linkroot.mkdir();(linkroot/'ok.txt').write_text('linked file');(linkroot/'index.html').write_text('<a href="ok.txt">ok</a><a href="missing.txt">missing</a>')
    links=verify_links(linkroot,['index.html','ok.txt']);assert links['checked_relative_links']==1 and len(links['broken_relative_links'])==1
    receipts.append(dict(test='actual_HTML_present_and_missing_relative_link',status='passed'))
    (linkroot/'README.md').write_text('[present](ok.txt) [absent](missing.txt)')
    links=verify_links(linkroot,['README.md','ok.txt']);assert links['checked_relative_links']==1 and len(links['broken_relative_links'])==1
    receipts.append(dict(test='actual_Markdown_present_and_missing_relative_link',status='passed'))
    # A real archive-only JSON copy repairs an original report link, without
    # creating anything at that historical path in the source workspace.
    compatibility_source=base/'compatibility_source';compatibility_source.mkdir();compatibility_records=[]
    report_name,target=next(iter(HISTORICAL_LINK_DEFECTS));canonical=HISTORICAL_LINK_DEFECTS[(report_name,target)]
    for name,data in [(report_name,('[original historical link]('+target+')\n').encode()),(canonical,b'{"actual_historical_receipt":true}\n')]:
        path=compatibility_source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);stat=path.stat()
        compatibility_records.append(dict(path=name,bytes=len(data),mtime_ns=stat.st_mtime_ns,sha256=digest(path)))
    alias_name=str(((compatibility_source/report_name).parent/target).resolve().relative_to(compatibility_source.resolve()))
    original=verify_links(compatibility_source,[r['path'] for r in compatibility_records]);assert len(original['broken_relative_links'])==1
    receipts.append(dict(test='original_historical_missing_link_cannot_pass_without_archive_member',status='passed'))
    alias_data=(compatibility_source/canonical).read_bytes()
    alias=dict(path=alias_name,compatibility_alias_of=canonical,bytes=len(alias_data),sha256=hashlib.sha256(alias_data).hexdigest(),original_workspace_target_missing=True)
    cmanifest=dict(files=compatibility_records,virtual_files=[alias],archive_compatibility_aliases=[alias],
        volumes=[dict(name='compatibility.tar.gz',paths=[r['path'] for r in compatibility_records],link_targets={})])
    craw=(json.dumps(cmanifest)+'\n').encode();cpath=base/'compatibility.tar.gz'
    write_tar(cpath,compatibility_source,compatibility_records,{},craw,envpath,env,{alias_name:alias_data})
    verify_archive(cpath,hashlib.sha256(craw).hexdigest(),base/'compatibility_extracted')
    verified=verify_extracted(base/'compatibility_extracted',base/'compatibility_extracted'/SUPPORT,False,True)
    assert verified['links']['status']=='passed' and verified['links']['broken_relative_links']==0
    assert verified['archive_compatibility_aliases_SHA_verified']==1 and verified['links']['repaired_original_source_links']==1
    assert not (compatibility_source/alias_name).exists()
    receipts.append(dict(test='archive_only_compatibility_member_actual_extract_SHA_and_resolved_original_link',receipt=verified))
    # Execute a harmless real extracted shell entry. Equal bytes with different
    # execute bits must remain separate inodes; equal bytes+mode may share one.
    mode_source=base/'permission_source';mode_source.mkdir();mode_records=[]
    shell=b'#!/bin/sh\nprintf "archive executable fixture\\n"\n'
    for name,mode in [('REPRODUCE.sh',0o755),('same_mode_copy.sh',0o755),('non_executable_copy.sh',0o644)]:
        path=mode_source/name;path.write_bytes(shell);os.chmod(path,mode);stat=path.stat()
        mode_records.append(dict(path=name,bytes=len(shell),mtime_ns=stat.st_mtime_ns,archive_mode=ordinary_archive_mode(stat),sha256=digest(path)))
    mode_aliases=deduplicate_records(mode_records)
    assert mode_aliases=={'same_mode_copy.sh':'REPRODUCE.sh'}
    assert ordinary_archive_mode(type('FixtureStat',(),{'st_mode':0o106755})())==0o755
    mode_manifest=dict(files=mode_records,virtual_files=[dict(path=envpath,bytes=len(env),archive_mode=0o644,sha256=hashlib.sha256(env).hexdigest())],
        volumes=[dict(name='permissions.tar.gz',paths=[r['path'] for r in mode_records],link_targets=mode_aliases)])
    mode_payload=(json.dumps(mode_manifest)+'\n').encode();mode_tar=base/'permissions.tar.gz'
    write_tar(mode_tar,mode_source,mode_records,mode_aliases,mode_payload,envpath,env)
    mode_receipt=verify_archive(mode_tar,hashlib.sha256(mode_payload).hexdigest(),base/'permissions_extracted')
    mode_decoded=base/'permissions_extracted'
    assert (mode_decoded/'REPRODUCE.sh').stat().st_mode&0o7777==0o755
    assert (mode_decoded/'REPRODUCE.sh').stat().st_ino==(mode_decoded/'same_mode_copy.sh').stat().st_ino
    assert (mode_decoded/'REPRODUCE.sh').stat().st_ino!=(mode_decoded/'non_executable_copy.sh').stat().st_ino
    assert not os.access(mode_decoded/'non_executable_copy.sh',os.X_OK)
    command=subprocess.run([str(mode_decoded/'REPRODUCE.sh')],cwd=mode_decoded,check=True,capture_output=True,text=True)
    assert command.stdout=='archive executable fixture\n'
    mode_extracted=verify_extracted(mode_decoded,mode_decoded/SUPPORT,False,False)
    receipts.append(dict(test='ordinary_exec_bits_retained_and_actual_extracted_REPRODUCE_entry_executes',status='passed',
        archive_receipt=mode_receipt,extracted_receipt=mode_extracted,actual_stdout=command.stdout))
    receipts.append(dict(test='same_SHA_different_exec_modes_are_independent_payloads_and_inodes',status='passed',
        same_mode_hardlink_verified=True,different_mode_regular_payload_verified=True,special_permission_bits_removed=True))
    unsafe_aliases={'same_mode_copy.sh':'REPRODUCE.sh','non_executable_copy.sh':'REPRODUCE.sh'}
    unsafe_manifest=json.loads(mode_payload);unsafe_manifest['volumes'][0]['name']='bad_permissions.tar.gz';unsafe_manifest['volumes'][0]['link_targets']=unsafe_aliases
    unsafe_tar=base/'bad_permissions.tar.gz';write_tar(unsafe_tar,mode_source,mode_records,unsafe_aliases,json.dumps(unsafe_manifest).encode(),envpath,env)
    rejects('same_SHA_hardlink_with_different_exec_permissions_is_rejected',lambda:verify_archive(unsafe_tar))
    mode_zip=base/'permissions.zip';zip_modes=json.loads(mode_payload);zip_modes['volumes'][0].update(name=mode_zip.name,link_targets={})
    with zipfile.ZipFile(mode_zip,'w',compression=zipfile.ZIP_STORED) as archive:
        for row in mode_records:
            archive.write(mode_source/row['path'],arcname=row['path'])
            archive.getinfo(row['path']).external_attr=(0o100000|row['archive_mode'])<<16
        archive.writestr(str(MANIFEST),json.dumps(zip_modes).encode());archive.getinfo(str(MANIFEST)).external_attr=(0o100000|0o644)<<16
        archive.writestr(envpath,env);archive.getinfo(envpath).external_attr=(0o100000|0o644)<<16
    zip_permission_receipt=verify_archive(mode_zip)
    with zipfile.ZipFile(mode_zip) as archive:
        assert archive.getinfo('REPRODUCE.sh').external_attr>>16&0o7777==0o755
        assert archive.getinfo('non_executable_copy.sh').external_attr>>16&0o7777==0o644
    receipts.append(dict(test='ZIP_fallback_safe_executable_and_virtual_member_modes_verified',status='passed',receipt=zip_permission_receipt))
    result=dict(status='passed',created_utc=now(),fixture_directory=str(base),actual_cases=len(receipts),cases=receipts,final_archive_created=False,
        source_code_sha256=digest(Path(__file__)))
    write(out/'self_test_validation.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path);parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--build',action='store_true');parser.add_argument('--ready',action='store_true')
    parser.add_argument('--audit-duplicates',action='store_true');parser.add_argument('--archive-format',choices=['tar-hardlinks','zip64'],default='tar-hardlinks')
    parser.add_argument('--skip-extracted-validation',action='store_true',help='Only for explicitly scoped partial validation; final complete builds decode and independently validate all archived files and 35 scores by default.')
    parser.add_argument('--volume-gib',type=float,default=2.0);parser.add_argument('--verify-archive',type=Path)
    parser.add_argument('--extract-to',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--verify-extracted',action='store_true');parser.add_argument('--axes',action='store_true');parser.add_argument('--links',action='store_true')
    args=parser.parse_args();root=args.root.resolve();out=(args.output or root/SUPPORT).resolve()
    assert out.is_relative_to(root/SUPPORT),'Writes are confined to delivery_support'
    out.mkdir(parents=True,exist_ok=True)
    if args.self_test:result=self_test(out)
    elif args.verify_archive:result=verify_archive(args.verify_archive.resolve(),extract_to=args.extract_to)
    elif args.verify_extracted:result=verify_extracted(root,out,args.axes,args.links)
    elif args.build:result=build(root,out,args.ready,int(args.volume_gib*2**30),args.archive_format,not args.skip_extracted_validation)
    elif args.audit_duplicates:result=audit_duplicates(root,out)
    else:result=inventory(root,out)
    if not args.inventory and (args.self_test or args.verify_archive or args.verify_extracted or args.build):print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
