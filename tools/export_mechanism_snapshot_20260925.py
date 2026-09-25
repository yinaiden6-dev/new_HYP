#!/usr/bin/env python3
"""Archive the completed POST-LLM and attribution round, preserving source bytes."""
import ast
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

WS = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = WS/'ICLR/new ROUTEA/RC'
OUT = WS/'github_exports/new_HYP_20260919'
BACKUP = OUT/'backup/postllm_and_attribution_20260925'
ROOTS = '''rc_internal_m_v4_probe_v1 rc_mass_information_transport_v1
rc_postllm_h593_attribution_v1 rc_postllm_h593_decomposition_v1 rc_postllm_m_v1
rc_pair_relation_localization_v1 rc_roma_position_factor_v1 rc_internal_m_condition_scale_v4
rc_existing_evidence_attribution_v1 rc_postllm_h593_v1 rc_colnomic_relation_reader_v1
rc_postllm_m_signal_decomposition_v1 rc_postllm_train_pool_common_v1
rc_h593_mass_main_effects_v1 rc_postllm_train_pool_common_v2
rc_mass_discrimination_source_v1 rc_internal_m_content_preserving_readout_v1
rc_postllm_m_signal_decomposition_v2'''.split()
TEXT = {'.py','.md','.json','.jsonl','.csv','.tsv','.txt','.sh','.sbatch','.yaml','.yml','.toml'}
PRUNE = {'__pycache__','loading','encoder_cache','cache','logs','locks','patch_evidence'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16})')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def clean_read(p):
    if p.is_symlink():
        raise RuntimeError('Unexpected symlink: '+str(p))
    data=p.read_bytes()
    if SECRET.search(data):
        raise RuntimeError('Secret review required: '+str(p))
    if p.suffix=='.json':
        json.loads(data)
    return data

def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.export-tmp')
    tmp.write_bytes(data)
    os.replace(tmp,path)

def export_file(p):
    data=clean_read(p)
    rel=p.relative_to(WS)
    target=OUT/rel
    compressed=len(data)>60*1024**2
    if compressed:
        target=Path(str(target)+'.gz')
        output=gzip.compress(data,compresslevel=6,mtime=0)
    else:
        output=data
    if len(output)>=95*1024**2:
        raise RuntimeError('Archive exceeds Git file limit: '+str(p))
    if not target.is_file() or target.read_bytes()!=output:
        put(target,output)
    return dict(source=str(rel),export=str(target.relative_to(OUT)),source_bytes=len(data),
                bytes=len(output),source_sha256=digest(data),sha256=digest(output),
                encoding='gzip' if compressed else 'identity')

def export_candidates(folder):
    members=[]
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',compresslevel=6,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(folder.iterdir()):
                if p.suffix!='.json':
                    raise RuntimeError('Unexpected candidate payload: '+str(p))
                data=clean_read(p)
                info=tarfile.TarInfo(p.name);info.size=len(data);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(data))
                members.append(dict(name=p.name,bytes=len(data),sha256=digest(data)))
            data=(json.dumps(members,indent=2)+'\n').encode()
            info=tarfile.TarInfo('_EXPORT_MEMBERS.json');info.size=len(data);info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
    data=buffer.getvalue()
    target=OUT/folder.relative_to(WS)
    target=target.with_name('candidates.tar.gz')
    if len(data)>=95*1024**2:raise RuntimeError('Archive too large')
    put(target,data)
    return dict(source=str(folder.relative_to(WS)),export=str(target.relative_to(OUT)),
                files=len(members),source_bytes=sum(x['bytes'] for x in members),bytes=len(data),
                sha256=digest(data),encoding='tar.gz',members_manifest='_EXPORT_MEMBERS.json')

def main():
    started=time.time()
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OUT,text=True).strip()
    cutoff=int(subprocess.check_output(['git','show','-s','--format=%ct','HEAD'],cwd=OUT))
    files=set();excluded=[];archives=[]
    catalog={}
    for area in ['reports','plan','programs','registry','slurm']:
        for p in (RC/area).iterdir():
            if p.is_file() and p.suffix in TEXT:
                catalog[p.name]=p
                if p.stat().st_mtime>=cutoff:files.add(p)
    # Include local implementation dependencies and contracts of the new round.
    queue=list(files);seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p);text=clean_read(p).decode('utf-8')
        names=set(re.findall(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sbatch|sh|json|md)',text))
        if p.suffix=='.py':
            for node in ast.walk(ast.parse(text)):
                if isinstance(node,ast.Import):names.update(x.name.split('.')[0]+'.py' for x in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:names.add(node.module.split('.')[0]+'.py')
        for name in names:
            dep=catalog.get(name)
            if dep is not None and dep not in files:
                files.add(dep);queue.append(dep)
    for name in ROOTS:
        root=RC/'results'/name
        if not root.is_dir():raise RuntimeError('Missing result '+name)
        for path,dirs,names in os.walk(root,followlinks=False):
            path=Path(path)
            for d in list(dirs):
                child=path/d
                if child.is_symlink() or d in PRUNE:
                    excluded.append(dict(path=str(child.relative_to(WS)),reason='cache_or_runtime_directory'))
                    dirs.remove(d)
                elif d=='candidates':
                    archives.append(child);dirs.remove(d)
            for n in names:
                p=path/n
                if p.suffix in TEXT and n not in ('stdout.txt','stderr.txt') and not p.is_symlink():
                    files.add(p)
                else:excluded.append(dict(path=str(p.relative_to(WS)),reason='binary_model_cache_or_runtime_file'))
    print(json.dumps(dict(stage='inventory',files=len(files),candidate_archives=len(archives),excluded=len(excluded))),flush=True)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(export_file,p) for p in sorted(files)]
        futures += [pool.submit(export_candidates,p) for p in sorted(archives)]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            records.append(f.result())
            if i%1000==0:print(json.dumps(dict(stage='export',done=i,total=len(futures))),flush=True)
    BACKUP.mkdir(parents=True,exist_ok=True)
    record=dict(schema='new_hyp_mechanism_export_v1',source_workspace=str(WS),base_commit=base,
                result_roots=ROOTS,records=sorted(records,key=lambda x:x['export']),exclusions=excluded,
                summary=dict(exported_files=len(records),archived_candidate_files=sum(r.get('files',0) for r in records),
                             source_bytes=sum(r['source_bytes'] for r in records),export_bytes=sum(r['bytes'] for r in records)))
    put(BACKUP/'files.json',(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
    tool=OUT/'tools/export_mechanism_snapshot_20260925.py'
    put(tool,Path(__file__).read_bytes())
    print(json.dumps(dict(stage='DONE',**record['summary'],seconds=time.time()-started)),flush=True)

if __name__=='__main__':main()
