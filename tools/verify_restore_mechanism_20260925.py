#!/usr/bin/env python3
"""Verify the published experiment snapshot; optionally restore compressed text."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BACKUP=ROOT/'backup/postllm_and_attribution_20260925'

def sha(data):return hashlib.sha256(data).hexdigest()

def restore_file(path,data):
    if path.is_symlink():raise RuntimeError('Refusing symlink: '+str(path))
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Existing file differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.restore-tmp');tmp.write_bytes(data);os.replace(tmp,path)

def verify(record,restore):
    p=ROOT/record['export'];data=p.read_bytes()
    assert len(data)==record['bytes'] and sha(data)==record['sha256'], str(p)
    kind=record['encoding'];count=1
    if kind=='gzip':
        decoded=gzip.decompress(data)
        assert len(decoded)==record['source_bytes'] and sha(decoded)==record['source_sha256']
        json.loads(decoded)
        if restore:restore_file(ROOT/record['source'],decoded)
    elif kind=='tar.gz':
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
            members=json.load(tar.extractfile('_EXPORT_MEMBERS.json'))
            assert len(members)==record['files']
            assert set(tar.getnames())=={m['name'] for m in members}|{'_EXPORT_MEMBERS.json'}
            for m in members:
                name=m['name'];assert Path(name).name==name and name.endswith('.json')
                info=tar.getmember(name);assert info.isfile()
                decoded=tar.extractfile(info).read()
                assert len(decoded)==m['bytes'] and sha(decoded)==m['sha256']
                json.loads(decoded)
                if restore:restore_file(ROOT/record['source']/name,decoded)
            count=len(members)
    else:
        assert kind=='identity' and sha(data)==record['source_sha256']
        if p.suffix=='.json':json.loads(data)
    return count

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--restore-text',action='store_true')
    args=ap.parse_args();m=json.loads((BACKUP/'files.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        counts=list(pool.map(lambda r:verify(r,args.restore_text),m['records']))
    d=json.loads((ROOT/'ICLR/new ROUTEA/RC/results/rc_postllm_h593_decomposition_v1/result.json').read_text())
    p=json.loads((ROOT/'ICLR/new ROUTEA/RC/results/rc_postllm_h593_v1/result.json').read_text())
    assert d['status']=='ALL593_DECOMPOSITION_AUDITED' and d['n']==593
    assert p['status']=='H593_POSTLLM_FIVEFOLD_INDEPENDENT_JOIN_COMPLETE' and p['n']==593
    assert p['totals']['RAW']['correct']==426 and p['totals']['POST_REAL']['correct']==478
    expected={'NATIVE':478,'COMMON_ONLY':477,'SPATIAL_ONLY':430,'NATIVE_FIXED_ARGMAX':469,'COMMON_FIXED_ARGMAX':468}
    assert {k:d['totals'][k]['correct'] for k in expected}==expected
    candidate_count=sum(r.get('files',0) for r in m['records'] if 'rc_postllm_h593_decomposition_v1/queries/' in r['source'])
    assert candidate_count==593*128
    value=dict(status='EXPORTED_BYTES_ARCHIVE_MEMBERS_AND_RESULT_COUNTS_VERIFIED',
               manifest_sha256=sha((BACKUP/'files.json').read_bytes()),exported_files=len(counts),
               original_text_files_verified=sum(counts),h593_decomposition_candidate_records=candidate_count,
               text_restored=args.restore_text,binary_caches_included=False,
               scientific_validation='Existing independent experiment validations preserved; this is export verification.')
    (BACKUP/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
