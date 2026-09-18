#!/usr/bin/env python3
"""Verify source-copy hashes and the tracked-file inventory without loading models."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    rows=json.loads((ROOT/'backup/copied_files.json').read_text())
    for r in rows:
        p=ROOT/r['path']
        assert p.is_file() and not p.is_symlink(),r['path']
        assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['path']
    full=ROOT/'backup/tracked_content_manifest.json'
    if full.exists():
        for r in json.loads(full.read_text()):
            p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['path']
    model=json.loads((ROOT/'model_assets_manifest.json').read_text())
    for r in model['models']:
        assert sum(p['bytes'] for p in r['parts'])==r['bytes']
        assert all(p['bytes']<2*1024**3 for p in r['parts'])
    assert sum(r['bytes'] for r in model['models'])==model['total_bytes']
    print(json.dumps({'status':'ARCHIVE_SOURCE_HASHES_AND_MODEL_MANIFEST_PASS',
                     'source_files':len(rows),'model_files':len(model['models']),
                     'model_parts':sum(len(x['parts']) for x in model['models'])}))

if __name__=='__main__':main()
