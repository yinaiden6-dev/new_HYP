#!/usr/bin/env python3
"""Split only the exact mainline model weights; preserve original byte streams."""
from pathlib import Path
import hashlib,json,os

WS=Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
STAGE=WS/'github_exports/new_HYP_20260919'
ASSETS=WS/'github_exports/model_assets_20260919'
TAG='mainline-backup-20260919'
CHUNK=1024**3

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    paths=[WS/'models/downloaded_models/colnomic-embed-multimodal-7b/adapter_model.safetensors']
    paths+=sorted((WS/'models/downloaded_models/colqwen2.5-7B-base').glob('model-*-of-*.safetensors'))
    paths+=[WS/'third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt']
    assert len(paths)==9
    records=[]
    for i,path in enumerate(paths):
        before=path.stat();whole=hashlib.sha256();parts=[]
        with path.open('rb') as src:
            index=0
            while True:
                data=src.read(CHUNK)
                if not data:break
                name=f'model{i:02d}-{path.name}.part{index:03d}'
                dst=ASSETS/name;digest=hashlib.sha256(data).hexdigest();whole.update(data)
                if dst.exists():
                    assert dst.stat().st_size==len(data) and hashlib.sha256(dst.read_bytes()).hexdigest()==digest, 'EXISTING_ASSET_DIFFERS'
                else:
                    temp=dst.with_name(dst.name+'.partial')
                    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                    temp.rename(dst)
                parts.append(dict(asset=name,bytes=len(data),sha256=digest))
                index+=1
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'MODEL_CHANGED_WHILE_READING'
        rec=dict(path=str(path.relative_to(WS)),bytes=before.st_size,sha256=whole.hexdigest(),parts=parts)
        records.append(rec)
        print(json.dumps(dict(model=rec['path'],GiB=round(rec['bytes']/2**30,3),sha256=rec['sha256'],parts=len(parts))),flush=True)
    manifest=dict(schema=1,repository='yinaiden6-dev/new_HYP',release_tag=TAG,part_bytes=CHUNK,
                  format='raw byte chunks in listed order; no compression or conversion',
                  source_files=len(records),total_bytes=sum(r['bytes'] for r in records),models=records)
    data=(json.dumps(manifest,indent=2)+'\n').encode()
    (ASSETS/'model_assets_manifest.json').write_bytes(data)
    (STAGE/'model_assets_manifest.json').write_bytes(data)
    print(json.dumps(dict(status='MODEL_ASSETS_PREPARED_NOT_YET_UPLOADED',total_bytes=manifest['total_bytes'],parts=sum(len(x['parts']) for x in records))),flush=True)

if __name__=='__main__':main()
