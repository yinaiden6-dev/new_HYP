#!/usr/bin/env python3
"""Restore exact model bytes from the private repository's release assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def restore(manifest,root,assets,download=False,keep_parts=False):
    root=root.resolve();assets=assets.resolve();assets.mkdir(parents=True,exist_ok=True)
    for model in manifest['models']:
        target=(root/model['path']).resolve()
        if not target.is_relative_to(root):raise ValueError('Unsafe model destination')
        if target.exists():
            if target.stat().st_size==model['bytes'] and digest(target)==model['sha256']:
                print('Already verified:',model['path'],flush=True);continue
            raise RuntimeError('Existing model differs; refusing overwrite: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.restore.partial')
        whole=hashlib.sha256();count=0
        with temp.open('wb') as output:
            for part in model['parts']:
                name=part['asset']
                if Path(name).name!=name:raise ValueError('Unsafe asset name')
                p=assets/name
                if not p.exists():
                    if not download:raise FileNotFoundError(str(p)+'; pass --download or supply --assets-dir')
                    subprocess.run(['gh','release','download',manifest['release_tag'],'--repo',manifest['repository'],
                                    '--pattern',name,'--dir',str(assets)],check=True)
                if p.stat().st_size!=part['bytes'] or digest(p)!=part['sha256']:
                    raise RuntimeError('Asset checksum mismatch: '+name)
                with p.open('rb') as src:
                    for data in iter(lambda:src.read(8*1024**2),b''):
                        output.write(data);whole.update(data);count+=len(data)
            output.flush();os.fsync(output.fileno())
        if count!=model['bytes'] or whole.hexdigest()!=model['sha256']:
            raise RuntimeError('Reconstructed model checksum mismatch: '+model['path'])
        # Fail if another writer created the final file during restoration.
        os.link(temp,target);temp.unlink()
        print('Restored and SHA256 verified:',model['path'],flush=True)
        if not keep_parts:
            for part in model['parts']:(assets/part['asset']).unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--assets-dir',type=Path)
    parser.add_argument('--download',action='store_true',help='Use authenticated GitHub CLI to fetch missing parts')
    parser.add_argument('--keep-parts',action='store_true')
    args=parser.parse_args()
    path=args.manifest or args.root/'model_assets_manifest.json'
    restore(json.loads(path.read_text()),args.root,args.assets_dir or args.root/'.model_assets_cache',args.download,args.keep_parts)


if __name__=='__main__':main()
