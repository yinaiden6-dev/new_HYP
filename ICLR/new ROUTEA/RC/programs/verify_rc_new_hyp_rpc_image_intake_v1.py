#!/usr/bin/env python3
"""Verify the complete fixed RPC image intake; no model imports or scoring."""
import collections
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_new_hyp_rpc_transfer_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    assert bind(b['path'])==b
    return read(b['path'])
def main():
    metadata=read(OUT/'metadata_validation.json')
    for b in metadata['manifest_sources'].values():checked(b)
    manifest=read(OUT/'download_manifest.json')
    complete=read(OUT/'download_receipt.json')
    assert complete['status']=='RPC800_PUBLIC_IMAGES_RECEIVED_NOT_SCORED'
    assert complete['manifest_sha256']==sha(OUT/'download_manifest.json')
    assert complete['program_sha256']==sha(ROOT/'programs/download_rc_new_hyp_rpc_transfer_v1.py')
    assert len(complete['files'])==len(manifest['records'])==800
    saved={r['asset_id']:r for r in complete['files']}
    byte_groups=collections.defaultdict(list);pixel_groups=collections.defaultdict(list)
    for row in manifest['records']:
        receipt=read(OUT/'downloads'/(row['asset_id']+'.json'))
        assert receipt==saved[row['asset_id']]
        assert all(receipt[k]==row[k] for k in ('asset_id','role','image_path','source_filename'))
        p=Path(row['image_path']);assert sha(p)==receipt['image_sha256'] and p.stat().st_size==receipt['bytes']
        with Image.open(p) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB')
            assert list(rgb.size)==receipt['size']
            digest=hashlib.sha256(str(rgb.size).encode()+b'|RGB|'+rgb.tobytes()).hexdigest()
            assert digest==receipt['oriented_RGB_sha256']
        byte_groups[receipt['image_sha256']].append(row['asset_id'])
        pixel_groups[digest].append(row['asset_id'])
    workers=read(OUT/'worker_manifest.json')['records'];gallery=read(OUT/'gallery_manifest.json')['records'];roles=read(OUT/'curator_roles.json')['records']
    assert len(workers)==len(roles)==600 and len(gallery)==200
    assert all(set(w)=={'query_id','execution_ordinal','image_path'} for w in workers)
    assert [w['execution_ordinal'] for w in workers]==list(range(600))
    assert {w['query_id'] for w in workers}=={r['query_id'] for r in roles}
    assert set(r['identity'] for r in roles)==set(g['identity'] for g in gallery)
    groups=collections.defaultdict(list)
    for r in roles:groups[r['identity']].append(r)
    assert all(len(v)==3 and {r['camera'] for r in v}=={1,2,3} and len({r['stratum'] for r in v})==1 for v in groups.values())
    assert len({r['stratum'] for r in roles})==17
    byte_duplicates=[v for v in byte_groups.values() if len(v)>1]
    pixel_duplicates=[v for v in pixel_groups.values() if len(v)>1]
    ledger_path=ROOT/'cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json'
    ledger=read(ledger_path)['queries'];assert len(ledger)==987
    query_hits=[r['query_id'] for r in ledger if r['source_image_sha256'] in byte_groups]
    previous_path=ROOT/'results/rc_new_hyp_grozi120_external_v1/image_receipt.json'
    previous=read(previous_path)['records'];assert len(previous)==600
    grozi_hits=[r['asset_id'] for r in previous if r['image_sha256'] in byte_groups or r['oriented_RGB_sha256'] in pixel_groups]
    gallery_root=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
    oldpaths=[]
    for directory,subdirs,names in os.walk(gallery_root):
        subdirs.sort()
        for name in sorted(names):
            if name.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):oldpaths.append(Path(directory)/name)
    assert len(oldpaths)==5413
    gallery_hits=[str(p) for p in oldpaths if sha(p) in byte_groups]
    qualified=not any((byte_duplicates,pixel_duplicates,query_hits,grozi_hits,gallery_hits))
    result=dict(status='RPC800_IMAGE_INTAKE_PASS' if qualified else 'RPC800_IMAGE_INTAKE_CONFLICT',timestamp_utc=datetime.now(timezone.utc).isoformat(),program=bind(__file__),download_receipt=bind(OUT/'download_receipt.json'),metadata_validation=bind(OUT/'metadata_validation.json'),images=800,queries=600,references=200,identities=200,supercategories=17,byte_duplicates=byte_duplicates,RGB_duplicates=pixel_duplicates,historical_query_hits=query_hits,historical_gallery_hits=gallery_hits,grozi_hits=grozi_hits,query_ledger=bind(ledger_path),grozi_receipt=bind(previous_path),training_updates=0,model_inferences=0,images_visualized=0,limits=['Exact duplicates only; semantic SKU overlap and pretraining exposure are not excluded.'])
    with (OUT/'image_intake_validation_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    assert qualified,'FROZEN_INTAKE_CONFLICT_NO_REPLACEMENT'
if __name__=='__main__':main()
