#!/usr/bin/env python3
"""Metadata-only deterministic external retrieval manifests; no spatial fields."""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_new_hyp_external_head_freeze_v1/data_intake/rpc_train_source_metadata.json'
OUT = ROOT / 'results/rc_new_hyp_rpc_transfer_v1'
PLAN = ROOT / 'plan/RC_NEW_HYP_RPC_TRANSFER_V1_20260913.md'


def bind(p):
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def key(salt, text):
    return hashlib.sha256(('RPC_TRANSFER_V1_20260913|' + salt + '|' + text).encode()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def main():
    src = json.loads(SOURCE.read_text())
    labels = {}
    for a in src['annotations']:
        assert a['image_id'] not in labels, 'SINGLE_PRODUCT_REQUIRED'
        labels[a['image_id']] = a['category_id']
    cats = {c['id']: c for c in src['categories']}
    groups = collections.defaultdict(list)
    images = {}
    for im in src['images']:
        assert im['id'] not in images and im['id'] in labels
        match = re.fullmatch(r'(.+)_camera([0-3])-(\d+)\.jpg', im['file_name'])
        assert match, 'UNEXPECTED_FILENAME'
        row = dict(image_id=im['id'], filename=im['file_name'], camera=int(match[2]),
                   back='-back' in match[1], identity=labels[im['id']])
        images[im['id']] = row
        groups[labels[im['id']]].append(row)
    assert len(images) == 53739 and len(groups) == len(cats) == 200
    gallery, queries = [], []
    for cid in sorted(cats):
        pool = [r for r in groups[cid] if r['camera'] == 0 and not r['back']]
        assert pool, ('NO_ENROLLMENT_IMAGE', cid)
        gallery.append(min(pool, key=lambda r: key('reference', r['filename'])))
        for camera in (1, 2, 3):
            pool = [r for r in groups[cid] if r['camera'] == camera]
            assert pool, ('MISSING_CAMERA', cid, camera)
            queries.append(min(pool, key=lambda r: key('query', r['filename'])))
    queries.sort(key=lambda r: key('query_order', r['filename']))
    assert len({r['image_id'] for r in gallery + queries}) == 800
    downloads, worker, curator, references = [], [], [], []
    for ordinal, r in enumerate(gallery):
        identifier = f'RPC-REF-{ordinal:04d}'
        path = str(OUT / 'images' / (identifier + '.jpg'))
        references.append(dict(reference_id=identifier, physical_row=ordinal,
                               identity=f'RPC-SKU-{r["identity"]:03d}', image_path=path))
        downloads.append(dict(asset_id=identifier, image_path=path, source_filename=r['filename'], role='reference'))
    for ordinal, r in enumerate(queries):
        identifier = f'RPC-Q-{ordinal:04d}'
        path = str(OUT / 'images' / (identifier + '.jpg'))
        worker.append(dict(query_id=identifier, execution_ordinal=ordinal, image_path=path))
        curator.append(dict(query_id=identifier, identity=f'RPC-SKU-{r["identity"]:03d}',
                            component=f'RPC-SKU-{r["identity"]:03d}',
                            stratum=cats[r['identity']]['supercategory'], camera=r['camera'],
                            source_image_id=r['image_id'], source_filename=r['filename']))
        downloads.append(dict(asset_id=identifier, image_path=path, source_filename=r['filename'], role='query'))
    source = bind(SOURCE)
    write('worker_manifest.json', dict(records=worker, count=600, labels_included=False))
    write('gallery_manifest.json', dict(records=references, count=200))
    write('curator_roles.json', dict(records=curator, source=source))
    write('download_manifest.json', dict(records=downloads, source=source,
          endpoint_prefix='https://www.kaggle.com/api/v1/datasets/download/diyer22/retail-product-checkout-dataset/',
          dataset_path_prefix='retail_product_checkout/train2019/'))
    write('metadata_validation.json', dict(status='RPC600_METADATA_SPLIT_FROZEN_IMAGES_PENDING',
          source=source, plan=bind(PLAN), program=bind(Path(__file__)),
          manifest_sources={n: bind(OUT / (n + '.json')) for n in ('worker_manifest', 'gallery_manifest', 'curator_roles', 'download_manifest')},
          query_count=600, reference_count=200, identity_count=200, strata=17,
          query_cameras=[1, 2, 3], enrollment_camera=0, images_read=0, scores_computed=0,
          spatial_annotation_fields_exported=[], HYP_GO_claimed=False))
    print('RPC600 metadata frozen: 200 references, 600 queries, 200 SKU, 17 strata; zero images read.')


if __name__ == '__main__':
    main()
