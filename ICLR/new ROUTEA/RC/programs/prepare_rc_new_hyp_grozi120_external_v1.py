#!/usr/bin/env python3
"""Read-only sealed-source intake after explicit user release; no model calls."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[2] / 'external/_staging/grozi120_v1'
OUT = ROOT / 'results/rc_new_hyp_grozi120_external_v1'
PLAN = ROOT / 'plan/RC_NEW_HYP_GROZI120_FROZEN_DATA_V1_20260913.md'
HEADS = ROOT / 'results/rc_new_hyp_external_head_freeze_v1/bundle.json'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def bind(p): return dict(path=str(p), sha256=digest(p))
def key(salt, value): return hashlib.sha256(f'GROZI_EXTERNAL_V1_20260913|{salt}|{value}'.encode()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def prepare():
    receipt = json.loads((SOURCE / 'receipts/source_receipt.json').read_text())
    archives = {}
    for r in receipt['archives']:
        p = SOURCE / 'downloads' / r['name']
        assert digest(p) == r['sha256'] and p.stat().st_size == r['bytes'], 'SOURCE_DRIFT'
        archives[r['name']] = bind(p)
    refs, pool, mapping_count = [], {}, 0
    with zipfile.ZipFile(archives['inVitro.zip']['path']) as z:
        names = z.namelist()
        for identity in range(1, 121):
            candidates = [s for s in names if re.fullmatch(rf'inVitro/{identity}/web/JPEG/web\d+\.jpg', s)]
            assert candidates
            refs.append(dict(identity=identity, member=min(candidates, key=lambda s: key('reference', s))))
    with zipfile.ZipFile(archives['inSitu.zip']['path']) as z:
        names = set(z.namelist())
        for identity in range(1, 121):
            lines = [s for s in z.read(f'inSitu/{identity}/coordinates.txt').decode().splitlines() if s.strip()]
            if identity == 82:
                assert len(lines) == 122 and lines[:61] == lines[61:], 'KNOWN_EXACT_DUPLICATE_BLOCK'
                lines = lines[:61]
            entries = []
            for index, line in enumerate(lines, 1):
                video, frame = map(int, line.split()[:2])
                member = f'inSitu/{identity}/video/video{index}.png'
                assert member in names, 'QUERY_INDEX_MISSING'
                entries.append(dict(identity=identity, video=video, frame=frame, member=member))
            actual = {s for s in names if re.fullmatch(rf'inSitu/{identity}/video/video\d+\.png', s)}
            assert actual == {r['member'] for r in entries}, 'EXACT_IMAGE_COORDINATE_AXIS'
            mapping_count += len(entries)
            by_video = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in entries: by_video[r['video']][r['frame']].append(r)
            candidates = [v for v, frames in by_video.items() if len(frames) >= 4]
            assert candidates
            video = min(candidates, key=lambda v: key('video', f'{identity}|{v}'))
            ordered = [min(rs, key=lambda r: key('frame_instance', r['member'])) for _, rs in sorted(by_video[video].items())]
            indices = [((2 * k + 1) * len(ordered)) // 8 for k in range(4)]
            assert len(set(indices)) == 4
            pool[identity] = [ordered[i] for i in indices]
    queries = [r for identity in sorted(pool) for r in pool[identity]]
    queries.sort(key=lambda r: key('query_order', r['member']))
    assert mapping_count == 11194 and len(queries) == 480
    assert len({r['video'] for r in queries}) == 27
    worker, curator, gallery, assets = [], [], [], []
    for i, r in enumerate(refs):
        rid = f'GZ120-REF-{i:04d}'
        path = str(OUT / 'images' / (rid + '.jpg'))
        gallery.append(dict(reference_id=rid, physical_row=5413 + i, identity=f'GROZI120:{r["identity"]}', image_path=path))
        assets.append(dict(asset_id=rid, role='reference', archive='inVitro.zip', member=r['member'], image_path=path))
    for i, r in enumerate(queries):
        qid = f'GZ120-Q-{i:04d}'
        path = str(OUT / 'images' / (qid + '.png'))
        worker.append(dict(query_id=qid, execution_ordinal=i, image_path=path))
        curator.append(dict(query_id=qid, identity=f'GROZI120:{r["identity"]}', component=f'GZ120-video-{r["video"]}', video=r['video'], frame=r['frame'], source_member=r['member']))
        assets.append(dict(asset_id=qid, role='query', archive='inSitu.zip', member=r['member'], image_path=path))
    write('worker_manifest.json', dict(records=worker, count=480, target_fields=False))
    write('gallery_append_manifest.json', dict(records=gallery, count=120, preserved_legacy_physical_rows=5413))
    write('curator_roles.json', dict(records=curator))
    write('asset_manifest.json', dict(records=assets, archives=archives))
    write('metadata_validation.json', dict(status='GROZI120_FROZEN480_METADATA_PASS_IMAGES_PENDING',
          authorization='User explicitly allowed read-only GroZi-120 external confirmation; no training/tuning.',
          source_receipt=bind(SOURCE / 'receipts/source_receipt.json'), source_marker=bind(SOURCE / 'DO_NOT_CONSUME.md'),
          archives=archives, plan=bind(PLAN), program=bind(Path(__file__)), frozen_head_bundle=bind(HEADS),
          manifests={n: bind(OUT / (n + '.json')) for n in ('worker_manifest', 'gallery_append_manifest', 'curator_roles', 'asset_manifest')},
          complete_query_inventory=11194, query_count=480, reference_count=120, identity_count=120, video_groups=27,
          coordinate_duplicate_fix=dict(identity=82, original_rows=122, exact_repeat_rows=61, retained_rows=61),
          image_decodes=0, model_forwards=0, spatial_fields_exported=[], source_mutations=0, HYP_GO_claimed=False))
    print('GroZi metadata frozen: 480 queries / 120 reference / 120 identities / 27 videos; zero image decodes.')


def extract():
    validation = json.loads((OUT / 'metadata_validation.json').read_text())
    for b in list(validation['manifests'].values()) + [validation['plan'], validation['program'], validation['frozen_head_bundle']]:
        assert digest(Path(b['path'])) == b['sha256'], 'PIN_DRIFT'
    manifest = json.loads((OUT / 'asset_manifest.json').read_text())
    for b in manifest['archives'].values(): assert digest(Path(b['path'])) == b['sha256']
    records = []
    handles = {k: zipfile.ZipFile(b['path']) for k, b in manifest['archives'].items()}
    try:
        for r in manifest['records']:
            data = handles[r['archive']].read(r['member'])
            with Image.open(io.BytesIO(data)) as src:
                image = ImageOps.exif_transpose(src).convert('RGB')
                pixel_sha = hashlib.sha256(str(image.size).encode() + b'|RGB|' + image.tobytes()).hexdigest()
                record = dict(**r, image_sha256=hashlib.sha256(data).hexdigest(), oriented_RGB_sha256=pixel_sha,
                              size=list(image.size), bytes=len(data), exif_orientation=int(src.getexif().get(274, 1)))
            p = Path(r['image_path'])
            assert p.parent == OUT / 'images'
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f: f.write(data)
            records.append(record)
    finally:
        for z in handles.values(): z.close()
    collisions = []
    for kind in ('image_sha256', 'oriented_RGB_sha256'):
        groups = collections.defaultdict(list)
        for r in records: groups[r[kind]].append(r['asset_id'])
        collisions.extend(dict(kind=kind, assets=g) for g in groups.values() if len(g) > 1)
    write('image_receipt.json', dict(status='GROZI600_EXTRACTED' if not collisions else 'GROZI600_EXTRACTED_COLLISION_REVIEW_REQUIRED',
          records=records, collisions=collisions, source_archives=manifest['archives'],
          metadata_validation=bind(OUT / 'metadata_validation.json'), model_forwards=0,
          source_mutations=0, image_visual_inspections=0, historical_overlap_check='PENDING', HYP_GO_claimed=False))
    print(json.dumps(dict(extracted=len(records), collisions=collisions, model_forwards=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extract'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else extract()
