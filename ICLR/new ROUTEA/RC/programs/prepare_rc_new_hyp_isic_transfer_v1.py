#!/usr/bin/env python3
"""Freeze an opened ISIC cohort using image/lesion/patient IDs only."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[2] / 'data/isic_ima_pp'
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1'
PLAN = ROOT / 'plan/RC_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def bind(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def key(salt, text):
    return hashlib.sha256(('ISIC_TRANSFER_V1_20260914|' + salt + '|' + text).encode()).hexdigest()


def main():
    meta_path = SOURCE / 'img_metadata.csv'
    # Ignore all diagnosis/spatial/demographic columns while parsing metadata.
    with meta_path.open() as f:
        meta = {r['isic_id']: {k: r[k] for k in ('isic_id', 'lesion_id', 'patient_id')}
                for r in csv.DictReader(f)}
    grouped = collections.defaultdict(list)
    local = sorted((SOURCE / 'images').glob('*.jpg'))
    assert len(local) == 989, 'LOCAL_POPULATION_DRIFT'
    for p in local:
        r = dict(meta[p.stem], source_path=str(p))
        grouped[r['lesion_id']].append(r)
    selected, excluded = {}, []
    for lesion, rows in grouped.items():
        reason = ('missing_lesion' if not lesion else 'less_than_two_images' if len(rows) < 2
                  else 'missing_patient' if any(not r['patient_id'] for r in rows)
                  else 'inconsistent_patient' if len({r['patient_id'] for r in rows}) != 1 else None)
        if reason: excluded.append(dict(lesion_id=lesion, images=len(rows), reason=reason))
        else: selected[lesion] = rows
    assert len(selected) == 390 and sum(map(len, selected.values())) == 927
    patients = sorted({rows[0]['patient_id'] for rows in selected.values()})
    assert len(patients) == 346
    lesions = sorted(selected, key=lambda x: key('lesion-order', x))
    refs, queries = [], []
    for lesion in lesions:
        rows = sorted(selected[lesion], key=lambda r: key('reference', r['isic_id']))
        refs.append(rows[0]); queries.extend(rows[1:])
    queries.sort(key=lambda r: key('query-order', r['isic_id']))
    assert len(queries) == 537
    write(OUT / 'selection_preimage_seal.json', dict(plan=bind(PLAN), metadata=bind(meta_path),
          program=bind(__file__), selection='fixed IDs before image decoding',
          references=[r['isic_id'] for r in refs], queries=[r['isic_id'] for r in queries],
          excluded=excluded, image_forwards=0))
    gallery, worker, roles, files = [], [], [], []
    byte_seen, pixel_seen = {}, {}
    for role, rows in [('reference', refs), ('query', queries)]:
        for i, r in enumerate(rows):
            identifier = f"ISIC-{'REF' if role == 'reference' else 'Q'}-{i:04d}"
            p = Path(r['source_path']); byte_hash = sha(p)
            with Image.open(p) as raw:
                rgb = ImageOps.exif_transpose(raw).convert('RGB')
                digest = hashlib.sha256()
                digest.update(f'{rgb.width},{rgb.height}|RGB|'.encode())
                digest.update(rgb.tobytes()); pixel_hash = digest.hexdigest()
                size = list(rgb.size)
            assert byte_hash not in byte_seen, ('DUPLICATE_BYTES', p, byte_seen.get(byte_hash))
            assert pixel_hash not in pixel_seen, ('DUPLICATE_PIXELS', p, pixel_seen.get(pixel_hash))
            byte_seen[byte_hash], pixel_seen[pixel_hash] = str(p), str(p)
            dest = OUT / 'images' / (identifier + '.jpg')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with p.open('rb') as src, dest.open('xb') as dst: shutil.copyfileobj(src, dst)
            assert sha(dest) == byte_hash
            files.append(dict(asset_id=identifier, role=role, image_path=str(dest),
                              source_path=str(p), image_sha256=byte_hash, rgb_sha256=pixel_hash, size=size))
            identity = f'ISIC-LESION-{lesions.index(r["lesion_id"]):04d}'
            if role == 'reference':
                gallery.append(dict(reference_id=identifier, physical_row=i, identity=identity, image_path=str(dest)))
            else:
                worker.append(dict(query_id=identifier, execution_ordinal=i, image_path=str(dest)))
                roles.append(dict(query_id=identifier, identity=identity,
                    component='ISIC-PATIENT-' + key('patient', r['patient_id'])[:20],
                    source_image_id=r['isic_id']))
            if (len(files) % 100) == 0: print('checked_images', len(files), flush=True)
    write(OUT / 'gallery_manifest.json', dict(count=390, records=gallery))
    write(OUT / 'worker_manifest.json', dict(count=537, labels_included=False, records=worker))
    write(OUT / 'curator_roles.json', dict(records=roles, source=bind(meta_path)))
    write(OUT / 'image_receipt.json', dict(status='ISIC927_IMAGES_RECEIVED_NOT_SCORED', files=files,
          source=bind(meta_path), cohort_history='previously opened selected IMA++ cohort'))
    write(OUT / 'dataset_manifest.json', dict(reference_count=390, query_count=537, patient_count=346,
          shard_size=8, shard_count=68, last_shard_size=1, candidate_count=128))
    write(OUT / 'image_intake_validation_v1.json', dict(status='ISIC927_IMAGE_INTAKE_PASS',
          selection=bind(OUT / 'selection_preimage_seal.json'), images=bind(OUT / 'image_receipt.json'),
          gallery=bind(OUT / 'gallery_manifest.json'), worker=bind(OUT / 'worker_manifest.json'),
          byte_unique=927, exif_rgb_unique=927, excluded_images=62, excluded_lesions=28,
          reference_count=390, query_count=537, patient_count=346, training_updates=0,
          mask_reads=0, diagnosis_used=False, untouched_external_claim=False,
          independently_captured_provenance='distinct ISIC images of same lesion; capture sessions not independently verified',
          near_duplicate_exclusion_claimed=False))
    print('ISIC927_IMAGE_INTAKE_PASS', flush=True)


if __name__ == '__main__': main()
