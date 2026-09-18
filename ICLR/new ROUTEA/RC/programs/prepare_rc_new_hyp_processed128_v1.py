#!/usr/bin/env python3
"""Freeze outcome-blind processed query subset and unchanged runtime sources."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parents[2]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
from rc_aslo_xf.gallery_identity_repair import build_identity_map

OUT = ROOT / 'results/rc_new_hyp_processed128_regression_v1'
AUTH = ROOT / 'registry/rc_new_hyp_processed128_authority_v1_20260916.json'
SEED = 'processed128-regression-20260916-v1'


def key(value):
    return hashlib.sha256((SEED + '\0' + value).encode()).hexdigest()


def prepare():
    source = WS / '1/processed'
    mapping = source / 'origin name/processed_origin_name.csv'
    with mapping.open(newline='') as f:
        rows = list(csv.DictReader(f))
    actual = {p.name for p in source.iterdir() if p.is_file()}
    assert len(rows) == len(actual) == 2000
    assert {r['file_name'] for r in rows} == actual
    assert len({r['file_name'] for r in rows}) == 2000
    paths = []
    gallery_root = WS / 'dailymed/data/box_flat_20000_images/data/raw_images'
    for directory, subdirs, names in os.walk(gallery_root):
        subdirs.sort()
        paths.extend(Path(directory) / n for n in sorted(names)
                     if (Path(directory) / n).is_file()
                     and Path(n).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
    assert len(paths) == 5413
    identity_map = build_identity_map([p.stem.strip() for p in paths])
    identities = identity_map.labels
    gallery = [dict(physical_row=i, identity=identities[i], image_path=str(p)) for i, p in enumerate(paths)]
    by_name = collections.defaultdict(list)
    for g in gallery:
        by_name[Path(g['image_path']).stem].append(g)
    processed = collections.defaultdict(list)
    for r in rows:
        r['source_stem'] = r['file_name'].rsplit('__', 1)[0]
        assert r['origin_name'].rstrip(' _') == r['source_stem'].rstrip(' _')
        assert Path(r['image_path']) == source / r['file_name']
        processed[r['source_stem']].append(r)
    eligible = sorted((name for name in processed if len(by_name[name]) == 1), key=key)
    excluded = [dict(origin_name=name, images=len(processed[name]), gallery_matches=by_name[name],
                     reason='AMBIGUOUS_REFERENCE_NAME' if by_name[name] else 'NO_EXACT_REFERENCE_NAME')
                for name in sorted(processed) if len(by_name[name]) != 1]
    selected = [min(processed[name], key=lambda r: key(r['file_name'])) for name in eligible[:128]]
    assert len(selected) == 128
    # Outcome-blind selection is immutable before opening selected image bytes.
    M.write(OUT / 'selection.json', dict(status='PROCESSED128_OUTCOME_BLIND_SELECTION_FROZEN',
            seed=SEED, mapping=M.bind(mapping), total_images=2000, original_names=len(processed),
            eligible_names=len(eligible), eligible_images=sum(len(processed[n]) for n in eligible),
            selected=selected, exclusions=excluded, outcome_reads=0))
    M.write(OUT / 'gallery_manifest.json', dict(records=gallery, physical_count=5413, identity_count=5412,
            corrected_mapping_sha256=identity_map.corrected_row_identity_mapping_sha256))
    training_path = ROOT / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    training = M.read(training_path)['records']
    train_identities = {r['identity'] for r in training}
    train_bytes = {r['source_image_sha256'] for r in training}
    xml_path = WS / '1/image_xml_mapping.csv'
    xmls = collections.defaultdict(set)
    with xml_path.open(newline='') as f:
        for r in csv.DictReader(f):
            xmls[(r['category'], r['image_file'])].add(r['category'] + '/' + r['xml_file'])
    # All uniquely mapped sources, not just sampled ones, connect shared XML sources.
    parent = {name: name for name in eligible}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    xml_owner = {}
    name_xmls = {}
    for name in sorted(eligible):
        p = Path(by_name[name][0]['image_path'])
        xx = sorted(xmls[(p.parent.name, p.name)])
        name_xmls[name] = xx
        for x in xx:
            if x in xml_owner:
                a, b = find(name), find(xml_owner[x]); parent[max(a, b)] = min(a, b)
            else:
                xml_owner[x] = name
    worker, curator, receipts = [], [], []
    alias_root = OUT / 'images'; alias_root.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for i, r in enumerate(selected):
        path = Path(r['image_path']); sha = M.sha(path)
        qid = f'PROC-Q-{i:04d}'
        alias = alias_root / (qid + path.suffix.lower())
        alias.symlink_to(path)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            size = list(im.size); orientation = int(im.getexif().get(274, 1))
        identity = by_name[r['source_stem']][0]['identity']
        worker.append(dict(query_id=qid, execution_ordinal=i, query_image_path=str(alias),
                           source_image_sha256=sha, track='new_difficult_train'))
        corruption = path.stem.rsplit('__', 1)[1].rsplit('_', 1)[0]
        curator.append(dict(query_id=qid, execution_ordinal=i, identity=identity,
                            origin_name=r['origin_name'], source_stem=r['source_stem'], original_path=str(path),
                            target_physical_row=by_name[r['source_stem']][0]['physical_row'],
                            component=key(find(r['source_stem'])), source_xmls=name_xmls[r['source_stem']],
                            corruption=corruption, training_identity_overlap=identity in train_identities,
                            training_image_byte_overlap=sha in train_bytes))
        receipts.append(dict(query_id=qid, source_image_sha256=sha, size_wh=size, exif_orientation=orientation))
    assert len({r['identity'] for r in curator}) == 128
    assert len({r['source_image_sha256'] for r in worker}) == 128
    M.write(OUT / 'worker_manifest.json', dict(records=worker, queries=128, shards=16, per_shard=8))
    M.write(OUT / 'curator_roles.json', dict(records=curator, training_source=M.bind(training_path), xml_source=M.bind(xml_path)))
    M.write(OUT / 'image_receipt.json', dict(records=receipts))
    stats = dict(status='PROCESSED128_METADATA_IMAGE_INTEGRITY_PASS', queries=128,
                 eligible_origins=len(eligible), excluded_origins=len(excluded),
                 excluded_images=sum(x['images'] for x in excluded),
                 sampled_components=len({r['component'] for r in curator}),
                 training_identity_overlap=sum(r['training_identity_overlap'] for r in curator),
                 training_image_byte_overlap=sum(r['training_image_byte_overlap'] for r in curator),
                 corruptions=dict(collections.Counter(r['corruption'] for r in curator)),
                 worker=M.bind(OUT / 'worker_manifest.json'), selection=M.bind(OUT / 'selection.json'),
                 raw_forwards=0, outcome_reads=0)
    M.write(OUT / 'metadata_validation.json', stats)
    print(json.dumps(stats, ensure_ascii=False))


def freeze():
    old = M.read(ROOT / 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json')
    bundle_path = ROOT / 'results/rc_new_hyp_external_head_freeze_v1/bundle.json'
    bundle = M.read(bundle_path)
    sources = {k: old['sources'][k] for k in ('feature_program', 'old_preflight', 'materializer',
               'raw_program', 'roma_program', 'roma_profile')}
    files = dict(program=ROOT / 'programs/run_rc_new_hyp_processed128_v1.py',
                 join_program=ROOT / 'programs/analyze_rc_new_hyp_processed128_v1.py',
                 prepare_program=Path(__file__),
                 preflight_program=ROOT / 'programs/preflight_rc_new_hyp_processed128_v1.py',
                 plan=ROOT / 'plan/RC_NEW_HYP_PROCESSED128_REGRESSION_V1_20260916.md',
                 launcher=ROOT / 'slurm/rc_new_hyp_processed128_v1.sbatch',
                 heads_bundle=bundle_path)
    files.update({k: OUT / f'{v}.json' for k, v in dict(worker='worker_manifest', gallery='gallery_manifest',
                 image_receipt='image_receipt', metadata_validation='metadata_validation', selection='selection').items()})
    sources.update({k: M.bind(p) for k, p in files.items()})
    raw_body = M.segment(M.RAW_SOURCE, 'run_gpu', ' from colpali_engine.models', ' parity=[original_parity')
    M.write(AUTH, dict(status='PROCESSED128_FROZEN_REGRESSION_AUTHORIZED', sources=sources,
            operator_sources=old['operator_sources'], heads=bundle['heads'],
            curator=M.bind(OUT / 'curator_roles.json'), primary='COST1', secondary='CE',
            raw_numeric_loop_sha256=hashlib.sha256(raw_body.encode()).hexdigest(),
            roma_numeric_loop_sha256=old['roma_numeric_loop_sha256'], queries=128, shards=16,
            training_updates=0, target_insertion=False, retune_threshold=False))
    print(json.dumps(dict(authority=M.bind(AUTH))))


if __name__ == '__main__':
    {'prepare': prepare, 'freeze': freeze}[sys.argv[1]]()
