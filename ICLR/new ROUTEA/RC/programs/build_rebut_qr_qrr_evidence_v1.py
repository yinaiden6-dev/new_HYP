#!/usr/bin/env python3
"""CPU-only, label-free pooled-F71 evidence for matched QR/QR-vec/QRR arms.

Warp sampling is performed on original token cells BEFORE fixed 8x8 pooling.
No backbone calls, source-candidate truncation, or certainty thresholding.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / 'results/rc_h593_feature_fusion_cache_v1'
COORDINATES = ROOT / 'results/rc_h593_roma_coordinate_precision_v2'
OUT = ROOT / 'results/rc_rebut_qr_qrr_v1/evidence'
MANIFEST = OUT.parent / 'evidence_manifest.json'
SEED = 20260927
GRID = 8
WIDTH = 32
DIM = 72
VERSION = 'POOLED_F71_COARSE17_NATIVE_CENTER_WARP_V1'
CHANNELS = [
    dict(name='self_descriptor', start=0, stop=32),
    dict(name='other_descriptor_sampled_at_native_warp', start=32, stop=64),
    dict(name='self_support', start=64, stop=65),
    dict(name='other_support_sampled_at_native_warp', start=65, stop=66),
    dict(name='self_cell_center_xy_normalized_minus1_plus1', start=66, stop=68),
    dict(name='mapped_other_center_xy_normalized_minus1_plus1', start=68, stop=70),
    dict(name='source_geometry_valid_fraction', start=70, stop=71),
    dict(name='mapping_valid_fraction_of_source_valid', start=71, stop=72),
]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def same_sha(path, expected):
    actual = bind(path)
    need(actual['sha256'] == expected['sha256'], 'SOURCE_SHA:' + str(path))
    return actual


def tensor_sha(value):
    a = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def install_label_guard():
    """Builder may see linkage IDs and frozen scores, never identity labels."""
    forbidden = ('curator_roles', 'heldout_roles', 'train_roles', '/target_join/',
                 'm_scalar_headroom', 'failure_audit', '/grozi/', '/isic/',
                 'd1-mi', 'd1_mi', 'formal392')
    import sys
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).lower()
            need(not any(s in name for s in forbidden), 'LABEL_OR_PROTECTED_READ:' + name)
    sys.addaudithook(audit)


def projection():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    return ((torch.randint(0, 2, (1024, WIDTH), generator=gen).float() * 2 - 1)
            / math.sqrt(WIDTH))


def geometry_index(boxes, valid):
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    need(boxes.shape == (len(valid), 4) and np.isfinite(boxes).all(), 'BOX_DOMAIN')
    centers = (boxes[:, :2] + boxes[:, 2:]) * .5
    area = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    need(not np.any(valid & ~area), 'VALID_CELL_HAS_AREA')
    source_valid = valid & area
    grid_xy = np.floor(centers * GRID).astype(np.int64).clip(0, GRID - 1)
    bins = grid_xy[:, 1] * GRID + grid_xy[:, 0]
    usable = np.flatnonzero(area)
    need(len(usable) > 0, 'GEOMETRY_HAS_CELLS')
    return dict(boxes=boxes, centers=centers, source_valid=source_valid, bins=bins,
                tree=cKDTree(centers[usable]), tree_indices=usable,
                geometric_cell_count=np.bincount(bins, minlength=GRID * GRID))


def locate(other, warp):
    """Nearest cell center with explicit containment, bounds, and padding checks.

    Existing token cells partition a transformed regular image grid. Checking
    actual oriented boxes prevents implicit raw-frame/EXIF reshape assumptions.
    """
    warp = np.asarray(warp, dtype=np.float64)
    finite = np.isfinite(warp).all(axis=1)
    in_bounds = finite & (np.abs(warp) <= 1).all(axis=1)
    point = (np.where(finite[:, None], warp, 0) + 1) * .5
    _, local = other['tree'].query(point, k=1, workers=1)
    index = other['tree_indices'][local]
    box = other['boxes'][index]
    contained = ((point >= box[:, :2] - 1e-12) & (point <= box[:, 2:] + 1e-12)).all(axis=1)
    mapped = in_bounds & contained & other['source_valid'][index]
    return index, mapped


def pooled(values, bins, valid):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    total = np.zeros((GRID * GRID, values.shape[1]), dtype=np.float64)
    np.add.at(total, bins[valid], values[valid])
    n = np.bincount(bins[valid], minlength=GRID * GRID)
    return total / np.maximum(n[:, None], 1), n


def side_evidence(own, other, own_support, other_support, warp):
    own_support = np.asarray(own_support, dtype=np.float64)
    other_support = np.asarray(other_support, dtype=np.float64)
    warp = np.asarray(warp, dtype=np.float64)
    need(own_support.shape == (len(own['boxes']),) and warp.shape == (len(own_support), 2), 'OWN_AXIS')
    need(other_support.shape == (len(other['boxes']),), 'OTHER_AXIS')
    need(np.isfinite(own_support).all() and np.isfinite(other_support).all(), 'FINITE_SUPPORT')
    need(((own_support >= 0) & (own_support <= 1)).all() and
         ((other_support >= 0) & (other_support <= 1)).all(), 'SUPPORT_RANGE')
    index, mapped = locate(other, warp)
    valid = own['source_valid']; mapped &= valid
    a, counts = pooled(own['projected'], own['bins'], valid)
    b, map_counts = pooled(other['projected'][index], own['bins'], mapped)
    u, _ = pooled(own_support, own['bins'], valid)
    v, _ = pooled(other_support[index], own['bins'], mapped)
    xy, _ = pooled(own['centers'] * 2 - 1, own['bins'], valid)
    wxy, _ = pooled(warp, own['bins'], mapped)
    fractions = np.column_stack((counts / np.maximum(own['geometric_cell_count'], 1),
                                 map_counts / np.maximum(counts, 1)))
    x = np.concatenate((a, b, u, v, xy, wxy, fractions), axis=1).astype(np.float32)
    need(x.shape == (64, DIM) and np.isfinite(x).all(), 'SIDE_EVIDENCE_SHAPE')
    need((x[counts == 0] == 0).all(), 'EMPTY_CELL_ZERO')
    need((x[map_counts == 0, 32:64] == 0).all(), 'UNMAPPED_DESCRIPTOR_ZERO')
    need((x[map_counts == 0, 65] == 0).all(), 'UNMAPPED_SUPPORT_ZERO')
    return x, counts > 0, map_counts > 0, dict(source_tokens=int(valid.sum()),
        mapping_tokens=int(mapped.sum()), source_bins=int((counts > 0).sum()),
        mapping_bins=int((map_counts > 0).sum()), warp_index_sha256=tensor_sha(index),
        mapping_mask_sha256=tensor_sha(mapped))


class ImageBank:
    def __init__(self, ready, matrix):
        self.ready = ready; self.matrix = matrix; self.cache = OrderedDict()
        self.used = {}

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key); self.cache[key] = value; return value
        source = self.ready['features'][key]
        path = OUT / 'projected_images' / f'{key}.pt'
        if path.exists():
            p = load(path)
            need(p['version'] == VERSION and p['source'] == source and
                 p['projection_seed'] == SEED, 'PROJECTED_BANK_RESUME')
        else:
            original = load(source['path'])
            need(original['item']['key'] == key, 'IMAGE_KEY')
            f = original['components']['coarse_17']
            need(f.ndim == 2 and f.shape[1] == 1024 and torch.isfinite(f).all(), 'COARSE17')
            projected = f.float() @ self.matrix
            p = dict(version=VERSION, source=source, item=original['item'], geometry=original['geometry'],
                projected=projected, boxes=torch.as_tensor(original['cell_boxes_xyxy']).clone(),
                valid=torch.as_tensor(original['valid_patch_mask']).bool().clone(),
                projection_seed=SEED, projection_sha256=tensor_sha(self.matrix),
                source_component='coarse_17', source_component_sha256=tensor_sha(f),
                source_boxes_sha256=tensor_sha(original['cell_boxes_xyxy']),
                source_valid_sha256=tensor_sha(original['valid_patch_mask']),
                labels_read=0, backbone_forwards=0)
            atomic_torch(path, p)
        info = geometry_index(p['boxes'], p['valid'])
        info.update(projected=p['projected'].numpy(), item=p['item'], geometry=p['geometry'],
                    source=source, projected_binding=bind(path))
        self.used[key] = dict(source=source, projected=info['projected_binding'],
                             consumed_descriptor_sha256=p['source_component_sha256'])
        self.cache[key] = info
        if len(self.cache) > 192:
            self.cache.popitem(last=False)
        return info


def check_cell_binding(image, coordinate, weights):
    boxes = np.asarray(coordinate['cell_boxes_xyxy'])
    valid = np.asarray(coordinate['valid_patch_mask'])
    need(np.array_equal(image['boxes'], boxes), 'DESCRIPTOR_WARP_BOXES')
    need(np.array_equal(image['source_valid'], valid), 'DESCRIPTOR_WARP_VALIDITY')
    need(len(weights) == len(boxes), 'SUPPORT_DESCRIPTOR_AXIS')


def build_query(index, catalog_rows, bank):
    path = OUT / f'query{index:03d}.pt'
    receipt_path = path.with_suffix('.json')
    if path.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        need(receipt['version'] == VERSION and receipt['status'] == 'POOLED_F_QUERY_PASS', 'QUERY_RESUME')
        same_sha(path, receipt['payload'])
        print(json.dumps(dict(stage='resume', index=index, path=str(path))), flush=True)
        return receipt
    start = time.monotonic()
    entry = catalog_rows[index]
    original = load(entry['payload']['path'])
    cp = COORDINATES / f'query{index:03d}'
    validation = read(cp / 'validation.json')
    metadata = read(cp / 'payload.json')
    need(validation['status'] == 'ROMA_COORDINATE_QUERY_PASS', 'COORD_VALIDATION')
    same_sha(cp / 'payload.json', validation['payload'])
    need(entry['query_id'] == original['query_id'] == metadata['query_id'], 'QUERY_JOIN')
    need(entry['execution_ordinal'] == original['execution_ordinal'] == metadata['execution_ordinal'] == index, 'ORDINAL')
    axis = list(original['candidate_physical_rows'])
    need(axis == metadata['candidate_physical_rows'] and len(axis) == len(set(axis)) == 128, 'C128_AXIS')
    qi = bank.get(original['query_image_key'])
    pair_data = original['pairs']
    rows = []; summaries = []; parts = []; consumed = []
    keys = {original['query_image_key']}
    for part_binding in metadata['parts']:
        chunk = load(part_binding['path'])
        need(chunk['query_id'] == entry['query_id'], 'PART_QUERY')
        parts.append(part_binding)
        for pair in chunk['pairs']:
            pos = pair['candidate_position']; old = pair_data[pos]
            need(pos == len(rows) and pair['physical_row'] == axis[pos] == old['physical_row'], 'PAIR_AXIS')
            ri = bank.get(old['image_key']); keys.add(old['image_key'])
            need(ri['item']['tokens_sha256'] == pair['reference_tokens_sha256'], 'REF_TOKENS_BIND')
            native = pair['arms']['NATIVE']; u = native['query_visibility']; v = native['reference_visibility']
            need(torch.equal(u, old['query_visibility']) and torch.equal(v, old['reference_visibility']), 'OLD_UV_BITS')
            qcoord = native['intermediate']['query_coordinates']; rcoord = native['intermediate']['reference_coordinates']
            check_cell_binding(qi, qcoord, u); check_cell_binding(ri, rcoord, v)
            qe, qvalid, qmapped, qa = side_evidence(qi, ri, u.numpy(), v.numpy(), qcoord['center_xy'].numpy())
            re, rvalid, rmapped, ra = side_evidence(ri, qi, v.numpy(), u.numpy(), rcoord['center_xy'].numpy())
            m0 = float(torch.sqrt(u.mean() * v.mean()))
            need(m0.hex() == float(native['scores']['visibility_mass']).hex(), 'M0_BITS')
            rows.append((qe, re, qvalid, rvalid, qmapped, rmapped, m0))
            summaries.append(dict(position=pos, physical_row=axis[pos], query=qa, reference=ra))
            consumed.append(dict(position=pos, physical_row=axis[pos], reference_image_key=old['image_key'],
                query_visibility_sha256=tensor_sha(u), reference_visibility_sha256=tensor_sha(v),
                warp_AB_center_sha256=tensor_sha(qcoord['center_xy']),
                warp_BA_center_sha256=tensor_sha(rcoord['center_xy'])))
    need(len(rows) == 128, 'ALL128')
    columns = list(zip(*rows))
    payload = dict(version=VERSION, query_id=entry['query_id'], execution_ordinal=index,
        axis=axis, winner=int(original['winner']), challenger_positions=original['challenger_positions'],
        query=torch.from_numpy(np.stack(columns[0])), reference=torch.from_numpy(np.stack(columns[1])),
        query_valid=torch.from_numpy(np.stack(columns[2])), reference_valid=torch.from_numpy(np.stack(columns[3])),
        query_mapping_valid=torch.from_numpy(np.stack(columns[4])),
        reference_mapping_valid=torch.from_numpy(np.stack(columns[5])),
        M0=torch.tensor(columns[6], dtype=torch.float64),
        native_X=torch.tensor(metadata['modes']['NATIVE']['X'], dtype=torch.float64),
        raw_candidate_scores=torch.tensor(original['candidate_raw_scores'], dtype=torch.float64),
        source=dict(fusion_query=entry['payload'], coordinate_validation=bind(cp / 'validation.json'),
                    coordinate_payload=validation['payload'], coordinate_parts=parts,
                    images={key: bank.used[key] for key in sorted(keys)}, consumed_arrays=consumed),
        channels=CHANNELS, projection_seed=SEED, projection_sha256=tensor_sha(bank.matrix),
        grid=[GRID, GRID], labels_read=0, backbone_forwards=0, new_roma_forwards=0)
    need(payload['query'].shape == payload['reference'].shape == (128, 64, 72), 'FINAL_SHAPE')
    for mask in ('query_valid', 'reference_valid', 'query_mapping_valid', 'reference_mapping_valid'):
        need(payload[mask].shape == (128, 64), 'MASK_SHAPE')
    need(torch.equal(payload['query_valid'], payload['query_valid'][0].expand_as(payload['query_valid'])), 'COMMON_QUERY_GRID')
    need(torch.allclose(payload['query'][:, :, :32], payload['query'][0, :, :32].expand_as(payload['query'][:, :, :32]), atol=0, rtol=0), 'COMMON_QUERY_CONTENT')
    atomic_torch(path, payload)
    receipt = dict(status='POOLED_F_QUERY_PASS', version=VERSION, query_id=entry['query_id'],
        execution_ordinal=index, payload=bind(path), candidates=128, query_shape=[128, 64, 72],
        reference_shape=[128, 64, 72], source=payload['source'], geometry=summaries,
        seconds=time.monotonic() - start, labels_read=0, backbone_forwards=0, new_roma_forwards=0)
    atomic_json(receipt_path, receipt)
    print(json.dumps(dict(stage='query_done', index=index, query_id=entry['query_id'],
                         seconds=receipt['seconds'], bytes=receipt['payload']['bytes'])), flush=True)
    return receipt


def update_manifest(expected, bindings):
    records = []
    for e in expected:
        path = OUT / f"query{e['execution_ordinal']:03d}.json"
        if not path.exists():
            continue
        r = read(path)
        need(r['status'] == 'POOLED_F_QUERY_PASS' and r['version'] == VERSION, 'RECORD_VERSION')
        need(r['query_id'] == e['query_id'], 'FROZEN_MANIFEST_QUERY')
        records.append({k: r[k] for k in ('query_id', 'execution_ordinal', 'payload', 'candidates', 'seconds')})
    manifest = dict(status='POOLED_F71_COMPLETE' if len(records) == 71 else 'POOLED_F71_PARTIAL',
        version=VERSION, expected_queries=71, completed_queries=len(records),
        expected_records=expected, records=records, sources=bindings, channels=CHANNELS,
        query_shape=[128, 64, 72], reference_shape=[128, 64, 72], dtype='float32',
        M0_dtype='float64', mask_semantics='source geometry valid; separate warp mapping valid; no certainty cutoff',
        descriptor='RoMa coarse_17 FP64-area-pooled then saved FP32 on original ColNomic image cells',
        sampling='native final warp at original cell centers; containing-cell sample; THEN geometric 8x8 mean pool',
        projection=dict(kind='shared label-free Rademacher 1024-to-32 scaled by 1/sqrt(32)', seed=SEED),
        selection='All and only existing complete coordinate query indices 0..70; no outcome selection',
        large_source_sha_policy='Inherited sealed SHA; newly consumed component/geometry/warp arrays hashed explicitly; new output files fully hashed',
        labels_read=0, training_updates=0, backbone_forwards=0, new_roma_forwards=0,
        evidence_boundary='Pooled F71 development input, not full H593 F and not native dense activations')
    atomic_json(MANIFEST, manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, default=71); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--wall-seconds', type=float, default=0,
                    help='Stop after a completed query with exit 75 when this budget is reached; 0 disables.')
    args = ap.parse_args(); need(0 <= args.start <= args.stop <= 71, 'INDEX_RANGE')
    started = time.monotonic()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    install_label_guard()
    ready = read(FEATURES / 'ready.json'); need(ready['status'] == 'FUSION_ALL593_FEATURE_CACHE_PASS', 'FEATURE_READY')
    same_sha(FEATURES / 'catalog.json', ready['catalog'])
    catalog = read(FEATURES / 'catalog.json'); rows = {r['execution_ordinal']: r for r in catalog['queries']}
    available = sorted(int(p.name[-3:]) for p in COORDINATES.glob('query[0-9][0-9][0-9]') if (p / 'validation.json').exists())
    need(available == list(range(71)), 'EXPECTED_FIXED_CACHED_INTERSECTION71')
    expected = [dict(query_id=rows[i]['query_id'], execution_ordinal=i) for i in available]
    bindings = dict(program=bind(__file__), feature_ready=bind(FEATURES / 'ready.json'),
                    feature_catalog=ready['catalog'])
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = projection(); bank = ImageBank(ready, matrix)
    update_manifest(expected, bindings)
    for index in range(args.start, args.stop):
        build_query(index, rows, bank); update_manifest(expected, bindings)
        if args.wall_seconds > 0 and time.monotonic() - started >= args.wall_seconds and index + 1 < args.stop:
            print(json.dumps(dict(stage='wall_budget', next_index=index + 1,
                seconds=time.monotonic() - started, exit_code=75)), flush=True)
            return 75
    print(json.dumps({k: v for k, v in update_manifest(expected, bindings).items()
                      if k in ('status', 'expected_queries', 'completed_queries')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
