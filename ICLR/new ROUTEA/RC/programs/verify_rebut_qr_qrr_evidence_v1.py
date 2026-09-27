#!/usr/bin/env python3
"""Independent shape/M0 accounting and source geometry replay for pooled F71."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_rebut_qr_qrr_v1'
VERSION = 'POOLED_F71_COARSE17_NATIVE_CENTER_WARP_V1'


def require(ok, what):
    if not ok:
        raise AssertionError(what)


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def binding(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest(), bytes=p.stat().st_size)


def write(path, data):
    p = Path(path); tmp = p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def independent_side(own, other, support, other_support, warp):
    """Brute force cell-center search and explicit per-bin means; no builder calls."""
    boxes = array(own['boxes']).astype(np.float64)
    other_boxes = array(other['boxes']).astype(np.float64)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    other_centers = (other_boxes[:, :2] + other_boxes[:, 2:]) / 2
    own_valid = array(own['valid']).astype(bool)
    other_valid = array(other['valid']).astype(bool)
    available = np.flatnonzero((other_boxes[:, 2:] > other_boxes[:, :2]).all(1))
    point = (np.where(np.isfinite(warp).all(1)[:, None], warp, 0) + 1) / 2
    distances = ((point[:, None, :] - other_centers[available][None, :, :]) ** 2).sum(2)
    nearest = available[np.argmin(distances, axis=1)]
    box = other_boxes[nearest]
    mapped = own_valid & np.isfinite(warp).all(1) & (np.abs(warp) <= 1).all(1)
    mapped &= ((point >= box[:, :2] - 1e-12) & (point <= box[:, 2:] + 1e-12)).all(1)
    mapped &= other_valid[nearest]
    bins = np.floor(centers * 8).astype(np.int64).clip(0, 7)
    bins = bins[:, 1] * 8 + bins[:, 0]
    result = np.zeros((64, 72), dtype=np.float32)
    source_mask = np.zeros(64, dtype=bool); mapping_mask = np.zeros(64, dtype=bool)
    a = array(own['projected']); b = array(other['projected'])
    for cell in range(64):
        members = bins == cell; valid = members & own_valid; pairs = members & mapped
        source_mask[cell] = valid.any(); mapping_mask[cell] = pairs.any()
        if not valid.any():
            continue
        result[cell, :32] = a[valid].astype(np.float64).mean(0)
        result[cell, 64] = support[valid].mean()
        result[cell, 66:68] = (centers[valid] * 2 - 1).mean(0)
        result[cell, 70] = valid.sum() / members.sum()
        result[cell, 71] = pairs.sum() / valid.sum()
        if pairs.any():
            result[cell, 32:64] = b[nearest[pairs]].astype(np.float64).mean(0)
            result[cell, 65] = other_support[nearest[pairs]].mean()
            result[cell, 68:70] = warp[pairs].astype(np.float64).mean(0)
    return result, source_mask, mapping_mask


def synthetic_checks():
    path = ROOT / 'programs/build_rebut_qr_qrr_evidence_v1.py'
    spec = importlib.util.spec_from_file_location('builder_subject', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    boxes = np.array([[0, 0, .5, .5], [.5, 0, 1, .5], [0, .5, .5, 1], [.5, .5, 1, 1]], dtype=np.float64)
    features = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    own = mod.geometry_index(boxes, np.ones(4, bool)); own['projected'] = features
    perm = [3, 1, 0, 2]
    other = mod.geometry_index(boxes[perm], np.ones(4, bool)); other['projected'] = features[perm] + 1000
    warp = own['centers'] * 2 - 1
    u = np.array([0, 1e-15, .25, .75]); v = np.array([.1, .2, .3, .4])[perm]
    result, mask, mapped, _ = mod.side_evidence(own, other, u, v, warp)
    require(mapped.sum() == 4 and mask.sum() == 4, 'ZERO_SUPPORT_NOT_DROPPED')
    occupied = own['bins']
    require(np.array_equal(result[occupied, 32:64], features + 1000), 'REFERENCE_ARRAY_PERMUTATION_GEOMETRY')
    require(np.allclose(result[occupied, 64], u, atol=1e-8, rtol=0), 'LOW_SUPPORT_RETAINED')
    # Three unmappable states must retain self content, separate from target evidence.
    altered = warp.copy(); altered[0] = [1.01, 0]; altered[1] = [np.nan, 0]
    other['source_valid'][perm.index(2)] = False
    result, mask, mapped, _ = mod.side_evidence(own, other, u, v, altered)
    require(mask.sum() == 4 and mapped.sum() == 1, 'MAPPING_SEPARATE_FROM_SOURCE_VALIDITY')
    require(np.array_equal(result[occupied[:3], :32], features[:3]), 'OWN_CONTENT_SURVIVES_UNMAPPABLE')
    require((result[occupied[:3], 32:64] == 0).all(), 'UNMAPPABLE_OTHER_ZERO')
    require((result[occupied[:3], 71] == 0).all(), 'UNMAPPABLE_FLAG_ZERO')
    # Two distant mapped reference cells are sampled before being averaged.
    tiny_boxes = np.array([[0, 0, .04, .04], [.04, 0, .08, .04]])
    own2 = mod.geometry_index(tiny_boxes, np.ones(2, bool)); own2['projected'] = features[:2]
    other2 = mod.geometry_index(boxes, np.ones(4, bool)); other2['projected'] = features
    distant = np.array([[-.5, -.5], [.5, .5]])
    result, _, _, _ = mod.side_evidence(own2, other2, np.ones(2), np.ones(4), distant)
    require(np.array_equal(result[0, 32:64], (features[0] + features[3]) / 2), 'SAMPLE_BEFORE_POOL')
    return ['zero_and_low_support_retained', 'reference_array_permutation_geometry',
            'out_of_bounds_nan_padding_masks', 'self_content_survives_unmappable',
            'sample_before_spatial_pool', 'shared_query_grid']


def source_replay(record, positions):
    payload = load(record['payload']['path'])
    source = payload['source']; fusion = load(source['fusion_query']['path'])
    qkey = fusion['query_image_key']
    own = load(source['images'][qkey]['projected']['path'])
    # Projection check from the actual frozen coarse descriptor, with a fresh seed generator.
    original = load(source['images'][qkey]['source']['path'])
    generator = torch.Generator().manual_seed(20260927)
    matrix = (torch.randint(0, 2, (1024, 32), generator=generator).float() * 2 - 1) / math.sqrt(32)
    expected_projection = original['components']['coarse_17'].float() @ matrix
    require(torch.equal(expected_projection, own['projected']), 'FRESH_PROJECTION_REPLAY')
    errors = []; checks = 0; part_cache = {}
    for position in positions:
        index = position // 16
        if index not in part_cache:
            part_cache[index] = load(source['coordinate_parts'][index]['path'])
        pair = next(x for x in part_cache[index]['pairs'] if x['candidate_position'] == position)
        arm = pair['arms']['NATIVE']; rkey = fusion['pairs'][position]['image_key']
        other = load(source['images'][rkey]['projected']['path'])
        for side, first, second, u, v, coord in (
            ('query', own, other, arm['query_visibility'], arm['reference_visibility'], arm['intermediate']['query_coordinates']),
            ('reference', other, own, arm['reference_visibility'], arm['query_visibility'], arm['intermediate']['reference_coordinates'])):
            expected, valid, mapped = independent_side(first, second, array(u), array(v), array(coord['center_xy']))
            actual = array(payload[side][position]); error = float(np.max(np.abs(expected - actual)))
            require(error < 2e-5, 'INDEPENDENT_GEOMETRY_POOL:' + str((position, side, error)))
            require(np.array_equal(valid, array(payload[side + '_valid'][position])), 'INDEPENDENT_SOURCE_MASK')
            require(np.array_equal(mapped, array(payload[side + '_mapping_valid'][position])), 'INDEPENDENT_MAPPING_MASK')
            errors.append(error); checks += expected.size
    return dict(query_id=payload['query_id'], execution_ordinal=payload['execution_ordinal'],
                positions=positions, value_checks=checks, max_abs_error=max(errors))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--threads', type=int, default=2); args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    manifest_path = OUT / 'evidence_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if args.allow_partial:
        # The builder atomically refreshes its live manifest. Preserve exactly
        # the snapshot audited here instead of binding a later live revision.
        manifest_path = OUT / f"evidence_engineering_manifest_snapshot_{len(manifest['records']):03d}.json"
        manifest_path.write_bytes(manifest_bytes)
    manifest_binding = binding(manifest_path)
    require(manifest_binding['sha256'] == hashlib.sha256(manifest_bytes).hexdigest(), 'MANIFEST_SNAPSHOT')
    require(manifest['version'] == VERSION, 'VERSION')
    if not args.allow_partial:
        require(manifest['status'] == 'POOLED_F71_COMPLETE' and manifest['completed_queries'] == 71, 'COMPLETE71')
    require([r['execution_ordinal'] for r in manifest['expected_records']] == list(range(71)), 'FIXED_INTERSECTION')
    checks = synthetic_checks(); counted = 0; ids = []
    for record in manifest['records']:
        b = binding(record['payload']['path']); require(b == record['payload'], 'OUTPUT_SHA')
        p = load(b['path']); q = p['query']; r = p['reference']; n = p['execution_ordinal']
        require(q.shape == r.shape == (128, 64, 72), 'SHAPES')
        require(q.dtype == r.dtype == torch.float32 and torch.isfinite(q).all() and torch.isfinite(r).all(), 'DTYPE_FINITE')
        require(len(p['axis']) == len(set(p['axis'])) == 128 and 0 <= p['winner'] < 128, 'C128')
        require(sorted([p['winner']] + p['challenger_positions']) == list(range(128)), 'CHALLENGER_AXIS')
        require(p['labels_read'] == p['backbone_forwards'] == p['new_roma_forwards'] == 0, 'READ_ONLY')
        native = read(ROOT / f'results/rc_h593_roma_coordinate_precision_v2/query{n:03d}/payload.json')
        require(native['query_id'] == p['query_id'] == record['query_id'] and native['candidate_physical_rows'] == p['axis'], 'SOURCE_AXES')
        mass = torch.tensor([s['visibility_mass'] for s in native['modes']['NATIVE']['scores']], dtype=torch.float64)
        require(torch.equal(mass, p['M0']), 'SOURCE_M0_BITS')
        require(torch.equal(torch.tensor(native['modes']['NATIVE']['X'], dtype=torch.float64), p['native_X']), 'SOURCE_X_BITS')
        for side in ('query', 'reference'):
            valid = p[side + '_valid']; mapped = p[side + '_mapping_valid']
            require(valid.dtype == mapped.dtype == torch.bool and valid.shape == mapped.shape == (128, 64), 'MASK_DOMAIN')
            require(not bool((mapped & ~valid).any()), 'MAPPING_IMPLIES_SOURCE')
            require(bool((p[side][~valid] == 0).all()), 'SOURCE_INVALID_ZERO')
            require(bool((p[side][:, :, 32:64][~mapped] == 0).all()), 'MAPPING_INVALID_ZERO')
        require(torch.equal(q[:, :, :32], q[0, :, :32].expand_as(q[:, :, :32])), 'COMMON_QUERY_CONTENT')
        ids.append(p['query_id']); counted += 128
    require(len(ids) == len(set(ids)), 'UNIQUE_QUERIES')
    replay = source_replay(manifest['records'][0], [0, 1, 127]) if manifest['records'] else None
    complete = len(ids) == 71
    result = dict(status='POOLED_F_EVIDENCE_INDEPENDENT_PASS' if complete else 'POOLED_F_PARTIAL_INDEPENDENT_PASS',
        manifest=manifest_binding, verifier=binding(__file__),
        queries=len(ids), candidates=counted, synthetic_checks=checks, independent_source_replay=replay,
        note='All currently sealed outputs checked for SHA, shape, source M0/X and masks. First-query three pairs replayed by independent brute-force geometry and explicit means.',
        new_encoder_forwards=0, new_roma_forwards=0, labels_read=0)
    dest = OUT / ('evidence_validation.json' if complete else 'evidence_engineering_checks.json')
    write(dest, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
