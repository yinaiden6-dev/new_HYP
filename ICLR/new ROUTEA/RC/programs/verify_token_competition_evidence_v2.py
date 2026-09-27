#!/usr/bin/env python3
"""Independent NumPy replay and resumable SHA/axis/mask validation of V2 F128.

Replay candidates 0,63,127 and source-axis token positions 0,N//3,2N//3,N-1
plus the first/last valid tokens, fixed without labels. Every reference token
participates in every replayed argmax. Only a complete verification publishes
evidence_manifest.json. Pilot validation publishes pilot_manifest.json.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

# Share I/O and source allowlisting only; scientific replay below is NumPy.
import build_token_competition_evidence_v2 as B

OUT = B.OUT
REPLAY_CANDIDATES = [0, 63, 127]
ATOL = 2e-7


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def stat_signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class SourceHasher:
    """Rehash every changed file, reuse hashes only with all stat fields equal."""
    def __init__(self, prior):
        self.records = dict(prior)
        self.touched = {}

    def check(self, b):
        path = Path(b['path']).resolve()
        name = str(path)
        sig = stat_signature(path)
        old = self.records.get(name)
        if old is None or old['sha256'] != b['sha256'] or old['stat'] != sig:
            actual = B.binding(path)
            require(actual['sha256'] == b['sha256'], 'INDEPENDENT_SOURCE_SHA:' + name)
            require(stat_signature(path) == sig, 'SOURCE_CHANGED_DURING_HASH:' + name)
            self.records[name] = dict(sha256=b['sha256'], stat=sig)
        self.touched[name] = self.records[name]
        return path

    def unchanged(self, records):
        return all(Path(p).is_file() and stat_signature(p) == b['stat'] and
                   self.records.get(p) == b for p, b in records.items())


def np_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def np_image(ctx, key, descriptor, hashes):
    source = ctx.ready['features'][key]
    require(descriptor['key'] == key and descriptor['payload'] == source, 'IMAGE_SOURCE_BINDING')
    original = B.load(hashes.check(source))
    require(original['item'] == ctx.catalog['images'][key]['item'] == descriptor['item'], 'IMAGE_ITEM')
    raw = original['original_colnomic_tokens']
    h = hashlib.sha256()
    h.update(str(raw.dtype).encode('ascii'))
    h.update(json.dumps(list(raw.shape), separators=(',', ':')).encode('ascii'))
    h.update(raw.contiguous().view(torch.uint8).numpy().tobytes())
    require(h.hexdigest() == original['item']['tokens_sha256'] == descriptor['tokens_sha256'], 'IMAGE_TOKEN_SHA')
    x = raw.numpy().astype(np.float64)
    boxes0 = torch.as_tensor(original['cell_boxes_xyxy']).numpy()
    visual = torch.as_tensor(original['valid_patch_mask']).numpy().astype(bool)
    boxes = boxes0.astype(np.float64)
    require(x.ndim == 2 and x.shape[1] == 128 and len(x) > 0, 'TOKEN_SHAPE')
    require(boxes.shape == (len(x), 4) and visual.shape == (len(x),), 'GEOMETRY_AXIS')
    require(original['geometry'] == descriptor['geometry'] and
            list(original['geometry']['grid_shape']) == original['item']['grid'], 'GEOMETRY_BINDING')
    require(np_sha(boxes0) == descriptor['boxes_sha256'] and
            np_sha(visual) == descriptor['visual_mask_sha256'], 'GEOMETRY_COMPONENT_SHA')
    finite = np.isfinite(x).all(axis=1)
    clean = np.where(np.isfinite(x), x, 0.)
    norms = np.sqrt(np.sum(clean * clean, axis=1))
    valid = (visual & np.isfinite(boxes).all(axis=1) &
             (boxes[:, 2:] > boxes[:, :2]).all(axis=1) & finite &
             np.isfinite(norms) & (norms > 0))
    normalized = np.where(valid[:, None], clean / np.where(valid, norms, 1.)[:, None], 0.)
    require(np_sha(valid) == descriptor['valid_mask_sha256'] and
            len(x) == descriptor['original_tokens'] and int(valid.sum()) == descriptor['valid_tokens'], 'VALIDITY_SOURCE')
    return normalized, valid


def max_error(actual, expected, message):
    require(actual.shape == expected.shape, message + '_SHAPE')
    err = float(np.max(np.abs(actual.astype(np.float64) - expected))) if actual.size else 0.
    require(np.isfinite(err) and err <= ATOL, message + '_ERROR:' + str(err))
    return err


def replay(q, qvalid, r, rvalid, u, v, selected):
    """Independent scalar construction, including safe all-invalid behavior."""
    result = np.zeros((len(selected), 8), np.float64)
    refs = np.zeros((len(selected), 128), np.float64)
    indices = np.full(len(selected), -1, np.int64)
    ridx = np.flatnonzero(rvalid)
    if len(ridx) == 0:
        return result, refs, indices
    for k, t in enumerate(selected):
        if not qvalid[t]:
            continue
        scores = np.sum(r[ridx] * q[t][None, :], axis=1)
        local = int(np.argmax(scores))
        j = int(ridx[local])
        best = float(scores[local])
        second = float(np.sort(scores)[-2]) if len(scores) >= 2 else best
        result[k] = [best, best - second, u[t], v[j], u[t] * best,
                     v[j] * best, u[t] * v[j] * best, np.mean(v)]
        refs[k] = r[j]
        indices[k] = j
    return result, refs, indices


def validate_one(ctx, index, hashes):
    hashes.touched = {}
    receipt_path = OUT / 'evidence' / f'query{index:03d}.json'
    receipt_binding = B.binding(receipt_path)
    hashes.check(receipt_binding)
    receipt = B.read(receipt_path)
    require(receipt['status'] == 'TOKEN_COMPETITION_QUERY_PASS' and receipt['version'] == B.VERSION,
            'QUERY_RECEIPT')
    require(receipt['context'] == ctx.context_binding, 'QUERY_CONTEXT')
    path = hashes.check(receipt['payload'])
    require(path == OUT / 'evidence' / f'query{index:03d}.pt', 'OWN_PAYLOAD_ROOT')
    p = B.load(path)
    common = ctx.common[index]
    require(p['version'] == B.VERSION and p['context'] == ctx.context_binding, 'PAYLOAD_VERSION_CONTEXT')
    require(p['query_id'] == common['query_id'] == receipt['query_id'] and
            p['execution_ordinal'] == receipt['execution_ordinal'] == index, 'PAYLOAD_IDENTITY')
    require(p['axis128'] == p['axis'] == common['axis'] and p['winner'] == common['winner'], 'C128_AXIS_WINNER')
    require(len(p['axis128']) == len(set(p['axis128'])) == 128, 'C128_UNIQUE')
    require(p['source'] == receipt['source'] and p['source']['fusion_query'] == ctx.rows[index]['payload'], 'QUERY_SOURCE')
    require(p['scalar_names'] == B.SCALARS, 'SCALAR_ORDER')
    for value in (p, receipt):
        require(value['labels_read'] == value['backbone_forwards'] == value['new_roma_forwards'] == 0, 'NO_FORWARD_LABEL')
    require(p['M0'].dtype == torch.float64 and p['M0'].shape == (128,) and
            torch.equal(p['M0'], common['M0']), 'COMMON_M0_BITS')
    original = B.load(hashes.check(p['source']['fusion_query']))
    require(original['query_id'] == p['query_id'] and original['execution_ordinal'] == index and
            original['candidate_physical_rows'] == p['axis128'] and original['winner'] == p['winner'] and
            original['label_reads'] == 0 and len(original['pairs']) == 128, 'ORIGINAL_QUERY_AXIS')
    q, qvalid = np_image(ctx, original['query_image_key'], p['source']['query_image'], hashes)
    n = len(q)
    require(p['query_tokens'].shape == (n, 128) and p['query_tokens'].dtype == torch.float32, 'QUERY_TOKENS_SHAPE')
    require(p['query_valid'].shape == (n,) and p['query_valid'].dtype == torch.bool and
            np.array_equal(p['query_valid'].numpy(), qvalid), 'QUERY_VALIDITY')
    require(p['matched_reference'].shape == (128, n, 128) and p['matched_reference'].dtype == torch.float32,
            'MATCHED_REFERENCE_SHAPE')
    require(p['local_scalars'].shape == (128, n, 8) and p['local_scalars'].dtype == torch.float32, 'SCALAR_SHAPE')
    require(p['matched_indices'].shape == (128, n) and p['matched_indices'].dtype == torch.long, 'INDEX_SHAPE')
    require(p['matched_valid'].shape == (128, n) and p['matched_valid'].dtype == torch.bool, 'MATCHED_VALID_SHAPE')
    for name in ('query_tokens', 'matched_reference', 'local_scalars', 'M0'):
        require(torch.isfinite(p[name]).all(), 'FINITE_' + name)
    require((p['query_tokens'][~p['query_valid']] == 0).all(), 'INVALID_QUERY_ZERO')
    errors = dict(query_normalization=max_error(p['query_tokens'].numpy(), q, 'QUERY_NORMALIZATION'),
                  matched_reference=0., local_scalars=0., mean_v=0.)
    valid_indices = np.flatnonzero(qvalid)
    tokens = sorted(set([0, n // 3, 2 * n // 3, n - 1] +
                        ([int(valid_indices[0]), int(valid_indices[-1])] if len(valid_indices) else [])))
    require(len(p['source']['references']) == len(p['source']['parts']) == 128, 'SOURCE_C128')
    replay_records = []
    for pos, pair in enumerate(original['pairs']):
        require(pair['position'] == pos and pair['physical_row'] == p['axis128'][pos], 'SOURCE_PAIR_AXIS')
        part = p['source']['parts'][pos]
        require(part['position'] == pos, 'PART_POSITION')
        part_receipt_path = hashes.check(part['receipt'])
        require(part_receipt_path == OUT / 'candidate_parts' / f'query{index:03d}' / f'candidate{pos:03d}.json', 'PART_ROOT')
        cr = B.read(part_receipt_path)
        require(cr['status'] == 'TOKEN_COMPETITION_CANDIDATE_PASS' and cr['version'] == B.VERSION and
                cr['payload'] == part['payload'] and cr['reference'] == p['source']['references'][pos], 'PART_RECEIPT')
        part_path = hashes.check(part['payload'])
        require(part_path == part_receipt_path.with_suffix('.pt'), 'PART_PAYLOAD_ROOT')
        cp = B.load(part_path)
        require(cp['version'] == B.VERSION and cp['query_id'] == p['query_id'] and
                cp['execution_ordinal'] == index and cp['position'] == pos and
                cp['physical_row'] == p['axis128'][pos] and cp['reference'] == cr['reference'],
                'PART_PAYLOAD_BINDING')
        for name in ('matched_reference', 'local_scalars', 'matched_indices', 'matched_valid', 'M0'):
            require(torch.equal(cp[name], p[name][pos]), 'PART_FINAL_EXACT:' + name)
        u = pair['query_visibility']
        v = pair['reference_visibility']
        refdesc = p['source']['references'][pos]
        require(u.dtype == v.dtype == torch.float64 and u.shape == (n,) and
                v.shape == (refdesc['original_tokens'],), 'ORIGINAL_FP64_VISIBILITY_AXES')
        require(torch.isfinite(u).all() and torch.isfinite(v).all() and
                ((u >= 0) & (u <= 1)).all() and ((v >= 0) & (v <= 1)).all(), 'VISIBILITY_RANGE')
        expected_context = dict(context=ctx.context_binding, fusion_query=ctx.rows[index]['payload'],
            query_image=p['source']['query_image'], reference_source=ctx.ready['features'][pair['image_key']],
            position=pos, physical_row=pair['physical_row'], query_visibility_sha256=np_sha(u.numpy()),
            reference_visibility_sha256=np_sha(v.numpy()))
        require(cr['context'] == expected_context, 'CANDIDATE_SOURCE_CONTEXT')
        require(torch.equal(torch.sqrt(u.mean() * v.mean()), p['M0'][pos]), 'SOURCE_M0_BITS')
        r, rvalid = np_image(ctx, pair['image_key'], refdesc, hashes)
        mv = p['matched_valid'][pos].numpy()
        ids = p['matched_indices'][pos].numpy()
        require(np.array_equal(mv, qvalid & bool(rvalid.any())), 'MATCH_VALIDITY')
        require(np.all(ids[~mv] == -1) and np.all(ids[mv] >= 0) and np.all(ids[mv] < len(r)), 'SAFE_INDICES')
        require(np.all(rvalid[ids[mv]]), 'MATCH_POINTS_TO_VALID_REFERENCE')
        require((p['matched_reference'][pos][~torch.from_numpy(mv)] == 0).all() and
                (p['local_scalars'][pos][~torch.from_numpy(mv)] == 0).all(), 'ALL_INVALID_ZERO')
        if mv.any():
            error = max_error(p['local_scalars'][pos, torch.from_numpy(mv), 7].numpy(),
                              np.full(int(mv.sum()), float(v.mean())), 'MEAN_V')
            errors['mean_v'] = max(errors['mean_v'], error)
            # All serialized matched descriptors must actually be the indexed
            # normalized source descriptors, even outside sampled argmax replay.
            error = max_error(p['matched_reference'][pos].numpy()[mv], r[ids[mv]], 'MATCH_DESCRIPTOR')
            errors['matched_reference'] = max(errors['matched_reference'], error)
            # Independently recompute every scalar except runner-up gap at every
            # valid query token. Full-reference top1/top2 is sampled below.
            cos = np.sum(q[mv] * r[ids[mv]], axis=1)
            ui, vi = u.numpy()[mv], v.numpy()[ids[mv]]
            all_scalars = np.stack((cos, ui, vi, ui * cos, vi * cos, ui * vi * cos,
                                    np.full(int(mv.sum()), np.mean(v.numpy()))), axis=1)
            got = p['local_scalars'][pos].numpy()[mv][:, [0, 2, 3, 4, 5, 6, 7]]
            errors['local_scalars'] = max(errors['local_scalars'], max_error(got, all_scalars, 'ALL_TOKEN_SCALARS'))
            gaps = p['local_scalars'][pos].numpy()[mv, 1]
            require(np.all(gaps >= 0) and np.all(gaps <= 2 + ATOL), 'TOP2_GAP_RANGE')
        if pos in REPLAY_CANDIDATES:
            scalars, matched, expected_indices = replay(q, qvalid, r, rvalid, u.numpy(), v.numpy(), tokens)
            require(np.array_equal(ids[tokens], expected_indices), 'INDEPENDENT_ARGMAX:' + str((index, pos)))
            errors['local_scalars'] = max(errors['local_scalars'],
                max_error(p['local_scalars'][pos, tokens].numpy(), scalars, 'INDEPENDENT_SCALARS'))
            errors['matched_reference'] = max(errors['matched_reference'],
                max_error(p['matched_reference'][pos, tokens].numpy(), matched, 'INDEPENDENT_MATCHED'))
            replay_records.append(dict(candidate_position=pos, query_token_positions=tokens,
                                       full_reference_tokens=len(r), valid_reference_tokens=int(rvalid.sum())))
    return dict(query_id=p['query_id'], execution_ordinal=index, payload=receipt['payload'],
                receipt=receipt_binding, candidates=128, query_tokens=n, valid_query_tokens=int(qvalid.sum()),
                max_errors=errors, independent_replays=replay_records, source_stats=dict(hashes.touched))


def synthetic():
    # Different axes, tied reference vectors, NaNs/invalids are represented by
    # independent source validity masks. Compare NumPy construction to builder.
    q = np.zeros((4, 128), np.float64)
    q[0, 0] = q[1, 1] = 1.
    r = np.zeros((3, 128), np.float64)
    r[0, 0] = r[1, 0] = r[2, 1] = 1.
    qv = np.array([1, 1, 0, 0], bool)
    rv = np.ones(3, bool)
    u, v = np.array([.1, .2, .3, .4]), np.array([.3, .7, .9])
    expected = replay(q, qv, r, rv, u, v, list(range(4)))
    actual = B.compete(*[torch.from_numpy(x) for x in (q, qv, r, rv, u, v)])
    require(np.array_equal(actual[2].numpy(), expected[2]), 'SYNTHETIC_ARGMAX')
    err = max_error(actual[1].numpy(), expected[0], 'SYNTHETIC_SCALARS')
    require(expected[2].tolist() == [0, 2, -1, -1], 'SYNTHETIC_FIRST_TIE')
    empty = replay(q, qv, r, np.zeros_like(rv), u, v, list(range(4)))
    require(np.all(empty[0] == 0) and np.all(empty[1] == 0) and np.all(empty[2] == -1), 'SYNTHETIC_EMPTY')
    return dict(status='INDEPENDENT_NUMPY_SYNTHETIC_PASS', max_scalar_error=err, builder_checks=B.self_test())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=OUT / 'protocol.json')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--wall-seconds', type=float, default=480)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if a.self_test:
        print(json.dumps(synthetic()), flush=True)
        return 0
    require(os.environ.get('SLURM_JOB_ID'), 'BATCH_REQUIRED_FOR_SOURCE_VERIFICATION')
    started = time.monotonic()
    ctx = B.Context(a.protocol)
    verifier = B.binding(__file__)
    require(any(b == verifier for b in ctx.code_sources.values()), 'VERIFIER_NOT_FROZEN')
    context = dict(**ctx.context_binding, verifier=verifier)
    progress_path = OUT / ('pilot_evidence_verification_progress.json' if a.pilot else 'evidence_verification_progress.json')
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if progress_path.exists():
            progress = B.read(progress_path)
            require(progress['context'] == context, 'VERIFICATION_RESUME_CONTEXT')
        else:
            progress = dict(context=context, records={}, hashed_sources={})
        hashes = SourceHasher(progress['hashed_sources'])
        indices = [ctx.pilot_index()] if a.pilot else list(range(128))
        for index in indices:
            old = progress['records'].get(str(index))
            if old and hashes.unchanged(old['source_stats']):
                continue
            if B.expired(started, a.wall_seconds):
                progress['hashed_sources'] = hashes.records
                B.atomic_json(progress_path, progress)
                print(json.dumps(dict(status='TOKEN_COMPETITION_VERIFY_WALL_BUDGET', next_index=index,
                                      completed_queries=len(progress['records']), exit_code=75)), flush=True)
                return 75
            progress['records'][str(index)] = validate_one(ctx, index, hashes)
            progress['hashed_sources'] = hashes.records
            B.atomic_json(progress_path, progress)
            print(json.dumps(dict(stage='INDEPENDENT_QUERY_PASS', index=index,
                                  max_errors=progress['records'][str(index)]['max_errors'])), flush=True)
        # Sources from completed resumptions must still be unchanged now.
        for i in indices:
            require(hashes.unchanged(progress['records'][str(i)]['source_stats']), 'SOURCE_CHANGED_BEFORE_SEAL')
        checked_records = [progress['records'][str(i)] for i in indices]
        records = [{k: r[k] for k in ('query_id', 'execution_ordinal', 'payload', 'receipt', 'candidates',
                                      'query_tokens', 'valid_query_tokens')} for r in checked_records]
        status = 'TOKEN_COMPETITION_EVIDENCE_PILOT_PASS' if a.pilot else 'TOKEN_COMPETITION_EVIDENCE128_PASS'
        manifest = dict(status=status, version=B.VERSION, records=records,
            expected_queries=128, completed_queries=len(records), candidates_per_query=128,
            expected_records=[dict(query_id=ctx.common[i]['query_id'], execution_ordinal=i) for i in range(128)],
            sources=context, scalar_names=B.SCALARS, query_axis='original native visual token axis; invalid rows retained as zero',
            matching='FP64 cosine free argmax over all valid reference visual patches, ties first original index',
            singleton_reference_gap=0., empty_reference='zero descriptors/scalars; matched_valid=false; matched_indices=-1',
            M0='sqrt(original FP64 query_visibility.mean() * original FP64 reference_visibility.mean())',
            mean_v='original FP64 reference_visibility.mean() on its full original axis; masked positions unchanged',
            storage_dtype='float32', M0_dtype='float64', labels_read=0, backbone_forwards=0, new_roma_forwards=0)
        manifest_path = OUT / ('pilot_manifest.json' if a.pilot else 'evidence_manifest.json')
        B.atomic_json(manifest_path, manifest)
        maxima = {k: max(r['max_errors'][k] for r in checked_records) for k in checked_records[0]['max_errors']}
        validation = dict(status=status, manifest=B.binding(manifest_path), verifier=verifier,
            builder=ctx.context_binding['builder'], protocol=ctx.protocol_binding,
            scientific_module_sources=ctx.code_sources, queries=len(records), candidates=128 * len(records),
            max_errors=maxima, tolerance=ATOL, independent_numpy=True,
            fixed_replay_candidates=REPLAY_CANDIDATES,
            replay_query_token_rule='sorted unique {0,N//3,2N//3,N-1,first_valid,last_valid}',
            source_replays=[dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'],
                                replay=r['independent_replays'], max_errors=r['max_errors']) for r in checked_records],
            source_sha_checked=len(hashes.records), source_hash_resume_policy='reuse only matching dev,ino,size,mtime_ns,ctime_ns',
            common_axis_and_M0_exact=True, labels_read=0, backbone_forwards=0, new_roma_forwards=0,
            synthetic_checks=synthetic())
        B.atomic_json(OUT / ('pilot_evidence_validation.json' if a.pilot else 'evidence_validation.json'), validation)
        print(json.dumps(dict(status=status, queries=len(records), max_errors=maxima)), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
