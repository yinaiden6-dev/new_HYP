#!/usr/bin/env python3
"""Label-free native ColNomic token competition, CPU only, candidate resumable.

No projection, pooling, RoMa coordinate lookup, encoder call or label join.
The only correspondence is an exact FP64 cosine argmax over every valid visual
reference token.  Serialized descriptors/scalars are FP32; M0 remains FP64.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_token_competition_f128_v2'
VERSION = 'NATIVE_COLNOMIC_TOKEN_COMPETITION_F128_V2'
SCALARS = ['free_cos', 'free_top1_minus_top2', 'u', 'v_at_free',
           'u_times_free_cos', 'v_at_free_times_free_cos',
           'u_times_v_at_free_times_free_cos', 'mean_v']


def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(value):
    path = Path(value['path']).resolve()
    need(binding(path)['sha256'] == value['sha256'], 'SOURCE_SHA:' + str(path))
    return path


def array_sha(value):
    t = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def token_sha(value):
    t = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode('ascii'))
    h.update(json.dumps(list(t.shape), separators=(',', ':')).encode('ascii'))
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


class ReadGuard:
    """Exact source-file allowlist, plus own new output root; never label files."""
    def __init__(self, protocol_path):
        self.allowed = {Path(protocol_path).resolve()}
        self.protected = ('curator_roles', 'heldout_roles', 'train_roles', '/roles/',
                          '/target_join/', 'm_scalar_headroom', 'failure_audit',
                          '/grozi/', '/isic/', 'd1-mi', 'd1_mi', 'formal392', '/reports/')
        sys.addaudithook(self.audit)

    def allow(self, values):
        for b in values:
            if isinstance(b, dict) and 'path' in b and 'sha256' in b:
                self.allowed.add(Path(b['path']).resolve())

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        name = str(path).lower()
        need(not any(word in name for word in self.protected), 'LABEL_OR_PROTECTED_READ:' + name)
        if RC / 'results' in path.parents:
            need(OUT in path.parents or path in self.allowed, 'UNLISTED_RESULT:' + str(path))


class Context:
    def __init__(self, protocol_path):
        protocol_path = Path(protocol_path).resolve()
        self.guard = ReadGuard(protocol_path)
        p = read(protocol_path)
        self.protocol = p
        self.protocol_binding = binding(protocol_path)
        inputs = p
        if any(k not in p for k in ('feature_ready', 'feature_catalog', 'common_features')):
            self.guard.allow([p['evidence_inputs']])
            inputs = dict(read(checked(p['evidence_inputs'])), **p)
        self.sources = {k: inputs[k] for k in ('feature_ready', 'feature_catalog', 'common_features')}
        self.guard.allow(self.sources.values())
        self.ready = read(checked(self.sources['feature_ready']))
        self.catalog = read(checked(self.sources['feature_catalog']))
        need(self.ready['status'] == 'FUSION_ALL593_FEATURE_CACHE_PASS', 'FEATURE_READY')
        need(self.ready['catalog'] == self.sources['feature_catalog'], 'READY_CATALOG_BINDING')
        need(self.ready['queries'] == 593 and len(self.catalog['queries']) == 593, 'SOURCE593')
        self.rows = {r['execution_ordinal']: r for r in self.catalog['queries']}
        need(sorted(self.rows) == list(range(593)), 'SOURCE_ORDINALS')
        self.guard.allow([r['payload'] for r in self.catalog['queries']])
        self.guard.allow(self.ready['features'].values())
        common = load(checked(self.sources['common_features']))
        need(common['labels_included'] is False, 'COMMON_LABEL_FREE')
        self.common = {int(r['execution_ordinal']): r for r in common['records']}
        need(len(self.common) == len(common['records']), 'COMMON_UNIQUE_ORDINALS')
        expected = [self.common[i]['query_id'] for i in range(128)]
        need(len(set(expected)) == 128 and p['panel_query_ids'] == expected, 'NATURAL_F128_PANEL')
        for i in range(128):
            need(self.rows[i]['query_id'] == expected[i], 'COMMON_CATALOG_QUERY')
        self.code_sources = dict(p.get('code_sources', {}))
        for b in self.code_sources.values():
            if isinstance(b, dict) and 'sha256' in b:
                checked(b)
        own = binding(__file__)
        need(any(b == own for b in self.code_sources.values()), 'BUILDER_NOT_FROZEN')
        self.context_binding = dict(protocol=self.protocol_binding, builder=own, **self.sources)

    def pilot_index(self):
        p = self.protocol
        i = int(p['pilot_execution_ordinal'])
        qid = p['pilot_query_id']
        need(i == 0 and self.rows[i]['query_id'] == qid, 'FIXED_PILOT_ZERO')
        need(qid in p['folds']['0']['inner_fit_query_ids'], 'PILOT_INNER_TRAIN_ONLY')
        return i

    def query(self, index):
        need(0 <= index < 128, 'F128_INDEX')
        entry = self.rows[index]
        q = load(checked(entry['payload']))
        c = self.common[index]
        need(q['query_id'] == entry['query_id'] == c['query_id'], 'QUERY_ID')
        need(q['execution_ordinal'] == index, 'EXECUTION_ORDINAL')
        need(q['query_image_key'] == entry['query_image_key'], 'QUERY_IMAGE_KEY')
        axis = q['candidate_physical_rows']
        need(axis == c['axis'] and len(axis) == len(set(axis)) == 128, 'C128_AXIS')
        need(q['winner'] == c['winner'] and 0 <= q['winner'] < 128, 'RAW_WINNER')
        need(q['label_reads'] == 0 and len(q['pairs']) == 128, 'QUERY_LABEL_FREE')
        return q

    def image(self, key):
        b = self.ready['features'][key]
        p = load(checked(b))
        item = p['item']
        need(item == self.catalog['images'][key]['item'] and item['key'] == key, 'IMAGE_ITEM')
        tokens = p['original_colnomic_tokens']
        need(tokens.ndim == 2 and tokens.shape[1] == 128 and len(tokens) > 0, 'NATIVE128_TOKENS')
        need(token_sha(tokens) == item['tokens_sha256'], 'TOKEN_SOURCE_SHA')
        boxes = torch.as_tensor(p['cell_boxes_xyxy'])
        visual = torch.as_tensor(p['valid_patch_mask']).bool()
        need(boxes.shape == (len(tokens), 4) and visual.shape == (len(tokens),), 'GEOMETRY_AXIS')
        need(list(p['geometry']['grid_shape']) == list(item['grid']), 'GEOMETRY_GRID')
        normalized, valid = normalized_visual(tokens, boxes, visual)
        descriptor = dict(key=key, payload=b, item=item, geometry=p['geometry'],
            tokens_sha256=token_sha(tokens), boxes_sha256=array_sha(boxes),
            visual_mask_sha256=array_sha(visual), valid_mask_sha256=array_sha(valid),
            normalized_fp64_sha256=array_sha(normalized), original_tokens=len(tokens),
            valid_tokens=int(valid.sum()), source_dtype=str(tokens.dtype))
        return normalized, valid, descriptor


def normalized_visual(tokens, boxes, visual):
    """Retain source axis. Nonvisual, nonfinite and zero-norm rows are zero."""
    x = tokens.double()
    boxes = boxes.double()
    finite = torch.isfinite(x).all(dim=1)
    clean = torch.where(torch.isfinite(x), x, 0)
    norms = torch.linalg.vector_norm(clean, dim=1)
    geometry_valid = (torch.isfinite(boxes).all(dim=1) &
                      (boxes[:, 2:] > boxes[:, :2]).all(dim=1))
    valid = visual & geometry_valid & finite & torch.isfinite(norms) & (norms > 0)
    y = clean / torch.where(valid, norms, torch.ones_like(norms))[:, None]
    return torch.where(valid[:, None], y, 0), valid


def visibility(value, n, name):
    need(value.dtype == torch.float64 and value.shape == (n,), name + '_ORIGINAL_FP64_AXIS')
    need(torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), name + '_RANGE')
    return value


def compete(q, qvalid, r, rvalid, u, v):
    """Full-reference FP64 cosine. Ties use first original reference index."""
    n = len(q)
    matched = torch.zeros((n, 128), dtype=torch.float32)
    local = torch.zeros((n, 8), dtype=torch.float32)
    indices = torch.full((n,), -1, dtype=torch.long)
    matched_valid = torch.zeros(n, dtype=torch.bool)
    ridx = torch.nonzero(rvalid, as_tuple=False).flatten()
    qidx = torch.nonzero(qvalid, as_tuple=False).flatten()
    if not len(ridx) or not len(qidx):
        return matched, local, indices, matched_valid
    ref = r[ridx]
    # Chunk only the query axis; every chunk sees every valid reference patch.
    for start in range(0, len(qidx), 256):
        positions = qidx[start:start + 256]
        sim = q[positions] @ ref.T
        best, free = torch.max(sim, dim=1)
        if len(ridx) > 1:
            top2 = torch.topk(sim, k=2, dim=1).values[:, 1]
            gap = best - top2
        else:
            gap = torch.zeros_like(best)
        original = ridx[free]
        ui, vi = u[positions], v[original]
        scalars = torch.stack((best, gap, ui, vi, ui * best, vi * best,
                               ui * vi * best, v.mean().expand_as(best)), dim=1)
        matched[positions] = r[original].float()
        local[positions] = scalars.float()
        indices[positions] = original
        matched_valid[positions] = True
    return matched, local, indices, matched_valid


def expired(start, seconds):
    return seconds > 0 and time.monotonic() - start >= seconds


def build(ctx, index, wall_seconds):
    start = time.monotonic()
    target = OUT / 'evidence' / f'query{index:03d}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists() and target.with_suffix('.json').exists():
            old = read(target.with_suffix('.json'))
            need(old['context'] == ctx.context_binding and old['version'] == VERSION and
                 old['status'] == 'TOKEN_COMPETITION_QUERY_PASS', 'QUERY_RESUME_CONTEXT')
            checked(old['payload'])
            print(json.dumps(dict(status='QUERY_REUSED', index=index)), flush=True)
            return 0
        source = ctx.query(index)
        q, qvalid, qdesc = ctx.image(source['query_image_key'])
        parts = OUT / 'candidate_parts' / f'query{index:03d}'
        parts.mkdir(parents=True, exist_ok=True)
        records, payloads = [], []
        for pos, pair in enumerate(source['pairs']):
            if expired(start, wall_seconds):
                print(json.dumps(dict(status='TOKEN_COMPETITION_EVIDENCE_WALL_BUDGET',
                    index=index, next_candidate=pos, completed_candidates=len(records), exit_code=75)), flush=True)
                return 75
            need(pair['position'] == pos and pair['physical_row'] == source['candidate_physical_rows'][pos], 'PAIR_AXIS')
            path = parts / f'candidate{pos:03d}.pt'
            receipt_path = path.with_suffix('.json')
            u = visibility(pair['query_visibility'], len(q), 'QUERY_VISIBILITY')
            pair_context = dict(context=ctx.context_binding, fusion_query=ctx.rows[index]['payload'],
                query_image=qdesc, reference_source=ctx.ready['features'][pair['image_key']],
                position=pos, physical_row=pair['physical_row'], query_visibility_sha256=array_sha(u),
                reference_visibility_sha256=array_sha(pair['reference_visibility']))
            if path.exists() and receipt_path.exists():
                receipt = read(receipt_path)
                need(receipt['context'] == pair_context and receipt['status'] == 'TOKEN_COMPETITION_CANDIDATE_PASS', 'PART_RESUME')
                checked(receipt['payload'])
                # Source byte binding remains checked on every resumed candidate.
                checked(pair_context['reference_source'])
                part = load(path)
            else:
                r, rvalid, rdesc = ctx.image(pair['image_key'])
                v = visibility(pair['reference_visibility'], len(r), 'REFERENCE_VISIBILITY')
                matched, local, indices, valid = compete(q, qvalid, r, rvalid, u, v)
                m0 = torch.sqrt(u.mean() * v.mean())
                need(torch.equal(m0, ctx.common[index]['M0'][pos]), 'ORIGINAL_M0_EXACT')
                part = dict(version=VERSION, position=pos, physical_row=pair['physical_row'],
                    query_id=source['query_id'], execution_ordinal=index,
                    matched_reference=matched, local_scalars=local, matched_indices=indices,
                    matched_valid=valid, M0=m0, reference=rdesc)
                atomic_torch(path, part)
                receipt = dict(status='TOKEN_COMPETITION_CANDIDATE_PASS', version=VERSION,
                    context=pair_context, payload=binding(path), reference=rdesc)
                atomic_json(receipt_path, receipt)
            need(part['version'] == VERSION and part['position'] == pos and
                 part['physical_row'] == pair['physical_row'] and part['query_id'] == source['query_id'] and
                 part['execution_ordinal'] == index, 'PART_IDENTITY')
            need(torch.equal(part['M0'], ctx.common[index]['M0'][pos]), 'PART_M0_EXACT')
            records.append(dict(position=pos, receipt=binding(receipt_path), payload=receipt['payload']))
            payloads.append(part)
        payload = dict(version=VERSION, query_id=source['query_id'], execution_ordinal=index,
            axis128=source['candidate_physical_rows'], axis=source['candidate_physical_rows'], winner=source['winner'],
            query_tokens=q.float(), query_valid=qvalid,
            matched_reference=torch.stack([p['matched_reference'] for p in payloads]),
            local_scalars=torch.stack([p['local_scalars'] for p in payloads]),
            matched_indices=torch.stack([p['matched_indices'] for p in payloads]),
            matched_valid=torch.stack([p['matched_valid'] for p in payloads]),
            M0=torch.stack([p['M0'] for p in payloads]), scalar_names=SCALARS,
            context=ctx.context_binding, source=dict(fusion_query=ctx.rows[index]['payload'],
                query_image=qdesc, references=[p['reference'] for p in payloads], parts=records),
            labels_read=0, backbone_forwards=0, new_roma_forwards=0)
        need(torch.equal(payload['M0'], ctx.common[index]['M0']), 'COMMON_M0_EXACT')
        atomic_torch(target, payload)
        receipt = dict(status='TOKEN_COMPETITION_QUERY_PASS', version=VERSION,
            query_id=source['query_id'], execution_ordinal=index, payload=binding(target),
            context=ctx.context_binding, source=payload['source'], candidates=128,
            original_query_tokens=len(q), valid_query_tokens=int(qvalid.sum()),
            seconds=time.monotonic() - start, labels_read=0, backbone_forwards=0, new_roma_forwards=0)
        atomic_json(target.with_suffix('.json'), receipt)
        print(json.dumps(dict(status=receipt['status'], index=index, seconds=receipt['seconds'],
                              tokens=len(q), candidates=128)), flush=True)
        return 0


def self_test():
    q = torch.zeros((5, 128), dtype=torch.float64)
    q[0, 0] = q[1, 1] = q[3, 0] = 1
    q[2, 0] = float('nan')
    boxes = torch.tensor([[0., 0., 1., 1.]] * 5)
    q, qv = normalized_visual(q, boxes, torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool))
    r = torch.zeros((4, 128), dtype=torch.float64)
    r[0, 0] = r[1, 0] = r[2, 1] = 1
    r[3, 1] = float('nan')
    r, rv = normalized_visual(r, boxes[:4], torch.ones(4, dtype=torch.bool))
    u = torch.tensor([.2, .3, .4, .5, .6], dtype=torch.float64)
    v = torch.tensor([.7, .8, .9, 0.], dtype=torch.float64)
    matched, local, indices, mv = compete(q, qv, r, rv, u, v)
    need(indices.tolist() == [0, 2, -1, -1, -1], 'SELFTEST_TIE_NAN_INVALID')
    need(local[0, 1] == 0 and local[1, 1] == 1, 'SELFTEST_TOP2')
    need(torch.equal(mv, qv) and (local[~mv] == 0).all() and (matched[~mv] == 0).all(), 'SELFTEST_MASK')
    empty = compete(q, qv, r, torch.zeros_like(rv), u, v)
    need((empty[0] == 0).all() and (empty[1] == 0).all() and (empty[2] == -1).all() and not empty[3].any(), 'SELFTEST_EMPTY_REF')
    one = compete(q, qv, r[:1], rv[:1], u, v[:1])
    need((one[1][:, 1] == 0).all(), 'SELFTEST_SINGLE_REF_GAP_ZERO')
    return dict(status='TOKEN_COMPETITION_EVIDENCE_SYNTHETIC_PASS', labels_read=0,
                checks=['ties_first_original_index', 'top2_gap', 'NaN_mask', 'nonvisual_mask',
                        'zero_norm_mask', 'all_invalid_reference_zero', 'singleton_gap_zero'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, default=OUT / 'protocol.json')
    ap.add_argument('--index', type=int)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--wall-seconds', type=float, default=480)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if a.self_test:
        print(json.dumps(self_test()), flush=True)
        return 0
    need(os.environ.get('SLURM_JOB_ID'), 'BATCH_REQUIRED_FOR_FULL_C128')
    ctx = Context(a.protocol)
    index = ctx.pilot_index() if a.pilot else a.index
    need(index is not None and 0 <= index < 128, 'INDEX_REQUIRED')
    return build(ctx, index, a.wall_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
