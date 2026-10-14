"""Exact, resumable full-gallery scoring for the official FORB Subset.

Embedding sequences may be memory mapped shards. A worker holds one advisory
file lock for each complete query chunk; its output includes every gallery row.
No candidate selection, token downsampling, or reduced precision scoring occurs.
"""
from bisect import bisect_right
from collections import Counter, OrderedDict
from collections.abc import Sequence
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import uuid

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .data import digest, file_hash, read_json
from .scoring import tokenize, valid_tokens


SCORING_VERSION = 'forb.full_gallery_fp32.v1'


def _temporary(path):
    return path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_tensor(path, value):
    temporary = _temporary(path)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ShardedEmbeddingBank(Sequence):
    """Read only requested shards, sharing mmap pages across local processes.

    ``index.json`` contains ``ids``, ``signature``, and consecutive ``shards``
    with ``path``, ``start``, ``stop``, ``sha256``. Paths are relative to the
    index. Each torch shard contains ``ids``, ``signature``, ``embeddings``.
    File hashes are verified once per unchanged shard per bank instance.
    """

    def __init__(self, index_path, cache_shards=2, verify_hashes=True):
        self.index_path = Path(index_path).resolve()
        self.index = read_json(self.index_path)
        self.ids = self.index['ids']
        self.signature = self.index['signature']
        self.shards = self.index['shards']
        self.cache_shards = int(cache_shards)
        self.verify_hashes = verify_hashes
        self._cache = OrderedDict()
        self._verified = {}
        if self.cache_shards < 1:
            raise ValueError('cache_shards must be positive')
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Empty or duplicate embedding bank IDs')
        position = 0
        for shard in self.shards:
            relative = Path(shard['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Shard path must stay inside its bank directory')
            if shard['start'] != position or not position < shard['stop'] <= len(self.ids):
                raise ValueError('Embedding shards must cover consecutive bank rows')
            if not shard.get('sha256'):
                raise ValueError('Missing embedding shard checksum')
            position = shard['stop']
        if position != len(self.ids):
            raise ValueError('Incomplete embedding bank index')
        self._starts = [s['start'] for s in self.shards]

    def __len__(self):
        return len(self.ids)

    def _load(self, number):
        if number in self._cache:
            self._cache.move_to_end(number)
            return self._cache[number]
        spec = self.shards[number]
        path = self.index_path.parent / spec['path']
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if self.verify_hashes and self._verified.get(number) != identity:
            if file_hash(path) != spec['sha256']:
                raise ValueError(f'Embedding shard checksum differs: {path}')
            self._verified[number] = identity
        bank = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if (bank.get('signature') != self.signature
                or bank.get('ids') != self.ids[spec['start']:spec['stop']]
                or len(bank.get('embeddings', [])) != spec['stop'] - spec['start']):
            raise ValueError(f'Embedding shard metadata differs: {path}')
        self._cache[number] = bank['embeddings']
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return bank['embeddings']

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            values = []
            while start < stop:
                number = bisect_right(self._starts, start) - 1
                spec = self.shards[number]
                end = min(stop, spec['stop'])
                part = self._load(number)[start - spec['start']:end - spec['start']]
                values.extend(part)
                start = end
            return values
        if not isinstance(key, int):
            raise TypeError('Embedding index must be an integer or slice')
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        number = bisect_right(self._starts, key) - 1
        return self._load(number)[key - self.shards[number]['start']]


def _positive_batch_sizes(**values):
    if any(not isinstance(v, int) or v < 1 for v in values.values()):
        raise ValueError(f'Batch and token block sizes must be positive integers: {values}')


def _dense(values):
    matrix = values.float() if isinstance(values, torch.Tensor) else torch.stack(list(values)).float()
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError('Invalid dense embedding matrix')
    if not ((matrix.norm(dim=-1) - 1).abs() < .03).all():
        raise ValueError('Dense scoring expects the normalized encoder output')
    return matrix


@torch.inference_mode()
def dense_scores(queries, references, device='cuda:0', query_batch=256,
                 reference_batch=4096, progress=None):
    """FP32 dot products of already normalized encoder vectors, as in MILK10k."""
    _positive_batch_sizes(query_batch=query_batch, reference_batch=reference_batch)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = torch.empty((len(queries), len(references)), dtype=torch.float32)
    if not len(queries) or not len(references):
        return output
    query = _dense(queries[:])
    for ri in range(0, len(references), reference_batch):
        reference = _dense(references[ri:ri + reference_batch]).to(device)
        if query.shape[1] != reference.shape[1]:
            raise ValueError('Query and reference embedding dimensions differ')
        for qi in range(0, len(query), query_batch):
            block = query[qi:qi + query_batch].to(device)
            output[qi:qi + len(block), ri:ri + len(reference)] = (block @ reference.T).cpu()
        if progress:
            progress(min(ri + reference_batch, len(references)), len(references))
    return output


def _tokens(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError('Invalid multivector embedding')
    value = valid_tokens(value)
    if not len(value):
        raise ValueError('An image contains no effective tokens')
    return value


@torch.inference_mode()
def maxsim_scores(queries, references, device='cuda:0', query_batch=2,
                  reference_batch=8, token_block=256, progress=None):
    """FP32 mean MaxSim over all nonzero query and reference tokens.

    Both token axes are blocked. A reference batch is loaded once per complete
    query chunk, and its GPU tensors are reused for every query minibatch.
    Padding is masked before maxima, including for entirely negative matches.
    """
    _positive_batch_sizes(query_batch=query_batch, reference_batch=reference_batch,
                          token_block=token_block)
    torch.backends.cuda.matmul.allow_tf32 = False
    qs = [_tokens(v) for v in queries]
    output = torch.empty((len(qs), len(references)), dtype=torch.float32)
    if not qs or not len(references):
        return output
    dimension = qs[0].shape[1]
    if any(v.shape[1] != dimension for v in qs):
        raise ValueError('Query embedding dimensions differ')
    for ri in range(0, len(references), reference_batch):
        rs = [_tokens(v) for v in references[ri:ri + reference_batch]]
        if any(v.shape[1] != dimension for v in rs):
            raise ValueError('Query and reference embedding dimensions differ')
        lengths = torch.tensor([len(r) for r in rs], device=device)
        reference = pad_sequence(rs, batch_first=True).to(device)
        reference_mask = torch.arange(reference.shape[1], device=device)[None] < lengths[:, None]
        reference_blocks = []
        for rt in range(0, reference.shape[1], token_block):
            block = reference[:, rt:rt + token_block]
            reference_blocks.append((block.flatten(0, 1).T.contiguous(), block.shape[1],
                                     reference_mask[:, rt:rt + token_block]))
        for qi in range(0, len(qs), query_batch):
            values = qs[qi:qi + query_batch]
            qlengths = torch.tensor([len(q) for q in values], device=device)
            query = pad_sequence(values, batch_first=True).to(device)
            query_mask = torch.arange(query.shape[1], device=device)[None] < qlengths[:, None]
            total = torch.zeros((len(query), len(reference)), device=device, dtype=torch.float32)
            for qt in range(0, query.shape[1], token_block):
                block = query[:, qt:qt + token_block]
                maxima = torch.full((len(query), len(reference), block.shape[1]),
                                    -torch.inf, device=device, dtype=torch.float32)
                for flat_reference, rtokens, mask in reference_blocks:
                    similarity = (block.flatten(0, 1) @ flat_reference).reshape(
                        len(query), block.shape[1], len(reference), rtokens).permute(0, 2, 1, 3)
                    maxima = torch.maximum(maxima, similarity.masked_fill(
                        ~mask[None, :, None, :], -torch.inf).amax(-1))
                total += maxima.masked_fill(~query_mask[:, None, qt:qt + block.shape[1]], 0).sum(-1)
            output[qi:qi + len(query), ri:ri + len(reference)] = (total / qlengths[:, None]).cpu()
        if progress:
            progress(min(ri + reference_batch, len(references)), len(references))
    return output


class BM25Index:
    """Build global gallery statistics once; query chunking cannot change IDF.

    Formula and per-token FP32 accumulation match ``scoring.bm25_scores``.
    Empty OCR queries deliberately retain their complete all-zero score rows.
    """

    def __init__(self, reference_texts, k1=1.2, b=.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Invalid BM25 parameters')
        self.k1, self.b = k1, b
        documents = [Counter(tokenize(t)) for t in reference_texts]
        if not documents:
            raise ValueError('BM25 needs a nonempty gallery')
        self.count = len(documents)
        self.lengths = np.asarray([sum(d.values()) for d in documents], dtype=np.float64)
        self.average = max(float(self.lengths.mean()), 1e-12)
        postings = {}
        for i, document in enumerate(documents):
            for token, tf in document.items():
                postings.setdefault(token, []).append((i, tf))
        self.postings = {}
        for token, hits in postings.items():
            indices, tf = map(np.array, zip(*hits))
            idf = math.log(1 + (self.count - len(hits) + .5) / (len(hits) + .5))
            self.postings[token] = indices, tf, idf

    def score(self, query_texts):
        scores = np.zeros((len(query_texts), self.count), dtype=np.float32)
        for qi, text in enumerate(query_texts):
            for token, qf in Counter(tokenize(text)).items():
                if token not in self.postings:
                    continue
                indices, tf, idf = self.postings[token]
                scores[qi, indices] += (qf * idf * tf * (self.k1 + 1) /
                    (tf + self.k1 * (1 - self.b + self.b * self.lengths[indices] / self.average)))
        return torch.from_numpy(scores)


def bm25_index(reference_texts, k1=1.2, b=.75):
    return BM25Index(reference_texts, k1=k1, b=b)


def score_bm25(query_texts, index):
    return index.score(query_texts)


def _score_manifest(query_ids, reference_ids, signature, chunk_size):
    _positive_batch_sizes(chunk_size=chunk_size)
    query_ids, reference_ids = list(query_ids), list(reference_ids)
    for ids in (query_ids, reference_ids):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Score inventories must be nonempty with unique IDs')
    value = dict(schema='forb.score_chunks.v1', scoring_version=SCORING_VERSION,
                 scoring_source_sha256=file_hash(Path(__file__)),
                 shared_scoring_source_sha256=file_hash(Path(__file__).with_name('scoring.py')),
                 recipe=signature,
                 query_ids=query_ids, reference_ids=reference_ids,
                 shape=[len(query_ids), len(reference_ids)], dtype='float32', chunk_size=chunk_size)
    value['signature'] = digest(value)
    return value


def _ensure_manifest(output_dir, expected):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output_dir / 'manifest.json'
        if path.exists():
            if read_json(path) != expected:
                raise ValueError(f'Stale score chunk manifest: {path}')
        else:
            _save_json(path, expected)


def _ranges(manifest):
    for start in range(0, manifest['shape'][0], manifest['chunk_size']):
        yield start, min(start + manifest['chunk_size'], manifest['shape'][0])


def _chunk_path(output_dir, start, stop):
    return output_dir / 'chunks' / f'{start:06d}-{stop:06d}.pt'


def _tensor_hash(scores):
    return hashlib.sha256(memoryview(scores.numpy()).cast('B')).hexdigest()


def _load_chunk(path, manifest, start, stop):
    chunk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    scores = chunk.get('scores')
    if (chunk.get('signature') != manifest['signature']
            or chunk.get('query_start') != start or chunk.get('query_stop') != stop
            or chunk.get('query_ids') != manifest['query_ids'][start:stop]
            or chunk.get('reference_ids') != manifest['reference_ids']):
        raise ValueError(f'Score chunk metadata differs: {path}')
    if (not isinstance(scores, torch.Tensor) or scores.dtype != torch.float32
            or tuple(scores.shape) != (stop - start, manifest['shape'][1])
            or not scores.is_contiguous() or not torch.isfinite(scores).all()
            or chunk.get('scores_sha256') != _tensor_hash(scores)):
        raise ValueError(f'Invalid score chunk: {path}')
    return chunk


def run_score_worker(output_dir, query_ids, reference_ids, signature, score_chunk,
                     chunk_size=32, max_chunks=None, progress=None):
    """Claim available query chunks and atomically cache exact full-gallery rows.

    ``score_chunk(start, stop)`` returns a CPU/GPU tensor or NumPy matrix. Workers
    must share a filesystem supporting cross-host ``flock``. An interrupted
    worker releases its lock automatically; the next invocation resumes it.
    Return ``complete=False`` if another worker still owns unfinished chunks.
    """
    if max_chunks is not None:
        _positive_batch_sizes(max_chunks=max_chunks)
    output_dir = Path(output_dir)
    manifest = _score_manifest(query_ids, reference_ids, signature, chunk_size)
    _ensure_manifest(output_dir, manifest)
    (output_dir / 'chunks').mkdir(exist_ok=True)
    (output_dir / 'locks').mkdir(exist_ok=True)
    result = dict(computed=0, cached=0, busy=0, signature=manifest['signature'])
    for start, stop in _ranges(manifest):
        path = _chunk_path(output_dir, start, stop)
        if path.exists():
            _load_chunk(path, manifest, start, stop)
            result['cached'] += 1
            continue
        with (output_dir / 'locks' / f'{start:06d}-{stop:06d}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result['busy'] += 1
                continue
            if path.exists():
                _load_chunk(path, manifest, start, stop)
                result['cached'] += 1
                continue
            started = time.monotonic()
            scores = torch.as_tensor(score_chunk(start, stop)).detach().cpu().contiguous()
            if (scores.dtype != torch.float32
                    or tuple(scores.shape) != (stop - start, manifest['shape'][1])
                    or not torch.isfinite(scores).all()):
                raise ValueError(f'Invalid returned scores for query rows {start}:{stop}')
            chunk = dict(schema='forb.score_chunk.v1', signature=manifest['signature'],
                         recipe=signature, query_start=start, query_stop=stop,
                         query_ids=manifest['query_ids'][start:stop],
                         reference_ids=manifest['reference_ids'], scores=scores,
                         scores_sha256=_tensor_hash(scores),
                         scoring_seconds=time.monotonic() - started,
                         worker=dict(host=socket.gethostname(), pid=os.getpid()))
            _save_tensor(path, chunk)
            result['computed'] += 1
            if progress:
                progress(start, stop, manifest['shape'][0])
        if max_chunks is not None and result['computed'] >= max_chunks:
            break
    result['complete'] = all(_chunk_path(output_dir, start, stop).exists()
                             for start, stop in _ranges(manifest))
    return result


def merge_score_chunks(output_dir, query_ids=None, reference_ids=None, signature=None,
                       chunk_size=32):
    """Atomically merge every chunk into FP32 ``scores.npy`` with full provenance.

    Returns metadata, including ``scores_file`` relative to ``output_dir``. The
    returned matrix can be read with ``np.load(path, mmap_mode='r')``. A missing
    chunk raises ``FileNotFoundError``; no partial gallery is ever published.
    Optional inventories/recipe revalidate the manifest against the caller.
    """
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / 'manifest.json')
    if digest({k: v for k, v in manifest.items() if k != 'signature'}) != manifest.get('signature'):
        raise ValueError('Invalid score manifest signature')
    if any(v is not None for v in (query_ids, reference_ids, signature)):
        if any(v is None for v in (query_ids, reference_ids, signature)):
            raise ValueError('Supply both inventories and the recipe to validate a merge')
        expected = _score_manifest(query_ids, reference_ids, signature, chunk_size)
        if expected != manifest:
            raise ValueError('Merge manifest differs from the requested score recipe')
    with (output_dir / '.merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = output_dir / 'scores.npy'
        metadata_path = output_dir / 'metadata.json'
        if target.exists() and metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata.get('signature') != manifest['signature']:
                raise ValueError('Stale merged score metadata')
            if file_hash(target) != metadata.get('scores_sha256'):
                raise ValueError('Merged score checksum differs')
            return metadata
        temporary = _temporary(target)
        chunks = []
        try:
            output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                               shape=tuple(manifest['shape']))
            for start, stop in _ranges(manifest):
                path = _chunk_path(output_dir, start, stop)
                chunk = _load_chunk(path, manifest, start, stop)
                output[start:stop] = chunk['scores'].numpy()
                chunks.append(dict(path=str(path.relative_to(output_dir)), start=start, stop=stop,
                                   sha256=file_hash(path), scores_sha256=chunk['scores_sha256'],
                                   scoring_seconds=chunk['scoring_seconds'], worker=chunk['worker']))
            output.flush()
            del output
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = dict(manifest, status='complete', scores_file=target.name,
                        scores_sha256=file_hash(target), scores_bytes=target.stat().st_size,
                        chunks=chunks, scoring_seconds=sum(c['scoring_seconds'] for c in chunks))
        _save_json(metadata_path, metadata)
        return metadata


def verify_scores(queries, references, scores, method, atol=3e-5, rtol=3e-5,
                  k1=1.2, b=.75):
    """Independent NumPy/FP64 audit for a caller-selected, small cached sample.

    Use ``method`` = ``dense``, ``maxsim``, or ``bm25``. For BM25, references must
    include the whole gallery so document statistics remain those of the run.
    This function checks score math; callers separately verify image encodings.
    """
    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    actual = array(scores)
    expected = np.zeros((len(queries), len(references)), dtype=np.float64)
    if method == 'dense':
        expected = np.stack([array(v) for v in queries]) @ np.stack([array(v) for v in references]).T
    elif method == 'maxsim':
        qvalues = [array(v) for v in queries]
        rvalues = [array(v) for v in references]
        qvalues = [v[np.linalg.norm(v, axis=1) > 1e-8] for v in qvalues]
        rvalues = [v[np.linalg.norm(v, axis=1) > 1e-8] for v in rvalues]
        if any(not len(v) for v in qvalues + rvalues):
            raise ValueError('No effective tokens in verification sample')
        for qi, query in enumerate(qvalues):
            for ri, reference in enumerate(rvalues):
                expected[qi, ri] = np.max(query @ reference.T, axis=1).mean()
    elif method == 'bm25':
        # Intentionally separate, document-at-a-time implementation, not postings.
        terms = lambda text: re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE)
        docs = [Counter(terms(text)) for text in references]
        lengths = [sum(doc.values()) for doc in docs]
        average = max(float(np.mean(lengths)), 1e-12)
        frequencies = Counter(token for doc in docs for token in doc)
        for qi, text in enumerate(queries):
            for ri, doc in enumerate(docs):
                for token, qcount in Counter(terms(text)).items():
                    tf = doc.get(token, 0)
                    if tf:
                        df = frequencies[token]
                        idf = np.log1p((len(docs) - df + .5) / (df + .5))
                        expected[qi, ri] += (qcount * idf * tf * (k1 + 1) /
                            (tf + k1 * (1 - b + b * lengths[ri] / average)))
    else:
        raise ValueError(f'Unknown verification method: {method}')
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid score verification matrix')
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return dict(method=method, shape=list(actual.shape), reference='independent_numpy_float64',
                max_abs_error=float(np.max(np.abs(actual - expected), initial=0)),
                atol=atol, rtol=rtol, passed=True)
