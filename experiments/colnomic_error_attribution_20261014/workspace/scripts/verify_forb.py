#!/usr/bin/env python3
"""CPU-only audit of complete official FORB caches, score arithmetic and metrics."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

# Cache replay must never initialize an encoder or a CUDA context.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import digest, file_hash, read_json
from evaluation.forb_data import DOMAINS, PROTOCOL, inventory
from evaluation.forb_scoring import ShardedEmbeddingBank, verify_scores

DATA = ROOT / 'other_datasets/FORB'
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')
DIMENSIONS = {'dinov3': 1024, 'siglip2': 1024, 'Qwen3-VL-Embedding-2B': 2048}
METRICS = ('R@1', 'R@5', 'R@10', 'MRR', 'mAP@1', 'mAP@5', 'mAP@10', 'mAP@100', 'mAP')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def result_path(data, model):
    return Path(data) / 'results' / model / ('bm25' if model == 'bm25' else 'image')


def ready(data, model):
    directory = result_path(data, model)
    paths = [directory / name for name in ('config.json', 'metrics.json', 'predictions.jsonl')]
    paths.extend(Path(data) / 'cache/scores' / model / name for name in
                 ('scores.npy', 'metadata.json', 'manifest.json'))
    return all(path.is_file() for path in paths)


def duplicate_audit(manifest):
    groups = defaultdict(lambda: {'query_ids': [], 'reference_ids': []})
    for population, kind in [('photos', 'query_ids'), ('references', 'reference_ids')]:
        for row in manifest[population]:
            groups[row['sha256']][kind].append(row['id'])
    duplicates = [dict(sha256=sha, **value) for sha, value in groups.items()
                  if len(value['query_ids']) + len(value['reference_ids']) > 1]
    overlap = [entry for entry in duplicates if entry['query_ids'] and entry['reference_ids']]
    return dict(policy='every official row retained; only encoding/OCR computation may reuse identical bytes',
                official_rows=sum(len(manifest[p]) for p in ('photos', 'references')),
                unique_image_sha256=len(groups), duplicate_sha256_groups=len(duplicates),
                cross_query_gallery_sha256_groups=len(overlap),
                cross_query_gallery_image_pairs=sum(len(v['query_ids']) * len(v['reference_ids']) for v in overlap),
                groups=duplicates)


def check_sources(sources):
    for relative, expected in sources.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, f'Invalid source path: {relative}')
        require(file_hash(ROOT / path) == expected, f'Execution source differs: {relative}')


def check_banks(data, model, manifest, settings):
    unique = {}
    for row in manifest['photos'] + manifest['references']:
        unique.setdefault(row['sha256'], row)
    source_files = ('evaluation/forb_encoding.py', 'evaluation/dense.py',
                    'evaluation/encoders.py', 'evaluation/milk10k.py')
    encoding_recipe = dict(model=model, spec=settings['models'][model],
        source={p: file_hash(ROOT / p) for p in source_files},
        inputs=[[r['sha256'], r['image_path']] for r in unique.values()],
        task_size=settings['encoding_task_size'])
    encoding_signature = digest(encoding_recipe)
    banks, reports = {}, {}
    for side, population in [('photo', 'photos'), ('ref', 'references')]:
        directory = data / 'cache/image_embeddings' / model / side
        bank = ShardedEmbeddingBank(directory / 'index.json')
        index, rows = bank.index, manifest[population]
        recipe = dict(encoding=encoding_signature, population=side,
                      rows=[[r['id'], r['image_path'], r['sha256']] for r in rows])
        require(index.get('schema') == 'forb.embeddings.v1' and index.get('model') == model
                and index.get('population') == side, f'Wrong bank contract: {directory}')
        require(index.get('ids') == [r['id'] for r in rows], f'Bank ID/order mismatch: {directory}')
        require(index.get('image_paths') == [r['image_path'] for r in rows]
                and index.get('image_sha256') == [r['sha256'] for r in rows],
                f'Bank input provenance mismatch: {directory}')
        require(index.get('encoding_recipe') == encoding_recipe and index.get('recipe') == recipe
                and index.get('signature') == digest(recipe), f'Stale embedding recipe: {directory}')
        require(index.get('unique_images') == len(unique)
                and np.isfinite(index.get('unique_encoding_seconds', np.nan))
                and index['unique_encoding_seconds'] >= 0, f'Invalid bank compute provenance: {directory}')
        check_sources(index['encoding_recipe']['source'])
        norms_max_error = 0.
        token_counts = []
        for shard in index['shards']:
            path = directory / shard['path']
            require(path.stat().st_size == shard['bytes'] and file_hash(path) == shard['sha256'],
                    f'Embedding shard file differs: {path}')
            content = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            first, stop = shard['start'], shard['stop']
            require(content.get('signature') == index['signature']
                    and content.get('ids') == index['ids'][first:stop]
                    and content.get('image_sha256') == index['image_sha256'][first:stop],
                    f'Embedding shard provenance differs: {path}')
            values = content['embeddings']
            require(len(values) == stop - first, f'Incomplete embedding shard: {path}')
            for value in values:
                require(isinstance(value, torch.Tensor) and torch.isfinite(value).all(),
                        f'Nonfinite embedding: {path}')
                if model in DIMENSIONS:
                    require(value.shape == (DIMENSIONS[model],) and value.dtype == torch.float32,
                            f'Incorrect dense embedding shape/dtype: {path}')
                else:
                    require(value.ndim == 2 and value.shape[1] == 128 and value.dtype == torch.float16,
                            f'Incorrect token embedding shape/dtype: {path}')
                norms = value.double().norm(dim=-1).reshape(-1)
                norms = norms[norms > 1e-8]
                require(len(norms) > 0, f'Image has no effective tokens: {path}')
                error = float((norms - 1).abs().max())
                require(error < .03, f'Embedding normalization failed: {path}')
                norms_max_error = max(norms_max_error, error)
                token_counts.append(len(norms))
        reports[side] = dict(images=len(rows), index_sha256=file_hash(directory / 'index.json'),
            shards=len(index['shards']), max_norm_error=norms_max_error,
            effective_tokens=dict(min=min(token_counts), median=float(np.median(token_counts)), max=max(token_counts)))
        banks[side] = bank
    return banks, reports


def check_score_cache(data, model, manifest, settings, recorded, recipe):
    directory = data / 'cache/scores' / model
    meta = read_json(directory / 'metadata.json')
    plan = read_json(directory / 'manifest.json')
    qids = [r['id'] for r in manifest['photos']]
    rids = [r['id'] for r in manifest['references']]
    require(plan['signature'] == digest({k: v for k, v in plan.items() if k != 'signature'}),
            'Invalid score manifest signature')
    require(plan['recipe'] == recipe and plan['query_ids'] == qids and plan['reference_ids'] == rids
            and plan['shape'] == [len(qids), len(rids)] and plan['dtype'] == 'float32'
            and plan['chunk_size'] == settings['score_chunk_size'], 'Score inventory/recipe mismatch')
    require(plan['scoring_source_sha256'] == file_hash(ROOT / 'evaluation/forb_scoring.py')
            and plan['shared_scoring_source_sha256'] == file_hash(ROOT / 'evaluation/scoring.py'),
            'Scoring execution source changed')
    require(all(meta.get(k) == v for k, v in plan.items()) and meta.get('status') == 'complete',
            'Score metadata does not match the complete chunk plan')
    require(recorded['score_metadata'] == meta and recorded['scoring'] == recipe,
            'Reported score provenance differs from cached scores')
    score_path = directory / 'scores.npy'
    require(meta['scores_file'] == 'scores.npy' and recorded['score_path'] == str(score_path.relative_to(data))
            and file_hash(score_path) == meta['scores_sha256']
            and score_path.stat().st_size == meta['scores_bytes'], 'Score matrix checksum/path differs')
    scores = np.load(score_path, mmap_mode='r')
    require(scores.shape == (len(qids), len(rids)) and scores.dtype == np.float32, 'Invalid score matrix')
    first = 0
    for chunk_meta in meta['chunks']:
        start, stop = chunk_meta['start'], chunk_meta['stop']
        require(start == first and stop == min(start + plan['chunk_size'], len(qids)),
                'Nonconsecutive score chunk coverage')
        relative = Path(chunk_meta['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid score chunk path')
        path = directory / relative
        require(file_hash(path) == chunk_meta['sha256'], f'Score chunk checksum differs: {path}')
        chunk = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        require(chunk.get('signature') == plan['signature'] and chunk.get('recipe') == recipe
                and chunk.get('query_ids') == qids[start:stop] and chunk.get('reference_ids') == rids
                and chunk.get('query_start') == start and chunk.get('query_stop') == stop,
                f'Score chunk provenance differs: {path}')
        values = chunk['scores'].numpy()
        require(values.dtype == np.float32 and values.shape == (stop - start, len(rids))
                and np.isfinite(values).all(), f'Invalid chunk score matrix: {path}')
        checksum = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        require(checksum == chunk['scores_sha256'] == chunk_meta['scores_sha256']
                and np.array_equal(values, scores[start:stop]), f'Merged score/chunk mismatch: {path}')
        first = stop
    require(first == len(qids), 'Score chunks do not cover all official queries')
    return scores, meta


def check_rankings(scores, manifest, predictions, report, query_valid=None, reference_valid=None):
    """Independent lexicographic sorting, direct rank counts, and binary AP."""
    queries, references = manifest['photos'], manifest['references']
    nq, nr = len(queries), len(references)
    require(len(predictions) == nq, 'Prediction count differs from official queries')
    qvalid = np.ones(nq, dtype=bool) if query_valid is None else np.asarray(query_valid, dtype=bool)
    rvalid = np.ones(nr, dtype=bool) if reference_valid is None else np.asarray(reference_valid, dtype=bool)
    values = np.zeros((nq, len(METRICS)), dtype=np.float64)
    failures = []
    rows = np.arange(nr)
    for i, (query, prediction) in enumerate(zip(queries, predictions, strict=True)):
        positive = query['positive_indices']
        require(prediction['photo_id'] == query['id'] and prediction['domain'] == query['domain']
                and prediction['difficulty'] == query['difficulty']
                and prediction['positive_indices'] == positive
                and prediction['positive_reference_ids'] == [references[j]['id'] for j in positive],
                f'Prediction identity/positive mismatch: {query["id"]}')
        row = scores[i]
        require(np.isfinite(row).all(), f'Nonfinite score row: {query["id"]}')
        order = np.lexsort((rows, -row))
        # Count all strictly better scores and preceding equal scores directly.
        ranks = [int(1 + np.count_nonzero(row > row[j]) + np.count_nonzero(row[:j] == row[j]))
                 if qvalid[i] and rvalid[j] else None for j in positive]
        available = [rank for rank in ranks if rank is not None]
        rank = min(available) if available else None
        failure = None if rank is not None else ('empty_query_ocr' if not qvalid[i]
                                                else 'empty_positive_reference_ocr')
        require(prediction['positive_ranks'] == ranks and prediction['rank'] == rank
                and prediction['failure'] == failure, f'Positive ranks/failure mismatch: {query["id"]}')
        if failure:
            failures.append((i, failure))
        top = order[:10] if qvalid[i] else np.array([], dtype=int)
        require(prediction['top10_indices'] == top.tolist()
                and prediction['top10_ids'] == [references[j]['id'] for j in top]
                and prediction['top10_scores'] == [float(row[j]) for j in top],
                f'Top-10 ranking/tie mismatch: {query["id"]}')
        if rank is not None:
            hits = np.isin(order, [j for j in positive if rvalid[j]])
            precision = np.cumsum(hits, dtype=np.float64) / np.arange(1, nr + 1)
            contributions = precision * hits
            values[i, :4] = (rank <= 1, rank <= 5, rank <= 10, 1. / rank)
            for column, k in enumerate((1, 5, 10, 100, nr), start=4):
                numerator = float(contributions[:k].sum())
                values[i, column] = numerator / max(len(positive), numerator, 1)
        for column, name in enumerate(METRICS):
            require(abs(prediction['metrics'][name] - values[i, column]) < 1e-12,
                    f'Per-query {name} mismatch: {query["id"]}')
        if not qvalid[i]:
            require(not row.any(), f'Empty OCR query has nonzero BM25 scores: {query["id"]}')
    failed = np.zeros(nq, dtype=bool)
    for i, _ in failures:
        failed[i] = True

    def aggregate(indices, recorded):
        selected = values[indices]
        require(recorded['queries'] == len(indices)
                and recorded['correct_at_1'] == int(selected[:, 0].sum())
                and recorded['failures'] == int(failed[indices].sum()), 'Aggregate counts differ')
        for column, name in enumerate(METRICS):
            require(abs(recorded[name] - selected[:, column].mean()) < 1e-12,
                    f'Aggregate {name} differs')

    aggregate(np.arange(nq), report)
    for attribute in ('domain', 'difficulty'):
        labels = np.asarray([q[attribute] for q in queries])
        require(set(report['by_' + attribute]) == set(labels), f'Missing {attribute} groups')
        for label in dict.fromkeys(labels):
            aggregate(np.flatnonzero(labels == label), report['by_' + attribute][label])
    for column, name in enumerate(METRICS):
        independent = np.mean([values[[q['domain'] == domain for q in queries], column].mean()
                               for domain in dict.fromkeys(q['domain'] for q in queries)])
        require(abs(independent - report['domain_macro_average'][name]) < 1e-12,
                f'Domain macro {name} differs')
    require(report['references'] == nr and report['protocol'] == PROTOCOL
            and report['official_metrics'] == ['mAP@1', 'mAP@5']
            and report['ap_denominator'] == 'max(number_of_all_groundtruths, precision_sum, 1)'
            and report['tie_break'] == 'ascending original gallery row index'
            and report['failure_reasons'] == dict(Counter(reason for _, reason in failures)),
            'Metrics protocol/failure counts differ')
    return dict(queries=nq, gallery=nr, positive_links=sum(len(q['positive_indices']) for q in queries),
                full_rankings_verified=nq, all_positive_ranks_verified=True,
                official_map_denominator_verified=True, domain_and_difficulty_verified=True,
                failures=len(failures))


def arithmetic_check(model, banks, scores, manifest, texts=None, query_block=128):
    if model in DIMENSIONS:
        # At most 715 MB for the largest FP64 reference matrix, plus query chunks.
        reference = np.empty((len(banks['ref']), DIMENSIONS[model]), dtype=np.float64)
        for start in range(0, len(reference), 1024):
            reference[start:start + 1024] = torch.stack(banks['ref'][start:start + 1024]).numpy()
        largest = 0.
        for start in range(0, len(banks['photo']), query_block):
            query = torch.stack(banks['photo'][start:start + query_block]).numpy().astype(np.float64)
            expected = query @ reference.T
            actual = scores[start:start + len(query)]
            error = float(np.max(np.abs(expected - actual)))
            largest = max(largest, error)
            require(np.allclose(actual, expected, atol=3e-5, rtol=3e-5), f'Dense FP64 mismatch: {model}/{start}')
        return dict(method='full dense matrix independently recomputed with NumPy FP64',
                    score_pairs=int(scores.size), max_abs_error=largest, atol=3e-5, rtol=3e-5)
    by_domain = {domain: [i for i, row in enumerate(manifest['photos']) if row['domain'] == domain]
                 for domain in DOMAINS}
    if model == 'colnomic7b':
        pairs = []
        for indices in by_domain.values():
            qi = indices[len(indices) // 2]
            positive = manifest['photos'][qi]['positive_indices'][0]
            candidate = (qi * 7919 + 31) % scores.shape[1]
            if candidate == positive:
                candidate = (candidate + 1) % scores.shape[1]
            pairs.extend([(qi, positive), (qi, candidate)])
        checks = []
        for qi, ri in pairs:
            checked = verify_scores([banks['photo'][qi]], [banks['ref'][ri]],
                                     scores[qi:qi + 1, ri:ri + 1], 'maxsim')
            checks.append(dict(query_index=qi, reference_index=ri, **checked))
        return dict(method='16 deterministic MaxSim pairs independently recomputed with NumPy FP64',
                    score_pairs=len(checks), max_abs_error=max(c['max_abs_error'] for c in checks), pairs=checks)
    query_texts, reference_texts, parameters = texts
    indices = sorted({i for domain_indices in by_domain.values() for i in (domain_indices[0], domain_indices[-1])})
    for predicate in (lambda text: not re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE),):
        empty = next((i for i, text in enumerate(query_texts) if predicate(text)), None)
        if empty is not None:
            indices = sorted(set(indices) | {empty})
    checked = verify_scores([query_texts[i] for i in indices], reference_texts, scores[indices],
                            'bm25', **parameters)
    return dict(checked, method='fixed complete BM25 query rows, independent NumPy FP64 and full-gallery IDF',
                query_indices=indices, score_pairs=len(indices) * len(reference_texts))


def verify_model(data, model, manifest, settings):
    started = time.monotonic()
    directory = result_path(data, model)
    recorded = read_json(directory / 'config.json')
    report = read_json(directory / 'metrics.json')
    predictions = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines() if line]
    mode = 'bm25' if model == 'bm25' else 'image'
    require(recorded['protocol'] == PROTOCOL and recorded['model'] == model and recorded['mode'] == mode
            and recorded['dataset_sha256'] == file_hash(data / 'cache/manifest.json')
            and recorded['training'] == 'none' and recorded['labels_used_only_for_evaluation'] is True
            and recorded['metrics_source'] == file_hash(ROOT / 'evaluation/forb_data.py')
            and recorded['empty_ocr'] == settings['empty_ocr'] and recorded['tie_break'] == settings['tie_break'],
            f'Result contract differs: {model}')
    expected_spec = settings['bm25'] if model == 'bm25' else settings['models'][model]
    require(recorded['specification'] == expected_spec, f'Model specification differs: {model}')
    recipe = dict(protocol=PROTOCOL, model=model, dataset=file_hash(data / 'cache/manifest.json'))
    banks, bank_checks, texts = {}, {}, None
    qvalid = rvalid = None
    if model == 'bm25':
        from evaluation.forb_ocr import verify_cache
        ocr_meta = verify_cache(data)
        populations = [read_json(data / 'cache/ocr' / f'{side}.json') for side in ('photo', 'ref')]
        qtexts, rtexts = [[bank['entries'][r['id']]['text'] for r in manifest[population]]
                          for bank, population in zip(populations, ('photos', 'references'), strict=True)]
        has_tokens = lambda text: bool(re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE))
        qvalid, rvalid = [has_tokens(t) for t in qtexts], [has_tokens(t) for t in rtexts]
        recipe.update(bm25=settings['bm25'], query_ocr=file_hash(data / 'cache/ocr/photo.json'),
                      reference_ocr=file_hash(data / 'cache/ocr/ref.json'))
        texts = (qtexts, rtexts, settings['bm25'])
        bank_checks = dict(ocr_signature=ocr_meta['signature'], empty_token_queries=qvalid.count(False),
                           empty_token_references=rvalid.count(False))
    else:
        banks, bank_checks = check_banks(data, model, manifest, settings)
        base = data / 'cache/image_embeddings' / model
        recipe.update(model_spec=settings['models'][model], query=file_hash(base / 'photo/index.json'),
                      reference=file_hash(base / 'ref/index.json'),
                      arithmetic='FP32 mean over valid query-token maxima' if model == 'colnomic7b'
                      else 'FP32 normalized dot product')
        if model == 'colnomic7b':
            block_names = ('query_batch', 'reference_batch', 'token_block')
            execution_path = data / 'maxsim_execution.json'
            execution = read_json(execution_path) if execution_path.exists() else settings
            recipe['score_blocks'] = {name: execution[name] for name in block_names}
            if execution.get('backend'):
                require(execution['backend'] == 'resident_query_cached_gallery_fp32_v1',
                        'Unsupported MaxSim execution backend')
                recipe['maxsim_backend'] = dict(name=execution['backend'],
                    source_sha256=file_hash(ROOT / 'evaluation/forb_maxsim_fast.py'))
    scores, metadata = check_score_cache(data, model, manifest, settings, recorded, recipe)
    print(f'{model}: cache provenance passed; checking every ranking', flush=True)
    ranking = check_rankings(scores, manifest, predictions, report, qvalid, rvalid)
    if rvalid is not None:
        empty = np.flatnonzero(~np.asarray(rvalid))
        for start in range(0, len(scores), 256):
            require(not scores[start:start + 256, empty].any(), 'Empty OCR reference has nonzero BM25 scores')
    print(f'{model}: complete rankings passed; checking independent FP64 arithmetic', flush=True)
    arithmetic = arithmetic_check(model, banks, scores, manifest, texts)
    require(not torch.cuda.is_initialized(), 'Cache-only verifier initialized CUDA')
    output = dict(schema='forb.verification.v1', model=model, mode=mode, passed=True,
        verified_at_utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=file_hash(Path(__file__)),
        dataset_sha256=file_hash(data / 'cache/manifest.json'), score_sha256=metadata['scores_sha256'],
        score_metadata_sha256=file_hash(data / 'cache/scores' / model / 'metadata.json'),
        result_files={name: file_hash(directory / name) for name in ('config.json', 'metrics.json', 'predictions.jsonl')},
        banks=bank_checks, ranking=ranking, arithmetic=arithmetic,
        cache_only=True, cuda_initialized=False, verification_seconds=time.monotonic() - started)
    save_json(directory / 'verification.json', output)
    print(f'{model}: PASS, FP64 maximum absolute error {arithmetic["max_abs_error"]:.3g}', flush=True)
    return output


def run(data, models=None, ready_only=False):
    data = Path(data).resolve()
    requested = models or list(MODELS)
    selected = [model for model in requested if ready(data, model)] if ready_only else requested
    if not selected:
        print('No complete FORB result is ready for verification.', flush=True)
        return None
    for model in selected:
        require(ready(data, model), f'Incomplete FORB result: {model}; use --ready to skip unfinished models')
    manifest = read_json(data / 'cache/manifest.json')
    actual = inventory(data, verify_images=True)
    require(all(manifest.get(key) == value for key, value in actual.items()), 'Official inventory or image bytes changed')
    require(manifest['protocol'] == PROTOCOL and len(manifest['photos']) == 8995
            and len(manifest['references']) == 43585, 'Expected the complete official FORB Subset')
    settings = read_json(data / 'baseline_config.json')
    require(settings['protocol'] == PROTOCOL and settings['bm25'] == {'k1': 1.2, 'b': .75},
            'Unexpected FORB protocol/BM25 parameters')
    fresh = {model: verify_model(data, model, manifest, settings) for model in selected}
    result = data / 'results'
    result.mkdir(parents=True, exist_ok=True)
    with (result / '.verification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verified = dict(fresh)
        for model in MODELS:
            path = result_path(data, model) / 'verification.json'
            if model in verified or not ready(data, model) or not path.exists():
                continue
            old = read_json(path)
            directory = result_path(data, model)
            if (old.get('passed') and old.get('dataset_sha256') == file_hash(data / 'cache/manifest.json')
                    and old.get('verifier_sha256') == file_hash(Path(__file__))
                    and all(old.get('result_files', {}).get(name) == file_hash(directory / name)
                            for name in ('config.json', 'metrics.json', 'predictions.jsonl'))
                    and old.get('score_metadata_sha256') == file_hash(data / 'cache/scores' / model / 'metadata.json')):
                verified[model] = old
        audit = dict(schema='forb.verification_summary.v1', protocol=PROTOCOL, passed=True,
            all_planned_results_verified=set(verified) == set(MODELS),
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
            dataset_sha256=file_hash(data / 'cache/manifest.json'), verified_models=list(verified),
            pending_models=[model for model in MODELS if model not in verified],
            duplicates=duplicate_audit(manifest), results=verified,
            cache_only=True, cuda_initialized=False, verifier_sha256=file_hash(Path(__file__)))
        save_json(result / 'verification.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', action='append', choices=MODELS, help='Repeat to select several baselines')
    parser.add_argument('--ready', action='store_true', help='Verify only selected models with complete final results')
    parser.add_argument('--threads', type=int, default=4, help='CPU BLAS/PyTorch threads, default 4')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        run(args.data, args.model, args.ready)
    else:
        with threadpool_limits(limits=args.threads):
            run(args.data, args.model, args.ready)


if __name__ == '__main__':
    main()
