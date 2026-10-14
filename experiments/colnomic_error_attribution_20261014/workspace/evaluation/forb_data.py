"""Official FORB Subset inventory and retrieval metrics, without input filtering.

Labels and difficulty are evaluation metadata only. Encoders consume image_path;
no domain, ground truth, difficulty, or original URL belongs in a model prompt.
"""
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import digest, file_hash, read_json, write_json


PROTOCOL = 'forb.official_subset.query8995.gallery43585.v1'
DOMAINS = ('animated_cards', 'photorealistic_cards', 'bookcovers', 'paintings',
           'currency', 'logos', 'packaged_goods', 'movie_posters')
COUNTS = {'animated_cards': (4294, 9588), 'photorealistic_cards': (1850, 935),
          'bookcovers': (849, 7443), 'paintings': (207, 317),
          'currency': (448, 1002), 'logos': (469, 329),
          'packaged_goods': (519, 2302), 'movie_posters': (359, 21669)}
DIFFICULTIES = ('easy', 'medium', 'hard')


def _ndjson(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _build_inventory(data, domains=DOMAINS, expected_counts=COUNTS):
    """Build from immutable official lists; separate from costly image hashing."""
    data = Path(data)
    folder = data / 'metadata/FORB_subset'
    provenance_path = data / 'provenance/image_files.csv'
    with provenance_path.open(newline='') as stream:
        provenance_rows = list(csv.DictReader(stream))
    provenance = {row['path']: row for row in provenance_rows}
    _require(len(provenance) == len(provenance_rows), 'Duplicate provenance image paths')
    gallery_paths = (folder / 'all_database_images.txt').read_text().splitlines()
    _require(all(gallery_paths), 'Empty path in official gallery')
    photos, references, offsets = [], [], {}
    hashes = {'metadata/FORB_subset/all_database_images.txt':
              file_hash(folder / 'all_database_images.txt'),
              'provenance/image_files.csv': file_hash(provenance_path)}

    def record(domain, role, local_index, path, global_index):
        _require(path in provenance, f'Image absent from provenance: {path}')
        _require(path.startswith(f'{domain}/{role}/') and '..' not in Path(path).parts,
                 f'Unexpected image path: {path}')
        sha256 = provenance[path]['sha256']
        _require(len(sha256) == 64 and all(c in '0123456789abcdef' for c in sha256),
                 f'Invalid provenance SHA-256: {path}')
        return dict(index=global_index, id=f'{domain}:{role}:{local_index + 1:05d}',
                    image_path=path, sha256=sha256, domain=domain)

    for domain in domains:
        lists = {}
        for kind in ('query', 'database', 'gt'):
            path = folder / f'{domain}_{kind}.ndjson'
            lists[kind] = _ndjson(path)
            hashes[str(path.relative_to(data))] = file_hash(path)
        queries, database, groundtruth = lists['query'], lists['database'], lists['gt']
        if expected_counts is not None:
            _require((len(queries), len(database)) == expected_counts[domain],
                     f'Official subset count mismatch: {domain}')
        _require(len(queries) == len(groundtruth), f'Ground truth count mismatch: {domain}')
        offset = len(references)
        offsets[domain] = offset
        database_paths = [r['image'] for r in database]
        _require(gallery_paths[offset:offset + len(database)] == database_paths,
                 f'Official gallery order/offset mismatch: {domain}')
        for j, path in enumerate(database_paths):
            references.append(record(domain, 'database', j, path, offset + j))
        for j, (query, gt) in enumerate(zip(queries, groundtruth, strict=True)):
            _require(gt['query_id'] == j + 1 and gt['query_image'] == query['image'],
                     f'Query identity/order mismatch: {domain}/{j + 1}')
            local_ids, positive_paths = gt['groundtruth_ids'], gt['groundtruth_images']
            _require(bool(local_ids) and len(local_ids) == len(positive_paths),
                     f'Missing or inconsistent positives: {domain}/{j + 1}')
            indices = []
            for local_id, positive_path in zip(local_ids, positive_paths, strict=True):
                _require(isinstance(local_id, int) and 1 <= local_id <= len(database),
                         f'Invalid 1-based positive index: {domain}/{j + 1}/{local_id}')
                _require(database_paths[local_id - 1] == positive_path,
                         f'Positive path/index mismatch: {domain}/{j + 1}/{local_id}')
                index = offset + local_id - 1
                _require(gallery_paths[index] == positive_path,
                         f'Positive gallery offset mismatch: {domain}/{j + 1}')
                indices.append(index)
            _require(gt['difficulty'] in DIFFICULTIES,
                     f'Unknown difficulty: {domain}/{j + 1}')
            row = record(domain, 'query', j, query['image'], len(photos))
            row.update(positive_indices=indices,
                       positive_ids=[references[i]['id'] for i in indices],
                       difficulty=gt['difficulty'])
            photos.append(row)
    _require(len(references) == len(gallery_paths), 'Unaccounted official gallery rows')
    return dict(schema='forb.manifest.v1', protocol=PROTOCOL,
                dataset_sha256=digest(hashes), source_files=hashes,
                counts=dict(photos=len(photos), references=len(references),
                            positive_links=sum(len(p['positive_indices']) for p in photos)),
                domain_offsets=offsets,
                gallery='all eight domains; original all_database_images.txt row order',
                duplicate_policy='preserve every official row; no path/hash deduplication',
                tie_break='ascending original gallery row index',
                labels_usage='evaluation only; image encoders receive image pixels only',
                photos=photos, references=references)


def inventory(data, verify_images=True):
    """Read all 8,995 queries and 43,585 references, optionally hash every file."""
    data = Path(data)
    manifest = _build_inventory(data)
    if verify_images:
        # Repeated paths still remain distinct manifest rows; read each file once.
        expected = {r['image_path']: r['sha256']
                    for side in ('photos', 'references') for r in manifest[side]}

        def verify(item):
            relative, expected_hash = item
            _require(file_hash(data / relative) == expected_hash,
                     f'Image SHA-256 mismatch: {relative}')

        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(verify, expected.items()):
                pass
    return manifest


def prepare_manifest(data, verify_images=True):
    """Verify and freeze the manifest; never silently replace a different run."""
    data = Path(data)
    manifest = inventory(data, verify_images=verify_images)
    path = data / 'cache/manifest.json'
    if path.exists():
        existing = read_json(path)
        _require(all(existing.get(k) == v for k, v in manifest.items()),
                 f'Existing FORB manifest disagrees with official inputs: {path}')
        return existing
    write_json(path, manifest)
    return manifest


def metrics(scores, manifest, query_valid=None, reference_valid=None, include_full_map=True):
    """Return (report, predictions) with official AP denominator and stable ties.

    The official get_map_at_k divides the precision sum by max(number of all
    ground truths, precision sum, 1), *not* min(number of positives, k).
    OCR-unavailable positives earn no credit but remain in that denominator.
    Every query, including empty OCR, remains in overall and grouped means.
    """
    if hasattr(scores, 'detach'):
        scores = scores.detach().cpu().float().numpy()
    a = np.asarray(scores)
    photos, references = manifest['photos'], manifest['references']
    nq, nr = len(photos), len(references)
    _require(nq > 0 and nr > 0 and a.shape == (nq, nr), 'Invalid FORB score matrix shape')
    _require(np.isfinite(a).all(), 'Non-finite FORB scores')
    query_valid = np.ones(nq, dtype=bool) if query_valid is None else np.asarray(query_valid, dtype=bool)
    reference_valid = np.ones(nr, dtype=bool) if reference_valid is None else np.asarray(reference_valid, dtype=bool)
    _require(query_valid.shape == (nq,), 'Invalid query validity shape')
    _require(reference_valid.shape == (nr,), 'Invalid reference validity shape')
    cutoffs = (1, 5, 10, 100)
    names = ['R@1', 'R@5', 'R@10', 'MRR'] + [f'mAP@{k}' for k in cutoffs]
    if include_full_map:
        names.append('mAP')
    values = np.zeros((nq, len(names)), dtype=np.float64)
    predictions, failed = [], np.zeros(nq, dtype=bool)
    for i, row in enumerate(photos):
        positives = np.asarray(row['positive_indices'], dtype=np.int64)
        _require(positives.ndim == 1 and len(positives) > 0
                 and ((positives >= 0) & (positives < nr)).all(),
                 f'Invalid positive indices for query {i}')
        order = np.argsort(-a[i], kind='stable')
        inverse = np.empty(nr, dtype=np.int64)
        inverse[order] = np.arange(1, nr + 1)
        available = reference_valid[positives]
        valid = bool(query_valid[i] and available.any())
        failed[i] = not valid
        positive_ranks = [int(inverse[p]) if query_valid[i] and usable else None
                          for p, usable in zip(positives, available, strict=True)]
        rank = min(r for r in positive_ranks if r is not None) if valid else None
        if valid:
            # A gallery row occurs once in the ranking, including repeated files.
            hit_ranks = np.sort(inverse[np.unique(positives[available])])
            precisions = np.arange(1, len(hit_ranks) + 1) / hit_ranks
            values[i, :4] = (rank <= 1, rank <= 5, rank <= 10, 1. / rank)
            for column, k in enumerate(cutoffs, start=4):
                numerator = float(precisions[hit_ranks <= k].sum())
                values[i, column] = numerator / max(len(positives), numerator, 1.)
            if include_full_map:
                numerator = float(precisions.sum())
                values[i, -1] = numerator / max(len(positives), numerator, 1.)
        top = order[:10] if query_valid[i] else np.array([], dtype=int)
        predictions.append(dict(photo_id=row['id'], domain=row['domain'],
            difficulty=row['difficulty'], positive_reference_ids=[references[p]['id'] for p in positives],
            positive_indices=positives.tolist(), positive_ranks=positive_ranks, rank=rank,
            failure=None if valid else ('empty_query_ocr' if not query_valid[i]
                                      else 'empty_positive_reference_ocr'),
            top10_indices=top.tolist(), top10_ids=[references[j]['id'] for j in top],
            top10_scores=[float(a[i, j]) for j in top],
            metrics=dict(zip(names, values[i].tolist()))))

    def aggregate(indices):
        sub = values[indices]
        return dict(queries=len(sub), **dict(zip(names, sub.mean(axis=0).tolist())),
                    correct_at_1=int(sub[:, 0].sum()), failures=int(failed[indices].sum()))

    report = aggregate(np.arange(nq))
    report.update(references=nr, protocol=manifest.get('protocol', PROTOCOL),
                  official_metrics=['mAP@1', 'mAP@5'],
                  ap_denominator='max(number_of_all_groundtruths, precision_sum, 1)',
                  tie_break='ascending original gallery row index')
    for key in ('domain', 'difficulty'):
        labels = np.array([row[key] for row in photos])
        report['by_' + key] = {label: aggregate(np.flatnonzero(labels == label))
                               for label in dict.fromkeys(labels.tolist())}
    report['domain_macro_average'] = {
        name: float(np.mean([part[name] for part in report['by_domain'].values()])) for name in names}
    report['failure_reasons'] = dict(Counter(p['failure'] for p in predictions if p['failure']))
    return report, predictions
