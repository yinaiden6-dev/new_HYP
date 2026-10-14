"""Independent gamma/lambda audit from frozen, complete-gallery geometry scores.

The coordinate shuffle changes coordinate-to-feature association only.  It leaves
the accepted OCR alignments, homographies, embeddings and availability unchanged.
Run with the project's rebut Python, -B, and --output to an independent directory.
"""
import argparse
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import anchors, grid, rank_rows, metrics
from evaluation.scoring import fuse

SEED = 20261013
FOCAL_IDS = [
    'photo_4a578988d89f5973', 'photo_65dbff1a3b69197f',
    'photo_da31992b29a6c690', 'photo_41de3eb383934ec3',
    'photo_21bd7df9ed6795e7', 'photo_1f48d45127db4129',
    'photo_0e87e0717ab1c215', 'photo_2bdbf879ee1d7dd8',
]


def stable_seed(*items):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, items)).encode()).digest()[:8], 'little')


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_inputs(folder):
    d = data()
    old = ROOT / 'results/token_mechanism_experiments/20261013_v1/stage2_geometry'
    oldaxes = read(old / 'score_axes.json')
    assert oldaxes['query_ids'] == d['qb']['ids']
    assert oldaxes['reference_ids'] == d['rb']['ids']
    assert oldaxes['reference_sha256'] == d['rb']['sha256']
    cache = np.load(old / 'full_gallery_scores.npz', allow_pickle=False)
    base = cache['original']
    penalty = cache['geo_penalty']
    lambdas = cache['lambdas']
    assert base.shape == (1117, 5405) and np.array_equal(base, d['image']['scores'].numpy())
    assert penalty.shape == (len(lambdas), *base.shape) and np.array_equal(penalty[0], base)
    fits = read(old / 'alignments.json')
    qindex = {qid: i for i, qid in enumerate(d['qb']['ids'])}
    rindex = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    ledger = read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')
    assert len(ledger) == len(qindex) and {r['photo_id'] for r in ledger} == set(qindex)
    d['ledger_by_id'] = {r['photo_id']: r for r in ledger}
    accepted = np.zeros(base.shape, bool)
    available = np.zeros(base.shape, np.float32)
    fitrows = defaultdict(list)
    token_counts = [int((v.float().norm(dim=1) > 1e-8).sum()) for v in d['qb']['embeddings']]
    for fit in fits:
        qi = qindex[fit['photo_id']]
        ri = rindex[fit['reference_id']]
        assert not accepted[qi, ri], 'duplicate accepted pair'
        accepted[qi, ri] = True
        available[qi, ri] = fit['subject_tokens'] / token_counts[qi]
        fitrows[qi].append((ri, fit))
    historical_bonus = cache['geo_bonus']
    decomposition_error = max(float(np.max(np.abs(
        historical_bonus[k] - (penalty[k] + weight * available))))
        for k, weight in enumerate(lambdas))
    assert decomposition_error < 2e-6
    assert np.all(penalty[:, ~accepted] == base[~accepted])
    has_text = [bool(d['qo'][qid]['text'].strip()) for qid in d['qb']['ids']]
    replay_fusion = fuse(torch.from_numpy(base), d['bm25']['scores'], has_text, .5).numpy()
    fusion_error = float(np.max(np.abs(replay_fusion - d['fusion']['scores'].numpy())))
    assert fusion_error < 2e-5
    paths = [
        'cache/image_embeddings/colnomic7b/photos.pt',
        'cache/image_embeddings/colnomic7b/references.pt',
        'cache/scores/colnomic7b/image.pt', 'cache/scores/bm25/bm25.pt',
        'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt',
        'dataset/annotations/subject_boxes.json', 'cache/ocr/photos.json',
        'cache/ocr/references.json', 'dataset/lists/photos.csv',
        'dataset/lists/train.csv', 'dataset/lists/val.csv', 'dataset/lists/test.csv',
        'dataset/lists/groups.csv', 'dataset/lists/positives.csv',
        'results/token_mechanism_analysis/F03/reproduction/token_metadata.json',
        'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json',
        str((old / 'full_gallery_scores.npz').relative_to(ROOT)),
        str((old / 'alignments.json').relative_to(ROOT)),
        str((old / 'selection.json').relative_to(ROOT)),
        str((old / 'score_axes.json').relative_to(ROOT)),
    ]
    inputs = {path: {'bytes': (ROOT / path).stat().st_size, 'sha256': sha(ROOT / path)} for path in paths}
    protocol = dict(
        schema='colnomic.split_geometry.v1', seed=SEED,
        source_code_sha256=sha(Path(__file__)), cpu_threads=2, device='cpu',
        full_gallery_shape=list(base.shape), gamma_grid=lambdas.tolist(), lambda_grid=lambdas.tolist(),
        formula='mean_i max_j(dot_ij - lambda*d_ij) + gamma*subject_fraction, for accepted alignments only',
        spatial_scope='confirmed subject visual Query tokens; reference template tokens distance 1; other Query tokens unchanged',
        selection='all validation identities; package-macro R1, then macro MRR, then smaller gamma+lambda, then smaller lambda, then gamma',
        shuffle='per accepted Query/candidate pair deterministic permutation of valid visual reference coordinates; dot columns and template coordinates unchanged',
        shuffle_parameters='parameters copied from corresponding real-coordinate validation-selected method; no shuffle/test selection',
        alignment='reuse frozen accepted OCR-content homographies; not pure geometry or a box-free deployable system',
        controls='standalone OCR alignment support, BM25 text, OCR exact anchor overlap, coordinate-only; no ColNomic features in standalone controls',
        historical_exposure='historical test was inspected; this is not a new blind test',
        prior_focal_cases='verification only; never used to choose train/validation/test or parameters',
        original_inputs_modified=False, inputs=inputs)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'protocol.json').exists():
        assert read(folder / 'protocol.json') == protocol, 'different protocol/input; use another output directory'
    else:
        write(folder / 'protocol.json', protocol)
    write(folder / 'score_axes.json', oldaxes)
    write(folder / 'split_verification.json', d['split_verification'])
    return d, cache, base, penalty, lambdas, fits, fitrows, accepted, available, has_text, protocol, {
        'historical_bonus_decomposition_max_error': decomposition_error,
        'baseline_fusion_replay_max_error': fusion_error,
    }


def select_parameters(d, base, penalty, lambdas, available, has_text, folder):
    selections, records = {}, []
    val_packages = sorted({d['photos'][qid]['package_id'] for qid in d['qb']['ids'] if d['splits'][qid] == 'val'})
    for config in ['B03', 'F03']:
        options = {'availability_only': [], 'spatial_only': [], 'combined': []}
        for li, lam in enumerate(lambdas):
            for gamma in lambdas:
                if li == 0 and gamma == 0:
                    raw = base
                else:
                    raw = penalty[li] + float(gamma) * available
                scores = raw if config == 'B03' else fuse(torch.from_numpy(raw), d['bm25']['scores'], has_text, .5).numpy()
                m = metrics(rank_rows(scores, d, 'val'))
                assert m['packages'] == len(val_packages)
                record = dict(configuration=config, gamma=float(gamma), lambda_value=float(lam), **m)
                records.append(record)
                objective = (m['package_macro_r1'], m['package_macro_mrr'], -float(gamma + lam), -float(lam), -float(gamma))
                item = (objective, dict(gamma=float(gamma), lambda_value=float(lam), lambda_index=li, validation_metrics=m))
                options['combined'].append(item)
                if li == 0: options['availability_only'].append(item)
                if gamma == 0: options['spatial_only'].append(item)
        selections[config] = {method: max(items, key=lambda item: item[0])[1] for method, items in options.items()}
    result = dict(selections=selections, validation_package_ids=val_packages,
                  validation_only=True, test_used_for_selection=False,
                  criterion='package_macro_r1, package_macro_mrr, smaller gamma+lambda, lambda, gamma',
                  historical_diagonal_selections=read(ROOT / 'results/token_mechanism_experiments/20261013_v1/stage2_geometry/selection.json')['selections'])
    if (folder / 'selection.json').exists(): assert read(folder / 'selection.json') == result
    else: write(folder / 'selection.json', result)
    csvout(folder / 'validation_grid.csv', records)
    return result


def shuffled_pair(query, reference, qxy, rxy, box, fit, lambdas, qid, rid):
    qi = torch.nonzero(query.float().norm(dim=1) > 1e-8).flatten().numpy()
    ri = torch.nonzero(reference.float().norm(dim=1) > 1e-8).flatten().numpy()
    q = query[qi].float()
    r = reference[ri].float()
    dot = q @ r.T
    maxima = dot.max(1).values
    original = float(maxima.mean())
    subject = np.isfinite(qxy[qi]).all(1) & ((qxy[qi] >= box[:2]) & (qxy[qi] < box[2:])).all(1)
    assert int(subject.sum()) == fit['subject_tokens']
    mapped = cv2.perspectiveTransform(qxy[qi[subject]][None].astype(np.float32), np.asarray(fit['homography'], np.float32))[0]
    finite = np.flatnonzero(np.isfinite(rxy[ri]).all(1))
    seed = stable_seed(SEED, qid, rid, 'target-coordinate-permutation')
    permutation = np.random.default_rng(seed).permutation(len(finite))
    shuffled = rxy[ri].copy()
    shuffled[finite] = shuffled[finite[permutation]]
    assert np.array_equal(np.sort(shuffled[finite], axis=0), np.sort(rxy[ri[finite]], axis=0))
    assert np.array_equal(np.isfinite(shuffled), np.isfinite(rxy[ri]))
    distances = np.ones((int(subject.sum()), len(ri)), np.float32)
    if len(finite):
        distances[:, finite] = np.clip(np.linalg.norm(mapped[:, None] - shuffled[finite][None], axis=2), 0, 1)
    distances[~np.isfinite(distances)] = 1
    dist = torch.from_numpy(distances)
    selected = torch.from_numpy(subject)
    subject_dot = dot[selected]
    baseline_subject = maxima[selected]
    changes = []
    for lam in lambdas:
        changes.append(float(((subject_dot - float(lam) * dist).max(1).values - baseline_subject).sum() / len(qi)))
    coordinate = float(np.exp(-10 * distances[:, finite].min(1).mean())) if len(finite) else 0.
    return np.asarray(changes, np.float32), original, dict(
        seed=seed, permuted_visual_tokens=len(finite), subject_tokens=int(subject.sum()),
        query_valid_tokens=len(qi), reference_valid_tokens=len(ri),
        permutation_sha256=hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest(),
        nonidentity_permutation=bool(len(finite) > 1 and np.any(permutation != np.arange(len(finite)))),
        shuffled_coordinate_only=coordinate)


def build_shuffle(d, base, lambdas, fitrows, available, folder, protocol):
    shard = folder / 'shuffle_query_checkpoints'
    shard.mkdir(exist_ok=True)
    signature = hashlib.sha256(str(protocol).encode()).hexdigest()
    path = folder / 'shuffled_geo_penalty.npy'
    resumed = path.exists()
    result = np.lib.format.open_memmap(path, mode='r+' if resumed else 'w+', dtype=np.float32, shape=(len(lambdas), *base.shape))
    if not resumed:
        for k in range(len(lambdas)): result[k] = base
        result.flush()
    audit_rows = []
    maximum = 0.
    started = time.monotonic()
    for qi, qid in enumerate(d['qb']['ids']):
        record_path = shard / f'{qi:04d}.npz'
        expected = sorted(fitrows.get(qi, []), key=lambda item: item[0])
        indices = np.asarray([ri for ri, fit in expected], np.int32)
        metadata_path = shard / f'{qi:04d}.json'
        if record_path.exists() and metadata_path.exists():
            metadata = read(metadata_path)
            cached = np.load(record_path, allow_pickle=False)
            assert metadata['signature'] == signature and metadata['photo_id'] == qid
            assert np.array_equal(cached['reference_indices'], indices)
            assert sha(record_path) == metadata['shard_sha256']
            scores = cached['scores']
            assert scores.shape == (len(lambdas), len(indices)) and np.isfinite(scores).all()
            rows = metadata['pairs']
            error = metadata['max_original_score_replay_error']
        else:
            scores = np.empty((len(lambdas), len(indices)), np.float32)
            rows = []
            error = 0.
            if expected:
                annotation = d['boxes'][qid]
                box = np.asarray(annotation['bbox'], np.float32) / np.tile([annotation['width'], annotation['height']], 2)
                qxy = grid(d['metadata']['photos'][qi])
                query = d['qb']['embeddings'][qi]
                for j, (ri, fit) in enumerate(expected):
                    rid = d['rb']['ids'][ri]
                    change, replay, audit = shuffled_pair(query, d['rb']['embeddings'][ri], qxy,
                        grid(d['metadata']['references'][ri]), box, fit, lambdas, qid, rid)
                    pair_error = abs(replay - float(base[qi, ri]))
                    assert pair_error < 2e-6
                    error = max(error, pair_error)
                    assert abs(audit['subject_tokens'] / audit['query_valid_tokens'] - float(available[qi, ri])) < 1e-7
                    scores[:, j] = base[qi, ri] + change
                    rows.append(dict(photo_id=qid, reference_id=rid, **audit, original_score_replay_error=pair_error))
            atomic_npz(record_path, reference_indices=indices, scores=scores)
            write(metadata_path, dict(signature=signature, photo_id=qid, pairs=rows,
                max_original_score_replay_error=error, shard_sha256=sha(record_path)))
        assert np.array_equal(scores[0], base[qi, indices])
        for k in range(len(lambdas)): result[k, qi, indices] = scores[k]
        result.flush()
        maximum = max(maximum, error)
        audit_rows.extend(rows)
        if (qi + 1) % 25 == 0 or qi + 1 == len(d['qb']['ids']):
            print(f'Shuffle {qi + 1}/1117 pairs={len(audit_rows)} elapsed={time.monotonic() - started:.1f}s', flush=True)
        write(folder / 'progress.json', dict(stage='coordinate_shuffle', queries_completed=qi + 1,
            total_queries=len(d['qb']['ids']), accepted_pairs_completed=len(audit_rows), last_query=qid,
            max_original_score_replay_error=maximum, elapsed_seconds=time.monotonic() - started))
    csvout(folder / 'coordinate_permutations.csv', audit_rows)
    assert np.array_equal(result[0], base)
    return result, maximum, audit_rows


def anchor_text_control(d):
    result = np.zeros(d['image']['scores'].shape, np.float32)
    postings = defaultdict(list)
    for ri, rid in enumerate(d['rb']['ids']):
        for text in anchors(d['ro'][rid]): postings[text].append(ri)
    for qi, qid in enumerate(d['qb']['ids']):
        annotation = d['boxes'][qid]
        box = np.asarray(annotation['bbox'], np.float32) / np.tile([annotation['width'], annotation['height']], 2)
        query_texts = anchors(d['qo'][qid], box)
        counts = Counter(ri for text in query_texts for ri in postings.get(text, []))
        for ri, count in counts.items(): result[qi, ri] = count / max(1, len(query_texts))
    return result


def describe_ranks(scores, d):
    ledger = d['ledger_by_id']
    result = []
    hashes = d['rb']['sha256']
    ids = d['rb']['ids']
    lookup = {rid: i for i, rid in enumerate(ids)}
    for qi, qid in enumerate(d['qb']['ids']):
        case = ledger[qid]
        positives = set(case['positive_reference_sha256'])
        wanted = positives | {hashes[lookup[case[k]]] for k in ['image_strongest_wrong', 'fusion_strongest_wrong']}
        seen, wanted_ranks = set(), {}
        top, strongest_wrong, best_positive = None, None, None
        for ri in np.argsort(-scores[qi], kind='stable'):
            value = hashes[ri]
            if value in seen: continue
            seen.add(value)
            top = ids[ri] if top is None else top
            if value in positives and best_positive is None: best_positive = len(seen)
            if value not in positives and strongest_wrong is None: strongest_wrong = ids[ri]
            if value in wanted: wanted_ranks[value] = len(seen)
            if len(wanted_ranks) == len(wanted) and strongest_wrong is not None: break
        assert best_positive is not None and strongest_wrong is not None
        result.append(dict(photo_id=qid, package_id=d['photos'][qid]['package_id'],
            group_id=d['groups'][d['photos'][qid]['package_id']], split=d['splits'][qid],
            rank=best_positive, top1=top, correct=best_positive == 1, strongest_wrong=strongest_wrong,
            positive_ranks={rid: wanted_ranks[hashes[lookup[rid]]] for rid in case['positive_reference_ids']},
            fixed_negative_ranks={case[k]: wanted_ranks[hashes[lookup[case[k]]]] for k in ['image_strongest_wrong', 'fusion_strongest_wrong']}))
    return result


def group_bootstrap(current, baseline, config, method, split, unit):
    groupkeys = sorted({r[unit] for r in current})
    groups = {key: [r for r in current if r[unit] == key] for key in groupkeys}
    values = []
    for key in groupkeys:
        pairs = [(r, baseline[r['photo_id']]) for r in groups[key]]
        values.append([np.mean([float(fn(r)) - float(fn(old)) for r, old in pairs]) for fn in [
            lambda r: r['rank'] == 1, lambda r: r['rank'] <= 5,
            lambda r: r['rank'] <= 10, lambda r: 1 / r['rank']]])
    values = np.asarray(values)
    seed = stable_seed(SEED, config, split, unit, 'paired-bootstrap')
    indices = np.random.default_rng(seed).integers(0, len(groupkeys), (5000, len(groupkeys)))
    distribution = values[indices].mean(1)
    return [dict(configuration=config, method=method, split=split, unit=unit,
        groups=len(groupkeys), queries=len(current), metric=key,
        group_macro_delta=float(values[:, k].mean()), ci95=np.quantile(distribution[:, k], [.025, .975]).tolist(),
        resamples=5000, seed=seed) for k, key in enumerate(['r1', 'r5', 'r10', 'mrr'])]


def evaluate(d, cache, base, penalty, lambdas, available, accepted, shuffle, selection, has_text, folder):
    text_control = anchor_text_control(d)
    array_path = folder / 'selected_full_gallery_scores.npz'
    score_arrays = dict(query_ids=np.asarray(d['qb']['ids']), reference_ids=np.asarray(d['rb']['ids']),
        reference_sha256=np.asarray(d['rb']['sha256']), subject_fraction=available,
        accepted_alignment=accepted, gamma_grid=lambdas, lambda_grid=lambdas)
    allpred, alltables, changes, bootstraps, pairrows = {}, [], [], [], []
    qindex = {qid: i for i, qid in enumerate(d['qb']['ids'])}
    rindex = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    baseline_image = describe_ranks(base, d)
    baseline_fusion = describe_ranks(d['fusion']['scores'].numpy(), d)
    joint = {a['photo_id']: ('both_correct' if a['correct'] and b['correct'] else
        'image_wrong_fusion_correct' if not a['correct'] and b['correct'] else
        'image_correct_fusion_wrong' if a['correct'] and not b['correct'] else 'both_wrong')
        for a, b in zip(baseline_image, baseline_fusion)}
    for config in ['B03', 'F03']:
        chosen = selection['selections'][config]
        historical = selection['historical_diagonal_selections'][config]
        hk = historical['index']
        hl = historical['lambda_value']
        methods = {
            'baseline': base,
            'availability_only': base + chosen['availability_only']['gamma'] * available,
            'spatial_only': penalty[chosen['spatial_only']['lambda_index']],
            'combined': penalty[chosen['combined']['lambda_index']] + chosen['combined']['gamma'] * available,
            'shuffled_spatial_only': shuffle[chosen['spatial_only']['lambda_index']],
            'shuffled_combined': shuffle[chosen['combined']['lambda_index']] + chosen['combined']['gamma'] * available,
            'historical_diagonal': penalty[hk] + hl * available,
            'historical_availability_only': base + hl * available,
            'historical_spatial_only': penalty[hk],
            'historical_shuffled_diagonal': shuffle[hk] + hl * available,
            'ocr_alignment_support': cache['geometry_only'],
            'coordinate_only': cache['coordinate_only'],
            'bm25_text_control': d['bm25']['scores'].numpy(),
            'ocr_anchor_overlap_control': text_control,
        }
        baseline = {r['photo_id']: r for r in (baseline_image if config == 'B03' else baseline_fusion)}
        for method, raw in methods.items():
            standalone = method in ['ocr_alignment_support', 'coordinate_only', 'bm25_text_control', 'ocr_anchor_overlap_control']
            scores = raw if config == 'B03' or standalone else fuse(torch.from_numpy(np.asarray(raw)), d['bm25']['scores'], has_text, .5).numpy()
            if config == 'F03' and method == 'baseline': scores = d['fusion']['scores'].numpy()
            assert scores.shape == (1117, 5405) and np.isfinite(scores).all()
            key = config + '__' + method
            score_arrays[key + '__raw'] = np.asarray(raw)
            score_arrays[key + '__scored'] = np.asarray(scores)
            predictions = describe_ranks(scores, d)
            allpred[key] = predictions
            for split in ['train', 'val', 'test', 'all']:
                subset = [r for r in predictions if split == 'all' or r['split'] == split]
                m = metrics(subset)
                repaired = sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in subset)
                harmed = sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in subset)
                alltables.append(dict(configuration=config, method=method, split=split,
                    repaired=repaired, harmed=harmed, net=repaired-harmed, **m))
                if method != 'baseline':
                    for unit in ['package_id', 'group_id']:
                        bootstraps.extend(group_bootstrap(subset, baseline, config, method, split, unit))
                for joint_class in sorted(set(joint.values())):
                    stratum = [r for r in subset if joint[r['photo_id']] == joint_class]
                    if stratum:
                        alltables.append(dict(configuration=config, method=method, split=split,
                            stratum=joint_class, repaired=sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in stratum),
                            harmed=sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in stratum), **metrics(stratum)))
            for row in predictions:
                old = baseline[row['photo_id']]
                changes.append(dict(configuration=config, method=method, **row,
                    baseline_joint_outcome=joint[row['photo_id']], baseline_rank=old['rank'], baseline_top1=old['top1'],
                    rank_delta=row['rank']-old['rank'], change='repaired' if row['correct'] and not old['correct'] else
                    'harmed' if not row['correct'] and old['correct'] else 'unchanged'))
            # Every original positive is preserved; repeated negative candidate IDs merge roles.
            for case in d['ledger_by_id'].values():
                qi = qindex[case['photo_id']]
                negative_roles = defaultdict(list)
                for role in ['image_strongest_wrong', 'fusion_strongest_wrong']:
                    negative_roles[case[role]].append(role)
                for positive in case['positive_reference_ids']:
                    pi = rindex[positive]
                    for negative, roles in negative_roles.items():
                        ni = rindex[negative]
                        pa, na = bool(accepted[qi, pi]), bool(accepted[qi, ni])
                        layer = 'both_align' if pa and na else 'positive_only' if pa else 'negative_only' if na else 'neither_align'
                        pairrows.append(dict(configuration=config, method=method, photo_id=case['photo_id'],
                            package_id=case['package_id'], split=case['split'], positive=positive, negative=negative,
                            negative_roles=roles, alignment_stratum=layer, positive_alignment=pa, negative_alignment=na,
                            positive_subject_fraction=float(available[qi, pi]), negative_subject_fraction=float(available[qi, ni]),
                            raw_positive_score=float(raw[qi, pi]), raw_negative_score=float(raw[qi, ni]),
                            raw_gap=float(raw[qi, pi]-raw[qi, ni]), scored_positive=float(scores[qi, pi]),
                            scored_negative=float(scores[qi, ni]), scored_gap=float(scores[qi, pi]-scores[qi, ni])))
            print('Evaluated', key, metrics(predictions)['r1'], flush=True)
    atomic_npz(array_path, **score_arrays)
    write(folder / 'predictions.json', allpred)
    csvout(folder / 'metrics.csv', alltables)
    csvout(folder / 'per_photo_rank_changes.csv', changes)
    csvout(folder / 'fixed_candidate_comparisons.csv', pairrows)
    csvout(folder / 'paired_group_bootstrap.csv', bootstraps)
    strata = []
    for config in ['B03', 'F03']:
        for method in methods:
            for layer in ['both_align', 'positive_only', 'negative_only', 'neither_align']:
                subset = [r for r in pairrows if r['configuration'] == config and r['method'] == method and r['alignment_stratum'] == layer]
                strata.append(dict(configuration=config, method=method, alignment_stratum=layer,
                    pair_denominator=len(subset), distinct_queries=len({r['photo_id'] for r in subset}),
                    positive_over_negative_count=sum(r['scored_gap'] > 0 for r in subset),
                    exact_tie_count=sum(r['scored_gap'] == 0 for r in subset),
                    positive_over_negative_fraction=float(np.mean([r['scored_gap'] > 0 for r in subset])) if subset else None,
                    mean_scored_gap=float(np.mean([r['scored_gap'] for r in subset])) if subset else None))
    csvout(folder / 'fixed_pair_alignment_strata.csv', strata)
    historical_checks = []
    for config in ['B03', 'F03']:
        bymethod = {method: {r['photo_id']: r for r in allpred[config + '__' + method]} for method in methods}
        for qid in FOCAL_IDS:
            assert qid in qindex
            historical_checks.append(dict(configuration=config, photo_id=qid, split=d['splits'][qid],
                historical_lambda=selection['historical_diagonal_selections'][config]['lambda_value'],
                ranks={method: bymethod[method][qid]['rank'] for method in methods},
                top1={method: bymethod[method][qid]['top1'] for method in methods},
                historical_combined_repair=(not bymethod['baseline'][qid]['correct'] and bymethod['historical_diagonal'][qid]['correct']),
                historical_availability_repair=(not bymethod['baseline'][qid]['correct'] and bymethod['historical_availability_only'][qid]['correct']),
                historical_spatial_repair=(not bymethod['baseline'][qid]['correct'] and bymethod['historical_spatial_only'][qid]['correct']),
                historical_spatial_increment_required=(bymethod['historical_diagonal'][qid]['correct'] and not bymethod['historical_availability_only'][qid]['correct']),
                historical_spatial_alone_repairs=(not bymethod['baseline'][qid]['correct'] and bymethod['historical_spatial_only'][qid]['correct']),
                historical_coordinate_sensitive=(bymethod['historical_diagonal'][qid]['correct'] and not bymethod['historical_shuffled_diagonal'][qid]['correct']),
                interpretation='spatial increment required inside OCR-conditioned scoring is narrower than repair by spatial term alone; controlled scoring response, not encoder causality'))
    write(folder / 'historical_and_focal_case_checks.json', historical_checks)
    write(folder / 'baseline_joint_outcomes.json', dict(counts=dict(Counter(joint.values())), denominator=len(joint)))
    return allpred, alltables, historical_checks


def verify(folder, d, base, penalty, available, accepted, shuffle, selection, audit, initial):
    matrices = np.load(folder / 'selected_full_gallery_scores.npz', allow_pickle=False)
    assert matrices['query_ids'].tolist() == d['qb']['ids'] and matrices['reference_ids'].tolist() == d['rb']['ids']
    assert matrices['reference_sha256'].tolist() == d['rb']['sha256']
    assert np.array_equal(matrices['subject_fraction'], available)
    assert np.array_equal(matrices['accepted_alignment'], accepted)
    assert len(audit) == int(accepted.sum()) and len({(r['photo_id'], r['reference_id']) for r in audit}) == len(audit)
    assert np.all(shuffle[:, ~accepted] == base[~accepted])
    predictions = read(folder / 'predictions.json')
    metrics_rows = list(__import__('csv').DictReader((folder / 'metrics.csv').open()))
    maximum = 0.
    ranked = 0
    for key, saved in predictions.items():
        scores = matrices[key + '__scored']
        assert scores.shape == (1117, 5405) and np.isfinite(scores).all()
        observed = describe_ranks(scores, d)
        assert saved == observed
        ranked += len(saved)
        config, method = key.split('__')
        if method in ['availability_only', 'spatial_only', 'combined', 'shuffled_spatial_only', 'shuffled_combined']:
            real = method.removeprefix('shuffled_')
            parameter = selection['selections'][config][real]
            source = shuffle if method.startswith('shuffled_') else penalty
            expected = source[parameter['lambda_index']] + parameter['gamma'] * available
            error = float(np.max(np.abs(matrices[key + '__raw'] - expected)))
            maximum = max(maximum, error)
            assert error < 2e-6
        baseline = {r['photo_id']: r for r in predictions[config + '__baseline']}
        for split in ['train', 'val', 'test', 'all']:
            subset = [r for r in observed if split == 'all' or r['split'] == split]
            table = next(row for row in metrics_rows if row['configuration'] == config and row['method'] == method
                         and row['split'] == split and not row.get('stratum'))
            for name, value in metrics(subset).items(): assert abs(float(table[name])-value) < 1e-12
            repaired = sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in subset)
            harmed = sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in subset)
            assert repaired == int(table['repaired']) and harmed == int(table['harmed']) and repaired-harmed == int(table['net'])
    protocol = read(folder / 'protocol.json')
    final_hashes = {path: sha(ROOT / path) for path in protocol['inputs']}
    changed = [path for path, digest in final_hashes.items() if digest != protocol['inputs'][path]['sha256']]
    assert not changed
    artifacts = {str(p.relative_to(folder)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(folder.iterdir()) if p.is_file() and p.name not in ['validation.json', 'artifacts_sha256.json', 'progress.json']}
    write(folder / 'artifacts_sha256.json', artifacts)
    report = dict(status='passed', validated_query_ids=len(d['qb']['ids']), validated_reference_ids=len(d['rb']['ids']),
        deduplicated_reference_hashes=len(set(d['rb']['sha256'])), accepted_pairs=int(accepted.sum()),
        queries_with_alignment=int(accepted.any(1).sum()), shuffled_pair_audit_records=len(audit),
        nonidentity_coordinate_permutations=sum(r['nonidentity_permutation'] for r in audit),
        accepted_alignment_and_subject_fraction_unchanged=bool(np.array_equal(matrices['accepted_alignment'], accepted)
            and np.array_equal(matrices['subject_fraction'], available)),
        methods_replayed=len(predictions), query_rank_records_replayed=ranked,
        selected_score_formula_max_error=maximum,
        shuffle_original_MaxSim_max_error=max(r['original_score_replay_error'] for r in audit),
        real_validation_packages=len(selection['validation_package_ids']), shuffle_uses_real_selected_parameters=True,
        input_files_hash_checked=len(final_hashes), changed_input_files=changed,
        split_verification=d['split_verification'], **initial)
    write(folder / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/colnomic_full_audit/20261013_resume/split_geometry')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    folder = args.output.resolve()
    d, cache, base, penalty, lambdas, fits, fitrows, accepted, available, has_text, protocol, initial = load_inputs(folder)
    print('Inputs verified:', len(d['qb']['ids']), len(d['rb']['ids']), len(fits), flush=True)
    if args.verify_only:
        selection = read(folder / 'selection.json')
        shuffle = np.load(folder / 'shuffled_geo_penalty.npy', mmap_mode='r', allow_pickle=False)
        import csv
        audit = list(csv.DictReader((folder / 'coordinate_permutations.csv').open()))
        for row in audit:
            row['nonidentity_permutation'] = row['nonidentity_permutation'] == 'True'
            row['original_score_replay_error'] = float(row['original_score_replay_error'])
        report = verify(folder, d, base, penalty, available, accepted, shuffle, selection, audit, initial)
        print('Independent verification passed', report['query_rank_records_replayed'], flush=True)
        return
    selection = select_parameters(d, base, penalty, lambdas, available, has_text, folder)
    print('Validation parameters fixed:', selection['selections'], flush=True)
    shuffle, maximum, audit = build_shuffle(d, base, lambdas, fitrows, available, folder, protocol)
    evaluate(d, cache, base, penalty, lambdas, available, accepted, shuffle, selection, has_text, folder)
    report = verify(folder, d, base, penalty, available, accepted, shuffle, selection, audit, initial)
    write(folder / 'progress.json', dict(stage='complete', queries_completed=1117, total_queries=1117,
        accepted_pairs_completed=len(fits), validated=True, validation_file='validation.json'))
    print('Complete:', report, flush=True)


if __name__ == '__main__':
    main()
