"""Append-only CPU audit of frozen nonlinear Token-set readouts and full-gallery heads.

The RBF kernel mean is permutation invariant but preserves distribution information
beyond first/second moments. It is a limited nonlinear summary, not an exhaustive
test of all information in a Token set. No coordinates, Token indices, IDs, or
identity labels enter any feature vector. IDs only join artifacts/splits.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT, data, read, write, rows, csvout, sha
from scripts.colnomic_mechanism.probes import (
    logistic_fit, logistic_predict, classification_metrics, paired_metrics,
    validation_threshold, shuffled_labels, save_head,
)

SEED = 20261014
SHUFFLES = [17, 31, 53]
KINDS = ['maxsim_only', 'pooled_only', 'pooled_plus_maxsim', 'kernel_set_only']


def log(message):
    print(message, flush=True)


def head(path):
    stored = torch.load(path, map_location='cpu', weights_only=True)
    return {k: np.asarray(stored[k]) for k in ['mean', 'scale', 'coefficient']}, stored


def visual(z, spec):
    layout = spec['layout']
    result = z[layout['start']:layout['start'] + layout['count']].float().numpy()
    assert result.ndim == 2 and result.shape[1] == 128 and np.isfinite(result).all()
    return result


def moments(z, spec):
    value = visual(z, spec)
    return np.r_[value.mean(0), value.std(0)].astype(np.float32)


def kernel_fit(named_arrays, folder, seed=SEED, dimensions=64):
    """Unsupervised landmarks, equal maximum sampling per training image."""
    rng = np.random.default_rng(seed)
    samples, provenance = [], []
    for name, array in named_arrays:
        choices = np.sort(rng.choice(len(array), min(16, len(array)), replace=False))
        samples.append(array[choices])
        provenance.extend(dict(source_id=name, visual_offset=int(i)) for i in choices)
    sample = np.concatenate(samples).astype(np.float32)
    assert len(sample) >= dimensions
    selected = rng.choice(len(sample), dimensions, replace=False)
    landmarks = sample[selected]
    a = sample[rng.integers(len(sample), size=2048)]
    b = sample[rng.integers(len(sample), size=2048)]
    sigma2 = float(np.median(np.square(a - b).sum(1)))
    assert sigma2 > 0 and np.isfinite(sigma2)
    kernel = dict(landmarks=landmarks, sigma2=np.array(sigma2, dtype=np.float32))
    np.savez_compressed(folder / 'kernel.npz', **kernel)
    csvout(folder / 'kernel_training_sample_index.csv', provenance)
    csvout(folder / 'kernel_landmark_index.csv', [dict(landmark=i, **provenance[int(j)]) for i, j in enumerate(selected)])
    write(folder / 'kernel_protocol.json', dict(seed=seed, dimensions=dimensions,
          training_images=len(named_arrays), sampled_training_tokens=len(sample), sigma_squared=sigma2,
          landmark_selection='unlabelled training images only; up to 16 randomly sampled visual tokens per image',
          features='mean(exp(-squared_raw_feature_distance/(2*sigma_squared))) over ALL visual tokens',
          input_coordinates=False, input_token_indices=False, input_ids=False, input_identity_labels=False,
          limitation='permutation-invariant nonlinear distribution summary, not a complete Token-set information test'))
    return kernel


def kernel_summary(array, kernel):
    landmarks = kernel['landmarks']
    total = np.zeros(len(landmarks), dtype=np.float64)
    ln = np.square(landmarks).sum(1)
    for start in range(0, len(array), 1024):
        chunk = array[start:start + 1024]
        distance = np.maximum(np.square(chunk).sum(1)[:, None] + ln[None] - 2 * (chunk @ landmarks.T), 0)
        total += np.exp(-distance / (2 * float(kernel['sigma2']))).sum(0, dtype=np.float64)
    result = (total / len(array)).astype(np.float32)
    assert np.isfinite(result).all()
    return result


def pair_features(q, r):
    return np.concatenate([np.abs(q - r), q * r], axis=-1).astype(np.float32)


def weights_for(packages):
    count = Counter(packages)
    return np.asarray([1. / count[p] for p in packages])


def bootstrap(values, packages, seed=SEED, rounds=2000):
    packages = np.asarray(packages)
    names = sorted(set(packages))
    macro = np.asarray([np.mean(np.asarray(values)[packages == p]) for p in names])
    rng = np.random.default_rng(seed)
    sampled = macro[rng.integers(len(macro), size=(rounds, len(macro)))].mean(1)
    return dict(identity_macro_mean=float(macro.mean()), ci95_low=float(np.quantile(sampled, .025)),
                ci95_high=float(np.quantile(sampled, .975)), independent_identities=len(names), bootstrap_rounds=rounds)


def metrics_for_samples(y, logits, qids, packages, splits, threshold):
    result = []
    for part in ['train', 'val', 'test']:
        keep = splits == part
        pm, _ = paired_metrics(y[keep], logits[keep], qids[keep], packages[keep])
        for strategy, cutoff in [('fixed_zero', 0.), ('validation_calibrated', threshold)]:
            result.append(dict(split=part, threshold_strategy=strategy,
                          **{**classification_metrics(y[keep], logits[keep], packages[keep], cutoff), **pm}))
    return result


def train_selected(x, y, qids, packages, splits, folder, kind):
    train, val = splits == 'train', splits == 'val'
    choices, selection = [], []
    weight = weights_for(packages)
    for C in [.01, .1, 1., 10.]:
        model = logistic_fit(x[train], y[train], C, weight[train])
        logits = logistic_predict(model, x[val])
        pm, _ = paired_metrics(y[val], logits, qids[val], packages[val])
        auc = classification_metrics(y[val], logits, packages[val])['auc']
        selection.append(dict(C=C, **pm, auc=auc))
        choices.append((pm['package_macro_pair_accuracy'], auc, -C, model, C))
    _, _, _, model, C = max(choices, key=lambda r: r[:3])
    threshold = validation_threshold(y[val], logistic_predict(model, x[val]), packages[val])
    save_head(folder / (kind + '_head.pt'), model, dict(kind=kind, C=C, threshold=threshold))
    write(folder / (kind + '_selection.json'), dict(C=C, threshold=threshold, candidates=selection,
          selection='validation identity-macro within-query/source pair accuracy, then AUC, then smaller C',
          threshold_selection='validation identity-macro classification accuracy',
          standardization_fitted_on_training_only=True, train_samples=int(train.sum()), val_samples=int(val.sum())))
    return model, C, threshold


def natural_descriptors(d, ledger, folder):
    path = folder / 'descriptors.npz'
    if path.exists():
        record = np.load(path)
        assert list(record['query_ids']) == d['qb']['ids'] and list(record['reference_ids']) == d['rb']['ids']
        return {k: record[k] for k in record.files}
    ri = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    training_refs = sorted({ri[rid] for qid, row in ledger.items() if d['splits'][qid] == 'train' for rid in row['positive_reference_ids']})
    named = [(qid, visual(z, spec)) for qid, z, spec in zip(d['qb']['ids'], d['qb']['embeddings'], d['metadata']['photos']) if d['splits'][qid] == 'train']
    named += [(d['rb']['ids'][i], visual(d['rb']['embeddings'][i], d['metadata']['references'][i])) for i in training_refs]
    kernel = kernel_fit(named, folder)
    del named
    values = {}
    for key, bank, meta in [('query', d['qb'], d['metadata']['photos']), ('reference', d['rb'], d['metadata']['references'])]:
        sets, pools = [], []
        for i, (z, spec) in enumerate(zip(bank['embeddings'], meta)):
            assert spec['id'] == bank['ids'][i]
            array = visual(z, spec)
            sets.append(kernel_summary(array, kernel))
            pools.append(np.r_[array.mean(0), array.std(0)].astype(np.float32))
            if (i + 1) % 500 == 0:
                log(f'{key} descriptors {i + 1}/{len(meta)}')
        values[key + '_set'] = np.asarray(sets)
        values[key + '_moments'] = np.asarray(pools)
        values[key + '_ids'] = np.asarray(bank['ids'])
    np.savez_compressed(path, **values)
    write(folder / 'descriptor_provenance.json', dict(query_bank_sha256=sha(ROOT / 'cache/image_embeddings/colnomic7b/photos.pt'),
          reference_bank_sha256=sha(ROOT / 'cache/image_embeddings/colnomic7b/references.pt'),
          training_landmark_query_ids=[qid for qid in d['qb']['ids'] if d['splits'][qid] == 'train'],
          training_landmark_reference_ids=[d['rb']['ids'][i] for i in training_refs]))
    return values


def restricted(d, descriptors, prior, folder):
    inputs = np.load(prior / 'identity_probe_inputs.npz')
    index = rows(prior / 'identity_sample_index.csv')
    qi = {qid: i for i, qid in enumerate(d['qb']['ids'])}
    ri = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    qindices = np.array([qi[r['photo_id']] for r in index])
    rindices = np.array([ri[r['reference_id']] for r in index])
    assert np.array_equal(inputs['query_ids'], [r['photo_id'] for r in index])
    y, qids, packages, splits = [inputs[k] for k in ['labels', 'query_ids', 'package_ids', 'splits']]
    set_x = pair_features(descriptors['query_set'][qindices], descriptors['reference_set'][rindices])
    candidates = dict(maxsim_only=inputs['maxsim'], pooled_only=inputs['pooled'],
                      pooled_plus_maxsim=np.column_stack([inputs['pooled'], inputs['maxsim']]), kernel_set_only=set_x)
    np.savez_compressed(folder / 'restricted_inputs.npz', kernel_set_only=set_x, labels=y, query_ids=qids,
                        reference_ids=np.array([r['reference_id'] for r in index]), package_ids=packages, splits=splits)
    models, predictions, metrics, pairrows, nullpredictions, replay = {}, [], [], [], [], []
    for kind, x in candidates.items():
        if kind == 'kernel_set_only':
            model, C, threshold = train_selected(x, y, qids, packages, splits, folder, kind)
        else:
            model, stored = head(prior / (kind + '_identity_head.pt'))
            C, threshold = float(stored['C']), float(stored['threshold'])
            save_head(folder / (kind + '_head.pt'), model, dict(kind=kind, C=C, threshold=threshold, original_head=str(prior / (kind + '_identity_head.pt'))))
        logits = logistic_predict(model, x)
        models[kind] = (model, C, threshold)
        metrics.extend(dict(probe=kind, scope='same_original_restricted_pairs', **r) for r in metrics_for_samples(y, logits, qids, packages, splits, threshold))
        _, pairs = paired_metrics(y, logits, qids, packages)
        pairrows.extend(dict(probe=kind, split=d['splits'][r['query_id']], **r) for r in pairs)
        predictions.extend(dict(probe=kind, **r, logit=float(s), threshold=threshold) for r, s in zip(index, logits))
        if kind != 'kernel_set_only':
            old = [r for r in rows(prior / 'identity_predictions.csv') if r['probe'] == kind]
            error = float(np.max(np.abs(logits - np.array([float(r['logit']) for r in old]))))
            assert error < 1e-10
            replay.append(dict(probe=kind, maximum_saved_prediction_error=error, sample_count=len(old)))
        train, val = splits == 'train', splits == 'val'
        for seed in SHUFFLES:
            shuffled = shuffled_labels(y[train], packages[train], seed)
            null = logistic_fit(x[train], shuffled, C, weights_for(packages)[train])
            cutoff = validation_threshold(y[val], logistic_predict(null, x[val]), packages[val])
            save_head(folder / (kind + f'_shuffle_{seed}_head.pt'), null, dict(kind=kind, seed=seed, C=C, threshold=cutoff))
            null_logits = logistic_predict(null, x)
            metrics.extend(dict(probe=kind, scope='same_original_restricted_pairs', control='label_shuffle', seed=seed, **r)
                           for r in metrics_for_samples(y, null_logits, qids, packages, splits, cutoff))
            traininglabels = np.full(len(y), -1); traininglabels[train] = shuffled
            nullpredictions.extend(dict(probe=kind, seed=seed, **r, shuffled_training_label=int(sy) if part == 'train' else None,
                                    logit=float(s), threshold=cutoff)
                                   for r, s, sy, part in zip(index, null_logits, traininglabels, splits))
        log('Restricted head complete ' + kind)
    csvout(folder / 'restricted_predictions.csv', predictions)
    csvout(folder / 'restricted_metrics.csv', metrics)
    csvout(folder / 'restricted_pair_gaps.csv', pairrows)
    csvout(folder / 'shuffle_predictions.csv', nullpredictions)
    write(folder / 'existing_identity_head_replay.json', replay)
    return models


def pair_moment_features(q, r, scores):
    return np.column_stack([np.abs(q[:, :128] - r[:, :128]), q[:, :128] * r[:, :128],
                            np.abs(q[:, 128:] - r[:, 128:]), q[:, 128:] * r[:, 128:], scores]).astype(np.float32)


def natural_hard_pairs(d, ledger, descriptors, models, folder):
    qi = {qid: i for i, qid in enumerate(d['qb']['ids'])}
    ri = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    hashindices = {}
    for i, h in enumerate(d['rb']['sha256']):
        hashindices.setdefault(h, []).append(i)
    sample_index, pairs = [], []
    for qid in d['qb']['ids']:
        row = ledger[qid]
        positives = sorted({i for h in row['positive_reference_sha256'] for i in hashindices[h]})
        negatives = list(dict.fromkeys([ri[row['image_strongest_wrong']], ri[row['fusion_strongest_wrong']]]))
        assert not set(positives) & set(negatives)
        samplepositions = {}
        for label, indices in [(1, positives), (0, negatives)]:
            for i in indices:
                samplepositions[i] = len(sample_index)
                sample_index.append(dict(photo_id=qid, reference_id=d['rb']['ids'][i], package_id=row['package_id'],
                                         split=d['splits'][qid], label=label, query_index=qi[qid], reference_index=i))
        for p in positives:
            for n in negatives:
                pairs.append(dict(photo_id=qid, package_id=row['package_id'], split=d['splits'][qid],
                     positive_reference_id=d['rb']['ids'][p], negative_reference_id=d['rb']['ids'][n],
                     negative_role=('image' if n == ri[row['image_strongest_wrong']] else '') +
                         ('+fusion' if n == ri[row['fusion_strongest_wrong']] else ''),
                     positive_index=samplepositions[p], negative_index=samplepositions[n]))
    qis = np.array([r['query_index'] for r in sample_index]); ris = np.array([r['reference_index'] for r in sample_index])
    scalars = d['image']['scores'][qis, ris].numpy().astype(np.float32)
    pool = pair_moment_features(descriptors['query_moments'][qis], descriptors['reference_moments'][ris], scalars)
    candidates = dict(maxsim_only=pool[:, -1:], pooled_only=pool[:, :-1], pooled_plus_maxsim=pool,
                      kernel_set_only=pair_features(descriptors['query_set'][qis], descriptors['reference_set'][ris]))
    outputs, metrics, predictions = [], [], []
    baseline_correct = np.array([float(scalars[r['positive_index']] > scalars[r['negative_index']]) +
                         .5 * float(scalars[r['positive_index']] == scalars[r['negative_index']]) for r in pairs])
    for kind, x in candidates.items():
        model, _, threshold = models[kind]
        logits = logistic_predict(model, x)
        predictions.extend(dict(probe=kind, **row, logit=float(s), threshold=threshold) for row, s in zip(sample_index, logits))
        result = []
        for row in pairs:
            gap = float(logits[row['positive_index']] - logits[row['negative_index']])
            result.append(dict(probe=kind, **row, positive_logit=float(logits[row['positive_index']]),
                               negative_logit=float(logits[row['negative_index']]), gap=gap,
                               correct=1. if gap > 0 else .5 if gap == 0 else 0.))
        outputs.extend(result)
        for part in ['all', 'train', 'val', 'test']:
            keep = np.array([part == 'all' or r['split'] == part for r in result])
            samplekeep = np.array([part == 'all' or r['split'] == part for r in sample_index])
            correct = np.array([r['correct'] for r in result])[keep]
            gap = np.array([r['gap'] for r in result])[keep]
            packages = np.array([r['package_id'] for r in result])[keep]
            queryids = np.array([r['photo_id'] for r in result])[keep]
            metrics.append(dict(probe=kind, split=part, scope='ALL1117 natural strongest image/fusion negatives; ALL SHA-correct aliases',
                pair_accuracy=float(correct.mean()), query_macro_pair_accuracy=float(np.mean([correct[queryids == q].mean() for q in set(queryids)])),
                **bootstrap(correct, packages), gap_mean=float(gap.mean()), gap_median=float(np.median(gap)),
                gap_p05=float(np.quantile(gap, .05)), gap_p95=float(np.quantile(gap, .95)),
                queries=len(set(queryids)), pairs=int(keep.sum()),
                paired_change_vs_raw_maxsim=bootstrap(correct - baseline_correct[keep], packages),
                threshold_classification=classification_metrics(np.array([r['label'] for r in sample_index])[samplekeep],
                     logits[samplekeep], np.array([r['package_id'] for r in sample_index])[samplekeep], threshold)))
    csvout(folder / 'all1117_hard_pair_predictions.csv', predictions)
    csvout(folder / 'all1117_hard_pair_gaps.csv', outputs)
    write(folder / 'all1117_hard_pair_metrics.json', metrics)


def affine(model):
    weights = model['coefficient'][:-1] / model['scale']
    bias = float(model['coefficient'][-1] - model['mean'] @ weights)
    return weights, bias


def gallery_score(kind, model, qpool, rpool, qset, rset, scalar):
    weights, bias = affine(model)
    if kind == 'maxsim_only':
        return scalar.astype(np.float64) * weights[0] + bias
    if kind == 'kernel_set_only':
        dim = len(qset)
        return np.abs(rset - qset) @ weights[:dim] + (rset * qset) @ weights[dim:] + bias
    wabs = np.r_[weights[:128], weights[256:384]]
    wprod = np.r_[weights[128:256], weights[384:512]]
    scores = np.abs(rpool - qpool) @ wabs + (rpool * qpool) @ wprod + bias
    if kind == 'pooled_plus_maxsim':
        scores += scalar.astype(np.float64) * weights[-1]
    return scores


def retrieval_rows(d, ledger, scores, folder, save_ranks=True):
    nq, nr = scores.shape
    assert (nq, nr) == (1117, 5405)
    hashes = np.asarray(d['rb']['sha256'])
    uniquenames, hashcodes = np.unique(hashes, return_inverse=True)
    rankings = np.lib.format.open_memmap(folder / 'stable_sorted_reference_indices.npy', mode='w+', dtype=np.int32, shape=(nq, nr)) if save_ranks else None
    designranks = np.lib.format.open_memmap(folder / 'sha_design_ranks.npy', mode='w+', dtype=np.int32, shape=(nq, nr)) if save_ranks else None
    result = []
    for i, qid in enumerate(d['qb']['ids']):
        order = np.argsort(-scores[i], kind='stable')
        if rankings is not None:
            rankings[i] = order
        first = np.full(len(uniquenames), nr, dtype=np.int32)
        np.minimum.at(first, hashcodes[order], np.arange(nr))
        hashorder = np.argsort(first, kind='stable')
        hash_rank = np.empty(len(uniquenames), dtype=np.int32)
        hash_rank[hashorder] = np.arange(1, len(uniquenames) + 1)
        ranks = hash_rank[hashcodes]
        if designranks is not None:
            designranks[i] = ranks
        positive = set(ledger[qid]['positive_reference_sha256'])
        positive_indices = np.flatnonzero(np.isin(hashes, list(positive)))
        rank = int(ranks[positive_indices].min())
        top = int(order[0])
        result.append(dict(photo_id=qid, package_id=ledger[qid]['package_id'], group_id=ledger[qid]['group_id'], split=d['splits'][qid],
              positive_rank=rank, top_reference_id=d['rb']['ids'][top], top_reference_sha256=hashes[top], top_score=float(scores[i, top]),
              best_positive_score=float(scores[i, positive_indices].max()), correct=rank == 1))
    if rankings is not None:
        rankings.flush(); designranks.flush()
    csvout(folder / 'query_rankings.csv', result)
    return result


def retrieval_metrics(result, baseline):
    metrics = []
    for part in ['all', 'train', 'val', 'test']:
        keep = np.array([part == 'all' or r['split'] == part for r in result])
        rank = np.array([r['positive_rank'] for r in result])[keep]
        base = np.array([r['positive_rank'] for r in baseline])[keep]
        packages = np.array([r['package_id'] for r in result])[keep]
        metrics.append(dict(split=part, queries=len(rank), identities=len(set(packages)), recall_at_1=float((rank <= 1).mean()),
           recall_at_5=float((rank <= 5).mean()), recall_at_10=float((rank <= 10).mean()), mrr=float((1. / rank).mean()),
           identity_macro_recall_at_1=float(np.mean([(rank[packages == p] <= 1).mean() for p in set(packages)])),
           identity_macro_mrr=float(np.mean([(1. / rank[packages == p]).mean() for p in set(packages)])),
           repaired=int(((base > 1) & (rank == 1)).sum()), harmed=int(((base == 1) & (rank > 1)).sum()),
           net_change=int((rank == 1).sum() - (base == 1).sum()),
           paired_identity_bootstrap_recall_change=bootstrap((rank == 1).astype(float) - (base == 1).astype(float), packages)))
    return metrics


def full_gallery(d, ledger, descriptors, models, folder):
    gallery = folder / 'full_gallery'
    gallery.mkdir(exist_ok=True)
    baselinefolder = gallery / 'raw_maxsim'; baselinefolder.mkdir(exist_ok=True)
    baseline = retrieval_rows(d, ledger, d['image']['scores'].numpy(), baselinefolder)
    metrics = dict(raw_maxsim=retrieval_metrics(baseline, baseline))
    replay = []
    for kind in KINDS:
        destination = gallery / kind; destination.mkdir(exist_ok=True)
        scorepath = destination / 'scores.npy'
        statepath = destination / 'progress.json'
        if statepath.exists():
            state = read(statepath); done = int(state['completed_queries'])
            scores = np.lib.format.open_memmap(scorepath, mode='r+')
        else:
            done = 0
            scores = np.lib.format.open_memmap(scorepath, mode='w+', dtype=np.float64, shape=(1117, 5405))
        model, _, _ = models[kind]
        for i in range(done, 1117):
            scalar = d['image']['scores'][i].numpy()
            scores[i] = gallery_score(kind, model, descriptors['query_moments'][i], descriptors['reference_moments'],
                                       descriptors['query_set'][i], descriptors['reference_set'], scalar)
            if i % 50 == 0 or i == 1116:
                scores.flush()
                write(statepath, dict(completed_queries=i + 1, total_queries=1117, reference_rows=5405))
                log(f'Full gallery {kind} {i + 1}/1117')
        assert np.isfinite(scores).all()
        # Independent direct-feature replay probes float32 pair products and column ordering.
        errors = []
        for i in [0, 111, 1116]:
            choice = np.array([0, 37, 1001, 5404])
            raw = d['image']['scores'][i, choice].numpy()
            pooled = pair_moment_features(np.repeat(descriptors['query_moments'][i:i+1], len(choice), axis=0), descriptors['reference_moments'][choice], raw)
            direct = raw[:, None] if kind == 'maxsim_only' else pooled[:, :-1] if kind == 'pooled_only' else pooled if kind == 'pooled_plus_maxsim' else pair_features(np.repeat(descriptors['query_set'][i:i+1], len(choice), axis=0), descriptors['reference_set'][choice])
            errors.append(float(np.max(np.abs(logistic_predict(model, direct) - scores[i, choice]))))
        assert max(errors) < 1e-7, (kind, errors)
        replay.append(dict(probe=kind, vectorization_maximum_error=max(errors), checked_pairs=12))
        result = retrieval_rows(d, ledger, scores, destination)
        metrics[kind] = retrieval_metrics(result, baseline)
        log('Full gallery ranking complete ' + kind)
    write(gallery / 'metrics.json', metrics)
    write(gallery / 'vectorization_replay.json', replay)
    write(gallery / 'score_axes.json', dict(query_ids=d['qb']['ids'], reference_ids=d['rb']['ids'], reference_sha256=d['rb']['sha256'],
          dedup='stable descending scores; first row per SHA; stable original row order resolves ties',
          unique_reference_designs=len(set(d['rb']['sha256'])), evaluated_rows=5405, queries=1117))


def layouts(layout_folder, folder):
    sources = read(layout_folder / 'sources.json')
    metadata = {r['id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['photos']}
    signed = (layout_folder / 'embeddings' / (sources[0]['photo_id'] + '_left.pt')).exists()
    targets = ['left', 'right'] if signed else ['shift_1', 'shift_2']
    editors = ['editor_left', 'editor_right'] if signed else ['editor_1', 'editor_2']
    target_description = ('within-source equal-amplitude signed_word_position classes (left=0/right=1); NOT natural identity/layout discrimination'
                          if signed else 'within-source two synthetic displacement-amplitude classes; NOT natural identity/layout discrimination')
    inputpath = layout_folder / 'analysis/probe_inputs.npz'
    a = np.load(inputpath) if inputpath.exists() and not signed else None
    named, records = [], {}
    for source in sources:
        qid = source['photo_id']
        records[qid] = {}
        for arm in targets + editors + ['sham', 'fresh_base']:
            record = torch.load(layout_folder / 'embeddings' / (qid + '_' + arm + '.pt'), map_location='cpu', weights_only=True)
            array = visual(record['embedding'], metadata[qid])
            records[qid][arm] = dict(array=array, embedding=record['embedding'], word_bbox=record.get('word_bbox'))
            if source['split'] == 'train' and arm in targets:
                named.append((qid + '_' + arm, array))
    if a is None:
        from scripts.colnomic_mechanism.analyze_controls import sim, roi_features
        baseinputs = dict(moments=[], maxsim=[], word_ROI=[], labels=[], photo_ids=[], package_ids=[], splits=[])
        for source in sources:
            qid = source['photo_id']
            for label, arm in enumerate(targets):
                record = records[qid][arm]; array = record['array']
                baseinputs['moments'].append(np.r_[array.mean(0), array.std(0)].astype(np.float32))
                baseinputs['maxsim'].append([sim(records[qid]['fresh_base']['embedding'], record['embedding'])[0]])
                baseinputs['word_ROI'].append(roi_features(record['embedding'], metadata[qid], record['word_bbox'])[0])
                for key, value in [('labels', label), ('photo_ids', qid), ('package_ids', source['package_id']), ('splits', source['split'])]:
                    baseinputs[key].append(value)
        a = {key: np.asarray(value) for key, value in baseinputs.items()}
        np.savez_compressed(folder / 'new_population_baseline_inputs.npz', **a)
    kernel = kernel_fit(named, folder, seed=SEED + 1)
    x = np.array([kernel_summary(records[qid][targets[int(label)]]['array'], kernel) for qid, label in zip(a['photo_ids'], a['labels'])])
    y, qids, packages, splits = [a[k] for k in ['labels', 'photo_ids', 'package_ids', 'splits']]
    model, C, cutoff = train_selected(x, y, qids, packages, splits, folder, 'kernel_set_only')
    logits = logistic_predict(model, x)
    np.savez_compressed(folder / 'layout_set_inputs.npz', kernel_set=x, labels=y, photo_ids=qids, package_ids=packages, splits=splits)
    metrics = [dict(probe='kernel_set_only', **r) for r in metrics_for_samples(y, logits, qids, packages, splits, cutoff)]
    predictions = [dict(probe='kernel_set_only', photo_id=qid, package_id=package, split=part, label=int(label), arm=targets[int(label)],
                         logit=float(value), threshold=cutoff) for qid, package, part, label, value in zip(qids, packages, splits, y, logits)]
    pairrows, editorrows, nullpredictions, bootstraprows = [], [], [], []
    baselineheads = dict(maxsim_only=a['maxsim'], pooled_only=a['moments'], pooled_plus_maxsim=np.column_stack([a['moments'], a['maxsim']]), word_ROI_assisted=a['word_ROI'])
    replay = []
    savedpredictions = rows(layout_folder / 'analysis/per_sample_predictions.csv') if inputpath.exists() and not signed else None
    layoutheads = {'kernel_set_only': (model, C, cutoff, x)}
    for kind, values in baselineheads.items():
        if savedpredictions is not None:
            hm, hs = head(layout_folder / 'analysis' / (kind + '_head.pt'))
            hC, ht = float(hs['C']), float(hs['threshold'])
            save_head(folder / (kind + '_head.pt'), hm, dict(kind=kind, C=hC, threshold=ht,
                       original_head=str(layout_folder / 'analysis' / (kind + '_head.pt'))))
        else:
            hm, hC, ht = train_selected(values, y, qids, packages, splits, folder, kind)
        scores = logistic_predict(hm, values)
        layoutheads[kind] = (hm, hC, ht, values)
        if savedpredictions is not None:
            old = np.array([float(r['logit']) for r in savedpredictions if r['probe'] == kind])
            error = float(np.max(np.abs(old - scores)))
            assert error < 1e-10
            replay.append(dict(probe=kind, saved_prediction_maximum_error=error, samples=len(scores)))
        metrics.extend(dict(probe=kind, **r) for r in metrics_for_samples(y, scores, qids, packages, splits, ht))
        predictions.extend(dict(probe=kind, photo_id=qid, package_id=package, split=part, label=int(label), arm=targets[int(label)],
                                logit=float(value), threshold=ht) for qid, package, part, label, value in zip(qids, packages, splits, y, scores))
    for kind, (hm, hC, ht, values) in layoutheads.items():
        scores = logistic_predict(hm, values)
        for part in ['train', 'val', 'test']:
            keep = splits == part
            _, pairs = paired_metrics(y[keep], scores[keep], qids[keep], packages[keep])
            bm, _, _, bx = layoutheads['maxsim_only']
            _, baselinepairs = paired_metrics(y[keep], logistic_predict(bm, bx)[keep], qids[keep], packages[keep])
            assert [r['query_id'] for r in pairs] == [r['query_id'] for r in baselinepairs]
            pairrows.extend(dict(probe=kind, split=part, **r) for r in pairs)
            bootstraprows.append(dict(probe=kind, split=part, quantity=target_description,
                 **bootstrap(np.array([r['correct'] for r in pairs]), np.array([r['package_id'] for r in pairs])),
                 paired_change_vs_maxsim_readout=bootstrap(np.array([r['correct'] for r in pairs]) -
                     np.array([r['correct'] for r in baselinepairs]), np.array([r['package_id'] for r in pairs]))))
    sourcebyid = {r['photo_id']: r for r in sources}
    for qid, record in records.items():
        source = sourcebyid[qid]
        from scripts.colnomic_mechanism.analyze_controls import sim, roi_features
        controlfeatures = {}
        for arm in editors + ['fresh_base', 'sham'] + targets:
            r = record[arm]; v = r['array']; pool = np.r_[v.mean(0), v.std(0)]
            scalar = np.array([sim(record['fresh_base']['embedding'], r['embedding'])[0]])
            roi_bbox = r['word_bbox'] if r['word_bbox'] is not None else record['sham']['word_bbox']
            controlfeatures[arm] = dict(kernel_set_only=kernel_summary(v, kernel), maxsim_only=scalar,
                 pooled_only=pool, pooled_plus_maxsim=np.r_[pool, scalar],
                 word_ROI_assisted=roi_features(r['embedding'], metadata[qid], roi_bbox)[0])
        for kind, (hm, hC, ht, values) in layoutheads.items():
            editor = logistic_predict(hm, np.array([controlfeatures[k][kind] for k in editors]))
            arms = logistic_predict(hm, np.array([controlfeatures[k][kind] for k in ['fresh_base', 'sham'] + targets]))
            editorrows.append(dict(probe=kind, photo_id=qid, package_id=source['package_id'], split=source['split'],
                editor1_logit=float(editor[0]), editor2_logit=float(editor[1]), editor_gap=float(editor[1] - editor[0]),
                base_logit=float(arms[0]), sham_logit=float(arms[1]), sham_minus_base=float(arms[1]-arms[0]),
                class0_minus_sham=float(arms[2]-arms[1]), class1_minus_sham=float(arms[3]-arms[1]),
                layout_gap=float(arms[3]-arms[2]), layout_gap_minus_editor_gap=float((arms[3]-arms[2])-(editor[1]-editor[0]))))
    train, val = splits == 'train', splits == 'val'
    for kind, (hm, hC, ht, values) in layoutheads.items():
        for seed in SHUFFLES:
            shuffled = shuffled_labels(y[train], packages[train], seed)
            null = logistic_fit(values[train], shuffled, hC, weights_for(packages)[train])
            threshold = validation_threshold(y[val], logistic_predict(null, values[val]), packages[val])
            save_head(folder / f'{kind}_shuffle_{seed}_head.pt', null, dict(kind=kind, seed=seed, C=hC, threshold=threshold))
            null_logits = logistic_predict(null, values)
            traininglabels = np.full(len(y), -1); traininglabels[train] = shuffled
            metrics.extend(dict(probe=kind, control='label_shuffle', seed=seed, **r)
                           for r in metrics_for_samples(y, null_logits, qids, packages, splits, threshold))
            nullpredictions.extend(dict(probe=kind, seed=seed, photo_id=qid, package_id=package, split=part, label=int(label),
                                shuffled_training_label=int(shuffledlabel) if part == 'train' else None,
                                logit=float(score), threshold=threshold)
                               for qid, package, part, label, shuffledlabel, score in zip(qids, packages, splits, y, traininglabels, null_logits))
    csvout(folder / 'layout_metrics.csv', metrics); csvout(folder / 'layout_predictions.csv', predictions)
    csvout(folder / 'layout_pair_gaps.csv', pairrows); csvout(folder / 'editing_sham_responses.csv', editorrows)
    csvout(folder / 'shuffle_predictions.csv', nullpredictions)
    write(folder / 'existing_layout_head_replay.json', replay); write(folder / 'paired_identity_bootstrap.json', bootstraprows)
    intersections = {}
    for field in ['package_id', 'group_id', 'image_sha256', 'source_sha256']:
        scope = {part: {r.get(field, r.get(field.replace('sha256', 'SHA256'))) for r in sources if r['split'] == part} for part in ['train', 'val', 'test']}
        assert all(None not in values for values in scope.values())
        intersections[field] = {a + '__' + b: sorted(scope[a] & scope[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert all(not v for check in intersections.values() for v in check.values())
    write(folder / 'scope_and_split_verification.json', dict(split_intersections=intersections,
          source_counts={part: sum(r['split'] == part for r in sources) for part in ['train', 'val', 'test']},
          target=target_description,
          no_layout_editor_controls='positions of words unchanged; editor gap has no classification correctness label',
          historical_test_exposure='known historical test identities, not a fresh blind test'))
    log('Layout kernel-set readout complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/colnomic_full_audit/20261013_resume/set_probes')
    parser.add_argument('--prior', type=Path, default=ROOT / 'results/token_mechanism_experiments/20261013_v2/stage3_probes')
    parser.add_argument('--layout-folder', type=Path, default=ROOT / 'results/token_mechanism_experiments/20261013_v2/stage4_extended_photos_v3')
    parser.add_argument('--layout-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except ImportError:
        pass
    args.output.mkdir(parents=True, exist_ok=True)
    executed_sha = sha(Path(__file__))
    snapshot = args.output / ('source_executed_' + executed_sha[:12] + '.py')
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    declared_sources = read(args.layout_folder / 'sources.json')
    declared_signed = (args.layout_folder / 'embeddings' / (declared_sources[0]['photo_id'] + '_left.pt')).exists()
    write(args.output / 'protocol.json', dict(seed=SEED, shuffle_seeds=SHUFFLES, encoder='frozen existing embeddings; CPU readouts only',
          natural_training='EXACT original 2234 restricted samples; train-only unlabelled query/gallery-positive landmarks',
          full_gallery='all 1117 x 5405 scalar scores; stable SHA dedup; no candidate truncation or truth insertion',
          feature_limitation='64-D nonlinear kernel mean preserves more than moments but remains a finite summary',
          layout_target='equal-amplitude signed word position, not natural identity layout' if declared_signed else 'synthetic displacement amplitude, not identity layout',
          executed_source_sha256=executed_sha, observed_test=True))
    if not args.layout_only:
        d = data()
        ledger = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
        assert set(ledger) == set(d['qb']['ids']) and len(ledger) == 1117
        natural = args.output / 'natural'; natural.mkdir(exist_ok=True)
        write(natural / 'split_verification.json', d['split_verification'])
        descriptors = natural_descriptors(d, ledger, natural)
        models = restricted(d, descriptors, args.prior, natural)
        natural_hard_pairs(d, ledger, descriptors, models, natural)
        full_gallery(d, ledger, descriptors, models, natural)
        del d, descriptors
    layoutout = args.output / 'layout'; layoutout.mkdir(exist_ok=True)
    layouts(args.layout_folder, layoutout)
    write(args.output / 'completion.json', dict(status='completed', natural_executed=not args.layout_only,
           layout_folder=str(args.layout_folder), source_code_sha256=executed_sha, executed_source_snapshot=str(snapshot),
           validation='asserted bank axes, split intersections, original head replay, vectorized score replay, finite full gallery scores'))


if __name__ == '__main__':
    main()
