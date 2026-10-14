from evaluation.paths import project_path
from collections import defaultdict

import numpy as np
import torch

from .common import WORK, read, save

METRICS = ('R@1', 'R@5', 'R@10', 'MRR')


def full_rank(scores, gallery, positives):
    """Rank the best positive, preserving the gallery's stable tie order."""
    scores = np.asarray(scores)
    gallery = np.asarray(gallery)
    values = scores[gallery]
    present = np.flatnonzero(np.isin(gallery, list(positives)))
    if not len(present):
        return None
    winner = present[np.argmax(values[present])]
    return int(1 + (values > values[winner]).sum() +
               ((values == values[winner]) & (np.arange(len(values)) < winner)).sum())


def deduplicated_rank(scores, canonical, positives):
    scores = np.asarray(scores)
    canonical = np.asarray(canonical)
    values = np.full(len(scores),-np.inf)
    np.maximum.at(values,canonical,scores)
    earliest = np.full(len(scores),len(scores))
    eligible = scores == values[canonical]
    np.minimum.at(earliest,canonical[eligible],np.flatnonzero(eligible))
    winner = min(positives,key=lambda p:(-values[p],earliest[p]))
    return int(1+(values>values[winner]).sum()+
               ((values==values[winner]) & (earliest<earliest[winner])).sum())


def restored_rank(raw_rank, candidates, positives, permutation):
    assert sorted(permutation) == list(range(len(candidates)))
    found = [i + 1 for i, p in enumerate(permutation) if candidates[p] in positives]
    return min(found) if found else raw_rank


def baseline_ranks(manifest):
    path = WORK / manifest['dataset'] / 'full_ranks.json'
    if path.exists():
        cached = read(path)
        assert cached['signature'] == manifest['signature']
        if 'positive_ranks' in cached:
            return cached['ranks']
    source = str(project_path(manifest['source']['score_bank']))
    if source.endswith('.npy'):
        scores = np.load(source, mmap_mode='r')
    else:
        scores = torch.load(source, weights_only=True, mmap=True)
        scores = scores['scores'] if isinstance(scores, dict) else scores
        scores = scores.numpy()
    labels = read(WORK / manifest['dataset'] / 'labels.json')
    gallery = manifest['gallery_indices']
    ranks, positive_ranks = {}, {}
    for row in manifest['rows']:
        if row['split'] == 'train':
            continue
        values = scores[row['query_index']]
        assert np.isfinite(values).all()
        order = np.asarray(gallery)[np.argsort(-values[gallery], kind='stable')]
        assert order[:len(row['candidates'])].tolist() == row['candidates']
        inverse = np.empty(len(values),dtype=np.int64)
        inverse[order] = np.arange(1,len(order)+1)
        positives = {str(p):int(inverse[p]) for p in labels[row['id']]}
        positive_ranks[row['id']] = positives
        ranks[row['id']] = min(positives.values())
    save(path, dict(signature=manifest['signature'], ranks=ranks, positive_ranks=positive_ranks,
                    scope='complete fixed gallery; unchanged tail beyond the reranking prefix'))
    return ranks


def values(rows, key='rank'):
    ranks = np.array([r[key] if r[key] is not None else np.inf for r in rows])
    return np.column_stack([ranks <= 1, ranks <= 5, ranks <= 10, 1 / ranks]).astype(float)


def bootstrap(rows, repetitions=10000, seed=20260925):
    delta = values(rows) - values(rows, 'raw_rank')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['group']].append(i)
    rng = np.random.default_rng(seed)
    totals = np.array([delta[ids].sum(0) for ids in groups.values()])
    counts = np.array([len(ids) for ids in groups.values()])
    samples = []
    for start in range(0, repetitions, 100):
        weights = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups),
                                  size=min(100, repetitions-start))
        samples.append(weights @ totals / (weights @ counts)[:, None])
    bounds = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    return dict(unit='declared dataset group', groups=len(groups), repetitions=repetitions,
                seed=seed, direction='method minus frozen ColNomic 7B',
                micro={name: dict(low=float(bounds[0, i]), high=float(bounds[1, i]))
                       for i, name in enumerate(METRICS)})


def summarize(rows, intervals=True):
    assert rows and len({r['id'] for r in rows}) == len(rows)
    data = values(rows)
    classes = defaultdict(list)
    for i, row in enumerate(rows):
        classes[row['identity']].append(i)
    result = dict(n=len(rows), identities=len(classes), correct=int(data[:, 0].sum()),
                  micro=dict(zip(METRICS, data.mean(0).tolist())),
                  macro=dict(zip(METRICS, np.mean([data[ix].mean(0) for ix in classes.values()], 0).tolist())),
                  rescued=sum(r['rank'] == 1 and r['raw_rank'] != 1 for r in rows),
                  harmed=sum(r['raw_rank'] == 1 and r['rank'] != 1 for r in rows),
                  candidate_recall=sum(r.get('target_in_candidates', False) for r in rows) / len(rows),
                  mrr_scope='complete fixed gallery, including the unchanged tail',
                  raw=dict(zip(METRICS, values(rows, 'raw_rank').mean(0).tolist())))
    if intervals:
        result['paired_intervals'] = bootstrap(rows)
    if all('positive_ranks' in r for r in rows):
        aps = [average_precisions(r['positive_ranks']) for r in rows]
        result['official_ap_denominator'] = 'max(number_of_all_groundtruths, precision_sum, 1)'
        for name in aps[0]:
            result['micro'][name] = float(np.mean([a[name] for a in aps]))
            result['macro'][name] = float(np.mean([np.mean([aps[i][name] for i in ix]) for ix in classes.values()]))
    return result


def paired_comparison(method, reference, reference_name):
    by_id = {r['id']: r for r in reference}
    assert len(by_id) == len(reference) == len(method)
    assert set(by_id) == {r['id'] for r in method}
    paired = []
    for row in method:
        other = by_id[row['id']]
        assert (row['group'], row['identity']) == (other['group'], other['identity'])
        paired.append(dict(row, raw_rank=other['rank']))
    result = summarize(paired)
    result['paired_intervals']['direction'] = 'method minus ' + reference_name
    result['reference'] = reference_name
    result['delta'] = {name: result['micro'][name] - result['raw'][name] for name in METRICS}
    return result


def average_precisions(positive_ranks):
    hit_ranks = np.sort(list(positive_ranks.values()))
    precision = np.arange(1,len(hit_ranks)+1) / hit_ranks
    result = {}
    for k in (1,5,10,100,None):
        numerator = float(precision.sum() if k is None else precision[hit_ranks<=k].sum())
        result['mAP' if k is None else f'mAP@{k}'] = numerator / max(len(hit_ranks),numerator,1.)
    return result
