"""Stable byte-deduplicated ranking, type summaries and source-group bootstrap."""
from collections import defaultdict
import numpy as np
import torch

METRICS = ('recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr')


def values(rows):
    ranks = np.array([r['rank'] or np.inf for r in rows])
    return np.column_stack([ranks <= 1, ranks <= 5, ranks <= 10, 1/ranks]).astype(float)


def summarize(rows):
    packages = sorted({r['package_id'] for r in rows})
    result = dict(queries=len(rows), packages=len(packages),
                  source_groups=len({r['group_id'] for r in rows}),
                  failures=sum(r['rank'] is None for r in rows))
    v = values(rows) if rows else np.empty((0,4))
    package_values = [v[[r['package_id'] == p for r in rows]].mean(0) for p in packages]
    result['photo_average'] = dict(zip(METRICS, v.mean(0).tolist() if rows else [None]*4))
    result['package_average'] = dict(zip(METRICS, np.mean(package_values,axis=0).tolist() if rows else [None]*4))
    result['correct'] = sum(r['rank'] == 1 for r in rows)
    return result


def evaluate(scores, dataset, valid=None):
    expected = (len(dataset.photos),len(dataset.references))
    if tuple(scores.shape) != expected or not torch.isfinite(scores).all():
        raise ValueError(f'Invalid scores; expected {expected}')
    if valid is None:
        valid = [True]*len(dataset.photos)
    if len(valid) != len(dataset.photos):
        raise ValueError('Missing query validity flags')
    sha_of = {r['reference_id']:r['sha256'] for r in dataset.references}
    positives = {p:{sha_of[r] for r in refs} for p,refs in dataset.positives.items()}
    order = torch.argsort(scores,dim=-1,descending=True,stable=True).tolist()
    predictions = []
    for qi,(photo,ranking) in enumerate(zip(dataset.photos,order,strict=True)):
        selected, seen, rank = [], set(), None
        if valid[qi]:
            for idx in ranking:
                ref = dataset.references[idx]
                if ref['sha256'] not in seen:
                    seen.add(ref['sha256'])
                    if rank is None and ref['sha256'] in positives[photo['package_id']]:
                        rank = len(seen)
                    if len(selected) < 10:
                        selected.append(ref['reference_id'])
        predictions.append(dict(photo_id=photo['photo_id'],package_id=photo['package_id'],
            group_id=dataset.groups[photo['package_id']],rank=rank,correct=rank == 1,
            selected_reference_id=selected[0] if selected else None,top_reference_ids=selected,
            positive_reference_ids=sorted(dataset.positives[photo['package_id']]),
            failure=None if valid[qi] else 'empty_ocr',type=dataset.labels[photo['photo_id']]))
    report = summarize(predictions)
    report.update(protocol=getattr(dataset,'full_protocol',f'Full-{len(dataset.photos)}'),reference_rows=len(dataset.references),
                  unique_reference_images=len(set(sha_of.values())),
                  tie_break='formal reference row order',deduplication='image SHA-256')
    report['confidence_intervals'] = bootstrap(predictions)
    by_type = {}
    for placement in (None,'tabletop','handheld'):
        for background in (None,'single_object','multi_object'):
            if placement is None and background is None:
                continue
            key = '/'.join(v for v in (placement,background) if v)
            by_type[key] = summarize([r for r in predictions
                if (placement is None or r['type']['placement'] == placement)
                and (background is None or r['type']['background'] == background)])
    if any('occlusion' in r['type'] for r in predictions):
        for occlusion in ('none','foreground','unclear'):
            by_type['occlusion='+occlusion] = summarize([r for r in predictions
                if r['type'].get('occlusion') == occlusion])
    return predictions,report,by_type


def bootstrap(rows, other=None, repetitions=10000, seed=20260924):
    """Resample complete source groups, retaining their photo/package weights."""
    if other is not None and [r['photo_id'] for r in rows] != [r['photo_id'] for r in other]:
        raise ValueError('Paired bootstrap requires identical ordered photos')
    groups = sorted({r['group_id'] for r in rows})
    if not groups:
        return None
    v = values(rows)
    if other is not None:
        v -= values(other)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=repetitions)
    result = dict(unit='source_group',groups=len(groups),repetitions=repetitions,seed=seed,
                  interval='percentile 95%',direction='first minus second' if other is not None else 'absolute')
    for averaging in ('photo_average','package_average'):
        totals, counts = [], []
        for group in groups:
            indices = [i for i,r in enumerate(rows) if r['group_id'] == group]
            if averaging == 'photo_average':
                gv = v[indices]
            else:
                packages = sorted({rows[i]['package_id'] for i in indices})
                gv = np.array([v[[i for i in indices if rows[i]['package_id'] == p]].mean(0) for p in packages])
            totals.append(gv.sum(0));counts.append(len(gv))
        samples = (weights @ np.array(totals))/(weights @ np.array(counts))[:,None]
        bounds = np.quantile(samples,[0.025,0.975],axis=0)
        result[averaging] = {name:dict(low=float(bounds[0,i]),high=float(bounds[1,i]))
                             for i,name in enumerate(METRICS)}
    return result


def paired_report(first,second):
    return dict(first_queries=len(first),second_queries=len(second),
        rescued=sum(a['correct'] and not b['correct'] for a,b in zip(first,second,strict=True)),
        harmed=sum(b['correct'] and not a['correct'] for a,b in zip(first,second,strict=True)),
        intervals=bootstrap(first,second))
