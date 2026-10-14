import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import RUNS, WORK, RESULTS, key, read, save, signature

KINDS = ('INTERNAL3', 'RAW_M', 'ADDITIVE4', 'PRODUCT5', 'WEIGHTED7', 'RAW_U', 'RAW_V', 'ADDITIVE_UV')


def features(raw, values, kind='INTERNAL3', content=None):
    raw, values = raw.double(), values.double()
    content = values[:, 0] if content is None else content.double()
    mass = values[:, 1]
    def delta(x):
        return (x[1:] - x[0]) / (x[1:].abs() + x[0].abs() + 1e-12)
    cols = [(raw[1:]-raw[0]) / raw.std(unbiased=False).clamp_min(1e-12)]
    if kind in ('RAW_M','RAW_U','RAW_V'):
        cols += [delta(values[:, {'RAW_M':1,'RAW_U':6,'RAW_V':7}[kind]])]
    elif kind == 'INTERNAL3':
        cols += [delta(content)]
    elif kind == 'ADDITIVE4':
        cols += [delta(mass), delta(content)]
    elif kind == 'ADDITIVE_UV':
        cols += [delta(values[:,6]), delta(values[:,7]), delta(content)]
    elif kind == 'PRODUCT5':
        cols += [delta(mass*content), delta(mass), delta(content)]
    elif kind == 'WEIGHTED7':
        cols += [delta(values[:, 3]), delta(mass), delta(values[:, 2]),
                 delta(mass*values[:, 4]), delta(mass*values[:, 5])]
    else:
        raise ValueError(kind)
    cols.append(torch.ones_like(cols[0]))
    return torch.stack(cols, -1)


def mass_columns(kind):
    return {'RAW_U':(6,), 'RAW_V':(7,), 'ADDITIVE_UV':(6,7)}.get(kind, (1,))


def intervene_mass(values, candidates, intervention, columns=(1,)):
    result = values.clone()
    if intervention == 'constant':
        for column in columns:
            result[:, column] = result[:, column].mean()
    elif intervention == 'shuffled':
        physical = torch.as_tensor(candidates, device=result.device).argsort()
        for column in columns:
            result[physical, column] = result[physical, column].roll(-1)
    else:
        return result
    if 6 in columns or 7 in columns:
        result[:, 1] = (result[:, 6] * result[:, 7]).clamp_min(0).sqrt()
    result[:, 3] = result[:, 1] * result[:, 2]
    return result


def cost(logits, targets, loss='cost'):
    batch = logits.ndim == 2
    if not batch:
        logits = logits[None]
        targets = torch.as_tensor([targets], device=logits.device)
    if loss == 'ce':
        full = torch.cat([torch.zeros_like(logits[:, :1]), logits], 1)
        return F.cross_entropy(full, targets.clamp_min(0))
    chosen = (targets-1).clamp_min(0)
    true = logits.gather(1, chosen[:, None]).squeeze(1)
    wrong = logits.scatter(1, chosen[:, None], -torch.inf)
    return torch.where(targets > 0, F.softplus(-true) + F.softplus(wrong.amax(1)),
                       F.softplus(logits.amax(1))).mean()


def load_rows(dataset, split, budget=100):
    manifest = read(WORK / dataset / 'manifest.json')
    labels = read(WORK / dataset / 'labels.json')
    rows = [r for r in manifest['rows'] if r['split'] == split]
    if split == 'train' and budget != 100:
        from .prepare import csvread
        from .common import ROOT
        subset = {r['photo_id'] for r in csvread(ROOT / f'dataset/lists/train_subsets/{budget}.csv')}
        rows = [r for r in rows if r['original_id'] in subset]
    result = []
    for row in rows:
        value = torch.load(WORK / dataset / 'features' / (key(row) + '.pt'), weights_only=True)
        assert value['id'] == row['id'] and value['signature'] == manifest['signature']
        assert value['candidates'] == row['candidates']
        positions = [i for i, r in enumerate(row['candidates']) if r in labels[row['id']]]
        if split == 'train' and not positions:
            continue
        result.append(dict(row=row, features=value, positions=positions,
                           target=positions[0] if positions else -1))
    return manifest, result


def wait_rows(dataset, split, budget=100):
    while True:
        try:
            return load_rows(dataset, split, budget)
        except FileNotFoundError:
            time.sleep(20)


def summary(rows, selections):
    per_query, by_group, by_class = [], defaultdict(list), defaultdict(list)
    for item, order in zip(rows, selections):
        row, positions = item['row'], set(item['positions'])
        rank = next((i+1 for i, j in enumerate(order) if j in positions), None)
        raw_rank = min(positions)+1 if positions else None
        values = np.array([rank == 1, rank is not None and rank <= 5,
                           rank is not None and rank <= 10, 1/rank if rank else 0.], dtype=float)
        record = dict(id=row['id'], group=row['group'], identity=row['identity'], rank=rank,
                      raw_rank=raw_rank, correct=rank == 1, raw_correct=raw_rank == 1,
                      top_reference_indices=[row['candidates'][i] for i in order[:10]],
                      target_in_candidates=bool(positions))
        per_query.append(record)
        by_group[row['group']].append(values)
        by_class[row['identity']].append(values)
    all_values = np.array([v for g in by_group.values() for v in g])
    macro = np.mean([np.mean(g, 0) for g in by_class.values()], 0)
    report = dict(n=len(rows), correct=sum(r['correct'] for r in per_query),
                  micro=dict(zip(['R@1','R@5','R@10','MRR'], all_values.mean(0).tolist())),
                  macro=dict(zip(['R@1','R@5','R@10','MRR'], macro.tolist())),
                  rescued=sum(r['correct'] and not r['raw_correct'] for r in per_query),
                  harmed=sum(r['raw_correct'] and not r['correct'] for r in per_query),
                  candidate_recall=sum(bool(x['positions']) for x in rows)/len(rows),
                  mrr_scope='candidate-truncated diagnostic; final report restores the unchanged full-gallery tail')
    return report, per_query


def evaluate(rows, theta, kind, candidate_k=128, intervention=None, contents=None):
    orders = []
    for i, item in enumerate(rows):
        data = item['features']
        n = min(candidate_k, len(data['raw']))
        raw = data['raw'][:n]
        vals = intervene_mass(data['values'][:n], item['row']['candidates'][:n], intervention, mass_columns(kind))
        content = None if contents is None else contents[i][:n]
        logits = features(raw, vals, kind, content) @ theta.double().cpu()
        scores = torch.cat([torch.zeros(1, dtype=torch.float64), logits])
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        orders.append(order + list(range(n, len(data['raw']))))
    return summary(rows, orders)


def fit(rows, kind, loss='cost', steps=2000):
    xs = torch.stack([features(r['features']['raw'], r['features']['values'], kind) for r in rows])
    targets = torch.tensor([r['target'] for r in rows])
    assert (targets >= 0).all(), 'Training target missing from natural candidates'
    theta = torch.nn.Parameter(torch.zeros(xs.shape[-1], dtype=torch.float64))
    if kind in ('ADDITIVE4', 'PRODUCT5', 'ADDITIVE_UV'):
        warm = fit(rows, 'INTERNAL3', loss, steps)
        initial = [warm[0], 0., warm[1], warm[2]] if kind == 'ADDITIVE4' else [warm[0], 0., 0., warm[1], warm[2]]
        theta.data.copy_(torch.tensor(initial))
    optimizer = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective = cost(xs @ theta, targets, loss)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1.)
        optimizer.step()
    assert torch.isfinite(theta).all()
    return theta.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=KINDS, default='ADDITIVE4')
    parser.add_argument('--dataset', default='medicine')
    parser.add_argument('--loss', choices=['cost','ce'], default='cost')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    while True:
        try:
            manifest, train = load_rows(args.dataset, 'train', args.budget)
            _, val = load_rows(args.dataset, 'val')
            break
        except FileNotFoundError:
            if not args.wait:
                raise
            time.sleep(20)
    run = f'external_{args.kind}_{args.loss}_b{args.budget}'
    if args.dataset != 'medicine':
        run = args.dataset + '_' + run
    folder = RUNS / run
    cfg = dict(kind=args.kind, loss=args.loss, budget=args.budget, steps=2000,
               manifest=manifest['signature'],feature_recipe=train[0]['features']['support_signature'],
               seed=None, training_rows=[x['row']['id'] for x in train])
    sig = signature(cfg)
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature'] == sig
        return
    start = time.monotonic()
    theta = fit(train, args.kind, args.loss)
    save(folder/'checkpoint.pt', dict(theta=theta, config=cfg, signature=sig))
    validations = {}
    for k in (8,16,32,64,128):
        report, predictions = evaluate(val, theta, args.kind, k)
        validations[str(k)] = report
        save(folder/f'val_k{k}.json', dict(metrics=report, predictions=predictions))
    save(folder/'complete.json', dict(signature=sig, config=cfg, validation=validations,
                                     theta=theta.tolist(), seconds=time.monotonic()-start,
                                     train=evaluate(train,theta,args.kind)[0]))
    print(run, validations, flush=True)


if __name__ == '__main__':
    main()
