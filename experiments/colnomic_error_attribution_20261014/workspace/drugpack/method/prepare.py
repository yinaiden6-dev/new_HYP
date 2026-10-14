import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, WORK, read, save, sha, signature


def csvread(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def medicine():
    from evaluation.data import Dataset
    ds = Dataset(ROOT)
    lists = ROOT / 'dataset/lists'
    protocol = read(lists / 'protocol.json')
    roles, seen = {}, set()
    for split in ('train', 'val', 'test'):
        assert sha(lists / f'{split}.csv') == protocol['output_sha256'][f'{split}.csv']
        items = csvread(lists / f'{split}.csv')
        groups = {ds.groups[r['package_id']] for r in items}
        assert not groups & seen
        seen |= groups
        roles.update({r['photo_id']: split for r in items})
    first = {}
    for i, r in enumerate(ds.references):
        first.setdefault(r['sha256'], i)
    index = {r['reference_id']: first[r['sha256']] for r in ds.references}
    positives = {p: sorted({index[r] for r in rr}) for p, rr in ds.positives.items()}
    train_refs = sorted({i for r in ds.photos if roles[r['photo_id']] == 'train'
                         for i in positives[r['package_id']]})
    held_refs = {i for r in ds.photos if roles[r['photo_id']] != 'train'
                 for i in positives[r['package_id']]}
    assert not held_refs.intersection(train_refs)
    photos = [dict(id='medicine:' + r['photo_id'], original_id=r['photo_id'],
                   path=str(ROOT / 'dataset' / r['path']), sha256=r['sha256'], split=roles[r['photo_id']],
                   group=ds.groups[r['package_id']], identity=r['package_id']) for r in ds.photos]
    refs = [dict(id=r['reference_id'], path=str(ROOT / 'dataset' / r['path']), sha256=r['sha256'])
            for r in ds.references]
    labels = {q['id']: positives[r['package_id']] for q, r in zip(photos, ds.photos)}
    scores = torch.load(ROOT / 'cache/scores/colnomic7b/image.pt', weights_only=True, mmap=True)
    assert scores['query_ids'] == [r['photo_id'] for r in ds.photos]
    assert scores['reference_ids'] == [r['reference_id'] for r in ds.references]
    return photos, refs, labels, scores['scores'], train_refs, list(first.values()), dict(
        protocol=protocol['name'], protocol_sha256=sha(lists / 'protocol.json'), inventory=ds.hashes,
        historical_exposure=protocol['historical_exposure'],
        query_bank=str(ROOT / 'cache/image_embeddings/colnomic7b/photos.pt'),
        reference_bank=str(ROOT / 'cache/image_embeddings/colnomic7b/references.pt'),
        score_bank=str(ROOT / 'cache/scores/colnomic7b/image.pt'))


def external(name):
    if name == 'grozi':
        data = ROOT / 'other_datasets/GroZi-3.2k'
        meta = read(data / 'retrieval_subset/metadata.json')
        photos = [dict(id='grozi:' + r['photo_id'], original_id=r['photo_id'],
                       path=str(data / 'retrieval_subset' / r['image_path']), sha256=r['sha256'],
                       split='transfer', group=str(r['source_image_id']), identity=str(r['class_id']))
                  for r in meta['photos']]
        refs = [dict(id=r['reference_id'], path=str(data / 'retrieval_subset' / r['image_path']),
                     sha256=r['sha256']) for r in meta['references']]
        labels = {q['id']: [r['positive_reference_index']] for q, r in zip(photos, meta['photos'])}
        score_path = data / 'cache/scores/colnomic7b/image.pt'
        score = torch.load(score_path, weights_only=True, mmap=True)
        score = score['scores'] if isinstance(score, dict) else score
        source = dict(protocol=meta['subset'], query_bank=str(data / 'cache/image_embeddings/colnomic7b/photo.pt'),
                      reference_bank=str(data / 'cache/image_embeddings/colnomic7b/ref.pt'), score_bank=str(score_path))
    elif name == 'milk':
        data = ROOT / 'other_datasets/MILK10K'
        meta = read(data / 'cache/test_manifest.json')
        pairs = meta['pairs']
        photos = [dict(id='milk:' + r['clinical']['image_id'], original_id=r['clinical']['image_id'],
                       path=str(data / r['clinical']['path']), sha256=r['clinical']['sha256'],
                       split='transfer', group=r['lesion_id'], identity=r['lesion_id']) for r in pairs]
        refs = [dict(id=r['dermoscopic']['image_id'], path=str(data / r['dermoscopic']['path']),
                     sha256=r['dermoscopic']['sha256']) for r in pairs]
        labels = {q['id']: [i] for i, q in enumerate(photos)}
        score_path = data / 'cache/scores/colnomic7b.pt'
        score = torch.load(score_path, weights_only=True, mmap=True)
        score = score['scores'] if isinstance(score, dict) else score
        source = dict(protocol=meta['protocol'], query_bank=str(data / 'cache/image_embeddings/colnomic7b/clinical.pt'),
                      reference_bank=str(data / 'cache/image_embeddings/colnomic7b/dermoscopic.pt'), score_bank=str(score_path))
    elif name in ('forb','forb32'):
        data = ROOT / 'other_datasets/FORB'
        meta = read(data/'cache/manifest.json')
        photos = [dict(id='forb:'+r['id'],original_id=r['id'],path=str(data/r['image_path']),
                       sha256=r['sha256'],split='transfer',group=r['id'],identity=r['domain'],
                       domain=r['domain'],difficulty=r['difficulty']) for r in meta['photos']]
        refs = [dict(id=r['id'],path=str(data/r['image_path']),sha256=r['sha256']) for r in meta['references']]
        labels = {q['id']:r['positive_indices'] for q,r in zip(photos,meta['photos'])}
        score_path = data/'cache/scores/colnomic7b/scores.npy'
        score = torch.from_numpy(np.load(score_path,mmap_mode='r'))
        source = dict(protocol=meta['protocol'],query_bank=str(data/'cache/image_embeddings/colnomic7b/photo'),
                      reference_bank=str(data/'cache/image_embeddings/colnomic7b/ref'),score_bank=str(score_path))
        axis = read(score_path.parent/'manifest.json')
        assert axis['query_ids'] == [x['original_id'] for x in photos]
        assert axis['reference_ids'] == [x['id'] for x in refs]
    else:
        raise ValueError(name)
    return photos, refs, labels, score, [], list(range(len(refs))), source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='medicine', choices=['medicine', 'medicine_catalog', 'grozi', 'milk', 'forb', 'forb32'])
    parser.add_argument('--candidates', type=int, default=128)
    args = parser.parse_args()
    if args.dataset == 'forb32':
        args.candidates = 32
    torch.set_num_threads(2)
    photos, refs, labels, scores, train_refs, unique, source = medicine() if args.dataset.startswith('medicine') else external(args.dataset)
    assert tuple(scores.shape) == (len(photos), len(refs)) and torch.isfinite(scores).all()
    rows = []
    for qi, photo in enumerate(photos):
        pool = train_refs if photo['split'] == 'train' and args.dataset == 'medicine' else unique
        values = scores[qi, pool].numpy()
        n = min(args.candidates,len(values))
        cutoff = np.partition(values,len(values)-n)[len(values)-n]
        above = np.flatnonzero(values > cutoff)
        tied = np.flatnonzero(values == cutoff)[:n-len(above)]
        selected = np.sort(np.concatenate([above,tied]))
        rank = selected[np.argsort(-values[selected], kind='stable')]
        candidates = [pool[i] for i in rank[:args.candidates]]
        rows.append(dict(**photo, query_index=qi, candidates=candidates,
                         raw_scores=[float(scores[qi, i]) for i in candidates]))
    result = dict(dataset=args.dataset, source=source, references=refs, rows=rows,
                  train_references=train_refs, gallery_indices=unique, candidates=args.candidates,
                  training_reference_policy='Only train-positive designs; held designs excluded from all training inputs.',
                  transfer_policy='Frozen medicine-selected checkpoint; no fitting or selection on transfer outcomes.')
    if args.dataset == 'medicine_catalog':
        result.update(hidden_dataset='medicine',hidden_signature=read(WORK/'medicine/manifest.json')['signature'],
                      roma_parent='medicine',training_reference_policy='Full fixed design catalog available as negatives; only train-photo identity labels fit parameters. This is catalog-visible training, not unseen-reference training.')
    if args.dataset == 'forb32':
        result.update(hidden_dataset='forb',hidden_signature=read(WORK/'forb/manifest.json')['signature'],
                      candidate_budget_reason='Top-32 reranking fixed before any new-model transfer evaluation to bound full 8995-query runtime. Initial retrieval searches all 43585 references; no query filtering or target insertion.')
    result['signature'] = signature(result)
    path = WORK / args.dataset / 'manifest.json'
    if path.exists():
        assert read(path) == result, 'Existing manifest changed'
    save(path, result)
    save(path.parent / 'labels.json', labels)
    print(json_summary(result), flush=True)


def json_summary(result):
    return {k: v for k, v in dict(dataset=result['dataset'], queries=len(result['rows']),
                                references=len(result['references']),
                                pairs=sum(len(r['candidates']) for r in result['rows']),
                                signature=result['signature']).items()}


if __name__ == '__main__':
    main()
