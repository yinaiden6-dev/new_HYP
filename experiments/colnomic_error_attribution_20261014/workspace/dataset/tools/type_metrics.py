"""Summarize retrieval ranks using current capture labels and manual corrections."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from photo_metadata import ROOT, SCENE_FIELDS, PhotoStore, read_csv, sha, write_json


def metrics(rows):
    if not rows:
        return dict(photos=0, packages=0, groups=0, recall_at_1=None,
                    recall_at_5=None, recall_at_10=None, mrr=None, package_macro=None)
    def scores(values):
        result = {'recall_at_' + str(k): sum(r['rank'] is not None and r['rank'] <= k
                   for r in values) / len(values) for k in (1, 5, 10)}
        result['mrr'] = sum(1 / r['rank'] if r['rank'] else 0 for r in values) / len(values)
        return result
    packages = defaultdict(list)
    for row in rows:
        packages[row['package_id']].append(row)
    package_scores = [scores(values) for values in packages.values()]
    macro = {k: sum(p[k] for p in package_scores) / len(package_scores) for k in package_scores[0]}
    return dict(photos=len(rows), packages=len(packages), groups=len({r['group_id'] for r in rows}),
                **scores(rows), package_macro=macro)


def evaluate(store, predictions, collection='photos', allow_prelabels=True):
    filenames = {'photos': 'photos.csv', 'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv',
                 'train_25': 'train_subsets/25.csv', 'train_50': 'train_subsets/50.csv'}
    # Hold the same lock as the editor for a consistent metadata/list snapshot.
    with store.locked():
        manifest = store.verify_active()
        labels = store.records()
        cohort = read_csv(store.root / 'lists' / filenames[collection])
    frozen = {r['photo_id'] for r in cohort}
    active = frozen
    excluded = frozen & {pid for pid, p in labels.items() if p['excluded']}
    if excluded:
        print(f'WARNING: {len(excluded)} formal photos are excluded only in active; using the full formal collection.')
    lookup = {}
    max_rank = len({p['sha256'] for p in read_csv(store.root / 'lists/references.csv')})
    for row in predictions:
        if not isinstance(row, dict) or 'photo_id' not in row or 'rank' not in row:
            raise ValueError('Each prediction needs photo_id and rank (positive integer or null)')
        pid, rank = row['photo_id'], row['rank']
        if pid in lookup or pid not in frozen:
            raise ValueError('Duplicate ID or photo outside the requested collection: ' + str(pid))
        if rank is not None and (type(rank) is not int or not 1 <= rank <= max_rank):
            raise ValueError('Invalid retrieval rank: ' + str(pid))
        lookup[pid] = rank
    missing = active - set(lookup)
    if missing:
        raise ValueError(f'Missing {len(missing)} active predictions; record failures with rank=null')
    unreviewed = [pid for pid in active if not labels[pid]['reviewed']]
    if unreviewed and not allow_prelabels:
        raise ValueError(f'{len(unreviewed)} active photos lack explicit human review; omit --require-reviewed to use current labels.')
    if any(any(labels[pid]['type'][key] is None for key in SCENE_FIELDS) for pid in active):
        raise ValueError('Some active photos have no type labels')
    groups = {r['package_id']: r['group_id'] for r in read_csv(store.root / 'lists/groups.csv')}
    rows = [dict(photo_id=pid, package_id=labels[pid]['package_id'],
                 group_id=groups[labels[pid]['package_id']], rank=lookup[pid], **labels[pid]['type'])
            for pid in sorted(active)]
    result = dict(collection=collection,
                  label_status='model_labels_with_optional_corrections' if unreviewed else 'human_reviewed',
                  reviewed_photos=len(active) - len(unreviewed), unreviewed_photos=len(unreviewed),
                  ignored_excluded_predictions=0,
                  annotation_sha256=manifest['annotation_sha256'], protocol_sha256=manifest['protocol_sha256'],
                  references_sha256=manifest['references_sha256'], overall=metrics(rows))
    for dimension in ('placement', 'background', 'joint'):
        subsets = defaultdict(list)
        for row in rows:
            key = row[dimension] if dimension != 'joint' else row['placement'] + '/' + row['background']
            subsets[key].append(row)
        result[dimension] = {key: metrics(values) for key, values in sorted(subsets.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, help='JSONL rows with photo_id and rank; null means retrieval failure')
    parser.add_argument('--collection', choices=['photos', 'train', 'val', 'test', 'train_25', 'train_50'], default='photos')
    parser.add_argument('--root', type=Path, default=ROOT)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--require-reviewed', action='store_true', help='Optionally require explicit human review for every active photo')
    policy.add_argument('--allow-prelabels', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    result = evaluate(PhotoStore(args.root), predictions, args.collection, allow_prelabels=not args.require_reviewed)
    result['predictions_sha256'] = sha(args.predictions.read_bytes())
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), overall=result['overall'], label_status=result['label_status']), indent=2))


if __name__ == '__main__':
    main()
