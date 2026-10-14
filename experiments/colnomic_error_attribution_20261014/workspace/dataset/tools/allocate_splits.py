"""Reproduce source-group allocations from a fixed rule and archived inputs."""
import argparse
import csv
import io
import json
import hashlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from splits import checksum, read_csv


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def apportion(total, weights):
    """Largest remainders, with category names breaking exact ties."""
    denominator = sum(weights.values())
    if not denominator:
        if total:
            raise ValueError('Cannot allocate a nonzero budget to an empty pool')
        return {name: 0 for name in weights}
    counts = {name: total * weight // denominator for name, weight in weights.items()}
    order = sorted(weights, key=lambda name: (-(total * weights[name] % denominator), name))
    for name in order[:total - sum(counts.values())]:
        counts[name] += 1
    return counts


def balance_scenes(splits, category, features, protected, seed):
    """Greedily swap same-category groups using metadata, never model scores."""
    roles = {gid: name for name, ids in splits.items() for gid in ids}
    counts = {name: len(splits[name]) for name in ('train', 'val', 'test')}
    total_groups = sum(counts.values())
    totals = [sum(v[i] for g, v in features.items() if roles[g] != 'review') for i in range(3)]
    current = {s: [sum(v[i] for g, v in features.items() if roles[g] == s) for i in range(3)]
               for s in counts}

    def score(name, values):
        return sum(Fraction((values[i] * total_groups - totals[i] * counts[name]) ** 2,
                            max(totals[i] * counts[name], 1) ** 2) for i in range(3))

    initial = sum(score(s, v) for s, v in current.items())
    movable = sorted(g for g in roles if g not in protected and roles[g] != 'review')
    swaps = []
    while True:
        best = None
        for index, a in enumerate(movable):
            for b in movable[index + 1:]:
                left, right = roles[a], roles[b]
                if left == right or category[a] != category[b]:
                    continue
                va = [v - features[a][i] + features[b][i] for i, v in enumerate(current[left])]
                vb = [v - features[b][i] + features[a][i] for i, v in enumerate(current[right])]
                delta = score(left, va) + score(right, vb) - score(left, current[left]) - score(right, current[right])
                tie = hashlib.sha256(f'{seed}:balance:{a}:{b}'.encode()).hexdigest()
                candidate = (delta, tie, a, b, va, vb)
                if delta < 0 and (best is None or candidate[:2] < best[:2]):
                    best = candidate
        if best is None:
            break
        _, _, a, b, va, vb = best
        left, right = roles[a], roles[b]
        swaps.append(dict(groups=[a, b], previous_splits=[left, right]))
        roles[a], roles[b] = right, left
        current[left], current[right] = va, vb
    balanced = {name: {g for g, role in roles.items() if role == name} for name in splits}
    audit = dict(features=['photos', 'handheld', 'multi_object'],
                 objective='Sum of squared relative deviations from source-group-proportional feature targets; exact rational arithmetic.',
                 initial_objective=float(initial), final_objective=float(sum(score(s, v) for s, v in current.items())),
                 swaps=swaps)
    return balanced, audit


def generate(root):

    root = Path(root).resolve()
    rules = json.loads((root / 'lists/allocation.json').read_text())
    if rules.get('revision') == 'package_supplement':
        from supplement_splits import generate as generate_supplement
        return generate_supplement(root, rules)
    if rules.get('revision') == 'human_reference_correction':
        return generate_corrected(root, rules)
    previous = root / rules['previous_lists']
    if checksum(previous / 'protocol.json') != rules['previous_protocol_sha256']:
        raise ValueError('Archived protocol changed')
    old = json.loads((previous / 'protocol.json').read_text())
    for filename, expected in old['output_sha256'].items():
        if checksum(previous / filename) != expected:
            raise ValueError('Archived split changed: ' + filename)
    for filename, expected in old['source_sha256'].items():
        if checksum(root / filename) != expected:
            raise ValueError('Dataset inventory changed: ' + filename)
    photos = read_csv(root / 'lists/photos.csv')
    packages = read_csv(root / 'lists/packages.csv')
    old_groups = read_csv(previous / 'groups.csv')
    group_of = {row['package_id']: row['group_id'] for row in old_groups}
    members = defaultdict(list)
    for package in packages:
        members[group_of[package['package_id']]].append(package)
    if any(len({p['category'] for p in rows}) != 1 for rows in members.values()):
        raise ValueError('A source group spans categories; specify its stratum explicitly')
    category = {gid: rows[0]['category'] for gid, rows in members.items()}
    held = {group_of[pid] for pid in rules['review']}
    eligible = set(members) - held
    targets = apportion(len(eligible), rules['group_percentages'])
    train = ({r['group_id'] for r in old_groups if r['split'] == 'train'} - held
             if rules['preserve_previous_train'] else set())
    protected = set(train)
    archive_root = previous.parent
    labels = {}
    label_hashes = {}
    for package in packages:
        path = archive_root / package['metadata_path']
        metadata = json.loads(path.read_text())
        labels.update({Path(p['image']).stem: p['type'] for p in metadata['photos']})
        label_hashes[package['metadata_path']] = checksum(path)

    def key(tag, gid):
        return hashlib.sha256(f"{rules['seed']}:{tag}:{gid}".encode()).hexdigest()

    def select(ids, count, tag):
        if not 0 <= count <= len(ids):
            raise ValueError('Requested budget conflicts with preserved groups')
        buckets = defaultdict(list)
        for gid in sorted(ids):
            buckets[category[gid]].append(gid)
        quotas = apportion(count, {c: len(v) for c, v in buckets.items()})
        return {gid for c, values in buckets.items()
                for gid in sorted(values, key=lambda g: key(tag, g))[:quotas[c]]}

    train |= select(eligible - train, targets['train'] - len(train), 'train')
    val = select(eligible - train, targets['val'], 'val')
    test = eligible - train - val
    splits = dict(train=train, val=val, test=test, review=held)
    features = defaultdict(lambda: [0, 0, 0])
    for photo in photos:
        values = features[group_of[photo['package_id']]]
        label = labels[photo['photo_id']]
        if label['placement'] not in ('tabletop', 'handheld') or label['background'] not in ('single_object', 'multi_object'):
            raise ValueError('Scene balancing needs complete two-category labels')
        values[0] += 1
        values[1] += label['placement'] == 'handheld'
        values[2] += label['background'] == 'multi_object'
    if rules['scene_balance'] != 'same_category_group_swaps':
        raise ValueError('Unknown scene balancing rule')
    splits, balance = balance_scenes(splits, category, features, protected, rules['seed'])
    train, val, test = (splits[name] for name in ('train', 'val', 'test'))
    assigned = {gid: name for name, ids in splits.items() for gid in ids}
    groups = [dict(row, split=assigned[row['group_id']]) for row in old_groups]
    split_of = {r['package_id']: r['split'] for r in groups}

    # Interleave categories proportionally; every budget keeps complete groups.
    buckets = {c: sorted((g for g in train if category[g] == c), key=lambda g: key('budget', g))
               for c in sorted({category[g] for g in train})}
    used = Counter()
    ordered = []
    for index in range(len(train)):
        chosen = min((c for c in buckets if used[c] < len(buckets[c])),
                     key=lambda c: (-(index + 1) * len(buckets[c]) + used[c] * len(train), c))
        ordered.append(buckets[chosen][used[chosen]])
        used[chosen] += 1
    collections = dict(splits)
    for percent in (25, 50):
        count = (len(train) * percent + 50) // 100
        collections[f'train_{percent}'] = set(ordered[:count])

    files = {}
    entries = {}
    strata = {}
    for name, ids in collections.items():
        rows = [p for p in photos if group_of[p['package_id']] in ids]
        filename = f'train_subsets/{name[6:]}.csv' if name.startswith('train_') else name + '.csv'
        files[filename] = csv_bytes(rows, list(photos[0]))
        entries[name] = dict(file='lists/' + filename, photos=len(rows),
                             packages=len({r['package_id'] for r in rows}), groups=len(ids),
                             categories=dict(Counter(r['category'] for r in rows)))
        strata[name] = {field: dict(Counter(labels[r['photo_id']][field] for r in rows))
                        for field in ('placement', 'background')}
    files['groups.csv'] = csv_bytes(groups, list(old_groups[0]))
    changed = [dict(package_id=r['package_id'], group_id=r['group_id'],
                    previous_split=r['split'], split=split_of[r['package_id']])
               for r in old_groups if r['split'] != split_of[r['package_id']]]
    owners = {r['reference_id']: r['package_id'] for r in read_csv(root / 'lists/positives.csv')}
    annotation_path = archive_root / 'annotations/confusion_metadata.json'
    annotation = json.loads(annotation_path.read_text())
    pairs = []
    for identifier, pair in sorted(annotation['pairs'].items()):
        if pair['status'] != 'confusing' or pair.get('source') != 'human':
            continue
        roles = [split_of.get(owners.get(r), 'catalog_only') for r in pair['reference_ids']]
        pairs.append(dict(pair_id=identifier, reference_ids=pair['reference_ids'], splits=roles,
                          training_candidate=all(s == 'train' for s in roles)))
    report = dict(protocol=rules['name'], previous_protocol=old['name'],
                  target_group_counts=targets, collections=entries, scene_counts=strata, scene_balance=balance,
                  changed_packages=changed, training_reference_count=entries['train']['packages'],
                  human_confusion_pairs=pairs,
                  training_pair_note='Reference-level candidates only; query-visible distinctions still require review.',
                  snapshot_sha256={'package_metadata': label_hashes,
                                   'confusion_metadata': checksum(annotation_path)})
    files['allocation_report.json'] = json_bytes(report)

    protocol = dict(old)
    protocol.update(name=rules['name'], seed=rules['seed'], frozen_date=rules['date'],
                    finalized_date=rules['date'], validation_group_count=len(val),
                    primary_collections=list(splits), collections=entries,
                    allocation='Preserve prior training groups; allocate remaining source groups by category and fixed SHA-256 order to 70/15/15 group targets, then balance scene metadata with same-category group swaps. Hold unresolved label groups in review.',
                    training_subsets='Nested proportional category-interleaved groups; budgets round to nearest group, half up.',
                    historical_exposure='All 1000 photos have development/inspection history. New training holdouts are not a historically untouched blind test.',
                    selection='Photo inventory, reference inventory, relevance and source-group identities unchanged; split assignments revised for adapter training.',
                    training_evaluation=dict(query_lists=['val.csv', 'test.csv'],
                                             references='references.csv', withheld='review.csv',
                                             checkpoint_selection='val only; test after method and checkpoint selection'),
                    previous_protocol=dict(name=old['name'], directory=rules['previous_lists'],
                                           protocol_sha256=rules['previous_protocol_sha256'],
                                           groups_sha256=checksum(previous / 'groups.csv')))
    protocol['source_sha256'] = dict(old['source_sha256'])
    protocol['source_sha256']['lists/allocation.json'] = checksum(root / 'lists/allocation.json')
    for filename in ('groups.csv', 'protocol.json'):
        protocol['source_sha256'][rules['previous_lists'] + '/' + filename] = checksum(previous / filename)
    protocol['output_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    protocol['output_sha256']['positives.csv'] = checksum(root / 'lists/positives.csv')
    files['protocol.json'] = json_bytes(protocol)
    return files


def generate_corrected(root, rules):
    """Restore a reviewed group while preserving every other v2 membership."""
    previous = root / rules['previous_lists']
    if checksum(previous / 'protocol.json') != rules['previous_protocol_sha256']:
        raise ValueError('Archived protocol changed')
    old = json.loads((previous / 'protocol.json').read_text())
    for name, expected in old['output_sha256'].items():
        if checksum(previous / name) != expected:
            raise ValueError('Archived split changed: ' + name)
    for name, expected in rules['source_sha256'].items():
        if checksum(root / name) != expected:
            raise ValueError('Corrected inventory changed: ' + name)
    correction = json.loads((root / rules['correction']).read_text())
    groups = read_csv(previous / 'groups.csv')
    parent = next(r for r in groups if r['package_id'] == correction['original_package'])
    restored = parent['group_id']
    groups.append(dict(parent, package_id=correction['new_package']))
    for row in groups:
        if row['group_id'] == restored:
            row['split'] = 'train'
    group_of = {r['package_id']: r['group_id'] for r in groups}
    splits = {s: {r['group_id'] for r in groups if r['split'] == s} for s in ('train', 'val', 'test', 'review')}
    for pct in (25, 50):
        rows = read_csv(previous / f'train_subsets/{pct}.csv')
        splits[f'train_{pct}'] = {group_of[r['package_id']] for r in rows}
    splits['train_50'].add(restored)
    for pct in (25, 50):
        if len(splits[f'train_{pct}']) != (len(splits['train']) * pct + 50) // 100:
            raise ValueError('Restored group no longer meets the fixed nested budget')
    labels = {}
    for row in read_csv(previous / 'packages.csv'):
        metadata = json.loads((previous.parent / row['metadata_path']).read_text())
        labels.update({Path(p['image']).stem: p['type'] for p in metadata['photos']})
    photos = read_csv(root / 'lists/photos.csv')
    by_id = {r['photo_id']: r for r in photos}
    positives = {r['package_id']: r['reference_id'] for r in read_csv(root / 'lists/positives.csv')}
    for row in correction['assignments']:
        photo = by_id[row['photo_id']]
        if (photo['package_id'], photo['sha256'], positives[photo['package_id']]) != (row['package_id'], row['sha256'], row['reference_id']):
            raise ValueError('Human reference assignment differs: ' + row['photo_id'])
    entries, strata, files = {}, {}, {}
    for name, ids in splits.items():
        rows = [r for r in photos if group_of[r['package_id']] in ids]
        filename = f'train_subsets/{name[6:]}.csv' if name.startswith('train_') else name + '.csv'
        files[filename] = csv_bytes(rows, list(photos[0]))
        entries[name] = dict(file='lists/' + filename, photos=len(rows), packages=len({r['package_id'] for r in rows}),
                             groups=len(ids), categories=dict(Counter(r['category'] for r in rows)))
        strata[name] = {field: dict(Counter(labels[r['photo_id']][field] for r in rows)) for field in ('placement', 'background')}
    files['groups.csv'] = csv_bytes(groups, list(groups[0]))
    report = json.loads((previous / 'allocation_report.json').read_text())
    report.update(protocol=rules['name'], previous_protocol=old['name'], collections=entries, scene_counts=strata,
                  target_group_counts={s: len(splits[s]) for s in ('train', 'val', 'test')},
                  training_reference_count=entries['train']['packages'],
                  changed_packages=[dict(package_id=p, group_id=restored, previous_split='review', split='train')
                                    for p in (correction['original_package'], correction['new_package'])],
                  scene_balance={'rule': 'Keep all v2 memberships; add the resolved group to train. No new search.'})
    files['allocation_report.json'] = json_bytes(report)
    protocol = dict(old)
    protocol.update(name=rules['name'], frozen_date=rules['date'], finalized_date=rules['date'],
                    inventory_version='0.7.1-rc1', collections=entries,
                    allocation='Keep v2 memberships. Split the two human-confirmed designs within their original source group and restore all 12 reviewed photos to train, reaching 75/16/16 groups.',
                    training_subsets='Keep v2 25% and 50% groups; add the resolved group to 50%. Nested budgets: 19/38/75 groups.',
                    selection='Seven human-confirmed positive assignments corrected; photo bytes, reference bytes, IDs, order and photo-to-source-group membership unchanged.',
                    previous_protocol=dict(name=old['name'], directory=rules['previous_lists'],
                                           protocol_sha256=rules['previous_protocol_sha256'], groups_sha256=checksum(previous / 'groups.csv')))
    protocol['source_sha256'] = dict(rules['source_sha256'], **{'lists/allocation.json': checksum(root / 'lists/allocation.json')})
    for filename in ('protocol.json', 'groups.csv'):
        protocol['source_sha256'][rules['previous_lists'] + '/' + filename] = checksum(previous / filename)
    protocol['output_sha256'] = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    protocol['output_sha256']['positives.csv'] = checksum(root / 'lists/positives.csv')
    files['protocol.json'] = json_bytes(protocol)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='Empty staging directory; never overwrites formal lists')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty')
    files = generate(args.root)
    for name, data in files.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(json.loads(files['protocol.json'])['collections'], indent=2))


if __name__ == '__main__':
    main()
