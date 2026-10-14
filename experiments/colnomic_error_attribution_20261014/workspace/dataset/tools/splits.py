"""Check versioned source-group splits and expose allowed training references."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_reference_ids(root, collection='train'):
    names = {'train': 'train.csv', 'train_25': 'train_subsets/25.csv',
             'train_50': 'train_subsets/50.csv'}
    if collection not in names:
        raise ValueError('Expected a training collection')
    package_ids = {row['package_id'] for row in read_csv(root / 'lists' / names[collection])}
    return [row['reference_id'] for row in read_csv(root / 'lists/positives.csv')
            if row['package_id'] in package_ids]


def historical_inputs(protocol):
    previous = protocol.get('previous_protocol', {})
    if not previous.get('directory'):
        return set()
    return {str(Path(previous['directory']) / name) for name in ('protocol.json', 'groups.csv')}


def verify_input_hashes(root, protocol, *, reconstruct=True):
    """Only explicitly declared reconstruction inputs may be omitted."""
    errors = []
    omitted = historical_inputs(protocol) if not reconstruct else set()
    for relative, expected in protocol['source_sha256'].items():
        if relative in omitted:
            continue
        path = root / relative
        if not path.is_file() or checksum(path) != expected:
            errors.append('Protocol input checksum differs: ' + relative)
    return errors


def verify(root, *, reconstruct=True):
    root = Path(root)
    protocol = json.loads((root / 'lists/protocol.json').read_text())
    errors = verify_input_hashes(root, protocol, reconstruct=reconstruct)
    for relative, expected in protocol['output_sha256'].items():
        path = root / 'lists' / relative
        if not path.is_file() or checksum(path) != expected:
            errors.append('Frozen split checksum differs: ' + relative)
    photos = read_csv(root / 'lists/photos.csv')
    by_id = {row['photo_id']: row for row in photos}
    references = read_csv(root / 'lists/references.csv')
    groups = read_csv(root / 'lists/groups.csv')
    group_of = {row['package_id']: row['group_id'] for row in groups}
    split_of = {row['package_id']: row['split'] for row in groups}
    if len(group_of) != len(groups) or set(group_of) != {r['package_id'] for r in photos}:
        errors.append('Source groups must cover each photographed package exactly once')
    if len(by_id) != len(photos) or len(photos) != protocol['photo_count']:
        errors.append('Invalid photo inventory count or repeated IDs')
    if len(references) != protocol['reference_count']:
        errors.append('Reference inventory count differs')
    if len(set(group_of.values())) != protocol['group_count']:
        errors.append('Source-group count differs')
    group_splits = defaultdict(set)
    for row in groups:
        group_splits[row['group_id']].add(row['split'])
    if any(len(values) != 1 for values in group_splits.values()):
        errors.append('A source group crosses primary splits')
    collections = {}
    for name, entry in protocol['collections'].items():
        rows = read_csv(root / entry['file'])
        ids = [row['photo_id'] for row in rows]
        collections[name] = set(ids)
        if len(ids) != len(set(ids)) or any(row != by_id.get(row['photo_id']) for row in rows):
            errors.append('Invalid photo membership or metadata: ' + name)
        expected = dict(photos=len(rows), packages=len({r['package_id'] for r in rows}),
                        groups=len({group_of[r['package_id']] for r in rows}),
                        categories=dict(Counter(r['category'] for r in rows)))
        if any(entry[key] != value for key, value in expected.items()):
            errors.append('Protocol counts differ: ' + name)
        split = 'train' if name.startswith('train') else name
        if any(split_of[row['package_id']] != split for row in rows):
            errors.append('A photo changed its frozen split: ' + name)
        chosen = {group_of[r['package_id']] for r in rows}
        if set(ids) != {r['photo_id'] for r in photos if group_of[r['package_id']] in chosen}:
            errors.append('Collection does not retain complete remaining source groups: ' + name)
    primary = protocol.get('primary_collections', ['train', 'val', 'test'])
    counts = Counter(pid for name in primary for pid in collections[name])
    if set(counts) != set(by_id) or any(n != 1 for n in counts.values()):
        errors.append('Primary splits, including review holds, must partition the retained photos')
    train = collections['train']
    if not collections['train_25'] <= collections['train_50'] <= train:
        errors.append('Training subsets are not nested')
    associations = defaultdict(set)
    packages = [json.loads((root / row['metadata_path']).read_text())
                for row in read_csv(root / 'lists/packages.csv')]
    for package in packages:
        split = split_of[package['package_id']]
        associations[('identity', package['identity'])].add(split)
        associations[('design', package['design']['sha256'])].add(split)
        for document in package['documents']:
            for key in ('document_id', 'set_id'):
                associations[(key, document[key])].add(split)
            for product in document['products']:
                if product.get('code'):
                    associations[('product', product['code_system'], product['code'])].add(split)
    for row in photos:
        for key in ('sha256', 'source_sha256'):
            associations[(key, row[key])].add(split_of[row['package_id']])
    if any(len(value) > 1 for value in associations.values()):
        errors.append('A design, source identifier or photo hash crosses primary splits')
    expected = {(p['package_id'], r['reference_id']) for p in packages for r in references
                if p['design']['sha256'] == r['sha256']}
    positives = read_csv(root / 'lists/positives.csv')
    actual = {(r['package_id'], r['reference_id']) for r in positives}
    if actual != expected or len(actual) != len(positives):
        errors.append('Positive reference mapping differs from exact image-byte matches')
    if reconstruct and 'previous_protocol' in protocol:
        from allocate_splits import generate
        try:
            for name, data in generate(root).items():
                if (root / 'lists' / name).read_bytes() != data:
                    errors.append('Fixed allocation does not reproduce: ' + name)
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors.append('Cannot reproduce allocation: ' + str(error))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = verify(args.root)
    print(json.dumps(dict(status='failed' if errors else 'passed', errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
