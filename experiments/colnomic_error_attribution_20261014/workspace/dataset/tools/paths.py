"""Export image paths from a named collection using only the standard library."""
import argparse
import csv
from pathlib import Path


COLLECTIONS = {
    'references': 'references.csv', 'photos': 'photos.csv',
    'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv', 'review': 'review.csv',
    'train_25': 'train_subsets/25.csv', 'train_50': 'train_subsets/50.csv',
}
COLLECTIONS.update({f'active/{name}': 'active/' + filename
                    for name, filename in list(COLLECTIONS.items())
                    if name in ('photos', 'train', 'val', 'test', 'review', 'train_25', 'train_50')})


def get_paths(root, collection, relative=False):
    root = Path(root).resolve()
    if collection.startswith('active/'):
        from photo_metadata import PhotoStore
        PhotoStore(root).verify_active()
    with (root / 'lists' / COLLECTIONS[collection]).open() as stream:
        rows = list(csv.DictReader(stream))
    # Keep row order and duplicate reference entries from the collection.
    return [row['path'] if relative else str(root / row['path']) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', choices=list(COLLECTIONS))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--relative', action='store_true', help='Use paths relative to the dataset root')
    parser.add_argument('--output', type=Path, help='Write one image path per line to this file')
    args = parser.parse_args()
    result = '\n'.join(get_paths(args.root, args.collection, args.relative)) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
