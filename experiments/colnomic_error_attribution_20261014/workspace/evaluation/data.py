"""Read formal inventories without filtering or shortening retrieval inputs."""
import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def encoding_inputs(rows, population):
    key = 'photo_id' if population == 'photos' else 'reference_id'
    return [(r[key], r['sha256'], r.get('source_sha256', r['sha256'])) for r in rows]


def validate_encoding_manifest(root, population, manifest, bank=None):
    """Labels and paths may change; encoded image IDs, order and bytes may not."""
    recorded = read_csv(manifest)
    current = read_csv(Path(root) / 'dataset/lists' / (population + '.csv'))
    if encoding_inputs(recorded, population) != encoding_inputs(current, population):
        raise ValueError('Encoding inputs changed: ' + population)
    if bank is not None:
        key = 'photo_id' if population == 'photos' else 'reference_id'
        expected = dict(ids=[r[key] for r in recorded], paths=['dataset/' + r['path'] for r in recorded],
                        identities=[r['identity'] for r in recorded], sha256=[r['sha256'] for r in recorded],
                        source_sha256=[r.get('source_sha256', r['sha256']) for r in recorded],
                        manifest_sha256=file_hash(manifest))
        if any(bank.get(k) != value for k, value in expected.items()):
            raise ValueError('Bank differs from its recorded encoding manifest')
    return recorded


class Dataset:
    @property
    def full_protocol(self):
        return 'Full-1k' if len(self.photos) == 1000 else f'Full-{len(self.photos)}'

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lists = self.root / 'dataset/lists'
        self.photos = read_csv(self.lists / 'photos.csv')
        self.references = read_csv(self.lists / 'references.csv')
        self.groups = {r['package_id']: r['group_id'] for r in read_csv(self.lists / 'groups.csv')}
        self.positives = {}
        for row in read_csv(self.lists / 'positives.csv'):
            self.positives.setdefault(row['package_id'], set()).add(row['reference_id'])
        self.hashes = {p.name: file_hash(p) for p in [self.lists / (name + '.csv')
                      for name in ('photos', 'references', 'positives', 'groups')]}
        self.signature = digest(self.hashes)
        for pop in ('photos', 'references'):
            ids = [r[self.id_key(pop)] for r in self.records(pop)]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f'Empty or duplicate inventory: {pop}')
        if [int(r['row']) for r in self.references] != list(range(len(self.references))):
            raise ValueError('Reference rows are not consecutive')
        reference_ids = {r['reference_id'] for r in self.references}
        for row in self.photos:
            pkg = row['package_id']
            if pkg not in self.groups or not self.positives.get(pkg) or not self.positives[pkg] <= reference_ids:
                raise ValueError(f'Incomplete group/positive mapping: {pkg}')
        self.labels = {}
        for package in read_csv(self.lists / 'packages.csv'):
            for photo in read_json(self.root / 'dataset' / package['metadata_path'])['photos']:
                self.labels[Path(photo['image']).stem] = photo['type']
        if set(r['photo_id'] for r in self.photos) - self.labels.keys():
            raise ValueError('Missing photo type annotations')
        self.annotation_sha256 = digest({r['photo_id']:self.labels[r['photo_id']] for r in self.photos})
        active = self.lists / 'active/photos.csv'
        self.active_difference = None
        if active.exists() and read_csv(active) != self.photos:
            active_ids = {r['photo_id'] for r in read_csv(active)}
            formal_ids = {r['photo_id'] for r in self.photos}
            self.active_difference = dict(formal_only=sorted(formal_ids-active_ids),
                                          active_only=sorted(active_ids-formal_ids),
                                          metadata_or_order_differs=True)
            print('WARNING: active differs; using every formal photo:', self.active_difference, flush=True)
        self._verified = set()

    def matches_full_inventory(self, hashes):
        """Accept old Full-1k provenance only when split roles alone changed.

        Training runs must additionally pin protocol.json; this is not a
        training-split compatibility check.
        """
        if hashes == self.hashes:
            return True
        if not isinstance(hashes, dict) or set(hashes) != set(self.hashes):
            return False
        if any(hashes[k] != value for k, value in self.hashes.items() if k != 'groups.csv'):
            return False
        previous = read_json(self.lists / 'protocol.json').get('previous_protocol')
        if not previous or hashes['groups.csv'] != previous['groups_sha256']:
            return False
        path = self.root / 'dataset' / previous['directory'] / 'groups.csv'
        if not path.is_file() or file_hash(path) != hashes['groups.csv']:
            return False
        rows = read_csv(path)
        groups = {r['package_id']: r['group_id'] for r in rows}
        return len(rows) == len(groups) and groups == self.groups

    def matches_encoding_inventory(self, hashes):
        """Check reusable image inputs, never compatibility of old metrics."""
        if not isinstance(hashes, dict):
            return False
        for population in ('photos', 'references'):
            path = self.root / 'cache/image_embeddings/manifests' / (population + '.csv')
            if hashes.get(population + '.csv') == self.hashes[population + '.csv']:
                continue
            if not path.is_file() or hashes.get(population + '.csv') != file_hash(path):
                return False
            try:
                validate_encoding_manifest(self.root, population, path)
            except ValueError:
                return False
        return True

    @staticmethod
    def id_key(population):
        return 'photo_id' if population == 'photos' else 'reference_id'

    def records(self, population):
        if population not in ('photos', 'references'):
            raise ValueError(population)
        return getattr(self, population)

    def verify_images(self, population):
        for row in self.records(population):
            path = self.root / 'dataset' / row['path']
            key = (str(path), row['sha256'])
            if key not in self._verified:
                if file_hash(path) != row['sha256']:
                    raise ValueError(f'Image hash mismatch: {path}')
                self._verified.add(key)
