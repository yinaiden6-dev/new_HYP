"""Photo labels, review decisions, and reproducible filtered collections."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PLACEMENTS = ('tabletop', 'handheld', 'other', 'unclear')
BACKGROUNDS = ('single_object', 'multi_object', 'unclear')
OCCLUSIONS = ('foreground', 'none', 'unclear')
SCENE_FIELDS = ('placement', 'background')
COLLECTIONS = ('photos.csv', 'train.csv', 'val.csv', 'test.csv',
               'train_subsets/25.csv', 'train_subsets/50.csv')
ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data if isinstance(data, bytes) else data.encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_photo(photo):
    if set(photo) != {'image', 'type', 'reviewed', 'excluded'}:
        raise ValueError('Unexpected photo metadata fields')
    if not isinstance(photo['image'], str) or Path(photo['image']).parent.as_posix() != 'photos':
        raise ValueError('Invalid photo path')
    if (not isinstance(photo['type'], dict) or not set(SCENE_FIELDS).issubset(photo['type'])
            or set(photo['type']) - {*SCENE_FIELDS, 'occlusion'}):
        raise ValueError('Expected placement, background and optional occlusion')
    for key, values in [('placement', PLACEMENTS), ('background', BACKGROUNDS), ('occlusion', OCCLUSIONS)]:
        if photo['type'].get(key) not in (None, *values):
            raise ValueError('Invalid ' + key)
    if type(photo['reviewed']) is not bool or type(photo['excluded']) is not bool:
        raise ValueError('Review and exclusion flags must be booleans')
    if photo['reviewed'] and any(photo['type'][key] is None for key in SCENE_FIELDS):
        raise ValueError('Choose both labels before confirming a photo')


class Conflict(ValueError):
    pass


class PhotoStore:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.review = self.root / 'review/photos'
        self.review.mkdir(parents=True, exist_ok=True)
        self.packages = {p['package_id']: p for p in read_csv(self.root / 'lists/packages.csv')}
        self.photos = {p['photo_id']: p for p in read_csv(self.root / 'lists/photos.csv')}
        self.roles = {p['package_id']: p['split'] for p in read_csv(self.root / 'lists/groups.csv')}
        self.thread_lock = threading.RLock()

    @contextmanager
    def locked(self):
        with self.thread_lock:
            if getattr(self, '_in_lock', False):
                yield
                return
            with (self.review / '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._in_lock = True
                try:
                    yield
                finally:
                    self._in_lock = False
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def package_path(self, package_id):
        return self.root / self.packages[package_id]['metadata_path']

    def package(self, package_id):
        data = self.package_path(package_id).read_bytes()
        return json.loads(data), sha(data)

    def records(self):
        result = {}
        for package_id in self.packages:
            package, _ = self.package(package_id)
            for photo in package['photos']:
                validate_photo(photo)
                photo_id = Path(photo['image']).stem
                row = self.photos[photo_id]
                if row['package_id'] != package_id or row['path'] != self.packages[package_id]['directory'] + '/' + photo['image']:
                    raise ValueError('Photo path or package mismatch: ' + photo_id)
                if photo_id in result:
                    raise ValueError('Repeated photo: ' + photo_id)
                result[photo_id] = dict(photo, photo_id=photo_id, package_id=package_id,
                                       split=self.roles[package_id])
        if set(result) != set(self.photos):
            raise ValueError('Metadata does not cover the photo inventory')
        return result

    def event(self, record):
        with (self.review / 'edits.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), **record)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def update(self, package_id, revision, changes, actor='human', export=True):
        with self.locked():
            package, current = self.package(package_id)
            if revision is not None and revision != current:
                raise Conflict('This package changed. Reload before saving.')
            if not isinstance(changes, list) or not changes:
                raise ValueError('Expected a nonempty list of photo changes')
            index = {Path(p['image']).stem: p for p in package['photos']}
            seen, edits = set(), []
            for change in changes:
                if not isinstance(change, dict) or set(change) != {'photo_id', 'type', 'reviewed', 'excluded'}:
                    raise ValueError('Invalid change fields')
                photo_id = change['photo_id']
                if photo_id not in index or photo_id in seen:
                    raise ValueError('Unknown or repeated photo ID')
                seen.add(photo_id)
                old = index[photo_id]
                new = dict(image=old['image'], **{k: change[k] for k in ('type', 'reviewed', 'excluded')})
                validate_photo(new)
                # Older editor tabs and the two-label prelabeler omit this optional field.
                if 'occlusion' in old['type'] and 'occlusion' not in new['type']:
                    new['type'] = dict(new['type'], occlusion=old['type']['occlusion'])
                if actor == 'model':
                    # A resumed prelabel run never replaces a human decision or existing label.
                    if old['reviewed'] or any(v is not None for v in old['type'].values()):
                        continue
                    new['reviewed'], new['excluded'] = False, old['excluded']
                if old != new:
                    edits.append(dict(photo_id=photo_id, before=dict(old), after=dict(new)))
                    old.clear()
                    old.update(new)
            if edits:
                write_json(self.package_path(package_id), package)
                self.event(dict(actor=actor, package_id=package_id, edits=edits))
            if export:
                self.export_active_unlocked()
            return self.package(package_id)

    def active_payload(self):
        records = self.records()
        annotation_bytes = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
        outputs, counts, source_hashes = {}, {}, {}
        names = COLLECTIONS + (('review.csv',) if (self.root / 'lists/review.csv').exists() else ())
        for name in names:
            source = self.root / 'lists' / name
            rows = read_csv(source)
            selected = [r for r in rows if not records[r['photo_id']]['excluded']]
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=next(csv.reader(io.StringIO(source.read_text()))))
            writer.writeheader()
            writer.writerows(selected)
            outputs[name] = stream.getvalue().encode()
            counts[name] = len(selected)
            source_hashes[name] = sha(source.read_bytes())
        summary = dict(total=len(records), excluded=sum(p['excluded'] for p in records.values()),
                       reviewed=sum(p['reviewed'] for p in records.values()),
                       labeled=sum(all(p['type'][key] is not None for key in SCENE_FIELDS) for p in records.values()),
                       occlusion_labeled=sum(p['type'].get('occlusion') is not None for p in records.values()))
        manifest = dict(schema_version=1, annotation_sha256=sha(annotation_bytes),
                        protocol_sha256=sha((self.root / 'lists/protocol.json').read_bytes()),
                        references_sha256=sha((self.root / 'lists/references.csv').read_bytes()),
                        source_sha256=source_hashes, counts=counts, summary=summary,
                        output_sha256={n: sha(v) for n, v in outputs.items()})
        return outputs, manifest

    def export_active_unlocked(self):
        outputs, manifest = self.active_payload()
        for name, data in outputs.items():
            atomic_write(self.root / 'lists/active' / name, data)
        # Readers verify checksums against this last-written commit marker.
        write_json(self.root / 'lists/active/manifest.json', manifest)
        return manifest

    def export_active(self):
        with self.locked():
            return self.export_active_unlocked()

    def verify_active(self):
        with self.locked():
            outputs, expected = self.active_payload()
            actual = json.loads((self.root / 'lists/active/manifest.json').read_text())
            if actual != expected:
                raise ValueError('Active manifest is stale; run photo_metadata.py export')
            for name, data in outputs.items():
                if (self.root / 'lists/active' / name).read_bytes() != data:
                    raise ValueError('Active collection mismatch: ' + name)
            return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'check', 'summary'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    store = PhotoStore(args.root)
    if args.command == 'export':
        result = store.export_active()
    elif args.command == 'check':
        result = store.verify_active()
    else:
        records = list(store.records().values())
        result = {key: dict(Counter(str(p['type'].get(key)) for p in records))
                  for key in (*SCENE_FIELDS, 'occlusion')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
