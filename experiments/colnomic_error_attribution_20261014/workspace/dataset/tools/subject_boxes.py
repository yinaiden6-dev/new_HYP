"""Human subject boxes, independent of scene labels and evaluation manifests."""
import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache

from PIL import Image, ImageOps
if __package__:
    from .photo_metadata import Conflict, sha, write_json
else:
    from photo_metadata import Conflict, sha, write_json


def box_revision(record):
    return sha(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())


def validate_box(box, width, height):
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError('Box must contain four finite pixel coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Box must lie inside the oriented original image')
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError('Box must be at least 2 pixels on each side')
    return [round(float(v), 4) for v in box]


def resolve_records(photos, records, prelabels):
    """Human decisions override defaults, including an explicit clear or unclear."""
    resolved = dict(records)
    for photo in photos:
        pid = photo['photo_id']
        candidate = prelabels.get(pid)
        if pid in resolved or not candidate or candidate.get('image_sha256') != photo['sha256']:
            continue
        try:
            bbox = validate_box(candidate['bbox'], candidate['width'], candidate['height'])
        except (KeyError, ValueError, TypeError):
            continue
        resolved[pid] = dict(candidate, bbox=bbox, status='accepted', source='sam3', note='')
    return resolved


class SubjectBoxes:
    def __init__(self, store):
        self.store = store
        self.path = store.root / 'annotations/subject_boxes.json'
        self.prelabels_path = store.root / 'annotations/subject_box_prelabels.json'
        self.inventory = sha((store.root / 'lists/photos.csv').read_bytes())

    def read_prelabels(self):
        empty = dict(schema='release.subject_box_prelabels.v1', photos_sha256=self.inventory,
                     records={})
        if not self.prelabels_path.exists():
            return empty
        data = json.loads(self.prelabels_path.read_text())
        # Proposals from another inventory must never appear on the current photos.
        if data.get('schema') != empty['schema'] or data.get('photos_sha256') != self.inventory:
            return empty
        return data

    def proposal(self, photo_id, records, info=None):
        record = records.get(photo_id)
        if not record or record.get('image_sha256') != self.store.photos[photo_id]['sha256']:
            return None
        if info and any(record.get(k) != v for k, v in info.items()):
            return None
        try:
            box = validate_box(record['bbox'], record['width'], record['height'])
        except (KeyError, ValueError, TypeError):
            return None
        return dict(bbox=box, source=record.get('source', 'sam3'), confidence=record.get('confidence'))

    def save_prelabel(self, photo_id, bbox, **details):
        info = self.image_info(photo_id)
        bbox = validate_box(bbox, info['width'], info['height']) if bbox is not None else None
        with self.store.locked():
            data = self.read_prelabels()
            data['records'][photo_id] = dict(details, **info, bbox=bbox, source='sam3',
                updated_at=datetime.now(timezone.utc).isoformat())
            write_json(self.prelabels_path, data)

    def read(self):
        if not self.path.exists():
            return dict(schema='release.subject_boxes.v1', photos_sha256=self.inventory,
                        coordinate_system='EXIF-transposed original pixels; xyxy; exclusive right/bottom', records={})
        data = json.loads(self.path.read_text())
        if data.get('schema') != 'release.subject_boxes.v1' or data.get('photos_sha256') != self.inventory:
            raise ValueError('Subject boxes belong to a different dataset inventory')
        if set(data['records']) - set(self.store.photos):
            raise ValueError('Subject boxes contain unknown photos')
        return data

    @lru_cache(maxsize=2048)
    def image_info(self, photo_id):
        row = self.store.photos[photo_id]
        path = self.store.root / row['path']
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != row['sha256']:
            raise ValueError('Image checksum differs from the formal inventory')
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        return dict(photo_id=photo_id, package_id=row['package_id'], image_path=row['path'],
                    image_sha256=checksum, width=width, height=height)

    def get(self, photo_id):
        info = self.image_info(photo_id)
        with self.store.locked():
            record = self.read()['records'].get(photo_id)
            proposal = self.proposal(photo_id, self.read_prelabels()['records'], info)
        if record and any(record.get(k) != v for k, v in info.items()):
            raise ValueError('Saved region is stale for this image')
        return dict(**info, status=record['status'] if record else 'accepted' if proposal else 'pending',
                    bbox=record['bbox'] if record else proposal['bbox'] if proposal else None,
                    note=record.get('note', '') if record else '', proposal=proposal,
                    revision=box_revision(record))

    def summary(self):
        with self.store.locked():
            records = self.read()['records']
            prelabels = self.read_prelabels()['records']
        resolved = resolve_records(self.store.photos.values(), records, prelabels)
        states = {pid: resolved.get(pid, {}).get('status', 'pending') for pid in self.store.photos}
        regions = {pid: {k: row[k] for k in ('bbox', 'width', 'height', 'status', 'source')}
                   for pid, row in resolved.items()}
        return dict(total=len(states), counts={s: list(states.values()).count(s)
                    for s in ('pending', 'accepted', 'confirmed', 'unclear')}, states=states, regions=regions,
                    prelabels_done=sum(r.get('image_sha256') == self.store.photos.get(pid, {}).get('sha256')
                                       for pid, r in prelabels.items()))

    def update(self, photo_id, body):
        if not isinstance(body, dict) or set(body) != {'revision', 'status', 'bbox', 'note'}:
            raise ValueError('Expected revision, status, bbox and note')
        if body['status'] not in ('pending', 'confirmed', 'unclear'):
            raise ValueError('Invalid annotation status')
        if not isinstance(body['revision'], str) or not isinstance(body['note'], str) or len(body['note']) > 1000:
            raise ValueError('Invalid revision or note')
        info = self.image_info(photo_id)
        box = validate_box(body['bbox'], info['width'], info['height']) if body['status'] == 'confirmed' else None
        if body['status'] != 'confirmed' and body['bbox'] is not None:
            raise ValueError('Pending/unclear annotations must not contain a box')
        with self.store.locked():
            data = self.read()
            if box_revision(data['records'].get(photo_id)) != body['revision']:
                raise Conflict('This box changed in another window. Reopen the photo before saving.')
            data['records'][photo_id] = dict(**info, bbox=box, status=body['status'],
                note=body['note'].strip(), source='human', updated_at=datetime.now(timezone.utc).isoformat())
            write_json(self.path, data)
        return self.get(photo_id)
