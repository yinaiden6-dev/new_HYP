"""Region coordinates, concurrent persistence and original-image crop checks."""
import hashlib
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
import test_photo_review as fixture
from photo_metadata import Conflict, PhotoStore, write_json
from review_photos import ReviewServer
from subject_boxes import SubjectBoxes, validate_box, resolve_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.regions import open_photo, crop_region


class SubjectBoxTest(unittest.TestCase):
    def setUp(self):
        fixture.PhotoReviewTest.setUp(self)
        rows = list(self.store.photos.values())
        for row in rows:
            path = self.root / row['path']
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (80, 120), '#337799').save(path, exif=exif)
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixture.csv_file(self.root / 'lists/photos.csv', list(rows[0]), rows)
        self.store = PhotoStore(self.root)
        self.store.export_active()
        self.boxes = SubjectBoxes(self.store)

    tearDown = fixture.PhotoReviewTest.tearDown

    def body(self, pid='photo_1_0', **updates):
        return dict(dict(revision=self.boxes.get(pid)['revision'], status='confirmed',
                         bbox=[10, 20, 100, 60], note='target package'), **updates)

    def test_box_persistence_is_independent_and_oriented(self):
        metadata = self.store.package('package_001')[1]
        original = (self.root / 'lists/photos.csv').read_bytes()
        self.assertEqual((self.boxes.get('photo_1_0')['width'], self.boxes.get('photo_1_0')['height']), (120, 80))
        body = self.body()
        saved = self.boxes.update('photo_1_0', body)
        self.assertEqual(saved['bbox'], body['bbox'])
        self.assertEqual(SubjectBoxes(self.store).get('photo_1_0'), saved)
        with self.assertRaises(Conflict):
            self.boxes.update('photo_1_0', body)
        self.boxes.update('photo_2_0', self.body('photo_2_0'))
        self.assertEqual(self.boxes.summary()['counts']['confirmed'], 2)
        self.assertEqual(self.store.package('package_001')[1], metadata)
        self.assertEqual((self.root / 'lists/photos.csv').read_bytes(), original)

    def test_invalid_boxes_and_explicit_clear(self):
        for box in [[1,2,3], [10,10,9,12], [0,0,121,80], [0,0,float('nan'),20],
                    [0,0,float('inf'),20], [False,0,10,20], [1,1,1.5,10]]:
            with self.assertRaises(ValueError):
                self.boxes.update('photo_1_0', self.body(bbox=box))
        self.assertFalse(self.boxes.path.exists())
        self.boxes.update('photo_1_0', self.body())
        self.boxes.update('photo_1_0', self.body(status='unclear', bbox=None))
        self.assertIsNone(self.boxes.get('photo_1_0')['bbox'])
        self.boxes.update('photo_1_0', self.body(status='pending', bbox=None))
        self.assertEqual(self.boxes.summary()['counts']['pending'], 4)

    def test_prelabels_do_not_confirm_or_overwrite_human_work(self):
        metadata = self.store.package('package_001')[1]
        self.boxes.save_prelabel('photo_1_0', [5, 10, 110, 70], confidence=0.9)
        initial = self.boxes.get('photo_1_0')
        self.assertEqual(initial['status'], 'accepted')
        self.assertEqual(initial['bbox'], [5, 10, 110, 70])
        self.assertEqual(self.boxes.summary()['counts']['accepted'], 1)
        self.assertEqual(self.boxes.summary()['regions']['photo_1_0']['bbox'], [5, 10, 110, 70])
        self.assertFalse(self.boxes.path.exists())
        self.boxes.update('photo_1_0', self.body())
        human_bytes = self.boxes.path.read_bytes()
        self.boxes.save_prelabel('photo_1_0', [0, 0, 90, 65], confidence=0.95)
        self.assertEqual(self.boxes.get('photo_1_0')['bbox'], [10, 20, 100, 60])
        self.assertEqual(self.boxes.path.read_bytes(), human_bytes)
        self.boxes.update('photo_1_0', self.body(status='pending', bbox=None))
        self.assertIsNone(self.boxes.get('photo_1_0')['bbox'])
        self.assertEqual(self.boxes.get('photo_1_0')['proposal']['bbox'], [0, 0, 90, 65])
        self.assertEqual(self.store.package('package_001')[1], metadata)

    def test_effective_defaults_preserve_explicit_human_decisions(self):
        for pid in self.store.photos:
            self.boxes.save_prelabel(pid, [5, 10, 110, 70])
        self.boxes.update('photo_1_0', self.body())
        self.boxes.update('photo_1_1', self.body('photo_1_1', status='pending', bbox=None))
        self.boxes.update('photo_1_2', self.body('photo_1_2', status='unclear', bbox=None))
        resolved = resolve_records(self.store.photos.values(), self.boxes.read()['records'],
                                   self.boxes.read_prelabels()['records'])
        self.assertEqual(resolved['photo_2_0']['status'], 'accepted')
        self.assertEqual(resolved['photo_2_0']['source'], 'sam3')
        self.assertEqual(resolved['photo_1_0']['bbox'], [10, 20, 100, 60])
        self.assertEqual(resolved['photo_1_1']['status'], 'pending')
        self.assertIsNone(resolved['photo_1_1']['bbox'])
        self.assertEqual(resolved['photo_1_2']['status'], 'unclear')

    def test_missing_stale_and_invalid_prelabels_stay_pending(self):
        self.boxes.save_prelabel('photo_1_0', None, fallback_reason='no_detection')
        self.assertEqual(self.boxes.get('photo_1_0')['status'], 'pending')
        self.assertEqual(self.boxes.summary()['prelabels_done'], 1)
        self.boxes.save_prelabel('photo_1_0', [5, 10, 110, 70])
        valid = self.boxes.read_prelabels()
        for key, value in [('image_sha256', 'stale'), ('width', 121), ('bbox', [0, 0, 500, 80])]:
            data = json.loads(json.dumps(valid))
            data['records']['photo_1_0'][key] = value
            write_json(self.boxes.prelabels_path, data)
            self.assertEqual(self.boxes.get('photo_1_0')['status'], 'pending')
            self.assertIsNone(self.boxes.get('photo_1_0')['proposal'])
        valid['photos_sha256'] = 'old inventory'
        write_json(self.boxes.prelabels_path, valid)
        self.assertEqual(self.boxes.summary()['counts']['pending'], 4)

    def test_native_crop_and_white_background(self):
        path = self.root / self.store.photos['photo_1_0']['path']
        image = open_photo(path)
        self.assertEqual(image.size, (120,80))
        mask = Image.new('L', image.size)
        mask.paste(255, (20,20,60,60))
        crop, bounds = crop_region(image, [20,20,60,60], mask=mask)
        self.assertEqual(bounds, [18,18,62,62])
        self.assertEqual(crop.size, (44,44))
        self.assertEqual(crop.getpixel((0,0)), (255,255,255))
        self.assertEqual(crop.getpixel((5,5)), image.getpixel((23,23)))
        with self.assertRaises(ValueError):
            crop_region(image, [20,20,60,60], mask=Image.new('L',(60,40)))

    def test_box_http_conflict_and_session(self):
        server = ReviewServer(('127.0.0.1',0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        body = json.dumps(self.body()).encode()
        try:
            for headers, expected in [({},403), ({'X-Review-Token':server.token},200),
                                       ({'X-Review-Token':server.token},409)]:
                request = urllib.request.Request(url+'/api/box/photo_1_0', data=body, headers=headers)
                try:
                    with urllib.request.urlopen(request) as response:
                        self.assertEqual(response.status,expected)
                except urllib.error.HTTPError as error:
                    self.assertEqual(error.code,expected)
            with urllib.request.urlopen(url+'/api/box/photo_1_0') as response:
                self.assertEqual(json.load(response)['bbox'], [10,20,100,60])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        restarted = ReviewServer(('127.0.0.1', 0), self.store)
        try:
            self.assertEqual(restarted.token, server.token)
            self.assertEqual((self.store.review / '.session_token').stat().st_mode & 0o777, 0o600)
        finally:
            restarted.server_close()
