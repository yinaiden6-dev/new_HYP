"""Persistence and evaluation checks using an isolated, synthetic dataset."""
import csv
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/tools'))
from photo_metadata import Conflict, PhotoStore, write_json
from review_photos import ReviewServer
from type_metrics import evaluate


def csv_file(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


class PhotoReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'dataset'
        rows, packages, groups = [], [], []
        for n, count in [(1, 3), (2, 1)]:
            package_id = f'package_{n:03d}'
            directory = 'packages/' + package_id
            photos = []
            for i in range(count):
                photo_id = f'photo_{n}_{i}'
                relative = 'photos/' + photo_id + '.jpg'
                path = self.root / directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic fixture; not a real image')
                rows.append(dict(photo_id=photo_id, path=directory + '/' + relative,
                                 package_id=package_id, sha256='fixture'))
                photos.append(dict(image=relative, type=dict(placement='tabletop', background='single_object'),
                                   reviewed=False, excluded=False))
            metadata = directory + '/metadata.json'
            write_json(self.root / metadata, dict(package_id=package_id, name=package_id,
                       category='human/otc', documents=[], photos=photos))
            packages.append(dict(package_id=package_id, directory=directory,
                                 metadata_path=metadata, design_path=directory + '/design.jpg'))
            groups.append(dict(package_id=package_id, group_id=str(n), split='train' if n == 1 else 'test'))
        csv_file(self.root / 'lists/packages.csv', list(packages[0]), packages)
        csv_file(self.root / 'lists/groups.csv', list(groups[0]), groups)
        for filename, values in [('photos.csv', rows), ('train.csv', rows[:3]), ('val.csv', []),
                                 ('test.csv', rows[3:]), ('train_subsets/25.csv', rows[:1]),
                                 ('train_subsets/50.csv', rows[:2])]:
            csv_file(self.root / 'lists' / filename, list(rows[0]), values)
        csv_file(self.root / 'lists/references.csv', ['reference_id', 'sha256'],
                 [dict(reference_id=str(i), sha256=str(i)) for i in range(10)])
        write_json(self.root / 'lists/protocol.json', dict(version='synthetic-test-only'))
        self.store = PhotoStore(self.root)
        self.store.export_active()

    def tearDown(self):
        self.temp.cleanup()

    def change(self, **overrides):
        return dict(dict(photo_id='photo_1_0', type=dict(placement='handheld', background='multi_object'),
                         reviewed=True, excluded=False), **overrides)

    def test_remove_restore_preserves_original_files_and_splits(self):
        original = (self.root / 'lists/photos.csv').read_bytes()
        revision = self.store.package('package_001')[1]
        self.store.update('package_001', revision, [self.change(excluded=True)])
        manifest = self.store.verify_active()
        self.assertEqual(manifest['counts']['photos.csv'], 3)
        self.assertEqual(manifest['counts']['train.csv'], 2)
        self.assertEqual(manifest['counts']['test.csv'], 1)
        self.assertTrue((self.root / 'packages/package_001/photos/photo_1_0.jpg').exists())
        self.assertEqual((self.root / 'lists/photos.csv').read_bytes(), original)
        self.store.update('package_001', self.store.package('package_001')[1], [self.change()])
        self.assertEqual(self.store.verify_active()['counts']['photos.csv'], 4)

    def test_conflicts_and_invalid_changes_do_not_overwrite(self):
        revision = self.store.package('package_001')[1]
        self.store.update('package_001', revision, [self.change()])
        with self.assertRaises(Conflict):
            self.store.update('package_001', revision, [self.change(excluded=True)])
        before = self.store.package('package_001')[1]
        for bad in [self.change(reviewed='true'), self.change(photo_id='photo_2_0'),
                    self.change(type=dict(placement='invented', background='single_object'))]:
            with self.assertRaises(ValueError):
                self.store.update('package_001', before, [bad])
            self.assertEqual(self.store.package('package_001')[1], before)
        self.store.update('package_001', None, [self.change(excluded=True)], actor='model')
        self.assertEqual(self.store.package('package_001')[1], before)

    def test_corrupt_active_list_is_rejected(self):
        path = self.root / 'lists/active/train.csv'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.store.verify_active()

    def test_confirm_unchanged_prelabels_and_reopen(self):
        before = self.store.records()['photo_1_0']
        self.store.update('package_001', self.store.package('package_001')[1], [self.change(
            type=before['type'], reviewed=True)])
        reopened = PhotoStore(self.root)
        self.assertEqual(reopened.records()['photo_1_0'], dict(before, reviewed=True))
        self.assertEqual(reopened.verify_active()['summary']['reviewed'], 1)
        reopened.update('package_001', None, [self.change()], actor='model')
        self.assertEqual(reopened.records()['photo_1_0']['type'], before['type'])

    def test_occlusion_round_trip_and_older_editor_preserves_it(self):
        self.store.update('package_001', None, [self.change(
            type=dict(placement='tabletop', background='single_object', occlusion='foreground'))])
        reopened = PhotoStore(self.root)
        row = reopened.records()['photo_1_0']
        self.assertEqual(row['type']['occlusion'], 'foreground')
        self.assertEqual(row['type']['background'], 'single_object')
        self.assertEqual(reopened.verify_active()['summary']['occlusion_labeled'], 1)
        # A tab opened before the upgrade sends only the original two fields.
        reopened.update('package_001', None, [self.change(reviewed=False)])
        self.assertEqual(reopened.records()['photo_1_0']['type']['occlusion'], 'foreground')
        reopened.update('package_001', None, [self.change()], actor='model')
        self.assertEqual(reopened.records()['photo_1_0']['type']['occlusion'], 'foreground')
        reopened.update('package_001', None, [self.change(
            type=dict(placement='handheld', background='multi_object', occlusion=None))])
        manifest = reopened.verify_active()
        self.assertEqual(manifest['summary']['labeled'], 4)
        self.assertEqual(manifest['summary']['occlusion_labeled'], 0)
        report = evaluate(reopened, [dict(photo_id=pid, rank=1) for pid in reopened.photos])
        self.assertEqual(report['overall']['photos'], 4)

    def test_invalid_occlusion_rejected_without_modifying_metadata(self):
        revision = self.store.package('package_001')[1]
        for value in ['invented', True, 1]:
            with self.assertRaisesRegex(ValueError, 'Invalid occlusion'):
                self.store.update('package_001', revision, [self.change(type=dict(
                    placement='handheld', background='multi_object', occlusion=value))])
            self.assertEqual(self.store.package('package_001')[1], revision)

    def test_review_scope_uses_supplement_receipt(self):
        receipt = 'review/supplements/fixture/import.json'
        ids = ['photo_1_1', 'photo_2_0']
        write_json(self.root / receipt, dict(photo_ids=ids))
        write_json(self.root / 'release.json', dict(supplement=receipt))
        server = ReviewServer(('127.0.0.1', 0), self.store)
        try:
            self.assertEqual(server.review_scopes()[1]['photo_ids'], ids)
            write_json(self.root / receipt, dict(photo_ids=['missing_photo']))
            with self.assertRaisesRegex(ValueError, 'current inventory'):
                server.review_scopes()
            write_json(self.root / 'release.json', dict(supplement='../outside.json'))
            with self.assertRaisesRegex(ValueError, 'inside dataset/review'):
                server.review_scopes()
        finally:
            server.server_close()

    def test_rank_denominators_and_default_labels(self):
        predictions = [dict(photo_id=pid, rank=rank) for pid, rank in
                       zip(self.store.photos, [1, 7, None, 1])]
        with self.assertRaisesRegex(ValueError, 'human review'):
            evaluate(self.store, predictions, 'photos', allow_prelabels=False)
        report = evaluate(self.store, predictions, 'photos')
        self.assertEqual(report['label_status'], 'model_labels_with_optional_corrections')
        self.assertEqual(report['reviewed_photos'], 0)
        self.assertEqual(report['overall']['recall_at_1'], .5)
        self.assertEqual(report['overall']['recall_at_10'], .75)
        self.assertAlmostEqual(report['overall']['mrr'], (2 + 1 / 7) / 4)
        self.assertAlmostEqual(report['overall']['package_macro']['recall_at_1'], 2 / 3)
        for bad in [predictions[:-1], predictions + [predictions[0]],
                    [dict(predictions[0], rank=True)] + predictions[1:]]:
            with self.assertRaises(ValueError):
                evaluate(self.store, bad, 'photos', allow_prelabels=True)
        self.store.update('package_001', None, [self.change(excluded=True)])
        report = evaluate(self.store, predictions, 'photos', allow_prelabels=True)
        self.assertEqual(report['overall']['photos'], 4)
        self.assertEqual(report['ignored_excluded_predictions'], 0)
        self.assertAlmostEqual(report['overall']['recall_at_1'], .5)

    def test_http_editor_session_and_origin(self):
        server = ReviewServer(('127.0.0.1', 0), self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url + '/api/state') as response:
                state = json.load(response)
            self.assertEqual(state['review_scopes'], [dict(id='all', name='全部照片',
                             photo_ids=list(self.store.photos))])
            body = json.dumps(dict(revision=self.store.package('package_001')[1],
                                   changes=[self.change()])).encode()
            headers = {'Content-Type': 'application/json'}
            for extra in [{}, {'X-Review-Token': state['token'], 'Origin': 'https://example.org'}]:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(url + '/api/package/package_001',
                                           data=body, headers=dict(headers, **extra)))
                self.assertEqual(caught.exception.code, 403)
            with urllib.request.urlopen(urllib.request.Request(url + '/api/package/package_001',
                    data=body, headers=dict(headers, **{'X-Review-Token': state['token']}))) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
