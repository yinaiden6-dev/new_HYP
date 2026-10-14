"""FORB offset, duplicate, AP denominator and missing OCR regression checks."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.forb_data import _build_inventory, metrics


class ForbInventoryTests(unittest.TestCase):
    def test_path_mapping_offsets_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'metadata/FORB_subset'
            folder.mkdir(parents=True)
            (root / 'provenance').mkdir()
            gallery = ['first/database/a.jpg', 'first/database/b.jpg',
                       'second/database/a.jpg', 'second/database/b.jpg']
            queries = ['first/query/q.jpg', 'second/query/q.jpg']
            (folder / 'all_database_images.txt').write_text('\n'.join(gallery) + '\n')
            with (root / 'provenance/image_files.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path', 'sha256'])
                writer.writeheader()
                # Distinct rows with identical bytes must remain distinct.
                writer.writerows(dict(path=p, sha256='a' * 64) for p in gallery + queries)
            for d, domain in enumerate(('first', 'second')):
                db = [dict(image=p) for p in gallery[d * 2:d * 2 + 2]]
                query = [dict(image=queries[d])]
                gt = [dict(query_id=1, query_image=queries[d],
                           groundtruth_ids=[1, 2] if d else [2],
                           groundtruth_images=gallery[2:] if d else [gallery[1]],
                           difficulty='medium')]
                for kind, rows in (('query', query), ('database', db), ('gt', gt)):
                    (folder / f'{domain}_{kind}.ndjson').write_text(
                        '\n'.join(json.dumps(r) for r in rows) + '\n')
            manifest = _build_inventory(root, ('first', 'second'), None)
            self.assertEqual(manifest['domain_offsets'], {'first': 0, 'second': 2})
            self.assertEqual([p['positive_indices'] for p in manifest['photos']], [[1], [2, 3]])
            self.assertEqual(len(manifest['references']), 4)
            self.assertEqual(len({r['id'] for r in manifest['references']}), 4)
            self.assertEqual(manifest['photos'][1]['positive_ids'],
                             [r['id'] for r in manifest['references'][2:]])
            path = folder / 'second_gt.ndjson'
            gt[0]['groundtruth_ids'] = [2, 1]
            path.write_text(json.dumps(gt[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'Positive path/index mismatch'):
                _build_inventory(root, ('first', 'second'), None)


class ForbMetricTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(photos=[
            dict(id='q0', domain='first', difficulty='easy', positive_indices=[1, 3]),
            dict(id='q1', domain='second', difficulty='hard', positive_indices=[0]),
            dict(id='q2', domain='second', difficulty='easy', positive_indices=[2, 4])],
            references=[dict(id=f'r{i}') for i in range(5)])
        self.scores = np.array([[1., 1., 1., 1., 1.],
                                [5., 4., 3., 2., 1.],
                                [1., 2., 5., 3., 4.]])

    def test_stable_ties_multiple_positives_and_group_denominators(self):
        report, predictions = metrics(self.scores, self.manifest)
        self.assertEqual([p['rank'] for p in predictions], [2, 1, 1])
        self.assertEqual(predictions[0]['top10_indices'], [0, 1, 2, 3, 4])
        self.assertEqual(predictions[0]['positive_ranks'], [2, 4])
        self.assertAlmostEqual(report['R@1'], 2 / 3)
        self.assertAlmostEqual(report['mAP@1'], .5)
        self.assertAlmostEqual(report['by_domain']['first']['mAP@5'], .5)
        self.assertEqual(report['by_difficulty']['easy']['queries'], 2)

    def test_ocr_failures_keep_full_query_and_positive_denominators(self):
        report, predictions = metrics(self.scores, self.manifest,
            query_valid=[False, True, True], reference_valid=[False, True, False, True, True])
        self.assertEqual(report['queries'], 3)
        self.assertEqual(report['failures'], 2)
        self.assertEqual([p['rank'] for p in predictions], [None, None, 2])
        self.assertEqual(predictions[0]['top10_indices'], [])
        self.assertEqual(predictions[0]['failure'], 'empty_query_ocr')
        self.assertEqual(predictions[1]['failure'], 'empty_positive_reference_ocr')
        self.assertEqual(predictions[2]['positive_ranks'], [None, 2])
        self.assertAlmostEqual(report['R@5'], 1 / 3)
        self.assertAlmostEqual(report['MRR'], 1 / 6)
        self.assertAlmostEqual(report['mAP@5'], 1 / 12)

    def test_matches_authors_metric_for_each_cutoff(self):
        source = Path(__file__).resolve().parents[1] / 'other_datasets/FORB/upstream/metric_helper/metrics.py'
        spec = importlib.util.spec_from_file_location('forb_official_metrics', source)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        report, _ = metrics(self.scores, self.manifest)
        groundtruth = [p['positive_indices'] for p in self.manifest['photos']]
        order = np.argsort(-self.scores, axis=1, kind='stable').tolist()
        for k in (1, 5, 10, 100):
            self.assertAlmostEqual(report[f'mAP@{k}'], official.get_map_at_k(groundtruth, order, k))

    def test_ap_at_100_divides_by_all_positives(self):
        manifest = dict(photos=[dict(id='q', domain='a', difficulty='easy',
                                    positive_indices=list(range(120)))],
                        references=[dict(id=str(i)) for i in range(120)])
        report, _ = metrics(np.zeros((1, 120)), manifest)
        self.assertAlmostEqual(report['mAP@1'], 1 / 120)
        self.assertAlmostEqual(report['mAP@100'], 100 / 120)
        self.assertEqual(report['mAP'], 1.)

    def test_rejects_bad_score_shapes_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            metrics(np.zeros((3, 4)), self.manifest)
        self.scores[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            metrics(self.scores, self.manifest)


if __name__ == '__main__':
    unittest.main()
