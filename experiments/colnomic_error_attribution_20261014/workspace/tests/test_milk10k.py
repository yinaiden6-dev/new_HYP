import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from evaluation.milk10k import maxsim, metrics_and_predictions, test_pairs as load_test_pairs


class MilkTest(unittest.TestCase):
    def test_maxsim_padding_negative_and_asymmetry(self):
        q = [torch.tensor([[1., 0.], [0., 1.], [0., 0.]]), torch.tensor([[1., 0.]])]
        r = [torch.tensor([[-1., 0.], [0., 0.]]), torch.tensor([[1., 0.], [0., 1.]])]
        actual = maxsim(q, r, device='cpu', reference_batch=2)
        expected = torch.tensor([[-.5, 1.], [-1., 1.]])
        torch.testing.assert_close(actual, expected)
        reverse = maxsim(r, q, device='cpu')
        self.assertFalse(torch.equal(actual, reverse.T))

    def test_metrics_join_by_identity_and_stable_ties(self):
        scores = np.array([[.9, .9, .1], [.4, .3, .2], [.1, .2, .3]])
        metrics, rows = metrics_and_predictions(scores, ['q1', 'q2', 'q3'], ['r2', 'r1', 'r3'],
                                                ['a', 'b', 'c'], ['b', 'a', 'c'])
        self.assertEqual([r['rank'] for r in rows], [2, 1, 1])
        self.assertEqual(rows[0]['positive_id'], 'r1')
        self.assertAlmostEqual(metrics['MRR'], 5 / 6)
        self.assertEqual(metrics['correct_at_1'], 2)

    def test_manifest_uses_only_official_test_and_checks_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(2):
                path = root / 'MILK10k_Test_Input' / 'lesion1' / f'image{i}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), (i * 100, 30, 80)).save(path)
                paths.append(str(path.relative_to(root)))
            with (root / 'MILK10k_Test_Metadata.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['lesion_id', 'image_type', 'isic_id'])
                writer.writerows([['lesion1', 'clinical: close-up', 'image0'],
                                  ['lesion1', 'dermoscopic', 'image1']])
            def pairs(clinical):
                with (root / 'retrieval_pairs.csv').open('w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['split', 'lesion_id', 'clinical_image', 'dermoscopic_image'])
                    writer.writerow(['train', 'unused', 'missing.jpg', 'missing.jpg'])
                    writer.writerow(['test', 'lesion1', clinical, paths[1]])
            pairs(paths[0])
            self.assertEqual(len(load_test_pairs(root, expected_count=1)), 1)
            pairs(paths[0].replace('Test', 'Training'))
            with self.assertRaises(ValueError):
                load_test_pairs(root, expected_count=1)


if __name__ == '__main__':
    unittest.main()
