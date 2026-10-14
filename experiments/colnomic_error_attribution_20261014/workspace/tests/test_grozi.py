"""Protocol checks that guard against diagonal labels and OCR denominator errors."""
import unittest
import torch
from evaluation.grozi import metrics


class GroziMetricTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(
            photos=[dict(photo_id='q0', class_id=12, source_image_id=9,
                         positive_reference_index=2, positive_reference_id='r2'),
                    dict(photo_id='q1', class_id=10, source_image_id=9,
                         positive_reference_index=0, positive_reference_id='r0'),
                    dict(photo_id='q2', class_id=12, source_image_id=11,
                         positive_reference_index=2, positive_reference_id='r2')],
            references=[dict(reference_id=f'r{i}') for i in range(4)])
        self.scores = torch.tensor([[0., 1., 3., 2.], [3., 3., 0., 0.], [3., 2., 1., 0.]])

    def test_identity_mapping_and_stable_ties(self):
        result, predictions = metrics(self.scores, self.data)
        self.assertEqual([p['rank'] for p in predictions], [1, 1, 3])
        self.assertAlmostEqual(result['R@1'], 2/3)
        self.assertAlmostEqual(result['class_average']['R@1'], .75)
        self.assertEqual(result['confidence_intervals']['groups'], 2)

    def test_empty_ocr_counts_as_failure_in_full_denominator(self):
        result, predictions = metrics(self.scores, self.data, [False, True, True], [False, True, True, True])
        self.assertEqual([p['rank'] for p in predictions], [None, None, 3])
        self.assertEqual(result['queries'], 3)
        self.assertEqual(result['failures'], 2)
        self.assertAlmostEqual(result['R@5'], 1/3)
        self.assertAlmostEqual(result['MRR'], 1/9)


if __name__ == '__main__':
    unittest.main()
