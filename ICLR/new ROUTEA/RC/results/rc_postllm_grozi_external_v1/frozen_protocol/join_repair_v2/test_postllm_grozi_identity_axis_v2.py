"""Regression for physical-gallery versus identity-ranking semantics."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'programs'))
from join_rc_postllm_grozi_external_v2 import validate_identity_axis


class IdentityAxis(unittest.TestCase):
    def setUp(self):
        self.labels = [f'id{i}' for i in range(130)] + ['id0']
        self.scores = np.arange(131, dtype=float)
        self.scores[130] = -1
        self.order = list(reversed(range(130)))
        self.axis = sorted(self.order[:128])

    def test_original_identity_deduplication_is_accepted(self):
        validate_identity_axis(self.order, self.axis, self.labels, self.scores)

    def test_physical_permutation_and_missing_identity_are_rejected(self):
        for order in (self.order + [130], self.order[:-1], self.order[:-1] + [129]):
            with self.assertRaises(RuntimeError):
                validate_identity_axis(order, self.axis, self.labels, self.scores)

    def test_wrong_representative_and_wrong_order_are_rejected(self):
        for order in (self.order[:-1] + [130], self.order[1:2] + self.order[:1] + self.order[2:]):
            with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_IDENTITY_RANK'):
                validate_identity_axis(order, sorted(order[:128]), self.labels, self.scores)

    def test_tie_keeps_earlier_physical_row(self):
        scores = self.scores.copy(); scores[130] = scores[0]
        validate_identity_axis(self.order, self.axis, self.labels, scores)
        with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_IDENTITY_RANK'):
            validate_identity_axis(self.order[:-1] + [130], self.axis, self.labels, scores)

    def test_candidate_insertion_and_nonfinite_score_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'NATURAL_C128'):
            validate_identity_axis(self.order, sorted(self.order[-128:]), self.labels, self.scores)
        scores = self.scores.copy(); scores[0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'FULL_PHYSICAL_SCORES'):
            validate_identity_axis(self.order, self.axis, self.labels, scores)


if __name__ == '__main__':
    unittest.main()
