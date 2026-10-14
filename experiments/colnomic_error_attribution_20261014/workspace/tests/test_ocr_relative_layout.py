"""Absolute placement is irrelevant; changes to pairwise layout must affect scores."""
import json
from pathlib import Path
import unittest

import numpy as np

from evaluation.ocr_relative_layout import extract_relative_lines, relative_layout_score, reference_idf
from tests.test_ocr_layout import record

SETTINGS = json.loads((Path(__file__).resolve().parents[1] / 'configs/bm25_relative_layout_fusion.json').read_text())


class RelativeLayoutTests(unittest.TestCase):
    def setUp(self):
        self.words = ['brand', 'medicine', 'quantity', 'formula', 'warning', 'revision']
        self.points = [(0, 0), (75, 10), (10, 45), (100, 60), (20, 110), (110, 150)]
        self.query = record(self.words, self.points)

    def score(self, reference):
        idf = reference_idf([reference])
        q = extract_relative_lines(self.query, idf, SETTINGS)
        r = extract_relative_lines(reference, idf, SETTINGS)
        return relative_layout_score(q, r, idf, SETTINGS), q, r

    def test_absolute_coordinates_are_discarded(self):
        result, q, r = self.score(record(self.words, self.points))
        for features in [q, r]:
            self.assertFalse(hasattr(features, 'centers'))
            self.assertFalse(hasattr(features, 'widths'))
            self.assertFalse(hasattr(features, 'directions'))
        np.testing.assert_allclose(q.relative_xy[0, 0], [1., 4.5])
        self.assertEqual(result, (1., 6, 30))

    def test_translation_and_canvas_size_have_no_effect(self):
        reference = record(self.words, self.points, lambda p: p + [21350., -9281.])
        reference.update(width=999999, height=700000)
        self.query.update(width=300, height=250)
        result, q, r = self.score(reference)
        self.assertAlmostEqual(result[0], 1.)
        np.testing.assert_allclose(q.relative_xy, r.relative_xy)

    def test_uniform_scale_and_rotation_preserve_relative_graph(self):
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reference = record(self.words, self.points, lambda p: p @ rotation.T * 3.7 + [800, 1500])
        result, q, r = self.score(reference)
        self.assertAlmostEqual(result[0], 1.)
        np.testing.assert_allclose(q.relative_xy, r.relative_xy, atol=1e-12)
        np.testing.assert_allclose(q.relative_log_size, r.relative_log_size, atol=1e-12)

    def test_same_words_with_reordered_positions_are_penalized(self):
        wrong = record(self.words, [self.points[i] for i in [5, 2, 4, 0, 3, 1]])
        result, _, _ = self.score(wrong)
        self.assertLess(result[0], .9)

    def test_changed_relative_spacing_is_penalized(self):
        wrong = record(self.words, [(x * 4, y) for x, y in self.points])
        result, _, _ = self.score(wrong)
        self.assertLess(result[0], .9)

    def test_mirrored_above_below_relations_are_penalized(self):
        wrong = record(self.words, [(x, -y) for x, y in self.points])
        result, _, _ = self.score(wrong)
        self.assertLess(result[0], .5)

    def test_extra_unseen_reference_panels_do_not_change_fragment_score(self):
        reference = record(self.words, self.points)
        extra = record(['unseen', 'panel'], [(6000, 4000), (6500, 4500)])
        reference['lines'] += extra['lines']; reference['text'] += '\n' + extra['text']
        result, _, _ = self.score(reference)
        self.assertAlmostEqual(result[0], 1.)

    def test_text_alone_without_three_line_geometry_supplies_no_bonus(self):
        reference = record(self.words, self.points)
        self.query['lines'] = self.query['lines'][:2]
        result, _, _ = self.score(reference)
        self.assertEqual(result, (0., 0, 0))


if __name__ == '__main__':
    unittest.main()
