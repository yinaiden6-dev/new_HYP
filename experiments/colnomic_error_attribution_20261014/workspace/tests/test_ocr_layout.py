"""Coordinates must distinguish reordered text and preserve local view transforms."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from evaluation.bm25_layout_fusion import apply_layout, candidate_rows
from evaluation.ocr_layout import extract_lines, layout_score, reference_idf

SETTINGS = json.loads((Path(__file__).resolve().parents[1] / 'configs/bm25_layout_fusion.json').read_text())


def record(words, centers, transform=None):
    lines = []
    for word, (x, y) in zip(words, centers, strict=True):
        p = np.array([[x-20, y-5], [x+20, y-5], [x+20, y+5], [x-20, y+5]], dtype=float)
        if transform is not None:
            p = transform(p)
        lines.append(dict(text=word, polygon=p.tolist()))
    return dict(text='\n'.join(words), lines=lines)


class LayoutTests(unittest.TestCase):
    def test_same_text_at_different_positions_is_distinguished(self):
        words = ['brand', 'medicine', '100', 'ml', 'formula', 'revision']
        points = [(0, 0), (75, 10), (10, 45), (100, 60), (20, 110), (110, 150)]
        q = record(words, points)
        correct = record(words, points)
        wrong = record(words, [points[i] for i in [5, 2, 4, 0, 3, 1]])
        idf = reference_idf([correct, wrong])
        query = extract_lines(q, idf)
        good, count = layout_score(query, extract_lines(correct, idf), idf, SETTINGS)
        bad, _ = layout_score(query, extract_lines(wrong, idf), idf, SETTINGS)
        self.assertAlmostEqual(good, 1.)
        self.assertEqual(count, 6)
        self.assertLess(bad, good - .1)

    def test_rotation_scale_translation_and_extra_reference_panels(self):
        words = ['brand', 'formula', 'quantity', 'warning']
        points = [(0, 0), (100, 10), (10, 70), (90, 110)]
        q = record(words, points)
        rotation = np.array([[0., -1.], [1., 0.]])
        transformed = record(words, points, lambda p: (p @ rotation.T) * 2.5 + [800, 300])
        extra = record(['unseen', 'panel'], [(1800, 1500), (1900, 1700)])
        transformed['lines'] += extra['lines']
        transformed['text'] += '\n' + extra['text']
        idf = reference_idf([transformed])
        score, count = layout_score(extract_lines(q, idf), extract_lines(transformed, idf), idf, SETTINGS)
        self.assertAlmostEqual(score, 1.)
        self.assertEqual(count, 4)

    def test_repeated_query_lines_cannot_reuse_one_reference_line(self):
        q = record(['brand'] * 4, [(0, 0), (0, 20), (0, 40), (0, 60)])
        r = record(['brand'], [(0, 0)])
        idf = reference_idf([r])
        self.assertEqual(layout_score(extract_lines(q, idf), extract_lines(r, idf), idf, SETTINGS), (0., 0))

    def test_missing_and_invalid_polygons_keep_base_scores(self):
        q = dict(text='brand', lines=[dict(text='brand', polygon=[[float('nan'), 0]] * 4)])
        r = dict(text='brand', lines=[])
        idf = reference_idf([r])
        self.assertEqual(layout_score(extract_lines(q, idf), extract_lines(r, idf), idf, SETTINGS), (0., 0))
        base = torch.tensor([[1., 2., 3.]])
        refs = [dict(sha256=str(i)) for i in range(3)]
        self.assertTrue(torch.equal(base, apply_layout(base, [[2, 1]], [[0., 0.]], refs, .5)))

    def test_byte_duplicates_receive_identical_bonus_and_candidates_are_unique(self):
        refs = [dict(sha256=s) for s in ['a', 'a', 'b', 'c']]
        base = torch.tensor([[5., 5., 4., 3.]])
        candidates, _ = candidate_rows(base, refs, 2)
        self.assertEqual(candidates, [[0, 2]])
        result = apply_layout(base, candidates, [[.6, .8]], refs, .5)
        torch.testing.assert_close(result, torch.tensor([[5.3, 5.3, 4.4, 3.]]))


if __name__ == '__main__':
    unittest.main()
