"""Top-1 mask selection must stay identical while reducing full-size allocation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Sam3ImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/tools'))
from prelabel_subject_boxes import select_best_mask


class PrelabelMemoryTest(unittest.TestCase):
    def test_top_mask_matches_full_postprocessing(self):
        processor = Sam3ImageProcessor()
        generator = torch.Generator().manual_seed(42)
        for logits, threshold in [([1., 3., 2.], .2), ([3., 3., 1.], .2), ([-10., -9., -8.], .2)]:
            with self.subTest(logits=logits):
                output = SimpleNamespace(pred_logits=torch.tensor([logits]),
                    pred_boxes=torch.rand(1, 3, 4, generator=generator),
                    pred_masks=torch.randn(1, 3, 8, 8, generator=generator),
                    presence_logits=torch.tensor([[1.]]))
                settings = dict(threshold=threshold, mask_threshold=.5, target_sizes=[(48, 64)])
                full = processor.post_process_instance_segmentation(output, **settings)[0]
                selected, details = select_best_mask(output, threshold)
                reduced = processor.post_process_instance_segmentation(selected, **settings)[0]
                self.assertEqual(selected.pred_masks.shape[1], 1)
                self.assertEqual(details['candidates'], len(full['scores']))
                if len(full['scores']):
                    best = torch.argsort(full['scores'], descending=True, stable=True)[0]
                    self.assertTrue(torch.equal(full['masks'][best], reduced['masks'][0]))
                    self.assertEqual(full['scores'][best], reduced['scores'][0])
                    self.assertEqual(details['selected_index'], int(best))
                else:
                    self.assertEqual(len(reduced['scores']), 0)
