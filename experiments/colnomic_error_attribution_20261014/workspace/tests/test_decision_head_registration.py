"""Decision-only results must not fabricate full-gallery rank metrics."""
import unittest
from evaluation.content_box_ce_polish18 import decision_summary


class DecisionHeadRegistrationTest(unittest.TestCase):
    def test_decisions_are_counted_independently_of_unknown_error_ranks(self):
        rows=[dict(package_id='a',group_id='g1',correct=True,rank=1),
              dict(package_id='a',group_id='g1',correct=False,rank=None),
              dict(package_id='b',group_id='g2',correct=True,rank=1)]
        result=decision_summary(rows)
        self.assertEqual(result['correct'],2);self.assertEqual(result['incorrect'],1)
        self.assertEqual(result['failures'],0)
        self.assertEqual(result['photo_average']['recall_at_1'],2/3)
        self.assertEqual(result['package_average']['recall_at_1'],.75)
        for key in ['recall_at_5','recall_at_10','mrr']:
            self.assertIsNone(result['photo_average'][key]);self.assertIsNone(result['package_average'][key])

    def test_empty_scene_is_unavailable_instead_of_zero_accuracy(self):
        result=decision_summary([])
        self.assertEqual(result['queries'],0);self.assertEqual(result['correct'],0)
        self.assertTrue(all(value is None for value in result['photo_average'].values()))


if __name__=='__main__':unittest.main()
