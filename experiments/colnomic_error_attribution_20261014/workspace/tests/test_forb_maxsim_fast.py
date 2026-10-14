import unittest

import torch
from torch.nn.functional import normalize

from evaluation.forb_maxsim_fast import MaxSimEngine, maxsim_scores
from evaluation.forb_scoring import maxsim_scores as original, verify_scores


class FastMaxSimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_negative_matches_zero_padding_and_reused_gallery(self):
        q = [torch.tensor([[1., 0.], [0., 1.], [0., 0.]]), torch.tensor([[1., 0.]])]
        r = [torch.tensor([[-1., 0.], [0., 0.]]), torch.eye(2)]
        for cached in (False, True):
            engine = MaxSimEngine(r, 'cpu', 2, 2, 1, cache_gallery=cached)
            actual = engine.score(q)
            torch.testing.assert_close(actual, torch.tensor([[-.5, 1.], [-1., 1.]]))
            torch.testing.assert_close(engine.score(q[1:]), actual[1:])
            self.assertTrue(verify_scores(q, r, actual, 'maxsim')['passed'])

    def test_all_tokens_ragged_shapes_batches_and_float32_storage(self):
        g = torch.Generator().manual_seed(8261)
        for dtype in (torch.float16, torch.float32):
            q = [normalize(torch.randn(n, 11, generator=g), dim=-1).to(dtype) for n in (1, 9, 33)]
            r = [normalize(torch.randn(n, 11, generator=g), dim=-1).to(dtype) for n in (2, 17, 4, 35)]
            q[1] = torch.cat((q[1], torch.zeros(3, 11, dtype=dtype)))
            r[0] = torch.cat((r[0], torch.zeros(2, 11, dtype=dtype)))
            before = [v.clone() for v in q + r]
            expected = original(q, r, 'cpu', 2, 3, 7)
            for qb, rb, tb in ((1, 1, 1), (2, 3, 8), (8, 8, 1024)):
                for cached in (False, True):
                    actual = MaxSimEngine(r, 'cpu', qb, rb, tb, cache_gallery=cached).score(q)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                    self.assertTrue(verify_scores(q, r, actual, 'maxsim')['passed'])
            for actual, saved in zip(q + r, before):
                self.assertTrue(torch.equal(actual, saved))

    def test_invalid_inputs_and_empty_inventories(self):
        valid = [torch.ones(2, 3)]
        for bad in (torch.zeros(3, 3), torch.full((2, 3), float('nan')), torch.ones(2, 4), torch.ones(3)):
            with self.assertRaises(ValueError):
                maxsim_scores(valid, [bad], 'cpu')
        with self.assertRaises(ValueError):
            MaxSimEngine(valid, 'cpu', token_block=0)
        self.assertEqual(tuple(maxsim_scores([], valid, 'cpu').shape), (0, 1))
        self.assertEqual(tuple(maxsim_scores(valid, [], 'cpu').shape), (1, 0))


if __name__ == '__main__':
    unittest.main()
