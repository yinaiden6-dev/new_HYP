"""Numerical and checkpoint qualifications for the full-data POST runner."""
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'programs'))
import run_rc_postllm_h593_v1 as R


class Qualification(unittest.TestCase):
    def test_complete_action_cost_and_tie_gradient_parity(self):
        generator = torch.Generator().manual_seed(7)
        rows = [dict(winner_index=0, target_positions=[t], challenger_positions=list(range(1, 128)))
                for t in [0, 1, 64, 127] * 3]
        for d in [3, 4, 5]:
            x = torch.randn(len(rows), 127, d, dtype=torch.float64, generator=generator)
            self.assertEqual(len(R.verify_batched_cost(rows, x)), 2)

    def test_rng_checkpoint_restores_python_numpy_torch(self):
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        expected_state = R.rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(5))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'rng.pt'
            R.save(p, expected_state)
            restored = torch.load(p, map_location='cpu', weights_only=True)
        R.restore_rng(restored)
        actual = (random.random(), float(np.random.rand()), torch.rand(5))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_fresh_zero_adapter_shared_initialization(self):
        ctx = dict(config=dict(seed=17, bottleneck=16, condition_gain=3584**.5, residual_scale=.1),
                   normalization=dict(log_mean=-3., log_std=1.5, epsilon=1e-8))
        modules = [R.adapter(ctx, arm, 'cpu') for arm in R.ARMS]
        self.assertTrue(all(R.V.tree_equal(modules[0].state_dict(), m.state_dict()) for m in modules[1:]))
        h = torch.randn(5, 3584).to(torch.bfloat16)
        for m in modules:
            for mass in [0., .1, 1.]:
                self.assertTrue(torch.equal(m(h, mass), h))

    def test_fixed_candidate_mass_shift_and_constant_control(self):
        row = dict(M=[i / 128 for i in range(128)])
        original = list(row['M'])
        self.assertEqual(R.row_condition(row, 'POST_REAL'), (original, False))
        self.assertEqual(R.row_condition(row, 'POST_SHUFFLED'), (original[1:]+original[:1], False))
        self.assertEqual(R.row_condition(row, 'POST_REAL', 'shuffled'), (original[1:]+original[:1], False))
        self.assertEqual(R.row_condition(row, 'POST_REAL', 'constant'), (original, True))
        self.assertEqual(row['M'], original)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
