"""Boundary and independent finite-pattern checks, without natural query data."""

import math
import random
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rc_aslo_xf.h593_hold_net_gain_v1 import (  # noqa: E402
    MAX_FINITE_BITS, apply_hold_lift, first_crossing_alpha, fit_hold_net_gain,
)


def bits(x):
    return struct.pack(">d", x)


def direct_score(rows, alpha):
    switched = [r for r in rows if r["m"] <= 0 and r["m"] + alpha * r["h"] > 0]
    return sum(r["delta"] for r in switched), len(switched)


class HoldNetGainTests(unittest.TestCase):
    def test_rounding_boundaries_and_unreachable(self):
        tiny = math.ulp(0.0)
        cases = [(0.0, 1.0), (-0.0, .5), (-tiny, 1.0), (-tiny, .5),
                 (-1.0, .3), (-1e250, .8), (-1e-250, 1e-50)]
        for m, h in cases:
            a = first_crossing_alpha(m, h)
            self.assertIsNotNone(a)
            self.assertGreater(m + a * h, 0)
            self.assertLessEqual(m + math.nextafter(a, 0.0) * h, 0)
        self.assertEqual(first_crossing_alpha(-0.0, 1.0), tiny)
        self.assertEqual(first_crossing_alpha(0.0, .5), 2 * tiny)
        self.assertIsNone(first_crossing_alpha(-1.0, 0.0))
        self.assertIsNone(first_crossing_alpha(-sys.float_info.max, 1.0))
        self.assertIsNone(first_crossing_alpha(-1.0, tiny))

    def test_full_vector_identity_and_original_switch_lock(self):
        logits = (-0.0, -1.0, -2.0) + (-3.0,) * 124
        self.assertEqual([bits(v) for v in apply_hold_lift(logits, 1.0, 0.0)],
                         [bits(v) for v in logits])
        switched = (.01,) + logits[1:]
        self.assertEqual(apply_hold_lift(switched, 1, sys.float_info.max), switched)
        got = apply_hold_lift(logits, 1.0, 1.0)
        self.assertEqual(got[0], 1.0)
        self.assertEqual([bits(v) for v in got[1:]], [bits(v) for v in logits[1:]])
        tied = (-1.0, -1.0, -3.0)
        self.assertEqual(apply_hold_lift(tied, 1.0, 2.0), (1.0, -1.0, -3.0))

    def test_conflicts_neutral_and_safe_constraint(self):
        rows = [dict(m=-1, h=1, delta=1), dict(m=-1, h=1, delta=-1)]
        self.assertEqual(fit_hold_net_gain(rows)["alpha_hex"], 0.0.hex())
        rows += [dict(m=-2, h=1, delta=1), dict(m=-2, h=1, delta=0)]
        fit = fit_hold_net_gain(rows)
        self.assertEqual((fit["training_net_gain"], fit["training_changed_holds"]), (1, 4))
        self.assertEqual(fit["training_both_wrong"], 1)
        self.assertEqual(fit_hold_net_gain(rows, zero_break=True)["alpha"], 0.0)
        tie_rows = [dict(m=-1, h=1, delta=1), dict(m=-2, h=1, delta=0)]
        self.assertEqual(fit_hold_net_gain(tie_rows)["training_changed_holds"], 1)

    def test_independent_dense_breakpoint_enumeration(self):
        rng = random.Random(8671)
        for _ in range(20):
            # Integer powers of two yield analytically known crossing locations.
            rows = [dict(m=-float(rng.randrange(0, 12)), h=2.0 ** -rng.randrange(5),
                         delta=rng.choice((-1, 0, 1))) for _ in range(24)]
            rows.append(dict(m=.5, h=1.0, delta=-1))
            alphas = {0.0}
            for r in rows:
                if r["m"] <= 0:
                    boundary = -r["m"] / r["h"]
                    # Include neighboring floats to account for subnormal rounding.
                    x = boundary
                    for _j in range(33):
                        alphas.add(x)
                        x = math.nextafter(x, math.inf)
            def independent_key(alpha):
                gain, changed = direct_score(rows, alpha)
                return gain, -changed, -alpha

            best = max(alphas, key=independent_key)
            fit = fit_hold_net_gain(rows)
            self.assertEqual(fit["alpha"], best)
            self.assertEqual(fit["training_net_gain"], direct_score(rows, best)[0])
            for cert in fit["certificates"]:
                if cert["status"] == "crossing":
                    self.assertLessEqual(float.fromhex(cert["predecessor_logit_hex"]), 0)
                    self.assertGreater(float.fromhex(cert["crossing_logit_hex"]), 0)

    def test_empty_invalid_and_maximal_finite(self):
        self.assertEqual(fit_hold_net_gain([])["alpha"], 0.0)
        maxval = struct.unpack(">d", struct.pack(">Q", MAX_FINITE_BITS))[0]
        prev = math.nextafter(maxval, 0.0)
        self.assertEqual(first_crossing_alpha(-prev, 1.0), maxval)
        self.assertTrue(math.isfinite(apply_hold_lift((-prev,), 1.0, maxval)[0]))
        for h in (-1, 2, math.nan):
            with self.assertRaises(ValueError):
                first_crossing_alpha(-1, h)
        with self.assertRaises(ValueError):
            apply_hold_lift((-1,), 1, math.inf)
        with self.assertRaises(ValueError):
            fit_hold_net_gain([dict(m=-1, h=1, delta=2)])


if __name__ == "__main__":
    unittest.main()
