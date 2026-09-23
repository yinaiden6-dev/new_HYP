"""Synthetic contracts for constrained slope fitting and exact bias calibration."""
from pathlib import Path
import math
import struct
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf.h593_s_bias_competition_v1 import (
    MODES, MAX_FINITE, apply_head, calibrate_bias, first_crossing_bias,
    fit_head, gate_score, projected_gradient, surrogate_loss_gradient,
)


def bits(xs):
    return b"".join(struct.pack(">d", float(x)) for x in xs)


def row(m, h=0., d=0., delta=1):
    return {"m": m, "h_s": h, "d": d, "delta": delta}


class HeadTest(unittest.TestCase):
    def test_analytic_gradient_and_l2_bias(self):
        offsets = np.array([-2., -.4, 0., -1.])
        design = np.array([[.1, .3, 1.], [.8, 1.5, 1.], [0., 0., 1.], [1., .2, 1.]])
        targets = np.array([1., -1., 1., -1.])
        theta = np.array([.6, .2, -.3])
        loss, gradient = surrogate_loss_gradient(theta, offsets, design, targets)
        self.assertTrue(math.isfinite(loss))
        eps = 1e-6
        numeric = []
        for i in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric.append((surrogate_loss_gradient(plus, offsets, design, targets)[0] -
                            surrogate_loss_gradient(minus, offsets, design, targets)[0]) / (2 * eps))
        np.testing.assert_allclose(gradient, numeric, atol=1e-9, rtol=1e-7)

    def test_crossing_signed_extremes_and_subnormals(self):
        tiny = math.nextafter(0., 1.)
        for u in [0., -0., tiny, -tiny, 1., -1., 1e200, -1e200, MAX_FINITE,
                  math.nextafter(-MAX_FINITE, 0.)]:
            b = first_crossing_bias(u)
            self.assertIsNotNone(b)
            self.assertTrue(math.isfinite(b))
            self.assertGreater(float(u + b), 0.)
            self.assertLessEqual(float(u + math.nextafter(b, -math.inf)), 0.)
        self.assertIsNone(first_crossing_bias(-MAX_FINITE))
        self.assertEqual(first_crossing_bias(0.), tiny)

    def test_negative_bias_and_closest_zero_plateau(self):
        rows = [row(-1., h=2., delta=1), row(-1., h=1.2, delta=-1)]
        head = calibrate_bias(rows, 1., 0.)
        bad_u = float(-1. + 1.2)
        self.assertEqual(head["bias"], -bad_u)
        self.assertEqual(head["training_net_gain"], 1)
        self.assertEqual(head["training_changed_holds"], 1)
        self.assertGreater(gate_score(-1., 2., 0., head), 0.)
        self.assertEqual(gate_score(-1., 1.2, 0., head), 0.)

    def test_neutral_rows_count_for_ties(self):
        rows = [row(-1., delta=1), row(-2., delta=0), row(-3., delta=1), row(-3., delta=-1)]
        head = calibrate_bias(rows, 0., 0.)
        self.assertEqual(head["training_net_gain"], 1)
        self.assertEqual(head["training_changed_holds"], 1)
        self.assertEqual(head["bias"], math.nextafter(1., math.inf))
        intervals = head["calibration_certificate"]["intervals"]
        self.assertEqual([v["changed_holds"] for v in intervals], [0, 1, 2, 4])
        self.assertEqual([v["both_wrong"] for v in intervals], [0, 0, 1, 1])

    def test_grouped_ties_cannot_split_labels(self):
        head = calibrate_bias([row(-1., delta=1), row(-1., delta=-1)], 0., 0.)
        self.assertTrue(head["disabled"])
        self.assertEqual(head["training_changed_holds"], 0)
        self.assertEqual(head["calibration_certificate"]["distinct_crossings"], 1)

    def test_empty_no_positive_gain_and_unreachable(self):
        for rows in [[], [row(-1., delta=0)], [row(-1., delta=-1)],
                     [row(-MAX_FINITE, delta=1)], [row(.1, delta=1)]]:
            head = calibrate_bias(rows, 0., 0.)
            self.assertTrue(head["disabled"])
            self.assertEqual(head["training_net_gain"], 0)
        head = fit_head([row(-1., delta=0)], "S_GAP3")
        self.assertEqual(head["optimization"]["informative_hold_count"], 0)
        self.assertTrue(head["optimization"]["success"])

    def test_enumeration_matches_independent_bruteforce(self):
        rng = np.random.default_rng(831)
        for _ in range(20):
            rows = [row(-float(rng.uniform(0, 3)), float(rng.uniform(0, 1)),
                        float(rng.uniform(0, 4)), int(rng.integers(-1, 2))) for _ in range(25)]
            alpha, beta = .8, .17
            us = [float(float(r["m"] + float(alpha * r["h_s"])) + float(beta * r["d"])) for r in rows]
            candidates = {0.}
            for u in us:
                candidates.add(-u)
                candidates.add(math.nextafter(-u, math.inf))
            best_key = (0, 0, 0., 0.)
            best_bias = 0.
            for bias in candidates:
                switched = [r["delta"] for r, u in zip(rows, us) if float(u + bias) > 0]
                key = (sum(switched), -len(switched), -abs(bias), -bias)
                if key[0] > 0 and key > best_key:
                    best_key, best_bias = key, bias
            head = calibrate_bias(rows, alpha, beta)
            self.assertEqual(head["bias_hex"], best_bias.hex())
            self.assertEqual(head["training_net_gain"], best_key[0])
            self.assertEqual(head["training_changed_holds"], -best_key[1])

    def test_constrained_optimizer_and_projected_gradient(self):
        rows = [row(-.5, h=0., d=2., delta=1), row(-.5, h=1., d=0., delta=-1)] * 8
        for mode in MODES:
            head = fit_head(rows, mode)
            opt = head["optimization"]
            self.assertTrue(opt["success"], opt)
            self.assertLessEqual(opt["fitted_surrogate_loss"], opt["initial_surrogate_loss"])
            self.assertLess(opt["projected_gradient_inf_norm"], 1e-5)
            self.assertGreaterEqual(head["alpha"], 0.)
            self.assertGreaterEqual(head["beta"], 0.)
            self.assertEqual(head, fit_head(rows, mode))
        self.assertEqual(fit_head(rows, "S_BIAS2")["alpha"], 0.)
        self.assertGreater(fit_head(rows, "S_GAP3")["beta"], 0.)
        np.testing.assert_array_equal(projected_gradient([0, 2, 0], [2, -3, 4], 2), [0, -3, 4])

    def test_neutral_not_in_surrogate_existing_switch_locked(self):
        base = [row(-.5, h=.8, d=1., delta=1), row(-.2, h=.1, d=.2, delta=-1)]
        extra = base + [row(-.1, h=3., d=20., delta=0), row(.1, h=30., d=20., delta=1)]
        a, b = fit_head(base, "S_GAP3"), fit_head(extra, "S_GAP3")
        self.assertEqual(a["optimization"]["theta_hex"], b["optimization"]["theta_hex"])
        self.assertEqual(b["optimization"]["neutral_holds_omitted_from_surrogate"], 1)
        self.assertEqual(b["calibration_certificate"]["original_switch_count"], 1)

    def test_all_127_bits_protected_and_first_tie(self):
        head = {"alpha_hex": 1.0.hex(), "beta_hex": .5.hex(), "bias_hex": 0.0.hex(), "disabled": False}
        values = [-float(i + 1) for i in range(127)]
        values[2], values[100] = -.1, -.1
        switched = apply_head(values, .2, .1, head)
        self.assertGreater(switched[2], 0.)
        for i in range(127):
            if i != 2:
                self.assertEqual(bits([values[i]]), bits([switched[i]]))
        self.assertEqual(max(range(127), key=switched.__getitem__), 2)
        held = apply_head(values, .01, .01, head)
        self.assertEqual(bits(values), bits(held))
        values[10] = .2
        values[40] = -0.
        self.assertEqual(bits(values), bits(apply_head(values, 1., 1., head)))
        head["disabled"] = True
        values[10] = -.2
        self.assertEqual(bits(values), bits(apply_head(values, 1., 1., head)))

    def test_reject_invalid_nonfinite_and_overflow(self):
        with self.assertRaises(ValueError):
            fit_head([row(-1., h=-1)], "S_GAP3")
        with self.assertRaises(ValueError):
            calibrate_bias([row(float("nan"))], 0., 0.)
        with self.assertRaises(ValueError):
            calibrate_bias([row(-1., h=MAX_FINITE)], 2., 0.)
        with self.assertRaises(ValueError):
            fit_head([row(-1., delta=2)], "S_GAP3")
        with self.assertRaises(ValueError):
            fit_head([], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
