"""Synthetic tests only: conditional interaction fit and exact bias semantics."""
from pathlib import Path
import math
import struct
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf import h593_raw_incumbent_gate_v1 as old
from rc_aslo_xf.h593_conditional_gap_v1 import (
    MODES, apply_head, calibrate_bias, feature_value, fit_head, fit_scale,
    gate_score, projected_gradient, surrogate_loss_gradient,
)


def bits(xs):
    return b"".join(struct.pack(">d", float(x)) for x in xs)


def row(m, r=-.5, d=1., delta=1):
    return {"m": m, "h_raw": r, "d": d, "delta": delta}


def manual(mode="GAP_RAW_INTERACT4", gamma=.2, beta=.7, eta=.3,
           bias=.1, scale=1., disabled=False):
    return {"mode": mode, "gamma_hex": float(gamma).hex(),
            "beta_hex": float(beta).hex(), "eta_hex": float(eta).hex(),
            "bias_hex": float(bias).hex(), "scale_hex": float(scale).hex(),
            "disabled": disabled}


class ConditionalGapTest(unittest.TestCase):
    def test_eta_zero_bit_parity_with_frozen_core(self):
        rng = np.random.default_rng(497)
        for mode in MODES:
            for _ in range(100):
                m = float(rng.normal())
                r = -float(rng.uniform(0, 5))
                d = float(rng.uniform(0, 5))
                gamma, beta = map(float, rng.uniform(0, 3, 2))
                bias = float(rng.normal())
                head = manual(mode, gamma, beta, 0., bias, 3.)
                self.assertEqual(bits([gate_score(m, r, d, head)]),
                                 bits([old.gate_score(m, r, d, head)]))
            zero = manual(mode, gamma=0., beta=-0., eta=0., bias=-0.)
            self.assertEqual(bits([gate_score(-0., -1., 1., zero)]), bits([-0.]))
            self.assertEqual(bits([gate_score(-0., -1., 1., zero)]),
                             bits([old.gate_score(-0., -1., 1., zero)]))
        # phi itself would overflow, but eta=0 must not compute it.
        head = manual(gamma=0., beta=0., eta=0., bias=0.)
        self.assertEqual(gate_score(-1., -sys.float_info.max, 2., head), -1.)

    def test_rms_scale_all_holds_neutral_and_input_order(self):
        rows = [row(-1., -2., 3., 1), row(0., -4., 2., 0),
                row(.1, -sys.float_info.max, 2., -1)]
        for mode, raw in [("GAP_RAW_INTERACT4", [-6., -8.]),
                          ("GAP_CURVE4", [9., 4.])]:
            got = fit_scale(rows, mode)
            vals = np.asarray(raw, dtype=np.float64)
            expected = float(np.sqrt(np.mean(vals * vals, dtype=np.float64)))
            self.assertEqual(got["scale_hex"], expected.hex())
            self.assertEqual(got["count"], 2)
            self.assertEqual(got["raw_phi_hex"], [x.hex() for x in raw])
            self.assertFalse(got["centered"])
            self.assertFalse(got["old_inputs_rescaled"])
        for records in [[], [row(-1., 0., 0., 0)], [row(.1, -10., 10., 1)]]:
            for mode in MODES:
                scale = fit_scale(records, mode)
                self.assertEqual(scale["scale"], 1.)
                self.assertTrue(scale["empty_or_allzero_fallback"])

    def test_feature_literal_product_division(self):
        self.assertEqual(feature_value(-.3, .7, "GAP_RAW_INTERACT4", 1.2).hex(),
                         float(float(-.3 * .7) / 1.2).hex())
        self.assertEqual(feature_value(-.3, .7, "GAP_CURVE4", 1.2).hex(),
                         float(float(.7 * .7) / 1.2).hex())
        for eta in [-.9, .9]:
            head = manual(eta=eta, scale=1.3)
            u0 = float(float(-.8 + float(.2 * -.3)) + float(.7 * 1.2))
            expected = float(float(u0 + float(eta * float(float(-.3 * 1.2) / 1.3))) + .1)
            self.assertEqual(gate_score(-.8, -.3, 1.2, head).hex(), expected.hex())

    def test_four_parameter_analytic_gradient(self):
        offsets = np.array([-2., -.4, 0., -1.])
        design = np.array([[-.1, .3, -.7, 1.], [-.8, 1.5, -2., 1.],
                           [0., 0., 0., 1.], [-1., .2, -.3, 1.]])
        targets = np.array([1., -1., 1., -1.])
        theta = np.array([.6, .2, -.4, -.3])
        _, gradient = surrogate_loss_gradient(theta, offsets, design, targets)
        eps = 1e-6
        numeric = []
        for i in range(4):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric.append((surrogate_loss_gradient(plus, offsets, design, targets)[0] -
                            surrogate_loss_gradient(minus, offsets, design, targets)[0]) / (2 * eps))
        np.testing.assert_allclose(gradient, numeric, atol=1e-9, rtol=1e-7)

    def test_eta_can_fit_both_signs_and_kkt(self):
        for mode in MODES:
            etas = []
            for direction in [1, -1]:
                rows = [row(-.5, -1., 1., direction),
                        row(-.5, -1., 2., -direction)] * 8
                head = fit_head(rows, mode)
                opt = head["optimization"]
                self.assertTrue(opt["success"], opt)
                self.assertLess(opt["projected_gradient_inf_norm"], 1e-5)
                self.assertLessEqual(opt["fitted_surrogate_loss"], opt["initial_surrogate_loss"])
                self.assertGreaterEqual(head["gamma"], 0.)
                self.assertGreaterEqual(head["beta"], 0.)
                self.assertEqual(opt["bounds"], [[0., None], [0., None], [None, None], [None, None]])
                self.assertEqual(head, fit_head(rows, mode))
                etas.append(head["eta"])
            self.assertLess(etas[0] * etas[1], 0., etas)
        np.testing.assert_array_equal(projected_gradient([0, 2, 0, 0], [2, -3, 4, 5], 2),
                                      [0, -3, 4, 5])

    def test_exact_bias_matches_independent_bruteforce(self):
        rng = np.random.default_rng(541)
        for mode in MODES:
            for eta in [-.7, 0., .7]:
                for _ in range(5):
                    rows = [row(-float(rng.uniform(0, 3)), -float(rng.uniform(0, 2)),
                                float(rng.uniform(0, 4)), int(rng.integers(-1, 2))) for _ in range(25)]
                    gamma, beta, scale = .8, .17, 2.3
                    us = []
                    for r in rows:
                        u0 = float(float(r["m"] + float(gamma * r["h_raw"])) + float(beta * r["d"]))
                        raw = float(r["h_raw"] * r["d"] if mode == "GAP_RAW_INTERACT4" else r["d"] * r["d"])
                        phi = float(raw / scale)
                        us.append(u0 if eta == 0. else float(u0 + float(eta * phi)))
                    candidates = {0.}
                    for u in us:
                        candidates.update([-u, math.nextafter(-u, math.inf)])
                    best_key, best_bias = (0, 0, 0., 0.), 0.
                    for bias in candidates:
                        deltas = [r["delta"] for r, u in zip(rows, us) if float(u + bias) > 0.]
                        key = (sum(deltas), -len(deltas), -abs(bias), -bias)
                        if key[0] > 0 and key > best_key:
                            best_key, best_bias = key, bias
                    head = calibrate_bias(rows, gamma, beta, eta, mode, scale)
                    self.assertEqual(head["bias_hex"], best_bias.hex())
                    self.assertEqual(head["training_net_gain"], best_key[0])
                    self.assertEqual(head["training_changed_holds"], -best_key[1])

    def test_neutral_ties_disable_and_existing_switch_ignored(self):
        for mode in MODES:
            rows = [row(-1., delta=1), row(-2., delta=0),
                    row(-3., delta=1), row(-3., delta=-1)]
            head = calibrate_bias(rows, 0., 0., 0., mode, 1.)
            self.assertEqual(head["training_changed_holds"], 1)
            self.assertEqual(head["bias"], math.nextafter(1., math.inf))
            tied = calibrate_bias([row(-1., delta=1), row(-1., delta=-1)], 0., 0., 0., mode, 1.)
            self.assertTrue(tied["disabled"])
            base = [row(-.5, -1., 1., 1), row(-.2, -.3, .2, -1)]
            extra = base + [row(.1, -sys.float_info.max, 20., 1)]
            a, b = fit_head(base, mode), fit_head(extra, mode)
            self.assertEqual(a["optimization"]["theta_hex"], b["optimization"]["theta_hex"])
            self.assertEqual(a["scale_hex"], b["scale_hex"])
            self.assertEqual(b["calibration_certificate"]["original_switch_count"], 1)
            empty = fit_head([], mode)
            self.assertTrue(empty["disabled"])
            self.assertEqual(empty["optimization"]["informative_hold_count"], 0)

    def test_all_127_original_switch_and_retained_hold_bits(self):
        for mode in MODES:
            head = manual(mode, gamma=.1, beta=1., eta=-.01, bias=.1)
            values = [-float(i + 1) for i in range(127)]
            values[2], values[100] = -.1, -.1
            switched = apply_head(values, -.2, .3, head)
            self.assertGreater(switched[2], 0.)
            for i in range(127):
                if i != 2:
                    self.assertEqual(bits([values[i]]), bits([switched[i]]))
            self.assertEqual(max(range(127), key=switched.__getitem__), 2)
            head["bias_hex"] = (-3.).hex()
            self.assertEqual(bits(values), bits(apply_head(values, -.2, .3, head)))
            values[10], values[40] = .2, -0.
            self.assertEqual(bits(values), bits(apply_head(values, -1., 1., head)))
            values[10] = -.2
            head["disabled"] = True
            self.assertEqual(bits(values), bits(apply_head(values, -1., 1., head)))

    def test_finite_invalid_inputs_and_overflow(self):
        for bad in [row(float("nan")), row(-1., .1), row(-1., -.1, -.1), row(-1., delta=2)]:
            with self.assertRaises(ValueError):
                fit_head([bad], "GAP_RAW_INTERACT4")
        for scale in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                feature_value(-1., 1., "GAP_RAW_INTERACT4", scale)
        with self.assertRaises(ValueError):
            feature_value(-1., 1., "UNKNOWN", 1.)
        with self.assertRaises(ValueError):
            feature_value(-sys.float_info.max, 2., "GAP_RAW_INTERACT4", 1.)
        with self.assertRaises(ValueError):
            fit_scale([row(-1., -1e200, 1.)], "GAP_RAW_INTERACT4")
        with self.assertRaises(ValueError):
            fit_scale([row(-1., -1e-200, 1.)], "GAP_RAW_INTERACT4")
        with self.assertRaises(ValueError):
            gate_score(-1., -.1, 1., manual(eta=float("nan")))
        with self.assertRaises(ValueError):
            calibrate_bias([], -1., 1., 0., "GAP_RAW_INTERACT4", 1.)


if __name__ == "__main__":
    unittest.main()
