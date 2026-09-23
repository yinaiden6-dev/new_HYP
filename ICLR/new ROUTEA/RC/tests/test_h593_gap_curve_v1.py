"""Synthetic-only checks of the RAW-independent three-parameter gap curve."""
from pathlib import Path
import math
import struct
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf import h593_raw_incumbent_gate_v1 as old
from rc_aslo_xf.h593_gap_curve_v1 import (
    MODE, apply_head, calibrate_bias, feature_value, fit_head, fit_scale,
    gate_score, projected_gradient, surrogate_loss_gradient,
)


def bits(xs):
    return b"".join(struct.pack(">d", float(x)) for x in xs)


def row(m, r=-.5, d=1., delta=1):
    return {"m": m, "h_raw": r, "d": d, "delta": delta}


def manual(beta=.7, eta=.3, bias=.1, scale=1., disabled=False):
    return {"mode": MODE, "gamma_hex": 0.0.hex(),
            "beta_hex": float(beta).hex(), "eta_hex": float(eta).hex(),
            "bias_hex": float(bias).hex(), "scale_hex": float(scale).hex(),
            "disabled": disabled}


class GapCurveTest(unittest.TestCase):
    def test_raw_input_cannot_affect_training_or_application(self):
        rows = [row(-.5, -.1, .2, 1), row(-.5, -3., 2., -1),
                row(-.8, -0., 1., 0), row(.1, -9., 3., 1)] * 3
        baseline = fit_head(rows)
        self.assertEqual(baseline["mode"], MODE)
        self.assertEqual(baseline["training_scale"]["mode"], MODE)
        self.assertEqual(baseline["gamma_hex"], 0.0.hex())
        self.assertEqual(baseline["frozen_gamma"], 0.)
        self.assertTrue(baseline["ignored_raw_input"])
        # Ignored RAW input is not even validated. NaN/positive values and a
        # missing field cannot become accidental fit or score dependencies.
        for raw in [-0., 0., -.2, -1e200, float("nan"), 1.]:
            self.assertEqual(baseline, fit_head([{**r, "h_raw": raw} for r in rows]))
            for r in rows:
                self.assertEqual(bits([gate_score(r["m"], raw, r["d"], baseline)]),
                                 bits([gate_score(r["m"], 0., r["d"], baseline)]))
        self.assertEqual(baseline, fit_head([{k: v for k, v in r.items() if k != "h_raw"}
                                           for r in rows]))

    def test_eta_zero_old_gap_bit_parity(self):
        rng = np.random.default_rng(449)
        for _ in range(100):
            m = float(rng.normal())
            d = float(rng.uniform(0, 4))
            beta = float(rng.uniform(0, 3))
            bias = float(rng.normal())
            head = manual(beta=beta, eta=0., bias=bias)
            old_head = {**head, "mode": "GAP_BIAS2"}
            self.assertEqual(bits([gate_score(m, -3., d, head)]),
                             bits([old.gate_score(m, 0., d, old_head)]))
        for m in [-0., 0.]:
            for beta in [-0., 0.]:
                for bias in [-0., 0.]:
                    head = manual(beta=beta, eta=0., bias=bias)
                    old_head = {**head, "mode": "GAP_BIAS2"}
                    expected = old.gate_score(m, 0., 1., old_head)
                    for raw in [-0., 0., -1.]:
                        self.assertEqual(bits([gate_score(m, raw, 1., head)]), bits([expected]))
        # eta=0 skips the feature rather than evaluating overflowing d squared.
        head = manual(beta=0., eta=0., bias=0.)
        self.assertEqual(gate_score(-1., -1., sys.float_info.max, head), -1.)

    def test_scale_all_original_holds_and_neutral_only(self):
        rows = [row(-1., d=3., delta=1), row(0., d=2., delta=0),
                row(.1, d=1e200, delta=-1)]
        scale = fit_scale(rows)
        raw = np.array([9., 4.], dtype=np.float64)
        expected = float(np.sqrt(np.mean(raw * raw, dtype=np.float64)))
        self.assertEqual(scale["scale_hex"], expected.hex())
        self.assertEqual(scale["raw_phi_hex"], [9.0.hex(), 4.0.hex()])
        self.assertEqual(scale["count"], 2)
        self.assertFalse(scale["centered"])
        for records in [[], [row(-1., d=0., delta=0)], [row(.1, d=1e200)]]:
            self.assertEqual(fit_scale(records)["scale"], 1.)
        self.assertEqual(feature_value(-3., .7, MODE, 1.2).hex(),
                         float(float(.7 * .7) / 1.2).hex())

    def test_three_parameter_gradient_and_kkt(self):
        offsets = np.array([-2., -.4, 0., -1.])
        design = np.array([[.3, .09, 1.], [1.5, 2.25, 1.],
                           [0., 0., 1.], [.2, .04, 1.]])
        targets = np.array([1., -1., 1., -1.])
        theta = np.array([.2, -.4, -.3])
        _, gradient = surrogate_loss_gradient(theta, offsets, design, targets)
        eps = 1e-6
        numeric = []
        for i in range(3):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric.append((surrogate_loss_gradient(plus, offsets, design, targets)[0] -
                            surrogate_loss_gradient(minus, offsets, design, targets)[0]) / (2 * eps))
        np.testing.assert_allclose(gradient, numeric, atol=1e-9, rtol=1e-7)
        etas = []
        for direction in [1, -1]:
            rows = [row(-.5, d=1., delta=direction),
                    row(-.5, d=2., delta=-direction)] * 8
            head = fit_head(rows)
            opt = head["optimization"]
            self.assertTrue(opt["success"], opt)
            self.assertLess(opt["projected_gradient_inf_norm"], 1e-5)
            self.assertLessEqual(opt["fitted_surrogate_loss"], opt["initial_surrogate_loss"])
            self.assertEqual(opt["free_features"], ["d", "phi", "bias"])
            self.assertEqual(opt["coefficient_order"], ["beta", "eta", "bias"])
            self.assertEqual(opt["bounds"], [[0., None], [None, None], [None, None]])
            self.assertEqual(len(opt["theta_hex"]), 3)
            etas.append(head["eta"])
        self.assertLess(etas[0] * etas[1], 0.)
        np.testing.assert_array_equal(projected_gradient([0, 0, 0], [2, 4, 5], 1), [0, 4, 5])

    def test_bias_certificate_against_independent_bruteforce(self):
        rng = np.random.default_rng(553)
        for eta in [-.7, 0., .7]:
            for _ in range(5):
                rows = [row(-float(rng.uniform(0, 3)), -float(rng.uniform(0, 5)),
                            float(rng.uniform(0, 4)), int(rng.integers(-1, 2))) for _ in range(25)]
                beta, scale = .17, 2.3
                us = []
                for r in rows:
                    u = float(float(r["m"] + 0.) + float(beta * r["d"]))
                    phi = float(float(r["d"] * r["d"]) / scale)
                    us.append(u if eta == 0. else float(u + float(eta * phi)))
                candidates = {0.}
                for u in us:
                    candidates.update([-u, math.nextafter(-u, math.inf)])
                best_key, best_bias = (0, 0, 0., 0.), 0.
                for bias in candidates:
                    deltas = [r["delta"] for r, u in zip(rows, us) if float(u + bias) > 0.]
                    key = (sum(deltas), -len(deltas), -abs(bias), -bias)
                    if key[0] > 0 and key > best_key:
                        best_key, best_bias = key, bias
                head = calibrate_bias(rows, beta, eta, scale=scale)
                self.assertEqual(head["bias_hex"], best_bias.hex())
                self.assertEqual(head["training_net_gain"], best_key[0])
                self.assertEqual(head["training_changed_holds"], -best_key[1])

    def test_neutral_ties_and_empty_disable(self):
        rows = [row(-1., delta=1), row(-2., delta=0),
                row(-3., delta=1), row(-3., delta=-1)]
        head = calibrate_bias(rows, 0., 0.)
        self.assertEqual(head["training_changed_holds"], 1)
        self.assertEqual(head["bias"], math.nextafter(1., math.inf))
        tied = calibrate_bias([row(-1., delta=1), row(-1., delta=-1)], 0., 0.)
        self.assertTrue(tied["disabled"])
        self.assertTrue(fit_head([])["disabled"])

    def test_all_127_original_switch_and_retained_hold_bits(self):
        head = manual(beta=1., eta=-.01, bias=.1)
        values = [-float(i + 1) for i in range(127)]
        values[2], values[100] = -.1, -.1
        switched = apply_head(values, -3., .3, head)
        self.assertGreater(switched[2], 0.)
        for i in range(127):
            if i != 2:
                self.assertEqual(bits([values[i]]), bits([switched[i]]))
        for raw in [-0., 0., -1., -sys.float_info.max]:
            self.assertEqual(bits(switched), bits(apply_head(values, raw, .3, head)))
        head["bias_hex"] = (-3.).hex()
        self.assertEqual(bits(values), bits(apply_head(values, -1., .3, head)))
        values[10], values[40] = .2, -0.
        self.assertEqual(bits(values), bits(apply_head(values, -1., 1., head)))
        values[10] = -.2
        head["disabled"] = True
        self.assertEqual(bits(values), bits(apply_head(values, -1., 1., head)))

    def test_invalid_active_inputs_and_locked_gamma(self):
        for bad in [row(float("nan")), row(-1., d=-.1), row(-1., delta=2)]:
            with self.assertRaises(ValueError):
                fit_head([bad])
        with self.assertRaises(ValueError):
            fit_head([], "GAP_CURVE4")
        with self.assertRaises(ValueError):
            gate_score(-1., -1., 1., {**manual(), "gamma_hex": (.1).hex()})
        with self.assertRaises(ValueError):
            gate_score(-1., -1., 1., manual(eta=float("nan")))
        with self.assertRaises(ValueError):
            fit_scale([row(-1., d=1e200)])
        with self.assertRaises(ValueError):
            fit_scale([row(-1., d=1e-100)])


if __name__ == "__main__":
    unittest.main()
