"""Synthetic numerical/gradient checks; no natural data or scientific gate."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as old
from rc_aslo_xf.reference_visibility_pv_lossless_v1 import tensor_sha256
from rc_aslo_xf.shared_query_target_prior_v1 import (
    PRIOR_PARAMETER_COUNT, SCORE_FIELDS, SharedQueryTargetPrior, FixedActionHead,
    query_features, precompute_candidate_evidence, score_candidate, score_query,
    standardized_raw_gap, frozen_action_features, frozen_action_logit,
    pack_candidate_sources, score_candidates_batched, frozen_action_features_batched,
    frozen_action_logits_batched,
)


class SharedQueryTargetPriorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assets(self, count=4, shape=(3, 5), query_shift=None, reference_shift=None):
        rng = torch.Generator().manual_seed(1713)
        q = torch.randn((shape[0] * shape[1], 128), generator=rng, dtype=torch.float32)
        data, sources = {}, {}
        for index in range(count):
            rshape = (2 + index, 3)
            r = torch.randn((rshape[0] * rshape[1], 128), generator=rng, dtype=torch.float32)
            wq = torch.rand(len(q), generator=rng, dtype=torch.float64)
            wr = torch.rand(len(r), generator=rng, dtype=torch.float64)
            data[index] = (r, wq, wr, rshape)
            sources[index] = precompute_candidate_evidence(q, r, wq, wr, shape, rshape,
                query_control_shift=query_shift, reference_control_shift=reference_shift)
        return q, shape, data, sources

    def assertBits(self, actual, expected):
        self.assertEqual(tensor_sha256(actual), tensor_sha256(expected),
            msg=f"{actual.detach().tolist()} != {expected.detach().tolist()}")

    @staticmethod
    def literal(q, r, wq, wr):
        qn = F.normalize(q.to(torch.float64), dim=1)
        rn = F.normalize(r.to(torch.float64), dim=1)
        sim = qn @ rn.T
        local = (sim * wr[None]).max(1).values
        mass = torch.sqrt(wq.mean() * wr.mean())
        return mass * (wq * local).sum() / wq.sum().clamp_min(1e-12), mass

    def test_feature_axis_normalization_and_zero_initialization(self):
        q, shape, _, _ = self.assets(shape=(2, 3))
        features = query_features(q, shape)
        self.assertEqual(features.shape, (6, 133))
        self.assertBits(features[:, :128], F.normalize(q.double(), dim=1))
        x = torch.tensor([-2./3., 0., 2./3.] * 2, dtype=torch.float64)
        y = torch.tensor([-.5] * 3 + [.5] * 3, dtype=torch.float64)
        torch.testing.assert_close(features[:, 128:], torch.stack((x, y, x*x, y*y, x*y), dim=1), rtol=0., atol=3e-16)
        prior = SharedQueryTargetPrior()
        self.assertEqual(sum(p.numel() for p in prior.parameters()), PRIOR_PARAMETER_COUNT)
        self.assertBits(prior(q, shape), torch.ones(6, dtype=torch.float64))
        self.assertEqual(list(dict(prior.named_parameters())), ["theta"])

    def test_zero_prior_replays_literal_full_reference_four_fields_bits(self):
        for qshift, rshift in ((None, None), (1, 1)):
            q, shape, data, sources = self.assets(query_shift=qshift, reference_shift=rshift)
            output = score_query(SharedQueryTargetPrior(), q, shape, sources)
            for index, (r, wq, wr, _) in data.items():
                real, mass = self.literal(q, r, wq, wr)
                query, _ = self.literal(q, r, wq.roll(max(1, len(q)//2) if qshift is None else qshift), wr)
                reference, _ = self.literal(q, r, wq, wr.roll(max(1, len(r)//2) if rshift is None else rshift))
                for field, expected in zip(SCORE_FIELDS, (real, mass, query, reference)):
                    self.assertBits(getattr(output.records[index], field), expected)

    def test_original_six_features_and_frozen_head_are_bit_exact(self):
        q, shape, _, sources = self.assets()
        output = score_query(SharedQueryTargetPrior(), q, shape, sources)
        raw = [184.41, 179.0125, 183.119, 2.**-27]
        plain = {key: value.as_old_feature_record() for key, value in output.records.items()}
        for challenger in range(4):
            want = old.candidate_feature(raw, plain, challenger, 0)
            got = frozen_action_features(raw, output.records, challenger, 0)
            self.assertBits(got, want)
            self.assertEqual(float(frozen_action_logit(raw, output.records, challenger, 0).detach()).hex(),
                             old.logit(raw, plain, challenger, 0).hex())
            self.assertBits(standardized_raw_gap(raw, challenger, 0), want[0])

    def test_external_native_head_has_no_parameters_and_preserves_weights(self):
        q, shape, _, sources = self.assets()
        prior = SharedQueryTargetPrior()
        external = torch.tensor([.1, -.3, 2.25, 1.0, -.75, .2], dtype=torch.float64)
        head = FixedActionHead(external, -.15)
        self.assertEqual(list(head.parameters()), [])
        self.assertEqual(set(dict(head.named_buffers())), {"weight", "bias"})
        output = score_query(prior, q, shape, sources)
        features = frozen_action_features([4., 3., 2., 1.], output.records, 1, 0)
        result = frozen_action_logit([4., 3., 2., 1.], output.records, 1, 0, head=head)
        self.assertBits(result, (external * features).sum() + torch.tensor(-.15, dtype=torch.float64))
        result.backward()
        self.assertTrue(torch.isfinite(prior.theta.grad).all())
        self.assertGreater(float(prior.theta.grad.norm()), 0.)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(head.bias.grad)
        self.assertBits(head.weight, external)

    def test_query_prior_shared_under_candidate_reordering_and_replacement(self):
        q, shape, _, sources = self.assets()
        prior = SharedQueryTargetPrior()
        with torch.no_grad():
            prior.theta.copy_(torch.linspace(-.2, .3, 133, dtype=torch.float64))
        first = score_query(prior, q, shape, sources)
        second = score_query(prior, q, shape, {"different-reference": sources[3], "other": sources[0]})
        self.assertBits(first.alpha, second.alpha)
        for field in SCORE_FIELDS:
            self.assertBits(getattr(first.records[3], field), getattr(second.records["different-reference"], field))
        with self.assertRaises(TypeError):
            prior(q, shape, candidate_id="must-not-be-an-input")

    def test_prior_position_control_rolls_only_prior_then_recomputes_every_field(self):
        q, shape, data, sources = self.assets()
        prior = SharedQueryTargetPrior()
        with torch.no_grad():
            prior.theta[128] = 1.7
            prior.theta[129] = -.6
        alpha = prior(q, shape)
        got = score_query(prior, q, shape, sources, prior_position_control=True)
        rolled = alpha.roll(max(1, len(q)//2))
        self.assertBits(got.alpha, rolled)
        real_output = score_query(prior, q, shape, sources)
        changed = 0
        for index, (r, wq, wr, _) in data.items():
            effective = rolled * wq
            real, mass = self.literal(q, r, effective, wr)
            query, _ = self.literal(q, r, effective.roll(max(1, len(q)//2)), wr)
            reference, _ = self.literal(q, r, effective, wr.roll(max(1, len(r)//2)))
            for field, expected in zip(SCORE_FIELDS, (real, mass, query, reference)):
                self.assertBits(getattr(got.records[index], field), expected)
                changed += tensor_sha256(expected) != tensor_sha256(getattr(real_output.records[index], field))
            self.assertBits(sources[index].query_visibility, wq)
        self.assertEqual(changed, 16)

    def test_zero_mass_finite_gradient_for_query_reference_and_underflow(self):
        q, shape, data, sources = self.assets()
        for mode in ("query_zero", "reference_zero", "tiny_query", "underflow"):
            r, wq, wr, rshape = data[0]
            wq, wr = wq.clone(), wr.clone()
            if mode == "query_zero": wq.zero_()
            if mode == "reference_zero": wr.zero_()
            if mode == "tiny_query": wq.fill_(2.**-50)
            if mode == "underflow":
                wq.zero_(); wq[0] = 1.
            source = precompute_candidate_evidence(q, r, wq, wr, shape, rshape)
            prior = SharedQueryTargetPrior()
            if mode == "underflow":
                with torch.no_grad(): prior.theta[128] = 1.e6
            alpha = prior(q, shape)
            scores = score_candidate(alpha, source)
            objective = sum(getattr(scores, field) for field in SCORE_FIELDS)
            objective.backward()
            self.assertTrue(torch.isfinite(objective))
            self.assertTrue(torch.isfinite(prior.theta.grad).all(), mode)
            if mode in ("query_zero", "reference_zero", "underflow"):
                self.assertEqual(float(objective.detach()), 0.)
                self.assertEqual(float(prior.theta.grad.norm()), 0.)

    def test_axes_dtypes_frozen_sources_and_nonfinite_inputs_fail_closed(self):
        q, shape, data, sources = self.assets()
        alpha = torch.ones(len(q), dtype=torch.float64)
        for bad in (replace(sources[0], query_axis=sources[0].query_axis.roll(1)),
                    replace(sources[0], query_visibility=sources[0].query_visibility.float()),
                    replace(sources[0], local_values=sources[0].local_values[:-1]),
                    replace(sources[0], reference_mean=torch.tensor(float("nan"), dtype=torch.float64)),
                    replace(sources[0], query_visibility=sources[0].query_visibility.clone().requires_grad_())):
            with self.assertRaises(ValueError): score_candidate(alpha, bad)
        with self.assertRaises(ValueError): query_features(q, (1, 1))
        with self.assertRaises(ValueError): query_features(q[:, :127], shape)
        with self.assertRaises(ValueError): query_features(q, shape, query_axis=torch.arange(len(q)).roll(1))
        with self.assertRaises(ValueError): score_candidate(alpha.float(), sources[0])
        with self.assertRaises(ValueError):
            score_query(SharedQueryTargetPrior(), q, shape, sources, prior_control_shift=1)
        r, wq, wr, rshape = data[0]
        with self.assertRaises(ValueError):
            precompute_candidate_evidence(q, r, wq, wr, shape, rshape, reference_axis=torch.arange(len(r)).flip(0))

    def test_batched_values_and_gradients_match_literal_with_reported_fp64_tolerance(self):
        for shape in ((3, 5), (8, 8), (7, 11)):
            q, shape, _, sources = self.assets(shape=shape)
            packed = pack_candidate_sources(sources)
            prior = SharedQueryTargetPrior()
            with torch.no_grad(): prior.theta.copy_(torch.linspace(-.03, .07, 133, dtype=torch.float64))
            alpha = prior(q, shape)
            literal = [score_candidate(alpha, source) for source in sources.values()]
            batch = score_candidates_batched(alpha, packed)
            left = torch.cat([torch.stack([getattr(row, field) for row in literal]) for field in SCORE_FIELDS])
            right = torch.cat([getattr(batch, field) for field in SCORE_FIELDS])
            torch.testing.assert_close(right, left, rtol=2e-14, atol=2e-16)
            weights = torch.linspace(-1.1, 2.7, left.numel(), dtype=torch.float64)
            gleft = torch.autograd.grad((left * weights).sum(), prior.theta, retain_graph=True)[0]
            gright = torch.autograd.grad((right * weights).sum(), prior.theta)[0]
            torch.testing.assert_close(gright, gleft, rtol=2e-12, atol=2e-14)
            raw = [4., 3., 2., 1.]
            raw_gaps = torch.stack([standardized_raw_gap(raw, i, 0) for i in range(4)])
            features = frozen_action_features_batched(raw_gaps, batch, 0)
            records = batch.as_records(packed.candidate_keys)
            for index in range(4):
                self.assertBits(features[index], frozen_action_features(raw, records, index, 0))
            logits = frozen_action_logits_batched(raw_gaps, batch, 0, FixedActionHead())
            for index in range(4):
                torch.testing.assert_close(logits[index], frozen_action_logit(raw, records, index, 0), rtol=2e-14, atol=2e-15)

    def test_batched_zero_visibility_gradient_is_finite(self):
        q, shape, _, sources = self.assets()
        sources[0] = replace(sources[0], query_visibility=torch.zeros_like(sources[0].query_visibility))
        sources[1] = replace(sources[1], reference_mean=torch.zeros((), dtype=torch.float64),
                             reference_control_mean=torch.zeros((), dtype=torch.float64),
                             local_values=torch.zeros_like(sources[1].local_values),
                             reference_control_local_values=torch.zeros_like(sources[1].local_values))
        prior = SharedQueryTargetPrior()
        scores = score_candidates_batched(prior(q, shape), pack_candidate_sources(sources))
        objective = sum(getattr(scores, field).sum() for field in SCORE_FIELDS)
        objective.backward()
        self.assertTrue(torch.isfinite(prior.theta.grad).all())
        for field in SCORE_FIELDS:
            self.assertBits(getattr(scores, field)[:2], torch.zeros(2, dtype=torch.float64))

    def test_training_gradient_matches_finite_difference_without_updating_head(self):
        q, shape, _, sources = self.assets()
        packed = pack_candidate_sources(sources)
        prior, head = SharedQueryTargetPrior(), FixedActionHead()
        raw_gaps = torch.tensor([0., -.4, -.7, -.9], dtype=torch.float64)
        with torch.no_grad(): prior.theta.copy_(torch.linspace(-.03, .07, 133, dtype=torch.float64))
        direction = torch.linspace(.02, -.09, 133, dtype=torch.float64)
        def objective():
            logits = frozen_action_logits_batched(raw_gaps,
                score_candidates_batched(prior(q, shape), packed), 0, head)
            return F.softplus(-logits[1]) + F.softplus(logits[2])
        analytic = torch.autograd.grad(objective(), prior.theta)[0] @ direction
        initial = prior.theta.detach().clone()
        eps = 1.e-6
        with torch.no_grad():
            prior.theta.copy_(initial + eps * direction); plus = objective()
            prior.theta.copy_(initial - eps * direction); minus = objective()
            prior.theta.copy_(initial)
        torch.testing.assert_close(analytic, (plus-minus)/(2.*eps), rtol=2e-6, atol=2e-8)
        optimizer = torch.optim.SGD(prior.parameters(), lr=.001)
        optimizer.zero_grad(); objective().backward(); optimizer.step()
        self.assertFalse(torch.equal(prior.theta.detach(), initial))
        self.assertBits(head.weight, old.WEIGHT)
        self.assertEqual(list(head.parameters()), [])


if __name__ == "__main__":
    unittest.main()
