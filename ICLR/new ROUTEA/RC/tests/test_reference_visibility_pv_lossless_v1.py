"""Engineering parity with the verbatim legacy scalar function, no natural data."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf.reference_visibility_pv_lossless_v1 import (
    make_visibility_proposal, validate_visibility_proposal, verify_visibility,
    serialize_visibility_proposal, deserialize_visibility_proposal, tensor_sha256,
)

LEGACY_PATH = ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py"
LEGACY_SHA256 = "fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c"


def legacy_oracle():
    source = LEGACY_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != LEGACY_SHA256:
        raise AssertionError("Frozen legacy source changed")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "score"]
    if len(functions) != 1:
        raise AssertionError("Legacy score function missing")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LEGACY_PATH), "exec"), namespace)
    return namespace["score"]


class LosslessVisibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.oracle = staticmethod(legacy_oracle())

    def assets(self, qshape=(2, 3), rshape=(3, 4), dtype=torch.float32):
        rng = torch.Generator().manual_seed(17)
        q = torch.randn((qshape[0] * qshape[1], 128), generator=rng, dtype=dtype)
        r = torch.randn((rshape[0] * rshape[1], 128), generator=rng, dtype=dtype)
        wq = torch.rand(len(q), generator=rng, dtype=torch.float64)
        wr = torch.rand(len(r), generator=rng, dtype=torch.float64)
        return q, r, wq, wr

    def proposal(self, q, r, wq, wr, qshape=(2, 3), rshape=(3, 4)):
        source = {"query_tokens_sha256": tensor_sha256(q), "reference_tokens_sha256": tensor_sha256(r),
                  "source_maps_sha256": "a" * 64, "roma_checkpoint_sha256": "b" * 64}
        return make_visibility_proposal(query_resource_key="q", candidate_resource_key="c", reference_resource_key="r",
            query_grid_shape=qshape, reference_grid_shape=rshape, query_visibility=wq, reference_visibility=wr,
            source_binding=source)

    def assertBits(self, x, y):
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(tensor_sha256(x), tensor_sha256(y))

    def check_oracle(self, q, r, wq, wr, qshape, rshape):
        p = self.proposal(q, r, wq, wr, qshape, rshape)
        result = verify_visibility(p, q, r, expected_source_binding=p.source_binding)
        real = self.oracle(q, r, wq, wr)
        qc = self.oracle(q, r, wq.roll(max(1, len(wq) // 2)), wr)
        rc = self.oracle(q, r, wq, wr.roll(max(1, len(wr) // 2)))
        actual = ((result.real_score, result.visibility_mass, result.real_local_values),
                  (result.query_control_score, result.query_control_mass, result.query_control_local_values),
                  (result.reference_control_score, result.reference_control_mass, result.reference_control_local_values))
        for expected, got in zip((real, qc, rc), actual):
            for a, b in zip(expected, got):
                self.assertBits(a, b)
        self.assertBits(p.query_full_mean, wq.mean())
        self.assertBits(p.reference_full_mean, wr.mean())
        self.assertBits(p.visibility_mass, real[1])
        self.assertEqual(set(result.as_old_feature_record()), {"real_score", "visibility_mass", "query_control_score", "reference_control_score"})

    def test_different_grids_and_original_raw_dtypes(self):
        for qshape, rshape in (((1, 1), (1, 2)), ((2, 3), (3, 4)), ((3, 5), (2, 7))):
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(qshape=qshape, rshape=rshape, dtype=dtype):
                    self.check_oracle(*self.assets(qshape, rshape, dtype), qshape, rshape)

    def test_strongly_nonuniform_reference_visibility(self):
        q, r, wq, wr = self.assets()
        wr.zero_(); wr[0] = 1.; wr[5] = 2. ** -30; wr[11] = .03
        self.check_oracle(q, r, wq, wr, (2, 3), (3, 4))

    def test_zero_mass_and_small_query_denominator(self):
        for mode in ("query_zero", "reference_zero", "both_zero", "query_tiny"):
            q, r, wq, wr = self.assets()
            if mode in ("query_zero", "both_zero"):
                wq.zero_()
            if mode in ("reference_zero", "both_zero"):
                wr.zero_()
            if mode == "query_tiny":
                wq.fill_(2. ** -50)
            with self.subTest(mode=mode):
                self.check_oracle(q, r, wq, wr, (2, 3), (3, 4))

    def test_source_and_token_substitution_rejected(self):
        q, r, wq, wr = self.assets(); p = self.proposal(q, r, wq, wr)
        changed = dict(p.source_binding); changed["source_maps_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_visibility_proposal(replace(p, source_binding=changed))
        with self.assertRaises(ValueError):
            verify_visibility(p, q, r, expected_source_binding=changed)
        wrong = r.clone(); wrong[0, 0] += .1
        with self.assertRaises(ValueError):
            verify_visibility(p, q, wrong)
        with self.assertRaises(ValueError):
            validate_visibility_proposal(replace(p, candidate_resource_key="wrong_candidate"))

    def test_axis_visibility_and_saved_mass_tampering_rejected(self):
        q, r, wq, wr = self.assets(); p = self.proposal(q, r, wq, wr)
        for changed in (replace(p, query_axis=p.query_axis.roll(1)),
                        replace(p, reference_axis=p.reference_axis[:-1]),
                        replace(p, reference_visibility=p.reference_visibility.roll(1)),
                        replace(p, visibility_mass=p.visibility_mass + .01)):
            with self.subTest(field=changed):
                with self.assertRaises(ValueError):
                    validate_visibility_proposal(changed)

    def test_binary64_json_roundtrip_including_mass_and_zero(self):
        q, r, wq, wr = self.assets(); wq[0] = -0.; wr[0] = 2. ** -40
        p = self.proposal(q, r, wq, wr)
        loaded = deserialize_visibility_proposal(json.loads(json.dumps(serialize_visibility_proposal(p))))
        self.assertEqual(loaded.logical_sha256, p.logical_sha256)
        for field in ("query_visibility", "reference_visibility", "query_full_mean", "reference_full_mean", "visibility_mass"):
            self.assertBits(getattr(loaded, field), getattr(p, field))
        self.assertBits(verify_visibility(loaded, q, r).real_score, verify_visibility(p, q, r).real_score)
        malformed = serialize_visibility_proposal(p); malformed["query_axis"][0] = False
        with self.assertRaises(ValueError):
            deserialize_visibility_proposal(malformed)

    def test_constructor_does_not_alias_caller_visibility(self):
        q, r, wq, wr = self.assets(); p = self.proposal(q, r, wq, wr)
        wq.zero_(); wr.zero_()
        validate_visibility_proposal(p)
        self.assertGreater(float(p.visibility_mass), 0.)

    def test_sampled_reference_atoms_cannot_determine_full_mass(self):
        q = torch.zeros((2, 128), dtype=torch.float32); q[:, 0] = 1.
        r = torch.zeros((4, 128), dtype=torch.float32); r[:, 0] = 1.
        wq = torch.ones(2, dtype=torch.float64)
        wr_a = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        wr_b = torch.ones(4, dtype=torch.float64)
        # Every hypothetical RoMa source assignment can hit reference cell 0:
        # the sampled reference visibility is identical, the full mass is not.
        sampled = torch.tensor([0, 0], dtype=torch.int64)
        self.assertBits(wr_a[sampled], wr_b[sampled])
        pa = self.proposal(q, r, wq, wr_a, (1, 2), (1, 4))
        pb = self.proposal(q, r, wq, wr_b, (1, 2), (1, 4))
        self.assertEqual(float(pa.visibility_mass), .5)
        self.assertEqual(float(pb.visibility_mass), 1.)
        self.assertEqual(float(verify_visibility(pa, q, r).real_score), .5)
        self.assertEqual(float(verify_visibility(pb, q, r).real_score), 1.)


if __name__ == "__main__":
    unittest.main()
