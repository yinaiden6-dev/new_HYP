"""Independent synthetic scale, condition-binding and actual-shell V4 checks.

No model, image, real TRAIN/probe row, reference token or held label is read.
Temporary checkpoints model the sealed V3 initial-state source on CPU.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "tests")]
import rc_prellm_m_adapter_v1 as old_adapter
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
import run_rc_internal_m_condition_scale_v4 as v4
import run_rc_internal_m_learned_use_v3 as v3
import test_internal_m_learned_use_v3 as legacy_tests


class AdapterScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(921)

    def paired_modules(self, conditioning="real", gain=1.):
        arguments = dict(hidden_size=12, bottleneck=4, conditioning=conditioning,
                         mass_log_mean=-3., mass_log_std=.7, residual_scale=.1)
        old = old_adapter.QualityResidualAdapter(**arguments)
        with torch.no_grad():
            old.up.weight.normal_(0., .12)
            old.up.bias.normal_(0., .04)
        new = ScaledQualityResidualAdapter(**arguments, condition_gain=gain)
        new.load_state_dict(old.state_dict())
        return old, new

    def assert_forward_and_gradient_parity(self, old, new, dtype):
        generator = torch.Generator().manual_seed(534)
        tokens = torch.randn(9, 12, generator=generator).to(dtype)
        objective = torch.randn(9, 12, generator=generator)
        mass = torch.linspace(.01, .2, 9).requires_grad_()
        outputs, token_gradients, parameter_gradients = [], [], []
        for module in (old, new):
            module.zero_grad(set_to_none=True)
            local = tokens.clone().requires_grad_()
            output = module(local, mass)
            ((output.float() * objective).sum() + output.float().square().mean()).backward()
            outputs.append(output.detach())
            token_gradients.append(local.grad)
            parameter_gradients.append({name: value.grad for name, value in module.named_parameters()})
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertTrue(torch.equal(token_gradients[0], token_gradients[1]))
        self.assertEqual(parameter_gradients[0].keys(), parameter_gradients[1].keys())
        for name in parameter_gradients[0]:
            self.assertIsNotNone(parameter_gradients[0][name], name)
            self.assertTrue(torch.equal(parameter_gradients[0][name], parameter_gradients[1][name]), name)
        self.assertIsNone(mass.grad, "RoMa mass must remain detached")

    def test_gain_one_reproduces_old_nonzero_forward_and_all_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_forward_and_gradient_parity(*self.paired_modules(), dtype)

    def test_constant_any_gain_reproduces_old_forward_and_all_gradients(self):
        for gain in (.125, math.sqrt(3584), 1e6):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(gain=gain, dtype=dtype):
                    old, new = self.paired_modules("constant", gain)
                    self.assert_forward_and_gradient_parity(old, new, dtype)
                    self.assertEqual(float(new.down.weight.grad[:, -1].abs().max()), 0.)

    def test_real_mass_scale_and_zero_residual_initialization(self):
        gain = math.sqrt(3584)
        for dtype in (torch.float32, torch.bfloat16):
            module = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=16,
                conditioning="real", mass_log_mean=-4., mass_log_std=.5, condition_gain=gain)
            source = torch.randn(3, 3584).to(dtype)
            mass = torch.tensor([0., math.exp(-4.), .2])
            expected = ((mass.clamp_min(1e-8).log() + 4.) / .5).unsqueeze(-1) * gain
            self.assertTrue(torch.equal(module.standardized_mass(mass, source), expected))
            self.assertTrue(torch.equal(module(source, mass), source))
            self.assertTrue(torch.equal(module(source, .4), source))
            self.assertEqual(sum(value.numel() for value in module.parameters()), 118304)
            self.assertNotIn("condition_gain", module.state_dict())
        old, scaled = self.paired_modules(gain=gain)
        tokens = torch.randn(9, 12)
        self.assertGreater(float((scaled(tokens, .1) - old(tokens, .1)).detach().abs().max()), 0.)

    def test_invalid_gain_fails_closed(self):
        for gain in (0., -1., float("nan"), float("inf")):
            with self.subTest(gain=gain), self.assertRaisesRegex(ValueError, "positive finite"):
                ScaledQualityResidualAdapter(hidden_size=12, bottleneck=4, condition_gain=gain)


class ConditionBindingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fixed_shuffle_changes_only_mass_and_annotation(self):
        rows = [legacy_tests.synthetic_row(False), legacy_tests.synthetic_row(True)]
        rows[1]["query_id"] = "SYNTHETIC_TRAIN_QUERY_2"
        manifest = {"train_rows": rows, "mass_normalization": {"synthetic": True},
                    "probe_rows": [{"synthetic_unopened_marker": True}]}
        before = copy.deepcopy(manifest)
        actual = v4.shuffled_manifest(manifest, 1)
        self.assertEqual(manifest, before)
        self.assertEqual(actual["probe_rows"], before["probe_rows"])
        self.assertEqual(actual["mass_normalization"], before["mass_normalization"])
        for previous, shifted in zip(before["train_rows"], actual["train_rows"]):
            expected = copy.deepcopy(previous)
            expected["M"] = previous["M"][1:] + previous["M"][:1]
            expected["condition_assignment"] = "fixed_cyclic_shift_1"
            self.assertEqual(shifted, expected)
            self.assertEqual(sorted(shifted["M"]), sorted(previous["M"]))
        self.assertEqual(v4.shuffled_manifest(manifest, 1), actual)
        for shift in (0, 128, -1):
            with self.subTest(shift=shift), self.assertRaisesRegex(RuntimeError, "CONDITION_SHIFT"):
                v4.shuffled_manifest(manifest, shift)

    def test_real_scoring_consumes_shifted_mass_on_original_reference_axis(self):
        original = legacy_tests.synthetic_row()
        shifted = v4.shuffled_manifest({"train_rows": [original]}, 1)["train_rows"][0]
        references = list(range(128))
        for row, intervention, shift in ((original, "native", 0), (shifted, "native", 1),
                                         (original, "shuffled", 1)):
            calls, values = [], []
            def predict(model, cache, module, mass):
                self.assertFalse(torch.is_grad_enabled())
                calls.append(mass)
                return torch.tensor(mass, dtype=torch.float64)
            with mock.patch.object(v4, "predict", side_effect=predict), \
                 mock.patch.object(v4.OLD, "content_score", side_effect=lambda output, reference: output + reference), \
                 mock.patch.object(v4.time, "monotonic", return_value=0.):
                complete = v4.score_all(None, None, SimpleNamespace(conditioning="real"), row,
                    references, values, 0., 1., lambda value: None, intervention)
            self.assertTrue(complete)
            expected = original["M"][shift:] + original["M"][:shift]
            self.assertEqual(calls, expected)
            self.assertEqual(values, [mass + i for i, mass in enumerate(expected)])
            self.assertEqual(references, list(range(128)))

    def test_internal_head_has_no_mass_access_in_features_loss_gradients_or_action(self):
        for raw_correct in (False, True):
            row = legacy_tests.synthetic_row(raw_correct)
            absent = copy.deepcopy(row)
            del absent["M"]
            content = torch.tensor(row["L0"], dtype=torch.float64, requires_grad=True)
            head = torch.tensor([.4, 2., -.2], dtype=torch.float64, requires_grad=True)
            expected = legacy_tests.independent_features(row, content, "INTERNAL3")
            self.assertTrue(torch.equal(v4.features(absent, content), expected))
            original = v4.joint_derivatives(row, content, head)
            no_mass = v4.joint_derivatives(absent, content, head)
            for actual, reference in zip(no_mass, original):
                self.assertTrue(torch.equal(actual, reference))
            self.assertEqual(v4.choose(row, content.detach(), head.detach()),
                             v4.choose(absent, content.detach(), head.detach()))
            self.assertTrue(torch.equal(v4.loss(row, content, head), v3.loss(row, content, head)))


class SourceInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_factory_preserves_synthetic_v3_source_and_exact_initial_state(self):
        config = dict(v4.CONFIG)
        manifest = {"mass_normalization": {"log_mean": -4., "log_std": .5, "epsilon": 1e-8}}
        torch.manual_seed(config["seed"])
        baseline = old_adapter.QualityResidualAdapter(hidden_size=3584, bottleneck=config["bottleneck"],
            conditioning="real", mass_log_mean=-4., mass_log_std=.5, mass_epsilon=1e-8,
            residual_scale=config["residual_scale"])
        with tempfile.TemporaryDirectory(prefix="v4-init-", suffix=".synthetic") as temporary:
            path = Path(temporary) / "old_v3_initial.pt"
            torch.save({"adapter": v4.state_cpu(baseline), "head": torch.tensor([.1, 2., -.3])}, path)
            config["baseline_initial"] = v4.bind(path)
            original_bytes = path.read_bytes()
            for arm in v4.ARMS:
                # Exercise the production factory and all its lineage checks;
                # the only interception keeps the verified small module on CPU.
                with mock.patch.object(ScaledQualityResidualAdapter, "to", autospec=True,
                                       side_effect=lambda module, device: module) as movement:
                    created = v4.create_adapter(config, manifest, arm)
                movement.assert_called_once()
                self.assertEqual(movement.call_args.args[1], "cuda")
                self.assertTrue(v4.tree_equal(v4.state_cpu(created), v4.state_cpu(baseline)))
                self.assertEqual(created.condition_gain, math.sqrt(3584))
                self.assertEqual(created.conditioning, "real")
                self.assertEqual(path.read_bytes(), original_bytes)
                self.assertEqual(v4.bind(path), config["baseline_initial"])
            changed = v4.state_cpu(baseline)
            changed["down.weight"][0, 0] += .1
            torch.save({"adapter": changed}, path)
            with self.assertRaises(RuntimeError):
                v4.create_adapter(config, manifest, "PRE_REAL")
            config["baseline_initial"] = v4.bind(path)
            with self.assertRaisesRegex(RuntimeError, "SAME_V3_INITIAL_ADAPTER"):
                v4.create_adapter(config, manifest, "PRE_REAL")

    def test_scale_only_configuration_matches_v3_training_recipe(self):
        for name, value in v3.CONFIG.items():
            self.assertEqual(v4.CONFIG[name], value, name)
        self.assertEqual(set(v4.CONFIG) - set(v3.CONFIG), {"condition_gain", "condition_shift"})
        self.assertEqual(v4.CONFIG["condition_gain"], math.sqrt(3584))
        self.assertEqual(v4.CONFIG["condition_shift"], 1)
        self.assertEqual(v4.ARMS, ("PRE_REAL", "PRE_SHUFFLED"))
        self.assertNotEqual(v4.OUT, v3.OUT)
        self.assertEqual(v4.BASELINE, v3.OUT)


class BaselineSourceChainTests(unittest.TestCase):
    """An intact parsed JSON payload is insufficient when sealed bytes drift."""

    def fixture(self, root):
        def write(relative, payload):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n")
            return path
        authority = write("sealed/authority.json", {"synthetic": True})
        authority_binding = v4.bind(authority)
        heads = {
            kind: write(f"sealed/heads/{kind}.json", {"kind": kind, "authority": authority_binding,
                                                       "theta": [0.] * len(v4.FEATURES[kind])})
            for kind in ("INTERNAL3", "ADDITIVE4", "PRODUCT5")}
        endpoints, fits = {}, {}
        for arm in ("PRE_REAL", "PRE_CONSTANT"):
            endpoint_bindings = []
            for step in (16, 128):
                path = write(f"sealed/{arm}/endpoints/{step:04d}.json", {
                    "status": "FULL_TRAIN16_ENDPOINT_PASS", "arm": arm, "step": step,
                    "authority": authority_binding, "predictions": [], "summaries": {"synthetic": True}})
                endpoints[(arm, step)] = path
                endpoint_bindings.append(v4.bind(path))
            fits[arm] = write(f"sealed/{arm}/fit_validation.json", {
                "status": "FIXED128_UPDATES_TRAIN_ENDPOINTS_COMPLETE", "arm": arm, "steps": 128,
                "authority": authority_binding, "endpoints": endpoint_bindings})
        result = write("sealed/result.json", {
            "status": "TRAIN16_INTERNAL_M_LEARNED_USE_DIAGNOSTIC_COMPLETE", "authority": authority_binding,
            "external": {kind: {"binding": v4.bind(path)} for kind, path in heads.items()},
            "internal": {arm: {"validation": v4.bind(path)} for arm, path in fits.items()}})
        config = dict(v4.CONFIG, baseline_result=v4.bind(result), baseline_authority=authority_binding)
        # A live directory with a different source list must never override the
        # list reachable from the immutable result binding in the authority.
        decoy = write("live/cpu/decoy.json", {"synthetic_unbound": True})
        write("live/cpu/validation.json", {"authority": authority_binding, "heads": [v4.bind(decoy)]})
        for arm in fits:
            write(f"live/{arm}/endpoints/0128/validation.json", {"step": 128, "synthetic_unbound": True})
        return config, heads, fits, endpoints, result

    def test_head_bindings_follow_sealed_result_not_live_cpu_index(self):
        with tempfile.TemporaryDirectory(prefix="v4-head-chain-", suffix=".synthetic") as temporary:
            root = Path(temporary)
            config, heads, _, _, _ = self.fixture(root)
            with mock.patch.object(v4, "BASELINE", root / "live"):
                actual = v4.baseline_head_bindings(config)
            self.assertEqual(actual, [v4.bind(path) for path in heads.values()])

    def test_head_or_sealed_result_byte_drift_is_rejected(self):
        for changed in ("result", "INTERNAL3", "ADDITIVE4", "PRODUCT5"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(suffix=".synthetic") as temporary:
                config, heads, _, _, result = self.fixture(Path(temporary))
                source = result if changed == "result" else heads[changed]
                before = json.loads(source.read_text())
                source.write_text(source.read_text() + "\n")
                self.assertEqual(json.loads(source.read_text()), before)
                with self.assertRaises(RuntimeError):
                    v4.baseline_head_bindings(config)

    def test_endpoint_binding_follows_exact_arm_and_fixed_step(self):
        with tempfile.TemporaryDirectory(prefix="v4-endpoint-chain-", suffix=".synthetic") as temporary:
            root = Path(temporary)
            config, _, _, endpoints, _ = self.fixture(root)
            with mock.patch.object(v4, "BASELINE", root / "live"):
                for (arm, step), path in endpoints.items():
                    self.assertEqual(v4.baseline_endpoint_binding(config, arm, step), v4.bind(path))
                with self.assertRaises(RuntimeError):
                    v4.baseline_endpoint_binding(config, "PRE_REAL", 64)

    def test_endpoint_chain_rejects_result_fit_receipt_and_endpoint_byte_drift(self):
        for changed in ("result", "fit", "endpoint"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(suffix=".synthetic") as temporary:
                config, _, fits, endpoints, result = self.fixture(Path(temporary))
                source = {"result": result, "fit": fits["PRE_REAL"],
                          "endpoint": endpoints[("PRE_REAL", 128)]}[changed]
                before = json.loads(source.read_text())
                source.write_text(source.read_text() + "\n")
                self.assertEqual(json.loads(source.read_text()), before)
                with self.assertRaises(RuntimeError):
                    v4.baseline_endpoint_binding(config, "PRE_REAL", 128)


class V4LauncherExecutionTests(legacy_tests.LauncherExecutionTests):
    """Inherited tests execute this actual V4 shell with only tools redirected.

    Coverage includes success, worker75, timeout124, hard failure, scontrol
    failure, and the finite restart limit, in a temporary path with spaces.
    """
    launcher = ROOT / "slurm/rc_internal_m_condition_scale_v4.sbatch"

    def test_success_dispatches_exact_v4_worker_then_advance(self):
        result, calls = self.run_launcher(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            ["python", "-u", "programs/run_rc_internal_m_condition_scale_v4.py", "train",
             "--arm", "PRE_REAL", "--budget", "400"],
            ["python", "-u", "programs/run_rc_internal_m_condition_scale_v4.py", "advance-train",
             "--arm", "PRE_REAL"],
        ])

    def test_timeout_at_restart_limit_fails_closed_without_dispatch(self):
        result, calls = self.run_launcher(124, restart=v4.CONFIG["max_requeues"])
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "python")

    def test_last_permitted_requeue_uses_same_job(self):
        result, calls = self.run_launcher(75, restart=v4.CONFIG["max_requeues"] - 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[-1], ["scontrol", "requeue", "SYNTHETIC_JOB_7001"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
