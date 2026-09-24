"""Independent full-candidate arithmetic, resume and real-shell launcher checks.

These checks use synthetic retrieval labels and never open held curator labels.
They deliberately exercise production V3 helpers rather than copying a training
loop and only comparing that copy with itself.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
import run_rc_internal_m_learned_use_v3 as v3


def synthetic_row(target_is_raw: bool = False) -> dict:
    generator = torch.Generator().manual_seed(347)
    raw = torch.randn(128, dtype=torch.float64, generator=generator)
    winner = int(raw.argmax())
    target = winner if target_is_raw else (winner + 71) % 128
    return {
        "query_id": "SYNTHETIC_TRAIN_QUERY",
        "split": "train",
        "candidate_ids": list(range(128)),
        "candidate_identities": ["synthetic_" + str(i) for i in range(128)],
        "raw_scores": raw.tolist(),
        "M": torch.linspace(.007, .25, 128, dtype=torch.float64).tolist(),
        "L0": torch.linspace(.21, .67, 128, dtype=torch.float64).tolist(),
        "winner_index": winner,
        "challenger_positions": [i for i in range(128) if i != winner],
        "target_positions": [target],
        "target_position": target,
    }


def independent_features(row: dict, content: torch.Tensor, kind: str) -> torch.Tensor:
    """Literal definition, independent of the production scorer."""
    raw = torch.tensor(row["raw_scores"], dtype=torch.float64)
    mass = torch.tensor(row["M"], dtype=torch.float64)
    winner = row["winner_index"]
    challengers = [i for i in range(128) if i != winner]

    def sym(values):
        return (values[challengers] - values[winner]) / (
            values[challengers].abs() + values[winner].abs() + 1e-12
        )

    parts = [(raw[challengers] - raw[winner]) / raw.std(unbiased=False).clamp_min(1e-12)]
    if kind == "PRODUCT5":
        parts += [sym(mass * content), sym(mass), sym(content)]
    elif kind == "ADDITIVE4":
        parts += [sym(mass), sym(content)]
    elif kind == "INTERNAL3":
        parts += [sym(content)]
    else:
        raise ValueError(kind)
    parts.append(torch.ones(127, dtype=torch.float64))
    return torch.stack(parts, dim=1)


def independent_cost(row: dict, logits: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    winner = row["winner_index"]
    target = row["target_positions"][0]
    challengers = [i for i in range(128) if i != winner]
    if target == winner:
        return F.softplus(logits.amax())
    position = challengers.index(target)
    wrong = torch.cat((logits[:position], logits[position + 1:]))
    return F.softplus(-logits[position]) + F.softplus(wrong.amax())


class LearnedUseArithmeticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_feature_columns_full_axis_and_no_direct_internal_mass(self):
        row = synthetic_row()
        content = torch.tensor(row["L0"], dtype=torch.float64)
        for kind in ("INTERNAL3", "ADDITIVE4", "PRODUCT5"):
            with self.subTest(kind=kind):
                expected = independent_features(row, content, kind)
                actual = v3.features(row, content, kind)
                self.assertEqual(actual.shape, (127, len(v3.FEATURES[kind])))
                torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)
        no_mass = copy.deepcopy(row)
        del no_mass["M"]
        torch.testing.assert_close(v3.features(no_mass, content), v3.features(row, content), atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "FULL_C128"):
            v3.features(row, content[:-1])
        wrong_axis = copy.deepcopy(row)
        wrong_axis["challenger_positions"].reverse()
        with self.assertRaisesRegex(RuntimeError, "CHALLENGER_ORDER"):
            v3.features(wrong_axis, content)

    def test_original_cost1_both_label_types_and_tie_derivative(self):
        for correct in (False, True):
            row = synthetic_row(correct)
            z = torch.linspace(-.7, .8, 127, dtype=torch.float64, requires_grad=True)
            actual = v3.cost_from_logits(row, z)
            expected = independent_cost(row, z)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(torch.autograd.grad(actual, z)[0],
                                       torch.autograd.grad(expected, z)[0], atol=0, rtol=0)
        row = synthetic_row(True)
        tied = torch.zeros(127, dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(v3.cost_from_logits(row, tied), tied)[0]
        torch.testing.assert_close(grad, torch.full_like(grad, .5 / 127), atol=1e-16, rtol=0)

    def test_action_hold_zero_and_physical_candidate_tie_order(self):
        row = synthetic_row()
        row["candidate_ids"] = [7000 + i for i in range(128)]
        content = row["L0"]
        hold = v3.choose(row, content, [0., 0., 0.])
        self.assertEqual(hold["prediction_position"], row["winner_index"])
        self.assertEqual(hold["prediction_id"], row["candidate_ids"][row["winner_index"]])
        self.assertEqual(len(hold["logits"]), 127)
        self.assertEqual(hold["scores128"][row["winner_index"]], 0.)
        tie = v3.choose(row, content, [0., 0., .1])
        self.assertEqual(tie["prediction_position"], row["challenger_positions"][0])
        self.assertTrue(tie["switched"])
        label_free = copy.deepcopy(row)
        del label_free["target_positions"]
        prediction = v3.choose(label_free, content, [0., 0., .1])
        self.assertNotIn("correct", prediction)
        self.assertEqual(prediction["prediction_position"], tie["prediction_position"])

    def test_joint_content_and_head_vjp_matches_direct_autograd(self):
        generator = torch.Generator().manual_seed(771)
        for correct in (False, True):
            row = synthetic_row(correct)
            design = torch.randn(128, 5, generator=generator, dtype=torch.float64)
            for kind in ("INTERNAL3", "ADDITIVE4", "PRODUCT5"):
                with self.subTest(correct=correct, kind=kind):
                    latent = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)
                    theta = torch.randn(len(v3.FEATURES[kind]), generator=generator,
                                        dtype=torch.float64, requires_grad=True)
                    contents = (design @ latent).sigmoid()
                    direct_loss = independent_cost(row, independent_features(row, contents, kind) @ theta)
                    direct_adapter, direct_head = torch.autograd.grad(direct_loss, (latent, theta))
                    leaf = contents.detach().clone().requires_grad_()
                    replay_head = theta.detach().clone().requires_grad_()
                    actual_loss, derivative, head_grad = v3.joint_derivatives(row, leaf, replay_head, kind)
                    replay_latent = latent.detach().clone().requires_grad_()
                    active = derivative.nonzero().flatten().tolist()
                    self.assertTrue(active)
                    for i in active:
                        ((design[i] @ replay_latent).sigmoid() * derivative[i]).backward()
                    torch.testing.assert_close(actual_loss.detach(), direct_loss.detach(), atol=1e-13, rtol=0)
                    torch.testing.assert_close(replay_latent.grad, direct_adapter, atol=1e-12, rtol=0)
                    torch.testing.assert_close(head_grad, direct_head, atol=1e-12, rtol=0)
                    self.assertFalse(derivative.requires_grad)
                    self.assertFalse(head_grad.requires_grad)

    def test_shared_warm_start_identical_predictions_for_all_heads(self):
        row = synthetic_row()
        contents = torch.tensor(row["L0"], dtype=torch.float64)
        warm = [1.37, -2.4, .09]
        baseline = v3.features(row, contents, "INTERNAL3") @ torch.tensor(warm, dtype=torch.float64)
        for kind in ("INTERNAL3", "ADDITIVE4", "PRODUCT5"):
            expanded = v3.map_warm_head(warm, kind)
            actual = v3.features(row, contents, kind) @ torch.tensor(expanded, dtype=torch.float64)
            torch.testing.assert_close(actual, baseline, atol=1e-13, rtol=0)
            self.assertEqual(v3.choose(row, contents, expanded, kind)["prediction_position"],
                             v3.choose(row, contents, warm)["prediction_position"])


class CandidateReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_partial_candidate_resume_and_constant_forward_reuse(self):
        row = synthetic_row()
        refs = [torch.tensor(i / 1000, dtype=torch.float64) for i in range(128)]
        for conditioning in ("constant", "real"):
            with self.subTest(conditioning=conditioning):
                small = SimpleNamespace(conditioning=conditioning)
                calls = []

                def predict(model, cache, adapter, mass):
                    self.assertFalse(torch.is_grad_enabled())
                    calls.append(mass)
                    return torch.tensor(0.2 if adapter.conditioning == "constant" else mass,
                                        dtype=torch.float64)

                def score(output, reference):
                    return output + reference

                values, persisted = [], []
                clock = iter([0.] * 23 + [2.] * 200)
                with mock.patch.object(v3, "predict", side_effect=predict), \
                     mock.patch.object(v3.OLD, "content_score", side_effect=score), \
                     mock.patch.object(v3.time, "monotonic", side_effect=lambda: next(clock)):
                    done = v3.score_all(None, None, small, row, refs, values, 0., 1.,
                                        lambda v: persisted.append(list(v)))
                self.assertFalse(done)
                self.assertEqual(len(values), 23)
                self.assertEqual(persisted[-1], values)
                prefix = list(values)
                with mock.patch.object(v3, "predict", side_effect=predict), \
                     mock.patch.object(v3.OLD, "content_score", side_effect=score), \
                     mock.patch.object(v3.time, "monotonic", return_value=0.):
                    done = v3.score_all(None, None, small, row, refs, values, 0., 1., lambda v: None)
                self.assertTrue(done)
                self.assertEqual(len(values), 128)
                self.assertEqual(values[:23], prefix)
                expected = [(.2 if conditioning == "constant" else row["M"][i]) + i / 1000
                            for i in range(128)]
                torch.testing.assert_close(torch.tensor(values), torch.tensor(expected), atol=0, rtol=0)
                self.assertEqual(len(calls), 2 if conditioning == "constant" else 128)
                self.assertEqual(small.conditioning, conditioning)

    def test_mass_diagnostic_restores_module_even_on_exception(self):
        small = SimpleNamespace(conditioning="real")
        with mock.patch.object(v3, "predict", side_effect=RuntimeError("synthetic forward fault")):
            with self.assertRaisesRegex(RuntimeError, "synthetic forward fault"):
                v3.score_all(None, None, small, synthetic_row(), [None] * 128, [],
                             0., 10., lambda v: None, "constant")
        self.assertEqual(small.conditioning, "real")

    def test_corrupt_partial_axis_or_values_fail_before_forward(self):
        for invalid in ([0.] * 129, [float("nan")], [float("inf")], ["0.1"], (0.,)):
            with self.subTest(invalid=str(invalid)[:40]), \
                 mock.patch.object(v3, "predict") as forward:
                with self.assertRaisesRegex(RuntimeError, "PENDING_"):
                    v3.score_all(None, None, SimpleNamespace(conditioning="real"), synthetic_row(),
                                 [None] * 128, invalid, 0., 10., lambda v: None)
                forward.assert_not_called()

    def test_shared_query_backward_accumulates_all_active_candidates(self):
        # Constant conditioning produces one query graph. Its reuse must retain
        # every target/winner/strongest-wrong contribution before backward.
        row = synthetic_row(False)
        generator = torch.Generator().manual_seed(931)
        design = torch.randn(128, 6, generator=generator, dtype=torch.float64)
        adapter = torch.randn(6, generator=generator, dtype=torch.float64, requires_grad=True)
        head = torch.tensor([.4, 2.1, -.07], dtype=torch.float64, requires_grad=True)
        common = adapter.tanh()
        scores = (design @ common).sigmoid()
        direct = independent_cost(row, independent_features(row, scores, "INTERNAL3") @ head)
        expected_adapter, expected_head = torch.autograd.grad(direct, (adapter, head))
        leaf = scores.detach().clone().requires_grad_()
        h = head.detach().clone().requires_grad_()
        _, derivative, actual_head = v3.joint_derivatives(row, leaf, h, "INTERNAL3")
        replay_adapter = adapter.detach().clone().requires_grad_()
        reused = replay_adapter.tanh()
        active = derivative.nonzero().flatten().tolist()
        self.assertGreaterEqual(len(active), 2)
        torch.stack([(design[i] @ reused).sigmoid() * derivative[i] for i in active]).sum().backward()
        torch.testing.assert_close(replay_adapter.grad, expected_adapter, atol=1e-12, rtol=0)
        torch.testing.assert_close(actual_head, expected_head, atol=1e-12, rtol=0)


class JointCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4242)

    def fixture(self):
        small = torch.nn.Linear(4, 1, dtype=torch.float64)
        head = torch.nn.Parameter(torch.tensor([.5, 1.8, -.03], dtype=torch.float64))
        config = dict(v3.CONFIG)
        optimizer = v3.make_optimizer(small, head, config)
        row = synthetic_row()
        inputs = torch.randn(128, 4, dtype=torch.float64)
        return small, head, config, optimizer, row, inputs

    def update(self, small, head, optimizer, row, inputs):
        optimizer.zero_grad(set_to_none=True)
        value = v3.loss(row, small(inputs).flatten().sigmoid(), head)
        value.backward()
        torch.nn.utils.clip_grad_norm_(small.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_([head], 1.)
        optimizer.step()

    def test_optimizer_and_joint_parameters_resume_exactly_next_update(self):
        small, head, config, optimizer, row, inputs = self.fixture()
        self.update(small, head, optimizer, row, inputs)
        saved_small, saved_head, saved_opt = (v3.state_cpu(small), head.detach().clone(),
                                             v3.cpu_optimizer_state(optimizer))
        restored_small = torch.nn.Linear(4, 1, dtype=torch.float64)
        restored_small.load_state_dict(saved_small)
        restored_head = torch.nn.Parameter(saved_head.clone())
        restored_opt = v3.make_optimizer(restored_small, restored_head, config)
        restored_opt.load_state_dict(saved_opt)
        self.assertTrue(v3.tree_equal(saved_opt, v3.cpu_optimizer_state(restored_opt)))
        self.update(small, head, optimizer, row, inputs)
        self.update(restored_small, restored_head, restored_opt, row, inputs)
        self.assertTrue(v3.tree_equal(v3.state_cpu(small), v3.state_cpu(restored_small)))
        self.assertTrue(torch.equal(head, restored_head))
        self.assertTrue(v3.tree_equal(v3.cpu_optimizer_state(optimizer), v3.cpu_optimizer_state(restored_opt)))

    def test_pilot_receipt_recovery_after_committed_update_is_idempotent(self):
        small, head, config, optimizer, row, inputs = self.fixture()
        self.update(small, head, optimizer, row, inputs)
        with tempfile.TemporaryDirectory(prefix="internal-m-pilot-commit-") as temporary:
            root = Path(temporary)
            authority = root / "authority.json"
            v3.write(authority, {"synthetic": True})
            arm = "PRE_REAL"
            checkpoint = root / arm / "checkpoint.pt"
            record = root / arm / "steps/0001.json"
            v3.write(record, dict(adapter_parameter_change_max=.1, head_parameter_change_max=.1,
                                 gradient_forward_error=0., token_changed_fraction=.5))
            v3.save(checkpoint, dict(authority=v3.bind(authority), arm=arm, step=1, pending=[],
                                    adapter=v3.state_cpu(small), head=head.detach().clone(),
                                    optimizer=v3.cpu_optimizer_state(optimizer)))
            before = v3.bind(checkpoint)
            with mock.patch.object(v3, "AUTH", authority), mock.patch.object(v3, "OUT", root):
                first = v3.finish_pilot(config, arm, checkpoint, small, head, optimizer, 1.25)
                second = v3.finish_pilot(config, arm, checkpoint, small, head, optimizer, 1.25)
                self.assertEqual(first, second)
                self.assertEqual(first["step"], 1)
                self.assertTrue(first["resume_optimizer_exact"])
                self.assertEqual(v3.bind(checkpoint), before)
                self.assertEqual(v3.read(root / "pilot_validation.json"), second)
                with torch.no_grad():
                    head[0] += .01
                with self.assertRaisesRegex(RuntimeError, "HEAD_RELOAD"):
                    v3.finish_pilot(config, arm, checkpoint, small, head, optimizer, 1.25)


class LauncherExecutionTests(unittest.TestCase):
    """Run the actual shell logic, replacing runtime executables only."""

    launcher = ROOT / "slurm/rc_internal_m_learned_use_v3.sbatch"

    def run_launcher(self, code: int, restart: int = 0, requeue_code: int = 0):
        source = self.launcher.read_text()
        with tempfile.TemporaryDirectory(prefix="internal M v3 launcher ") as temporary:
            root = Path(temporary)
            trace = root / "calls.jsonl"
            executable = root / "mock_python"
            executable.write_text(
                "#!/usr/bin/python3\n"
                "import json,os,sys\n"
                "with open(os.environ['MOCK_TRACE'],'a') as f: "
                "f.write(json.dumps(['python']+sys.argv[1:])+'\\n')\n"
                "advance=any(x.startswith('advance-') for x in sys.argv[1:])\n"
                "sys.exit(0 if advance else int(os.environ['MOCK_EXIT']))\n"
            )
            executable.chmod(0o755)
            scontrol = root / "scontrol"
            scontrol.write_text(
                "#!/usr/bin/python3\n"
                "import json,os,sys\n"
                "with open(os.environ['MOCK_TRACE'],'a') as f: "
                "f.write(json.dumps(['scontrol']+sys.argv[1:])+'\\n')\n"
                "sys.exit(int(os.environ.get('MOCK_REQUEUE_EXIT','0')))\n"
            )
            scontrol.chmod(0o755)
            rewritten, replacements = re.subn(
                r"(?m)^PY=.*$", "PY=" + shlex.quote(str(executable)), source
            )
            self.assertEqual(replacements, 1, "launcher must have one bound Python path")
            rewritten = re.sub(
                r"(?m)^export PATH=.*$",
                "export PATH=" + shlex.quote(str(root) + ":/usr/local/bin:/usr/bin:/bin"),
                rewritten,
            )
            rewritten = re.sub(r"(?m)^cd .*$", "cd " + shlex.quote(str(root)), rewritten)
            rewritten = rewritten.replace("/usr/bin/scontrol", shlex.quote(str(scontrol)))
            script = root / "actual launcher body.sh"
            script.write_text(rewritten)
            env = dict(os.environ, MOCK_TRACE=str(trace), MOCK_EXIT=str(code),
                       MOCK_REQUEUE_EXIT=str(requeue_code), SLURM_JOB_ID="SYNTHETIC_JOB_7001",
                       SLURM_RESTART_COUNT=str(restart))
            completed = subprocess.run(
                ["/usr/bin/bash", str(script), "train", "PRE_REAL"],
                env=env, capture_output=True, text=True, timeout=10,
            )
            calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            return completed, calls

    def test_success_executes_with_literal_arm_argument(self):
        result, calls = self.run_launcher(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(calls and calls[0][0] == "python")
        self.assertIn("train", calls[0])
        self.assertIn("PRE_REAL", calls[0])
        self.assertFalse(any(call[0] == "scontrol" for call in calls))

    def test_partial_and_timeout_requeue_same_job_without_advancing(self):
        for exit_code in (75, 124):
            with self.subTest(exit_code=exit_code):
                result, calls = self.run_launcher(exit_code)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([call for call in calls if call[0] == "scontrol"],
                                 [["scontrol", "requeue", "SYNTHETIC_JOB_7001"]])
                self.assertEqual(sum(call[0] == "python" for call in calls), 1)

    def test_hard_failure_is_not_success_or_requeue(self):
        result, calls = self.run_launcher(2)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(calls), 1)

    def test_requeue_failure_is_not_silently_successful(self):
        result, calls = self.run_launcher(75, requeue_code=7)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(call[0] == "python" for call in calls), 1)

    def test_maximum_restarts_fail_closed(self):
        source = self.launcher.read_text()
        maximum = re.search(r"SLURM_RESTART_COUNT[^\n]*-ge\s+(\d+)", source)
        self.assertIsNotNone(maximum, "an explicit finite requeue ceiling is required")
        result, calls = self.run_launcher(75, restart=int(maximum.group(1)))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == "scontrol" for call in calls))


if __name__ == "__main__":
    unittest.main()
