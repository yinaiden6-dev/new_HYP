"""CPU completion watcher regressions: one-time audits, restarts and failures."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/forb_completion_watch.py'
spec = importlib.util.spec_from_file_location('forb_completion_watch', SOURCE)
watch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch_module)


class CompletionWatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name)
        (self.data / 'cache').mkdir()
        watch_module.save_json(self.data / 'cache/manifest.json', {'synthetic': True})
        watch_module.save_json(self.data / 'baseline_config.json', {'frozen': True})
        self.args = argparse.Namespace(data=self.data, poll_seconds=.01, threads=2, retry_failed=False)
        self.watch = watch_module.CompletionWatch(self.args)
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def make_ready(self, model):
        for path in self.watch.artifacts(model):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n')
        score = self.data / 'cache/scores' / model
        watch_module.save_json(score / 'metadata.json', {'scores_sha256': watch_module.file_hash(score / 'scores.npy')})

    def fake_script(self, script, arguments, name):
        self.calls.append((script, arguments, name))
        if script == 'verify_forb.py':
            model = arguments[arguments.index('--model') + 1]
            folder = self.watch.result_folder(model)
            scores = self.data / 'cache/scores' / model
            receipt = dict(model=model, passed=True, cache_only=True, cuda_initialized=False,
                dataset_sha256=self.watch.identity['manifest_sha256'],
                verifier_sha256=watch_module.file_hash(watch_module.ROOT / 'scripts/verify_forb.py'),
                result_files={name: watch_module.file_hash(folder / name)
                              for name in ('config.json', 'metrics.json', 'predictions.jsonl')},
                score_metadata_sha256=watch_module.file_hash(scores / 'metadata.json'),
                score_sha256=watch_module.file_hash(scores / 'scores.npy'))
            watch_module.save_json(folder / 'verification.json', receipt)
            verified = [model for model in watch_module.MODELS
                        if (self.watch.result_folder(model) / 'verification.json').exists()]
            watch_module.save_json(self.data / 'results/verification.json', dict(passed=True,
                all_planned_results_verified=len(verified) == 5, verified_models=verified))
        else:
            self.assertEqual(script, 'summarize_forb.py')
            self.assertEqual(arguments, ['--ready'])
            for name in ('REPORT.md', 'summary.csv', 'timing.csv'):
                (self.data / 'results' / name).write_text('complete report\n')
        return dict(returncode=0, log='synthetic', command=[script, *arguments])

    def test_all_five_are_verified_once_and_restart_does_no_work(self):
        for model in watch_module.MODELS:
            self.make_ready(model)
        with patch.object(self.watch, 'run_script', self.fake_script):
            self.watch.run()
        self.assertEqual(sum(script == 'verify_forb.py' for script, _, _ in self.calls), 5)
        self.assertEqual(sum(script == 'summarize_forb.py' for script, _, _ in self.calls), 5)
        self.assertEqual(self.watch.state['status'], 'complete')
        self.assertTrue(self.watch.state['all_models_verified'])
        resumed = watch_module.CompletionWatch(self.args)
        with patch.object(resumed, 'run_script', side_effect=AssertionError('repeated a completed command')):
            resumed.run()
        self.assertEqual(resumed.state['status'], 'complete')

    def test_incomplete_results_do_not_trigger_audit(self):
        with patch.object(self.watch, 'run_script', side_effect=AssertionError('verified incomplete artifacts')):
            self.watch.verify_ready_model('dinov3')
        self.assertEqual(self.watch.state['models']['dinov3']['status'], 'waiting')

    def test_failed_verification_is_persisted_and_not_retried_implicitly(self):
        self.make_ready('dinov3')
        with patch.object(self.watch, 'run_script', side_effect=subprocess.CalledProcessError(17, ['fake'])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.watch.verify_ready_model('dinov3')
        saved = watch_module.read_json(self.watch.state_path)
        self.assertEqual(saved['models']['dinov3']['status'], 'failed')
        self.assertEqual(saved['models']['dinov3']['returncode'], 17)
        resumed = watch_module.CompletionWatch(self.args)
        with patch.object(resumed, 'run_script', side_effect=AssertionError('implicitly retried failed audit')):
            with self.assertRaisesRegex(ValueError, 'Previous verification attempt'):
                resumed.verify_ready_model('dinov3')
        self.args.retry_failed = True
        with patch.object(resumed, 'run_script', self.fake_script):
            resumed.verify_ready_model('dinov3')
        self.assertEqual(resumed.state['models']['dinov3']['status'], 'verified')

    def test_changed_verified_artifacts_are_rejected_without_recomputation(self):
        self.make_ready('dinov3')
        with patch.object(self.watch, 'run_script', self.fake_script):
            self.watch.verify_ready_model('dinov3')
        (self.watch.result_folder('dinov3') / 'metrics.json').write_text('{"changed": true}\n')
        with patch.object(self.watch, 'run_script', side_effect=AssertionError('silently recomputed changed result')):
            with self.assertRaisesRegex(ValueError, 'Artifacts changed'):
                self.watch.verify_ready_model('dinov3')

    def test_child_processes_hide_all_gpus(self):
        class FinishedChild:
            pid = 123
            returncode = 0

            def poll(self):
                return 0

        with patch.object(watch_module.subprocess, 'Popen', return_value=FinishedChild()) as launch:
            self.watch.run_script('verify_forb.py', ['--model', 'dinov3'], 'fake-cpu')
        self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        command = launch.call_args.args[0]
        self.assertEqual(command[:2], ['bash', str(watch_module.ROOT / 'scripts/forb_python.sh')])


if __name__ == '__main__':
    unittest.main()
