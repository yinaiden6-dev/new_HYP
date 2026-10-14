#!/usr/bin/env python3
"""Watch final FORB artifacts, verify each baseline once on CPU, and export reports."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class CompletionWatch:
    def __init__(self, args):
        self.args = args
        self.data = args.data.resolve()
        self.folder = self.data / 'cache/intermediate'
        self.logs = self.folder / 'completion_logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.folder / 'completion.json'
        self.log_path = self.logs / 'watch.log'
        self.identity = dict(manifest_sha256=file_hash(self.data / 'cache/manifest.json'),
                             baseline_config_sha256=file_hash(self.data / 'baseline_config.json'))
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state.get('identity') != self.identity:
                raise ValueError('completion.json belongs to a different manifest/configuration')
        else:
            self.state = dict(schema='forb.completion.v1', identity=self.identity,
                started_unix=time.time(), status='waiting', models={model: dict(status='waiting') for model in MODELS},
                report=dict(status='waiting'), errors=[])
        self.state.update(pid=os.getpid(), poll_seconds=args.poll_seconds,
                          cpu_only=True, cuda_visible_devices='', watcher_sha256=file_hash(__file__))
        self.save()

    def save(self, **fields):
        self.state.update(fields, heartbeat_unix=time.time())
        save_json(self.state_path, self.state)

    def event(self, message, **details):
        line = json.dumps(dict(time_unix=time.time(), message=message, **details), ensure_ascii=False)
        with self.log_path.open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    def result_folder(self, model):
        return self.data / 'results' / model / ('bm25' if model == 'bm25' else 'image')

    def artifacts(self, model):
        return [self.result_folder(model) / name for name in ('config.json', 'metrics.json', 'predictions.jsonl')] + [
            self.data / 'cache/scores' / model / name for name in ('scores.npy', 'metadata.json', 'manifest.json')]

    def ready(self, model):
        return all(path.is_file() for path in self.artifacts(model))

    def fingerprint(self, model):
        paths = self.artifacts(model) + [self.result_folder(model) / 'verification.json']
        return {str(path.relative_to(self.data)): dict(bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, inode=path.stat().st_ino) for path in paths}

    def validate_verification(self, model, imported=False):
        path = self.result_folder(model) / 'verification.json'
        if not path.exists():
            return False
        value = read_json(path)
        expected = dict(model=model, passed=True, cache_only=True, cuda_initialized=False,
                        dataset_sha256=self.identity['manifest_sha256'],
                        verifier_sha256=file_hash(ROOT / 'scripts/verify_forb.py'))
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError(f'Invalid or stale independent verification: {path}')
        for name in ('config.json', 'metrics.json', 'predictions.jsonl'):
            if value.get('result_files', {}).get(name) != file_hash(self.result_folder(model) / name):
                raise ValueError(f'Verified result has changed: {model}/{name}')
        score_folder = self.data / 'cache/scores' / model
        metadata = read_json(score_folder / 'metadata.json')
        if (value.get('score_metadata_sha256') != file_hash(score_folder / 'metadata.json')
                or value.get('score_sha256') != metadata.get('scores_sha256')):
            raise ValueError(f'Verified score metadata has changed: {model}')
        # A receipt produced outside this watcher has no saved file fingerprint.
        # Hash the matrix once before adopting it, without recomputing scores.
        if imported and file_hash(score_folder / 'scores.npy') != value['score_sha256']:
            raise ValueError(f'Verified score matrix has changed: {model}')
        return True

    def run_script(self, script, arguments, name):
        command = ['bash', str(ROOT / 'scripts/forb_python.sh'), str(ROOT / 'scripts' / script),
                   '--data', str(self.data), *map(str, arguments)]
        log_path = self.logs / f'{name}.log'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(self.args.threads),
                           MKL_NUM_THREADS=str(self.args.threads), OPENBLAS_NUM_THREADS=str(self.args.threads))
        self.save(active_command=command, active_log=str(log_path), active_started_unix=time.time())
        self.event('command_started', name=name, command=command, log=str(log_path))
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            self.save(child_pid=child.pid)
            try:
                while child.poll() is None:
                    time.sleep(2)
                    self.save()
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        code = child.returncode
        self.save(last_returncode=code, active_command=None, child_pid=None)
        self.event('command_finished', name=name, returncode=code)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return dict(command=command, returncode=code, log=str(log_path), finished_unix=time.time())

    def mark_verified(self, model, reused=False):
        if not self.validate_verification(model, imported=reused):
            raise ValueError(f'Verifier returned without a verification receipt: {model}')
        path = self.result_folder(model) / 'verification.json'
        item = self.state['models'][model]
        item.update(status='verified', verified_unix=time.time(), reused_existing_verification=reused,
                    verification_path=str(path.relative_to(self.data)), verification_sha256=file_hash(path),
                    artifact_fingerprint=self.fingerprint(model))
        self.save()
        self.event('model_verified', model=model, reused_existing_verification=reused)

    def check_saved_success(self, model):
        item = self.state['models'][model]
        if item.get('status') != 'verified':
            return False
        if not self.ready(model) or self.fingerprint(model) != item.get('artifact_fingerprint'):
            raise ValueError(f'Artifacts changed after successful verification: {model}')
        receipt = read_json(self.result_folder(model) / 'verification.json')
        if receipt.get('verifier_sha256') != file_hash(ROOT / 'scripts/verify_forb.py'):
            raise ValueError(f'Verifier source changed after successful verification: {model}')
        return True

    def verify_ready_model(self, model):
        item = self.state['models'][model]
        if self.check_saved_success(model):
            return
        if not self.ready(model):
            item.update(status='waiting', final_artifacts_ready=False)
            return
        item['final_artifacts_ready'] = True
        receipt = self.result_folder(model) / 'verification.json'
        if receipt.exists():
            self.mark_verified(model, reused=True)
            return
        if item.get('status') in ('failed', 'verifying') and not self.args.retry_failed:
            raise ValueError(f'Previous verification attempt did not finish: {model}; inspect its log, then use --retry-failed')
        item.update(status='verifying', verification_started_unix=time.time(),
                    attempts=item.get('attempts', 0) + 1)
        self.save(status='verifying', active_model=model)
        try:
            result = self.run_script('verify_forb.py', ['--model', model, '--threads', self.args.threads],
                                     'verify_' + model)
            item['command_result'] = result
            self.mark_verified(model)
        except BaseException as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}',
                        returncode=getattr(error, 'returncode', None), failed_unix=time.time())
            self.save()
            raise

    def export_report(self):
        verified = {model: item['verification_sha256'] for model, item in self.state['models'].items()
                    if item.get('status') == 'verified'}
        if not verified:
            return False
        script = ROOT / 'scripts/summarize_forb.py'
        if not script.is_file():
            self.state['report'].update(status='waiting_for_summarize_script')
            return False
        target = dict(verified_receipts=verified, summarizer_sha256=file_hash(script))
        report = self.state['report']
        output_paths = [self.data / 'results' / name for name in ('REPORT.md', 'summary.csv', 'timing.csv')]
        if report.get('status') == 'complete' and report.get('inputs') == target:
            if not all(path.exists() for path in output_paths):
                raise ValueError('A previously generated FORB report is missing')
            return True
        if report.get('status') in ('failed', 'running') and not self.args.retry_failed:
            raise ValueError('Previous report export failed or was interrupted; inspect its log, then use --retry-failed')
        report.update(status='running', inputs=target, started_unix=time.time())
        self.save(status='reporting')
        try:
            result = self.run_script('summarize_forb.py', ['--ready'], 'summarize')
            if not all(path.is_file() for path in output_paths):
                raise ValueError('Summarizer returned without REPORT.md, summary.csv and timing.csv')
            report.update(status='complete', command_result=result, finished_unix=time.time(),
                          files={str(p.relative_to(self.data)): file_hash(p) for p in output_paths})
            self.save()
            return True
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}',
                          returncode=getattr(error, 'returncode', None), failed_unix=time.time())
            self.save()
            raise

    def run(self):
        self.event('watch_started', cpu_only=True, poll_seconds=self.args.poll_seconds)
        while True:
            # Keep a successful per-model receipt, so restarting this watcher
            # never repeats a completed expensive independent arithmetic audit.
            for model in MODELS:
                self.verify_ready_model(model)
                if self.state['models'][model]['status'] == 'verified':
                    self.export_report()
            verified = [model for model in MODELS if self.state['models'][model]['status'] == 'verified']
            self.save(verified_models=verified, pending_models=[m for m in MODELS if m not in verified],
                      all_final_artifacts_ready=all(self.ready(m) for m in MODELS),
                      all_models_verified=len(verified) == len(MODELS), active_model=None)
            if len(verified) == len(MODELS) and self.export_report():
                audit_path = self.data / 'results/verification.json'
                audit = read_json(audit_path)
                if not audit.get('passed') or not audit.get('all_planned_results_verified') or set(audit.get('verified_models', [])) != set(MODELS):
                    raise ValueError('Aggregate verification does not mark all five baselines verified')
                self.save(status='complete', report_generated=True, finished_unix=time.time(), returncode=0,
                          aggregate_verification_sha256=file_hash(audit_path))
                self.event('ALL_FIVE_BASELINES_VERIFIED_AND_REPORTED', returncode=0)
                return
            self.save(status='waiting')
            time.sleep(self.args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'other_datasets/FORB')
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--retry-failed', action='store_true', help='Explicitly retry a failed/interrupted verification or report command')
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60 or args.threads < 1:
        parser.error('--poll-seconds must be in (0,60], and --threads must be positive')
    folder = args.data.resolve() / 'cache/intermediate'
    folder.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    with (folder / '.completion_watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALERT: another FORB completion watcher already owns the lock', file=sys.stderr, flush=True)
            return 2
        watch = None
        try:
            watch = CompletionWatch(args)
            watch.run()
            return 0
        except BaseException as error:
            code = getattr(error, 'returncode', 130 if isinstance(error, KeyboardInterrupt) else 1)
            code = 128 - code if code < 0 else code
            if watch is not None:
                watch.state['errors'].append(dict(error=f'{type(error).__name__}: {error}',
                                                 returncode=code, time_unix=time.time()))
                watch.save(status='failed', returncode=code, failed_unix=time.time())
                watch.event('ALERT: FORB_COMPLETION_FAILED', error=f'{type(error).__name__}: {error}', returncode=code)
            else:
                print(f'ALERT: FORB completion watcher startup failed: {error}', file=sys.stderr, flush=True)
            return code


if __name__ == '__main__':
    sys.exit(main())
