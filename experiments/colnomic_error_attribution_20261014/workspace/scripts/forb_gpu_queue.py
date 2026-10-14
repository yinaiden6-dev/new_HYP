#!/usr/bin/env python3
"""Persistent FORB GPU queues; run after a server's initial encoding/OCR job.

Only schedules existing validated CLIs. All physical GPU selections are passed
through CUDA_VISIBLE_DEVICES; individual workers always use cuda:0 / gpu:0.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Queue:
    def __init__(self, args):
        self.args = args
        self.data = args.data.resolve()
        self.folder = self.data / 'cache/intermediate/queues'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hostname = socket.gethostname()
        self.name = args.name or f'{socket.gethostname()}_{args.role}_gpu{args.gpus.replace(",", "-")}_{os.getpid()}'
        self.state_path = self.folder / f'{self.name}.json'
        self.log_path = self.folder / f'{self.name}.log'
        self.event_path = self.folder / f'{self.name}.events.jsonl'
        self.manifest = read_json(self.data / 'cache/manifest.json')
        self.settings = read_json(self.data / 'baseline_config.json')
        self.identity = dict(manifest_sha256=sha256(self.data / 'cache/manifest.json'),
                             configuration_sha256=sha256(self.data / 'baseline_config.json'))
        unique = {}
        for row in self.manifest['photos'] + self.manifest['references']:
            unique.setdefault(row['sha256'], row)
        self.rows = list(unique.values())
        self.task_size = self.settings['encoding_task_size']
        self.chunk_size = self.settings['score_chunk_size']
        self.ocr_cache = {}
        self.children = []
        self.state = dict(name=self.name, hostname=socket.gethostname(), pid=os.getpid(),
                          role=args.role, first_model=args.first_model, physical_gpus=args.gpus,
                          identity=self.identity, status='starting', started_unix=time.time(),
                          log=str(self.log_path), commands_completed=0)
        self.update()

    def update(self, **fields):
        self.state.update(fields, heartbeat_unix=time.time())
        write_json(self.state_path, self.state)

    def event(self, event, **fields):
        item = dict(time_unix=time.time(), event=event, **fields)
        line = json.dumps(item, ensure_ascii=False)
        with self.event_path.open('a') as stream:
            stream.write(line + '\n')
        with self.log_path.open('a') as stream:
            stream.write(line + '\n')
        print(f'{self.name}: {line}', flush=True)

    def environment(self, gpus=None):
        return dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=self.args.gpus if gpus is None else gpus)

    def command(self, module, arguments, ocr=False):
        return ['bash', str(ROOT / 'scripts/forb_python.sh'), *(['--ocr'] if ocr else []),
                '-m', module, '--data', str(self.data), *map(str, arguments)]

    def run_command(self, command, phase, gpus=None):
        selected = self.args.gpus if gpus is None else gpus
        with self.gpu_leases(selected, phase) as descriptors:
            self._run_command(command, phase, selected, descriptors)

    @contextmanager
    def gpu_leases(self, gpus, phase):
        """Hold host-local physical GPU leases for the complete subprocess.

        Every multi-GPU claimant takes numeric order, avoiding cyclic waits.
        Children inherit the descriptors so killing only the scheduler cannot
        expose an occupied GPU to another scheduler while its child still runs.
        """
        devices = sorted({int(token) for token in gpus.split(',') if token})
        with ExitStack() as stack:
            descriptors = []
            for gpu in devices:
                path = self.folder / 'gpu_leases' / f'{self.hostname}_gpu_{gpu}.lock'
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(path.open('a'))
                while True:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        self.nap(f'{phase}:waiting_for_{self.hostname}_gpu_{gpu}_lease')
                descriptors.append(stream.fileno())
            yield descriptors

    def _run_command(self, command, phase, gpus, descriptors):
        self.update(status='running', phase=phase, command=command, command_started_unix=time.time())
        self.event('command_started', phase=phase, command=shlex.join(command))
        with self.log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment(gpus),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(descriptors))
            self.children.append(process)
            self.update(child_pid=process.pid)
            try:
                while process.poll() is None:
                    time.sleep(2)
                    self.update()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.children.remove(process)
        code = process.returncode
        self.update(last_returncode=code, child_pid=None,
                    commands_completed=self.state['commands_completed'] + 1)
        self.event('command_finished', phase=phase, returncode=code)
        if code:
            raise subprocess.CalledProcessError(code, command)

    def forb(self, model, action, *extra, gpus=None):
        self.run_command(self.command('evaluation.forb', ['--model', model, '--' + action,
            '--device', 'cpu' if model == 'bm25' else 'cuda:0', '--worker', self.name, *extra]),
            f'{model}:{action}', gpus=gpus)

    def nap(self, phase):
        self.update(status='waiting', phase=phase)
        time.sleep(self.args.poll_seconds)

    def encoding_status(self, model):
        base = self.data / 'cache/intermediate/encoding' / model
        missing, ready = [], 0
        for first in range(0, len(self.rows), self.task_size):
            marker = base / f'{first:06d}.done.json'
            path = base / f'{first:06d}.pt'
            if not marker.exists():
                missing.append(base / f'{first:06d}.lock')
                continue
            value = read_json(marker)
            expected = [r['sha256'] for r in self.rows[first:first + self.task_size]]
            if value.get('image_sha256') != expected or not path.exists() or path.stat().st_size != value.get('bytes'):
                raise ValueError(f'Invalid completed encoding marker: {marker}')
            ready += len(expected)
        return dict(complete=not missing, done=ready, total=len(self.rows)), missing

    def score_status(self, model):
        base = self.data / 'cache/scores' / model
        missing, done = [], 0
        count = len(self.manifest['photos'])
        for first in range(0, count, self.chunk_size):
            stop = min(first + self.chunk_size, count)
            stem = f'{first:06d}-{stop:06d}'
            if (base / 'chunks' / (stem + '.pt')).is_file():
                done += 1
            else:
                missing.append(base / 'locks' / (stem + '.lock'))
        return dict(complete=not missing, done=done, total=(count + self.chunk_size - 1) // self.chunk_size), missing

    @staticmethod
    def available(locks):
        for path in locks:
            with claim(path) as acquired:
                if acquired:
                    return True
        return False

    def banks_ready(self, model):
        folder = self.data / 'cache/image_embeddings' / model
        return all((folder / side / 'index.json').is_file() for side in ('photo', 'ref'))

    def receipt(self, model):
        return self.folder / f'completed_{model}.json'

    def finalized(self, model):
        path = self.receipt(model)
        if not path.exists():
            return False
        record = read_json(path)
        if record.get('identity') != self.identity:
            raise ValueError(f'Stale queue completion receipt: {path}')
        mode = 'bm25' if model == 'bm25' else 'image'
        folder = self.data / 'results' / model / mode
        if not all((folder / name).is_file() for name in ('metrics.json', 'config.json', 'predictions.jsonl')):
            raise ValueError(f'Missing finalized results: {folder}')
        return True

    def try_finalize(self, model):
        if self.finalized(model):
            return True
        if not self.score_status(model)[0]['complete']:
            return False
        with claim(self.folder / f'finalize_{model}.lock') as acquired:
            if not acquired:
                return False
            if not self.finalized(model):
                self.forb(model, 'finalize', gpus='')
                write_json(self.receipt(model), dict(identity=self.identity, model=model,
                    completed_unix=time.time(), queue=self.name,
                    score_metadata_sha256=sha256(self.data / 'cache/scores' / model / 'metadata.json')))
        return True

    def ensure_embeddings(self, model):
        while True:
            progress, missing = self.encoding_status(model)
            self.update(encoding_progress={model: progress})
            if progress['complete']:
                if self.merge_embeddings_once(model):
                    return
                self.nap(f'{model}:waiting_for_bank_merge')
                continue
            if self.available(missing):
                self.forb(model, 'encode')
            else:
                self.nap(f'{model}:waiting_for_busy_encoding_tasks')

    def merge_embeddings_once(self, model):
        """Prevent simultaneous helpers from serially rewriting huge banks."""
        with claim(self.folder / f'merge_{model}.lock') as acquired:
            if not acquired:
                return False
            if not self.banks_ready(model):
                self.forb(model, 'merge-embeddings', gpus='')
            return True

    def ensure_scores(self, model, help_ocr=False):
        while not self.finalized(model):
            progress, missing = self.score_status(model)
            self.update(scoring_progress={model: progress})
            if progress['complete']:
                if self.try_finalize(model):
                    return
            elif self.available(missing):
                self.forb(model, 'score')
                continue
            if help_ocr and self.ocr_one_task():
                continue
            self.nap(f'{model}:waiting_for_busy_scoring_or_finalize')

    def ocr_status(self):
        folder = self.data / 'cache/intermediate/ocr'
        plan_path = folder / 'plan.json'
        if not plan_path.exists():
            return dict(complete=False, done=0, total=None), [], 32
        plan = read_json(plan_path)
        missing, done = [], 0
        for index, hashes in enumerate(plan['tasks']):
            path = folder / 'tasks' / f'{index:06d}.json'
            if not path.exists():
                missing.append(path.with_suffix('.lock'))
                continue
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns)
            cached = self.ocr_cache.get(index)
            if cached is None or cached[0] != identity:
                record = read_json(path)
                if record.get('task_index') != index or record.get('image_hashes') != hashes:
                    raise ValueError(f'Invalid OCR task partition: {path}')
                complete = record.get('complete') is True
                if complete and set(record.get('entries', {})) != set(hashes):
                    raise ValueError(f'Invalid complete OCR task: {path}')
                self.ocr_cache[index] = identity, complete
            if self.ocr_cache[index][1]:
                done += 1
            else:
                missing.append(path.with_suffix('.lock'))
        return dict(complete=not missing, done=done, total=len(plan['tasks'])), missing, plan['task_size']

    def ocr_one_task(self):
        progress, missing, task_size = self.ocr_status()
        self.update(ocr_progress=progress)
        if progress['complete'] or (progress['total'] is not None and not self.available(missing)):
            return False
        self.run_command(self.command('evaluation.forb_ocr', ['--device', 'gpu:0',
            '--worker', self.name, '--task-size', task_size, '--max-tasks', 1], ocr=True), 'ocr:one_task')
        return True

    def try_finish_ocr_bm25(self):
        if self.finalized('bm25'):
            return True
        progress, _, task_size = self.ocr_status()
        if not progress['complete']:
            return False
        with claim(self.folder / 'ocr_and_bm25_finalize.lock') as acquired:
            if not acquired:
                return False
            if self.finalized('bm25'):
                return True
            metadata = self.data / 'cache/ocr/metadata.json'
            if not metadata.exists() or not read_json(metadata).get('complete'):
                # Cache merging uses only the standard library. Keep it on the
                # Python 3.12 runtime used by scoring/verification; the legacy
                # OCR runtime crashed during a full-bank merge on host43.
                self.run_command(self.command('evaluation.forb_ocr', ['--merge', '--task-size', task_size]),
                                 'ocr:merge', gpus='')
            while not self.score_status('bm25')[0]['complete']:
                _, missing = self.score_status('bm25')
                if self.available(missing):
                    self.forb('bm25', 'score', gpus='')
                else:
                    self.nap('bm25:waiting_for_busy_scoring_tasks')
            return self.try_finalize('bm25')

    def assist_colnomic_and_ocr(self):
        while True:
            # Finish the fast CPU baseline before every freed queue enters a
            # long Col7 encoding run. This coordinator holds no GPU lease, so
            # its peer can still claim both physical GPUs while it finalizes.
            if not self.finalized('bm25') and self.ocr_status()[0]['complete']:
                self.try_finish_ocr_bm25()
            if self.banks_ready('colnomic7b') and not self.finalized('colnomic7b'):
                self.ensure_scores('colnomic7b', help_ocr=True)
                continue
            if self.assist_colnomic_pair_encoding():
                continue
            if self.ocr_one_task():
                continue
            self.try_finish_ocr_bm25()
            if all(self.finalized(model) for model in MODELS):
                return
            self.nap('waiting_for_colnomic_banks_or_remaining_baseline_results')

    def assist_colnomic_pair_encoding(self):
        """Join Col7 encoding with an adjacent even/odd physical GPU pair.

        Single queues finish Qwen first; OCR queues finish global OCR first.
        A pair leader keeps its pair claim while waiting for either physical
        GPU. The peer then waits instead of repeatedly starting another OCR
        task, so both GPUs eventually become available for the pair leader.
        """
        if self.args.role == 'single':
            if not self.finalized('Qwen3-VL-Embedding-2B'):
                return False
        elif self.args.role == 'ocr':
            if not self.ocr_status()[0]['complete']:
                return False
        else:
            return False
        gpu = int(self.args.gpus)
        first = gpu - gpu % 2
        pair = f'{first},{first + 1}'
        path = self.folder / 'gpu_leases' / f'{self.hostname}_pair_{first}_{first + 1}.lock'
        with claim(path) as acquired:
            if not acquired:
                self.nap(f'colnomic7b:pair_{pair}_claimed_by_peer')
                return True
            progress, missing = self.encoding_status('colnomic7b')
            self.update(encoding_progress={'colnomic7b': progress})
            if progress['complete']:
                if not self.banks_ready('colnomic7b'):
                    if not self.merge_embeddings_once('colnomic7b'):
                        self.nap('colnomic7b:waiting_for_bank_merge')
                    return True
                return False
            if not self.available(missing):
                return False
            self.event('pair_encoding_assistance_started', physical_gpus=pair)
            self.forb('colnomic7b', 'encode', gpus=pair)
            if self.encoding_status('colnomic7b')[0]['complete']:
                self.merge_embeddings_once('colnomic7b')
            self.event('pair_encoding_assistance_finished', physical_gpus=pair)
            return True

    def pair_children(self):
        commands = []
        for gpu in self.args.gpus.split(','):
            commands.append(['bash', str(ROOT / 'scripts/forb_python.sh'), str(Path(__file__).resolve()),
                '--role', 'ocr', '--gpus', gpu, '--data', str(self.data),
                '--name', self.name + '_score_gpu' + gpu, '--poll-seconds', str(self.args.poll_seconds)])
        with self.log_path.open('a') as log:
            for command in commands:
                child = subprocess.Popen(command, cwd=ROOT, env=self.environment(), stdout=log,
                                         stderr=subprocess.STDOUT)
                self.children.append(child)
                self.event('single_gpu_child_started', pid=child.pid, command=shlex.join(command))
            try:
                while True:
                    codes = [child.poll() for child in self.children]
                    self.update(status='running', phase='pair_children',
                                child_pids=[p.pid for p in self.children], child_returncodes=codes)
                    for command, code in zip(commands, codes, strict=True):
                        if code is not None and code != 0:
                            raise subprocess.CalledProcessError(code, command)
                    if all(code is not None for code in codes):
                        return
                    time.sleep(5)
            finally:
                for child in self.children:
                    if child.poll() is None:
                        child.terminate()
                for child in self.children:
                    if child.poll() is None:
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                self.children.clear()

    def run(self):
        self.event('queue_started', role=self.args.role, physical_gpus=self.args.gpus)
        if self.args.role == 'single':
            for model in (self.args.first_model, 'Qwen3-VL-Embedding-2B'):
                if not self.finalized(model):
                    self.ensure_embeddings(model)
                    self.ensure_scores(model)
            self.assist_colnomic_and_ocr()
        elif self.args.role == 'colnomic-pair':
            self.ensure_embeddings('colnomic7b')
            self.pair_children()
        else:
            self.assist_colnomic_and_ocr()
        self.update(status='complete', phase='all_baselines_complete', finished_unix=time.time())
        self.event('queue_finished', returncode=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=('single', 'colnomic-pair', 'ocr'))
    parser.add_argument('--first-model', choices=('dinov3', 'siglip2'))
    parser.add_argument('--gpus', required=True, help='Physical GPU number, or comma-separated pair')
    parser.add_argument('--data', type=Path, default=ROOT / 'other_datasets/FORB')
    parser.add_argument('--name', help='Unique queue name for persistent logs/state')
    parser.add_argument('--poll-seconds', type=float, default=20.)
    args = parser.parse_args()
    physical = args.gpus.split(',')
    if any(not token.isdigit() for token in physical) or len(set(physical)) != len(physical):
        parser.error('--gpus must contain distinct physical GPU numbers')
    if len(physical) != (2 if args.role == 'colnomic-pair' else 1):
        parser.error('colnomic-pair needs two GPUs; single and ocr need one')
    if args.role == 'single' and args.first_model is None:
        parser.error('--first-model is required for the single role')
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error('--poll-seconds must be in (0, 60]')
    if args.name and (Path(args.name).name != args.name or args.name in ('.', '..')):
        parser.error('--name must be a plain filename component')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    queue = Queue(args)
    try:
        queue.run()
    except BaseException as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else (
            130 if isinstance(error, KeyboardInterrupt) else 1)
        code = 128 - code if code < 0 else code
        queue.update(status='failed', error=f'{type(error).__name__}: {error}',
                     returncode=code, finished_unix=time.time())
        queue.event('queue_failed', error=f'{type(error).__name__}: {error}', returncode=code)
        return code
    return 0


if __name__ == '__main__':
    sys.exit(main())
