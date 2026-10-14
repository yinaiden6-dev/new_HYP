"""Distributed, resumable image banks for the official FORB subset."""
import fcntl
import gc
import importlib.metadata
import os
from pathlib import Path
import socket
import time

import torch

from .cache import save_tensor, valid_embedding
from .data import digest, file_hash, read_json, write_json
from .dense import DenseEncoder, DENSE_DIMENSIONS
from .encoders import Encoder
from .milk10k import GenericQwenEncoder

ROOT = Path(__file__).resolve().parents[1]


def context(data, model):
    data = Path(data)
    manifest = read_json(data / 'cache/manifest.json')
    settings = read_json(data / 'baseline_config.json')
    spec = settings['models'][model]
    unique = {}
    for row in manifest['photos'] + manifest['references']:
        unique.setdefault(row['sha256'], row)
    rows = list(unique.values())
    sources = {p: file_hash(ROOT / p) for p in ('evaluation/forb_encoding.py', 'evaluation/dense.py',
                                              'evaluation/encoders.py', 'evaluation/milk10k.py')}
    recipe = dict(model=model, spec=spec, source=sources,
                  inputs=[(r['sha256'], r['image_path']) for r in rows],
                  task_size=settings['encoding_task_size'])
    return manifest, settings, rows, recipe, digest(recipe)


def task_paths(data, model, start):
    base = Path(data) / 'cache/intermediate/encoding' / model
    base.mkdir(parents=True, exist_ok=True)
    return base / f'{start:06d}.pt', base / f'{start:06d}.done.json', base / f'{start:06d}.lock'


def is_done(path, marker, signature, rows):
    if not marker.exists():
        return False
    value = read_json(marker)
    if value['signature'] != signature or value['image_sha256'] != [r['sha256'] for r in rows]:
        raise ValueError(f'Stale completed encoding task: {marker}')
    if not path.exists() or path.stat().st_size != value['bytes']:
        raise ValueError(f'Missing or changed completed encoding task: {path}')
    return True


def encode_worker(data, model, device='cuda:0', worker=None, max_tasks=None):
    data = Path(data)
    manifest, settings, rows, recipe, signature = context(data, model)
    task_size = settings['encoding_task_size']
    worker = worker or f'{socket.gethostname()}_{os.getpid()}'
    encoder = None
    completed = 0
    runtime_path = data / 'cache/intermediate/workers' / (worker + '.json')
    runtime = dict(worker=worker, model=model, hostname=socket.gethostname(), pid=os.getpid(),
                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), signature=signature,
                   packages={k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'numpy', 'Pillow')},
                   status='starting', tasks_completed=0, started_unix=time.time())
    write_json(runtime_path, runtime)
    try:
        for first in range(0, len(rows), task_size):
            batch_rows = rows[first:first+task_size]
            path, marker, lock_path = task_paths(data, model, first)
            if is_done(path, marker, signature, batch_rows):
                continue
            with lock_path.open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if is_done(path, marker, signature, batch_rows):
                    continue
                partial = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else {}
                if partial and (partial['signature'] != signature or
                                partial['image_sha256'] != [r['sha256'] for r in batch_rows]):
                    raise ValueError(f'Stale partial encoding: {path}')
                values = partial.get('embeddings', [])
                if isinstance(values, torch.Tensor):
                    values = list(values.unbind())
                if len(values) > len(batch_rows) or not all(valid_embedding(v, model, 'image') for v in values):
                    raise ValueError(f'Invalid partial encoding: {path}')
                elapsed = partial.get('encoding_seconds', 0.)
                contributors = partial.get('workers', [])
                if worker not in contributors:
                    contributors.append(worker)
                for row in batch_rows:
                    if file_hash(data / row['image_path']) != row['sha256']:
                        raise ValueError(f'Image changed: {row["image_path"]}')
                if encoder is None:
                    started = time.monotonic()
                    factory = DenseEncoder if model in ('dinov3', 'siglip2') else (
                        GenericQwenEncoder if model == 'Qwen3-VL-Embedding-2B' else Encoder)
                    encoder = factory(model, settings['models'][model], ROOT, device)
                    runtime['model_loading_seconds'] = time.monotonic() - started
                    runtime['hardware'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                    print(f'{worker}: loaded {model}', flush=True)
                batch_size = settings['dense_batch'] if model in ('dinov3', 'siglip2') else 1
                runtime.update(status='encoding', current_task=first)
                write_json(runtime_path, runtime)
                while len(values) < len(batch_rows):
                    images = [data / r['image_path'] for r in batch_rows[len(values):len(values)+batch_size]]
                    started = time.monotonic()
                    new = encoder.encode_images(images) if model in ('dinov3', 'siglip2') else [encoder.encode_image(images[0])]
                    elapsed += time.monotonic() - started
                    if len(new) != len(images) or not all(valid_embedding(v, model, 'image') for v in new):
                        raise ValueError('Invalid new image embedding')
                    values.extend(new)
                    if len(values) % 8 == 0 or len(values) == len(batch_rows):
                        saved = dict(signature=signature, image_sha256=[r['sha256'] for r in batch_rows],
                                     embeddings=torch.stack(values) if model in DENSE_DIMENSIONS else values,
                                     encoding_seconds=elapsed, workers=contributors)
                        save_tensor(path, saved)
                        print(f'{worker}: {model} unique {first}+{len(values)}/{len(rows)} ({elapsed:.1f}s task)', flush=True)
                write_json(marker, dict(signature=signature, image_sha256=[r['sha256'] for r in batch_rows],
                                        bytes=path.stat().st_size, sha256=file_hash(path),
                                        encoding_seconds=elapsed, workers=contributors))
                completed += 1
                runtime.update(tasks_completed=completed, completed_unix=time.time())
                write_json(runtime_path, runtime)
            if max_tasks is not None and completed >= max_tasks:
                break
        runtime.update(status='idle', finished_unix=time.time())
        write_json(runtime_path, runtime)
    except BaseException as error:
        runtime.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        write_json(runtime_path, runtime)
        raise
    finally:
        if encoder is not None:
            encoder.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def encoding_progress(data, model):
    _, settings, rows, _, signature = context(data, model)
    total = 0
    for first in range(0, len(rows), settings['encoding_task_size']):
        batch = rows[first:first+settings['encoding_task_size']]
        path, marker, _ = task_paths(data, model, first)
        if is_done(path, marker, signature, batch):
            total += len(batch)
    return dict(complete=total == len(rows), unique_done=total, unique_total=len(rows))


def merge_banks(data, model):
    data = Path(data)
    manifest, settings, rows, recipe, signature = context(data, model)
    lock_path = data / 'cache/intermediate/encoding' / model / 'merge.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        indexes = [data / 'cache/image_embeddings' / model / side / 'index.json'
                   for side in ('photo', 'ref')]
        if all(path.exists() for path in indexes):
            for path, key, side in zip(indexes, ('photos', 'references'), ('photo', 'ref'), strict=True):
                record = read_json(path)
                population = manifest[key]
                expected = dict(encoding=signature, population=side,
                                rows=[(r['id'], r['image_path'], r['sha256']) for r in population])
                if record['signature'] != digest(expected) or record['ids'] != [r['id'] for r in population]:
                    raise ValueError(f'Stale merged embedding bank: {path}')
                for shard in record['shards']:
                    source = path.parent / shard['path']
                    if not source.exists() or source.stat().st_size != shard['bytes'] or file_hash(source) != shard['sha256']:
                        raise ValueError(f'Changed merged embedding shard: {source}')
            print(f'Merged {model}: validated existing banks', flush=True)
            return
        if not encoding_progress(data, model)['complete']:
            raise ValueError(f'Encoding incomplete: {model}')
        values = {}
        seconds = 0.
        for first in range(0, len(rows), settings['encoding_task_size']):
            batch = rows[first:first+settings['encoding_task_size']]
            path, marker, _ = task_paths(data, model, first)
            record = read_json(marker)
            if file_hash(path) != record['sha256']:
                raise ValueError(f'Encoding task checksum differs: {path}')
            task = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if task['signature'] != signature or len(task['embeddings']) != len(batch):
                raise ValueError(f'Incomplete encoding task: {path}')
            for row, value in zip(batch, task['embeddings'], strict=True):
                if not valid_embedding(value, model, 'image'):
                    raise ValueError(f'Invalid stored embedding: {row["id"]}')
                values[row['sha256']] = value
            seconds += record['encoding_seconds']
        for side, key in [('photo', 'photos'), ('ref', 'references')]:
            population = manifest[key]
            destination = data / 'cache/image_embeddings' / model / side
            destination.mkdir(parents=True, exist_ok=True)
            ids = [r['id'] for r in population]
            bank_recipe = dict(encoding=signature, population=side,
                               rows=[(r['id'], r['image_path'], r['sha256']) for r in population])
            bank_signature = digest(bank_recipe)
            shards = []
            for first in range(0, len(population), 128):
                subset = population[first:first+128]
                path = destination / f'{first:06d}.pt'
                vectors = [values[r['sha256']] for r in subset]
                shard = dict(signature=bank_signature, ids=ids[first:first+len(subset)],
                             image_sha256=[r['sha256'] for r in subset],
                             embeddings=torch.stack(vectors) if model in DENSE_DIMENSIONS else vectors)
                save_tensor(path, shard)
                shards.append(dict(path=path.name, start=first, stop=first+len(subset),
                                   sha256=file_hash(path), bytes=path.stat().st_size))
            write_json(destination / 'index.json', dict(schema='forb.embeddings.v1', model=model,
                population=side, signature=bank_signature, recipe=bank_recipe, encoding_recipe=recipe,
                ids=ids, image_paths=[r['image_path'] for r in population],
                image_sha256=[r['sha256'] for r in population], shards=shards,
                unique_encoding_seconds=seconds, unique_images=len(rows),
                encoding_time_scope='sum of worker encoding wall times; shared between both banks'))
            print(f'Merged {model}/{side}: {len(population)} rows in {len(shards)} shards', flush=True)
