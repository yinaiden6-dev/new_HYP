"""Restartable, shared-filesystem PP-OCRv6 extraction for the official FORB subset.

Workers claim small immutable task lists with advisory file locks. A worker can
be stopped at any point; a subsequent worker resumes its last saved task. Image
bytes are deduplicated only for inference, never in the final retrieval rows.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import socket
import time
import unicodedata

from .data import digest, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/FORB'
SCHEMA = 'forb.ocr.v1'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path, blocking=False):
    """Keep the descriptor alive only while actually operating on its task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_inputs(data):
    data = Path(data).resolve()
    manifest_path = data / 'cache/manifest.json'
    manifest = read_json(manifest_path)
    configuration = dict(config=read_json(ROOT / 'configs/ppocrv6.json'),
                         models=read_json(ROOT / 'configs/ppocrv6_models.json'),
                         implementation=file_hash(ROOT / 'evaluation/ocr.py'),
                         extraction_implementation=file_hash(__file__))
    if read_json(data / 'baseline_config.json')['ocr'] != configuration['config']:
        raise ValueError('FORB OCR configuration differs from released PP-OCRv6 recipe')
    unique = {}
    for population in ('photos', 'references'):
        rows = manifest[population]
        if not rows or [r['index'] for r in rows] != list(range(len(rows))):
            raise ValueError(f'Invalid or empty ordered inventory: {population}')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate IDs in {population}')
        for row in rows:
            path = Path(row['image_path'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Expected dataset-relative image path: {path}')
            sha = row['sha256']
            if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError(f'Invalid image hash: {row["id"]}')
            unique.setdefault(sha, row)
    sources = dict(manifest_sha256=file_hash(manifest_path),
                   ocr_config_sha256=file_hash(ROOT / 'configs/ppocrv6.json'),
                   ocr_models_sha256=file_hash(ROOT / 'configs/ppocrv6_models.json'))
    signature = digest(dict(configuration=configuration, sources=sources))
    return dict(data=data, manifest=manifest, configuration=configuration,
                sources=sources, signature=signature, unique=unique)


def task_plan(inputs, task_size, create=True):
    directory = inputs['data'] / 'cache/intermediate/ocr'
    path = directory / 'plan.json'
    image_hashes = list(inputs['unique'])
    if create:
        with lock(directory / 'plan.lock', blocking=True):
            if not path.exists():
                write_json(path, dict(schema=SCHEMA, signature=inputs['signature'],
                    task_size=task_size, unique_images=len(image_hashes),
                    tasks=[image_hashes[i:i + task_size]
                           for i in range(0, len(image_hashes), task_size)]))
    plan = read_json(path)
    if plan.get('schema') != SCHEMA or plan.get('signature') != inputs['signature']:
        raise ValueError(f'Stale OCR task plan: {path}')
    if plan.get('task_size') != task_size:
        raise ValueError(f'Task size is pinned to {plan.get("task_size")}, requested {task_size}')
    expected_tasks = [image_hashes[i:i + task_size] for i in range(0, len(image_hashes), task_size)]
    if plan.get('tasks') != expected_tasks or plan.get('unique_images') != len(image_hashes):
        raise ValueError(f'Invalid OCR task partition: {path}')
    return plan


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def validate_entry(entry, sha, inputs):
    """Reject malformed, truncated or configuration-incompatible cached OCR."""
    if entry.get('image_sha256') != sha:
        raise ValueError(f'OCR entry has wrong image hash: {sha}')
    if not isinstance(entry.get('width'), int) or entry['width'] <= 0 or \
            not isinstance(entry.get('height'), int) or entry['height'] <= 0:
        raise ValueError(f'OCR entry has invalid dimensions: {sha}')
    if len(entry.get('stored_size', [])) != 2 or \
            not isinstance(entry.get('exif_orientation'), int) or \
            entry.get('coordinate_system') != 'EXIF-transposed image, pixel centers, top-left origin':
        raise ValueError(f'OCR entry is missing image geometry: {sha}')
    lines = entry.get('lines')
    if not isinstance(lines, list):
        raise ValueError(f'OCR entry has no line records: {sha}')
    minimum = inputs['configuration']['config']['minimum_retained_confidence']
    for line in lines:
        confidence = line.get('confidence')
        polygon, bbox = line.get('polygon', []), line.get('bbox', [])
        if not isinstance(line.get('text'), str) or not line['text'].strip() or \
                line.get('normalized_text') != normalized(line['text']) or \
                not isinstance(confidence, (float, int)) or not minimum <= confidence <= 1 or \
                len(polygon) < 3 or any(len(p) != 2 for p in polygon) or len(bbox) != 4 or \
                not all(isinstance(x, (float, int)) and math.isfinite(x)
                        for x in bbox + [x for p in polygon for x in p]):
            raise ValueError(f'OCR entry has invalid text or coordinates: {sha}')
    if entry.get('text') != '\n'.join(x['text'] for x in lines) or \
            entry.get('normalized_text') != '\n'.join(x['normalized_text'] for x in lines):
        raise ValueError(f'OCR line/text mismatch: {sha}')
    elapsed = entry.get('inference_seconds')
    if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0 or \
            not isinstance(entry.get('worker'), str) or not entry['worker']:
        raise ValueError(f'OCR entry lacks valid timing/worker: {sha}')


def task_path(inputs, index):
    return inputs['data'] / 'cache/intermediate/ocr/tasks' / f'{index:06d}.json'


def read_task(inputs, index, hashes):
    path = task_path(inputs, index)
    if not path.exists():
        return dict(schema=SCHEMA, signature=inputs['signature'], task_index=index,
                    image_hashes=hashes, entries={}, entry_sha256={}, sessions=[], complete=False)
    task = read_json(path)
    if task.get('schema') != SCHEMA or task.get('signature') != inputs['signature'] or \
            task.get('task_index') != index or task.get('image_hashes') != hashes:
        raise ValueError(f'Stale or mismatched OCR task: {path}')
    entries = task.get('entries', {})
    if not isinstance(entries, dict) or not set(entries) <= set(hashes) or \
            set(task.get('entry_sha256', {})) != set(entries):
        raise ValueError(f'Invalid OCR task entries: {path}')
    for sha, entry in entries.items():
        if digest(entry) != task['entry_sha256'][sha]:
            raise ValueError(f'OCR entry checksum mismatch: {path} / {sha}')
        validate_entry(entry, sha, inputs)
    if bool(task.get('complete')) != (set(entries) == set(hashes)):
        raise ValueError(f'OCR completion marker mismatch: {path}')
    return task


def save_task(inputs, task):
    task['complete'] = set(task['entries']) == set(task['image_hashes'])
    task['updated_at'] = timestamp()
    write_json(task_path(inputs, task['task_index']), task)


def run(data, device, worker, task_size=32, save_every=4, max_tasks=None, max_images=None):
    inputs = load_inputs(data)
    plan = task_plan(inputs, task_size)
    engine = None
    finished_tasks = generated = busy = 0
    worker = worker or f'{socket.gethostname()}:{device}:pid{os.getpid()}'
    print(f'OCR worker={worker} device={device} unique_images={plan["unique_images"]} '
          f'tasks={len(plan["tasks"])} signature={inputs["signature"]}', flush=True)
    for index, hashes in enumerate(plan['tasks']):
        path = task_path(inputs, index)
        with lock(path.with_suffix('.lock')) as acquired:
            if not acquired:
                busy += 1
                continue
            task = read_task(inputs, index, hashes)
            if task['complete']:
                continue
            session = dict(worker=worker, host=socket.gethostname(), pid=os.getpid(),
                           device=device, started_at=timestamp(), images=0,
                           inference_seconds=0., loading_seconds=0.)
            task['sessions'].append(session)
            if engine is None:
                from .ocr import OCR
                started = time.monotonic()
                engine = OCR(ROOT, device)
                session['loading_seconds'] = time.monotonic() - started
            try:
                for sha in hashes:
                    if sha in task['entries']:
                        continue
                    row = inputs['unique'][sha]
                    image_path = inputs['data'] / row['image_path']
                    if file_hash(image_path) != sha:
                        raise ValueError(f'Image content differs from manifest: {image_path}')
                    entry, elapsed = engine.recognize(dict(path=row['image_path'], sha256=sha), image_path)
                    entry.update(image_path=row['image_path'], inference_seconds=elapsed,
                                 worker=worker, device=device, completed_at=timestamp())
                    validate_entry(entry, sha, inputs)
                    task['entries'][sha] = entry
                    task['entry_sha256'][sha] = digest(entry)
                    session['images'] += 1
                    session['inference_seconds'] += elapsed
                    generated += 1
                    if session['images'] % save_every == 0:
                        save_task(inputs, task)
                    print(f'OCR {worker} task={index:06d} {len(task["entries"])}/{len(hashes)} '
                          f'image={row["id"]} lines={len(entry["lines"])} seconds={elapsed:.2f}', flush=True)
                    if max_images is not None and generated >= max_images:
                        break
            finally:
                session['finished_at'] = timestamp()
                save_task(inputs, task)
            finished_tasks += int(task['complete'])
        if (max_tasks is not None and finished_tasks >= max_tasks) or \
                (max_images is not None and generated >= max_images):
            break
    summary = dict(worker=worker, generated_images=generated,
                   completed_tasks=finished_tasks, locked_tasks_skipped=busy)
    print(summary, flush=True)
    return summary


def collect(inputs, plan):
    entries, tasks, incomplete = {}, [], []
    for index, hashes in enumerate(plan['tasks']):
        # Atomic task replacement allows consistent reads without holding a
        # worker's lock. Merge is permitted only after every task is complete.
        task = read_task(inputs, index, hashes)
        if not task['complete']:
            incomplete.append(dict(task=index, done=len(task['entries']), total=len(hashes)))
            continue
        entries.update(task['entries'])
        tasks.append(dict(task=index, file=str(task_path(inputs, index).relative_to(inputs['data'])),
                          sha256=file_hash(task_path(inputs, index)), sessions=task['sessions']))
    if incomplete:
        pending = sum(t['total'] - t['done'] for t in incomplete)
        raise ValueError(f'OCR incomplete: {len(incomplete)} tasks / {pending} unique images remain; '
                         f'first tasks: {incomplete[:8]}')
    if set(entries) != set(inputs['unique']):
        raise ValueError('OCR task union does not cover the manifest')
    return entries, tasks


def verify_images(inputs):
    verified = set()
    for population in ('photos', 'references'):
        for row in inputs['manifest'][population]:
            key = (row['image_path'], row['sha256'])
            if key not in verified:
                path = inputs['data'] / row['image_path']
                if file_hash(path) != row['sha256']:
                    raise ValueError(f'Image content differs from manifest: {path}')
                verified.add(key)
    return len(verified)


def merge(data, task_size=32):
    inputs = load_inputs(data)
    plan = task_plan(inputs, task_size, create=False)
    directory = inputs['data'] / 'cache/ocr'
    with lock(directory / 'merge.lock') as acquired:
        if not acquired:
            raise RuntimeError('Another worker is already merging OCR caches')
        unique, tasks = collect(inputs, plan)
        verified = verify_images(inputs)
        files, populations = {}, {}
        for side, population in (('photo', 'photos'), ('ref', 'references')):
            rows = inputs['manifest'][population]
            entries = {r['id']: dict(unique[r['sha256']], id=r['id'], index=r['index'],
                                    domain=r['domain'], image_path=r['image_path']) for r in rows}
            population_hashes = {r['sha256'] for r in rows}
            seconds = sum(unique[sha]['inference_seconds'] for sha in population_hashes)
            empty_ids = [r['id'] for r in rows if not entries[r['id']]['text'].strip()]
            bank = dict(schema=SCHEMA, signature=inputs['signature'],
                        configuration=inputs['configuration'], sources=inputs['sources'],
                        population=side, complete=True, ids=[r['id'] for r in rows],
                        count=len(rows), entries=entries, empty_ids=empty_ids,
                        image_path_base='FORB directory', unique_images=len(population_hashes),
                        inference_seconds=seconds,
                        timing_note='Sum over unique image bytes in this population; images shared between populations must not be counted twice.')
            path = directory / f'{side}.json'
            write_json(path, bank)
            files[side] = file_hash(path)
            populations[side] = dict(images=len(rows), unique_images=len(population_hashes),
                                    lines=sum(len(e['lines']) for e in entries.values()),
                                    inference_seconds=seconds, empty_ids=empty_ids,
                                    image_sha256=[r['sha256'] for r in rows])
        workers = {}
        for entry in unique.values():
            stats = workers.setdefault(entry['worker'], dict(images=0, inference_seconds=0.))
            stats['images'] += 1
            stats['inference_seconds'] += entry['inference_seconds']
        metadata = dict(schema=SCHEMA, signature=inputs['signature'], complete=True,
                        configuration=inputs['configuration'], sources=inputs['sources'],
                        source=file_hash(__file__), files=files, populations=populations,
                        unique_images=len(unique), verified_image_paths=verified,
                        inference_seconds=sum(e['inference_seconds'] for e in unique.values()),
                        loading_seconds=sum(s['loading_seconds'] for t in tasks for s in t['sessions']),
                        workers=workers, tasks=tasks, merged_at=timestamp(),
                        deduplication='OCR computation by image SHA-256; all official query/reference IDs and order preserved')
        write_json(directory / 'metadata.json', metadata)
        print(f'OCR merged: {len(unique)} unique images, '
              f'{len(inputs["manifest"]["photos"])} queries, '
              f'{len(inputs["manifest"]["references"])} references', flush=True)
        return metadata


def verify_cache(data, task_size=32):
    inputs = load_inputs(data)
    plan = task_plan(inputs, task_size, create=False)
    unique, tasks = collect(inputs, plan)
    directory = inputs['data'] / 'cache/ocr'
    metadata = read_json(directory / 'metadata.json')
    if metadata.get('signature') != inputs['signature'] or not metadata.get('complete') or \
            metadata.get('configuration') != inputs['configuration'] or \
            metadata.get('sources') != inputs['sources'] or metadata.get('tasks') != tasks:
        raise ValueError('Stale or incomplete OCR metadata')
    verify_images(inputs)
    for side, population in (('photo', 'photos'), ('ref', 'references')):
        path = directory / f'{side}.json'
        if file_hash(path) != metadata['files'][side]:
            raise ValueError(f'OCR bank file checksum mismatch: {path}')
        bank = read_json(path)
        rows = inputs['manifest'][population]
        ids = [r['id'] for r in rows]
        if bank.get('signature') != inputs['signature'] or not bank.get('complete') or \
                bank.get('configuration') != inputs['configuration'] or bank.get('sources') != inputs['sources'] or \
                bank.get('ids') != ids or bank.get('count') != len(ids) or list(bank.get('entries', {})) != ids:
            raise ValueError(f'Invalid OCR population or ordering: {path}')
        for row in rows:
            expected = dict(unique[row['sha256']], id=row['id'], index=row['index'],
                            domain=row['domain'], image_path=row['image_path'])
            if bank['entries'][row['id']] != expected:
                raise ValueError(f'Expanded OCR row differs from its task: {row["id"]}')
        empty_ids = [r['id'] for r in rows if not bank['entries'][r['id']]['text'].strip()]
        if bank.get('empty_ids') != empty_ids or metadata['populations'][side]['empty_ids'] != empty_ids or \
                metadata['populations'][side]['image_sha256'] != [r['sha256'] for r in rows]:
            raise ValueError(f'OCR empties/image hashes differ: {path}')
    print(f'OCR cache verified: {len(unique)} unique images, signature={inputs["signature"]}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--device', default='gpu:0')
    parser.add_argument('--worker', help='Stable host/GPU worker name; defaults to hostname/device/PID')
    parser.add_argument('--task-size', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=4)
    parser.add_argument('--max-tasks', type=int, help='Optional bounded worker run for smoke checks')
    parser.add_argument('--max-images', type=int, help='Optional bounded inference count; progress remains resumable')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--merge', action='store_true', help='Require all tasks complete and expand official rows')
    actions.add_argument('--cache-only', action='store_true', help='Verify final cache without importing OCR or using a GPU')
    args = parser.parse_args()
    if args.task_size < 1 or args.save_every < 1 or \
            (args.max_tasks is not None and args.max_tasks < 1) or \
            (args.max_images is not None and args.max_images < 1):
        parser.error('Task size, save interval and optional maximum tasks must be positive')
    if args.merge:
        merge(args.data, args.task_size)
    elif args.cache_only:
        verify_cache(args.data, args.task_size)
    else:
        run(args.data, args.device, args.worker, args.task_size, args.save_every, args.max_tasks, args.max_images)


if __name__ == '__main__':
    main()
