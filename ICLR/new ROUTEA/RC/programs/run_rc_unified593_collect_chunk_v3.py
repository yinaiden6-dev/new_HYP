#!/usr/bin/env python3
"""Resume one frozen acquisition chunk for the natural ordinals 128..592."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from run_rc_unified128_collect_chunk_v2 import matcher_with_internal_replay

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / 'registry/rc_unified593_long4_authority_v1_20260927.json'


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def guard(index):
    authority = json.loads(AUTH.read_text())
    for binding in authority['sources']:
        assert bind(binding['path']) == binding, ('LONG4_SOURCE_HASH_DRIFT', binding['path'])
    assert authority['indices'] == list(range(128, 593)), 'LONG4_NATURAL_ORDINAL_SCOPE'
    assert index in authority['indices']
    assert os.environ.get('SLURM_JOB_ID')
    return authority


def normalized_input(value):
    """Only identical image content may differ in its original source path."""
    assert isinstance(value, dict), 'SHARED_INPUT_PAYLOAD_TYPE'
    image = value['original_image']
    assert isinstance(image, dict) and isinstance(image['path'], str)
    assert image['sha256'] == value['image_sha256'], 'SHARED_INPUT_IMAGE_SHA_DRIFT'
    normalized = dict(value)
    normalized['original_image'] = dict(image, path='<same-image-sha256>')
    return normalized


def save_with_verified_shared_publish(original_save, load, equal_tree, input_root, counts):
    """Recover a publication race only when the existing input is identical."""
    input_root = Path(input_root).resolve()

    def save(path, value, *args, **kwargs):
        path = Path(path)
        try:
            return original_save(path, value, *args, **kwargs)
        except (RuntimeError, FileExistsError) as exc:
            eligible = path.suffix == '.pt' and path.parent.resolve() == input_root
            publication_race = isinstance(exc, FileExistsError) or str(exc) == 'IMMUTABLE_PART'
            if not eligible or not publication_race:
                raise
            # The original writer publishes using a hard link only after fsync.
            # Read its already-published file; never replace the winning file.
            published = load(path)
            assert equal_tree(normalized_input(published), normalized_input(value)), (
                'SHARED_INPUT_PUBLISH_DRIFT', str(path))
            counts['shared_input_publish_races_verified'] += 1
            counts['shared_input_source_path_aliases'] += int(
                published['original_image']['path'] != value['original_image']['path'])
            # A losing hard-link publication can leave this process's own temp.
            # Preserve it on any mismatch above; remove it only after equality.
            temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
            if isinstance(exc, FileExistsError) and temporary.exists():
                temporary.unlink()
            return None

    return save


def receipt_path(path, index, job_id, restart, attempt):
    path = Path(path)
    if path.parent.name == 'chunks' and path.name == job_id + '.json':
        return path.with_name(
            f'query{index:03d}_{job_id}_restart{restart:02d}_chunk{attempt:03d}.json')
    return path


def selftest():
    """No Torch import, filesystem writes, scheduler calls, or GPU work."""
    from run_rc_unified128_collect_chunk_v2 import selftest as matcher_selftest
    matcher_selftest()
    root = Path('/synthetic-unified-inputs')
    image_sha = 'a' * 64
    candidate = dict(authority={'path': '/authority', 'sha256': 'b' * 64},
                     image_sha256=image_sha, transform='NATIVE',
                     original_image={'path': '/source-a', 'sha256': image_sha},
                     low_resolution_thumbnail64=[1., 2.],
                     high_resolution_thumbnail64=[3., 4.])
    published = dict(candidate, original_image={'path': '/source-b', 'sha256': image_sha})
    counts = Counter()
    reads = []

    def load(path):
        reads.append(path)
        return published

    def immutable(*args, **kwargs):
        raise RuntimeError('IMMUTABLE_PART')

    def exists(*args, **kwargs):
        raise FileExistsError('already published')

    for failure in (immutable, exists):
        save = save_with_verified_shared_publish(failure, load, lambda a, b: a == b, root, counts)
        assert save(root / 'image_NATIVE.pt', candidate) is None
    assert len(reads) == 2 and counts['shared_input_publish_races_verified'] == 2
    assert counts['shared_input_source_path_aliases'] == 2
    save = save_with_verified_shared_publish(immutable, load, lambda a, b: a == b, root, counts)
    for bad in (dict(candidate, transform='GRAY'),
                dict(candidate, low_resolution_thumbnail64=[1., 99.]),
                dict(candidate, original_image={'path': '/source-a', 'sha256': 'c' * 64})):
        try:
            save(root / 'image_NATIVE.pt', bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unequal published shared input was accepted')
    before = len(reads)
    for path in (root / 'query128/pair000.pt', root / 'image_NATIVE.json',
                 Path('/synthetic-other/image_NATIVE.pt')):
        try:
            save(path, candidate)
        except RuntimeError as exc:
            assert str(exc) == 'IMMUTABLE_PART'
        else:
            raise AssertionError('Non-shared-input collision was accepted')
    assert len(reads) == before

    def unrelated(*args, **kwargs):
        raise RuntimeError('UNRELATED_FAILURE')

    try:
        save_with_verified_shared_publish(unrelated, load, lambda a, b: a == b, root, counts)(
            root / 'image_NATIVE.pt', candidate)
    except RuntimeError as exc:
        assert str(exc) == 'UNRELATED_FAILURE'
    else:
        raise AssertionError('Unrelated save error was suppressed')
    calls = []

    def successful(path, value):
        calls.append((path, value))
        return 'ORIGINAL_SAVE_RESULT'

    assert save_with_verified_shared_publish(successful, load, lambda a, b: a == b, root, counts)(
        root / 'image_NATIVE.pt', candidate) == 'ORIGINAL_SAVE_RESULT'
    assert len(calls) == 1 and len(reads) == before
    first = receipt_path('/cache/query128/chunks/123.json', 128, '123', 0, 0)
    second = receipt_path('/cache/query128/chunks/123.json', 128, '123', 0, 1)
    third = receipt_path('/cache/query128/chunks/123.json', 128, '123', 1, 0)
    assert len({first, second, third}) == 3
    assert receipt_path('/cache/query128/ready.json', 128, '123', 0, 1).name == 'ready.json'
    print({'status': 'UNIFIED593_SHARED_PUBLISH_AND_CHUNK_RECEIPTS_SELFTEST_PASS',
           'unequal_publish_rejected': True, 'scope_restriction_checked': True}, flush=True)


def collect(index):
    guard(index)
    job_id = os.environ['SLURM_JOB_ID']
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    attempt = int(os.environ['RC_UNIFIED_CHUNK_ATTEMPT'])
    assert restart >= 0 and attempt >= 0
    execution_binding = bind(AUTH)

    import run_rc_h593_unified_acquisition_v1 as original
    counts = Counter()
    original_matcher = original.G.JointObserver.memo_matcher
    original_save = original.save
    original_write = original.write

    def write(path, value, *args, **kwargs):
        revised = receipt_path(path, index, job_id, restart, attempt)
        if revised != Path(path):
            value = dict(value, index=index, slurm_job_id=job_id, restart=restart,
                         chunk_attempt=attempt, long4_execution_authority=execution_binding)
        return original_write(revised, value, *args, **kwargs)

    original.G.JointObserver.memo_matcher = matcher_with_internal_replay(
        original_matcher, original.equal_tree, original.V.tags, counts)
    original.save = save_with_verified_shared_publish(
        original_save,
        lambda path: original.torch.load(path, map_location='cpu', weights_only=True),
        original.equal_tree, original.OUT / 'inputs', counts)
    original.write = write
    returned = False
    try:
        original.collect(index)
        returned = True
    finally:
        original.G.JointObserver.memo_matcher = original_matcher
        original.save = original_save
        original.write = original_write
        original_write(original.OUT / f'query{index:03d}' / 'long4_execution'
                       / f'query{index:03d}_{job_id}_restart{restart:02d}_chunk{attempt:03d}.json',
                       dict(authority=execution_binding, index=index, slurm_job_id=job_id,
                            restart=restart, chunk_attempt=attempt,
                            collect_returned_normally=returned,
                            internal_hook_cache_bypasses=counts['internal_hook_cache_bypasses'],
                            internal_hook_replays_bit_exact=counts['internal_hook_replays_bit_exact'],
                            shared_input_publish_races_verified=counts['shared_input_publish_races_verified'],
                            shared_input_source_path_aliases=counts['shared_input_source_path_aliases'],
                            original_science_unchanged=True, existing_capsules_preserved=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int, nargs='?')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        assert args.index is not None
        collect(args.index)
