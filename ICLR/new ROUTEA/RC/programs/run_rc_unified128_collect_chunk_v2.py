#!/usr/bin/env python3
"""Resume sealed acquisition with internal-hook replay and distinct receipts."""
import argparse
from collections import Counter
import os
from pathlib import Path


def matcher_with_internal_replay(original_matcher, equal_tree, tags, counts):
    """Reuse the proven query70 repair and verify replayed matcher outputs."""
    def matcher(self, *args, **kwargs):
        key = tags(self.arm)
        cached = None
        if (self.want_inside and self.cache_identity == self.identity
                and key in self.pair_cache):
            cached = self.pair_cache[key]
            assert equal_tree((args, kwargs), cached['inputs']), 'JOINT_MATCHER_INPUT_DRIFT'
            del self.pair_cache[key]
            counts['internal_hook_cache_bypasses'] += 1
        result = original_matcher(self, *args, **kwargs)
        if cached is not None:
            assert equal_tree(result, cached['outputs']), 'INTERNAL_HOOK_REPLAY_OUTPUT_DRIFT'
            counts['internal_hook_replays_bit_exact'] += 1
        return result
    return matcher


def receipt_path(path, job_id, restart):
    path = Path(path)
    if path.parent.name == 'chunks' and path.name == job_id + '.json':
        return path.with_name(f'{job_id}_restart{restart:02d}.json')
    return path


def selftest():
    """Exercise the hook replay branch without importing Torch or using GPUs."""
    class Dummy:
        def __init__(self):
            self.want_inside = True
            self.identity = ('query', 'reference')
            self.cache_identity = self.identity
            self.arm = 'NATIVE'
            self.pair_cache = {}
            self.calls = 0

    def native(self, *args, **kwargs):
        if self.cache_identity != self.identity:
            self.pair_cache.clear()
            self.cache_identity = self.identity
        if self.arm in self.pair_cache:
            assert not self.want_inside
            return self.pair_cache[self.arm]['outputs']
        self.calls += 1
        result = {'warp': args[0], 'confidence': kwargs['confidence']}
        self.pair_cache[self.arm] = {'inputs': (args, kwargs), 'outputs': result}
        return result

    counts = Counter()
    wrapped = matcher_with_internal_replay(native, lambda a, b: a == b, lambda arm: arm, counts)
    observer = Dummy()
    first = wrapped(observer, 7, confidence=11)
    assert observer.calls == 1 and not counts
    assert wrapped(observer, 7, confidence=11) == first
    assert observer.calls == 2 and counts['internal_hook_replays_bit_exact'] == 1
    observer.want_inside = False
    assert wrapped(observer, 7, confidence=11) == first and observer.calls == 2
    observer.want_inside = True
    try:
        wrapped(observer, 8, confidence=11)
    except AssertionError as exc:
        assert str(exc) == 'JOINT_MATCHER_INPUT_DRIFT'
    else:
        raise AssertionError('Changed inputs were accepted')
    assert observer.calls == 2 and observer.arm in observer.pair_cache
    observer.pair_cache[observer.arm]['outputs'] = {'warp': -1, 'confidence': 11}
    try:
        wrapped(observer, 7, confidence=11)
    except AssertionError as exc:
        assert str(exc) == 'INTERNAL_HOOK_REPLAY_OUTPUT_DRIFT'
    else:
        raise AssertionError('Changed replay outputs were accepted')
    observer.identity = ('query', 'different_reference')
    assert wrapped(observer, 9, confidence=13) == {'warp': 9, 'confidence': 13}
    assert receipt_path('/cache/query041/chunks/123.json', '123', 2).name == '123_restart02.json'
    assert receipt_path('/cache/query041/ready.json', '123', 2).name == 'ready.json'
    print({'status': 'UNIFIED128_DUPLICATE_HOOK_AND_RESTART_SELFTEST_PASS',
           'input_drift_rejected': True, 'output_drift_rejected': True}, flush=True)


def collect(index):
    import run_rc_unified128_resume_v2 as execution
    authority = execution.guard()
    assert index in authority['indices']
    assert os.environ.get('SLURM_JOB_ID')
    job_id = os.environ['SLURM_JOB_ID']
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    assert 0 <= restart <= authority['max_restarts']

    import run_rc_h593_unified_acquisition_v1 as original
    counts = Counter()
    original_matcher = original.G.JointObserver.memo_matcher
    original_write = original.write

    def write(path, value, *args, **kwargs):
        revised = receipt_path(path, job_id, restart)
        if revised != Path(path):
            value = dict(value, slurm_job_id=job_id, restart=restart,
                         resume_execution_authority=execution.bind(execution.AUTH))
        return original_write(revised, value, *args, **kwargs)

    original.G.JointObserver.memo_matcher = matcher_with_internal_replay(
        original_matcher, original.equal_tree, original.V.tags, counts)
    original.write = write
    completed = False
    try:
        original.collect(index)
        completed = True
    finally:
        original.G.JointObserver.memo_matcher = original_matcher
        original.write = original_write
        original_write(original.OUT / f'query{index:03d}' / 'resume_v2_execution'
                       / f'{job_id}_restart{restart:02d}.json', dict(
            authority=execution.bind(execution.AUTH), index=index, slurm_job_id=job_id,
            restart=restart, collect_returned_normally=completed,
            internal_hook_cache_bypasses=counts['internal_hook_cache_bypasses'],
            internal_hook_replays_bit_exact=counts['internal_hook_replays_bit_exact'],
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
