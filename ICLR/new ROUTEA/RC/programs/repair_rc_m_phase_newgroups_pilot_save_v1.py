#!/usr/bin/env python3
"""Recover the sealed phase pilot's save-only failure from its four CPU caches.

The failed pilot completed its encoder and matcher repeats before attempting
``dict(protocol=pb, **first)`` although ``first`` already contained protocol.
This repair never changes the sealed source/protocol or reruns either forward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import math
from pathlib import Path
import resource
import tempfile

import run_rc_m_phase_newgroups_cpu_v1 as original

NAMES = ('pilot_encoder_0.pt', 'pilot_encoder_1.pt',
         'pilot_matcher_0.pt', 'pilot_matcher_1.pt')
MANIFEST = 'pilot_save_repair_manifest.json'


def prepare(torch):
    p = original.guard(torch, gate=False)
    pb = original.bind(original.OUT/'protocol.json')
    caches = {name: original.bind(original.OUT/name) for name in NAMES}
    for name, binding in caches.items():
        record = torch.load(original.checked(binding), weights_only=True,
                            mmap=True, map_location='cpu')
        assert record['protocol'] == pb, ('CACHE_PROTOCOL_DRIFT', name)
        assert math.isfinite(record['seconds']) and 0 < record['seconds'] <= 320
    manifest = dict(
        status='PHASE_PILOT_SAVE_REPAIR_PROTOCOL_FROZEN', version=1,
        repair_source=original.bind(Path(__file__)),
        original_source=original.bind(Path(original.__file__)),
        original_protocol=pb, caches=caches,
        original_checkpoint=p['checkpoint'], original_head_state=p['head_state'],
        old_gpu_pilot_pair=p['old_pilot_pair'], environment=original.environment(torch),
        failed_job='5167604',
        engineering_cause="dict(protocol=pb, **first) supplies protocol twice",
        modification='Merge the cached first matcher record without a duplicate protocol keyword; recover its missing validation.',
        qualification='Preserve original exact CPU repeats, finite tensors, strict checkpoint/head identity, original algebra; GPU differences remain diagnostic only.',
        new_encoder_forwards=0, new_matcher_forwards=0,
        modifies_original_sources=False, modifies_original_protocol=False,
        science_or_threshold_changes=False)
    original.write(original.OUT/MANIFEST, manifest)
    original.emit(status='PHASE_PILOT_SAVE_REPAIR_PREPARED',
                  manifest=original.bind(original.OUT/MANIFEST), caches=len(caches))
    return 0


def guard(torch):
    p = original.guard(torch, gate=False)
    manifest = original.read(original.OUT/MANIFEST)
    assert manifest['status'] == 'PHASE_PILOT_SAVE_REPAIR_PROTOCOL_FROZEN'
    for name in ('repair_source', 'original_source', 'original_protocol',
                 'original_checkpoint', 'original_head_state', 'old_gpu_pilot_pair'):
        original.checked(manifest[name])
    assert manifest['repair_source'] == original.bind(Path(__file__))
    assert manifest['original_protocol'] == original.bind(original.OUT/'protocol.json')
    assert manifest['environment'] == original.environment(torch)
    assert list(sorted(manifest['caches'])) == list(sorted(NAMES))
    for binding in manifest['caches'].values(): original.checked(binding)
    return p, manifest


def forbidden_forward(*args, **kwargs):
    raise AssertionError('REPAIR_MUST_NOT_RUN_MODEL_FORWARD')


def qualify(torch):
    """Repeat the original qualification using retained tensors only."""
    p, manifest = guard(torch)
    pb = manifest['original_protocol']
    records = {
        name: torch.load(original.checked(binding), weights_only=True,
                         mmap=True, map_location='cpu')
        for name, binding in manifest['caches'].items()}
    assert all(record['protocol'] == pb for record in records.values())
    encoded = [records[f'pilot_encoder_{i}.pt'] for i in range(2)]
    captured = [records[f'pilot_matcher_{i}.pt'] for i in range(2)]
    assert all(math.isfinite(row['seconds']) and 0 < row['seconds'] <= 320
               for row in encoded+captured), 'ORIGINAL_ATOMIC_BUDGET_GATE_FAILED'
    for row in encoded:
        assert row['image'] == p['pilot_encoder_image'] and row['CPU_encoder'] is True
        assert row['profile'] == p['profile'] and len(row['features']) == 2
        assert all(tuple(x.shape) == (1, 50, 50, 1024) and
                   bool(torch.isfinite(x).all()) for x in row['features'])
    for row in captured:
        assert len(row['sides']) == 2
        for side in row['sides']:
            assert set(side) == {'J', 'P', 'warp', 'confidence'}
            assert all(bool(torch.isfinite(value).all()) for value in side.values())

    # Loading the exact original model verifies checkpoint and head identity.
    # Its forward is not invoked, and the two acquisition entrypoints are
    # forbidden so later edits cannot accidentally repeat the successful work.
    encoder_fn, capture_fn = original.encoder, original.capture
    original.encoder = original.capture = forbidden_forward
    try:
        matcher, _, helper = original.model(p, torch)
        matcher.forward = forbidden_forward
        matcher_config = asdict(matcher.cfg)
        del matcher
    finally:
        original.encoder, original.capture = encoder_fn, capture_fn

    old_pair = torch.load(original.checked(p['old_pilot_pair']),
                          weights_only=True, mmap=True, map_location='cpu')
    old_cap = torch.load(original.checked(old_pair['capture']),
                         weights_only=True, mmap=True, map_location='cpu')
    fs = [[t.clone() for t in torch.load(original.checked(b), weights_only=True,
                                        mmap=True, map_location='cpu')['features']]
          for b in old_cap['descriptor_sources']]
    encoder_repeat = [helper.differences(a, b, torch) for a, b in
                      zip(encoded[0]['features'], encoded[1]['features'], strict=True)]
    assert all(x['bit_exact'] for x in encoder_repeat), 'CPU_ENCODER_REPEAT_NOT_EXACT'
    encoder_gpu_drift = [helper.differences(a, b, torch) for a, b in
                         zip(encoded[0]['features'], fs[0], strict=True)]
    repeats, drift = [], []
    first, second = captured
    for side in (0, 1):
        checks = {key: helper.differences(first['sides'][side][key],
                                         second['sides'][side][key], torch)
                  for key in ('J', 'P', 'warp', 'confidence')}
        assert all(row['bit_exact'] for row in checks.values()), 'CPU_MATCHER_NATIVE_REPEAT_NOT_EXACT'
        repeats.append(checks)
        gpu = old_cap['sides'][side]
        row = {key: helper.differences(
            first['sides'][side][key],
            gpu[key]['tensor'].to(dtype=getattr(torch, gpu[key]['original_dtype'])), torch)
            for key in ('J', 'P')}
        row['confidence'] = helper.differences(first['sides'][side]['confidence'],
            old_pair['branches']['NATIVE'][side]['confidence'], torch)
        drift.append(row)
    import rc_roma_property_restore_v2 as transform
    algebra = transform.self_test()
    # first already contains protocol. This is the sole scientific-output save
    # construction correction; all retained tensors remain the exact same ones.
    capture = dict(first, old_gpu_pair=p['old_pilot_pair'])
    assert capture['protocol'] == pb
    validation = dict(
        status='PHASE_NEWGROUP_CPU_PILOT_PASS', protocol=pb,
        pilot_source='Historical opened query000 pair000; no new replication group outcome used',
        repeat_checks=repeats, gpu_drift=drift,
        gpu_replay_bit_exact_claimed=False, gpu_tolerance_gate=False,
        acceptance='Same-CPU native full matcher repeat exact, finite outputs, checkpoint/head identity, original algebra',
        algebra=algebra, matcher_config=matcher_config,
        seconds=[first['seconds'], second['seconds']],
        encoder_repeat=encoder_repeat, encoder_gpu_drift=encoder_gpu_drift,
        encoder_seconds=[row['seconds'] for row in encoded],
        new_single_image_features_required=p['new_image_encoder_forwards_required'],
        execution_repair=original.bind(original.OUT/MANIFEST),
        saved_stage_sources=manifest['caches'], new_encoder_forwards=0,
        new_matcher_forwards=0, original_scientific_qualification_unchanged=True)
    return capture, validation


def equal_record(actual, expected, torch, location='capture'):
    """Reject a conflicting existing artifact instead of overwriting it."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.shape == expected.shape
        assert actual.dtype == expected.dtype, ('TENSOR_DTYPE_DRIFT', location)
        a = actual.reshape(-1).contiguous().view(torch.uint8)
        b = expected.reshape(-1).contiguous().view(torch.uint8)
        assert torch.equal(a, b), ('TENSOR_BYTES_DRIFT', location)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), location
        for key in expected: equal_record(actual[key], expected[key], torch, location+'.'+str(key))
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), location
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal_record(a, b, torch, location+f'[{i}]')
    else:
        assert actual == expected, ('CACHE_VALUE_DRIFT', location)


def ensure_capture(path, capture, torch):
    if path.exists():
        existing = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
        equal_record(existing, capture, torch)
    else:
        original.save(path, capture, torch)
    return original.bind(path)


def publish(folder, capture, validation, torch):
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(validation, capture=ensure_capture(folder/'pilot_cpu_capture.pt', capture, torch))
    path = folder/'pilot_validation.json'
    if path.exists():
        existing = original.read(path)
        assert {k: v for k, v in existing.items() if k != 'max_rss_kib'} == payload, 'EXISTING_VALIDATION_DRIFT'
    else:
        original.write(path, dict(payload, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return original.bind(path)


def recover(torch):
    with (original.OUT/'pilot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capture, validation = qualify(torch)
        output = publish(original.OUT, capture, validation, torch)
    original.emit(status='PHASE_NEWGROUP_CPU_PILOT_PASS', validation=output,
                  engineering_repair=True, new_encoder_forwards=0, new_matcher_forwards=0)
    return 0


def test(torch):
    """Use the real four caches; write only temporary recovery artifacts."""
    capture, validation = qualify(torch)
    before = {name: original.bind(original.OUT/name) for name in NAMES}
    try:
        dict(protocol=capture['protocol'], **capture)
    except TypeError:
        duplicate_bug_reproduced = True
    else:
        raise AssertionError('ORIGINAL_DUPLICATE_KEYWORD_BUG_NOT_REPRODUCED')
    with tempfile.TemporaryDirectory(prefix='phase-pilot-save-repair-', dir='/tmp') as temporary:
        base = Path(temporary)
        fresh = publish(base/'fresh', capture, validation, torch)
        assert publish(base/'fresh', capture, validation, torch) == fresh
        # Simulate interruption exactly after the tensor save, before JSON.
        pending = base/'interrupted'; pending.mkdir()
        saved = ensure_capture(pending/'pilot_cpu_capture.pt', capture, torch)
        resumed = publish(pending, capture, validation, torch)
        assert original.bind(pending/'pilot_cpu_capture.pt') == saved
        assert publish(pending, capture, validation, torch) == resumed
        rejected = base/'foreign'; rejected.mkdir()
        foreign = dict(capture, protocol={'path': 'wrong', 'sha256': 'wrong'})
        original.save(rejected/'pilot_cpu_capture.pt', foreign, torch)
        try: publish(rejected, capture, validation, torch)
        except AssertionError: pass
        else: raise AssertionError('FOREIGN_EXISTING_CAPTURE_NOT_REJECTED')
        assert not (rejected/'pilot_validation.json').exists()
    assert before == {name: original.bind(original.OUT/name) for name in NAMES}
    original.emit(status='PHASE_PILOT_SAVE_REPAIR_REGRESSION_PASS',
        duplicate_bug_reproduced=duplicate_bug_reproduced, cached_stages=4,
        encoder_repeat_exact=all(x['bit_exact'] for x in validation['encoder_repeat']),
        matcher_repeat_exact=all(v['bit_exact'] for row in validation['repeat_checks'] for v in row.values()),
        fresh_recovery=True, validation_idempotent=True, capture_only_interruption_recovered=True,
        foreign_capture_rejected=True, formal_outputs_written=False,
        new_encoder_forwards=0, new_matcher_forwards=0)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'recover', 'test'))
    parser.add_argument('--root', type=Path, default=original.OUT)
    args = parser.parse_args(); original.OUT = args.root.resolve()
    torch = original.configure()
    return {'prepare': prepare, 'recover': recover, 'test': test}[args.stage](torch)


if __name__ == '__main__': raise SystemExit(main())
