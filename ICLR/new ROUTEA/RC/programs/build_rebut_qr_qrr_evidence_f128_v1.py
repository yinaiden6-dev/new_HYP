#!/usr/bin/env python3
"""Extend sealed F71 to F128 on CPU, reusing old payloads and projected images."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import torch
import build_rebut_qr_qrr_evidence_v1 as science

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_f128_v1'
OLD = RC / 'results/rc_rebut_qr_qrr_v1'
SCHEMA = 'POOLED_F71_COARSE17_NATIVE_CENTER_WARP_V1'
MANIFEST_VERSION = 'POOLED_F128_SAME_V1_SCHEMA'


def checked(binding):
    return Path(science.same_sha(binding['path'], binding)['path'])


class ReusedImageBank(science.ImageBank):
    """Read existing projected images; write only genuinely absent projections."""
    def get(self, key):
        path = OLD / 'evidence/projected_images' / (key + '.pt')
        if key in self.cache or not path.exists():
            return super().get(key)
        source = self.ready['features'][key]
        p = science.load(path)
        science.need(p['version'] == SCHEMA and p['source'] == source and
                     p['projection_seed'] == science.SEED, 'REUSED_PROJECTION_BINDING')
        science.need(p['projection_sha256'] == science.tensor_sha(self.matrix), 'REUSED_PROJECTION_MATRIX')
        info = science.geometry_index(p['boxes'], p['valid'])
        info.update(projected=p['projected'].numpy(), item=p['item'], geometry=p['geometry'],
                    source=source, projected_binding=science.bind(path))
        self.used[key] = dict(source=source, projected=info['projected_binding'],
                             consumed_descriptor_sha256=p['source_component_sha256'])
        self.cache[key] = info
        if len(self.cache) > 192:
            self.cache.popitem(last=False)
        return info


def inputs():
    p = science.read(OUT / 'protocol.json')
    data = science.read(checked(p['evidence_inputs']))
    science.need(data['status'] == 'F128_EVIDENCE_INPUTS_FROZEN' and
                 data['panel_indices'] == list(range(128)), 'F128_FROZEN_INPUTS')
    science.need([r['execution_ordinal'] for r in data['expected_records']] == list(range(128)), 'F128_ORDER')
    for b in data['sources'].values():
        checked(b)
    checked(data['old_evidence_manifest'])
    checked(data['feature_ready']); checked(data['feature_catalog'])
    return p, data


def manifest(data):
    expected = {r['execution_ordinal']: r for r in data['expected_records']}
    records = []
    for old in data['reused_records']:
        checked(old['payload'])
        r = dict(old)
        science.need(r['execution_ordinal'] in range(71) and r['reused_from_F71'], 'OLD71_SCOPE')
        records.append(r)
    for index in range(71, 128):
        path = OUT / f'evidence/query{index:03d}.json'
        if not path.exists():
            continue
        r = science.read(path)
        science.need(r['status'] == 'POOLED_F_QUERY_PASS' and r['version'] == SCHEMA, 'NEW_QUERY_SEAL')
        checked(r['payload'])
        science.need(r['query_id'] == expected[index]['query_id'] and r['execution_ordinal'] == index, 'QUERY_ID')
        records.append(dict(query_id=r['query_id'], execution_ordinal=index, payload=r['payload'],
                            candidates=128, seconds=r['seconds'], reused_from_F71=False))
    records.sort(key=lambda r: r['execution_ordinal'])
    science.need(len({r['execution_ordinal'] for r in records}) == len(records), 'DUPLICATE_INDEX')
    complete = len(records) == 128
    value = dict(status='POOLED_F128_COMPLETE' if complete else 'POOLED_F128_PARTIAL',
        version=MANIFEST_VERSION, payload_schema_version=SCHEMA, expected_queries=128,
        completed_queries=len(records), expected_records=data['expected_records'], records=records,
        sources=dict(frozen_inputs=science.bind(OUT / 'evidence_inputs.json'),
            builder=science.bind(__file__), original_builder=science.bind(science.__file__),
            inherited_F71_manifest=data['old_evidence_manifest']),
        channels=science.CHANNELS, query_shape=[128, 64, 72], reference_shape=[128, 64, 72],
        dtype='float32', M0_dtype='float64', labels_read=0, new_roma_forwards=0, backbone_forwards=0,
        projection=dict(kind='shared label-free Rademacher1024-to32 scaled by1/sqrt32', seed=science.SEED),
        selection='Natural execution ordinals0..127; no correctness screening; original71 reused verbatim',
        scientific_array_definition_changed=False, old_payloads_reused=71,
        new_payloads_complete=len(records) - 71,
        mask_semantics='Source geometry valid; mapping valid separately; no certainty cutoff',
        sampling='Native warp on original token cells BEFORE same fixed8x8 geometric mean',
        evidence_boundary='F128 pooled projected development evidence; not native dense activations or all593')
    science.atomic_json(OUT / 'evidence_manifest.json', value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=int, default=71)
    ap.add_argument('--stop', type=int, default=128)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--wall-seconds', type=float, default=480)
    ap.add_argument('--manifest-only', action='store_true')
    a = ap.parse_args()
    science.need(71 <= a.start <= a.stop <= 128, 'NEW57_SCOPE')
    started = time.monotonic()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    science.install_label_guard()
    protocol, data = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'evidence_build.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = manifest(data)
        if a.manifest_only:
            print(json.dumps(initial), flush=True)
            return 0
        # Do not mutate original scientific files or their sealed projected image bank.
        science.OUT = OUT / 'evidence'
        science.OUT.mkdir(parents=True, exist_ok=True)
        ready = science.read(checked(data['feature_ready']))
        catalog = science.read(checked(data['feature_catalog']))
        rows = {r['execution_ordinal']: r for r in catalog['queries']}
        for index in range(128):
            receipt = science.COORDINATES / f'query{index:03d}/validation.json'
            science.need(receipt.is_file(), 'WAIT_FOR_ORIGINAL128_CPU_EXPORT:' + str(index))
            science.need(science.read(receipt)['status'] == 'ROMA_COORDINATE_QUERY_PASS', 'COORD_SOURCE_STATUS')
        bank = ReusedImageBank(ready, science.projection())
        for index in range(a.start, a.stop):
            receipt = science.build_query(index, rows, bank)
            row = data['expected_records'][index]
            payload = science.load(receipt['payload']['path'])
            science.need(payload['query_id'] == row['query_id'] and payload['axis'] == row['axis'] and
                         payload['winner'] == row['winner'], 'FROZEN_COMMON_AXIS')
            # Publish a manifest only at chunk boundaries; avoid rehashing all old71 after each query.
            if a.wall_seconds > 0 and time.monotonic() - started >= a.wall_seconds and index + 1 < a.stop:
                value = manifest(data)
                print(json.dumps(dict(status='F128_EVIDENCE_WALL_BUDGET', completed=value['completed_queries'],
                    next_index=index + 1, exit_code=75)), flush=True)
                return 75
        value = manifest(data)
        print(json.dumps(dict(status=value['status'], queries=value['completed_queries'],
                              reused=71, new=value['new_payloads_complete'])), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
