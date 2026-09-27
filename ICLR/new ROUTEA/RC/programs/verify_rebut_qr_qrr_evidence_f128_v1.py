#!/usr/bin/env python3
"""Independently verify F128 axes, M0, geometry, reuse and absence of label reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import verify_rebut_qr_qrr_evidence_v1 as independent

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_f128_v1'


def checked(binding):
    path = Path(binding['path'])
    assert independent.binding(path)['sha256'] == binding['sha256'], ('SHA', str(path))
    return path


def verify_payload(record, common):
    path = checked(record['payload'])
    p = independent.load(path)
    q, r = p['query'], p['reference']
    assert p['query_id'] == record['query_id'] == common['query_id']
    assert p['execution_ordinal'] == record['execution_ordinal'] == common['execution_ordinal']
    assert p['axis'] == common['axis'] and p['winner'] == common['winner']
    assert len(p['axis']) == len(set(p['axis'])) == 128 and 0 <= p['winner'] < 128
    assert sorted([p['winner']] + p['challenger_positions']) == list(range(128))
    assert q.shape == r.shape == (128, 64, 72) and q.dtype == r.dtype == torch.float32
    assert torch.isfinite(q).all() and torch.isfinite(r).all()
    assert p['M0'].dtype == torch.float64 and torch.equal(p['M0'], common['M0'])
    assert p['labels_read'] == p['backbone_forwards'] == p['new_roma_forwards'] == 0
    cp = RC / f"results/rc_h593_roma_coordinate_precision_v2/query{p['execution_ordinal']:03d}"
    cv = independent.read(cp / 'validation.json')
    assert cv['status'] == 'ROMA_COORDINATE_QUERY_PASS'
    native = independent.read(checked(cv['payload']))
    assert native['query_id'] == p['query_id'] and native['candidate_physical_rows'] == p['axis']
    mass = torch.tensor([s['visibility_mass'] for s in native['modes']['NATIVE']['scores']], dtype=torch.float64)
    assert torch.equal(mass, p['M0'])
    assert torch.equal(torch.tensor(native['modes']['NATIVE']['X'], dtype=torch.float64), p['native_X'])
    for side in ['query', 'reference']:
        valid, mapped = p[side + '_valid'], p[side + '_mapping_valid']
        assert valid.dtype == mapped.dtype == torch.bool and valid.shape == mapped.shape == (128, 64)
        assert not bool((mapped & ~valid).any())
        assert bool((p[side][~valid] == 0).all())
        assert bool((p[side][:, :, 32:64][~mapped] == 0).all())
    assert torch.equal(q[:, :, :32], q[0, :, :32].expand_as(q[:, :, :32]))
    assert torch.equal(p['query_valid'], p['query_valid'][0].expand_as(p['query_valid']))
    return p['query_id']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    mp = OUT / 'evidence_manifest.json'
    mb = independent.binding(mp); manifest = independent.read(mp)
    assert manifest['version'] == 'POOLED_F128_SAME_V1_SCHEMA'
    assert manifest['payload_schema_version'] == independent.VERSION
    assert [r['execution_ordinal'] for r in manifest['expected_records']] == list(range(128))
    assert manifest['completed_queries'] == len(manifest['records'])
    if not a.allow_partial:
        assert manifest['status'] == 'POOLED_F128_COMPLETE' and len(manifest['records']) == 128
    elif len(manifest['records']) < 128:
        assert manifest['status'] == 'POOLED_F128_PARTIAL'
    data = independent.read(checked(manifest['sources']['frozen_inputs']))
    common = {r['query_id']: r for r in independent.load(checked(data['common_features']))['records']}
    old = {r['execution_ordinal']: r for r in independent.read(checked(data['old_evidence_manifest']))['records']}
    seen = set()
    for row in manifest['records']:
        n = row['execution_ordinal']
        assert n in range(128) and n not in seen
        seen.add(n)
        assert row['query_id'] == manifest['expected_records'][n]['query_id']
        assert row['reused_from_F71'] == (n < 71)
        if n < 71:
            assert row['payload'] == old[n]['payload'], 'OLD71_BYTES_REUSED'
        else:
            assert Path(row['payload']['path']).parent == OUT / 'evidence', 'NEW57_OWN_ROOT'
        verify_payload(row, common[row['query_id']])
    assert set(range(71)) <= seen
    synthetic = independent.synthetic_checks()
    first = manifest['records'][0]
    replays = [independent.source_replay(first, [0, 1, 127])]
    new = [r for r in manifest['records'] if r['execution_ordinal'] >= 71]
    if new:
        replays.append(independent.source_replay(new[0], [0, 1, 127]))
    assert independent.binding(mp) == mb, 'MANIFEST_CHANGED_DURING_VALIDATION'
    complete = len(seen) == 128
    result = dict(status='POOLED_F128_EVIDENCE_INDEPENDENT_PASS' if complete else 'POOLED_F128_PARTIAL_INDEPENDENT_PASS',
        manifest=mb, verifier=independent.binding(__file__), queries=len(seen), candidates=128 * len(seen),
        reused71_exact=True, new_queries=len(seen) - 71, synthetic_checks=synthetic,
        source_replays=replays, common_axis_and_M0_exact=True, labels_read=0,
        new_encoder_forwards=0, new_roma_forwards=0, old_F71_changed=False)
    dest = OUT / ('evidence_validation.json' if complete else 'evidence_engineering_checks.json')
    independent.write(dest, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
