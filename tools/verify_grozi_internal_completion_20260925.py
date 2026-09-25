#!/usr/bin/env python3
"""Verify archived bytes and complete candidate axes; requires no GPU or model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / 'ICLR/new ROUTEA/RC'
EXP = RC / 'results/rc_postllm_grozi_external_v1'
BACKUP = ROOT / 'backup/grozi_internal_complete_20260925'
SOURCE_WS = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')


def read(p):
    return json.loads(p.read_text())


def check_bound(b):
    p = ROOT / Path(b['path']).relative_to(SOURCE_WS)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'], str(p)
    return read(p)


def main():
    manifest = read(BACKUP / 'files.json')
    for r in manifest['records']:
        p = ROOT / r['path']
        assert p.stat().st_size == r['bytes'], str(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'], str(p)
    authority = read(RC / 'registry/rc_postllm_grozi_external_authority_v1_20260925.json')
    workers = check_bound(authority['workers'])['records']
    expected = {w['query_id'] for w in workers}
    assert len(expected) == 480
    seal = read(EXP / 'all_predictions_prelabel_seal.json')
    assert seal['status'] == 'GROZI_POSTLLM_ALL480_PRELABEL_PASS'
    assert len(seal['shard_validations']) == 60
    seen = set()
    arm_names = {'POST_REAL', 'POST_REAL_CONSTANT', 'POST_REAL_SHUFFLED',
                 'NO_ADAPTER_INTERNAL3', 'EXTERNAL_ADDITIVE4', 'EXTERNAL_PRODUCT5'}
    for b in seal['shard_validations']:
        shard = check_bound(b)
        assert shard['status'] == 'GROZI_POSTLLM_SHARD_PASS'
        assert len(shard['queries']) == shard['query_count'] == 8
        for qb in shard['queries']:
            q = check_bound(qb)
            assert q['query_id'] not in seen
            seen.add(q['query_id'])
            assert q['status'] == 'GROZI_POSTLLM_FROZEN_QUERY_PASS'
            assert q['external_training_updates'] == 0 and q['held_label_reads'] == 0
            assert q['direct_M_in_internal_head'] is False
            assert set(q['models']) == arm_names
            for model in q['models'].values():
                decision = model['decision']
                assert len(model['L']) == len(decision['scores128']) == 128
                assert len(decision['logits']) == len(decision['challenger_positions']) == 127
    assert seen == expected
    partials = list((EXP / 'predictions').glob('*.partial.json'))
    parity = list((EXP / 'encoder_cache').glob('*/parity.json'))
    cache_seals = list((EXP / 'encoder_cache').glob('*/validation.json'))
    assert len(partials) == len(parity) == len(cache_seals) == 480
    result = read(EXP / 'result.json')
    assert result['summary']['RAW']['correct'] == 321
    assert result['summary']['POST_REAL']['correct'] == 339
    assert result['summary']['POST_REAL']['rescue'] == 18
    assert result['summary']['POST_REAL']['breaks'] == 0
    validation = read(EXP / 'validation.json')
    check_bound(validation['result'])
    check_bound(validation['predictions_seal'])
    check_bound(validation['authority'])
    check_bound(validation['join_repair'])
    assert validation['query_count'] == 480 and validation['video_groups'] == 27
    report = {
        'status': 'PASS_EXPORTED_BYTES_SEALED_AXES_AND_COMPLETE_PREDICTIONS',
        'exported_text_files': len(manifest['records']),
        'queries': 480, 'shards': 60, 'non_RAW_arms': 6,
        'candidate_content_scores': 480 * 6 * 128,
        'challenger_logits': 480 * 6 * 127,
        'saved_partial_states': len(partials),
        'cache_parity_metadata': len(parity), 'cache_seal_metadata': len(cache_seals),
        'binary_token_payloads_included': False,
        'scientific_validation': 'Existing independent NumPy and grouped-statistic validations retained; this script verifies export completeness and seal integrity.',
        'manifest_sha256': hashlib.sha256((BACKUP / 'files.json').read_bytes()).hexdigest(),
    }
    (BACKUP / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
