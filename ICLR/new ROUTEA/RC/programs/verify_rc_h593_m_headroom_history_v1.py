#!/usr/bin/env python3
"""Independently recount historical repairs of the audited 40 M-first errors."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_m_scalar_headroom_audit_20260927_v1'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(b):
    assert bind(b['path']) == b
    return read(b['path'])


def main():
    result = read(OUT / 'result.json')
    validation = read(OUT / 'independent_validation.json')
    assert validation['status'] == 'H593_M_SCALAR_HEADROOM_INDEPENDENT_PASS'
    assert validation['result'] == bind(OUT / 'result.json')
    history = read(OUT / 'historical_overlap.json')
    assert history['audit_result_sha256'] == validation['result']['sha256']
    roles = {r['query_id']: r for r in checked(result['sources']['curator'])['records']}
    labels = {r['physical_row']: r['identity'] for r in checked(result['sources']['gallery'])['records']}
    reference_rows = {r['query_id']: r for r in result['records']}
    opportunities = {q for q, r in reference_rows.items() if not r['cost1_correct'] and r.get('M_unique_first', False)}
    assert len(opportunities) == history['original_M_first_COST1_errors'] == 40
    sources = [checked(b) for b in history['sources']]
    recomputed = []
    for reported in history['models']:
        model = reported['model']
        source = sources[1] if model == 'CONTENT_BOX_CE_POLISH18' else sources[0]
        assert len(source['rows']) == 593
        correct_ids = set()
        for r in source['rows']:
            qid = r['query_id']
            assert r['fold'] == reference_rows[qid]['fold']
            assert r['component'] == reference_rows[qid]['component']
            correct = labels[r['selected'][model]] == roles[qid]['identity']
            assert correct == r['correct'][model]
            if correct:
                correct_ids.add(qid)
        rescues = sorted(opportunities & correct_ids)
        remaining = sorted(opportunities - correct_ids)
        calculated = {'model': model, 'correct': len(correct_ids),
                      'of_original40_M_first_recovered': len(rescues),
                      'of_original40_M_first_still_wrong': len(remaining),
                      'rescued_query_ids': rescues, 'remaining_query_ids': remaining}
        assert calculated == reported
        recomputed.append(calculated)
    receipt = {'status': 'H593_M_HEADROOM_HISTORY_INDEPENDENT_PASS',
               'source': bind(OUT / 'historical_overlap.json'),
               'native_M_audit_validation': bind(OUT / 'independent_validation.json'),
               'verifier': bind(__file__), 'sources': history['sources'],
               'verified_model_decisions': 4 * 593, 'models': recomputed,
               'new_training_updates': 0, 'new_encoder_or_matcher_forwards': 0}
    (OUT / 'historical_overlap_independent_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'counts': [{k: v for k, v in r.items() if not k.endswith('query_ids')} for r in recomputed]}, indent=2))


if __name__ == '__main__':
    main()
