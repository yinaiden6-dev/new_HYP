#!/usr/bin/env python3
"""Describe held-identity reference exposure under original H593 query folds."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_v1'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    cache_path = RC / 'results/rc_h593_simple_explanations_v1/cache.json'
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    gallery_path = RC / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'
    cache = {r['query_id']: r for r in read(cache_path)['rows']}
    roles = {r['query_id']: r for r in read(curator_path)['records']}
    gallery = {r['physical_row']: r['identity'] for r in read(gallery_path)['records']}
    assert set(cache) == set(roles) and len(cache) == 593
    present = {q: any(gallery[p] == roles[q]['identity'] for p in row['axis']) for q, row in cache.items()}
    assert sum(present.values()) == 570
    fold_rows = []
    detailed = []
    fold_sources = []
    for fold in range(5):
        path = RC / f'results/rc_six_cause_isolation_v1/loss_binding/fold{fold}/payload.json'
        payload = read(path)
        fold_sources.append(binding(path))
        train = set(payload['train_query_ids'])
        held = {r['query_id'] for r in payload['predictions']}
        assert not train & held and train | held == set(roles)
        assert held == {q for q, r in roles.items() if r['outer_fold'] == fold}
        train_ids = {roles[q]['identity'] for q in train}
        held_ids = {roles[q]['identity'] for q in held}
        train_components = {roles[q]['component'] for q in train}
        held_components = {roles[q]['component'] for q in held}
        train_sha = {roles[q]['source_image_sha256'] for q in train}
        held_sha = {roles[q]['source_image_sha256'] for q in held}
        for universe, train_subset in [
            ('ALL_ORIGINAL_TRAIN_QUERIES', train),
            ('TARGET_PRESENT_TRAIN_QUERIES', {q for q in train if present[q]}),
        ]:
            identity_exposure = Counter()
            identity_training_queries = {identity: set() for identity in held_ids}
            exposed_reference_rows = {identity: set() for identity in held_ids}
            train_queries_with_any_held_negative = set()
            for q in sorted(train_subset):
                assert len(cache[q]['axis']) == len(set(cache[q]['axis'])) == 128
                for physical in cache[q]['axis']:
                    identity = gallery[physical]
                    if identity in held_ids and identity != roles[q]['identity']:
                        identity_exposure[identity] += 1
                        identity_training_queries[identity].add(q)
                        exposed_reference_rows[identity].add(physical)
                        train_queries_with_any_held_negative.add(q)
            identities_exposed = {identity for identity in held_ids if identity_exposure[identity] > 0}
            queries_exposed = [q for q in held if roles[q]['identity'] in identities_exposed]
            fold_rows.append({
                'fold': fold,
                'train_universe': universe,
                'train_query_count': len(train_subset),
                'held_query_count': len(held),
                'held_identity_count': len(held_ids),
                'train_held_query_identity_overlap': len(train_ids & held_ids),
                'train_held_query_component_overlap': len(train_components & held_components),
                'train_held_query_image_sha256_overlap': len(train_sha & held_sha),
                'held_identities_exposed_as_train_reference_negative': len(identities_exposed),
                'held_identity_exposure_fraction': len(identities_exposed) / len(held_ids),
                'held_queries_whose_identity_is_exposed': len(queries_exposed),
                'held_query_exposure_fraction': len(queries_exposed) / len(held),
                'train_queries_containing_any_held_reference_negative': len(train_queries_with_any_held_negative),
                'train_query_fraction_with_held_reference_negative': len(train_queries_with_any_held_negative) / len(train_subset),
                'held_identity_negative_candidate_occurrences': sum(identity_exposure.values()),
                'distinct_exposed_held_identity_reference_rows': len(set().union(*exposed_reference_rows.values())),
            })
            for q in sorted(held):
                identity = roles[q]['identity']
                detailed.append({
                    'fold': fold, 'train_universe': universe, 'query_id': q,
                    'display_id': roles[q]['original_query_id'], 'identity': identity,
                    'target_in_C128': present[q],
                    'identity_exposed_as_training_reference_negative': identity in identities_exposed,
                    'negative_candidate_occurrences': identity_exposure[identity],
                    'distinct_training_queries_with_identity_negative': len(identity_training_queries[identity]),
                    'reference_physical_rows_seen_as_negatives': sorted(exposed_reference_rows[identity]),
                })
    assert all(r['train_held_query_identity_overlap'] == 0 for r in fold_rows)
    assert all(r['train_held_query_component_overlap'] == 0 for r in fold_rows)
    assert all(r['train_held_query_image_sha256_overlap'] == 0 for r in fold_rows)
    output = {
        'status': 'ORIGINAL_H593_FOLD_REFERENCE_EXPOSURE_AUDIT_PASS',
        'population': 593, 'natural_candidates_per_query': 128,
        'query_identity_overlap_across_train_held': 0,
        'query_component_overlap_across_train_held': 0,
        'query_image_sha256_overlap_across_train_held': 0,
        'by_fold': fold_rows,
        'by_held_query': detailed,
        'sources': {'curator': binding(curator_path), 'cache': binding(cache_path), 'gallery': binding(gallery_path), 'original_fold_payloads': fold_sources},
        'program': binding(__file__),
        'interpretation': [
            'Original query identity/component folds remain disjoint; no split or training inputs are changed.',
            'Natural C128 can contain gallery references of held query identities as negative candidates in other training queries.',
            'This is fixed-gallery reference exposure, distinct from seeing held positive query labels or held query images.',
            'Therefore query-group OOF is not an unseen-reference-gallery inductive protocol; report this scope explicitly.',
            'The target-present subset quantifies exposure if loss updates skip training queries whose target is absent; it does not assert which rule a future trainer uses.',
            'Labels are read only to describe exposure in this audit, not supplied to optimization or split selection.',
        ],
        'training_updates': 0, 'model_forward_calls': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fold_exposure_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'fold_exposure_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(fold_rows)
    print(json.dumps({'status': output['status'], 'by_fold': fold_rows}, indent=2))


if __name__ == '__main__':
    main()
