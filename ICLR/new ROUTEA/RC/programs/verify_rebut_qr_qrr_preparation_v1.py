#!/usr/bin/env python3
"""Independent frozen-panel/split/basis check; no training and no score tuning."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_v1'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(binding):
    assert bind(binding['path']) == binding
    return read(binding['path'])


def keyed(rows):
    d = {r['query_id']: r for r in rows}
    assert len(d) == len(rows)
    return d


def main():
    protocol_path = OUT / 'protocol.json'
    protocol = read(protocol_path)
    assert protocol['status'] == 'F71_PROTOCOL_FROZEN_BEFORE_NEW_FIT'
    assert bind(protocol['common_features']['path']) == protocol['common_features']
    common = torch.load(protocol['common_features']['path'], map_location='cpu', weights_only=False)
    assert common['labels_included'] is False
    data = keyed(common['records'])
    simple = keyed(checked(protocol['sources']['simple_cache'])['rows'])
    roles_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(roles_path)['records'])
    split = checked(protocol['sources']['grouped_split'])
    contents = {}
    for binding in protocol['sources']['content_features']:
        receipt = checked(binding['validation'])
        payload = checked(binding['payload'])
        assert receipt['status'] == 'CONTENT_ALL_PAIRS_NUMPY_PASS' and receipt['payload'] == binding['payload']
        for r in payload['records']:
            assert r['query_id'] not in contents
            contents[r['query_id']] = r
    assert len(data) == 593 and set(data) == set(simple) == set(curator) == set(contents)
    observed_ordinals = set()
    for q, r in data.items():
        assert set(r) == {'query_id', 'execution_ordinal', 'axis', 'winner', 'X0', 'M0'}, (q, 'unexpected learned/label fields')
        s = simple[q]
        c = contents[q]
        assert r['axis'] == s['axis'] == c['candidate_physical_rows']
        assert r['winner'] == s['winner'] == c['winner']
        assert r['execution_ordinal'] == s['execution_ordinal'] == curator[q]['execution_ordinal']
        observed_ordinals.add(r['execution_ordinal'])
        assert r['X0'].shape == (128, 18) and r['X0'].dtype == torch.float64
        assert r['M0'].shape == (128,) and r['M0'].dtype == torch.float64
        assert np.array_equal(r['M0'].numpy(), s['mass'])
        assert len(set(r['axis'])) == 128
        positions = s['challengers']
        assert positions == c['challenger_positions'] and set(positions) == set(range(128)) - {r['winner']}
        old = np.asarray(s['native_X'], dtype=np.float64)
        stats = np.asarray(c['statistics'], dtype=np.float64)
        delta = stats[positions] - stats[r['winner']]
        scale = np.abs(stats[positions]) + np.abs(stats[r['winner']]) + 1e-12
        new_content = delta / scale
        assert np.array_equal(new_content, np.asarray(c['X'])[:, 1:])
        expected = np.zeros((128, 18), dtype=np.float64)
        expected[positions, :6] = old
        expected[positions, 6:12] = old * old
        expected[positions, 12:17] = new_content
        expected[positions, 17] = 1
        assert np.array_equal(expected, r['X0'].numpy()), (q, 'independent 18-coordinate reconstruction')
    assert observed_ordinals == set(range(593))
    panel = {q for q, r in data.items() if 0 <= r['execution_ordinal'] <= 70}
    assert len(panel) == 71 and set(protocol['panel_query_ids']) == panel
    assert {data[q]['execution_ordinal'] for q in panel} == set(range(71))
    fold_summary = []
    held_union = set()
    for fold in range(5):
        original = split['folds'][fold]
        f = protocol['folds'][str(fold)]
        train = set(f['train_query_ids'])
        held = set(f['heldout_query_ids'])
        fit = set(f['inner_fit_query_ids'])
        val = set(f['inner_val_query_ids'])
        fold_roles = keyed(checked(f['train_roles'])['records'])
        assert train == set(original['train_query_ids']) & panel
        assert held == set(original['heldout_query_ids']) & panel
        assert fit and val and held and fit | val == train
        assert not fit & val and not train & held and train | held == panel
        assert not held_union & held
        held_union.update(held)
        assert train <= set(fold_roles) and not held & set(fold_roles)
        groups = sorted({fold_roles[q]['component'] for q in train}, key=lambda c: hashlib.sha256(('REBUT_QR_QRR_V1|' + c).encode()).hexdigest())
        selected_validation_groups = set(groups[:max(1, math.ceil(.2 * len(groups)))])
        assert val == {q for q in train if fold_roles[q]['component'] in selected_validation_groups}
        for field in ['identity', 'component', 'source_image_sha256']:
            sets = [{curator[q][field] for q in ids} for ids in [fit, val, held]]
            assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        fold_summary.append({'fold': fold, 'fit_queries': len(fit), 'validation_queries': len(val), 'held_queries': len(held), 'fit_components': len({curator[q]['component'] for q in fit}), 'validation_components': len({curator[q]['component'] for q in val}), 'held_components': len({curator[q]['component'] for q in held})})
    assert held_union == panel
    history = read(OUT / 'historical496_predictions.json')
    assert history['label_reads'] == 0 and history['new_training_updates'] == 0
    history_predictions = history['records']
    maximum_error = 0.0
    replayed = set()
    history_sources = []
    for fold in range(5):
        path = RC / f'results/rc_h593_box_ce_polish_v1/fold{fold}/payload.json'
        receipt = read(path.with_name('validation.json'))
        assert receipt['payload'] == bind(path)
        payload = read(path)
        history_sources.append(bind(path))
        theta = np.asarray([float.fromhex(x) for x in payload['parameters']['CONTENT_BOX_CE_POLISH18']])
        assert theta.shape == (18,)
        train = set(payload['train_query_ids'])
        for p in payload['predictions']:
            q = p['query_id']
            assert q not in train and q not in replayed
            replayed.add(q)
            r = data[q]
            values = r['X0'].numpy() @ theta
            challenger = p['challenger_positions']
            original_scores = np.asarray([float.fromhex(x) for x in p['models']['CONTENT_BOX_CE_POLISH18']['logits_hex']])
            error = float(np.max(np.abs(values[challenger] - original_scores)))
            maximum_error = max(maximum_error, error)
            assert error < 2e-10
            best = max(challenger, key=lambda j: values[j])
            selected = r['axis'][best if values[best] > 0 else r['winner']]
            assert selected == p['models']['CONTENT_BOX_CE_POLISH18']['selected'] == history_predictions[q]['selected']
            assert np.array_equal(values, history_predictions[q]['scores'])
            assert history_predictions[q]['fold'] == fold
    assert replayed == set(data)
    output = {
        'status': 'REBUT_QR_QRR_PREPARATION_INDEPENDENT_PASS',
        'protocol': bind(protocol_path), 'common_features': protocol['common_features'],
        'historical496_predictions': bind(OUT / 'historical496_predictions.json'),
        'historical496_source_folds': history_sources,
        'queries_reconstructed': 593, 'natural_candidate_rows': 593 * 128,
        'feature_coordinates': 18, 'feature_names': 'native six, their six squares, five full-reference content contrasts, challenger intercept',
        'panel_queries': 71, 'panel_ordinals': list(range(71)),
        'panel_selection': 'All existing ordinal0..70; no correctness/M/score conditions in selection',
        'original_outer_folds_preserved': True,
        'inner_query_identity_component_image_sha_disjoint': True,
        'folds': fold_summary,
        'historical496_max_abs_logit_error': maximum_error,
        'common_features_contain_labels_or_trained_head_weights': False,
        'checkpoint_initialization_review': 'Preparation stores no head parameters in common_features; protocol requires zero new head. Actual runner initialization must be checked separately.',
        'evidence_bundle_validation': 'Pending builder evidence_manifest; this receipt does not validate grid alignment or saved warp descriptors.',
        'scientific_scope': 'Opened 71-query development pilot; historical heads use larger outer TRAIN and are labelled comparators, not equal-training-budget ablations.',
        'sources': {'cache': protocol['sources']['simple_cache'], 'curator': bind(roles_path), 'split': protocol['sources']['grouped_split'], 'content': protocol['sources']['content_features']},
        'verifier': bind(__file__), 'new_training_updates': 0,
    }
    (OUT / 'preparation_validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['status', 'queries_reconstructed', 'feature_coordinates', 'panel_queries', 'folds', 'historical496_max_abs_logit_error']}, indent=2))


if __name__ == '__main__':
    main()
