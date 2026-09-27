#!/usr/bin/env python3
"""Independent read-only arithmetic review of sealed property add-ons.

Does not import either experiment, fit parameters, or invoke evaluation.
Only writes new audit artifacts; all experiment artifacts remain immutable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RC = Path(__file__).resolve().parents[1]
DEFAULT = RC / 'results/rc_m_causal128_attribution_chain_v1'


def read(p):
    return json.loads(Path(p).read_text())


def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write_new(p, d):
    text = json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, ('AUDIT_OUTPUT_EXISTS_CHANGED', str(p))
    else:
        p.write_text(text)


def verify_bindings(obj):
    if isinstance(obj, dict):
        if set(obj) == {'path', 'sha256'}:
            assert binding(obj['path']) == obj, obj['path']
            return 1
        return sum(verify_bindings(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(verify_bindings(v) for v in obj)
    return 0


def feature(meta, mm):
    a, b = meta['raw_scores_pair']; la, lb = meta['free_content_pair']; ma, mb = mm
    sym = lambda a, b: (a-b)/(abs(a)+abs(b)+1e-12)
    return [(a-b)/meta['raw_C128_population_std'], sym(ma*la, mb*lb), sym(ma, mb), sym(la, lb)]


def effects(values):
    def avg(prefix):
        return math.fsum(values[prefix+d] for d in ['X_PLUS', 'X_MINUS', 'Y_PLUS', 'Y_MINUS']) / 4
    ag, al = avg('GLOBAL_'), avg('LOCAL_')
    bg, bl = avg('AMP_PERMUTE__GLOBAL_'), avg('AMP_PERMUTE__LOCAL_')
    return dict(phase_restoration_after_amplitude_permutation=bg-bl,
                amplitude_restoration_under_local_phase=al-bl,
                factorial_interaction=(ag-al)-(bg-bl),
                phase_restoration_native_amplitude=ag-al)


def main(root):
    comp = root/'property_compensation_fixed_pairs_v1'
    bridge = root/'newgroup_property_bridge_v1'
    phase = root/'phase_replication'
    cp, cv, cr = [read(comp/f) for f in ['protocol.json', 'validation.json', 'result.json']]
    bp, bv, br = [read(bridge/f) for f in ['protocol.json', 'validation.json', 'result.json']]
    pv, pr = [read(phase/f) for f in ['validation.json', 'result.json']]
    assert cv['status'] == 'PROPERTY_FIXED_PAIR_COMPENSATION_INDEPENDENT_PASS'
    assert bv['status'] == 'NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS'
    assert pv['status'] == 'PHASE_NEWGROUP_REPLICATION_JOIN_PASS'
    hashes = sum(verify_bindings(d) for d in [cp, cv, cr, bp, bv, br, pv, pr])
    manifest = read(comp/'pair_manifest.json'); fv = read(comp/'fit_validation.json')
    train_input = read(comp/'training_inputs.json'); test_input = read(comp/'test_inputs.json')
    old = {r['index']: r for r in read(cp['sources']['old_result']['path'])['rows']}
    held = {r['index']: r for r in pr['rows']}
    original = {r['index']: r for r in read(root/'bridge/inputs.json')['rows']}
    cache = {r['index']: r for r in read(root/'paths/cache.json')['rows']}
    assert not {r['component'] for r in manifest['train']} & {r['component'] for r in manifest['test']}
    metadata_errors = []
    for m in manifest['train']+manifest['test']:
        i = m['index']; item = original[i]; c = cache[i]; row = item['post_row']
        t, w = m['target_position'], m['wrong_position']
        assert item['query_id'] == m['query_id'] == c['query_id']
        assert item['component'] == m['component'] and c['axis'] == m['candidate_axis']
        assert [c['axis'][t], c['axis'][w]] == m['physical_rows']
        assert set(m['all_target_positions']) == {j for j, v in enumerate(row['candidate_identities']) if v == m['target_identity']}
        ll = np.asarray(c['free_content']); rr = np.asarray(row['raw_scores'])
        assert w == max((j for j in range(128) if j not in m['all_target_positions']), key=lambda j: ll[j])
        metadata_errors += [abs(float(rr.std())-m['raw_C128_population_std'])]
        metadata_errors += list(abs(rr[[t,w]]-m['raw_scores_pair'])) + list(abs(ll[[t,w]]-m['free_content_pair']))
    arrays = {}; features_max = 0.
    for split, saved in [('train', train_input), ('test', test_input)]:
        seen = []; arrays[split] = {}
        for cell, arms in cp['cells'].items():
            groups = []
            for m in manifest[split]:
                group = []
                for arm in arms:
                    if split == 'train':
                        mm = [old[m['index']]['mass_by_arm'][arm][str(j)] for j in [m['target_position'], m['wrong_position']]]
                    else:
                        h = held[m['index']]
                        assert h['query_id'] == m['query_id'] and h['group'] == m['component']
                        mm = [h['masses'][role][arm] for role in ['target', 'fixed_wrong']]
                    x = feature(m, mm); group.append(x); seen.append((cell, m['index'], arm, mm, x))
                groups.append(group)
            arrays[split][cell] = np.asarray(groups)
        assert len(seen) == len(saved['rows'])
        for x, s in zip(seen, saved['rows']):
            assert x[:3] == (s['cell'], s['index'], s['arm']) and x[3] == s['masses']
            features_max = max(features_max, float(np.max(abs(np.asarray(x[4])-s['features']))))
            if split == 'train': assert s['split'] == 'TRAIN' and s['index'] not in cp['test_indices']
    scales = np.sqrt(np.mean(arrays['train'][cp['reference_cell']]**2, axis=(0,1)))
    assert np.array_equal(scales, np.asarray(train_input['scales']))
    fits = {n: read(v['path']) for n, v in fv['fits'].items()}
    certificates = {}; score_errors = []; loss_errors = []
    for name, f in fits.items():
        assert f['scales'] == scales.tolist() and f['protocol'] == fv['protocol'] and f['training_inputs'] == fv['training_inputs']
        aa = (arrays['train'][f['cell']]/scales)[:,:,f['feature_indices']].reshape(28, -1)
        theta = np.asarray(f['theta']); lam = cp['ridge_lambda']; n, d = aa.shape
        grad = lam*theta; hess = lam*np.eye(d); losses = []
        for row in aa:
            z = math.fsum(float(a*b) for a,b in zip(row,theta)); probability = 1/(1+math.exp(z))
            losses.append(max(0.,-z)+math.log1p(math.exp(-abs(z))))
            grad -= row*probability/n; hess += np.outer(row,row)*probability*(1-probability)/n
        obj = math.fsum(losses)/n+lam/2*math.fsum(theta*theta)
        err = max(abs(obj-f['certificate']['objective']), float(max(abs(grad-f['certificate']['gradient']))))
        cert = dict(independent_error=err, gradient_inf=float(max(abs(grad))),
                    hessian_min_eigenvalue=float(np.linalg.eigvalsh(hess).min()),
                    objective_gap_bound=float(grad@grad/(2*lam)))
        assert cert['gradient_inf'] <= 1.01e-10 and cert['hessian_min_eigenvalue'] >= .001-1e-12 and err < 1e-12
        certificates[name] = cert
    for cell in cp['cells']:
        for method, name in [('FROZEN_GLOBAL', cp['reference_cell']), ('REFIT', cell), ('RAW_L_ONLY', 'RAW_L_ONLY')]:
            f = fits[name]; xx = (arrays['test'][cell]/scales)[:,:,f['feature_indices']]
            margins = [[math.fsum(float(a*b) for a,b in zip(row,f['theta'])) for row in group] for group in xx]
            for meta, values, saved in zip(manifest['test'], margins, cr['evaluations'][cell][method]['rows']):
                assert saved['index'] == meta['index'] and saved['component'] == meta['component']
                score_errors.append(float(max(abs(np.asarray(values)-saved['margins']))))
                loss = math.fsum(max(0.,-z)+math.log1p(math.exp(-abs(z))) for z in values)/4
                loss_errors.append(abs(loss-saved['group_mean_loss']))
    assert max(score_errors) < 1e-12 and max(loss_errors) < 1e-12 and features_max < 1e-13
    log_errors = []; contrast_errors = []; aggregate_errors = []; endpoint_errors = []
    assert len(br['rows']) == len(pr['rows']) == 9
    for row in br['rows']:
        src = held[row['index']]; assert row['query_id'] == src['query_id'] and row['component'] == src['group']
        for role in ['target', 'fixed_wrong']:
            values = {}
            for arm, sides in row['individual_endpoints'].items():
                values[arm] = sides[role]['logM']
                log_errors += [abs(values[arm]-math.log(src['masses'][role][arm])), abs(sides[role]['M']-src['masses'][role][arm])]
            eff = effects(values)
            contrast_errors += [abs(v-src['contrasts'][role+'_logM'][k]) for k,v in eff.items()]
        for arm, g in row['arms'].items():
            sides = row['individual_endpoints'][arm]
            for metric, v in g.items():
                field = metric[:-4] if metric.endswith('_gap') else metric
                endpoint_errors.append(abs(v-(sides['target'][field]-sides['fixed_wrong'][field])))
            for role, position in [('target',row['target_position']),('fixed_wrong',row['fixed_wrong_position'])]:
                if position == row['RAW_winner']:
                    assert all(sides[role][k] == 0 for k in ['NATIVE7_action','M_FREE_action','POST_action'])
        gap = effects({arm: v['logM_gap'] for arm,v in row['arms'].items()})
        contrast_errors += [abs(v-src['contrasts']['fixed_target_wrong_logM_gap'][k]) for k,v in gap.items()]
    for key, summary in [('fixed_target_wrong_logM_gap',br['gap_summary']['logM_gap']),
                         ('target_logM',br['individual_summary']['target']['logM']),
                         ('fixed_wrong_logM',br['individual_summary']['fixed_wrong']['logM'])]:
        for contrast, values in summary.items():
            source = pr['summary'][key][contrast]
            aggregate_errors += [abs(values['mean']-source['group_equal_mean'])]
            assert values['positive_groups'] == source['positive_groups'] and values['negative_groups'] == source['negative_groups']
    assert max(log_errors+contrast_errors+aggregate_errors+endpoint_errors) < 2e-10
    assert not br['accuracy_claim'] and not br['full_C128_predictions_computed'] and not br['training']
    assert not cr['full_C128_action_test'] and not cr['parameters_refit_on_test'] and not cr['unique_cause_claim']
    audit = dict(status='PROPERTY_NEWGROUP_BRIDGE_AND_COMPENSATION_SECOND_REVIEW_PASS',
        reviewer_code=binding(__file__), source_validations=[binding(comp/'validation.json'),binding(bridge/'validation.json'),binding(phase/'validation.json')],
        bound_hash_checks=hashes, metadata_pairs=16, metadata_max_error=max(metadata_errors),
        train_groups=7, test_groups=9, component_overlap=0, train_features=112, test_features=144,
        train_only_shared_scale_exact=True, training_certificates=certificates,
        features_max_error=features_max, compensation_score_max_error=max(score_errors),
        compensation_loss_max_error=max(loss_errors), bridge_logM_and_M_source_max_error=max(log_errors),
        bridge_phase_per_group_contrast_max_error=max(contrast_errors),
        bridge_phase_12_aggregate_contrast_max_error=max(aggregate_errors),
        gap_equals_target_minus_wrong_max_error=max(endpoint_errors), zero_HOLD_anchor_verified=True,
        no_experiment_mutations=True, no_formal_evaluator_called=True,
        boundaries=['Fixed target/wrong diagnostic, not complete C128 action.',
            'Seven TRAIN groups and nine groups absent prior F71; all already opened in historical H593.',
            'Certificates establish numerical optimum in this fixed regularized scalar family only.',
            'GLOBAL is coherent phase translation, not native P; four signs/axes are within-group replicates.',
            'No uniqueness, universal necessity, or equivalence claim.'])
    out = root/'independent_property_review_v1'
    write_new(out/'validation.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    main(parser.parse_args().root.resolve())
