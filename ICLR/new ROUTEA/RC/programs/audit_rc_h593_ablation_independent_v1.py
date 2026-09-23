#!/usr/bin/env python3
"""Read-only recomputation of sealed ablations; no project scorer/trainer imports."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_ablation_independent_audit_v1'
SOURCE = ROOT / 'results/rc_h593_six_feature_ablation_v1'
FEATURES = ['RAW', 'S', 'M', 'L', 'Q', 'R']

def read(path):
    return json.loads(Path(path).read_text())

bindings = []
def checked(binding):
    path = Path(binding['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == binding['sha256'], str(path)
    bindings.append(binding)
    return path

def target(r, role, labels):
    found = [i for i, p in enumerate(r['candidate_physical_rows']) if labels[p] == role['identity']]
    assert len(found) <= 1
    if not found:
        return -2
    if found[0] == r['winner']:
        return -1
    return r['challenger_positions'].index(found[0])

def choose(z, r):
    # Explicit HOLD at position zero: tied zero selects HOLD, as the sealed rule requires.
    action = int(np.argmax(np.r_[0., z]))
    pos = r['winner'] if action == 0 else r['challenger_positions'][action-1]
    return r['candidate_physical_rows'][pos]

def objective(z, y, kind):
    if kind == 'CE':
        all_scores = np.c_[np.zeros(len(z)), z]
        vmax = all_scores.max(axis=1)
        return vmax + np.log(np.exp(all_scores-vmax[:, None]).sum(axis=1)) - all_scores[np.arange(len(y)), y+1]
    values = []
    for zi, yi in zip(z, y):
        if yi == -1:
            values.append(np.logaddexp(0., zi.max()))
        else:
            values.append(np.logaddexp(0., -zi[yi]) + np.logaddexp(0., np.delete(zi, yi).max()))
    return np.array(values)

def main():
    torch.set_num_threads(1)
    authority_path = ROOT / 'registry/rc_h593_six_feature_ablation_authority_v1_20260920.json'
    authority = read(authority_path)
    sealed_validation = read(SOURCE / 'validation.json')
    result = read(checked(sealed_validation['result']))
    roles = {r['query_id']: r for r in read(checked(authority['join_sources']['curator']))['records']}
    splits = read(checked(authority['public_sources']['split']))['folds']
    gallery_path = ROOT / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'
    gallery = read(gallery_path)
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    cached = []
    for sources in authority['features']:
        for name in ('receipt', 'validation'):
            checked(sources[name])
        cached.extend(torch.load(checked(sources['payload']), map_location='cpu', weights_only=True)['records'])
    cached.sort(key=lambda r: r['execution_ordinal'])
    data = {r['query_id']: r for r in cached}
    assert len(data) == len(cached) == len(roles) == 593
    sealed_rows = {r['query_id']: r for r in result['rows']}
    counts, switches = Counter(), Counter()
    outcomes = defaultdict(dict)
    folds, losses, changes = [], [], []
    max_error = max_delta_error = 0.
    logit_checks = 0
    minimum_decision_margin = math.inf
    seen = set()
    for fold in range(5):
        validation = read(SOURCE / f'fold{fold}/validation.json')
        payload = read(checked(validation['payload']))
        old = read(checked(authority['fold_sources'][str(fold)]['full_payload']))
        assert payload['gallery_mapping_sha256'] == gallery['corrected_mapping_sha256']
        train_ids = splits[fold]['train_query_ids']
        held_ids = splits[fold]['heldout_query_ids']
        assert set(train_ids).isdisjoint(held_ids)
        assert set(payload['train_query_ids']) == set(train_ids)
        train_roles = read(checked(authority['fold_sources'][str(fold)]['train_roles']))['records']
        assert {r['query_id'] for r in train_roles} == set(train_ids)
        train_components = {r['component'] for r in train_roles}
        train_identities = {r['identity'] for r in train_roles}
        train_images = {roles[q]['source_image_sha256'] for q in train_ids}
        parameters = {}
        for name, par in payload['parameters'].items():
            theta = np.array([float.fromhex(v) for v in par['theta_hex']])
            parameters[name] = (theta, par['retained_indices'])
            kind, method = name.split('_')[:2]
            full = payload['parameters'][kind+'_FULL']['theta_hex']
            if method == 'FULL':
                assert par['theta_hex'] == old['parameters']['ALL_CE' if kind == 'CE' else kind]
            elif method == 'ZERO':
                expected = list(full)
                expected[FEATURES.index(par['dropped'])] = float(0).hex()
                assert par['theta_hex'] == expected and par['retained_indices'] == list(range(6))
            else:
                assert par['retained_indices'] == [i for i, f in enumerate(FEATURES) if f != par['dropped']]
        fold_counts = Counter()
        for pred in payload['predictions']:
            q = pred['query_id']
            assert q in held_ids and q not in seen
            seen.add(q)
            role, r = roles[q], data[q]
            assert role['outer_fold'] == fold
            assert role['component'] not in train_components and role['identity'] not in train_identities
            assert role['source_image_sha256'] not in train_images
            for key in ('candidate_physical_rows','challenger_positions','winner','execution_ordinal'):
                assert r[key] == pred[key]
            assert len(r['candidate_physical_rows']) == 128
            assert sorted(r['challenger_positions'] + [r['winner']]) == list(range(128))
            assert len({labels[p] for p in r['candidate_physical_rows']}) == 128
            x = r['modes']['REAL']['X'].numpy()
            assert x.dtype == np.float64 and x.shape == (127,6) and np.isfinite(x).all()
            raw = r['candidate_physical_rows'][r['winner']]
            counts['RAW'] += labels[raw] == role['identity']
            assert (target(r, role, labels) != -2) == sealed_rows[q]['target_in_C128']
            recomputed = {}
            for name, (theta, keep) in parameters.items():
                z = np.sum(x[:, keep] * theta[:-1], axis=1) + theta[-1]
                saved = np.array([float.fromhex(v) for v in pred['models'][name]['logits_hex']])
                error = float(np.max(np.abs(z-saved)))
                max_error = max(max_error, error)
                assert error < 2e-10
                selected = choose(z, r)
                assert selected == pred['models'][name]['selected'] == sealed_rows[q]['selected'][name]
                correct = labels[selected] == role['identity']
                assert correct == sealed_rows[q]['correct'][name]
                counts[name] += correct
                fold_counts[name] += correct
                switches[name] += selected != raw
                outcomes[name][q] = bool(correct)
                logit_checks += len(z)
                ordered = np.sort(np.r_[0., z])
                minimum_decision_margin = min(minimum_decision_margin, float(ordered[-1]-ordered[-2]))
                recomputed[name] = (z, selected)
            zfull, old_sel = recomputed['COST1_FULL']
            zzero, new_sel = recomputed['COST1_ZERO_S']
            theta = parameters['COST1_FULL'][0]
            max_delta_error = max(max_delta_error, float(np.max(np.abs(zzero-zfull+theta[1]*x[:,1]))))
            if old_sel != new_sel:
                old_correct, new_correct = labels[old_sel] == role['identity'], labels[new_sel] == role['identity']
                entry = dict(query_id=q, original_query_id=role['original_query_id'], fold=fold,
                             component=role['component'], identity=role['identity'], raw_correct=labels[raw]==role['identity'],
                             old_correct=old_correct,new_correct=new_correct,
                             change='rescue' if new_correct and not old_correct else 'break' if old_correct and not new_correct else 'wrong_to_wrong',
                             old_action='HOLD' if old_sel==raw else 'SWITCH',new_action='HOLD' if new_sel==raw else 'SWITCH',
                             old_identity=labels[old_sel],new_identity=labels[new_sel],
                             old_selected_physical_row=old_sel,new_selected_physical_row=new_sel,
                             full_top_score=float(zfull.max()),zero_top_score=float(zzero.max()))
                for side, selected in [('old',old_sel),('new',new_sel)]:
                    if selected == raw:
                        entry[side+'_full_score'] = entry[side+'_zero_score'] = 0.
                        entry[side+'_contributions'] = None
                    else:
                        j = r['challenger_positions'].index(r['candidate_physical_rows'].index(selected))
                        entry[side+'_full_score'] = float(zfull[j]); entry[side+'_zero_score'] = float(zzero[j])
                        entry[side+'_contributions'] = dict(zip(FEATURES+['bias'],map(float,np.r_[x[j]*theta[:-1],theta[-1]])))
                changes.append(entry)
        assert {p['query_id'] for p in payload['predictions']} == set(held_ids)
        folds.append(dict(fold=fold, train_n=len(train_ids),heldout_n=len(held_ids),correct=dict(fold_counts)))
        # Evaluate frozen parameters against the original surrogate without any optimization.
        for stage, ids in [('TRAIN',train_ids),('OOF',held_ids)]:
            usable = [q for q in ids if target(data[q],roles[q],labels)>=-1]
            y = np.array([target(data[q],roles[q],labels) for q in usable])
            x = np.stack([data[q]['modes']['REAL']['X'].numpy() for q in usable])
            for name, (theta, keep) in parameters.items():
                z = np.sum(x[:,:,keep]*theta[:-1],axis=2)+theta[-1]
                losses.append(dict(fold=fold,stage=stage,model=name,n=len(y),
                                   loss=float(objective(z,y,name.split('_')[0]).mean()),
                                   correct=int((np.argmax(np.c_[np.zeros(len(z)),z],axis=1)==y+1).sum())))
    assert len(seen)==593
    for name in outcomes:
        s = result['summary'][name]
        assert counts[name] == s['correct'] and switches[name] == s['switches']
    transition = Counter((r['change'],r['old_action']+'->'+r['new_action']) for r in changes)
    group_deltas = defaultdict(int)
    for r in changes:
        group_deltas[r['component']] += int(r['new_correct'])-int(r['old_correct'])
    summary = dict(status='INDEPENDENT_CACHED_FEATURE_ACTION_AND_COUNT_PASS',
                   no_project_scorer_or_trainer_imports=True,no_training_or_parameter_selection=True,
                   authority_sha256=hashlib.sha256(authority_path.read_bytes()).hexdigest(),
                   gallery_manifest_sha256=hashlib.sha256(gallery_path.read_bytes()).hexdigest(),
                   query_count=593,feature_shards=len(authority['features']),logit_checks=logit_checks,
                   max_abs_logit_error=max_error,min_top_two_action_margin=minimum_decision_margin,
                   max_s_delta_identity_error=max_delta_error,counts=dict(counts),switches=dict(switches),folds=folds,
                   s_changes=len(changes),s_transition_counts={str(k):v for k,v in transition.items()},
                   s_changed_components=len(group_deltas),s_positive_components=sum(v>0 for v in group_deltas.values()),
                   s_negative_components=sum(v<0 for v in group_deltas.values()),s_component_net=dict(group_deltas),
                   source_bindings=bindings,
                   limits=['Reads already opened H593 outcomes for retrospective diagnostics',
                           'Verifies cached-feature scoring; does not regenerate image tokens or visibility maps',
                           'No fresh confirmation and no automatic deployment change'])
    OUT.mkdir(exist_ok=True)
    for name, value in [('validation.json',summary),('s_changed_cases.json',changes),('surrogate_diagnostics.json',losses)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'surrogate_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(losses[0]));w.writeheader();w.writerows(losses)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_bindings','folds','s_component_net')},indent=2))

if __name__ == '__main__':
    main()
