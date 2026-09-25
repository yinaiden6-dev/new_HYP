#!/usr/bin/env python3
"""Post-hoc, label-free TRAIN decomposition of sealed RoMa masses.

This is a fixed-head diagnostic, not a new model search.  Source artifacts are
read-only.  Identity labels are opened only in join after all held predictions
are sealed.  Natural-C128 reference sampling limits the interpretation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_mass_main_effects_v1'
SOURCE = ROOT / 'results/rc_h593_simple_explanations_v1'
PARENT = ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
ARMS = ('ORIGINAL_M', 'REFERENCE_MAIN_EFFECT', 'REMOVE_REFERENCE_MAIN_EFFECT')
ITERATIONS = 2000
EPS = 1e-12


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def checked(bound):
    need(bind(bound['path']) == bound, 'SOURCE_SHA:' + bound['path'])
    return Path(bound['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    os.replace(tmp, path)


def prepare():
    need(not (OUT / 'protocol.json').exists(), 'FRESH_PROTOCOL_REQUIRED')
    parent = read(PARENT)
    cv = read(SOURCE / 'cache_validation.json')
    need(cv['status'] == 'CACHE_PASS', 'SOURCE_CACHE_VALIDATED')
    checked(cv['payload'])
    files = {
        'cache': cv['payload'], 'cache_validation': bind(SOURCE / 'cache_validation.json'),
        'parent': bind(PARENT), 'split': parent['public']['split'],
        'gallery': parent['public']['gallery'], 'curator': parent['join']['curator'],
        'historical_full_result': parent['join']['old_result'],
        'historical_simple_result': bind(SOURCE / 'result.json'),
    }
    heads = {str(f): bind(SOURCE / f'fold{f}/COST1_PRODUCT5.json') for f in range(5)}
    write(OUT / 'protocol.json', {
        'status': 'PREPARED_POSTHOC_MAIN_EFFECT_DIAGNOSTIC',
        'program': bind(__file__), 'sources': files, 'heads': heads,
        'head': 'Frozen per-fold COST1_PRODUCT5, equivalent MASS5 interface',
        'fit_data': 'All TRAIN query natural C128 masses; no identity labels; no held query masses',
        'reference_unit': 'physical gallery row, not identity',
        'fit': 'Unpenalized log(M)=mu+a_query+b_reference least squares by fixed ALS',
        'iterations': ITERATIONS, 'log_floor': EPS, 'sym_epsilon': EPS,
        'gauge': 'mu=TRAIN edge mean logM; reference b has observation-weighted mean zero',
        'unseen_reference': 'b=0 global geometric-mean reference prior; never fit from held',
        'graph': 'Require one connected TRAIN query/reference incidence component',
        'arms': list(ARMS),
        'mass_definitions': {'ORIGINAL_M': 'M', 'REFERENCE_MAIN_EFFECT': 'exp(mu+b_ref)',
            'REMOVE_REFERENCE_MAIN_EFFECT': 'M/exp(b_ref), preserves original zero if any'},
        'acceptance': 'Original 127 challengers; frozen five-parameter head; strict max(z)>0 otherwise RAW HOLD',
        'no_new_head_fit': True, 'new_encoder_forwards': 0, 'new_roma_forwards': 0,
        'evidence': 'Opened H593 development OOF panel. Post-hoc mechanism diagnosis, not fresh confirmation.',
        'limits': [
            'Reference prior is estimated conditional on natural-C128 TRAIN sampling; it is not universal single-image quality.',
            'REMOVE_REFERENCE still contains query scale plus unexplained pair variation, including model error.',
            'Frozen-head degradation does not establish indispensability or rule out refit recovery.',
            'Shared query scale cancels algebraically only at zero epsilon; actual epsilon error is measured.',
            'No claim that all reference or query-only explanations have been exhausted.',
        ],
    })
    print(json.dumps({'stage': 'prepare', 'protocol': bind(OUT / 'protocol.json')}), flush=True)


def guard(stage):
    p = read(OUT / 'protocol.json')
    checked(p['program'])
    allowed = {Path(v['path']).resolve() for k, v in p['sources'].items()
               if stage == 'join' or k not in ('gallery', 'curator', 'historical_full_result', 'historical_simple_result')}
    allowed.update(Path(v['path']).resolve() for v in p['heads'].values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        low = str(path).lower()
        need(not any(s in low for s in ('d1-mi', 'd1_mi', 'formal392')), 'PROTECTED_SOURCE')
        if ROOT / 'results' in path.parents:
            need(OUT in path.parents or path in allowed, 'UNLISTED_RESULT:' + str(path))
        if 'curator_roles' in low:
            need(stage == 'join' and (OUT / 'all_predictions_prelabel_seal.json').exists(), 'NO_PRESEAL_LABELS')
    sys.addaudithook(audit)
    return p


def sym(a, b, epsilon=EPS):
    return (a - b) / (np.abs(a) + np.abs(b) + epsilon)


def features(row, mass, epsilon=EPS):
    c = np.asarray(row['challengers'], dtype=np.int64)
    w = row['winner']
    l = np.asarray(row['free_content'], dtype=np.float64)
    raw_delta = np.asarray(row['X'], dtype=np.float64)[:, 0]
    return np.stack([raw_delta, sym(mass[c]*l[c], mass[w]*l[w], epsilon),
                     sym(mass[c], mass[w], epsilon), sym(l[c], l[w], epsilon)], axis=1)


def decision(row, logits):
    j = int(np.argmax(logits))
    pos = row['challengers'][j] if logits[j] > 0 else row['winner']
    return {'selected_position': int(pos), 'selected_physical_row': row['axis'][pos],
            'switch': bool(logits[j] > 0), 'max_challenger_logit': float(logits[j])}


def connected_components(axis):
    # Each query's 128 references are a clique in the bipartite incidence graph.
    refs = np.unique(axis)
    parent = {int(x): int(x) for x in refs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in axis:
        first = find(int(row[0]))
        for g in row[1:]:
            parent[find(int(g))] = first
    return len({find(int(x)) for x in refs})


def fit_effects(train_rows):
    mass = np.asarray([r['mass'] for r in train_rows], dtype=np.float64)
    axis = np.asarray([r['axis'] for r in train_rows], dtype=np.int64)
    need(np.isfinite(mass).all() and (mass >= 0).all(), 'FINITE_NONNEGATIVE_M')
    components = connected_components(axis)
    need(components == 1, 'DISCONNECTED_REFERENCE_GAUGES_REQUIRE_NEW_PROTOCOL')
    count = np.bincount(axis.ravel(), minlength=5413)
    seen = count > 0
    y = np.log(np.maximum(mass, EPS))
    mu = float(y.mean())
    b = np.zeros(5413, dtype=np.float64)
    delta = math.inf
    for _ in range(ITERATIONS):
        a = (y - mu - b[axis]).mean(axis=1)
        sums = np.bincount(axis.ravel(), weights=(y - mu - a[:, None]).ravel(), minlength=5413)
        new_b = np.zeros_like(b)
        new_b[seen] = sums[seen] / count[seen]
        shift = float(np.dot(count, new_b) / count.sum())
        new_b[seen] -= shift
        a += shift
        delta = float(np.max(np.abs(new_b - b)))
        b = new_b
    residual = y - mu - a[:, None] - b[axis]
    normal_q = float(np.max(np.abs(residual.mean(axis=1))))
    normal_g_sum = np.bincount(axis.ravel(), weights=residual.ravel(), minlength=5413)
    normal_g = float(np.max(np.abs(normal_g_sum[seen] / count[seen])))
    need(max(delta, normal_q, normal_g) < 1e-9, 'FIXED_ALS_NOT_CONVERGED')
    total_sse = float(np.square(y - mu).sum())
    query_sse = float(np.square(y - y.mean(axis=1, keepdims=True)).sum())
    joint_sse = float(np.square(residual).sum())
    return mu, a, b, count, {
        'train_queries': len(train_rows), 'train_pairs': int(mass.size),
        'train_query_ids': [r['query_id'] for r in train_rows],
        'connected_components': components, 'reference_seen_count': int(seen.sum()),
        'reference_unseen_count': int((~seen).sum()), 'minimum_mass': float(mass.min()),
        'clamped_log_count': int((mass < EPS).sum()), 'iterations': ITERATIONS,
        'last_update_max': delta, 'query_normal_equation_max': normal_q,
        'reference_normal_equation_max': normal_g,
        'log_mass_total_sse': total_sse, 'query_only_residual_sse': query_sse,
        'query_reference_residual_sse': joint_sse,
        'reference_incremental_r2_after_query': 1 - joint_sse/query_sse,
        'joint_r2': 1 - joint_sse/total_sse,
        'label_reads': 0,
    }


def run_fold(p, fold):
    start = time.monotonic()
    d = OUT / f'fold{fold}'
    need(not (d / 'validation.json').exists(), 'FOLD_ALREADY_SEALED')
    cache = read(checked(p['sources']['cache']))
    split = read(checked(p['sources']['split']))['folds'][fold]
    rows = cache['rows']
    train_ids = set(split['train_query_ids'])
    held_ids = set(split['heldout_query_ids'])
    train = [r for r in rows if r['query_id'] in train_ids]
    held = [r for r in rows if r['query_id'] in held_ids]
    need(not train_ids & held_ids and len(train) + len(held) == 593, 'ORIGINAL_FOLD_SPLIT')
    need(not {r['source_image_sha256'] for r in train} & {r['source_image_sha256'] for r in held}, 'SOURCE_DISJOINT')
    old = read(checked(p['heads'][str(fold)]))
    need(old['model'] == 'COST1_PRODUCT5' and old['heldout_label_reads'] == 0, 'FROZEN_HEAD_SOURCE')
    theta = np.array([float.fromhex(x) for x in old['theta_hex']])
    need(theta.shape == (5,), 'MASS5_THETA')
    mu, a, b, count, fit = fit_effects(train)
    predictions = []
    max_original_feature_error = 0.
    max_original_logit_error = 0.
    scale_epsilon_feature_error = 0.
    scale_zero_epsilon_error = 0.
    query_scale_changed_actions = 0
    held_within_sse = 0.
    held_after_ref_sse = 0.
    for i, r in enumerate(held):
        m = np.array(r['mass'], dtype=np.float64)
        axis = np.asarray(r['axis'], dtype=np.int64)
        prior = np.exp(mu + b[axis])
        residual_m = m / np.exp(b[axis])
        ms = {'ORIGINAL_M': m, 'REFERENCE_MAIN_EFFECT': prior,
              'REMOVE_REFERENCE_MAIN_EFFECT': residual_m}
        models = {}
        for name, val in ms.items():
            x = features(r, val)
            z = x @ theta[:-1] + theta[-1]
            need(np.isfinite(z).all(), 'FINITE_LOGITS')
            models[name] = dict(decision(r, z), mass=val.tolist(),
                features=x.tolist(), logits=z.tolist(), logits_hex=[float(v).hex() for v in z])
        err = float(np.max(np.abs(np.asarray(models['ORIGINAL_M']['features']) - np.asarray(r['X']))))
        max_original_feature_error = max(max_original_feature_error, err)
        old_z = np.asarray(old['held_logits'][i])
        need(old_z.shape == (127,), 'OLD_LOGIT_AXIS')
        max_original_logit_error = max(max_original_logit_error, float(np.max(np.abs(old_z - models['ORIGINAL_M']['logits']))))
        scale = float(np.exp(np.log(np.maximum(m, EPS)).mean()))
        normalized = m / scale
        xnormalized = features(r, normalized)
        scale_epsilon_feature_error = max(scale_epsilon_feature_error, float(np.max(np.abs(xnormalized - models['ORIGINAL_M']['features']))))
        scale_zero_epsilon_error = max(scale_zero_epsilon_error, float(np.max(np.abs(features(r, normalized, 0.) - features(r, m, 0.)))))
        normalized_action = decision(r, xnormalized @ theta[:-1] + theta[-1])
        query_scale_changed_actions += normalized_action['selected_physical_row'] != models['ORIGINAL_M']['selected_physical_row']
        logm = np.log(np.maximum(m, EPS))
        centered = logm - logm.mean()
        centered_after = (logm - b[axis]) - (logm - b[axis]).mean()
        held_within_sse += float(np.square(centered).sum())
        held_after_ref_sse += float(np.square(centered_after).sum())
        predictions.append({
            'query_id': r['query_id'], 'execution_ordinal': r['execution_ordinal'], 'fold': fold,
            'axis': r['axis'], 'winner': r['winner'], 'challengers': r['challengers'],
            'raw_winner': r['axis'][r['winner']], 'free_content': r['free_content'],
            'raw_standardized_delta': np.asarray(r['X'])[:, 0].tolist(),
            'reference_train_observations': count[axis].tolist(),
            'reference_log_main_effect': b[axis].tolist(),
            'unseen_reference_positions': np.flatnonzero(count[axis] == 0).tolist(),
            'unseen_reference_count': int(np.sum(count[axis] == 0)),
            'held_query_geometric_mean_used_only_for_algebra_check': scale,
            'models': models,
        })
    need(max_original_feature_error < 2e-10 and max_original_logit_error < 2e-10, 'ORIGINAL_MASS5_REPLAY')
    fit['held_unlabeled_within_log_mass_sse'] = held_within_sse
    fit['held_after_train_reference_main_effect_sse'] = held_after_ref_sse
    fit['held_within_r2_explained_by_train_reference_prior'] = 1 - held_after_ref_sse/held_within_sse
    write(d / 'effects.json', {'protocol': bind(OUT / 'protocol.json'), 'fold': fold,
        'mu': mu, 'query_effects': [{'query_id': r['query_id'], 'a': float(v)} for r, v in zip(train, a)],
        'reference_effects': [{'physical_row': g, 'b': float(b[g]), 'observations': int(count[g]),
                               'fallback': bool(count[g] == 0)} for g in range(5413)],
        'fit': fit})
    write(d / 'predictions.json', {'protocol': bind(OUT / 'protocol.json'), 'fold': fold,
        'head': p['heads'][str(fold)], 'theta_hex': old['theta_hex'],
        'query_ids': [r['query_id'] for r in held], 'predictions': predictions, 'label_reads': 0})
    validation = {'status': 'FOLD_LABEL_FREE_MASS_DECOMPOSITION_AND_ORIGINAL_REPLAY_PASS',
        'protocol': bind(OUT / 'protocol.json'), 'fold': fold,
        'effects': bind(d / 'effects.json'), 'predictions': bind(d / 'predictions.json'),
        'train_queries': len(train), 'held_queries': len(held), 'fit': fit,
        'maximum_original_feature_error': max_original_feature_error,
        'maximum_original_logit_error': max_original_logit_error,
        'query_scale_actual_epsilon_max_feature_error': scale_epsilon_feature_error,
        'query_scale_zero_epsilon_max_feature_error': scale_zero_epsilon_error,
        'query_scale_changed_actions': int(query_scale_changed_actions),
        'seconds': time.monotonic()-start, 'label_reads': 0}
    write(d / 'validation.json', validation)
    print(json.dumps({k: v for k, v in validation.items() if k not in ('fit', 'effects', 'predictions', 'protocol')}), flush=True)


def stat(rows, model, base='RAW'):
    c = np.array([r['correct'][model] for r in rows], dtype=bool)
    b = np.array([r['correct'][base] for r in rows], dtype=bool)
    return {'count': len(rows), 'correct': int(c.sum()), 'baseline_correct': int(b.sum()),
            'rescue': int((c & ~b).sum()), 'breaks': int((~c & b).sum()),
            'net': int(c.sum()-b.sum())}


def join(p):
    validations = [read(OUT / f'fold{f}/validation.json') for f in range(5)]
    rows = []
    for f, v in enumerate(validations):
        need(v['status'] == 'FOLD_LABEL_FREE_MASS_DECOMPOSITION_AND_ORIGINAL_REPLAY_PASS', 'FOLD_PASS')
        need(v['protocol'] == bind(OUT / 'protocol.json') and v['fold'] == f, 'FOLD_PROTOCOL')
        checked(v['effects'])
        rows.extend(read(checked(v['predictions']))['predictions'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'EXACT_OOF593')
    write(OUT / 'all_predictions_prelabel_seal.json', {'protocol': bind(OUT / 'protocol.json'),
        'fold_validations': [bind(OUT / f'fold{f}/validation.json') for f in range(5)],
        'queries': 593, 'label_reads': 0})
    curator = {r['query_id']: r for r in read(checked(p['sources']['curator']))['records']}
    gallery = {r['physical_row']: r['identity'] for r in read(checked(p['sources']['gallery']))['records']}
    old_simple = {r['query_id']: r for r in read(checked(p['sources']['historical_simple_result']))['rows']}
    old_full = {r['query_id']: r for r in read(checked(p['sources']['historical_full_result']))['rows']}
    for r in rows:
        q = curator[r['query_id']]
        need(q['outer_fold'] == r['fold'], 'JOIN_FOLD')
        target_positions = [i for i, g in enumerate(r['axis']) if gallery[g] == q['identity']]
        r.update(component=q['component'], group=q['group'], target_identity=q['identity'],
            original_query_id=q['original_query_id'], target_in_C128=bool(target_positions),
            target_positions=target_positions,
            target_unseen_reference=bool(target_positions) and all(r['reference_train_observations'][i] == 0 for i in target_positions),
            target_any_reference_unseen=bool(target_positions) and any(r['reference_train_observations'][i] == 0 for i in target_positions),
            target_train_observation_counts=[r['reference_train_observations'][i] for i in target_positions])
        r['correct'] = {'RAW': gallery[r['raw_winner']] == q['identity'],
                        'ORIGINAL_NATIVE7': bool(old_full[r['query_id']]['correct']['COST1'])}
        for model in ARMS:
            r['correct'][model] = gallery[r['models'][model]['selected_physical_row']] == q['identity']
        old = old_simple[r['query_id']]
        need(old['selected']['COST1_PRODUCT5@zero'] == r['models']['ORIGINAL_M']['selected_physical_row'], 'SIMPLE_ORIGINAL_DECISION')
        need(old['correct']['COST1_PRODUCT5@zero'] == r['correct']['ORIGINAL_M'], 'SIMPLE_ORIGINAL_CORRECT')
        need(bool(old_full[r['query_id']]['correct']['RAW']) == r['correct']['RAW'], 'RAW_CORRECT')
    models = ('RAW', 'ORIGINAL_NATIVE7') + ARMS
    summary = {m: stat(rows, m) for m in models}
    comparisons = {m: stat(rows, m, 'ORIGINAL_M') for m in ARMS[1:]}
    strata = {
        'target_present_and_seen': [r for r in rows if r['target_in_C128'] and not r['target_unseen_reference']],
        'target_present_unseen': [r for r in rows if r['target_unseen_reference']],
        'target_absent': [r for r in rows if not r['target_in_C128']],
        'all_candidate_references_seen': [r for r in rows if not r['unseen_reference_count']],
        'some_candidate_reference_unseen': [r for r in rows if r['unseen_reference_count']],
    }
    result = {'status': 'POSTHOC_H593_FROZEN_HEAD_MASS_MAIN_EFFECTS_COMPLETE',
        'protocol': bind(OUT / 'protocol.json'), 'prelabel_seal': bind(OUT / 'all_predictions_prelabel_seal.json'),
        'summary': summary, 'comparisons_vs_original_mass': comparisons,
        'fold_summary': {str(f): {m: stat([r for r in rows if r['fold'] == f], m) for m in models} for f in range(5)},
        'stratified_summary': {name: {m: stat(sub, m) for m in models} for name, sub in strata.items()},
        'stratified_comparisons_vs_original': {name: {m: stat(sub, m, 'ORIGINAL_M') for m in ARMS[1:]} for name, sub in strata.items()},
        'groups': {g: {m: stat([r for r in rows if r['component'] == g], m) for m in models} for g in sorted({r['component'] for r in rows})},
        'reference_coverage': {'held_pairs': 593*128, 'held_unseen_pairs': sum(r['unseen_reference_count'] for r in rows),
            'held_unseen_pair_fraction': sum(r['unseen_reference_count'] for r in rows)/(593*128),
            'target_present_queries': sum(r['target_in_C128'] for r in rows),
            'target_present_unseen_queries': sum(r['target_unseen_reference'] for r in rows)},
        'fold_diagnostics': validations, 'rows': sorted(rows, key=lambda r: r['execution_ordinal']),
        'limits': p['limits'], 'new_head_training': False, 'new_gpu_forwards': 0,
        'head_unchanged': True, 'threshold_unchanged': True, 'external_GO': False}
    write(OUT / 'result.json', result)
    print(json.dumps({'stage': 'join', 'summary': summary, 'comparisons_vs_original_mass': comparisons,
                      'reference_coverage': result['reference_coverage']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'fold', 'join'))
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        protocol = guard(args.stage)
        if args.stage == 'fold':
            need(args.fold is not None, 'FOLD_REQUIRED')
            run_fold(protocol, args.fold)
        else:
            join(protocol)
