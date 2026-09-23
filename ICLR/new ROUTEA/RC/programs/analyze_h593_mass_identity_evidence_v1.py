#!/usr/bin/env python3
"""Read-only post-hoc interpretation of sealed H593 operator predictions; no fit."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_h593_quality_operator_eval_v1'
OUT = ROOT / 'reports/h593_mass_identity_evidence_20260922_v1'
MODEL = 'COST1_REFIT_M1Q0R0'
NATIVE = 'COST1_FROZEN_NATIVE'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(source):
    assert binding(source['path']) == source, source['path']
    return Path(source['path'])


def write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, str(path)
    else:
        path.write_text(text)


def action(logits, query):
    index = max(range(len(logits)), key=lambda i: logits[i])
    position = query['challenger_positions'][index] if logits[index] > 0 else query['winner']
    return query['candidate_physical_rows'][position]


def main():
    validation = read(SOURCE / 'validation.json')
    assert validation['status'] == 'QUALITY_OPERATOR_EVAL_ALL_COUNTS_PASS'
    result = read(checked(validation['result']))
    authority = read(checked(validation['authority']))
    gallery = read(checked(authority['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    folds, predictions = {}, {}
    for source in validation['fold_validations']:
        val = read(checked(source)); payload = read(checked(val['payload']))
        assert val['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS'
        assert val['authority'] == validation['authority'] == payload['authority']
        fold = payload['fold']; folds[fold] = payload
        for row in payload['predictions']:
            assert row['query_id'] not in predictions
            predictions[row['query_id']] = row
    rows, max_logit_error = [], 0.
    for old in result['rows']:
        index = old['execution_ordinal']
        val = read(checked(result['operator_validations'][index]))
        query = read(checked(val['payload']))
        assert val['status'] == 'QUALITY_OPERATOR_QUERY_PASS'
        assert query['query_id'] == old['query_id']
        assert query['authority'] == val['authority'] == authority['operator_authority']
        axis = query['candidate_physical_rows']; winner = query['winner']
        assert len(axis) == len(set(axis)) == 128
        targets = [i for i, physical in enumerate(axis) if labels[physical] == old['identity']]
        assert len(targets) <= 1 and bool(targets) == old['target_in_C128']
        scores = {
            'M': [x['visibility_mass'] for x in query['modes']['M1Q0R0']['scores']],
            'C': [x['real_score'] for x in query['modes']['M0Q0R0']['scores']],
            'MC': [x['real_score'] for x in query['modes']['M1Q0R0']['scores']],
        }
        theta = [float.fromhex(x) for x in folds[old['fold']]['parameters'][MODEL]['theta_hex']]
        x = query['modes']['M1Q0R0']['X']
        assert all(v[4] == v[5] == 0 for v in x)
        logits = [math.fsum(a*b for a, b in zip(theta[:-1], v)) + theta[-1] for v in x]
        expected = predictions[old['query_id']]['models'][MODEL]
        errors = [abs(a - float.fromhex(b)) for a, b in zip(logits, expected['logits_hex'])]
        max_logit_error = max(max_logit_error, max(errors))
        assert max(errors) < 2e-12
        assert action(logits, query) == expected['selected'] == old['selected'][MODEL]
        # Same frozen simplified head, all quality removed in the scorer, no refit.
        free_x = query['modes']['M0Q0R0']['X']
        no_quality = [math.fsum(a*b for a, b in zip(theta[:-1], v)) + theta[-1] for v in free_x]
        no_quality_selected = action(no_quality, query)
        row = dict(query_id=old['query_id'], original_query_id=old['original_query_id'],
                   execution_ordinal=index, fold=old['fold'], component=old['component'],
                   identity=old['identity'], raw_correct=old['correct']['RAW'],
                   simple_correct=old['correct'][MODEL], native_correct=old['correct'][NATIVE],
                   recall_present=bool(targets), source=val['payload'],
                   raw_physical_row=axis[winner], simple_physical_row=expected['selected'],
                   no_quality_selected=no_quality_selected,
                   no_quality_correct=labels[no_quality_selected] == old['identity'],
                   no_quality_logits=no_quality,
                   no_quality_logit_axis=[axis[i] for i in query['challenger_positions']],
                   theta=theta, scores={})
        for name, values in scores.items():
            chosen = max(range(128), key=lambda i: (values[i], -axis[i]))
            item = dict(selected=axis[chosen], correct=chosen in targets)
            if targets:
                t = targets[0]; wrong = [i for i in range(128) if i != t]
                item.update(target_value=values[t], raw_value=values[winner],
                            strongest_wrong_value=max(values[i] for i in wrong),
                            target_gt_raw=values[t] > values[winner],
                            target_rank=1 + sum(values[i] > values[t] for i in wrong),
                            wrong_candidates_beaten=sum(values[t] > values[i] for i in wrong))
            row['scores'][name] = item
        if targets and targets[0] != winner:
            j = query['challenger_positions'].index(targets[0])
            terms = [a*b for a, b in zip(theta[:-1], x[j])] + [theta[-1]]
            row['target_vs_raw_head'] = dict(
                feature_names=['RAW', 'MC', 'M', 'C', 'Q', 'R', 'bias'],
                contributions=terms, logit=logits[j],
                no_quality_logit=no_quality[j],
                quality_related_logit_change=logits[j]-no_quality[j])
        rows.append(row)
    assert len(rows) == len(predictions) == 593
    assert sum(r['recall_present'] for r in rows) == 570
    groups = {
        'all593': rows,
        'raw_correct': [r for r in rows if r['raw_correct']],
        'raw_wrong_recall_present': [r for r in rows if not r['raw_correct'] and r['recall_present']],
        'simple_rescues': [r for r in rows if r['simple_correct'] and not r['raw_correct']],
        'simple_breaks': [r for r in rows if not r['simple_correct'] and r['raw_correct']],
        'native_rescues': [r for r in rows if r['native_correct'] and not r['raw_correct']],
    }
    summary = {}
    for name, subset in groups.items():
        summary[name] = dict(n=len(subset),
            direct_top1={s: sum(r['scores'][s]['correct'] for r in subset) for s in scores},
            target_gt_raw_among_raw_errors={s: sum(r['scores'][s].get('target_gt_raw', False)
                for r in subset if not r['raw_correct']) for s in scores},
            no_quality_correct=sum(r['no_quality_correct'] for r in subset))
    direct_vs_raw = {}
    for name in scores:
        direct_vs_raw[name] = dict(correct=sum(r['scores'][name]['correct'] for r in rows),
            rescue=sum(r['scores'][name]['correct'] and not r['raw_correct'] for r in rows),
            loss=sum(not r['scores'][name]['correct'] and r['raw_correct'] for r in rows))
    rescues = groups['simple_rescues']
    rescue_readout = dict(
        n=len(rescues), target_gt_raw_M=sum(r['scores']['M']['target_gt_raw'] for r in rescues),
        quality_change_positive=sum(r['target_vs_raw_head']['quality_related_logit_change'] > 0 for r in rescues),
        target_logit_without_quality_positive=sum(r['target_vs_raw_head']['no_quality_logit'] > 0 for r in rescues),
        mean_contributions=[math.fsum(r['target_vs_raw_head']['contributions'][i] for r in rescues)/len(rescues) for i in range(7)],
        by_fold={str(f): dict(n=sum(r['fold']==f for r in rescues),
            target_gt_raw_M=sum(r['fold']==f and r['scores']['M']['target_gt_raw'] for r in rescues)) for f in range(5)})
    output = dict(status='SEALED_H593_MASS_IDENTITY_POSTHOC_COMPLETE',
        source_validation=binding(SOURCE/'validation.json'), source_result=validation['result'],
        gallery=authority['public_sources']['gallery'], code=binding(__file__),
        scope='Opened H593 OOF5, original natural RAW C128, descriptive post-hoc only; no fitting, no new forward, no model selection',
        direct_ranking='Diagnostic only: highest M, C or MC without action; ties by physical row',
        counterfactual='Frozen simplified COST1 head applied to sealed M0Q0R0 features; not a refitted model or pixel intervention',
        summary=summary, direct_vs_raw=direct_vs_raw, rescue_readout=rescue_readout,
        max_independent_logit_error=max_logit_error, rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'analysis.json', output)
    write(OUT/'validation.json', dict(status='MASS_IDENTITY_SOURCE_AND_LOGIT_CHECKS_PASS',
        analysis=binding(OUT/'analysis.json'), total=593, candidate_recall=570,
        checked_logit_count=593*127, max_error=max_logit_error, label_scope='Previously joined opened H593 only'))
    print(json.dumps({k:output[k] for k in ['status','summary','direct_vs_raw','rescue_readout','max_independent_logit_error']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
