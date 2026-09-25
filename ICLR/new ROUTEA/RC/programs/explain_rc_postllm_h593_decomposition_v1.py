#!/usr/bin/env python3
"""Post-hoc fixed-model pathway analysis after independent H593 verification.

No fitting/selection. Four already-frozen worlds support a conditional
common/remainder decomposition of the target versus a fixed native wrong rival.
These are model-output effects, not unique real-world causal percentages.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_rc_postllm_h593_decomposition_v1 import bind, checked, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/rc_postllm_h593_decomposition_v1'
OUT = BASE/'interpretation'


def full_scores(arm,winner):
    indices = [i for i in range(128) if i != winner]
    scores = np.zeros(128,dtype=np.float64)
    scores[indices] = arm['logits']
    return scores


def summarize(rows):
    if not rows:
        return dict(n=0)
    keys = ['common_shapley','remainder_shapley','interaction','total_margin_change',
            'common_sufficiency_gap','rematching_margin_gain']
    groups = defaultdict(list)
    for r in rows:groups[r['component']].append(r)
    stats = {}
    for key in keys:
        vals = np.asarray([r[key] for r in rows])
        gv = np.array([np.mean([r[key] for r in gr]) for gr in groups.values()])
        stats[key] = dict(query_mean=float(vals.mean()),query_median=float(np.median(vals)),
            group_mean=float(gv.mean()),positive_queries=int(np.sum(vals>0)),negative_queries=int(np.sum(vals<0)))
    return dict(n=len(rows),groups=len(groups),effects=stats)


def main():
    d = read(BASE/'result.json')
    assert d['status'] in ['ALL593_DECOMPOSITION_AUDITED','ALL593_AUDITED_CPU_GPU_ACTION_DIFFERENCES']
    assert d['n'] == len(d['rows']) == 593
    protocol = read(checked(d['protocol']))
    original = read(checked(protocol['source_result']))
    labels = {r['query_id']:r for r in original['rows']}
    manifest = read(checked(protocol['manifest']))
    inputs = {r['query_id']:r for r in manifest['rows']}
    source_rescues = {q for q,y in labels.items() if y['correct']['POST_REAL'] and not y['correct']['RAW']}
    breakdowns = {arm:Counter() for arm in d['totals']}
    margins = []
    for r in d['rows']:
        q = r['query_id']; row = inputs[q]; y = labels[q]
        w = row['winner_index']; idx = [i for i in range(128) if i != w]
        scores = {a:full_scores(v,w) for a,v in r['arms'].items()}
        if q in source_rescues:
            for arm,s in scores.items():
                chosen = r['arms'][arm]['prediction_position']
                challenger = idx[int(s[idx].argmax())]
                correct = row['candidate_identities'][chosen] == y['identity']
                best_correct = row['candidate_identities'][challenger] == y['identity']
                state = 'preserved' if correct else 'correct_challenger_below_HOLD' if best_correct else 'wrong_challenger_first'
                breakdowns[arm][state] += 1
        if not y['target_in_C128']:
            continue
        target_positions = y['target_positions']
        native = scores['NATIVE']
        target = max(target_positions,key=lambda i:native[i])
        wrong_positions = [i for i,x in enumerate(row['candidate_identities']) if x != y['identity']]
        wrong = max(wrong_positions,key=lambda i:native[i])
        z = {a:float(s[target]-s[wrong]) for a,s in scores.items()}
        c0,c1,s1,cs = (z[k] for k in ['CONSTANT','COMMON_ONLY','SPATIAL_ONLY','NATIVE'])
        common = .5*((c1-c0)+(cs-s1))
        remain = .5*((s1-c0)+(cs-c1))
        assert abs(common+remain-(cs-c0)) < 1e-10
        margins.append(dict(query_id=q,fold=r['fold'],component=y['component'],
            original_rescue=q in source_rescues,raw_correct=y['correct']['RAW'],
            native_correct=r['arms']['NATIVE']['prediction_identity']==y['identity'],
            target_position=target,fixed_native_wrong_position=wrong,margins=z,
            common_shapley=common,remainder_shapley=remain,interaction=cs-c1-s1+c0,
            total_margin_change=cs-c0,common_sufficiency_gap=cs-c1,
            rematching_margin_gain=cs-z['NATIVE_FIXED_ARGMAX']))
    assert len(margins)==570
    selected = dict(all_inpool=summarize(margins),
        original_54_rescues=summarize([r for r in margins if r['original_rescue']]),
        raw_correct=summarize([r for r in margins if r['raw_correct']]))
    result = dict(status='FROZEN_WORLD_CONTRASTS_ANALYZED',source=bind(BASE/'result.json'),
        program=bind(Path(__file__)),CPU_GPU_decision_differences=d['CPU_GPU_decision_differences'],
        original_54_rescue_failures={a:dict(x) for a,x in breakdowns.items()},
        conditional_margin_decomposition=selected,rows=margins,
        caveats=['Shapley values refer only to the chosen common/deviation intervention worlds and fixed rivals; not unique semantic causes.',
                 'No gain percentage is inferred from vector energy or unequal accuracy denominators.',
                 'All data are opened H593 development OOF; no new independent confirmation.'])
    write(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','program','caveats']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
