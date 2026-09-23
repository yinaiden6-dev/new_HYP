#!/usr/bin/env python3
"""Post-join accounting and fixed-head block interventions; no fitting."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
import run_rc_h593_joint_diagonal_v1 as J
from validate_rc_h593_joint_diagonal_v1 import basis


def main():
    a = J.read(J.AUTH)
    for b in a['code_sources'].values(): J.checked(b)
    v = J.read(J.OUT/'validation.json')
    assert v['status'] == 'JOINT_DIAGONAL_ALL_COUNTS_PASS' and v['authority'] == J.bind(J.AUTH)
    r = J.read(J.checked(v['result']))
    feature_rows, labels = J.C.features(a)
    features = {row['query_id']: row for row in feature_rows}
    roles = {row['query_id']: row for row in J.read(J.checked(a['join_sources']['curator']))['records']}
    result_rows = {row['query_id']: row for row in r['rows']}
    variants = ('DIAG_CE13','NO_RAW_SQUARE','NO_JOINT_SQUARE','NO_SQUARE','CBIND')
    rows, details, losses, validation = [], [], [], []
    for f in range(5):
        fv = J.read(J.checked(v['fold_validations'][f]))
        assert fv['status'] == 'JOINT_DIAGONAL_FRESH_NUMPY_PASS'
        p = J.read(J.checked(fv['payload']))
        theta = np.array([float.fromhex(x) for x in p['parameters']['DIAG_CE13']])
        held = p['predictions']
        xx = np.stack([features[pred['query_id']]['modes']['REAL']['X'].numpy() for pred in held])
        cb = np.stack([features[pred['query_id']]['modes']['CBIND']['X'].numpy() for pred in held])
        assert np.array_equal(xx[...,0],cb[...,0])
        ph = basis(xx)
        heads = {}
        for name in variants:
            t = theta.copy()
            if name == 'NO_RAW_SQUARE': t[6] = 0.
            if name == 'NO_JOINT_SQUARE': t[7:12] = 0.
            if name == 'NO_SQUARE': t[6:12] = 0.
            heads[name] = np.sum((basis(cb) if name == 'CBIND' else ph)*t[:-1],axis=-1)+t[-1]
        original = np.array([[float.fromhex(x) for x in pred['models']['DIAG_CE13']['logits_hex']] for pred in held])
        err = float(abs(heads['DIAG_CE13']-original).max())
        assert err < 2e-10
        validation.append(dict(fold=f, independent_logits=int(original.size), max_abs_error=err))
        fold_losses = {m:[] for m in ('CE_FULL','DIAG_CE13')}
        for i,pred in enumerate(held):
            q = pred['query_id']; ref = result_rows[q]; identity = roles[q]['identity']
            raw = pred['candidate_physical_rows'][pred['winner']]
            top = int(original[i].argmax())
            saved = pred['models']['DIAG_CE13']
            assert top == saved['top_index']
            selected, correct = {}, {}
            for name,zz in heads.items():
                j = int(zz[i].argmax())
                pos = pred['challenger_positions'][j] if zz[i,j]>0 else pred['winner']
                selected[name] = pred['candidate_physical_rows'][pos]
                correct[name] = labels[selected[name]] == identity
            assert selected['DIAG_CE13'] == saved['selected'] == ref['selected']['DIAG_CE13']
            assert correct['DIAG_CE13'] == ref['correct']['DIAG_CE13']
            for name in J.MODELS:
                assert (labels[ref['selected'][name]]==identity) == ref['correct'][name]
            selected['RAW'], correct['RAW'] = raw, labels[raw]==identity
            rows.append(dict(query_id=q, original_query_id=ref['original_query_id'], fold=f, component=ref['component'],
                selected=selected, correct=correct))
            if ref['target_in_C128']:
                target = [j for j,physical in enumerate(pred['candidate_physical_rows']) if labels[physical]==identity][0]
                target_action = 0 if target == pred['winner'] else pred['challenger_positions'].index(target)+1
                for name in fold_losses:
                    z = np.r_[0.,[float.fromhex(x) for x in pred['models'][name]['logits_hex']]]
                    peak = z.max(); fold_losses[name].append(float(peak+np.log(np.exp(z-peak).sum())-z[target_action]))
            if ref['correct']['DIAG_CE13'] != ref['correct']['CE_FULL'] or ref['correct']['DIAG_CE13'] != ref['correct']['GAP_BIAS2']:
                details.append(dict(query_id=q,original_query_id=ref['original_query_id'],fold=f,
                    component=ref['component'],raw_correct=ref['correct']['RAW'],
                    original40=ref['original40_ranking_blocked'],correct=ref['correct'],top_correct=ref['top_correct'],
                    top_score=float(original[i,top]),
                    selected_top_contributions=dict(linear=(xx[i,top]*theta[:6]).tolist(),
                        square=(xx[i,top]**2*theta[6:12]).tolist(),bias=float(theta[-1])),
                    fixed_head_intervention_correct=correct))
        losses.append(dict(fold=f,n=len(fold_losses['CE_FULL']),
            heldout_CE={m:float(np.mean(v)) for m,v in fold_losses.items()},train=p['train_metrics']))
    assert len(rows)==593
    stats = {}
    for m in variants:
        stats[m] = dict(correct=sum(row['correct'][m] for row in rows),
            rescue_vs_RAW=sum(row['correct'][m] and not row['correct']['RAW'] for row in rows),
            break_vs_RAW=sum(not row['correct'][m] and row['correct']['RAW'] for row in rows))
    decomposition = {}
    for base in ('CE_FULL','GAP_BIAS2'):
        gained, lost = Counter(), Counter()
        for row in r['rows']:
            raw=row['correct']['RAW']; new=row['correct']['DIAG_CE13']; old=row['correct'][base]
            top=row['top_correct']['DIAG_CE13']
            if new and not old:
                reason = 'old_RAW_break_reverted' if raw else 'correct_top_now_switched'
                if not raw and base in row['top_correct'] and not row['top_correct'][base]: reason='ranking_repair_and_switch'
                gained[reason]+=1
            elif old and not new:
                lost['new_RAW_break' if raw else 'correct_top_HELD' if top else 'ranking_loss']+=1
        decomposition[base]=dict(gains=dict(gained),losses=dict(lost))
    counts={m:sum(row['correct'][m] for row in r['rows']) for m in J.MODELS}
    assert counts=={m:s['correct'] for m,s in r['summary'].items()}
    # Recompute all paired group intervals from physical-reference correctness.
    for arm,bs in r['comparisons'].items():
        for base,saved in bs.items(): assert J.R.compare(r['rows'],base,arm)==saved
    output=dict(status='JOINT_DIAGONAL_POSTJOIN_AUDIT_PASS',result=v['result'],source=J.bind(__file__),
        natural_fits=0,validation=validation,counts=counts,losses=losses,decomposition=decomposition,
        fixed_head_interventions=stats,changed_details=details,
        intervention_scope='Post-hoc frozen-parameter sensitivity; not retrained ablations, not selected or deployed heads.',
        binding_scope='Existing cached CBIND preserves RAW and breaks candidate binding of joint evidence; not spatial ownership.',
        external_GO=False)
    J.write(J.OUT/'post_join_analysis.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('changed_details','losses')},ensure_ascii=False))
    print('losses',[(d['fold'],d['heldout_CE']) for d in losses])


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
