#!/usr/bin/env python3
"""Frozen result accounting, CE/gradient diagnostics; no optimization or selection."""
from collections import Counter
import numpy as np
import torch
import run_rc_h593_sml_extension_v1 as J
from validate_rc_h593_sml_extension_v1 import basis, logits_check
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis


def ce_gradient(phi, theta, y):
    z = np.sum(phi*theta[:-1], axis=-1)+theta[-1]
    full = np.column_stack((np.zeros(len(z)),z))
    peak = full.max(1); ex = np.exp(full-peak[:,None]); prob = ex/ex.sum(1)[:,None]
    losses = peak+np.log(ex.sum(1))-full[np.arange(len(y)),y+1]
    prob[np.arange(len(y)),y+1] -= 1.
    dz = prob[:,1:]/len(y)
    grad = np.r_[np.sum(phi*dz[...,None],axis=(0,1)),dz.sum()]
    return losses, grad


def main():
    a=J.read(J.AUTH)
    for b in a['code_sources'].values(): J.checked(b)
    v=J.read(J.OUT/'validation.json')
    assert v['status']=='SML_EXTENSION_ALL_COUNTS_PASS' and v['authority']==J.bind(J.AUTH)
    r=J.read(J.checked(v['result'])); ref={row['query_id']:row for row in r['rows']}
    roles={row['query_id']:row for row in J.read(J.checked(a['join_sources']['curator']))['records']}
    arms=('DIAG_CE13',)+J.ARMS
    losses={m:[] for m in arms}; fold_diagnostics=[]; checks=[]; changed=[]
    for f in range(5):
        fv=J.read(J.checked(v['fold_validations'][f])); p=J.read(J.checked(fv['payload']))
        assert fv['status']=='SML_EXTENSION_FRESH_NUMPY_PASS' and p['fold']==f
        train,held,keep,x,y,trainroles,labels,old=J.C.train_inputs(a,f)
        assert p['effective_train_query_ids']==[row['query_id'] for row in keep]
        features={row['query_id']:row for row in held}
        preds=p['predictions']; xx=np.stack([features[row['query_id']]['modes']['REAL']['X'].numpy() for row in preds])
        valid=[i for i,pred in enumerate(preds) if ref[pred['query_id']]['target_in_C128']]
        yy=np.array([J.C.target(features[preds[i]['query_id']],roles[preds[i]['query_id']]['identity'],labels) for i in valid])
        fm={}
        for m in arms:
            theta=np.array([float.fromhex(z) for z in p['parameters'][m]])
            ph=diagonal_basis(x.numpy()) if m=='DIAG_CE13' else basis(x.numpy(),m)
            hh=diagonal_basis(xx) if m=='DIAG_CE13' else basis(xx,m)
            lv,grad=ce_gradient(ph,theta,y.numpy()); hv,_=ce_gradient(hh[valid],theta,yy)
            assert abs(float(lv.mean())-p['train_metrics'][m]['CE'])<2e-10
            checks.append(dict(fold=f,model=m,**logits_check(xx,p['parameters'][m],preds,m)))
            fm[m]=dict(train_CE=float(lv.mean()),train_CE_gradient_linf=float(abs(grad).max()),
                       train_CE_gradient_l2=float(np.linalg.norm(grad)),train_CE_gradient=grad.tolist(),
                       heldout_CE=float(hv.mean()),train_n=len(y),heldout_present_n=len(yy))
            losses[m].extend(hv.tolist())
        fold_diagnostics.append(dict(fold=f,metrics=fm))
        for pred in preds:
            row=ref[pred['query_id']]; identity=roles[row['query_id']]['identity']
            for m in J.MODELS:
                assert (labels[row['selected'][m]]==identity)==row['correct'][m]
            for m in J.ARMS:
                if row['selected'][m]!=row['selected']['DIAG_CE13']:
                    changed.append(dict(query_id=row['query_id'],original_query_id=row['original_query_id'],fold=f,model=m,
                        raw_correct=row['correct']['RAW'],correct_old=row['correct']['DIAG_CE13'],correct_new=row['correct'][m],
                        top_correct_old=row['top_correct']['DIAG_CE13'],top_correct_new=row['top_correct'][m],
                        original40=row['original40_ranking_blocked']))
    decomposition={}
    for m in arms:
        reasons=Counter()
        for row in r['rows']:
            if row['correct'][m]: continue
            reason=('target_absent' if not row['target_in_C128'] else
                    'RAW_correct_broken' if row['correct']['RAW'] else
                    'correct_challenger_HELD' if row['top_correct'][m] else 'RAW_and_top_wrong')
            reasons[reason]+=1
        assert sum(reasons.values())==593-r['summary'][m]['correct']
        decomposition[m]=dict(reasons)
    for m,comp in r['comparisons'].items():
        for base,saved in comp.items(): assert J.R.compare(r['rows'],base,m)==saved
    assert J.R.compare(r['rows'],'SML_CUBE_CE16','SML_CROSS_CE16')==r['matched_parameter_comparison']
    result=dict(status='SML_EXTENSION_POSTJOIN_AUDIT_PASS',source=J.bind(__file__),result=v['result'],natural_fits=0,
        counts={m:sum(row['correct'][m] for row in r['rows']) for m in J.MODELS},fold_diagnostics=fold_diagnostics,
        independent_logits=checks,heldout_CE={m:float(np.mean(ls)) for m,ls in losses.items()},
        error_decomposition=decomposition,changed_decisions=changed,
        interpretation='Unregularized CE gradients are diagnostics only; AdamW finite-step recipe is not claimed to minimize CE or fixed L2.',
        external_GO=False)
    J.write(J.OUT/'post_join_analysis.json',result)
    print({k:result[k] for k in ('status','counts','heldout_CE','error_decomposition')})
    print('train gradient linf',[(d['fold'],{m:v['train_CE_gradient_linf'] for m,v in d['metrics'].items()}) for d in fold_diagnostics])


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
