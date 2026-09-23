#!/usr/bin/env python3
"""Independent closed-result accounting; no optimization or head selection."""
from collections import Counter
import numpy as np
import torch
import run_rc_h593_bounded_decision_v1 as J
from validate_rc_h593_bounded_decision_v1 import basis,logits_check,values_gradient,loss_gradient
from validate_rc_h593_content_complement_v1 import basis as content_basis
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis


def phi(x,m):
    if m=='CE_FULL':return x[...,:6]
    if m=='DIAG_CE13':return diagonal_basis(x[...,:6])
    if m=='JOINT_CONTENT_CE18':return content_basis(x,m)
    return basis(x,m)


def main():
    a=J.read(J.AUTH)
    for b in a['code_sources'].values():J.checked(b)
    v=J.read(J.OUT/'validation.json');assert v['status']=='BOUNDED_DECISION_ALL_COUNTS_PASS' and v['authority']==J.bind(J.AUTH)
    r=J.read(J.checked(v['result']));ref={row['query_id']:row for row in r['rows']}
    roles={row['query_id']:row for row in J.read(J.checked(a['join_sources']['curator']))['records']}
    models=('CE_FULL','DIAG_CE13','JOINT_CONTENT_CE18')+J.ARMS
    folds=[];checks=[];changes=[];initial_checks=[]
    for f in range(5):
        fv=J.read(J.checked(v['fold_validations'][f]));p=J.read(J.checked(fv['payload']))
        assert fv['status']=='BOUNDED_DECISION_FRESH_NUMPY_PASS' and p['fold']==f
        train,held,keep,x,y,trainroles,labels,old=J.C.train_inputs(a,f)
        assert p['effective_train_query_ids']==[row['query_id'] for row in keep]
        aug=J.content_inputs(a,train+held);x=np.stack([aug[row['query_id']].numpy() for row in keep]);yn=y.numpy()
        features={row['query_id']:row for row in held};preds=p['predictions']
        xx=np.stack([aug[row['query_id']].numpy() for row in preds]);valid=[i for i,pred in enumerate(preds) if ref[pred['query_id']]['target_in_C128']]
        yy=np.array([J.C.target(features[preds[i]['query_id']],roles[preds[i]['query_id']]['identity'],labels) for i in valid])
        metrics={}
        for m in models:
            t=np.array([float.fromhex(z) for z in p['parameters'][m]]);z=np.sum(phi(x,m)*t[:-1],axis=-1)+t[-1]
            hz=np.sum(phi(xx,m)*t[:-1],axis=-1)+t[-1]
            ce,_=loss_gradient(z,yn,False);bounded,_=loss_gradient(z,yn,True)
            assert abs(ce-p['train_metrics'][m]['CE'])<2e-10 and abs(bounded-p['train_metrics'][m]['BOUNDED'])<2e-10
            hc,_=loss_gradient(hz[valid],yy,False);hb,_=loss_gradient(hz[valid],yy,True)
            metrics[m]=dict(train_CE=ce,train_bounded=bounded,heldout_CE=hc,heldout_bounded=hb,
                train_correct=p['train_metrics'][m]['actual_correct'],train_target_top=p['train_metrics'][m]['target_top_RAWwrong'])
            checks.append(dict(fold=f,model=m,**logits_check(xx,p['parameters'][m],preds,m)))
            if m in J.ARMS:
                base='DIAG_CE13' if m.startswith('DIAG_') else 'JOINT_CONTENT_CE18'
                assert p['initial_parameters'][m]==p['parameters'][base]
                initial_checks.append(dict(fold=f,model=m,source_model=base,parameters_identical=True))
                val,grad=values_gradient(t,x,yn,m)
                assert abs(val-(bounded if 'BOUND' in m else ce))<2e-10
                metrics[m]['objective_gradient_linf']=float(abs(grad).max())
        folds.append(dict(fold=f,train_n=len(yn),heldout_present_n=len(yy),metrics=metrics))
        for pred in preds:
            row=ref[pred['query_id']];identity=roles[pred['query_id']]['identity']
            for m in J.MODELS:assert (labels[row['selected'][m]]==identity)==row['correct'][m]
            for base in ('DIAG_CE13','JOINT_CONTENT_CE18'):
                new='CONTENT_CONT_CE18'
                if row['correct'][new]!=row['correct'][base]:
                    why=('RAW_break_reverted' if row['correct']['RAW'] else 'ranking_repair_and_SWITCH' if not row['top_correct'][base] else 'correct_top_now_SWITCH') if row['correct'][new] else ('new_RAW_break' if row['correct']['RAW'] else 'ranking_loss' if not row['top_correct'][new] else 'correct_top_HELD')
                    changes.append(dict(query_id=row['query_id'],original_query_id=row['original_query_id'],fold=f,component=row['component'],baseline=base,
                        new_correct=row['correct'][new],raw_correct=row['correct']['RAW'],reason=why,original40=row['original40_ranking_blocked']))
    errors={}
    for m in models:
        c=Counter('target_absent' if not row['target_in_C128'] else 'RAW_correct_broken' if row['correct']['RAW'] else
                  'correct_challenger_HELD' if row['top_correct'][m] else 'RAW_and_top_wrong' for row in r['rows'] if not row['correct'][m])
        assert sum(c.values())==593-r['summary'][m]['correct'];errors[m]=dict(c)
    for m,cs in r['comparisons'].items():
        for b,saved in cs.items():assert J.R.compare(r['rows'],b,m)==saved
    for saved in [r['matched_parameter_comparison'],*r['new_arm_comparisons'].values()]:
        assert J.R.compare(r['rows'],saved['baseline'],saved['new'])==saved
    aggregate={m:{key:sum(d['heldout_present_n']*d['metrics'][m][key] for d in folds)/570 for key in ('heldout_CE','heldout_bounded')} for m in models}
    output=dict(status='BOUNDED_DECISION_POSTJOIN_AUDIT_PASS',source=J.bind(__file__),result=v['result'],natural_fits=0,
        counts={m:sum(row['correct'][m] for row in r['rows']) for m in J.MODELS},fold_diagnostics=folds,
        aggregate=aggregate,independent_logits=checks,initialization_checks=initial_checks,error_decomposition=errors,
        continued_content_changes=changes,external_GO=False,
        scope='Opened development result. Bounded primary failed; CE continuation control is secondary positive evidence. No global convergence claim.')
    J.write(J.OUT/'post_join_analysis.json',output)
    print({k:output[k] for k in ('status','counts','aggregate','error_decomposition')})


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
