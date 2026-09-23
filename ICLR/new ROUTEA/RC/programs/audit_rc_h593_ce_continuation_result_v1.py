#!/usr/bin/env python3
"""Read-only independent accounting of the closed CE continuation experiment."""
from collections import Counter
import numpy as np
import torch
import run_rc_h593_ce_continuation_v1 as J
from validate_rc_h593_ce_continuation_v1 import basis,logits_check,optimizer_check
from validate_rc_h593_bounded_decision_v1 import loss_gradient
from validate_rc_h593_content_complement_v1 import basis as content_basis
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis


def phi(x,m):
    if m=='CE_FULL':return x[...,:6]
    if m.startswith('DIAG_'):return diagonal_basis(x[...,:6])
    return content_basis(x,'JOINT_CONTENT_CE18')


def main():
    a=J.read(J.AUTH)
    for b in a['code_sources'].values():J.checked(b)
    v=J.read(J.OUT/'validation.json')
    assert v['status']=='CE_CONTINUATION_ALL_COUNTS_PASS' and v['authority']==J.bind(J.AUTH)
    r=J.read(J.checked(v['result']));ref={row['query_id']:row for row in r['rows']}
    roles={row['query_id']:row for row in J.read(J.checked(a['join_sources']['curator']))['records']}
    models=('CE_FULL','DIAG_CE13','JOINT_CONTENT_CE18','DIAG_CONT_CE13','CONTENT_CONT_CE18')+J.ARMS
    folds=[];checks=[];states=[];changes=[]
    for f in range(5):
        fv=J.read(J.checked(v['fold_validations'][f]));p=J.read(J.checked(fv['payload']))
        assert fv['status']=='CE_CONTINUATION_FRESH_NUMPY_PASS' and p['fold']==f
        train,held,keep,x,y,trainroles,labels,old=J.C.train_inputs(a,f)
        assert p['effective_train_query_ids']==[row['query_id'] for row in keep]
        aug=J.content_inputs(a,train+held);x=np.stack([aug[row['query_id']].numpy() for row in keep]);yn=y.numpy()
        features={row['query_id']:row for row in held};preds=p['predictions']
        xx=np.stack([aug[row['query_id']].numpy() for row in preds])
        valid=[i for i,pred in enumerate(preds) if ref[pred['query_id']]['target_in_C128']]
        yy=np.array([J.C.target(features[preds[i]['query_id']],roles[preds[i]['query_id']]['identity'],labels) for i in valid])
        metrics={}
        for m in models:
            t=np.array([float.fromhex(z) for z in p['parameters'][m]]);ph=phi(x,m)
            z=np.sum(ph*t[:-1],axis=-1)+t[-1];hz=np.sum(phi(xx,m)*t[:-1],axis=-1)+t[-1]
            ce,dz=loss_gradient(z,yn,False);bounded,_=loss_gradient(z,yn,True)
            assert abs(ce-p['train_metrics'][m]['CE'])<2e-10
            hc,_=loss_gradient(hz[valid],yy,False);hb,_=loss_gradient(hz[valid],yy,True)
            grad=np.r_[np.sum(ph*dz[...,None],axis=(0,1)),dz.sum()]
            metrics[m]=dict(train_CE=ce,train_bounded=bounded,heldout_CE=hc,heldout_bounded=hb,
                train_correct=p['train_metrics'][m]['actual_correct'],theta_linf=float(abs(t).max()),
                gradient_linf=float(abs(grad).max()),gradient_l1=float(abs(grad).sum()),
                gradient_dot_theta=float(grad@t),all_inside_box64=bool((abs(t)<=64).all()))
            checks.append(dict(fold=f,model=m,**logits_check(xx,p['parameters'][m],preds,m)))
            if m in J.ARMS:
                initial=p['parameters']['DIAG_CE13' if m.startswith('DIAG_') else 'JOINT_CONTENT_CE18']
                optimizer_check(p['optimizer_states'][m],p['parameters'][m],initial)
                states.append(dict(fold=f,model=m,midpoint_bitexact=True,midpoint_step=2000,final_step=4000))
        folds.append(dict(fold=f,train_n=len(yn),heldout_present_n=len(yy),metrics=metrics))
        for pred in preds:
            row=ref[pred['query_id']];identity=roles[pred['query_id']]['identity']
            for m in J.MODELS:assert (labels[row['selected'][m]]==identity)==row['correct'][m]
            for new,base in [('CONTENT_CONTINUOUS_CE18','JOINT_CONTENT_CE18'),('CONTENT_CONTINUOUS_CE18','CONTENT_CONT_CE18')]:
                if row['correct'][new]!=row['correct'][base]:
                    changes.append(dict(query_id=row['query_id'],original_query_id=row['original_query_id'],fold=f,component=row['component'],baseline=base,
                        model=new,new_correct=row['correct'][new],raw_correct=row['correct']['RAW'],old_top_correct=row['top_correct'][base],new_top_correct=row['top_correct'][new]))
    errors={}
    for m in models:
        c=Counter('target_absent' if not row['target_in_C128'] else 'RAW_correct_broken' if row['correct']['RAW'] else
                  'correct_challenger_HELD' if row['top_correct'][m] else 'RAW_and_top_wrong' for row in r['rows'] if not row['correct'][m])
        assert sum(c.values())==593-r['summary'][m]['correct'];errors[m]=dict(c)
    for m,cs in r['comparisons'].items():
        for b,saved in cs.items():assert J.R.compare(r['rows'],b,m)==saved
    aggregate={m:{key:sum(d['heldout_present_n']*d['metrics'][m][key] for d in folds)/570 for key in ('heldout_CE','heldout_bounded')} for m in models}
    output=dict(status='CE_CONTINUATION_POSTJOIN_AUDIT_PASS',source=J.bind(__file__),result=v['result'],natural_fits=0,
        counts={m:sum(row['correct'][m] for row in r['rows']) for m in J.MODELS},fold_diagnostics=folds,
        aggregate=aggregate,independent_logits=checks,optimizer_state_checks=states,error_decomposition=errors,
        paired_changes=changes,external_GO=False,
        scope='Opened development. Continuous4000 content494 versus reset4000 content496; neither data loss nor AdamW state establishes a global CE optimum.')
    J.write(J.OUT/'post_join_analysis.json',output)
    print({k:output[k] for k in ('status','counts','aggregate','error_decomposition')})
    for d in folds:print(dict(fold=d['fold'],metrics={m:d['metrics'][m] for m in J.ARMS}))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
