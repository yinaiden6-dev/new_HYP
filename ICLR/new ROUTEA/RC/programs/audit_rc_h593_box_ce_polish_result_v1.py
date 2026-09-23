#!/usr/bin/env python3
"""Independent post-join accounting and loss analysis; no fitting."""
from collections import Counter
from fractions import Fraction
import numpy as np
import torch
import run_rc_h593_box_ce_polish_v1 as J
from validate_rc_h593_bounded_decision_v1 import loss_gradient


def main():
    a=J.read(J.AUTH)
    for b in a['code_sources'].values():J.checked(b)
    v=J.read(J.OUT/'validation.json');assert v['status']=='BOX_CE_POLISH_ALL_COUNTS_PASS' and v['authority']==J.bind(J.AUTH)
    result=J.read(J.checked(v['result']));rows=result['rows'];ref={r['query_id']:r for r in rows}
    labels={r['physical_row']:r['identity'] for r in J.read(J.checked(a['public_sources']['gallery']))['records']}
    roles={r['query_id']:r for r in J.read(J.checked(a['join_sources']['curator']))['records']}
    for r in rows:
        for m in J.MODELS:assert (labels[r['selected'][m]]==roles[r['query_id']]['identity'])==r['correct'][m]
    folds=[];checks=[];changed=[]
    for f in range(5):
        fv=J.read(J.checked(v['fold_validations'][f]));p=J.read(J.checked(fv['payload']))
        assert fv['status']=='BOX_CE_POLISH_EXACT_REPLAY_PASS' and p['authority']==J.bind(J.AUTH)
        train,held,keep,x,y,xx,old=J.inputs(a,f)
        kept=[i for i,r in enumerate(held) if ref[r['query_id']]['target_in_C128']]
        yy=np.array([J.C.target(held[i],roles[held[i]['query_id']]['identity'],labels) for i in kept])
        metrics={}
        for m in J.ARMS:
            source=J.START[m];d=J.design(x,m);hd=J.design(xx,m)
            for name in (source,m):
                t=np.array([float.fromhex(v) for v in p['parameters'][name]])
                z=np.sum(d[:,1:]*t,axis=-1);hz=np.sum(hd[:,1:]*t,axis=-1)
                ce,_=loss_gradient(z,y-1,False);hc,_=loss_gradient(hz[kept],yy,False)
                metrics[name]=dict(train_CE=ce,heldout_CE=hc,train_correct=int((np.where(z.max(1)>0,z.argmax(1),-1)==y-1).sum()))
            s=p['solver_records'][m];c=s['certificate']
            assert abs(metrics[m]['train_CE']-s['final_CE'])<2e-10
            assert abs(metrics[source]['train_CE']-s['initial_CE'])<2e-10
            assert c['certified_box_gap_le_1e_6']==(Fraction(c['gap_upper'])<=Fraction(1,1000000))
            checks.append(dict(fold=f,model=m,**J.logits_check(xx,p['parameters'][m],p['predictions'],m)))
        folds.append(dict(fold=f,train_present=len(keep),held_present=len(kept),metrics=metrics))
    for m,cs in result['comparisons'].items():
        for b,saved in cs.items():assert J.R.compare(rows,b,m)==saved
    for m,source in J.START.items():
        for r in rows:
            if r['selected'][m]!=r['selected'][source]:changed.append(dict(query_id=r['query_id'],original_query_id=r['original_query_id'],model=m,source=source,old_correct=r['correct'][source],new_correct=r['correct'][m]))
    errors={m:dict(Counter('target_absent' if not r['target_in_C128'] else 'RAW_correct_broken' if r['correct']['RAW'] else 'correct_challenger_HELD' if r['top_correct'][m] else 'RAW_and_top_wrong' for r in rows if not r['correct'][m])) for m in J.ARMS}
    aggregate={m:sum(d['held_present']*d['metrics'][m]['heldout_CE'] for d in folds)/570 for m in (*J.START.values(),*J.ARMS)}
    output=dict(status='BOX_CE_POLISH_POSTJOIN_AUDIT_PASS',source=J.bind(__file__),result=v['result'],natural_fits=0,
        counts={m:sum(r['correct'][m] for r in rows) for m in J.MODELS},folds=folds,heldout_CE=aggregate,
        independent_logits=checks,error_decomposition=errors,changed=changed,external_GO=False,
        all_box_optima_certified=result['all_box_optima_certified'],optimization='Same frozen box/objective; certificate and accuracy gates reported separately.')
    J.write(J.OUT/'post_join_analysis.json',output)
    print({k:output[k] for k in ('status','counts','heldout_CE','error_decomposition','changed')})


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
