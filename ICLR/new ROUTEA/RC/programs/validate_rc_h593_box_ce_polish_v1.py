#!/usr/bin/env python3
"""Independent Torch CE/gradient, certificate replay, and action checks."""
from fractions import Fraction
import numpy as np
import torch
from validate_rc_h593_content_complement_v1 import basis as content_basis
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis
import rc_h593_box_ce_core_v1 as K

SIZES={'CONTENT_BOX_CE_POLISH18':18,'DIAG_BOX_CE_POLISH13':13}


def design(x,m):
    ph=content_basis(x,'JOINT_CONTENT_CE18') if SIZES[m]==18 else diagonal_basis(x[...,:6])
    rows=np.concatenate((ph,np.ones(ph.shape[:2]+(1,),dtype=np.float64)),axis=-1)
    return np.concatenate((np.zeros((len(x),1,rows.shape[-1]),dtype=np.float64),rows),axis=1)


def verify(theta,d,y,record):
    b=np.zeros(d.shape[:2]);val,grad,_=K.value_gradient(theta,d,b,y)
    t=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    z=torch.from_numpy(d)@t;loss=torch.nn.functional.cross_entropy(z,torch.from_numpy(y))
    g=torch.autograd.grad(loss,t)[0].numpy()
    assert abs(float(loss.detach())-val)<2e-10 and abs(g-grad).max()<2e-10
    c=record['certificate'];replayed=K.exact_certificate(d,b,y,theta,c['probability_counts'])
    assert replayed==c and abs(record['final_CE']-val)<2e-12
    assert float(Fraction(c['objective_lower']))-2e-12<=val<=float(Fraction(c['objective_upper']))+2e-12
    old=np.array([float.fromhex(v) for v in record['initial_parameters']])
    oldval,_,_=K.value_gradient(old,d,b,y)
    assert abs(oldval-record['initial_CE'])<2e-12 and val<=oldval+1e-10
    return dict(loss_error=abs(float(loss.detach())-val),gradient_error=float(abs(g-grad).max()),
        exact_certificate_replay=True,certified_box_gap_le_1e_6=c['certified_box_gap_le_1e_6'])


def logits_check(x,parameters,predictions,m):
    d=design(x,m);theta=np.array([float.fromhex(v) for v in parameters])
    z=np.sum(d[:,1:]*theta,axis=-1)
    wanted=np.array([[float.fromhex(v) for v in p['models'][m]['logits_hex']] for p in predictions])
    err=float(abs(z-wanted).max());assert err<2e-10
    for p,s in zip(predictions,z):
        top=int(s.argmax());pos=p['challenger_positions'][top] if s[top]>0 else p['winner']
        assert top==p['models'][m]['top_index'] and p['candidate_physical_rows'][pos]==p['models'][m]['selected']
    return dict(logit_count=z.size,max_abs_error=err,decisions=len(z))



def self_test():
    import time
    import rc_h593_box_ce_polish_core_v1 as N
    rng=np.random.default_rng(20260921);checks=0;started=time.monotonic()
    x=rng.normal(size=(4,127,11))*.1;y=np.array([0,1,64,127],dtype=np.int64)
    for m,n in SIZES.items():
        d=design(x,m);old,_=K.fit(d,y,rng.normal(size=n)*.1)
        t,r=N.fit(d,y,old);verify(t,d,y,r)
        assert r['certificate']['certified_box_gap_le_1e_6'];checks+=4
        tt=torch.tensor(t,requires_grad=True);dd=torch.from_numpy(d)
        h=torch.autograd.functional.hessian(lambda v:torch.nn.functional.cross_entropy(dd@v,torch.from_numpy(y)),tt).numpy()
        _,_,p=K.value_gradient(t,d,np.zeros(d.shape[:2]),y)
        assert abs(N.hessian(d,p)-h).max()<2e-10;checks+=1
        for bias,selected in ((0.,0),(1.,1),(-1.,0)):
            pred=[dict(winner=0,candidate_physical_rows=list(range(128)),challenger_positions=list(range(1,128)),models={m:dict(logits_hex=[bias.hex()]*127,top_index=0,selected=selected)})]
            logits_check(np.zeros((1,127,11)),[0.0.hex()]*(n-1)+[bias.hex()],pred,m);checks+=1
        rng.uniform(-64,64,size=(4,n))
    for t,g,want in ((-64.,1.,0.),(64.,-1.,0.),(-64.,-1.,-1.),(64.,1.,1.),(0.,2.,2.)):
        assert N.projected_gradient(np.array([t]),np.array([g]))[0]==want;checks+=1
    return dict(status='BOX_CE_POLISH_SYNTHETIC_PASS',checks=checks,synthetic_base_fits=2,synthetic_refinements=2,natural_fits=0,seconds=time.monotonic()-started)
