#!/usr/bin/env python3
"""Independent CE values/gradients and continuous optimizer boundary checks."""
import numpy as np
from validate_rc_h593_bounded_decision_v1 import loss_gradient,logits_check as parent_logits_check,content_check
from validate_rc_h593_content_complement_v1 import basis as content_basis
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis

SIZES=dict(CONTENT_CONTINUOUS_CE18=18,DIAG_CONTINUOUS_CE13=13)


def basis(x,arm):
    assert arm in SIZES
    return diagonal_basis(x[...,:6]) if SIZES[arm]==13 else content_basis(x,'JOINT_CONTENT_CE18')


def values_gradient(theta,x,y,arm):
    t=np.asarray(theta,dtype=np.float64);ph=basis(x,arm);assert t.shape==(SIZES[arm],)
    z=np.sum(ph*t[:-1],axis=-1)+t[-1];value,dz=loss_gradient(z,y,False)
    return value,np.r_[np.sum(ph*dz[...,None],axis=(0,1)),dz.sum()]


def logits_check(x,parameters,predictions,model):
    if model not in SIZES:return parent_logits_check(x,parameters,predictions,model)
    t=np.array([float.fromhex(v) for v in parameters]);z=np.sum(basis(x,model)*t[:-1],axis=-1)+t[-1]
    wanted=np.array([[float.fromhex(v) for v in r['models'][model]['logits_hex']] for r in predictions])
    error=float(abs(z-wanted).max());assert error<2e-10
    for r,score,want in zip(predictions,z,wanted):
        top=int(score.argmax());assert top==int(want.argmax()) and bool(score[top]>0)==bool(want[top]>0)
        pos=r['challenger_positions'][top] if score[top]>0 else r['winner']
        assert r['candidate_physical_rows'][pos]==r['models'][model]['selected']
    return dict(logit_count=int(z.size),max_abs_error=error,decisions=len(predictions))


def optimizer_check(states,parameters,expected_midpoint):
    assert states['midpoint']['step']==2000 and states['final']['step']==4000
    assert states['midpoint']['parameters']==expected_midpoint and states['final']['parameters']==parameters
    for state in states.values():
        for name in ('parameters','exp_avg','exp_avg_sq'):
            values=np.array([float.fromhex(v) for v in state[name]])
            assert values.shape==(len(parameters),) and np.isfinite(values).all()
            if name=='exp_avg_sq':assert (values>=0).all()


def self_test():
    import torch
    from rc_aslo_xf.h593_ce_continuation_v1 import expand,training_loss,fit
    from rc_aslo_xf.h593_joint_diagonal_v1 import fit as diag_fit
    from rc_aslo_xf.h593_content_complement_v1 import fit as content_fit
    rng=np.random.default_rng(20260921);checks=0
    for scale in (0.,.1,1.,3.):
        x=rng.normal(size=(4,127,11))*scale;y=np.array([-1,0,12,126]);xx=torch.from_numpy(x);yy=torch.from_numpy(y)
        for arm,n in SIZES.items():
            ph=expand(xx,arm);assert np.array_equal(ph.numpy(),basis(x,arm))
            theta=torch.tensor(rng.normal(size=n),dtype=torch.float64,requires_grad=True)
            loss=training_loss(theta,ph,yy,arm);grad=torch.autograd.grad(loss,theta)[0].numpy()
            value,g=values_gradient(theta.detach().numpy(),x,y,arm)
            assert abs(float(loss.detach())-value)<2e-10 and abs(grad-g).max()<2e-10;checks+=2
    x=torch.zeros((2,127,11),dtype=torch.float64);x[1,0,0]=-1.;x[1,0,1]=1.;y=torch.tensor([-1,0])
    for arm,n in SIZES.items():
        midpoint=diag_fit(x[...,:6].contiguous(),y,'DIAG_CE13') if n==13 else content_fit(x,y,'JOINT_CONTENT_CE18')
        theta,states=fit(x,y,arm,midpoint);z=expand(x,arm)@theta[:-1]+theta[-1]
        assert torch.equal(torch.where(z.max(1).values>0,z.argmax(1),-1),y)
        optimizer_check(states,[float(v).hex() for v in theta],[float(v).hex() for v in midpoint]);checks+=2
    p=[dict(winner=0,candidate_physical_rows=list(range(128)),challenger_positions=list(range(1,128)),models={})]
    for arm,n in SIZES.items():
        for bias,selected in ((0.,0),(1.,1),(-1.,0)):
            p[0]['models'][arm]=dict(logits_hex=[bias.hex()]*127,selected=selected)
            logits_check(np.zeros((1,127,11)),[0.0.hex()]*(n-1)+[bias.hex()],p,arm);checks+=1
    return dict(status='CE_CONTINUATION_SYNTHETIC_PASS',checks=checks,natural_fits=0,
        synthetic_2000_step_fits=2,synthetic_4000_step_fits=2)
