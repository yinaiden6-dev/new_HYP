#!/usr/bin/env python3
"""Independent full HOLD-plus-challenger bounded objective and tie gradients."""
import numpy as np
from validate_rc_h593_content_complement_v1 import basis as content_basis
from validate_rc_h593_content_complement_v1 import logits_check as content_logits_check, content_check
from validate_rc_h593_joint_diagonal_v1 import basis as diagonal_basis

SIZES=dict(DIAG_BOUND13=13,DIAG_CONT_CE13=13,CONTENT_BOUND18=18,CONTENT_CONT_CE18=18)


def basis(x,arm):
    assert arm in SIZES
    return diagonal_basis(x[...,:6]) if SIZES[arm]==13 else content_basis(x,'JOINT_CONTENT_CE18')


def loss_gradient(z,y,bounded):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    assert z.shape==(len(y),127) and len(y)>0 and np.isfinite(z).all() and np.isin(y,np.arange(-1,127)).all()
    full=np.column_stack((np.zeros(len(y)),z));dz=np.zeros_like(full);losses=[]
    for i,t in enumerate(y+1):
        if bounded:
            mask=np.arange(128)!=t;peak=full[i,mask].max();ties=mask&(full[i]==peak)
            v=peak-full[i,t];sig=float(np.exp(-np.logaddexp(0.,-v)))
            # Stable derivative independent of Torch sigmoid backward.
            derivative=float(np.exp(-np.logaddexp(0.,v)-np.logaddexp(0.,-v)))
            dz[i,t]=-derivative;dz[i,ties]=derivative/ties.sum();losses.append(sig)
        else:
            peak=full[i].max();ex=np.exp(full[i]-peak);probs=ex/ex.sum()
            losses.append(float(peak+np.log(ex.sum())-full[i,t]));dz[i]=probs;dz[i,t]-=1.
    return float(np.mean(losses)),dz[:,1:]/len(y)


def values_gradient(theta,x,y,arm):
    t=np.asarray(theta,dtype=np.float64);ph=basis(x,arm);assert t.shape==(SIZES[arm],)
    z=np.sum(ph*t[:-1],axis=-1)+t[-1]
    value,dz=loss_gradient(z,y,'BOUND' in arm)
    return value,np.r_[np.sum(ph*dz[...,None],axis=(0,1)),dz.sum()]


def logits_check(x,parameters,predictions,model):
    if model not in SIZES: return content_logits_check(x,parameters,predictions,model)
    t=np.array([float.fromhex(v) for v in parameters]);z=np.sum(basis(x,model)*t[:-1],axis=-1)+t[-1]
    wanted=np.array([[float.fromhex(v) for v in r['models'][model]['logits_hex']] for r in predictions])
    error=float(abs(z-wanted).max());assert error<2e-10
    for r,score,want in zip(predictions,z,wanted):
        top=int(score.argmax());assert top==int(want.argmax()) and bool(score[top]>0)==bool(want[top]>0)
        pos=r['challenger_positions'][top] if score[top]>0 else r['winner']
        assert r['candidate_physical_rows'][pos]==r['models'][model]['selected']
    return dict(logit_count=int(z.size),max_abs_error=error,decisions=len(predictions))


def self_test():
    import torch
    from rc_aslo_xf.h593_bounded_decision_v1 import expand,training_loss,fit,bounded_loss
    rng=np.random.default_rng(20260921);checks=0
    for scale in (0.,.1,1.,3.):
        x=rng.normal(size=(4,127,11))*scale;y=np.array([-1,0,12,126]);xx=torch.from_numpy(x);yy=torch.from_numpy(y)
        for arm,n in SIZES.items():
            phi=expand(xx,arm);assert np.array_equal(phi.numpy(),basis(x,arm))
            theta=torch.tensor(rng.normal(size=n),dtype=torch.float64,requires_grad=True)
            loss=training_loss(theta,phi,yy,arm);grad=torch.autograd.grad(loss,theta)[0].numpy()
            value,g=values_gradient(theta.detach().numpy(),x,y,arm)
            assert abs(float(loss.detach())-value)<2e-10 and abs(grad-g).max()<2e-10;checks+=2
    # All 128 actions tied: HOLD participates as a wrong rival for target challengers.
    z=torch.zeros((3,127),dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0,126])
    loss=bounded_loss(z,y);grad=torch.autograd.grad(loss,z)[0].numpy();value,g=loss_gradient(z.detach().numpy(),y.numpy(),True)
    assert value==float(loss.detach())==.5 and np.max(abs(grad-g))<1e-16
    wanted=np.full((3,127),.25/(127*3));wanted[1,0]=wanted[2,126]=-.25/3
    np.testing.assert_allclose(grad,wanted,rtol=1e-14,atol=1e-17);checks+=2
    # Large confident errors/correct decisions remain finite, and every loss is bounded.
    z=torch.full((3,127),1000.,dtype=torch.float64);z[1]=-1000.;z[2,126]=2000.;z.requires_grad_(True)
    loss=bounded_loss(z,y);grad=torch.autograd.grad(loss,z)[0].numpy();value,g=loss_gradient(z.detach().numpy(),y.numpy(),True)
    assert np.isfinite(grad).all() and 0<=value<=1 and abs(float(loss.detach())-value)<1e-14 and abs(grad-g).max()<1e-14;checks+=1
    x=torch.zeros((2,127,11),dtype=torch.float64);x[1,0,0]=-1.;x[1,0,1]=1.;y=torch.tensor([-1,0])
    for arm,n in SIZES.items():
        initial=torch.zeros(n,dtype=torch.float64);copy=initial.clone();t=fit(x,y,arm,initial);z=expand(x,arm)@t[:-1]+t[-1]
        assert torch.equal(initial,copy) and torch.equal(torch.where(z.max(1).values>0,z.argmax(1),-1),y);checks+=1
    p=[dict(winner=0,candidate_physical_rows=list(range(128)),challenger_positions=list(range(1,128)),models={})]
    for arm,n in SIZES.items():
        for bias,selected in ((0.,0),(1.,1),(-1.,0)):
            p[0]['models'][arm]=dict(logits_hex=[bias.hex()]*127,selected=selected)
            logits_check(np.zeros((1,127,11)),[0.0.hex()]*(n-1)+[bias.hex()],p,arm);checks+=1
    return dict(status='BOUNDED_DECISION_SYNTHETIC_PASS',checks=checks,natural_fits=0,synthetic_fits=4)
