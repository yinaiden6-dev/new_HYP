#!/usr/bin/env python3
"""Independent NumPy SML basis, FULL CE, gradients, and physical decisions."""
import numpy as np
from validate_rc_h593_joint_diagonal_v1 import basis as parent_basis
from validate_rc_h593_joint_diagonal_v1 import logits_check as parent_logits_check


def basis(x,arm):
    x=np.asarray(x,dtype=np.float64)
    base=parent_basis(x)
    if arm=='SML_CROSS_CE16': terms=np.stack([x[...,i]*x[...,j] for i,j in ((1,2),(1,3),(2,3))],axis=-1)
    else:
        assert arm=='SML_CUBE_CE16'
        terms=np.stack([(x[...,i]*x[...,i])*x[...,i] for i in (1,2,3)],axis=-1)
    result=np.concatenate((base,terms),axis=-1)
    assert result.shape[-1]==15 and np.isfinite(result).all()
    return result


def values_gradient(theta,x,y,arm):
    theta=np.asarray(theta,dtype=np.float64);phi=basis(x,arm)
    y=np.asarray(y,dtype=np.int64)
    assert theta.shape==(16,) and len(y)==len(x) and len(y)>0 and np.isin(y,np.arange(-1,127)).all()
    z=np.sum(phi*theta[:-1],axis=-1)+theta[-1]
    full=np.column_stack((np.zeros(len(x)),z))
    peak=full.max(1);exp=np.exp(full-peak[:,None]);probs=exp/exp.sum(1)[:,None]
    loss=np.mean(peak+np.log(exp.sum(1))-full[np.arange(len(y)),y+1])
    probs[np.arange(len(y)),y+1]-=1.
    dz=probs[:,1:]/len(y)
    return float(loss),np.r_[np.sum(phi*dz[...,None],axis=(0,1)),dz.sum()]


def logits_check(x,parameters,predictions,model):
    if len(parameters)!=16: return parent_logits_check(x,parameters,predictions,model)
    t=np.array([float.fromhex(v) for v in parameters])
    z=np.sum(basis(x,model)*t[:-1],axis=-1)+t[-1]
    wanted=np.array([[float.fromhex(v) for v in r['models'][model]['logits_hex']] for r in predictions])
    error=float(abs(z-wanted).max());assert error<2e-10
    for r,score,want in zip(predictions,z,wanted):
        top=int(score.argmax());assert top==int(want.argmax()) and bool(score[top]>0)==bool(want[top]>0)
        pos=r['challenger_positions'][top] if score[top]>0 else r['winner']
        assert r['candidate_physical_rows'][pos]==r['models'][model]['selected']
    return dict(logit_count=int(z.size),max_abs_error=error,decisions=len(predictions))


def self_test():
    import torch
    from rc_aslo_xf.h593_sml_extension_v1 import expand,training_loss,fit
    from rc_aslo_xf.h593_joint_diagonal_v1 import objective
    rng=np.random.default_rng(20260921);checks=0
    for scale in (0.,.1,1.,3.):
        x=rng.normal(size=(4,127,6))*scale;y=np.array([-1,0,12,126])
        for arm in ('SML_CROSS_CE16','SML_CUBE_CE16'):
            xx=torch.from_numpy(x);yy=torch.from_numpy(y)
            phi=expand(xx,arm)
            assert np.array_equal(phi.numpy(),basis(x,arm))
            theta=torch.tensor(rng.normal(size=16),dtype=torch.float64,requires_grad=True)
            loss=training_loss(theta,phi,yy,arm)
            grad=torch.autograd.grad(loss,theta)[0].numpy()
            v,g=values_gradient(theta.detach().numpy(),x,y,arm)
            assert abs(float(loss.detach())-v)<2e-10 and abs(grad-g).max()<2e-10
            # Zero extra coefficients embed the original diagonal function.
            small=torch.tensor(rng.normal(size=13),dtype=torch.float64)
            large=torch.cat((small[:12],torch.zeros(3,dtype=torch.float64),small[-1:]))
            torch.testing.assert_close(phi@large[:-1]+large[-1],expand(xx)@small[:-1]+small[-1],rtol=1e-13,atol=1e-13)
            checks+=3
    x=torch.zeros((2,127,6),dtype=torch.float64);x[1,0,0]=-1.;x[1,0,1]=1.;y=torch.tensor([-1,0])
    for arm in ('SML_CROSS_CE16','SML_CUBE_CE16'):
        t=fit(x,y,arm);z=expand(x,arm)@t[:-1]+t[-1]
        assert torch.equal(torch.where(z.max(1).values>0,z.argmax(1),-1),y)
        assert torch.equal(training_loss(t,expand(x,arm),y,arm),objective(z,y,'DIAG_CE13'))
        checks+=2
    # Exact zero ties HOLD, positive ties pick the first canonical challenger.
    p=[dict(winner=0,candidate_physical_rows=list(range(128)),challenger_positions=list(range(1,128)),models={})]
    for arm in ('SML_CROSS_CE16','SML_CUBE_CE16'):
        for bias,selected in ((0.,0),(1.,1),(-1.,0)):
            p[0]['models'][arm]=dict(logits_hex=[bias.hex()]*127,selected=selected)
            logits_check(np.zeros((1,127,6)),[0.0.hex()]*15+[bias.hex()],p,arm);checks+=1
    return dict(status='SML_EXTENSION_SYNTHETIC_PASS',checks=checks,natural_fits=0,synthetic_fits=2)
