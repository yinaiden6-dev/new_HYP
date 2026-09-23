#!/usr/bin/env python3
"""Independent fixed-mask FULL-CE and existing full13 logit validator."""
import numpy as np
from validate_rc_h593_joint_diagonal_v1 import basis, logits_check
from validate_rc_h593_joint_diagonal_v1 import values_gradient as parent_values_gradient


def mask(arm):
    assert arm in ('JOINT_SQUARE_CE12','RAW_SQUARE_CE8')
    m = np.ones(13)
    if arm == 'JOINT_SQUARE_CE12': m[6] = 0.
    else: m[7:12] = 0.
    return m


def values_gradient(theta, x, y, arm):
    m = mask(arm)
    loss, grad = parent_values_gradient(np.asarray(theta)*m,x,y,'DIAG_CE13')
    return loss, grad*m


def self_test():
    import torch
    from rc_aslo_xf.h593_square_blocks_v1 import active_mask, expand, training_loss, fit
    from rc_aslo_xf.h593_joint_diagonal_v1 import objective
    rng = np.random.default_rng(20260921)
    checks = 0
    for scale in (0.,.1,1.,10.):
        x = rng.normal(size=(4,127,6))*scale
        y = np.array([-1,0,12,126])
        for arm in ('JOINT_SQUARE_CE12','RAW_SQUARE_CE8'):
            assert np.array_equal(active_mask(arm).numpy(),mask(arm))
            theta = torch.tensor(rng.normal(size=13),dtype=torch.float64,requires_grad=True)
            xx, yy = torch.from_numpy(x), torch.from_numpy(y)
            loss = training_loss(theta,expand(xx),yy,arm)
            gradient = torch.autograd.grad(loss,theta)[0].numpy()
            lv,gv = values_gradient(theta.detach().numpy(),x,y,arm)
            assert abs(float(loss.detach())-lv)<2e-10 and np.max(abs(gradient-gv))<2e-10
            assert np.all(gradient[mask(arm)==0]==0)
            checks += 3
    x = torch.zeros((2,127,6),dtype=torch.float64)
    x[1,0,0] = -1.; x[1,0,1] = 1.
    y = torch.tensor([-1,0])
    for arm in ('JOINT_SQUARE_CE12','RAW_SQUARE_CE8'):
        t = fit(x,y,arm)
        assert bool((t[active_mask(arm)==0]==0).all())
        z = expand(x)@t[:-1]+t[-1]
        chosen = torch.where(z.max(1).values>0,z.argmax(1),-1)
        assert torch.equal(chosen,y)
        assert torch.equal(training_loss(t,expand(x),y,arm),objective(z,y,'DIAG_CE13'))
        checks += 3
    return dict(status='SQUARE_BLOCKS_SYNTHETIC_PASS',checks=checks,natural_fits=0,synthetic_fits=2)
