"""Add three fixed S/M/L readout terms to the successful diagonal FULL-CE head."""
import torch
from rc_aslo_xf.h593_joint_diagonal_v1 import expand as diagonal_basis
from rc_aslo_xf.h593_joint_diagonal_v1 import objective as parent_objective

ARMS = ('SML_CROSS_CE16', 'SML_CUBE_CE16')


def expand(x, arm=None):
    base = diagonal_basis(x)
    if arm is None: return base
    if arm == 'SML_CROSS_CE16':
        terms = torch.stack((x[...,1]*x[...,2], x[...,1]*x[...,3], x[...,2]*x[...,3]), dim=-1)
    elif arm == 'SML_CUBE_CE16':
        terms = (x[...,1:4]*x[...,1:4])*x[...,1:4]
    else: raise ValueError('Unknown SML extension')
    result = torch.cat((base,terms),dim=-1)
    if not bool(torch.isfinite(result).all()): raise ValueError('Nonfinite SML basis')
    return result


def objective(z,y,arm):
    return parent_objective(z,y,'DIAG_CE13' if arm in ARMS else arm)


def training_loss(theta,phi,y,arm):
    if arm not in ARMS or theta.shape!=(16,) or phi.shape[-1]!=15:
        raise ValueError('Expected full 16-parameter SML head')
    return objective(phi@theta[:-1]+theta[-1],y,arm)


def fit(x,y,arm):
    phi = expand(x.detach(),arm)
    theta = torch.nn.Parameter(torch.zeros(16,dtype=torch.float64))
    opt = torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
    for _ in range(2000):
        opt.zero_grad(); loss=training_loss(theta,phi,y,arm)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite CE')
        loss.backward(); opt.step()
    if not bool(torch.isfinite(theta).all()): raise RuntimeError('Nonfinite parameters')
    return theta.detach()
