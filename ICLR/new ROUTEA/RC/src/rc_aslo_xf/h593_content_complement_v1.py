"""Fixed joint evidence with independently cached pure-content contrasts."""
import torch
from rc_aslo_xf.h593_joint_diagonal_v1 import expand as diagonal_basis
from rc_aslo_xf.h593_joint_diagonal_v1 import objective as parent_objective

ARMS=('JOINT_CONTENT_CE18','JOINT_CURVE_CE18','LINEAR_CONTENT_CE12')
PARAMETERS=dict(JOINT_CONTENT_CE18=18,JOINT_CURVE_CE18=18,LINEAR_CONTENT_CE12=12)


def expand(x,arm=None):
    base=diagonal_basis(x[...,:6])
    if arm is None: return base
    if x.shape[-1]!=11: raise ValueError('Expected original6 plus five content contrasts')
    if arm=='JOINT_CONTENT_CE18': result=torch.cat((base,x[...,6:]),dim=-1)
    elif arm=='JOINT_CURVE_CE18':
        terms=(x[...,1:6]*x[...,1:6])*x[...,1:6]
        result=torch.cat((base,terms),dim=-1)
    elif arm=='LINEAR_CONTENT_CE12': result=x
    else: raise ValueError('Unknown content-complement arm')
    if not bool(torch.isfinite(result).all()): raise ValueError('Nonfinite features')
    return result


def objective(z,y,arm):
    return parent_objective(z,y,'DIAG_CE13' if arm in ARMS else arm)


def training_loss(theta,phi,y,arm):
    n=PARAMETERS[arm]
    if theta.shape!=(n,) or phi.shape[-1]!=n-1: raise ValueError('Feature/parameter shape mismatch')
    return objective(phi@theta[:-1]+theta[-1],y,arm)


def fit(x,y,arm):
    phi=expand(x.detach(),arm)
    theta=torch.nn.Parameter(torch.zeros(PARAMETERS[arm],dtype=torch.float64))
    opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
    for _ in range(2000):
        opt.zero_grad();loss=training_loss(theta,phi,y,arm)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite CE')
        loss.backward();opt.step()
    if not bool(torch.isfinite(theta).all()): raise RuntimeError('Nonfinite parameters')
    return theta.detach()
