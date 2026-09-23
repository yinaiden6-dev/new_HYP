"""Matched warm continuation: full CE versus bounded full-decision margin risk."""
import torch
from rc_aslo_xf.h593_content_complement_v1 import expand as content_basis
from rc_aslo_xf.h593_joint_diagonal_v1 import objective as legacy_objective

ARMS=('DIAG_BOUND13','DIAG_CONT_CE13','CONTENT_BOUND18','CONTENT_CONT_CE18')
PARAMETERS={m:(13 if m.startswith('DIAG_') else 18) for m in ARMS}


def expand(x,arm=None):
    if arm is None or arm.startswith('DIAG_'): return content_basis(x)
    if arm in ARMS or arm=='JOINT_CONTENT_CE18': return content_basis(x,'JOINT_CONTENT_CE18')
    raise ValueError('Unknown basis')


def bounded_loss(z,y):
    if z.dtype!=torch.float64 or z.shape!=(len(y),127) or y.dtype!=torch.long:
        raise ValueError('Expected FP64 logits and long targets')
    if not len(y) or bool(((y < -1)|(y > 126)).any()) or not bool(torch.isfinite(z).all()):
        raise ValueError('Invalid target-present batch')
    full=torch.cat((z.new_zeros((len(z),1)),z),dim=1)
    target=y+1
    values=full.gather(1,target[:,None]).squeeze(1)
    mask=torch.zeros_like(full,dtype=torch.bool).scatter_(1,target[:,None],True)
    rival=full.masked_fill(mask,-torch.inf).amax(dim=1)
    return torch.sigmoid(rival-values).mean()


def objective(z,y,arm):
    if arm=='FULL_BOUNDED' or arm in ('DIAG_BOUND13','CONTENT_BOUND18'): return bounded_loss(z,y)
    return legacy_objective(z,y,'DIAG_CE13' if arm in ARMS else arm)


def training_loss(theta,phi,y,arm):
    n=PARAMETERS[arm]
    if theta.shape!=(n,) or phi.shape[-1]!=n-1: raise ValueError('Wrong basis or parameters')
    return objective(phi@theta[:-1]+theta[-1],y,arm)


def fit(x,y,arm,initial):
    if initial.shape!=(PARAMETERS[arm],) or initial.dtype!=torch.float64:
        raise ValueError('Expected same-fold frozen CE initialization')
    phi=expand(x.detach(),arm);theta=torch.nn.Parameter(initial.detach().clone())
    opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
    for _ in range(2000):
        opt.zero_grad();loss=training_loss(theta,phi,y,arm)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite loss')
        loss.backward();opt.step()
    if not bool(torch.isfinite(theta).all()): raise RuntimeError('Nonfinite parameters')
    return theta.detach()
