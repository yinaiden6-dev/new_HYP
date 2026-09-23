"""Uninterrupted 4000-step CE with exact checks against frozen 2000-step heads."""
import torch
from rc_aslo_xf.h593_content_complement_v1 import expand as content_basis
from rc_aslo_xf.h593_joint_diagonal_v1 import objective as legacy_objective

ARMS=('CONTENT_CONTINUOUS_CE18','DIAG_CONTINUOUS_CE13')
PARAMETERS=dict(CONTENT_CONTINUOUS_CE18=18,DIAG_CONTINUOUS_CE13=13)


def expand(x,arm=None):
    if arm is None or arm in ('DIAG_CONTINUOUS_CE13','DIAG_CONT_CE13'):return content_basis(x)
    if arm in ('CONTENT_CONTINUOUS_CE18','CONTENT_CONT_CE18','JOINT_CONTENT_CE18'):
        return content_basis(x,'JOINT_CONTENT_CE18')
    raise ValueError('Unknown fixed basis')


def objective(z,y,arm):
    return legacy_objective(z,y,'DIAG_CE13' if arm in ARMS else arm)


def training_loss(theta,phi,y,arm):
    n=PARAMETERS[arm]
    if theta.shape!=(n,) or phi.shape[-1]!=n-1:raise ValueError('Wrong parameter or input shape')
    return objective(phi@theta[:-1]+theta[-1],y,arm)


def snapshot(theta,opt):
    state=opt.state[theta]
    return dict(step=int(state['step'].item()),parameters=[float(v).hex() for v in theta.detach()],
        exp_avg=[float(v).hex() for v in state['exp_avg']],exp_avg_sq=[float(v).hex() for v in state['exp_avg_sq']])


def fit(x,y,arm,expected_midpoint):
    if expected_midpoint.dtype!=torch.float64 or expected_midpoint.shape!=(PARAMETERS[arm],):
        raise ValueError('Expected frozen same-fold 2000-step parameter vector')
    phi=expand(x.detach(),arm);theta=torch.nn.Parameter(torch.zeros(PARAMETERS[arm],dtype=torch.float64))
    opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001);states={}
    for step in range(1,4001):
        opt.zero_grad();loss=training_loss(theta,phi,y,arm)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite CE')
        loss.backward();opt.step()
        if step==2000:
            if not torch.equal(theta.detach(),expected_midpoint):raise RuntimeError('2000-step frozen parameter mismatch')
            states['midpoint']=snapshot(theta,opt)
    if not bool(torch.isfinite(theta).all()):raise RuntimeError('Nonfinite parameters')
    states['final']=snapshot(theta,opt)
    if states['midpoint']['step']!=2000 or states['final']['step']!=4000:raise RuntimeError('Optimizer reset or wrong step count')
    return theta.detach(),states
