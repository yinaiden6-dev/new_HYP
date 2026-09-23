"""Two prespecified blocks of squared features under the original FULL CE.

Keep 13 storage coordinates for identical reduction layout. Disabled square
coordinates have zero gradient, zero initialization and remain exactly zero;
effective learned parameter counts (including bias) are eight and twelve.
"""
import torch
from rc_aslo_xf.h593_joint_diagonal_v1 import expand, objective as parent_objective

ARMS = ('JOINT_SQUARE_CE12', 'RAW_SQUARE_CE8')


def active_mask(arm):
    if arm not in ARMS:
        raise ValueError('Unknown square block')
    mask = torch.ones(13, dtype=torch.float64)
    if arm == 'JOINT_SQUARE_CE12': mask[6] = 0.
    else: mask[7:12] = 0.
    return mask


def objective(z, y, arm):
    # Legacy names remain available for matched TRAIN metric reporting.
    return parent_objective(z, y, 'DIAG_CE13' if arm in ARMS else arm)


def training_loss(theta, phi, y, arm):
    effective = theta * active_mask(arm)
    return objective(phi @ effective[:-1] + effective[-1], y, arm)


def fit(x, y, arm):
    phi = expand(x.detach())
    mask = active_mask(arm)
    theta = torch.nn.Parameter(torch.zeros(13, dtype=torch.float64))
    opt = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
    for _ in range(2000):
        opt.zero_grad()
        loss = training_loss(theta, phi, y, arm)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite CE')
        loss.backward(); opt.step()
    if not bool(torch.isfinite(theta).all()) or not bool((theta[mask == 0] == 0).all()):
        raise RuntimeError('Invalid fitted masked head')
    return theta.detach()
