import numpy as np
import torch
from scripts.colnomic_mechanism.weighted_maxsim import score_bounds, weight_mask


def test_bounds_cover_unweighted_and_weighted_unnormalized_similarities():
    generator=torch.Generator().manual_seed(17)
    query=torch.randn(17,8,generator=generator)*2
    refs=[torch.randn(n,8,generator=generator)*(1+k) for k,n in enumerate([8,13,23])]
    mask=weight_mask(query)
    maxima=np.stack([(query@r.T).max(1).values.numpy() for r in refs],1)
    means=maxima.mean(0)
    norms=np.array([float(r.double().norm(dim=1).max()) for r in refs])
    etas=np.array([0.,.05,.2,1.])
    lower,upper=score_bounds(query,mask,norms,means,etas)
    exact=(maxima.sum(0)[None]+etas[:,None]*maxima[mask].sum(0)[None])/(len(query)+etas[:,None]*mask.sum())
    assert np.all(lower<=exact+1e-6) and np.all(upper>=exact-1e-6)


def test_spatial_bounds_cover_penalty_and_alignment_rewards():
    generator=torch.Generator().manual_seed(31)
    query=torch.randn(11,5,generator=generator)
    reference=torch.randn(13,5,generator=generator)
    dot=query@reference.T
    distances=torch.rand(dot.shape,generator=generator)
    maxima=(dot-.2*distances).max(1).values.numpy()+.1
    mask=weight_mask(query)
    norm=np.array([float(reference.double().norm(dim=1).max())])
    etas=np.array([0.,.1,.5])
    lower,upper=score_bounds(query,mask,norm,np.array([maxima.mean()]),etas,spatial_lambda=.2,alignment_gamma=.1)
    exact=(maxima.sum()+etas*maxima[mask].sum())/(len(query)+etas*mask.sum())
    assert np.all(lower[:,0]<=exact+1e-6) and np.all(upper[:,0]>=exact-1e-6)
