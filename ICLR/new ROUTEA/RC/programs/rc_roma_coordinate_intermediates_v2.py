"""Compact, replayable coordinates and MaxSim decisions; no scorer mutation."""
import numpy as np
import torch
from torch.nn import functional as F


def coordinate_record(warp,geometry):
    # Exact float32 coordinate at each token-cell center; area moments are FP64.
    a=warp[0].detach().float().cpu().numpy();h,w,c=a.shape;assert c==2
    boxes=np.asarray(geometry.cell_boxes_xyxy,dtype=np.float64)
    valid=np.asarray(geometry.valid_patch_mask,dtype=bool)
    lo=np.floor(boxes[:,:2]*[w,h]).astype(np.int64);hi=np.ceil(boxes[:,2:]*[w,h]).astype(np.int64)
    lo=np.minimum(np.maximum(lo,0),[w-1,h-1]);hi=np.minimum(np.maximum(hi,lo+1),[w,h])
    x0,y0=lo.T;x1,y1=hi.T;area=(x1-x0)*(y1-y0)
    center=np.floor((boxes[:,:2]+boxes[:,2:])*.5*[w,h]).astype(np.int64);center=np.minimum(np.maximum(center,0),[w-1,h-1])
    means=[]
    x=a[...,0].astype(np.float64);y=a[...,1].astype(np.float64)
    for field in (x,y,x*x,y*y,x*y,((abs(x)>1)|(abs(y)>1)).astype(np.float64)):
        integral=np.pad(field.cumsum(0).cumsum(1),((1,0),(1,0)))
        value=(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/area;value[~valid]=0.;means.append(value)
    means=np.stack(means,axis=1);cov=np.column_stack([means[:,2]-means[:,0]**2,means[:,3]-means[:,1]**2,means[:,4]-means[:,0]*means[:,1]])
    yy=np.minimum(((np.arange(64)+.5)*h/64).astype(int),h-1);xx=np.minimum(((np.arange(64)+.5)*w/64).astype(int),w-1)
    return dict(grid_hw=[h,w],cell_boxes_xyxy=torch.from_numpy(boxes.copy()),valid_patch_mask=torch.from_numpy(valid.copy()),
        center_xy=torch.from_numpy(a[center[:,1],center[:,0]].copy()),cell_mean_xy=torch.from_numpy(means[:,:2].copy()),
        cell_cov_xx_yy_xy=torch.from_numpy(cov.copy()),cell_out_of_frame_fraction=torch.from_numpy(means[:,5].copy()),
        visualization_sample_64x64=torch.from_numpy(a[yy[:,None],xx[None,:]].copy()),
        coordinate_frame='target EXIF-oriented RoMa input, normalized [-1,1]; 64x64 is subsampling, not full dense output')


def content_record(q,r,u,v):
    sim=F.normalize(q.to(torch.float64),dim=1)@F.normalize(r.to(torch.float64),dim=1).T
    free,jfree=sim.max(1);weighted,jweighted=(sim*v[None]).max(1)
    mass=(u.mean()*v.mean()).sqrt();contribution=mass*u*weighted/u.sum().clamp_min(1e-12)
    return dict(free_maxsim_reference_token=jfree,free_maxsim_similarity=free,
        weighted_maxsim_reference_token=jweighted,weighted_maxsim_value=weighted,
        selected_unweighted_cosine=sim[torch.arange(len(u)),jweighted],query_token_score_contribution=contribution)


def self_test():
    from types import SimpleNamespace
    g=SimpleNamespace(cell_boxes_xyxy=np.array([[0.,0.,.5,.5],[.5,.5,1.,1.]]),valid_patch_mask=np.array([True,True]))
    w=torch.arange(4*4*2,dtype=torch.float32).reshape(1,4,4,2)/20-1
    p=coordinate_record(w,g)
    for i,a in enumerate((w[0,:2,:2],w[0,2:,2:])):
        m=a.double().reshape(-1,2).mean(0)
        assert torch.allclose(p['cell_mean_xy'][i],m,atol=1e-14,rtol=0)
        d=a.double().reshape(-1,2)-m;cov=torch.stack([d[:,0].square().mean(),d[:,1].square().mean(),(d[:,0]*d[:,1]).mean()])
        assert torch.allclose(p['cell_cov_xx_yy_xy'][i],cov,atol=1e-14,rtol=0)
    q=torch.eye(2,dtype=torch.float64);r=torch.tensor([[1.,0.],[.8,.6]],dtype=torch.float64);u=torch.tensor([.4,.9],dtype=torch.float64);v=torch.tensor([.1,.9],dtype=torch.float64)
    s=content_record(q,r,u,v);assert s['free_maxsim_reference_token'][0]==0 and s['weighted_maxsim_reference_token'][0]==1
    return dict(status='COORDINATE_INTERMEDIATES_PASS',checks=['cell_mean','cell_covariance','free_and_weighted_match_index'])
