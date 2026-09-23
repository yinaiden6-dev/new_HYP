"""Three operator factors, preserving the original native FP64 reduction order."""
import itertools
import torch
from torch.nn import functional as F
FACTORS={'NATIVE':(1,1,1)}
FACTORS.update({f'M{m}Q{q}R{r}':(m,q,r) for m,q,r in itertools.product((0,1),repeat=3) if (m,q,r)!=(1,1,1)})
LEVELS=tuple(FACTORS)+('FIXED_FREE_ARGMAX',)
SCORE_KEYS=('real_score','visibility_mass','query_control_score','reference_control_score')


def compute(q,r,u,v):
    q=F.normalize(q.to(torch.float64),dim=1);r=F.normalize(r.to(torch.float64),dim=1);sim=q@r.T
    u=u.to(torch.float64);v=v.to(torch.float64);ur=u.roll(max(1,len(u)//2));vr=v.roll(max(1,len(v)//2))
    free,jfree=sim.max(1);weighted,jweighted=(sim*v[None]).max(1);rolled=(sim*vr[None]).max(1).values
    fixed=free*v[jfree];fixedroll=free*vr[jfree]
    masses=[torch.sqrt(u.mean()*v.mean()),torch.sqrt(ur.mean()*v.mean()),torch.sqrt(u.mean()*vr.mean())]
    def score(w,local,m):return m*(w*local).sum()/w.sum().clamp_min(1e-12)
    scores={}
    for name,(mf,qf,rf) in FACTORS.items():
        w=u if qf else torch.ones_like(u);wc=ur if qf else w
        local=weighted if rf else free;lc=rolled if rf else local
        mass=masses[0] if mf else u.new_ones(())
        qm=masses[1] if mf and qf else mass;rm=masses[2] if mf and rf else mass
        scores[name]=dict(zip(SCORE_KEYS,map(float,(score(w,local,mass),mass,score(wc,local,qm),score(w,lc,rm)))))
    scores['FIXED_FREE_ARGMAX']=dict(zip(SCORE_KEYS,map(float,(score(u,fixed,masses[0]),masses[0],score(ur,fixed,masses[1]),score(u,fixedroll,masses[2])))))
    contributions=masses[0]*u/u.sum().clamp_min(1e-12)
    detail=dict(query_visibility=u,reference_visibility=v,free_reference_token=jfree,weighted_reference_token=jweighted,
        free_maxsim=free,weighted_maxsim=weighted,fixed_free_index_weighted_value=fixed,
        native_token_contribution=contributions*weighted,
        reference_value_change_at_free_match=contributions*(fixed-free),reference_reselection_gain=contributions*(weighted-fixed))
    assert bool(((weighted-fixed)>=-1e-14).all())
    assert abs(float(detail['native_token_contribution'].sum())-scores['NATIVE']['real_score'])<2e-12
    reference_delta=scores['NATIVE']['real_score']-scores['M1Q1R0']['real_score']
    assert abs(reference_delta-float((detail['reference_value_change_at_free_match']+detail['reference_reselection_gain']).sum()))<2e-12
    return scores,detail


def independent(q,r,u,v):
    import numpy as np
    q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64);u=np.asarray(u,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12);r=r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-12);sim=q@r.T
    ur=np.roll(u,max(1,len(u)//2));vr=np.roll(v,max(1,len(v)//2));j=np.argmax(sim,axis=1);free=sim[np.arange(len(u)),j]
    out={}
    def one(w,local,m):return float(m*np.sum(w*local)/max(float(w.sum()),1e-12))
    for name in LEVELS:
        mf,qf,rf=FACTORS[name] if name in FACTORS else (1,1,1)
        w=u if qf else np.ones_like(u);wc=ur if qf else w
        if name=='FIXED_FREE_ARGMAX':a=free*v[j];c=free*vr[j]
        elif rf:a=np.max(sim*v[None],axis=1);c=np.max(sim*vr[None],axis=1)
        else:a=free;c=free
        m=np.sqrt(u.mean()*v.mean()) if mf else 1.;qm=np.sqrt(ur.mean()*v.mean()) if mf and qf else m;rm=np.sqrt(u.mean()*vr.mean()) if mf and rf else m
        out[name]=dict(zip(SCORE_KEYS,(one(w,a,m),float(m),one(wc,a,qm),one(w,c,rm))))
    return out


def self_test():
    gen=torch.Generator().manual_seed(20260922);q=torch.randn(7,4,generator=gen,dtype=torch.float64);r=torch.randn(9,4,generator=gen,dtype=torch.float64);u=torch.rand(7,generator=gen,dtype=torch.float64);v=torch.rand(9,generator=gen,dtype=torch.float64)
    a,d=compute(q,r,u,v);b=independent(q.numpy(),r.numpy(),u.numpy(),v.numpy())
    assert max(abs(a[m][k]-b[m][k]) for m in LEVELS for k in SCORE_KEYS)<1e-13
    for m,(_,qf,rf) in FACTORS.items():
        if not qf:assert a[m]['real_score']==a[m]['query_control_score']
        if not rf:assert a[m]['real_score']==a[m]['reference_control_score']
    assert a['NATIVE']['real_score']>=a['FIXED_FREE_ARGMAX']['real_score']-1e-14
    return dict(status='QUALITY_OPERATOR_SYNTHETIC_PASS',arms=len(LEVELS),checks=['independent_numpy','disabled_query_control_identity','disabled_reference_control_identity','reselection_nonnegative','exact_reference_decomposition'])
