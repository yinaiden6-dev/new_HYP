"""Fixed-box FULL CE optimization and an exact dyadic Fenchel certificate.

Certificate implementation retained from the qualified 20260912 CE diagnostic.
No input loading; fixed FP64 feature endpoints, HOLD already included.
"""
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
import rc_convex_loss_cause_core_v1 as K
DEN=1<<52
BOUND=64

def need(ok,message):
    if not ok:raise AssertionError(message)

def value_gradient(theta,d,b,y):
    z=b+np.einsum('qck,k->qc',d,theta)
    lse=logsumexp(z,axis=1)
    probability=np.exp(z-lse[:,None]);r=np.arange(len(y))
    value=float(np.mean(lse-z[r,y]))
    grad=np.mean(np.einsum('qc,qck->qk',probability,d)-d[r,y],axis=0)
    return value,grad,probability


def exact_certificate(d,b,y,theta,counts=None):
    """Exact endpoint problem. The dyadic simplex witness supplies a valid cut."""
    K.iv.dps=50
    need(np.isfinite(d).all() and np.isfinite(b).all() and np.max(np.abs(theta))<=BOUND,'FINITE_BOX_POINT')
    if counts is None:
        _,_,p=value_gradient(theta,d,b,y)
        counts=np.floor(p*DEN).astype(np.int64)
        for i in range(len(counts)):counts[i,int(np.argmax(p[i]))]+=DEN-int(counts[i].sum())
    counts=np.asarray(counts,dtype=np.int64)
    need(counts.shape==b.shape and np.all(counts>=0) and np.all(counts.sum(1)==DEN),'EXACT_DYADIC_SIMPLEX')
    point=[K.rat(t) for t in theta]
    slope=[Fraction(0)]*len(theta);offset=Fraction(0);lo=hi=Fraction(0)
    for i in range(len(y)):
        bs=[K.rat(v) for v in b[i]];ds=[[K.rat(v) for v in row] for row in d[i]]
        logits=[bb+sum((x*t for x,t in zip(row,point)),Fraction(0)) for bb,row in zip(bs,ds)]
        maximum=max(logits)
        total=K.iv.mpf(0)
        for z in logits:total+=K.iv.exp(K._interval(z-maximum))
        ends=K._ends(K._interval(maximum-logits[int(y[i])])+K.iv.log(total))
        lo+=ends[0];hi+=ends[1]
        entropy=Fraction(0);q=[Fraction(int(v),DEN) for v in counts[i]]
        for p in q:
            if p:entropy+=K._ends(-K._interval(p)*K.iv.log(K._interval(p)))[0]
        offset+=sum((p*bb for p,bb in zip(q,bs)),Fraction(0))-bs[int(y[i])]+entropy
        for k in range(len(theta)):
            slope[k]+=sum((p*row[k] for p,row in zip(q,ds)),Fraction(0))-ds[int(y[i])][k]
    n=len(y);slope=[v/n for v in slope];offset/=n;lo/=n;hi/=n
    lower=max(Fraction(0),offset-BOUND*sum((abs(v) for v in slope),Fraction(0)))
    need(lower<=hi,'VALID_LOWER_UPPER_ORDER')
    gap=hi-lower
    return dict(probability_counts=counts.tolist(),probability_denominator=DEN,objective_lower=str(lo),objective_upper=str(hi),box_lower=str(lower),gap_upper=str(gap),gap_upper_float=K._up(gap),box_bound=BOUND,affine_slope=[str(v) for v in slope],affine_offset_lower=str(offset),certified_box_gap_le_1e_6=gap<=Fraction(1,1000000))


def fit(d,y,initial):
    b=np.zeros(d.shape[:2],dtype=np.float64)
    need(np.max(abs(initial))<=BOUND,'OLD_HEAD_INSIDE_FIXED_BOX')
    opt=minimize(lambda t:value_gradient(t,d,b,y)[:2],initial,jac=True,
        method='L-BFGS-B',bounds=[(-BOUND,BOUND)]*len(initial),
        options=dict(maxiter=2000,maxls=50,ftol=1e-14,gtol=1e-10))
    theta=np.asarray(opt.x,dtype=np.float64)
    oldval,_,_=value_gradient(initial,d,b,y)
    newval,grad,_=value_gradient(theta,d,b,y)
    need(newval<=oldval+1e-10,'NO_TRAIN_CE_INCREASE')
    cert=exact_certificate(d,b,y,theta)
    return theta,dict(initial_parameters=[float(v).hex() for v in initial],
        initial_CE=oldval,final_CE=newval,gradient_linf=float(abs(grad).max()),
        solver=dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),evaluations=int(opt.nfev)),
        boundary_coordinates=[i for i,v in enumerate(theta) if abs(v)==BOUND],certificate=cert)
