"""Refine an existing fixed-box CE point; unchanged objective and feature basis."""
import numpy as np
import rc_h593_box_ce_core_v1 as B

BOUND=B.BOUND


def hessian(d,p):
    mean=np.einsum('qc,qck->qk',p,d)
    centered=(d-mean[:,None,:])*np.sqrt(p)[...,None]
    flat=centered.reshape(-1,d.shape[-1])
    return (flat.T@flat)/len(d)


def projected_gradient(t,g):
    return np.where(((t<=-BOUND)&(g>=0))|((t>=BOUND)&(g<=0)),0.,g)


def fit(d,y,initial):
    t=np.asarray(initial,dtype=np.float64).copy();b=np.zeros(d.shape[:2])
    B.need(np.isfinite(t).all() and np.max(abs(t))<=BOUND,'VALID_INITIAL_POINT')
    initial_value,_,_=B.value_gradient(t,d,b,y);trace=[];message='iteration limit'
    for it in range(30):
        value,g,p=B.value_gradient(t,d,b,y);pg=projected_gradient(t,g)
        residual=float(abs(pg).max())
        if residual<=1e-11:message='projected gradient tolerance';break
        H=hessian(d,p)
        active=((t<=-BOUND)&(g>=0))|((t>=BOUND)&(g<=0))
        free=np.flatnonzero(~active & (np.diag(H)>0))
        step=np.zeros_like(t)
        if len(free):
            hf=H[np.ix_(free,free)];sc=np.sqrt(np.diag(hf))
            normalized=hf/sc[:,None]/sc[None,:]
            direction=np.linalg.lstsq(normalized,-g[free]/sc,rcond=1e-14)[0]/sc
            step[free]=direction
        step[((t<=-BOUND)&(step<0))|((t>=BOUND)&(step>0))]=0
        if float(g@step)>=0:step=-pg
        if not np.any(step):message='no feasible descent direction';break
        alpha=1.
        for k,s in enumerate(step):
            if s>0:alpha=min(alpha,(BOUND-t[k])/s)
            elif s<0:alpha=min(alpha,(-BOUND-t[k])/s)
        accepted=False
        for backtrack in range(50):
            candidate=np.clip(t+alpha*step,-BOUND,BOUND)
            nv,ng,_=B.value_gradient(candidate,d,b,y)
            tolerance=8*np.finfo(float).eps*max(1.,abs(value))
            armijo=nv<=value+1e-4*float(g@(candidate-t))
            roundoff_only=nv<=value+tolerance and abs(projected_gradient(candidate,ng)).max()<residual
            if armijo or roundoff_only:
                accepted=True;break
            alpha*=.5
        trace.append(dict(iteration=it,CE=value,projected_gradient_linf=residual,active_coordinates=np.flatnonzero(active).tolist(),
            alpha=float(alpha),line_search_steps=backtrack+1,accepted=accepted))
        if not accepted:message='line search unresolved';break
        t=candidate
    value,g,_=B.value_gradient(t,d,b,y);residual=float(abs(projected_gradient(t,g)).max())
    B.need(value<=initial_value+1e-12,'NO_TRAIN_CE_INCREASE')
    cert=B.exact_certificate(d,b,y,t)
    return t,dict(initial_parameters=[float(v).hex() for v in initial],initial_CE=initial_value,final_CE=value,
        gradient_linf=float(abs(g).max()),projected_gradient_linf=residual,
        solver=dict(success=residual<=1e-11,message=message,iterations=len(trace),method='box active-set damped Newton',trace=trace),
        boundary_coordinates=np.flatnonzero(abs(t)==BOUND).tolist(),certificate=cert)
