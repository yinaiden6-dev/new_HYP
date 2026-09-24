"""One shared TRAIN-fitted scalar for bounded predicted quality M."""
import numpy as np

EPS=1e-8


def apply(mass,bias):
    mass=np.asarray(mass,dtype=np.float64)
    assert np.isfinite(mass).all() and (mass>=0).all() and (mass<=1).all()
    clipped=np.clip(mass,1e-12,1-1e-12)
    z=np.log(clipped)-np.log1p(-clipped)+bias
    return np.exp(-np.logaddexp(0.,-z))


def fit(mass,teacher):
    mass,teacher=np.asarray(mass,dtype=np.float64),np.asarray(teacher,dtype=np.float64)
    assert mass.shape==teacher.shape and mass.ndim==2 and mass.shape[1]==128
    assert np.isfinite(teacher).all() and (teacher>=0).all() and (teacher<=1).all()
    def residual(b):return float(np.mean(np.log(apply(mass,b)+EPS)-np.log(teacher+EPS)))
    lo,hi=-30.,30.
    assert residual(lo)<=0<=residual(hi),'CALIBRATION_BRACKET'
    for _ in range(80):
        mid=(lo+hi)/2
        if residual(mid)>0:hi=mid
        else:lo=mid
    bias=(lo+hi)/2
    assert abs(residual(bias))<1e-12
    return bias
