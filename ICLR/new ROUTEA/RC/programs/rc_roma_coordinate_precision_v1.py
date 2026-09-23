"""Coordinate-only intervention at RoMa refinement inputs; no weight edits."""
import torch

LEVELS={'NATIVE':0,'GRID256':256,'GRID64':64,'GRID16':16}


def quantize(x,bins):
    if bins==0:return x
    if bins not in (16,64,256):raise ValueError('Unregistered coordinate precision')
    # Normalized coordinates span [-1,1]. Keep out-of-frame coordinates; no clamp.
    return torch.round(x*(bins/2))*(2/bins)


class CoordinateHook:
    def __init__(self,model):
        self.bins=0;self.calls=[];self.handles=[]
        for name,module in model.refiners.items():
            self.handles.append(module.register_forward_pre_hook(self.make_hook(name),with_kwargs=True))

    def make_hook(self,name):
        def hook(module,args,kwargs):
            if self.bins==0:return None
            original=kwargs['prev_warp'];q=quantize(original,self.bins);delta=q-original
            error=float(delta.abs().max());assert error<=1/self.bins+2e-7
            self.calls.append(dict(refiner=str(name),elements=original.numel(),changed=int(torch.count_nonzero(delta)),
                max_abs_normalized=error,squared_error_sum=float((delta.double()**2).sum())))
            changed=dict(kwargs);changed['prev_warp']=q
            assert changed['prev_confidence'] is kwargs['prev_confidence']
            for k in ('f_A','f_B','scale_factor'):assert changed[k] is kwargs[k]
            return args,changed
        return hook

    def select(self,name):
        self.bins=LEVELS[name];self.calls=[]

    def close(self):
        for h in self.handles:h.remove()


def self_test():
    import numpy as np
    class Refiner(torch.nn.Module):
        def forward(self,*,prev_warp,prev_confidence,f_A,f_B,scale_factor):
            return prev_warp,prev_confidence,f_A,f_B,scale_factor
    class Model:
        refiners={'4':Refiner(),'2':Refiner(),'1':Refiner()}
    model=Model();h=CoordinateHook(model);checks=0
    x=torch.tensor([-1.8,-1.,-.7,-.003,0.,.021,.62,1.,1.4],dtype=torch.float32).reshape(1,1,-1,1).repeat(1,1,1,2)
    c=torch.rand((1,1,9,4));a=torch.zeros((1,1,9,1));b=a.clone();s=torch.ones(2)
    for name,bins in LEVELS.items():
        h.select(name);out=model.refiners['4'](prev_warp=x,prev_confidence=c,f_A=a,f_B=b,scale_factor=s)
        if bins:
            expected=np.rint(x.numpy()*(bins/2))*(2/bins)
            assert np.array_equal(out[0].numpy(),expected) and h.calls[0]['changed']>0
            assert torch.equal(quantize(out[0],bins),out[0]);checks+=3
        else:assert out[0] is x and not h.calls;checks+=2
        assert out[1] is c and out[2] is a and out[3] is b and out[4] is s;checks+=1
    assert quantize(x,16).min()<-1 and quantize(x,16).max()>1;checks+=1
    h.close()
    assert model.refiners['4'](prev_warp=x,prev_confidence=c,f_A=a,f_B=b,scale_factor=s)[0] is x;checks+=1
    return dict(status='ROMA_COORDINATE_PRECISION_SYNTHETIC_PASS',checks=checks,natural_forwards=0)
