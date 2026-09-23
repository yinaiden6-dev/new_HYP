"""Original precise LR input and matcher, returning before any refinement."""
from collections import Counter
import torch
from torch.nn import functional as F

@torch.inference_mode()
def match_coarse(model, image_a, image_b):
    assert not model.training and model.bidirectional and model.threshold is None
    assert model.H_lr == model.W_lr == 800
    assert torch.get_float32_matmul_precision() == 'highest'
    a, b = model._load_image(image_a), model._load_image(image_b)
    a = F.interpolate(a, size=(800,800), mode='bicubic', align_corners=False, antialias=True)
    b = F.interpolate(b, size=(800,800), mode='bicubic', align_corners=False, antialias=True)
    fa, fb = model.f(a), model.f(b)
    out = model.matcher(fa, fb, img_A=a, img_B=b, bidirectional=True)
    return {**out, 'overlap_AB':out['confidence_AB'][...,:1].sigmoid(),
            'overlap_BA':out['confidence_BA'][...,:1].sigmoid()}

class Calls:
    def __init__(self,model):
        self.counts=Counter(); self.handles=[]; self.coarse=None
        for name,module in [('descriptor',model.f),('matcher',model.matcher),('fine_features',model.refiner_features),*[(f'refiner_{k}',m) for k,m in model.refiners.items()]]:
            def hook(mod,args,out,name=name):
                self.counts[name]+=1
                if name=='matcher':
                    self.coarse={k:out[k].detach().clone() for k in ('warp_AB','warp_BA','confidence_AB','confidence_BA')}
            self.handles.append(module.register_forward_hook(hook))
    def reset(self):self.counts.clear();self.coarse=None
    def close(self):
        for h in self.handles:h.remove()
