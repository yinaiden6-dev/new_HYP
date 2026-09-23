"""Lossless persistent coarse descriptors and literal, cached pooling geometry."""
from collections import Counter, OrderedDict
import hashlib
import math
from pathlib import Path
import time
import torch
from rc_roma_exact_feature_cache_v2 import clone_tree, equal_tree


def tensor_key(x):
    y=x.detach().contiguous().cpu()
    h=hashlib.sha256()
    h.update(str((str(y.dtype),tuple(y.shape))).encode())
    h.update(y.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class CoarseReader:
    def __init__(self,model,bindings,checked):
        self.original=model.f.forward;self.module=model.f
        self.bindings=bindings;self.checked=checked;self.verified=set()
        self.memory=OrderedDict();self.stats=Counter();self.enabled=True
        model.f.forward=self.forward

    def forward(self,x):
        if not self.enabled:return self.original(x)
        start=time.monotonic();key=tensor_key(x)
        if key not in self.bindings:
            # Transformed visual arms require genuinely new features.
            self.stats['uncached_input_computations']+=1
            return self.original(x)
        if key not in self.memory:
            b=self.bindings[key]
            path=self.checked(b) if key not in self.verified else Path(b['path'])
            self.verified.add(key)
            p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert p['input_key']==key
            self.memory[key]=p['features'];self.stats['disk_loads']+=1
            if len(self.memory)>4:self.memory.popitem(last=False)
        self.memory.move_to_end(key)
        value=[v.to(x.device).clone() for v in self.memory[key]]
        self.stats['hits']+=1;self.stats['read_seconds']+=time.monotonic()-start
        return value

    def close(self):
        self.module.forward=self.original;self.memory.clear()


class ExactPool:
    """Only cache integer boxes; preserve original FP64 slice.mean and stack."""
    def __init__(self):self.cache=OrderedDict();self.stats=Counter()
    def __call__(self,overlap,geometry):
        t=torch.as_tensor(overlap,dtype=torch.float64);h,w=t.shape
        key=(id(geometry),h,w)
        if key not in self.cache:
            bounds=[]
            for box,valid in zip(geometry.cell_boxes_xyxy,geometry.valid_patch_mask,strict=True):
                if not bool(valid):bounds.append(None);continue
                x0,y0,x1,y1=[float(v) for v in box]
                ix0=max(0,min(w-1,int(math.floor(x0*w))))
                iy0=max(0,min(h-1,int(math.floor(y0*h))))
                ix1=max(ix0+1,min(w,int(math.ceil(x1*w))))
                iy1=max(iy0+1,min(h,int(math.ceil(y1*h))))
                bounds.append((iy0,iy1,ix0,ix1))
            self.cache[key]=(geometry,bounds);self.stats['bounds_builds']+=1
            if len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(key);owner,bounds=self.cache[key]
        assert owner is geometry
        values=[t.new_zeros(()) if b is None else t[b[0]:b[1],b[2]:b[3]].mean() for b in bounds]
        self.stats['maps']+=1
        return torch.stack(values)
