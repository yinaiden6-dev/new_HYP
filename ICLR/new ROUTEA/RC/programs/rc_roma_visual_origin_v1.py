"""Deterministic RoMa-input interventions and non-mutating stage observation."""
from collections import Counter
import torch
from torch.nn import functional as F
from rc_roma_exact_feature_cache_v2 import clone_tree, equal_tree

ARMS = ('NATIVE', 'Q_GRAY', 'R_GRAY', 'Q_LOWPASS', 'R_LOWPASS', 'Q_SHUFFLE', 'R_SHUFFLE')
STAGES = ('COARSE', 'LR4', 'LR2', 'LR1', 'HR4', 'HR2', 'HR1')


def tags(arm):
    assert arm in ARMS
    if arm == 'NATIVE': return 'NATIVE', 'NATIVE'
    side, transform = arm.split('_')
    return (transform, 'NATIVE') if side == 'Q' else ('NATIVE', transform)


def transform(x, name):
    assert x.ndim == 4 and x.shape[1] == 3
    if name == 'NATIVE': return x
    if name == 'GRAY': return x.mean(1, keepdim=True).expand_as(x).contiguous()
    if name == 'LOWPASS':
        assert x.shape[-2] % 16 == x.shape[-1] % 16 == 0
        return F.interpolate(F.avg_pool2d(x, 16, 16), size=x.shape[-2:], mode='bilinear', align_corners=False)
    assert name == 'SHUFFLE'
    n, c, h, w = x.shape; assert h % 4 == w % 4 == 0
    order = torch.randperm(16, generator=torch.Generator().manual_seed(20260922)).to(x.device)
    blocks = x.reshape(n, c, 4, h//4, 4, w//4).permute(0, 2, 4, 1, 3, 5).reshape(n, 16, c, h//4, w//4)
    return blocks[:, order].reshape(n, 4, 4, c, h//4, w//4).permute(0, 3, 1, 4, 2, 5).reshape_as(x)


class Observer:
    def __init__(self, model, core, input_sink):
        self.model, self.core, self.input_sink = model, core, input_sink
        self.arm = 'NATIVE'; self.geometry = None; self.identity = None
        self.query_cache, self.reference_cache = {}, {}; self.reference_key = None
        self.calls, self.stats, self.stages = Counter(), Counter(), {}
        self.original_forward = model.forward
        self.original_modules = {n: getattr(model, n).forward for n in ('f', 'refiner_features')}
        model.forward = self.forward
        for name in self.original_modules: getattr(model, name).forward = self.memo(name)
        self.handles = [model.matcher.register_forward_hook(self.coarse)]
        for name, module in model.refiners.items():
            self.handles.append(module.register_forward_hook(self.refiner(name)))

    def select(self, arm, qgeom, rgeom, query_sha, reference_sha):
        if reference_sha != self.reference_key:
            self.reference_cache.clear(); self.reference_key = reference_sha
        self.arm, self.geometry, self.identity = arm, (qgeom, rgeom), (query_sha, reference_sha)
        self.calls.clear(); self.stages = {}

    def memo(self, name):
        def call(*args, **kwargs):
            count = self.calls[name]; self.calls[name] += 1
            assert count < (2 if name == 'f' else 4)
            side = count % 2; tag = tags(self.arm)[side]
            resolution = 'COARSE' if name == 'f' else 'LR' if count < 2 else 'HR'
            cache = self.query_cache if side == 0 else self.reference_cache
            key = (name, resolution, tag)
            if key not in cache:
                inputs = clone_tree((args, kwargs))
                value = self.original_modules[name](*args, **kwargs)
                cache[key] = (inputs, clone_tree(value)); self.stats[name+'_miss'] += 1
            else:
                assert equal_tree(cache[key][0], (args, kwargs)), ('FEATURE_INPUT_DRIFT', key)
                self.stats[name+'_hit'] += 1
            return clone_tree(cache[key][1])
        return call

    @torch.inference_mode()
    def forward(self, img_A_lr, img_B_lr, img_A_hr=None, img_B_hr=None):
        qtag, rtag = tags(self.arm)
        values = (transform(img_A_lr, qtag), transform(img_B_lr, rtag),
                  transform(img_A_hr, qtag), transform(img_B_hr, rtag))
        for side, tag in enumerate((qtag, rtag)):
            self.input_sink(self.identity[side], tag, values[side], values[side+2])
        result = self.original_forward(values[0], values[1], img_A_hr=values[2], img_B_hr=values[3])
        assert self.calls['f'] == 2 and self.calls['refiner_features'] == 4
        assert set(self.stages) == set(STAGES)
        assert all(set(v) == {'AB', 'BA'} for v in self.stages.values())
        return result

    def capture(self, stage, side, confidence, warp):
        # Clone values through sigmoid before RoMa's later in-place precision reset.
        overlap = confidence[0, ..., 0].sigmoid().detach().cpu()
        weights = self.core.cell_means(overlap, self.geometry[side])
        assert weights.dtype == torch.float64 and bool(torch.isfinite(weights).all())
        assert bool(((weights >= 0) & (weights <= 1)).all())
        grid = F.interpolate(overlap[None, None], size=(16,16), mode='area')[0,0]
        coordinates = F.interpolate(warp.detach().permute(0,3,1,2), size=(16,16), mode='bilinear', align_corners=False)[0].permute(1,2,0).cpu()
        self.stages.setdefault(stage, {})['AB' if side == 0 else 'BA'] = dict(
            weights=weights, overlap_grid16=grid, warp_grid16=coordinates,
            native_overlap_shape=list(overlap.shape))

    def coarse(self, module, args, output):
        for side, direction in enumerate(('AB','BA')):
            self.capture('COARSE', side, output['confidence_'+direction], output['warp_'+direction])

    def refiner(self, stride):
        def callback(module, args, output):
            key = 'refiner'+stride; count = self.calls[key]; self.calls[key] += 1
            assert count < 4
            self.capture(('LR' if count < 2 else 'HR')+stride, count % 2, output['confidence'], output['warp'])
        return callback

    def close(self):
        self.model.forward = self.original_forward
        for name, forward in self.original_modules.items(): getattr(self.model, name).forward = forward
        for handle in self.handles: handle.remove()
        self.query_cache.clear(); self.reference_cache.clear()


def self_test():
    x = torch.arange(3*32*48, dtype=torch.float32).reshape(1,3,32,48)/5000
    assert transform(x, 'NATIVE') is x
    g = transform(x, 'GRAY'); assert torch.equal(g[:,0], g[:,1]) and torch.equal(g[:,1], g[:,2])
    s = transform(x, 'SHUFFLE'); assert not torch.equal(s, x)
    assert torch.equal(s.flatten().sort().values, x.flatten().sort().values)
    assert torch.equal(s, transform(x, 'SHUFFLE'))
    c = torch.full_like(x, .25); assert torch.equal(transform(c, 'LOWPASS'), c)
    assert transform(x, 'LOWPASS').shape == x.shape
    assert all(tags(a)[0] == 'NATIVE' or tags(a)[1] == 'NATIVE' for a in ARMS)
    return dict(status='VISUAL_ORIGIN_TRANSFORMS_PASS', arms=list(ARMS), stages=list(STAGES),
                identity_exact=True, shuffle_histogram_exact=True, constant_lowpass_exact=True)
