"""Inference-only, bounded per-query/per-pair memoization of frozen RoMa modules.

RoMa.match, forward, interpolation, refiner hooks and scoring stay untouched.
Every replay returns fresh tensor storage, including nested matcher confidence.
"""
from collections import Counter
import torch


def clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_tree(v) for v in x)
    return x


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def tensor_bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(map(tensor_bytes, x))
    return 0


class ExactFeatureCache:
    def __init__(self, model):
        self.model = model
        self.enabled = True
        self.query_image = self.reference_image = None
        self.query = {}
        self.reference = {}
        self.pair = {}
        self.calls = Counter()
        self.stats = Counter()
        self.peak_cached_bytes = 0
        self.original_match = model.match
        self.original_forwards = {}
        for name in ('f', 'refiner_features', 'matcher'):
            module = getattr(model, name)
            self.original_forwards[name] = module.forward
            module.forward = self._wrapper(name)
        model.match = self.match

    def _wrapper(self, name):
        def forward(*args, **kwargs):
            if not self.enabled:
                return self.original_forwards[name](*args, **kwargs)
            n = self.calls[name]
            self.calls[name] += 1
            if name == 'f':
                assert n < 2, ('DESCRIPTOR_CALL_ORDER', n)
                bank = self.query if n == 0 else self.reference
                key = 'coarse'
            elif name == 'refiner_features':
                assert n < 4, ('FINE_CALL_ORDER', n)
                bank = self.query if n % 2 == 0 else self.reference
                key = 'fine_lr' if n < 2 else 'fine_hr'
            else:
                assert n == 0, ('MATCHER_CALL_ORDER', n)
                bank, key = self.pair, 'matcher'
            if key in bank:
                saved = bank[key]
                assert equal_tree((args, kwargs), saved['inputs']), ('CACHE_INPUT_DRIFT', name, key)
                self.stats[name + '_hits'] += 1
            else:
                # Store inputs before calling the original module. Retain no
                # reference to downstream-mutable output storage.
                inputs = clone_tree((args, kwargs))
                output = self.original_forwards[name](*args, **kwargs)
                bank[key] = dict(inputs=inputs, output=clone_tree(output))
                self.stats[name + '_misses'] += 1
                self.peak_cached_bytes = max(self.peak_cached_bytes, tensor_bytes((self.query, self.reference, self.pair)))
            return clone_tree(bank[key]['output'])
        return forward

    @torch.inference_mode()
    def match(self, image_a, image_b):
        if not self.enabled:
            return self.original_match(image_a, image_b)
        # Hold the actual input objects so Python cannot reuse their identity.
        # Repeated module inputs are additionally checked bit for bit above.
        if image_a is not self.query_image:
            self.query.clear()
            self.reference.clear()
            self.pair.clear()
            self.query_image = image_a
            self.reference_image = None
        if image_b is not self.reference_image:
            self.reference.clear()
            self.pair.clear()
            self.reference_image = image_b
        self.calls.clear()
        result = self.original_match(image_a, image_b)
        assert self.calls == Counter(f=2, refiner_features=4, matcher=1), ('INCOMPLETE_PRECISE_FORWARD', self.calls)
        self.stats['cached_match_calls'] += 1
        return result

    def summary(self):
        return dict(counts=dict(self.stats), peak_cached_tensor_bytes=self.peak_cached_bytes,
                    scope='Query descriptors retained; current reference and pair only; fresh output clones on every use')

    def close(self):
        self.model.match = self.original_match
        for name, forward in self.original_forwards.items():
            getattr(self.model, name).forward = forward
        self.query.clear()
        self.reference.clear()
        self.pair.clear()


@torch.inference_mode()
def self_test():
    class Descriptor(torch.nn.Module):
        def forward(self, x):
            return [x * 2, x + 1]
    class Fine(torch.nn.Module):
        def forward(self, x):
            return {1: x + 3, 2: x * 4, 4: x - 1}
    class Matcher(torch.nn.Module):
        def forward(self, a, b, **kwargs):
            return dict(warp=a[0] + b[0], confidence=a[1] - b[1])
    class Model:
        def __init__(self):
            self.f, self.refiner_features, self.matcher = Descriptor(), Fine(), Matcher()
        def match(self, a, b):
            m = self.matcher(self.f(a), self.f(b), bidirectional=True)
            # Reproduce downstream mutation hazard without altering inputs.
            m['confidence'].add_(7)
            fine = [self.refiner_features(x) for x in (a, b, a + 10, b + 10)]
            return dict(matcher=m, fine=fine)
    model = Model()
    a, b, c = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + i for i in range(3))
    expected_b, expected_c = model.match(a, b), model.match(a, c)
    cache = ExactFeatureCache(model)
    for ref, expected in ((b, expected_b), (c, expected_c)):
        for _ in range(4):
            result = model.match(a, ref)
            assert equal_tree(result, expected)
            result['matcher']['confidence'].zero_()
            result['fine'][0][1].zero_()
    assert cache.stats['f_misses'] == 3 and cache.stats['refiner_features_misses'] == 6 and cache.stats['matcher_misses'] == 2
    caught = False
    a.add_(1)
    try:
        model.match(a, c)
    except AssertionError as error:
        caught = 'CACHE_INPUT_DRIFT' in str(error)
    assert caught
    cache.close()
    assert equal_tree(model.match(a, c), Model().match(a, c))
    return dict(status='ROMA_EXACT_CACHE_SYNTHETIC_PASS',mutation_isolation=True,
                query_reuse_across_references=True,input_drift_rejected=True)
