from evaluation.paths import project_path
import argparse
from collections import OrderedDict
import os
import socket
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import ROOT, WORK, claim, key, read, save, sha, signature


class CachedModule(torch.nn.Module):
    def __init__(self, module, limit):
        super().__init__()
        self.module, self.limit = module, limit
        self.values = OrderedDict()
        self.used = 0

    def forward(self, image):
        token = image._cache_id
        if token in self.values:
            value, size = self.values.pop(token)
            self.values[token] = (value, size)
            return self.copy_container(value)
        value = self.module(image)
        tensors = value.values() if isinstance(value, dict) else value
        size = sum(t.numel() * t.element_size() for t in tensors)
        while self.values and self.used + size > self.limit:
            _, (_, old) = self.values.popitem(last=False)
            self.used -= old
        if size <= self.limit:
            self.values[token] = (value, size)
            self.used += size
        return self.copy_container(value)

    @staticmethod
    def copy_container(value):
        # RoMa's matching head replaces the last element of its descriptor list.
        if isinstance(value,list):
            return list(value)
        if isinstance(value,dict):
            return dict(value)
        return value


class Matcher:
    def __init__(self, resolution=512, amp=False):
        sys.path.insert(0, str(ROOT / 'weights/romav2/source/src'))
        sys.path.insert(0, str(WORK / 'intermediate/runtime'))
        os.environ['TORCH_HOME'] = str(ROOT / 'weights/romav2/torchhub')
        from romav2 import RoMaV2
        from romav2.features import Descriptor
        from romav2.matcher import Matcher as DenseMatcher
        torch.set_float32_matmul_precision('highest')
        original = torch.autocast

        class CompatibleAutocast(original):
            def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
                if device_type == 'cuda' and dtype == torch.bfloat16:
                    dtype, enabled = torch.float16, enabled and amp
                super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

        torch.autocast = CompatibleAutocast
        cfg = RoMaV2.Cfg(setting='fast', descriptor=Descriptor.Cfg(enable_amp=amp),
                        matcher=DenseMatcher.Cfg(enable_amp=amp))
        self.model = RoMaV2(cfg)
        self.model.H_lr = self.model.W_lr = resolution
        self.model.bidirectional = True
        self.model.requires_grad_(False)
        self.model.f = CachedModule(self.model.f, 512 << 20)
        self.model.refiner_features = CachedModule(self.model.refiner_features, 1024 << 20)
        self.images = OrderedDict()
        self.resolution = resolution
        self.precision = 'fp16_autocast' if amp else 'fp32'
        from colpali_engine.models import ColQwen2_5_Processor
        self.processor = ColQwen2_5_Processor.from_pretrained(ROOT / 'weights/colnomic7b/adapter', local_files_only=True)
        self.grids = {}

    def image(self, record):
        token = record['sha256']
        if token in self.images:
            im = self.images.pop(token)
            self.images[token] = im
            return im
        assert sha(record['path']) == token
        with Image.open(project_path(record['path'])) as source:
            rgb = source.convert('RGB')
            width, height = rgb.size
            array = np.array(rgb)
        from .token_layout import token_layout
        grid = token_layout(self.processor, 'colnomic7b', width, height)
        self.grids[token] = (grid['height'], grid['width'])
        im = torch.from_numpy(array).permute(2, 0, 1).to('cuda').float()[None] / 255.
        im = F.interpolate(im, size=(self.resolution, self.resolution), mode='bicubic', align_corners=False, antialias=True)
        im._cache_id = token
        self.images[token] = im
        while len(self.images) > 32:
            self.images.popitem(last=False)
        return im

    @torch.inference_mode()
    def pair(self, photo, reference):
        a, b = self.image(photo), self.image(reference)
        predictions = self.model(a, b)
        vectors = []
        for direction, record in [('AB', photo), ('BA', reference)]:
            confidence = predictions['confidence_' + direction][..., :1].sigmoid()
            if not torch.isfinite(confidence).all():
                raise FloatingPointError('Non-finite RoMa support')
            grid = self.grids[record['sha256']]
            support = F.interpolate(confidence.permute(0, 3, 1, 2).float(), size=grid, mode='area').flatten()
            vectors.append(support.cpu())
        u, v = vectors
        return dict(M=float((u.mean() * v.mean()).sqrt()), u=u.half(), v=v.half())


def audit_cache(precision='fp32'):
    manifest=read(WORK/'medicine/manifest.json')
    labels=read(WORK/'medicine/labels.json')
    rows=[r for r in manifest['rows'] if r['split']=='train'][:3]
    pairs=[]
    for row in rows:
        positive=labels[row['id']][0]
        negative=next(r for r in row['candidates'] if r not in labels[row['id']])
        pairs.extend([(row,manifest['references'][r],r==positive) for r in (positive,negative)])
    matcher=Matcher(amp=precision=='fp16')
    cached_f,cached_vgg=matcher.model.f,matcher.model.refiner_features
    matcher.model.f=cached_f.module
    matcher.model.refiner_features=cached_vgg.module
    started=time.perf_counter()
    native=[matcher.pair(a,b) for a,b,_ in pairs]
    uncached_seconds=time.perf_counter()-started
    for (a,b,_),expected in zip(pairs,native,strict=True):
        result=matcher.model.match(str(project_path(a['path'])),str(project_path(b['path'])))
        vectors=[]
        for name,record in (('AB',a),('BA',b)):
            vectors.append(F.interpolate(result['overlap_'+name].permute(0,3,1,2).float(),
                                         size=matcher.grids[record['sha256']],mode='area').flatten().cpu())
        u,v=vectors
        assert abs(float((u.mean()*v.mean()).sqrt())-expected['M'])<1e-6
        assert torch.equal(u.half(),expected['u']) and torch.equal(v.half(),expected['v'])
    matcher.model.f,matcher.model.refiner_features=cached_f,cached_vgg
    started=time.perf_counter()
    first=[matcher.pair(a,b) for a,b,_ in pairs]
    for a,b,_ in reversed(pairs):
        matcher.pair(a,b)
    repeated=[matcher.pair(a,b) for a,b,_ in pairs]
    cached_seconds=time.perf_counter()-started
    records=[]
    for (a,b,positive),expected,initial,actual in zip(pairs,native,first,repeated,strict=True):
        error=max(abs(expected['M']-actual['M']),abs(expected['M']-initial['M']))
        uv=max(float((expected[k].float()-value[k].float()).abs().max())
               for value in (initial,actual) for k in ('u','v'))
        assert error<1e-6 and uv<1e-6,(a['id'],b['id'],error,uv)
        records.append(dict(query=a['id'],reference=b['id'],positive=positive,M=expected['M'],
                            mass_error=error,support_error=uv))
    result=dict(status='PASS',host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),
                recipe=precision+' 512 bidirectional; copied descriptor containers',
                uncached_seconds_per_pair=uncached_seconds/len(pairs),cached_seconds_per_pair=cached_seconds/(3*len(pairs)),
                checked='Official match API, uncached forward, first cache use, and cache reuse after intervening pairs',
                records=records)
    save(WORK/'intermediate/audits'/f'roma_cache_{socket.gethostname()}_{precision}.json',result)
    print(result,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['medicine', 'grozi', 'milk'])
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test', 'transfer'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.audit:
        audit_cache(args.precision)
        return
    host = socket.gethostname()
    worker = host + '_' + os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',', '-')
    state_path = WORK / 'intermediate/workers' / (worker + '.json')
    model = None
    completed = 0
    def passes():
        while True:
            yield from args.datasets
            if not args.watch:
                return
            pending = any(not (WORK/d/f'roma{args.resolution}'/(key(r)+'.pt')).exists()
                          for d in args.datasets for r in read(WORK/d/'manifest.json')['rows']
                          if r['split'] in args.splits)
            if not pending:
                return
            time.sleep(10)
    for dataset in passes():
        manifest = read(WORK / dataset / 'manifest.json')
        recipe = dict(manifest=manifest['signature'], resolution=args.resolution,
                      model='RoMaV2-v2.0.1', revision='95c9968145c8906b7b59383258e9f73b02853d89',
                      bidirectional=True, threshold=None, high_resolution=False,
                      pooling='area to native ColNomic image grid; sqrt(mean(u)*mean(v))',
                      descriptor_cache='immutable cached features; copy list/dict containers before each consumer v2',
                      numeric=args.precision+'; TF32 disabled; finite checked; confidence pooling in FP32')
        sig = signature(recipe)
        for split in args.splits:
            for row in manifest['rows']:
                if row['split'] != split:
                    continue
                path = WORK / dataset / f'roma{args.resolution}' / (key(row) + '.pt')
                if path.exists():
                    continue
                parent = {}
                if manifest.get('roma_parent'):
                    parent_path = WORK/manifest['roma_parent']/f'roma{args.resolution}'/(key(row)+'.pt')
                    if not parent_path.exists():
                        continue
                    source = torch.load(parent_path,weights_only=True)
                    parent = dict(zip(source['candidates'],source['pairs']))
                with claim(path.with_suffix('.lock')) as acquired:
                    if not acquired or path.exists():
                        continue
                    if model is None and not set(row['candidates']) <= parent.keys():
                        model = Matcher(args.resolution,amp=args.precision=='fp16')
                    partial = path.with_suffix('.partial.pt')
                    old = torch.load(partial, weights_only=True) if partial.exists() else {}
                    if old:
                        assert old['signature'] == sig and old['id'] == row['id']
                    pairs = old.get('pairs', [])
                    seconds = old.get('seconds', 0.)
                    for i in range(len(pairs), len(row['candidates'])):
                        save(state_path, dict(host=host, pid=os.getpid(), dataset=dataset, split=split,
                                              query=row['id'], candidate=i, candidates=len(row['candidates']),
                                              complete=completed, updated=time.time(), status='running'))
                        start = time.monotonic()
                        rid = row['candidates'][i]
                        pairs.append(parent[rid] if rid in parent else model.pair(row, manifest['references'][rid]))
                        seconds += time.monotonic()-start
                        if (i+1) % 8 == 0 or i+1 == len(row['candidates']):
                            save(partial, dict(id=row['id'], signature=sig, recipe=recipe, candidates=row['candidates'],
                                               pairs=pairs, seconds=seconds, host=host, precision=args.precision,
                                               reused_pairs=len(set(row['candidates']) & parent.keys())))
                    os.replace(partial, path)
                    completed += 1
                    print(dataset, split, row['query_index'], 'pairs', len(pairs), 'seconds', round(seconds, 2), flush=True)
                    if args.max_queries and completed >= args.max_queries:
                        save(state_path, dict(host=host, pid=os.getpid(), status='limit_complete', complete=completed, updated=time.time()))
                        return
    save(state_path, dict(host=host, pid=os.getpid(), status='complete', complete=completed, updated=time.time()))


if __name__ == '__main__':
    main()
