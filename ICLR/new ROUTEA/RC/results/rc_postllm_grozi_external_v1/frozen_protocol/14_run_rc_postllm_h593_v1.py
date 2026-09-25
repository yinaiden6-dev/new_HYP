#!/usr/bin/env python3
"""Fresh five-fold POST-LLM M training on the sealed H593 natural C128.

The frozen projector and sparse full-C128 VJP are inherited from the qualified
24-image pilot. This runner never loads a language model or RoMa. It does not
reuse any pilot adapter/head and never chooses a checkpoint on held outcomes.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import run_rc_internal_m_condition_scale_v4 as V

ROOT = V.ROOT
DEFAULT_AUTH = ROOT / 'registry/rc_postllm_h593_authority_v1_20260924.json'
DEFAULT_OUT = ROOT / 'results/rc_postllm_h593_v1'
ARMS = ('POST_REAL', 'POST_CONSTANT', 'POST_SHUFFLED')
read, write, bind, checked, need, save, emit = V.read, V.write, V.bind, V.checked, V.need, V.save, V.emit
STOP = False


def interrupted(signum, frame):
    global STOP
    STOP = True


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def setup(seed, threads):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for s in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(s, interrupted)


def rng_state():
    import numpy as np
    import torch
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def nbytes(value):
    import torch
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(map(nbytes, value))
    return 0


class LRU:
    def __init__(self, maximum):
        self.maximum = maximum
        self.items = OrderedDict()
        self.used = 0
        self.hits = self.misses = 0

    def get(self, key, factory):
        if key in self.items:
            self.hits += 1
            value, size = self.items.pop(key)
            self.items[key] = (value, size)
            return value
        self.misses += 1
        value = factory()
        size = nbytes(value)
        if size <= self.maximum:
            while self.items and self.used + size > self.maximum:
                _, (_, oldsize) = self.items.popitem(last=False)
                self.used -= oldsize
            self.items[key] = (value, size)
            self.used += size
        return value


class Inputs:
    """Bounded CPU query/reference banks and normalized GPU reference LRU."""
    def __init__(self, authority, device, cpu_gb=8., gpu_gb=3.):
        self.authority = authority
        self.device = device
        self.queries = LRU(int(cpu_gb * (1 << 29)))
        self.references = LRU(int(cpu_gb * (1 << 29)))
        self.device_references = LRU(int(gpu_gb * (1 << 30)))
        self.verified = {}
        self.query_sources = {}

    def verified_path(self, binding):
        path = str(Path(binding['path']).resolve())
        key = (path, binding['sha256'])
        stat = Path(path).stat()
        current = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in self.verified:
            checked(binding)
            self.verified[key] = current
        need(self.verified[key] == current, 'IMMUTABLE_CACHE_CHANGED')
        return Path(path)

    def cache_cpu(self, row):
        import torch
        q = row['query_id']
        def load():
            seal_binding = row.get('hidden_validation')
            path = self.verified_path(seal_binding) if seal_binding else Path(row['hidden_cache_dir']) / 'validation.json'
            seal = read(path)
            need('PASS' in seal['status'], 'HIDDEN_CACHE_NOT_QUALIFIED')
            if not seal_binding:
                need(seal['authority'] == self.authority, 'NEW_HIDDEN_AUTHORITY')
            if 'candidate_ids' in seal:
                need(seal['candidate_ids'] == row['candidate_ids'], 'HIDDEN_CONTENT_CANDIDATE_AXIS')
            if 'candidate_axis_sha256' in seal:
                need(seal['candidate_axis_sha256'] == object_sha(row['candidate_ids']), 'HIDDEN_CONTENT_CANDIDATE_AXIS_SHA')
            payload = seal['payload']
            cache = torch.load(self.verified_path(payload), map_location='cpu', weights_only=True)
            need(cache.get('query_id', q) == q, 'HIDDEN_QUERY_ID')
            need(cache['hidden'].dtype == torch.bfloat16 and not cache['hidden'].requires_grad,
                 'BF16_FROZEN_HIDDEN_REQUIRED')
            need(cache['hidden'].shape[:2] == cache['image_mask'].shape, 'HIDDEN_MASK_AXIS')
            need(cache['hidden'].shape[-1] == 3584 and cache['image_mask'].dtype == torch.bool,
                 'HIDDEN_SHAPE')
            need('fresh_L0' in cache and len(cache['fresh_L0']) == 128, 'QUALIFIED_FRESH_FULL128_L0')
            self.query_sources[q] = dict(validation=bind(path), payload=payload,
                                        candidate_axis_sha256=object_sha(row['candidate_ids']))
            # Pixel/vision intermediates are not needed for post-LLM replay.
            return dict(query_id=q, hidden=cache['hidden'], image_mask=cache['image_mask'],
                        batch={'attention_mask': cache['batch']['attention_mask']},
                        native_tokens=cache['native_tokens'], fresh_L0=torch.as_tensor(cache['fresh_L0'], dtype=torch.float64))
        return self.queries.get(q, load)

    def cache(self, row):
        import torch
        src = self.cache_cpu(row)
        def move(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            return {k: move(v) for k, v in x.items()} if isinstance(x, dict) else x
        return move(src)

    def refs(self, row):
        import torch
        from torch.nn import functional as F
        result = []
        need(len(row['reference_tokens']) == 128, 'NATURAL_C128_REFERENCES')
        for item in row['reference_tokens']:
            b = item.get('token_file', item)
            key = (str(Path(b['path']).resolve()), b['sha256'])
            def load(b=b):
                x = torch.load(self.verified_path(b), map_location='cpu', weights_only=True)['tokens']
                need(x.ndim == 2 and x.shape[1] == 128 and torch.isfinite(x).all(), 'REFERENCE_TOKEN_SHAPE')
                return x
            def normalize(key=key, load=load):
                return F.normalize(self.references.get(key, load).to(self.device, dtype=torch.float64), dim=-1)
            result.append(self.device_references.get(key, normalize))
        return result

    def stats(self):
        return {k: dict(hits=v.hits, misses=v.misses, bytes=v.used, entries=len(v.items))
                for k, v in [('query_cpu', self.queries), ('reference_cpu', self.references),
                             ('reference_device', self.device_references)]}


def experiment(authority_path, out, fold):
    a = read(authority_path)
    ab = bind(authority_path)
    for b in a.get('code_sources', []):
        checked(b)
    manifest = read(checked(a['manifest']))
    fd = manifest['folds'][str(fold)]
    rows = {r['query_id']: r for r in manifest['rows']}
    need(len(rows) == len(manifest['rows']) == 593, 'EXACT_H593')
    train_ids, held_ids = fd['train_query_ids'], fd['held_query_ids']
    need(not set(train_ids) & set(held_ids), 'DISJOINT_FOLD_QUERY')
    label_binding = fd.get('train_labels_path', fd.get('train_labels'))
    records = read(checked(label_binding))['records']
    labels = records if isinstance(records, dict) else {r['query_id']: r for r in records}
    need(set(train_ids) <= set(labels) <= set(fd['train_all_query_ids']), 'TRAIN_ONLY_LABEL_SCOPE')
    train = []
    for q in train_ids:
        r = copy.deepcopy(rows[q]); r.update(labels[q])
        need(len(r['target_positions']) == 1, 'INPOOL_TRAIN_ONLY')
        need(r['candidate_identities'][r['target_positions'][0]] == r['target_id'], 'TRAIN_LABEL_AXIS')
        r['raw_correct'] = r['candidate_identities'][r['winner_index']] == r['target_id']
        train.append(r)
    for r in rows.values():
        need(len(r['candidate_ids']) == len(r['M']) == len(r['raw_scores']) == 128, 'FULL_C128_AXIS')
        need(r['challenger_positions'] == [i for i in range(128) if i != r['winner_index']], 'PHYSICAL_CHALLENGER_AXIS')
        need(not any(k in r for k in ('target_id', 'target_positions', 'raw_correct')), 'LABEL_FREE_MANIFEST_ROWS')
    config = dict(V.CONFIG)
    config.update(a.get('training', {}))
    for key in V.CONFIG:
        if key in a:
            config[key] = a[key]
    config['passes'] = int(a.get('training', {}).get('passes', a.get('passes', a.get('epochs', 8))))
    config['updates'] = config['passes'] * len(train)
    need(config['passes'] == 8 and config['seed'] == 17, 'FROZEN_EIGHT_PASSES_SEED17')
    normalization = fd['mass_normalization']
    import numpy as np
    logs = np.log(np.maximum(np.asarray([r['M'] for r in train]), normalization['epsilon']))
    need(abs(float(logs.mean()) - normalization['log_mean']) < 1e-10, 'TRAIN_ONLY_MASS_MEAN')
    need(abs(float(logs.std()) - normalization['log_std']) < 1e-10, 'TRAIN_ONLY_MASS_STD')
    d = Path(out) / f'fold{fold}'
    d.mkdir(parents=True, exist_ok=True)
    return dict(a=a, authority=ab, manifest=manifest, fold=fold, fold_data=fd, train=train,
                held=[rows[q] for q in held_ids], config=config, normalization=normalization, out=d,
                input_scope=dict(manifest=a['manifest'], labels=label_binding,
                    train_axis_sha256=object_sha(train_ids), held_axis_sha256=object_sha(held_ids)))


def adapter(ctx, arm, device):
    import torch
    from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
    c, n = ctx['config'], ctx['normalization']
    torch.manual_seed(c['seed'])
    small = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=c['bottleneck'],
        conditioning='constant' if arm == 'POST_CONSTANT' else 'real', condition_gain=c['condition_gain'],
        mass_log_mean=n['log_mean'], mass_log_std=n['log_std'], mass_epsilon=n['epsilon'],
        residual_scale=c['residual_scale'])
    need(not bool(small.up.weight.any()) and not bool(small.up.bias.any()), 'ZERO_OUTPUT_INITIALIZATION')
    return small.to(device)


def row_condition(row, arm, intervention='native'):
    need(arm in ARMS and intervention in ('native', 'constant', 'shuffled'), 'ARM_INTERVENTION')
    masses = row['M']
    if arm == 'POST_SHUFFLED' or intervention == 'shuffled':
        masses = masses[1:] + masses[:1]
    return masses, arm == 'POST_CONSTANT' or intervention == 'constant'


def batched_cost(rows, logits):
    """Vectorized full-C128 COST1; amax retains tie-split gradients."""
    import torch
    from torch.nn import functional as F
    onwinner = torch.tensor([r['target_positions'][0] == r['winner_index'] for r in rows],
                            dtype=torch.bool, device=logits.device)
    target = torch.tensor([r['challenger_positions'].index(r['target_positions'][0]) if not w else 0
                           for r, w in zip(rows, onwinner.tolist())], device=logits.device)
    positive = logits.gather(1, target[:, None]).squeeze(1)
    wrong = logits.scatter(1, target[:, None], -torch.inf)
    costs = torch.where(onwinner, F.softplus(logits.amax(dim=1)),
                        F.softplus(-positive) + F.softplus(wrong.amax(dim=1)))
    return costs.mean()


def verify_batched_cost(rows, xs):
    import torch
    gen = torch.Generator().manual_seed(113)
    errors = []
    subset = rows[:min(12, len(rows))]
    xx = xs[:len(subset)]
    for zero in (False, True):
        theta = torch.zeros(xs.shape[-1], dtype=torch.float64) if zero else torch.randn(xs.shape[-1], generator=gen, dtype=torch.float64)
        t1 = theta.clone().requires_grad_(True); t2 = theta.clone().requires_grad_(True)
        original = torch.stack([V.cost_from_logits(r, x @ t1) for r, x in zip(subset, xx)]).mean()
        new = batched_cost(subset, xx @ t2)
        g1, = torch.autograd.grad(original, t1); g2, = torch.autograd.grad(new, t2)
        e = dict(loss_error=abs(float((original-new).detach())), gradient_error=float((g1-g2).abs().max()))
        need(e['loss_error'] < 1e-12 and e['gradient_error'] < 1e-12, 'BATCHED_HEAD_LOSS_GRADIENT_PARITY')
        errors.append(e)
    return errors


def warm(ctx, inputs, budget):
    import torch
    started = time.monotonic()
    d = ctx['out'] / 'warm'
    d.mkdir(exist_ok=True)
    lock = (d / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seal = d / 'validation.json'
    if seal.exists():
        old = read(seal)
        need(old['authority'] == ctx['authority'] and old['input_scope'] == ctx['input_scope'], 'WARM_RESUME_SCOPE')
        for b in old['heads'].values():
            checked(b)
        return True
    rows, c = ctx['train'], ctx['config']
    contents = {r['query_id']: inputs.cache_cpu(r)['fresh_L0'].to(dtype=torch.float64) for r in rows}
    sources = inputs.query_sources.copy()
    heads = {}
    for kind in ('INTERNAL3', 'ADDITIVE4', 'PRODUCT5'):
        p = d / f'{kind}.json'
        if p.exists():
            old = read(p)
            need(old['authority'] == ctx['authority'] and old['input_scope'] == ctx['input_scope'], 'WARM_HEAD_SCOPE')
            heads[kind] = bind(p); continue
        if STOP or time.monotonic()-started > budget-10:
            return False
        initial = [0., 0., 0.] if kind == 'INTERNAL3' else V.map_warm_head(read(d / 'INTERNAL3.json')['theta'], kind)
        theta = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
        xs = torch.stack([V.features(r, contents[r['query_id']], kind) for r in rows])
        parity = verify_batched_cost(rows, xs)
        opt = torch.optim.AdamW([theta], lr=c['head_lr'], weight_decay=c['weight_decay'])
        steps = c['warmstart_steps'] if kind == 'INTERNAL3' else c['external_steps']
        cp = d / f'{kind}.partial.pt'; start = 0; history = []
        if cp.exists():
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            need(saved['authority'] == ctx['authority'] and saved['input_scope'] == ctx['input_scope'] and saved['kind'] == kind,
                 'WARM_CHECKPOINT_SCOPE')
            theta.data.copy_(saved['theta']); opt.load_state_dict(saved['optimizer'])
            start, history = saved['step'], saved['history']; restore_rng(saved['rng'])
        def checkpoint(step):
            save(cp, dict(authority=ctx['authority'], input_scope=ctx['input_scope'], kind=kind,
                theta=theta.detach().clone(), optimizer=opt.state_dict(), step=step, history=history, rng=rng_state()))
        for step in range(start, steps):
            if STOP or time.monotonic()-started > budget-5:
                checkpoint(step); return False
            opt.zero_grad(set_to_none=True)
            value = batched_cost(rows, xs @ theta)
            value.backward()
            need(bool(torch.isfinite(theta.grad).all()), 'FINITE_WARM_GRADIENT')
            norm = float(torch.nn.utils.clip_grad_norm_([theta], c['clip_norm']))
            opt.step()
            if (step+1) % 100 == 0 or step == 0 or step+1 == steps:
                history.append(dict(step=step+1, loss=float(value.detach()), gradient_norm=norm))
                checkpoint(step+1)
        predictions = {r['query_id']: V.choose(r, contents[r['query_id']], theta.detach(), kind) for r in rows}
        write(p, dict(status='FOLD_TRAIN_ONLY_HEAD_FIT_COMPLETE', authority=ctx['authority'], input_scope=ctx['input_scope'],
            kind=kind, theta=theta.detach().tolist(), initial_theta=initial, steps=steps, history=history,
            feature_order=V.FEATURES[kind], predictions=predictions, summary=V.summary(rows, predictions),
            fit_queries=[r['query_id'] for r in rows], native_content_sources=sources,
            batch_loss_gradient_parity=parity, held_label_reads=0, old_pilot_head_reused=False))
        heads[kind] = bind(p)
        emit(stage='warm', fold=ctx['fold'], kind=kind, summary=read(p)['summary'])
    write(seal, dict(status='FOLD_FRESH_WARM_AND_EXTERNAL_HEADS_PASS', authority=ctx['authority'],
                     input_scope=ctx['input_scope'], heads=heads, held_label_reads=0))
    return True


def score_all(model, cache, small, row, refs, values, ctx, arm, intervention, started, budget, persist):
    import torch
    from rc_postllm_m_backend_v1 import predict_projection
    masses, constant = row_condition(row, arm, intervention)
    previous = small.conditioning
    small.conditioning = 'constant' if constant else 'real'
    try:
        with torch.no_grad():
            common = predict_projection(model, cache, small, masses[0]) if constant and len(values) < 128 else None
            for i in range(len(values), 128):
                if STOP or time.monotonic() - started > budget:
                    persist(values); return False
                tokens = common if constant else predict_projection(model, cache, small, masses[i])
                values.append(float(V.OLD.content_score(tokens, refs[i])))
        return True
    finally:
        small.conditioning = previous


def worker(ctx, inputs, arm, device, budget, pilot=False):
    import torch
    from rc_postllm_m_backend_v1 import load_projection, predict_projection, validate_cached_projection
    started = time.monotonic()
    c, rows = ctx['config'], ctx['train']
    d = ctx['out'] / arm
    d.mkdir(exist_ok=True)
    lock = (d / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fitseal = d / 'fit_validation.json'
    if fitseal.exists():
        old = read(fitseal)
        need(old['authority'] == ctx['authority'] and old['input_scope'] == ctx['input_scope'], 'COMPLETED_FIT_SCOPE')
        checked(old['snapshot']); return True
    warmseal = read(ctx['out'] / 'warm/validation.json')
    need(warmseal['authority'] == ctx['authority'] and warmseal['input_scope'] == ctx['input_scope'], 'WARM_REQUIRED')
    wb = warmseal['heads']['INTERNAL3']; w = read(checked(wb))
    model = load_projection(ctx['a'], device=device)
    small = adapter(ctx, arm, device)
    theta = torch.nn.Parameter(torch.tensor(w['theta'], device=device, dtype=torch.float64))
    optimizer = V.make_optimizer(small, theta, c)
    cp = d / 'checkpoint.pt'
    step = 0; pending = []; parity_records = []
    def checkpoint(values):
        V.validate_pending(values)
        save(cp, dict(authority=ctx['authority'], input_scope=ctx['input_scope'], arm=arm, step=step,
            target_updates=c['updates'], adapter=V.state_cpu(small), head=theta.detach().cpu().clone(),
            optimizer=V.cpu_optimizer_state(optimizer), rng=rng_state(), pending=list(values),
            pending_query_id=rows[step % len(rows)]['query_id'] if step < c['updates'] else None,
            warmstart=wb, parity_records=parity_records, query_sources=inputs.query_sources,
            conditioning=small.conditioning))
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        need(saved['authority'] == ctx['authority'] and saved['input_scope'] == ctx['input_scope'] and saved['arm'] == arm
             and saved['warmstart'] == wb and saved['target_updates'] == c['updates'], 'TRAIN_CHECKPOINT_SCOPE')
        small.load_state_dict(saved['adapter']); theta.data.copy_(saved['head'].to(device))
        optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        step, pending, parity_records = saved['step'], saved['pending'], saved['parity_records']
        inputs.query_sources.update(saved.get('query_sources', {}))
        need(saved['pending_query_id'] == (rows[step % len(rows)]['query_id'] if step < c['updates'] else None), 'TRAIN_ORDER_RESUME')
        need(V.tree_equal(saved['optimizer'], V.cpu_optimizer_state(optimizer)), 'EXACT_OPTIMIZER_RELOAD')
        need(saved['conditioning'] == small.conditioning, 'CONDITIONING_RESUME')
    else:
        checkpoint([])
        save(d / 'initial.pt', dict(authority=ctx['authority'], input_scope=ctx['input_scope'], arm=arm,
            adapter=V.state_cpu(small), head=theta.detach().cpu().clone(), warmstart=wb, step=0,
            pilot_checkpoint_reused=False, seed=c['seed'], normalization=ctx['normalization']))
    target = min(1, c['updates']) if pilot else c['updates']
    need(0 <= step <= c['updates'], 'VALID_RESUMED_STEP')
    while step < target:
        if STOP or time.monotonic() - started > budget - 12:
            checkpoint(pending); return False
        row = rows[step % len(rows)]
        cache = inputs.cache(row); refs = inputs.refs(row)
        if step == 0 and not pending:
            native_parity = validate_cached_projection(model, cache, atol=0.)
            need(native_parity['exact_equal'], 'GPU_NATIVE_PROJECTOR_PARITY')
            parity_records.append(native_parity)
        s = time.monotonic()
        if not score_all(model, cache, small, row, refs, pending, ctx, arm, 'native', started, budget-8, checkpoint):
            return False
        scoring_seconds = time.monotonic() - s
        if step == 0:
            error = float((torch.tensor(pending, dtype=torch.float64)-cache['fresh_L0'].cpu().double()).abs().max())
            need(error < 2e-8, 'ZERO_FULL_C128_NATIVE_L_PARITY')
            parity_records.append(dict(query_id=row['query_id'], zero_content_max_error=error))
        if STOP or time.monotonic() - started > budget - 8:
            checkpoint(pending); return False
        content = torch.tensor(pending, dtype=torch.float64, device=device, requires_grad=True)
        value, derivative, head_gradient = V.joint_derivatives(row, content, theta)
        active = derivative.nonzero().flatten().tolist()
        need(bool(torch.isfinite(derivative).all()) and bool(torch.isfinite(head_gradient).all()), 'FINITE_FULL_C128_VJP')
        masses, constant = row_condition(row, arm)
        optimizer.zero_grad(set_to_none=True); theta.grad = head_gradient.clone()
        s = time.monotonic(); replay_error = 0.
        if active and constant:
            tokens = predict_projection(model, cache, small, masses[0])
            terms = []
            for i in active:
                score = V.OLD.content_score(tokens, refs[i])
                replay_error = max(replay_error, abs(float(score.detach()) - pending[i]))
                terms.append(score * derivative[i])
            torch.stack(terms).sum().backward()
        else:
            for i in active:
                tokens = predict_projection(model, cache, small, masses[i])
                score = V.OLD.content_score(tokens, refs[i])
                replay_error = max(replay_error, abs(float(score.detach()) - pending[i]))
                (score * derivative[i]).backward()
        need(replay_error < 2e-8, 'FULL_VJP_FORWARD_PARITY')
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in small.parameters()), 'FINITE_ADAPTER_GRADIENT')
        # A zero content coefficient or saturated case can legally have no
        # adapter gradient; record it rather than forcing synthetic updates.
        anorm = float(torch.nn.utils.clip_grad_norm_(small.parameters(), c['clip_norm']))
        hnorm = float(torch.nn.utils.clip_grad_norm_([theta], c['clip_norm']))
        head_before = theta.detach().cpu().clone()
        before = V.state_cpu(small) if step == 0 else None
        optimizer.step()
        need(bool(torch.isfinite(theta).all()) and all(bool(torch.isfinite(p).all()) for p in small.parameters()), 'FINITE_UPDATED_PARAMETERS')
        record = dict(authority=ctx['authority'], fold=ctx['fold'], arm=arm, step=step+1,
            pass_index=step // len(rows), query_id=row['query_id'], content=pending,
            derivative=derivative.cpu().tolist(), head_gradient=head_gradient.cpu().tolist(),
            head_before=head_before.tolist(), head_after=theta.detach().cpu().tolist(),
            decision_before_update=V.choose(row, pending, head_before), active_candidates=active,
            loss_before_update=float(value.detach()), gradient_forward_error=replay_error,
            adapter_gradient_norm=anorm, head_gradient_norm=hnorm, scoring_seconds=scoring_seconds,
            backward_seconds=time.monotonic()-s, frozen_backbone=True, direct_M_in_head=False)
        if before is not None:
            record['adapter_parameter_change_max'] = max(float((small.state_dict()[k].cpu()-v).abs().max()) for k,v in before.items())
            record['head_parameter_change_max'] = float((theta.detach().cpu()-head_before).abs().max())
        write(d / 'steps' / f'{step+1:05d}.json', record)
        step += 1; pending = []
        if step == 1 or step % 16 == 0 or step == target:
            checkpoint([])
        emit(stage='fit', fold=ctx['fold'], arm=arm, step=step, total=c['updates'],
             loss=record['loss_before_update'], scoring_seconds=scoring_seconds,
             adapter_gradient_norm=anorm, head_gradient_norm=hnorm)
        del refs, cache, content, value
    checkpoint([])
    if pilot:
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        need(V.tree_equal(saved['adapter'], V.state_cpu(small)) and torch.equal(saved['head'], theta.detach().cpu())
             and V.tree_equal(saved['optimizer'], V.cpu_optimizer_state(optimizer)), 'PILOT_RELOAD_PARITY')
        first = read(d / 'steps/00001.json')
        need(first['adapter_parameter_change_max'] > 0 and first['head_parameter_change_max'] > 0,
             'PILOT_BOTH_MODULES_UPDATE')
        write(d / 'pilot_validation.json', dict(status='H593_POST_FULL128_UPDATE_RESUME_PASS', authority=ctx['authority'],
            input_scope=ctx['input_scope'], arm=arm, checkpoint=bind(cp), step=step, candidate_count=128,
            parity=parity_records, seconds=time.monotonic()-started, cache_stats=inputs.stats(), model_report=model.report))
        return True
    snapshot = d / 'final.pt'
    save(snapshot, dict(authority=ctx['authority'], input_scope=ctx['input_scope'], arm=arm, step=step,
        adapter=V.state_cpu(small), head=theta.detach().cpu().clone(), warmstart=wb, normalization=ctx['normalization']))
    write(fitseal, dict(status='H593_FIXED_EIGHT_PASSES_COMPLETE', authority=ctx['authority'], input_scope=ctx['input_scope'],
        arm=arm, steps=step, passes=c['passes'], train_queries=len(rows), snapshot=bind(snapshot), checkpoint=bind(cp),
        held_label_reads=0, old_pilot_parameters_reused=False, parity=parity_records, cache_stats=inputs.stats(),
        final_checkpoint_selection='fixed eight complete TRAIN passes', direct_M_in_head=False,
        adapter_parameters=sum(p.numel() for p in small.parameters()), head_features=V.FEATURES['INTERNAL3']))
    return True


def evaluate(ctx, inputs, arm, device, budget, split='held', shard_index=0, shard_count=1):
    import torch
    from rc_postllm_m_backend_v1 import load_projection
    started = time.monotonic(); d = ctx['out'] / arm
    fit = read(d / 'fit_validation.json')
    need(fit['authority'] == ctx['authority'] and fit['input_scope'] == ctx['input_scope']
         and fit['steps'] == ctx['config']['updates'], 'FINAL_FIXED_ENDPOINT_BEFORE_EVAL')
    snapshot = fit['snapshot']
    state = torch.load(checked(snapshot), map_location='cpu', weights_only=True)
    model = load_projection(ctx['a'], device=device); small = adapter(ctx, arm, device)
    small.load_state_dict(state['adapter']); small.eval()
    rows = ctx[split][shard_index::shard_count]
    modes = ['native', 'constant', 'shuffled'] if split == 'held' and arm == 'POST_REAL' else ['native']
    destination = d / ('held' if split == 'held' else 'train_endpoint')
    records = []
    warmseal = read(ctx['out'] / 'warm/validation.json')
    external_thetas = {kind: read(checked(b))['theta'] for kind, b in warmseal['heads'].items()}
    for row in rows:
        cache = refs = None
        for mode in modes:
            p = destination / mode / (row['query_id']+'.json')
            if p.exists():
                old = read(p)
                need(old['authority'] == ctx['authority'] and old['snapshot'] == snapshot
                     and old['candidate_ids'] == row['candidate_ids'] and old['intervention'] == mode, 'EVAL_RESUME_SCOPE')
                records.append(bind(p)); continue
            if STOP or time.monotonic()-started > budget-10:
                return False
            if cache is None:
                cache = inputs.cache(row); refs = inputs.refs(row)
            pp = p.with_suffix('.partial.json'); values = []
            if pp.exists():
                old = read(pp)
                need(old['authority'] == ctx['authority'] and old['snapshot'] == snapshot
                     and old['query_id'] == row['query_id'] and old['intervention'] == mode, 'EVAL_PARTIAL_SCOPE')
                values = old['L']; V.validate_pending(values)
            def persist(values):
                write(pp, dict(authority=ctx['authority'], snapshot=snapshot, query_id=row['query_id'],
                               intervention=mode, candidate_ids=row['candidate_ids'], L=values))
            if not score_all(model, cache, small, row, refs, values, ctx, arm, mode, started, budget-5, persist):
                return False
            decision = V.choose(row, values, state['head'])
            external = {}
            if mode == 'native':
                for kind, coefficients in external_thetas.items():
                    external[kind] = V.choose(row, cache['fresh_L0'].cpu(), coefficients, kind)
            write(p, dict(authority=ctx['authority'], input_scope=ctx['input_scope'], snapshot=snapshot,
                split=split, fold=ctx['fold'], arm=arm, intervention=mode, query_id=row['query_id'],
                candidate_ids=row['candidate_ids'], candidate_identities=row['candidate_identities'],
                winner_index=row['winner_index'], raw_scores=row['raw_scores'], M=row['M'], L=values,
                L0=cache['fresh_L0'].cpu().tolist(), decision=decision, external=external,
                hidden_source=inputs.query_sources[row['query_id']], direct_M_in_head=False,
                held_label_reads=0, training_updates=0))
            records.append(bind(p))
            emit(stage='evaluation', fold=ctx['fold'], arm=arm, split=split, intervention=mode,
                 query_id=row['query_id'], completed=len(records), total=len(rows)*len(modes))
        del cache, refs
    seal = destination / f'validation-{shard_index:03d}-of-{shard_count:03d}.json'
    write(seal, dict(status='FULL_C128_ENDPOINT_SHARD_COMPLETE', authority=ctx['authority'],
        input_scope=ctx['input_scope'], snapshot=snapshot, split=split, arm=arm, fold=ctx['fold'],
        shard_index=shard_index, shard_count=shard_count, query_ids=[r['query_id'] for r in rows],
        modes=modes, predictions=records, held_label_reads=0, cache_stats=inputs.stats()))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['warm', 'pilot', 'fit', 'eval', 'train-endpoint', 'worker'])
    p.add_argument('--authority', type=Path, default=DEFAULT_AUTH)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--fold', type=int, required=True, choices=range(5))
    p.add_argument('--arm', choices=ARMS, default='POST_REAL')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--budget', type=float, default=500)
    p.add_argument('--cpu-cache-gb', type=float, default=8.)
    p.add_argument('--gpu-cache-gb', type=float, default=3.)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    need(0 <= args.shard_index < args.shard_count, 'VALID_EVAL_SHARD')
    ctx = experiment(args.authority, args.out, args.fold)
    setup(ctx['config']['seed'], args.threads)
    inputs = Inputs(ctx['authority'], args.device, args.cpu_cache_gb, args.gpu_cache_gb)
    started = time.monotonic()
    if args.stage == 'warm':
        done = warm(ctx, inputs, args.budget)
    elif args.stage in ('pilot', 'fit', 'worker'):
        done = worker(ctx, inputs, args.arm, args.device, args.budget, pilot=args.stage == 'pilot')
        if done and args.stage == 'worker':
            left = args.budget-(time.monotonic()-started)
            done = left > 15 and evaluate(ctx, inputs, args.arm, args.device, left, 'train', args.shard_index, args.shard_count)
            if done:
                left = args.budget-(time.monotonic()-started)
                done = left > 15 and evaluate(ctx, inputs, args.arm, args.device, left, 'held', args.shard_index, args.shard_count)
    else:
        done = evaluate(ctx, inputs, args.arm, args.device, args.budget,
                        'train' if args.stage == 'train-endpoint' else 'held', args.shard_index, args.shard_count)
    emit(stage='exit', fold=args.fold, arm=args.arm, complete=done, seconds=time.monotonic()-started, cache_stats=inputs.stats())
    if not done:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
