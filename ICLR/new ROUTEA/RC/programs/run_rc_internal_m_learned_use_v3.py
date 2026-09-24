#!/usr/bin/env python3
"""TRAIN16 ability test: true RoMa M enters frozen ColNomic before the LLM.

The internal output head receives RAW, new free content, and a bias, never M.
The adapter and head learn together. Every update uses the complete C128 loss.
This opened-data fit diagnostic neither selects nor evaluates held checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_prellm_m_pilot_v1 as OLD

OUT = ROOT / 'results/rc_internal_m_learned_use_v3'
AUTH = ROOT / 'registry/rc_internal_m_learned_use_v3_authority_20260924.json'
PLAN = ROOT / 'plan/RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_EXECUTION_20260924.md'
LAUNCH = ROOT / 'slurm/rc_internal_m_learned_use_v3.sbatch'
V2 = ROOT / 'results/rc_prellm_m_adapter_v2'
V2_AUTH = ROOT / 'registry/rc_prellm_m_adapter_authority_v2_20260924.json'
ARMS = ('PRE_CONSTANT', 'PRE_REAL')
FEATURES = {
    'INTERNAL3': ['raw_standardized_delta', 'sym_L', 'bias'],
    'ADDITIVE4': ['raw_standardized_delta', 'sym_M', 'sym_L', 'bias'],
    'PRODUCT5': ['raw_standardized_delta', 'sym_ML', 'sym_M', 'sym_L', 'bias'],
}
CONFIG = dict(updates=128, endpoint_updates=[16, 128], checkpoint_updates=[16, 64, 128],
              warmstart_steps=2000, external_steps=2000, seed=17, bottleneck=16,
              residual_scale=.1, adapter_lr=3e-4, head_lr=.03, weight_decay=.001,
              clip_norm=1., max_requeues=48, worker_budget=400,
              diagnostic_interventions=['constant', 'shuffled'], diagnostic_updates=[128],
              candidates=128, train_count=16, frozen_opened_probe_count=8,
              attention='sdpa', processor_backend='torchvision')

need, read, bind, checked, write, save, emit = (OLD.need, OLD.read, OLD.bind, OLD.checked,
                                              OLD.write, OLD.save, OLD.emit)


def features(row, content, kind='INTERNAL3'):
    """All 127 challengers in the frozen physical candidate order, FP64."""
    import torch
    need(kind in FEATURES, 'UNKNOWN_HEAD')
    need(content.ndim == 1 and len(content) == 128, 'FULL_C128_CONTENT_REQUIRED')
    need(bool(torch.isfinite(content).all()), 'FINITE_CONTENT')
    raw = torch.as_tensor(row['raw_scores'], dtype=torch.float64, device=content.device)
    winner = int(row['winner_index'])
    idx = [i for i in range(128) if i != winner]
    need(idx == row['challenger_positions'], 'CHALLENGER_ORDER')
    def sym(x):
        return (x[idx] - x[winner]) / (x[idx].abs() + x[winner].abs() + 1e-12)
    columns = [(raw[idx] - raw[winner]) / raw.std(unbiased=False).clamp_min(1e-12)]
    if kind != 'INTERNAL3':
        mass = torch.as_tensor(row['M'], dtype=torch.float64, device=content.device)
        if kind == 'PRODUCT5':
            columns.append(sym(mass * content))
        columns.append(sym(mass))
    columns.extend([sym(content), torch.ones(127, dtype=torch.float64, device=content.device)])
    return torch.stack(columns, dim=1)


def cost_from_logits(row, z):
    """Original target-free action COST1, with retrieval identity labels in TRAIN."""
    import torch
    from torch.nn import functional as F
    need(z.ndim == 1 and z.numel() == 127, 'FULL_127_CHALLENGER_LOSS')
    need(len(row['target_positions']) == 1, 'SINGLE_TRAIN_IDENTITY')
    target = row['target_positions'][0]
    if target == row['winner_index']:
        return F.softplus(z.amax())
    position = row['challenger_positions'].index(target)
    wrong = z.clone()
    wrong[position] = -torch.inf
    return F.softplus(-z[position]) + F.softplus(wrong.amax())


def loss(row, content, theta, kind='INTERNAL3'):
    return cost_from_logits(row, features(row, content, kind) @ theta)


def joint_derivatives(row, content, theta, kind='INTERNAL3'):
    """Derivative of the *same* full action loss for both trainable paths."""
    import torch
    value = loss(row, content, theta, kind)
    dcontent, dhead = torch.autograd.grad(value, (content, theta))
    return value, dcontent.detach(), dhead.detach()


def map_warm_head(theta3, kind):
    """Embed the no-M readout without changing a single initial logit."""
    need(len(theta3) == 3 and kind in FEATURES, 'COMMON_HEAD_SHAPE')
    w = list(map(float, theta3))
    if kind == 'INTERNAL3':
        return w
    return [w[0], 0., w[1], w[2]] if kind == 'ADDITIVE4' else [w[0], 0., 0., w[1], w[2]]


def validate_pending(values):
    need(isinstance(values, list) and len(values) <= 128, 'PENDING_FULL_AXIS_LENGTH')
    need(all(isinstance(x, (float, int)) and math.isfinite(x) for x in values), 'PENDING_FINITE')


def tree_equal(left, right):
    """Exact recursive optimizer/checkpoint state comparison, also used by tests."""
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def choose(row, content, theta, kind='INTERNAL3'):
    import torch
    content = torch.as_tensor(content, dtype=torch.float64, device='cpu')
    theta = torch.as_tensor(theta, dtype=torch.float64, device='cpu')
    need(bool(torch.isfinite(theta).all()), 'FINITE_HEAD')
    z = (features(row, content, kind) @ theta).detach()
    winner = row['winner_index']
    pos = row['challenger_positions'][int(z.argmax())] if float(z.max()) > 0 else winner
    scores = torch.zeros(128, dtype=torch.float64)
    scores[row['challenger_positions']] = z
    result = dict(prediction_position=pos, prediction_id=row['candidate_ids'][pos],
                  prediction_identity=row['candidate_identities'][pos], switched=pos != winner,
                  challenger_positions=row['challenger_positions'], logits=z.tolist(),
                  scores128=scores.tolist(), theta=theta.tolist(), features=FEATURES[kind])
    if 'target_positions' in row:
        target = row['target_positions'][0]
        target_score = scores[target]
        others = scores.clone(); others[target] = -torch.inf
        result.update(correct=row['candidate_identities'][pos] == row['candidate_identities'][target],
                      cost1=float(cost_from_logits(row, z)),
                      target_position=target,
                      target_vs_strongest_wrong_margin=float(target_score - others.max()),
                      target_rank=1 + int((scores > target_score).sum()),
                      free_content_target_rank=1 + int((content > content[target]).sum()))
    return result


def state_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def make_optimizer(small, theta, a):
    import torch
    return torch.optim.AdamW([
        dict(params=list(small.parameters()), lr=a['adapter_lr'], weight_decay=a['weight_decay']),
        dict(params=[theta], lr=a['head_lr'], weight_decay=a['weight_decay']),
    ])


def setup(a):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(a['seed']); np.random.seed(a['seed']); random.seed(a['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


def check_sources(a):
    for b in a['code_sources']:
        checked(b)
    checked(a['v2_authority']); checked(a['manifest'])
    for b in a['cache_bindings']:
        checked(b)
    # The v2 cache source receipt already checked every immutable model SHA.
    # Device numbers vary across GPFS hosts and must not be compared.
    receipt = read(checked(a['model_source_validation']))
    need(receipt['authority'] == a['v2_authority'], 'V2_MODEL_SOURCE_AUTHORITY')
    for item in receipt['files']:
        stat = Path(item['binding']['path']).stat()
        need(item['stat'][1:] == [stat.st_ino, stat.st_size, stat.st_mtime_ns],
             'FROZEN_MODEL_SOURCE_STAT_DRIFT')


def guard(stage):
    a = read(AUTH); check_sources(a)
    m = read(checked(a['manifest']))
    need(len(m['train_rows']) == a['train_count'] == 16, 'TRAIN16_ONLY')
    for row in m['train_rows']:
        need(len(row['candidate_ids']) == len(set(row['candidate_ids'])) == 128,
             'UNIQUE_PHYSICAL_C128')
        need(len(row['candidate_identities']) == len(set(row['candidate_identities'])) == 128,
             'UNIQUE_IDENTITY_C128')
        need(row['target_positions'] == [row['target_position']], 'SINGLE_TARGET_AXIS')
    if stage not in ('preflight',):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    # This entire execution is TRAIN-only. Even its final summarizer cannot
    # open curator files or read the original opened probe label join.
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(s in p for s in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/isic/')),
             'PROTECTED_INPUT')
        need('curator_roles' not in p and '/target_join/' not in p, 'NO_HELD_LABELS')
        need(p != str(V2 / 'result.json').lower(), 'NO_OLD_PROBE_RESULT_INPUT')
    sys.addaudithook(audit)
    setup(a)
    return a, m


def prepare():
    """Freeze only after new program, launcher, tests and execution plan exist."""
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    v2a = read(V2_AUTH)
    manifest_path = checked(v2a['manifest']); m = read(manifest_path)
    need(len(m['train_rows']) == 16 and len(m['probe_rows']) == 8, 'INHERITED_PANEL')
    validations = []
    for row in m['train_rows']:
        d = V2 / 'encoder_cache' / row['query_id']
        v = read(d / 'validation.json')
        need(v['status'] == 'QUERY_ENCODER_CACHE_PASS' and v['authority'] == bind(V2_AUTH), 'V2_CACHE_PASS')
        checked(v['payload']); checked(v['parity'])
        validations.extend([bind(d / 'validation.json'), v['payload'], v['parity']])
    required_codes = [Path(__file__), LAUNCH, PLAN,
                      ROOT / 'tests/test_internal_m_learned_use_v3.py',
                      ROOT / 'programs/rc_prellm_m_adapter_v1.py',
                      ROOT / 'programs/run_rc_prellm_m_pilot_v1.py']
    for p in required_codes:
        need(p.exists(), 'MISSING_AUTHORITY_SOURCE:' + str(p))
    a = dict(CONFIG)
    a.update(status='TRAIN16_INTERNAL_REAL_M_LEARNED_USE_AUTHORIZED',
             user_authorization='2026-09-24 保留 RoMa，内部学会使用 M；先推上轮，然后开始下轮',
             code_sources=[bind(p) for p in required_codes], manifest=bind(manifest_path),
             v2_authority=bind(V2_AUTH), cache_bindings=validations,
             model_source_validation=bind(V2 / 'model_source_validation.json'),
             encoder_cache_validation=bind(V2 / 'encoder_cache/validation.json'),
             model=v2a['model'], arms=list(ARMS), source_fold=0,
             mass_normalization=m['mass_normalization'],
             score_dtype='float64', adapter_dtype='float32', backbone_dtype='bfloat16',
             encoder_trainable=False, retrieval_lora_trainable=False,
             training_order=[r['query_id'] for r in m['train_rows']],
             feature_definitions=FEATURES,
             external_supervision='Same TRAIN16, original cached fresh full-reference L and original M',
             warmstart='INTERNAL3 full TRAIN16 mean COST1, zero init,2000 AdamW updates',
             external_initialization='Embed same warm INTERNAL3; added M/ML coefficients zero',
             internal_initialization='Same warm INTERNAL3 plus zero residual adapter',
             selection='Fixed128 updates; no probe, held, hyperparameter or checkpoint selection',
             diagnostic_scope='TRAIN16 repeated fit diagnostic only; neither OOF nor external evidence',
             new_roma_forwards=0, new_visual_encoder_forwards=0,
             grad_rule='Full128 scalar loss; head gradient directly plus exact nonzero content VJP',
             continuation='One PRE_REAL update must pass actual engineering gates, then resume both arms',
             candidate_sampling=False)
    write(AUTH, a)
    emit(stage='prepare', authority=bind(AUTH), candidates=128, updates=a['updates'])


def cached_row(row, device='cpu'):
    import torch
    v = read(V2 / 'encoder_cache' / row['query_id'] / 'validation.json')
    need(v['authority'] == bind(V2_AUTH), 'CACHE_V2_AUTHORITY')
    return torch.load(checked(v['payload']), map_location=device, weights_only=True)


def train_contents(m):
    return {r['query_id']: cached_row(r)['fresh_L0'].to(dtype=__import__('torch').float64)
            for r in m['train_rows']}


def summary(rows, predictions):
    count = len(rows)
    correct = sum(predictions[r['query_id']]['correct'] for r in rows)
    mean_loss = sum(predictions[r['query_id']]['cost1'] for r in rows) / count
    rescue = [r['query_id'] for r in rows if not r['raw_correct'] and predictions[r['query_id']]['correct']]
    breaks = [r['query_id'] for r in rows if r['raw_correct'] and not predictions[r['query_id']]['correct']]
    return dict(queries=count, correct=correct, mean_cost1=mean_loss,
                target_margin_mean=sum(predictions[r['query_id']]['target_vs_strongest_wrong_margin'] for r in rows)/count,
                rescues_vs_raw=rescue, breaks_vs_raw=breaks,
                changed_vs_raw=sum(predictions[r['query_id']]['switched'] for r in rows))


def fit_cpu_head(rows, contents, theta0, kind, steps, a):
    import torch
    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float64).clone())
    xs = [features(r, contents[r['query_id']], kind) for r in rows]
    opt = torch.optim.AdamW([theta], lr=a['head_lr'], weight_decay=a['weight_decay'])
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        value = torch.stack([cost_from_logits(r, x @ theta) for r, x in zip(rows, xs)]).mean()
        value.backward()
        need(torch.isfinite(theta.grad).all(), 'CPU_HEAD_GRAD_FINITE')
        norm = float(torch.nn.utils.clip_grad_norm_([theta], a['clip_norm']))
        opt.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            with torch.no_grad():
                endpoint = float(torch.stack([cost_from_logits(r, x @ theta) for r, x in zip(rows, xs)]).mean())
            history.append(dict(step=step+1, mean_cost1=endpoint, head_gradient_norm=norm))
    predictions = {r['query_id']: choose(r, contents[r['query_id']], theta.detach(), kind) for r in rows}
    return dict(theta=theta.detach().tolist(), feature_order=FEATURES[kind], steps=steps,
                history=history, predictions=predictions, summary=summary(rows, predictions),
                fit_queries=[r['query_id'] for r in rows], held_label_reads=0,
                initial_theta=list(map(float, theta0)), update_unit='Full TRAIN16 mean COST1')


def cpu(a, m):
    d = OUT / 'cpu'; d.mkdir(parents=True, exist_ok=True)
    if (d / 'validation.json').exists():
        v = read(d / 'validation.json'); need(v['authority'] == bind(AUTH), 'CPU_RESUME_AUTHORITY')
        for b in v['heads']:
            checked(b)
        return True
    contents = train_contents(m); rows = m['train_rows']
    wp = d / 'warmstart_INTERNAL3.json'
    if wp.exists():
        warm = read(wp); need(warm['authority'] == bind(AUTH), 'WARM_AUTHORITY')
    else:
        warm = fit_cpu_head(rows, contents, [0.,0.,0.], 'INTERNAL3', a['warmstart_steps'], a)
        warm.update(authority=bind(AUTH), kind='INTERNAL3', explicit_M=False)
        write(wp, warm)
    need(abs(warm['theta'][1]) > 1e-8, 'WARM_CONTENT_GRADIENT_MUST_EXIST')
    bindings = [bind(wp)]
    for kind in ('ADDITIVE4', 'PRODUCT5'):
        p = d / (kind + '.json')
        if p.exists():
            value = read(p); need(value['authority'] == bind(AUTH), 'EXTERNAL_AUTHORITY')
        else:
            w = warm['theta']; initial = map_warm_head(w, kind)
            for row in rows:
                baseline = features(row, contents[row['query_id']], 'INTERNAL3') @ __import__('torch').tensor(w,dtype=__import__('torch').float64)
                restored = features(row, contents[row['query_id']], kind) @ __import__('torch').tensor(initial,dtype=__import__('torch').float64)
                need(__import__('torch').allclose(baseline, restored, atol=1e-12, rtol=0), 'COMMON_HEAD_INITIALIZATION')
            value = fit_cpu_head(rows, contents, initial, kind, a['external_steps'], a)
            value.update(authority=bind(AUTH), kind=kind, warmstart=bind(wp), explicit_M=True)
            write(p, value)
        bindings.append(bind(p))
    write(d / 'validation.json', dict(status='TRAIN16_COMMON_WARM_AND_EXTERNAL_HEADS_PASS',
          authority=bind(AUTH), heads=bindings, no_probe_evaluation=True, held_label_reads=0,
          same_label_budget=True, identical_optimization_budget=False,
          optimization_difference='External full TRAIN16 batch2000; adapter joint128 ordered per-query updates'))
    emit(stage='cpu', status='TRAIN16_COMMON_WARM_AND_EXTERNAL_HEADS_PASS',
         summaries={Path(b['path']).stem: read(b['path'])['summary'] for b in bindings})
    return True


def preflight(a, m):
    import torch
    generator = torch.Generator().manual_seed(73)
    records = []
    for kind in FEATURES:
        for row in m['train_rows']:
            design = torch.randn(128, 5, generator=generator, dtype=torch.float64)
            latent = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)
            theta = torch.randn(len(FEATURES[kind]), generator=generator, dtype=torch.float64, requires_grad=True)
            direct_content = (design @ latent).sigmoid()
            direct = loss(row, direct_content, theta, kind)
            direct_w, direct_h = torch.autograd.grad(direct, (latent, theta))
            detached = direct_content.detach().requires_grad_()
            hh = theta.detach().clone().requires_grad_()
            _, dc, dh = joint_derivatives(row, detached, hh, kind)
            ww = latent.detach().clone().requires_grad_()
            replay = sum((design[i] @ ww).sigmoid()*dc[i].detach() for i in dc.nonzero().flatten().tolist())
            replay.backward()
            e1 = float((direct_w - ww.grad).abs().max()); e2 = float((direct_h - dh).abs().max())
            need(e1 < 1e-11 and e2 < 1e-11, 'JOINT_HEAD_ADAPTER_VJP')
            changed = dict(row); changed['M'] = list(reversed(row['M']))
            invariant = float((features(row, detached, 'INTERNAL3') - features(changed, detached, 'INTERNAL3')).abs().max())
            need(invariant == 0, 'INTERNAL_HEAD_HAS_M_LEAKAGE')
            records.append(dict(kind=kind, query_id=row['query_id'], adapter_error=e1, head_error=e2,
                                active_candidates=int((dc != 0).sum()), direct_M_feature_error=invariant))
    write(OUT / 'preflight.json', dict(status='JOINT_FULL_C128_VJP_AND_NO_DIRECT_M_PASS',
          authority=bind(AUTH), checks=records, held_label_reads=0))
    emit(stage='preflight', status='JOINT_FULL_C128_VJP_AND_NO_DIRECT_M_PASS')
    return True


def load_model(a):
    import torch
    from rc_prellm_m_adapter_v1 import _load_frozen_colnomic_weights
    need(torch.cuda.is_available(), 'GPU_REQUIRED')
    started = time.monotonic()
    model, report = _load_frozen_colnomic_weights(a['model'], 'cuda', attention=a['attention'], verify_weight_hashes=False)
    need(not any(p.requires_grad for p in model.parameters()), 'FROZEN_BACKBONE')
    # No images are reprocessed in v3. The precise qualified merger tensors and
    # image token masks already live in the v2 payloads frozen in this authority.
    report.update(processor_forward_calls=0, visual_forward_calls=0,
                  cache_processor_backend=a['processor_backend'], loading_seconds=time.monotonic()-started)
    write(OUT/'loading'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
          dict(authority=bind(AUTH), report=report))
    emit(stage='load_model', seconds=report['loading_seconds'], gpu=torch.cuda.get_device_name())
    return model


def create_adapter(a, m, arm):
    import torch
    from rc_prellm_m_adapter_v1 import QualityResidualAdapter
    torch.manual_seed(a['seed'])
    return QualityResidualAdapter(hidden_size=3584, bottleneck=a['bottleneck'],
            conditioning='real' if arm == 'PRE_REAL' else 'constant',
            mass_log_mean=m['mass_normalization']['log_mean'],
            mass_log_std=m['mass_normalization']['log_std'],
            mass_epsilon=m['mass_normalization']['epsilon'],
            residual_scale=a['residual_scale']).to('cuda')


def predict(model, cache, small, mass):
    return OLD.predict_tokens(model, cache, small, mass, 'prellm')


def score_all(model, cache, small, row, refs, values, started, budget, persist, intervention='native'):
    import torch
    need(intervention in ('native','constant','shuffled'), 'INTERVENTION')
    validate_pending(values)
    need(len(refs) == len(row['M']) == 128, 'SCORE_FULL_C128_AXIS')
    masses = row['M'][1:] + row['M'][:1] if intervention == 'shuffled' else row['M']
    original = small.conditioning
    constant = original == 'constant' or intervention == 'constant'
    if intervention == 'constant':
        small.conditioning = 'constant'
    try:
        with torch.no_grad():
            common = predict(model, cache, small, masses[0]) if constant and len(values) < 128 else None
            for i in range(len(values),128):
                if time.monotonic()-started > budget:
                    persist(values); return False
                output = common if constant else predict(model, cache, small, masses[i])
                values.append(float(OLD.content_score(output, refs[i])))
                if (i+1) % 16 == 0:
                    persist(values)
        persist(values)
        return True
    finally:
        small.conditioning = original


def evaluate_endpoint(a, m, arm, step, model, small, theta, started, budget, snapshot):
    """Checkpoint fixed; per-query complete and partial predictions bind its SHA."""
    import torch
    d = OUT / arm / 'endpoints' / f'{step:04d}'
    interventions = ['native']
    if arm == 'PRE_REAL' and step in a['diagnostic_updates']:
        interventions += a['diagnostic_interventions']
    seal = d / 'validation.json'
    if seal.exists():
        v = read(seal); need(v['snapshot'] == snapshot and v['authority'] == bind(AUTH), 'ENDPOINT_RESUME')
        for b in v['predictions']:
            checked(b)
        return True
    all_predictions = []
    for row in m['train_rows']:
        cache = None; refs = None
        for intervention in interventions:
            p = d / intervention / (row['query_id'] + '.json')
            if p.exists():
                record = read(p)
                need(record['snapshot'] == snapshot and record['authority'] == bind(AUTH), 'PREDICTION_RESUME')
                all_predictions.append(bind(p)); continue
            if time.monotonic()-started > budget-20:
                return False
            if cache is None:
                cache = cached_row(row, 'cuda'); refs = OLD.references(row)
            partial = p.with_suffix('.partial.json')
            values = []
            if partial.exists():
                old = read(partial)
                need(old['snapshot'] == snapshot and old['authority'] == bind(AUTH)
                     and old['query_id'] == row['query_id'] and old['intervention'] == intervention,
                     'PARTIAL_SNAPSHOT_AXIS')
                values = old['L']; validate_pending(values)
            def persist(v):
                write(partial,dict(authority=bind(AUTH),snapshot=snapshot,query_id=row['query_id'],intervention=intervention,L=v))
            if not score_all(model,cache,small,row,refs,values,started,budget,persist,intervention):
                return False
            prediction = choose(row, values, theta.detach().cpu())
            write(p,dict(authority=bind(AUTH),snapshot=snapshot,step=step,arm=arm,intervention=intervention,
                         query_id=row['query_id'],candidate_ids=row['candidate_ids'],
                         L=values,M=row['M'],raw_scores=row['raw_scores'],decision=prediction,
                         direct_M_in_head=False,held_label_reads=0))
            all_predictions.append(bind(p))
            emit(stage='train_endpoint', arm=arm, step=step, intervention=intervention,
                 query_id=row['query_id'],loss=prediction['cost1'],correct=prediction['correct'])
        del refs,cache; torch.cuda.empty_cache()
    aggregate = {}
    for intervention in interventions:
        predictions = {r['query_id']: read(d/intervention/(r['query_id']+'.json'))['decision'] for r in m['train_rows']}
        aggregate[intervention] = summary(m['train_rows'],predictions)
    write(seal,dict(status='FULL_TRAIN16_ENDPOINT_PASS',authority=bind(AUTH),snapshot=snapshot,
                   step=step,arm=arm,predictions=all_predictions,summaries=aggregate,held_label_reads=0))
    emit(stage='train_endpoint_complete',arm=arm,step=step,summaries=aggregate)
    return True


def cpu_optimizer_state(optimizer):
    import torch
    def rec(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:rec(v) for k,v in value.items()}
        if isinstance(value,list): return [rec(v) for v in value]
        if isinstance(value,tuple): return tuple(rec(v) for v in value)
        return value
    return rec(optimizer.state_dict())


def finish_pilot(a, arm, checkpoint_path, small, theta, optimizer, elapsed_seconds):
    """Idempotently seal a committed first update, including after preemption."""
    import torch
    cp = Path(checkpoint_path); d = cp.parent
    loaded = torch.load(cp, map_location='cpu', weights_only=True)
    need(loaded['authority'] == bind(AUTH) and loaded['arm'] == arm, 'PILOT_CHECKPOINT_LINEAGE')
    need(loaded['step'] == 1 and not loaded['pending'], 'ATOMIC_STEP_COMMIT')
    need(torch.equal(loaded['head'], theta.detach().cpu()), 'HEAD_RELOAD')
    for name, tensor in small.state_dict().items():
        need(torch.equal(loaded['adapter'][name], tensor.detach().cpu()), 'ADAPTER_RELOAD')
    need(tree_equal(loaded['optimizer'], cpu_optimizer_state(optimizer)), 'PILOT_OPTIMIZER_RELOAD')
    receipt = read(d / 'steps/0001.json')
    need(receipt['adapter_parameter_change_max'] > 0 and receipt['head_parameter_change_max'] > 0,
         'PILOT_BOTH_MODULES_CHANGED')
    need(receipt['gradient_forward_error'] < 2e-8, 'PILOT_FORWARD_PARITY')
    need(receipt['token_changed_fraction'] > 0, 'PILOT_TOKEN_CHANGE_VISIBLE')
    result = dict(status='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS',
                  authority=bind(AUTH), arm=arm, step=1, checkpoint=bind(cp),
                  step_record=bind(d/'steps/0001.json'), full_candidates=128,
                  head_updates=True, adapter_updates=True, head_M_input=False,
                  resume_parameters_exact=True, resume_optimizer_exact=True,
                  opened_probe_evaluations=0, measured_seconds=elapsed_seconds)
    write(OUT / 'pilot_validation.json', result)
    return result


def train(a,m,arm,budget,pilot=False):
    import torch
    started = time.monotonic()
    d = OUT / arm; d.mkdir(parents=True,exist_ok=True)
    lock = (d/'worker.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    need(arm in ARMS and (not pilot or arm == 'PRE_REAL'),'TRAIN_ARM')
    if (d/'fit_validation.json').exists():
        need(not pilot,'PILOT_AFTER_TRAIN')
        return True
    cpu_receipt = read(OUT/'cpu/validation.json')
    need(cpu_receipt['status']=='TRAIN16_COMMON_WARM_AND_EXTERNAL_HEADS_PASS'
         and cpu_receipt['authority']==bind(AUTH),'CPU_FIRST')
    for binding in cpu_receipt['heads']:
        checked(binding)
    if not pilot:
        need(read(OUT/'pilot_validation.json')['status']=='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS','PILOT_REQUIRED')
    model = load_model(a); small = create_adapter(a,m,arm)
    warm_binding = bind(OUT/'cpu/warmstart_INTERNAL3.json'); warm = read(warm_binding['path'])
    need(warm_binding in cpu_receipt['heads'], 'SEALED_COMMON_WARM_HEAD')
    theta = torch.nn.Parameter(torch.tensor(warm['theta'],dtype=torch.float64,device='cuda'))
    optimizer = make_optimizer(small,theta,a)
    cp = d/'checkpoint.pt'; step=0; pending=[]; history=[]
    def checkpoint(values):
        validate_pending(values)
        save(cp,dict(authority=bind(AUTH),arm=arm,adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                     optimizer=cpu_optimizer_state(optimizer),step=step,history=history,pending=list(values),
                     pending_query_id=m['train_rows'][step % 16]['query_id'] if step < a['updates'] else None,
                     warmstart=warm_binding,candidate_count=128,conditioning=small.conditioning))
    if cp.exists():
        saved = torch.load(cp,map_location='cpu',weights_only=True)
        need(saved['authority']==bind(AUTH) and saved['arm']==arm and saved['warmstart']==warm_binding,'CHECKPOINT_BINDING')
        small.load_state_dict(saved['adapter']); theta.data.copy_(saved['head'].to('cuda'))
        optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];pending=saved['pending'];history=saved['history']
        validate_pending(pending)
        need(0 <= step <= a['updates'] and len(history) == step, 'CHECKPOINT_STEP_HISTORY')
        expected_query = m['train_rows'][step % 16]['query_id'] if step < a['updates'] else None
        need(saved['pending_query_id'] == expected_query and saved['conditioning'] == small.conditioning,
             'CHECKPOINT_PENDING_QUERY_CONDITION')
        need(tree_equal(saved['optimizer'], cpu_optimizer_state(optimizer)), 'OPTIMIZER_RELOAD_EXACT')
    else:
        checkpoint([])
        save(d/'initial.pt',dict(authority=bind(AUTH),adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                                warmstart=warm_binding,step=0,arm=arm))
    if pilot and step == 1:
        # A preemption can occur after the atomic update commit and before its
        # receipt. Recover that receipt without repeating or skipping an update.
        finish_pilot(a,arm,cp,small,theta,optimizer,time.monotonic()-started)
    target = 1 if pilot else a['updates']
    while True:
        if step in a['checkpoint_updates']:
            snap = d/'snapshots'/f'{step:04d}.pt'
            if not snap.exists():
                save(snap,dict(authority=bind(AUTH),adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                               arm=arm,step=step,warmstart=warm_binding))
            frozen = torch.load(snap,map_location='cpu',weights_only=True)
            need(frozen['authority']==bind(AUTH) and frozen['arm']==arm and frozen['step']==step,
                 'SNAPSHOT_LINEAGE')
            need(tree_equal(frozen['adapter'],state_cpu(small))
                 and torch.equal(frozen['head'],theta.detach().cpu()),'SNAPSHOT_STATE_PARITY')
            if step in a['endpoint_updates'] and not evaluate_endpoint(a,m,arm,step,model,small,theta,started,budget,bind(snap)):
                return False
        if step >= target:
            break
        if time.monotonic()-started > budget-30:
            return False
        row=m['train_rows'][step % len(m['train_rows'])]
        cache=cached_row(row,'cuda');refs=OLD.references(row)
        score_started=time.monotonic(); resumed_candidates=len(pending)
        if step==0 and not pending:
            # Actual engineering parity: the zero residual starts at the exact
            # cached content and the same common head on both conditions.
            with torch.no_grad():
                encoded=predict(model,cache,small,row['M'][0])
                native=cache['native_tokens'][cache['image_mask']]
                need(float((encoded-native).abs().max())<=1e-6,'ZERO_ADAPTER_NATIVE_PARITY')
        if not score_all(model,cache,small,row,refs,pending,started,budget,checkpoint):
            return False
        scoring_seconds = time.monotonic()-score_started
        if step == 0:
            zero_content_error = float((torch.tensor(pending,dtype=torch.float64)-cache['fresh_L0'].cpu()).abs().max())
            need(zero_content_error < 2e-8,'ZERO_FULL128_FRESH_CONTENT_PARITY')
        content=torch.tensor(pending,dtype=torch.float64,device='cuda',requires_grad=True)
        value,derivative,head_gradient=joint_derivatives(row,content,theta)
        active=derivative.nonzero().flatten().tolist()
        need(active and torch.isfinite(derivative).all() and torch.isfinite(head_gradient).all(),'JOINT_GRAD_FINITE')
        if time.monotonic()-started > budget-35:
            checkpoint(pending);return False
        optimizer.zero_grad(set_to_none=True);theta.grad=head_gradient.clone()
        grad_started=time.monotonic();replay_error=0.;last_output=None
        if arm=='PRE_CONSTANT':
            # One identical query representation contributes to all active
            # candidates; summing all terms preserves the full C128 gradient.
            last_output=predict(model,cache,small,row['M'][0])
            terms=[]
            for i in active:
                score=OLD.content_score(last_output,refs[i])
                replay_error=max(replay_error,abs(float(score.detach())-pending[i]))
                terms.append(score*derivative[i])
            torch.stack(terms).sum().backward()
        else:
            for i in active:
                last_output=predict(model,cache,small,row['M'][i])
                score=OLD.content_score(last_output,refs[i])
                replay_error=max(replay_error,abs(float(score.detach())-pending[i]))
                (score*derivative[i]).backward()
        need(replay_error < 2e-8,'NO_GRAD_VJP_FORWARD_PARITY')
        gradients=[p.grad for p in small.parameters() if p.grad is not None]
        need(gradients and all(torch.isfinite(g).all() for g in gradients),'ADAPTER_GRAD_FINITE')
        adapter_norm=float(torch.nn.utils.clip_grad_norm_(small.parameters(),a['clip_norm']))
        head_norm=float(torch.nn.utils.clip_grad_norm_([theta],a['clip_norm']))
        need(adapter_norm > 0,'ADAPTER_GRAD_NONZERO')
        need(not any(p.grad is not None for p in model.parameters()),'FROZEN_BACKBONE_GRAD')
        before=state_cpu(small);head_before=theta.detach().cpu().clone()
        prior_tokens=last_output.detach().clone()
        optimizer.step()
        need(bool(torch.isfinite(theta).all()) and all(bool(torch.isfinite(p).all()) for p in small.parameters()),
             'FINITE_UPDATED_PARAMETERS')
        adapter_change=max(float((small.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in before.items())
        head_change=float((theta.detach().cpu()-head_before).abs().max())
        need(adapter_change > 0 and head_change > 0,'BOTH_MODULES_UPDATE')
        record=dict(step=step+1,query_id=row['query_id'],arm=arm,loss_before_update=float(value.detach()),
                    content=pending,derivative=derivative.cpu().tolist(),head_gradient=head_gradient.cpu().tolist(),
                    decision_before_update=choose(row,pending,head_before),head_before=head_before.tolist(),
                    head_after=theta.detach().cpu().tolist(),active_candidates=active,
                    adapter_gradient_norm=adapter_norm,head_gradient_norm=head_norm,
                    adapter_parameter_change_max=adapter_change,head_parameter_change_max=head_change,
                    gradient_forward_error=replay_error,scoring_seconds=scoring_seconds,
                    backward_seconds=time.monotonic()-grad_started,resumed_candidates=resumed_candidates,
                    backbone_trainable=0,direct_M_in_head=False,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if step==0 or step+1 in a['checkpoint_updates']:
            with torch.no_grad():
                updated=predict(model,cache,small,row['M'][active[-1]])
                delta=updated.float()-prior_tokens.float()
                record['token_max_change']=float(delta.abs().max())
                record['token_changed_fraction']=float((delta!=0).float().mean())
                source=cache['merged'];modulated=small(source,row['M'][active[-1]])
                record['prellm_residual_max']=float((modulated.float()-source.float()).abs().max())
                save(d/'traces'/f'{step+1:04d}.pt',dict(query_id=row['query_id'],candidate_position=active[-1],
                     M=row['M'][active[-1]],source=source.cpu(),modulated_source=modulated.cpu(),
                     before_tokens=prior_tokens.cpu(),after_tokens=updated.cpu(),image_mask=cache['image_mask'].cpu()))
        write(d/'steps'/f'{step+1:04d}.json',record)
        history.append({k:v for k,v in record.items() if k not in ('content','derivative','decision_before_update','head_gradient')})
        step+=1;pending=[];checkpoint([])
        if step==1 and pilot:
            finish_pilot(a,arm,cp,small,theta,optimizer,time.monotonic()-started)
        emit(stage='fit',arm=arm,step=step,total=a['updates'],loss=record['loss_before_update'],
             scoring_seconds=record['scoring_seconds'],adapter_gradient_norm=adapter_norm,head_gradient_norm=head_norm)
        del refs,cache,last_output,content,value,prior_tokens;torch.cuda.empty_cache()
    if not pilot:
        endpoint_bindings=[bind(d/'endpoints'/f'{u:04d}'/'validation.json') for u in a['endpoint_updates']]
        write(d/'fit_validation.json',dict(status='FIXED128_UPDATES_TRAIN_ENDPOINTS_COMPLETE',authority=bind(AUTH),
              arm=arm,steps=step,checkpoint=bind(cp),endpoints=endpoint_bindings,
              snapshots=[bind(d/'snapshots'/f'{u:04d}.pt') for u in a['checkpoint_updates']],
              head_trainable_parameters=3,adapter_parameters=sum(p.numel() for p in small.parameters()),
              head_features=FEATURES['INTERNAL3'],direct_M_in_head=False,held_label_reads=0,
              selection='Fixed terminal128; no model or epoch selection'))
    return True


def summarize(a,m):
    output=dict(status='TRAIN16_INTERNAL_M_LEARNED_USE_DIAGNOSTIC_COMPLETE',authority=bind(AUTH),
                evidence_level='Opened TRAIN16 fit diagnostic; no held/probe result',
                raw_correct=sum(r['raw_correct'] for r in m['train_rows']),train_count=16,
                external={},internal={},no_generalization_claim=True,opened_probe_evaluations=0)
    cpu_validation=read(OUT/'cpu/validation.json')
    for b in cpu_validation['heads']:
        value=read(checked(b));output['external'][value['kind']]=dict(binding=b,theta=value['theta'],summary=value['summary'])
    for arm in ARMS:
        value=read(OUT/arm/'fit_validation.json')
        need(value['authority']==bind(AUTH) and value['steps']==a['updates'],'FINAL_ARM_PASS')
        checked(value['checkpoint'])
        endpoints={}
        for b in value['endpoints']:
            item=read(checked(b))
            for prediction in item['predictions']: checked(prediction)
            endpoints[str(item['step'])]=item['summaries']
        output['internal'][arm]=dict(validation=bind(OUT/arm/'fit_validation.json'),endpoints=endpoints)
    base=output['external']['INTERNAL3']['summary']['mean_cost1']
    output['train_loss_relative_change']={arm:{step:(base-v['native']['mean_cost1'])/max(abs(base),1e-12)
                                            for step,v in data['endpoints'].items()}
                                          for arm,data in output['internal'].items()}
    write(OUT/'result.json',output)
    lines=['# ColNomic internal M v3：TRAIN16 拟合能力诊断','',
           'RoMa 保留；PRE_REAL 使用真实配对 M，PRE_CONSTANT 使用恒定标准化条件。',
           '内部末端仅读取 RAW、自由内容 Lθ 与 bias；适配器和三参数头共同训练。',
           '原自然 C128、16 张 TRAIN、固定128次更新。没有评估或选择 probe。','',
           '| 路径 | 正确/16 | 全 TRAIN 平均 COST1 | 对 RAW 救回/损失 |',
           '|---|---:|---:|---:|']
    for kind,data in output['external'].items():
        s=data['summary'];lines.append(f"| {kind} CPU | {s['correct']}/16 | {s['mean_cost1']:.9f} | {len(s['rescues_vs_raw'])}/{len(s['breaks_vs_raw'])} |")
    for arm,data in output['internal'].items():
        for step,interventions in data['endpoints'].items():
            for name,s in interventions.items():
                lines.append(f"| {arm} step{step} {name} | {s['correct']}/16 | {s['mean_cost1']:.9f} | {len(s['rescues_vs_raw'])}/{len(s['breaks_vs_raw'])} |")
    lines+=['','CPU 外部头使用同16图2000次全批量更新；内部使用128次逐 query 更新。标签预算相同，优化成本和步数不同。',
            '这里回答是否学动、能否在 TRAIN 使用 M，不证明跨组泛化、超过完整 COST1 或必须改造编码器。','',
            '完整标量、梯度、checkpoint、每张图C128内容与127动作分数均保留在本目录。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',dict(status='TRAIN16_RESULT_ARTIFACTS_PASS',authority=bind(AUTH),
          result=bind(OUT/'result.json'),report=bind(OUT/'report_zh.md'),opened_probe_evaluations=0))
    emit(stage='summarize',status=output['status'])
    return True


def advance(a,stage,arm):
    """Only an engineering-qualified pilot can launch fixed-budget successors."""
    d=OUT/'dispatch';d.mkdir(parents=True,exist_ok=True)
    lock=(d/'advance.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    record=d/(stage+'_'+arm+'.json');state=read(record) if record.exists() else dict(complete=False,jobs=[])
    if state['complete']: return
    def submit(next_stage,next_arm='',dependencies=None):
        for job in state['jobs']:
            if job['stage']==next_stage and job['arm']==next_arm:return job['job_id']
        command=['sbatch','--parsable']
        if next_stage=='summarize':command+=['--partition=cpuonly','--gres=none','--mem=16G']
        else:command+=['--partition=accelerated']
        if dependencies:command+=['--dependency=afterok:'+':'.join(dependencies)]
        command += [str(LAUNCH),next_stage,next_arm]
        completed=subprocess.run(command,capture_output=True,text=True,timeout=45)
        need(completed.returncode==0,'SBATCH:'+completed.stderr)
        job=completed.stdout.strip().split(';')[0];need(job.isdigit(),'SBATCH_JOB_ID')
        state['jobs'].append(dict(stage=next_stage,arm=next_arm,job_id=job,command=command));write(record,state)
        return job
    if stage=='pilot':
        check=read(OUT/'pilot_validation.json')
        need(check['status']=='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS' and check['authority']==bind(AUTH),'QUALIFIED_PILOT')
        jobs=[submit('train',ar,[os.environ['SLURM_JOB_ID']]) for ar in ARMS]
        submit('summarize','',jobs)
    state['complete']=True;write(record,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','cpu','pilot','train','summarize','advance-pilot','advance-train','advance-cpu','advance-summarize'])
    p.add_argument('--arm',default='');p.add_argument('--budget',type=float,default=400)
    args=p.parse_args()
    if args.stage=='prepare':prepare();return
    a,m=guard(args.stage)
    if args.stage.startswith('advance-'):advance(a,args.stage[8:],args.arm);return
    if args.stage=='preflight':preflight(a,m);return
    need(read(OUT/'preflight.json')['status']=='JOINT_FULL_C128_VJP_AND_NO_DIRECT_M_PASS','PREFLIGHT_REQUIRED')
    started=time.monotonic()
    try:
        if args.stage=='cpu':done=cpu(a,m)
        elif args.stage=='summarize':done=summarize(a,m)
        else:done=train(a,m,args.arm or 'PRE_REAL',args.budget,pilot=args.stage=='pilot')
    finally:
        import torch
        write(OUT/'runtime'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
              dict(authority=bind(AUTH),stage=args.stage,arm=args.arm,seconds=time.monotonic()-started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
    if not done:sys.exit(75)


if __name__=='__main__':main()
