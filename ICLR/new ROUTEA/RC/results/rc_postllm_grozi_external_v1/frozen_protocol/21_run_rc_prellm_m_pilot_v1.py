#!/usr/bin/env python3
"""Small, fixed, real-M injection pilot; original full C128 and frozen COST1.

Query-conditioned pre-LLM and post-LLM controls share parameters and labels.
This is an opened-data development experiment, never an external HYP test.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[2]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
OUT = ROOT / 'results/rc_prellm_m_adapter_v1'
AUTH = ROOT / 'registry/rc_prellm_m_adapter_authority_v1_20260924.json'
PLAN = ROOT / 'plan/RC_COLNOMIC_INTERNAL_M_PILOT_V1_20260924.md'
LAUNCH = ROOT / 'slurm/rc_prellm_m_pilot_v1.sbatch'
ARMS = ('PRE_REAL', 'PRE_CONSTANT', 'POST_REAL', 'POST_CONSTANT')
MODEL = WORK / 'models/downloaded_models/colnomic-embed-multimodal-7b'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(value):
    need(bind(value['path']) == value, 'SHA_DRIFT:' + value['path'])
    return Path(value['path'])


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False, allow_nan=False), flush=True)


def prepare():
    need(not AUTH.exists(), 'EXISTING_AUTHORITY')
    manifest = read(OUT / 'input_manifest.json')
    need(len(manifest['train_rows']) == 16 and len(manifest['probe_rows']) == 8, 'PILOT_16_8')
    codes = [Path(__file__), ROOT / 'programs/rc_prellm_m_adapter_v1.py',
             ROOT / 'programs/prepare_rc_prellm_m_v1.py',
             ROOT / 'programs/rc_prellm_m_join_v1.py',
             ROOT / 'tests/test_prellm_m_adapter_v1.py', PLAN, LAUNCH]
    a = dict(status='INTERNAL_M_PILOT_AUTHORIZED', user_authorization='2026-09-24 开始第一种改造方式; 也加在其他地方做对照',
             code_sources=[bind(p) for p in codes], manifest=bind(OUT / 'input_manifest.json'),
             arms=list(ARMS), fold=0, train_count=16, probe_count=8, candidates=128,
             epochs=1, updates=16, seed=17, bottleneck=16, residual_scale=.1,
             learning_rate=3e-4, weight_decay=1e-3, clip_norm=1.,
             model=str(MODEL), conditioning='TRAIN-only log M standardized; constant z=0',
             selection='Fixed terminal update; no probe model selection',
             dtype=dict(backbone='bfloat16', adapter='float32', scoring='float64'),
             attention='sdpa', use_fast=False, max_requeues=16,
             output_head='frozen COST1_REFIT_M1Q0R0, original fold0 full TRAIN',
             source_parity=dict(mean_cosine_min=.999, max_content_error=.005, action_exact=True),
             cache_parity_atol=1e-6, new_roma_forwards=0,
             scope='16 TRAIN adapter feasibility and 8 opened outer-held probes; not generalization confirmation')
    write(AUTH, a)
    emit(stage='prepare', authority=bind(AUTH), arms=ARMS)


def guard(stage):
    a = read(AUTH)
    for b in a['code_sources']:
        checked(b)
    m = read(checked(a['manifest']))
    if stage not in ('preflight',):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    # Labels are permitted only for TRAIN during fit, and for join after seal.
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(s in p for s in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/isic/')), 'PROTECTED_READ')
        if 'curator_roles' in p or '/target_join/' in p:
            need(stage == 'join' and (OUT / 'all_predictions_prelabel_seal.json').exists(), 'HELD_LABEL_READ_BEFORE_SEAL')
    sys.addaudithook(audit)
    return a, m


def torch_setup(a):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(a['seed']); np.random.seed(a['seed']); random.seed(a['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


def load_model(a):
    import torch
    from rc_prellm_m_adapter_v1 import load_frozen_colnomic
    need(torch.cuda.is_available(), 'GPU_REQUIRED')
    start = time.monotonic()
    model, processor, loading = load_frozen_colnomic(a['model'],'cuda',attention=a['attention'],
                         use_fast=a['use_fast'],verify_weight_hashes=False)
    write(OUT/'loading'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
          dict(authority=bind(AUTH),report=loading))
    need(not any(p.requires_grad for p in model.parameters()), 'FROZEN_BACKBONE')
    need(any('lora_' in n for n, _ in model.named_parameters()), 'COLNOMIC_RETRIEVAL_LORA_REQUIRED')
    emit(event='model_loaded', seconds=time.monotonic()-start, gpu=torch.cuda.get_device_name(),
         attention=a['attention'], parameters=sum(p.numel() for p in model.parameters()))
    return model, processor


def verify_model_sources(m):
    """Hash weights once in the GPU batch, then reject any filesystem drift."""
    p=OUT/'model_source_validation.json'
    def signature(path):
        s=Path(path).stat()
        return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    if p.exists():
        prior=read(p);need(prior['authority']==bind(AUTH),'MODEL_SOURCE_AUTH')
        for x in prior['files']:
            need(signature(x['binding']['path'])==x['stat'],'MODEL_SOURCE_STAT_DRIFT')
        return
    files=[]
    for b in m['encoder']['model_sources'].values():
        path=checked(b);files.append(dict(binding=b,stat=signature(path)))
    write(p,dict(status='MODEL_SOURCE_HASHES_PASS',authority=bind(AUTH),files=files))


def adapter(a, m, arm):
    import torch
    from rc_prellm_m_adapter_v1 import QualityResidualAdapter
    torch.manual_seed(a['seed'])
    return QualityResidualAdapter(hidden_size=3584, bottleneck=a['bottleneck'],
        conditioning='real' if arm.endswith('REAL') else 'constant',
        mass_log_mean=m['mass_normalization']['log_mean'],
        mass_log_std=m['mass_normalization']['log_std'], residual_scale=a['residual_scale']).to('cuda')


def state_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def head_inputs(row, mass, content):
    import torch
    raw = torch.as_tensor(row['raw_scores'], dtype=torch.float64, device=content.device)
    winner = row['winner_index']
    idx = [i for i in range(128) if i != winner]
    def sym(x):
        return (x[idx] - x[winner]) / (x[idx].abs() + x[winner].abs() + 1e-12)
    return torch.stack(((raw[idx] - raw[winner])/raw.std(unbiased=False).clamp_min(1e-12),
                        sym(mass*content), sym(mass), sym(content)), dim=1)


def logits(row, content, m):
    import torch
    mass = torch.as_tensor(row['M'], dtype=torch.float64, device=content.device)
    theta = torch.as_tensor(m['frozen_head']['theta'], dtype=torch.float64, device=content.device)
    return head_inputs(row, mass, content) @ theta[:4] + theta[4]


def cost(row, content, m):
    import torch
    from torch.nn import functional as F
    z = logits(row, content, m)
    idx = [i for i in range(128) if i != row['winner_index']]
    targets = row['target_positions']
    need(len(targets) == 1, 'PILOT_SINGLE_IDENTITY_TARGET')
    t = targets[0]
    if t == row['winner_index']:
        return F.softplus(z.amax())
    pos = idx.index(t)
    wrong = z.clone(); wrong[pos] = -torch.inf
    return F.softplus(-z[pos]) + F.softplus(wrong.amax())


def decision(row, content, m):
    import torch
    z = logits(row, torch.as_tensor(content, dtype=torch.float64), m).detach().cpu()
    idx = [i for i in range(128) if i != row['winner_index']]
    pos = idx[int(z.argmax())] if float(z.max()) > 0 else row['winner_index']
    return dict(prediction_position=pos, prediction_id=row['candidate_ids'][pos],
                switched=pos != row['winner_index'], challenger_positions=idx, logits=z.tolist())


def old_tokens(binding):
    import torch
    binding=binding.get('token_file', binding)
    p = torch.load(checked(binding), map_location='cpu', weights_only=True)
    return p['tokens']


def references(row, device='cuda'):
    import torch
    from torch.nn import functional as F
    return [F.normalize(old_tokens(b).to(device=device, dtype=torch.float64), dim=-1)
            for b in row['reference_tokens']]


def content_score(tokens, ref):
    import torch
    from torch.nn import functional as F
    q = F.normalize(tokens.to(dtype=torch.float64), dim=-1)
    return (q @ ref.T).max(dim=1).values.mean()


def qcache(row, device='cuda'):
    import torch
    d = OUT / 'encoder_cache' / row['query_id']
    v = read(d / 'validation.json')
    need(v['authority'] == bind(AUTH) and v['status'] == 'QUERY_ENCODER_CACHE_PASS', 'QUERY_CACHE')
    x = torch.load(checked(v['payload']), map_location=device, weights_only=True)
    return x


def predict_tokens(model, cache, small, mass, location):
    from rc_prellm_m_adapter_v1 import conditioned_colnomic_forward, project_cached_hidden
    if location == 'prellm':
        result = conditioned_colnomic_forward(model, cache['batch'], small, [mass],
                    cached_visual=cache['merged'], location=location)
    else:
        result = project_cached_hidden(model, cache['hidden'], cache['batch'], small, [mass])
    return result[cache['image_mask']]


def encoder_cache(a, m, budget):
    import torch
    from PIL import Image, ImageOps
    from torch.nn import functional as F
    from rc_prellm_m_adapter_v1 import (cache_merged_visual_tokens, capture_frozen_hidden,
        conditioned_colnomic_forward, project_cached_hidden)
    started = time.monotonic(); model, processor = load_model(a)
    validations = []
    for row in m['train_rows'] + m['probe_rows']:
        d = OUT / 'encoder_cache' / row['query_id']
        if (d / 'validation.json').exists():
            v = read(d / 'validation.json'); checked(v['payload'])
            need(v['authority'] == bind(AUTH), 'CACHE_AUTH'); validations.append(bind(d / 'validation.json')); continue
        if time.monotonic() - started > budget-90:
            return False
        need(bind(row['image_path'])['sha256']==row['source_image_sha256'],'SOURCE_IMAGE_SHA')
        with Image.open(row['image_path']) as im:
            if row['frame'] in ('EXIF_TRANSPOSED','EXIF_NORMALIZED','EXIF_ORIENTED_BEFORE_RESIZE'):
                im = ImageOps.exif_transpose(im)
            else:
                need(row['frame'] == 'DECODED_RAW_BEFORE_EXIF', 'UNKNOWN_IMAGE_FRAME')
            batch = processor.process_images([im.convert('RGB')]).to('cuda')
        with torch.no_grad():
            native = model(**batch)
            merged = cache_merged_visual_tokens(model, batch)
            hidden = capture_frozen_hidden(model, batch, cached_visual=merged)
            replay = conditioned_colnomic_forward(model, batch, None, [row['M'][0]], cached_visual=merged)
            projected = project_cached_hidden(model, hidden, batch, None, [row['M'][0]])
        image_mask = batch['input_ids'].eq(model.config.image_token_id)
        need(int(image_mask.sum()) == merged.shape[0], 'IMAGE_TOKEN_AXIS')
        error = max(float((native-replay).abs().max()), float((native-projected).abs().max()))
        need(error <= a['cache_parity_atol'], f'CACHE_FORWARD_PARITY:{error}')
        prior = old_tokens(row['query_tokens']).to('cuda').float()
        current = native[image_mask].float()
        need(prior.shape == current.shape, 'ORIGINAL_QUERY_TOKEN_SHAPE')
        cosine = float(F.cosine_similarity(prior, current, dim=-1).mean())
        refs = references(row)
        with torch.no_grad():
            prior_l = torch.stack([content_score(prior, r) for r in refs]).cpu()
            fresh_l = torch.stack([content_score(current, r) for r in refs]).cpu()
        need(float((prior_l-torch.tensor(row['L0'], dtype=torch.float64)).abs().max()) < 2e-10, 'ORIGINAL_CONTENT_REPLAY')
        drift = float((fresh_l-prior_l).abs().max())
        parity = dict(query_id=row['query_id'], cache_forward_error=error, token_mean_cosine=cosine,
            content_max_drift=drift, fresh_L0=fresh_l.tolist(), original_L0=prior_l.tolist(),
            original=decision(row, prior_l, m), fresh=decision(row, fresh_l, m))
        write(d / 'parity.json', parity)
        need(cosine >= a['source_parity']['mean_cosine_min'], 'SOURCE_TOKEN_PARITY')
        need(drift <= a['source_parity']['max_content_error'], 'SOURCE_CONTENT_PARITY')
        need(parity['original']['prediction_position'] == parity['fresh']['prediction_position'], 'SOURCE_ACTION_PARITY')
        checks=[]
        for arm in ARMS:
            small = adapter(a, m, arm)
            for mass in (min(row['M']), max(row['M'])):
                with torch.no_grad():
                    encoded = predict_tokens(model, dict(batch=batch, merged=merged, hidden=hidden,
                                            image_mask=image_mask), small, mass, 'prellm' if arm.startswith('PRE') else 'postllm')
                e=float((encoded-native[image_mask]).abs().max()); need(e <= a['cache_parity_atol'], 'ZERO_ADAPTER_PARITY')
                checks.append(dict(arm=arm, mass=mass, error=e))
        # Validate the actual 7B backward on the largest TRAIN input before dispatch.
        largest=max(m['train_rows'], key=lambda r:r['query_tokens']['token_shape'][0])['query_id']
        if row['query_id']==largest:
            gradient_checks=[]
            for location in ('prellm','postllm'):
                trial=adapter(a,m,'PRE_REAL'); trial.zero_grad(set_to_none=True)
                output=predict_tokens(model,dict(batch=batch,merged=merged,hidden=hidden,image_mask=image_mask),
                                      trial,row['M'][0],location)
                (-content_score(output,refs[0])).backward()
                grads=[p.grad for p in trial.parameters() if p.grad is not None]
                need(grads and all(torch.isfinite(g).all() for g in grads),'ACTUAL_BACKWARD_FINITE')
                norm=float(sum(g.double().square().sum() for g in grads).sqrt())
                need(norm>0 and not any(p.grad is not None for p in model.parameters()),'ACTUAL_BACKWARD_FROZEN')
                gradient_checks.append(dict(location=location,gradient_norm=norm,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),image_tokens=int(image_mask.sum())))
                del output,trial,grads
            write(OUT/'actual_model_gradient_validation.json',dict(status='ACTUAL_7B_BOTH_LOCATIONS_GRAD_PASS',
                  authority=bind(AUTH),query_id=row['query_id'],checks=gradient_checks,
                  fitting_updates=0,diagnostic='Negative full-reference content for fixed position0; trial modules discarded'))
        cache = dict(authority=bind(AUTH), query_id=row['query_id'],
            batch={k:v.detach().cpu() for k,v in batch.items() if k!='pixel_values'},
            merged=merged.cpu(), hidden=hidden.cpu(), image_mask=image_mask.cpu(),
            native_tokens=native.cpu(), fresh_L0=fresh_l, original_L0=prior_l)
        save(d/'payload.pt', cache)
        write(d/'validation.json', dict(status='QUERY_ENCODER_CACHE_PASS', authority=bind(AUTH),
            payload=bind(d/'payload.pt'), parity=bind(d/'parity.json'), zero_adapter=checks,
            image_sha256=bind(row['image_path'])['sha256'], held_labels_read=False))
        validations.append(bind(d/'validation.json'))
        emit(stage='cache', query_id=row['query_id'], done=len(validations), total=a['train_count']+a['probe_count'], cosine=cosine, drift=drift)
        del refs, native, merged, hidden, batch, small; torch.cuda.empty_cache()
    need(read(OUT/'actual_model_gradient_validation.json')['status']=='ACTUAL_7B_BOTH_LOCATIONS_GRAD_PASS','ACTUAL_BACKWARD_REQUIRED')
    write(OUT/'encoder_cache/validation.json', dict(status='ENCODER_CACHE_PASS', authority=bind(AUTH),
        queries=validations, versions={k:importlib.metadata.version(k) for k in ('torch','transformers','colpali-engine','peft')},
        seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name()))
    return True


def score_all(model, cache, small, row, refs, location, values, start_time, budget, persist, shuffled=False):
    import torch
    constant = small.conditioning == 'constant'
    masses = row['M'][1:] + row['M'][:1] if shuffled else row['M']
    with torch.no_grad():
        common = predict_tokens(model, cache, small, masses[0], location) if constant else None
        for i in range(len(values), 128):
            if time.monotonic()-start_time > budget:
                persist(values); return False
            z = common if constant else predict_tokens(model, cache, small, masses[i], location)
            values.append(float(content_score(z, refs[i])))
            if (i+1) % 16 == 0:
                persist(values)
    persist(values)
    return True


def fit(a, m, arm, budget):
    import torch
    started=time.monotonic(); d=OUT/arm; d.mkdir(parents=True, exist_ok=True)
    lock=(d/'worker.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (d/'fit_validation.json').exists():
        return True
    need(read(OUT/'encoder_cache/validation.json')['status']=='ENCODER_CACHE_PASS','ENCODER_CACHE_REQUIRED')
    model, _ = load_model(a); small=adapter(a,m,arm)
    opt=torch.optim.AdamW(small.parameters(), lr=a['learning_rate'], weight_decay=a['weight_decay'])
    location='prellm' if arm.startswith('PRE') else 'postllm'
    cp=d/'checkpoint.pt'; step=0; history=[]; pending=[]
    if cp.exists():
        x=torch.load(cp,map_location='cpu',weights_only=True)
        need(x['authority']==bind(AUTH) and x['arm']==arm, 'TRAIN_RESUME')
        small.load_state_dict(x['adapter']); opt.load_state_dict(x['optimizer'])
        step=x['step']; history=x['history']; pending=x['pending']
    def checkpoint(values):
        save(cp,dict(authority=bind(AUTH),arm=arm,adapter=state_cpu(small),optimizer=opt.state_dict(),
            step=step,history=history,pending=values))
    while step<a['updates']:
        row=m['train_rows'][step % len(m['train_rows'])]
        cache=qcache(row); refs=references(row)
        if not score_all(model,cache,small,row,refs,location,pending,started,budget,checkpoint):
            return False
        content=torch.tensor(pending, dtype=torch.float64, device='cuda', requires_grad=True)
        loss=cost(row, content, m); derivative=torch.autograd.grad(loss,content)[0].detach()
        active=derivative.nonzero().flatten().tolist()
        need(active, 'COST1_VJP_SUPPORT')
        # amax splits derivatives across exact ties; retain every nonzero entry.
        # Always finish an update atomically; a timeout replays from its preceding checkpoint.
        if time.monotonic()-started>budget-35:
            checkpoint(pending); return False
        opt.zero_grad(set_to_none=True); grad_start=time.monotonic(); replay_error=0.
        for i in active:
            output=predict_tokens(model,cache,small,row['M'][i],location)
            v=content_score(output,refs[i])
            replay_error=max(replay_error,abs(float(v.detach())-pending[i]))
            need(replay_error<2e-8, 'NO_GRAD_GRAD_FORWARD_MISMATCH')
            (v*derivative[i]).backward()
        grads=[p.grad for p in small.parameters() if p.grad is not None]
        need(grads and all(torch.isfinite(g).all() for g in grads), 'FINITE_ADAPTER_GRAD')
        norm=float(torch.nn.utils.clip_grad_norm_(small.parameters(),a['clip_norm']))
        need(norm>0, 'NONZERO_ADAPTER_GRAD')
        need(not any(p.grad is not None for p in model.parameters()),'FROZEN_BACKBONE_GRAD')
        before=state_cpu(small); previous_output=output.detach().clone(); opt.step()
        changed=max(float((small.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in before.items())
        need(changed>0,'ADAPTER_CHANGED')
        record=dict(step=step,query_id=row['query_id'],loss=float(loss.detach()),gradient_norm=norm,
            active_candidates=active,content=pending,decision=decision(row,pending,m),
            derivative=derivative.cpu().tolist(),gradient_forward_error=replay_error,
            backward_seconds=time.monotonic()-grad_start,parameter_change_max=changed,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if step in (0,a['updates']-1):
            with torch.no_grad():
                updated=predict_tokens(model,cache,small,row['M'][active[-1]],location)
                delta=updated.float()-previous_output.float()
                record['token_max_change']=float(delta.abs().max())
                record['token_changed_fraction']=float((delta!=0).float().mean())
                source=cache['merged'] if location=='prellm' else cache['hidden'][cache['image_mask']]
                changed_source=small(source,row['M'][active[-1]])
                save(d/'traces'/f'step{step:03d}.pt',dict(query_id=row['query_id'],candidate_position=active[-1],
                    M=row['M'][active[-1]],location=location,source=source.cpu(),modulated_source=changed_source.cpu(),
                    before_tokens=previous_output.cpu(),after_tokens=updated.cpu(),image_mask=cache['image_mask'].cpu()))
        write(d/'steps'/f'{step:03d}.json',record)
        history.append({k:v for k,v in record.items() if k not in ('content','decision','derivative')})
        step+=1; pending=[]; checkpoint(pending)
        emit(stage='fit',arm=arm,step=step,total=a['updates'],loss=record['loss'],gradient_norm=norm)
        del refs, cache, output, v, content, loss, previous_output; torch.cuda.empty_cache()
        if time.monotonic()-started>budget:
            return False
    save(d/'adapter.pt',dict(authority=bind(AUTH),adapter=state_cpu(small),arm=arm,step=step))
    write(d/'fit_validation.json',dict(status='FIXED_TRAIN_UPDATES_COMPLETE',authority=bind(AUTH),
        adapter=bind(d/'adapter.pt'),checkpoint=bind(cp),steps=step,history=history,
        backbone_trainable=0,head_trainable=0,adapter_parameters=sum(p.numel() for p in small.parameters()),
        held_label_reads=0,training_selection='Fixed update16; no selection by held probe'))
    return True


def predict(a,m,arm,budget):
    import torch
    started=time.monotonic(); d=OUT/arm; model,_=load_model(a); small=adapter(a,m,arm)
    fitinfo=read(d/'fit_validation.json'); model_binding=fitinfo['adapter']
    small.load_state_dict(torch.load(checked(model_binding),weights_only=True,map_location='cpu')['adapter'])
    location='prellm' if arm.startswith('PRE') else 'postllm'; seals=[]
    for split,rows in (('train',m['train_rows']),('probe',m['probe_rows'])):
        for row in rows:
            cache=qcache(row); refs=references(row)
            for intervention in (('native','shuffled') if arm.endswith('REAL') else ('native',)):
                p=d/'predictions'/split/(row['query_id']+'_'+intervention+'.json')
                if p.exists():
                    prior=read(p); need(prior['adapter']==model_binding,'PREDICTION_ADAPTER');seals.append(bind(p));continue
                partial=p.with_suffix('.partial.json'); values=[]
                if partial.exists():
                    prior=read(partial);need(prior['adapter']==model_binding,'PARTIAL_ADAPTER');values=prior['L']
                def persist(v):write(partial,dict(adapter=model_binding,L=v))
                if not score_all(model,cache,small,row,refs,location,values,started,budget,persist,intervention=='shuffled'):
                    return False
                write(p,dict(authority=bind(AUTH),adapter=model_binding,query_id=row['query_id'],
                    arm=arm,intervention=intervention,L=values,M=row['M'],raw_scores=row['raw_scores'],
                    candidate_ids=row['candidate_ids'],decision=decision(row,values,m),
                    source_baseline=decision(row,row['L0'],m),fresh_baseline=decision(row,cache['fresh_L0'].cpu(),m),
                    held_label_reads=0))
                seals.append(bind(p));emit(stage='predict',arm=arm,split=split,query_id=row['query_id'],intervention=intervention)
            del refs,cache;torch.cuda.empty_cache()
    write(d/'prediction_seal.json',dict(status='PREDICTIONS_SEALED',authority=bind(AUTH),adapter=model_binding,predictions=seals,held_label_reads=0))
    return True


def preflight(a,m):
    import torch
    # Exact full-C128 scalar VJP versus ordinary autograd, both HOLD and SWITCH labels.
    generator=torch.Generator().manual_seed(73); tests=[]
    for row in m['train_rows']:
        q=torch.randn(128,4,generator=generator,dtype=torch.float64)
        w=torch.randn(4,generator=generator,dtype=torch.float64,requires_grad=True)
        c=(q@w).sigmoid(); ordinary=cost(row,c,m)
        direct=torch.autograd.grad(ordinary,w)[0]
        wd=w.detach().clone().requires_grad_(); detached=(q@wd).sigmoid().detach().requires_grad_()
        dl=torch.autograd.grad(cost(row,detached,m),detached)[0]
        for i in dl.nonzero().flatten().tolist():
            (((q[i]@wd).sigmoid())*dl[i]).backward()
        err=float((direct-wd.grad).abs().max());need(err<1e-11,'VJP_EQUIVALENCE')
        tests.append(dict(query_id=row['query_id'],gradient_error=err,active_count=int((dl!=0).sum())))
    write(OUT/'preflight.json',dict(status='FULL_C128_VJP_PASS',authority=bind(AUTH),tests=tests))
    emit(stage='preflight',status='FULL_C128_VJP_PASS')


def advance(stage,arm):
    # Called only after successful completion; all downstream workers validate authority.
    d=OUT/'dispatch'; d.mkdir(parents=True,exist_ok=True)
    lock=(d/'advance.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    key=d/(stage+'_'+arm+'.json')
    prior=read(key) if key.exists() else dict(jobs=[],complete=False)
    if prior['complete']:return
    jobs=prior['jobs']
    def submit(st, ar='', dependency=None):
        existing=[x for x in jobs if x['stage']==st and x['arm']==ar]
        if existing:return existing[0]['job_id']
        cmd=['sbatch','--parsable','--partition=accelerated']
        if dependency:cmd+=['--dependency=afterok:'+dependency]
        cmd += [str(LAUNCH),st,ar]
        out=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        need(out.returncode==0,'SBATCH_FAILED:'+out.stderr)
        job=out.stdout.strip().split(';')[0];need(job.isdigit(),'SBATCH_JOB_ID')
        jobs.append(dict(stage=st,arm=ar,job_id=job,command=cmd));write(key,dict(jobs=jobs,complete=False))
        return job
    if stage=='cache':
        for name in ARMS:
            fitjob=submit('fit',name,os.environ['SLURM_JOB_ID'])
            submit('predict',name,fitjob)
    elif stage=='predict' and all((OUT/x/'prediction_seal.json').exists() for x in ARMS):
        # Join only after every arm is fixed and sealed; it has no GPU need.
        all_seals=[bind(OUT/x/'prediction_seal.json') for x in ARMS]
        write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),seals=all_seals,held_label_reads=0))
        cmd=['sbatch','--parsable','--partition=cpuonly','--gres=none','--mem=16G',str(LAUNCH),'join','']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        need(result.returncode==0,'JOIN_SBATCH:'+result.stderr)
        jobs.append(dict(stage='join',arm='',job_id=result.stdout.strip().split(';')[0],command=cmd))
    write(key,dict(jobs=jobs,complete=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','cache','fit','predict','join','advance-cache','advance-fit','advance-predict','advance-join'])
    p.add_argument('--arm',default='');p.add_argument('--budget',type=float,default=430)
    args=p.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage.startswith('advance-'):advance(args.stage[8:],args.arm);return
    a,m=guard(args.stage);torch_setup(a)
    if args.stage=='preflight':preflight(a,m);return
    need(read(OUT/'preflight.json')['status']=='FULL_C128_VJP_PASS','PREFLIGHT')
    if args.stage=='join':
        from rc_prellm_m_join_v1 import join
        join(a,m);return
    verify_model_sources(m)
    started=time.monotonic()
    try:
        if args.stage=='cache':done=encoder_cache(a,m,args.budget)
        else:
            need(args.arm in ARMS,'ARM')
            done=(fit if args.stage=='fit' else predict)(a,m,args.arm,args.budget)
    finally:
        import torch
        write(OUT/'runtime'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
              dict(stage=args.stage,arm=args.arm,seconds=time.monotonic()-started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated(),authority=bind(AUTH)))
    if not done:sys.exit(75)


if __name__=='__main__':main()
