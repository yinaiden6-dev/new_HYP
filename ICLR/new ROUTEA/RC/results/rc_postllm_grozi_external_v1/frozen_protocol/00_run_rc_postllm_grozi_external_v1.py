#!/usr/bin/env python3
"""Frozen H593 fold0 POST_REAL transfer to the existing GroZi480 natural C128.

One missing query encoder forward is cached; references and RoMa are reused.
No external labels, gradients, fitting, candidate changes or threshold choice.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import cache_rc_postllm_h593_v1 as C
import run_rc_postllm_h593_v1 as H
import run_rc_simple_external_replay_v1 as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_postllm_grozi_external_v1'
AUTH = ROOT / 'registry/rc_postllm_grozi_external_authority_v1_20260925.json'
read, write, bind, checked, need = C.read, C.write, C.bind, C.checked, C.need
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def guard():
    a = read(AUTH)
    for b in a['code_sources']:
        checked(b)
    for k in ('source_authority', 'source_snapshot', 'source_fit', 'external_authority',
              'workers', 'legacy_gallery', 'gallery_append'):
        checked(a[k])
    for b in a['source_heads'].values():
        checked(b)
    need(a['source_fold'] == 0 and a['training_updates'] == 0 and a['switch_threshold'] == 0,
         'FROZEN_TRANSFER_PROTOCOL')
    external = read(a['external_authority']['path'])
    need(external['sources']['worker'] == a['workers'], 'EXTERNAL_WORKER_SOURCE')
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        need(not any(x in s for x in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', 'formal392')),
             'FORBIDDEN_LABEL_OR_PROTECTED_PATH')
        need(p != Path(external['curator_after_all_seals']['path']).resolve(), 'EXTERNAL_LABEL_READ')
        need(not (p.name in ('result.json', 'result_validation.json') and
                  E.DATA['grozi']['root'] in p.parents), 'NO_OLD_RESULTS_DURING_INFERENCE')
    sys.addaudithook(audit)
    return a, bind(AUTH)


def load_runtime(a, device):
    import torch
    from rc_postllm_m_backend_v1 import load_projection
    parent = read(a['source_authority']['path'])
    state = torch.load(a['source_snapshot']['path'], map_location='cpu', weights_only=True)
    fit = read(a['source_fit']['path'])
    need(fit['snapshot'] == a['source_snapshot'] and fit['steps'] == state['step'] == 3656,
         'SOURCE_FIXED_EIGHT_PASS_ENDPOINT')
    need(state['authority'] == a['source_authority'] and state['arm'] == 'POST_REAL', 'SOURCE_STATE')
    ctx = {'config': parent, 'normalization': state['normalization']}
    small = H.adapter(ctx, 'POST_REAL', device)
    small.load_state_dict(state['adapter']); small.eval().requires_grad_(False)
    model = load_projection(parent, device=device)
    heads = {k: read(v['path'])['theta'] for k, v in a['source_heads'].items()}
    return dict(parent=parent, small=small, projection=model, state=state, heads=heads)


def make_row(q, r, labels):
    axis = q['candidate_physical_rows']
    w = axis.index(q['raw_ranked_physical_rows'][0])
    return dict(query_id=q['query_id'], candidate_ids=axis,
        candidate_identities=[labels[i] for i in axis], winner_index=w,
        challenger_positions=[i for i in range(128) if i != w],
        raw_scores=q['candidate_raw_scores'].double().tolist(),
        M=[float(c['old_scores']['visibility_mass']) for c in r['candidates']])


def cache_existing(q, row, ab):
    import torch
    p = OUT/'encoder_cache'/q['query_id']/'validation.json'
    if not p.exists():
        return None
    v = read(p)
    need(v['status'] == 'GROZI_POSTLLM_QUERY_CACHE_PASS' and v['authority'] == ab,
         'CACHE_AUTHORITY')
    need(v['query_id'] == q['query_id'] and v['image_sha256'] == q['query_source_sha256']
         and v['candidate_ids'] == row['candidate_ids']
         and v['old_query_tokens_sha256'] == q['query_tokens_sha256'], 'CACHE_QUERY_LINEAGE')
    checked(v['parity'])
    payload = torch.load(checked(v['payload']), map_location='cpu', weights_only=True)
    C.tensor_contract(payload, row)
    return payload, bind(p)


def collect_query(q, row, w, refs, encoder, processor, runtime, a, ab):
    import torch
    from PIL import Image, ImageOps
    from torch.nn import functional as F
    from rc_prellm_m_adapter_v1 import find_colnomic_base
    from rc_postllm_m_backend_v1 import validate_cached_projection
    old = cache_existing(q, row, ab)
    if old is not None:
        return old
    started = time.monotonic()
    need(C.sha(w['image_path']) == q['query_source_sha256'], 'QUERY_IMAGE_SHA')
    with Image.open(w['image_path']) as image:
        batch = processor.process_images([ImageOps.exif_transpose(image).convert('RGB')]).to('cuda')
    base = find_colnomic_base(encoder); captured = []
    handle = base.custom_text_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            native = encoder(**batch)
    finally:
        handle.remove()
    need(len(captured) == 1, 'ONE_FULL_HIDDEN_CAPTURE')
    cache = dict(query_id=q['query_id'], hidden=captured.pop(),
        image_mask=batch['input_ids'].eq(base.config.image_token_id), native_tokens=native,
        batch={k:v for k,v in batch.items() if k != 'pixel_values'})
    parity = validate_cached_projection(runtime['projection'], cache, atol=0.)
    need(parity['exact_equal'], 'UNMERGED_GPU_PROJECTION_EXACT_PARITY')
    current = native[cache['image_mask']]
    original = q['query_tokens'].to('cuda')
    need(original.shape == current.shape, 'HISTORICAL_QUERY_SHAPE')
    exact = bool(torch.equal(current.float().half(), original.half()))
    # Freeze the same historical-token equality gate as H593. Do not adjust on outcomes.
    need(exact, 'HISTORICAL_FP16_QUERY_TOKEN_EQUALITY')
    oldunit = F.normalize(original.double(), dim=-1)
    freshunit = F.normalize(current.double(), dim=-1)
    oldL = torch.stack([(oldunit @ ref.T).max(1).values.mean() for ref in refs])
    freshL = torch.stack([(freshunit @ ref.T).max(1).values.mean() for ref in refs])
    drift = float((oldL-freshL).abs().max())
    need(drift <= a['source_parity']['max_content_error'], 'CONTENT_DRIFT')
    payload = {k:({kk:vv.detach().cpu() for kk,vv in v.items()} if k=='batch'
                   else v.detach().cpu() if torch.is_tensor(v) else v) for k,v in cache.items()}
    payload.update(fresh_L0=freshL.cpu(), original_L0=oldL.cpu(), authority=ab)
    C.tensor_contract(payload, row)
    dest = OUT/'encoder_cache'/q['query_id'];dest.mkdir(parents=True, exist_ok=True)
    H.save(dest/'payload.pt', payload)
    parity.update(historical_fp16_exact=exact, content_max_drift=drift,
        source_query_tokens_sha256=q['query_tokens_sha256'], image_frame='EXIF_ORIENTED_BEFORE_RESIZE',
        query_encoder_forwards=1, reference_encoder_forwards=0, roma_forwards=0, held_label_reads=0)
    write(dest/'parity.json', parity)
    validation = dict(status='GROZI_POSTLLM_QUERY_CACHE_PASS', authority=ab, query_id=q['query_id'],
        image_sha256=q['query_source_sha256'], old_query_tokens_sha256=q['query_tokens_sha256'],
        candidate_ids=row['candidate_ids'], payload=bind(dest/'payload.pt'), parity=bind(dest/'parity.json'),
        seconds=time.monotonic()-started, external_training_updates=0, held_label_reads=0)
    write(dest/'validation.json', validation)
    H.emit(stage='external_query_cache', query_id=q['query_id'], seconds=validation['seconds'], parity=True)
    return payload, bind(dest/'validation.json')


def predict_query(q, row, cache, cv, refs, runtime, sources, shard, a, ab, start, budget):
    import torch
    from rc_postllm_m_backend_v1 import predict_projection, validate_cached_projection
    dest=OUT/'predictions'/(q['query_id']+'.json')
    if dest.exists():
        prev=read(dest)
        need(prev['authority']==ab and prev['source_snapshot']==a['source_snapshot']
             and prev['row']==row and prev['cache_validation']==cv, 'RESUME_PREDICTION_SOURCE')
        return bind(dest)
    cache=C.cuda_payload(cache)
    parity=validate_cached_projection(runtime['projection'],cache,atol=0.)
    need(parity['exact_equal'], 'REPLAY_GPU_CACHE_PARITY')
    small=runtime['small'];model=runtime['projection']
    partial=dest.with_suffix('.partial.json')
    models={};values={}
    if partial.exists():
        prev=read(partial)
        need(prev['authority']==ab and prev['source_snapshot']==a['source_snapshot'] and prev['row']==row
             and prev['cache_validation']==cv and prev['input_sources']==sources,
             'PARTIAL_SCOPE')
        models,values=prev['models'],prev['values']
    def persist():
        write(partial,dict(authority=ab,source_snapshot=a['source_snapshot'],row=row,
            cache_validation=cv,input_sources=sources,models=models,values=values))
    modes=[('POST_REAL','native'),('POST_REAL_CONSTANT','constant'),('POST_REAL_SHUFFLED','shuffled')]
    with torch.no_grad():
        for name,mode in modes:
            if name in models:
                continue
            masses,constant=H.row_condition(row,'POST_REAL',mode)
            small.conditioning='constant' if constant else 'real'
            pending=values.setdefault(name,[]);H.V.validate_pending(pending)
            common=predict_projection(model,cache,small,masses[0]) if constant else None
            for i in range(len(pending),128):
                if STOP or time.monotonic()-start > budget-15:
                    persist();return None
                tokens=common if constant else predict_projection(model,cache,small,masses[i])
                pending.append(float(H.V.OLD.content_score(tokens,refs[i])))
            models[name]=dict(kind='INTERNAL3',L=list(pending),
                decision=H.V.choose(row,pending,runtime['state']['head']))
            persist()
        small.conditioning='real'
        fresh=cache['fresh_L0'].cpu().double()
        for name,kind in [('NO_ADAPTER_INTERNAL3','INTERNAL3'),('EXTERNAL_ADDITIVE4','ADDITIVE4'),
                          ('EXTERNAL_PRODUCT5','PRODUCT5')]:
            models[name]=dict(kind=kind,L=fresh.tolist(),
                decision=H.V.choose(row,fresh,runtime['heads'][kind],kind))
    write(dest,dict(status='GROZI_POSTLLM_FROZEN_QUERY_PASS',authority=ab,source_snapshot=a['source_snapshot'],
        query_id=q['query_id'],shard=shard,row=row,raw_ranked_physical_rows=q['raw_ranked_physical_rows'],
        models=models,cache_validation=cv,input_sources=sources,external_training_updates=0,
        held_label_reads=0,direct_M_in_internal_head=False,projection_parity=parity))
    H.emit(stage='external_prediction',query_id=q['query_id'],models=len(models))
    return bind(dest)


def worker(task, pilot, budget):
    import torch
    from torch.nn import functional as F
    need(os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'GPU_BATCH_REQUIRED')
    a,ab=guard();start=time.monotonic()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    shards=[0] if pilot else list(range(1+task,60,a['workers_parallel']))
    need(pilot or 0<=task<a['workers_parallel'], 'TASK_RANGE')
    runtime=load_runtime(a,'cuda');encoder=processor=None
    # Public gallery identity metadata are distinct from query target labels.
    gallery=read(a['legacy_gallery']['path'])['records']+read(a['gallery_append']['path'])['records']
    need([g['physical_row'] for g in gallery]==list(range(5533)), 'PUBLIC_GALLERY_AXIS')
    labels=[g['identity'] for g in gallery]
    C.OUT=OUT
    for shard in shards:
        directory=OUT/'shards'/f'{shard:02d}';directory.mkdir(parents=True,exist_ok=True)
        lock=(directory/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seal=directory/'validation.json'
        if seal.exists():
            v=read(seal);need(v['authority']==ab and v['status']=='GROZI_POSTLLM_SHARD_PASS','SHARD_SCOPE')
            for b in v['queries']:checked(b)
            lock.close();continue
        if STOP or time.monotonic()-start>budget-30:return False
        loaded,sources,workers=E.source_envelope('grozi',shard)
        raw,roma=loaded['raw'],loaded['roma']
        E.axes_and_integrity('grozi',shard,raw,roma,sources,workers)
        queries=[]
        for q,r,w in zip(raw['records'],roma['records'],workers):
            if STOP or time.monotonic()-start>budget-20:return False
            row=make_row(q,r,labels)
            refs=[F.normalize(raw['references'][i]['tokens'].to(device='cuda',dtype=torch.float64),dim=-1)
                  for i in row['candidate_ids']]
            old=cache_existing(q,row,ab)
            if old is None:
                if encoder is None:
                    C.verify_model_sources(runtime['parent'],read(checked(runtime['parent']['manifest'])))
                    encoder,processor=C.load_model(runtime['parent'],ab)
                old=collect_query(q,row,w,refs,encoder,processor,runtime,a,ab)
            cache,cv=old
            output=predict_query(q,row,cache,cv,refs,runtime,sources,shard,a,ab,start,budget)
            if output is None:return False
            queries.append(output)
            del cache,refs
        write(seal,dict(status='GROZI_POSTLLM_SHARD_PASS',authority=ab,shard=shard,
            query_count=len(queries),queries=queries,external_training_updates=0,held_label_reads=0,
            source_snapshot=a['source_snapshot']))
        H.emit(stage='external_shard_complete',shard=shard,queries=len(queries),seconds=time.monotonic()-start)
        lock.close()
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=int,default=0);p.add_argument('--pilot',action='store_true')
    p.add_argument('--budget',type=float,default=490)
    args=p.parse_args()
    for s in (signal.SIGUSR1,signal.SIGTERM):signal.signal(s,stop)
    if not worker(args.task,args.pilot,args.budget):raise SystemExit(75)


if __name__=='__main__':main()
