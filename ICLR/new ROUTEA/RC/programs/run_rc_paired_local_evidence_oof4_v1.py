#!/usr/bin/env python3
"""Same-query-token candidate contrast, using sealed TRAIN128-only inputs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
sys.dont_write_bytecode = True
import run_rc_query_content_routing_oof4_v1 as R
P = R.P
OUT = ROOT/'results/rc_paired_local_evidence_oof4_v1'
AUTH = ROOT/'registry/rc_paired_local_evidence_oof4_authority_v2_20260912.json'
VALIDATOR = ROOT/'programs/validate_rc_paired_local_evidence_oof4_v1.py'
MODELS = ('BIAS1', 'MEAN2', 'CURVE3', 'JOINT3')
need, read, write, write_bytes, sha, bind, checked = P.need, P.read, P.write, P.write_bytes, P.sha, P.bind, P.checked


def guard(stage, index=None):
    a = read(AUTH)
    need(a['status'] == 'PAIRED_LOCAL_TRAIN128_OOF4_BOUNDED_AUTHORIZED', 'AUTHORITY_STATUS')
    for b in a['code_sources'].values(): checked(b)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM_BINDING')
    if stage != 'preflight': need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    if stage.startswith('cache'): need(index in range(16), 'SHARD_RANGE')
    if stage in ('fit', 'replay', 'verify'): need(index in range(4), 'FOLD_RANGE')
    allowed = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage.startswith('cache'):
        for kind in ('raw_shards', 'roma_shards'):
            allowed.update(Path(a[kind][index][k]['path']).resolve() for k in ('payload', 'receipt', 'validation'))
    if stage in ('fit', 'replay', 'verify'):
        allowed.update(Path(b['path']).resolve() for k,b in a['fold_sources'][str(index)].items() if k != 'heldout_roles')
    if stage == 'join':
        for bundle in (*a['fold_sources'].values(), *a['global_sources'].values()):
            allowed.update(Path(b['path']).resolve() for b in bundle.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        need(not any(x in s for x in ('rc_opened_', '/reports/', 'curator_roles', 'd1-mi', 'd1_mi', '/grozi/', '/target_join/')), 'PROTECTED_READ:'+s)
        if ROOT/'results' in p.parents:
            own = OUT in p.parents
            if own and stage != 'join':
                rel = p.relative_to(OUT).parts
                preflight_output = rel[0] == 'preflight.json' or (stage == 'preflight' and rel[0].startswith('.preflight.json.') and rel[0].endswith('.tmp'))
                own = preflight_output or (stage.startswith('cache') and rel[0] == f'cache{index:02d}') or (stage in ('fit','replay','verify') and (rel[0].startswith('cache') or rel[0] == f'fold{index:02d}'))
            need(own or p in allowed, 'UNLISTED_RESULT_INPUT:'+s)
        if p.name == 'heldout_roles.json': need(stage == 'join', 'NO_HELDOUT_LABELS_BEFORE_SEAL')
    sys.addaudithook(audit)
    for b in a['public_sources'].values(): checked(b)
    if stage != 'preflight':
        pf = read(OUT/'preflight.json')
        need(pf['authority'] == bind(AUTH) and pf['status'] == 'PAIRED_LOCAL_SYNTHETIC_PREFLIGHT_PASS', 'PREFLIGHT')
    return a


def tsha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode()+json.dumps(list(t.shape),separators=(',', ':')).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def pair_statistics(profiles, weights, winner):
    u = weights + weights[winner]
    rho = u / u.sum(1, keepdim=True).clamp_min(1e-12)
    d = profiles - profiles[winner]
    return torch.stack(((rho*d).sum(1), (rho*d*d.abs()).sum(1)), dim=-1)


def cache_compute(a, shard, independent=False):
    source = read(checked(a['public_sources']['input_validation']))
    need(source['status'] == 'ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS', 'SOURCE_QUALIFIED')
    payloads = []
    for kind in ('raw_shards', 'roma_shards'):
        b = a[kind][shard]
        need(b == source[kind][shard], 'EXACT_SOURCE_BINDING')
        receipt, validation = read(checked(b['receipt'])), read(checked(b['validation']))
        need(receipt['payload'] == validation['payload'] == b['payload'] and validation['receipt'] == b['receipt'], 'SHARD_CHAIN')
        payloads.append(torch.load(checked(b['payload']), map_location='cpu', weights_only=True, mmap=True))
    raw, roma = payloads
    need(raw['shard'] == roma['shard'] == shard and len(raw['records']) == len(roma['records']) == 8, 'SHARD_HEADER')
    need(roma['token_source'] == {k:a['raw_shards'][shard][k] for k in ('payload','receipt','validation')}, 'ROMA_TOKEN_CHAIN')
    refs = {int(k):v for k,v in raw['references'].items()}
    for ref in refs.values(): need(tsha(ref['tokens']) == ref['tokens_sha256'], 'REFERENCE_TOKEN_BYTES')
    features = read(checked(a['public_sources']['native_features']))
    with np.load(checked(a['public_sources']['disagreement_npz']), allow_pickle=False) as old:
        old_f = old['features'][:,:,0].copy()
    arrays, meta, stat, err = {}, [], [], 0.
    for offset, (qrow, rrow) in enumerate(zip(raw['records'], roma['records'], strict=True)):
        i = shard*8 + offset
        f = features[i]
        need(qrow['query_id'] == rrow['query_id'] == f['query_id'] and qrow['execution_ordinal'] == rrow['execution_ordinal'] == i, 'QUERY_AXIS')
        need(qrow['query_source_sha256'] == rrow['query_source_sha256'] == f['source_image_sha256'], 'QUERY_IMAGE')
        q = qrow['query_tokens']
        need(q.dtype == torch.float16 and q.ndim == 2 and q.shape[1] == 128 and len(q) == np.prod(f['query_grid_shape']), 'QUERY_DOMAIN')
        need(tsha(q) == qrow['query_tokens_sha256'] == rrow['query_tokens_sha256'] == f['query_tokens_sha256'], 'QUERY_TOKEN_BYTES')
        axis = list(qrow['candidate_physical_rows'])
        need(axis == list(rrow['candidate_physical_rows']) == f['candidate_physical_rows'] == sorted(axis) and len(set(axis)) == 128, 'FULL_PHYSICAL_C128')
        need(tsha(qrow['candidate_raw_scores']) == tsha(rrow['candidate_raw_scores']), 'RAW_BITS')
        need([float(v).hex() for v in qrow['candidate_raw_scores']] == f['candidate_raw_scores_binary64'], 'NATIVE_RAW_BITS')
        need(len(rrow['candidates']) == 128, 'ALL_CANDIDATES')
        qt = q.to(torch.float64)
        qn = np.asarray(qt) / np.maximum(np.linalg.norm(np.asarray(qt),axis=1,keepdims=True),1e-12) if independent else F.normalize(qt, dim=1)
        ps, ws = [], []
        for position, c in enumerate(rrow['candidates']):
            ref = refs[axis[position]]
            r, wq, wr = ref['tokens'], c['query_visibility'], c['reference_visibility']
            need(c['candidate_position'] == position and c['physical_row'] == axis[position] and c['reference_tokens_sha256'] == ref['tokens_sha256'], 'CANDIDATE_REFERENCE_BINDING')
            need(r.dtype == torch.float16 and r.ndim == 2 and r.shape[1] == 128, 'REFERENCE_DOMAIN')
            for w,n,key in ((wq,len(q),'query'),(wr,len(r),'reference')):
                need(w.dtype == torch.float64 and tuple(w.shape) == (n,) and bool(torch.isfinite(w).all()) and bool(((w>=0)&(w<=1)).all()), 'WEIGHT_DOMAIN')
                need(hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest() == c[key+'_map_sha256'] == c['old_scores'][key+'_map_sha256'], 'VISIBILITY_MAP_BYTES')
            rt = r.to(torch.float64)
            if independent:
                rn = np.asarray(rt) / np.maximum(np.linalg.norm(np.asarray(rt),axis=1,keepdims=True),1e-12)
                p = np.max(qn @ rn.T, axis=1)
                free_score = float(np.dot(np.asarray(wq),p)/max(float(np.sum(np.asarray(wq))),1e-12))
            else:
                p = (qn @ F.normalize(rt, dim=1).T).max(1).values
                free_score = float((wq*p).sum()/wq.sum().clamp_min(1e-12))
            error = abs(free_score-old_f[i,position])
            need(error <= 2e-12, 'EXISTING_FREE_SCORE_REPLAY')
            err = max(err,error)
            ps.append(np.asarray(p)); ws.append(np.asarray(wq))
        profiles, weights = np.stack(ps), np.stack(ws)
        w = f['base_winner_position']
        if independent:
            u = weights + weights[w]
            rho = u / np.maximum(u.sum(axis=1,keepdims=True),1e-12)
            d = profiles - profiles[w]
            # Explicit dot products, separate from the Torch producer reductions.
            value = np.array([[np.dot(rr,dd), np.dot(rr,dd*np.abs(dd))] for rr,dd in zip(rho,d,strict=True)])
        else:
            value = pair_statistics(torch.from_numpy(profiles),torch.from_numpy(weights),w).numpy()
        need(np.isfinite(value).all() and np.array_equal(value[w],np.zeros(2)), 'FINITE_ZERO_SELF_CONTRAST')
        arrays[f'free_{i:03d}'], arrays[f'wq_{i:03d}'] = profiles, weights
        stat.append(value)
        meta.append(dict(query_id=qrow['query_id'],execution_ordinal=i,source_image_sha256=f['source_image_sha256'],query_tokens_sha256=f['query_tokens_sha256'],candidate_physical_rows=axis,token_count=len(q),base_winner_position=w))
        print(json.dumps(dict(stage='cache-replay' if independent else 'cache',shard=shard,completed_queries=offset+1)),flush=True)
    arrays['statistics'] = np.stack(stat)
    return arrays, meta, err


def cache(shard, replay=False):
    started = time.monotonic()
    a = read(AUTH)
    arrays, meta, err = cache_compute(a,shard,replay)
    folder = OUT/f'cache{shard:02d}'
    if replay:
        receipt = read(folder/'receipt.json')
        need(receipt['authority'] == bind(AUTH) and receipt['records'] == meta, 'CACHE_RECEIPT')
        maxerr = 0.
        with np.load(checked(receipt['payload']),allow_pickle=False) as saved:
            need(set(saved.files) == set(arrays), 'CACHE_KEYS')
            for k,v in arrays.items():
                need(v.shape == saved[k].shape and v.dtype == saved[k].dtype, 'CACHE_ARRAY_DOMAIN')
                diff = float(np.max(np.abs(v-saved[k])))
                need(diff <= 2e-12, 'FULL_NUMPY_CACHE_REPLAY:'+k)
                maxerr = max(maxerr,diff)
        write(folder/'validation.json',dict(status='PAIRED_LOCAL_FRESH_NUMPY_CACHE_PASS',shard=shard,authority=bind(AUTH),payload=receipt['payload'],receipt=bind(folder/'receipt.json'),query_count=8,candidate_occurrences=1024,max_abs_error=maxerr,old_F_replay_max_abs_error=err,label_reads=0))
    else:
        stream = io.BytesIO(); np.savez(stream,**arrays); write_bytes(folder/'payload.npz',stream.getvalue())
        write(folder/'receipt.json',dict(status='PAIRED_LOCAL_CACHE_SEALED',shard=shard,authority=bind(AUTH),payload=bind(folder/'payload.npz'),records=meta,old_F_replay_max_abs_error=err,label_reads=0))
        subprocess.run([sys.executable,__file__,'cache-replay','--shard',str(shard)],check=True)
    print(json.dumps(dict(stage='cache-replay-complete' if replay else 'cache-complete',shard=shard,elapsed_seconds=time.monotonic()-started)),flush=True)


def load_statistics():
    stats, meta, receipts = [], [], []
    for shard in range(16):
        folder = OUT/f'cache{shard:02d}'
        v = read(folder/'validation.json'); r = read(checked(v['receipt']))
        need(v['status'] == 'PAIRED_LOCAL_FRESH_NUMPY_CACHE_PASS' and v['authority'] == r['authority'] == bind(AUTH) and v['payload'] == r['payload'] and v['shard'] == r['shard'] == shard, 'QUALIFIED_JOINT_CACHE')
        with np.load(checked(r['payload']),allow_pickle=False) as npz: stats.append(npz['statistics'].copy())
        meta.extend(r['records']); receipts.append(bind(folder/'validation.json'))
    x = np.concatenate(stats)
    need(x.shape == (128,128,2) and [r['execution_ordinal'] for r in meta] == list(range(128)), 'WHOLE_TRAIN128_CACHE')
    return x, meta, receipts


def designs(stats, rows, train):
    v = np.stack([stats[i,r['challenger_positions']] for i,r in enumerate(rows)])
    m1,m2 = v[:,:,0],v[:,:,1]
    raw = {'BIAS1':np.empty((128,127,0)), 'MEAN2':m1[:,:,None], 'CURVE3':np.stack((m1,m1*np.abs(m1)),axis=-1), 'JOINT3':np.stack((m1,m2),axis=-1)}
    matrices, transforms = {}, {}
    for m,x in raw.items():
        if x.shape[-1]:
            mean,scale = R.standardize_fit(x[train].reshape(-1,x.shape[-1]))
            standardized = (x-mean)/scale
        else:
            mean,scale = np.empty(0),np.empty(0); standardized=x
        matrices[m] = torch.from_numpy(np.concatenate((np.ones((128,127,1)),standardized),axis=-1))
        transforms[m] = dict(mean=mean.tolist(),scale=scale.tolist(),fit_ordinals=train)
    return matrices,transforms


def load_fold(fold):
    a = read(AUTH); b = a['fold_sources'][str(fold)]
    seal,validation = read(checked(b['seal'])),read(checked(b['validation']))
    need(validation['status'] == 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and validation['seal'] == b['seal'] and seal['parameters'] == b['parameters'] and seal['predictions'] == b['predictions'], 'FROZEN_BASE_QUALIFICATION')
    rows = P.qualified_rows()
    split = read(checked(a['public_sources']['fold_manifest']))['records']
    train = [i for i,r in enumerate(split) if r['fold'] != fold]
    held = [i for i,r in enumerate(split) if r['fold'] == fold]
    need(train == seal['closure']['correction_FULL_ordinals'] and held == seal['closure']['heldout_ordinals'], 'ORIGINAL_FOLD_AXES')
    roles = read(checked(b['train_roles']))['records']; rolemap = {r['query_id']:r for r in roles}
    need(set(rolemap) == {rows[i]['query_id'] for i in train}, 'EXACT_TRAIN_LABELS')
    labels,mapping = P.gallery_labels(); need(mapping == seal['closure']['gallery_mapping_sha256'], 'GALLERY_MAPPING')
    y = []
    for i in train:
        r = rows[i]; positions = [p for p,g in enumerate(r['candidate_physical_rows']) if labels[g] == rolemap[r['query_id']]['identity']]
        need(len(positions) == 1, 'TRAIN_TARGET_IN_C128')
        y.append(0 if positions[0] == r['base_winner_position'] else r['challenger_positions'].index(positions[0])+1)
    stats,meta,receipts = load_statistics()
    need([r['query_id'] for r in meta] == [r['query_id'] for r in rows] and all(r['candidate_physical_rows'] == m['candidate_physical_rows'] for r,m in zip(rows,meta,strict=True)), 'NATIVE_CACHE_JOIN')
    mats,transforms = designs(stats,rows,train)
    par = read(checked(b['parameters']))['BASE7']
    w = torch.tensor([float.fromhex(v) for v in par['weight_binary64']],dtype=torch.float64); bias = float.fromhex(par['bias_binary64'])
    base = torch.stack([r['X']@w+bias for r in rows])
    old = {r['query_id']:r for r in read(checked(b['predictions']))}
    for i in held: need([float(z).hex() for z in base[i]] == old[rows[i]['query_id']]['predictions']['BASE7']['all127_logits_binary64'], 'FROZEN_BASE_BIT_PARITY')
    return rows,train,held,mats,base,torch.tensor(y),transforms,receipts


def fit(fold,replay=False,nonce=None):
    rows,train,held,mats,base,y,transforms,receipts = load_fold(fold)
    parameters,allz = {}, {'BASE7':base}
    for m in MODELS:
        theta,loss = R.optimize(mats[m][train],base[train],y)
        parameters[m] = dict(theta_binary64=[float(v).hex() for v in theta],parameter_count=len(theta),last_preupdate_loss=loss)
        allz[m] = base+mats[m]@theta
        print(json.dumps(dict(stage='replay' if replay else 'fit',fold=fold,model=m,steps=2000,loss=loss)),flush=True)
    predictions = []
    for i in held:
        r = rows[i]; models = {}
        for m,z in allz.items():
            scores=z[i]; j=int(torch.argmax(scores)); pos=r['challenger_positions'][j] if scores[j]>0 else r['base_winner_position']
            models[m]=dict(logits_binary64=[float(v).hex() for v in scores],selected_physical_row=r['candidate_physical_rows'][pos])
        predictions.append(dict(query_id=r['query_id'],execution_ordinal=i,models=models))
    value = dict(status='PAIRED_LOCAL_FOLD_PREDICTIONS_SEALED',fold=fold,authority=bind(AUTH),parameters=parameters,transforms=transforms,train_ordinals=train,heldout_ordinals=held,predictions=predictions,cache_validations=receipts,base_training_updates=0,heldout_label_reads=0,EVAL_reads=0)
    folder=OUT/f'fold{fold:02d}'
    if replay:
        need(nonce and nonce == os.environ.get('PAIRED_LOCAL_NONCE'), 'FRESH_PROCESS_NONCE')
        need(value == read(folder/'fit_predictions.json'), 'FRESH_TRAIN_PARAMETERS_PREDICTIONS_BIT_REPLAY')
        write(folder/'validation.json',dict(status='PAIRED_LOCAL_FRESH_FIT_REPLAY_PASS',fold=fold,authority=bind(AUTH),payload=bind(folder/'fit_predictions.json'),heldout_label_reads=0,nonce=nonce))
    else:
        write(folder/'fit_predictions.json',value)
        nonce=uuid.uuid4().hex
        subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],check=True,env=dict(os.environ,PAIRED_LOCAL_NONCE=nonce))
        subprocess.run([sys.executable,str(VALIDATOR),'verify','--fold',str(fold)],check=True)


def preflight():
    a=torch.tensor([[1.,.5,0.],[0.,.875,.625],[.875,0.,.625]],dtype=torch.float64)
    weights=torch.ones_like(a)
    left=pair_statistics(a,weights,1)[0];right=pair_statistics(a,weights,2)[0]
    need(abs(float(left[0])) < 1e-15 and abs(float(right[0])) < 1e-15 and abs(float(left[1])-5/32)<1e-15 and abs(float(right[1])+1/24)<1e-15, 'MARGINAL_COLLISION_JOINT_SEPARATION')
    need(torch.equal(pair_statistics(a,torch.zeros_like(a),0),torch.zeros((3,2),dtype=torch.float64)), 'ZERO_SUPPORT')
    need(torch.allclose(pair_statistics(a[:2],weights[:2],1)[0],-pair_statistics(a[:2],weights[:2],0)[1],atol=1e-15,rtol=0),'CANDIDATE_SWAP_ANTISYMMETRY')
    torch.manual_seed(17)
    z=torch.randn((4,127),dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,64,127])
    loss=R.objective(z,y);other=F.cross_entropy(torch.cat((z.new_zeros((4,1)),z),1),y)
    need(torch.allclose(loss,other,atol=1e-12,rtol=0) and torch.allclose(torch.autograd.grad(loss,z,retain_graph=True)[0],torch.autograd.grad(other,z)[0],atol=1e-12,rtol=0),'FULL_C128_LOSS_GRADIENT')
    stats=np.random.default_rng(17).normal(size=(128,128,2));rows=[dict(challenger_positions=list(range(1,128))) for _ in range(128)];train=list(range(96))
    mats,t=designs(stats,rows,train);changed=stats.copy();changed[96:]+=100
    mats2,t2=designs(changed,rows,train)
    need(t == t2 and all(torch.equal(mats[m][train],mats2[m][train]) for m in MODELS),'TRAIN_ONLY_STANDARDIZATION')
    need(all(torch.equal(z.detach()+mats[m][:4]@torch.zeros(mats[m].shape[-1],dtype=torch.float64),z.detach()) for m in MODELS),'ZERO_RESIDUAL_BASE_PARITY')
    R.optimize(mats['JOINT3'][:4],z.detach(),y,steps=2)
    write(OUT/'preflight.json',dict(status='PAIRED_LOCAL_SYNTHETIC_PREFLIGHT_PASS',authority=bind(AUTH),natural_training_updates=0,checks=['marginal_collision_witness','zero_support','swap_antisymmetry','full128_loss_gradient','TRAIN_only_standardization','zero_residual','two_optimizer_steps']))
    print('PAIRED_LOCAL_SYNTHETIC_PREFLIGHT_PASS',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','cache','cache-replay','fit','replay']);ap.add_argument('--shard',type=int);ap.add_argument('--fold',type=int);ap.add_argument('--nonce');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    guard(args.stage,args.shard if args.stage.startswith('cache') else args.fold)
    if args.stage == 'preflight': preflight()
    elif args.stage.startswith('cache'): cache(args.shard,args.stage=='cache-replay')
    else: fit(args.fold,args.stage=='replay',args.nonce)
