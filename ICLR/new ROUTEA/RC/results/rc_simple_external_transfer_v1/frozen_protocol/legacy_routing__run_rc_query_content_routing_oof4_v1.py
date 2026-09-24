#!/usr/bin/env python3
"""TRAIN128-only content-conditioned residuals; reuse frozen OOF base heads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
sys.dont_write_bytecode = True
import run_rc_train128_disagreement_oof4_v1 as P

OUT = ROOT/'results/rc_query_content_routing_oof4_v1'
AUTH = ROOT/'registry/rc_query_content_routing_oof4_authority_v1_20260912.json'
MODELS = ('GLOBAL7', 'STATS28', 'QUERY28', 'QUERY_PERM28')
STEPS = 2000
NEED = P.need
read, sha, bind, checked = P.read, P.sha, P.bind, P.checked
write, write_bytes = P.write, P.write_bytes


def guard(stage, fold):
    authority = read(AUTH)
    NEED(authority['program'] == bind(__file__), 'FROZEN_PROGRAM')
    checked(authority['parent_program'])
    checked(authority['feature_core'])
    if stage != 'preflight':
        NEED(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    sources = {Path(b['path']).resolve() for b in authority['public_sources'].values()}
    if stage in ('fit', 'replay', 'verify'):
        sources.update(Path(b['path']).resolve() for b in authority['fold_sources'][str(fold)].values())
        sources.discard((P.OUT/f'fold{fold:02d}'/'heldout_roles.json').resolve())
    elif stage == 'join':
        for bundle in authority['fold_sources'].values():
            sources.update(Path(b['path']).resolve() for b in bundle.values())
    if stage in ('cache', 'cache-replay'):
        for b in authority['raw_sources']:
            sources.update(Path(x['path']).resolve() for x in b.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        NEED(not any(x in s for x in ('rc_opened_', '/reports/', 'curator_roles', 'd1-mi', 'd1_mi', '/grozi/', '/target_join/')), 'PROTECTED_READ:'+s)
        if ROOT/'results' in p.parents:
            own = OUT in p.parents
            if own and stage in ('fit', 'replay', 'verify') and p.parent.name.startswith('fold'):
                NEED(p.parent == OUT/f'fold{fold:02d}', 'OWN_FOLD_OUTPUT_ONLY')
            NEED(own or p in sources, 'UNLISTED_RESULT_INPUT:'+s)
        if p.name == 'heldout_roles.json':
            NEED(stage == 'join', 'NO_HELDOUT_LABELS_BEFORE_ALL_PREDICTIONS')
    sys.addaudithook(audit)
    if stage not in ('preflight',):
        pf = read(OUT/'preflight.json')
        NEED(pf['authority'] == bind(AUTH) and pf['status'] == 'QUERY_ROUTING_SYNTHETIC_PREFLIGHT_PASS', 'PREFLIGHT')
    return authority


def tsha(t):
    return hashlib.sha256(str(t.dtype).encode()+json.dumps(list(t.shape), separators=(',', ':')).encode()+t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cache(replay=False):
    a = read(AUTH)
    source = read(checked(a['public_sources']['input_validation']))
    workers = read(checked(a['public_sources']['worker_manifest']))['records']
    feature_rows = read(checked(a['public_sources']['native_features']))
    descriptors, records = [], []
    maxerr = 0.
    for shard, binding in enumerate(a['raw_sources']):
        NEED(binding == source['raw_shards'][shard] or all(binding[k] == source['raw_shards'][shard][k] for k in binding), 'SOURCE_SHARD_BINDING')
        receipt = read(checked(binding['receipt']))
        validation = read(checked(binding['validation']))
        NEED(receipt['payload'] == validation['payload'] == binding['payload'] and validation['receipt'] == binding['receipt'], 'RAW_VALIDATED_CHAIN')
        payload = torch.load(checked(binding['payload']), map_location='cpu', weights_only=True, mmap=True)
        for offset, r in enumerate(payload['records']):
            i = shard*8+offset
            f, w = feature_rows[i], workers[i]
            NEED(r['query_id'] == f['query_id'] == w['query_id'] and r['execution_ordinal'] == i, 'QUERY_AXIS')
            NEED(r['query_source_sha256'] == f['source_image_sha256'] == w['source_image_sha256'], 'IMAGE_BINDING')
            q = r['query_tokens']
            NEED(q.dtype == torch.float16 and q.ndim == 2 and q.shape[1] == 128, 'TOKEN_DOMAIN')
            NEED(len(q) == np.prod(f['query_grid_shape']), 'ONLY_VALID_IMAGE_GRID_TOKENS')
            NEED(tsha(q) == r['query_tokens_sha256'] == f['query_tokens_sha256'], 'QUERY_TOKEN_HASH')
            NEED(list(r['candidate_physical_rows']) == f['candidate_physical_rows'], 'SAME_RAW_C128')
            qt = q.to(torch.float64)
            mean = qt.mean(0).numpy()
            independent = np.asarray(q, dtype=np.float64).sum(axis=0)/len(q)
            err = float(np.max(np.abs(mean-independent)))
            NEED(np.isfinite(mean).all() and err <= 2e-12, 'INDEPENDENT_QUERY_MEAN')
            maxerr = max(maxerr, err)
            descriptors.append(independent if replay else mean)
            records.append(dict(query_id=r['query_id'], execution_ordinal=i, source_image_sha256=w['source_image_sha256'], query_tokens_sha256=tsha(q), token_count=len(q)))
        print(json.dumps(dict(stage='cache-replay' if replay else 'cache', completed_shards=shard+1)), flush=True)
    d = np.stack(descriptors)
    NEED(d.shape == (128,128) and len({r['query_id'] for r in records}) == 128, 'ALL128_QUERY_DESCRIPTORS')
    if replay:
        saved = np.load(OUT/'query_descriptors.npy', allow_pickle=False)
        NEED(np.allclose(d, saved, atol=2e-12, rtol=0), 'FRESH_NUMPY_DESCRIPTOR_REPLAY')
        receipt = read(OUT/'cache_receipt.json')
        NEED(receipt['records'] == records and receipt['descriptor'] == bind(OUT/'query_descriptors.npy'), 'DESCRIPTOR_RECEIPT')
        write(OUT/'cache_validation.json', dict(status='QUERY_DESCRIPTOR_FRESH_NUMPY_REPLAY_PASS', descriptor=receipt['descriptor'], receipt=bind(OUT/'cache_receipt.json'), max_abs_error=float(np.max(np.abs(d-saved))), query_count=128, label_reads=0))
    else:
        import io
        stream=io.BytesIO();np.save(stream,d,allow_pickle=False)
        write_bytes(OUT/'query_descriptors.npy',stream.getvalue())
        write(OUT/'cache_receipt.json',dict(status='QUERY_DESCRIPTOR_CACHE_SEALED',descriptor=bind(OUT/'query_descriptors.npy'),authority=bind(AUTH),records=records,max_independent_mean_error=maxerr,source_validation=a['public_sources']['input_validation'],label_reads=0))
        subprocess.run([sys.executable,__file__,'cache-replay'],check=True)


def standardize_fit(x):
    mean=x.mean(axis=0)
    scale=np.maximum(np.sqrt(np.mean((x-mean)**2,axis=0)),1e-12)
    return mean,scale


def query_projection(desc, train):
    mean=desc[train].mean(0)
    _,s,vh=np.linalg.svd(desc[train]-mean, full_matrices=False)
    basis=vh[:3].T.copy()
    for j in range(3):
        if basis[np.argmax(np.abs(basis[:,j])),j] < 0:basis[:,j]*=-1
    projection=(desc-mean)@basis
    pm,ps=standardize_fit(projection[train])
    value=(projection-pm)/ps
    return value,dict(mean=mean.tolist(),basis=basis.tolist(),projection_mean=pm.tolist(),projection_scale=ps.tolist(),singular_values=s.tolist(),explained_variance_ratio=(s[:3]**2/np.sum(s**2)).tolist(),fit_ordinals=list(map(int,train)))


def permutation(ordinals, ids, namespace):
    ordered=sorted(map(int,ordinals),key=lambda i:hashlib.sha256((namespace+'|'+ids[i]).encode()).hexdigest())
    NEED(len(ordered)>1,'PERMUTATION_NONTRIVIAL')
    return {i:ordered[(j+1)%len(ordered)] for j,i in enumerate(ordered)}


def design(x, context):
    # Context-major ordering; the first seven columns are the global residual.
    augmented=torch.cat((x,x.new_ones((*x.shape[:-1],1))),dim=-1)
    return (context[..., :, None]*augmented[..., None, :]).flatten(-2)


def objective(z, y):
    allz=torch.cat((z.new_zeros((len(z),1)),z),dim=1)
    return (torch.logsumexp(allz,dim=1)-allz[torch.arange(len(y)),y]).mean()


def optimize(x, base, y, steps=STEPS):
    theta=torch.nn.Parameter(torch.zeros(x.shape[-1],dtype=torch.float64))
    opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
    for step in range(steps):
        opt.zero_grad();z=base+x@theta;loss=objective(z,y)
        NEED(torch.isfinite(loss),'FINITE_LOSS');loss.backward()
        NEED(torch.isfinite(theta.grad).all(),'FINITE_GRADIENT');opt.step()
        NEED(torch.isfinite(theta).all(),'FINITE_PARAMETER')
    return theta.detach(),float(loss.detach())


def load_data(fold):
    a=read(AUTH);b=a['fold_sources'][str(fold)]
    oldseal=read(checked(b['seal']));oldvalidation=read(checked(b['validation']))
    NEED(oldvalidation['seal']==b['seal'] and oldvalidation['status']=='TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS','OLD_BASE_QUALIFIED')
    NEED(oldseal['parameters']==b['parameters'] and oldseal['predictions']==b['predictions'],'OLD_BASE_PARAMETER_BINDING')
    oldpar=read(checked(b['parameters']))['BASE7']
    roles=read(checked(b['train_roles']))['records'];rolemap={r['query_id']:r for r in roles}
    rows=P.qualified_rows();P.cache_rows(rows)
    split=read(checked(a['public_sources']['fold_manifest']))['records']
    train=[i for i,s in enumerate(split) if s['fold']!=fold]
    held=[i for i,s in enumerate(split) if s['fold']==fold]
    NEED({rows[i]['query_id'] for i in train}==set(rolemap),'EXACT_FOLD_TRAIN_LABELS')
    NEED(train==oldseal['closure']['correction_FULL_ordinals'] and held==oldseal['closure']['heldout_ordinals'],'SAME_PREVIOUS_CORRECTION_DATA')
    labels,mapping=P.gallery_labels()
    NEED(mapping==oldseal['closure']['gallery_mapping_sha256'],'FROZEN_GALLERY_MAPPING')
    targets=[]
    for i in train:
        r=rows[i];pos=[p for p,g in enumerate(r['candidate_physical_rows']) if labels[g]==rolemap[r['query_id']]['identity']]
        NEED(len(pos)==1,'TRAIN_TARGET_IN_C128')
        targets.append(0 if pos[0]==r['base_winner_position'] else r['challenger_positions'].index(pos[0])+1)
    cv=read(OUT/'cache_validation.json');cr=read(OUT/'cache_receipt.json')
    NEED(cv['status']=='QUERY_DESCRIPTOR_FRESH_NUMPY_REPLAY_PASS' and cv['receipt']==bind(OUT/'cache_receipt.json') and cv['descriptor']==cr['descriptor'],'DESCRIPTOR_VALIDATION')
    desc=np.load(checked(cr['descriptor']),allow_pickle=False)
    NEED([r['query_id'] for r in cr['records']]==[r['query_id'] for r in rows],'DESCRIPTOR_NATIVE_JOIN')
    q,pc=query_projection(desc,train)
    ids=[r['query_id'] for r in rows]
    donors={**permutation(train,ids,'QUERY_ROUTE_V1_TRAIN'),**permutation(held,ids,'QUERY_ROUTE_V1_HELDOUT')}
    NEED(set(donors)==set(range(128)) and all((i in train)==(j in train) and i!=j for i,j in donors.items()),'WITHIN_SPLIT_DERANGEMENT')
    x=torch.stack([r['X'] for r in rows]);stats=np.stack([r['gate_X'].numpy()[:,1:] for r in rows])
    sm,ss=standardize_fit(stats[train].reshape(-1,3));st=(stats-sm)/ss
    contexts={'GLOBAL7':np.ones((128,127,1)), 'STATS28':np.concatenate((np.ones((128,127,1)),st),axis=-1)}
    for name,values in [('QUERY28',q),('QUERY_PERM28',q[[donors[i] for i in range(128)]])]:
        contexts[name]=np.broadcast_to(np.concatenate((np.ones((128,1)),values),axis=-1)[:,None,:],(128,127,4)).copy()
    matrices={m:design(x,torch.from_numpy(contexts[m])) for m in MODELS}
    w=torch.tensor([float.fromhex(v) for v in oldpar['weight_binary64']],dtype=torch.float64)
    bias=float.fromhex(oldpar['bias_binary64'])
    base=torch.stack([r['X']@w+bias for r in rows])
    oldpred={r['query_id']:r for r in read(checked(b['predictions']))}
    for i in held:
        old=oldpred[ids[i]]['predictions']['BASE7']['all127_logits_binary64']
        NEED([float(z).hex() for z in base[i]]==old,'FROZEN_BASE_ALL_LOGITS_BIT_PARITY')
    transforms=dict(query_pca=pc,stats_mean=sm.tolist(),stats_scale=ss.tolist(),permutation_donors={str(i):j for i,j in donors.items()})
    return rows,train,held,matrices,base,torch.tensor(targets),transforms


def run_fold(fold,replay=False,nonce=None):
    rows,train,held,matrices,base,y,transforms=load_data(fold)
    params={};allz={'BASE7':base}
    for m in MODELS:
        theta,loss=optimize(matrices[m][train],base[train],y)
        params[m]=dict(theta_binary64=[float(v).hex() for v in theta],parameter_count=len(theta),last_preupdate_loss=loss)
        allz[m]=base+matrices[m]@theta
        print(json.dumps(dict(stage='replay' if replay else 'fit',fold=fold,model=m,steps=STEPS,train_loss=loss)),flush=True)
    predictions=[]
    for i in held:
        row=rows[i];models={}
        for m,z in allz.items():
            scores=z[i];j=int(torch.argmax(scores));pos=row['challenger_positions'][j] if float(scores[j])>0 else row['base_winner_position']
            models[m]=dict(logits_binary64=[float(v).hex() for v in scores],selected_physical_row=row['candidate_physical_rows'][pos])
        predictions.append(dict(query_id=row['query_id'],execution_ordinal=i,models=models))
    value=dict(status='QUERY_ROUTING_FOLD_PREDICTIONS_SEALED',fold=fold,authority=bind(AUTH),parameters=params,transforms=transforms,train_ordinals=train,heldout_ordinals=held,predictions=predictions,base_training_updates=0,new_training_updates_per_model=STEPS,heldout_label_reads=0,EVAL_reads=0,descriptor=bind(OUT/'query_descriptors.npy'))
    folder=OUT/f'fold{fold:02d}'
    if replay:
        NEED(nonce==os.environ.get('QUERY_ROUTE_NONCE'),'FRESH_PROCESS_NONCE')
        NEED(value==read(folder/'fit_predictions.json'),'FRESH_PARAMETERS_AND_PREDICTIONS_BIT_REPLAY')
        write(folder/'validation.json',dict(status='QUERY_ROUTING_FRESH_FOLD_REPLAY_PASS',payload=bind(folder/'fit_predictions.json'),authority=bind(AUTH),fold=fold,nonce=nonce,base_training_updates=0,heldout_label_reads=0))
    else:
        write(folder/'fit_predictions.json',value)
        nonce=uuid.uuid4().hex
        subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],check=True,env=dict(os.environ,QUERY_ROUTE_NONCE=nonce))
        subprocess.run([sys.executable,str(ROOT/'programs/validate_rc_query_content_routing_oof4_v1.py'),'verify','--fold',str(fold)],check=True)


def synthetic():
    torch.manual_seed(17)
    x=torch.randn(4,3,6,dtype=torch.float64);h=torch.randn(4,3,4,dtype=torch.float64);h[:,:,0]=1
    d=design(x,h);theta=torch.randn(28,dtype=torch.float64,requires_grad=True)
    base=torch.randn(4,3,dtype=torch.float64);y=torch.tensor([0,1,2,3]);z=base+d@theta
    manual=torch.stack([torch.stack([base[i,c]+sum((torch.cat((x[i,c],x.new_ones(1)))*theta[k*7:(k+1)*7]).sum()*h[i,c,k] for k in range(4)) for c in range(3)]) for i in range(4)])
    NEED(torch.allclose(z,manual,atol=1e-12,rtol=0),'BILINEAR_DESIGN_INDEPENDENT_LOOP')
    loss=objective(z,y);other=F.cross_entropy(torch.cat((base.new_zeros((4,1)),manual),1),y)
    g=torch.autograd.grad(loss,theta,retain_graph=True)[0];g2=torch.autograd.grad(other,theta)[0]
    NEED(torch.allclose(loss,other,atol=1e-12,rtol=0) and torch.allclose(g,g2,atol=1e-12,rtol=0),'LOSS_AND_GRADIENT')
    NEED(torch.equal(base+d@torch.zeros(28,dtype=torch.float64),base),'ZERO_RESIDUAL_PARITY')
    a=np.random.default_rng(17).normal(size=(12,128));q,t=query_projection(a,list(range(8)))
    changed=a.copy();changed[8:]+=100
    q2,t2=query_projection(changed,list(range(8)))
    NEED(t==t2 and np.array_equal(q[:8],q2[:8]),'NO_HELDOUT_PCA_FIT')
    perm=permutation(range(4),list('abcd'),'synthetic')
    NEED(set(perm.values())==set(range(4)) and all(i!=j for i,j in perm.items()),'NO_SELF_PERMUTATION')
    theta,loss=optimize(d,base,y,steps=2)
    NEED(torch.isfinite(theta).all(),'SYNTHETIC_OPTIMIZER_RUN')
    return dict(status='QUERY_ROUTING_SYNTHETIC_PREFLIGHT_PASS',authority=bind(AUTH),natural_training_updates=0,checks=['bilinear_formula','full_CE_loss_and_gradient','zero_residual','no_heldout_PCA_fit','split_local_permutation','two_optimizer_updates'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','cache','cache-replay','fit','replay']);ap.add_argument('--fold',type=int);ap.add_argument('--nonce');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage,args.fold)
    if args.stage=='preflight':write(OUT/'preflight.json',synthetic());print('QUERY_ROUTING_SYNTHETIC_PREFLIGHT_PASS',flush=True)
    elif args.stage in ('cache','cache-replay'):cache(args.stage=='cache-replay')
    else:run_fold(args.fold,args.stage=='replay',args.nonce)
