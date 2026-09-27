#!/usr/bin/env python3
"""Cached COARSE J/P entry deletion and matched PRODUCT5 compensation.

This is attribution on the first 128 cached H593 queries, not a new retrieval
reader. J/P are direct predictor inputs: P itself was computed from native
matching, so J=0 does not remove all interaction information. All scientific
arms use the same original COARSE GPU cache. No new encoder/matcher forward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

RC = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RC / 'results/rc_m_causal128_attribution_chain_v1/paths'
INSIDE = RC / 'results/rc_h593_m_inside_v1'
SIMPLE = RC / 'results/rc_h593_simple_explanations_v1'
AUTH = RC / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
ARMS = ('A1J1P1', 'A1J1P0', 'A1J0P1', 'A1J0P0')
FEATURES = ('standardized_raw_gap', 'symmetric_M_times_free_L', 'symmetric_M', 'symmetric_free_L')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, once=False):
    path = Path(path)
    if once and path.exists():
        need(read(path) == value, 'IMMUTABLE_OUTPUT:' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def checked(record):
    actual = binding(record['path'])
    need(actual['sha256'] == record['sha256'], 'SHA:' + record['path'])
    return Path(record['path'])


def emit(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sym(a, b):
    return (a - b) / (np.abs(a) + np.abs(b) + 1e-12)


def design(raw, free, masses, winner, challengers):
    m, l = np.asarray(masses, np.float64), np.asarray(free, np.float64)
    c = np.asarray(challengers, int)
    need(m.shape == l.shape == (128,), 'C128_SHAPE')
    need(np.isfinite(m).all() and (m > 0).all() and (m <= 1).all(), 'VALID_M')
    need(np.isfinite(l).all(), 'VALID_FREE_CONTENT')
    # No native u/v, S, original M, or native local-control terms are read here.
    return np.stack([np.asarray(raw, np.float64), sym((m*l)[c], (m*l)[winner]),
                     sym(m[c], m[winner]), sym(l[c], l[winner])], axis=-1)


def choice(z):
    z = np.asarray(z)
    return np.where(z.max(axis=-1) > 0, z.argmax(axis=-1) + 1, 0)


def protocol(root):
    p = read(root / 'protocol.json')
    need(p['status'] == 'M_CAUSAL128_PATH_PROTOCOL_FROZEN', 'PROTOCOL')
    for b in p['code_sources']:
        checked(b)
    return p, binding(root / 'protocol.json')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'prepare.lock'):
        if (root / 'protocol.json').exists():
            p, pb = protocol(root)
            emit(status='PREPARE_ALREADY_COMPLETE', protocol=pb)
            return 0
        a = read(AUTH)
        v = read(SIMPLE / 'cache_validation.json')
        need(v['status'] == 'CACHE_PASS' and v['authority'] == binding(AUTH), 'PARENT_CACHE')
        cache = read(checked(v['payload']))
        split = read(checked(a['public']['split']))
        fold_by_q = {}
        for f, s in enumerate(split['folds']):
            for q in s['heldout_query_ids']:
                need(q not in fold_by_q, 'UNIQUE_ORIGINAL_FOLD')
                fold_by_q[q] = f
        src = []
        for i in range(128):
            d = INSIDE / f'query{i:03d}'
            iv, nv = read(d / 'inside_validation.json'), read(d / 'validation.json')
            need(iv['status'] == 'M_INSIDE_PATHS_PASS' and iv['candidates'] == 128, 'INSIDE128')
            need(nv['status'] == 'M_VISUAL_ORIGIN_QUERY_PASS', 'SOURCE_NATIVE')
            im, na = read(checked(iv['payload'])), read(checked(nv['payload']))
            r = cache['rows'][i]
            need(r['execution_ordinal'] == i == na['execution_ordinal'], 'ORDINAL')
            need(r['query_id'] == im['query_id'] == na['query_id'], 'QUERY_ID')
            need(r['axis'] == na['candidate_physical_rows'], 'AXIS')
            need(len(im['records']) == len(im['sources']) == 128, 'ALL128_SOURCES')
            need(im['authority'] == iv['authority'] == nv['authority'], 'SOURCE_AUTHORITY')
            src.append(dict(index=i, query_id=r['query_id'], fold=fold_by_q[r['query_id']],
                inside_validation=binding(d / 'inside_validation.json'), inside_manifest=iv['payload'],
                native_validation=binding(d / 'validation.json'), native_payload=nv['payload']))
        code = [Path(__file__), RC / 'programs/run_rc_six_cause_loss_binding_v1.py',
                RC / 'programs/run_rc_h593_simple_explanations_v1.py',
                RC / 'programs/rc_roma_m_inside_v1.py']
        p = dict(status='M_CAUSAL128_PATH_PROTOCOL_FROZEN', code_sources=[binding(c) for c in code],
            source_authority=binding(AUTH), source_cache=v['payload'], cache_validation=binding(SIMPLE/'cache_validation.json'),
            public={k:a['public'][k] for k in ('split','gallery')},
            train_roles={str(f):a['folds'][str(f)]['train_roles'] for f in range(5)},
            curator=a['join']['curator'], sources=src, queries=128, candidates=128,
            arms=list(ARMS), endpoint='COARSE', feature_names=list(FEATURES),
            training=dict(loss='COST1', family='PRODUCT5', parameters=5, dtype='float64', seed=17,
                          updates=2000, learning_rate=.03, weight_decay=.001, optimizer='AdamW',
                          initialization='all_zero', threshold=0., held_selection=False,
                          split='Original H593 five identity/component folds intersected with cached first128'),
            source_tensor_validation='Every retained per-pair tensor SHA and both side u/v means are rechecked during extract.',
            scope='Opened first128 H593 queries, natural ColNomic C128; no new reader or model selection.',
            causal_boundary='Direct J/P input deletion at frozen COARSE predictor; native P may already contain interaction. Not deletion of every upstream source of corresponding information.',
            causal_level='DIRECT_PREDICTOR_INPUT_PATH',
            not_tested=['Phase-specific irreplaceability after retraining', 'Amplitude-specific irreplaceability after retraining'],
            compensation_boundary='Matched original PRODUCT5 refit tests compensation in this family/budget; a failure is not information absence or uniqueness.',
            labels_in_extraction=False, GPU=False)
        write(root/'protocol.json', p, once=True)
        write(root/'preparation_validation.json', dict(status='M_CAUSAL128_PREPARED', protocol=binding(root/'protocol.json'),queries=128,labels_read=False), once=True)
        emit(status='M_CAUSAL128_PREPARED', queries=128)
        return 0


def extract(root, shard, shards, budget):
    import torch
    torch.set_num_threads(1)
    p, pb = protocol(root)
    need(0 <= shard < shards, 'SHARD')
    cache = read(checked(p['source_cache']))
    started = time.monotonic()
    for i in range(shard, 128, shards):
        d = root/'queries'/f'{i:03d}'
        with lock(d/'worker.lock'):
            if (d/'validation.json').exists():
                v = read(d/'validation.json')
                need(v['protocol'] == pb, 'QUERY_RESUME_PROTOCOL')
                checked(v['payload'])
                continue
            src = p['sources'][i]
            iv = read(checked(src['inside_validation']))
            im = read(checked(src['inside_manifest']))
            na = read(checked(src['native_payload']))
            need(iv['payload'] == src['inside_manifest'], 'MANIFEST_SEAL')
            r = cache['rows'][i]
            progress = d/'checkpoint.json'
            prior = read(progress) if progress.exists() else dict(protocol=pb,index=i,rows=[])
            need(prior['protocol'] == pb and prior['index'] == i, 'EXTRACT_CHECKPOINT')
            records = prior['rows']
            for pos in range(len(records), 128):
                if time.monotonic()-started >= budget-10:
                    write(progress, dict(protocol=pb,index=i,rows=records))
                    emit(status='NORMAL_BUDGET_CHECKPOINT', query=i,pairs=len(records))
                    return 75
                mr = im['records'][pos]
                need(mr['candidate_position'] == pos and mr['physical_row'] == r['axis'][pos], 'PAIR_AXIS')
                b = im['sources'][pos]
                data = torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
                need(data['authority'] == im['authority'] and data['index'] == i, 'PAIR_SOURCE')
                need(data['query_sha'] == na['query_image']['sha256'], 'PAIR_QUERY_SHA')
                means, masses, maximum = {}, {}, 0.
                for arm in ARMS:
                    side_means = []
                    for s in (0,1):
                        item = data['sides'][s]['coarse'][arm]
                        weights = item['weights'].numpy()
                        need(weights.dtype == np.float64 and np.isfinite(weights).all(), 'FP64_WEIGHTS')
                        need((weights >= 0).all() and (weights <= 1).all(), 'WEIGHTS_RANGE')
                        mean = float(np.mean(weights))
                        error = abs(mean-item['mean_weight'])
                        need(error < 2e-12, 'SIDE_MEAN_REPLAY')
                        maximum = max(maximum,error)
                        side_means.append(mean)
                    mass = math.sqrt(side_means[0]*side_means[1])
                    error = abs(mass-mr['branches'][arm]['COARSE'])
                    need(error < 2e-12, 'M_REPLAY')
                    maximum = max(maximum,error)
                    # Preserve originally sealed M; independent replay is diagnostic.
                    masses[arm] = mr['branches'][arm]['COARSE']
                    means[arm] = side_means
                records.append(dict(position=pos,physical_row=r['axis'][pos],source=b,
                    mean_weights=means,M=masses,max_mass_replay_error=maximum))
                if (pos+1)%16 == 0:
                    write(progress, dict(protocol=pb,index=i,rows=records))
            need(len(records)==128, 'C128_COMPLETE')
            mm = {a:[z['M'][a] for z in records] for a in ARMS}
            raw = np.asarray(r['X'],np.float64)[:,0].tolist()
            # Recheck original M*L formula before copying M-independent evidence.
            ox = design(raw,r['free_content'],r['mass'],r['winner'],r['challengers'])
            need(np.max(np.abs(ox-np.asarray(r['X']))) < 2e-10, 'ORIGINAL_FREE_FORMULA')
            row = dict(index=i,execution_ordinal=i,query_id=r['query_id'],fold=src['fold'],
                source_image_sha256=r['source_image_sha256'],axis=r['axis'],candidate_physical_rows=r['axis'],
                winner=r['winner'],winner_index=r['winner'],challengers=r['challengers'],challenger_positions=r['challengers'],
                raw_feature=raw,free_content=r['free_content'],original_M=r['mass'],native_X=r['native_X'],M=mm,
                X={a:design(raw,r['free_content'],mm[a],r['winner'],r['challengers']).tolist() for a in ARMS},
                pair_audit=records,sources=src,labels_included=False)
            write(d/'payload.json',dict(protocol=pb,row=row),once=True)
            write(d/'validation.json',dict(status='M_CAUSAL128_QUERY_INDEPENDENT_M_PASS',protocol=pb,payload=binding(d/'payload.json'),
                pairs=128,arms=list(ARMS),max_mass_replay_error=max(z['max_mass_replay_error'] for z in records),
                tensor_shas_checked=128,side_weights_recomputed=1024,labels_read=False),once=True)
            emit(status='QUERY_EXTRACTED',index=i,pairs=128,seconds=time.monotonic()-started)
    return 0


def validate(root):
    p,pb = protocol(root)
    with lock(root/'validate.lock'):
        rows,seals = [],[]
        for i in range(128):
            d=root/'queries'/f'{i:03d}'
            if not (d/'validation.json').exists():
                emit(status='WAITING_FOR_EXTRACTION',index=i)
                return 75
            v=read(d/'validation.json');value=read(checked(v['payload']))
            need(v['status']=='M_CAUSAL128_QUERY_INDEPENDENT_M_PASS' and v['protocol']==value['protocol']==pb,'QUERY_VALIDATION')
            r=value['row'];need(r['index']==i and len(r['axis'])==len(set(r['axis']))==128,'AXIS128')
            need(sorted([r['winner']]+r['challengers'])==list(range(128)),'COMPLETE_ACTION_AXIS')
            im=read(checked(r['sources']['inside_manifest']))
            for a in ARMS:
                x=design(r['raw_feature'],r['free_content'],r['M'][a],r['winner'],r['challengers'])
                need(np.array_equal(x,np.asarray(r['X'][a])),'FEATURE_REPLAY')
                for k,z in enumerate(r['pair_audit']):
                    need(z['source']==im['sources'][k] and z['physical_row']==r['axis'][k],'AUDIT_SOURCE_AXIS')
                    need(r['M'][a][k]==im['records'][k]['branches'][a]['COARSE'],'M_SEALED_VALUE')
            rows.append(r);seals.append(binding(d/'validation.json'))
        old=read(checked(p['source_cache']))
        payload=dict(protocol=pb,rows=rows,prior_product_parameters=old['prior_product_parameters'],
                     native_parameters=old['native_parameters'],labels_included=False)
        write(root/'cache.json',payload,once=True)
        write(root/'cache_validation.json',dict(status='M_CAUSAL128_EXTRACT_ALL128_PASS',protocol=pb,cache=binding(root/'cache.json'),
            query_validations=seals,queries=128,candidates=16384,arms=list(ARMS),tensor_shas_checked=16384,
            independent_side_means=131072,labels_read=False),once=True)
        emit(status='M_CAUSAL128_EXTRACT_ALL128_PASS',queries=128)
    return 0


def cache_rows(root,pb):
    v=read(root/'cache_validation.json')
    need(v['status']=='M_CAUSAL128_EXTRACT_ALL128_PASS' and v['protocol']==pb,'CACHE_SEAL')
    payload=read(checked(v['cache']))
    need(payload['protocol']==pb and not payload['labels_included'],'CACHE_SCOPE')
    return payload['rows'],v['cache']


def train_split(p,rows,fold):
    split=read(checked(p['public']['split']))['folds'][fold]
    roles={r['query_id']:r for r in read(checked(p['train_roles'][str(fold)]))['records']}
    gallery={r['physical_row']:r['identity'] for r in read(checked(p['public']['gallery']))['records']}
    train=[i for i,r in enumerate(rows) if r['query_id'] in set(split['train_query_ids'])]
    held=[i for i,r in enumerate(rows) if r['query_id'] in set(split['heldout_query_ids'])]
    need(len(train)+len(held)==128 and not set(train)&set(held),'SPLIT128')
    need(set(roles)==set(split['train_query_ids']),'TRAIN_ROLES_ONLY')
    need(not {rows[i]['source_image_sha256'] for i in train}&{rows[i]['source_image_sha256'] for i in held},'IMAGE_DISJOINT')
    labels={}
    for i in train:
        r=rows[i];identity=roles[r['query_id']]['identity']
        order=[r['winner']]+r['challengers']
        labels[i]=next((j-1 for j,pos in enumerate(order) if gallery[r['axis'][pos]]==identity),-2)
    effective=[i for i in train if labels[i]>=-1]
    need(effective and held,'NONEMPTY_TRAIN_HELD')
    return train,held,effective,labels,roles


def unit_loss(torch,z,y):
    # Exact COST1 expression and tie gradient from the original LOSS.unit.
    raw=y==-1;ii=torch.arange(len(y));idx=y.clamp_min(0);wrong=z.clone()
    wrong[ii[~raw],idx[~raw]]=-torch.inf
    return torch.where(raw,torch.nn.functional.softplus(torch.amax(z,1)),
        torch.nn.functional.softplus(-z[ii,idx])+torch.nn.functional.softplus(torch.amax(wrong,1))).mean()


def save_torch(torch,path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def fit(root,fold,budget,threads):
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    p,pb=protocol(root);rows,cb=cache_rows(root,pb)
    need(fold in range(5),'FOLD')
    tr,held,eff,ys,roles=train_split(p,rows,fold)
    targets=torch.tensor([ys[i] for i in eff],dtype=torch.long)
    started=time.monotonic();d=root/f'fold{fold}'
    with lock(d/'fit.lock'):
        split=dict(protocol=pb,fold=fold,train_indices=tr,held_indices=held,effective_train_indices=eff,
                   effective_train_query_ids=[rows[i]['query_id'] for i in eff],
                   train_target_absent=len(tr)-len(eff),train_roles=p['train_roles'][str(fold)],
                   heldout_label_reads=0,cache=cb)
        write(d/'split.json',split,once=True)
        for arm in ARMS:
            resultpath=d/(arm+'.json')
            if resultpath.exists():
                old=read(resultpath);need(old['protocol']==pb and old['cache']==cb,'FIT_RESUME_SCOPE')
                continue
            if time.monotonic()-started>=budget-15:
                return 75
            x=torch.tensor([rows[i]['X'][arm] for i in eff],dtype=torch.float64)
            xh=torch.tensor([rows[i]['X'][arm] for i in held],dtype=torch.float64)
            torch.manual_seed(17)
            theta=torch.nn.Parameter(torch.zeros(5,dtype=torch.float64))
            opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
            cp=d/(arm+'.checkpoint.pt');step=0;trace=[];train_seconds=0.
            if cp.exists():
                state=torch.load(cp,map_location='cpu',weights_only=True)
                need(state['protocol']==pb and state['cache']==cb and state['arm']==arm and state['fold']==fold,'CHECKPOINT_SCOPE')
                need(state['threads']==threads,'CHECKPOINT_THREADS')
                with torch.no_grad():theta.copy_(state['theta'])
                opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
                step=state['step'];trace=state['trace'];train_seconds=state['train_seconds']
            tick=time.monotonic()
            def checkpoint():
                save_torch(torch,cp,dict(protocol=pb,cache=cb,fold=fold,arm=arm,threads=threads,
                    theta=theta.detach(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=step,
                    trace=trace,train_seconds=train_seconds+time.monotonic()-tick))
            while step<2000:
                if step%25==0 and time.monotonic()-started>=budget-10:
                    checkpoint();emit(status='FIT_CHECKPOINT',fold=fold,arm=arm,step=step);return 75
                opt.zero_grad();z=x@theta[:-1]+theta[-1];loss=unit_loss(torch,z,targets)
                need(bool(torch.isfinite(loss)),'FINITE_LOSS');loss.backward();opt.step();step+=1
                if step%100==0:
                    trace.append(dict(step=step,preupdate_cost=float(loss.detach())))
                if step%250==0:checkpoint()
            with torch.no_grad():
                z=x@theta[:-1]+theta[-1];zh=xh@theta[:-1]+theta[-1]
                fit_cost=float(unit_loss(torch,z,targets));target_action=targets.numpy()+1
                train_correct=int(np.sum(choice(z.numpy())==target_action))
            checkpoint()
            record=dict(status='M_CAUSAL128_PRODUCT5_COMPENSATION_COMPLETE',protocol=pb,cache=cb,fold=fold,arm=arm,
                feature_names=list(FEATURES),theta_hex=[float(v).hex() for v in theta.detach()],
                train_indices=eff,held_indices=held,train_logits=z.numpy().tolist(),held_logits=zh.numpy().tolist(),
                train_fit=dict(cost=fit_cost,correct=train_correct,total=len(eff),target_absent_excluded=len(tr)-len(eff)),
                train_trace=trace,training=p['training'],steps=step,threads=threads,
                seconds=train_seconds+time.monotonic()-tick,heldout_label_reads=0)
            # NumPy score/action reconstruction, not just serializing Torch results.
            par=theta.detach().numpy();nz=np.sum(xh.numpy()*par[:-1],axis=-1)+par[-1]
            error=float(np.max(np.abs(nz-zh.numpy())))
            need(error<2e-10 and np.array_equal(choice(nz),choice(zh.numpy())),'NUMPY_HELD_REPLAY')
            record['numpy_held_max_error']=error
            write(resultpath,record,once=True)
            emit(status=record['status'],fold=fold,arm=arm,fit_cost=fit_cost)
        models=[binding(d/(a+'.json')) for a in ARMS]
        write(d/'validation.json',dict(status='M_CAUSAL128_COMPENSATION_FOLD_PASS',protocol=pb,cache=cb,
            fold=fold,models=models,split=binding(d/'split.json'),heldout_label_reads=0),once=True)
    return 0


def join(root):
    p,pb=protocol(root);rows,cb=cache_rows(root,pb)
    with lock(root/'join.lock'):
        folds=[];seals=[];maximum=0.
        for f in range(5):
            d=root/f'fold{f}'
            if not (d/'validation.json').exists():return 75
            v=read(d/'validation.json');need(v['status']=='M_CAUSAL128_COMPENSATION_FOLD_PASS' and v['protocol']==pb and v['cache']==cb,'FOLD_VALIDATION')
            sp=read(checked(v['split']));mods={}
            for b in v['models']:
                m=read(checked(b));need(m['protocol']==pb and m['cache']==cb and m['fold']==f,'MODEL_SCOPE')
                need(m['training']==p['training'] and m['steps']==2000,'MATCHED_TRAINING')
                arm=m['arm'];need(arm in ARMS and arm not in mods,'MODEL_ARM')
                par=np.array([float.fromhex(z) for z in m['theta_hex']])
                for splitname in ('train','held'):
                    inds=m[splitname+'_indices'];xx=np.asarray([rows[i]['X'][arm] for i in inds])
                    z=np.sum(xx*par[:-1],axis=-1)+par[-1];stored=np.asarray(m[splitname+'_logits'])
                    error=float(np.max(np.abs(z-stored)));maximum=max(maximum,error)
                    need(error<2e-10 and np.array_equal(choice(z),choice(stored)),'NUMPY_FINAL_REPLAY')
                need(m['held_indices']==sp['held_indices'],'HELD_AXIS')
                mods[arm]=m
            need(set(mods)==set(ARMS),'ALL_FOUR_ARMS')
            folds.append((sp,mods));seals.append(binding(d/'validation.json'))
        write(root/'all_predictions_prelabel_seal.json',dict(protocol=pb,folds=seals),once=True)
        curator={r['query_id']:r for r in read(checked(p['curator']))['records']}
        gallery={r['physical_row']:r['identity'] for r in read(checked(p['public']['gallery']))['records']}
        resultrows=[]
        for f,(sp,mods) in enumerate(folds):
            trainroles={r['query_id']:r for r in read(checked(p['train_roles'][str(f)]))['records']}
            ids={r['identity'] for r in trainroles.values()};groups={r['component'] for r in trainroles.values()}
            for j,i in enumerate(sp['held_indices']):
                r=rows[i];role=curator[r['query_id']]
                need(role['outer_fold']==f and role['identity'] not in ids and role['component'] not in groups,'IDENTITY_COMPONENT_DISJOINT')
                order=[r['winner']]+r['challengers'];selected={'RAW':r['axis'][r['winner']]}
                logits={};margins={}
                target=next((k for k,pos in enumerate(order) if gallery[r['axis'][pos]]==role['identity']),None)
                for arm,m in mods.items():
                    z=np.asarray(m['held_logits'][j]);action=int(choice(z))
                    selected[arm]=r['axis'][order[action]];logits[arm]=z.tolist()
                    allz=np.r_[0.,z]
                    margins[arm]=None if target is None else float(allz[target]-np.max(np.delete(allz,target)))
                resultrows.append(dict(index=i,query_id=r['query_id'],fold=f,component=role['component'],
                    target_in_C128=target is not None,target_action=target,selected=selected,
                    correct={a:gallery[pos]==role['identity'] for a,pos in selected.items()},
                    logits=logits,target_vs_strongest_wrong_margin=margins))
        resultrows.sort(key=lambda r:r['index']);need([r['index'] for r in resultrows]==list(range(128)),'ALL128_ONCE')
        summary={}
        for a in ('RAW',)+ARMS:
            summary[a]=dict(correct=sum(r['correct'][a] for r in resultrows),total=128,
                rescue_vs_RAW=sum(r['correct'][a] and not r['correct']['RAW'] for r in resultrows),
                break_vs_RAW=sum(not r['correct'][a] and r['correct']['RAW'] for r in resultrows))
        comparisons={}
        for a in ARMS[1:]:
            group=defaultdict(list)
            for r in resultrows:group[r['component']].append(int(r['correct'][a])-int(r['correct']['A1J1P1']))
            values=np.array([np.mean(v) for _,v in sorted(group.items())]);rng=np.random.default_rng(20260927)
            boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
            comparisons[a]=dict(baseline='A1J1P1',group_equal_delta=float(values.mean()),groups=len(values),
                exploratory_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                caveat='Opened groups; intervals describe this fixed compensation protocol, not necessity or fresh confirmation.')
        value=dict(status='M_CAUSAL128_COMPENSATION_ALL5_INDEPENDENT_PASS',protocol=pb,cache=cb,
            summary=summary,comparisons=comparisons,rows=resultrows,
            fits=[dict(fold=f,arms={a:m['train_fit'] for a,m in mods.items()}) for f,(_,mods) in enumerate(folds)],
            maximum_numpy_logit_error=maximum,queries=128,models=20,external_GO=False,
            attribution_only=True,unique_cause_claim=False,information_absence_claim=False)
        value['causal_level']='DIRECT_PREDICTOR_INPUT_PATH'
        value['phase_or_amplitude_necessity_test']=False
        write(root/'result.json',value,once=True)
        report=['# H128 cached COARSE J/P entry deletion: matched compensation', '',
            'This is an opened H593 first128 mechanistic panel with its original five group folds and natural ColNomic C128. '
            'Four original PRODUCT5 heads are fit with the same COST1 loss, 2000 updates, seed17, FP64 and fixed zero threshold.', '',
            '| COARSE input | Correct | Rescue vs RAW | Break vs RAW |','|---|---:|---:|---:|']
        for a,s in summary.items():report.append(f"| {a} | {s['correct']}/128 | {s['rescue_vs_RAW']} | {s['break_vs_RAW']} |")
        report += ['', 'All product and mass features use the intervention M; free L is unchanged and contains no native RoMa u/v. '
            'No held labels select parameters or thresholds. Fitting quality is retained in result.json.', '',
            'P=0 removes the direct correspondence-distribution embedding input, not all spatial information. '
            'J=0 removes direct joint-context input, while P still comes from native interaction. '
            'The causal level is direct predictor input paths; this experiment does not complete the phase/amplitude property-specific removal-and-compensation step. '
            'Compensation failure does not prove information absence, unique causality, or irreducible necessity. '
            'Compensation success means this retained information and this readout can substitute within the evaluated protocol.', '']
        (root/'report.md').write_text('\n'.join(report))
        write(root/'validation.json',dict(status=value['status'],protocol=pb,result=binding(root/'result.json'),
             report=binding(root/'report.md'),fold_validations=seals,models=20,queries=128,maximum_numpy_logit_error=maximum),once=True)
        emit(status=value['status'],summary=summary)
    return 0


def self_test():
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(27);m=rng.uniform(.01,.8,128);l=rng.uniform(-.5,.8,128)
    w=17;c=[i for i in range(128) if i!=w];raw=rng.normal(size=127)
    x=design(raw,l,m,w,c);changed=design(raw,l,m[::-1],w,c)
    need(np.array_equal(x[:,0],changed[:,0]) and np.array_equal(x[:,3],changed[:,3]),'UNCHANGED_RAW_FREE_L')
    need(not np.array_equal(x[:,1:3],changed[:,1:3]),'ALL_M_DESCENDANTS_REPLACED')
    expected=[]
    for k in c:
        def scalar(a,b):return (a-b)/(abs(a)+abs(b)+1e-12)
        expected.append([raw[c.index(k)],scalar(m[k]*l[k],m[w]*l[w]),scalar(m[k],m[w]),scalar(l[k],l[w])])
    need(np.array_equal(x,np.asarray(expected)),'SCALAR_DESIGN_PARITY')
    sys.path.insert(0,str(RC/'programs'))
    import run_rc_six_cause_loss_binding_v1 as loss_source
    z=torch.tensor(rng.normal(size=(5,127)),dtype=torch.float64,requires_grad=True)
    y=torch.tensor([-1,2,11,9,-1]);a=unit_loss(torch,z,y);b=loss_source.unit(z,y)
    need(torch.equal(a,b),'ORIGINAL_LOSS_PARITY')
    need(torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0]),'ORIGINAL_GRADIENT_PARITY')
    # Serialization/resume uses identical optimizer state, including Adam step.
    xx=torch.tensor(np.stack([x]*5));g=torch.Generator().manual_seed(9)
    xx=xx+torch.randn(xx.shape,generator=g,dtype=torch.float64)*.01
    def run(n,t=None,o=None):
        t=torch.nn.Parameter(torch.zeros(5,dtype=torch.float64)) if t is None else t
        o=torch.optim.AdamW([t],lr=.03,weight_decay=.001) if o is None else o
        for _ in range(n):o.zero_grad();unit_loss(torch,xx@t[:-1]+t[-1],y).backward();o.step()
        return t,o
    import io
    full,_=run(30);part,opt=run(13);buf=io.BytesIO();torch.save(dict(t=part.detach(),o=opt.state_dict()),buf);buf.seek(0)
    state=torch.load(buf,weights_only=True);res=torch.nn.Parameter(state['t']);op=torch.optim.AdamW([res],lr=.03,weight_decay=.001);op.load_state_dict(state['o'])
    resumed,_=run(17,res,op);need(torch.equal(full,resumed),'BIT_EXACT_OPTIMIZER_RESUME')
    emit(status='M_CAUSAL128_SYNTHETIC_PASS',checks=['M_descendants','scalar_feature_formula','original_COST1_value_gradient','optimizer_resume_bit_exact'])
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','extract','validate','fit','join','self-test'])
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1);ap.add_argument('--budget',type=float,default=430)
    ap.add_argument('--fold',type=int,default=0);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    if a.command=='self-test':return self_test()
    if a.command=='prepare':return prepare(a.root)
    if a.command=='extract':return extract(a.root,a.shard,a.shards,a.budget)
    if a.command=='validate':return validate(a.root)
    if a.command=='fit':return fit(a.root,a.fold,a.budget,a.threads)
    return join(a.root)


if __name__=='__main__':
    sys.exit(main())
