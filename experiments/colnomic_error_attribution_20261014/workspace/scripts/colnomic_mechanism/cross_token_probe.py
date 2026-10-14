"""Cross-Token relation statistics, frozen identity heads, complete-gallery replay.

The head consumes a finite permutation-invariant summary of the complete raw
q x r dot matrix. It does not consume coordinates, raw Token indices, image IDs,
or identity IDs. Counts normalize summaries; a separate count-only control
explicitly receives Token cardinalities. Failed
readout is not proof that Token sets contain no identity/layout information.
All original nonzero Tokens, including templates, enter the matrix and MaxSim.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, rows, write, csvout, sha
from scripts.colnomic_mechanism.probes import logistic_fit, logistic_predict, paired_metrics, classification_metrics, validation_threshold, shuffled_labels, save_head
from scripts.colnomic_mechanism.completion_audit import fit as independent_fit, pairs as independent_pairs, classification as independent_classification, threshold_from_validation as independent_threshold

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
OUT=BASE/'cross_token_relation'
QUANTILES=[.05,.25,.5,.75,.95]
SUMMARIES=['mean','std','q05','q25','q50','q75','q95']
NAMES=[direction+'_'+measure+'_'+stat for direction in ['forward','reverse'] for measure in ['top1','top1_minus_top2'] for stat in SUMMARIES]
NAMES += [direction+'_'+stat for direction in ['forward','reverse'] for stat in ['assignment_entropy','maximum_target_load','target_coverage','effective_target_fraction','mutual_nearest_fraction']]
SEEDS=[17,31,53]
COUNT_NAMES=['log_valid_query_tokens','log_valid_reference_tokens','absolute_log_count_difference','query_reference_count_ratio']


def log(*message):print(*message,flush=True)


def load_head(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    return {k:saved[k].numpy() for k in ['mean','scale','coefficient']},saved


def valid(z):
    result=z.float()
    return result[torch.linalg.vector_norm(result,dim=1)>1e-8].contiguous()


def gpu_distribution(value, counts):
    width=value.shape[1]
    keep=torch.arange(width,device=value.device)[None] < counts[:,None]
    clean=torch.where(keep,value,0.)
    mean=clean.sum(1)/counts
    std=torch.sqrt((((clean-mean[:,None])**2)*keep).sum(1)/counts)
    ordered=torch.sort(torch.where(keep,value,-torch.inf),dim=1,descending=True).values
    outputs=[mean,std]
    for quantile in QUANTILES:
        index=(1-quantile)*(counts-1).float();lower=index.floor().long();upper=index.ceil().long()
        lo=ordered.gather(1,lower[:,None]).squeeze(1);hi=ordered.gather(1,upper[:,None]).squeeze(1)
        outputs.append(lo+(hi-lo)*(index-lower))
    return torch.stack(outputs,dim=1)


def gpu_relation(matrix, reference_counts):
    """Each batch row contains EVERY valid Token of one complete candidate."""
    batch,nq,nr=matrix.shape
    rkeep=torch.arange(nr,device=matrix.device)[None] < reference_counts[:,None]
    # Padding never competes for either best or runner-up matches.
    matrix=matrix.masked_fill(~rkeep[:,None,:],-torch.inf)
    fw=torch.topk(matrix,k=2,dim=2,sorted=True).values
    rv=torch.topk(matrix,k=2,dim=1,sorted=True).values
    ftop=fw[:,:,0];fgap=fw[:,:,0]-fw[:,:,1]
    rtop=rv[:,0,:];rgap=rv[:,0,:]-rv[:,1,:]
    qc=torch.full_like(reference_counts,nq)
    parts=[gpu_distribution(ftop,qc),gpu_distribution(fgap,qc),gpu_distribution(rtop,reference_counts),gpu_distribution(rgap,reference_counts)]
    # Ties share matching mass. This avoids order/ID-dependent argmax features.
    fequal=(matrix==ftop[:,:,None]) & rkeep[:,None,:]
    requal=(matrix==rtop[:,None,:]) & rkeep[:,None,:]
    fassignment=fequal.float()/fequal.sum(2,keepdim=True).clamp_min(1)
    fload=fassignment.sum(1)/nq
    mutual=(fassignment*requal).sum((1,2))/nq
    entropy=-(fload*fload.clamp_min(1e-30).log()).sum(1)/reference_counts.float().log()
    parts.append(torch.stack([entropy,fload.max(1).values,((fload>0)&rkeep).sum(1)/reference_counts,
                              1/(fload.square().sum(1)*reference_counts),mutual],1))
    del fassignment
    rassignment=requal.float()/requal.sum(1,keepdim=True).clamp_min(1)
    rload=rassignment.sum(2)/reference_counts[:,None]
    mutual=(rassignment*fequal).sum((1,2))/reference_counts
    entropy=-(rload*rload.clamp_min(1e-30).log()).sum(1)/math.log(nq)
    parts.append(torch.stack([entropy,rload.max(1).values,(rload>0).float().mean(1),1/(rload.square().sum(1)*nq),mutual],1))
    result=torch.cat(parts,1)
    assert result.shape==(batch,len(NAMES)) and bool(torch.isfinite(result).all())
    return result


def cpu_relation(matrix):
    """Independent NumPy implementation; consumes a serialized matrix directly."""
    value=np.asarray(matrix,dtype=np.float32);nq,nr=value.shape
    forward=np.sort(value,axis=1)[:,-2:];reverse=np.sort(value,axis=0)[-2:,:]
    ft=forward[:,-1];fg=forward[:,-1]-forward[:,-2];rt=reverse[-1];rg=reverse[-1]-reverse[-2]
    parts=[]
    for array in [ft,fg,rt,rg]:
        parts.extend([float(array.mean(dtype=np.float64)),float(array.std(dtype=np.float64))]+np.quantile(array,QUANTILES).tolist())
    f=(value==ft[:,None]);r=(value==rt[None,:]);fa=f/f.sum(1,keepdims=True);ra=r/r.sum(0,keepdims=True)
    for load,count,mutual in [(fa.sum(0)/nq,nr,float((fa*r).sum()/nq)),(ra.sum(1)/nr,nq,float((ra*f).sum()/nr))]:
        nonzero=load>0
        parts.extend([float(-(load[nonzero]*np.log(load[nonzero])).sum()/np.log(count)),float(load.max()),float(nonzero.mean()),
                      float(1/(np.square(load).sum()*count)),mutual])
    return np.asarray(parts,dtype=np.float32)


def relation_features(query_tokens,reference_tokens):
    """Public CPU helper: [nq,d] and [nr,d] -> float32[38], in NAMES order.

    Callers may select a declared ROI beforehand; this function receives only
    feature vectors and does not receive that ROI's coordinates or Token IDs.
    Whole-image main protocol includes every valid template Token. ROI callers
    must separately declare their selection supervision and image context.
    """
    def array(value):
        return value.detach().cpu().float().numpy() if isinstance(value,torch.Tensor) else np.asarray(value,dtype=np.float32)
    q=array(query_tokens);r=array(reference_tokens)
    q=q[np.linalg.norm(q,axis=1)>1e-8];r=r[np.linalg.norm(r,axis=1)>1e-8]
    assert q.ndim==r.ndim==2 and q.shape[1]==r.shape[1] and len(q)>=2 and len(r)>=2
    assert np.isfinite(q).all() and np.isfinite(r).all()
    return cpu_relation(q @ r.T)


def statistics_tests(output,device):
    rng=np.random.default_rng(13);matrix=rng.normal(size=(5,7)).astype(np.float32)
    original=cpu_relation(matrix);permuted=cpu_relation(matrix[[4,1,2,0,3]][:,[6,3,0,4,2,1,5]])
    permutation_error=float(np.max(np.abs(original-permuted)));assert permutation_error<1e-6
    tied=cpu_relation(np.ones((3,4),dtype=np.float32)*2)
    assert np.allclose(tied[[0,14]],2) and np.allclose(tied[[7,21]],0)
    assert np.allclose(tied[[28,30,31,32,33,35,36,37]],1)
    assert np.isclose(tied[29],.25) and np.isclose(tied[34],1/3)
    batch=np.stack([matrix,matrix.copy()]);batch[1,:,5:]=1000.
    actual=gpu_relation(torch.tensor(batch,device=device),torch.tensor([7,5],device=device)).cpu().numpy()
    expected=np.stack([cpu_relation(matrix),cpu_relation(matrix[:,:5])]);error=float(np.max(np.abs(actual-expected)));assert error<3e-6
    result=dict(status='passed',CPU_permutation_error=permutation_error,padded_GPU_reduction_max_error=error,
                checks=['query and reference permutations preserve relation statistics','exact tied maxima share mass','invalid padded candidate Tokens cannot win top1/top2'])
    write(output/'relation_statistics_tests.json',result)


def batches(order,size):
    return [np.asarray(order[i:i+size],dtype=np.int32) for i in range(0,len(order),size)]


def matrix_batch(q, indices, references, device):
    counts=np.array([len(references[int(i)]) for i in indices],dtype=np.int64);maximum=int(counts.max())
    block=torch.zeros((len(indices),maximum,128),dtype=torch.float32)
    for j,ri in enumerate(indices):block[j,:counts[j]]=references[int(ri)]
    block=block.to(device);counts=torch.tensor(counts,device=device)
    # One GEMM; candidate boundaries are restored before reductions.
    dot=(q @ block.flatten(0,1).T).reshape(len(q),len(indices),maximum).permute(1,0,2)
    return dot,counts


def cpu_predict(model,x):
    return ((x-model['mean'])/model['scale']) @ model['coefficient'][:-1]+model['coefficient'][-1]


def count_features(query_count,reference_counts):
    counts=np.asarray(reference_counts,dtype=np.float64)
    qlog=np.full(counts.shape,np.log(float(query_count)));rlog=np.log(counts)
    return np.c_[qlog,rlog,np.abs(qlog-rlog),float(query_count)/counts].astype(np.float32)


def inputs_for_head(kind,value,query_count,reference_counts):
    return count_features(query_count,reference_counts) if kind.startswith('count_only') else value


def benchmark(d,references,output,args):
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','unspecified').replace(',','_')
    destination=output/'benchmarks'/f'GPU{physical}_batch{args.batch_size}';destination.mkdir(parents=True,exist_ok=True)
    qi=0;q=valid(d['qb']['embeddings'][qi]).to(args.device)
    rng=np.random.default_rng(20261013);chosen=np.sort(rng.choice(len(references),100,replace=False));group=batches(chosen,args.batch_size)
    elapsed=[];records=[];errors=[];pointwise=[]
    # Warm kernels; do not include initialization in steady-state estimate.
    dot,count=matrix_batch(q,group[0],references,args.device);gpu_relation(dot,count);torch.cuda.synchronize()
    start=time.monotonic()
    for bi,indices in enumerate(group):
        tick=time.monotonic();dot,count=matrix_batch(q,indices,references,args.device);features=gpu_relation(dot,count).cpu().numpy()
        torch.cuda.synchronize();elapsed.append(time.monotonic()-tick)
        for j,ri in enumerate(indices):
            original=float(d['image']['scores'][qi,int(ri)]);error=abs(float(features[j,0])-original)
            records.append(dict(query_index=qi,reference_index=int(ri),raw_MaxSim_error=error))
            if len(errors)<8:
                m=dot[j,:,:int(count[j])].cpu().numpy();reconstructed=cpu_relation(m)
                e=float(np.max(np.abs(features[j]-reconstructed)));errors.append(e)
                old=valid(d['qb']['embeddings'][qi]) @ references[int(ri)].T
                pointwise.append(float(np.max(np.abs(old.numpy()-m))))
                np.savez_compressed(destination/f'matrix_{qi}_{int(ri)}.npz',matrix=m,features=features[j],reference_index=np.array(int(ri)),query_index=np.array(qi))
    wall_seconds=time.monotonic()-start;duration=sum(elapsed);pair_rate=100/duration;total=1117*5405
    result=dict(status='passed' if max(errors)<3e-6 and max(r['raw_MaxSim_error'] for r in records)<3e-6 else 'failed',
                pairs=100,seconds=duration,wall_seconds_with_independent_checks=wall_seconds,pairs_per_second=pair_rate,full_gallery_pairs=total,
                arithmetic_ETA_seconds=total/pair_rate,batch_size=args.batch_size,device=args.device,
                matrix_reduction_replay_errors=errors,CPU_GPU_dot_max_errors=pointwise,
                raw_MaxSim_max_error=max(r['raw_MaxSim_error'] for r in records),
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope='100 uniformly sampled reference rows, fixed original first query; ETA excludes disk/hash/ranking and variability')
    write(destination/'result.json',result);csvout(destination/'pairs.csv',records)
    log('Benchmark',json.dumps(result,ensure_ascii=False))
    assert result['status']=='passed','Benchmark independent reductions/MaxSim failed'
    return result


def features_full(d,references,output,args):
    folder=output/'features';folder.mkdir(exist_ok=True)
    worker=output/'workers'/f'partition_{args.partition_id}';worker.mkdir(parents=True,exist_ok=True)
    lock=(worker/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pattern=args.partition_pattern or list(range(args.partition_count))
    selected=[i for i in range(len(d['qb']['ids'])) if pattern[i%len(pattern)]==args.partition_id]
    order=sorted(range(len(references)),key=lambda i:(len(references[i]),i));plans=batches(order,args.batch_size)
    maximum=0.;times=[];done=0
    for qi,qid in enumerate(d['qb']['ids']):
        if pattern[qi%len(pattern)]!=args.partition_id:continue
        featurepath=folder/(qid+'.npz');commit=folder/(qid+'.json')
        if commit.exists():
            record=read(commit);assert sha(featurepath)==record['sha256'] and record['photo_id']==qid
            maximum=max(maximum,record['raw_MaxSim_max_error']);done+=1;continue
        started=time.monotonic();q=valid(d['qb']['embeddings'][qi]).to(args.device)
        value=np.empty((5405,len(NAMES)),dtype=np.float32)
        for indices in plans:
            dot,count=matrix_batch(q,indices,references,args.device)
            value[indices]=gpu_relation(dot,count).cpu().numpy()
        error=float(np.max(np.abs(value[:,0]-d['image']['scores'][qi].numpy())))
        assert error<3e-6,(qid,error)
        assert np.isfinite(value).all()
        temporary=featurepath.with_suffix('.tmp.npz');np.savez_compressed(temporary,features=value,query_index=np.array(qi,dtype=np.int32),
                valid_query_tokens=np.array(len(q),dtype=np.int32),valid_reference_tokens=np.array([len(r) for r in references],dtype=np.int32))
        temporary.replace(featurepath)
        seconds=time.monotonic()-started;times.append(seconds);maximum=max(maximum,error)
        write(commit,dict(photo_id=qid,query_index=qi,reference_rows=5405,features=len(NAMES),sha256=sha(featurepath),bytes=featurepath.stat().st_size,
                          raw_MaxSim_max_error=error,seconds=seconds,device=args.device,batch_size=args.batch_size,valid_query_tokens=len(q),source_sha256=args.executed_source_sha256))
        done+=1
        write(worker/'progress.json',dict(stage='complete_gallery_relation_features',partition_id=args.partition_id,partition_count=args.partition_count,
                completed_queries=done,target_queries=len(selected),global_total_queries=1117,reference_rows=5405,last_query_index=qi,last_photo_id=qid,
                partition_pattern=pattern,
                maximum_raw_MaxSim_error=maximum,mean_recent_query_seconds=float(np.mean(times[-20:])),ETA_seconds=float(np.mean(times[-20:])*(len(selected)-done))))
        log('Partition',args.partition_id,'full relation features',done,'/',len(selected),'query index',qi,'seconds',round(seconds,2),'MaxSim error',error)
    write(worker/'receipt.json',dict(status='complete',query_indices=selected,queries=len(selected),reference_rows=5405,feature_dimensions=len(NAMES),
            raw_MaxSim_max_error=maximum,partition_id=args.partition_id,partition_count=args.partition_count,source_sha256=args.executed_source_sha256,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),batch_size=args.batch_size))
    if args.partition_count==1:aggregate_features(d,output)


def aggregate_features(d,output):
    records=[];maximum=0.
    for qi,qid in enumerate(d['qb']['ids']):
        featurepath=output/'features'/(qid+'.npz');commit=read(featurepath.with_suffix('.json'))
        assert commit['photo_id']==qid and commit['query_index']==qi and commit['reference_rows']==5405 and commit['features']==len(NAMES)
        assert sha(featurepath)==commit['sha256']
        value=np.load(featurepath)['features'];assert value.shape==(5405,len(NAMES)) and value.dtype==np.float32 and np.isfinite(value).all()
        error=float(np.max(np.abs(value[:,0]-d['image']['scores'][qi].numpy())));assert error<3e-6
        maximum=max(maximum,error);records.append(commit)
    assert len(records)==len({r['photo_id'] for r in records})==1117
    receipts=[read(p) for p in (output/'workers').glob('partition_*/receipt.json')]
    covered=[i for r in receipts for i in r['query_indices']]
    assert sorted(covered)==list(range(1117)) and len({r['source_sha256'] for r in receipts})==1,'missing/duplicate partition or mixed executed source'
    assert {r['source_sha256'] for r in records}=={r['source_sha256'] for r in receipts},'feature commits do not match worker executed sources'
    write(output/'features_complete.json',dict(status='complete',queries=len(records),reference_rows=5405,feature_dimensions=len(NAMES),
            raw_MaxSim_max_error=maximum,query_commit_files=len(records),partitions=receipts,no_candidate_truncation=True))
    write(output/'feature_file_manifest.json',records)
    log('All full-gallery feature partitions independently aggregated',len(records))


def bootstrap(values,packages,seed=20261013):
    ids=sorted(set(packages));means=np.array([np.mean(np.asarray(values)[packages==p]) for p in ids]);rng=np.random.default_rng(seed)
    draw=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    return dict(identity_macro_mean=float(means.mean()),ci95_low=float(np.quantile(draw,.025)),ci95_high=float(np.quantile(draw,.975)),identities=len(ids),rounds=2000)


def independent_bootstrap(values,packages):
    package_names=np.unique(packages)
    group_values=np.array([np.asarray(values)[np.flatnonzero(packages==pid)].sum()/np.count_nonzero(packages==pid) for pid in package_names])
    resampling=np.random.default_rng(20261013).integers(0,len(package_names),(2000,len(package_names)))
    samples=np.take(group_values,resampling).sum(axis=1)/len(package_names)
    low,high=np.percentile(samples,[2.5,97.5])
    return dict(identity_macro_mean=float(group_values.sum()/len(package_names)),ci95_low=float(low),ci95_high=float(high),identities=len(package_names),rounds=2000)


def restricted_inputs(d,output):
    prior=ROOT/'results/token_mechanism_experiments/20261013_v2/stage3_probes'
    index=rows(prior/'identity_sample_index.csv');ri={rid:i for i,rid in enumerate(d['rb']['ids'])};qi={qid:i for i,qid in enumerate(d['qb']['ids'])}
    x=[];cx=[];last=None;value=None;counts=None
    for row in index:
        if row['photo_id']!=last:
            stored=np.load(output/'features'/(row['photo_id']+'.npz'));value=stored['features'];last=row['photo_id']
            counts=count_features(int(stored['valid_query_tokens']),stored['valid_reference_tokens'])
        x.append(value[ri[row['reference_id']]])
        cx.append(counts[ri[row['reference_id']]])
    x=np.array(x);y=np.array([int(r['label']) for r in index]);p=np.array([r['package_id'] for r in index]);s=np.array([r['split'] for r in index]);q=np.array([r['photo_id'] for r in index])
    original=np.load(prior/'identity_probe_inputs.npz')
    assert np.array_equal(y,original['labels']) and np.array_equal(p,original['package_ids']) and np.array_equal(s,original['splits']) and np.array_equal(q,original['query_ids'])
    allowed={rid for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json') if d['splits'][r['photo_id']]=='train' for rid in r['positive_reference_ids']}
    assert all(r['reference_id'] in allowed for r in index if r['split']=='train' and int(r['label'])==0),'training negatives cross identity split'
    np.savez_compressed(output/'restricted_inputs.npz',features=x,labels=y,package_ids=p,splits=s,query_ids=q,
                        reference_ids=np.array([r['reference_id'] for r in index]))
    np.savez_compressed(output/'restricted_count_inputs.npz',features=np.array(cx),labels=y,package_ids=p,splits=s,query_ids=q,
                        reference_ids=np.array([r['reference_id'] for r in index]))
    csvout(output/'restricted_sample_index.csv',index)
    return x,y,p,s,q,index


def fit_heads(d,output):
    x,y,p,s,q,index=restricted_inputs(d,output);train=s=='train';val=s=='val';weight=np.array([1./Counter(p)[pid] for pid in p])
    families={'relation':x,'count_only':np.load(output/'restricted_count_inputs.npz')['features']};heads={};predictions=[];metrics=[]
    for family,values in families.items():
        candidates=[];grid=[]
        for c in [.01,.1,1.,10.]:
            model=logistic_fit(values[train],y[train],c,weight[train]);scores=logistic_predict(model,values[val]);pm,_=paired_metrics(y[val],scores,q[val],p[val]);auc=classification_metrics(y[val],scores,p[val])['auc']
            candidates.append((pm['package_macro_pair_accuracy'],auc,-c,c,model));grid.append(dict(C=c,**pm,auc=auc))
        best=max(candidates,key=lambda r:r[:3]);c=best[3];model=best[4];threshold=validation_threshold(y[val],logistic_predict(model,values[val]),p[val]);names=NAMES if family=='relation' else COUNT_NAMES
        save_head(output/(family+'_head.pt'),model,dict(C=c,threshold=threshold,feature_names=names))
        write(output/('selection.json' if family=='relation' else 'count_only_selection.json'),dict(C=c,threshold=threshold,candidates=grid,validation_query_ids=q[val].tolist(),validation_package_ids=sorted(set(p[val])),
                        train_samples=int(train.sum()),val_samples=int(val.sum()),test_used_for_selection=False,
                        selection='validation identity-macro within-query accuracy, AUC, smaller C',threshold_selection='validation identity-macro classification',feature_names=names))
        heads[family]=(model,threshold)
        for seed in SEEDS:
            shuffled=shuffled_labels(y[train],p[train],seed);null=logistic_fit(values[train],shuffled,c,weight[train]);cutoff=validation_threshold(y[val],logistic_predict(null,values[val]),p[val])
            save_head(output/f'{family}_shuffle_{seed}_head.pt',null,dict(C=c,threshold=cutoff,seed=seed,feature_names=names))
            heads[f'shuffle_{seed}' if family=='relation' else f'count_only_shuffle_{seed}']=(null,cutoff)
    for kind,(head,cutoff) in heads.items():
        values=families['count_only' if kind.startswith('count_only') else 'relation'];scores=logistic_predict(head,values)
        shuffled=shuffled_labels(y[train],p[train],int(kind.rsplit('_',1)[-1])) if 'shuffle_' in kind else None;training_labels=np.full(len(y),-1,dtype=np.int8)
        if shuffled is not None:training_labels[train]=shuffled
        predictions.extend(dict(head=kind,**r,logit=float(v),threshold=cutoff,shuffled_training_label=int(sy) if sy>=0 else None) for r,v,sy in zip(index,scores,training_labels))
        for part in ['train','val','test']:
            keep=s==part;pm,_=paired_metrics(y[keep],scores[keep],q[keep],p[keep])
            for strategy,t in [('fixed_zero',0.),('validation_calibrated',cutoff)]:
                metrics.append(dict(head=kind,scope='original2234 restricted same-split pairs',split=part,threshold_strategy=strategy,
                    **{**classification_metrics(y[keep],scores[keep],p[keep],t),**pm}))
    csvout(output/'restricted_predictions.csv',predictions);csvout(output/'restricted_metrics.csv',metrics)
    return heads


def natural_hard(d,output,heads):
    ri={rid:i for i,rid in enumerate(d['rb']['ids'])};samples=[];contrasts=[];prediction=[];gaps=[];allmetrics=[]
    for qi,qid in enumerate(d['qb']['ids']):
        comparisons=read(BASE/'full_tokens/queries'/qid/'comparisons.json');stored=np.load(output/'features'/(qid+'.npz'));value=stored['features'];photo=d['photos'][qid]
        selected=sorted({r['positive'] for r in comparisons}|{r['negative'] for r in comparisons},key=lambda rid:ri[rid]);positive={r['positive'] for r in comparisons}
        positions={rid:len(samples)+j for j,rid in enumerate(selected)}
        for rid in selected:samples.append(dict(photo_id=qid,reference_id=rid,package_id=photo['package_id'],split=d['splits'][qid],label=int(rid in positive),query_index=qi,reference_index=ri[rid]))
        for row in comparisons:contrasts.append(dict(photo_id=qid,package_id=photo['package_id'],split=d['splits'][qid],positive_reference_id=row['positive'],negative_reference_id=row['negative'],
                    negative_roles=row['negative_roles'],positive_index=positions[row['positive']],negative_index=positions[row['negative']]))
        for kind,(head,threshold) in heads.items():
            values=inputs_for_head(kind,value,int(stored['valid_query_tokens']),stored['valid_reference_tokens'])
            scores=logistic_predict(head,values[[ri[rid] for rid in selected]])
            prediction.extend(dict(head=kind,**row,logit=float(v),threshold=threshold) for row,v in zip(samples[-len(selected):],scores))
    assert len(contrasts)==1648 and len({r['photo_id'] for r in contrasts})==1117
    lookup={(r['head'],r['photo_id'],r['reference_id']):r['logit'] for r in prediction}
    for kind,(head,threshold) in heads.items():
        result=[]
        for pair in contrasts:
            ps=lookup[kind,pair['photo_id'],pair['positive_reference_id']];ns=lookup[kind,pair['photo_id'],pair['negative_reference_id']];gap=ps-ns
            result.append(dict(head=kind,**pair,positive_logit=ps,negative_logit=ns,gap=gap,correct=1. if gap>0 else .5 if gap==0 else 0.))
        gaps.extend(result)
        for part in ['all','train','val','test']:
            keep=[r for r in result if part=='all' or r['split']==part];packages=np.array([r['package_id'] for r in keep]);qs=np.array([r['photo_id'] for r in keep]);correct=np.array([r['correct'] for r in keep]);margin=np.array([r['gap'] for r in keep])
            pr=[r for r in prediction if r['head']==kind and (part=='all' or r['split']==part)]
            allmetrics.append(dict(head=kind,split=part,scope='1648 original positive-vs-image/fusion strongest wrong contrasts',pairs=len(keep),queries=len(set(qs)),pair_accuracy=float(correct.mean()),
                      query_macro_pair_accuracy=float(np.mean([correct[qs==qid].mean() for qid in set(qs)])),gap_mean=float(margin.mean()),gap_median=float(np.median(margin)),
                      identity_bootstrap=bootstrap(correct,packages),classification=classification_metrics(np.array([r['label'] for r in pr]),np.array([r['logit'] for r in pr]),np.array([r['package_id'] for r in pr]),threshold)))
    csvout(output/'natural_hard_sample_index.csv',samples);csvout(output/'natural_hard_contrast_index.csv',contrasts)
    csvout(output/'natural_hard_predictions.csv',prediction);csvout(output/'natural_hard_pair_gaps.csv',gaps);write(output/'natural_hard_metrics.json',allmetrics)


def gallery_ranks(d,output,heads):
    gallery=output/'full_gallery';gallery.mkdir(exist_ok=True);ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    hashes=np.asarray(d['rb']['sha256']);names,hashcodes=np.unique(hashes,return_inverse=True)
    results={};metrics=[]
    for kind,(model,threshold) in heads.items():
        folder=gallery/kind;folder.mkdir(exist_ok=True)
        scores=np.lib.format.open_memmap(folder/'scores.npy',mode='w+',dtype=np.float64,shape=(1117,5405));sortedindex=np.lib.format.open_memmap(folder/'stable_sorted_reference_indices.npy',mode='w+',dtype=np.int32,shape=(1117,5405));sharank=np.lib.format.open_memmap(folder/'sha_design_ranks.npy',mode='w+',dtype=np.int32,shape=(1117,5405))
        result=[];contributions=[];positive_indices=[];negative_indices=[]
        for qi,qid in enumerate(d['qb']['ids']):
            stored=np.load(output/'features'/(qid+'.npz'));value=inputs_for_head(kind,stored['features'],int(stored['valid_query_tokens']),stored['valid_reference_tokens'])
            score=logistic_predict(model,value);scores[qi]=score;order=np.argsort(-score,kind='stable');sortedindex[qi]=order
            unique,first=np.unique(hashcodes[order],return_index=True);firstorder=np.argsort(first);ranks=np.empty(len(names),dtype=np.int32);ranks[unique[firstorder]]=np.arange(1,len(names)+1);sharank[qi]=ranks[hashcodes]
            positives=set(ledger[qid]['positive_reference_sha256']);rank=min(ranks[np.flatnonzero(names==h)[0]] for h in positives)
            positive=next(int(i) for i in order if hashes[i] in positives);negative=next(int(i) for i in order if hashes[i] not in positives)
            feature_difference=(value[positive].astype(np.float64)-value[negative].astype(np.float64))/model['scale']
            component=feature_difference*model['coefficient'][:-1];gap=float(score[positive]-score[negative])
            assert abs(float(component.sum())-gap)<1e-10
            contributions.append(component);positive_indices.append(positive);negative_indices.append(negative)
            base=int(ledger[qid]['image_positive_rank'])
            result.append(dict(head=kind,photo_id=qid,package_id=d['photos'][qid]['package_id'],split=d['splits'][qid],positive_rank=int(rank),baseline_image_rank=base,
                        best_positive_reference_id=d['rb']['ids'][positive],strongest_wrong_reference_id=d['rb']['ids'][negative],positive_logit=float(score[positive]),wrong_logit=float(score[negative]),logit_gap=gap,
                        original_MaxSim_same_pair_gap=float(d['image']['scores'][qi,positive]-d['image']['scores'][qi,negative]),
                        top_reference_id=d['rb']['ids'][order[0]],correct=int(rank)==1,repaired=base>1 and rank==1,harmed=base==1 and rank>1))
        scores.flush();sortedindex.flush();sharank.flush();csvout(folder/'query_rankings.csv',result);results[kind]=result
        np.savez_compressed(folder/'per_query_feature_gap_decomposition.npz',contributions=np.array(contributions,dtype=np.float64),
                            positive_reference_indices=np.array(positive_indices,dtype=np.int32),negative_reference_indices=np.array(negative_indices,dtype=np.int32),
                            feature_names=np.array(COUNT_NAMES if kind.startswith('count_only') else NAMES),query_ids=np.array(d['qb']['ids']))
        for part in ['all','train','val','test']:
            selected=[r for r in result if part=='all' or r['split']==part];rank=np.array([r['positive_rank'] for r in selected]);base=np.array([r['baseline_image_rank'] for r in selected]);packages=np.array([r['package_id'] for r in selected]);delta=(rank==1).astype(float)-(base==1)
            metrics.append(dict(head=kind,split=part,queries=len(rank),identities=len(set(packages)),recall_at_1=float((rank<=1).mean()),recall_at_5=float((rank<=5).mean()),recall_at_10=float((rank<=10).mean()),mrr=float((1/rank).mean()),
                          identity_macro_recall_at_1=float(np.mean([(rank[packages==p]<=1).mean() for p in set(packages)])),identity_macro_mrr=float(np.mean([(1/rank[packages==p]).mean() for p in set(packages)])),
                          repaired=int(((base>1)&(rank==1)).sum()),harmed=int(((base==1)&(rank>1)).sum()),net_change=int((rank==1).sum()-(base==1).sum()),identity_paired_bootstrap=bootstrap(delta,packages)))
        log('Full-gallery head ranked',kind)
    csvout(gallery/'metrics.csv',metrics);write(gallery/'score_axes.json',dict(query_ids=d['qb']['ids'],reference_ids=d['rb']['ids'],reference_sha256=d['rb']['sha256'],
                unique_reference_designs=len(set(hashes)),ranking='stable descending scalar then original reference row; first SHA occurrence',all_reference_rows_scored=True))


def independent_verify(d,output):
    checks={};a=dict(np.load(output/'restricted_inputs.npz'));x,y,p,s,q=[a[k] for k in ['features','labels','package_ids','splits','query_ids']]
    ca=dict(np.load(output/'restricted_count_inputs.npz'));checks['count_control_input_axes_match']=all(np.array_equal(ca[k],a[k]) for k in ['labels','package_ids','splits','query_ids','reference_ids'])
    families={'relation':x,'count_only':ca['features']};train=s=='train';val=s=='val';count=Counter(p);w=np.array([1./count[pid] for pid in p])
    allpred=rows(output/'restricted_predictions.csv');heads={};errors={}
    for family,values in families.items():
        selection=read(output/('selection.json' if family=='relation' else 'count_only_selection.json'));model,stored=load_head(output/(family+'_head.pt'))
        mean=np.average(values[train],axis=0,weights=w[train]);scale=np.maximum(np.sqrt(np.average((values[train]-mean)**2,axis=0,weights=w[train])),1e-5)
        checks[family+'_train_only_mean_scale']=np.array_equal(model['mean'],mean) and np.array_equal(model['scale'],scale)
        choices=[];candidate_error=0.
        for c in [.01,.1,1.,10.]:
            fitted=independent_fit(values[train],y[train],c,w[train]);scores=cpu_predict(fitted,values[val]);pm=independent_pairs(y[val],scores,q[val],p[val]);auc=independent_classification(y[val],scores,p[val],0.)['auc'];choices.append((pm['package_macro_pair_accuracy'],auc,-c,c,fitted))
            original=next(r for r in selection['candidates'] if r['C']==c);candidate_error=max(candidate_error,abs(original['package_macro_pair_accuracy']-pm['package_macro_pair_accuracy']),abs(original['auc']-auc))
        best=max(choices,key=lambda r:r[:3]);checks[family+'_selected_C_and_refit_coefficients']=best[3]==selection['C']==stored['C'] and np.max(np.abs(best[4]['coefficient']-model['coefficient']))<1e-7 and candidate_error<1e-10
        cutoff=independent_threshold(y[val],cpu_predict(model,values[val]),p[val]);checks[family+'_validation_threshold_replayed']=abs(cutoff-stored['threshold'])<1e-10
        heads[family]=(model,float(stored['threshold']))
        for seed in SEEDS:
            key=f'shuffle_{seed}' if family=='relation' else f'count_only_shuffle_{seed}';nm,ns=load_head(output/f'{family}_shuffle_{seed}_head.pt');shuffled=y[train].copy();rng=np.random.default_rng(seed)
            for pid in sorted(set(p[train])):
                pos=np.flatnonzero(p[train]==pid);shuffled[pos]=shuffled[rng.permutation(pos)]
            fit=independent_fit(values[train],shuffled,stored['C'],w[train]);nr=[r for r in allpred if r['head']==key];observed=np.array([int(r['shuffled_training_label']) for r in nr if r['split']=='train'])
            checks[key+'_labels_coefficients_standardization']=np.array_equal(shuffled,observed) and np.max(np.abs(fit['coefficient']-nm['coefficient']))<1e-7 and np.array_equal(nm['mean'],mean) and np.array_equal(nm['scale'],scale)
            threshold=independent_threshold(y[val],cpu_predict(nm,values[val]),p[val]);checks[key+'_threshold']=abs(threshold-ns['threshold'])<1e-10
            heads[key]=(nm,float(ns['threshold']))
    hardpred=rows(output/'natural_hard_predictions.csv');contrasts=rows(output/'natural_hard_contrast_index.csv');checks['natural_1648_contrast_1117_query_coverage']=len(contrasts)==1648 and {r['photo_id'] for r in contrasts}==set(d['qb']['ids'])
    axes=read(output/'full_gallery/score_axes.json');checks['full_gallery_axes_match_original_banks']=axes['query_ids']==d['qb']['ids'] and axes['reference_ids']==d['rb']['ids'] and axes['reference_sha256']==d['rb']['sha256']
    ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')};hashes=np.asarray(d['rb']['sha256']);ri={rid:i for i,rid in enumerate(d['rb']['ids'])};rankmismatch=0;maxscoreerror=0.;maxraw=0.;featurecommits=0
    arrays={kind:{k:np.load(output/'full_gallery'/kind/(k+'.npy'),mmap_mode='r') for k in ['scores','stable_sorted_reference_indices','sha_design_ranks']} for kind in heads}
    queryrows={kind:rows(output/'full_gallery'/kind/'query_rankings.csv') for kind in heads}
    decompositions={kind:dict(np.load(output/'full_gallery'/kind/'per_query_feature_gap_decomposition.npz')) for kind in heads}
    true_reference_counts=np.array([len(valid(z)) for z in d['rb']['embeddings']],dtype=np.int32)
    maximum_count_input_error=0.
    rng=np.random.default_rng(2026101401)
    feature_spot_indices={int(qi):int(ri) for qi,ri in zip(rng.choice(1117,32,replace=False),rng.integers(5405,size=32))}
    feature_spot=[];feature_spot_error=0.
    reference_to_column={rid:i for i,rid in enumerate(d['rb']['ids'])}
    restricted_positions={qid:np.flatnonzero(q==qid) for qid in set(q)}
    for qi,qid in enumerate(d['qb']['ids']):
        path=output/'features'/(qid+'.npz');commit=read(path.with_suffix('.json'));checks[qid+'_features_SHA']=sha(path)==commit['sha256'];featurecommits+=1
        record=np.load(path);value=record['features'];maxraw=max(maxraw,float(np.max(np.abs(value[:,0]-d['image']['scores'][qi].numpy()))))
        true_query_count=len(valid(d['qb']['embeddings'][qi]));checks[qid+'_valid_counts_replayed_from_original_tensors']=int(record['valid_query_tokens'])==true_query_count and np.array_equal(record['valid_reference_tokens'],true_reference_counts)
        if qi in feature_spot_indices:
            rindex=feature_spot_indices[qi]
            dot=valid(d['qb']['embeddings'][qi]).numpy()@valid(d['rb']['embeddings'][rindex]).numpy().T
            independently_reduced=cpu_relation(dot)
            error=np.abs(independently_reduced-value[rindex]);feature_spot_error=max(feature_spot_error,float(error.max()))
            feature_spot.append(dict(photo_id=qid,reference_id=d['rb']['ids'][rindex],query_index=qi,reference_index=rindex,
                              query_tokens=dot.shape[0],reference_tokens=dot.shape[1],maximum_feature_error=float(error.max()),
                              maximum_error_feature=NAMES[int(error.argmax())]))
        rcount=true_reference_counts.astype(np.float64);ql=np.full(len(rcount),math.log(true_query_count));rl=np.log(rcount)
        independent_counts=np.column_stack([ql,rl,np.abs(ql-rl),true_query_count/rcount]).astype(np.float32)
        positions=restricted_positions[qid];columns=[reference_to_column[rid] for rid in a['reference_ids'][positions]]
        maximum_count_input_error=max(maximum_count_input_error,float(np.max(np.abs(independent_counts[columns]-ca['features'][positions]))))
        for kind,(head,threshold) in heads.items():
            values=inputs_for_head(kind,value,int(record['valid_query_tokens']),record['valid_reference_tokens'])
            score=arrays[kind]['scores'][qi];direct=cpu_predict(head,values)
            maxscoreerror=max(maxscoreerror,float(np.max(np.abs(direct-score))));order=np.lexsort((np.arange(5405),-score));saved=arrays[kind]['stable_sorted_reference_indices'][qi]
            rankmismatch+=int(np.count_nonzero(order!=saved));seen={}
            for j in order:
                if hashes[j] not in seen:seen[hashes[j]]=len(seen)+1
            actual=np.array([seen[h] for h in hashes],dtype=np.int32);savedrank=arrays[kind]['sha_design_ranks'][qi]
            rankmismatch+=int(np.count_nonzero(actual!=savedrank))
            qr=queryrows[kind][qi];rank=min(seen[h] for h in ledger[qid]['positive_reference_sha256']);base=int(ledger[qid]['image_positive_rank'])
            checks[kind+'_'+qid+'_actual_query_ranking']=qr['photo_id']==qid and int(qr['positive_rank'])==rank and int(qr['baseline_image_rank'])==base and qr['top_reference_id']==d['rb']['ids'][order[0]]
            positives=set(ledger[qid]['positive_reference_sha256']);positive=next(int(i) for i in order if hashes[i] in positives);negative=next(int(i) for i in order if hashes[i] not in positives)
            component=((values[positive].astype(np.float64)-values[negative].astype(np.float64))/head['scale'])*head['coefficient'][:-1];saved_component=decompositions[kind]['contributions'][qi];gap=float(score[positive]-score[negative])
            checks[kind+'_'+qid+'_feature_contribution_gap_conserved']=np.max(np.abs(component-saved_component))<1e-10 and abs(float(component.sum())-gap)<1e-10 and abs(float(qr['logit_gap'])-gap)<1e-10 and qr['best_positive_reference_id']==d['rb']['ids'][positive] and qr['strongest_wrong_reference_id']==d['rb']['ids'][negative]
        if qi%100==0:log('Independent full-gallery replay',qi+1,'/1117')
    checks['all1117x5405x8_scores_direct_head_replay']=maxscoreerror<1e-10;checks['all_full_gallery_stable_SHA_ranks_replay']=rankmismatch==0;checks['all_complete_gallery_original_MaxSim_replay']=maxraw<3e-6
    checks['count_control_features_reconstructed_without_labels']=maximum_count_input_error==0.
    checks['32_response_independent_original_tensor_CPU_relation_replays']=len(feature_spot)==32 and feature_spot_error<3e-5
    csvout(output/'independent_relation_feature_spots.csv',feature_spot)
    gaprows=rows(output/'natural_hard_pair_gaps.csv');hardlookup={(r['head'],r['photo_id'],r['reference_id']):float(r['logit']) for r in hardpred};gaperror=0.;correct_mismatches=0
    for row in gaprows:
        positive=hardlookup[row['head'],row['photo_id'],row['positive_reference_id']];negative=hardlookup[row['head'],row['photo_id'],row['negative_reference_id']];gap=positive-negative
        gaperror=max(gaperror,abs(gap-float(row['gap'])),abs(positive-float(row['positive_logit'])),abs(negative-float(row['negative_logit'])))
        correct=1. if gap>0 else .5 if gap==0 else 0.;correct_mismatches+=correct!=float(row['correct'])
    checks['all1648x8_natural_hard_gap_and_correctness_replayed']=len(gaprows)==1648*8 and gaperror<1e-10 and correct_mismatches==0
    for kind,(head,threshold) in heads.items():
        values=families['count_only' if kind.startswith('count_only') else 'relation'];records=[r for r in allpred if r['head']==kind];error=float(np.max(np.abs(cpu_predict(head,values)-np.array([float(r['logit']) for r in records]))));errors[kind+'_restricted_logits']=error;checks[kind+'_restricted_prediction_replay']=error<1e-10 and len(records)==2234
        hard=[r for r in hardpred if r['head']==kind];maximum=0.;last=None;value=None
        for r in hard:
            if r['photo_id']!=last:
                record=np.load(output/'features'/(r['photo_id']+'.npz'));value=inputs_for_head(kind,record['features'],int(record['valid_query_tokens']),record['valid_reference_tokens']);last=r['photo_id']
            score=float(cpu_predict(head,value[ri[r['reference_id']]:ri[r['reference_id']]+1])[0]);maximum=max(maximum,abs(score-float(r['logit'])))
        errors[kind+'_natural_logits']=maximum;checks[kind+'_natural_hard_prediction_replay']=maximum<1e-10
        for metric in [r for r in rows(output/'restricted_metrics.csv') if r['head']==kind]:
            keep=s==metric['split'];scores=cpu_predict(head,values[keep]);cutoff=0. if metric['threshold_strategy']=='fixed_zero' else threshold
            actual={**independent_classification(y[keep],scores,p[keep],cutoff),**independent_pairs(y[keep],scores,q[keep],p[keep])}
            checks[kind+'_'+metric['split']+'_'+metric['threshold_strategy']+'_restricted_metrics_replayed']=all(abs(float(metric[k])-v)<1e-10 for k,v in actual.items())
        for metric in [r for r in read(output/'natural_hard_metrics.json') if r['head']==kind]:
            pairs=[r for r in gaprows if r['head']==kind and (metric['split']=='all' or r['split']==metric['split'])]
            correct=np.array([float(r['correct']) for r in pairs]);margin=np.array([float(r['gap']) for r in pairs]);qids=np.array([r['photo_id'] for r in pairs])
            pr=[r for r in hardpred if r['head']==kind and (metric['split']=='all' or r['split']==metric['split'])]
            actual=independent_classification(np.array([int(r['label']) for r in pr]),np.array([float(r['logit']) for r in pr]),np.array([r['package_id'] for r in pr]),threshold)
            checks[kind+'_'+metric['split']+'_natural_metrics_replayed']=metric['pairs']==len(pairs) and metric['queries']==len(set(qids)) and abs(metric['pair_accuracy']-float(correct.mean()))<1e-12 and abs(metric['query_macro_pair_accuracy']-float(np.mean([correct[qids==qid].mean() for qid in set(qids)])))<1e-12 and abs(metric['gap_mean']-float(margin.mean()))<1e-12 and abs(metric['gap_median']-float(np.median(margin)))<1e-12 and all(abs(metric['classification'][k]-v)<1e-12 for k,v in actual.items())
            actual=independent_bootstrap(correct,np.array([r['package_id'] for r in pairs]))
            checks[kind+'_'+metric['split']+'_natural_identity_bootstrap_replayed']=all(abs(metric['identity_bootstrap'][k]-v)<1e-12 for k,v in actual.items())
        qr=queryrows[kind];checks[kind+'_rank_coverage']=len(qr)==1117 and [r['photo_id'] for r in qr]==d['qb']['ids']
        for m in [r for r in rows(output/'full_gallery/metrics.csv') if r['head']==kind]:
            selected=[r for r in qr if m['split']=='all' or r['split']==m['split']];rank=np.array([int(r['positive_rank']) for r in selected]);base=np.array([int(r['baseline_image_rank']) for r in selected])
            repaired=int(((base>1)&(rank==1)).sum());harmed=int(((base==1)&(rank>1)).sum())
            expected={'recall_at_1':float((rank<=1).mean()),'recall_at_5':float((rank<=5).mean()),'recall_at_10':float((rank<=10).mean()),'mrr':float((1/rank).mean())}
            checks[kind+'_'+m['split']+'_repair_harm_metrics_replayed']=int(m['repaired'])==repaired and int(m['harmed'])==harmed and int(m['net_change'])==repaired-harmed and all(abs(float(m[k])-v)<1e-12 for k,v in expected.items())
            packages=np.array([r['package_id'] for r in selected]);delta=(rank==1).astype(float)-(base==1)
            actual=independent_bootstrap(delta,packages);reported=json.loads(m['identity_paired_bootstrap'])
            checks[kind+'_'+m['split']+'_identity_paired_bootstrap_replayed']=all(abs(reported[k]-v)<1e-12 for k,v in actual.items()) and abs(float(m['identity_macro_recall_at_1'])-np.mean([(rank[packages==pid]<=1).mean() for pid in np.unique(packages)]))<1e-12 and abs(float(m['identity_macro_mrr'])-np.mean([(1/rank[packages==pid]).mean() for pid in np.unique(packages)]))<1e-12
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,maximum_errors=errors,full_gallery_logit_error=maxscoreerror,original_MaxSim_error=maxraw,rank_axis_mismatches=rankmismatch,
                count_input_max_error=maximum_count_input_error,natural_gap_max_error=gaperror,natural_correctness_mismatches=correct_mismatches,independent_CPU_relation_feature_max_error=feature_spot_error,
                coverage=dict(queries=1117,references=5405,heads=len(heads),relation_heads=4,count_control_heads=4,shuffle_heads=6,feature_files=featurecommits,restricted_samples=2234,natural_contrasts=1648),
                scope='direct saved-feature scoring, independent objective refit, independent lexsort/SHA ranks, null labels, selections and conserved original mean MaxSim; finite relation statistics do not exhaust Token-set information')
    write(output/'independent_validation.json',result);log('Independent relation status',result['status'],'failed',[k for k,v in checks.items() if not v]);assert result['status']=='passed'
    return result


def gallery_diagnostics(output):
    """Post-score count-support diagnostics; never changes model selection."""
    axes=read(output/'full_gallery/score_axes.json');ri={rid:i for i,rid in enumerate(axes['reference_ids'])}
    reference_counts=np.load(output/'features'/(axes['query_ids'][0]+'.npz'))['valid_reference_tokens']
    index=rows(output/'restricted_sample_index.csv');allowed={r['reference_id'] for r in index if r['split']=='train'}
    training_counts=np.array([reference_counts[ri[r['reference_id']]] for r in index if r['split']=='train'])
    minimum,maximum=training_counts.min(),training_counts.max()
    def summary(a):
        a=np.asarray(a)
        return dict(N=len(a),minimum=float(a.min()),q05=float(np.quantile(a,.05)),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),q95=float(np.quantile(a,.95)),maximum=float(a.max()))
    result=dict(purpose='Descriptive candidate distribution diagnostics after frozen scoring; no model/threshold/feature choices change',
                relation_full_retrieval_limit='Restricted-pair readout accuracy does not imply complete gallery success',
                all_reference_token_counts=summary(reference_counts),train_restricted_reference_token_counts=summary(training_counts),
                training_distinct_reference_IDs=len(allowed),reference_rows=5405,gallery_heads=[])
    query_rows=[]
    for folder in sorted((output/'full_gallery').iterdir()):
        if not folder.is_dir():continue
        records=rows(folder/'query_rankings.csv');counts=np.array([reference_counts[ri[r['top_reference_id']]] for r in records]);frequencies=Counter(r['top_reference_id'] for r in records)
        result['gallery_heads'].append(dict(head=folder.name,winning_reference_token_counts=summary(counts),
                        winning_reference_token_counts_by_split={part:summary(counts[np.array([r['split']==part for r in records])]) for part in ['train','val','test']},
                        winning_reference_count_outside_train_minmax=int(((counts<minimum)|(counts>maximum)).sum()),
                        winners_in_actual_train_pair_referenceIDs=sum(r['top_reference_id'] in allowed for r in records),distinct_top_reference_IDs=len(frequencies),
                        most_frequent_top_referenceIDs=[dict(reference_id=rid,queries=n,valid_tokens=int(reference_counts[ri[rid]])) for rid,n in frequencies.most_common(10)]))
        query_rows.extend(dict(head=folder.name,photo_id=r['photo_id'],split=r['split'],top_reference_id=r['top_reference_id'],winning_reference_token_count=int(count),
                          winner_in_train_pair_references=r['top_reference_id'] in allowed,winner_count_outside_train_reference_range=bool(count<minimum or count>maximum),positive_rank=int(r['positive_rank'])) for r,count in zip(records,counts))
    write(output/'gallery_candidate_shift_audit.json',result);csvout(output/'gallery_candidate_shift_queries.csv',query_rows)
    return result


def report(output):
    validation=read(output/'independent_validation.json');selection=read(output/'selection.json');metrics=rows(output/'full_gallery/metrics.csv')
    count_selection=read(output/'count_only_selection.json');features=read(output/'features_complete.json')
    feature_source='results/colnomic_full_audit/20261013_resume/cross_token_relation/source_executed_'+features['partitions'][0]['source_sha256'][:12]+'.py'
    restricted=next(r for r in rows(output/'restricted_metrics.csv') if r['head']=='relation' and r['split']=='test' and r['threshold_strategy']=='fixed_zero')
    hard=next(r for r in read(output/'natural_hard_metrics.json') if r['head']=='relation' and r['split']=='test')
    diagnostic=read(output/'gallery_candidate_shift_audit.json');relation_shift=next(r for r in diagnostic['gallery_heads'] if r['head']=='relation');count_shift=next(r for r in diagnostic['gallery_heads'] if r['head']=='count_only')
    lines=['# 冻结 cross-Token 关系特征头','','本实验直接对完整 Query×Reference 原始 FP32 点积矩阵读取38维关系摘要，包含双向最佳/次优匹配分布与匹配质量聚集。与先分别池化或kernel mean后再配对不同，关系在汇总前被计算。它仍是有限关系摘要，不是完整 Token 集合信息测试。','',
           '输入没有坐标、原Token索引、图像ID或身份ID；ID只用于数据连接、划分和评价。全部非零Token（包括模板Token）参与，未重新归一化。精确并列的匹配质量均分，避免索引顺序成为特征。','',
           f'原2234受限配对用于训练/验证/测试；训练身份独立，标准化只用train，C={selection["C"]}与阈值只在验证身份选择。1648自然困难对照覆盖1117查询，另有3个训练标签打乱对照。历史测试已被观察，不能称新的盲测。','',
           f'全部1117×5405图库记录均保存38维float32关系特征。额外count-only基线输入4维显式有效Token数代理（log q数、log r数、绝对log差及q/r比），独立train标准化、val选C={count_selection["C"]}，并有3个null。两类共8个头均保存全图库标量、稳定排序和SHA设计排名。未插入真值、未限制top-K、未用bounds代替未算scalar。','',
           '|头|split|N|R@1|R@5|R@10|MRR|修复|损害|净变化|','|---|---|---|---|---|---|---|---|---|---|']
    for r in metrics:
        if r['split'] in ['all','test']:
            lines.append('|'+ '|'.join([r['head'],r['split'],r['queries']]+[f'{float(r[k]):.6f}' for k in ['recall_at_1','recall_at_5','recall_at_10','mrr']]+[r['repaired'],r['harmed'],r['net_change']])+'|')
    lines += ['',f'候选范围的差异：relation在原受限test配对的准确率为{float(restricted["pair_accuracy"]):.6f}，自然困难test的{hard["pairs"]}对为{hard["pair_accuracy"]:.6f}，但全图库出现上表中的大幅损害。完整图库结果不支持将这个通用配对头作为检索改进。','',
              f'训练配对只覆盖{diagnostic["training_distinct_reference_IDs"]}个Reference ID，有效Token数范围{diagnostic["train_restricted_reference_token_counts"]["minimum"]:g}–{diagnostic["train_restricted_reference_token_counts"]["maximum"]:g}，图库最小为{diagnostic["all_reference_token_counts"]["minimum"]:g}。relation的{relation_shift["winning_reference_count_outside_train_minmax"]}/1117赢家数量超出训练范围；数量控制头的{count_shift["distinct_top_reference_IDs"]}个最高Reference ID覆盖全部查询。gallery_candidate_shift_audit.json保留各头赢家分布。这里只报告训练支持范围与数量差异，不能断言下降全部由数量导致。该盘点发生在固定模型评分之后，不据此重选参数。','',
              f'独立数值验收：{validation["status"]}，{len(validation["checks"])}项。完整评分直接头重放误差{validation["full_gallery_logit_error"]}；原mean MaxSim最大误差{validation["original_MaxSim_error"]}；稳定排名与SHA轴不一致数{validation["rank_axis_mismatches"]}。另以固定种子在32对原始Token矩阵执行独立CPU关系摘要重算，最大误差{validation["independent_CPU_relation_feature_max_error"]}。','',
              '本头评估自然身份配对及完整图库，不把结果升级为自然同文字布局证据；文字内容、包装面及其它可见因素仍可能共同作用。Probe成功不证明原MaxSim使用信息，失败不证明特征没有信息。自然共同短语ROI Probe另行记录额外ROI监督与整图上下文。','',
              '## 复现与只验入口','',
              '两个小显存矩阵worker最终按query index奇偶分配不重叠查询，原缓存只读。前段硬件基准后因两卡持续速度相近调整分区，已有提交复用；rebalance_receipt.json保留该记录，未按模型效果分配。每worker allocator上限约675MiB，CPU1线程，FP32禁TF32。分区锁、逐Query提交与receipt支持断点续跑。重放特征需使用实际运行的冻结源码快照，使提交与receipt源码SHA一致；CPU训练和验证使用当前模块。','',
              '```bash',
              f"PYTHONPATH=/mnt/lsdf_zyin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /home/jwei/miniconda3/envs/rebut/bin/python -B {feature_source} --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 0 --cpu-threads 1",
              f"PYTHONPATH=/mnt/lsdf_zyin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 /home/jwei/miniconda3/envs/rebut/bin/python -B {feature_source} --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 1 --cpu-threads 1",
              "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage aggregate",
              "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage train",
              "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage verify",'```','',
              '`restricted_*`、`natural_hard_*`与`full_gallery/`保留不同候选范围。`feature_file_manifest.json`、`source_executed_*`、各worker runtime/receipt与`independent_validation.json`提供对应证据。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--stage',choices=['all','benchmark','features','aggregate','train','verify'],default='all');parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--partition-id',type=int,default=0);parser.add_argument('--partition-count',type=int,default=1);parser.add_argument('--partition-pattern',default='');parser.add_argument('--cpu-threads',type=int,default=2)
    args=parser.parse_args();output=args.output.resolve();assert output==OUT or OUT in output.parents;assert 0<=args.partition_id<args.partition_count
    assert args.stage!='all' or args.partition_count==1,'partitioned workers only generate features; aggregate/train separately'
    args.partition_pattern=[int(v) for v in args.partition_pattern.split(',')] if args.partition_pattern else []
    assert not args.partition_pattern or set(args.partition_pattern)==set(range(args.partition_count))
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.cpu_threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.cpu_threads)
    except ImportError:
        pass  # OMP_NUM_THREADS/OPENBLAS_NUM_THREADS are fixed by the run entry.
    d=data();source=Path(__file__);sourcehash=sha(source);args.executed_source_sha256=sourcehash;snapshot=output/('source_executed_'+sourcehash[:12]+'.py')
    if args.partition_id==0 and not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    protocol=dict(version=1,feature_names=NAMES,feature_dimensions=len(NAMES),encoder='frozen original FP16 banks; raw FP32 dot; no normalization; all valid template Tokens included',
                count_only_control_feature_names=COUNT_NAMES,count_only_control='separate structural count-only baseline; independent validation selection and3 null heads',
                feature_scope='complete cross-token matrix distribution/top-two/bidirectional matching graph; finite relation summary, NOT complete information test',
                graph_ties='uniform allocation across every exactly tied maximum; no index/ID-based tie winner',no_coordinate_inputs=True,no_token_index_inputs=True,no_image_ID_inputs=True,no_identity_ID_inputs=True,
                IDs='join and split/evaluation only',training='exact original2234 same-split samples; training negatives restricted to train positive designs',
                selection='validation identities only; C .01/.1/1/10; identity macro pair accuracy then AUC then smaller C',shuffle_seeds=SEEDS,
                full_gallery='all1117x5405 scalar predictions; no top-K truncation, no truth insertion; stable SHA dedup; null heads also fully scored',
                matrix_batching='batch complete reference records; EVERY candidate Token is present before top-two/matching reductions; candidate Tokens not split or truncated',
                observed_historical_test=True,executed_source_sha256=sourcehash,original_artifacts_read_only=True)
    if args.partition_id==0:write(output/'protocol.json',protocol)
    if args.stage=='aggregate':aggregate_features(d,output);return
    if args.stage in ['all','benchmark','features']:
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        device=torch.device(args.device);free,total=torch.cuda.mem_get_info(device);assert free>1024**3,('GPU0 requires >1GiB currently free',free)
        torch.cuda.set_per_process_memory_fraction(.06,device)
        worker=output/'workers'/f'partition_{args.partition_id}';worker.mkdir(parents=True,exist_ok=True)
        write(worker/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                  device=args.device,GPU_name=torch.cuda.get_device_name(device),GPU_initial_free_bytes=free,GPU_total_bytes=total,
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),TF32=False,matmul_precision=torch.get_float32_matmul_precision(),batch_size=args.batch_size,memory_fraction=.06,
                  source_sha256=sourcehash,partition_id=args.partition_id,partition_count=args.partition_count,CPU_threads=args.cpu_threads))
        references=[valid(z) for z in d['rb']['embeddings']]
        statistics_tests(worker,args.device)
        if args.stage in ['all','benchmark']:benchmark(d,references,output,args)
        if args.stage=='benchmark':return
        features_full(d,references,output,args);del references;torch.cuda.empty_cache()
        if args.stage=='features':return
    if args.stage in ['all','train']:
        assert read(output/'features_complete.json')['status']=='complete';heads=fit_heads(d,output);natural_hard(d,output,heads);gallery_ranks(d,output,heads)
    if args.stage in ['all','train','verify']:independent_verify(d,output)
    gallery_diagnostics(output)
    report(output)
    write(output/'completion.json',dict(status='complete',stage=args.stage,full_gallery=True,feature_dimensions=len(NAMES),source_code_sha256=sourcehash,
                limitation='finite cross-token relation statistics; no claim of natural same-text identity layout or encoder-layer causality'))


if __name__=='__main__':main()
