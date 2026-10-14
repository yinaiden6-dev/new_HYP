"""Independent frozen-input/standardization/selection/metric replay and block CI."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,data,read,write,rows,csvout,sha
from scripts.colnomic_mechanism.completion_audit import model_at,predict,fit,pairs,classification,threshold_from_validation


def relation(matrix):
    value=np.asarray(matrix,dtype=np.float64);nq,nr=value.shape
    topf=np.partition(value,-2,axis=1)[:,-2:];topr=np.partition(value,-2,axis=0)[-2:,:]
    f=topf.max(1);r=topr.max(0);parts=[]
    for v in [f,topf.max(1)-topf.min(1),r,topr.max(0)-topr.min(0)]:
        parts.extend([v.mean(),v.std(),*np.quantile(v,[.05,.25,.5,.75,.95])])
    forward=value==f[:,None];reverse=value==r[None,:]
    massq=forward/forward.sum(1)[:,None];massr=reverse/reverse.sum(0)[None]
    for load,k,mutual in [(massq.sum(0)/nq,nr,(massq*reverse).sum()/nq),(massr.sum(1)/nr,nq,(massr*forward).sum()/nr)]:
        nz=load[load>0];parts.extend([-(nz*np.log(nz)).sum()/np.log(k),load.max(),len(nz)/k,1/(np.square(load).sum()*k),mutual])
    return np.array(parts,dtype=np.float32)


def main():
    torch.set_num_threads(2);folder=ROOT/'results/colnomic_full_audit/20261013_resume/natural_shared_phrase_layout'
    destination=folder/'independent';destination.mkdir(exist_ok=True);d=data();inputs=np.load(folder/'inputs.npz')
    samples=rows(folder/'sample_index.csv');indices=rows(folder/'token_input_indices.csv')
    assert len(samples)==len(indices)==len(inputs['labels'])
    qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])};maximum=Counter();checks=[]
    for i,(sample,index) in enumerate(zip(samples,indices)):
        assert sample['case_id']==index['case_id'] and sample['reference_id']==index['reference_id']
        qix=json.loads(index['query_indices']);rix=json.loads(index['reference_indices'])
        assert len(qix)==len(set(qix)) and len(rix)==len(set(rix))
        q=d['qb']['embeddings'][qi[sample['photo_id']]][qix].float().numpy();r=d['rb']['embeddings'][ri[sample['reference_id']]][rix].float().numpy()
        qm=np.r_[q.mean(0),q.std(0)];rm=np.r_[r.mean(0),r.std(0)];wanted=np.r_[np.abs(qm-rm),qm*rm]
        dot=q@r.T;assert np.isfinite(dot).all()
        for name,actual,expected in [('pooled',inputs['pooled'][i],wanted),('maxsim',inputs['maxsim'][i],np.array([dot.max(1).mean()])),
            ('relations',inputs['relations'][i],relation(dot)),('token_counts',inputs['token_counts'][i],np.array([np.log1p(len(q)),np.log1p(len(r)),len(q)/len(r)]))]:
            error=float(np.max(np.abs(actual-expected)));maximum[name]=max(maximum[name],error);assert error<2e-6
        assert sample['split']==d['splits'][sample['photo_id']] and sample['package_id']==d['photos'][sample['photo_id']]['package_id']
    y=inputs['labels'];pkg=inputs['package_ids'];qids=inputs['query_ids'];splits=inputs['splits'];train=splits=='train';val=splits=='val'
    weights=np.array([1/Counter(pkg)[p] for p in pkg]);scopes={s:{d['groups'][r['package_id']] for r in samples if r['split']==s} for s in ['train','val','test']}
    assert all(not scopes[a]&scopes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    ledger=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')
    held={h for row in ledger if d['splits'][row['photo_id']]!='train' for h in row['positive_reference_sha256']}
    assert all(d['rb']['sha256'][ri[r['reference_id']]] not in held for r in samples if r['split']=='train' and r['label']=='0')
    feature_sets={'local_maxsim_only':inputs['maxsim'],'shared_phrase_pooled_only':inputs['pooled'],
        'shared_phrase_pooled_plus_maxsim':np.column_stack([inputs['pooled'],inputs['maxsim']]),
        'shared_phrase_relation_only':inputs['relations'],'ROI_token_count_control':inputs['token_counts']}
    stored_predictions=rows(folder/'predictions.csv');stored_nulls=rows(folder/'null_predictions.csv');metrics=rows(folder/'metrics.csv');cis=[]
    for kind,x in feature_sets.items():
        model,saved=model_at(folder/(kind+'_head.pt'));selection=read(folder/(kind+'_selection.json'))
        mean=np.average(x[train],axis=0,weights=weights[train]);scale=np.maximum(np.sqrt(np.average((x[train]-mean)**2,axis=0,weights=weights[train])),1e-5)
        assert np.array_equal(mean,model['mean']) and np.array_equal(scale,model['scale'])
        candidates=[]
        for c in [.01,.1,1.,10.]:
            candidate=fit(x[train],y[train],c,weights[train]);scores=predict(candidate,x[val]);p=pairs(y[val],scores,qids[val],pkg[val]);auc=classification(y[val],scores,pkg[val],0)['auc']
            candidates.append((p['package_macro_pair_accuracy'],auc,-c,c,candidate))
        best=max(candidates,key=lambda z:z[:3]);assert best[3]==float(saved['C'])==selection['C']
        assert np.max(np.abs(best[4]['coefficient']-model['coefficient']))<1e-8
        validation_scores=predict(model,x[val]);cutoff=threshold_from_validation(y[val],validation_scores,pkg[val])
        assert abs(cutoff-float(saved['threshold']))<1e-12
        assert np.array_equal(validation_scores>=cutoff,validation_scores>=float(saved['threshold']))
        maximum['threshold_roundoff']=max(maximum['threshold_roundoff'],abs(cutoff-float(saved['threshold'])))
        scores=predict(model,x);prior=np.array([float(r['score']) for r in stored_predictions if r['probe']==kind]);assert np.max(np.abs(scores-prior))<1e-10
        for seed in [17,31,53]:
            sy=y[train].copy();pp=pkg[train];rng=np.random.default_rng(seed)
            for package in sorted(set(pp)):
                mask=np.flatnonzero(pp==package);sy[mask]=sy[rng.permutation(mask)]
            null,record=model_at(folder/(kind+f'_shuffle_{seed}_head.pt'));refit=fit(x[train],sy,float(saved['C']),weights[train])
            assert np.max(np.abs(refit['coefficient']-null['coefficient']))<1e-8
            null_scores=predict(null,x);prior=np.array([float(r['score']) for r in stored_nulls if r['probe']==kind and int(r['seed'])==seed]);assert np.max(np.abs(null_scores-prior))<1e-10
            nt=threshold_from_validation(y[val],null_scores[val],pkg[val]);st=float(record['threshold'])
            assert abs(nt-st)<1e-12 and np.array_equal(null_scores>=nt,null_scores>=st)
            maximum['threshold_roundoff']=max(maximum['threshold_roundoff'],abs(nt-st))
        for split in ['train','val','test']:
            mask=splits==split;p=pairs(y[mask],scores[mask],qids[mask],pkg[mask])
            for strategy,t in [('fixed_zero',0),('validation_calibrated',cutoff)]:
                c=classification(y[mask],scores[mask],pkg[mask],t);actual=next(r for r in metrics if r['probe']==kind and r['split']==split and r['threshold_strategy']==strategy and not r['control'])
                for k,v in {**p,**c}.items():assert abs(float(actual[k])-v)<1e-10
        mask=splits=='test';gaps=[];packages=[]
        for qid in sorted(set(qids[mask])):
            positions=np.flatnonzero(mask&(qids==qid));positive=positions[y[positions]==1];negative=positions[y[positions]==0]
            assert len(positive)==len(negative)==1;gaps.append(scores[positive[0]]-scores[negative[0]]);packages.append(pkg[positive[0]])
        gaps=np.array(gaps);packages=np.array(packages);correct=(gaps>0).astype(float)+.5*(gaps==0)
        names=sorted(set(packages));pm=np.array([correct[packages==p].mean() for p in names]);rng=np.random.default_rng(20261014)
        sampled=pm[rng.integers(len(pm),size=(5000,len(pm)))].mean(1)
        cis.append(dict(probe=kind,test_identities=len(pm),test_pairs=len(gaps),identity_macro_pair_accuracy=pm.mean(),
            ci95_low=np.quantile(sampled,.025),ci95_high=np.quantile(sampled,.975),bootstrap_seed=20261014,
            bootstrap_rounds=5000,scope='package-block bootstrap conditional on fitted/selected head;does not include training-selection uncertainty'))
        checks.append(dict(probe=kind,training_standardization_exact=True,selected_C_refit=True,threshold_val_only_replayed=True,three_null_heads_refit=True,all_saved_metrics_replayed=True))
    csvout(destination/'package_bootstrap.csv',cis)
    write(destination/'validation.json',dict(status='passed',samples=len(samples),pairs=len(samples)//2,
        feature_replay_maximum_errors=dict(maximum),head_checks=checks,identities_by_split={s:len(set(pkg[splits==s])) for s in ['train','val','test']},
        source_groups_disjoint=scopes['train'].isdisjoint(scopes['val']) and scopes['val'].isdisjoint(scopes['test']) and scopes['train'].isdisjoint(scopes['test']),
        held_identity_training_negatives_absent=True,source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in [folder/'inputs.npz',folder/'sample_index.csv',folder/'token_input_indices.csv',folder/'manual_reviews.json']},
        limitation='four OCR phrase-selected patch sets;whole-image context/color/fonts/cardinality confounds;restricted pairs not complete-gallery or layout-only causality'))
    print('Independent shared phrase Probe verification passed',flush=True)

if __name__=='__main__':main()
