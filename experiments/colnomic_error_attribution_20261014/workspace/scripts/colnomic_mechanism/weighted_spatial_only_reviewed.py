"""Gamma=0 reviewed spatial matching × query-only aggregation, certified full gallery.

Reuse exact nonaligned candidates from two immutable certified weighted caches.
Recompute every accepted homography with FP32 dot/penalty maxima, then evaluate
any remaining candidate whose conservative bound can reach top10 or any GT.
Uncomputed scalar scores remain NaN. Parameters are selected on validation only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2,numpy as np,torch
from torch.nn.utils.rnn import pad_sequence
from scripts.colnomic_mechanism.common import ROOT,data,read,write,csvout,sha,rows
from scripts.colnomic_mechanism.weighted_maxsim import ETAS,weight_mask,score_bounds
from scripts.colnomic_mechanism.geometry import grid,metrics
from scripts.colnomic_mechanism.full_fusion import bootstrap
from evaluation.scoring import valid_tokens

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
OUT=BASE/'weighted_spatial_only_reviewed'
SOURCES=['weighted_original','weighted_spatial_reviewed']

def load_inputs(d):
    manifest_path=BASE/'reviewed_boxes/spatial_scoring_manifest.json';m=read(manifest_path)
    for name in ['scores','alignments','boxes','selection','score_axes']:assert sha(ROOT/m[name])==m[name+'_sha256'],name
    cache=np.load(ROOT/m['scores'],allow_pickle=False)
    assert cache['query_ids'].tolist()==d['qb']['ids'] and cache['reference_ids'].tolist()==d['rb']['ids']
    assert cache['reference_sha256'].tolist()==d['rb']['sha256']
    chosen=read(ROOT/m['selection'])['selections']['B03']['spatial_only'];assert chosen['gamma']==0 and chosen['lambda_value']==.1
    means=cache['geo_penalty'][chosen['lambda_index']].copy()
    qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])}
    fits={(qi[f['photo_id']],ri[f['reference_id']]):f for f in read(ROOT/m['alignments'])}
    assert len(fits)==20329
    boxes=read(ROOT/m['boxes'])['records'];d['boxes']=boxes
    return means,fits,chosen,manifest_path,m

def setup():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    d=data();means,fits,chosen,mp,m=load_inputs(d)
    qv=[valid_tokens(z) for z in d['qb']['embeddings']];rv=[valid_tokens(z) for z in d['rb']['embeddings']]
    rnorm=np.array([float(z.double().norm(dim=1).max()) for z in rv])
    grouped=defaultdict(list)
    for qi,ri in fits:grouped[qi].append(ri)
    ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    protocol=dict(schema='colnomic.weighted_spatial_only_reviewed.v1',etas=ETAS.tolist(),spatial_lambda=chosen['lambda_value'],alignment_gamma=0.,
        selection_key='B03.spatial_only',spatial_lambda_selected_on_validation_only=True,eta_selection='validation package_macro_r1, then package_macro_mrr, then smaller eta',
        query_weights='1+eta iff frozen query token distance to within-query feature mean is strictly greater than its 75th percentile; otherwise1',
        query_weight_inputs='query vectors only; no labels/OCR/coords/box/identity',
        matching='For accepted reviewed homography: subject maxima of FP32 dot-lambda*clipped normalized geometric distance, no availability reward; other tokens/candidates original MaxSim',
        uniform_eta_zero='Reuse exact reviewed_boxes gamma0 full-gallery mean, preserving stored FP32 baseline and pure-spatial ranks.',
        weighted_reduction='Float64 token/subset sums from FP32 adjusted maxima, divided by N+eta*K; eta0 overwritten by stored uniform means, matching historical weighted protocol.',
        bound='Cauchy token norm subset bounds intersected with N*(stored mean +/- tolerance) minus complement bounds',safety_tolerance=2e-6,
        all_gallery_rows_bounded=True,uncomputed_scalar_scores='NaN; every potentially top10/GT-outranking/tie candidate receives exact scores',
        reuse='Exact cached union restricted to no accepted H; every accepted H is recalculated, never subtract gamma from already-rounded maxima.',
        cache_sources=SOURCES,spatial_manifest_path=str(mp.relative_to(ROOT)),spatial_manifest_SHA256=sha(mp),
        reviewed_box_snapshot=m['boxes'],reviewed_box_snapshot_SHA256=m['boxes_sha256'],
        input_scores_SHA256=sha(ROOT/'cache/scores/colnomic7b/image.pt'),metadata_SHA256=sha(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json'),
        original_eta_source_SHA256=sha(BASE/'weighted_original/selection.json'),
        GPU='cuda:3, math only, memory fraction .1, TF32 disabled, deterministic algorithms',CPU_threads=2,
        historical_test_exposure=True,human_reviewed_boxes_include_observed_test=True,GT_free_automatic_boxes_claimed=False,encoder_rerun=False)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:write(OUT/'protocol.json',protocol)
    return d,means,fits,chosen,qv,rv,rnorm,grouped,ledger,protocol

def ranks_and_thresholds(exact,evaluated,qi,d,ledger):
    hashes=d['rb']['sha256'];qid=d['qb']['ids'][qi];pos=set(ledger[qid]['positive_reference_sha256'])
    pr={j for j,h in enumerate(hashes) if h in pos};rankings=[];thresholds=[]
    for ei,eta in enumerate(ETAS):
        order=sorted(evaluated,key=lambda j:(-float(exact[ei,j]),j));unique=[];seen=set()
        for j in order:
            if hashes[j] not in seen:unique.append(j);seen.add(hashes[j])
        rank=next((k+1 for k,j in enumerate(unique) if hashes[j] in pos),None)
        threshold=min(exact[ei,unique[9]],max(exact[ei,j] for j in pr)) if len(unique)>=10 and rank and pr<=set(evaluated) else -np.inf
        thresholds.append(threshold)
        rankings.append(dict(eta=float(eta),photo_id=qid,package_id=d['photos'][qid]['package_id'],split=d['splits'][qid],rank=rank,correct=rank==1,
            top1=d['rb']['ids'][unique[0]] if unique else None,top10=[d['rb']['ids'][j] for j in unique[:10]],scoring='reviewed_spatial_penalty_gamma0_query_only_weighted'))
    return rankings,np.asarray(thresholds)

def reuse_query(qi,d,q,aligned,means,ledger):
    qid=d['qb']['ids'][qi];mask=weight_mask(q);n=len(d['rb']['ids']);exact=np.full((len(ETAS),n),np.nan);sources=[];aligned_set=set(aligned)
    for source in SOURCES:
        folder=BASE/source/'queries'/qid;complete=read(folder/'complete.json');array_path=folder/'scores_and_bounds.npz'
        assert complete['query_tensor_SHA256']==d['metadata']['photos'][qi]['all_tokens_sha256']
        assert complete['scores_SHA256']==sha(array_path)
        with np.load(array_path,allow_pickle=False) as a:
            assert np.array_equal(mask,a['query_weight_mask'])
            keep=np.asarray([int(ri) for ri in a['evaluated'] if int(ri) not in aligned_set],np.int64)
            old=exact[:,keep];incoming=a['exact'][:,keep]
            overlap=np.isfinite(old);assert np.array_equal(old[overlap],incoming[overlap]),qid+' no-H cache union values disagree'
            assert np.array_equal(incoming[0],means[qi,keep]),qid+' no-H uniform mean differs'
            exact[:,keep]=incoming
        sources.append(dict(source=source,scores_sha256=complete['scores_SHA256'],complete_sha256=sha(folder/'complete.json'),kept_nonaligned_candidates=len(keep)))
    evaluated=set(np.flatnonzero(np.isfinite(exact[0])).tolist());assert np.isfinite(exact[:,list(evaluated)]).all()
    return exact,evaluated,mask,sources

def preflight():
    d,means,fits,chosen,qv,rv,rnorm,grouped,ledger,protocol=setup();records=[];total=0;operations=0
    for qi,qid in enumerate(d['qb']['ids']):
        exact,evaluated,mask,sources=reuse_query(qi,d,qv[qi],grouped[qi],means,ledger)
        flags=np.zeros(len(rv));flags[grouped[qi]]=1
        lower,upper=score_bounds(qv[qi],mask,rnorm,means[qi],spatial_lambda=chosen['lambda_value']*flags,alignment_gamma=0.)
        # Replacing not-yet-recomputed H scores by conservative lower bounds
        # gives a safe lower threshold, and hence an upper bound on new dot work.
        optimistic=exact.copy();optimistic[:,grouped[qi]]=lower[:,grouped[qi]]
        known=evaluated|set(grouped[qi]);_,threshold=ranks_and_thresholds(optimistic,known,qi,d,ledger)
        needed=[j for j in range(len(rv)) if j not in known and np.any(upper[:,j]>=threshold)]
        # If a GT is neither cached nor aligned, explicitly count it as required.
        assert np.isfinite(threshold).all(),qid+' expected all GT already in source-cache union or H'
        work=sum(len(qv[qi])*len(rv[j]) for j in grouped[qi]+needed);operations+=work
        total+=len(evaluated)
        records.append(dict(photo_id=qid,nonaligned_cache_union=len(evaluated),accepted_H_recompute=len(grouped[qi]),additional_noH_dot_upper_bound=len(needed),
            token_dot_products_upper_bound=work,query_valid_tokens=len(qv[qi]),query_weighted_tokens=int(mask.sum())))
        if qi%100==0:print('Pure spatial preflight',qi+1,'/',len(qv),'reuse',total,flush=True)
    csvout(OUT/'workload_by_query.csv',records)
    report=dict(status='prepared_and_sources_validated',queries=len(qv),gallery_rows=len(rv),nonaligned_cached_candidates_reusable=total,
        accepted_H_candidates_must_recompute=sum(r['accepted_H_recompute'] for r in records),
        additional_noH_dot_upper_bound=sum(r['additional_noH_dot_upper_bound'] for r in records),
        token_dot_products_upper_bound=operations,FP32_multiply_add_FLOPs_upper_bound=operations*128*2,
        original_full_gallery_pairs=len(qv)*len(rv),all_source_masks_and_noH_overlapping_scores_exact=True,
        remaining_work_note='Conservative bound before actual gamma0 H scoring; actual new no-H computations can only decrease.',
        spatial_lambda=chosen['lambda_value'],gamma=0.,GPU_memory_fraction=.1,encoder_rerun=False)
    write(OUT/'workload_preflight.json',report);print(report,flush=True)

def save_partial(target,exact,evaluated,mask,recomputed,added,source_records,qid,d,protocol):
    target.mkdir(parents=True,exist_ok=True)
    temporary=target/'partial.tmp.npz'
    np.savez_compressed(temporary,exact=exact,evaluated=np.asarray(sorted(evaluated),np.int64),query_weight_mask=mask,
        recomputed_H=np.asarray(sorted(recomputed),np.int64),new_noH=np.asarray(sorted(added),np.int64))
    temporary.replace(target/'partial.npz')
    write(target/'partial.json',dict(photo_id=qid,query_tensor_SHA256=d['metadata']['photos'][d['qb']['ids'].index(qid)]['all_tokens_sha256'],
        protocol_sha256=sha(OUT/'protocol.json'),partial_sha256=sha(target/'partial.npz'),cache_sources=source_records,
        accepted_H_recomputed=len(recomputed),new_noH_computed=len(added)))

@torch.inference_mode()
def calculate_batch(qi,indices,d,qv,rv,mask,fits,means,spatial_lambda,device):
    q=qv[qi];query=q.to(device);lengths=[len(rv[ri]) for ri in indices]
    references=pad_sequence([rv[ri] for ri in indices],batch_first=True).to(device)
    dots=torch.bmm(query[None].expand(len(indices),-1,-1),references.transpose(1,2))
    for bi,length in enumerate(lengths):dots[bi,:,length:]=-torch.inf
    maxima=dots.amax(2)
    valid_q=d['qb']['embeddings'][qi].float().norm(dim=1).numpy()>1e-8
    qxy=grid(d['metadata']['photos'][qi])[valid_q]
    record=d['boxes'][d['qb']['ids'][qi]]
    box=np.asarray(record['bbox'],np.float32)/np.tile([record['width'],record['height']],2)
    subject=np.isfinite(qxy).all(1)&((qxy>=box[:2])&(qxy<box[2:])).all(1)
    subject_gpu=torch.from_numpy(subject).to(device)
    for bi,ri in enumerate(indices):
        fit=fits.get((qi,ri))
        if fit is None:continue
        assert int(subject.sum())==fit['subject_tokens'],(qi,ri,int(subject.sum()),fit['subject_tokens'])
        valid_r=d['rb']['embeddings'][ri].float().norm(dim=1).numpy()>1e-8
        rxy=grid(d['metadata']['references'][ri])[valid_r]
        mapped=cv2.perspectiveTransform(qxy[subject][None].astype(np.float32),np.asarray(fit['homography'],np.float32))[0]
        distance=np.ones((int(subject.sum()),lengths[bi]),np.float32);rvalid=np.isfinite(rxy).all(1)
        distance[:,rvalid]=np.clip(np.linalg.norm(mapped[:,None]-rxy[rvalid][None],axis=2),0,1);distance[~np.isfinite(distance)]=1.
        maxima[bi,subject_gpu]=(dots[bi,subject_gpu,:lengths[bi]]-spatial_lambda*torch.from_numpy(distance).to(device)).amax(1)
    values=maxima.cpu().numpy();total=values.astype(np.float64).sum(1);subset=values[:,mask].astype(np.float64).sum(1)
    result=(total[:,None]+ETAS[None]*subset[:,None])/(len(q)+ETAS[None]*int(mask.sum()))
    mean_errors=[]
    for bi,ri in enumerate(indices):
        error=abs(float(values[bi].mean())-float(means[qi,ri]));assert error<3e-6,(qi,ri,error)
        mean_errors.append(error);result[bi,0]=means[qi,ri]
    return result.T,max(mean_errors,default=0.)

def run(batch_size=16):
    d,means,fits,chosen,qv,rv,rnorm,grouped,ledger,protocol=setup();device=torch.device('cuda:3')
    torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.1,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    started=time.monotonic();records=[];counts=[];maximum_mean_error=0.
    uniform_expected={r['photo_id']:r for r in read(BASE/'reviewed_boxes/predictions.json')['B03__spatial_only']}
    for qi,qid in enumerate(d['qb']['ids']):
        target=OUT/'queries'/qid;completed=target/'complete.json'
        if completed.exists():
            done=read(completed);assert done['query_tensor_SHA256']==d['metadata']['photos'][qi]['all_tokens_sha256']
            assert sha(target/'scores_and_bounds.npz')==done['scores_SHA256'] and done['protocol_sha256']==sha(OUT/'protocol.json')
            records.extend(done['rankings']);counts.append(done['workload']);continue
        exact,evaluated,mask,source_records=reuse_query(qi,d,qv[qi],grouped[qi],means,ledger)
        initial_reuse=len(evaluated);recomputed=set();added=set();local_mean_error=0.
        if (target/'partial.json').exists():
            p=read(target/'partial.json');assert p['protocol_sha256']==sha(OUT/'protocol.json') and p['cache_sources']==source_records
            assert p['query_tensor_SHA256']==d['metadata']['photos'][qi]['all_tokens_sha256'] and p['partial_sha256']==sha(target/'partial.npz')
            with np.load(target/'partial.npz',allow_pickle=False) as a:
                assert np.array_equal(mask,a['query_weight_mask']);exact=a['exact'].copy();evaluated=set(a['evaluated'].tolist())
                recomputed=set(a['recomputed_H'].tolist());added=set(a['new_noH'].tolist())
        flags=np.zeros(len(rv));flags[grouped[qi]]=1
        lower,upper=score_bounds(qv[qi],mask,rnorm,means[qi],spatial_lambda=chosen['lambda_value']*flags,alignment_gamma=0.)
        for offset in range(0,len(grouped[qi]),batch_size):
            batch=[ri for ri in grouped[qi][offset:offset+batch_size] if ri not in recomputed]
            if not batch:continue
            values,error=calculate_batch(qi,batch,d,qv,rv,mask,fits,means,chosen['lambda_value'],device)
            exact[:,batch]=values;evaluated.update(batch);recomputed.update(batch);local_mean_error=max(local_mean_error,error)
            assert np.all(values<=upper[:,batch]+1e-8) and np.all(values>=lower[:,batch]-1e-8)
            save_partial(target,exact,evaluated,mask,recomputed,added,source_records,qid,d,protocol)
        while True:
            rankings,thresholds=ranks_and_thresholds(exact,evaluated,qi,d,ledger)
            remaining=np.asarray([ri for ri in range(len(rv)) if ri not in evaluated],np.int64)
            possible=remaining[(upper[:,remaining]>=thresholds[:,None]).any(0)]
            if not len(possible):break
            batch=sorted(possible.tolist(),key=lambda ri:(-float(upper[:,ri].max()),ri))[:batch_size]
            assert not any((qi,ri) in fits for ri in batch),'Every H must have been recomputed before supplementation'
            values,error=calculate_batch(qi,batch,d,qv,rv,mask,fits,means,chosen['lambda_value'],device)
            exact[:,batch]=values;evaluated.update(batch);added.update(batch);local_mean_error=max(local_mean_error,error)
            assert np.all(values<=upper[:,batch]+1e-8) and np.all(values>=lower[:,batch]-1e-8)
            save_partial(target,exact,evaluated,mask,recomputed,added,source_records,qid,d,protocol)
        rankings,thresholds=ranks_and_thresholds(exact,evaluated,qi,d,ledger)
        slack=float(np.min(thresholds[:,None]-upper[:,remaining])) if len(remaining) else 1.
        assert slack>0 and all(r['rank'] is not None and len(r['top10'])==10 for r in rankings)
        assert rankings[0]['rank']==uniform_expected[qid]['rank'] and rankings[0]['top1']==uniform_expected[qid]['top1']
        assert set(grouped[qi])<=recomputed and len(recomputed)==len(grouped[qi])
        target.mkdir(parents=True,exist_ok=True);temporary=target/'scores_and_bounds.tmp.npz'
        np.savez_compressed(temporary,exact=exact,lower=lower,upper=upper,evaluated=np.asarray(sorted(evaluated),np.int64),query_weight_mask=mask,
            recomputed_H=np.asarray(sorted(recomputed),np.int64),new_noH=np.asarray(sorted(added),np.int64))
        temporary.replace(target/'scores_and_bounds.npz')
        work=dict(photo_id=qid,nonaligned_cached_reused=initial_reuse,accepted_H_recomputed=len(recomputed),new_noH_computed=len(added),
            token_dot_products=sum(len(qv[qi])*len(rv[ri]) for ri in recomputed|added),exact_candidates=len(evaluated),uncomputed_candidates=len(remaining),
            maximum_uniform_mean_replay_error=local_mean_error)
        done=dict(query_tensor_SHA256=d['metadata']['photos'][qi]['all_tokens_sha256'],gallery_rows=len(rv),evaluated_rows=len(evaluated),
            bounded_not_scored_rows=len(remaining),proof_min_slack=slack,scores_SHA256=sha(target/'scores_and_bounds.npz'),
            protocol_sha256=sha(OUT/'protocol.json'),rankings=rankings,workload=work,cache_sources=source_records)
        write(completed,done);records.extend(rankings);counts.append(work);maximum_mean_error=max(maximum_mean_error,local_mean_error)
        write(OUT/'progress.json',dict(completed_queries=len(counts),total_queries=len(qv),last=qid,elapsed_seconds=time.monotonic()-started,
            accepted_H_recomputed=sum(r['accepted_H_recomputed'] for r in counts),new_noH_computed=sum(r['new_noH_computed'] for r in counts)))
        print('Pure spatial',qi+1,'/',len(qv),'reuse',initial_reuse,'H',len(recomputed),'new',len(added),'slack',slack,flush=True)
    finish(records,counts,d,chosen,protocol)
    write(OUT/'runtime.json',dict(elapsed_seconds=time.monotonic()-started,device=str(device),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),GPU_memory_fraction=.1,TF32=False,CPU_threads=2,encoder_rerun=False,
        maximum_uniform_mean_replay_error=maximum_mean_error))

def cached_rankings(name,eta):
    result=[]
    for row in rows(BASE/name/'all_parameter_rankings.csv'):
        if float(row['eta'])!=eta:continue
        row['rank']=int(row['rank']);row['correct']=row['correct']=='True';row['eta']=eta
        result.append(row)
    assert len(result)==1117;return result

def finish(records,counts,d,chosen,protocol):
    assert len(records)==1117*len(ETAS)
    validation=[dict(eta=float(eta),**metrics([r for r in records if r['eta']==eta and r['split']=='val'])) for eta in ETAS]
    eta=max(validation,key=lambda r:(r['package_macro_r1'],r['package_macro_mrr'],-r['eta']))['eta']
    write(OUT/'selection.json',dict(eta=eta,criterion=protocol['eta_selection'],validation_only=True,validation_queries=173,validation_packages=21,
        spatial_lambda=chosen['lambda_value'],alignment_gamma=0.))
    csvout(OUT/'validation_grid.csv',validation);csvout(OUT/'all_parameter_rankings.csv',records);csvout(OUT/'actual_workload_by_query.csv',counts)
    before=[r for r in records if r['eta']==0];old={r['photo_id']:r for r in before};tables=[]
    for value in dict.fromkeys([0.,eta]):
        selected=[r for r in records if r['eta']==value]
        for split in ['train','val','test','all']:
            subset=[r for r in selected if split=='all' or r['split']==split]
            repaired=sum(r['correct'] and not old[r['photo_id']]['correct'] for r in subset);harmed=sum(not r['correct'] and old[r['photo_id']]['correct'] for r in subset)
            tables.append(dict(eta=value,split=split,**metrics(subset),rescued=repaired,harmed=harmed,net=repaired-harmed,**bootstrap(before,selected,d,split)))
    csvout(OUT/'metrics.csv',tables)
    original_eta=read(BASE/'weighted_original/selection.json')['eta'];combined_eta=read(BASE/'weighted_spatial_reviewed/selection.json')['eta']
    cells=dict(original_uniform=cached_rankings('weighted_original',0.),original_query_weighted=cached_rankings('weighted_original',original_eta),
        spatial_only_uniform=before,spatial_only_query_weighted=[r for r in records if r['eta']==eta],
        availability_spatial_uniform=cached_rankings('weighted_spatial_reviewed',0.),availability_spatial_query_weighted=cached_rankings('weighted_spatial_reviewed',combined_eta))
    factorial=[];transitions=[]
    for name,pred in cells.items():
        for split in ['train','val','test','all']:
            subset=[r for r in pred if split=='all' or r['split']==split];base={r['photo_id']:r for r in cells['original_uniform']}
            factorial.append(dict(cell=name,split=split,eta=pred[0]['eta'],gamma=.2 if name.startswith('availability_') else 0.,
                spatial_lambda=.1 if name.startswith(('spatial_only','availability_')) else 0.,**metrics(subset),
                repaired_vs_original_uniform=sum(r['correct'] and not base[r['photo_id']]['correct'] for r in subset),
                harmed_vs_original_uniform=sum(not r['correct'] and base[r['photo_id']]['correct'] for r in subset),
                **bootstrap(cells['original_uniform'],pred,d,split)))
    pairs=[('weighting_original','original_uniform','original_query_weighted'),('weighting_spatial_only','spatial_only_uniform','spatial_only_query_weighted'),
        ('spatial_matching_uniform','original_uniform','spatial_only_uniform'),('spatial_matching_weighted','original_query_weighted','spatial_only_query_weighted')]
    comparisons=[]
    for label,old_name,new_name in pairs:
        old={r['photo_id']:r for r in cells[old_name]}
        for row in cells[new_name]:
            prior=old[row['photo_id']];transitions.append(dict(comparison=label,photo_id=row['photo_id'],package_id=row['package_id'],split=row['split'],
                before_rank=prior['rank'],after_rank=row['rank'],before_top1=prior['top1'],after_top1=row['top1'],
                repaired=row['correct'] and not prior['correct'],harmed=not row['correct'] and prior['correct']))
        for split in ['train','val','test','all']:
            subset=[r for r in cells[new_name] if split=='all' or r['split']==split]
            comparisons.append(dict(comparison=label,split=split,photos=len(subset),repaired=sum(r['correct'] and not old[r['photo_id']]['correct'] for r in subset),
                harmed=sum(not r['correct'] and old[r['photo_id']]['correct'] for r in subset),**bootstrap(cells[old_name],cells[new_name],d,split)))
    csvout(OUT/'factorial_2x2_and_reward_controls.csv',factorial);csvout(OUT/'factorial_effects.csv',comparisons);csvout(OUT/'per_photo_factorial_transitions.csv',transitions)
    write(OUT/'workload_completed.json',dict(status='all1117_mathematical_scoring_complete',queries=1117,gallery_rows=5405,
        nonaligned_cached_candidates_reused=sum(r['nonaligned_cached_reused'] for r in counts),accepted_H_recomputed=sum(r['accepted_H_recomputed'] for r in counts),
        additional_noH_computed=sum(r['new_noH_computed'] for r in counts),exact_candidate_pairs=sum(r['exact_candidates'] for r in counts),
        uncomputed_candidate_pairs=sum(r['uncomputed_candidates'] for r in counts),all_candidate_pairs=1117*5405,
        actual_token_dot_products=sum(r['token_dot_products'] for r in counts),selected_eta=eta,selected_on_validation_only=True,encoder_rerun=False))

def independent_metrics(pred):
    ranks=np.asarray([r['rank'] for r in pred]);packages=np.asarray([r['package_id'] for r in pred]);out=dict(photos=len(pred),packages=len(set(packages)))
    for name,values in [('r1',(ranks==1).astype(float)),('r5',(ranks<=5).astype(float)),('r10',(ranks<=10).astype(float)),('mrr',1./ranks)]:
        out[name]=float(values.mean());out['package_macro_'+name]=float(np.mean([values[packages==p].mean() for p in sorted(set(packages))]))
    return out

def independent_bootstrap(before,after,d,split,seed=20261013):
    old={r['photo_id']:r for r in before};grouped=defaultdict(list)
    for r in after:
        if split!='all' and r['split']!=split:continue
        grouped[d['groups'][r['package_id']]].append(int(r['correct'])-int(old[r['photo_id']]['correct']))
    values=np.asarray([np.mean(grouped[g]) for g in sorted(grouped)]);rng=np.random.default_rng(seed)
    resampled=values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
    return dict(source_groups=len(values),source_group_macro_r1_difference=float(values.mean()),ci95=np.quantile(resampled,[.025,.975]).tolist(),seed=seed,draws=5000)

def CPU_score(qi,ri,d,qv,rv,mask,fits,means,lam):
    # Independent CPU implementation: no padding/bmm and no GPU maxima reused.
    query=qv[qi];reference=rv[ri];matrix=query@reference.T;maxima=torch.amax(matrix,dim=1).numpy().copy()
    fit=fits.get((qi,ri))
    if fit:
        qm=d['qb']['embeddings'][qi].float().norm(dim=1).numpy()>1e-8;rm=d['rb']['embeddings'][ri].float().norm(dim=1).numpy()>1e-8
        qp=grid(d['metadata']['photos'][qi])[qm];rp=grid(d['metadata']['references'][ri])[rm]
        b=d['boxes'][d['qb']['ids'][qi]];box=np.asarray(b['bbox'],np.float32)/np.tile([b['width'],b['height']],2)
        take=np.isfinite(qp).all(1)&(qp[:,0]>=box[0])&(qp[:,0]<box[2])&(qp[:,1]>=box[1])&(qp[:,1]<box[3])
        projected=cv2.perspectiveTransform(qp[take][None].astype(np.float32),np.asarray(fit['homography'],np.float32))[0]
        dist=np.ones((int(take.sum()),len(reference)),np.float32);valid=np.isfinite(rp).all(1)
        dist[:,valid]=np.minimum(np.sqrt(((projected[:,None]-rp[valid][None])**2).sum(axis=2)),1)
        dist[~np.isfinite(dist)]=1.
        adjusted=matrix[torch.from_numpy(take)]-lam*torch.from_numpy(dist);maxima[take]=torch.amax(adjusted,dim=1).numpy()
    result=np.asarray([(maxima.astype(np.float64).sum()+eta*maxima[mask].astype(np.float64).sum())/(len(query)+eta*int(mask.sum())) for eta in ETAS])
    result[0]=means[qi,ri];return result

def verify():
    d,means,fits,chosen,qv,rv,norms,grouped,ledger,protocol=setup();all_records=[];proofs=[];CPU_checks=[];max_bound_error=0.;sample_max=0.
    qids=d['qb']['ids'];hashes=np.asarray(d['rb']['sha256']);source_hashes={};actual_work=[]
    uniform_expected={r['photo_id']:r for r in read(BASE/'reviewed_boxes/predictions.json')['B03__spatial_only']}
    sample_queries=set(range(0,len(qids),50))|{qids.index(q) for q in ['photo_88e7f77db7da30c3','photo_41de3eb383934ec3','photo_67f24002bcb357c9','photo_c81d2a7abb5af160','photo_d33ae6dfc0cb239b']}
    for qi,qid in enumerate(qids):
        folder=OUT/'queries'/qid;c=read(folder/'complete.json');assert sha(folder/'scores_and_bounds.npz')==c['scores_SHA256']
        assert c['protocol_sha256']==sha(OUT/'protocol.json') and c['query_tensor_SHA256']==d['metadata']['photos'][qi]['all_tokens_sha256']
        with np.load(folder/'scores_and_bounds.npz',allow_pickle=False) as a:
            exact=a['exact'];lo=a['lower'];hi=a['upper'];evaluated=a['evaluated'];mask=a['query_weight_mask'];hr=a['recomputed_H'];new=a['new_noH']
            assert exact.shape==lo.shape==hi.shape==(5,5405) and np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo<=hi)
            assert np.array_equal(mask,weight_mask(qv[qi]));assert set(hr.tolist())==set(grouped[qi])
            assert not (set(hr.tolist())&set(new.tolist()))
            unknown=np.flatnonzero(~np.isfinite(exact[0]));assert set(evaluated.tolist())==set(np.flatnonzero(np.isfinite(exact[0])).tolist())
            assert np.isnan(exact[:,unknown]).all() and np.isfinite(exact[:,evaluated]).all()
            assert np.array_equal(exact[0,evaluated],means[qi,evaluated])
            # Independent algebra from token Cauchy bounds and the complement.
            n=len(qv[qi]);k=int(mask.sum());qn=qv[qi].double().norm(dim=1).numpy();penalty=np.zeros(5405);penalty[grouped[qi]]=chosen['lambda_value']
            sum_hi=n*(means[qi].astype(np.float64)+2e-6);sum_lo=n*(means[qi].astype(np.float64)-2e-6)
            sub_hi=np.minimum(qn[mask].sum()*norms,sum_hi+qn[~mask].sum()*norms+penalty*(n-k))
            sub_lo=np.maximum(-qn[mask].sum()*norms-penalty*k,sum_lo-qn[~mask].sum()*norms)
            expected_hi=(sum_hi[None]+ETAS[:,None]*sub_hi[None])/(n+ETAS[:,None]*k)
            expected_lo=(sum_lo[None]+ETAS[:,None]*sub_lo[None])/(n+ETAS[:,None]*k)
            error=max(float(abs(expected_hi-hi).max()),float(abs(expected_lo-lo).max()));assert error<1e-12;max_bound_error=max(max_bound_error,error)
            assert (exact[:,evaluated]<=hi[:,evaluated]+1e-8).all() and (exact[:,evaluated]>=lo[:,evaluated]-1e-8).all()
            positives=np.flatnonzero(np.isin(hashes,ledger[qid]['positive_reference_sha256']));assert set(positives)<=set(evaluated)
            minslack=np.inf
            for ei,eta in enumerate(ETAS):
                order=np.argsort(-exact[ei],kind='stable');order=order[np.isfinite(exact[ei,order])];unique=[];seen=set()
                for ri in order:
                    if hashes[ri] not in seen:unique.append(int(ri));seen.add(hashes[ri])
                rank=next(i+1 for i,ri in enumerate(unique) if hashes[ri] in set(ledger[qid]['positive_reference_sha256']))
                stored=c['rankings'][ei];assert stored['eta']==float(eta) and stored['rank']==rank and stored['correct']==(rank==1)
                assert stored['top10']==[d['rb']['ids'][ri] for ri in unique[:10]] and stored['top1']==d['rb']['ids'][unique[0]]
                threshold=min(exact[ei,unique[9]],float(exact[ei,positives].max()));slack=float((threshold-hi[ei,unknown]).min()) if len(unknown) else 1.
                assert slack>0;minslack=min(minslack,slack);all_records.append(stored)
                proofs.append(dict(photo_id=qid,eta=float(eta),GT_rank=rank,top10_certified=True,all_GT_rows_scored=True,uncomputed_rows=len(unknown),strict_positive_slack=slack))
            assert abs(minslack-c['proof_min_slack'])<1e-12
            assert c['rankings'][0]['rank']==uniform_expected[qid]['rank'] and c['rankings'][0]['top1']==uniform_expected[qid]['top1']
            for s in c['cache_sources']:
                source=BASE/s['source']/'queries'/qid;assert sha(source/'scores_and_bounds.npz')==s['scores_sha256'] and sha(source/'complete.json')==s['complete_sha256']
                source_hashes[str(source/'scores_and_bounds.npz')]=s['scores_sha256']
            actual_work.append(c['workload'])
            if qi in sample_queries:
                candidates=set(positives.tolist())|{evaluated[0]}|set(grouped[qi][:1])|{d['rb']['ids'].index(c['rankings'][-1]['top1'])}
                for ri in sorted(candidates):
                    cpu=CPU_score(qi,int(ri),d,qv,rv,mask,fits,means,chosen['lambda_value']);err=float(abs(cpu-exact[:,ri]).max())
                    assert err<1e-6,(qid,ri,err);sample_max=max(sample_max,err)
                    CPU_checks.append(dict(photo_id=qid,reference_id=d['rb']['ids'][ri],accepted_H=(qi,int(ri)) in fits,maximum_all_eta_absolute_score_error=err))
        if qi%100==0:print('Independent pure spatial cert',qi+1,'/',len(qids),flush=True)
    saved=rows(OUT/'all_parameter_rankings.csv');assert len(saved)==len(all_records)==5585
    for a,b in zip(saved,all_records):assert a['photo_id']==b['photo_id'] and float(a['eta'])==b['eta'] and int(a['rank'])==b['rank'] and json.loads(a['top10'])==b['top10']
    expected_grid=[dict(eta=float(eta),**independent_metrics([r for r in all_records if r['eta']==eta and r['split']=='val'])) for eta in ETAS]
    chosen_eta=max(expected_grid,key=lambda r:(r['package_macro_r1'],r['package_macro_mrr'],-r['eta']))['eta'];assert chosen_eta==read(OUT/'selection.json')['eta']
    for actual,expected in zip(rows(OUT/'validation_grid.csv'),expected_grid):
        for key,value in expected.items():assert abs(float(actual[key])-value)<1e-12
    base=[r for r in all_records if r['eta']==0];byid={r['photo_id']:r for r in base}
    for m in rows(OUT/'metrics.csv'):
        pred=[r for r in all_records if r['eta']==float(m['eta'])];subset=[r for r in pred if m['split']=='all' or r['split']==m['split']]
        for key,value in independent_metrics(subset).items():assert abs(float(m[key])-value)<1e-12
        rescued=sum(r['correct'] and not byid[r['photo_id']]['correct'] for r in subset);harmed=sum(not r['correct'] and byid[r['photo_id']]['correct'] for r in subset)
        assert int(m['rescued'])==rescued and int(m['harmed'])==harmed and int(m['net'])==rescued-harmed
        bs=independent_bootstrap(base,pred,d,m['split']);assert abs(float(m['source_group_macro_r1_difference'])-bs['source_group_macro_r1_difference'])<1e-12
        assert np.max(np.abs(np.asarray(json.loads(m['ci95']))-bs['ci95']))<1e-12
    original_eta=read(BASE/'weighted_original/selection.json')['eta'];original_uniform=cached_rankings('weighted_original',0.);original_weighted=cached_rankings('weighted_original',original_eta)
    new_weighted=[r for r in all_records if r['eta']==chosen_eta];cells=dict(original_uniform=original_uniform,original_query_weighted=original_weighted,
        spatial_only_uniform=base,spatial_only_query_weighted=new_weighted,
        availability_spatial_uniform=cached_rankings('weighted_spatial_reviewed',0.),
        availability_spatial_query_weighted=cached_rankings('weighted_spatial_reviewed',read(BASE/'weighted_spatial_reviewed/selection.json')['eta']))
    for m in rows(OUT/'factorial_2x2_and_reward_controls.csv'):
        pred=cells[m['cell']];subset=[r for r in pred if m['split']=='all' or r['split']==m['split']]
        for key,value in independent_metrics(subset).items():assert abs(float(m[key])-value)<1e-12
        bs=independent_bootstrap(original_uniform,pred,d,m['split']);assert abs(float(m['source_group_macro_r1_difference'])-bs['source_group_macro_r1_difference'])<1e-12
        assert np.max(abs(np.asarray(json.loads(m['ci95']))-bs['ci95']))<1e-12
    maps={name:{r['photo_id']:r for r in pred} for name,pred in cells.items()};interaction=[]
    for split in ['train','val','test','all']:
        grouped_values=defaultdict(list);vals=[]
        for qid in qids:
            if split!='all' and d['splits'][qid]!=split:continue
            value=(int(maps['spatial_only_query_weighted'][qid]['correct'])-int(maps['spatial_only_uniform'][qid]['correct']))-(int(maps['original_query_weighted'][qid]['correct'])-int(maps['original_uniform'][qid]['correct']))
            vals.append(value);grouped_values[d['groups'][d['photos'][qid]['package_id']]].append(value)
        values=np.asarray([np.mean(grouped_values[g]) for g in sorted(grouped_values)]);rng=np.random.default_rng(20261013)
        draws=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
        interaction.append(dict(split=split,photos=len(vals),source_groups=len(values),micro_r1_interaction=float(np.mean(vals)),
            source_group_macro_r1_interaction=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),seed=20261013,draws=5000,
            definition='(spatial_only_weighted-spatial_only_uniform)-(original_weighted-original_uniform)',original_eta=original_eta,spatial_only_eta=chosen_eta))
    csvout(OUT/'independent_rank_certificates.csv',proofs);csvout(OUT/'independent_CPU_math_checks.csv',CPU_checks);csvout(OUT/'factorial_interaction_bootstrap.csv',interaction)
    write(OUT/'cell_predictions.json',cells)
    rawbanks=[ROOT/'cache/image_embeddings/colnomic7b/photos.pt',ROOT/'cache/image_embeddings/colnomic7b/references.pt'];expected=['e3505044319ab374c20bd7ecb42f5342d818d0b764899329b2f4a5e3c47fe8ca','500d4e15d167f7f4749348816a489e6d24c647cb20e89bef216d18d265f6b17d']
    bank_hashes={str(p.relative_to(ROOT)):sha(p) for p in rawbanks};assert list(bank_hashes.values())==expected
    report=dict(status='passed',independently_executed=True,queries=1117,reference_rows=5405,parameter_rankings_certified=5585,
        every_eta_exact_top10_and_GT_rank=True,every_GT_SHA_duplicate_row_explicitly_scored=True,stable_score_ties_keep_gallery_row_order=True,
        every_gallery_pair_has_bounds=1117*5405,uncomputed_scalar_values_remain_NaN=True,selected_eta=chosen_eta,selection_replayed_validation_only=True,
        uniform_spatial_all1117_GT_rank_top1_equal_to_reviewed_geometry=True,accepted_H_fresh_score_pairs=sum(r['accepted_H_recomputed'] for r in actual_work),
        nonaligned_cached_reused=sum(r['nonaligned_cached_reused'] for r in actual_work),additional_noH_computed=sum(r['new_noH_computed'] for r in actual_work),
        source_cache_NPZ_SHA_rechecked=len(source_hashes),raw_bank_SHA256=bank_hashes,
        independently_reconstructed_bounds_max_error=max_bound_error,minimum_uncomputed_certificate_slack=min(r['strict_positive_slack'] for r in proofs),
        independent_CPU_dot_penalty_pairs=len(CPU_checks),independent_CPU_dot_penalty_max_score_error=sample_max,
        macro_metrics_replayed=True,repairs_harms_replayed=True,source_group_bootstrap_5000_replayed=True,
        pure_matching_factorial_cells=4,availability_reward_control_cells_retained=2,
        partial_checkpoints_supported=True,TF32_disabled=True,GPU_math_only=True,encoder_rerun=False,full_scalar_scores_for_all_weighted_candidates=False,
        executed_scoring_source_SHA256=sha(OUT/'source_executed.py'),verification_source_SHA256=sha(Path(__file__)),
        limitations=['Human-reviewed boxes include observed test annotations; auxiliary assistance, not a GT-free automatic method.',
            'GPU FP32 bmm and cached CPU FP32 scalar reductions can differ slightly; independent CPU direct-dot samples and all saved bounds are checked.',
            'Frozen banks/scoring are replayable; historical gallery neural encoding upstream remains external.'])
    write(OUT/'independent_verification.json',report);print(report,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:verify()
    elif args.run:run(args.batch_size)
    else:preflight()

if __name__=='__main__':main()
