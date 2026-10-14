"""CPU-only frozen audit verification and replay into a fresh directory.

This entry validates numerical artifacts and source links. It does not encode
images, reopen an unavailable historical encoder, or certify causal mechanisms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

RELATIVE_BASE=Path('results/colnomic_full_audit/20261013_resume')
HEADS=['relation','shuffle_17','shuffle_31','shuffle_53','count_only','count_only_shuffle_17','count_only_shuffle_31','count_only_shuffle_53']
MODULES=[
    ('tokens','verify_full_tokens','all exported Token/region accounting; all2765 raw matrices; semantic proxies are not causal labels'),
    ('set_probes','verify_set_probes','natural finite pooled/kernel readouts and full-gallery/null replay'),
    ('signed_layout','completion_audit','41-identity synthetic signed left/right edits, pixel conservation and five readout heads'),
    ('positions','verify_reviewed_positions','reviewed-box position targets, selections/main/nulls; OCR vertical order is a proxy'),
    ('position_content','position_content_control','OCR train-vocabulary content/vertical-order correlation control; association uses extra OCR/box supervision; no layout-only causality'),
    ('shared_phrase','verify_shared_phrase_probe','local common-phrase ROI identity Probe with extra ROI supervision; not layout-only/full-text sameness'),
    ('crops','verify_reviewed_crops','query-only reviewed crop scores/ranks and selected raw dot matrices; fixed historical gallery'),
    ('weighted','verify_weighted','four weighted methods including reviewed pure spatial lambda; exact rank certificates; uncomputed scalar values remain NaN'),
    ('crossview','verify_crossview_layouts','controlled cross-view pilot cached feature matrices/traces; no held-test causal generalization'),
    ('auxiliary','verify_auxiliary_reports','exported correspondence matrices and visual-label/35-config derived statistics'),
]


def log(*items):print(*items,flush=True)


def initialize(workspace):
    global common
    common=importlib.import_module('scripts.colnomic_mechanism.common')
    common.ROOT=workspace
    common.OUT=workspace/'results/token_mechanism_experiments/20261013_v1'
    os.environ['COLNOMIC_WORKSPACE']=str(workspace)
    return common


def inside(path,root):return path==root or root in path.parents


def fresh_output(path,workspace):
    path=path.expanduser().resolve()
    if not (inside(path,workspace/'results') or inside(path,Path('/tmp'))):
        raise ValueError('Output must be a fresh directory under workspace/results or /tmp')
    base=workspace/RELATIVE_BASE
    if inside(path,base) and not inside(path,base/'reproduction_support'):
        raise ValueError('Output cannot overwrite historical audit branches')
    if path.exists() and any(path.iterdir()):raise FileExistsError('Choose a new empty --output directory')
    path.mkdir(parents=True,exist_ok=True)
    return path


def install_guard(workspace,output):
    """Reject old-workspace fallback reads and any writes to historical inputs."""
    historical=Path('/mnt/lsdf_zyin')
    def resolve(value,dirfd=None):
        path=Path(os.fsdecode(value))
        if not path.is_absolute() and isinstance(dirfd,int) and dirfd>=0:
            path=Path(os.readlink('/proc/self/fd/'+str(dirfd)))/path
        return path.absolute().resolve()
    def guard(event,args):
        if event=='open':
            value=args[0]
            if not isinstance(value,(str,bytes,os.PathLike)):return
            path=resolve(value)
            if workspace!=historical and inside(path,historical) and not inside(path,workspace):
                raise PermissionError('Forbidden fallback read of original workspace: '+str(path))
            flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
            mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
            writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or any(x in mode for x in 'wax+')
            if writing and not inside(path,output):
                raise PermissionError('Frozen verification cannot write outside fresh output: '+str(path))
        elif event in ['os.mkdir','os.remove','os.rename','os.rmdir']:
            value=args[0]
            if not isinstance(value,(str,bytes,os.PathLike)):return
            fd=(args[2] if len(args)>2 else None) if event=='os.mkdir' else (args[2] if len(args)>2 else None) if event=='os.rename' else (args[1] if len(args)>1 else None)
            path=resolve(value,fd)
            if event=='os.mkdir' and path.is_dir():return
            if not inside(path,output):raise PermissionError('Frozen history mutation rejected: '+str(path))
            if event=='os.rename':
                target=resolve(args[1],args[3] if len(args)>3 else None)
                if not inside(target,output):raise PermissionError('Frozen history rename rejected')
    sys.addaudithook(guard)


def payload_paths(base,output):
    keys={'path','original_path','spatial_manifest_path','evidence_image','image_path','source_path','snapshot_path'}
    names=['full_tokens/dossiers_all.json','symmetric_layouts/sources.json','detail_interventions/sources.json',
           'reviewed_crops/encoding_manifest.json','controlled_crossview_layouts/encoding_manifest.json',
           'controlled_crossview_layouts/sources_reviewed.json','weighted_original/protocol.json',
           'weighted_spatial/protocol.json','weighted_spatial_reviewed/protocol.json']
    records=[];hashes={}
    def visit(value,origin):
        if isinstance(value,list):
            for row in value:visit(row,origin)
        elif isinstance(value,dict):
            for key,item in value.items():
                if key in keys and isinstance(item,str) and ('/' in item or Path(item).suffix):
                    # source_path may be descriptive provenance without a file suffix.
                    if key=='source_path' and not Path(item).suffix:continue
                    target=common.workspace_path(item,must_exist=True)
                    if not target.is_file():raise ValueError('Payload input is not a file: '+item)
                    if target not in hashes:hashes[target]=common.sha(target)
                    expected=value.get('image_sha256',value.get('image_SHA256'))
                    if key=='original_path':expected=value.get('image_sha256',value.get('image_SHA256',value.get('sha256',expected)))
                    if expected is not None and hashes[target]!=expected:raise ValueError('Payload image SHA mismatch: '+item)
                    records.append(dict(payload=origin,field=key,archived_path=item,current_relative_path=str(target.relative_to(common.ROOT)),sha256=hashes[target],image_SHA_checked=expected is not None))
                elif isinstance(item,(list,dict)):visit(item,origin)
    for name in names:
        path=base/name
        if path.is_file():visit(common.read(path),name)
    result=dict(status='passed',policy='Only current ROOT paths or explicit /mnt/lsdf_zyin rebasing; all resolved paths stay within ROOT; no external fallback',
                checked_payload_paths=len(records),distinct_files=len(hashes),records=records)
    common.write(output/'input_path_validation.json',result)
    return result


def ranks(score,hashes,positives):
    order=np.lexsort((np.arange(len(score)),-score));seen={}
    for i in order:
        h=hashes[int(i)]
        if h not in seen:seen[h]=len(seen)+1
    full=np.array([seen[h] for h in hashes],dtype=np.int32)
    return order.astype(np.int32),full,min(seen[h] for h in positives)


def choose_queries(d,quick):
    if not quick:return list(range(1117))
    rng=np.random.default_rng(2026101402);selected=[]
    for part,n in [('train',3),('val',3),('test',2)]:
        eligible=[i for i,q in enumerate(d['qb']['ids']) if d['splits'][q]==part]
        selected.extend(map(int,rng.choice(eligible,n,replace=False)))
    return sorted(selected)


def output_arrays(folder,shape,reproduce):
    if not reproduce:return None
    folder.mkdir(parents=True,exist_ok=True)
    return {name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',shape=shape,dtype=dtype)
            for name,dtype in [('scores',np.float64),('stable_sorted_reference_indices',np.int32),('sha_design_ranks',np.int32)]}


def baseline_replay(d,base,output,selected,reproduce):
    from evaluation.scoring import bm25_scores,fuse
    ids=d['qb']['ids'];rids=d['rb']['ids'];hashes=d['rb']['sha256'];ledger={r['photo_id']:r for r in common.read(common.ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    recipe=d['bm25']['recipe'];text=bm25_scores([d['qo'][ids[i]]['text'] for i in selected],[d['ro'][r]['text'] for r in rids],k1=recipe['k1'],b=recipe['b'])
    image=d['image']['scores'][selected].clone();fusion=fuse(image,text,[bool(d['qo'][ids[i]]['text'].strip()) for i in selected],.5)
    errors=dict(BM25=float((text-d['bm25']['scores'][selected]).abs().max()),fusion=float((fusion-d['fusion']['scores'][selected]).abs().max()))
    assert max(errors.values())<1e-5,errors
    variants={'image':image.numpy(),'BM25':text.numpy(),'fusion':fusion.numpy()};report=[]
    for name,matrix in variants.items():
        exported=output_arrays(output/'baseline'/name,matrix.shape,reproduce)
        for local,qi in enumerate(selected):
            qid=ids[qi];order,full,rank=ranks(matrix[local],hashes,set(ledger[qid]['positive_reference_sha256']))
            if name in ['image','fusion']:assert rank==int(ledger[qid][name+'_positive_rank'])
            report.append(dict(method=name,photo_id=qid,query_index=qi,split=d['splits'][qid],positive_rank=rank,top_reference_id=rids[int(order[0])],correct=rank==1))
            if exported:
                exported['scores'][local]=matrix[local];exported['stable_sorted_reference_indices'][local]=order;exported['sha_design_ranks'][local]=full
        if exported:
            for array in exported.values():array.flush()
    axes=dict(query_ids=[ids[i] for i in selected],original_query_indices=selected,reference_ids=rids,reference_sha256=hashes,reference_rows=5405,unique_SHA_designs=len(set(hashes)))
    common.write(output/'baseline/score_axes.json',axes);common.csvout(output/'baseline/query_rankings.csv',report)
    result=dict(status='passed',queries=len(selected),reference_rows=5405,scalar_values_replayed=3*len(selected)*5405,maximum_errors=errors,
                image_scope='Frozen original image scalars; all selected full-gallery raw means independently checked from saved relation features below; no full encoder rerun',
                BM25_scope='Full document-corpus analytic scoring from OCR text and archived k1/b',fusion_scope='FP32 full-row population z-score and original empty-OCR fallback',
                ranking='numpy.lexsort with original-row ties then complete SHA scan',exported_arrays=reproduce)
    common.write(output/'baseline/validation.json',result);log('Frozen baseline replay passed',len(selected),'x5405x3')
    return result


def relation_replay(d,base,output,selected,reproduce,quick):
    from scripts.colnomic_mechanism.completion_audit import fit,pairs,classification,threshold_from_validation
    from scripts.colnomic_mechanism.cross_token_probe import cpu_relation,NAMES,COUNT_NAMES
    source=base/'cross_token_relation';a=dict(np.load(source/'restricted_inputs.npz'));ca=dict(np.load(source/'restricted_count_inputs.npz'))
    y,p,s,q=[a[k] for k in ['labels','package_ids','splits','query_ids']];train=s=='train';val=s=='val';counts=Counter(p);weights=np.array([1/counts[pid] for pid in p]);heads={};checks={};selection_errors={}
    for family,x in [('relation',a['features']),('count_only',ca['features'])]:
        selection=common.read(source/('selection.json' if family=='relation' else 'count_only_selection.json'))
        choices=[]
        for c in [.01,.1,1.,10.]:
            model=fit(x[train],y[train],c,weights[train]);score=((x[val]-model['mean'])/model['scale'])@model['coefficient'][:-1]+model['coefficient'][-1]
            pm=pairs(y[val],score,q[val],p[val]);auc=classification(y[val],score,p[val],0)['auc'];choices.append((pm['package_macro_pair_accuracy'],auc,-c,c,model))
        chosen=max(choices,key=lambda row:row[:3]);assert chosen[3]==selection['C']
        for seed in [None,17,31,53]:
            filename=family+('_head.pt' if seed is None else f'_shuffle_{seed}_head.pt')
            saved=torch.load(source/filename,map_location='cpu',weights_only=True);model={k:saved[k].numpy() for k in ['mean','scale','coefficient']}
            labels=y[train].copy()
            if seed is not None:
                rng=np.random.default_rng(seed)
                for pid in sorted(set(p[train])):
                    ix=np.flatnonzero(p[train]==pid);labels[ix]=labels[rng.permutation(ix)]
                refitted=fit(x[train],labels,chosen[3],weights[train])
            else:refitted=chosen[4]
            error=float(np.max(np.abs(refitted['coefficient']-model['coefficient'])));assert error<1e-7
            assert np.array_equal(refitted['mean'],model['mean']) and np.array_equal(refitted['scale'],model['scale'])
            validation=((x[val]-model['mean'])/model['scale'])@model['coefficient'][:-1]+model['coefficient'][-1]
            assert abs(threshold_from_validation(y[val],validation,p[val])-float(saved['threshold']))<1e-10
            key=family if seed is None else (f'shuffle_{seed}' if family=='relation' else f'count_only_shuffle_{seed}')
            heads[key]=model;selection_errors[key]=error;checks[key+'_train_standardization_C_threshold_refit']=True
    axes=common.read(source/'full_gallery/score_axes.json');assert axes['query_ids']==d['qb']['ids'] and axes['reference_ids']==d['rb']['ids'] and axes['reference_sha256']==d['rb']['sha256']
    ledger={r['photo_id']:r for r in common.read(common.ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    original={key:{name:np.load(source/'full_gallery'/key/(name+'.npy'),mmap_mode='r') for name in ['scores','stable_sorted_reference_indices','sha_design_ranks']} for key in HEADS}
    exported={key:output_arrays(output/'relation'/key,(len(selected),5405),reproduce) for key in HEADS}
    rawmeans=np.lib.format.open_memmap(output/'relation_raw_MaxSim_means.npy',mode='w+',dtype=np.float32,shape=(len(selected),5405)) if reproduce else None
    rcounts=np.array([int((z.float().norm(dim=1)>1e-8).sum()) for z in d['rb']['embeddings']]);refmap={rid:i for i,rid in enumerate(d['rb']['ids'])}
    maximum=0.;maxmean=0.;maxjoin=0.;spoterror=0.;rank_errors=0;records=[];feature_hashes=[]
    rng=np.random.default_rng(2026101403);spots={int(qi):int(ri) for qi,ri in zip(rng.choice(selected,min(32,len(selected)),replace=False),rng.integers(5405,size=min(32,len(selected))))}
    for local,qi in enumerate(selected):
        qid=d['qb']['ids'][qi];path=source/'features'/(qid+'.npz');receipt=common.read(path.with_suffix('.json'));assert common.sha(path)==receipt['sha256'];record=np.load(path);features=record['features']
        assert features.shape==(5405,38) and features.dtype==np.float32 and np.isfinite(features).all()
        qcount=int((d['qb']['embeddings'][qi].float().norm(dim=1)>1e-8).sum());assert int(record['valid_query_tokens'])==qcount and np.array_equal(record['valid_reference_tokens'],rcounts)
        rc=rcounts.astype(np.float64);cq=np.full(5405,np.log(float(qcount)));cr=np.log(rc);cx=np.column_stack([cq,cr,np.abs(cq-cr),qcount/rc]).astype(np.float32)
        error=float(np.max(np.abs(features[:,0]-d['image']['scores'][qi].numpy())));maxmean=max(maxmean,error);assert error<3e-6
        if rawmeans is not None:rawmeans[local]=features[:,0]
        positions=np.flatnonzero(q==qid);columns=[refmap[r] for r in a['reference_ids'][positions]]
        joined=max(float(np.max(np.abs(features[columns]-a['features'][positions]))),float(np.max(np.abs(cx[columns]-ca['features'][positions]))));maxjoin=max(maxjoin,joined);assert joined==0.
        if qi in spots:
            j=spots[qi];qt=d['qb']['embeddings'][qi].float().numpy();rt=d['rb']['embeddings'][j].float().numpy();qt=qt[np.linalg.norm(qt,axis=1)>1e-8];rt=rt[np.linalg.norm(rt,axis=1)>1e-8]
            spoterror=max(spoterror,float(np.max(np.abs(cpu_relation(qt@rt.T)-features[j]))));assert spoterror<3e-5
        for key,model in heads.items():
            x=cx if key.startswith('count_only') else features;score=((x-model['mean'])/model['scale'])@model['coefficient'][:-1]+model['coefficient'][-1]
            error=float(np.max(np.abs(score-original[key]['scores'][qi])));maximum=max(maximum,error);assert error<1e-10
            order,full,rank=ranks(score,d['rb']['sha256'],set(ledger[qid]['positive_reference_sha256']))
            rank_errors+=int(np.count_nonzero(order!=original[key]['stable_sorted_reference_indices'][qi]))+int(np.count_nonzero(full!=original[key]['sha_design_ranks'][qi]))
            target=exported[key]
            if target:
                target['scores'][local]=score;target['stable_sorted_reference_indices'][local]=order;target['sha_design_ranks'][local]=full
            records.append(dict(head=key,photo_id=qid,query_index=qi,positive_rank=rank,top_reference_id=d['rb']['ids'][int(order[0])],correct=rank==1))
        feature_hashes.append(dict(photo_id=qid,path=str(path.relative_to(common.ROOT)),sha256=receipt['sha256']))
        if (local+1)%100==0:log('Frozen relation replay',local+1,'/',len(selected))
    assert rank_errors==0
    if rawmeans is not None:rawmeans.flush()
    for target in exported.values():
        if target:
            for array in target.values():array.flush()
    if reproduce:
        destination=output/'relation/models';destination.mkdir(parents=True,exist_ok=True)
        for file in list(source.glob('*head.pt'))+[source/'selection.json',source/'count_only_selection.json',source/'protocol.json']:shutil.copyfile(file,destination/file.name)
    common.write(output/'relation/score_axes.json',dict(query_ids=[d['qb']['ids'][i] for i in selected],original_query_indices=selected,reference_ids=d['rb']['ids'],reference_sha256=d['rb']['sha256']))
    common.csvout(output/'relation/query_rankings.csv',records);common.write(output/'relation/feature_sources.json',feature_hashes)
    result=dict(status='passed',checks=checks,queries=len(selected),references=5405,heads=8,scalar_values_replayed=len(selected)*5405*8,score_error=maximum,raw_MaxSim_mean_error=maxmean,independent_relation_spots=len(spots),spot_feature_error=spoterror,restricted_join_error=maxjoin,rank_mismatches=rank_errors,refit_coefficient_errors=selection_errors,exported_arrays=reproduce,
                scope='Finite38D saved cross-token relation and4D count control; original counts and sampled raw dot features; main/null train refit and val-only selections; no exhaustive information/causal claim',quick=quick)
    common.write(output/'relation/validation.json',result);log('Frozen relation replay passed',len(selected),'x5405x8')
    return result


def verify_position_content(d,base,output):
    """Rebuild the content-only inputs and independently refit saved heads."""
    from scripts.colnomic_mechanism.completion_audit import regression_fit,regression_predict,regression_metrics
    source=base/'position_content_control';a=dict(np.load(source/'head_inputs.npz'));selection=common.read(source/'selection.json')
    original=common.rows(base/'positions_reviewed/ocr_vertical_order_sample_index.csv');saved_samples=common.rows(source/'sample_index_predictions.csv')
    boxes=common.read(base/'manual_box_review/subject_boxes_after_user_review.json')['records'];line_sets={};texts=[];checks={}
    for row,stored in zip(original,saved_samples):
        assert all(stored[k]==v for k,v in row.items())
        qid=row['photo_id']
        if qid not in line_sets:
            b=boxes[qid];box=np.array(b['bbox'])/np.tile([b['width'],b['height']],2);record=d['qo'][qid];size=np.array([record['width'],record['height']])
            eligible=[line for line in record['lines'] if np.all((np.mean(line['polygon'],axis=0)/size>=box[:2])&(np.mean(line['polygon'],axis=0)/size<box[2:]))]
            line_sets[qid]=sorted(eligible,key=lambda line:np.mean(line['polygon'],axis=0)[1]);assert len(eligible)>=4
        target=json.loads(row['target'])[0];index=int(round(target*len(line_sets[qid])-.5));assert abs((index+.5)/len(line_sets[qid])-target)<1e-12
        text=line_sets[qid][index]['text'];assert text==stored['OCR_text'] and index==int(stored['OCR_vertical_index'])
        texts.append(set(re.findall(r'[^\W_]+',text.lower(),flags=re.UNICODE)))
    y=np.array([json.loads(r['target']) for r in original]);packages=np.array([r['package_id'] for r in original]);parts=np.array([r['split'] for r in original]);train=parts=='train';val=parts=='val'
    frequency=Counter(word for i,words in enumerate(texts) if train[i] for word in words);vocabulary=sorted(sorted(frequency,key=lambda word:(-frequency[word],word))[:256])
    x=np.array([[int(word in words) for word in vocabulary] for words in texts],dtype=np.float64)
    photo_counts=Counter(r['package_id'] for r in d['photos'].values());sample_counts=Counter(r['photo_id'] for r in original)
    weights=np.array([1/(sample_counts[r['photo_id']]*photo_counts[r['package_id']]) for r in original])
    checks['all46412_OCR_associations_and_historical_targets']=len(original)==len(saved_samples)==46412
    checks['all_inputs_independently_rebuilt']=np.array_equal(x,a['word_presence']) and np.array_equal(y,a['targets']) and np.array_equal(weights,a['weights']) and np.array_equal(parts,a['splits']) and np.array_equal(packages,a['package_ids'])
    checks['256_training_only_vocabulary_exact']=len(vocabulary)==256 and vocabulary==selection['vocabulary']
    choices=[]
    for alpha in [.001,.01,.1,1.,10.,100.]:
        model=regression_fit(x[train],y[train],alpha,weights[train]);m=regression_metrics(y[val],regression_predict(model,x[val]),packages[val]);choices.append((m['package_macro_mae'],alpha,model))
    chosen=min(choices,key=lambda row:(row[0],-row[1]));record=torch.load(source/'OCR_content_only_head.pt',map_location='cpu',weights_only=True)
    model={key:record[key].numpy() for key in ['mean','scale','coefficient']};coefficient_errors={};pred_errors={};metrics=common.rows(source/'metrics.csv');metric_error=0.
    checks['validation_ridge_selection_exact']=chosen[1]==selection['alpha']==record['alpha'] and record['vocabulary']==vocabulary
    for seed in [None,17,31,53]:
        if seed is None:
            key='main';refitted=chosen[2];observed=np.array([float(r['content_prediction']) for r in saved_samples])[:,None]
        else:
            key='shuffle_'+str(seed);labels=y[train].copy();rng=np.random.default_rng(seed)
            for pid in sorted(set(packages[train])):
                ix=np.flatnonzero(packages[train]==pid);labels[ix]=labels[rng.permutation(ix)]
            refitted=regression_fit(x[train],labels,chosen[1],weights[train]);record=torch.load(source/f'OCR_content_shuffle_{seed}_head.pt',map_location='cpu',weights_only=True);model={k:record[k].numpy() for k in ['mean','scale','coefficient']}
            prior=np.load(source/f'shuffle_{seed}_predictions.npz');observed=prior['predictions'];checks[key+'_labels_exact']=np.array_equal(labels,prior['shuffled_training_targets'])
        coefficient_errors[key]=float(np.max(np.abs(refitted['coefficient']-model['coefficient'])));pred=regression_predict(model,x);pred_errors[key]=float(np.max(np.abs(pred-observed)))
        checks[key+'_refit_standardization_prediction']=coefficient_errors[key]<1e-8 and pred_errors[key]<1e-10 and np.array_equal(refitted['mean'],model['mean']) and np.array_equal(refitted['scale'],model['scale'])
        for part in (['train','val','test'] if seed is None else ['test']):
            mask=parts==part;actual=regression_metrics(y[mask],pred[mask],packages[mask]);saved=next(r for r in metrics if r['split']==part and r['control']==('OCR_content_only' if seed is None else 'label_shuffle') and (seed is None or int(r['seed'])==seed))
            metric_error=max(metric_error,max(abs(float(saved[k])-v) for k,v in actual.items()))
    checks['all_saved_metrics_replayed']=metric_error<1e-10
    sources=common.read(source/'validation.json')['source_SHA256'];checks['actual_source_SHA_links']=all(common.sha(common.workspace_path(path,must_exist=True))==digest for path,digest in sources.items())
    result=dict(status='passed' if all(checks.values()) else 'failed',checks={k:bool(v) for k,v in checks.items()},samples=len(original),vocabulary_dimensions=len(vocabulary),alpha=chosen[1],coefficient_errors=coefficient_errors,prediction_errors=pred_errors,metric_max_error=metric_error,source_SHA256=sources,
                scope='Content-only correlation control: no Token features/coordinates/indices/IDs/line count enter the head; OCR/polygon/box association is extra supervision; not actual reading lines/layout identity or causal proof')
    common.write(output/'validation.json',result);assert result['status']=='passed';log('Independent content/position control replay passed',len(original),len(vocabulary))
    return result


@contextmanager
def redirected_verifier(module,source,output):
    """Legacy algorithms read history; every generated artifact goes to output."""
    originals={};old_numpy=np.savez_compressed;old_mkdir=Path.mkdir;generated={}
    def target(path):
        path=Path(path).resolve()
        if inside(path,output):return path
        if not inside(path,source):raise PermissionError('Unexpected verifier output: '+str(path))
        result=output/path.relative_to(source);old_mkdir(result.parent,parents=True,exist_ok=True);generated[path]=result;return result
    def mapped_sha(path):
        path=Path(path).resolve()
        if path in generated:return common.sha(generated[path])
        if inside(path,output):return common.sha(path)
        return common.sha(common.workspace_path(path,must_exist=True))
    def mapped_mkdir(path,*args,**kwargs):
        path=Path(path).resolve()
        if inside(path,output):return old_mkdir(path,*args,**kwargs)
        if inside(path,source):
            destination=output/path.relative_to(source);old_mkdir(destination.parent,parents=True,exist_ok=True)
            return old_mkdir(destination,*args,**kwargs)
        raise PermissionError('Unexpected verifier directory mutation: '+str(path))
    for key,fn in [('write',lambda path,value:common.write(target(path),value)),('save',lambda path,value:common.write(target(path),value)),('csvout',lambda path,value:common.csvout(target(path),value)),('table',lambda path,value:common.csvout(target(path),value)),('sha',mapped_sha),('digest',mapped_sha)]:
        if hasattr(module,key):originals[key]=getattr(module,key);setattr(module,key,fn)
    np.savez_compressed=lambda path,*args,**kwargs:old_numpy(target(path),*args,**kwargs)
    Path.mkdir=mapped_mkdir
    try:yield
    finally:
        np.savez_compressed=old_numpy
        Path.mkdir=old_mkdir
        for key,value in originals.items():setattr(module,key,value)


def run_modules(d,base,output,quick,requested=None):
    if quick:return [dict(module=name,status='not_run_quick',scope=scope) for name,_,scope in MODULES]
    results=[];saved_data=common.data;common.data=lambda:d
    try:
        for name,filename,scope in MODULES:
            if requested is not None and name not in requested:continue
            module=importlib.import_module('scripts.colnomic_mechanism.'+filename);module.ROOT=common.ROOT
            if hasattr(module,'BASE'):module.BASE=base/'controlled_crossview_layouts' if name=='crossview' else base
            if hasattr(module,'data'):module.data=lambda:d
            receipt=output/'modules'/name;receipt.mkdir(parents=True,exist_ok=True)
            oldargs=sys.argv;arguments=[filename]
            if name=='tokens':arguments+=['--output',str(base/'full_tokens'),'--receipt-dir',str(receipt),'--threads','2','--matrix-replay','all']
            elif name=='set_probes':arguments+=['--folder',str(base/'set_probes'),'--output',str(receipt/'validation.json')]
            elif name=='weighted':arguments+=['--output',str(receipt/'validation.json'),'--folders','weighted_original','weighted_spatial','weighted_spatial_reviewed','weighted_spatial_only_reviewed']
            sys.argv=arguments;started=time.monotonic()
            try:
                with redirected_verifier(module,base,receipt):
                    if name=='signed_layout':module.verify_layout(receipt)
                    elif name=='position_content':verify_position_content(d,base,receipt)
                    else:module.main()
            finally:sys.argv=oldargs
            results.append(dict(module=name,status='passed',scope=scope,elapsed_seconds=time.monotonic()-started,verifier_sha256=common.sha(Path(module.__file__)),receipts=str(receipt.relative_to(output))))
            log('Module verifier passed',name)
    finally:common.data=saved_data
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only',action='store_true');mode.add_argument('--reproduce',action='store_true')
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true',help='Fixed8 queries/all5405 refs/all8 heads; module-level long replays excluded')
    parser.add_argument('--plan-only',action='store_true',help='Validate entry plan only; does not claim numerical verification')
    args=parser.parse_args();workspace=args.workspace.resolve();initialize(workspace);torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ['',None]:raise ValueError('Run with CUDA_VISIBLE_DEVICES empty; this entry is CPU-only')
    output=fresh_output(args.output or workspace/RELATIVE_BASE/'reproduction_support/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid())),workspace)
    scratch=output/'runtime_temp';scratch.mkdir();tempfile.tempdir=str(scratch);os.environ['TMPDIR']=str(scratch)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(scratch/'torchinductor')
    install_guard(workspace,output);base=workspace/RELATIVE_BASE
    plan=dict(mode='reproduce' if args.reproduce else 'verify-only',workspace=str(workspace),output=str(output),GPU=False,encoder=False,quick=args.quick,
              baseline='all1117x5405 three frozen baseline axes/ranks; BM25/fusion rescored from OCR',relation='all1117x5405 saved38D features/rawmean and8 fitted/null heads scores/ranks',
              quick_scope='only8 stratified seeded queries, all5405 refs, all8 heads; full main/null parameter refit; other modules not run' if args.quick else None,
              module_plan=[dict(module=name,program='scripts.colnomic_mechanism.'+filename,scope=scope) for name,filename,scope in MODULES],
              source_path_policy='workspace_path explicit old-prefix mapping; no symlink/../other-prefix escape; audit hook forbids external old-workspace reads and historic writes',
              limitations=['frozen artifacts only; model weights are not reopened or bundled by this entry','historical gallery encoding source invocation is unavailable; no exact fresh encoder reproduction claimed','passed numerical artifacts do not certify semantic labels, encoder-layer causality or complete information'])
    common.write(output/'plan.json',plan)
    if args.plan_only:
        common.write(output/'validation.json',dict(status='planned_not_numerically_verified',plan_only=True,causal_complete=False));log('Plan smoke passed; no numerical acceptance claimed',output);return
    paths=payload_paths(base,output);d=common.data();selected=choose_queries(d,args.quick)
    baseline=baseline_replay(d,base,output,selected,args.reproduce);relation=relation_replay(d,base,output,selected,args.reproduce,args.quick);modules=run_modules(d,base,output,args.quick)
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',scope='quick_partial_numerical_verification' if args.quick else 'full_frozen_numerical_verification',overall_goal_acceptance=False,causal_complete=False,
                plan=plan,baselines=baseline,relation=relation,module_verification=modules,payload_source_links=dict(paths=paths['checked_payload_paths'],distinct_files=paths['distinct_files']),
                runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,CPU_threads=2,GPU_initialized=False),
                entry_sha256=common.sha(Path(__file__)),workspace_path_implementation_sha256=common.sha(Path(common.__file__)),limitations=plan['limitations'])
    common.write(output/'validation.json',result);log('Frozen audit entry passed',result['scope'],output)


if __name__=='__main__':main()
