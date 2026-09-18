#!/usr/bin/env python3
"""Independent NumPy readout; heldout labels are joined only after four seals."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_query_content_routing_oof4_v1 as R


def validate_fold(fold):
    a=R.read(R.AUTH);bundle=a['fold_sources'][str(fold)]
    folder=R.OUT/f'fold{fold:02d}'
    fv=R.read(folder/'validation.json');saved=R.read(R.checked(fv['payload']))
    R.NEED(fv['status']=='QUERY_ROUTING_FRESH_FOLD_REPLAY_PASS' and saved['authority']==fv['authority']==R.bind(R.AUTH),'FRESH_FIT_VALIDATION')
    source=R.read(R.checked(a['public_sources']['native_features']))
    fm=R.read(R.checked(a['public_sources']['fold_manifest']))['records']
    train=[i for i,r in enumerate(fm) if r['fold']!=fold];held=[i for i,r in enumerate(fm) if r['fold']==fold]
    R.NEED(saved['train_ordinals']==train and saved['heldout_ordinals']==held,'EXACT_ROLE_AXES')
    descriptor=np.load(R.checked(saved['descriptor']),allow_pickle=False)
    transforms=saved['transforms'];pc=transforms['query_pca']
    R.NEED(pc['fit_ordinals']==train,'TRAIN_ONLY_PCA_ROWS')
    mean=np.mean(descriptor[train],axis=0);centered=descriptor[train]-mean
    eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered)
    basis=eigenvectors[:,-3:][:,::-1]
    for j in range(3):
        if basis[np.argmax(abs(basis[:,j])),j]<0:basis[:,j]*=-1
    R.NEED(np.allclose(mean,pc['mean'],atol=1e-12,rtol=0) and np.allclose(basis,pc['basis'],atol=1e-8,rtol=0),'INDEPENDENT_COVARIANCE_EIGEN_PCA')
    projection=(descriptor-np.asarray(pc['mean']))@np.asarray(pc['basis'])
    pm=np.mean(projection[train],axis=0);ps=np.maximum(np.std(projection[train],axis=0),1e-12)
    R.NEED(np.allclose(pm,pc['projection_mean'],atol=1e-12,rtol=0) and np.allclose(ps,pc['projection_scale'],atol=1e-12,rtol=0),'TRAIN_ONLY_PROJECTION_SCALE')
    q=(projection-pm)/ps
    donors={}
    for ix,namespace in [(train,'QUERY_ROUTE_V1_TRAIN'),(held,'QUERY_ROUTE_V1_HELDOUT')]:
        ordered=sorted(ix,key=lambda i:hashlib.sha256((namespace+'|'+source[i]['query_id']).encode()).hexdigest())
        for k,i in enumerate(ordered):donors[str(i)]=ordered[(k+1)%len(ordered)]
    R.NEED(donors==transforms['permutation_donors'],'LABEL_FREE_SPLIT_LOCAL_DONORS')
    with np.load(R.checked(a['public_sources']['disagreement_npz']),allow_pickle=False) as npz:
        f=npz['features'].copy()
    xs=[];stats=[]
    for i,row in enumerate(source):
        x=np.array([[float.fromhex(v) for v in r] for r in row['features_binary64']['REAL']])
        w=row['base_winner_position'];cs=row['challenger_positions']
        fw=f[i,w];fc=f[i,cs]
        stats.append(np.stack((-x[:,0],fw[2]-fc[:,2],(fc[:,3]-fw[3])/(np.abs(fc[:,3])+abs(fw[3])+1e-12)),axis=-1))
        xs.append(x)
    stats=np.stack(stats);sm=stats[train].reshape(-1,3).mean(0);ss=np.maximum(stats[train].reshape(-1,3).std(0),1e-12)
    R.NEED(np.allclose(sm,transforms['stats_mean'],atol=1e-12,rtol=0) and np.allclose(ss,transforms['stats_scale'],atol=1e-12,rtol=0),'TRAIN_ONLY_STATISTICS_SCALE')
    stats=(stats-sm)/ss
    old=R.read(R.checked(bundle['parameters']))['BASE7'];w=np.array([float.fromhex(v) for v in old['weight_binary64']]);b=float.fromhex(old['bias_binary64'])
    err=0.;checks=0
    for pred,i in zip(saved['predictions'],held,strict=True):
        row=source[i];x=xs[i];aug=np.concatenate((x,np.ones((127,1))),axis=1);base=x@w+b
        R.NEED(pred['query_id']==row['query_id'] and pred['execution_ordinal']==i,'PREDICTION_ORDER')
        for model in ('BASE7',*R.MODELS):
            if model=='BASE7':z=base
            else:
                if model=='GLOBAL7':context=np.ones((127,1))
                elif model=='STATS28':context=np.concatenate((np.ones((127,1)),stats[i]),axis=1)
                else:
                    qi=i if model=='QUERY28' else donors[str(i)]
                    context=np.broadcast_to(np.concatenate(([1.],q[qi])),(127,4))
                beta=np.asarray([float.fromhex(v) for v in saved['parameters'][model]['theta_binary64']]).reshape(-1,7)
                # Independent two-matrix formula, no producer design tensor.
                z=base+np.sum((aug@beta.T)*context,axis=1)
            recorded=pred['models'][model]
            expected=np.asarray([float.fromhex(v) for v in recorded['logits_binary64']]);difference=float(np.max(abs(z-expected)))
            R.NEED(difference<=2e-10,'INDEPENDENT_NUMPY_LOGITS')
            j=int(np.argmax(z));pos=row['challenger_positions'][j] if z[j]>0 else row['base_winner_position']
            R.NEED(row['candidate_physical_rows'][pos]==recorded['selected_physical_row'],'EXACT_INDEPENDENT_ACTION')
            err=max(err,difference);checks+=127
    value=dict(status='QUERY_ROUTING_INDEPENDENT_NUMPY_FOLD_PASS',fold=fold,payload=fv['payload'],fresh_validation=R.bind(folder/'validation.json'),authority=R.bind(R.AUTH),logit_checks=checks,max_abs_logit_error=err,heldout_label_reads=0,EVAL_reads=0)
    R.write(folder/'independent_validation.json',value)
    print(json.dumps(value),flush=True)


def compare(rows,new,base):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    differences=[];ledger=[]
    for g,rs in sorted(groups.items()):
        delta=sum(int(r['correct'][new])-int(r['correct'][base]) for r in rs)
        v=Fraction(delta,len(rs));differences.append(v)
        ledger.append(dict(group=g,count=len(rs),net=delta,accuracy_difference_fraction=str(v)))
    values=np.asarray([float(v) for v in differences])
    indices=np.random.default_rng(20260912).integers(0,len(values),(10000,len(values)))
    ci=np.quantile(values[indices].mean(1),[.025,.975])
    rescue=[r['original_query_id'] for r in rows if r['correct'][new] and not r['correct'][base]]
    loss=[r['original_query_id'] for r in rows if not r['correct'][new] and r['correct'][base]]
    return dict(rescue=len(rescue),loss=len(loss),net=len(rescue)-len(loss),rescue_query_ids=rescue,loss_query_ids=loss,equal_group_difference=float(np.mean(values)),group_bootstrap_95=ci.tolist(),positive_groups=sum(v>0 for v in differences),negative_groups=sum(v<0 for v in differences),groups=ledger,exact_group_signflip=R.P.exact_group_signflip(differences))


def join():
    a=R.read(R.AUTH);payloads=[];receipts=[]
    for fold in range(4):
        folder=R.OUT/f'fold{fold:02d}';v=R.read(folder/'independent_validation.json');fv=R.read(R.checked(v['fresh_validation']));p=R.read(R.checked(v['payload']))
        R.NEED(v['status']=='QUERY_ROUTING_INDEPENDENT_NUMPY_FOLD_PASS' and v['authority']==fv['authority']==p['authority']==R.bind(R.AUTH),'ALL_FOLDS_VALIDATED')
        R.NEED(v['payload']==fv['payload'] and v['fold']==p['fold']==fold,'FOLD_PAYLOAD')
        payloads.append(p);receipts.append(R.bind(folder/'independent_validation.json'))
    R.NEED(sorted(r['execution_ordinal'] for p in payloads for r in p['predictions'])==list(range(128)),'ALL128_PREDICTIONS_BEFORE_LABELS')
    R.write(R.OUT/'all_predictions_prelabel_seal.json',dict(status='ALL_QUERY_ROUTING_OOF_PREDICTIONS_VALIDATED_BEFORE_LABELS',fold_validations=receipts,authority=R.bind(R.AUTH),query_count=128))
    source=R.read(R.checked(a['public_sources']['native_features']));labels,mapping=R.P.gallery_labels()
    rows=[]
    for fold,p in enumerate(payloads):
        bundle=a['fold_sources'][str(fold)]
        trainroles=R.read(R.checked(bundle['train_roles']))['records']
        roles=R.read(R.checked(bundle['heldout_roles']))['records'];rolemap={r['query_id']:r for r in roles}
        for key in ('query_id','identity','group','source_image_sha256'):
            R.NEED(not ({r[key] for r in roles}&{r[key] for r in trainroles}),'TRAIN_HELDOUT_DISJOINT:'+key)
        for pred in p['predictions']:
            role=rolemap[pred['query_id']];f=source[pred['execution_ordinal']];raw=f['raw_ranked_physical_rows'];target=role['identity']
            R.NEED(role['source_image_sha256']==f['source_image_sha256'] and role['execution_ordinal']==pred['execution_ordinal'],'CURATOR_POSTSEAL_JOIN')
            rawphysical=f['candidate_physical_rows'][f['base_winner_position']]
            rawrank=next(i+1 for i,g in enumerate(raw) if labels[g]==target)
            selected={'RAW':rawphysical,**{m:d['selected_physical_row'] for m,d in pred['models'].items()}}
            correct={m:labels[g]==target for m,g in selected.items()}
            ranks={m:1 if correct[m] else rawrank+int(raw.index(g)+1>rawrank) for m,g in selected.items()}
            R.NEED(ranks['RAW']==rawrank,'RAW_RANK_REPLAY')
            rows.append(dict(query_id=pred['query_id'],original_query_id=role['original_query_id'],execution_ordinal=pred['execution_ordinal'],fold=fold,group=role['group'],identity=target,correct=correct,ranks=ranks,selected_physical_rows=selected,target_in_C128=target in {labels[g] for g in f['candidate_physical_rows']}))
    rows.sort(key=lambda r:r['execution_ordinal']);R.NEED(len(rows)==128 and len({r['identity'] for r in rows})==len({r['group'] for r in rows})==32,'POPULATION_128_32_32')
    models=['RAW','BASE7',*R.MODELS];scores={}
    for m in models:
        correct=sum(r['correct'][m] for r in rows);rr=sum((Fraction(1,r['ranks'][m]) for r in rows),Fraction(0))/128
        scores[m]=dict(correct=correct,accuracy=correct/128,MRR=float(rr),MRR_fraction=str(rr))
    pairs=[('QUERY28',b) for b in ('BASE7','GLOBAL7','STATS28','QUERY_PERM28')]+[(m,'BASE7') for m in ('GLOBAL7','STATS28','QUERY_PERM28')]
    comparisons={f'{base}__to__{new}':compare(rows,new,base) for new,base in pairs}
    result=dict(status='QUERY_CONTENT_ROUTING_TRAIN128_OOF4_DEVELOPMENT_READOUT_COMPLETE',authority=R.bind(R.AUTH),evidence_level='historically opened TRAIN grouped OOF development; no EVAL32/EVAL128/593 evaluation',query_count=128,group_count=32,identity_count=32,candidate_source='unchanged RAW full-gallery C128',action='all127 challengers, max>0 SWITCH else RAW HOLD',scores=scores,comparisons=comparisons,rows=rows,fold_metrics={str(f):{m:sum(r['correct'][m] for r in rows if r['fold']==f) for m in models} for f in range(4)},PCA_explained_variance={str(p['fold']):p['transforms']['query_pca']['explained_variance_ratio'] for p in payloads},fold_validations=receipts,base_training_updates=0,new_parameters={'GLOBAL7':7,'STATS28':28,'QUERY28':28,'QUERY_PERM28':28},query_observed_net_gain_vs_base=comparisons['BASE7__to__QUERY28']['net']>0,query_observed_above_all_matched_controls=all(comparisons[b+'__to__QUERY28']['net']>0 for b in ('GLOBAL7','STATS28','QUERY_PERM28')),HYP_GO_claimed=False,deployment_changed=False)
    R.write(R.OUT/'result.json',result)
    # Independent boolean-array counting cross-check after the row ledger exists.
    boolean=np.asarray([[r['correct'][m] for m in models] for r in rows],dtype=np.int64)
    R.NEED(boolean.sum(0).tolist()==[scores[m]['correct'] for m in models],'COLUMNWISE_COUNTS')
    for new,base in pairs:
        delta=boolean[:,models.index(new)]-boolean[:,models.index(base)];c=comparisons[base+'__to__'+new]
        R.NEED(int((delta==1).sum())==c['rescue'] and int((delta==-1).sum())==c['loss'] and int(delta.sum())==c['net'],'INDEPENDENT_PAIRED_COUNTS')
    R.write(R.OUT/'result_validation.json',dict(status='QUERY_ROUTING_ALL_FOLD_ACTIONS_RANKS_COUNTS_PASS',result=R.bind(R.OUT/'result.json'),prelabel_seal=R.bind(R.OUT/'all_predictions_prelabel_seal.json'),logit_checks=128*127*5,query_count=128,authority=R.bind(R.AUTH),EVAL_reads=0))
    report=['# 照片内容条件化：TRAIN128四折结果','', '本轮只在原TRAIN128/32身份/32来源组开发。各折原BASE7直接复用；新四臂同FULL-C128交叉熵、2000步。128不是原99/128的EVAL面板。','', '| 模型 | 正确/128 | MRR |','|---|---:|---:|']
    report += [f"| {m} | {scores[m]['correct']} | {scores[m]['MRR']:.9f} |" for m in models]
    report += ['', '| QUERY28相对对照 | 救回 | 损失 | 净增 | 正/负组 | 等组差95%区间 |','|---|---:|---:|---:|---|---|']
    for base in ('BASE7','GLOBAL7','STATS28','QUERY_PERM28'):
        c=comparisons[base+'__to__QUERY28'];report.append(f"| {base} | {c['rescue']} | {c['loss']} | {c['net']} | {c['positive_groups']}/{c['negative_groups']} | {c['group_bootstrap_95']} |")
    report += ['', '照片描述是全图token均值的TRAIN内前三主成分；不是人工类型标签、数字识别器或候选细粒度差异读取器。PCA所保留的方差见result.json。没有根据留出结果更改维度、seed、epoch或阈值。','', '主臂对原基头观察净增：'+str(result['query_observed_net_gain_vs_base'])+'；同时超过全部匹配对照：'+str(result['query_observed_above_all_matched_controls'])+'。组统计不确定性另列，不恢复零损失门。','', '该结果仅检验当前粗内容摘要和读出配方；无收益不等于全部照片条件化不可行，有收益也不直接等于new HYP理论成立。没有旧EVAL重测或部署替换。','', '[机器结果](result.json) · [独立核算](result_validation.json)']
    R.write_bytes(R.OUT/'report.md',('\n'.join(report)+'\n').encode())
    print(json.dumps(dict(status=result['status'],scores=scores,comparisons={k:{f:v[f] for f in ('rescue','loss','net')} for k,v in comparisons.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['verify','join']);ap.add_argument('--fold',type=int);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);R.guard(args.stage,args.fold)
    R.NEED(R.bind(__file__)==R.read(R.AUTH)['validator'],'FROZEN_VALIDATOR')
    if args.stage=='verify':validate_fold(args.fold)
    else:join()
