#!/usr/bin/env python3
"""Independent feature/readout algebra; heldout labels only after all seals."""
import argparse
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_local_evidence_oof4_v1 as N
import validate_rc_query_content_routing_oof4_v1 as V


def independent_stats():
    all_stats=[]
    for shard in range(16):
        folder=N.OUT/f'cache{shard:02d}'
        v=N.read(folder/'validation.json');r=N.read(N.checked(v['receipt']))
        N.need(v['status']=='PAIRED_LOCAL_FRESH_NUMPY_CACHE_PASS' and v['authority']==r['authority']==N.bind(N.AUTH) and v['payload']==r['payload'],'CACHE_QUALIFIED')
        with np.load(N.checked(r['payload']),allow_pickle=False) as data:
            for offset,meta in enumerate(r['records']):
                i=shard*8+offset;N.need(meta['execution_ordinal']==i,'CACHE_ORDINAL')
                p=data[f'free_{i:03d}'];wq=data[f'wq_{i:03d}'];winner=meta['base_winner_position']
                # Per-candidate dot products over a common query-token axis.
                result=[]
                for c in range(128):
                    weight=wq[c]+wq[winner];d=p[c]-p[winner];den=max(float(np.sum(weight)),1e-12)
                    result.append([float(np.dot(weight,d)/den),float(np.dot(weight,d*np.abs(d))/den)])
                result=np.asarray(result)
                N.need(np.allclose(result,data['statistics'][offset],atol=2e-12,rtol=0),'INDEPENDENT_JOINT_REDUCTION')
                all_stats.append(result)
    return np.stack(all_stats)


def verify(fold):
    a=N.read(N.AUTH);b=a['fold_sources'][str(fold)];folder=N.OUT/f'fold{fold:02d}'
    v=N.read(folder/'validation.json');p=N.read(N.checked(v['payload']))
    N.need(v['status']=='PAIRED_LOCAL_FRESH_FIT_REPLAY_PASS' and p['authority']==v['authority']==N.bind(N.AUTH),'FRESH_FIT_QUALIFIED')
    rows=N.read(N.checked(a['public_sources']['native_features']))
    fm=N.read(N.checked(a['public_sources']['fold_manifest']))['records']
    train=[i for i,r in enumerate(fm) if r['fold']!=fold];held=[i for i,r in enumerate(fm) if r['fold']==fold]
    N.need(p['train_ordinals']==train and p['heldout_ordinals']==held,'ORIGINAL_FOLD_AXES')
    values=independent_stats()
    diff=np.stack([values[i,r['challenger_positions']] for i,r in enumerate(rows)])
    m1,m2=diff[:,:,0],diff[:,:,1]
    features={'BIAS1':np.empty((128,127,0)),'MEAN2':m1[:,:,None],'CURVE3':np.stack((m1,m1*np.abs(m1)),axis=-1),'JOINT3':np.stack((m1,m2),axis=-1)}
    design={}
    for model,f in features.items():
        t=p['transforms'][model];N.need(t['fit_ordinals']==train,'TRAIN_TRANSFORM_FIT_AXIS')
        if f.shape[-1]:
            fit=f[train].reshape(-1,f.shape[-1]);mean=fit.mean(0);scale=np.maximum(fit.std(0),1e-12)
            N.need(np.allclose(mean,t['mean'],atol=2e-12,rtol=0) and np.allclose(scale,t['scale'],atol=2e-12,rtol=0),'INDEPENDENT_TRAIN_STANDARDIZATION')
            # Use recorded transform after independently checking it.
            f=(f-np.asarray(t['mean']))/np.asarray(t['scale'])
        else:N.need(t['mean']==t['scale']==[],'BIAS_UNSTANDARDIZED')
        design[model]=np.concatenate((np.ones((128,127,1)),f),axis=-1)
    old=N.read(N.checked(b['parameters']))['BASE7'];w=np.asarray([float.fromhex(x) for x in old['weight_binary64']]);bias=float.fromhex(old['bias_binary64'])
    err=0.;checks=0
    for pred,i in zip(p['predictions'],held,strict=True):
        r=rows[i];N.need(pred['query_id']==r['query_id'] and pred['execution_ordinal']==i,'PREDICTION_JOIN')
        x=np.asarray([[float.fromhex(v) for v in row] for row in r['features_binary64']['REAL']]);base=x@w+bias
        for m in ('BASE7',*N.MODELS):
            z=base if m=='BASE7' else base+np.einsum('jk,k->j',design[m][i],np.asarray([float.fromhex(v) for v in p['parameters'][m]['theta_binary64']]))
            saved=pred['models'][m];expected=np.asarray([float.fromhex(v) for v in saved['logits_binary64']]);error=float(np.max(np.abs(z-expected)))
            N.need(error<=2e-10,'INDEPENDENT_NUMPY_ALL_LOGITS')
            j=int(np.argmax(z));pos=r['challenger_positions'][j] if z[j]>0 else r['base_winner_position']
            N.need(r['candidate_physical_rows'][pos]==saved['selected_physical_row'],'INDEPENDENT_ACTION')
            err=max(err,error);checks+=127
    N.write(folder/'independent_validation.json',dict(status='PAIRED_LOCAL_INDEPENDENT_NUMPY_FOLD_PASS',fold=fold,authority=N.bind(N.AUTH),payload=v['payload'],fresh_validation=N.bind(folder/'validation.json'),logit_checks=checks,max_abs_logit_error=err,heldout_label_reads=0,EVAL_reads=0))
    print(dict(stage='independent-fold',fold=fold,logit_checks=checks,max_abs_error=err),flush=True)


def join():
    a=N.read(N.AUTH);payloads=[];receipts=[];global_predictions=[]
    for fold in range(4):
        folder=N.OUT/f'fold{fold:02d}';v=N.read(folder/'independent_validation.json');fresh=N.read(N.checked(v['fresh_validation']));p=N.read(N.checked(v['payload']))
        N.need(v['status']=='PAIRED_LOCAL_INDEPENDENT_NUMPY_FOLD_PASS' and v['authority']==fresh['authority']==p['authority']==N.bind(N.AUTH) and v['payload']==fresh['payload'] and p['fold']==v['fold']==fold,'ALL_FOLDS_QUALIFIED')
        payloads.append(p);receipts.append(N.bind(folder/'independent_validation.json'))
        gb=a['global_sources'][str(fold)];gv=N.read(N.checked(gb['validation']));gp=N.read(N.checked(gb['payload']))
        N.need(gv['status']=='QUERY_ROUTING_INDEPENDENT_NUMPY_FOLD_PASS' and gv['payload']==gb['payload'] and gp['fold']==fold and gp['train_ordinals']==p['train_ordinals'] and gp['heldout_ordinals']==p['heldout_ordinals'],'GLOBAL_SAME_FOLD_QUALIFIED')
        global_predictions.append({r['query_id']:r for r in gp['predictions']})
    N.need(sorted(r['execution_ordinal'] for p in payloads for r in p['predictions'])==list(range(128)),'COMPLETE_PRELABEL_PREDICTIONS')
    N.write(N.OUT/'all_predictions_prelabel_seal.json',dict(status='PAIRED_LOCAL_ALL128_VALIDATED_BEFORE_LABELS',authority=N.bind(N.AUTH),fold_validations=receipts,query_count=128))
    source=N.read(N.checked(a['public_sources']['native_features']));labels,mapping=N.P.gallery_labels();rows=[]
    for fold,p in enumerate(payloads):
        bundle=a['fold_sources'][str(fold)]
        oldseal=N.read(N.checked(bundle['seal']));N.need(mapping==oldseal['closure']['gallery_mapping_sha256'],'JOIN_GALLERY_MAPPING')
        train=N.read(N.checked(bundle['train_roles']))['records'];held=N.read(N.checked(bundle['heldout_roles']))['records'];rolemap={r['query_id']:r for r in held}
        for key in ('query_id','identity','group','source_image_sha256'):N.need(not ({r[key] for r in train}&{r[key] for r in held}),'TRAIN_HELDOUT_DISJOINT:'+key)
        for pred in p['predictions']:
            i=pred['execution_ordinal'];f=source[i];role=rolemap[pred['query_id']]
            N.need(role['execution_ordinal']==i and role['source_image_sha256']==f['source_image_sha256'],'POSTSEAL_LABEL_JOIN')
            gp=global_predictions[fold][pred['query_id']]
            N.need(gp['models']['BASE7']==pred['models']['BASE7'],'GLOBAL_BASE_EXACT_SAME_PREDICTIONS')
            raw=f['raw_ranked_physical_rows'];rawrow=f['candidate_physical_rows'][f['base_winner_position']]
            selected={'RAW':rawrow,**{m:d['selected_physical_row'] for m,d in pred['models'].items()},'GLOBAL7':gp['models']['GLOBAL7']['selected_physical_row']}
            target=role['identity'];correct={m:labels[g]==target for m,g in selected.items()};ranks={}
            for m,g in selected.items():
                # Materialize final physical gallery order independently of shortcut rank formula.
                order=[g]+[x for x in raw if x!=g]
                N.need(len(order)==5413 and len(set(order))==5413,'FULL_GALLERY_ACTION_RANKING')
                ranks[m]=next(j+1 for j,x in enumerate(order) if labels[x]==target)
                N.need((ranks[m]==1)==correct[m],'RANK_CORRECT_PARITY')
            rows.append(dict(query_id=pred['query_id'],original_query_id=role['original_query_id'],execution_ordinal=i,fold=fold,identity=target,group=role['group'],correct=correct,ranks=ranks,selected_physical_rows=selected))
    rows.sort(key=lambda r:r['execution_ordinal']);N.need(len(rows)==128 and len({r['identity'] for r in rows})==len({r['group'] for r in rows})==32,'TRAIN128_32_IDENTITIES_32_GROUPS')
    models=('RAW','BASE7','GLOBAL7',*N.MODELS);scores={}
    for m in models:
        rr=sum((Fraction(1,r['ranks'][m]) for r in rows),Fraction(0))/128
        scores[m]=dict(correct=sum(r['correct'][m] for r in rows),MRR=float(rr),MRR_fraction=str(rr))
    pairs=[(m,'BASE7') for m in N.MODELS]+[('JOINT3',b) for b in ('GLOBAL7','BIAS1','MEAN2','CURVE3')]
    comparisons={b+'__to__'+m:V.compare(rows,m,b) for m,b in pairs}
    boolean=np.asarray([[r['correct'][m] for m in models] for r in rows],dtype=np.int64)
    N.need(boolean.sum(0).tolist()==[scores[m]['correct'] for m in models],'INDEPENDENT_COLUMN_COUNTS')
    for m,b in pairs:
        d=boolean[:,models.index(m)]-boolean[:,models.index(b)];c=comparisons[b+'__to__'+m]
        N.need(int((d==1).sum())==c['rescue'] and int((d==-1).sum())==c['loss'] and int(d.sum())==c['net'],'INDEPENDENT_PAIRED_COUNTS')
    screen=all(comparisons[b+'__to__JOINT3']['net']>0 for b in ('BASE7','GLOBAL7','BIAS1','MEAN2','CURVE3'))
    result=dict(status='PAIRED_LOCAL_TRAIN128_OOF4_DEVELOPMENT_COMPLETE',authority=N.bind(N.AUTH),evidence_level='historically opened TRAIN128 grouped OOF development; no fixed EVAL32/EVAL128/other593',query_count=128,identity_count=32,group_count=32,candidate_source='unchanged RAW full-gallery C128',action='all127 physical-order challengers; max>0 SWITCH else HOLD',scores=scores,comparisons=comparisons,rows=rows,fold_metrics={str(k):{m:sum(r['correct'][m] for r in rows if r['fold']==k) for m in models} for k in range(4)},primary='JOINT3',screen_positive=screen,HYP_GO_claimed=False,deployment_changed=False,EVAL_reads=0,base_training_updates=0,fold_validations=receipts)
    N.write(N.OUT/'result.json',result)
    N.write(N.OUT/'result_validation.json',dict(status='PAIRED_LOCAL_ALL_LOGITS_ACTIONS_RANKS_COUNTS_PASS',result=N.bind(N.OUT/'result.json'),prelabel_seal=N.bind(N.OUT/'all_predictions_prelabel_seal.json'),authority=N.bind(N.AUTH),new_fold_logit_checks=128*127*5,query_count=128,EVAL_reads=0))
    report=['# 同位置候选内容差：TRAIN128四折结果','','原TRAIN128、32身份/32组，RAW C128/127 challenger；旧EVAL32和EVAL128未重测。原折BASE7和既有GLOBAL7预测直接复用。','', '| 模型 | 正确/128 | MRR |','|---|---:|---:|']
    report += [f"| {m} | {scores[m]['correct']} | {scores[m]['MRR']:.9f} |" for m in models]
    report += ['', '| JOINT3 相对 | 救回 | 损失 | 净增 | 等组差95%区间 |','|---|---:|---:|---:|---|']
    for b in ('BASE7','GLOBAL7','BIAS1','MEAN2','CURVE3'):
        c=comparisons[b+'__to__JOINT3'];report.append(f"| {b} | {c['rescue']} | {c['loss']} | {c['net']} | {c['group_bootstrap_95']} |")
    report += ['', 'JOINT3=m2加平均差与bias；m2是同query位置的候选内容差的加权有符号二阶量。CURVE3仅对平均差做相同非线性，两臂新参数数目相同。','', f'超过全部固定对照的开发screen：{screen}。允许损失，组不确定性及精确组检验见result.json。该结果不自动成为HYP GO、空间因果或旧EVAL技术提升。','', '本统计无增益只否定本配方；有增益需后续独立迁移检验。没有根据留出结果追加角度、温度、维度、门或epoch。']
    N.write_bytes(N.OUT/'report.md',('\n'.join(report)+'\n').encode())
    print(dict(status=result['status'],scores=scores,screen_positive=screen),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['verify','join']);ap.add_argument('--fold',type=int);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);N.guard(args.stage,args.fold)
    N.need(N.bind(__file__)==N.read(N.AUTH)['code_sources']['validator'],'FROZEN_VALIDATOR')
    if args.stage=='verify':verify(args.fold)
    else:join()
