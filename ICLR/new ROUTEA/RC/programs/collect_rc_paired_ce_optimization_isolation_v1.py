#!/usr/bin/env python3
"""Post-seal grouped readout, preserving all old-model row ledgers."""
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_ce_optimization_isolation_v1 as C
import validate_rc_query_content_routing_oof4_v1 as V


def join():
    a=C.read(C.AUTH);payloads=[];validations=[]
    for fold in range(4):
        folder=C.OUT/f'fold{fold:02d}';v=C.read(folder/'validation.json');p=C.read(C.checked(v['payload']))
        C.need(v['status']=='PAIRED_CE_OPT_EXACT_BOUND_AND_PREDICTION_REPLAY_PASS' and v['authority']==p['authority']==C.bind(C.AUTH) and v['fold']==p['fold']==fold,'ALL_FOUR_FOLDS_VALIDATED')
        payloads.append(p);validations.append(C.bind(folder/'validation.json'))
    C.need(sorted(r['execution_ordinal'] for p in payloads for r in p['predictions'])==list(range(128)),'ALL128_PRELABEL_PREDICTIONS')
    C.write(C.OUT/'all_predictions_prelabel_seal.json',dict(status='ALL_CE_OPT_FOLDS_SEALED_BEFORE_LABELS',authority=C.bind(C.AUTH),fold_validations=validations,query_count=128))
    previous=C.read(C.checked(a['join_sources']['previous_result']));pv=C.read(C.checked(a['join_sources']['previous_validation']))
    C.need(pv['result']==a['join_sources']['previous_result'] and pv['status']=='PAIRED_LOCAL_ALL_LOGITS_ACTIONS_RANKS_COUNTS_PASS','PREVIOUS_QUALIFIED_READOUT')
    lookup={r['query_id']:r for r in previous['rows']};source=C.read(C.checked(a['public_sources']['native_features']))
    labels,mapping=C.N.P.gallery_labels();rows=[];fold_losses={}
    for fold,p in enumerate(payloads):
        b=a['fold_sources'][str(fold)];train=C.read(C.checked(b['train_roles']))['records'];held=C.read(C.checked(b['heldout_roles']))['records'];roles={r['query_id']:r for r in held}
        C.need(mapping==C.read(C.checked(b['seal']))['closure']['gallery_mapping_sha256'],'GALLERY_MAPPING')
        for key in ('query_id','identity','group','source_image_sha256'):C.need(not ({r[key] for r in train}&{r[key] for r in held}),'DISJOINT_'+key)
        losses={m+'_'+s:[] for m in C.MODELS for s in ('ADAMW','CEOPT')}
        for pred in p['predictions']:
            i=pred['execution_ordinal'];f=source[i];role=roles[pred['query_id']];old=lookup[pred['query_id']]
            C.need(i==role['execution_ordinal']==old['execution_ordinal'] and f['source_image_sha256']==role['source_image_sha256'],'POSTSEAL_QUERY_JOIN')
            raw=f['raw_ranked_physical_rows'];target=role['identity'];candidates=f['candidate_physical_rows'];winner=f['base_winner_position'];cs=f['challenger_positions']
            target_positions=[j for j,g in enumerate(candidates) if labels[g]==target];C.need(len(target_positions)==1,'HELD_TARGET_IN_C128')
            t=0 if target_positions[0]==winner else cs.index(target_positions[0])+1
            selected={m:old['selected_physical_rows'][m] for m in ('RAW','BASE7','GLOBAL7')}
            for m,d in pred['models'].items():
                selected[m]=d['selected_physical_row'];z=np.array([0.]+[float.fromhex(v) for v in d['logits_binary64']]);losses[m].append(float(logsumexp(z)-z[t]))
                if m.endswith('_ADAMW'):C.need(selected[m]==old['selected_physical_rows'][m.removesuffix('_ADAMW')],'OLD_HEAD_SELECTION_PARITY')
            correct={m:labels[g]==target for m,g in selected.items()};ranks={}
            for m,g in selected.items():
                order=[g]+[x for x in raw if x!=g]
                C.need(g in raw and len(order)==len(set(order))==len({labels[x] for x in order})==5412,'COMPLETE_5412_IDENTITY_RANKING')
                ranks[m]=next(k+1 for k,x in enumerate(order) if labels[x]==target)
                parent=m.removesuffix('_ADAMW')
                if m in ('RAW','BASE7','GLOBAL7') or m.endswith('_ADAMW'):
                    C.need(correct[m]==old['correct'][parent] and ranks[m]==old['ranks'][parent],'ALL_OLD_COUNTS_RANKS_PARITY')
            rows.append(dict(query_id=pred['query_id'],original_query_id=role['original_query_id'],execution_ordinal=i,fold=fold,group=role['group'],identity=target,correct=correct,ranks=ranks,selected_physical_rows=selected,heldout_ce={m:losses[m][-1] for m in losses}))
        fold_losses[str(fold)]={}
        for m in C.MODELS:
            q=p['parameters'][m]
            fold_losses[str(fold)][m]=dict(old_TRAIN_CE=q['old_loss'],new_TRAIN_CE=q['new_loss'],TRAIN_improvement=q['old_loss']-q['new_loss'],certified_box_gap=q['new_certificate']['gap_upper_float'],certified_near_optimum=q['new_certificate']['certified_box_gap_le_1e_6'],old_suboptimality_interval=q['old_box_suboptimality_interval'],boundary_coordinates=q['boundary_coordinates'],old_HELD_CE=float(np.mean(losses[m+'_ADAMW'])),new_HELD_CE=float(np.mean(losses[m+'_CEOPT'])))
    rows.sort(key=lambda r:r['execution_ordinal']);C.need(len(rows)==128 and len({r['identity'] for r in rows})==len({r['group'] for r in rows})==32,'ORIGINAL_TRAIN128_POPULATION')
    models=['RAW','BASE7','GLOBAL7']+[m+'_'+s for m in C.MODELS for s in ('ADAMW','CEOPT')]
    scores={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=float(sum((Fraction(1,r['ranks'][m]) for r in rows),Fraction(0))/128)) for m in models}
    pairs=[(m+'_CEOPT',m+'_ADAMW') for m in C.MODELS]+[('JOINT3_CEOPT',b) for b in ('MEAN2_CEOPT','CURVE3_CEOPT','BASE7','GLOBAL7')]
    comparisons={b+'__to__'+m:V.compare(rows,m,b) for m,b in pairs}
    boolean=np.asarray([[r['correct'][m] for m in models] for r in rows],dtype=np.int64)
    C.need(boolean.sum(0).tolist()==[scores[m]['correct'] for m in models],'COLUMNWISE_COUNTS')
    for m,b in pairs:
        delta=boolean[:,models.index(m)]-boolean[:,models.index(b)];v=comparisons[b+'__to__'+m]
        C.need(v['rescue']==int((delta==1).sum()) and v['loss']==int((delta==-1).sum()) and v['net']==int(delta.sum()),'INDEPENDENT_PAIRED_COUNTS')
    result=dict(status='PAIRED_CE_OPTIMIZATION_ISOLATION_TRAIN_OOF4_COMPLETE',authority=C.bind(C.AUTH),query_count=128,identity_count=32,group_count=32,evidence_level='opened TRAIN128 grouped OOF optimization diagnostic; no fixed EVAL or other593',candidate_source='frozen RAW C128',action='all127 challengers; max>0 SWITCH else HOLD',scores=scores,comparisons=comparisons,fold_loss_diagnosis=fold_losses,rows=rows,fold_metrics={str(f):{m:sum(r['correct'][m] for r in rows if r['fold']==f) for m in models} for f in range(4)},all_box_gaps_certified=all(d['certified_near_optimum'] for f in fold_losses.values() for d in f.values()),HYP_GO_claimed=False,deployment_changed=False,EVAL_reads=0,scope='supplied FP64 endpoints viewed as exact reals, standardized parameter box [-64,64], unregularized CE data loss; not AdamW decay equivalence')
    C.write(C.OUT/'result.json',result)
    C.write(C.OUT/'result_validation.json',dict(status='PAIRED_CE_OPT_CERTIFICATES_ACTIONS_RANKS_COUNTS_PASS',authority=C.bind(C.AUTH),result=C.bind(C.OUT/'result.json'),fold_validations=validations,prelabel_seal=C.bind(C.OUT/'all_predictions_prelabel_seal.json'),logit_checks=128*127*6,old_models_preserved=['RAW','BASE7','GLOBAL7',*[m+'_ADAMW' for m in C.MODELS]],EVAL_reads=0))
    report=['# 同函数类CE优化：TRAIN128四折','','固定原TRAIN四折及完整RAW C128，仅优化无额外正则CE；不等同继续原AdamW权重衰减过程。','', '| 模型 | 正确/128 | MRR |','|---|---:|---:|']
    report += [f"| {m} | {scores[m]['correct']} | {scores[m]['MRR']:.9f} |" for m in models]
    report += ['', '| 折 | 模型 | TRAIN旧→新CE | HELD旧→新CE | 盒内剩余间隙上界 |','|---|---|---|---|---:|']
    for f,items in fold_losses.items():
        for m,d in items.items():report.append(f"| {f} | {m} | {d['old_TRAIN_CE']:.9f}→{d['new_TRAIN_CE']:.9f} | {d['old_HELD_CE']:.9f}→{d['new_HELD_CE']:.9f} | {d['certified_box_gap']:.3g} |")
    report += ['', '全部盒内间隙≤1e-6：'+str(result['all_box_gaps_certified'])+'。这些界只覆盖预先固定参数盒及已封存实数端点目标；不是无界参数域、统计最优或HYP GO。组不确定性、逐图变化及完整证书见机器产物。','', '旧EVAL32/128没有重测，当前准确率不得拼入原28/32、99/128账本。']
    C.write_bytes(C.OUT/'report.md',('\n'.join(report)+'\n').encode())
    print(dict(status=result['status'],scores=scores,all_box_gaps_certified=result['all_box_gaps_certified']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(8);torch.set_num_interop_threads(1);C.guard('join');C.need(C.read(C.AUTH)['code_sources']['collector']==C.bind(__file__),'COLLECTOR_PIN');join()
