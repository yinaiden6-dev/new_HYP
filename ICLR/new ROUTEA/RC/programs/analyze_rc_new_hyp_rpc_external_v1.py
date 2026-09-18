#!/usr/bin/env python3
"""Join sealed RPC600 predictions; frozen SKU-stratified bootstrap."""
import collections
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=ROOT/'results/rc_new_hyp_rpc_transfer_v1'
AUTH=ROOT/'registry/rc_new_hyp_rpc_inference_authority_v1_20260913.json'

def comparison(rows,baseline,model,draws=100000):
    sku=collections.defaultdict(list)
    for row in rows:sku[row['identity']].append(row)
    assert len(sku)==200 and all(len(v)==3 for v in sku.values())
    strata=collections.defaultdict(list)
    for identity,rr in sorted(sku.items()):
        assert len({r['stratum'] for r in rr})==1
        d=sum(int(r['correct'][model])-int(r['correct'][baseline]) for r in rr)/3
        strata[rr[0]['stratum']].append(d)
    assert len(strata)==17
    rng=np.random.default_rng(20260913);boot=np.zeros(draws,dtype=np.float64)
    for name,ds in sorted(strata.items()):
        ds=np.array(ds,dtype=np.float64)
        boot+=ds[rng.integers(len(ds),size=(draws,len(ds)))].sum(axis=1)/200
    ci=np.quantile(boot,[.025,.975]).tolist()
    group_means=np.array([np.mean(ds) for _,ds in sorted(strata.items())])
    rng=np.random.default_rng(20260913)
    group_boot=group_means[rng.integers(17,size=(draws,17))].mean(axis=1)
    group_ci=np.quantile(group_boot,[.025,.975]).tolist()
    rescue=sum(not r['correct'][baseline] and r['correct'][model] for r in rows)
    loss=sum(r['correct'][baseline] and not r['correct'][model] for r in rows)
    mean=sum(sum(ds) for ds in strata.values())/200
    assert abs(mean-(rescue-loss)/600)<1e-12
    return dict(rescue=rescue,loss=loss,net=rescue-loss,query_difference=(rescue-loss)/600,equal_sku_difference=mean,
                sku_stratified_bootstrap95=ci,sku_count=200,supercategory_count=17,
                supercategory_equal_mean_difference=float(group_means.mean()),supercategory_bootstrap95=group_ci,
                reliable_positive=bool(rescue>loss and mean>0 and ci[0]>0))

def summarize(rows,models):
    counts={m:sum(r['correct'][m] for r in rows) for m in models}
    return dict(queries=len(rows),correct=counts,MRR={m:math.fsum(1/r['ranks'][m] for r in rows)/len(rows) for m in models},
                recall_C128=sum(r['target_in_C128'] for r in rows))

def main():
    M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    a=M.read(AUTH);ab=M.bind(AUTH)
    M.checked(a['sources']['join_program'])
    predictions=[];validations=[]
    for shard in range(75):
        folder=OUT/'predictions'/f'shard{shard:02d}'
        v=M.read(folder/'validation.json')
        M.need(v['status']=='RPC_FROZEN_HEAD_NUMPY_ACTION_PASS' and v['authority']==ab,'ALL75_QUALIFIED')
        p=M.read(M.checked(v['payload']))
        M.need(p['authority']==ab and len(p['records'])==8 and p['target_reads']==p['training_updates']==0,'SEALED_WORKER')
        predictions.extend(p['records']);validations.append(M.bind(folder/'validation.json'))
    workers=M.read(M.checked(a['sources']['worker']))['records']
    M.need([p['query_id'] for p in predictions]==[w['query_id'] for w in workers] and len(predictions)==600,'COMPLETE600_ORDER')
    M.write(OUT/'all_predictions_prejoin_seal.json',dict(authority=ab,validations=validations,queries=600))
    # First and only label join occurs after every fixed-head prediction is sealed.
    gallery=M.read(M.checked(a['sources']['gallery']))['records']
    labels=[r['identity'] for r in gallery]
    M.need([r['physical_row'] for r in gallery]==list(range(200)) and len(set(labels))==200,'GALLERY200')
    roles={r['query_id']:r for r in M.read(M.checked(a['curator_after_all_seals']))['records']}
    rows=[];independent=collections.Counter();logit_actions=0
    for p in predictions:
        role=roles[p['query_id']];order=p['raw_ranked_physical_rows']
        M.need(len(order)==200 and set(order)==set(range(200)) and p['raw_selected']==order[0],'NATURAL_FULL200')
        target=labels.index(role['identity']);raw_rank=order.index(target)+1
        chosen={'RAW':p['raw_selected']};correct={'RAW':chosen['RAW']==target};ranks={'RAW':raw_rank};holds={}
        challengers=[x for x in sorted(order[:128]) if x!=order[0]]
        for model,v in p['models'].items():
            z=[float.fromhex(h) for h in v['logits_hex']]
            M.need(len(z)==127 and all(math.isfinite(x) for x in z),'LOGIT_VECTOR127')
            k=max(range(127),key=lambda j:z[j]);selected=challengers[k] if z[k]>0 else order[0]
            M.need(selected==v['selected'],'INDEPENDENT_LOGIT_ACTION');logit_actions+=1
            chosen[model]=selected;correct[model]=labels[selected]==role['identity'];holds[model]=selected==order[0]
            ranks[model]=1 if correct[model] else raw_rank+int(order.index(selected)+1>raw_rank)
        for model,selected in chosen.items():
            independent[model]+=selected==target
            direct=[selected]+[x for x in order if x!=selected]
            M.need(direct.index(target)+1==ranks[model],'INDEPENDENT_MRR_REORDER')
        rows.append(dict(query_id=p['query_id'],identity=role['identity'],component=role['component'],camera=role['camera'],stratum=role['stratum'],target_in_C128=raw_rank<=128,correct=correct,ranks=ranks,selected=chosen,holds=holds))
    M.need(len({r['identity'] for r in rows})==200 and len({r['stratum'] for r in rows})==17,'FIXED_DATASET_UNITS')
    M.need(set(r['camera'] for r in rows)=={1,2,3},'THREE_QUERY_CAMERAS')
    models=list(rows[0]['correct']);overall=summarize(rows,models)
    M.need(dict(independent)==overall['correct'],'INDEPENDENT_COUNTS')
    pairs=[(b,'COST1') for b in ('RAW','COST4','GROUP_COST4','RAW2_CE','COST1_CBIND')]+[('COST1','CE'),('RAW','CE'),('GROUP_COST4','CE'),('CE_CBIND','CE')]
    comparisons={b+'__to__'+m:comparison(rows,b,m) for b,m in pairs}
    performance=all(comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW','COST4','GROUP_COST4'))
    mechanism=all(comparisons[b+'__to__COST1']['reliable_positive'] for b in ('RAW2_CE','COST1_CBIND'))
    by_camera={str(c):summarize([r for r in rows if r['camera']==c],models) for c in (1,2,3)}
    by_stratum={c:summarize([r for r in rows if r['stratum']==c],models) for c in sorted({r['stratum'] for r in rows})}
    by_sku={c:summarize([r for r in rows if r['identity']==c],models) for c in sorted({r['identity'] for r in rows})}
    actions={}
    for m in models:
        if m=='RAW':continue
        switched=[r for r in rows if not r['holds'][m]]
        actions[m]=dict(holds=600-len(switched),switches=len(switched),rescues_vs_RAW=sum(not r['correct']['RAW'] and r['correct'][m] for r in rows),breaks_vs_RAW=sum(r['correct']['RAW'] and not r['correct'][m] for r in rows),wrong_to_other_wrong=sum(not r['correct']['RAW'] and not r['correct'][m] for r in switched))
    result=dict(status='RPC_EXTERNAL_FROZEN600_COMPLETE',authority=ab,primary='COST1',secondary='CE',
                model_lineage='frozen full-H593 heads; zero RPC training/calibration',query_panel='600 original single-product images;200 SKU;three query cameras;17 supercategories',
                candidate_source='independent200-reference gallery;naturalC128',action='all127 challengers; SWITCH iff maxlogit>0 else HOLD',
                counts=overall['correct'],MRR=overall['MRR'],target_recall_C128=overall['recall_C128'],comparisons=comparisons,
                by_camera=by_camera,by_supercategory=by_stratum,by_sku=by_sku,action_breakdown=actions,rows=rows,
                performance_gate=performance,binding_gate=mechanism,scoped_external_new_HYP_GO=performance and mechanism,
                GO_scope='Fixed RPC single-product cross-camera retrieval only; no multi-object checkout, ownership, open-set or universal generalization claim.',
                bootstrap=dict(primary='SKU resampling within17 supercategories, three queries per SKU kept together',draws=100000,seed=20260913,sensitivity='equal17-supercategory cluster bootstrap'),
                old_EVAL32_128_retested=False,GroZi_result_unchanged=True)
    M.write(OUT/'result.json',result)
    M.write(OUT/'result_validation.json',dict(status='RPC600_ALL_SEALS_COUNTS_MRR_ACTIONS_PASS',authority=ab,result=M.bind(OUT/'result.json'),all_prediction_validations=validations,independent_logit_actions=logit_actions,query_count=600))
    print(json.dumps(dict(counts=result['counts'],performance_gate=performance,binding_gate=mechanism,scoped_external_new_HYP_GO=performance and mechanism)),flush=True)
if __name__=='__main__':main()
