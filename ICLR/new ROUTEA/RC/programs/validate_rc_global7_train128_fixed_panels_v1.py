#!/usr/bin/env python3
"""Independently verify all sealed fixed-panel scores before opening EVAL labels."""
from collections import defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_global7_train128_fixed_panels_v1 as G


def comparison(rows,base,new):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(r)
    diffs=[Fraction(sum(int(r['correct'][new])-int(r['correct'][base]) for r in members),len(members)) for _,members in sorted(groups.items())]
    values=np.asarray([float(v) for v in diffs]);indices=np.random.default_rng(20260912).integers(0,len(values),(10000,len(values)))
    rescue=[r['original_query_id'] for r in rows if r['correct'][new] and not r['correct'][base]]
    loss=[r['original_query_id'] for r in rows if not r['correct'][new] and r['correct'][base]]
    return dict(rescue=len(rescue),loss=len(loss),net=len(rescue)-len(loss),rescue_query_ids=rescue,loss_query_ids=loss,positive_components=sum(v>0 for v in diffs),negative_components=sum(v<0 for v in diffs),equal_component_difference=float(values.mean()),component_bootstrap_95=np.quantile(values[indices].mean(1),[.025,.975]).tolist(),exact_component_signflip=G.P.exact_group_signflip(diffs))


def main():
    a=G.guard('validate');G.need(a['validator']==G.bind(__file__),'FROZEN_VALIDATOR')
    fitval=G.read(G.OUT/'fit_validation.json');fit=G.read(G.checked(fitval['fit']))
    G.need(fitval['status']=='GLOBAL7_TRAIN128_FRESH_PARAMETER_REPLAY_PASS' and fitval['authority']==fit['authority']==G.bind(G.AUTH),'FIT_VALIDATION')
    G.need(fit['train_count']==128 and fit['new_residual_training_updates']==2000 and fit['original_head_training_updates']==0,'EXACT_TRAIN_SCOPE')
    G.need(fit['function_source']==a['screened_helper'] and fit['train_roles']==G.bind(G.OUT/'train_roles.json'),'SCREENED_FUNCTION_AND_LABELS')
    seal=G.read(G.OUT/'prediction_seal.json');payload=torch.load(G.checked(seal['payload']),weights_only=True,map_location='cpu')
    G.need(seal['status']=='GLOBAL7_TRAIN128_ALL160_REAL_CBIND_PRELABEL_SEALED' and seal['authority']==payload['authority']==G.bind(G.AUTH),'PREDICTION_SEAL')
    for key,path in [('fit',G.OUT/'fit.json'),('fit_validation',G.OUT/'fit_validation.json')]:
        G.need(seal[key]==payload[key]==G.bind(path),'SEALED_FIT_SOURCE')
    theta=np.asarray([float.fromhex(v) for v in fit['base_theta_binary64']]);delta=np.asarray([float.fromhex(v) for v in fit['delta_binary64']])
    G.need(np.array_equal(theta,G.head().numpy()),'FROZEN_EC7_BASE')
    par=payload['parameters'][G.MODEL]
    G.need(np.array_equal(par['base_theta'].numpy(),theta) and np.array_equal(par['delta'].numpy(),delta),'PREDICTION_PARAMETERS')
    previous,previous_seal=G.previous_predictions();oldpred={r['query_id']:r for r in previous['predictions']}
    G.need(payload['previous_payload']==previous_seal['payload'],'PREVIOUS_PREDICTIONS_BINDING')
    for model in G.BASELINES:G.need(G.E.same(payload['parameters'][model],previous['parameters'][model]),'BASELINE_PARAMETER_BITS')
    features,sources=G.E.features();fi={r['query_id']:r for r in features}
    G.need(sources==payload['feature_sources']==previous['feature_sources'],'FROZEN_FEATURE_SOURCE_PARITY')
    predictions={r['query_id']:r for r in payload['predictions']}
    G.need(len(predictions)==160 and predictions.keys()==oldpred.keys(),'EXACT_FIXED160')
    chosen={};maxerr=0.;checks=0
    for q,p in predictions.items():
        r=fi[q];prior=oldpred[q]
        keys=('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')
        G.need(all(p[k]==r[k]==prior[k] for k in keys),'FROZEN_AXIS_PARITY')
        axis=r['candidate_physical_rows'];raw=r['raw_ranked_physical_rows'];cs=r['challenger_positions']
        G.need(axis==sorted(set(axis)) and len(axis)==128 and set(axis)==set(raw[:128]),'NATURAL_RAW_C128')
        G.need(cs==[i for i in range(128) if i!=r['winner']] and axis[r['winner']]==raw[0],'FULL127_PHYSICAL_ORDER')
        selected={'RAW':raw[0]}
        for model in (*G.BASELINES,G.MODEL):
            for mode in ('REAL','CBIND'):
                x=r['modes'][mode]['X'].numpy();observed=p['models'][model][mode]
                if model==G.MODEL:z=x@theta[:6]+theta[6]+np.concatenate((x,np.ones((127,1))),axis=1)@delta
                else:
                    b=payload['parameters'][model]['theta'].numpy();z=x@b[:6]+b[6]
                    G.need(G.E.same(observed,prior['models'][model][mode]),'ORIGINAL_AND_LISTWISE_PREDICTIONS_BIT_EXACT')
                err=float(np.max(abs(z-observed['logits'].numpy())));G.need(np.isfinite(z).all() and err<1e-12,'INDEPENDENT_NUMPY_LOGITS')
                j=int(np.argmax(z));pos=cs[j] if z[j]>0 else r['winner'];G.need(pos==observed['selected_position'],'EXACT_INDEPENDENT_ACTION')
                selected[model if mode=='REAL' else model+'_CBIND']=axis[pos];maxerr=max(maxerr,err);checks+=127
        chosen[q]=selected
    G.need(checks==121920,'ALL121920_LOGITS_BEFORE_LABELS')
    G.write(G.OUT/'independent_prelabel_seal.json',dict(status='GLOBAL7_TRAIN128_ALL160_NUMPY_ACTIONS_BEFORE_LABELS_PASS',prediction_seal=G.bind(G.OUT/'prediction_seal.json'),logit_checks=checks,max_abs_error=maxerr,authority=G.bind(G.AUTH),evaluation_label_reads=0))

    G.LABELS_OPEN=True
    for binding in a['evaluation_sources'].values():G.checked(binding)
    roles={r['query_id']:r for r in G.read(a['evaluation_sources']['eval_roles']['path'])['records']}
    panels=G.read(a['evaluation_sources']['panel_manifest']['path'])['panels']
    old=G.read(a['evaluation_sources']['previous_result']['path']);oldval=G.read(a['evaluation_sources']['previous_validation']['path'])
    G.need(oldval['result']==a['evaluation_sources']['previous_result'],'PREVIOUS_OUTCOME_VALIDATION')
    train=G.read(G.OUT/'train_roles.json')['records'];labels,mapping=G.P.gallery_labels();outpanels={};overlaps={}
    G.need(mapping==fit['gallery_mapping_sha256'],'IDENTITY_MAPPING_PARITY')
    G.need(set(roles)==set(predictions)=={q for qs in panels.values() for q in qs},'EXACT_PANEL_LABEL_POPULATION')
    for name,qs in panels.items():
        G.need(len(qs)=={'EVAL32':32,'EVAL128':128}[name],'FROZEN_PANEL_SIZE')
        overlaps[name]={}
        for tkey,ekey in [('canonical_query_id','query_id'),('original_query_id','original_query_id'),('source_image_sha256','source_image_sha256'),('identity','identity'),('group','group'),('component','component')]:
            shared={r[tkey] for r in train}&{roles[q][ekey] for q in qs};G.need(not shared,'TRAIN_EVAL_DISJOINT:'+tkey);overlaps[name][tkey]=len(shared)
        oldrows={r['query_id']:r for r in old['panels'][name]['rows']};rows=[]
        for q in qs:
            role=roles[q];r=fi[q];raw=r['raw_ranked_physical_rows'];selection=chosen[q]
            G.need(role['execution_ordinal']==r['execution_ordinal'] and role['source_image_sha256']==r['source_image_sha256'],'ROLE_IMAGE_JOIN')
            targets=[p for p in raw if labels[p]==role['identity']];G.need(len(targets)==1,'UNIQUE_TARGET_FULL_GALLERY')
            target=targets[0];rank=raw.index(target)+1
            correct={m:labels[p]==role['identity'] for m,p in selection.items()}
            ranks={m:1 if correct[m] else rank+int(raw.index(p)+1>rank) for m,p in selection.items()}
            row=dict(query_id=q,original_query_id=role['original_query_id'],group=role['group'],component=role['component'],target_in_C128=target in r['candidate_physical_rows'],correct=correct,ranks=ranks,selected_physical_rows=selection)
            for key in ('correct','ranks','selected_physical_rows'):
                for model in ('RAW',*G.BASELINES,*(m+'_CBIND' for m in G.BASELINES)):
                    G.need(row[key][model]==oldrows[q][key][model],'BASELINE_PERQUERY_OUTCOME_PARITY')
            rows.append(row)
        scores={}
        for model in rows[0]['correct']:
            count=sum(r['correct'][model] for r in rows);mrr=sum((Fraction(1,r['ranks'][model]) for r in rows),Fraction(0))/len(rows)
            scores[model]=dict(correct=count,queries=len(rows),accuracy=count/len(rows),MRR=float(mrr),MRR_fraction=str(mrr))
        comparisons={base+'__to__'+G.MODEL:comparison(rows,base,G.MODEL) for base in G.BASELINES}
        for base in G.BASELINES:
            c=comparisons[base+'__to__'+G.MODEL];G.need(c['net']==scores[G.MODEL]['correct']-scores[base]['correct'],'PAIRED_NET_COUNT')
        rescues=[r for r in rows if r['correct'][G.MODEL] and not r['correct']['RAW']]
        outpanels[name]=dict(population=len(rows),recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons=comparisons,rows=rows,C_BIND=dict(REAL_rescues_vs_RAW=len(rescues),retained_rescues=sum(r['correct'][G.MODEL+'_CBIND'] for r in rescues)))
    result=dict(status='GLOBAL7_TRAIN128_FIXED_PANELS_DEVELOPMENT_COMPLETE',authority=G.bind(G.AUTH),fit=G.bind(G.OUT/'fit.json'),fit_validation=G.bind(G.OUT/'fit_validation.json'),prediction_seal=G.bind(G.OUT/'prediction_seal.json'),independent_prelabel_seal=G.bind(G.OUT/'independent_prelabel_seal.json'),training_population=dict(images=128,identities=32,source_groups=32,supervision='FULL C128 retrieval labels'),candidate_source='unchanged RAW full-gallery C128',production_arithmetic=fit['production_arithmetic'],optimizer=fit['optimizer'],panels=outpanels,train_eval_overlaps=overlaps,original_head_training_updates=0,old_model_predictions_bit_exact=True,independent_logit_checks=checks,max_abs_numpy_error=maxerr,observed_net_vs_original_by_panel={n:p['comparisons']['ORIGINAL7__to__'+G.MODEL]['net']>0 for n,p in outpanels.items()},observed_net_vs_listwise_by_panel={n:p['comparisons']['LISTWISE_UNIT1__to__'+G.MODEL]['net']>0 for n,p in outpanels.items()},HYP_GO_claimed=False,external_confirmation=False,deployment_changed=False,OOF114_not_relabelled_as_EVAL=True)
    G.write(G.OUT/'result.json',result)
    G.write(G.OUT/'result_validation.json',dict(status='GLOBAL7_TRAIN128_INDEPENDENT_FIXED_PANEL_ACTION_RANK_COUNTS_PASS',result=G.bind(G.OUT/'result.json'),authority=G.bind(G.AUTH),logit_checks=checks,predictions_before_labels=True,baseline_per_query_parity=True,train_eval_overlaps=overlaps))
    lines=['# GLOBAL7统一微调：原固定32/128面板结果','','原各折TRAIN OOF 114/128是先前开发结果；本页评价最终全TRAIN128残差。原ec7冻结、七维残差按固定2000步配方拟合。RAW C128及完整127 challenger HOLD/SWITCH不变。','', '| 面板 | RAW | 原ec7 | LISTWISE_UNIT1 | GLOBAL7_T128 | 对原头救/损/净 | 对LISTWISE救/损/净 |','|---|---:|---:|---:|---:|---|---|']
    for n in ('EVAL32','EVAL128'):
        p=outpanels[n];s=p['scores'];c=p['comparisons'];o=c['ORIGINAL7__to__'+G.MODEL];l=c['LISTWISE_UNIT1__to__'+G.MODEL]
        lines.append(f"| {n} | {s['RAW']['correct']} | {s['ORIGINAL7']['correct']} | {s['LISTWISE_UNIT1']['correct']} | {s[G.MODEL]['correct']} | {o['rescue']}/{o['loss']}/{o['net']:+d} | {l['rescue']}/{l['loss']}/{l['net']:+d} |")
    lines+=['','允许损失，按每面板rescue-loss报告观察净增，来源组不确定性及CBIND结果保存在机器结果。两套已打开开发面板不是外部确认。没有改写原114 TRAIN留出、28/32或99/128原模型成绩，没有自动替换部署或宣称new HYP GO。','','[机器结果](result.json) · [独立核算](result_validation.json)']
    G.write_bytes(G.OUT/'report.md',('\n'.join(lines)+'\n').encode())
    brief={n:dict(scores=p['scores'],comparisons={k:{f:v[f] for f in ('rescue','loss','net')} for k,v in p['comparisons'].items()}) for n,p in outpanels.items()}
    print(json.dumps(dict(status=result['status'],panels=brief)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(8);torch.set_num_interop_threads(1);main()
