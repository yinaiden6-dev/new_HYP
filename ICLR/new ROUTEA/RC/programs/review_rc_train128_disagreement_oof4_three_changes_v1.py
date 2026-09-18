#!/usr/bin/env python3
"""Post-result, three-case accounting of immutable OOF logits; no new fitting."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / 'results/rc_train128_disagreement_oof4_v1'
CACHE = ROOT / 'results/rc_train128_disagreement_features_v1'
INPUT = ROOT / 'results/rc_original7_train128_inputs_v1'
OUT = ROOT / 'results/rc_train128_disagreement_oof4_three_changes_v1'
CASES = ('OUTCOME-0389', 'OUTCOME-0087', 'DIFFICULT-0102')
MODELS = ('BASE7', 'CONSTANT1', 'CONDITIONAL4')
PINS = {OOF / 'result.json': '9fdf72f68ab3b1444c1961a9d842b4a8d9267bdfebaee30a081f7b7947a79d0d', OOF / 'validation.json': '43ccd7e30a50e52f4c4783d4d17304f9b4ec205bac1ca09674e2a82fca20d8c0', INPUT / 'validation.json': '770b9f9ce46f431c7bbe8479e320934c9d1dacdf23f3c3eea650f2812c1a77bf'}
ALLOWED = {OOF / 'result.json', OOF / 'validation.json', INPUT / 'validation.json', INPUT / 'feature_records.json', *(CACHE / n for n in ('result.json','manifest.json','validation.json','features.npz'))}
for fold in (0,1,2):
    ALLOWED.update(OOF / f'fold{fold:02d}' / n for n in ('seal.json','validation.json','parameters.json','predictions.json'))
BLOCKED = []
def audit(event,args):
    if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=Path(os.fsdecode(args[0])).resolve()
    if 'curator' in str(p).lower() or (ROOT/'results' in p.parents and p not in ALLOWED and OUT not in p.parents):
        BLOCKED.append(str(p));raise PermissionError('THREE_CASE_READ_BOUNDARY:'+str(p))
sys.addaudithook(audit)
def need(v,m):
    if not bool(v):raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):
    p=Path(b['path']);need(sha(p)==b['sha256'],'SOURCE_SHA_DRIFT:'+str(p));return p
def val(x):return {'value':float(x),'binary64':float(x).hex()}
def sym(a,b):return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
def hxvec(v):return [float(x).hex() for x in v]
def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    for p,h in PINS.items():need(sha(p)==h,'PIN_DRIFT:'+str(p))
    result, validation=read(OOF/'result.json'),read(OOF/'validation.json')
    need(validation['status']=='TRAIN128_DISAGREEMENT_OOF4_ALL_FOLDS_AND_JOIN_REPLAY_PASS' and validation['result']==bind(OOF/'result.json'),'QUALIFIED_OOF_RESULT')
    srcv=read(INPUT/'validation.json');checked(srcv['feature_records'])
    source_rows=read(INPUT/'feature_records.json')
    cr,cv,cm=(read(CACHE/n) for n in ('result.json','validation.json','manifest.json'))
    need(cr['status']=='TRAIN128_DISAGREEMENT_CACHE_READY' and cv['status']=='TRAIN128_DISAGREEMENT_CACHE_INDEPENDENT_REPLAY_PASS','QUALIFIED_CACHE')
    need(cr['validation']==bind(CACHE/'validation.json') and cr['manifest']==cv['manifest']==bind(CACHE/'manifest.json'),'CACHE_RECEIPT_CHAIN')
    need(cr['cache']==cv['cache']==cm['cache']==bind(CACHE/'features.npz') and cm['source_validation']==bind(INPUT/'validation.json'),'CACHE_SOURCE_CHAIN')
    with np.load(checked(cm['cache']),allow_pickle=False) as c:
        cache=c['features'].copy();axes=c['candidate_physical_rows'].copy();qids=c['query_ids'].copy()
    need(cache.shape==(128,128,5) and cache.dtype==np.float64,'COMPLETE_QUALIFIED_CACHE')
    byid={r['original_query_id']:r for r in result['rows']}
    source_bindings=[bind(p) for p in sorted(ALLOWED) if p.exists()]
    cases=[]
    for name in CASES:
        role=byid[name];fold=role['fold'];ordinal=role['execution_ordinal'];directory=OOF/f'fold{fold:02d}'
        fv=read(directory/'validation.json');seal=read(checked(fv['seal']))
        need(result['fold_validations'][str(fold)]==bind(directory/'validation.json') and fv['status']=='TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS','FOLD_REPLAY_BOUND')
        parameters=read(checked(seal['parameters']));preds=read(checked(seal['predictions']))
        pred=next(p for p in preds if p['query_id']==role['query_id'])
        row=source_rows[ordinal];axis=row['candidate_physical_rows'];raw=row['base_winner_position'];cs=row['challenger_positions']
        need(row['query_id']==pred['query_id']==str(qids[ordinal]) and axis==pred['candidate_physical_rows']==axes[ordinal].tolist(),'QUERY_AXIS_BINDING')
        need(row['source_image_sha256']==pred['source_image_sha256'],'QUERY_IMAGE_BINDING')
        target_rows={pred['predictions'][m]['final_physical_row'] for m in MODELS if role['correct'][m]}
        need(len(target_rows)==1 and not role['RAW_correct'],'TARGET_FROM_SEALED_CORRECT_FINAL')
        target_row=next(iter(target_rows));target=axis.index(target_row)
        x=torch.tensor([[float.fromhex(v) for v in r] for r in row['features_binary64']['REAL']],dtype=torch.float64)
        fw,aw,dw,rw,qw=cache[ordinal,raw]
        context=[];df=[]
        for j,c in enumerate(cs):
            fc,ac,dc,rc,qc=cache[ordinal,c]
            context.append([1.,-float(x[j,0]),float(dw)-float(dc),sym(rc,rw)])
            df.append(sym(fc,fw))
        context=torch.tensor(context,dtype=torch.float64);df=torch.tensor(df,dtype=torch.float64)
        w=torch.tensor([float.fromhex(v) for v in parameters['BASE7']['weight_binary64']],dtype=torch.float64)
        b=float.fromhex(parameters['BASE7']['bias_binary64'])
        base=x@w+b;all_logits={};alphas={};delta={}
        for model in MODELS:
            logits=base
            if model!='BASE7':
                beta=torch.tensor([float.fromhex(v) for v in parameters[model]['parameters_binary64']],dtype=torch.float64)
                alphas[model]=F.softplus(context[:,:len(beta)]@beta)
                delta[model]=alphas[model]*df
                logits=base+delta[model]
            need(hxvec(logits)==pred['predictions'][model]['all127_logits_binary64'],'ALL127_LOGIT_BITS:'+name+':'+model)
            scores=np.empty(128,np.float64);scores[raw]=0.
            for j,c in enumerate(cs):scores[c]=float(logits[j])
            top=min(range(128),key=lambda p:(-scores[p],axis[p]))
            # A score-zero challenger never overrides RAW under the frozen rule.
            best=min(cs,key=lambda p:(-scores[p],axis[p]));final=best if scores[best]>0 else raw
            need(final==pred['predictions'][model]['final_position'],'FINAL_ACTION_REPLAY')
            all_logits[model]=scores
        keyroles={raw:['RAW_WINNER'],target:['TARGET']};decisions={}
        for model in MODELS:
            scores=all_logits[model]
            wrong=min((p for p in range(128) if p!=target),key=lambda p:(-scores[p],axis[p]))
            wrong_c=min((p for p in cs if p!=target),key=lambda p:(-scores[p],axis[p]))
            final=pred['predictions'][model]['final_position']
            for p,tag in ((wrong,'STRONGEST_WRONG_'+model),(wrong_c,'STRONGEST_WRONG_CHALLENGER_'+model),(final,'FINAL_'+model)):
                keyroles.setdefault(p,[]).append(tag)
            decisions[model]={'action':pred['predictions'][model]['action'],'final_physical_row':axis[final],'target_logit':val(scores[target]),'target_above_zero':bool(scores[target]>0),'strongest_wrong_including_RAW_row':axis[wrong],'strongest_wrong_including_RAW_logit':val(scores[wrong]),'target_minus_strongest_wrong_including_RAW':val(scores[target]-scores[wrong]),'strongest_wrong_challenger_row':axis[wrong_c],'strongest_wrong_challenger_logit':val(scores[wrong_c]),'target_minus_strongest_wrong_challenger':val(scores[target]-scores[wrong_c]),'target_challenger_rank':int(1+sum(scores[c]>scores[target] or (scores[c]==scores[target] and axis[c]<axis[target]) for c in cs if c!=target)),'correct_from_qualified_result':role['correct'][model]}
        candidate_rows=[]
        for p in sorted(keyroles,key=lambda p:axis[p]):
            entry={'candidate_position':p,'physical_row':axis[p],'roles':keyroles[p],'evidence':{k:val(v) for k,v in zip(('F','A','D','mean_wr','mean_wq'),cache[ordinal,p],strict=True)},'logits':{m:val(all_logits[m][p]) for m in MODELS},'RAW_policy_zero':p==raw}
            if p==raw:
                entry.update({'dF':val(0.),'gate_context':None,'alpha':None,'formula_correction':None,'note':'RAW policy logit remains zero; it is not evaluated by either correction.'})
            else:
                j=cs.index(p);entry.update({'dF':val(df[j]),'gate_context':{k:val(v) for k,v in zip(('intercept','minus_xRAW','D_w_minus_D_c','symmetric_mean_wr'),context[j],strict=True)},'alpha':{m:val(alphas[m][j]) for m in ('CONSTANT1','CONDITIONAL4')},'formula_correction':{m:val(delta[m][j]) for m in ('CONSTANT1','CONDITIONAL4')},'observed_rounded_logit_change':{m:val(all_logits[m][p]-all_logits['BASE7'][p]) for m in ('CONSTANT1','CONDITIONAL4')}})
            candidate_rows.append(entry)
        if name=='OUTCOME-0389':interpretation='Target remains the highest challenger; positive content correction crosses the zero SWITCH threshold. CONSTANT1 also rescues it.'
        elif name=='OUTCOME-0087':interpretation='Target remains the highest challenger, but negative dF correction pushes it below zero. Wrong RAW is retained; CONSTANT1 also loses it.'
        else:interpretation='Target remains above zero. CONDITIONAL4 boosts a wrong challenger more than target and reverses their competition; CONSTANT1 retains target.'
        cases.append({'original_query_id':name,'opaque_query_id':role['query_id'],'fold':fold,'execution_ordinal':ordinal,'target_physical_row':target_row,'target_source':'unanimous final physical row of models marked correct in already sealed OOF result','RAW_physical_row':axis[raw],'parameters':parameters,'decisions':decisions,'key_candidates':candidate_rows,'interpretation':interpretation,'all127_logit_bit_checks':381})
    need(not BLOCKED,'UNEXPECTED_SOURCE_ACCESS')
    output={'status':'TRAIN_OOF4_THREE_CHANGED_DECISIONS_POSTRESULT_ACCOUNTING_PASS','claim_level':'THREE_SELECTED_POSTRESULT_COMPUTATIONAL_DECISION_EXPLANATIONS_ONLY','source_bindings':source_bindings,'cases':cases,'total_logit_bit_checks':1143,'new_fitting_updates':0,'EVAL_reads':0,'curator_reads':0,'threshold_or_parameter_selection':False,'feature_capacity_claim':False,'alpha_method':'actual FP64 gate_X @ sealed beta then Torch softplus; never reverse divide rounded logits','rounding_note':'Formula correction alpha*dF and subtraction of rounded output logits are recorded separately. RAW policy score is exactly zero.','non_target_semantics':'Every other position on the qualified naturally deduplicated C128 axis; target row comes only from a correct sealed decision.'}
    need(not (OUT/'result.json').exists(), 'IMMUTABLE_OUTPUT_EXISTS')
    OUT.mkdir(exist_ok=True)
    path=OUT/'result.json';path.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n');path.chmod(0o444)
    for case in cases:
        print(case['original_query_id'],case['interpretation'])
        for c in case['key_candidates']:
            print(c['physical_row'],','.join(c['roles']), 'F',c['evidence']['F']['value'],'D',c['evidence']['D']['value'],'dF',c['dF']['value'],'alpha',None if c['alpha'] is None else {k:v['value'] for k,v in c['alpha'].items()},'z',{k:v['value'] for k,v in c['logits'].items()})
        print('DECISIONS',json.dumps(case['decisions']))
    print('RESULT_SHA',sha(path))
if __name__=='__main__':main()
