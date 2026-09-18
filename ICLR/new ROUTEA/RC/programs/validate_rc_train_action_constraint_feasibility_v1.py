#!/usr/bin/env python3
"""Fresh exact-rational validation of TRAIN feasibility certificates."""
from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,json,pickle,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_train_action_constraint_feasibility_v1'

def need(x,m):
 if not bool(x):raise RuntimeError(m)
def vector(row,key,pos):
 cs=[int(p) for p in row['challenger_positions']]
 return [Fraction.from_float(float(x)) for x in row[key]['C_PAIRED'][cs.index(pos)]]+[Fraction(1)]
def main():
 value=json.loads((OUT/'result.json').read_text());p=Path(value['constraint_systems']['path'])
 need(hashlib.sha256(p.read_bytes()).hexdigest()==value['constraint_systems']['sha256'],'SYSTEM_HASH')
 with p.open('rb') as f:systems=pickle.load(f)
 source=ROOT/'programs/run_rc_absolute_evidence_scale_calibration_v1.py'
 authority=json.loads((ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json').read_text())
 need(hashlib.sha256(source.read_bytes()).hexdigest()==authority['sources']['program']['sha256'],'SOURCE_PROGRAM')
 spec=importlib.util.spec_from_file_location('verified_feature_source',source);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);u.helper();u.deadline()
 frozen,pure,pair,train,unused,entries,labels,closure=u.prepare()
 need(closure==u.H.read(ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json'),'SOURCE_FEATURES')
 tm={r['query_id']:r for r in train};pm={r['query_id']:r for r in pair['records']};checks=[]
 for model,variants in value['models'].items():
  key=u.KEYS[model][0]
  for tag,result in variants.items():
   saved=systems[model+'_'+tag];rows=[];expected_meta=[]
   sign=tag.startswith('SIGN');with_pair=tag.endswith('PLUS_PAIR')
   for r in train:
    w,t=int(r['base_winner_position']),int(r['target_position'])
    cs=[int(c) for c in r['challenger_positions']]
    if t==w:
     for c in cs:
      rows.append([-x for x in vector(r,key,c)]);expected_meta.append({'query_id':r['query_id'],'kind':'BASE_CORRECT_WRONG_BELOW_ZERO','candidate_position':c})
    else:
     target=vector(r,key,t);rows.append(target);expected_meta.append({'query_id':r['query_id'],'kind':'TRUE_CHALLENGER_ABOVE_HOLD','candidate_position':t})
     for c in cs:
      if c==t:continue
      other=vector(r,key,c);rows.append([-x for x in other] if sign else [a-b for a,b in zip(target,other)])
      expected_meta.append({'query_id':r['query_id'],'kind':'WRONG_BELOW_ZERO' if sign else 'TRUE_CHALLENGER_ABOVE_WRONG','candidate_position':c})
   if with_pair:
    for r in pair['records']:
     x=[Fraction.from_float(float(v)) for v in r[key]['C_PAIRED'][0]]+[Fraction(1)]
     rows.append(x if r['switch_label'] else [-v for v in x]);expected_meta.append({'query_id':r['query_id'],'kind':'PAIR_POSITIVE' if r['switch_label'] else 'PAIR_NEGATIVE','candidate_position':None})
   need(expected_meta==saved['metadata'] and len(rows)==result['constraint_count'],'COMPLETE_CONSTRAINT_ORDER')
   d=result['parameter_count'];need(all(len(r)==d for r in rows),'DIMENSION')
   if result['status'].endswith('_INFEASIBLE'):
    cert=result['certificate'];weights=[Fraction(c['weight']) for c in cert]
    need(all(w>=0 for w in weights) and sum(weights)==1,'NONNEGATIVE_NORMALIZED_DUAL')
    for c in cert:need(c['constraint']==expected_meta[c['constraint_index']],'CERTIFICATE_SOURCE_REFERENCE')
    need(all(sum((w*rows[c['constraint_index']][j] for w,c in zip(weights,cert)),Fraction(0))==0 for j in range(d)),'EXACT_DUAL_ZERO')
   elif result['status'].endswith('_FEASIBLE'):
    theta=[Fraction(x) for x in result['rational_unit_margin_theta']]
    need(all(sum((a*b for a,b in zip(row,theta)),Fraction(0))>=1 for row in rows),'EXACT_PRIMAL_UNIT_MARGINS')
   else:raise RuntimeError('UNRESOLVED_NUMERIC_ONLY_RESULT')
   checks.append({'model':model,'constraint_system':tag,'exact_certificate_valid':True})
 need(len(checks)==12 and u.H.BARRIER.blocked==0 and not u.H.BARRIER.released,'CLOSURE')
 u.H.atomic(OUT/'independent_validation.json',{'status':'TRAIN_CONSTRAINT_INDEPENDENT_EXACT_VALIDATION_PASS',
   'result_sha256':u.H.sha(OUT/'result.json'),'validator':u.H.binding(__file__),'systems':checks,
   'source_rows_independently_rebuilt':True,'exact_fraction_arithmetic':True,'EVAL_target_reads':0,'EVAL_scores_computed':0,
   'scope':'Certificates concern strict linear decisions on the fixed TRAIN features, not an all-model bound or a 32/32 scientific gate.'})
 print('TRAIN_CONSTRAINT_INDEPENDENT_EXACT_VALIDATION_PASS')
if __name__=='__main__':main()
