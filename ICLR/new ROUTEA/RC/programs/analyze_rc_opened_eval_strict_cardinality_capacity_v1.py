#!/usr/bin/env python3
"""Finite strict-cardinality capacity certificate cover; quarantined EVAL oracle."""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_opened_eval_strict_cardinality_capacity_v1';PLAN=ROOT/'plan/RC_OPENED_EVAL_STRICT_CARDINALITY_CAPACITY_V1_20260909.md'
VALIDATOR=ROOT/'programs/validate_rc_opened_eval_strict_cardinality_capacity_v1.py'
PRIOR=ROOT/'results/rc_opened_eval_strict_no_break_capacity_v1';SOURCE=ROOT/'programs/analyze_rc_opened_eval_strict_no_break_capacity_v1.py'
SOURCE_SHA='eae15db5a0d4aab8797891ce68e1ff618d17cee372fff89a8b760f82dc7984a1'
PINS={'result.json':'3194c5fa41c8d49d37adfb3f262446f84fbf7ebacaba6aec7b88e493b3536870','independent_validation.json':'bd803ccf5bce7edce82c4bc5a5d3f25dc6b44ced76c9197f8a9350c72a3ee8d9','source_feature_endpoints.json':'b1a722f64059ac4c816b7cf38aa15c94f889403d278d00f6347eb40a05502268'}
LABEL='NON_DEPLOYABLE_LABEL_AWARE_EVAL_DIAGNOSTIC';MAX_SOLVES=64;SOFT_SECONDS=300.
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def atomic(p,x):need(not p.exists(),'APPEND_ONLY:'+str(p));p.write_bytes(encode(x)+b'\n')
def check_sources(node):
 if isinstance(node,dict):
  if set(('path','sha256'))<=set(node):need(sha(ROOT/node['path'])==node['sha256'],'SOURCE_CHANGED')
  else:
   for value in node.values():check_sources(value)
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def prepare(independent=False):
 need(sha(SOURCE)==SOURCE_SHA,'PINNED_ENDPOINT_PRODUCER')
 for name,digest in PINS.items():need(sha(PRIOR/name)==digest,'PRIOR_EXACT_PROOF_SOURCE:'+name)
 old=read(PRIOR/'result.json');v=read(PRIOR/'independent_validation.json');need(v['result_sha256']==PINS['result.json'] and all(x['exact_certificate_valid'] for x in v['systems']),'PRIOR_EXACT_VALIDATION')
 p=load(SOURCE,'cardinality_original_endpoint_source');base_sources,rows,endpoints,correct,theta=p.inputs(independent=independent)
 need(base_sources==old['sources'] and read(PRIOR/'source_feature_endpoints.json')=={'diagnostic_label':LABEL,'rows':endpoints},'EXACT_PRIOR_ENDPOINT_REPLAY')
 helper=p.load(ROOT/p.PINS['certificate_helper'][0],'cardinality_frozen_certificate_search')
 # Numeric LP rows may be rounded differences; exact proof always starts from
 # endpoint terms or fresh rational endpoints, never from these numeric rows.
 a,terms,metadata=helper.build({'records':[]},rows,'real_native_features',False,False);need(a.shape==(4064,7),'ALL32_ORIGINAL_ACTION_CONSTRAINTS')
 keys=[(x['query_id'],x['kind'],x['candidate_position']) for x in metadata];need(len(set(keys))==4064,'SEMANTIC_KEY_UNIQUENESS')
 sources={'endpoint_source':binding(SOURCE),'prior_artifacts':{k:binding(PRIOR/k) for k in PINS},'original_endpoint_sources':base_sources,
 'program':binding(Path(__file__).resolve()),'validator':binding(VALIDATOR),'plan':binding(PLAN)}
 return p,helper,sources,rows,endpoints,correct,theta,old,a,terms,metadata,dict(zip(keys,range(len(keys))))

def all_subsets(rows):
 order=[r['query_id'] for r in rows];raw={r['query_id'] for r in rows if int(r['target_position'])==int(r['base_winner_position'])};need(len(raw)==25 and len(order)==32,'RAW25_ALL32')
 lex=[]
 for n,combo in enumerate(combinations(order,29)):
  required=set(combo);lex.append({'lex_index':n,'required_query_ids':list(combo),'omitted_query_ids':[q for q in order if q not in required],'preserves_RAW25':raw<=required})
 need(len(lex)==4960 and sum(x['preserves_RAW25'] for x in lex)==35,'COMBINATORIAL_COUNTS')
 ordered=[x for x in lex if x['preserves_RAW25']]+[x for x in lex if not x['preserves_RAW25']]
 for n,item in enumerate(ordered):item['priority_index']=n
 return ordered,order,sorted(raw)

def verify_dual(cert,exact_rows,metadata):
 rows=cert['rows'];weights=[Fraction(x['weight']) for x in rows];need(rows and len({x['global_constraint_index'] for x in rows})==len(rows),'DUAL_SUPPORT_UNIQUE')
 need(all(w>=0 for w in weights) and sum(weights)==1,'DUAL_NONNEGATIVE_NORMALIZED')
 for r in rows:need(r['constraint']==metadata[r['global_constraint_index']],'SEMANTIC_ROW_BINDING')
 need(all(sum((w*exact_rows(r['global_constraint_index'])[j] for w,r in zip(weights,rows)),Fraction(0))==0 for j in range(7)),'FARKAS_EXACT_ZERO')
 need(cert['support_query_ids']==sorted({r['constraint']['query_id'] for r in rows}),'EXACT_SUPPORT_QUERIES')

def covering(pool,required):
 s=set(required)
 for name,c in pool.items():
  if set(c['support_query_ids'])<=s:return name
 return None

def make_dual(name,rows,origin):
 return {'certificate_id':name,'diagnostic_label':LABEL,'origin':origin,'rows':rows,'support_query_ids':sorted({x['constraint']['query_id'] for x in rows}),
 'semantics':'Nonnegative rational weights sum to1; weighted exact seven-dimensional inequality rows sum tozero, contradicting unit right-hand sides.'}

def main():
 import torch,scipy,numpy as np
 torch.set_num_threads(8);torch.set_num_interop_threads(1);need(not OUT.exists(),'APPEND_ONLY_OUTPUT')
 p,h,sources,rows,endpoints,correct,theta,old,a,terms,meta,keymap=prepare();subsets,order,raw=all_subsets(rows);exact_cache={}
 def exact(i):
  if i not in exact_cache:exact_cache[i]=h.row_exact(terms[i])
  return exact_cache[i]
 pool={}
 for n,q in enumerate(p.ERRORS):
  c=old['systems'][q];need(c['status']=='EXACT_RATIONAL_STRICT_ACTION_INFEASIBLE','PRIOR_CERTIFICATE_NOT_EXACT')
  rr=[{'global_constraint_index':keymap[(x['constraint']['query_id'],x['constraint']['kind'],x['constraint']['candidate_position'])],
       'weight':x['weight'],'constraint':x['constraint']} for x in c['certificate']]
  name=f'PRIOR_{n:02d}';pool[name]=make_dual(name,rr,{'type':'previous_independently_validated_certificate','query_id':q,'prior_result_sha256':PINS['result.json']});verify_dual(pool[name],exact,meta)
 initial={x['lex_index']:covering(pool,x['required_query_ids']) for x in subsets};need(sum(v is not None for v in initial.values())==4365 and sum(initial[x['lex_index']] is not None for x in subsets if x['preserves_RAW25'])==19,'INITIAL_EXACT_COVER_COUNTS')
 baseline_rows=[i for i,m in enumerate(meta) if m['query_id'] in correct];oldmin=min(sum((v*w for v,w in zip(exact(i),theta)),Fraction(0)) for i in baseline_rows);need(oldmin>0 and str(oldmin)==old['baseline_28_exact_min_margin'],'OLD28_EXACT_LOWER_BOUND')
 manifest={'sources':sources,'diagnostic_label':LABEL,'query_order':order,'RAW_correct_query_ids':raw,'all29_subset_count':4960,'RAW25_preserving_subset_count':35,
 'subset_order':'RAW25-preserving35 then remaining4925; each in execution-ordered lexicographic required29 tuple order','parameter_count':7,'constraints_per_required_subset':3683,
 'maximum_new_solver_calls':MAX_SOLVES,'soft_solver_seconds':SOFT_SECONDS,'initial_exact_certificate_count':4,'initial_exact_covered_subsets':4365,'initial_RAW25_exact_covered_subsets':19,
 'strict_cardinality_claim_excludes_exact_ties_zero_boundaries_and_rounding_special_cases':True,'baseline28_exact_positive_margin':str(oldmin),
 'no_new_model_or_checkpoint_or_witness_prediction':True}
 OUT.mkdir();atomic(OUT/'execution_manifest.json',manifest);atomic(OUT/'source_feature_endpoints.json',{'diagnostic_label':LABEL,'rows':endpoints});trace=[];calls=[];witness=None;elapsed=0.;stop='ALL_REQUIRED_SUBSETS_PROCESSED'
 for subset in subsets:
  hit=covering(pool,subset['required_query_ids'])
  if hit:
   trace.append({'priority_index':subset['priority_index'],'lex_index':subset['lex_index'],'method':'EXACT_CERTIFICATE_COVER','certificate_id':hit});continue
  if len(calls)>=MAX_SOLVES or elapsed>=SOFT_SECONDS:stop='SOLVER_WORK_BUDGET_REACHED';break
  chosen=set(subset['required_query_ids']);ix=[i for i,x in enumerate(meta) if x['query_id'] in chosen];need(len(ix)==3683,'SUBSET_ALL_SEMANTIC_ROWS');tt=[terms[i] for i in ix];mm=[meta[i] for i in ix]
  before=elapsed;started=time.monotonic()
  try:answer=h.certify(a[ix],tt,mm)
  except RuntimeError as err:answer={'status':'NUMERIC_SEARCH_UNRESOLVED','exact_constraint_validation':False,'reason':str(err)}
  duration=time.monotonic()-started;elapsed+=duration;record={'solver_call_index':len(calls),'priority_index':subset['priority_index'],'lex_index':subset['lex_index'],
   'solver_seconds_before':before,'solver_seconds':duration,'solver_seconds_after':elapsed,'status':answer['status'],'preserves_RAW25':subset['preserves_RAW25']}
  if answer['status']=='EXACT_RATIONAL_STRICT_ACTION_INFEASIBLE':
   name=f'NEW_{len(calls):02d}';rr=[{'global_constraint_index':ix[x['constraint_index']],'weight':x['weight'],'constraint':x['constraint']} for x in answer['certificate']]
   pool[name]=make_dual(name,rr,{'type':'new_fixed_solver_certificate','solver_call_index':len(calls),'required_subset_lex_index':subset['lex_index']});verify_dual(pool[name],exact,meta);record['certificate_id']=name
  elif answer['status']=='EXACT_RATIONAL_STRICT_ACTION_FEASIBLE':
   th=[Fraction(x) for x in answer['rational_unit_margin_theta']];need(len(th)==7 and all(sum((x*y for x,y in zip(exact(i),th)),Fraction(0))>=1 for i in ix),'EXACT_EXISTENTIAL_WITNESS')
   witness={'diagnostic_label':LABEL,'priority_index':subset['priority_index'],'lex_index':subset['lex_index'],'required_query_ids':subset['required_query_ids'],'omitted_query_ids':subset['omitted_query_ids'],
    'preserves_RAW25':subset['preserves_RAW25'],'rational_unit_margin_coefficients':answer['rational_unit_margin_theta'],'constraint_count':3683,'parameter_count':7,
    'not_a_deployable_or_trained_model':True,'no_predictions_on_omitted_queries':True};record['witness_found']=True;stop='FIRST_EXACT_FEASIBLE_WITNESS'
  else:
   record['unresolved_reason']=answer.get('reason',answer.get('solver_message','No exact certificate'));record['exact_certificate_valid']=False
  calls.append(record);trace.append({'priority_index':subset['priority_index'],'lex_index':subset['lex_index'],'method':'NEW_SOLVER_CALL','solver_call_index':record['solver_call_index']})
  print(json.dumps({'event':'STRICT_CARDINALITY_NEW_SOLVE','call':len(calls),'status':answer['status'],'RAW25':subset['preserves_RAW25'],'solver_seconds_total':elapsed,'dual_pool_size':len(pool)}),flush=True)
  if witness is not None:break
 # Complete proof bookkeeping uses only already verified certificates; it does
 # not issue additional solves after budget/first-witness termination.
 ledger=[]
 for s in subsets:
  hit=covering(pool,s['required_query_ids']);is_witness=witness is not None and s['lex_index']==witness['lex_index'];need(not(is_witness and hit),'EXACT_FEASIBLE_INFEASIBLE_CONTRADICTION')
  ledger.append({'priority_index':s['priority_index'],'lex_index':s['lex_index'],'omitted_query_ids':s['omitted_query_ids'],'preserves_RAW25':s['preserves_RAW25'],
   'status':'EXACT_FEASIBLE_WITNESS' if is_witness else 'EXACT_INFEASIBLE_CERTIFICATE_COVER' if hit else 'UNRESOLVED','certificate_id':hit})
 covered=sum(x['status']=='EXACT_INFEASIBLE_CERTIFICATE_COVER' for x in ledger);rawcovered=sum(x['status']=='EXACT_INFEASIBLE_CERTIFICATE_COVER' for x in ledger if x['preserves_RAW25'])
 status='STRICT_CARDINALITY_AT_LEAST29_EXACT_FEASIBLE' if witness is not None else 'STRICT_CARDINALITY_AT_LEAST29_EXACT_INFEASIBLE' if covered==4960 else 'STRICT_CARDINALITY_CAPACITY_UNRESOLVED'
 rawstatus='RAW25_PRESERVING_AT_LEAST29_EXACT_FEASIBLE' if witness is not None and witness['preserves_RAW25'] else 'RAW25_PRESERVING_AT_LEAST29_EXACT_INFEASIBLE' if rawcovered==35 else 'RAW25_PRESERVING_CAPACITY_UNRESOLVED'
 atomic(OUT/'certificate_pool.json',{'diagnostic_label':LABEL,'certificates':pool});atomic(OUT/'subset_coverage_ledger.json',{'diagnostic_label':LABEL,'subsets':ledger});atomic(OUT/'search_trace.json',{'trace':trace,'new_solver_calls':calls})
 if witness is not None:atomic(OUT/'existential_certificate.json',witness)
 check_sources(sources)
 result={'status':status,'diagnostic_label':LABEL,'theory_name':'new HYP','strict_RAW25_preserving_status':rawstatus,'stop_reason':stop,'sources':sources,
 'artifacts':{n:binding(OUT/n) for n in ['execution_manifest.json','source_feature_endpoints.json','certificate_pool.json','subset_coverage_ledger.json','search_trace.json']},
 'head_class':'Real linear readout of original frozen native6 plus intercept; strict positive action margins','dataset':'previously opened matched EVAL32','candidate_source':'original RAW C128 with127 challenger comparisons',
 'required_subset_count':4960,'exact_infeasible_covered_subset_count':covered,'RAW25_preserving_subset_count':35,'RAW25_exact_infeasible_covered_subset_count':rawcovered,
 'unresolved_subset_count':sum(x['status']=='UNRESOLVED' for x in ledger),'exact_feasible_witness_found':witness is not None,'witness_preserves_RAW25':None if witness is None else witness['preserves_RAW25'],
 'new_solver_call_count':len(calls),'solver_work_seconds':elapsed,'maximum_new_solver_calls':MAX_SOLVES,'soft_solver_work_budget_seconds':SOFT_SECONDS,'last_call_may_overshoot_soft_budget':True,
 'unique_exact_dual_certificate_count':len(pool),'baseline28_exact_positive_margin':str(oldmin),'all32_original_actions_and4064_original_logits_replayed':True,
 'solver':{'scipy':scipy.__version__,'numpy':np.__version__,'method':'unchanged prior HiGHS primal/dual LP search; exact rational certification'},
 'new_model_accuracy':None,'witness_scored_on_omitted_queries':False,'new_trainable_or_deployable_checkpoint_created':False,'original_head_modified':False,'threshold_scan_count':0,'new_encoder_or_RoMa_forward_count':0,'scheduler_calls':0,'HYP_GO_claimed':False,
 'interpretation':['All4960 required29 sets allow the original28 correctness set to change; RAW25-preserving35 is separately reported, not conflated with general cardinality.',
 'A feasible certificate proves only existence of at least29 strict correct decisions in this fixed finite-sample real-linear class, not a learned29/32 result or the maximum.',
 'Only complete exact certificate coverage of all4960 excludes strict29-or-more; together with the old28 exact witness this identifies strict class maximum28.',
 'Unresolved budget or numeric cases are not infeasibility; no automatic budget extension is authorized.',
 'Exact ties, zero-boundary decisions, special floating rounding effects, different features/nonlinear models/newdata are outside scope.',
 'EVAL-label-aware mathematical coefficients are quarantined and cannot feed training, thresholds, subset selection, or deployment.']}
 if witness is not None:result['artifacts']['existential_certificate.json']=binding(OUT/'existential_certificate.json')
 atomic(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ['status','strict_RAW25_preserving_status','stop_reason','exact_infeasible_covered_subset_count','RAW25_exact_infeasible_covered_subset_count','new_solver_call_count','solver_work_seconds','unique_exact_dual_certificate_count']},sort_keys=True),flush=True)
if __name__=='__main__':main()
