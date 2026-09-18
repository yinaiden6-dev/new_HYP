#!/usr/bin/env python3
"""Independently validate exact endpoint rows, sparse proofs and all4960 coverage."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_opened_eval_strict_cardinality_capacity_v1';PRODUCER=ROOT/'programs/analyze_rc_opened_eval_strict_cardinality_capacity_v1.py'
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_sources(node):
 if isinstance(node,dict):
  if set(('path','sha256'))<=set(node):need(sha(ROOT/node['path'])==node['sha256'],'SOURCE_BINDING')
  else:
   for x in node.values():check_sources(x)

def exact_equations(rows):
 values=[];meta=[]
 for r in rows:
  cs=[int(x) for x in r['challenger_positions']];t=int(r['target_position']);w=int(r['base_winner_position']);vectors={c:[Fraction.from_float(float(v)) for v in r['real_native_features']['C_PAIRED'][i]]+[Fraction(1)] for i,c in enumerate(cs)}
  if t==w:
   for c in cs:values.append([-v for v in vectors[c]]);meta.append({'query_id':r['query_id'],'kind':'BASE_CORRECT_WRONG_BELOW_ZERO','candidate_position':c})
  else:
   values.append(vectors[t]);meta.append({'query_id':r['query_id'],'kind':'TRUE_CHALLENGER_ABOVE_HOLD','candidate_position':t})
   for c in cs:
    if c!=t:values.append([x-y for x,y in zip(vectors[t],vectors[c])]);meta.append({'query_id':r['query_id'],'kind':'TRUE_CHALLENGER_ABOVE_WRONG','candidate_position':c})
 return values,meta

def main():
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);result=read(OUT/'result.json');manifest=read(OUT/'execution_manifest.json');check_sources(result['sources']);check_sources(result['artifacts'])
 need(sha(PRODUCER)==manifest['sources']['program']['sha256'] and sha(Path(__file__))==manifest['sources']['validator']['sha256'],'PRODUCER_VALIDATOR_FROZEN')
 spec=importlib.util.spec_from_file_location('strict_cardinality_inputs',PRODUCER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 p,h,sources,rows,endpoints,correct,oldtheta,prior,unused_numeric,unused_terms,producer_meta,unused_map=m.prepare(independent=True)
 need(sources==result['sources']==manifest['sources'] and read(OUT/'source_feature_endpoints.json')=={'diagnostic_label':m.LABEL,'rows':endpoints},'SOURCE_ENDPOINTS_INDEPENDENT_REBUILD')
 a,meta=exact_equations(rows);need(len(a)==4064 and meta==producer_meta and all(len(x)==7 for x in a),'INDEPENDENT_GLOBAL_SEMANTIC_EQUATIONS')
 keys={(x['query_id'],x['kind'],x['candidate_position']):i for i,x in enumerate(meta)};need(len(keys)==4064,'UNIQUE_GLOBAL_KEYS')
 oldmin=min(sum((x*y for x,y in zip(vec,oldtheta)),Fraction(0)) for vec,info in zip(a,meta) if info['query_id'] in correct)
 need(oldmin>0 and str(oldmin)==result['baseline28_exact_positive_margin']==manifest['baseline28_exact_positive_margin'],'BASELINE28_EXACT_STRICT_WITNESS')
 order=[r['query_id'] for r in rows];raw={r['query_id'] for r in rows if int(r['base_winner_position'])==int(r['target_position'])};need(len(raw)==25,'RAW25')
 alllex=[tuple(x) for x in combinations(order,29)];need(len(alllex)==4960,'4960_SUBSETS');priority=[i for i,x in enumerate(alllex) if raw<=set(x)]+[i for i,x in enumerate(alllex) if not raw<=set(x)];need(sum(raw<=set(x) for x in alllex)==35,'35_RAW_SUBSETS')
 need(manifest['query_order']==order and manifest['RAW_correct_query_ids']==sorted(raw),'QUERY_ORDER_MANIFEST')
 pool=read(OUT/'certificate_pool.json')['certificates'];proofs={};prior_names=[]
 for name,c in pool.items():
  need(c['certificate_id']==name and c['diagnostic_label']==m.LABEL,'CERTIFICATE_SCHEMA');rr=c['rows'];weights=[Fraction(x['weight']) for x in rr];need(rr and len({x['global_constraint_index'] for x in rr})==len(rr),'UNIQUE_DUAL_ROWS');need(all(w>=0 for w in weights) and sum(weights)==1,'DUAL_SIMPLEX')
  for r in rr:
   i=r['global_constraint_index'];need(0<=i<4064 and r['constraint']==meta[i] and keys[(r['constraint']['query_id'],r['constraint']['kind'],r['constraint']['candidate_position'])]==i,'EXACT_SEMANTIC_ROW_IDENTITY')
  need(all(sum((w*a[r['global_constraint_index']][j] for w,r in zip(weights,rr)),Fraction(0))==0 for j in range(7)),'EXACT_FARKAS_CERTIFICATE')
  support={r['constraint']['query_id'] for r in rr};need(c['support_query_ids']==sorted(support),'SPARSE_SUPPORT');proofs[name]=support
  if c['origin']['type']=='previous_independently_validated_certificate':
   prior_names.append(name);q=c['origin']['query_id'];need(c['origin']['prior_result_sha256']==m.PINS['result.json'],'PRIOR_PROOF_HASH')
   expected=[{'global_constraint_index':keys[(x['constraint']['query_id'],x['constraint']['kind'],x['constraint']['candidate_position'])],'weight':x['weight'],'constraint':x['constraint']} for x in prior['systems'][q]['certificate']]
   need(rr==expected,'PRIOR_EXACT_CERTIFICATE_REUSE')
  else:need(c['origin']['type']=='new_fixed_solver_certificate','PROOF_ORIGIN')
 need(sorted(prior_names)==['PRIOR_00','PRIOR_01','PRIOR_02','PRIOR_03'],'FOUR_PRIOR_PROOFS')
 initial=sum(any(proofs[k]<=set(x) for k in prior_names) for x in alllex);initial_raw=sum(any(proofs[k]<=set(x) for k in prior_names) for x in alllex if raw<=set(x));need(initial==4365 and initial_raw==19,'INITIAL_PROOF_COVER_INDEPENDENT')
 ledger=read(OUT/'subset_coverage_ledger.json')['subsets'];need(len(ledger)==4960,'ENTIRE4960_COVERAGE_LEDGER');witness_path=OUT/'existential_certificate.json';witness=read(witness_path) if witness_path.exists() else None;witness_min=None
 if witness is not None:
  need(witness['diagnostic_label']==m.LABEL and witness['not_a_deployable_or_trained_model'] is True and witness['no_predictions_on_omitted_queries'] is True,'QUARANTINED_WITNESS')
  chosen=alllex[witness['lex_index']];need(witness['required_query_ids']==list(chosen) and witness['omitted_query_ids']==[q for q in order if q not in chosen] and witness['preserves_RAW25']==(raw<=set(chosen)),'WITNESS_SUBSET')
  theta=[Fraction(v) for v in witness['rational_unit_margin_coefficients']];need(len(theta)==7 and witness['constraint_count']==3683 and witness['parameter_count']==7,'WITNESS_DIMENSION')
  required=[row for row,info in zip(a,meta) if info['query_id'] in set(chosen)];need(len(required)==3683,'WITNESS_ALL127');witness_min=min(sum((x*y for x,y in zip(row,theta)),Fraction(0)) for row in required);need(witness_min>=1,'EXACT_FEASIBLE_UNIT_MARGINS')
 counts={'covered':0,'raw_covered':0,'witness':0,'unknown':0}
 for idx,item in enumerate(ledger):
  li=priority[idx];subset=set(alllex[li]);need(item['priority_index']==idx and item['lex_index']==li and item['omitted_query_ids']==[q for q in order if q not in subset] and item['preserves_RAW25']==(raw<=subset),'STRATIFIED_SUBSET_ORDER')
  existing=[name for name,support in proofs.items() if support<=subset];status=item['status']
  if status=='EXACT_INFEASIBLE_CERTIFICATE_COVER':
   name=item['certificate_id'];need(name in existing,'ACTUAL_CERTIFICATE_COVERS_SUBSET')
   # Membership of all semantic inequalities is verified, not merely a count.
   need(all(r['constraint']['query_id'] in subset and r['constraint']==meta[r['global_constraint_index']] for r in pool[name]['rows']),'EVERY_REUSED_INEQUALITY_INCLUDED')
   counts['covered']+=1;counts['raw_covered']+=item['preserves_RAW25']
  elif status=='EXACT_FEASIBLE_WITNESS':
   need(witness is not None and li==witness['lex_index'] and item['certificate_id'] is None and not existing,'WITNESS_NOT_CONTRADICTED');counts['witness']+=1
  else:need(status=='UNRESOLVED' and item['certificate_id'] is None and not existing,'UNRESOLVED_NOT_HIDDEN_PROOF');counts['unknown']+=1
 need(counts['witness']==int(witness is not None),'SINGLE_FIRST_WITNESS')
 search=read(OUT/'search_trace.json');trace=search['trace'];calls=search['new_solver_calls'];need(len(calls)<=64 and len(trace)<=4960,'WORK_LIMITS');elapsed=0.;discovered=set(prior_names);call_index=0;seen_witness=False
 for ti,t in enumerate(trace):
  need(t['priority_index']==ti and t['lex_index']==priority[ti] and not seen_witness,'TRACE_PREFIX_NO_AFTER_WITNESS');required=set(alllex[t['lex_index']]);cover=[k for k in discovered if proofs[k]<=required]
  if t['method']=='EXACT_CERTIFICATE_COVER':need(t['certificate_id'] in cover,'REUSE_PREVIOUSLY_KNOWN_EXACT_CERTIFICATE')
  else:
   need(t['method']=='NEW_SOLVER_CALL' and not cover and t['solver_call_index']==call_index and call_index<len(calls),'SOLVE_ONLY_NEW_UNCOVERED_SUBSET');c=calls[call_index]
   need(c['solver_call_index']==call_index and c['priority_index']==ti and c['lex_index']==t['lex_index'] and c['preserves_RAW25']==(raw<=required),'CALL_FIXED_ORDER')
   need(c['solver_seconds_before']==elapsed and elapsed<300 and c['solver_seconds']>=0,'SOFT_BUDGET_BEFORE_CALL');elapsed+=c['solver_seconds'];need(c['solver_seconds_after']==elapsed,'RECORDED_CALL_TIME_SUM')
   if c['status']=='EXACT_RATIONAL_STRICT_ACTION_INFEASIBLE':
    name=c['certificate_id'];origin=pool[name]['origin'];need(origin=={'type':'new_fixed_solver_certificate','solver_call_index':call_index,'required_subset_lex_index':t['lex_index']} and proofs[name]<=required,'NEW_PROOF_SOURCE_SUBSET');discovered.add(name)
   elif c['status']=='EXACT_RATIONAL_STRICT_ACTION_FEASIBLE':need(witness is not None and c['witness_found'] is True and witness['lex_index']==t['lex_index'],'FIRST_EXACT_FEASIBLE_CALL');seen_witness=True
   else:need(c.get('exact_certificate_valid') is False,'NUMERIC_UNRESOLVED_NOT_PROOF')
   call_index+=1
 need(call_index==len(calls) and discovered==set(pool),'ALL_NEW_PROOFS_AND_CALLS_ACCOUNTED')
 if result['stop_reason']=='FIRST_EXACT_FEASIBLE_WITNESS':need(seen_witness and witness is not None and trace[-1]['lex_index']==witness['lex_index'],'STOP_FIRST_WITNESS')
 elif result['stop_reason']=='SOLVER_WORK_BUDGET_REACHED':need(witness is None and (len(calls)>=64 or elapsed>=300),'BUDGET_STOP_NO_INFEASIBILITY_INFERENCE')
 else:need(result['stop_reason']=='ALL_REQUIRED_SUBSETS_PROCESSED' and len(trace)==4960 and witness is None,'ALL_SUBSETS_PROCESSED')
 status='STRICT_CARDINALITY_AT_LEAST29_EXACT_FEASIBLE' if witness is not None else 'STRICT_CARDINALITY_AT_LEAST29_EXACT_INFEASIBLE' if counts['covered']==4960 else 'STRICT_CARDINALITY_CAPACITY_UNRESOLVED'
 rawstatus='RAW25_PRESERVING_AT_LEAST29_EXACT_FEASIBLE' if witness is not None and witness['preserves_RAW25'] else 'RAW25_PRESERVING_AT_LEAST29_EXACT_INFEASIBLE' if counts['raw_covered']==35 else 'RAW25_PRESERVING_CAPACITY_UNRESOLVED'
 need(result['status']==status and result['strict_RAW25_preserving_status']==rawstatus and result['exact_infeasible_covered_subset_count']==counts['covered'] and result['RAW25_exact_infeasible_covered_subset_count']==counts['raw_covered'] and result['unresolved_subset_count']==counts['unknown'],'FINAL_CLASSIFICATION')
 need(result['new_solver_call_count']==len(calls) and result['solver_work_seconds']==elapsed and result['unique_exact_dual_certificate_count']==len(pool),'BUDGET_POOL_TOTALS')
 value={'status':'OPENED_EVAL_STRICT_CARDINALITY_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS','diagnostic_label':m.LABEL,'result_sha256':m.sha(OUT/'result.json'),'validator':m.binding(Path(__file__).resolve()),
 'validated_conclusion':status,'validated_RAW25_conclusion':rawstatus,'all4960_subset_assignments_independently_verified':True,'all_unique_dual_certificates_exact':True,'unique_dual_certificate_count':len(pool),
 'covered_subset_count':counts['covered'],'RAW25_covered_subset_count':counts['raw_covered'],'unresolved_subset_count':counts['unknown'],'exact_feasible_witness_verified':witness is not None,'exact_witness_minimum_margin':None if witness_min is None else str(witness_min),
 'baseline28_strict_positive_margin_verified':True,'source_native6_independently_rebuilt_from_original_scalars':True,'semantic_rows_rebuilt_as_exact_Fractions':True,'initial_old_proof_coverage_verified':[4365,19],
 'stratified_order_budget_and_first_witness_stop_verified':True,'solver_called':False,'witness_model_predictions_computed':False,'witness_used_in_training_or_deployment':False,'new_accuracy_claimed':False,'HYP_GO_claimed':False}
 m.atomic(OUT/'independent_validation.json',value);print(json.dumps({k:value[k] for k in ['status','validated_conclusion','validated_RAW25_conclusion','covered_subset_count','RAW25_covered_subset_count','unresolved_subset_count','exact_feasible_witness_verified','unique_dual_certificate_count']},sort_keys=True),flush=True)
if __name__=='__main__':main()
