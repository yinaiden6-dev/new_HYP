#!/usr/bin/env python3
"""Independent parallel exact certificate check; no LP or role access."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'programs/prepare_rc_reference_support_maxmin_cache_v1.py'
M=None

def initialize():
 global M
 if M is not None:return
 spec=importlib.util.spec_from_file_location('frozen_full_maxmin_cache_driver',DRIVER);M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M);M.bootstrap()

def validate_episode(payload):
 initialize();desc,entry=payload;started=time.monotonic();record=M.read(M.path_of(entry));P,V=M.P,M.V
 for key in ('kind','query_id','execution_ordinal','candidate_physical_rows','score_positions','base_winner_position','challenger_positions','fixed_J_binary64','query_token_count','query_tokens_sha256','expected_all128_a_sha256','source'):
  M.need(record[key]==desc[key],'RECORD_SOURCE_DESCRIPTOR:'+key)
 if desc['kind']=='FULL':M.need(record['cbind_source_positions']==desc['cbind_source_positions'],'CBIND_DONOR_AXIS')
 else:
  for key in ('pair_cohort','pair_row_ordinal','switch_label'):M.need(record[key]==desc[key],'PAIR_ORIGINAL_INTERFACE:'+key)
 a,w,profile,oldfp=M.profiles(desc,produce=False);M.need(record['profile_source']==profile,'PERSISTED_PROFILE_HASH')
 cert=M.read(M.path_of(record['certificates']))
 M.need(cert['profile_source']==profile and cert['operator_source']==M.bind(M.PINS['pilot_operator'][0]),'CERTIFICATE_SOURCE')
 M.need((cert['kind'],cert['query_id'],cert['execution_ordinal'])==(desc['kind'],desc['query_id'],desc['execution_ordinal']),'CERTIFICATE_QUERY_KEY')
 games=cert['games'];n=desc['query_token_count'];positions=[r['candidate_position'] for r in games]
 M.need(positions==desc['score_positions'][:len(games)] and len(games)<=len(desc['score_positions']),'ALL_ORIGINAL_GAME_AXIS_PREFIX')
 matrix=V.EndpointFractions(a);expected_T={};reused=[];qualified=True
 for local,r in enumerate(games):
  g=desc['score_positions'][local]
  M.need(r['query_id']==desc['query_id'] and r['execution_ordinal']==desc['execution_ordinal'] and r['physical_row']==desc['candidate_physical_rows'][g],'CANDIDATE_SOURCE_POSITION')
  M.need(r['opponent_positions']==[h for h in range(128) if h!=g],'FULL127_OPPONENTS')
  if desc['reuse_pilot']:
   M.need(r==M.pilot_records()[(desc['execution_ordinal'],g)],'PILOT_REUSE_EXACT_RECORD');reused.append(g)
  distributions=[];errors=[]
  for call,dim in ((r['primal_solver'],n),(r['dual_solver'],127)):
   M.need(call.get('attempted') is True and call['seconds']>=0,'RAW_LP_ATTEMPT_RECORD')
   raw=call.get('raw_solution_binary64')
   if raw is not None:M.need(len(raw)==dim+1,'RAW_SOLUTION_GENERIC_LENGTH')
   d,error=V.independent_distribution(P,None if raw is None else raw[:-1],dim);distributions.append(d);errors.append(error)
  p,alpha=distributions;pe,ae=errors
  M.need(r['primal_distribution']==p and r['dual_distribution']==alpha and r['distribution_errors']=={'primal':pe,'dual':ae},'EXACT_RAW_SOLUTION_PROBABILITIES')
  original,oe=V.independent_distribution(P,w[local].tolist(),n);M.need(oe in(None,'NO_POSITIVE_COEFFICIENT_MASS'),'ORIGINAL_WQ_FINITE')
  baseline=None if original is None else P.frac(matrix.own_minus_best_other(g,original))
  M.need(r['original_support_distribution']==original and r['original_exact_support_margin']==baseline,'EXACT_FIXED_SUPPORT_BASELINE')
  M.need(r['old_FP64']==oldfp[g] and oldfp[g]['J_binary64']==desc['fixed_J_binary64'][str(g)],'FROZEN_FIXED_J_BITS')
  expected=None if pe or ae else V.expected_certificate(P,matrix,g,p,alpha)
  M.need(expected==r['certificate'],'INDEPENDENT_ORIGINAL_ENDPOINT_CERTIFICATE')
  if expected is not None:
   lower=Fraction(int(expected['lower']['numerator']),int(expected['lower']['denominator']))
   expected_T[str(g)]=float(lower).hex()
   if baseline is not None:M.need(P.unfrac(baseline)<=P.unfrac(expected['upper']),'FIXED_SIMPLEX_VS_DUAL')
  qualified=qualified and expected is not None and expected['gap_qualified']
 qualified=qualified and len(games)==len(desc['score_positions'])
 M.need(record['T_binary64']==expected_T,'T_NEAREST_BINARY64_OF_EXACT_LOWER_L')
 M.need(record['all_game_gap_qualified']==cert['all_game_gap_qualified']==qualified,'QUERY_QUALIFICATION')
 M.need(cert['pilot_reused_positions']==reused and record['pilot_reused_games']==len(reused),'PILOT_REUSE_COUNT')
 M.need(cert['pilot_source']==(M.bind(M.PILOT/'prejoin_seal.json') if reused else None),'PILOT_SOURCE_BINDING')
 M.need(record['completed_games']==len(games) and record['new_games']==len(games)-len(reused),'QUERY_GAME_COUNT')
 M.need(not M._BLOCKED,'FORBIDDEN_READ_ATTEMPT')
 return {'kind':desc['kind'],'query_id':desc['query_id'],'execution_ordinal':desc['execution_ordinal'],'games':len(games),
  'pilot_reuses':len(reused),'new_games':len(games)-len(reused),'qualified':qualified,'seconds':time.monotonic()-started}

def validate():
 initialize();M.need(not(M.OUT/'validation.json').exists(),'APPEND_ONLY_VALIDATION_EXISTS');auth,pf=M.authority();ds,counts=M.inventory()
 M.need(ds==pf['inventory'] and counts==pf['counts'],'SOURCE_INVENTORY_FROZEN')
 result=M.read(M.OUT/'result.json');manifest=M.read(M.OUT/'manifest.json')
 M.need(result['manifest']==M.bind(M.OUT/'manifest.json'),'MANIFEST_BINDING')
 M.need(manifest['sources']==result['sources']==auth['sources'] and manifest['contract']==M.CONTRACT,'SOURCE_CONTRACT')
 M.need(manifest['authority']==M.bind(M.AUTH),'AUTHORITY_BINDING')
 inv=M.read(M.path_of(manifest['inventory']));M.need(inv['records']==ds and inv['counts']==counts and inv['sources']==auth['sources'],'CACHED_INVENTORY')
 bykey={(d['kind'],d['query_id'],d['execution_ordinal']):d for d in ds}
 entries=manifest['records'];failures=manifest['engineering_failures']
 all_keys=[(e['kind'],e['query_id'],e['execution_ordinal']) for e in entries+failures]
 M.need(len(all_keys)==128 and len(set(all_keys))==128 and set(all_keys)==set(bykey),'FULL_QUERY_COVERAGE')
 for e in failures:
  failure=M.read(M.path_of(e['failure']));M.need(all(failure[k]==e[k] for k in ('kind','query_id','execution_ordinal')),'ENGINEERING_FAILURE_SOURCE')
 checked=[]
 with ProcessPoolExecutor(max_workers=M.WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as executor:
  futures=[executor.submit(validate_episode,(bykey[(e['kind'],e['query_id'],e['execution_ordinal'])],e)) for e in entries]
  for future in as_completed(futures):
   checked.append(future.result())
   print(M.json.dumps({'event':'MAXMIN_CACHE_INDEPENDENT_QUERY_VALIDATED','records':len(checked),'total':len(entries)}),flush=True)
 total=sum(r['games'] for r in checked);reuse=sum(r['pilot_reuses'] for r in checked)
 ready=len(checked)==128 and not failures and total==8320 and reuse==512 and all(r['qualified'] for r in checked)
 summary={'records':len(checked),'engineering_failures':len(failures),'games':total,'pilot_reuses':reuse,'new_games':total-reuse,'expected_games':8320,'expected_new_games':7808}
 M.need(summary==result['summary']==manifest['summary'] and ready==result['all8320_gap_qualified']==manifest['all8320_gap_qualified'],'AGGREGATE_COVERAGE_QUALIFICATION')
 expected='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_COMPLETE' if ready else 'RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_ENGINEERING_INCOMPLETE'
 M.need(manifest['status']==result['status']==expected,'STATUS_FAIL_CLOSED')
 M.need(result['role_reads']==result['EVAL_outcome_reads']==result['new_training_updates']==0 and result['scientific_GO_or_NO_GO'] is None,'NO_ROLE_OR_TRAINING_CLAIM')
 out={'status':'RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_INDEPENDENT_VALIDATION_PASS' if ready else 'RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_INCOMPLETE_ARTIFACT_VALIDATION_PASS',
  'sources':auth['sources'],'contract':M.CONTRACT,'authority':M.bind(M.AUTH),'manifest':M.bind(M.OUT/'manifest.json'),'result':M.bind(M.OUT/'result.json'),
  'all8320_gap_qualified':ready,'summary':summary,'fresh_explicit_subprocess':True,'worker_processes':8,'threads_per_worker':1,
  'independent_exact_original_endpoints':True,'raw_p_and_alpha_exactly_rebuilt':True,'T_nearest_binary64_of_exact_L_verified':True,
  'pilot_reuse_records_revalidated':reuse,'new_LP_calls':0,'role_reads':0,'EVAL_outcome_reads':0,'scientific_GO_or_NO_GO':None}
 M.write(M.OUT/'validation.json',out);print(M.json.dumps({'status':out['status'],'summary':summary,'validation':M.bind(M.OUT/'validation.json')}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validate-child',required=True);a=p.parse_args()
 if not(a.validate_child==os.environ.get('RC_MAXMIN_CACHE_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('RC_MAXMIN_CACHE_VALIDATOR_PARENT_PID')):raise RuntimeError('EXPLICIT_CHILD_REQUIRED')
 validate()
if __name__=='__main__':main()
