#!/usr/bin/env python3
"""Fixed zero-training C128 projection argmax, separately sealed and described.

produce-prejoin reads only the qualified label-free cache. describe-validated
requires the completed four-head independent reexecution before reading its
already-opened TRAIN/EVAL target descriptions. No fifth head or action is run.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
PLAN=ROOT/'plan/RC_DIRECT_PROJECTION_CANDIDATE_DIAGNOSTIC_V1_20260910.md'
OUT=ROOT/'results/rc_direct_projection_candidate_diagnostic_v1'
CACHE=ROOT/'results/rc_reference_support_maxmin_cache_v1'
CACHE_AUTH=ROOT/'registry/rc_reference_support_maxmin_cache_authority_v1_20260910.json'
READOUT=ROOT/'results/rc_reference_support_maxmin_readout_v1'
READOUT_AUTH=ROOT/'registry/rc_reference_support_maxmin_readout_authority_v1_20260910.json'
READOUT_PROGRAM=ROOT/'programs/run_rc_reference_support_maxmin_readout_v1.py'
ADDENDUM=ROOT/'plan/RC_REFERENCE_SUPPORT_PROJECTION_DISTANCE_ADDENDUM_V2_20260910.md'
HEADS=('ORIGINAL7','FIXED8','MAXMIN8','PROJECTION8')
CONTRACT={'theory_name':'new HYP','rule':'d_g=abs(binary64(J_g+T_g))/math.sqrt(2.0)',
 'selection':'maximum_d_then_minimum_physical_row','queries':'ALL_FULL64','candidates_each':128,
 'new_training_updates':0,'new_LP_calls':0,'HOLD_SWITCH_action_used':False,'fifth_trained_head':False,
 'projection_axis':'FIXED_V2_DIAGONAL','postjoin':'ONLY_AFTER_ALL_FOUR_HEADS_INDEPENDENTLY_VALIDATED',
 'evidence':'previously_opened_internal_TRAIN32_EVAL32_direct_candidate_ranking_diagnostic',
 'automatic_adoption':False,'scientific_GO_or_NO_GO':None}
HEAD_VERIFICATION=False;HEAD_RELEASED=False;HASH_DEPTH=0;BLOCKED=[]

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p);bad=None
 if any(x in s.lower() for x in ('/role_shards/','/rc_opened_eval_strict_','d1_mi','d1-mi','grozi','gisc_prerecall_universe')):bad='ROLE_OR_ORACLE_OR_PROTECTED_READ'
 if p.is_relative_to(READOUT) and not HEAD_VERIFICATION:bad='FOUR_HEAD_ARTIFACT_BEFORE_DIRECT_PREJOIN'
 if p==READOUT/'result.json' and not HEAD_RELEASED and not HASH_DEPTH:bad='FOUR_HEAD_OUTCOME_BEFORE_QUALIFICATION'
 if p.name=='result.json' and '/RC/results/' in s and p.parent not in (OUT,CACHE,READOUT) and not HASH_DEPTH:bad='UNRELATED_OUTCOME_READ'
 if p.suffix in('.pt','.npz'):bad='NO_TENSOR_CACHE_OR_MODEL_FORWARD_NEEDED'
 if bad:BLOCKED.append(s);raise RuntimeError(bad)
def sha(p):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for c in iter(lambda:f.read(8<<20),b''):h.update(c)
  return h.hexdigest()
 finally:HASH_DEPTH-=1

def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=encode(v)+b'\n'
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_DRIFT:'+str(p));return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)
def checked(v,root=None):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
 if root is not None:need(p.is_relative_to(root),'BOUND_FILE_OUTSIDE_EXPECTED_ROOT')
 need(sha(p)==v['sha256'],'BOUND_FILE_SHA:'+str(p));return p

def cache_ready():
 m=read(CACHE/'manifest.json');v=read(CACHE/'validation.json');a=read(CACHE_AUTH)
 need(m['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_COMPLETE' and m['all8320_gap_qualified'] is True,'COMPLETE_CACHE_REQUIRED')
 need(v['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_INDEPENDENT_VALIDATION_PASS' and v['all8320_gap_qualified'] is True,'INDEPENDENT_CACHE_QUALIFICATION_REQUIRED')
 need(checked(v['manifest'],CACHE)==(CACHE/'manifest.json').resolve() and checked(v['result'],CACHE)==(CACHE/'result.json').resolve(),'CACHE_VALIDATION_BINDINGS')
 need(checked(m['authority'])==CACHE_AUTH.resolve() and a['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_AUTHORIZED','CACHE_AUTHORITY')
 need(m['sources']==v['sources']==a['sources'] and m['contract']==a['contract'] and m['contract']['games']==8320,'CACHE_SOURCE_OPERATOR_CLOSURE')
 need(m['contract']['T']=='float(Fraction(exact_lower_L)).hex();nearest_binary64;retain_negative','EXACT_SIGNED_T_DEFINITION')
 need(len(m['records'])==128 and sum(e['kind']=='FULL' for e in m['records'])==64 and sum(e['kind']=='PAIR' for e in m['records'])==64,'COMPLETE_CACHE_QUERY_POPULATION')
 return m,{'manifest':bind(CACHE/'manifest.json'),'validation':bind(CACHE/'validation.json'),'result':bind(CACHE/'result.json'),'authority':bind(CACHE_AUTH)}

def predictions(manifest,independent=False):
 output=[]
 entries=sorted((e for e in manifest['records'] if e['kind']=='FULL'),key=lambda e:e['execution_ordinal'])
 need(len({e['query_id'] for e in entries})==len({e['execution_ordinal'] for e in entries})==64,'FULL64_UNIQUE_QUERY_KEYS')
 for entry in entries:
  rp=checked(entry,CACHE);r=read(rp);axis=r['candidate_physical_rows'];positions=list(range(128))
  need(r['kind']=='FULL' and r['query_id']==entry['query_id'] and r['execution_ordinal']==entry['execution_ordinal'],'RECORD_QUERY_KEY')
  need(r['all_game_gap_qualified'] is True and r['score_positions']==positions and len(axis)==len(set(axis))==128,'EVERY_FULL_CANDIDATE_QUALIFIED')
  j={int(k):float.fromhex(x) for k,x in r['fixed_J_binary64'].items()};t={int(k):float.fromhex(x) for k,x in r['T_binary64'].items()}
  need(set(j)==set(t)==set(positions),'COMPLETE_J_T_AXIS')
  cert=read(checked(r['certificates'],CACHE))
  need(cert['all_game_gap_qualified'] is True and [g['candidate_position'] for g in cert['games']]==positions,'CERTIFIED_FULL_AXIS')
  for g in cert['games']:
   c=g['certificate'];lo=Fraction(int(c['lower']['numerator']),int(c['lower']['denominator']));hi=Fraction(int(c['upper']['numerator']),int(c['upper']['denominator']))
   need(c['gap_qualified'] is True and lo<=hi and hi-lo<=Fraction(1,100000000),'EXACT_QUALIFIED_INTERVAL')
   need(t[g['candidate_position']].hex()==float(lo).hex(),'T_EXACT_L_ROUNDING')
  if independent:
   aa=np.asarray([j[p] for p in positions],dtype=np.float64);bb=np.asarray([t[p] for p in positions],dtype=np.float64)
   sums=np.add(aa,bb,dtype=np.float64);distance=np.divide(np.absolute(sums),np.sqrt(np.float64(2.)),dtype=np.float64)
   ds=[float(x) for x in distance];ss=[float(x) for x in sums]
   order=sorted(positions,key=lambda p:(-ds[p],axis[p]))
   largest=max(ds);winner=min((p for p in positions if ds[p]==largest),key=lambda p:axis[p])
  else:
   ss=[j[p]+t[p] for p in positions];ds=[abs(x)/math.sqrt(2.0) for x in ss]
   winner=max(positions,key=lambda p:(ds[p],-axis[p]))
   order=sorted(positions,key=lambda p:(ds[p],-axis[p]),reverse=True)
  need(all(math.isfinite(x) for x in ss+ds) and all(x>=0 for x in ds),'FINITE_DISTANCE')
  need(order[0]==winner and set(order)==set(positions),'FULL_RANKING_AND_ARGMAX')
  output.append({'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],'candidate_physical_rows':axis,
   'base_winner_position':r['base_winner_position'],'fixed_J_binary64':[j[p].hex() for p in positions],
   'T_binary64':[t[p].hex() for p in positions],'J_plus_T_binary64':[x.hex() for x in ss],
   'distance_binary64':[x.hex() for x in ds],'ranked_candidate_positions':order,
   'predicted_position':winner,'predicted_physical_row':axis[winner],
   'cache_record':bind(rp),'certificates':r['certificates']})
 need(not BLOCKED,'FORBIDDEN_READ_ATTEMPT');return output

def produce():
 need(not OUT.exists(),'APPEND_ONLY_PREJOIN_EXISTS');m,cache=cache_ready()
 sources={'program':bind(PROGRAM),'plan':bind(PLAN),'projection_V2_addendum':bind(ADDENDUM),
  'four_head_program':bind(READOUT_PROGRAM),'cache':cache}
 rows=predictions(m);need(rows==predictions(m,independent=True),'INDEPENDENT_LITERAL_DISTANCE_AND_RANKING_REPLAY')
 OUT.mkdir();pre={'status':'DIRECT_PROJECTION_FULL64_PREJOIN_READY','sources':sources,'contract':CONTRACT,'records':rows,
  'query_count':64,'candidate_scores':8192,'target_or_four_head_output_reads':0,'training_updates':0}
 write(OUT/'candidate_prejoin.json',pre)
 seal={'status':'DIRECT_PROJECTION_FULL64_PREDICTIONS_SEALED','prejoin':bind(OUT/'candidate_prejoin.json'),'sources':sources,
  'contract':CONTRACT,'all64x128_scores_and_rankings_sealed':True,'independent_literal_arithmetic_and_tie_break':True,
  'target_or_four_head_output_reads':0,'new_training_updates':0,'HOLD_SWITCH_action_used':False}
 write(OUT/'candidate_prejoin_seal.json',seal)
 print(json.dumps({'status':seal['status'],'prejoin_seal':bind(OUT/'candidate_prejoin_seal.json')}),flush=True)

def describe():
 global HEAD_VERIFICATION,HEAD_RELEASED
 seal=read(OUT/'candidate_prejoin_seal.json');pre=read(checked(seal['prejoin'],OUT))
 need(seal['status']=='DIRECT_PROJECTION_FULL64_PREDICTIONS_SEALED' and seal['contract']==pre['contract']==CONTRACT,'DIRECT_PREJOIN_SEAL')
 need(seal['sources']==pre['sources'] and seal['sources']['program']==bind(PROGRAM) and seal['sources']['plan']==bind(PLAN),'DIRECT_SOURCE_BINDING')
 m,cache=cache_ready();need(cache==seal['sources']['cache'],'QUALIFIED_CACHE_UNCHANGED')
 need(pre['records']==predictions(m,independent=True),'FRESH_LITERAL_PREJOIN_REBUILD')
 HEAD_VERIFICATION=True
 validation=read(READOUT/'independent_validation.json')
 need(validation['status']=='MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS' and validation['checks']['all_four_heads_retrained_exactly'] is True and all(v is True for v in validation['checks'].values()),'FOUR_HEAD_INDEPENDENT_VALIDATION_REQUIRED')
 need(validation['result_sha256']==sha(READOUT/'result.json'),'FOUR_HEAD_VALIDATED_RESULT_SHA')
 authority=read(READOUT_AUTH);headseal=read(READOUT/'eval_prejoin_seal.json')
 need(checked(authority['sources']['program'])==READOUT_PROGRAM.resolve() and seal['sources']['four_head_program']==bind(READOUT_PROGRAM),'FOUR_HEAD_PROGRAM_AUTHORITY')
 need(checked(authority['sources']['user_PROJECTION8_addendum'])==ADDENDUM.resolve() and authority['sources']['user_PROJECTION8_addendum']['sha256']==seal['sources']['projection_V2_addendum']['sha256'],'PROJECTION_RULE_UNCHANGED')
 need(headseal['parameters_sha256']==sha(READOUT/'parameters.json') and headseal['eval_prejoin_sha256']==sha(READOUT/'eval_prejoin.json'),'FOUR_HEAD_PARAMETERS_AND_PREDICTIONS_SEALED')
 need(tuple(headseal['models'])==HEADS and headseal['TRAIN_count']==headseal['EVAL_count']==32 and headseal['candidate_count']==128 and headseal['runtime_EVAL_target_reads']==0,'FOUR_HEAD_PREJOIN_BOUNDARY')
 params=read(READOUT/'parameters.json');heads_predictions=read(READOUT/'eval_prejoin.json')
 need(set(params)==set(HEADS) and len(heads_predictions)==64,'FOUR_HEAD_COMPLETE_POPULATION')
 expected={(r['query_id'],r['execution_ordinal']) for r in pre['records']}
 need({(r['query_id'],r['execution_ordinal']) for r in heads_predictions}==expected,'IDENTICAL_FULL64_QUERY_UNIVERSE')
 HEAD_RELEASED=True;old=read(READOUT/'result.json')
 need(old['parameters_sha256']==sha(READOUT/'parameters.json') and old['eval_prejoin_seal_sha256']==sha(READOUT/'eval_prejoin_seal.json') and old['authority_sha256']==sha(READOUT_AUTH),'FOUR_HEAD_OUTCOME_SOURCE_CLOSURE')
 bykey={(r['query_id'],r['execution_ordinal']):r for r in pre['records']};rows=[];summaries={};comparisons={};head_counts={};seen=set()
 for role in ('TRAIN','EVAL'):
  labels=old['actions'][role]['ORIGINAL7']['REAL'];need(len(labels)==32,'WHOLE32_TARGET_DESCRIPTIONS')
  other={name:{(a['query_id'],a['execution_ordinal']):a for a in old['actions'][role][name]['REAL']} for name in HEADS}
  rr=[]
  for label in labels:
   key=(label['query_id'],label['execution_ordinal']);need(key in bykey and key not in seen,'UNIQUE_VALIDATED_TARGET_JOIN');seen.add(key)
   direct=bykey[key];target=int(label['target_position']);axis=direct['candidate_physical_rows']
   need(0<=target<128 and axis[target]==label['target_physical_row'] and label['base_winner']==direct['base_winner_position'],'TARGET_AND_RAW_AXIS_MATCH')
   for name in HEADS:
    a=other[name][key];need(a['target_position']==target and a['target_physical_row']==axis[target],'FOUR_HEAD_TARGET_AGREEMENT')
   rank=direct['ranked_candidate_positions'].index(target)+1
   record={'role':role,'query_id':key[0],'execution_ordinal':key[1],'target_position':target,'target_physical_row':axis[target],
    'direct_position':direct['predicted_position'],'direct_physical_row':direct['predicted_physical_row'],
    'direct_correct':rank==1,'direct_target_rank':rank,'raw_correct':bool(label['base_correct']),
    'four_head_final_correct':{name:bool(other[name][key]['final_correct']) for name in HEADS}}
   rr.append(record);rows.append(record)
  summaries[role]={'query_count':32,'candidate_count':128,'top1_correct':sum(r['direct_correct'] for r in rr),
   'R_at_1':sum(r['direct_correct'] for r in rr)/32,'MRR':sum(1/r['direct_target_rank'] for r in rr)/32,
   'scoring_path':'untrained_C128_distance_argmax_and_full_distance_ranking_no_action'}
  comparisons[role]={}
  for base in ('RAW',)+HEADS:
   bc=lambda r:r['raw_correct'] if base=='RAW' else r['four_head_final_correct'][base]
   rescue=[r['query_id'] for r in rr if r['direct_correct'] and not bc(r)]
   breaks=[r['query_id'] for r in rr if bc(r) and not r['direct_correct']]
   comparisons[role]['DIRECT_vs_'+base]={'rescue':len(rescue),'break':len(breaks),'net':len(rescue)-len(breaks),'rescue_query_ids':rescue,'break_query_ids':breaks}
  head_counts[role]={'scoring_path':'separately_trained_four_head_all127_challenger_HOLD_SWITCH',
   'RAW_correct':sum(r['raw_correct'] for r in rr),'head_correct':{name:sum(r['four_head_final_correct'][name] for r in rr) for name in HEADS}}
  for name in HEADS:need(head_counts[role]['head_correct'][name]==old['metrics'][role][name]['REAL']['final_top1'],'VALIDATED_HEAD_COUNT_REPLAY')
 need(seen==expected and not BLOCKED,'WHOLE64_JOIN_NO_FORBIDDEN_READ')
 result={'status':'DIRECT_PROJECTION_VALIDATED_DESCRIPTIVE_DIAGNOSTIC_READY','contract':CONTRACT,
  'direct_prejoin_seal':bind(OUT/'candidate_prejoin_seal.json'),
  'four_head_qualification':{name:bind(READOUT/name) for name in ('parameters.json','eval_prejoin.json','eval_prejoin_seal.json','result.json','independent_validation.json')},
  'four_head_authority':bind(READOUT_AUTH),'direct_candidate_ranking':summaries,'separate_trained_action_counts':head_counts,
  'paired_descriptive_comparisons':comparisons,'target_descriptions_and_direct_ranks':rows,
  'independent_numpy_binary64_distance_and_physical_tie_break_replayed':True,'new_training_updates':0,'new_role_file_reads':0,
  'deployment_or_adoption':False,'scientific_GO_or_NO_GO':None}
 write(OUT/'result.json',result)
 write(OUT/'validation.json',{'status':'DIRECT_PROJECTION_LITERAL_REPLAY_AND_POSTJOIN_VALIDATION_PASS','result':bind(OUT/'result.json'),
  'direct_prejoin_seal':bind(OUT/'candidate_prejoin_seal.json'),'all64x128_scores_and_rankings_rebuilt':True,
  'four_head_independent_qualification_required':True,'all64_target_descriptions_from_validated_readout':True,'new_training_updates':0,'new_role_file_reads':0})
 print(json.dumps({'status':result['status'],'direct_candidate_ranking':summaries,'separate_trained_action_counts':head_counts,'result':bind(OUT/'result.json')}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('produce-prejoin','describe-validated'));a=p.parse_args();sys.addaudithook(audit)
 if a.phase=='produce-prejoin':produce()
 else:describe()
if __name__=='__main__':main()
