#!/usr/bin/env python3
"""Fresh process validation of exact max-min bounds; never invokes a solver."""
from __future__ import annotations
import argparse
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'programs/run_rc_reference_support_maxmin_train_pilot_v1.py'

def independent_distribution(D,raw,dimension):
 if raw is None:return None,'NO_SOLUTION_VECTOR'
 D.need(len(raw)==dimension,'INDEPENDENT_SOLUTION_DIMENSION')
 values=[float.fromhex(x) if isinstance(x,str) else float(x) for x in raw]
 if not all(math.isfinite(x) for x in values):return None,'NONFINITE_SOLUTION_VECTOR'
 fractions=[Fraction.from_float(x) if x>0 else Fraction(0) for x in values]
 mass=sum(fractions,Fraction(0))
 if mass==0:return None,'NO_POSITIVE_COEFFICIENT_MASS'
 normalized=[f/mass for f in fractions];positions=[i for i,f in enumerate(normalized) if f]
 denominator=math.lcm(*(normalized[i].denominator for i in positions))
 nums=[normalized[i].numerator*(denominator//normalized[i].denominator) for i in positions]
 D.need(sum(normalized,Fraction(0))==1 and all(f>=0 for f in normalized),'EXACT_SIMPLEX')
 return {'dimension':dimension,'nonzero_positions':positions,'numerators':[str(n) for n in nums],
  'denominator':str(denominator),'negative_coefficients_clipped':sum(x<0 for x in values)},None

class EndpointFractions:
 def __init__(self,a):
  fractions=[[Fraction.from_float(float(x)) for x in row] for row in a]
  self.scale=max(f.denominator for row in fractions for f in row)
  self.rows=[[f.numerator*(self.scale//f.denominator) for f in row] for row in fractions]
  self.n=a.shape[1]
 def own_minus_best_other(self,g,prob):
  positions=prob['nonzero_positions'];nums=[int(x) for x in prob['numerators']]
  # Independent grouping: first pool each complete reference, then subtract.
  pooled=[sum(row[i]*n for i,n in zip(positions,nums)) for row in self.rows]
  best=max(pooled[h] for h in range(128) if h!=g)
  return Fraction(pooled[g]-best,int(prob['denominator'])*self.scale)
 def dual_upper(self,g,prob):
  others=[h for h in range(128) if h!=g];den=int(prob['denominator'])
  opponents=[others[j] for j in prob['nonzero_positions']];nums=[int(n) for n in prob['numerators']]
  # Sum(alpha)=1 exactly, so endpoint difference is formed after competitor pooling.
  margins=[self.rows[g][i]*den-sum(self.rows[h][i]*n for h,n in zip(opponents,nums)) for i in range(self.n)]
  return Fraction(max(margins),den*self.scale)

def expected_certificate(D,matrix,g,p,alpha):
 lo=matrix.own_minus_best_other(g,p);hi=matrix.dual_upper(g,alpha)
 D.need(lo<=hi,'INDEPENDENT_EXACT_WEAK_DUALITY')
 sign='POSITIVE_SEPARATION' if lo>0 else ('NONPOSITIVE_OPTIMUM' if hi<=0 else 'UNRESOLVED_SIGN')
 return {'lower':D.frac(lo),'upper':D.frac(hi),'gap':D.frac(hi-lo),'gap_qualified':hi-lo<=Fraction(1,100000000),'sign_classification':sign}

def validate(D):
 D.need(not (D.OUT/'validation.json').exists(),'APPEND_ONLY_VALIDATION_EXISTS')
 auth,pf=D.authority();inputs=D.source_inputs();D.need([D.public_input(r) for r in inputs]==pf['inputs'],'TRAIN_SOURCES_FROZEN')
 result=D.read(D.OUT/'result.json');seal=D.read(D.path_of(result['prejoin_seal']))
 D.need(seal['sources']==auth['sources'] and seal['contract']==D.CONTRACT and result['contract']==D.CONTRACT,'RESULT_SOURCE_CONTRACT')
 D.need(seal['inputs']==pf['inputs'] and seal['TRAIN_label_reads']==0 and seal['EVAL_reads']==0,'PREJOIN_ROLE_BOUNDARY')
 D.need(seal['authority']==D.bind(D.AUTH) and result['authority']==D.bind(D.AUTH),'AUTHORITY_BINDING')
 header=D.read(D.path_of(seal['header']));D.need(header['sources']==auth['sources'] and header['inputs']==pf['inputs'] and header['TRAIN_label_reads']==0,'RUN_HEADER')
 progress=D.path_of(seal['progress']);lines=[D.json.loads(line) for line in progress.read_text().splitlines()]
 records=seal['records'];D.need(lines==records,'APPEND_PROGRESS_SEAL_EXACT')
 expected_order=[(ex,g) for ex in D.EXECS for g in range(128)]
 D.need([(r['execution_ordinal'],r['candidate_position']) for r in records]==expected_order[:len(records)] and len(records)<=512,'ALL_CANDIDATE_PREFIX_ORDER')
 by_query={r['execution_ordinal']:r for r in inputs};matrices={ex:EndpointFractions(r['a']) for ex,r in by_query.items()}
 checked=0
 for r in records:
  ex,g=r['execution_ordinal'],r['candidate_position'];row=by_query[ex];matrix=matrices[ex]
  D.need(r['query_id']==row['query_id'] and r['physical_row']==row['candidate_physical_rows'][g],'CANDIDATE_SOURCE_POSITION')
  D.need(r['opponent_positions']==[h for h in range(128) if h!=g],'ALL127_OTHER_POSITIONS')
  vectors=[];errors=[]
  for call,dim in ((r['primal_solver'],768),(r['dual_solver'],127)):
   D.need(call['seconds']>=0 and isinstance(call['attempted'],bool),'SOLVER_TIMING_AND_ATTEMPT')
   raw=call.get('raw_solution_binary64')
   if raw is not None:D.need(len(raw)==dim+1,'RAW_FULL_LP_VECTOR')
   dist,error=independent_distribution(D,None if raw is None else raw[:-1],dim)
   vectors.append(dist);errors.append(error)
  p,alpha=vectors;pe,ae=errors
  D.need(r['primal_distribution']==p and r['dual_distribution']==alpha and r['distribution_errors']=={'primal':pe,'dual':ae},'RAW_SOLUTION_EXACT_PROBABILITIES')
  old,old_error=independent_distribution(D,row['wq'][g].tolist(),768)
  D.need(old_error in (None,'NO_POSITIVE_COEFFICIENT_MASS'),'ORIGINAL_SUPPORT_FINITE')
  baseline=None if old is None else D.frac(matrix.own_minus_best_other(g,old))
  D.need(r['original_support_distribution']==old and r['original_exact_support_margin']==baseline,'EXACT_FIXED_SUPPORT_BOUND')
  D.need(r['old_FP64']==row['old_FP64'][g],'ORIGINAL_CLAMPED_FP64_REPLAY')
  expected=None if pe or ae else expected_certificate(D,matrix,g,p,alpha)
  D.need(r['certificate']==expected,'EXACT_PRIMAL_DUAL_CERTIFICATE')
  if expected is not None and baseline is not None:D.need(D.unfrac(baseline)<=D.unfrac(expected['upper']),'BASELINE_BOUNDED_BY_DUAL')
  D.need(r['seconds']>=0,'GAME_TIME')
  checked+=1
  if checked%64==0:print(D.json.dumps({'event':'INDEPENDENT_EXACT_MAXMIN_VALIDATION','games':checked,'planned_games':512}),flush=True)
 complete=len(records)==512 and all(r['dual_solver'].get('attempted') for r in records)
 summary=D.result_summary(records,complete)
 D.need(seal['summary']==result['summary']==summary,'COMPLETE_ALL_RECORD_SUMMARY')
 D.need(seal['status']==('TRAIN_MAXMIN_PREJOIN_COMPLETE' if complete else 'TRAIN_MAXMIN_PREJOIN_INCOMPLETE'),'PREJOIN_COMPLETENESS')
 status='TRAIN_MAXMIN_PILOT_COMPLETE_GAP_QUALIFIED' if summary['all512_gap_qualified'] else ('TRAIN_MAXMIN_PILOT_COMPLETE_NUMERICALLY_UNRESOLVED' if complete else 'TRAIN_MAXMIN_PILOT_INCOMPLETE')
 D.need(result['status']==status,'STATUS_NOT_SOLVER_SUCCESS')
 if summary['all512_gap_qualified']:post=D.postjoin(inputs,records,result['prejoin_seal'])
 else:post={'scope':'INCOMPLETE_OR_UNQUALIFIED_NO_LABEL_JOIN','TRAIN_role_reads':0,'EVAL_role_reads':0,'targets':[]}
 D.need(post==result['TRAIN_postjoin'],'ONLY_SELECTED_TRAIN_ROLE_DESCRIPTION')
 D.need(result['scientific_GO_or_NO_GO'] is None and result['retrieval_accuracy_claimed'] is False and result['no_forbidden_read_attempts'] is True and not D.BLOCKED,'NO_ACCURACY_OR_PROTECTED_CLAIM')
 out={'status':'TRAIN_MAXMIN_PILOT_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS','result':D.bind(D.OUT/'result.json'),
  'prejoin_seal':result['prejoin_seal'],'authority':D.bind(D.AUTH),'sources':auth['sources'],'contract':D.CONTRACT,
  'summary':summary,'fresh_explicit_subprocess':True,'validator_pid':os.getpid(),'validator_parent_pid':os.getppid(),
  'original_endpoints_rebuilt_with_Fraction_from_float':True,'bounds_independently_pool_then_difference':True,
  'raw_solver_probabilities_exactly_rebuilt':True,'new_LP_calls':0,'EVAL_reads':0,'scientific_GO_or_NO_GO':None}
 D.write(D.OUT/'validation.json',out);print(D.json.dumps({'status':out['status'],'summary':summary,'validation':D.bind(D.OUT/'validation.json')}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validate-child',required=True);a=p.parse_args()
 if not(a.validate_child==os.environ.get('RC_MAXMIN_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('RC_MAXMIN_VALIDATOR_PARENT_PID')):raise RuntimeError('EXPLICIT_CHILD_REQUIRED')
 spec=importlib.util.spec_from_file_location('frozen_maxmin_pilot_driver',DRIVER);D=importlib.util.module_from_spec(spec);sys.modules[spec.name]=D;spec.loader.exec_module(D)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);sys.path.insert(0,str(ROOT/'src'));sys.addaudithook(D.barrier)
 validate(D)
if __name__=='__main__':main()
