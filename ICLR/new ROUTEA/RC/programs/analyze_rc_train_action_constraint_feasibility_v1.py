#!/usr/bin/env python3
"""TRAIN-only action-vs-sign constraint feasibility; no EVAL scoring."""
from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.optimize import linprog
import scipy
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_train_action_constraint_feasibility_v1'
SOURCE=ROOT/'programs/run_rc_absolute_evidence_scale_calibration_v1.py'
PARENT=ROOT/'results/rc_absolute_evidence_scale_calibration_v1'

def need(x,m):
 if not bool(x):raise RuntimeError(m)
def hx(x):return float(x).hex()
def row_exact(terms):
 return [sum((Fraction.from_float(float(vector[j]))*sign for sign,vector in terms),Fraction(0)) for j in range(len(terms[0][1]))]
def rational_solve(matrix,b):
 a=[list(r)+[v] for r,v in zip(matrix,b)];m=len(a);n=len(a[0])-1;row=0;pivots=[]
 for c in range(n):
  pivot=next((r for r in range(row,m) if a[r][c]),None)
  if pivot is None:continue
  a[row],a[pivot]=a[pivot],a[row];scale=a[row][c];a[row]=[x/scale for x in a[row]]
  for r in range(m):
   if r!=row and a[r][c]:
    factor=a[r][c];a[r]=[x-factor*y for x,y in zip(a[r],a[row])]
  pivots.append((row,c));row+=1
  if row==m:break
 if any(all(v==0 for v in r[:n]) and r[n]!=0 for r in a):return None
 x=[Fraction(0) for _ in range(n)]
 for r,c in pivots:x[c]=a[r][n]
 return x

def build(pair,train,key,sign_required,pair_required):
 terms=[];meta=[]
 def add(vectors,query,kind,candidate=None):
  terms.append(vectors);meta.append({'query_id':query,'kind':kind,'candidate_position':candidate})
 for r in train:
  f=r[key]['C_PAIRED'].numpy();x=np.column_stack([f,np.ones(len(f))]);cs=[int(c) for c in r['challenger_positions']]
  t=int(r['target_position']);w=int(r['base_winner_position'])
  if t==w:
   for i,c in enumerate(cs):add([(-1,x[i])],r['query_id'],'BASE_CORRECT_WRONG_BELOW_ZERO',c)
  else:
   ti=cs.index(t);add([(1,x[ti])],r['query_id'],'TRUE_CHALLENGER_ABOVE_HOLD',t)
   for i,c in enumerate(cs):
    if i==ti:continue
    add([(-1,x[i])] if sign_required else [(1,x[ti]),(-1,x[i])],r['query_id'],
        'WRONG_BELOW_ZERO' if sign_required else 'TRUE_CHALLENGER_ABOVE_WRONG',c)
 if pair_required:
  for r in pair['records']:
   f=r[key]['C_PAIRED'].numpy();need(f.shape[0]==1,'PAIR_ONE_ROW')
   x=np.append(f[0],1.)
   add([(1 if r['switch_label'] else -1,x)],r['query_id'],'PAIR_POSITIVE' if r['switch_label'] else 'PAIR_NEGATIVE')
 a=np.stack([sum(sign*v for sign,v in parts) for parts in terms])
 return a,terms,meta

def certify(a,terms,meta):
 m,d=a.shape;options={'presolve':True,'time_limit':30.,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
 # theta = theta_plus-theta_minus; minimizing L1 gives a bounded objective.
 result=linprog(np.ones(2*d),A_ub=-np.column_stack([a,-a]),b_ub=-np.ones(m),bounds=(0,None),method='highs',options=options)
 out={'solver_status':int(result.status),'solver_message':result.message,'constraint_count':m,'parameter_count':d}
 if result.success:
  theta=result.x[:d]-result.x[d:];tf=[Fraction.from_float(float(x)) for x in theta]
  margins=[sum((x*y for x,y in zip(row_exact(v),tf)),Fraction(0)) for v in terms]
  minimum=min(margins);need(minimum>0,'NUMERIC_PRIMAL_HAS_NO_EXACT_POSITIVE_MARGIN')
  # A common positive rescaling by this exact minimum provides unit margins.
  certificate=[v/minimum for v in tf]
  out.update(status='EXACT_RATIONAL_STRICT_ACTION_FEASIBLE',float_theta_binary64=[hx(x) for x in theta],
             float_witness_L1=float(np.abs(theta).sum()),exact_min_margin_before_rescale=str(minimum),
             rational_unit_margin_theta=[str(v) for v in certificate],exact_constraint_validation=True)
 else:
  need(result.status==2,'SOLVER_DID_NOT_RESOLVE_FEASIBILITY')
  dual=linprog(np.zeros(m),A_eq=np.vstack([a.T,np.ones(m)]),b_eq=np.r_[np.zeros(d),1.],bounds=(0,None),method='highs',options=options)
  out['dual_solver_status']=int(dual.status)
  if not dual.success:
   out.update(status='NUMERIC_INFEASIBLE_WITHOUT_EXACT_CERTIFICATE',exact_constraint_validation=False);return out
  support=np.flatnonzero(dual.x>0).tolist();exact_rows=[row_exact(terms[i]) for i in support]
  matrix=[[row[j] for row in exact_rows] for j in range(d)]+[[Fraction(1)]*len(support)]
  weights=rational_solve(matrix,[Fraction(0)]*d+[Fraction(1)])
  if weights is None or any(v<0 for v in weights):
   out.update(status='NUMERIC_INFEASIBLE_WITHOUT_EXACT_CERTIFICATE',exact_constraint_validation=False,numeric_dual_support=support);return out
  need(sum(weights)==1 and all(sum((weights[i]*exact_rows[i][j] for i in range(len(weights))),Fraction(0))==0 for j in range(d)),'EXACT_DUAL_CHECK')
  out.update(status='EXACT_RATIONAL_STRICT_ACTION_INFEASIBLE',exact_constraint_validation=True,
             certificate=[{'constraint_index':idx,'weight':str(weight),'constraint':meta[idx]} for idx,weight in zip(support,weights) if weight],
             contradiction='Nonnegative weights sum to1 and weighted exact left hand sides sum to0, whereas unit-margin right hand sides sum to1.')
 return out

def main():
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 need(not OUT.exists(),'APPEND_ONLY_OUTPUT')
 authority=json.loads((ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json').read_text())
 need(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==authority['sources']['program']['sha256'],'FROZEN_PRODUCER_SHA')
 spec=importlib.util.spec_from_file_location('frozen_abs_calibration_inputs',SOURCE);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 u.helper();u.deadline();need(u.sources()==authority['sources'],'FROZEN_INPUT_SOURCES')
 validation=u.H.read(PARENT/'independent_validation.json')
 need(validation['result_sha256']==u.H.sha(PARENT/'result.json') and all(validation['checks'].values()),'PARENT_NOT_VALIDATED')
 frozen,pure,pair,train,unused_eval,entries,labels,closure=u.prepare()
 need(closure==u.H.read(PARENT/'input_closure.json'),'TRAIN_FEATURE_RMS_REBUILD')
 tables={};started=time.monotonic();systems={}
 for name in u.MODELS:
  key=u.KEYS[name][0];tables[name]={}
  for sign,paired in [(False,False),(True,False),(False,True),(True,True)]:
   tag=('SIGN' if sign else 'ACTION')+('_PLUS_PAIR' if paired else '_FULL_ONLY')
   a,terms,meta=build(pair,train,key,sign,paired);answer=certify(a,terms,meta)
   tables[name][tag]=answer
   systems[name+'_'+tag]={'A_float64':a,'terms':terms,'metadata':meta}
   print(json.dumps({'model':name,'system':tag,'status':answer['status'],'seconds':time.monotonic()-started}),flush=True)
 need(u.H.BARRIER.blocked==0 and not u.H.BARRIER.released,'NO_EVAL_OR_OUTCOME_READ')
 OUT.mkdir()
 # The underlying frozen feature rows are stored alongside the certificates,
 # allowing exact rational reconstruction without relying on rounded A diffs.
 import pickle
 with (OUT/'constraint_systems.pkl').open('xb') as f:pickle.dump(systems,f,protocol=5)
 (OUT/'constraint_systems.pkl').chmod(0o444)
 out={'status':'TRAIN_CONSTRAINT_FEASIBILITY_COMPLETE','program':u.H.binding(__file__),
      'source_model_authority':u.H.binding(ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json'),
      'source_validation':u.H.binding(PARENT/'independent_validation.json'),'training_feature_closure':u.H.binding(PARENT/'input_closure.json'),
      'constraint_systems':u.H.binding(OUT/'constraint_systems.pkl'),'solver':{'scipy':scipy.__version__,'numpy':np.__version__,'method':'highs'},
      'models':tables,'TRAIN_queries':32,'PAIR_queries':64,'EVAL_scores_computed':0,'EVAL_target_reads':0,
      'optimization_head_candidate_created':False,'HYP_GO_claimed':False,
      'interpretation':['These are feasibility certificates on existing TRAIN features, not generalization or EVAL accuracy.',
                        'SIGN imposes the additional wrong-logit-below-zero requirement used by the original surrogate.',
                        'ACTION requires only the actual target to beat HOLD0 and every wrong candidate.',
                        'Exact checks treat the original cached floating-point feature values as exact rationals; they do not assert arbitrary finite-precision arithmetic is exact.',
                        'No model, feature, threshold, seed or EVAL decision is selected using these certificates.']}
 u.H.atomic(OUT/'result.json',out)
 print(json.dumps({'status':out['status'],'result_sha256':u.H.sha(OUT/'result.json'),'elapsed_seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
