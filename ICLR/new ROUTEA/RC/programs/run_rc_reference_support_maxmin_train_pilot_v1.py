#!/usr/bin/env python3
"""TRAIN4 only: max-min query-support LP with exact feasible game bounds.

No natural LP is run by preflight or freeze. Natural execution requires a frozen
source authority and Slurm allocation. Labels are joined only after all512 games
have been sealed; incomplete work remains explicitly incomplete.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
from functools import reduce
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import warnings
import numpy as np
import scipy
from scipy.optimize import linprog,OptimizeWarning
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
VALIDATOR=ROOT/'programs/validate_rc_reference_support_maxmin_train_pilot_v1.py'
LAUNCH=ROOT/'slurm/rc_reference_support_maxmin_train_pilot_v1_dev_cpuonly_15m.sbatch'
PLAN=ROOT/'plan/RC_REFERENCE_SUPPORT_MAXMIN_TRAIN_PILOT_V1_20260910.md'
OUT=ROOT/'results/rc_reference_support_maxmin_train_pilot_v1'
PREFLIGHT=ROOT/'results/rc_reference_support_maxmin_train_pilot_v1_preflight'
AUTH=ROOT/'registry/rc_reference_support_maxmin_train_pilot_authority_v1_20260910.json'
SHARED=ROOT/'results/rc_shared_query_target_prior_cache_v1'
FULLREF=ROOT/'results/rc_full64_full_reference_cache_v1'
ROLE=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2'
SPEC=ROOT/'programs/prepare_rc_same_support_specificity_inputs_v1.py'
EXECS=(56,66,72,75)
OPTIONS={'presolve':True,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,
 'threads':1,'parallel':False,'random_seed':0,'time_limit':5.0}
GAP=Fraction(1,100000000)
BUDGET=600.0
PINS={
 'shared_manifest':(SHARED/'manifest.json','f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c'),
 'shared_validation':(SHARED/'validation.json','0258615f000d0d82f757f57298b803420395ebc3a35e3a46bfa53c3116b7488f'),
 'full_reference_manifest':(FULLREF/'manifest.json','51618b74113bb8c3064e36116c6db0f6d559419d1da7a0fcb4b0b3725ce12234'),
 'full_reference_validation':(FULLREF/'validation.json','65760c2a528e31c6fe9246b6bbe76869a74d81599d5b8cdeedf60ecaaca431f8'),
 'specificity_input_program':(SPEC,'4bd77d8e7aaab3f51155f2721dbba553e899b1b7e28f7dde138b586509320627'),
 'specificity_input_manifest':(ROOT/'results/rc_same_support_specificity_inputs_v1/manifest.json','378dbf79e540653f6bb1b555d3e1079c7153cd350ed7d689fc3b147cd728c574'),
 'specificity_input_validation':(ROOT/'results/rc_same_support_specificity_inputs_v1/validation.json','545b10d603064e2df9d746e9e16fdebdb64de2dc56048aced04c44bc76e48a75'),
 'identity_manifest':(ROOT/'registry/gallery_identity_repair_v1.json','9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f'),
 'identity_core':(ROOT/'src/rc_aslo_xf/gallery_identity_repair.py','995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d'),
 'identity_contract':(ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json','867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650'),
 'role_manifest':(ROLE/'role_manifest.json','2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454'),
 'role_validation':(ROLE/'independent_validation.json','ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac')}
CONTRACT={'theory_name':'new HYP','scope':'TRAIN_SUPPORT_MAXMIN_ENGINEERING_PILOT_NOT_ACCURACY',
 'execution_order':list(EXECS),'query_count':4,'candidates_per_query':128,'games':512,'LP_calls_if_complete':1024,
 'method':'highs-ds','options':OPTIONS,'scipy_version':'1.16.3','CPU_threads':1,
 'primal_margin_bounds':[None,None],'dual_margin_bounds':[None,None],
 'distribution':'negative_numeric_coefficients_clipped_zero_then_exact_binary64_rational_normalization',
 'certificate':'exact_original_binary64_a_endpoint_differences_not_rounded_D',
 'gap_qualification_rational':{'numerator':'1','denominator':'100000000'},'soft_budget_seconds':BUDGET,
 'signs':'L>0_POSITIVE;U<=0_NONPOSITIVE;otherwise_UNRESOLVED',
 'old_wq_baseline':'exact_normalized_positive_mass_separate_from_FP64_clamped_original_J',
 'postjoin':'ALL512_GAMES_WITH_QUALIFIED_EXACT_BOUNDS_SEALED_BEFORE_ONLY4_TRAIN_IDENTITY_LABEL_READS',
 'new_model_training_updates':0,'new_encoder_or_RoMa_forwards':0,'EVAL_source_or_outcome_reads':0,
 'scientific_GO_or_NO_GO':None,'deployable_model_result':False,'spatial_or_pixel_mask_claimed':False}
ALLOWED_EPISODES=set();ALLOWED_ROLES=set();ROLES_RELEASED=False;BLOCKED=[]

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def barrier(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p)
 reason=None
 if any(x in s.lower() for x in ('d1_mi','d1-mi','grozi','gisc_prerecall_universe','/rc_opened_eval_strict_')):reason='PROTECTED_OR_ORACLE'
 if '/role_shards/' in s and (not ROLES_RELEASED or s not in ALLOWED_ROLES):reason='ROLE_BEFORE_SEAL_OR_NOT_SELECTED_TRAIN'
 if any('/'+x+'/episodes/' in s for x in ('rc_shared_query_target_prior_cache_v1','rc_full64_full_reference_cache_v1')) and s not in ALLOWED_EPISODES:reason='NON_SELECTED_EPISODE'
 if p.name=='result.json' and '/RC/results/' in s and p.parent!=OUT and p.parent!=PREFLIGHT:reason='OTHER_RESULT'
 if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'):reason='IMAGE_FILE_READ'
 if reason:BLOCKED.append({'path':s,'reason':reason});raise RuntimeError(reason)
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=encode(v)+b'\n'
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_DRIFT:'+str(p));return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)
def path_of(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==v['sha256'],'SOURCE_SHA:'+str(p));return p

def tensor_sha(a):
 a=np.ascontiguousarray(a);dtype={'float64':'torch.float64','float16':'torch.float16'}[str(a.dtype)]
 return hashlib.sha256(dtype.encode()+encode(list(a.shape))+a.tobytes()).hexdigest()
def frac(v):return {'numerator':str(v.numerator),'denominator':str(v.denominator),'approximate_binary64':float(v).hex()}
def unfrac(d):return Fraction(int(d['numerator']),int(d['denominator']))
def deadline():need(datetime.now(timezone.utc)<datetime(2026,9,11,16,tzinfo=timezone.utc),'USER_RESEARCH_DEADLINE')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_inputs():
 m=read(SHARED/'manifest.json');v=read(SHARED/'validation.json')
 need(v['manifest_sha256']==PINS['shared_manifest'][1] and all(x is True for x in v['checks'].values()),'SHARED_CACHE_VALIDATED')
 entries=sorted((e for e in m['records'] if e['kind']=='FULL' and e['role']=='TRAIN'),key=lambda e:int(e['execution_ordinal']))[:4]
 need(tuple(int(e['execution_ordinal']) for e in entries)==EXECS,'EXACT_FIRST4_TRAIN_SELECTION')
 fm=read(FULLREF/'manifest.json');fv=read(FULLREF/'validation.json')
 need(fv['manifest_sha256']==PINS['full_reference_manifest'][1] and all(x is True for x in fv['checks'].values()),'FULL_REFERENCE_CACHE_VALIDATED')
 fentries={(e['query_id'],int(e['execution_ordinal'])):e for e in fm['records']}
 sm,sv=read(PINS['specificity_input_manifest'][0]),read(PINS['specificity_input_validation'][0])
 need(sv['status']=='RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_INDEPENDENT_VALIDATION_PASS' and sv['manifest']['sha256']==PINS['specificity_input_manifest'][1],'SPECIFICITY_INPUT_QUALIFIED')
 need(sm['sources']['program']['sha256']==PINS['specificity_input_program'][1],'SPECIFICITY_SCORER_BINDING')
 X=module(SPEC,'frozen_specificity_input_math')
 output=[]
 for entry in entries:
  mp=Path(entry['path']).resolve();ALLOWED_EPISODES.add(str(mp));mp=path_of(entry);meta=read(mp)
  need(meta['kind']=='FULL' and meta['role']=='TRAIN','ONLY_TRAIN_METADATA')
  ap=Path(meta['arrays']['path']).resolve();ALLOWED_EPISODES.add(str(ap));ap=path_of(meta['arrays'])
  fe=fentries[(meta['query_id'],int(meta['execution_ordinal']))];fp=Path(fe['path']).resolve();ALLOWED_EPISODES.add(str(fp));fp=path_of(fe);fmeta=read(fp)
  with np.load(ap,allow_pickle=False) as z:a=z['a'].copy();w=z['wq'].copy();q=z['q_tokens'].copy();positions=z['candidate_positions'].tolist()
  need(a.dtype==w.dtype==np.float64 and a.shape==w.shape==(128,768),'FP64_FULL_AXIS')
  need(np.isfinite(a).all() and np.isfinite(w).all() and (w>=0).all(),'FINITE_SUPPORT_PROFILE')
  need(positions==list(range(128)) and len(set(meta['axis']))==128 and not {714,715}.issubset(meta['axis']),'ORIGINAL_REFERENCE_AXIS')
  need(tensor_sha(q)==meta['q_tokens_sha256']==fmeta['query_tokens_sha256'],'QUERY_TOKEN_HASH')
  need(meta['axis']==fmeta['candidate_physical_rows'] and tensor_sha(a)==fmeta['a_RIMAGE_same_source_sha256'],'DUAL_QUALIFIED_FREE_VECTOR_HASH')
  ta=torch.from_numpy(a);tw=torch.from_numpy(w);original=[]
  for g in range(128):
   f,j,h,_=X.cues(ta,tw[g],g,meta['axis'])
   original.append({'free_binary64':float(f).hex(),'J_binary64':float(j).hex(),'strongest_other_position':h,'wq_sum_binary64':float(tw[g].sum()).hex()})
  output.append({'query_id':meta['query_id'],'execution_ordinal':int(meta['execution_ordinal']),'candidate_physical_rows':meta['axis'],
   'query_tokens_sha256':tensor_sha(q),'a_sha256':tensor_sha(a),'wq_sha256':tensor_sha(w),
   'source':{'metadata':bind(mp),'arrays':bind(ap),'independent_full_reference_metadata':bind(fp)},'a':a,'wq':w,'old_FP64':original})
 need(not BLOCKED,'FORBIDDEN_READ_ATTEMPT');return output

def public_input(r):return {k:v for k,v in r.items() if k not in ('a','wq')}
def sources():
 out={'program':bind(PROGRAM),'validator':bind(VALIDATOR),'launcher':bind(LAUNCH),'plan':bind(PLAN)}
 for k,(p,s) in PINS.items():out[k]=bind(p);need(out[k]['sha256']==s,'SOURCE_PIN:'+k)
 need(scipy.__version__=='1.16.3','FROZEN_SCIPY')
 out['runtime']={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads()}
 return out

class ExactMatrix:
 def __init__(self,a):
  pairs=[[float(x).as_integer_ratio() for x in row] for row in a]
  self.exponent=max(d.bit_length()-1 for row in pairs for _,d in row)
  self.scale=1<<self.exponent
  self.rows=[[n<<(self.exponent-(d.bit_length()-1)) for n,d in row] for row in pairs]
  self.shape=a.shape

def distribution(raw):
 if raw is None:return None,'NO_SOLUTION_VECTOR'
 vals=[float(x) for x in raw]
 if not all(math.isfinite(x) for x in vals):return None,'NONFINITE_SOLUTION_VECTOR'
 positive=[(i,x.as_integer_ratio()) for i,x in enumerate(vals) if x>0]
 if not positive:return None,'NO_POSITIVE_COEFFICIENT_MASS'
 exponent=max(d.bit_length()-1 for _,(_,d) in positive)
 nums=[n<<(exponent-(d.bit_length()-1)) for _,(n,d) in positive]
 divisor=reduce(math.gcd,nums);nums=[n//divisor for n in nums];den=sum(nums)
 return {'dimension':len(vals),'nonzero_positions':[i for i,_ in positive],'numerators':[str(n) for n in nums],
  'denominator':str(den),'negative_coefficients_clipped':sum(x<0 for x in vals)},None

def sparse(d):return list(zip(d['nonzero_positions'],map(int,d['numerators'])))
def lower_bound(matrix,g,dist):
 row=matrix.rows[g];p=sparse(dist);den=int(dist['denominator'])*matrix.scale
 nums=[sum((row[i]-matrix.rows[h][i])*n for i,n in p) for h in range(128) if h!=g]
 return Fraction(min(nums),den)
def upper_bound(matrix,g,dist):
 others=[h for h in range(128) if h!=g];alpha=[(others[j],n) for j,n in sparse(dist)]
 nums=[sum((matrix.rows[g][i]-matrix.rows[h][i])*n for h,n in alpha) for i in range(matrix.shape[1])]
 return Fraction(max(nums),int(dist['denominator'])*matrix.scale)
def certificate(matrix,g,p,alpha):
 lo=lower_bound(matrix,g,p);hi=upper_bound(matrix,g,alpha);need(lo<=hi,'EXACT_WEAK_DUALITY')
 sign='POSITIVE_SEPARATION' if lo>0 else ('NONPOSITIVE_OPTIMUM' if hi<=0 else 'UNRESOLVED_SIGN')
 return {'lower':frac(lo),'upper':frac(hi),'gap':frac(hi-lo),'gap_qualified':hi-lo<=GAP,'sign_classification':sign}

def lp_call(d,dual=False):
 h,n=d.shape
 if dual:
  c=np.r_[np.zeros(h),1.];aub=np.c_[d.T,-np.ones(n)];beq=[1.];aeq=np.r_[np.ones(h),0.][None];bounds=[(0,None)]*h+[(None,None)]
 else:
  c=np.r_[np.zeros(n),-1.];aub=np.c_[-d,np.ones(h)];beq=[1.];aeq=np.r_[np.ones(n),0.][None];bounds=[(0,None)]*n+[(None,None)]
 started=time.monotonic()
 try:
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always',OptimizeWarning)
   r=linprog(c,A_ub=aub,b_ub=np.zeros(aub.shape[0]),A_eq=aeq,b_eq=beq,bounds=bounds,method='highs-ds',options=OPTIONS)
  x=None if r.x is None else [float(v).hex() for v in r.x]
  return {'attempted':True,'success':bool(r.success),'status':int(r.status),'message':str(r.message),'nit':int(r.nit),
   'raw_solution_binary64':x,'raw_minimized_objective_binary64':None if r.fun is None else float(r.fun).hex(),
   'seconds':time.monotonic()-started,'warnings':sorted(set(str(w.message) for w in ws))}
 except Exception as e:
  return {'attempted':True,'success':False,'status':None,'message':type(e).__name__+':'+str(e),'nit':None,
   'raw_solution_binary64':None,'raw_minimized_objective_binary64':None,'seconds':time.monotonic()-started,'warnings':[]}

def solution_distribution(call,dimension):
 x=call.get('raw_solution_binary64')
 if x is None:return None,'NO_SOLUTION_VECTOR'
 need(len(x)==dimension+1,'SOLVER_SOLUTION_DIMENSION')
 return distribution([float.fromhex(v) for v in x[:-1]])

def e0():
 V=module(VALIDATOR,'synthetic_independent_maxmin_validator');D=sys.modules[__name__]
 cases=[]
 # Positive, genuinely negative, exactly zero, and rounded-D endpoint test.
 for name,a,g,expected in (
  ('positive',np.array([[.75,.75],[1.,0.],[0.,1.]]+[[0.,0.]]*125),0,Fraction(1,4)),
  ('negative',np.array([[0.,0.]]+[[.5,.5]]*127),0,Fraction(-1,2)),
  ('zero',np.zeros((128,2)),0,Fraction(0)),
  ('rounded_difference',np.array([[1.,1.]]+[[2.**-54,2.**-54]]*127),0,Fraction((1<<54)-1,1<<54))):
  d=a[g]-np.delete(a,g,axis=0);pr,du=lp_call(d),lp_call(d,True)
  p,pe=solution_distribution(pr,2);alpha,ae=solution_distribution(du,127)
  need(not pe and not ae,'E0_SOLVER_VECTORS')
  c=certificate(ExactMatrix(a),g,p,alpha)
  need(unfrac(c['lower'])==unfrac(c['upper'])==expected,'E0_EXACT_BOUNDS:'+name)
  vp,vpe=V.independent_distribution(D,pr['raw_solution_binary64'][:-1],2)
  va,vae=V.independent_distribution(D,du['raw_solution_binary64'][:-1],127)
  need(vp==p and va==alpha and not vpe and not vae,'E0_INDEPENDENT_EXACT_DISTRIBUTIONS')
  need(V.expected_certificate(D,V.EndpointFractions(a),g,vp,va)==c,'E0_INDEPENDENT_BOUNDS')
  cases.append({'name':name,'expected':frac(expected),'certificate':c,'primal_success':pr['success'],'dual_success':du['success']})
 dist,err=distribution([-1e-12,.25,.75]);need(not err and dist['nonzero_positions']==[1,2] and sum(map(int,dist['numerators']))==int(dist['denominator']),'E0_CLIPPED_EXACT_SIMPLEX')
 zero,zerr=distribution([0.,0.]);need(zero is None and zerr=='NO_POSITIVE_COEFFICIENT_MASS','E0_ZERO_ORIGINAL_SUPPORT_NO_SIMPLEX')
 return {'status':'TRAIN_MAXMIN_PILOT_SYNTHETIC_E0_PASS','zero_original_support_not_a_simplex_baseline':True,'independent_probability_and_bounds_replay':True,'cases':cases,'negative_margin_explicitly_allowed':True,'raw_endpoint_not_rounded_difference_certified':True,'natural_LP_solves':0}

def preflight():
 deadline();s=sources();inputs=source_inputs();toy=e0()
 value={'status':'TRAIN_MAXMIN_PILOT_PREFLIGHT_PASS','sources':s,'contract':CONTRACT,'inputs':[public_input(r) for r in inputs],
  'E0':toy,'TRAIN_source_arrays_read':4,'natural_LP_solves':0,'TRAIN_label_reads':0,'EVAL_reads':0}
 p=PREFLIGHT/(s['program']['sha256']+'.json');write(p,value)
 print(json.dumps({'status':value['status'],'preflight':bind(p)}),flush=True);return p

def freeze():
 need(not AUTH.exists(),'AUTHORITY_EXISTS');s=sources();p=PREFLIGHT/(s['program']['sha256']+'.json');f=read(p)
 need(f['sources']==s and f['contract']==CONTRACT and f['status']=='TRAIN_MAXMIN_PILOT_PREFLIGHT_PASS','PREFLIGHT_MISMATCH')
 value={'status':'TRAIN_MAXMIN_PILOT_AUTHORIZED','sources':s,'contract':CONTRACT,'preflight':bind(p),'output':str(OUT)}
 write(AUTH,value);print(json.dumps({'status':value['status'],'authority':bind(AUTH)}),flush=True)
def authority():
 a=read(AUTH);need(a['status']=='TRAIN_MAXMIN_PILOT_AUTHORIZED' and a['sources']==sources() and a['contract']==CONTRACT and a['output']==str(OUT),'AUTHORITY_MISMATCH')
 pf=read(path_of(a['preflight']));need(pf['sources']==a['sources'] and pf['contract']==CONTRACT,'PREFLIGHT_BOUNDARY');return a,pf

def result_summary(records,complete):
 valid=[r for r in records if r['certificate'] is not None]
 return {'attempted_games':len(records),'planned_games':512,'LP_calls':sum(int(c.get('attempted',False)) for r in records for c in (r['primal_solver'],r['dual_solver'])),
  'complete_512_games':complete,'exact_bound_certificates':len(valid),'gap_qualified_games':sum(r['certificate']['gap_qualified'] for r in valid),
  'sign_counts':dict(Counter(r['certificate']['sign_classification'] for r in valid)),
  'numerical_or_incomplete_games':len(records)-len(valid),'original_zero_support_games':sum(r['original_support_distribution'] is None for r in records),
  'original_clamp_active_games':sum(float.fromhex(r['old_FP64']['wq_sum_binary64'])<1e-12 for r in records),
  'solver_seconds':sum(c.get('seconds',0.) for r in records for c in (r['primal_solver'],r['dual_solver'])),
  'all512_gap_qualified':complete and len(valid)==512 and all(r['certificate']['gap_qualified'] for r in valid)}

def postjoin(inputs,records,seal_binding):
 global ROLES_RELEASED
 need(len(records)==512 and all(r['dual_solver'].get('attempted') and r['certificate'] is not None and r['certificate']['gap_qualified'] for r in records),'NO_LABEL_JOIN_ON_PARTIAL_OR_UNQUALIFIED')
 need(path_of(seal_binding).is_file(),'PREJOIN_SEAL_REQUIRED')
 m=read(ROLE/'role_manifest.json');entries={int(e['execution_ordinal']):e for e in m['shards'] if int(e['execution_ordinal']) in EXECS}
 need(set(entries)==set(EXECS),'ONLY_SELECTED_ROLE_ENTRIES')
 ALLOWED_ROLES.update(str(Path(e['path']).resolve()) for e in entries.values());ROLES_RELEASED=True
 from rc_aslo_xf.gallery_identity_repair import build_identity_map,PHYSICAL_ROW_COUNT
 gallery=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images';paths=[]
 for directory,subdirs,names in os.walk(gallery):
  subdirs.sort();paths.extend(Path(directory)/n for n in sorted(names) if Path(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'} and (Path(directory)/n).is_file())
 need(len(paths)==PHYSICAL_ROW_COUNT,'GALLERY_CATALOGUE_COUNT')
 labels=build_identity_map([p.stem.strip() for p in paths]).labels;out=[]
 for row in inputs:
  entry=entries[row['execution_ordinal']];rp=path_of(entry);role=read(rp)
  need(role['query_id']==row['query_id'] and role['target_insertion_count']==0 and role['raw_d1_field_count']==0 and role['target_spatial_supervision_count']==0,'TRAIN_ROLE_BOUNDARY')
  positions=[i for i,p in enumerate(row['candidate_physical_rows']) if labels[p]==role['identity']];need(len(positions)==1,'UNIQUE_NATURAL_TRAIN_TARGET')
  target=positions[0];record=next(r for r in records if r['execution_ordinal']==row['execution_ordinal'] and r['candidate_position']==target)
  difference=None
  if record['certificate'] is not None and record['original_exact_support_margin'] is not None:
   difference={'lower_minus_fixed_exact':frac(unfrac(record['certificate']['lower'])-unfrac(record['original_exact_support_margin'])),
    'upper_minus_fixed_exact':frac(unfrac(record['certificate']['upper'])-unfrac(record['original_exact_support_margin']))}
  out.append({'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'target_position':target,
   'target_physical_row':row['candidate_physical_rows'][target],'target_identity':role['identity'],'supergroup':role['supergroup'],
   'role_source':bind(rp),'certificate':record['certificate'],'old_FP64_J':record['old_FP64']['J_binary64'],
   'original_exact_support_margin':record['original_exact_support_margin'],'bound_minus_original_exact_margin':difference})
 return {'scope':'ONLY_EXISTING_TRAIN_LABEL_DESCRIPTION_NOT_RETRIEVAL_SCORE','TRAIN_role_reads':4,'EVAL_role_reads':0,'targets':out}

def run():
 deadline();need(bool(os.environ.get('SLURM_JOB_ID')),'NATURAL_LP_REQUIRES_SLURM_ALLOCATION');need(not OUT.exists(),'APPEND_ONLY_RUN_EXISTS')
 started=time.monotonic();auth,pf=authority();inputs=source_inputs()
 need([public_input(r) for r in inputs]==pf['inputs'],'FROZEN_TRAIN_SOURCE_DRIFT');OUT.mkdir()
 header={'sources':auth['sources'],'authority':bind(AUTH),'contract':CONTRACT,'inputs':[public_input(r) for r in inputs],
  'start_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'TRAIN_label_reads':0}
 write(OUT/'run_header.json',header);records=[];stopped=False
 with (OUT/'progress.jsonl').open('xb') as progress:
  for row in inputs:
   matrix=ExactMatrix(row['a'])
   for g in range(128):
    if time.monotonic()-started>=BUDGET:stopped=True;break
    game_started=time.monotonic();others=[h for h in range(128) if h!=g]
    d=row['a'][g]-row['a'][others]
    pr=lp_call(d)
    if time.monotonic()-started>=BUDGET:
     du={'attempted':False,'success':False,'status':None,'message':'SOFT_BUDGET_BEFORE_DUAL','raw_solution_binary64':None,'seconds':0.};stopped=True
    else:du=lp_call(d,True)
    p,pe=solution_distribution(pr,768);alpha,ae=solution_distribution(du,127)
    old,old_error=distribution(row['wq'][g]);need(old_error in (None,'NO_POSITIVE_COEFFICIENT_MASS'),'NONFINITE_ORIGINAL_SUPPORT')
    baseline=None if old is None else frac(lower_bound(matrix,g,old))
    cert=None if pe or ae else certificate(matrix,g,p,alpha)
    # A feasible fixed-support point must be below the certified dual upper.
    if cert is not None and baseline is not None:need(unfrac(baseline)<=unfrac(cert['upper']),'FIXED_SUPPORT_VERSUS_DUAL_BOUND')
    rec={'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'candidate_position':g,'physical_row':row['candidate_physical_rows'][g],
     'opponent_positions':others,'primal_solver':pr,'dual_solver':du,'primal_distribution':p,'dual_distribution':alpha,
     'distribution_errors':{'primal':pe,'dual':ae},'original_support_distribution':old,'original_exact_support_margin':baseline,
     'old_FP64':row['old_FP64'][g],'certificate':cert,'seconds':time.monotonic()-game_started}
    records.append(rec);progress.write(encode(rec)+b'\n');progress.flush();os.fsync(progress.fileno())
    if len(records)%32==0:print(json.dumps({'event':'TRAIN_SUPPORT_MAXMIN_PROGRESS','games':len(records),'planned_games':512,'elapsed_seconds':time.monotonic()-started}),flush=True)
    if stopped:break
   if stopped:break
 (OUT/'progress.jsonl').chmod(0o444)
 complete=len(records)==512 and all(r['dual_solver'].get('attempted') for r in records)
 seal={'status':'TRAIN_MAXMIN_PREJOIN_COMPLETE' if complete else 'TRAIN_MAXMIN_PREJOIN_INCOMPLETE',
  'sources':auth['sources'],'authority':bind(AUTH),'contract':CONTRACT,'inputs':[public_input(r) for r in inputs],
  'records':records,'summary':result_summary(records,complete),'TRAIN_label_reads':0,'EVAL_reads':0,
  'elapsed_seconds_before_labels':time.monotonic()-started,'progress':bind(OUT/'progress.jsonl'),'header':bind(OUT/'run_header.json')}
 write(OUT/'prejoin_seal.json',seal)
 labels=postjoin(inputs,records,bind(OUT/'prejoin_seal.json')) if seal['summary']['all512_gap_qualified'] else {'scope':'INCOMPLETE_OR_UNQUALIFIED_NO_LABEL_JOIN','TRAIN_role_reads':0,'EVAL_role_reads':0,'targets':[]}
 status='TRAIN_MAXMIN_PILOT_COMPLETE_GAP_QUALIFIED' if seal['summary']['all512_gap_qualified'] else ('TRAIN_MAXMIN_PILOT_COMPLETE_NUMERICALLY_UNRESOLVED' if complete else 'TRAIN_MAXMIN_PILOT_INCOMPLETE')
 result={'status':status,'prejoin_seal':bind(OUT/'prejoin_seal.json'),'authority':bind(AUTH),'contract':CONTRACT,
  'summary':seal['summary'],'TRAIN_postjoin':labels,'scientific_GO_or_NO_GO':None,'retrieval_accuracy_claimed':False,
  'no_forbidden_read_attempts':not BLOCKED,'elapsed_seconds_total':time.monotonic()-started}
 write(OUT/'result.json',result);print(json.dumps({'status':status,'summary':result['summary'],'result':bind(OUT/'result.json')}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('preflight','freeze','run','validate'));a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);sys.path.insert(0,str(ROOT/'src'));sys.addaudithook(barrier)
 if a.phase=='preflight':preflight()
 elif a.phase=='freeze':freeze()
 elif a.phase=='run':run()
 else:
  authority();nonce=uuid.uuid4().hex
  env=dict(os.environ,RC_MAXMIN_VALIDATOR_NONCE=nonce,RC_MAXMIN_VALIDATOR_PARENT_PID=str(os.getpid()))
  subprocess.run([sys.executable,str(VALIDATOR),'--validate-child',nonce],check=True,env=env)
if __name__=='__main__':main()
