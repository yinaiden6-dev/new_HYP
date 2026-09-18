#!/usr/bin/env python3
"""Complete label-free max-min cache:8320 games,512 frozen pilot reuses.

Eight single-thread workers; no natural LP in preflight/freeze. Every cached
profile and certificate remains source-bound. Any incomplete/uncertified game
closes the subsequent head stage. Only exact L rounded once to FP64 becomes T.
"""
from __future__ import annotations
import argparse
from collections import Counter,OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import scipy
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
VALIDATOR=ROOT/'programs/validate_rc_reference_support_maxmin_cache_v1.py'
LAUNCH=ROOT/'slurm/rc_reference_support_maxmin_cache_v1_dev_cpuonly_30m.sbatch'
PLAN=ROOT/'plan/RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md'
OUT=ROOT/'results/rc_reference_support_maxmin_cache_v1'
PREFLIGHT=ROOT/'results/rc_reference_support_maxmin_cache_v1_preflight'
AUTH=ROOT/'registry/rc_reference_support_maxmin_cache_authority_v1_20260910.json'
SHARED=ROOT/'results/rc_shared_query_target_prior_cache_v1'
FULLREF=ROOT/'results/rc_full64_full_reference_cache_v1'
SPEC=ROOT/'results/rc_same_support_specificity_inputs_v1'
PILOT=ROOT/'results/rc_reference_support_maxmin_train_pilot_v1'
GALLERY=ROOT.parents[2]/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
PINS={
 'pilot_operator':(ROOT/'programs/run_rc_reference_support_maxmin_train_pilot_v1.py','a554f3ed002119d39ff2548aff96b277fb24b7d5bb814f3c5218b18bfbb48dea'),
 'pilot_independent_math':(ROOT/'programs/validate_rc_reference_support_maxmin_train_pilot_v1.py','96a63e78ad74ee5490907b180ba30223ac169198f39bd4c248667abf50f13772'),
 'pilot_prejoin_seal':(PILOT/'prejoin_seal.json','d7347bcfcd0132a14b717db95cce561546470077dd623ed20c2e822282e113ee'),
 'pilot_validation':(PILOT/'validation.json','75bf7b3c73bafa4e5279912e40ae2be9e94db79ade0cf7b69e862870d13e9426'),
 'shared_manifest':(SHARED/'manifest.json','f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c'),
 'shared_validation':(SHARED/'validation.json','0258615f000d0d82f757f57298b803420395ebc3a35e3a46bfa53c3116b7488f'),
 'full_reference_manifest':(FULLREF/'manifest.json','51618b74113bb8c3064e36116c6db0f6d559419d1da7a0fcb4b0b3725ce12234'),
 'full_reference_validation':(FULLREF/'validation.json','65760c2a528e31c6fe9246b6bbe76869a74d81599d5b8cdeedf60ecaaca431f8'),
 'specificity_program':(ROOT/'programs/prepare_rc_same_support_specificity_inputs_v1.py','4bd77d8e7aaab3f51155f2721dbba553e899b1b7e28f7dde138b586509320627'),
 'specificity_result':(SPEC/'result.json','13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2'),
 'specificity_manifest':(SPEC/'manifest.json','378dbf79e540653f6bb1b555d3e1079c7153cd350ed7d689fc3b147cd728c574'),
 'specificity_validation':(SPEC/'validation.json','545b10d603064e2df9d746e9e16fdebdb64de2dc56048aced04c44bc76e48a75'),
 'raw_gallery':(GALLERY,'11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc'),
 'identity_repair':(ROOT/'registry/gallery_identity_repair_v1.json','9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f')}
WORKERS=8
SOFT_RUN_SECONDS=1500.
CONTRACT={'theory_name':'new HYP','scope':'LABEL_FREE_MAXMIN_CACHE_NOT_RETRIEVAL_ACCURACY',
 'FULL_queries':64,'PAIR_queries':64,'FULL_games':8192,'PAIR_games':128,'games':8320,'pilot_reuses':512,'new_games':7808,
 'opponents_per_game':127,'query_token_lengths':[720,768],'worker_processes':8,'threads_per_worker':1,
 'solver':'FROZEN_PILOT_HIGHS_DS_PRIMAL_AND_DUAL_OPTIONS','exact_math':'ORIGINAL_BINARY64_ENDPOINTS_NOT_ROUNDED_D',
 'qualification_gap':{'numerator':'1','denominator':'100000000'},'T':'float(Fraction(exact_lower_L)).hex();nearest_binary64;retain_negative',
 'PAIR_profiles':'PERSIST_FULL128_AND_MATCH_OLD_FULL_MATRIX_SHA_AND_ORIGINAL2_VECTORS',
 'cache_soft_run_seconds':SOFT_RUN_SECONDS,'roles_or_EVAL_outcomes_read':0,'oracle_reads':0,'new_training_updates':0,
 'scientific_GO_or_NO_GO':None,'readout_gate':'ALL8320_PRESENT_GAP_QUALIFIED_PLUS_INDEPENDENT_VALIDATION'}
P=V=X=None
_BOOTED=False
_BLOCKED=[]
_GALLERY=None
_REFS=OrderedDict()
_PILOT_RECORDS=None
_PREFLIGHT_BARRIER=None

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def barrier(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p);bad=None
 if any(x in s.lower() for x in ('/role_shards/','d1_mi','d1-mi','grozi','gisc_prerecall_universe','/rc_opened_eval_strict_')):bad='ROLE_OR_PROTECTED_OR_ORACLE'
 if p.name=='result.json' and '/RC/results/' in s and p!=SPEC/'result.json' and not p.is_relative_to(OUT):bad='OTHER_RESULT'
 if p.suffix=='.pt' and p!=GALLERY:bad='UNAUTHORIZED_TENSOR_PAYLOAD'
 if p.suffix=='.npz' and not(p.is_relative_to(SHARED) or p.is_relative_to(OUT)):bad='UNAUTHORIZED_PROFILE_ARCHIVE'
 if bad:_BLOCKED.append(s);raise RuntimeError(bad)
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
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

def tsha(t):
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(str(t.dtype).encode()+encode(list(t.shape))+t.numpy().tobytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def bootstrap():
 global P,V,X,_BOOTED
 if _BOOTED:return
 torch.set_num_threads(1);torch.set_num_interop_threads(1);sys.path.insert(0,str(ROOT/'src'));sys.addaudithook(barrier)
 for k in ('pilot_operator','pilot_independent_math','specificity_program'):need(sha(PINS[k][0])==PINS[k][1],'FROZEN_MATH_PIN:'+k)
 P=module(PINS['pilot_operator'][0],'frozen_maxmin_pilot_math')
 V=module(PINS['pilot_independent_math'][0],'frozen_maxmin_fraction_validation_math')
 X=module(PINS['specificity_program'][0],'frozen_same_support_specificity_math')
 need(scipy.__version__=='1.16.3' and P.GAP==Fraction(1,100000000),'SOLVER_MATH_VERSION')
 need(P.OPTIONS=={'presolve':True,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':1,'parallel':False,'random_seed':0,'time_limit':5.0},'FROZEN_SOLVER_OPTIONS')
 _BOOTED=True

def sources():
 s={'program':bind(PROGRAM),'validator':bind(VALIDATOR),'launcher':bind(LAUNCH),'plan':bind(PLAN)}
 for k,(p,h) in PINS.items():s[k]=bind(p);need(s[k]['sha256']==h,'SOURCE_PIN:'+k)
 s['runtime']={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'threads':torch.get_num_threads()}
 return s

def inventory():
 sm,sv=read(SHARED/'manifest.json'),read(SHARED/'validation.json')
 need(sv['manifest_sha256']==PINS['shared_manifest'][1] and all(v is True for v in sv['checks'].values()),'SHARED_VALIDATED')
 fm,fv=read(FULLREF/'manifest.json'),read(FULLREF/'validation.json')
 need(fv['manifest_sha256']==PINS['full_reference_manifest'][1] and all(v is True for v in fv['checks'].values()),'FULL_REF_VALIDATED')
 sr,sv=read(SPEC/'result.json'),read(SPEC/'validation.json')
 need(sv['status']=='RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_INDEPENDENT_VALIDATION_PASS' and sv['result']['sha256']==PINS['specificity_result'][1],'SPECIFICITY_VALIDATED')
 bykey={(r['kind'],r['query_id'],int(r['execution_ordinal'])):r for r in sr['records']};need(len(bykey)==128,'SOURCE_UNIQUE_KEYS')
 fkeys={(e['query_id'],int(e['execution_ordinal'])):e for e in fm['records']}
 pv=read(PILOT/'validation.json');ps=read(PILOT/'prejoin_seal.json')
 need(pv['status']=='TRAIN_MAXMIN_PILOT_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS' and pv['summary']['all512_gap_qualified'] is True,'PILOT_QUALIFIED')
 need(pv['prejoin_seal']['sha256']==PINS['pilot_prejoin_seal'][1] and ps['TRAIN_label_reads']==0 and ps['EVAL_reads']==0,'PILOT_PREJOIN_ONLY')
 need(pv['result']['sha256']=='d2f059554bbe62e74dd112b7802449063dae4aa74204a0906c86e82917c941b0','PILOT_FINAL_LINEAGE_METADATA_ONLY')
 pinputs={r['execution_ordinal']:r for r in ps['inputs']}
 descriptors=[]
 for entry in sm['records']:
  mp=path_of(entry);meta=read(mp);key=(meta['kind'],meta['query_id'],int(meta['execution_ordinal']));old=bykey[key]
  positions=[int(c['candidate_position']) for c in meta['candidates']]
  need(meta['axis']==old['candidate_physical_rows'] and positions==[c['candidate_position'] for c in old['candidates']],'SOURCE_AXIS_AND_SCORE_SUBAXIS')
  need(len(meta['axis'])==128 and len(set(meta['axis']))==128 and not{714,715}.issubset(meta['axis']),'FULL128_IDENTITY_AXIS')
  n=meta['array_schema']['a']['shape'][1];need(n in (720,768),'SUPPORTED_QUERY_TOKEN_LENGTH')
  source={'metadata':bind(mp),'arrays':meta['arrays']}
  if key[0]=='FULL':
   need(positions==list(range(128)),'FULL_SCORE_AXIS');fp=path_of(fkeys[(key[1],key[2])]);fmeta=read(fp)
   need(fmeta['candidate_physical_rows']==meta['axis'] and fmeta['query_tokens_sha256']==meta['q_tokens_sha256'],'FULL_REF_TOKEN_AXIS')
   need(fmeta['a_RIMAGE_same_source_sha256']==old['all128_free_maxsim_sha256'],'FULL_PROFILE_SHA_AGREEMENT')
   source['independent_full_reference_metadata']=bind(fp)
  else:need(len(positions)==2 and meta['pair_cohort']==old['pair_cohort'] and meta['switch_label']==old['switch_label'],'PAIR_ORIGINAL_INTERFACE')
  d={'kind':key[0],'query_id':key[1],'execution_ordinal':key[2],'query_token_count':n,
   'query_tokens_sha256':meta['q_tokens_sha256'],'candidate_physical_rows':meta['axis'],'score_positions':positions,
   'base_winner_position':old['base_winner_position'],'challenger_positions':old['challenger_positions'],
   'expected_all128_a_sha256':old['all128_free_maxsim_sha256'],'fixed_J_binary64':old['specificity_margin_binary64'],
   'source':source,'episode_name':key[0]+'_'+format(key[2],'04d'),'reuse_pilot':key[0]=='FULL' and key[2] in pinputs}
  if key[0]=='FULL':d['cbind_source_positions']=old['cbind_source_positions']
  else:d.update(pair_cohort=old['pair_cohort'],pair_row_ordinal=old['pair_row_ordinal'],switch_label=old['switch_label'])
  if d['reuse_pilot']:
   pi=pinputs[key[2]]
   need(pi['candidate_physical_rows']==d['candidate_physical_rows'] and pi['query_tokens_sha256']==d['query_tokens_sha256'] and pi['a_sha256']==d['expected_all128_a_sha256'],'PILOT_REUSE_AXIS_PROFILE')
   need(pi['source']['metadata']==d['source']['metadata'] and pi['source']['arrays']==bind(path_of(d['source']['arrays'])),'PILOT_REUSE_SOURCE_BITS')
  descriptors.append(d)
 counts={'FULL_queries':sum(d['kind']=='FULL' for d in descriptors),'PAIR_queries':sum(d['kind']=='PAIR' for d in descriptors),
  'games':sum(len(d['score_positions']) for d in descriptors),'pilot_reuses':sum(len(d['score_positions']) for d in descriptors if d['reuse_pilot']),
  'length_counts':dict(Counter(str(d['query_token_count']) for d in descriptors))}
 need(counts['FULL_queries']==counts['PAIR_queries']==64 and counts['games']==8320 and counts['pilot_reuses']==512,'INVENTORY_COVERAGE')
 return descriptors,counts

def pilot_records():
 global _PILOT_RECORDS
 if _PILOT_RECORDS is None:
  need(sha(PILOT/'prejoin_seal.json')==PINS['pilot_prejoin_seal'][1],'PILOT_PREJOIN_PIN')
  ps=read(PILOT/'prejoin_seal.json')
  _PILOT_RECORDS={(r['execution_ordinal'],r['candidate_position']):r for r in ps['records']}
  need(len(_PILOT_RECORDS)==512,'PILOT_RECORD_COUNT')
 return _PILOT_RECORDS

def load_local(desc):
 meta=read(path_of(desc['source']['metadata']));ap=path_of(desc['source']['arrays'])
 with np.load(ap,allow_pickle=False) as z:q=torch.from_numpy(z['q_tokens'].copy());old_a=torch.from_numpy(z['a'].copy());w=torch.from_numpy(z['wq'].copy());pos=z['candidate_positions'].tolist()
 n=desc['query_token_count'];need(q.shape==(n,128) and q.dtype==torch.float16 and tsha(q)==desc['query_tokens_sha256'],'QUERY_SOURCE')
 need(pos==desc['score_positions'] and old_a.shape==w.shape==(len(pos),n) and old_a.dtype==w.dtype==torch.float64,'LOCAL_PROFILE_AXIS')
 need(bool(torch.isfinite(old_a).all()) and bool(torch.isfinite(w).all()) and bool((w>=0).all()),'FINITE_INPUTS')
 return meta,ap,q,old_a,w

def full_pair_profile(q,axis):
 global _GALLERY
 if _GALLERY is None:_GALLERY=torch.load(GALLERY,map_location='cpu',mmap=True,weights_only=False)['passage_emb']
 qn=F.normalize(q.double(),dim=1);rows=[]
 for physical in axis:
  if physical not in _REFS:
   r=_GALLERY[physical][4:-7].contiguous();need(r.dtype==torch.float16 and r.ndim==2 and r.shape[1]==128,'IMAGE_TOKEN_SLICE')
   _REFS[physical]=F.normalize(r.double(),dim=1)
   if len(_REFS)>128:_REFS.popitem(last=False)
  else:_REFS.move_to_end(physical)
  rows.append((qn@_REFS[physical].T).max(dim=1).values)
 return torch.stack(rows)

def profiles(desc,produce=False):
 meta,ap,q,old_a,w=load_local(desc);folder=OUT/'episodes'/desc['episode_name']
 if desc['kind']=='FULL':a=old_a;source={**bind(ap),'array_key':'a','a_sha256':tsha(a),'reused_original_array':True}
 else:
  pp=folder/'profiles.npz'
  if produce:
   a=full_pair_profile(q,desc['candidate_physical_rows'])
   need(tsha(a)==desc['expected_all128_a_sha256'],'PAIR_FULL_PROFILE_PREVIOUSLY_SEALED_HASH')
   folder.mkdir(parents=True,exist_ok=True)
   with pp.open('xb') as f:np.savez(f,a=a.numpy());f.flush();os.fsync(f.fileno())
   pp.chmod(0o444)
  else:
   with np.load(pp,allow_pickle=False) as z:a=torch.from_numpy(z['a'].copy())
  source={**bind(pp),'array_key':'a','a_sha256':tsha(a),'reused_original_array':False}
  for i,pos in enumerate(desc['score_positions']):need(tsha(a[pos])==tsha(old_a[i]),'PAIR_ORIGINAL_TWO_FREE_VECTORS')
 need(a.dtype==torch.float64 and a.shape==(128,desc['query_token_count']) and tsha(a)==desc['expected_all128_a_sha256'],'FULL_C128_PROFILE_BITS')
 oldfp={}
 for local,g in enumerate(desc['score_positions']):
  f,j,h,_=X.cues(a,w[local],g,desc['candidate_physical_rows'])
  need(float(j).hex()==desc['fixed_J_binary64'][str(g)],'FIXED_J_BINARY64_REPLAY')
  oldfp[g]={'free_binary64':float(f).hex(),'J_binary64':float(j).hex(),'strongest_other_position':h,'wq_sum_binary64':float(w[local].sum()).hex()}
 return a.numpy(),w.numpy(),source,oldfp

def game(desc,a,w,oldfp,matrix,local,g):
 others=[h for h in range(128) if h!=g];d=a[g]-a[others];started=time.monotonic()
 pr,du=P.lp_call(d),P.lp_call(d,True);p,pe=P.solution_distribution(pr,a.shape[1]);alpha,ae=P.solution_distribution(du,127)
 original,oe=P.distribution(w[local]);need(oe in(None,'NO_POSITIVE_COEFFICIENT_MASS'),'ORIGINAL_SUPPORT_DISTRIBUTION')
 baseline=None if original is None else P.frac(P.lower_bound(matrix,g,original))
 cert=None if pe or ae else P.certificate(matrix,g,p,alpha)
 if cert is not None and baseline is not None:need(P.unfrac(baseline)<=P.unfrac(cert['upper']),'FIXED_SUPPORT_VS_DUAL')
 return {'query_id':desc['query_id'],'execution_ordinal':desc['execution_ordinal'],'candidate_position':g,
  'physical_row':desc['candidate_physical_rows'][g],'opponent_positions':others,'primal_solver':pr,'dual_solver':du,
  'primal_distribution':p,'dual_distribution':alpha,'distribution_errors':{'primal':pe,'dual':ae},
  'original_support_distribution':original,'original_exact_support_margin':baseline,'old_FP64':oldfp[g],
  'certificate':cert,'seconds':time.monotonic()-started}

def produce_episode(payload):
 desc,stop_at=payload;bootstrap();started=time.monotonic();folder=OUT/'episodes'/desc['episode_name'];folder.mkdir(parents=True,exist_ok=False)
 try:
  a,w,profile,oldfp=profiles(desc,produce=True);matrix=P.ExactMatrix(a);games=[];reused=[]
  for local,g in enumerate(desc['score_positions']):
   if time.time()>=stop_at:break
   if desc['reuse_pilot']:
    r=pilot_records()[(desc['execution_ordinal'],g)]
    need(r['query_id']==desc['query_id'] and r['physical_row']==desc['candidate_physical_rows'][g] and r['old_FP64']==oldfp[g],'PILOT_RECORD_SOURCE_MATCH')
    need(r['certificate'] is not None and r['certificate']['gap_qualified'],'PILOT_RECORD_QUALIFIED');reused.append(g)
   else:r=game(desc,a,w,oldfp,matrix,local,g)
   games.append(r)
  qualified=len(games)==len(desc['score_positions']) and all(r['certificate'] is not None and r['certificate']['gap_qualified'] for r in games)
  cert={'schema':'maxmin_cache_certificates_v1','kind':desc['kind'],'query_id':desc['query_id'],'execution_ordinal':desc['execution_ordinal'],
   'profile_source':profile,'operator_source':bind(PINS['pilot_operator'][0]),'pilot_reused_positions':reused,
   'pilot_source':bind(PILOT/'prejoin_seal.json') if reused else None,'games':games,'all_game_gap_qualified':qualified}
  write(folder/'certificates.json',cert)
  record={k:v for k,v in desc.items() if k not in('episode_name','reuse_pilot')}
  record.update(schema='maxmin_cache_record_v1',profile_source=profile,certificates=bind(folder/'certificates.json'),
   T_binary64={str(r['candidate_position']):float(P.unfrac(r['certificate']['lower'])).hex() for r in games if r['certificate'] is not None},
   all_game_gap_qualified=qualified,completed_games=len(games),pilot_reused_games=len(reused),new_games=len(games)-len(reused),seconds=time.monotonic()-started)
  write(folder/'record.json',record)
  return {'kind':desc['kind'],'query_id':desc['query_id'],'execution_ordinal':desc['execution_ordinal'],**bind(folder/'record.json')}
 except Exception as e:
  error={'kind':desc['kind'],'query_id':desc['query_id'],'execution_ordinal':desc['execution_ordinal'],'error':type(e).__name__+':'+str(e),'seconds':time.monotonic()-started}
  write(folder/'engineering_failure.json',error);return {'failure':bind(folder/'engineering_failure.json'),**{k:error[k] for k in ('kind','query_id','execution_ordinal')}}

def synthetic():
 base=P.e0()
 # Explicit variable-length certificate and exact T rounding checks; no natural LP.
 checks=[]
 for n in (720,768):
  a=np.zeros((128,n));a[0]=.75;a[1]=1.;m=P.ExactMatrix(a)
  p,_=P.distribution([1.]+[0.]*(n-1));alpha,_=P.distribution([1.]+[0.]*126)
  c=P.certificate(m,0,p,alpha);need(P.unfrac(c['lower'])==P.unfrac(c['upper'])==Fraction(-1,4),'VARIABLE_N_NEGATIVE_BOUND')
  ic=V.expected_certificate(P,V.EndpointFractions(a),0,p,alpha);need(ic==c,'VARIABLE_N_INDEPENDENT_BOUNDS')
  need(float(P.unfrac(c['lower'])).hex()==(-.25).hex(),'NEGATIVE_T_RETAINED')
  checks.append({'query_tokens':n,'T_binary64':(-.25).hex(),'negative_bound_preserved':True})
 return {'status':'MAXMIN_FULL_CACHE_SYNTHETIC_PASS','pilot_E0':base,'variable_lengths':checks,'natural_LP_solves':0}

def spawn_preflight_initializer(barrier_object):
 global _PREFLIGHT_BARRIER
 bootstrap();_PREFLIGHT_BARRIER=barrier_object

def spawn_synthetic_worker(n):
 _PREFLIGHT_BARRIER.wait(timeout=90)
 a=np.zeros((128,n));a[0]=.75;a[1]=1.
 p,_=P.distribution([1.]+[0.]*(n-1));alpha,_=P.distribution([1.]+[0.]*126)
 c=P.certificate(P.ExactMatrix(a),0,p,alpha)
 ic=V.expected_certificate(P,V.EndpointFractions(a),0,p,alpha)
 need(c==ic and float(P.unfrac(c['lower'])).hex()==(-.25).hex(),'SPAWN_SYNTHETIC_EXACT_BOUND')
 return {'pid':os.getpid(),'threads':torch.get_num_threads(),'n':n,'T_binary64':(-.25).hex()}

def spawn_preflight():
 context=mp.get_context('spawn');b=context.Barrier(WORKERS)
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=context,initializer=spawn_preflight_initializer,initargs=(b,)) as executor:
  rows=list(executor.map(spawn_synthetic_worker,[720,768]*4))
 need(len({r['pid'] for r in rows})==8 and all(r['threads']==1 for r in rows),'EIGHT_SINGLE_THREAD_SPAWN_WORKERS')
 return {'status':'EIGHT_SINGLE_THREAD_SPAWN_SYNTHETIC_PASS','workers':8,'threads_each':1,'natural_LP_solves':0,
  'cases':[{'n':r['n'],'T_binary64':r['T_binary64']} for r in rows]}

def preflight():
 P.deadline();s=sources();ds,counts=inventory();value={'status':'MAXMIN_CACHE_PREFLIGHT_PASS','sources':s,'contract':CONTRACT,
  'inventory':ds,'counts':counts,'synthetic':synthetic(),'spawn_preflight':spawn_preflight(),'natural_LP_solves':0,'role_reads':0}
 key=hashlib.sha256(encode(s)).hexdigest();p=PREFLIGHT/(key+'.json');write(p,value)
 print(json.dumps({'status':value['status'],'counts':counts,'preflight':bind(p)}),flush=True);return p

def freeze():
 need(not AUTH.exists(),'AUTHORITY_EXISTS');s=sources();p=PREFLIGHT/(hashlib.sha256(encode(s)).hexdigest()+'.json');f=read(p)
 need(f['sources']==s and f['contract']==CONTRACT and f['status']=='MAXMIN_CACHE_PREFLIGHT_PASS','PREFLIGHT_CLOSURE')
 a={'status':'RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_AUTHORIZED','sources':s,'contract':CONTRACT,'preflight':bind(p),'output':str(OUT)}
 write(AUTH,a);print(json.dumps({'status':a['status'],'authority':bind(AUTH)}),flush=True)
def authority():
 a=read(AUTH);need(a['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_AUTHORIZED' and a['sources']==sources() and a['contract']==CONTRACT and a['output']==str(OUT),'AUTHORITY_CLOSURE')
 f=read(path_of(a['preflight']));need(f['sources']==a['sources'] and f['contract']==CONTRACT,'PREFLIGHT_BINDING');return a,f

def run():
 P.deadline();need(bool(os.environ.get('SLURM_JOB_ID')),'NATURAL_LP_REQUIRES_SLURM');need(not OUT.exists(),'APPEND_ONLY_OUTPUT')
 started=time.monotonic();a,f=authority();ds,counts=inventory();need(ds==f['inventory'] and counts==f['counts'],'FROZEN_INVENTORY')
 OUT.mkdir();write(OUT/'inventory.json',{'sources':a['sources'],'contract':CONTRACT,'records':ds,'counts':counts})
 stop_at=time.time()+SOFT_RUN_SECONDS;entries=[]
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=bootstrap) as executor:
  futures={executor.submit(produce_episode,(d,stop_at)):d for d in ds}
  for future in as_completed(futures):
   try:entry=future.result()
   except Exception as e:
    d=futures[future];failure={'kind':d['kind'],'query_id':d['query_id'],'execution_ordinal':d['execution_ordinal'],'error':'WORKER_DISPATCH:'+type(e).__name__+':'+str(e)}
    fp=OUT/'worker_failures'/(d['episode_name']+'.json');write(fp,failure);entry={'failure':bind(fp),**{k:failure[k] for k in ('kind','query_id','execution_ordinal')}}
   entries.append(entry);print(json.dumps({'event':'MAXMIN_CACHE_QUERY_CLOSED','query_records':len(entries),'total':128,'elapsed_seconds':time.monotonic()-started}),flush=True)
 bykey={(e['kind'],e['query_id'],e['execution_ordinal']):e for e in entries}
 ordered=[bykey[(d['kind'],d['query_id'],d['execution_ordinal'])] for d in ds]
 good=[e for e in ordered if 'failure' not in e];records=[read(path_of(e)) for e in good]
 total=sum(r['completed_games'] for r in records);reuse=sum(r['pilot_reused_games'] for r in records)
 ready=len(good)==128 and total==8320 and reuse==512 and all(r['all_game_gap_qualified'] for r in records)
 summary={'records':len(good),'engineering_failures':len(ordered)-len(good),'games':total,'pilot_reuses':reuse,'new_games':total-reuse,'expected_games':8320,'expected_new_games':7808}
 status='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_COMPLETE' if ready else 'RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_ENGINEERING_INCOMPLETE'
 manifest={'status':status,'sources':a['sources'],'contract':CONTRACT,'authority':bind(AUTH),'inventory':bind(OUT/'inventory.json'),
  'all8320_gap_qualified':ready,'records':good,'engineering_failures':[e for e in ordered if 'failure' in e],'summary':summary}
 write(OUT/'manifest.json',manifest)
 result={'status':status,'sources':a['sources'],'manifest':bind(OUT/'manifest.json'),'all8320_gap_qualified':ready,'summary':summary,
  'elapsed_seconds':time.monotonic()-started,'role_reads':0,'EVAL_outcome_reads':0,'new_training_updates':0,'scientific_GO_or_NO_GO':None}
 write(OUT/'result.json',result);print(json.dumps({'status':status,'summary':summary,'result':bind(OUT/'result.json')}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('preflight','freeze','run','validate'));args=p.parse_args();bootstrap()
 if args.phase=='preflight':preflight()
 elif args.phase=='freeze':freeze()
 elif args.phase=='run':run()
 else:
  authority();n=uuid.uuid4().hex;env=dict(os.environ,RC_MAXMIN_CACHE_VALIDATOR_NONCE=n,RC_MAXMIN_CACHE_VALIDATOR_PARENT_PID=str(os.getpid()))
  subprocess.run([sys.executable,str(VALIDATOR),'--validate-child',n],check=True,env=env)
if __name__=='__main__':main()
