#!/usr/bin/env python3
"""Same-support C128 free-content and strongest-other-reference input cues.

Only cached token arithmetic is executed. Original C4 and native6 evidence must
replay exactly. PAIR supports remain2 but every support compares all128 original
reference candidates. Pools precede strongest-other-reference competition.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
OUT=ROOT/'results/rc_same_support_specificity_inputs_v1'
PLAN=ROOT/'plan/RC_SAME_SUPPORT_REFERENCE_SPECIFICITY_V1_20260910.md'
HELPER=ROOT/'programs/prepare_rc_evidence_dispersion_inputs_v1.py'
SHARED=ROOT/'results/rc_shared_query_target_prior_cache_v1'
FULLREF=ROOT/'results/rc_full64_full_reference_cache_v1'
GALLERY=ROOT.parents[2]/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
PINS={
 'qualified_C_replay_helper':(HELPER,'cb86c50b987c713400ede717990f35ef1ab4d3810fbcbdaae9fe7d82259600a5'),
 'full_reference_manifest':(FULLREF/'manifest.json','51618b74113bb8c3064e36116c6db0f6d559419d1da7a0fcb4b0b3725ce12234'),
 'full_reference_validation':(FULLREF/'validation.json','65760c2a528e31c6fe9246b6bbe76869a74d81599d5b8cdeedf60ecaaca431f8'),
 'gallery_identity_repair':(ROOT/'registry/gallery_identity_repair_v1.json','9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f'),
 'original_raw_gallery':(GALLERY,'11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc')}
BOUNDARY={'theory_name':'new HYP','scope':'SAME_SUPPORT_REFERENCE_SPECIFICITY_INPUTS_ONLY',
 'training_supervision':'ORIGINAL_PAIR_IDENTITY_SWITCH_LABELS_ONLY',
 'support':'EACH_CANDIDATE_ORIGINAL_SOFT_QUERY_VISIBILITY',
 'reference_readout':'FULL_ORIGINAL_C128_IMAGE_TOKEN_FREE_MAXSIM',
 'canonical_pool':'literal_for_g_for_h_torch_sum(wq_g_times_a_h)_over_clamped_sum_wq_g_FP64',
 'free_cue':'A_g_g', 'specificity_cue':'A_g_g_minus_max_over_original127_other_positions_A_g_h',
 'pool_precedes_competitor_max':True,'reference_identity_deduplication':False,
 'C_BIND':'ORIGINAL_WHOLE_EVIDENCE_SOURCE_POSITION_TRANSFER_OF_FREE_AND_SPECIFICITY',
 'new_training_updates':0,'new_encoder_RoMa_SAM_forwards':0,'evaluation_target_or_outcome_reads':0,
 'oracle_certificate_reads':0,'candidate_generation_or_insertion':0,'correct_identity_or_accuracy_claimed':False}
BLOCKED=[]
H=None

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def barrier(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p)
 if '/role_shards/' in s or '/rc_opened_eval_strict_' in s or (p.name=='result.json' and '/results/' in s and p.parent!=OUT):
  BLOCKED.append(s);raise RuntimeError('TARGET_OUTCOME_OR_ORACLE_READ_FORBIDDEN')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write(p,v):
 with Path(p).open('xb') as f:f.write(encode(v)+b'\n');f.flush();os.fsync(f.fileno())
 Path(p).chmod(0o444)
def path_of(d):
 p=Path(d['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==d['sha256'],'SOURCE_SHA:'+str(p));return p
def tsha(t):
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(str(t.dtype).encode()+encode(list(t.shape))+t.view(torch.uint8).numpy().tobytes()).hexdigest()
def rawsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def hexes(v):return [float(x).hex() for x in v]
def hx(e):return {k:float(e[k]).hex() for k in H.SCALARS}
def load_helper():
 global H
 need(sha(HELPER)==PINS['qualified_C_replay_helper'][1],'HELPER_PIN')
 spec=importlib.util.spec_from_file_location('frozen_dispersion_input_replay',HELPER)
 H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
def sources():
 need(PLAN.is_file(),'PLAN_REQUIRED')
 d={'plan':bind(PLAN),'program':bind(PROGRAM)}
 for k,(p,expected) in {**PINS,**H.PINS}.items():
  d[k]=bind(p);need(d[k]['sha256']==expected,'SOURCE_PIN:'+k)
 return d

def canonical_pool(a,w):
 denominator=w.sum().clamp_min(1e-12)
 return torch.stack([(w*a[h]).sum()/denominator for h in range(128)])
def independent_pool(a,w):
 outputs=[]
 for reference_position in range(128):
  weighted=torch.mul(a[reference_position],w)
  outputs.append(torch.sum(weighted)/torch.clamp(torch.sum(w),min=1e-12))
 return torch.stack(outputs)
def cues(a,w,g,axis,independent=False):
 pooled=(independent_pool if independent else canonical_pool)(a,w)
 others=[h for h in range(128) if h!=g]
 need(len(others)==127 and len(set(others))==127 and g not in others,'ALL127_SELF_EXCLUDED')
 if independent:
  best_score=max(float(pooled[h]) for h in others)
  tied=[h for h in others if float(pooled[h])==best_score]
  strongest=min(tied,key=lambda h:int(axis[h]))
 else:strongest=max(others,key=lambda h:(float(pooled[h]),-int(axis[h])))
 free=float(pooled[g]);specific=free-float(pooled[strongest])
 need(bool(torch.isfinite(pooled).all()),'FINITE_POOL')
 return free,specific,strongest,pooled

def toy(independent):
 a=torch.zeros((128,2),dtype=torch.float64)
 a[0]=torch.tensor([.75,.75]);a[1]=torch.tensor([1.,0.]);a[2]=torch.tensor([0.,1.])
 w=torch.ones(2,dtype=torch.float64);axis=list(range(128))
 free,specific,strong,pooled=cues(a,w,0,axis,independent)
 need(free==.75 and specific==.25 and strong==1,'E0_ORIGINAL_PLAN_POOL_ORDER')
 envelope=float((w*a[1:].max(dim=0).values).sum()/w.sum())
 need(free-envelope==-.25 and envelope!=float(pooled[strong]),'E0_POOL_BEFORE_MAX')
 scaled=cues(a,w*3,0,axis,independent)
 need(scaled[:3]==(free,specific,strong) and torch.equal(scaled[3],pooled),'E0_WEIGHT_NORMALIZATION')
 # PAIR local support rows need not have original candidate positions0or1.
 pair_a=torch.zeros((128,2),dtype=torch.float64)
 pair_a[5]=torch.tensor([.75,.75]);pair_a[9]=torch.tensor([.5,.5]);pair_a[127]=torch.tensor([.875,.875])
 pair5=cues(pair_a,w,5,axis,independent);pair9=cues(pair_a,w,9,axis,independent)
 need(pair5[:3]==(.75,-.125,127) and pair9[:3]==(.5,-.375,127),'E0_PAIR_LOCAL_AND_CANDIDATE_POSITION_DISTINCT')
 zero=cues(pair_a,torch.zeros_like(w),5,axis,independent)
 need(zero[0]==zero[1]==0 and torch.equal(zero[3],torch.zeros(128,dtype=torch.float64)),'E0_ZERO_SUPPORT_RETAINED')
 return {'status':'E0_SAME_SUPPORT_SELF_EXCLUSION_AND_POOL_ORDER_PASS','C128':128,'other_reference_count':127,
  'free_binary64':free.hex(),'specificity_binary64':specific.hex(),'strongest_other_position':strong,
  'pool_then_max_binary64':float(pooled[strong]).hex(),'max_then_pool_wrong_margin_binary64':(free-envelope).hex(),
  'uniform_weight_scale_invariant':True,'PAIR_support_positions':[5,9],'PAIR_strongest_other_positions':[pair5[2],pair9[2]],
  'zero_support_retained_with_F_J_zero':True,'correct_identity_or_accuracy_claimed':False}

class ReferenceCache:
 def __init__(self):
  self.gallery=torch.load(GALLERY,map_location='cpu',mmap=True,weights_only=False)['passage_emb']
  self.normalized=OrderedDict();self.hashes={}
 def image(self,physical):
  need(0<=physical<len(self.gallery),'REFERENCE_PHYSICAL_RANGE')
  r=self.gallery[physical][4:-7].contiguous()
  need(r.dtype==torch.float16 and r.ndim==2 and r.shape[1]==128,'ORIGINAL_REFERENCE_IMAGE_DOMAIN')
  if physical not in self.hashes:self.hashes[physical]={'image_token_count':len(r),'image_tokens_sha256':tsha(r)}
  return r
 def norm(self,physical):
  if physical not in self.normalized:
   self.normalized[physical]=F.normalize(self.image(physical).to(torch.float64),dim=1)
   if len(self.normalized)>128:self.normalized.popitem(last=False)
  else:self.normalized.move_to_end(physical)
  return self.normalized[physical]
 def all_maxsim(self,q,axis):
  qn=F.normalize(q.to(torch.float64),dim=1)
  return torch.stack([(qn@self.norm(int(physical)).T).max(dim=1).values for physical in axis])

def recompute(independent=False):
 from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature,symmetric
 feature=H.independent_feature if independent else candidate_feature
 scalar=H.independent_scalars if independent else H.scalar_replay
 rows,shards=H.load_rows();sm=read(SHARED/'manifest.json');sv=read(SHARED/'validation.json')
 need(sv['manifest_sha256']==H.PINS['shared_manifest'][1] and all(x is True for x in sv['checks'].values()),'QUALIFIED_SHARED_CACHE')
 fm,fv=read(FULLREF/'manifest.json'),read(FULLREF/'validation.json')
 need(fv['manifest_sha256']==PINS['full_reference_manifest'][1] and all(x is True for x in fv['checks'].values()),'QUALIFIED_FULL_REFERENCE_CACHE')
 fullentries={(e['query_id'],int(e['execution_ordinal'])):e for e in fm['records']}
 refs=ReferenceCache();records=[];counts=Counter();seen=set()
 for entry in sm['records']:
  mp=path_of(entry);meta=read(mp);key=(meta['kind'],meta['query_id'],int(meta['execution_ordinal']));row=rows[key]
  need(key not in seen,'UNIQUE_QUERY_RECORD');seen.add(key)
  need(meta['axis']==row['candidate_physical_rows'] and meta['winner']==row['base_winner_position'],'C128_AXIS_WINNER')
  axis=meta['axis'];need(len(axis)==128 and len(set(axis))==128,'ORIGINAL_C128_PHYSICAL_AXIS')
  need(not {714,715}.issubset(set(axis)),'NO_FROZEN_EQUIVALENT_REFERENCE_PAIR_IN_C128')
  need(meta.get('challenger_positions',[i for i in range(128) if i!=meta['winner']])==row['challenger_positions'],'ORIGINAL_CHALLENGER_ORDER')
  ap=path_of(meta['arrays'])
  with np.load(ap,allow_pickle=False) as pack:
   arrays={k:torch.from_numpy(pack[k].copy()) for k in ('q_tokens','a','wq','b','br','wr_mean','wr_rolled_mean')}
   positions=pack['candidate_positions'].tolist();raw=pack['raw_scores'].tolist();qs,rs=pack['qshift'].tolist(),pack['rshift'].tolist()
  q,weights,old_a=arrays['q_tokens'],arrays['wq'],arrays['a']
  need(q.dtype==torch.float16 and tsha(q)==meta['q_tokens_sha256'],'ORIGINAL_QUERY_TOKENS')
  need(weights.dtype==old_a.dtype==arrays['b'].dtype==arrays['br'].dtype==torch.float64,'FP64_LOCAL_INPUTS')
  need(weights.shape==old_a.shape==arrays['b'].shape==arrays['br'].shape and weights.shape[1]==len(q),'LOCAL_QUERY_AXIS')
  need(positions==[c['candidate_position'] for c in meta['candidates']],'SUPPORT_SUBAXIS')
  need(hexes(raw)==hexes(row['base_scores']) and len(raw)==128,'RAW_BITS')
  source={'metadata':bind(mp),'arrays':bind(ap)}
  if key[0]=='FULL':
   need(positions==list(range(128)),'FULL128_SUPPORT_AXIS')
   fmp=path_of(fullentries[(key[1],key[2])]);fmeta=read(fmp)
   need(fmeta['candidate_physical_rows']==axis and fmeta['query_tokens_sha256']==tsha(q),'FULL_REFERENCE_AXIS')
   need(tsha(old_a)==fmeta['a_RIMAGE_same_source_sha256'],'FULL_FREE_VECTOR_INDEPENDENT_CACHE_HASH')
   all_a=old_a;source['qualified_full_reference_metadata']=bind(fmp)
   counts['FULL_cached_free_vectors_source_qualified']+=128
  else:
   need(len(positions)==2 and meta['pair_cohort']==row['pair_cohort'] and meta['switch_label']==row['switch_label'],'PAIR_ORIGINAL_SUPERVISION')
   all_a=refs.all_maxsim(q,axis)
   for local,pos in enumerate(positions):need(tsha(all_a[pos])==tsha(old_a[local]),'PAIR_TWO_ORIGINAL_FREE_VECTORS_BIT_EXACT')
   counts['PAIR_full128_free_vectors_freshly_computed']+=128;counts['PAIR_old_free_vectors_bit_exact']+=2
  need(all_a.shape==(128,len(q)) and all_a.dtype==torch.float64,'ALL_C128_FREE_VECTOR_AXIS')
  ev,free,specific,strong,poolhash={}, {}, {}, {}, {};candidates=[]
  all_reference_sources=[]
  for h,physical in enumerate(axis):
   refs.image(int(physical));all_reference_sources.append({'candidate_position':h,'physical_row':physical,**refs.hashes[int(physical)]})
  for local,c in enumerate(meta['candidates']):
   pos,physical=int(c['candidate_position']),int(c['physical_row']);w=weights[local]
   need(physical==axis[pos] and rawsha(w)==c['query_map_sha256'],'SUPPORT_MAP_SOURCE')
   rsinfo=refs.hashes[physical]
   need(rsinfo['image_tokens_sha256']==c['reference_tokens_sha256'] and rsinfo['image_token_count']==c['reference_token_count'],'ORIGINAL_REFERENCE_TOKEN_BYTES')
   expected=(1,1) if key[0]=='PAIR' and row['pair_cohort']=='BALANCED32_V1' else (max(1,len(w)//2),max(1,c['reference_token_count']//2))
   need((int(qs[local]),int(rs[local]))==expected,'ORIGINAL_MIXED_CONTROL_SHIFTS')
   need(bool(torch.isfinite(w).all()) and bool((w>=0).all()),'FINITE_NONNEGATIVE_SUPPORT')
   if key[0]=='PAIR':
    fixed=row['fixed_candidate_maps'][pos]
    need(rawsha(w)==fixed['query_map_sha256'] and rawsha(fixed['reference_map'])==c['reference_map_sha256'],'PAIR_SOURCE_MAPS')
    need(tsha(q)==fixed['query_tokens_sha256'] and c['reference_tokens_sha256']==fixed['reference_tokens_sha256'],'PAIR_SOURCE_TOKENS')
   e=scalar(w,arrays['wr_mean'][local],arrays['wr_rolled_mean'][local],arrays['b'][local],arrays['br'][local],int(qs[local]))
   need(hx(e)==hx(row['evidence']['C_PAIRED'][pos])==c['old_scalars_binary64'],'ORIGINAL_C4_BITS:'+key[1]+':'+str(pos))
   fg,jg,hg,pg=cues(all_a,w,pos,axis,independent)
   ev[pos],free[pos],specific[pos],strong[pos],poolhash[pos]=e,fg,jg,hg,tsha(pg)
   candidates.append({'candidate_position':pos,'physical_row':physical,'original_query_control_shift':int(qs[local]),
    'original_reference_control_shift':int(rs[local]),'query_map_sha256':c['query_map_sha256'],'reference_map_sha256':c['reference_map_sha256'],
    'reference_tokens_sha256':c['reference_tokens_sha256'],'free_pool_mean_binary64':fg.hex(),'specificity_margin_binary64':jg.hex(),
    'strongest_other_position':hg,'strongest_other_physical_row':axis[hg], 'pooled_all128_sha256':tsha(pg),
    'strongest_other_pool_binary64':float(pg[hg]).hex(),'wq_sum_binary64':float(w.sum()).hex()})
   counts[key[0]+'_support_occurrences']+=1;counts['original_C4_scalar_bit_exact_checks']+=4;counts['same_support_reference_pools']+=128
  modes=('REAL','CBIND') if key[0]=='FULL' else ('REAL',);features={}
  for mode in modes:
   donor={p:int(row['cbind_source_positions'][p]) for p in ev} if mode=='CBIND' else {p:p for p in ev}
   need(set(donor.values())==set(ev),'WHOLE_EVIDENCE_DONOR_PERMUTATION')
   e={p:ev[donor[p]] for p in ev};f={p:free[donor[p]] for p in ev};j={p:specific[donor[p]] for p in ev}
   native=[list(map(float,feature(raw,e,c,row['base_winner_position']))) for c in row['challenger_positions']]
   expected=row[('real_' if mode=='REAL' else 'cbind_')+'native_features']['C_PAIRED']
   need([hexes(x) for x in native]==[hexes(x) for x in expected],'ORIGINAL_NATIVE6_BITS:'+mode)
   features[mode]={'ORIGINAL_C':[hexes(x) for x in native]}
   for name,values in (('FREE',f),('SPECIFICITY',j)):
    extended=[]
    for x,c in zip(native,row['challenger_positions']):
     a,b=values[c],values[row['base_winner_position']]
     extra=(a-b)/(abs(a)+abs(b)+1e-12) if independent else symmetric(a,b)
     extended.append(hexes(x+[extra]))
    features[mode][name]=extended;counts['new_seventh_feature_checks']+=len(extended)
   counts['original_native6_bit_exact_checks']+=len(native)*6
  out={'kind':key[0],'query_id':key[1],'execution_ordinal':key[2],'candidate_physical_rows':axis,
   'base_winner_position':row['base_winner_position'],'challenger_positions':row['challenger_positions'],'base_scores_binary64':hexes(raw),
   'evidence_binary64':{str(p):hx(e) for p,e in ev.items()},'free_pool_mean_binary64':{str(p):f.hex() for p,f in free.items()},
   'specificity_margin_binary64':{str(p):j.hex() for p,j in specific.items()},'strongest_other_position':{str(p):h for p,h in strong.items()},
   'feature_binary64':features,'candidates':candidates,'all128_reference_sources':all_reference_sources,
   'all128_free_maxsim_sha256':tsha(all_a),'query_tokens_sha256':tsha(q),'source':source}
  if key[0]=='PAIR':out.update(pair_cohort=row['pair_cohort'],pair_row_ordinal=row['pair_row_ordinal'],switch_label=bool(row['switch_label']))
  else:out['cbind_source_positions']=row['cbind_source_positions']
  records.append(out);counts[key[0]+'_query_count']+=1
  if counts[key[0]+'_query_count']%16==0:print(json.dumps({'event':'SAME_SUPPORT_INPUT_PROGRESS','kind':key[0],'queries':counts[key[0]+'_query_count'],'independent':independent}),flush=True)
 need(set(rows)==seen and counts['FULL_query_count']==counts['PAIR_query_count']==64,'FINAL_QUERY_POPULATION')
 need(counts['FULL_support_occurrences']==8192 and counts['PAIR_support_occurrences']==128,'FINAL_SUPPORT_POPULATION')
 need(not BLOCKED,'FORBIDDEN_READ_ATTEMPT')
 return {'records':records,'counts':dict(counts),'source_shards':shards,'E0':toy(independent)}

def validate_child(nonce):
 need(nonce==os.environ.get('RC_SPECIFICITY_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('RC_SPECIFICITY_VALIDATOR_PARENT_PID'),'EXPLICIT_SUBPROCESS_REQUIRED')
 need(not (OUT/'validation.json').exists(),'APPEND_ONLY_VALIDATION')
 sealed=sources();m=read(OUT/'manifest.json');r=read(OUT/'result.json')
 need(m['sources']==r['sources']==sealed and r['boundary']==BOUNDARY,'SOURCE_PROVENANCE')
 need(m['result']==bind(OUT/'result.json'),'RESULT_SHA')
 replay=recompute(independent=True)
 for k in replay:need(replay[k]==r[k],'INDEPENDENT_REPLAY:'+k)
 need(bind(PLAN)==sealed['plan'] and bind(PROGRAM)==sealed['program'],'SOURCE_DRIFT_DURING_VALIDATION')
 v={'status':'RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_INDEPENDENT_VALIDATION_PASS','sources':sealed,
  'manifest':bind(OUT/'manifest.json'),'result':bind(OUT/'result.json'),'counts':replay['counts'],'boundary':BOUNDARY,
  'fresh_explicit_subprocess':True,'validator_pid':os.getpid(),'validator_parent_pid':os.getppid(),
  'independent_literal_pool_old_scalar_and_feature_arithmetic':True,'PAIR_all128_free_vectors_recomputed_from_original_tokens':True,
  'FULL_all128_free_vectors_reuse_two_qualified_cache_hashes':True,'E0':replay['E0']}
 write(OUT/'validation.json',v);print(json.dumps({'status':v['status'],'counts':v['counts'],'validation':bind(OUT/'validation.json')}),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
 m.add_argument('--produce',action='store_true');m.add_argument('--validate',action='store_true');m.add_argument('--validator-child');args=p.parse_args()
 torch.set_num_threads(8);torch.set_num_interop_threads(1);sys.path.insert(0,str(ROOT/'src'));sys.addaudithook(barrier);load_helper()
 if args.validate:
  nonce=uuid.uuid4().hex;env=dict(os.environ,RC_SPECIFICITY_VALIDATOR_NONCE=nonce,RC_SPECIFICITY_VALIDATOR_PARENT_PID=str(os.getpid()))
  subprocess.run([sys.executable,str(PROGRAM),'--validator-child',nonce],check=True,env=env)
 elif args.validator_child:
  with torch.inference_mode():validate_child(args.validator_child)
 else:
  need(not OUT.exists(),'APPEND_ONLY_OUTPUT_EXISTS');sealed=sources()
  with torch.inference_mode():replay=recompute()
  need(bind(PLAN)==sealed['plan'] and bind(PROGRAM)==sealed['program'],'SOURCE_DRIFT_DURING_PRODUCTION')
  OUT.mkdir();r={'status':'RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_REPLAY_PASS','sources':sealed,'boundary':BOUNDARY,**replay}
  write(OUT/'result.json',r);write(OUT/'manifest.json',{'status':'RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_PRODUCED','sources':sealed,'result':bind(OUT/'result.json'),'counts':replay['counts'],'producer_pid':os.getpid(),'boundary':BOUNDARY})
  print(json.dumps({'status':r['status'],'counts':r['counts'],'result':bind(OUT/'result.json')}),flush=True)
if __name__=='__main__':main()
