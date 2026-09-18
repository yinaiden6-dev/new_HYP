#!/usr/bin/env python3
"""Exact, label-free token-interface witness of lost cross-reference information.

No image, task label, natural feature cache, learned weight, or scheduler is read.
This is not a claim that an encoder produces these tokens for a realizable image.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
OUT=ROOT/'results/rc_same_support_information_witness_v1'
PINS={
 'legacy_scorer':(ROOT/'src/rc_aslo_xf/reference_visibility_pv_lossless_v1.py','f5fcfa1ce6fa6582a3628035e2ba54414185160fa36515fa27fd55d8aa81adcc'),
 'legacy_feature':(ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7'),
 'specificity_interface':(ROOT/'programs/prepare_rc_same_support_specificity_inputs_v1.py','4bd77d8e7aaab3f51155f2721dbba553e899b1b7e28f7dde138b586509320627')}
SCALARS=('real_score','visibility_mass','query_control_score','reference_control_score')
BOUNDARY={'scope':'LABEL_FREE_TOKEN_INTERFACE_INFORMATION_WITNESS','natural_image_or_token_cache_reads':0,
 'task_label_reads':0,'natural_prediction_or_accuracy_reads':0,'training_updates':0,'scheduler_queries':0,
 'query_tokens':'FP16_E1_E2_E3_E4_PADDED_TO128','reference_token_norms':'EXACT_UNIT_NORM_DYADIC_FP16',
 'pixel_or_encoder_realizability_claimed':False,'natural_accuracy_gain_claimed':False,'spatial_ownership_claimed':False}
BLOCKED=[]
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def barrier(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p)
 if p.suffix in ('.pt','.npz') or '/role_shards/' in s or '/rc_opened_eval_strict_' in s or ('/RC/results/' in s and p.parent!=OUT):
  BLOCKED.append(s);raise RuntimeError('NATURAL_DATA_OR_OTHER_RESULT_READ_FORBIDDEN')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write(p,v):
 with Path(p).open('xb') as f:f.write(encode(v)+b'\n');f.flush();os.fsync(f.fileno())
 Path(p).chmod(0o444)
def read(p):return json.loads(Path(p).read_text())
def tsha(t):
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(str(t.dtype).encode()+encode(list(t.shape))+t.numpy().tobytes()).hexdigest()
def hx(x):return float(x).hex()
def hs(v):return [hx(x) for x in v]
def sources():
 out={'program':bind(PROGRAM)}
 for k,(p,s) in PINS.items():
  out[k]=bind(p);need(out[k]['sha256']==s,'SOURCE_PIN:'+k)
 return out

def reference(profile):
 r=torch.zeros((4,128),dtype=torch.float16)
 for i,a in enumerate(profile):
  r[i,i]=a
  if a==.25:r[i,64:79]=.25
  elif a==.5:r[i,64:67]=.5
  elif a==.75:r[i,64:71]=.25
  else:raise RuntimeError('UNSPECIFIED_EXACT_TOKEN_CONSTRUCTION')
 need(torch.equal(r.double().square().sum(dim=1),torch.ones(4,dtype=torch.float64)),'DYADIC_TOKEN_SQUARED_NORM_EXACT_ONE')
 need(torch.equal(F.normalize(r.double(),dim=1),r.double()),'NORMALIZATION_EXACT_IDENTITY')
 return r

def independent_score(q,r,wq,wr):
 sim=F.normalize(q.double(),dim=1)@F.normalize(r.double(),dim=1).T
 values=[];masses=[];locals_=[]
 for qw,rw in ((wq,wr),(torch.roll(wq,shifts=2),wr),(wq,torch.roll(wr,shifts=max(1,len(wr)//2)))):
  local=torch.max(torch.mul(sim,rw[None,:]),dim=1).values
  mass=torch.sqrt(torch.mean(qw)*torch.mean(rw))
  values.append(float(torch.mul(mass,torch.sum(torch.mul(qw,local)))/torch.clamp(torch.sum(qw),min=1e-12)))
  masses.append(float(mass));locals_.append(local)
 return dict(zip(SCALARS,(values[0],masses[0],values[1],values[2]))),locals_[0]
def independent_features(raw,evidence,c,w):
 mu=sum(float(x) for x in raw)/len(raw)
 std=(sum((float(x)-mu)**2 for x in raw)/len(raw))**.5
 def values(e):
  s,m,q,r=(float(e[k]) for k in SCALARS)
  return s,m,s/max(m,1e-12),s-q,s-r
 a,b=values(evidence[c]),values(evidence[w])
 return [(raw[c]-raw[w])/max(std,1e-12)]+[(x-y)/(abs(x)+abs(y)+1e-12) for x,y in zip(a,b)]
def independent_cues(profiles,w,g):
 scores=[]
 for h in range(128):scores.append(float(torch.sum(torch.mul(w,profiles[h]))/torch.clamp(torch.sum(w),min=1e-12)))
 competitor=max(scores[h] for h in range(128) if h!=g)
 return scores[g],scores[g]-competitor

def replay(independent=False):
 from rc_aslo_xf.reference_visibility_pv_lossless_v1 import _legacy_score
 from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature,symmetric
 spec=importlib.util.spec_from_file_location('frozen_specificity_interface',PINS['specificity_interface'][0])
 X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X)
 q=torch.zeros((4,128),dtype=torch.float16);q[:,:4]=torch.eye(4,dtype=torch.float16)
 need(torch.equal(F.normalize(q.double(),dim=1),q.double()),'QUERY_NORMALIZATION_EXACT')
 gref=reference((.5,.5,.5,.5));haref=reference((.25,.5,.75,.5));hbref=reference((.75,.5,.25,.5))
 dummy=torch.zeros((1,128),dtype=torch.float16);dummy[0,96]=1
 axis=list(range(128));raw=[2.,2.]+[0.]*126
 winner=max(range(128),key=lambda p:(raw[p],-axis[p]));need(winner==0,'FIXED_RAW_WINNER_H')
 weights=[torch.tensor((0.,1.,0.,1.),dtype=torch.float64),torch.tensor((1.,0.,0.,0.),dtype=torch.float64)]+[torch.ones(4,dtype=torch.float64) for _ in range(126)]
 scenes=[]
 for name,href in (('A',haref),('B',hbref)):
  references=[href,gref]+[dummy]*126
  profiles=torch.stack([(F.normalize(q.double(),dim=1)@F.normalize(r.double(),dim=1).T).max(dim=1).values for r in references])
  expected=torch.tensor((.25,.5,.75,.5) if name=='A' else (.75,.5,.25,.5),dtype=torch.float64)
  need(torch.equal(profiles[0],expected) and torch.equal(profiles[1],torch.full((4,),.5,dtype=torch.float64)),'EXACT_FREE_PROFILE_REALIZATION')
  need(torch.equal(profiles[2:],torch.zeros((126,4),dtype=torch.float64)),'OTHER126_EXACT_ZERO_PROFILES')
  evidence={};variance={};free={};specific={};native=[]
  for p,(r,wq) in enumerate(zip(references,weights)):
   wr=torch.ones(len(r),dtype=torch.float64)
   if independent:e,b=independent_score(q,r,wq,wr)
   else:
    s,m,b=_legacy_score(q,r,wq,wr)
    qc,_,_=_legacy_score(q,r,wq.roll(2),wr)
    rc,_,_=_legacy_score(q,r,wq,wr.roll(max(1,len(wr)//2)))
    e=dict(zip(SCALARS,map(float,(s,m,qc,rc))))
   evidence[p]=e
   center=e['real_score']/max(e['visibility_mass'],1e-12)
   variance[p]=float(torch.sum(wq*torch.square(b-center))/torch.clamp(torch.sum(wq),min=1e-12))
   if independent:free[p],specific[p]=independent_cues(profiles,wq,p)
   else:free[p],specific[p],_,_=X.cues(profiles,wq,p,axis)
  for c in range(1,128):
   fn=independent_features if independent else candidate_feature
   native.append(list(map(float,fn(raw,evidence,c,winner))))
  def extra(v):
   a,b=v[1],v[0]
   return (a-b)/(abs(a)+abs(b)+1e-12) if independent else symmetric(a,b)
  scenes.append({'name':name,'query_tokens_sha256':tsha(q),'reference_h_tokens_sha256':tsha(href),
   'reference_g_tokens_sha256':tsha(gref),'raw_scores_binary64':hs(raw),'raw_mean_binary64':hx(sum(raw)/128),
   'base_winner_position':winner,'h_position':0,'g_position':1,
   'profiles_h_binary64':hs(profiles[0]),'profiles_g_binary64':hs(profiles[1]),
   'all128_original_C4_binary64':{str(p):{k:hx(v) for k,v in e.items()} for p,e in evidence.items()},
   'all128_own_free_binary64':{str(p):hx(v) for p,v in free.items()},
   'all128_own_variance_binary64':{str(p):hx(v) for p,v in variance.items()},
   'all128_specificity_binary64':{str(p):hx(v) for p,v in specific.items()},
   'all127_native6_binary64':[hs(x) for x in native],
   'g_vs_h_extra_FREE_binary64':hx(extra(free)),'g_vs_h_extra_SPECIFICITY_binary64':hx(extra(specific))})
 a,b=scenes
 invariant=['query_tokens_sha256','reference_g_tokens_sha256','raw_scores_binary64','raw_mean_binary64',
  'base_winner_position','all128_original_C4_binary64','all128_own_free_binary64','all128_own_variance_binary64','all127_native6_binary64','g_vs_h_extra_FREE_binary64']
 for k in invariant:need(a[k]==b[k],'OWN_SUMMARY_MUST_BE_IDENTICAL:'+k)
 need(a['reference_h_tokens_sha256']!=b['reference_h_tokens_sha256'],'DISTINCT_INTERFACE_STATES')
 need(a['all128_specificity_binary64']['1']==hx(.25) and b['all128_specificity_binary64']['1']==hx(-.25),'G_SPECIFICITY_SIGN_FLIPS')
 need(a['all128_specificity_binary64']['0']==b['all128_specificity_binary64']['0']==hx(0.),'H_SPECIFICITY_FIXED_ZERO')
 need(a['g_vs_h_extra_FREE_binary64']==b['g_vs_h_extra_FREE_binary64']==hx(0.),'FREE_CONTRAST_FIXED_ZERO')
 need(float.fromhex(a['g_vs_h_extra_SPECIFICITY_binary64'])>0>float.fromhex(b['g_vs_h_extra_SPECIFICITY_binary64']),'SPECIFICITY_CONTRAST_SIGN_FLIPS')
 for scene in scenes:need(all(float.fromhex(v)==0 for v in scene['all128_own_variance_binary64'].values()),'ALL_OWN_VARIANCES_ZERO')
 need(not BLOCKED,'FORBIDDEN_READ_ATTEMPT')
 return {'scenes':scenes,'checks':{'FP16_dyadic_unit_norm_tokens':True,'old_C4_identical_all128':True,
  'old_native6_identical_all127':True,'own_F_identical_all128':True,'own_V_identical_all128':True,
  'RAW_scores_mean_and_winner_identical':True,'FREE_added_contrast_identical_zero':True,
  'g_J_changes_positive_quarter_to_negative_quarter':True,'h_J_fixed_zero':True,'SPECIFICITY_added_contrast_changes_sign':True},
  'counts':{'query_tokens':4,'reference_candidates':128,'other_competitors_per_support':127,'old_C4_equal_values':512,'old_native_equal_values':762},
  'supported_statement':'The vector of own C4, own F, own V and fixed RAW summaries cannot universally determine same-support cross-reference J on this exact token interface.',
  'unsupported_statement':'No natural recognition gain, learned encoder/pixel realizability, or spatial ownership is established.'}

def validate_child(nonce):
 need(nonce==os.environ.get('RC_WITNESS_CHILD_NONCE') and str(os.getppid())==os.environ.get('RC_WITNESS_PARENT_PID'),'EXPLICIT_CHILD_REQUIRED')
 need(not (OUT/'validation.json').exists(),'APPEND_ONLY_VALIDATION')
 s=sources();m=read(OUT/'manifest.json');r=read(OUT/'result.json')
 need(m['sources']==r['sources']==s and r['boundary']==BOUNDARY and m['result']==bind(OUT/'result.json'),'BOUND_PROVENANCE')
 v=replay(independent=True)
 for k in v:need(v[k]==r[k],'INDEPENDENT_ARITHMETIC_REPLAY:'+k)
 out={'status':'RC_SAME_SUPPORT_INFORMATION_WITNESS_V1_INDEPENDENT_VALIDATION_PASS','sources':s,
  'manifest':bind(OUT/'manifest.json'),'result':bind(OUT/'result.json'),'boundary':BOUNDARY,'checks':v['checks'],
  'fresh_explicit_subprocess':True,'independent_scorer_feature_pool_implementation':True,'validator_pid':os.getpid(),'validator_parent_pid':os.getppid()}
 write(OUT/'validation.json',out);print(json.dumps({'status':out['status'],'validation':bind(OUT/'validation.json')}),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
 m.add_argument('--produce',action='store_true');m.add_argument('--validate',action='store_true');m.add_argument('--validator-child');a=p.parse_args()
 torch.set_num_threads(8);torch.set_num_interop_threads(1);sys.path.insert(0,str(ROOT/'src'));sys.addaudithook(barrier)
 if a.validate:
  n=uuid.uuid4().hex;env=dict(os.environ,RC_WITNESS_CHILD_NONCE=n,RC_WITNESS_PARENT_PID=str(os.getpid()))
  subprocess.run([sys.executable,str(PROGRAM),'--validator-child',n],check=True,env=env)
 elif a.validator_child:
  with torch.inference_mode():validate_child(a.validator_child)
 else:
  need(not OUT.exists(),'APPEND_ONLY_OUTPUT');s=sources()
  with torch.inference_mode():v=replay()
  need(bind(PROGRAM)==s['program'],'SOURCE_DRIFT');OUT.mkdir()
  r={'status':'RC_SAME_SUPPORT_INFORMATION_WITNESS_V1_PASS','sources':s,'boundary':BOUNDARY,**v}
  write(OUT/'result.json',r);write(OUT/'manifest.json',{'sources':s,'result':bind(OUT/'result.json'),'producer_pid':os.getpid()})
  print(json.dumps({'status':r['status'],'checks':r['checks'],'result':bind(OUT/'result.json')}),flush=True)
if __name__=='__main__':main()
