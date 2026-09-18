#!/usr/bin/env python3
"""Label-free RAW/C4 qualification and ORIGINAL7 sealing, followed by an embargoed curator join."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys,time
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
PLAN=ROOT/'plan/RC_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md'
COMMON=ROOT/'programs/rc_original7_eval128_execution_common_v1.py'
COHORT=ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
WORKER=COHORT/'worker_manifest.json'
CURATOR=COHORT/'curator_roles.json'
METADATA=COHORT/'independent_metadata_validation.json'
RAW=ROOT/'results/rc_original7_eval128_token_raw_v1'
ROMA=ROOT/'results/rc_original7_eval128_roma_v1'
OUT=ROOT/'results/rc_original7_eval128_full_evidence_v1'
REPORT=ROOT/'reports/REPORT_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md'
LAUNCHERS={phase:ROOT/'slurm'/name for phase,name in {'raw_aggregate':'rc_original7_eval128_raw_aggregate_v1_dev_cpuonly_15m.sbatch','finalize_prejoin':'rc_original7_eval128_finalize_prejoin_v1_dev_cpuonly_30m.sbatch','join_validated':'rc_original7_eval128_join_validated_v1_dev_cpuonly_15m.sbatch'}.items()}
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
FEATURE_CORE=ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py'
SCORE_CORE=ROOT/'src/rc_aslo_xf/reference_visibility_pv_lossless_v1.py'
IDENTITY_CORE=ROOT/'src/rc_aslo_xf/gallery_identity_repair.py'
IDENTITY_CONTRACT=ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'
IDENTITY_REGISTRY=ROOT/'registry/gallery_identity_repair_v1.json'
GALLERY_IMAGES=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
PINS={WORKER:'f2c2e6f9181e103a61a181edf43e3309aa1de72743cfa6cf6097a9deb108e4e7',CURATOR:'47277ce66f7aba445fb80a93643039c7e8a47c1f5cc8237f60157a98e3e93381',METADATA:'a70936a24fc2f5c6996228aee673a6de505c3022487b903878d268ad9a34cd94',HEAD:'42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b',FEATURE_CORE:'96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7',SCORE_CORE:'f5fcfa1ce6fa6582a3628035e2ba54414185160fa36515fa27fd55d8aa81adcc',IDENTITY_CORE:'995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d',IDENTITY_CONTRACT:'867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650',IDENTITY_REGISTRY:'9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f'}
OLD_TRAIN_TOKEN=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1/shard03/payload.pt'
OLD_TRAIN_MAPS=ROOT/'results/rc_original_raw_visibility_pv_inputs_v1/shard03/payload.pt'
OLD_NATIVE_PREJOIN=ROOT/'results/rc_reference_support_maxmin_readout_v1/eval_prejoin.json'
OLD_NATIVE_PARAMS=ROOT/'results/rc_reference_support_maxmin_readout_v1/parameters.json'
OLD_NATIVE_SEAL=ROOT/'results/rc_reference_support_maxmin_readout_v1/eval_prejoin_seal.json'
PINS.update({OLD_TRAIN_TOKEN:'5c05fb4cf96bda8214fe4dbfdd0d237e90ef9fa571713f034e1005005182ca5d',OLD_TRAIN_MAPS:'c70f75ba1023ab0366c3509e4597c5289dfacc274dfa58b535e9ac3b9fc7b8fc',OLD_NATIVE_PREJOIN:'4dbd5c5e4fe359e35075493846df1cfd59d7354ea1e0f222fff9aaedc31e5d1c',OLD_NATIVE_PARAMS:'32969083441ec1071cc88777e253e4d0a333894962d5cb30d32d35d2d6bb3ec1',OLD_NATIVE_SEAL:'41bcd273e324599f2c02457233be2566463cff11670407b5e82032d560383b40'})
PARAMETER_SHA='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
SCORE_KEYS=('real_score','visibility_mass','query_control_score','reference_control_score')
FEATURE_NAMES=('RAW','S','M','L','Q','R')
ZERO_COLUMNS={'DROP_RAW':[0],'DROP_S':[1],'DROP_M':[2],'DROP_L':[3],'DROP_Q':[4],'DROP_R':[5],'DROP_QR':[4,5],'DROP_S_QR':[1,4,5]}
CONDITIONS=('REAL','CBIND',*ZERO_COLUMNS)
PRIVATE_PATHS={CURATOR.resolve(),METADATA.resolve(),(COHORT/'selection_receipt.json').resolve()}
RELEASED=False;HASH_DEPTH=0;BLOCKED=[];C=None;FC=None


def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if any(x in s for x in ('d1_mi','d1-mi','/grozi/','gisc_prerecall_universe','/rc_opened_eval_strict_','direction_capacity','angle_capacity')):
  BLOCKED.append(str(p));raise PermissionError('PROTECTED_INPUT_FORBIDDEN')
 if not RELEASED and not HASH_DEPTH and (p in PRIVATE_PATHS or '/target_join/' in s or '/role_shards/' in s):
  BLOCKED.append(str(p));raise PermissionError('CURATOR_OR_ROLE_PAYLOAD_BEFORE_ALL128_PREJOIN_SEAL')
sys.addaudithook(audit)
def sha(p):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for v in iter(lambda:f.read(8<<20),b''):h.update(v)
  return h.hexdigest()
 finally:HASH_DEPTH-=1
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hx(v):return float(v).hex()
def tsha(v):
 v=v.detach().cpu().contiguous()
 return hashlib.sha256(str(v.dtype).encode('ascii')+json.dumps(list(v.shape),separators=(',',':')).encode('ascii')+v.view(torch.uint8).numpy().tobytes()).hexdigest()
def mapsha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def checked(binding,root=ROOT):
 p=Path(binding['path']);p=p if p.is_absolute() else root/p;p=p.resolve()
 need(sha(p)==binding['sha256'],'ARTIFACT_BINDING_DRIFT:'+str(p));return p

def setup():
 global C,FC
 spec=importlib.util.spec_from_file_location('eval128_execution_common',COMMON);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
 for p,h in PINS.items():need(sha(p)==h,'FROZEN_SOURCE_PIN:'+p.name)
 sys.path.insert(0,str(ROOT/'src'))
 import rc_aslo_xf.romav2_colnomic_frozen_gate_v1 as core
 FC=core

def require(stage):
 global HASH_DEPTH
 # The frozen common helper only hashes public bindings; the permit does not cover stage payload parsing.
 HASH_DEPTH+=1
 try:a=C.require_authority(stage,PROGRAM)
 finally:HASH_DEPTH-=1
 need(not any(C.bound_path(v) in {CURATOR.resolve()} for v in a['source_bindings'].values()),'CURATOR_BINDING_MUST_REMAIN_PRIVATE')
 need('cpu_preflight' in a['source_bindings'],'CPU_PREFLIGHT_AUTHORITY_REQUIRED')
 pre=read(checked(a['source_bindings']['cpu_preflight']))
 need(pre['status']=='ORIGINAL7_EVAL128_CPU_SYNTHETIC_PREFLIGHT_PASS' and pre['head_parameter_sha256']==PARAMETER_SHA,'CPU_PREFLIGHT_STATUS')
 for key,value in pre['public_source_bindings'].items():need(a['source_bindings'].get(key)==value,'CPU_PUBLIC_SOURCE_AUTHORITY:'+key)
 need(pre['runtime']==runtime(),'CPU_RUNTIME_DRIFT')
 return a

def runtime():
 return {'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}

def worker_manifest():
 m=read(WORKER)
 need(m['status']=='EXPANDED_EVAL128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and m['query_count']==128 and m['curator_ledger_sha256']==PINS[CURATOR],'FROZEN_WORKER_MANIFEST')
 need(m['frozen_head_parameters']==bind(HEAD),'FROZEN_HEAD_BINDING')
 rr=m['records'];need(len(rr)==128 and [r['execution_ordinal'] for r in rr]==list(range(128)),'WORKER_EXECUTION_AXIS')
 need(len({r['query_id'] for r in rr})==len({r['source_image_sha256'] for r in rr})==128,'DISTINCT_WORKER_IMAGES')
 for r in rr:need(not({'target','identity','group','target_identity','target_position','original_query_id'}&set(r)),'NO_WORKER_TARGET_FIELDS')
 return {r['execution_ordinal']:r for r in rr}

def parameters():
 p=read(HEAD);w=torch.tensor([float.fromhex(v) for v in p['weight_binary64']],dtype=torch.float64);b=float.fromhex(p['bias_binary64'])
 need(w.shape==(6,) and bool(torch.isfinite(w).all()) and math.isfinite(b),'FINITE_CURRENT_ORIGINAL7')
 need(p['parameter_sha256']==PARAMETER_SHA==hashlib.sha256(encode({'weight':list(map(float,w)),'bias':b})).hexdigest(),'CURRENT_NATIVE7_NOT_DEFAULT_LEGACY_WEIGHTS')
 return w,b,{'source':bind(HEAD),'parameter_sha256':PARAMETER_SHA,'weight_binary64':p['weight_binary64'],'bias_binary64':p['bias_binary64'],'parameter_count':7}

def gallery_labels():
 from rc_aslo_xf.gallery_identity_repair import build_identity_map
 paths=[]
 for directory,subdirs,names in os.walk(GALLERY_IMAGES):
  subdirs.sort()
  paths.extend(Path(directory)/n for n in sorted(names) if (Path(directory)/n).is_file() and Path(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})
 need(len(paths)==5413,'FROZEN_GALLERY_PHYSICAL_COUNT')
 mapping=build_identity_map([p.stem.strip() for p in paths]);need(len(set(mapping.labels))==5412,'CORRECTED_GALLERY_IDENTITY_COUNT')
 return tuple(mapping.labels),mapping.corrected_row_identity_mapping_sha256

def as_tensor(value,dtype,shape,label):
 need(isinstance(value,torch.Tensor) and value.dtype==dtype and tuple(value.shape)==tuple(shape) and value.device.type=='cpu','TENSOR_DOMAIN:'+label)
 need(bool(torch.isfinite(value).all()),'NONFINITE:'+label);return value

def verify_raw_row(r,worker,labels):
 need(r['query_id']==worker['query_id'] and r['execution_ordinal']==worker['execution_ordinal'] and r['track']==worker['track'],'RAW_WORKER_JOIN')
 need(r['query_source_sha256']==worker['source_image_sha256'] and r['query_source_path']==worker['query_image_path'],'RAW_QUERY_IMAGE_SHA_AND_ALIAS')
 need(r['target_or_label_read_count']==r['target_insertion_count']==0,'RAW_QUERY_NO_TARGETS_OR_INSERTION')
 expected_frame='EXIF_ORIENTED_BEFORE_RESIZE' if worker['track']=='new_difficult_train' else 'DECODED_RAW_BEFORE_EXIF'
 need(r['processor_input_frame']==expected_frame,'ORIGINAL_QUERY_INPUT_FRAME')
 need(not({'target','identity','group','target_identity','target_position','original_query_id'}&set(r)),'RAW_QUERY_TARGET_FIELDS_FORBIDDEN')
 q=r['query_tokens'];template=r['template_tokens'];grid=list(r['query_grid_shape'])
 need(q.dtype==template.dtype==torch.float16 and q.ndim==template.ndim==2 and q.shape[1]==template.shape[1]==128,'RAW_QUERY_TOKEN_DOMAIN')
 need(len(grid)==2 and math.prod(grid)==len(q)>0 and len(template)>0,'CURRENT_ENCODER_GRID')
 need(tsha(q)==r['query_tokens_sha256'] and tsha(template)==r['template_tokens_sha256'],'ACTUAL_CURRENT_RUNTIME_TOKEN_HASHES')
 physical=as_tensor(r['raw_physical_scores'],torch.float64,(5413,),'RAW5413')
 ordered=sorted(range(5413),key=lambda i:(-float(physical[i]),i));seen=set();ranked=[]
 for i in ordered:
  if labels[i] not in seen:ranked.append(i);seen.add(labels[i])
 need(len(ranked)==5412 and ranked==list(r['raw_ranked_physical_rows']),'INDEPENDENT_FULL5412_RAW_RANKING')
 scored=as_tensor(r['raw_ranked_scores'],torch.float64,(5412,),'RANKED5412')
 need([hx(v) for v in scored]==[hx(physical[i]) for i in ranked],'RANKED_SCORES_SOURCE')
 axis=list(r['candidate_physical_rows']);top=ranked[:128]
 need(axis==sorted(top) and list(r['candidate_ranked_physical_rows'])==top and len(set(axis))==128,'ORIGINAL_TOP128_CANONICAL_PHYSICAL_AXIS')
 candidate=as_tensor(r['candidate_raw_scores'],torch.float64,(128,),'C128_RAW')
 need([hx(v) for v in candidate]==[hx(physical[i]) for i in axis],'C128_RAW_SCORE_BITS')
 return {'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],'track':r['track'],'source_image_sha256':r['query_source_sha256'],
  'query_tokens_sha256':r['query_tokens_sha256'],'query_grid_shape':grid,'processor_input_frame':r['processor_input_frame'],
  'candidate_physical_rows':axis,'candidate_raw_scores_binary64':[hx(v) for v in candidate],
  'base_winner_position':axis.index(top[0]),'challenger_positions':[i for i in range(128) if axis[i]!=top[0]],
  'raw_ranked_physical_rows':ranked,'raw_ranked_scores_binary64':[hx(v) for v in scored],
  'raw_physical_scores_sha256':tsha(physical),'template_tokens_sha256':r['template_tokens_sha256']}

def load_raw_shard(shard,a,workers,labels):
 folder=RAW/f'shard{shard:02d}';receipt=read(folder/'receipt.json');validation=read(folder/'validation.json')
 need(receipt['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_SHARD_READY' and validation['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_CPU_REPLAY_PASS','RAW_STAGE_QUALIFIED')
 payload_path=checked(receipt['payload']);need(payload_path==(folder/'payload.pt').resolve(),'RAW_PAYLOAD_PATH')
 need(checked(validation['payload'])==payload_path and checked(validation['receipt'])==(folder/'receipt.json').resolve(),'RAW_VALIDATOR_BINDINGS')
 need(checked(validation['authority'])==C.AUTHORITY.resolve(),'RAW_COMMON_AUTHORITY')
 data=torch.load(payload_path,map_location='cpu',weights_only=True)
 need(data['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_SHARD_READY' and data['shard']==shard and data['shard_count']==16,'RAW_SHARD_HEADER')
 need(checked(data['bindings']['authority'])==C.AUTHORITY.resolve() and checked(data['bindings']['worker_manifest'])==WORKER.resolve(),'RAW_PAYLOAD_AUTHORITY_WORKER_BINDING')
 need(validation['fresh_CPU_process'] is True and all(x is True for x in validation['checks'].values()),'RAW_FRESH_CPU_QUALIFICATION')
 need(data['access']['query_target_reads']==data['access']['curator_reads']==data['access']['new_training_updates']==data['access']['forbidden_read_attempts']==0,'RAW_ACCESS_BOUNDARY')
 records=data['records'];need(len(records)==8 and [r['execution_ordinal'] for r in records]==list(range(shard*8,shard*8+8)),'RAW_COMPLETE_SHARD8')
 qualified=[verify_raw_row(r,workers[r['execution_ordinal']],labels) for r in records]
 refs={int(k):v for k,v in data['references'].items()};required={p for r in records for p in r['candidate_physical_rows']}
 need(required==set(refs),'EXACT_C128_REFERENCE_UNION_PRESENT')
 for p in required:
  ref=refs[p];t=ref['tokens'];need(ref['physical_row']==p and t.dtype==torch.float16 and t.ndim==2 and t.shape[1]==128 and math.prod(ref['grid_shape'])==len(t)>0,'REFERENCE_IMAGE_TOKEN_DOMAIN')
  need(tsha(t)==ref['tokens_sha256'],'REFERENCE_IMAGE_TOKEN_HASH')
 return records,refs,qualified,{'shard':shard,'payload':bind(payload_path),'receipt':bind(folder/'receipt.json'),'validation':bind(folder/'validation.json')}

def raw_aggregate(a):
 workers=worker_manifest();labels,mapping=gallery_labels();entries=[];records=[]
 for shard in range(16):
  _,_,qualified,source=load_raw_shard(shard,a,workers,labels);entries.append(source);records.extend(qualified)
 need([r['execution_ordinal'] for r in records]==list(range(128)) and len({r['query_id'] for r in records})==128 and not BLOCKED,'ALL128_RAW_QUALIFIED_WITHOUT_CURATOR')
 value={'status':'ORIGINAL7_EVAL128_RAW_AGGREGATE_PASS','authority':bind(C.AUTHORITY),'program':bind(PROGRAM),'worker_manifest':bind(WORKER),
  'query_count':128,'gallery_physical_count':5413,'gallery_identity_count':5412,'corrected_gallery_mapping_sha256':mapping,
  'shards':entries,'records':records,'independent_full_RAW_order_and_C128_replayed':True,'current_runtime_tokens_from_bound_payloads':True,
  'legacy_redacted_tokens_used':False,'curator_payload_reads':0,'candidate_insertion_count':0,'new_training_updates':0}
 C.write_json(OUT/'raw_aggregate.json',value);print(json.dumps({'status':value['status'],'output':bind(OUT/'raw_aggregate.json'),'query_count':128}),flush=True)


def independent_score(q,r,wq,wr):
 # Deliberately repeat normalize/matmul for each control; preserve literal mass*sum/den order.
 nq=F.normalize(q.to(torch.float64),dim=1);nr=F.normalize(r.to(torch.float64),dim=1)
 similarity=nq@nr.T;local=(similarity*wr[None]).max(1).values
 mass=torch.sqrt(wq.mean()*wr.mean())
 return mass*(wq*local).sum()/wq.sum().clamp_min(1e-12),mass

def independent_c4(q,r,wq,wr):
 real,mass=independent_score(q,r,wq,wr)
 query,_=independent_score(q,r,torch.roll(wq,max(1,len(wq)//2)),wr)
 reference,_=independent_score(q,r,wq,torch.roll(wr,max(1,len(wr)//2)))
 return {'real_score':float(real),'visibility_mass':float(mass),'query_control_score':float(query),'reference_control_score':float(reference)}

def literal_feature(raw,evidence,c,w):
 mean=sum(float(x) for x in raw)/128;sd=(sum((float(x)-mean)**2 for x in raw)/128)**.5
 ec,ew=evidence[c],evidence[w]
 def sym(x,y):return (x-y)/(abs(x)+abs(y)+1e-12)
 lc=ec['real_score']/max(ec['visibility_mass'],1e-12);lw=ew['real_score']/max(ew['visibility_mass'],1e-12)
 return torch.tensor([(raw[c]-raw[w])/max(sd,1e-12),sym(ec['real_score'],ew['real_score']),sym(ec['visibility_mass'],ew['visibility_mass']),sym(lc,lw),sym(ec['real_score']-ec['query_control_score'],ew['real_score']-ew['query_control_score']),sym(ec['real_score']-ec['reference_control_score'],ew['real_score']-ew['reference_control_score'])],dtype=torch.float64)

def predict(row,z,independent=False):
 vals=list(map(float,z));cs=row['challenger_positions'];axis=row['candidate_physical_rows'];winner=row['base_winner_position']
 need(len(vals)==127 and all(math.isfinite(v) for v in vals),'ALL127_FINITE_LOGITS')
 k=sorted(range(127),key=lambda j:(-vals[j],axis[cs[j]]))[0] if independent else max(range(127),key=lambda j:(vals[j],-axis[cs[j]]))
 best=cs[k];decision='SWITCH' if vals[k]>0 else 'HOLD';final=best if decision=='SWITCH' else winner
 return {'decision':decision,'proposed_challenger':best,'proposed_physical_row':axis[best],'max_logit_binary64':hx(vals[k]),
  'final_position':final,'final_physical_row':axis[final],'all127_logits_binary64':[hx(v) for v in vals]}

def score_record(row,evidence,w,b,independent=False):
 raw=[float.fromhex(v) for v in row['candidate_raw_scores_binary64']];winner=row['base_winner_position'];cs=row['challenger_positions']
 donor=[(i+64)%128 for i in range(128)];cb={i:evidence[donor[i]] for i in range(128)}
 feature=literal_feature if independent else FC.candidate_feature
 real=torch.stack([feature(raw,evidence,c,winner) for c in cs]);control=torch.stack([feature(raw,cb,c,winner) for c in cs])
 outputs={}
 for condition in CONDITIONS:
  matrix=control if condition=='CBIND' else real
  if condition in ZERO_COLUMNS:matrix=matrix.clone();matrix[:,ZERO_COLUMNS[condition]]=0.
  outputs[condition]=predict(row,matrix@w+b,independent)
 return {'features_binary64':{'REAL':[[hx(v) for v in r] for r in real],'CBIND':[[hx(v) for v in r] for r in control]},
  'predictions':outputs,'C_BIND_source_positions':donor,'REAL_all127_signed_terms_binary64':[[hx(v) for v in r] for r in real*w]}


def exact_group_signflip(group_rows):
 values=[Fraction(g['net'],g['query_count']) for g in group_rows];nonzero=[v for v in values if v]
 scale=math.lcm(*(v.denominator for v in values)) if values else 1
 integer=[int(v*scale) for v in nonzero];observed=abs(sum(integer));distribution={0:1}
 for value in integer:
  updated=defaultdict(int)
  for previous,count in distribution.items():updated[previous+value]+=count;updated[previous-value]+=count
  distribution=dict(updated)
 denominator=1<<len(nonzero);numerator=sum(count for statistic,count in distribution.items() if abs(statistic)>=observed)
 need(sum(distribution.values())==denominator,'EXACT_SIGNFLIP_ENUMERATION_COUNTS')
 return {'two_sided_p':numerator/denominator,'p_numerator':numerator,'p_denominator':denominator,'nonzero_groups':len(nonzero),
  'total_groups':len(values),'integer_scale_LCM':scale,'observed_absolute_scaled_sum':observed,'method':'exact dynamic enumeration of all group sign flips; >= observed absolute paired group-mean statistic'}

def primary_statistics(actions):
 groups={}
 for a in actions:
  g=groups.setdefault(a['group'],{'group':a['group'],'query_count':0,'RAW_correct':0,'REAL_correct':0})
  g['query_count']+=1;g['RAW_correct']+=int(a['base_correct']);g['REAL_correct']+=int(a['final_correct'])
 ordered=[groups[k] for k in sorted(groups)]
 for g in ordered:g['net']=g['REAL_correct']-g['RAW_correct'];g['accuracy_difference']=g['net']/g['query_count']
 means=np.array([g['accuracy_difference'] for g in ordered],dtype=np.float64)
 rng=np.random.default_rng(20260910);indices=rng.integers(0,len(means),size=(10000,len(means)));boot=means[indices].mean(axis=1)
 interval=np.quantile(boot,[.025,.975],method='linear')
 exact_mean=sum((Fraction(g['net'],g['query_count']) for g in ordered),Fraction())/len(ordered)
 return {'groups':ordered,'group_count':len(ordered),'equal_group_mean_accuracy_difference':float(exact_mean),
  'group_bootstrap':{'seed':20260910,'draws':10000,'sampling':'source groups with replacement, all group means equally weighted',
   'interval_level':.95,'percentile_method':'numpy.quantile linear','lower':float(interval[0]),'upper':float(interval[1]),
   'numpy_version':np.__version__,'bit_generator':type(rng.bit_generator).__name__,'samples_binary64_sha256':hashlib.sha256(boot.tobytes()).hexdigest()},
  'two_sided_group_signflip':exact_group_signflip(ordered)}


def load_roma_shard(shard,raw_source,raw_records,refs):
 folder=ROMA/f'shard{shard:02d}';receipt=read(folder/'receipt.json');validation=read(folder/'validation.json')
 need(receipt['status']=='RC_ORIGINAL7_EVAL128_ROMA_SHARD_READY' and validation['status']=='RC_ORIGINAL7_EVAL128_ROMA_CPU_REPLAY_PASS','ROMA_SHARD_QUALIFIED')
 p=checked(receipt['payload']);need(p==(folder/'payload.pt').resolve() and checked(validation['payload'])==p,'ROMA_PAYLOAD_BINDING')
 need(checked(validation['receipt'])==(folder/'receipt.json').resolve() and checked(validation['authority'])==C.AUTHORITY.resolve(),'ROMA_VALIDATOR_BINDING')
 need(validation['query_count']==8 and validation['candidate_occurrence_count']==1024 and validation['independently_recomputed_C4_scalars']==4096 and validation['all_C4_bits_exact'] is True,'ALL_C4_SHARD_QUALIFICATION_COUNTS')
 need(validation['target_role_read_count']==validation['model_update_count']==0 and validation['fresh_child_nonce'],'ROMA_NO_TARGETS_OR_FIT')
 data=torch.load(p,map_location='cpu',weights_only=True)
 need(data['status']=='RC_ORIGINAL7_EVAL128_ROMA_SHARD_READY' and data['kind']=='shard' and data['shard']==shard and len(data['records'])==8,'ROMA_SHARD_HEADER')
 need(checked(data['authority'])==C.AUTHORITY.resolve() and checked(data['RAW_aggregate'])==(OUT/'raw_aggregate.json').resolve(),'ROMA_AUTHORITY_RAW_AGGREGATE_BINDING')
 for key in ('payload','receipt','validation'):need(checked(data['token_source'][key])==checked(raw_source[key]),'ROMA_EXACT_RAW_SHARD_SOURCE')
 results=[];scalar_count=0;map_count=0
 for row,raw in zip(data['records'],raw_records,strict=True):
  for key in ('query_id','execution_ordinal','track','query_source_sha256','query_grid_shape','processor_input_frame','query_tokens_sha256','candidate_physical_rows'):
   need(row[key]==raw[key],'ROMA_QUERY_TOKEN_FRAME_AXIS:'+key)
  need(tsha(row['candidate_raw_scores'])==tsha(raw['candidate_raw_scores']) and len(row['candidates'])==128,'ROMA_C128_RAW_SCORE_BITS')
  evidence={};map_bindings=[]
  for pos,candidate in enumerate(row['candidates']):
   physical=raw['candidate_physical_rows'][pos];ref=refs[physical];q=raw['query_tokens'];r=ref['tokens']
   need(candidate['candidate_position']==pos and candidate['physical_row']==physical,'ROMA_COMPLETE_CANDIDATE_ORDER')
   need(candidate['reference_tokens_sha256']==ref['tokens_sha256'] and list(candidate['reference_grid_shape'])==list(ref['grid_shape']) and candidate['reference_image_sha256']==ref['source_image_sha256'],'REFERENCE_TOKEN_IMAGE_GEOMETRY_BINDING')
   wq=as_tensor(candidate['query_visibility'],torch.float64,(len(q),),'QUERY_VISIBILITY');wr=as_tensor(candidate['reference_visibility'],torch.float64,(len(r),),'REFERENCE_VISIBILITY')
   need(bool(((wq>=0)&(wq<=1)).all()) and bool(((wr>=0)&(wr<=1)).all()),'VISIBILITY_RANGE')
   need(mapsha(wq)==candidate['query_map_sha256']==candidate['old_scores']['query_map_sha256'] and mapsha(wr)==candidate['reference_map_sha256']==candidate['old_scores']['reference_map_sha256'],'SAVED_MAP_HASHES')
   need(candidate['control_shifts']=={'query':max(1,len(wq)//2),'reference':max(1,len(wr)//2)},'ORIGINAL_HALF_AXIS_Q_R_CONTROLS')
   independent=independent_c4(q,r,wq,wr)
   for key in SCORE_KEYS:need(hx(independent[key])==hx(candidate['old_scores'][key]),'INDEPENDENT_CPU_C4_BITS:'+row['query_id']+':'+str(pos)+':'+key)
   need(hx(candidate['old_scores']['raw_score'])==hx(raw['candidate_raw_scores'][pos]),'C4_RAW_PRIOR_BINDING')
   evidence[pos]=independent;scalar_count+=4;map_count+=2
   map_bindings.append({'candidate_position':pos,'physical_row':physical,'query_map_sha256':mapsha(wq),'reference_map_sha256':mapsha(wr),'reference_tokens_sha256':ref['tokens_sha256']})
  results.append({'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'evidence':evidence,'map_token_bindings':map_bindings})
 need(scalar_count==4096 and map_count==2048,'WHOLE_C4_SHARD_RECOMPUTE')
 return results,{'shard':shard,'payload':bind(p),'receipt':bind(folder/'receipt.json'),'validation':bind(folder/'validation.json'),
  'independently_recomputed_scalar_count':scalar_count,'map_binding_count':map_count}


def validate_sealed_records(records,w,b):
 need(len(records)==128 and [r['execution_ordinal'] for r in records]==list(range(128)) and len({r['query_id'] for r in records})==128,'SEALED_ALL128_QUERY_AXIS')
 comparisons=0
 for row in records:
  ev={int(k):{f:float.fromhex(v) for f,v in value.items()} for k,value in row['C4_binary64'].items()}
  need(set(ev)==set(range(128)) and all(set(x)==set(SCORE_KEYS) for x in ev.values()),'SEALED_FULL_C4_AXIS')
  replay=score_record(row,ev,w,b,independent=True)
  for key in ('features_binary64','predictions','C_BIND_source_positions','REAL_all127_signed_terms_binary64'):
   need(replay[key]==row[key],'INDEPENDENT_PREJOIN_FEATURE_LOGIT_ACTION_REPLAY:'+key)
  comparisons+=len(CONDITIONS)*127
 need(comparisons==162560,'ALL128_TEN_CONDITIONS_ALL127')
 return comparisons


def finalize_prejoin(a):
 aggregate=read(OUT/'raw_aggregate.json')
 need(aggregate['status']=='ORIGINAL7_EVAL128_RAW_AGGREGATE_PASS' and aggregate['query_count']==128 and checked(aggregate['authority'])==C.AUTHORITY.resolve(),'QUALIFIED_ALL128_RAW_REQUIRED')
 need(not (OUT/'prejoin_seal.json').exists(),'APPEND_ONLY_PREJOIN_SEAL')
 workers=worker_manifest();labels,mapping=gallery_labels();need(mapping==aggregate['corrected_gallery_mapping_sha256'],'RAW_GALLERY_MAPPING_UNCHANGED')
 raw_sources=[];roma_sources=[];public_records=[];qualified_evidence=[];started=time.monotonic()
 # Complete all128 token/map/scalar qualifications before the first natural head score.
 for shard in range(16):
  raw_records,refs,raw_public,raw_source=load_raw_shard(shard,a,workers,labels)
  need(raw_source==aggregate['shards'][shard] and raw_public==aggregate['records'][shard*8:(shard+1)*8],'RAW_AGGREGATE_EXACT_REPLAY')
  c4,c4_source=load_roma_shard(shard,raw_source,raw_records,refs)
  raw_sources.append(raw_source);roma_sources.append(c4_source);public_records.extend(raw_public);qualified_evidence.extend(c4)
  print(json.dumps({'event':'CPU_C4_SHARD_QUALIFIED','shard':shard,'queries_qualified':len(public_records),'scalar_checks':sum(s['independently_recomputed_scalar_count'] for s in roma_sources),'elapsed_seconds':time.monotonic()-started}),flush=True)
 need(len(public_records)==len(qualified_evidence)==128 and sum(s['independently_recomputed_scalar_count'] for s in roma_sources)==65536 and not BLOCKED,'ALL128_ALL16384_C4_QUALIFIED_BEFORE_HEAD')
 qualification={'status':'ORIGINAL7_EVAL128_ALL_INPUTS_CPU_QUALIFIED','authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),
  'raw_aggregate':bind(OUT/'raw_aggregate.json'),'raw_shards':raw_sources,'roma_shards':roma_sources,'query_count':128,
  'candidate_occurrences':16384,'independent_C4_scalar_checks':65536,'head_predictions_computed_at_this_gate':0,
  'curator_payload_reads':0,'new_encoder_or_RoMa_forwards':0,'new_training_updates':0,'program':bind(PROGRAM)}
 C.write_json(OUT/'input_qualification.json',qualification)
 w,b,parameter_state=parameters();records=[]
 for row,c4 in zip(public_records,qualified_evidence,strict=True):
  need((row['query_id'],row['execution_ordinal'])==(c4['query_id'],c4['execution_ordinal']),'C4_RAW_JOIN_BY_OPAQUE_ID')
  computed=score_record(row,c4['evidence'],w,b)
  records.append({**row,'C4_binary64':{str(i):{k:hx(v) for k,v in ev.items()} for i,ev in c4['evidence'].items()},
   'candidate_map_token_bindings':c4['map_token_bindings'],**computed})
 count=validate_sealed_records(records,w,b)
 C.write_json(OUT/'prejoin_parameters.json',parameter_state)
 C.write_json(OUT/'prejoin_records.json',records)
 seal={'status':'ORIGINAL7_EVAL128_ALL_PREDICTIONS_SEALED_BEFORE_CURATOR','authority':bind(C.AUTHORITY),'program':bind(PROGRAM),
  'input_qualification':bind(OUT/'input_qualification.json'),'parameters':bind(OUT/'prejoin_parameters.json'),
  'records':bind(OUT/'prejoin_records.json'),'query_count':128,'condition_names':list(CONDITIONS),'all127_logit_count':count,
  'head_parameter_sha256':PARAMETER_SHA,'curator_ledger_sha256':PINS[CURATOR],'curator_payload_reads':0,'forbidden_read_attempts':len(BLOCKED),
  'producer_process':{'hostname':os.uname().nodename,'pid':os.getpid()},'method':'all128 RAW plus all16384 C4 qualified before head scoring; complete logits/actions sealed before curator join'}
 C.write_json(OUT/'prejoin_seal.json',seal)
 C.write_json(OUT/'prejoin_validation.json',{'status':'ORIGINAL7_EVAL128_PREJOIN_INDEPENDENT_CPU_REPLAY_PASS','seal':bind(OUT/'prejoin_seal.json'),
  'records':bind(OUT/'prejoin_records.json'),'parameters':bind(OUT/'prejoin_parameters.json'),'input_qualification':bind(OUT/'input_qualification.json'),
  'checks':{'all128_full_RAW_axes_independently_rebuilt':True,'all65536_C4_scalars_independently_recomputed_from_saved_maps_and_tokens':True,
   'all_original_and_zero_column_features_independently_rebuilt':True,'all162560_logits_and1280_actions_replayed':True,
   'full128_retained_before_any_target_join':True},'curator_payload_reads':0,'new_training_updates':0})
 print(json.dumps({'status':seal['status'],'seal':bind(OUT/'prejoin_seal.json'),'logits':count,'query_count':128}),flush=True)


def target_action(row,pred,role,labels):
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];base=row['base_winner_position'];ranked=row['raw_ranked_physical_rows']
 target_identity=role['identity'];target_full=[i for i,p in enumerate(ranked) if labels[p]==target_identity]
 need(len(target_full)==1,'CURATOR_TARGET_NOT_UNIQUE_IN_FULL5412_GALLERY:'+row['query_id'])
 raw_rank=target_full[0]+1;target_positions=[i for i,p in enumerate(axis) if labels[p]==target_identity]
 need(len(target_positions)<=1,'IDENTITY_DEDUP_C128_INVARIANT');target=target_positions[0] if target_positions else None
 values=[float.fromhex(v) for v in pred['all127_logits_binary64']]
 j=sorted(range(127),key=lambda i:(-values[i],axis[cs[i]]))[0];proposed=cs[j];switch=values[j]>0.;final=proposed if switch else base
 need(pred['final_position']==final and pred['decision']==('SWITCH' if switch else 'HOLD') and pred['max_logit_binary64']==hx(values[j]),'POSTJOIN_LITERAL_ACTION_REPLAY')
 chosen_rank=ranked.index(axis[proposed])+1
 if not switch:final_rank=raw_rank
 elif labels[axis[proposed]]==target_identity:final_rank=1
 else:final_rank=raw_rank+1 if raw_rank<chosen_rank else raw_rank
 # Independent direct move-to-front check over the complete5412 identity rank.
 direct=list(ranked)
 if switch:direct.remove(axis[proposed]);direct.insert(0,axis[proposed])
 need(direct[final_rank-1]==ranked[raw_rank-1],'FULL_GALLERY_MOVE_TO_FRONT_RANK')
 base_correct=labels[axis[base]]==target_identity;final_correct=labels[axis[final]]==target_identity
 if target is None:target_logit=None;threshold=None;competition=None;reason='CANDIDATE_RECALL_MISS'
 elif target==base:target_logit=None;threshold=-max(values);competition=None;reason='RAW_CORRECT_HELD' if final_correct else 'RAW_CORRECT_BROKEN'
 else:
  ti=cs.index(target);target_logit=values[ti];threshold=target_logit;competition=target_logit-max(v for i,v in enumerate(values) if cs[i]!=target)
  reason='RAW_WRONG_RESCUED' if final_correct else 'TARGET_CANNOT_TRIGGER_SWITCH' if threshold<=0 else 'TARGET_LOSES_CHALLENGER_COMPETITION'
 wrong=[i for i,c in enumerate(cs) if c!=target];wi=max(wrong,key=lambda i:(values[i],-axis[cs[i]]))
 return {'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'original_query_id':role['original_query_id'],
  'identity':target_identity,'group':role['group'],'track':row['track'],'candidate_recall':target is not None,'target_position':target,
  'target_physical_row_in_full_rank':ranked[raw_rank-1],'RAW_winner_physical_row':axis[base],
  'proposed_challenger':proposed,'proposed_physical_row':axis[proposed],'switch_logit':values[j],
  'decision':'SWITCH' if switch else 'HOLD','final_position':final,'final_physical_row':axis[final],
  'base_correct':base_correct,'final_correct':final_correct,'wrong_to_wrong':not base_correct and not final_correct and final!=base,
  'raw_target_rank_full_gallery':raw_rank,'final_target_rank_full_gallery':final_rank,
  'target_challenger_logit':target_logit,'RAW_winner_policy_score':0.,'switch_or_HOLD_protection_margin':threshold,
  'target_vs_strongest_wrong_margin':competition,'strongest_wrong_challenger':cs[wi],'strongest_wrong_physical_row':axis[cs[wi]],
  'strongest_wrong_logit':values[wi],'outcome_kind':reason}


def metrics(actions):
 n=len(actions)
 if not n:return {'query_count':0,'candidate_recall_count':0,'candidate_recall_at128':None,'RAW_top1':0,'final_top1':0,'RAW_accuracy':None,'final_accuracy':None,'rescue':0,'break':0,'net':0,'RAW_full_gallery_MRR':None,'final_full_gallery_MRR':None,'switch_count':0,'wrong_to_wrong_count':0}
 base=sum(a['base_correct'] for a in actions);final=sum(a['final_correct'] for a in actions);recall=sum(a['candidate_recall'] for a in actions)
 return {'query_count':n,'candidate_recall_count':recall,'candidate_recall_at128':recall/n,'RAW_top1':base,'final_top1':final,'RAW_accuracy':base/n,'final_accuracy':final/n,
  'rescue':sum(not a['base_correct'] and a['final_correct'] for a in actions),'break':sum(a['base_correct'] and not a['final_correct'] for a in actions),'net':final-base,
  'RAW_full_gallery_MRR':float(sum((Fraction(1,a['raw_target_rank_full_gallery']) for a in actions),Fraction())/n),
  'final_full_gallery_MRR':float(sum((Fraction(1,a['final_target_rank_full_gallery']) for a in actions),Fraction())/n),
  'switch_count':sum(a['decision']=='SWITCH' for a in actions),'wrong_to_wrong_count':sum(a['wrong_to_wrong'] for a in actions)}

def paired_actions(new,base):
 need([a['query_id'] for a in new]==[a['query_id'] for a in base],'PAIRED_QUERY_ORDER')
 rescue=[a['query_id'] for a,b in zip(new,base) if a['final_correct'] and not b['final_correct']]
 broken=[a['query_id'] for a,b in zip(new,base) if b['final_correct'] and not a['final_correct']]
 return {'rescue':len(rescue),'break':len(broken),'net':len(rescue)-len(broken),'rescue_query_ids':rescue,'break_query_ids':broken,
  'full_gallery_MRR_difference':float(sum((Fraction(1,a['final_target_rank_full_gallery'])-Fraction(1,b['final_target_rank_full_gallery']) for a,b in zip(new,base)),Fraction())/len(new))}


def six_feature_ledger(row,action,w,b):
 matrix=torch.tensor([[float.fromhex(v) for v in values] for values in row['features_binary64']['REAL']],dtype=torch.float64)
 z=torch.tensor([float.fromhex(v) for v in row['predictions']['REAL']['all127_logits_binary64']],dtype=torch.float64)
 cs=row['challenger_positions'];target=action['target_position'];winner=row['base_winner_position'];wrong=action['strongest_wrong_challenger'];wi=cs.index(wrong)
 def endpoint(i):
  terms=[Fraction.from_float(float(matrix[i,k]))*Fraction.from_float(float(w[k])) for k in range(6)]
  ideal=sum(terms,Fraction.from_float(b));observed=Fraction.from_float(float(z[i]))
  return {'candidate_position':cs[i],'physical_row':row['candidate_physical_rows'][cs[i]],'six_terms':{n:float(terms[k]) for k,n in enumerate(FEATURE_NAMES)},
   'bias':b,'logit':float(z[i]),'FP64_minus_exact_affine_roundoff':float(observed-ideal)}
 selected=endpoint(cs.index(action['proposed_challenger']));strongest=endpoint(wi)
 result={'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'candidate_recall':action['candidate_recall'],
  'outcome_kind':action['outcome_kind'],'selected_challenger_terms':selected,'strongest_wrong_terms':strongest,
  'target_challenger_terms':None,'target_minus_wrong_six_terms':None,'bias_cancels_in_challenger_comparison':True,
  'RAW_winner_policy_score':0.,'threshold_or_HOLD_protection_margin':action['switch_or_HOLD_protection_margin'],
  'target_vs_wrong_margin':action['target_vs_strongest_wrong_margin']}
 if target is not None and target!=winner:
  ti=cs.index(target);result['target_challenger_terms']=endpoint(ti)
  result['target_minus_wrong_six_terms']={n:float((Fraction.from_float(float(matrix[ti,k]))-Fraction.from_float(float(matrix[wi,k])))*Fraction.from_float(float(w[k]))) for k,n in enumerate(FEATURE_NAMES)}
 elif target==winner:
  result['HOLD_protection_six_terms']={n:-strongest['six_terms'][n] for n in FEATURE_NAMES};result['HOLD_protection_bias']=-b
 else:result['absence_note']='Target absent from original naturalC128; no target logit or invented target feature is computed.'
 return result


def join_validated(a):
 global RELEASED
 need(not (OUT/'result.json').exists(),'APPEND_ONLY_EVALUATION_RESULT')
 seal=read(OUT/'prejoin_seal.json');validation=read(OUT/'prejoin_validation.json')
 need(seal['status']=='ORIGINAL7_EVAL128_ALL_PREDICTIONS_SEALED_BEFORE_CURATOR' and validation['status']=='ORIGINAL7_EVAL128_PREJOIN_INDEPENDENT_CPU_REPLAY_PASS','ALL128_PREJOIN_QUALIFICATION_REQUIRED')
 for key,filename in (('records','prejoin_records.json'),('parameters','prejoin_parameters.json'),('input_qualification','input_qualification.json')):
  need(checked(seal[key])==checked(validation[key])==(OUT/filename).resolve(),'PREJOIN_COMPONENT_BINDING:'+key)
 need(checked(validation['seal'])==(OUT/'prejoin_seal.json').resolve() and checked(seal['authority'])==C.AUTHORITY.resolve(),'PREJOIN_VALIDATION_AUTHORITY_BINDING')
 need(seal['query_count']==128 and seal['condition_names']==list(CONDITIONS) and seal['all127_logit_count']==162560 and seal['curator_ledger_sha256']==PINS[CURATOR],'ALL128_CONDITIONS_PRESEALED')
 need(seal['curator_payload_reads']==seal['forbidden_read_attempts']==0 and all(v is True for v in validation['checks'].values()),'PREJOIN_GUARD_AND_CHECKS')
 need(seal['producer_process']!={'hostname':os.uname().nodename,'pid':os.getpid()},'POSTJOIN_REQUIRES_FRESH_CPU_PROCESS')
 w,b,par=parameters();need(read(OUT/'prejoin_parameters.json')==par,'FROZEN_CURRENT_HEAD_UNCHANGED')
 records=read(OUT/'prejoin_records.json');logits=validate_sealed_records(records,w,b)
 qualification=read(OUT/'input_qualification.json');need(qualification['query_count']==128 and qualification['candidate_occurrences']==16384 and qualification['independent_C4_scalar_checks']==65536,'COMPLETE_C4_QUALIFICATION')
 for source in qualification['raw_shards']+qualification['roma_shards']:
  for key in ('payload','receipt','validation'):checked(source[key])
 need(not BLOCKED and not RELEASED,'NO_CURATOR_ACCESS_BEFORE_CURRENT_PROCESS_PREDICTION_REPLAY')
 private=a['postjoin_source_bindings'];curator_bindings=[v for v in private.values() if C.bound_path(v)==CURATOR.resolve()]
 need(len(curator_bindings)==1 and curator_bindings[0]['sha256']==PINS[CURATOR] and sha(CURATOR)==PINS[CURATOR],'EXACT_PRIVATE_CURATOR_AUTHORITY')
 RELEASED=True
 curator=read(CURATOR);meta=read(METADATA)
 need(curator['status']=='CURATOR_ONLY_EXPANDED_EVAL128_ROLES_FROZEN' and meta['status']=='EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS','QUALIFIED_CURATOR_SELECTION')
 need(checked(meta['worker_manifest'])==WORKER.resolve() and checked(meta['curator_ledger'])==CURATOR.resolve(),'SELECTION_VALIDATOR_BINDINGS')
 roles=curator['records'];byid={r['query_id']:r for r in roles};workers=worker_manifest()
 need(len(roles)==len(byid)==128 and set(byid)=={r['query_id'] for r in records},'ALL128_CURATOR_JOIN_NO_FILTERING')
 need(len({r['group'] for r in roles})==21 and len({r['identity'] for r in roles})==24,'FROZEN21_GROUPS_24_IDENTITIES')
 for r in records:
  role=byid[r['query_id']];worker=workers[r['execution_ordinal']]
  need(role['execution_ordinal']==r['execution_ordinal'] and role['track']==r['track'] and role['source_image_sha256']==worker['source_image_sha256']==r['source_image_sha256'],'CURATOR_IMAGE_TRACK_JOIN')
 labels,mapping=gallery_labels();actions={name:[] for name in CONDITIONS};errors=[]
 for row in records:
  role=byid[row['query_id']]
  if role['identity'] not in labels:
   errors.append({'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'error':'CURATOR_TARGET_NOT_IN_FROZEN_GALLERY','identity':role['identity']});continue
  for condition in CONDITIONS:actions[condition].append(target_action(row,row['predictions'][condition],role,labels))
 if errors:
  C.write_json(OUT/'join_metadata_errors.json',{'status':'ORIGINAL7_EVAL128_METADATA_ERROR_NO_SCIENTIFIC_RESULT','errors':errors,
   'all128_prejoin_records_preserved':True,'excluded_or_inserted_targets':0,'prejoin_seal':bind(OUT/'prejoin_seal.json')})
  raise RuntimeError('CURATOR_METADATA_ERRORS; all128predictions preserved, aggregate scientific result withheld')
 need(all(len(v)==128 for v in actions.values()),'ALL128_RETAINED_AFTER_JOIN')
 all_metrics={name:metrics(aa) for name,aa in actions.items()};conditional={name:metrics([a for a in aa if a['candidate_recall']]) for name,aa in actions.items()}
 contrasts={name:paired_actions(aa,actions['REAL']) for name,aa in actions.items() if name!='REAL'}
 groups=primary_statistics(actions['REAL']);need(groups['group_count']==21,'GROUP21_PRIMARY_UNIT')
 raw_rescues=[a['query_id'] for a in actions['REAL'] if not a['base_correct'] and a['final_correct']]
 retention={}
 for name in CONDITIONS:
  good={a['query_id'] for a in actions[name] if a['final_correct']}
  retention[name]={'REAL_RAW_rescue_count':len(raw_rescues),'retained_query_ids':[q for q in raw_rescues if q in good],'lost_query_ids':[q for q in raw_rescues if q not in good]}
 ledger=[six_feature_ledger(row,action,w,b) for row,action in zip(records,actions['REAL'],strict=True)]
 real=all_metrics['REAL'];net=real['net']>0 and groups['equal_group_mean_accuracy_difference']>0;preserved=net and real['break']==0
 status='ORIGINAL7_EVAL128_INTERNAL_GAIN_AND_PRESERVATION' if preserved else 'ORIGINAL7_EVAL128_INTERNAL_NET_GAIN_WITH_BREAKS' if net else 'ORIGINAL7_EVAL128_NO_INTERNAL_POSITIVE_PAIRED_GAIN'
 value={'status':status,'theory_name':'new HYP','model':'frozen ORIGINAL7/NATIVE7 current seven parameters','parameter_sha256':PARAMETER_SHA,
  'authority':bind(C.AUTHORITY),'program':bind(PROGRAM),'plan':bind(PLAN),'worker_manifest':bind(WORKER),'curator_ledger':bind(CURATOR),
  'curator_metadata_validation':bind(METADATA),'prejoin_seal':bind(OUT/'prejoin_seal.json'),'prejoin_validation':bind(OUT/'prejoin_validation.json'),
  'query_count':128,'identity_count':24,'source_group_count':21,'corrected_gallery_mapping_sha256':mapping,
  'metrics_all128':all_metrics,'metrics_conditional_target_in_C128':conditional,'paired_controls_vs_REAL':contrasts,
  'primary_group_statistics':groups,'actions':actions,'six_feature_decision_ledger':ledger,'REAL_rescue_retention':retention,
  'primary_rescue_query_ids':raw_rescues,'primary_break_query_ids':[a['query_id'] for a in actions['REAL'] if a['base_correct'] and not a['final_correct']],
  'candidate_recall_miss_query_ids':[a['query_id'] for a in actions['REAL'] if not a['candidate_recall']],
  'internal_gain_and_preservation':preserved,'internal_net_gain_with_group_positive':net,'technical_or_metadata_errors':[],
  'source_context':'metadata-held-out from current43 identity/groups; source directories historically opened; not untouched external confirmation',
  'inference':'all originalRAW full5412 ranked identities; unchanged naturalC128; all127 challenger X@w+b; >0 SWITCH; physical-row ties',
  'statistical_scope':'21 source groups are the resampling/sign-flip units, not128 independent images or identities',
  'target_insertion_count':0,'target_absent_queries_dropped':0,'new_training_updates':0,'new_J_T_or_LP_calls':0,
  'deployment_changed':False,'universal_HYP_claimed':False,'ownership_claimed':False,
  'limits':['Column zeroing and C_BIND are fixed-head computational dependencies, not physical removal or spatial causality.',
   'All128 and conditional-on-C128 metrics remain separate; absent targets remain misses with full-gallery ranks.',
   'Bootstrap and sign-flip describe this historically opened source population and do not establish universal performance.',
   'No threshold, head coefficient, model, group, or image selection is performed using these evaluation outcomes.']}
 C.write_json(OUT/'result.json',value)
 C.write_json(OUT/'validation.json',{'status':'ORIGINAL7_EVAL128_POSTJOIN_LITERAL_REPLAY_PASS','result':bind(OUT/'result.json'),
  'checks':{'fresh_process_all162560_prejoin_logits_and_actions_replayed_before_curator':True,'all128_curator_bindings_checked':True,
   'candidate_absent_queries_retained':True,'full5412_MRR_checked_by_literal_move_to_front':True,'group21_bootstrap_and_exact_signflip_completed':True,
   'all_zero_column_and_binding_conditions_reported':True},'prejoin_logit_checks':logits,'fitted_models':0})
 write_report(value)
 print(json.dumps({'status':status,'result':bind(OUT/'result.json'),'REAL':real,'group_mean':groups['equal_group_mean_accuracy_difference'],
  'group_bootstrap95':groups['group_bootstrap'],'group_signflip':groups['two_sided_group_signflip'],'gain_and_preservation':preserved}),flush=True)


def write_report(value):
 m=value['metrics_all128']['REAL'];g=value['primary_group_statistics'];cond=value['metrics_conditional_target_in_C128']['REAL']
 lines=['# new HYP：冻结ORIGINAL7的128图完整证据评价','',
  '模型为当前ORIGINAL7/NATIVE7七参数，候选来自新计算的完整5412个去重reference RAW排名；自然C128不插入target。',
  '新128图与当前训练及旧EVAL的身份/组分离，覆盖24身份、21来源组；来源目录历史已打开，不称未触碰外部确认。','',
  f"主结果：candidate recall@128={m['candidate_recall_count']}/128；RAW {m['RAW_top1']}/128 → ORIGINAL7 {m['final_top1']}/128，救回{m['rescue']}、损失{m['break']}。",
  f"完整gallery MRR：{m['RAW_full_gallery_MRR']:.9f} → {m['final_full_gallery_MRR']:.9f}。",
  f"保留原正确且有新增：{value['internal_gain_and_preservation']}；技术/标签错误数：{len(value['technical_or_metadata_errors'])}。",'',
  f"仅target进入C128的条件子集：{cond['query_count']}张，RAW {cond['RAW_top1']} → REAL {cond['final_top1']}。此子集不替代全128指标。",'',
  f"21组等权准确率差：{g['equal_group_mean_accuracy_difference']:.9f}；10,000次组bootstrap 95%区间 [{g['group_bootstrap']['lower']:.9f}, {g['group_bootstrap']['upper']:.9f}]，seed20260910。",
  f"双侧精确组sign-flip p={g['two_sided_group_signflip']['two_sided_p']:.9g}；非零组{g['two_sided_group_signflip']['nonzero_groups']}。128图不是128个独立统计单位。",'',
  '| 条件 | 正确/128 | 救/损（对RAW） | 完整gallery MRR | REAL救回保留 |','| --- | ---: | --- | ---: | --- |']
 for name in CONDITIONS:
  row=value['metrics_all128'][name];ret=value['REAL_rescue_retention'][name]
  lines.append(f"| {name} | {row['final_top1']} | {row['rescue']}/{row['break']} | {row['final_full_gallery_MRR']:.9f} | {len(ret['retained_query_ids'])}/{ret['REAL_RAW_rescue_count']} |")
 lines+=['','所有128图保留；未进C128的target不伪造logit，按召回失败单列。六项、bias、SWITCH门及最强wrong竞争margin保存在机器账本中。',
  'C_BIND与置零是固定模型的计算依赖诊断，六列来源和代数耦合；它们不是图像物理干预、空间ownership或新训练模型。',
  '当前结果与旧matched32、旧FROZEN_C difficult90分别保留，不拼接准确率；没有自动部署或普遍HYP结论。',
  f"结果SHA：{sha(OUT/'result.json')}。",'']
 p=REPORT;p.parent.mkdir(parents=True,exist_ok=True);data='\n'.join(lines).encode()
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_REPORT_DRIFT');return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)


def known_TRAIN56_cross_runtime_fixture(w,b):
 # Public already-used TRAIN56 only; no curator/role joins and no new128 tensor or image reads.
 tokens=torch.load(OLD_TRAIN_TOKEN,map_location='cpu',weights_only=True,mmap=True)
 maps=torch.load(OLD_TRAIN_MAPS,map_location='cpu',weights_only=True,mmap=True)
 rows=[r for r in tokens['records'] if r['execution_ordinal']==56]
 visibility=[r for r in maps['records'] if r['execution_ordinal']==56]
 need(len(rows)==len(visibility)==1 and rows[0]['role']==visibility[0]['role']=='TRAIN','ONLY_KNOWN_TRAIN56_FIXTURE')
 row=rows[0];original=visibility[0];refs=tokens['references']
 need(row['query_id']==original['query_id'] and row['candidate_physical_rows']==original['candidate_physical_rows'] and tsha(row['query_tokens'])==original['query_tokens_sha256'],'TRAIN56_TOKEN_MAP_AXIS')
 axis=row['candidate_physical_rows'];evidence={};checks=0
 need(len(axis)==len(original['candidates'])==128,'ALL128_TRAIN56_CANDIDATES')
 for pos,c in enumerate(original['candidates']):
  physical=axis[pos];ref=refs[physical]
  need(c['physical_row']==physical and c['reference_tokens_sha256']==tsha(ref['tokens']),'TRAIN56_REFERENCE_TOKEN_BINDING')
  observed=independent_c4(row['query_tokens'],ref['tokens'],c['query_visibility'],c['reference_visibility'])
  for key in SCORE_KEYS:need(hx(observed[key])==hx(c['old_scores'][key]),'CROSS_RUNTIME_TRAIN56_C4_BITS:'+str(pos)+':'+key);checks+=1
  evidence[pos]=observed
 raw=row['candidate_raw_scores'];winner=axis.index(row['candidate_ranked_physical_rows'][0])
 public={'candidate_physical_rows':axis,'candidate_raw_scores_binary64':[hx(v) for v in raw],
  'base_winner_position':winner,'challenger_positions':[i for i in range(128) if i!=winner]}
 current=score_record(public,evidence,w,b)
 previous=next(r for r in read(OLD_NATIVE_PREJOIN) if r['execution_ordinal']==56)
 need(previous['query_id']==row['query_id'],'TRAIN56_OLD_HEAD_PREDICTION_QUERY')
 old_seal=read(OLD_NATIVE_SEAL);old_parameters=read(OLD_NATIVE_PARAMS)['ORIGINAL7']
 need(old_seal['parameters_sha256']==PINS[OLD_NATIVE_PARAMS] and old_seal['eval_prejoin_sha256']==PINS[OLD_NATIVE_PREJOIN] and old_parameters['parameter_sha256']==PARAMETER_SHA,'OLD_NATIVE_PREJOIN_BINDINGS')
 for mode in ('REAL','CBIND'):
  old=previous['predictions']['ORIGINAL7'][mode];new=current['predictions'][mode]
  for key in ('all127_logits_binary64','decision','final_position'):need(old[key]==new[key],'CROSS_RUNTIME_TRAIN56_HEAD_'+mode+':'+key)
 return {'status':'KNOWN_TRAIN56_CROSS_RUNTIME_C4_AND_ORIGINAL7_BIT_EXACT','execution_ordinal':56,'query_id':row['query_id'],
  'candidate_count':128,'C4_scalar_bit_checks':checks,'ORIGINAL7_REAL_and_CBIND_logit_bit_checks':254,
  'old_token_payload':bind(OLD_TRAIN_TOKEN),'old_visibility_payload':bind(OLD_TRAIN_MAPS),'old_ORIGINAL7_prejoin':bind(OLD_NATIVE_PREJOIN),
  'old_ORIGINAL7_parameters':bind(OLD_NATIVE_PARAMS),'old_ORIGINAL7_seal':bind(OLD_NATIVE_SEAL),
  'curator_or_role_reads':0,'new128_payload_reads':0,'new_encoder_or_RoMa_forwards':0,'model_training_updates':0,
  'runtime':runtime()}


def preflight():
 worker_manifest();w,b,par=parameters();old_fixture=known_TRAIN56_cross_runtime_fixture(w,b)
 from rc_aslo_xf.reference_visibility_pv_lossless_v1 import _legacy_score
 rng=torch.Generator().manual_seed(17);q=torch.randn((9,7),generator=rng,dtype=torch.float16);r=torch.randn((11,7),generator=rng,dtype=torch.float16)
 aq=torch.rand(9,generator=rng,dtype=torch.float64);ar=torch.rand(11,generator=rng,dtype=torch.float64)
 for wq,wr in ((aq,ar),(torch.zeros_like(aq),ar),(aq,torch.zeros_like(ar))):
  expected=independent_c4(q,r,wq,wr);s,m,_=_legacy_score(q,r,wq,wr)
  values=(s,m,_legacy_score(q,r,wq.roll(max(1,len(wq)//2)),wr)[0],_legacy_score(q,r,wq,wr.roll(max(1,len(wr)//2)))[0])
  need(all(hx(expected[k])==hx(v) for k,v in zip(SCORE_KEYS,values,strict=True)),'SYNTHETIC_LITERAL_C4_PARITY')
 raw=[float(i)/128 for i in range(128)];ev={i:{'real_score':.1+(i%9)*.01,'visibility_mass':.2+(i%7)*.01,'query_control_score':.03,'reference_control_score':.04} for i in range(128)}
 for c in range(127):need(tsha(FC.candidate_feature(raw,ev,c,127))==tsha(literal_feature(raw,ev,c,127)),'SYNTHETIC_ORIGINAL6_ARITHMETIC')
 need(exact_group_signflip([{'net':0,'query_count':6} for _ in range(21)])['two_sided_p']==1.,'ZERO_SIGNFLIP')
 need(exact_group_signflip([{'net':1,'query_count':6} for _ in range(3)])['p_numerator']==2,'EXACT_SIGNFLIP_EXTREME_TAIL')
 toy_labels=[f'toy_{i}' for i in range(5413)];toy_labels[715]=toy_labels[714]
 toy_rank=[i for i in range(5413) if i!=715]
 toy={'query_id':'SYNTHETIC_ONLY','execution_ordinal':0,'track':'outcome','candidate_physical_rows':list(range(128)),
  'challenger_positions':list(range(1,128)),'base_winner_position':0,'raw_ranked_physical_rows':toy_rank}
 def toy_action(target,switched):
  zz=torch.full((127,),-1.,dtype=torch.float64)
  if switched:zz[9]=.25
  role={'identity':toy_labels[target],'group':'TOY','original_query_id':'SYNTHETIC_ONLY'}
  return target_action(toy,predict(toy,zz),role,toy_labels)
 need(toy_action(0,False)['final_correct'] and toy_action(0,True)['final_target_rank_full_gallery']==2,'RAW_HOLD_SCORE0_AND_BREAK_RANK')
 need(toy_action(10,True)['final_target_rank_full_gallery']==1,'SWITCH_TARGET_MOVES_TO_TOP')
 need(toy_action(3,True)['final_target_rank_full_gallery']==5 and toy_action(20,True)['final_target_rank_full_gallery']==21,'FULL_GALLERY_TARGET_BEFORE_AFTER_MOVED_REFERENCE')
 for moved in (False,True):
  absent=toy_action(150,moved)
  need(absent['candidate_recall'] is False and absent['target_challenger_logit'] is None and absent['final_target_rank_full_gallery']==151 and not absent['final_correct'],'ABSENT_TARGET_RETAINED_NO_FAKE_LOGIT')
 need(metrics([])['query_count']==0 and metrics([])['final_accuracy'] is None,'EMPTY_CONDITIONAL_SUBSET_DEFINED')
 zero_group_actions=[{'group':f'toy_group_{g:02d}','base_correct':True,'final_correct':True} for g in range(21) for _ in range(7 if g<2 else 6)]
 zero_stats=primary_statistics(zero_group_actions)
 need(len(zero_group_actions)==128 and zero_stats['group_count']==21 and zero_stats['equal_group_mean_accuracy_difference']==0. and zero_stats['group_bootstrap']['lower']==zero_stats['group_bootstrap']['upper']==0. and zero_stats['two_sided_group_signflip']['two_sided_p']==1.,'PREDECLARED_21_GROUP_BOOTSTRAP_AND_SIGNFLIP_ZERO_CASE')
 value={'status':'ORIGINAL7_EVAL128_CPU_SYNTHETIC_PREFLIGHT_PASS','program':bind(PROGRAM),'plan':bind(PLAN),'common_program':bind(COMMON),
  'public_source_bindings':{'cpu_program':bind(PROGRAM),'cpu_plan':bind(PLAN),'cpu_worker_manifest':bind(WORKER),'cpu_current_ORIGINAL7_parameters':bind(HEAD),
   'cpu_feature_core':bind(FEATURE_CORE),'cpu_legacy_C4_score_core':bind(SCORE_CORE),'cpu_gallery_identity_core':bind(IDENTITY_CORE),
   'cpu_gallery_identity_contract':bind(IDENTITY_CONTRACT),'cpu_gallery_identity_registry':bind(IDENTITY_REGISTRY),
   'cpu_old_TRAIN56_tokens':bind(OLD_TRAIN_TOKEN),'cpu_old_TRAIN56_maps':bind(OLD_TRAIN_MAPS),'cpu_old_NATIVE7_prejoin':bind(OLD_NATIVE_PREJOIN),
   'cpu_old_NATIVE7_parameters':bind(OLD_NATIVE_PARAMS),'cpu_old_NATIVE7_seal':bind(OLD_NATIVE_SEAL),**{'cpu_launcher_'+stage:bind(path) for stage,path in LAUNCHERS.items()}},
  'head_parameter_sha256':par['parameter_sha256'],'known_TRAIN56_cross_runtime_fixture':old_fixture,'conditions':list(CONDITIONS),'synthetic_C4_checks':12,'synthetic_feature_checks':127,
  'new128_query_payload_reads':0,'known_TRAIN56_public_token_and_map_payload_reads':2,'curator_payload_reads':0,'model_training_updates':0,'new_encoder_or_RoMa_forwards':0,
  'synthetic_absent_target_and_full_gallery_MRR_cases':7,'synthetic_group_bootstrap_cases':1,
  'runtime':runtime()}
 need(not BLOCKED and not RELEASED,'PREFLIGHT_PRIVATE_GUARD')
 p=OUT.parent/(OUT.name+'_preflight')/sha(PROGRAM)/'result.json';C.write_json(p,value)
 print(json.dumps({'status':value['status'],'preflight':bind(p),'parameter_sha256':par['parameter_sha256']}),flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('preflight','raw_aggregate','finalize_prejoin','join_validated'));phase=parser.parse_args().phase
 torch.set_num_threads(8);torch.set_num_interop_threads(1);setup()
 if phase=='preflight':preflight();return
 a=require(phase)
 try:
  with torch.inference_mode():
   if phase=='raw_aggregate':raw_aggregate(a)
   elif phase=='finalize_prejoin':finalize_prejoin(a)
   else:join_validated(a)
 except Exception as e:
  C.write_json(OUT/'engineering_errors'/f'{phase}-{os.environ.get("SLURM_JOB_ID","unknown")}-{os.getpid()}.json',
   {'status':'ORIGINAL7_EVAL128_ENGINEERING_OR_METADATA_INCOMPLETE','phase':phase,'error_type':type(e).__name__,'message':str(e),
    'authority':bind(C.AUTHORITY),'program':bind(PROGRAM),'completed_upstream_shards_preserved':True,'partial_result_is_scientific_NO_GO':False})
  raise

if __name__=='__main__':main()
