#!/usr/bin/env python3
"""TRAIN128 anonymous RAW/C4 qualification; no labels, head prediction or fitting."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time,ast
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];PROGRAM=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'programs')]
import rc_original7_train128_execution_common_v1 as C
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
PLAN=ROOT/'plan/RC_ORIGINAL7_TRAIN128_INPUTS_V1_20260910.md'
WORKER=ROOT/'results/rc_original7_train128_manifest_v1/worker_manifest.json'
CURATOR=WORKER.with_name('curator_roles.json')
RAW=ROOT/'results/rc_original7_train128_token_raw_v1'
ROMA=ROOT/'results/rc_original7_train128_roma_v1'
OUT=ROOT/'results/rc_original7_train128_inputs_v1'
OLD_CPU=ROOT/'programs/run_rc_original7_eval128_full_evidence_v2_compat.py'
OLD_CPU_SHA='806a8506693e2d9e88aba4331702c2d2c8df58e4d3f693b9b30e0601de49304c'
GALLERY_IMAGES=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
SCORE_KEYS=('real_score','visibility_mass','query_control_score','reference_control_score')
HASH_DEPTH=0;BLOCKED=[]
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if any(x in s for x in ('d1_mi','d1-mi','/grozi/','gisc_prerecall_universe','/rc_opened_eval_strict_','direction_capacity','angle_capacity')):
  BLOCKED.append(str(p));raise PermissionError('PROTECTED_INPUT_FORBIDDEN')
 if not HASH_DEPTH and (p==CURATOR.resolve() or '/target_join/' in s or '/role_shards/' in s or '/rc_original7_eval128_full_evidence_v1/' in s):
  BLOCKED.append(str(p));raise PermissionError('LABEL_OR_EVAL_READ_DURING_TRAIN_INPUT_MATERIALIZATION')
sys.addaudithook(audit)

def need(v,m):
 if not bool(v):raise RuntimeError(m)

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

def runtime():
 return {'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}

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
 need(receipt['status']=='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_SHARD_READY' and validation['status']=='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_CPU_REPLAY_PASS','RAW_STAGE_QUALIFIED')
 payload_path=checked(receipt['payload']);need(payload_path==(folder/'payload.pt').resolve(),'RAW_PAYLOAD_PATH')
 need(checked(validation['payload'])==payload_path and checked(validation['receipt'])==(folder/'receipt.json').resolve(),'RAW_VALIDATOR_BINDINGS')
 need(checked(validation['authority'])==C.AUTHORITY.resolve(),'RAW_COMMON_AUTHORITY')
 data=torch.load(payload_path,map_location='cpu',weights_only=True)
 need(data['status']=='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_SHARD_READY' and data['shard']==shard and data['shard_count']==16,'RAW_SHARD_HEADER')
 need(checked(data['bindings']['authority'])==C.AUTHORITY.resolve() and checked(data['bindings']['worker_manifest'])==WORKER.resolve(),'RAW_PAYLOAD_AUTHORITY_WORKER_BINDING')
 need(validation['fresh_CPU_process'] is True and all(x is True for x in validation['checks'].values()),'RAW_FRESH_CPU_QUALIFICATION')
 need(validation['legacy_full32_parity_count']==(8 if shard<4 else 0) and validation['all_original_TRAIN_bits_exact'] is True,'ORIGINAL_FULL32_RAW_TOKEN_PARITY')
 need(data['legacy_full32_parity']==receipt['legacy_full32_parity'] and len(data['legacy_full32_parity'])==(8 if shard<4 else 0),'ORIGINAL_FULL32_PARITY_RECORDS')
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
 value={'status':'ORIGINAL7_TRAIN128_RAW_AGGREGATE_PASS','authority':bind(C.AUTHORITY),'program':bind(PROGRAM),'worker_manifest':bind(WORKER),
  'query_count':128,'gallery_physical_count':5413,'gallery_identity_count':5412,'corrected_gallery_mapping_sha256':mapping,
  'shards':entries,'records':records,'independent_full_RAW_order_and_C128_replayed':True,'current_runtime_tokens_from_bound_payloads':True,
  'legacy_redacted_tokens_used':False,'curator_payload_reads':0,'candidate_insertion_count':0,'new_training_updates':0,'old_FULL32_token_RAW_parity_count':32,'all32_original_token_RAW_bits_exact':True}
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

def load_roma_shard(shard,raw_source,raw_records,refs):
 folder=ROMA/f'shard{shard:02d}';receipt=read(folder/'receipt.json');validation=read(folder/'validation.json')
 need(receipt['status']=='RC_ORIGINAL7_TRAIN128_ROMA_SHARD_READY' and validation['status']=='RC_ORIGINAL7_TRAIN128_ROMA_CPU_REPLAY_PASS','ROMA_SHARD_QUALIFIED')
 p=checked(receipt['payload']);need(p==(folder/'payload.pt').resolve() and checked(validation['payload'])==p,'ROMA_PAYLOAD_BINDING')
 need(checked(validation['receipt'])==(folder/'receipt.json').resolve() and checked(validation['authority'])==C.AUTHORITY.resolve(),'ROMA_VALIDATOR_BINDING')
 need(validation['query_count']==8 and validation['candidate_occurrence_count']==1024 and validation['independently_recomputed_C4_scalars']==4096 and validation['all_C4_bits_exact'] is True,'ALL_C4_SHARD_QUALIFICATION_COUNTS')
 need(validation['target_role_read_count']==validation['model_update_count']==0 and validation['fresh_child_nonce'],'ROMA_NO_TARGETS_OR_FIT')
 need(validation['legacy_full32_parity_count']==(8 if shard<4 else 0) and validation['all_original_TRAIN_maps_C4_bits_exact'] is True,'ORIGINAL_FULL32_ROMA_MAP_C4_PARITY')
 data=torch.load(p,map_location='cpu',weights_only=True)
 need(data['status']=='RC_ORIGINAL7_TRAIN128_ROMA_SHARD_READY' and data['kind']=='shard' and data['shard']==shard and len(data['records'])==8,'ROMA_SHARD_HEADER')
 need(data['legacy_full32_parity']==receipt['legacy_full32_parity'] and len(data['legacy_full32_parity'])==(8 if shard<4 else 0) and all(x['all_bits_exact'] for x in data['legacy_full32_parity']),'ORIGINAL_FULL32_ROMA_PARITY_RECORDS')
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


def worker_manifest():
 m=read(WORKER);need(m['status']=='TRAIN128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and m['query_count']==128,'FROZEN_TRAIN128_WORKER')
 rows=m['records'];need(len(rows)==128 and [r['execution_ordinal'] for r in rows]==list(range(128)),'EXACT128_EXECUTION_AXIS')
 need(len({r['query_id'] for r in rows})==len({r['source_image_sha256'] for r in rows})==128,'DISTINCT128_IMAGES')
 for r in rows:need(r['query_id'].startswith('T128-') and not ({'target','identity','group','target_position','original_query_id'}&set(r)),'ANONYMOUS_WORKER')
 return {r['execution_ordinal']:r for r in rows}

def require(phase):
 need(sha(OLD_CPU)==OLD_CPU_SHA,'FROZEN_OPERATOR_SOURCE')
 a=C.require_authority(phase,PROGRAM);C.require_reused_engineering_bridges()
 need(a['source_bindings']['cpu_worker_manifest']==bind(WORKER),'WORKER_BINDING')
 pre=read(checked(a['source_bindings']['cpu_preflight']))
 need(pre['status']=='ORIGINAL7_TRAIN128_INPUTS_CPU_PREFLIGHT_PASS' and pre['runtime']==runtime(),'CPU_PREFLIGHT_RUNTIME')
 return a

def qualify_train(a):
 aggregate=read(OUT/'raw_aggregate.json');need(aggregate['status']=='ORIGINAL7_TRAIN128_RAW_AGGREGATE_PASS' and aggregate['authority']==bind(C.AUTHORITY),'TRAIN_RAW_AGGREGATE')
 workers=worker_manifest();labels,mapping=gallery_labels();need(mapping==aggregate['corrected_gallery_mapping_sha256'],'IDENTITY_MAPPING')
 records=[];raw_sources=[];roma_sources=[];started=time.monotonic()
 for shard in range(16):
  raw,refs,public,rs=load_raw_shard(shard,a,workers,labels)
  need(rs==aggregate['shards'][shard] and public==aggregate['records'][shard*8:(shard+1)*8],'RAW_REPLAY')
  evidence,es=load_roma_shard(shard,rs,raw,refs);raw_sources.append(rs);roma_sources.append(es)
  for row,ev in zip(public,evidence,strict=True):
   need(row['query_id']==ev['query_id'] and row['execution_ordinal']==ev['execution_ordinal'],'C4_JOIN')
   scores=[float.fromhex(x) for x in row['candidate_raw_scores_binary64']];winner=row['base_winner_position'];features={}
   for mode in ('REAL','CBIND'):
    values=ev['evidence'] if mode=='REAL' else {i:ev['evidence'][(i+64)%128] for i in range(128)}
    matrix=torch.stack([FC.candidate_feature(scores,values,c,winner) for c in row['challenger_positions']])
    literal=torch.stack([literal_feature(scores,values,c,winner) for c in row['challenger_positions']])
    need(tsha(matrix)==tsha(literal),'ALL_ORIGINAL_FEATURE_BITS');features[mode]=[[hx(x) for x in r] for r in matrix]
   records.append({**row,'C4_binary64':{str(i):{k:hx(x) for k,x in v.items()} for i,v in ev['evidence'].items()},'features_binary64':features,'candidate_map_token_bindings':ev['map_token_bindings']})
  print(json.dumps({'event':'TRAIN128_INPUT_SHARD_QUALIFIED','shard':shard,'queries':len(records),'C4_scalar_checks':sum(x['independently_recomputed_scalar_count'] for x in roma_sources),'elapsed_seconds':time.monotonic()-started}),flush=True)
 need(len(records)==128 and sum(x['independently_recomputed_scalar_count'] for x in roma_sources)==65536 and not BLOCKED,'ALL128_INPUTS_QUALIFIED')
 C.write_json(OUT/'feature_records.json',records)
 v={'status':'ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS','authority':bind(C.AUTHORITY),'program':bind(PROGRAM),'worker_manifest':bind(WORKER),'raw_aggregate':bind(OUT/'raw_aggregate.json'),'feature_records':bind(OUT/'feature_records.json'),
  'query_count':128,'candidate_occurrences':16384,'independent_C4_scalar_checks':65536,'independent_feature_scalar_checks':128*127*6*2,'raw_shards':raw_sources,'roma_shards':roma_sources,'head_predictions':0,'training_updates':0,'curator_reads':0,'EVAL_result_reads':0,'candidate_insertions':0,'forbidden_read_attempts':len(BLOCKED),'original_FULL32_token_RAW_map_C4_parity_count':32,'all32_original_input_bits_exact':True}
 C.write_json(OUT/'validation.json',v);print(json.dumps({'status':v['status'],'validation':bind(OUT/'validation.json')}),flush=True)

def preflight():
 worker_manifest();need(sha(OLD_CPU)==OLD_CPU_SHA,'PARENT_SOURCE_SHA')
 # Verify the copied numerical operators were not changed in this workload adapter.
 original=ast.parse(OLD_CPU.read_text());current=ast.parse(PROGRAM.read_text())
 names=('gallery_labels','as_tensor','verify_raw_row','independent_score','independent_c4','literal_feature')
 for n in names:
  x=next(v for v in original.body if isinstance(v,ast.FunctionDef) and v.name==n);y=next(v for v in current.body if isinstance(v,ast.FunctionDef) and v.name==n)
  need(ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),'UNCHANGED_NUMERICAL_OPERATOR:'+n)
 from rc_aslo_xf.reference_visibility_pv_lossless_v1 import _legacy_score
 gen=torch.Generator().manual_seed(17);q=torch.randn((9,7),generator=gen,dtype=torch.float16);r=torch.randn((11,7),generator=gen,dtype=torch.float16);aq=torch.rand(9,generator=gen,dtype=torch.float64);ar=torch.rand(11,generator=gen,dtype=torch.float64)
 for wq,wr in ((aq,ar),(aq*0,ar),(aq,ar*0)):
  got=independent_c4(q,r,wq,wr);s,m,_=_legacy_score(q,r,wq,wr)
  values=(s,m,_legacy_score(q,r,wq.roll(max(1,len(wq)//2)),wr)[0],_legacy_score(q,r,wq,wr.roll(max(1,len(wr)//2)))[0])
  need(all(hx(got[k])==hx(v) for k,v in zip(SCORE_KEYS,values,strict=True)),'SYNTHETIC_C4_BITS')
 raw=[i/128 for i in range(128)];ev={i:{'real_score':.1+(i%9)*.01,'visibility_mass':.2+(i%7)*.01,'query_control_score':.03,'reference_control_score':.04} for i in range(128)}
 for c in range(127):need(tsha(FC.candidate_feature(raw,ev,c,127))==tsha(literal_feature(raw,ev,c,127)),'SYNTHETIC_FEATURE_BITS')
 value={'status':'ORIGINAL7_TRAIN128_INPUTS_CPU_PREFLIGHT_PASS','program':bind(PROGRAM),'common_program':bind(Path(C.__file__)),'parent_source':bind(OLD_CPU),'worker_manifest':bind(WORKER),'plan':bind(PLAN),'runtime':runtime(),'unchanged_operator_AST':list(names),'synthetic_C4_checks':12,'synthetic_feature_checks':127,'new_natural_data_reads':0,'curator_reads':0,'training_updates':0}
 p=OUT.parent/(OUT.name+'_preflight')/sha(PROGRAM)/'result.json';C.write_json(p,value);print(json.dumps({'status':value['status'],'preflight':bind(p)}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('preflight','raw_aggregate','qualify_train'),required=True);phase=p.parse_args().phase
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 if phase=='preflight':preflight();return
 a=require(phase)
 with torch.inference_mode():
  if phase=='raw_aggregate':raw_aggregate(a)
  else:qualify_train(a)
if __name__=='__main__':main()
