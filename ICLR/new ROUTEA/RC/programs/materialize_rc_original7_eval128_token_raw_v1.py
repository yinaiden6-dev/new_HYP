#!/usr/bin/env python3
"""Fresh current-runtime ColNomic TOKEN/RAW stage for the opaque frozen128.

Old TRAIN56 closes the encoder/RAW/reference interface first. Expansion never
loads legacy query tokens, curator data, or the RGHFull600 query loader. Output
payloads contain only plain containers/scalars and CPU tensors.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
from importlib import metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
import torch
from PIL import Image,ImageOps
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'programs')]
import rc_original7_eval128_execution_common_v1 as C
PROGRAM=Path(__file__).resolve()
PLAN=ROOT/'plan/RC_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md'
MANIFEST_ROOT=ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
WORKER=MANIFEST_ROOT/'worker_manifest.json'
WORKER_SHA='f2c2e6f9181e103a61a181edf43e3309aa1de72743cfa6cf6097a9deb108e4e7'
METADATA_VALID=MANIFEST_ROOT/'independent_metadata_validation.json'
METADATA_VALID_SHA='a70936a24fc2f5c6996228aee673a6de505c3022487b903878d268ad9a34cd94'
OUT=ROOT/'results/rc_original7_eval128_token_raw_v1'
PREFLIGHT=ROOT/'results/rc_original7_eval128_token_raw_v1_preflight'
ROMA_BRIDGE=ROOT/'results/rc_original7_eval128_roma_v1/bridge/validation.json'
MODEL=WORKSPACE/'models/downloaded_models/colnomic-embed-multimodal-7b'
GALLERY=WORKSPACE/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
GALLERY_SHA='11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc'
SPATIAL=ROOT/'cache/conditional_colnomic_p_spatial_v1'
SHARED=ROOT/'results/rc_shared_query_target_prior_cache_v1'
OLD_TOKENS=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1'
OLD_INPUTS=ROOT/'results/rc_original_raw_visibility_pv_inputs_v1'
LEGACY_INDEX=ROOT/'cache/l0_natural_hardneg_v2_redacted_query_tokens_v1/index_v1.json'
LAUNCH_BRIDGE=ROOT/'slurm/rc_original7_eval128_token_raw_bridge_v1_30m.sbatch'
LAUNCH_SHARDS=ROOT/'slurm/rc_original7_eval128_token_raw_shards_v1_30m.sbatch'
FRAMES={'outcome':'DECODED_RAW_BEFORE_EXIF','difficult':'DECODED_RAW_BEFORE_EXIF','new_difficult_train':'EXIF_ORIENTED_BEFORE_RESIZE'}
PHASE=None;BLOCKED=[]
CONTRACT={'query_count':128,'shards':16,'queries_per_shard':8,'encoder':'ColQwen2_5.from_pretrained(frozen_ColNomic7B,torch_dtype=bfloat16).eval.requires_grad_false',
 'query_frame':FRAMES,'query_image_and_template_tokens':'FP16_from_encoded.float_image_mask_preserving_original_order',
 'RAW':'frozen_v2.full_gallery_scores;FP32_sum_MaxSim;batch16;highest_matmul_precision;full5413',
 'identity_reduction':'stable_descending_physical_score_then_corrected_identity_dedup_full5412',
 'candidate_axis':'top128_corrected_identity_representatives_sorted_physical',
 'reference_resolver':'frozen_HybridSpatialReferenceResolver_gallery_only_validated4976_and_same_CPU_recovery',
 'first_bridge_execution_ordinal':56,'expansion_requires_token_and_RoMa_bridges':True,
 'target_insertion':False,'drop_target_absent_queries':False,'query_labels_or_private_curator_reads':0,
 'new_training_updates':0,'J_T_LP_required':False}

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 # Deliberately keep opaque symlink path spelling, rather than expose its target.
 p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();bad=None
 if 'curator_roles' in s or any(x in s for x in ('/role_shards/','/target_join/','/rc_opened_eval_strict_','d1_mi','d1-mi','grozi','gisc_prerecall_universe')):bad='PRIVATE_ROLE_OR_PROTECTED_SOURCE_FORBIDDEN'
 if '/worker_assets/tokens/' in s:bad='NEW128_LEGACY_TOKENS_NEVER_USED'
 if '/worker_assets/images/' in s and PHASE not in ('shard','validate-shard'):bad='NEW128_PIXELS_BEFORE_EXPANSION_PHASE'
 if '/conditional_colnomic_p_spatial_v1/query/' in s:bad='UNRELATED_QUERY_SPATIAL_CACHE_FORBIDDEN'
 if bad:BLOCKED.append(str(p));raise RuntimeError(bad)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 # Scientific source files use explicit paths; query aliases are never sent here.
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def tensor_sha(t):
 t=t.detach().cpu().contiguous();return hashlib.sha256(str(t.dtype).encode('ascii')+json.dumps(list(t.shape),separators=(',',':')).encode('ascii')+t.view(torch.uint8).numpy().tobytes()).hexdigest()
def path_of(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==v['sha256'],'SOURCE_HASH_DRIFT:'+str(p));return p

def runtime_libraries():
 versions={}
 for name in ('torch','transformers','colpali-engine','peft','accelerate','safetensors','tokenizers','numpy','pillow'):
  try:versions[name]=metadata.version(name)
  except metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
 return {'python':sys.version,'executable':sys.executable,'packages':versions,'torch_cuda_build':torch.version.cuda,
  'torch_float32_matmul_precision':torch.get_float32_matmul_precision(),'torch_threads':torch.get_num_threads()}

def worker_manifest():
 need(sha(WORKER)==WORKER_SHA and sha(METADATA_VALID)==METADATA_VALID_SHA,'FROZEN_OPAQUE_COHORT_BINDING')
 d=read(WORKER);need(d['status']=='EXPANDED_EVAL128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and d['query_count']==128,'WORKER_MANIFEST_SCOPE')
 expected={'execution_ordinal','grid_hw','image_tokens_sha256','query_id','query_image_path','redacted_token_artifact',
  'redacted_token_artifact_declared_sha256','source_image_sha256','template_tokens_sha256','track'}
 need([r['execution_ordinal'] for r in d['records']]==list(range(128)) and len({r['query_id'] for r in d['records']})==128,'FULL_OPAQUE128_AXIS')
 for r in d['records']:
  need(set(r)==expected and r['query_id'].startswith('E128-') and r['track'] in FRAMES,'WORKER_ROW_SCHEMA')
  p=Path(r['query_image_path']);need(p.is_absolute() and p.parent==MANIFEST_ROOT/'worker_assets/images' and p.is_symlink(),'ANONYMOUS_IMAGE_ALIAS')
 return d

def old_train_metadata():
 need(sha(SHARED/'manifest.json')=='f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c','OLD_SHARED_SOURCE')
 rows=sorted((e for e in read(SHARED/'manifest.json')['records'] if e['kind']=='FULL' and e['role']=='TRAIN'),key=lambda e:e['execution_ordinal'])[:4]
 need([e['execution_ordinal'] for e in rows]==[56,66,72,75],'FIRST_FOUR_OLD_TRAIN_FIXED')
 return [read(path_of(e)) for e in rows]

def cpu_legacy_probe():
 ix=read(LEGACY_INDEX);byq={r['query_id']:r for r in ix['records']};records=[]
 for row in old_train_metadata():
  ap=path_of(row['arrays'])
  with np.load(ap,allow_pickle=False) as z:current=torch.from_numpy(z['q_tokens'].copy())
  old=byq[row['query_id']];p=Path(old['artifact']);p=p if p.is_absolute() else LEGACY_INDEX.parent/p
  need(sha(p)==old['artifact_file_sha256'],'OLD_TRAIN_REDACTED_SOURCE')
  payload=torch.load(p,map_location='cpu',mmap=True,weights_only=True);legacy=payload['image_tokens']
  same=current.shape==legacy.shape and current.dtype==legacy.dtype and tensor_sha(current)==tensor_sha(legacy)
  records.append({'execution_ordinal':row['execution_ordinal'],'query_id':row['query_id'],'current_shape':list(current.shape),'legacy_shape':list(legacy.shape),
   'current_tokens_sha256':tensor_sha(current),'legacy_tokens_sha256':tensor_sha(legacy),'image_token_bits_exact':same,
   'maximum_abs_difference':float((current.float()-legacy.float()).abs().max()) if current.shape==legacy.shape else None,
   'current_source':bind(ap),'legacy_source':bind(p),'legacy_template_shape':list(payload['template_tokens'].shape)})
 return {'status':'OLD_TRAIN_ONLY_CPU_TOKEN_SOURCE_COMPARISON_COMPLETE','records':records,'new128_payload_reads':0,'GPU_forwards':0,
  'decision':'FRESH_CURRENT_RUNTIME_ENCODER_ALWAYS;legacy_tokens_not_substituted'}

def reference_inventory():
 bindings={};union=[]
 for lo,hi in ((0,1244),(1244,2488),(2488,3732),(3732,4976)):
  rp=SPATIAL/'receipts'/f'gallery_{lo:04d}_{hi:04d}.json';vp=SPATIAL/'validation'/f'gallery_{lo:04d}_{hi:04d}.json'
  r,v=read(rp),read(vp);need(v['status']=='PASS' and all(x is True for x in v['checks'].values()),'REFERENCE_SPATIAL_VALIDATION')
  p=ROOT/r['cache'];need(sha(p)==r['cache_file_sha256']==v['cache_file_sha256'],'REFERENCE_SPATIAL_HASH')
  d=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
  need(d['kind']=='gallery' and d['range']==[lo,hi] and len(d['entries'])==hi-lo,'REFERENCE_SPATIAL_RANGE')
  for i,e in enumerate(d['entries']):need(e['kind']=='gallery' and e['work_ordinal']==lo+i,'REFERENCE_ORDINAL_ORDER');union.append(int(e['physical_row']))
  for tag,path in (('payload',p),('receipt',rp),('validation',vp)):bindings[f'token_raw_reference_{lo:04d}_{tag}']=bind(path)
 need(len(union)==4976 and union==sorted(set(union)),'QUALIFIED_4976_REFERENCE_UNION')
 return union,bindings

def public_sources():
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver,PROCESSOR_CONFIG_FILES
 files={'token_raw_program':PROGRAM,'token_raw_plan':PLAN,'token_raw_worker_manifest':WORKER,'token_raw_metadata_validation':METADATA_VALID,
  'token_raw_common_program':Path(C.__file__),'token_raw_bridge_launcher':LAUNCH_BRIDGE,'token_raw_shard_launcher':LAUNCH_SHARDS,
  'token_raw_legacy_encoder_program':ROOT/'programs/materialize_romav2_colnomic_current_runtime_bridge_shard_v1.py',
  'token_raw_fullrank_program':ROOT/'programs/run_romav2_colnomic_sealed_source_e0_v2.py',
  'token_raw_reference_resolver_program':Path(inspect.getfile(HybridSpatialReferenceResolver)),
  'token_raw_gallery_source_program':ROOT/'src/rc_aslo_xf/conditional_rep_sources.py',
  'token_raw_reference_cache_program':ROOT/'src/rc_aslo_xf/conditional_colnomic_ms_proposal.py',
  'token_raw_gallery_identity_core':ROOT/'src/rc_aslo_xf/gallery_identity_repair.py',
  'token_raw_gallery_identity_manifest':ROOT/'registry/gallery_identity_repair_v1.json',
  'token_raw_gallery_identity_contract':ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json',
  'token_raw_gallery_cache':GALLERY,'token_raw_old_shared_manifest':SHARED/'manifest.json',
  'token_raw_old_runtime_validation':OLD_TOKENS/'validation.json','token_raw_encoder_class_source':Path(inspect.getfile(ColQwen2_5)),
  'token_raw_processor_class_source':Path(inspect.getfile(ColQwen2_5_Processor))}
 for name in PROCESSOR_CONFIG_FILES:files['token_raw_model_'+name.replace('.','_')]=MODEL/name
 files['token_raw_adapter_weights']=MODEL/'adapter_model.safetensors'
 base=Path(read(MODEL/'adapter_config.json')['base_model_name_or_path'])
 need(base.is_absolute() and base.is_dir(),'FROZEN_LOCAL_BASE_MODEL')
 for name in ('config.json','generation_config.json','model.safetensors.index.json'):files['token_raw_base_'+name.replace('.','_')]=base/name
 index=read(base/'model.safetensors.index.json')
 for n,name in enumerate(sorted(set(index['weight_map'].values()))):files[f'token_raw_base_weights_{n:02d}']=base/name
 for module_name in ('transformers.models.qwen2_5_vl.modeling_qwen2_5_vl','transformers.models.qwen2_5_vl.processing_qwen2_5_vl','transformers.models.qwen2_vl.image_processing_qwen2_vl','peft.tuners.lora.layer'):
  module=importlib.import_module(module_name);files['token_raw_library_'+module_name.replace('.','_')]=Path(module.__file__)
 old_meta=old_train_metadata()[0];files['token_raw_bridge_expected_payload']=path_of(old_meta['source']['tokens'])
 bindings={}
 for k,p in files.items():
  bindings[k]=bind(p)
  if k.startswith('token_raw_base_weights_'):print(json.dumps({'event':'MODEL_WEIGHT_SOURCE_HASHED','source':k,'bytes':Path(p).stat().st_size}),flush=True)
 need(bindings['token_raw_gallery_cache']['sha256']==GALLERY_SHA,'RAW_GALLERY_PIN')
 union,refs=reference_inventory();bindings.update(refs)
 return bindings,union

def preflight():
 worker_manifest();sources,union=public_sources();probe=cpu_legacy_probe()
 for i,r in enumerate(probe['records']):
  sources[f'token_raw_cpu_probe_current_{i}']=r['current_source'];sources[f'token_raw_cpu_probe_legacy_{i}']=r['legacy_source']
 value={'status':'RC_ORIGINAL7_EVAL128_TOKEN_RAW_PREFLIGHT_PASS','public_source_bindings':sources,'legacy_reference_union':union,
  'contract':CONTRACT,'runtime_libraries':runtime_libraries(),'old_train_cpu_probe':probe,'old_TRAIN56_fixture_probe':fixture_probe(),
  'synthetic_RAW_rank_probe':rank_probe(),'new128_pixels_or_tokens_read':0,'GPU_forwards':0}
 key=hashlib.sha256(encode(sources)).hexdigest();p=PREFLIGHT/(key+'.json');C.write_json(p,value)
 print(json.dumps({'status':value['status'],'preflight':bind(p),'old_train_image_bits_equal':[r['image_token_bits_exact'] for r in probe['records']],'new128_payload_reads':0}),flush=True)

def checked_authority(stage):
 a=C.require_authority(stage,PROGRAM)
 need('token_raw_preflight' in a['source_bindings'],'TOKEN_PREFLIGHT_BINDING_REQUIRED')
 p=C.bound_path(a['source_bindings']['token_raw_preflight']);pre=read(p)
 need(pre['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_PREFLIGHT_PASS' and pre['contract']==CONTRACT,'TOKEN_PREFLIGHT_CONTRACT')
 for k,b in pre['public_source_bindings'].items():need(a['source_bindings'].get(k)==b,'PUBLIC_SOURCE_MISSING_FROM_MASTER:'+k)
 need(pre['runtime_libraries']==runtime_libraries(),'CURRENT_LIBRARY_OR_FP_SETTINGS_DRIFT')
 return a,pre

def old_bridge_expected():
 row=old_train_metadata()[0];need(row['execution_ordinal']==56 and row['role']=='TRAIN','ONLY_FIXED_OLD_TRAIN56')
 binding=row['source']['tokens'];p=path_of(binding);payload=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
 old=payload['records'][int(binding['record_index'])]
 need(old['execution_ordinal']==56 and old['query_id']==row['query_id'] and old['role']=='TRAIN','OLD_TRAIN_TOKEN_QUERY_BINDING')
 need(tensor_sha(old['query_tokens'])==row['q_tokens_sha256'] and old['candidate_physical_rows']==row['axis'],'OLD_TRAIN_TOKEN_AXIS_BINDING')
 return row,old,payload['references'],bind(p)

def gallery_source_and_resolver(pre,processor,passages):
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 from rc_aslo_xf.conditional_colnomic_ms_proposal import ValidatedSpatialCacheResolver
 from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver
 class GalleryOnlyValidatedResolver(ValidatedSpatialCacheResolver):
  def _scan(self,kind):return [] if kind=='query' else super()._scan(kind)
 gallery=build_gallery_source(verify_cache_file_sha256=True)
 cache=GalleryOnlyValidatedResolver(SPATIAL)
 resolver=HybridSpatialReferenceResolver(gallery_source=gallery,spatial_cache_resolver=cache,
  legacy_candidate_union=pre['legacy_reference_union'],processor_factory=lambda:processor,gallery_embedding_loader=lambda:passages)
 return gallery,resolver

def gallery_bindings(gallery):
 return {key:getattr(gallery,key) for key in ('gallery_cache_sha256','raw_path_sequence_sha256','legacy_setid_sequence_sha256',
  'corrected_mapping_sha256','repair_contract_sha256','repair_manifest_sha256')}

def complete_rank(scores,labels):
 seen=set();order=[]
 for physical in torch.argsort(scores,descending=True,stable=True).tolist():
  label=labels[int(physical)]
  if label in seen:continue
  seen.add(label);order.append(int(physical))
 need(len(order)==5412 and len(set(order))==5412,'FULL_CORRECTED_IDENTITY_POPULATION')
 return order

def plain_only(value,path='payload'):
 if isinstance(value,torch.Tensor):need(value.device.type=='cpu','CPU_TENSORS_ONLY:'+path);return
 if type(value) in (str,int,float,bool,type(None)):return
 if type(value) in (list,tuple):
  for i,x in enumerate(value):plain_only(x,path+'.'+str(i))
  return
 if type(value) is dict:
  for k,x in value.items():need(type(k) in(str,int),'PLAIN_KEY_ONLY');plain_only(x,path+'.'+str(k))
  return
 raise RuntimeError('CUSTOM_PICKLE_CLASS_FORBIDDEN:'+path+':'+str(type(value)))

def fixture_probe():
 _,old,references,source=old_bridge_expected()
 selected={int(p):references[int(p)] for p in old['candidate_physical_rows']}
 fixture={'records':[old],'references':selected};plain_only(fixture)
 stream=io.BytesIO();torch.save(fixture,stream);stream.seek(0)
 replay=torch.load(stream,map_location='cpu',weights_only=True);plain_only(replay)
 need(tensor_sha(replay['records'][0]['query_tokens'])==tensor_sha(old['query_tokens']),'FIXTURE_QUERY_PLAIN_ROUNDTRIP')
 need(all(tensor_sha(replay['references'][p]['tokens'])==tensor_sha(selected[p]['tokens']) for p in selected),'FIXTURE_REFERENCE_PLAIN_ROUNDTRIP')
 return {'status':'OLD_TRAIN56_PLAIN_CPU_FIXTURE_PASS','source':source,'execution_ordinal':56,'query_tokens_dtype':str(old['query_tokens'].dtype),
  'query_token_count':old['query_tokens'].shape[0],'reference_count':128,'old_current_template_stored':'template_tokens' in old,
  'weights_only_deserialization':True,'plain_container_tensor_roundtrip':True,'new128_payload_reads':0,'GPU_forwards':0}

def rank_probe():
 scores=torch.tensor([(p%31)-15. for p in range(5413)],dtype=torch.float64)
 labels=[str(p) for p in range(5413)];labels[715]=labels[714];scores[714]=scores[715]=20.
 observed=complete_rank(scores,labels);expected=[];seen=set()
 for p in sorted(range(5413),key=lambda p:(-float(scores[p]),p)):
  if labels[p] not in seen:seen.add(labels[p]);expected.append(p)
 need(observed==expected and observed[0]==714 and 715 not in observed and len(observed)==5412,'SYNTHETIC_RAW_TIES_IDENTITY_DEDUP')
 return {'status':'SYNTHETIC_RAW_FULLRANK_PASS','physical_count':5413,'identity_count':5412,'stable_physical_ties':True,'natural_score_reads':0}

def gpu_runtime():
 need(torch.cuda.is_available(),'CUDA_REQUIRED_FOR_CURRENT_RUNTIME_BRIDGE')
 p=torch.cuda.get_device_properties(0)
 return {'device_name':p.name,'compute_capability':[p.major,p.minor],'total_memory':p.total_memory,
  'cuda_runtime_build':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
  'float32_matmul_precision':torch.get_float32_matmul_precision(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
  'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}

def expansion_bridges():
 v=read(OUT/'bridge/validation.json');need(v['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_PASS','TOKEN_RAW_BRIDGE_REQUIRED')
 path_of(v['payload']);path_of(v['receipt']);need(v['authority']==bind(C.AUTHORITY),'TOKEN_BRIDGE_AUTHORITY')
 rv=read(ROMA_BRIDGE);need(rv['status']=='RC_ORIGINAL7_EVAL128_ROMA_BRIDGE_PASS','ROMA_ENGINEERING_BRIDGE_REQUIRED_BEFORE_NEW128_PIXELS')
 master=read(C.AUTHORITY);need(rv['authority']==bind(C.AUTHORITY),'ROMA_BRIDGE_CURRENT_AUTHORITY')
 need(path_of(rv['payload'])==ROMA_BRIDGE.parent/'payload.pt' and path_of(rv['receipt'])==ROMA_BRIDGE.parent/'receipt.json','ROMA_BRIDGE_PAYLOAD_RECEIPT')
 need(rv['program']==master['source_bindings']['roma_program'] and path_of(rv['program']).is_file(),'ROMA_BRIDGE_PROGRAM')
 profile=bind(ROOT/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json')
 need(rv['source_profile']==profile and profile in master['source_bindings'].values(),'ROMA_BRIDGE_SOURCE_PROFILE')
 expected_token={name:bind(OUT/'bridge'/filename) for name,filename in [('payload','payload.pt'),('receipt','receipt.json'),('validation','validation.json')]}
 need(rv['token_source']==expected_token and rv['query_count']==1 and rv['candidate_occurrence_count']==128,'ROMA_BRIDGE_SAME_TOKEN_FIXTURE')
 need(rv['engineering_original_maps_C4_exact'] is True and rv['all_C4_bits_exact'] is True and rv['target_role_read_count']==rv['model_update_count']==0,'ROMA_BRIDGE_ENGINEERING_CLOSURE')
 return {'token_raw_bridge_validation':bind(OUT/'bridge/validation.json'),'roma_bridge_validation':bind(ROMA_BRIDGE)}

def encode_query(record,processor,encoder,device):
 path=Path(record['query_source_path'])
 need(sha(path)==record['query_source_sha256'],'QUERY_IMAGE_BYTES')
 with Image.open(path) as raw:
  raw_size=[int(raw.width),int(raw.height)];orientation=int(raw.getexif().get(274,1))
  image=ImageOps.exif_transpose(raw).convert('RGB') if record['track']=='new_difficult_train' else raw.convert('RGB')
  inputs=processor.process_images([image]).to(device)
 encoded=encoder(**inputs)[0].float()
 mask=inputs['input_ids'][0]==processor.image_token_id
 image_tokens=encoded[mask].detach().half().cpu().contiguous();template_tokens=encoded[~mask].detach().half().cpu().contiguous()
 temporal,height,width=[int(x) for x in inputs['image_grid_thw'][0].tolist()];merge=int(processor.image_processor.merge_size)
 grid=[height//merge,width//merge]
 need(temporal==1 and image_tokens.shape==(grid[0]*grid[1],128) and template_tokens.ndim==2 and template_tokens.shape[1]==128,'QUERY_IMAGE_TEMPLATE_LAYOUT')
 return image_tokens,template_tokens,grid,{'raw_size_wh':raw_size,'exif_orientation':orientation}

def run_gpu(shard=None):
 bridge=shard is None;stage='token_raw_bridge' if bridge else 'token_raw_shard';authority,pre=checked_authority(stage)
 if bridge:
  meta,old,_,_=old_bridge_expected()
  work=[{'execution_ordinal':56,'query_id':old['query_id'],'track':old['track'],'query_source_path':old['query_source_path'],
   'query_source_sha256':old['query_source_sha256']}];folder=OUT/'bridge';bridge_bindings=None
 else:
  need(shard in range(16),'SHARD0_TO15');bridge_bindings=expansion_bridges()
  entries=worker_manifest()['records'][shard*8:(shard+1)*8];need(len(entries)==8,'EIGHT_QUERIES_PER_SHARD')
  work=[{'execution_ordinal':e['execution_ordinal'],'query_id':e['query_id'],'track':e['track'],
   'query_source_path':e['query_image_path'],'query_source_sha256':e['source_image_sha256']} for e in entries];folder=OUT/f'shard{shard:02d}'
 need(not folder.exists(),'APPEND_ONLY_GPU_OUTPUT_EXISTS')
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 import run_romav2_colnomic_sealed_source_e0_v2 as ranking
 device=torch.device('cuda');gpu=gpu_runtime()
 gallery_payload=torch.load(GALLERY,map_location='cpu',mmap=True,weights_only=False);passages=gallery_payload['passage_emb']
 encoder=ColQwen2_5.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False)
 processor=ColQwen2_5_Processor.from_pretrained(str(MODEL))
 gallery,resolver=gallery_source_and_resolver(pre,processor,passages);labels=gallery.corrected_identities
 records=[];reference_rows=set()
 for source in work:
  image,template,grid,image_metadata=encode_query(source,processor,encoder,device)
  scores=ranking.full_gallery_scores(torch.cat((image,template)),passages,device,batch_size=16)
  need(scores.dtype==torch.float64 and scores.shape==(5413,) and bool(torch.isfinite(scores).all()),'FULL5413_RAW_SCORES')
  order=complete_rank(scores,labels);ranked=ranking.top128_rows(scores,labels);need(order[:128]==ranked,'CANONICAL_V2_TOP128')
  axis=sorted(ranked);reference_rows.update(axis)
  records.append({**source,'processor_input_frame':FRAMES[source['track']],**image_metadata,
   'query_grid_shape':grid,'query_tokens':image,'query_tokens_sha256':tensor_sha(image),
   'template_tokens':template,'template_tokens_sha256':tensor_sha(template),
   'candidate_physical_rows':axis,'candidate_raw_scores':scores[axis].clone().contiguous(),
   'candidate_ranked_physical_rows':ranked,'raw_physical_scores':scores.clone().contiguous(),
   'raw_ranked_physical_rows':order,'raw_ranked_scores':scores[order].clone().contiguous(),
   'target_or_label_read_count':0,'target_insertion_count':0})
  print(json.dumps({'event':'TOKEN_RAW_QUERY_READY','query_id':source['query_id'],'execution_ordinal':source['execution_ordinal'],'candidate_count':128,'raw_identity_count':5412}),flush=True)
 references={}
 for physical in sorted(reference_rows):
  ref=resolver.resolve(physical)
  references[physical]={'physical_row':physical,'source_path':str(gallery.raw_paths[physical]),'grid_shape':list(ref.grid_shape),
   'tokens':ref.tokens.detach().cpu().contiguous(),'tokens_sha256':ref.tokens_sha256,'source_kind':ref.source_kind,
   'source_logical_sha256':ref.source_logical_sha256,'source_image_sha256':sha(gallery.raw_paths[physical])}
  need(tensor_sha(references[physical]['tokens'])==ref.tokens_sha256,'REFERENCE_TOKEN_SOURCE_BITS')
 checks={}
 if bridge:
  _,old,old_refs,expected_source=old_bridge_expected();row=records[0]
  missing_old_reference_rows=sorted(set(old['candidate_physical_rows'])-set(references))
  checks={'query_image_tokens_exact':tensor_sha(row['query_tokens'])==tensor_sha(old['query_tokens']),
   'query_grid_exact':list(row['query_grid_shape'])==list(old['query_grid_shape']),
   'candidate_axis_exact':row['candidate_physical_rows']==old['candidate_physical_rows'],
   'candidate_ranked_order_exact':row['candidate_ranked_physical_rows']==old['candidate_ranked_physical_rows'],
   'candidate_raw_scores_exact':tensor_sha(row['candidate_raw_scores'])==tensor_sha(old['candidate_raw_scores']),
   'all128_reference_tokens_exact':all(p in references and tensor_sha(references[p]['tokens'])==tensor_sha(old_refs[p]['tokens']) for p in old['candidate_physical_rows']),
   'all128_reference_grids_exact':all(p in references and references[p]['grid_shape']==list(old_refs[p]['grid_shape']) for p in old['candidate_physical_rows'])}
  template_available='template_tokens' in old
  if template_available:checks['query_template_tokens_exact']=tensor_sha(row['template_tokens'])==tensor_sha(old['template_tokens'])
  template_check='BIT_EXACT' if template_available else 'NOT_STORED_IN_QUALIFIED_OLD_RUNTIME;NO_TEMPLATE_PARITY_CLAIM'
 else:expected_source=None;template_check=None;missing_old_reference_rows=None
 ready=not bridge or all(checks.values())
 status=('RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_READY' if ready else 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_MISMATCH') if bridge else 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_SHARD_READY'
 value={'schema':'rc_original7_eval128_token_raw_payload_v1','status':status,'stage':'bridge' if bridge else 'shard','shard':shard,'shard_count':16,
  'records':records,'references':references,'bindings':{'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),
   'preflight':authority['source_bindings']['token_raw_preflight'],'gallery':gallery_bindings(gallery),
   'old_bridge_expected_payload':expected_source,'engineering_bridges':bridge_bindings},
  'contract':CONTRACT,'runtime_libraries':runtime_libraries(),'GPU_runtime':gpu,'model_config':{'encoder_class':type(encoder).__module__+'.'+type(encoder).__name__,
   'attn_implementation':str(getattr(encoder.config,'_attn_implementation',None)),'model_dtype':str(next(encoder.parameters()).dtype)},
  'bridge_checks':checks,'old_template_check':template_check,'bridge_missing_old_reference_rows':missing_old_reference_rows,
  'access':{'query_target_reads':0,'curator_reads':0,'formal392_P_RoMa_result_reads':0,'new_query_encoder_forwards':len(records) if not bridge else 0,
   'old_TRAIN_bridge_encoder_forwards':1 if bridge else 0,'reference_encoder_forwards':0,'new_training_updates':0,'forbidden_read_attempts':len(BLOCKED)}}
 need(not BLOCKED,'FORBIDDEN_SOURCE_ATTEMPT');plain_only(value);folder.mkdir(parents=True,exist_ok=False)
 with (folder/'payload.pt').open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 (folder/'payload.pt').chmod(0o444)
 receipt={'status':status,'stage':value['stage'],'shard':shard,'query_count':len(records),'reference_count':len(references),
  'payload':bind(folder/'payload.pt'),'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),'contract':CONTRACT,
  'bridge_checks':checks,'old_template_check':template_check,'bridge_missing_old_reference_rows':missing_old_reference_rows,'runtime_libraries':runtime_libraries(),'GPU_runtime':gpu,'access':value['access']}
 C.write_json(folder/'receipt.json',receipt)
 print(json.dumps({'status':status,'receipt':bind(folder/'receipt.json'),'bridge_checks':checks}),flush=True)
 need(ready,'BRIDGE_NOT_BIT_EXACT_EXPANSION_CLOSED')
 run_validation_child(shard)


def validate_cpu(shard=None):
 bridge=shard is None;stage='token_raw_bridge' if bridge else 'token_raw_shard';authority,pre=checked_authority(stage)
 folder=OUT/('bridge' if bridge else f'shard{shard:02d}');need(not(folder/'validation.json').exists(),'APPEND_ONLY_VALIDATION')
 receipt=read(folder/'receipt.json');expected_status='RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_READY' if bridge else 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_SHARD_READY'
 need(receipt['status']==expected_status and receipt['authority']==bind(C.AUTHORITY),'READY_BOUND_TOKEN_PAYLOAD')
 payload=torch.load(path_of(receipt['payload']),map_location='cpu',mmap=True,weights_only=True);plain_only(payload)
 need(payload['status']==expected_status and payload['contract']==CONTRACT and payload['bindings']['authority']==bind(C.AUTHORITY),'PAYLOAD_SCHEMA_AUTHORITY')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities
 need(payload['bindings']['gallery']==gallery_bindings(gallery),'GALLERY_IDENTITY_AND_PHYSICAL_ORDER')
 if bridge:
  _,old,oldrefs,_=old_bridge_expected();expected=[old]
 else:expected=worker_manifest()['records'][shard*8:(shard+1)*8];expansion_bridges()
 need(len(payload['records'])==len(expected)==(1 if bridge else 8),'QUERY_COUNT')
 references=payload['references'];seen=set()
 for row,e in zip(payload['records'],expected):
  need(row['query_id']==e['query_id'] and row['execution_ordinal']==e['execution_ordinal'] and row['track']==e['track'],'EXACT_QUERY_SEQUENCE')
  expected_image=e['query_source_path'] if bridge else e['query_image_path'];expected_sha=e['query_source_sha256'] if bridge else e['source_image_sha256']
  need(row['query_source_path']==expected_image and row['query_source_sha256']==expected_sha==sha(Path(expected_image)),'OPAQUE_IMAGE_SOURCE')
  need(row['processor_input_frame']==FRAMES[row['track']],'PREPROCESSING_FRAME')
  q,t=row['query_tokens'],row['template_tokens'];grid=row['query_grid_shape']
  need(q.dtype==t.dtype==torch.float16 and q.shape==(math.prod(grid),128) and t.ndim==2 and t.shape[1]==128,'TOKEN_GRID_LAYOUT')
  need(tensor_sha(q)==row['query_tokens_sha256'] and tensor_sha(t)==row['template_tokens_sha256'],'QUERY_TEMPLATE_TOKEN_HASHES')
  scores=row['raw_physical_scores'];need(scores.dtype==torch.float64 and scores.shape==(5413,) and bool(torch.isfinite(scores).all()),'RAW_SCORE_FINITE_FULL_PHYSICAL_AXIS')
  ranked=[];identities=set()
  for p in sorted(range(5413),key=lambda p:(-float(scores[p]),p)):
   if labels[p] in identities:continue
   identities.add(labels[p]);ranked.append(p)
  need(len(ranked)==5412 and ranked==row['raw_ranked_physical_rows'],'INDEPENDENT_COMPLETE_IDENTITY_RANK')
  need(tensor_sha(scores[ranked])==tensor_sha(row['raw_ranked_scores']),'FULL_RANKED_SCORE_BITS')
  top=ranked[:128];axis=sorted(top)
  need(top==row['candidate_ranked_physical_rows'] and axis==row['candidate_physical_rows'] and tensor_sha(scores[axis])==tensor_sha(row['candidate_raw_scores']),'INDEPENDENT_NATURAL_C128')
  if bridge:
   need(tensor_sha(q)==tensor_sha(old['query_tokens']) and grid==list(old['query_grid_shape']),'OLD_TRAIN_QUERY_TOKEN_BIT_REPLAY')
   need(axis==old['candidate_physical_rows'] and top==old['candidate_ranked_physical_rows'] and tensor_sha(row['candidate_raw_scores'])==tensor_sha(old['candidate_raw_scores']),'OLD_TRAIN_RAW_C128_BIT_REPLAY')
   if 'template_tokens' in old:need(tensor_sha(t)==tensor_sha(old['template_tokens']),'OLD_TRAIN_TEMPLATE_BITS')
  for p in axis:
   ref=references[p];need(ref['physical_row']==p and ref['source_path']==str(gallery.raw_paths[p]),'REFERENCE_PHYSICAL_SOURCE')
   need(ref['tokens'].dtype==torch.float16 and ref['tokens'].shape==(math.prod(ref['grid_shape']),128) and tensor_sha(ref['tokens'])==ref['tokens_sha256'],'REFERENCE_GRID_TOKEN_BYTES')
   need(sha(Path(ref['source_path']))==ref['source_image_sha256'],'REFERENCE_IMAGE_BYTES')
   if bridge:need(tensor_sha(ref['tokens'])==tensor_sha(oldrefs[p]['tokens']) and ref['grid_shape']==list(oldrefs[p]['grid_shape']),'OLD_TRAIN_REFERENCE_TOKEN_GRID_BITS')
   seen.add(p)
 need(set(references)==seen and not BLOCKED,'EXACT_REFERENCE_UNION_NO_FORBIDDEN_READ')
 if bridge:need(payload['bridge_checks'] and all(v is True for v in payload['bridge_checks'].values()),'ALL_BIT_EXACT_BRIDGE_CHECKS')
 status='RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_PASS' if bridge else 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_CPU_REPLAY_PASS'
 v={'status':status,'query_count':len(payload['records']),'reference_count':len(references),'payload':bind(folder/'payload.pt'),'receipt':bind(folder/'receipt.json'),
  'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),'contract':CONTRACT,
  'checks':{'plain_tensor_payload':True,'full5412_identity_rank_rebuilt':True,'natural_C128_axis_and_scores_exact':True,
   'query_reference_hash_grid_bindings':True,'no_query_targets_or_curator':True},
  'old_TRAIN_bit_exact_bridge_checked':bridge,'fresh_CPU_process':True,'new_model_forwards':0,'new_training_updates':0,'old_template_check':payload['old_template_check']}
 C.write_json(folder/'validation.json',v);print(json.dumps({'status':status,'validation':bind(folder/'validation.json')}),flush=True)

def run_validation_child(shard):
 nonce=uuid.uuid4().hex;env=dict(os.environ,E128_TOKEN_RAW_VALIDATOR_NONCE=nonce,E128_TOKEN_RAW_VALIDATOR_PARENT_PID=str(os.getpid()))
 cmd=[sys.executable,str(PROGRAM),'--phase','validate-existing','--validator-nonce',nonce]
 if shard is not None:cmd+=['--shard',str(shard)]
 subprocess.run(cmd,check=True,env=env)

def main():
 global PHASE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('preflight','bridge-existing','shard','validate-existing'))
 p.add_argument('--shard',type=int);p.add_argument('--validator-nonce');args=p.parse_args()
 PHASE='validate-shard' if args.phase=='validate-existing' and args.shard is not None else args.phase
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(17);sys.addaudithook(audit)
 if args.phase=='preflight':need(args.shard is None,'NO_SHARD_IN_PREFLIGHT');preflight()
 elif args.phase=='bridge-existing':need(args.shard is None,'ONLY_TRAIN56_BRIDGE');run_gpu()
 elif args.phase=='shard':need(args.shard in range(16),'EXPANSION_SHARD_REQUIRED');run_gpu(args.shard)
 else:
  need(args.validator_nonce==os.environ.get('E128_TOKEN_RAW_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('E128_TOKEN_RAW_VALIDATOR_PARENT_PID'),'EXPLICIT_FRESH_CPU_VALIDATOR_REQUIRED')
  validate_cpu(args.shard)
if __name__=='__main__':main()
