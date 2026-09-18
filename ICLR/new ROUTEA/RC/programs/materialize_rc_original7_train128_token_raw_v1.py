#!/usr/bin/env python3
"""TRAIN128 input materialization using the qualified original TOKEN/RAW math.

No model fitting or evaluation outcome inputs. The first 32 anonymous records
must replay every qualified original FULL TRAIN token/C128/reference bit.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, subprocess, sys, uuid
from importlib import metadata
from pathlib import Path
import torch
from PIL import Image,ImageOps
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'programs')]
import rc_original7_train128_execution_common_v1 as C
PROGRAM=Path(__file__).resolve()
PLAN=ROOT/'plan/RC_ORIGINAL7_TRAIN128_INPUTS_V1_20260910.md'
MANIFEST_ROOT=ROOT/'results/rc_original7_train128_manifest_v1'
WORKER=MANIFEST_ROOT/'worker_manifest.json'
METADATA_VALID=MANIFEST_ROOT/'independent_metadata_validation.json'
PARITY_INDEX=MANIFEST_ROOT/'legacy_full32_parity_index.json'
METADATA_PINS={'worker_manifest.json':'50d894c9643ca1ef200c18ba79cf700b598be15c5ea322010853c201fc0f22f7','independent_metadata_validation.json':'03176db194d4b5f9d68cc226a1ce48b019675e99b037205bae71552202fa907d','legacy_full32_parity_index.json':'31c337c676dc17d38fb85cdcb902b3aeddfe104aebbb6840baeaaba5837c6b03'}
OUT=ROOT/'results/rc_original7_train128_token_raw_v1'
PREFLIGHT=ROOT/'results/rc_original7_train128_token_raw_v1_preflight'
QUALIFIED_PROGRAM=ROOT/'programs/materialize_rc_original7_eval128_token_raw_v2_compat.py'
QUALIFIED_PROGRAM_SHA='708318dee09a387a2d2fb22680a184ed43b5653cc335ab56311fc8364707c97c'
QUALIFIED_PREFLIGHT=ROOT/'results/rc_original7_eval128_token_raw_v2_compat_preflight/1ce5aaffabf17d5816a31181517535f0305820bcfac553fea586c82999393ce7.json'
QUALIFIED_PREFLIGHT_SHA='630e8b88044b1ab972cc05eba1eccee12431ccda6f662b691c1999c6bff673df'
MODEL=WORKSPACE/'models/downloaded_models/colnomic-embed-multimodal-7b'
GALLERY=WORKSPACE/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
GALLERY_SHA='11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc'
SPATIAL=ROOT/'cache/conditional_colnomic_p_spatial_v1'
LAUNCH_SHARDS=ROOT/'slurm/rc_original7_train128_token_raw_shards_v1_30m.sbatch'
FRAMES={'outcome':'DECODED_RAW_BEFORE_EXIF','difficult':'DECODED_RAW_BEFORE_EXIF','new_difficult_train':'EXIF_ORIENTED_BEFORE_RESIZE'}
PHASE=None;BLOCKED=[]
CONTRACT={'query_count':128,'shards':16,'queries_per_shard':8,'encoder':'ColQwen2_5.from_pretrained(frozen_ColNomic7B,torch_dtype=bfloat16).eval.requires_grad_false',
 'query_frame':FRAMES,'query_image_and_template_tokens':'FP16_from_encoded.float_image_mask_preserving_original_order',
 'RAW':'frozen_v2.full_gallery_scores;FP32_sum_MaxSim;batch16;highest_matmul_precision;full5413',
 'identity_reduction':'stable_descending_physical_score_then_corrected_identity_dedup_full5412',
 'candidate_axis':'top128_corrected_identity_representatives_sorted_physical',
 'reference_resolver':'frozen_HybridSpatialReferenceResolver_gallery_only_validated4976_and_same_CPU_recovery',
 'old_FULL_TRAIN32_parity':'first32 query tokens, grid, C128 axis/order/scores, reference tokens/grids all binary exact',
 'original_TRAIN56_engineering_bridges':'reuse qualified TOKEN v1 and RoMa v2; no rerun',
 'target_insertion':False,'drop_target_absent_queries':False,'query_labels_or_private_curator_reads':0,
 'new_training_updates':0,'J_T_LP_required':False}

def need(v,m):
 if not bool(v):raise RuntimeError(m)

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

def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();bad=None
 if any(x in s for x in ('curator_roles','/role_shards/','/target_join/','/rc_opened_eval_strict_','d1_mi','d1-mi','grozi','gisc_prerecall_universe')):bad='PRIVATE_ROLE_OR_PROTECTED_SOURCE_FORBIDDEN'
 if any(x in s for x in ('/results/rc_original7_eval128_inputs_','/results/rc_original7_eval128_full_evidence_','/results/rc_eval128_0337_')):bad='EVAL_RESULT_FORBIDDEN'
 if '/worker_assets/tokens/' in s:bad='LEGACY_REDACTED_TOKENS_NEVER_SUBSTITUTED'
 if '/worker_assets/images/' in s:
  if not s.startswith(str(MANIFEST_ROOT/'worker_assets/images').lower()+'/'):bad='ONLY_TRAIN128_ANONYMOUS_IMAGES_ALLOWED'
  elif PHASE not in ('shard','validate-shard'):bad='NATURAL_IMAGES_BEFORE_SLURM_PHASE'
 if '/conditional_colnomic_p_spatial_v1/query/' in s:bad='UNRELATED_QUERY_SPATIAL_CACHE_FORBIDDEN'
 if bad:BLOCKED.append(str(p));raise RuntimeError(bad)

def worker_manifest():
 for name,digest in METADATA_PINS.items():need(sha(MANIFEST_ROOT/name)==digest,'FROZEN_METADATA_PIN:'+name)
 d=read(WORKER);v=read(METADATA_VALID)
 need(d['status']=='TRAIN128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and d['query_count']==128 and d['role']=='TRAIN','FROZEN_TRAIN128_SCOPE')
 expected={'execution_ordinal','grid_hw','image_tokens_sha256','query_id','query_image_path','redacted_token_artifact',
  'redacted_token_artifact_declared_sha256','source_image_sha256','template_tokens_sha256','track'}
 need([x['execution_ordinal'] for x in d['records']]==list(range(128)) and len({x['query_id'] for x in d['records']})==128,'OPAQUE128_AXIS')
 need(v['status']=='TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS' and v['counts']['training_images']==128 and v['fresh_explicit_subprocess'],'INDEPENDENT_METADATA_VALIDATION')
 need(v['worker_manifest']==bind(WORKER) and v['legacy_full32_parity_index']==d['legacy_full32_parity_index']==bind(PARITY_INDEX),'METADATA_VALIDATION_BINDS_CURRENT_WORKER_AND_PARITY')
 need(v['original_FULL32_and_PAIR64_images_and_internal_order_preserved'] and v['old_and_new_EVAL_identities_groups_images_excluded'] and v['formal392_ID_and_image_SHA_negative_exclusions'],'FIXED_TRAIN_SELECTION_BOUNDARY')
 for x in d['records']:
  need(set(x)==expected and x['query_id'].startswith('T128-') and x['track'] in FRAMES,'TRAIN_WORKER_ROW_SCHEMA')
  p=Path(x['query_image_path']);need(p.is_absolute() and p.parent==MANIFEST_ROOT/'worker_assets/images' and p.is_symlink(),'ANONYMOUS_TRAIN_IMAGE_ALIAS')
 return d

def parity_index():
 d=read(PARITY_INDEX);need(d['status']=='ORIGINAL7_TRAIN128_LEGACY_FULL32_PARITY_INDEX_METADATA_ONLY' and d['query_count']==32,'FROZEN_PARITY_INDEX_SCOPE');rows=d['records'];worker=worker_manifest()['records']
 need(len(rows)==32 and [x['execution_ordinal'] for x in rows]==list(range(32)),'EXACT_FIRST32_FULL_TRAIN_PARITY')
 for x,w in zip(rows,worker[:32]):
  need(x['query_id']==w['query_id'],'PARITY_ANONYMOUS_ID')
  m=read(path_of(x['metadata']))
  need(m['kind']=='FULL' and m['role']=='TRAIN' and m['execution_ordinal']==x['original_execution_ordinal'],'ONLY_ORIGINAL_FULL_TRAIN')
  need(m['query_source_image_sha256']==w['source_image_sha256'],'OLD_TRAIN_IMAGE_BINDING')
  for kind in ('tokens','maps'):
   original={**m['source'][kind],'path':str(C.bound_path(m['source'][kind]))};selected={**x[kind],'path':str(C.bound_path(x[kind]))}
   need(original==selected,'ORIGINAL_TOKEN_AND_MAP_SOURCE_BINDINGS:'+kind)
 return {x['query_id']:x for x in rows}

def unchanged_math_probe():
 need(sha(QUALIFIED_PROGRAM)==QUALIFIED_PROGRAM_SHA,'QUALIFIED_TOKEN_RAW_PROGRAM_PIN')
 old=QUALIFIED_PROGRAM.read_text();new=PROGRAM.read_text()
 def bodies(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 a,b=bodies(old),bodies(new)
 names=('tensor_sha','runtime_libraries','gallery_source_and_resolver','gallery_bindings','complete_rank','plain_only','rank_probe','gpu_runtime','encode_query')
 need(all(a[n]==b[n] for n in names),'QUALIFIED_MATH_BODY_CHANGED')
 return {'status':'QUALIFIED_TOKEN_RAW_MATH_AST_EXACT','functions':list(names),'qualified_program':bind(QUALIFIED_PROGRAM)}

def public_sources():
 need(sha(QUALIFIED_PREFLIGHT)==QUALIFIED_PREFLIGHT_SHA,'QUALIFIED_PREFLIGHT_PIN')
 old=read(QUALIFIED_PREFLIGHT);need(old['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_PREFLIGHT_PASS','QUALIFIED_PREFLIGHT_STATUS')
 changed={'token_raw_program','token_raw_plan','token_raw_worker_manifest','token_raw_metadata_validation','token_raw_common_program','token_raw_bridge_launcher','token_raw_shard_launcher'}
 sources={k:v for k,v in old['public_source_bindings'].items() if k not in changed and not k.startswith('token_raw_cpu_probe_')}
 files={'token_raw_program':PROGRAM,'token_raw_plan':PLAN,'token_raw_worker_manifest':WORKER,'token_raw_metadata_validation':METADATA_VALID,
  'token_raw_common_program':Path(C.__file__),'token_raw_shard_launcher':LAUNCH_SHARDS,'token_raw_qualified_program':QUALIFIED_PROGRAM,
  'token_raw_qualified_preflight':QUALIFIED_PREFLIGHT,'token_raw_legacy_full32_parity_index':PARITY_INDEX}
 for k,p in files.items():sources[k]=bind(p)
 for q,x in parity_index().items():
  for typ in ('metadata','tokens','maps'):
   v=x[typ];sources['token_raw_old_FULL_'+str(x['original_execution_ordinal'])+'_'+typ]={'path':str(path_of(v).resolve()),'sha256':v['sha256']}
 # Verify source-file bytes only. This does not deserialize any natural token/image payload.
 for k,v in sources.items():path_of(v)
 need(sources['token_raw_gallery_cache']['sha256']==GALLERY_SHA,'ORIGINAL_GALLERY_PIN')
 need(old['runtime_libraries']==runtime_libraries(),'QUALIFIED_RUNTIME_VERSION_OR_FP_SETTINGS_DRIFT')
 return sources,old['legacy_reference_union']

def preflight():
 worker_manifest();index=parity_index();sources,union=public_sources();math_check=unchanged_math_probe()
 value={'status':'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_PREFLIGHT_PASS','public_source_bindings':sources,'legacy_reference_union':union,
  'contract':CONTRACT,'runtime_libraries':runtime_libraries(),'unchanged_math':math_check,'synthetic_RAW_rank_probe':rank_probe(),
  'old_FULL_TRAIN_parity_index_count':len(index),'natural_image_reads':0,'natural_tensor_deserializations':0,'GPU_forwards':0,'new_training_updates':0}
 p=PREFLIGHT/(hashlib.sha256(encode(sources)).hexdigest()+'.json');C.write_json(p,value)
 print(json.dumps({'status':value['status'],'preflight':bind(p),'natural_image_reads':0,'natural_tensor_deserializations':0,'GPU_forwards':0}),flush=True)

def checked_authority():
 a=C.require_authority('token_raw_shard',PROGRAM);p=C.bound_path(a['source_bindings']['token_raw_preflight']);pre=read(p)
 need(pre['status']=='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_PREFLIGHT_PASS' and pre['contract']==CONTRACT,'TRAIN_PREFLIGHT_CONTRACT')
 for k,v in pre['public_source_bindings'].items():need(a['source_bindings'].get(k)==v,'PUBLIC_SOURCE_MISSING_FROM_MASTER:'+k)
 need(pre['runtime_libraries']==runtime_libraries(),'CURRENT_LIBRARY_OR_FP_SETTINGS_DRIFT')
 return a,pre

def engineering_bridges():
 d=C.require_reused_engineering_bridges();need(set(d)=={'token_raw','roma'},'EXACT_REUSED_ENGINEERING_BRIDGES')
 for kind in d:
  need(set(d[kind])=={'payload','receipt','validation'},'COMPLETE_REUSED_BRIDGE_BINDINGS')
  for v in d[kind].values():path_of(v)
 return d

def old_expected(entry):
 meta=read(path_of(entry['metadata']));p=path_of(entry['tokens'])
 payload=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
 old=payload['records'][int(entry['tokens']['record_index'])]
 need(old['role']=='TRAIN' and old['execution_ordinal']==entry['original_execution_ordinal']==meta['execution_ordinal'],'ONLY_OLD_TRAIN_RECORD')
 need(tensor_sha(old['query_tokens'])==meta['q_tokens_sha256'] and old['candidate_physical_rows']==meta['axis'],'ORIGINAL_TRAIN_TOKEN_AXIS_BINDING')
 return old,payload['references']

def original_parity(row,references,entry):
 old,oldrefs=old_expected(entry)
 missing=sorted(set(old['candidate_physical_rows'])-set(references))
 checks={'query_image_source_exact':row['query_source_sha256']==old['query_source_sha256'],
  'query_track_exact':row['track']==old['track'],'query_image_tokens_exact':tensor_sha(row['query_tokens'])==tensor_sha(old['query_tokens']),
  'query_grid_exact':row['query_grid_shape']==list(old['query_grid_shape']),
  'candidate_axis_exact':row['candidate_physical_rows']==old['candidate_physical_rows'],
  'candidate_ranked_order_exact':row['candidate_ranked_physical_rows']==old['candidate_ranked_physical_rows'],
  'candidate_raw_scores_exact':tensor_sha(row['candidate_raw_scores'])==tensor_sha(old['candidate_raw_scores']),
  'all128_reference_tokens_exact':all(p in references and tensor_sha(references[p]['tokens'])==tensor_sha(oldrefs[p]['tokens']) for p in old['candidate_physical_rows']),
  'all128_reference_grids_exact':all(p in references and references[p]['grid_shape']==list(oldrefs[p]['grid_shape']) for p in old['candidate_physical_rows'])}
 if 'template_tokens' in old:checks['query_template_tokens_exact']=tensor_sha(row['template_tokens'])==tensor_sha(old['template_tokens'])
 return {'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'original_execution_ordinal':entry['original_execution_ordinal'],
  'source':entry,'checks':checks,'all_bits_exact':all(checks.values()),'missing_old_reference_rows':missing,
  'template_parity':'BIT_EXACT' if 'template_tokens' in old else 'NOT_STORED_IN_QUALIFIED_OLD_RUNTIME;NO_TEMPLATE_PARITY_CLAIM'}

def run_gpu(shard):
 authority,pre=checked_authority();need(shard in range(16),'SHARD0_TO15');bridges=engineering_bridges()
 entries=worker_manifest()['records'][shard*8:(shard+1)*8];index=parity_index();need(len(entries)==8,'EIGHT_QUERIES_PER_SHARD')
 folder=OUT/f'shard{shard:02d}';need(not folder.exists(),'APPEND_ONLY_GPU_OUTPUT_EXISTS')
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 import run_romav2_colnomic_sealed_source_e0_v2 as ranking
 device=torch.device('cuda');gpu=gpu_runtime()
 gallery_payload=torch.load(GALLERY,map_location='cpu',mmap=True,weights_only=False);passages=gallery_payload['passage_emb']
 encoder=ColQwen2_5.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False)
 processor=ColQwen2_5_Processor.from_pretrained(str(MODEL))
 gallery,resolver=gallery_source_and_resolver(pre,processor,passages);labels=gallery.corrected_identities
 records=[];reference_rows=set()
 for entry in entries:
  source={'execution_ordinal':entry['execution_ordinal'],'query_id':entry['query_id'],'track':entry['track'],
   'query_source_path':entry['query_image_path'],'query_source_sha256':entry['source_image_sha256']}
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
 parity=[original_parity(row,references,index[row['query_id']]) for row in records if row['query_id'] in index]
 need(len(parity)==(8 if shard<4 else 0),'FIRST32_PARITY_COVERAGE')
 ready=all(x['all_bits_exact'] for x in parity)
 status='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_SHARD_READY' if ready else 'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_OLD32_MISMATCH'
 value={'schema':'rc_original7_train128_token_raw_payload_v1','status':status,'stage':'shard','shard':shard,'shard_count':16,
  'records':records,'references':references,'bindings':{'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),
   'preflight':authority['source_bindings']['token_raw_preflight'],'gallery':gallery_bindings(gallery),
   'legacy_full32_parity_index':bind(PARITY_INDEX),'engineering_bridges':bridges},
  'contract':CONTRACT,'runtime_libraries':runtime_libraries(),'GPU_runtime':gpu,'model_config':{'encoder_class':type(encoder).__module__+'.'+type(encoder).__name__,
   'attn_implementation':str(getattr(encoder.config,'_attn_implementation',None)),'model_dtype':str(next(encoder.parameters()).dtype)},
  'legacy_full32_parity':parity,'old_template_check':'PER_RECORD_IN_LEGACY_FULL32_PARITY',
  'access':{'query_target_reads':0,'curator_reads':0,'formal392_P_RoMa_result_reads':0,'new_query_encoder_forwards':len(records),
   'old_TRAIN_bridge_encoder_forwards':0,'reference_encoder_forwards':0,'new_training_updates':0,'forbidden_read_attempts':len(BLOCKED)}}
 need(not BLOCKED,'FORBIDDEN_SOURCE_ATTEMPT');plain_only(value);folder.mkdir(parents=True,exist_ok=False)
 with (folder/'payload.pt').open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
 (folder/'payload.pt').chmod(0o444)
 receipt={'status':status,'stage':'shard','shard':shard,'query_count':len(records),'reference_count':len(references),
  'payload':bind(folder/'payload.pt'),'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),'contract':CONTRACT,
  'legacy_full32_parity':parity,'runtime_libraries':runtime_libraries(),'GPU_runtime':gpu,'access':value['access']}
 C.write_json(folder/'receipt.json',receipt)
 print(json.dumps({'status':status,'receipt':bind(folder/'receipt.json'),'old_FULL_TRAIN_parity_count':len(parity)}),flush=True)
 need(ready,'OLD_TRAIN32_NOT_BIT_EXACT_NO_DATA_EXPANSION_CLAIM')
 run_validation_child(shard)

def validate_cpu(shard):
 authority,pre=checked_authority();need(shard in range(16),'SHARD0_TO15')
 folder=OUT/f'shard{shard:02d}';need(not(folder/'validation.json').exists(),'APPEND_ONLY_VALIDATION')
 receipt=read(folder/'receipt.json');expected_status='RC_ORIGINAL7_TRAIN128_TOKEN_RAW_SHARD_READY'
 need(receipt['status']==expected_status and receipt['authority']==bind(C.AUTHORITY),'READY_BOUND_TRAIN_TOKEN_PAYLOAD')
 payload=torch.load(path_of(receipt['payload']),map_location='cpu',mmap=True,weights_only=True);plain_only(payload)
 need(payload['status']==expected_status and payload['contract']==CONTRACT and payload['bindings']['authority']==bind(C.AUTHORITY),'PAYLOAD_SCHEMA_AUTHORITY')
 need(payload['bindings']['worker_manifest']==bind(WORKER) and payload['bindings']['legacy_full32_parity_index']==bind(PARITY_INDEX),'FROZEN_INPUT_METADATA')
 need(payload['bindings']['engineering_bridges']==engineering_bridges(),'REUSED_ENGINEERING_BRIDGES')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities
 need(payload['bindings']['gallery']==gallery_bindings(gallery),'GALLERY_IDENTITY_AND_PHYSICAL_ORDER')
 expected=worker_manifest()['records'][shard*8:(shard+1)*8];index=parity_index()
 need(len(payload['records'])==len(expected)==8,'QUERY_COUNT')
 references=payload['references'];seen=set();parity=[]
 for row,e in zip(payload['records'],expected):
  need(row['query_id']==e['query_id'] and row['execution_ordinal']==e['execution_ordinal'] and row['track']==e['track'],'EXACT_QUERY_SEQUENCE')
  need(row['query_source_path']==e['query_image_path'] and row['query_source_sha256']==e['source_image_sha256']==sha(Path(e['query_image_path'])),'OPAQUE_IMAGE_SOURCE')
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
  for p in axis:
   ref=references[p];need(ref['physical_row']==p and ref['source_path']==str(gallery.raw_paths[p]),'REFERENCE_PHYSICAL_SOURCE')
   need(ref['tokens'].dtype==torch.float16 and ref['tokens'].shape==(math.prod(ref['grid_shape']),128) and tensor_sha(ref['tokens'])==ref['tokens_sha256'],'REFERENCE_GRID_TOKEN_BYTES')
   need(sha(Path(ref['source_path']))==ref['source_image_sha256'],'REFERENCE_IMAGE_BYTES');seen.add(p)
  if row['query_id'] in index:
   one=original_parity(row,references,index[row['query_id']]);need(one['all_bits_exact'],'OLD_TRAIN_QUERY_C128_REFERENCE_BIT_REPLAY');parity.append(one)
 need(set(references)==seen and not BLOCKED,'EXACT_REFERENCE_UNION_NO_FORBIDDEN_READ')
 need(len(parity)==(8 if shard<4 else 0) and parity==payload['legacy_full32_parity']==receipt['legacy_full32_parity'],'FULL_TRAIN32_PARITY_RECEIPT_REPLAY')
 v={'status':'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_CPU_REPLAY_PASS','query_count':len(payload['records']),'reference_count':len(references),
  'payload':bind(folder/'payload.pt'),'receipt':bind(folder/'receipt.json'),'authority':bind(C.AUTHORITY),'worker_manifest':bind(WORKER),'contract':CONTRACT,
  'legacy_full32_parity_count':len(parity),'all_original_TRAIN_bits_exact':all(x['all_bits_exact'] for x in parity),
  'checks':{'plain_tensor_payload':True,'full5412_identity_rank_rebuilt':True,'natural_C128_axis_and_scores_exact':True,
   'query_reference_hash_grid_bindings':True,'old_FULL_TRAIN_bit_replay':True,'no_query_targets_or_curator':True},
  'fresh_CPU_process':True,'new_model_forwards':0,'new_training_updates':0}
 C.write_json(folder/'validation.json',v);print(json.dumps({'status':v['status'],'validation':bind(folder/'validation.json')}),flush=True)

def run_validation_child(shard):
 nonce=uuid.uuid4().hex;env=dict(os.environ,T128_TOKEN_RAW_VALIDATOR_NONCE=nonce,T128_TOKEN_RAW_VALIDATOR_PARENT_PID=str(os.getpid()))
 subprocess.run([sys.executable,str(PROGRAM),'--phase','validate-existing','--shard',str(shard),'--validator-nonce',nonce],check=True,env=env)

def main():
 global PHASE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('preflight','shard','validate-existing'))
 p.add_argument('--shard',type=int);p.add_argument('--validator-nonce');args=p.parse_args()
 PHASE='validate-shard' if args.phase=='validate-existing' else args.phase
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(17);sys.addaudithook(audit)
 if args.phase=='preflight':need(args.shard is None,'NO_SHARD_IN_PREFLIGHT');preflight()
 elif args.phase=='shard':need(args.shard in range(16),'TRAIN_SHARD_REQUIRED');run_gpu(args.shard)
 else:
  need(args.validator_nonce==os.environ.get('T128_TOKEN_RAW_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('T128_TOKEN_RAW_VALIDATOR_PARENT_PID'),'EXPLICIT_FRESH_CPU_VALIDATOR_REQUIRED')
  validate_cpu(args.shard)
if __name__=='__main__':main()
