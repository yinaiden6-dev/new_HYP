#!/usr/bin/env python3
"""Generalized missing-input adapter; original numerical loop bodies are reused verbatim."""
import argparse,ast,hashlib,importlib.util,json,math,os,subprocess,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
OUT=ROOT/'results/rc_new_hyp593_oof5_v1';META=OUT/'metadata';AUTH=ROOT/'registry/rc_new_hyp593_oof5_authority_v1_20260911.json'
RAW_SOURCE=ROOT/'programs/materialize_rc_original7_train128_token_raw_v1.py'
ROMA_SOURCE=ROOT/'programs/materialize_rc_original7_train128_roma_v1.py'
PROFILE=ROOT/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
RAW_PREFLIGHT=ROOT/'results/rc_original7_eval128_token_raw_v2_compat_preflight/1ce5aaffabf17d5816a31181517535f0305820bcfac553fea586c82999393ce7.json'
DEADLINE=datetime(2026,9,11,16,tzinfo=timezone.utc)
def need(v,m):
 if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def checked(b):
 p=Path(b['path']);need(sha(p)==b['sha256'],'SOURCE_DRIFT:'+str(p));return p
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def loadmodule(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def guard(stage):
 need(datetime.now(timezone.utc)<DEADLINE,'USER_DEADLINE');a=read(AUTH)
 need(a['status']=='H593_NEW_GROUPED_EXPERIMENT_AUTHORIZED','AUTHORITY')
 for b in a['adapter_sources'].values():checked(b)
 need(a['metadata_validation']==bind(META/'validation.json'),'METADATA_BINDING')
 need(read(META/'validation.json')['status']=='H593_METADATA_GROUP_SEPARATION_PASS','METADATA_VALIDATION')
 if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
 return a

def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 s=os.path.abspath(os.fsdecode(args[0])).lower()
 need(not any(x in s for x in ('curator_roles','target_join','d1-mi','d1_mi','grozi','rc_opened_','gisc_prerecall_universe','/reports/')),'LABEL_OR_PROTECTED_READ_DENIED')

def segment(path,function,start,end):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
 lines=path.read_text().splitlines(True);body=''.join(lines[node.lineno-1:node.end_lineno]);a=body.index(start);b=body.index(end,a)
 import textwrap
 return textwrap.dedent(body[a:b])
def compile_raw(m):
 body=segment(RAW_SOURCE,'run_gpu',' from colpali_engine.models',' parity=[original_parity')
 ns=dict(vars(m));exec('def compute(entries,pre):\n'+''.join(' '+line+'\n' for line in body.splitlines())+' return records,references,gpu,gallery_bindings(gallery)\n',ns);return ns['compute'],hashlib.sha256(body.encode()).hexdigest()
def compile_roma(m):
 body=segment(ROMA_SOURCE,'produce','    core = legacy_core(profile)','    reused = C.require_reused_engineering_bridges()')
 ns=dict(vars(m));exec('def compute(token,profile):\n'+''.join(' '+line+'\n' for line in body.splitlines())+' return records\n',ns);return ns['compute'],hashlib.sha256(body.encode()).hexdigest()
def entries(shard):
 w=read(META/'worker_manifest.json');rs=[r for r in w['records'] if 'reuse' not in r]
 need(shard in range(w['missing_shards']),'SHARD_INDEX');return rs[8*shard:8*(shard+1)]
def savepayload(folder,payload,extra):
 import torch
 folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';need(not p.exists(),'IMMUTABLE_PAYLOAD')
 tmp=folder/('.partial-'+str(os.getpid()));torch.save(payload,tmp);os.link(tmp,p);tmp.unlink()
 write(folder/'receipt.json',dict(payload=bind(p),authority=bind(AUTH),worker_manifest=bind(META/'worker_manifest.json'),**extra))
def child(stage,shard):
 nonce=uuid.uuid4().hex;env=dict(os.environ,CUDA_VISIBLE_DEVICES='',H593_VALIDATOR_NONCE=nonce)
 subprocess.run([sys.executable,__file__,stage,'--shard',str(shard),'--nonce',nonce],env=env,check=True)

def raw(shard):
 import torch
 m=loadmodule(RAW_SOURCE,'h593_qualified_raw');pre=read(RAW_PREFLIGHT)
 need(m.runtime_libraries()==pre['runtime_libraries'],'ORIGINAL_RAW_RUNTIME')
 # Stable encoder/gallery/ranking sources; workload-specific old manifests are not opened.
 a=read(AUTH)
 for b in a['raw_operator_sources'].values():checked(b)
 compute,digest=compile_raw(m);need(digest==a['numeric_loop_sha256']['raw'],'RAW_LOOP_PIN')
 with torch.inference_mode():records,refs,gpu,gallery=compute(entries(shard),pre)
 value=dict(status='H593_RAW_READY',shard=shard,records=records,references=refs,gallery=gallery,authority=bind(AUTH),worker_manifest=bind(META/'worker_manifest.json'),runtime_libraries=m.runtime_libraries(),GPU_runtime=gpu,numeric_loop_sha256=digest)
 m.plain_only(value);savepayload(OUT/'raw'/f'shard{shard:02d}',value,dict(status=value['status'],query_count=len(records),numeric_loop_sha256=digest));child('validate-raw',shard)

def envelope(kind,shard):
 folder=OUT/kind/f'shard{shard:02d}';r=read(folder/'receipt.json');need(r['authority']==bind(AUTH) and r['worker_manifest']==bind(META/'worker_manifest.json'),'ENVELOPE_BINDINGS')
 import torch
 p=torch.load(checked(r['payload']),map_location='cpu',weights_only=True,mmap=True)
 return p,dict(payload=r['payload'],receipt=bind(folder/'receipt.json'))

def validate_raw(shard):
 import torch
 m=loadmodule(RAW_SOURCE,'h593_raw_validator');p,b=envelope('raw',shard)
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 g=build_gallery_source(verify_cache_file_sha256=True);labels=g.corrected_identities
 expected=entries(shard);need(len(expected)==len(p['records']),'RAW_COMPLETE_SHARD')
 for r,e in zip(p['records'],expected):
  need(r['query_id']==e['query_id'] and r['execution_ordinal']==e['execution_ordinal'],'RAW_WORKER_AXIS')
  need(r['query_source_sha256']==e['source_image_sha256']==sha(e['query_image_path']),'QUERY_IMAGE_SHA')
  need(r['query_source_path']==e['query_image_path'] and r['processor_input_frame']==m.FRAMES[e['track']],'RAW_FRAME')
  for k in ('query_tokens','template_tokens'):need(m.tensor_sha(r[k])==r[k+'_sha256'],'QUERY_TOKEN_HASH')
  need(r['query_tokens'].shape==(math.prod(r['query_grid_shape']),128) and r['query_tokens'].dtype==torch.float16,'QUERY_GRID')
  scores=r['raw_physical_scores'];need(scores.shape==(5413,) and scores.dtype==torch.float64 and bool(torch.isfinite(scores).all()),'RAW_POPULATION')
  seen=set();order=[]
  for physical in sorted(range(5413),key=lambda x:(-float(scores[x]),x)):
   if labels[physical] not in seen:seen.add(labels[physical]);order.append(physical)
  need(len(order)==5412 and order==r['raw_ranked_physical_rows'],'INDEPENDENT_FULL_RANK')
  need(order[:128]==r['candidate_ranked_physical_rows'] and sorted(order[:128])==r['candidate_physical_rows'],'NATURAL_C128')
  need(torch.equal(scores[r['candidate_physical_rows']],r['candidate_raw_scores']),'RAW_C128_SCORES')
 for physical,ref in p['references'].items():
  need(ref['physical_row']==physical and ref['source_path']==str(g.raw_paths[physical]),'REFERENCE_AXIS')
  need(m.tensor_sha(ref['tokens'])==ref['tokens_sha256'] and ref['tokens'].shape==(math.prod(ref['grid_shape']),128),'REFERENCE_TOKEN_GRID')
  need(sha(ref['source_path'])==ref['source_image_sha256'],'REFERENCE_IMAGE_SHA')
 write(OUT/'raw'/f'shard{shard:02d}'/'validation.json',dict(status='H593_RAW_CPU_REPLAY_PASS',**b,authority=bind(AUTH),query_count=len(expected),fresh_nonce=os.environ['H593_VALIDATOR_NONCE'],target_reads=0))

def raw_validated(shard):
 v=read(OUT/'raw'/f'shard{shard:02d}'/'validation.json');need(v['status']=='H593_RAW_CPU_REPLAY_PASS','RAW_VALIDATED');p,b=envelope('raw',shard)
 need(v['payload']==b['payload'] and v['receipt']==b['receipt'],'RAW_REPLAY_BINDING');b['validation']=bind(OUT/'raw'/f'shard{shard:02d}'/'validation.json');return p,b

def roma(shard):
 import torch
 m=loadmodule(ROMA_SOURCE,'h593_qualified_roma');profile=read(PROFILE)
 for b in profile['sources'].values():checked(b)
 for tree in profile['source_trees'].values():
  need({str(p) for p in Path(tree['root']).rglob('*.py')}=={b['path'] for b in tree['python_files']},'ROMA_TREE_COVERAGE')
  for b in tree['python_files']:checked(b)
 token,tb=raw_validated(shard);compute,digest=compile_roma(m);need(digest==read(AUTH)['numeric_loop_sha256']['roma'],'ROMA_LOOP_PIN')
 with torch.inference_mode():records=compute(token,profile)
 value=dict(status='H593_ROMA_READY',shard=shard,records=records,token_source=tb,source_profile=bind(PROFILE),authority=bind(AUTH),numeric_loop_sha256=digest)
 savepayload(OUT/'roma'/f'shard{shard:02d}',value,dict(status=value['status'],token_source=tb,source_profile=bind(PROFILE),query_count=len(records),numeric_loop_sha256=digest));child('validate-roma',shard)

def validate_roma(shard):
 import torch
 m=loadmodule(ROMA_SOURCE,'h593_roma_validator');token,tb=raw_validated(shard);p,b=envelope('roma',shard);need(p['token_source']==tb,'ROMA_TOKEN_BINDING');n=0
 need(len(p['records'])==len(token['records']),'ROMA_QUERY_COUNT')
 for q,r in zip(token['records'],p['records']):
  need(q['query_id']==r['query_id'] and q['query_tokens_sha256']==r['query_tokens_sha256'] and q['query_source_sha256']==r['query_source_sha256'],'ROMA_QUERY_BINDING')
  need(q['candidate_physical_rows']==r['candidate_physical_rows'] and len(r['candidates'])==128,'ROMA_C128')
  for j,c in enumerate(r['candidates']):
   ref=token['references'][q['candidate_physical_rows'][j]];wq,wr=c['query_visibility'],c['reference_visibility']
   need(c['candidate_position']==j and c['physical_row']==ref['physical_row'] and c['reference_tokens_sha256']==ref['tokens_sha256'],'ROMA_REFERENCE')
   for w,count in ((wq,len(q['query_tokens'])),(wr,len(ref['tokens']))):need(w.dtype==torch.float64 and w.shape==(count,) and bool(torch.isfinite(w).all()) and bool(((w>=0)&(w<=1)).all()),'VISIBILITY_DOMAIN')
   need(m.hten(wq)==c['query_map_sha256'] and m.hten(wr)==c['reference_map_sha256'],'MAP_BYTES')
   scores=m.replay_c4(q['query_tokens'],ref['tokens'],wq,wr)
   need(all(float(v).hex()==float(c['old_scores'][k]).hex() for k,v in scores.items()),'INDEPENDENT_FP64_C4_REPLAY');n+=4
 write(OUT/'roma'/f'shard{shard:02d}'/'validation.json',dict(status='H593_ROMA_CPU_REPLAY_PASS',**b,token_source=tb,authority=bind(AUTH),independent_C4_scalars=n,all_C4_bits_exact=True,fresh_nonce=os.environ['H593_VALIDATOR_NONCE']))

def preflight():
 # Syntax of both extracted loops, no model or natural tensor read.
 for p in (RAW_SOURCE,ROMA_SOURCE):compile(p.read_text(),str(p),'exec')
 for name,p,fn,start,end in [('raw',RAW_SOURCE,'run_gpu',' from colpali_engine.models',' parity=[original_parity'),('roma',ROMA_SOURCE,'produce','    core = legacy_core(profile)','    reused = C.require_reused_engineering_bridges()')]:
  b=segment(p,fn,start,end);compile('def probe():\n'+''.join(' '+x+'\n' for x in b.splitlines()),str(p),'exec');need(hashlib.sha256(b.encode()).hexdigest()==read(AUTH)['numeric_loop_sha256'][name],'NUMERICAL_LOOP_FREEZE')
 w=read(META/'worker_manifest.json');flat=[r['query_id'] for i in range(43) for r in entries(i)];need(flat==w['missing_query_ids'] and len(flat)==len(set(flat))==337,'COMPLETE_MISSING_SHARDS')
 write(OUT/'input_preflight.json',dict(status='H593_INPUT_ADAPTER_PREFLIGHT_PASS',authority=bind(AUTH),missing_shards=43,last_shard_size=len(entries(42)),natural_tensor_reads=0,GPU_forwards=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','raw','roma','validate-raw','validate-roma']);ap.add_argument('--shard',type=int);ap.add_argument('--nonce');a=ap.parse_args();guard(a.stage);sys.addaudithook(audit)
 if a.stage=='preflight':preflight()
 else:
  import torch
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(17)
  if a.stage.startswith('validate'):need(a.nonce==os.environ.get('H593_VALIDATOR_NONCE'),'FRESH_VALIDATOR_NONCE')
  {'raw':raw,'roma':roma,'validate-raw':validate_raw,'validate-roma':validate_roma}[a.stage](a.shard)
