#!/usr/bin/env python3
"""Explicit untrained-retrieval base control; native C128 and candidate-bound M."""
import argparse,copy,fcntl,hashlib,json,math,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[2]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_colpali_mass_transfer_v1 as U
read,write,bind,checked=U.read,U.write,U.bind,U.checked
OUT=ROOT/'results/rc_colqwen_base_native_v2';AUTH=ROOT/'registry/rc_colqwen_base_native_authority_v2_20260924.json'
MODEL=WORK/'models/downloaded_models/colqwen2.5-base';GALLERY=WORK/'colqwen/difficult/raw_gallery/cache/colqwen_gallery_emb_difficult.pt'
HEAD_AUTH=ROOT/'registry/rc_colqwen_base_head_authority_v2_20260924.json'
GPU=ROOT/'slurm/rc_colqwen_base_native_gpu_v2.sbatch';CPU=ROOT/'slurm/rc_colqwen_base_native_cpu_v2.sbatch'
PROFILE=ROOT/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
def save(p,v):
 import torch
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def prepare():
 assert not AUTH.exists()
 previous=ROOT/'registry/rc_colqwen_base_native_authority_v1_20260924.json'
 a=copy.deepcopy(read(previous));w=read(checked(a['workers']))
 write(OUT/'workers.json',w)
 sources=[Path(__file__),GPU,CPU,ROOT/'plan/RC_COLQWEN_BASE_NATIVE_V2_20260924.md']
 for b in a['sources']:
  path=Path(b['path'])
  if path.name not in ['run_rc_colqwen_base_native_v1.py','rc_colqwen_base_native_gpu_v1.sbatch','rc_colqwen_base_native_cpu_v1.sbatch','RC_COLQWEN_BASE_NATIVE_V1_20260924.md']:
   checked(b);sources.append(path)
 sources.append(Path(U.__file__))
 a.update(sources=[bind(x) for x in sources],workers=bind(OUT/'workers.json'),parent=bind(previous),
  repair='Do not mix unmatched encoders. Label-free three-anchor compatibility chooses fixed recipe; if none pass, encode a new full gallery with the same recipe as queries.',
  compatibility_recipes=[dict(name='sdpa_slow',attention='sdpa',use_fast=False),dict(name='eager_slow',attention='eager',use_fast=False)],
  fallback_recipe=dict(name='sdpa_slow',attention='sdpa',use_fast=False),
  original_failure=dict(job='5161030',min_cosine=0.13808980742087462,relative_l2=0.1885817209583228),
  max_requeues=16)
 write(AUTH,a);print({'prepared':True,'authority':bind(AUTH)},flush=True)
def guard():
 a=read(AUTH)
 for b in a['sources']:checked(b)
 assert os.environ.get('SLURM_JOB_ID')
 return a,read(checked(a['workers']))
def encode(model,processor,path):
 import torch
 from PIL import Image
 with Image.open(path) as im:inp=processor.process_images([im.convert('RGB')])
 valid=inp['attention_mask'][0].bool();ids=inp['input_ids'][0][valid];mask=ids==model.config.image_token_id
 where=mask.nonzero().flatten();assert len(where)>0 and torch.equal(where,torch.arange(int(where[0]),int(where[-1])+1))
 with torch.inference_mode():t=model(**{k:v.to('cuda') for k,v in inp.items()})[0][valid.to('cuda')].to('cpu',dtype=torch.float16)
 return dict(tokens=t,image_mask=mask,grid=inp['image_grid_thw'][0].tolist(),prefix=int(where[0]),suffix=len(ids)-1-int(where[-1]))
def load_encoder(a,recipe,d,verify_weights=False):
 import torch
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 torch.set_num_threads(8);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
 for b in a['model_config']:checked(b)
 if verify_weights:
  for b in a['model_weights']:checked(b)
 mapping={r'^model\.':'language_model.'}
 model,info=ColQwen2_5.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=recipe['attention'],key_mapping=mapping,output_loading_info=True)
 clean={k:list(info.get(k,[])) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
 assert not any(clean.values()),clean
 model=model.to('cuda').eval();processor=ColQwen2_5_Processor.from_pretrained(MODEL,local_files_only=True,use_fast=recipe['use_fast'])
 write(d/'model_loading.json',dict(authority=bind(AUTH),issues=clean,key_mapping=mapping,recipe=recipe,processor_class=type(processor.image_processor).__name__))
 return model,processor

def compare_tokens(old,new):
 import torch
 import torch.nn.functional as F
 if old.shape!=new.shape:return dict(passed=False,old_shape=list(old.shape),new_shape=list(new.shape))
 old=old.double();new=new.double();cos=F.cosine_similarity(old,new,dim=1);l2=float(torch.linalg.vector_norm(old-new)/torch.linalg.vector_norm(old))
 return dict(passed=float(cos.min())>=.999 and l2<=.03,min_cosine=float(cos.min()),mean_cosine=float(cos.mean()),relative_l2=l2)

def compatibility(a,w):
 import torch,gc
 dest=OUT/'compatibility';dest.mkdir(parents=True,exist_ok=True)
 if (OUT/'compatibility.json').exists():assert read(OUT/'compatibility.json')['authority']==bind(AUTH);return 0
 old=torch.load(checked(a['gallery_cache']),map_location='cpu',weights_only=True,mmap=True)['passage_emb']
 attempts=[];chosen=None
 for recipe in a['compatibility_recipes']:
  d=dest/recipe['name'];d.mkdir(parents=True,exist_ok=True)
  model,processor=load_encoder(a,recipe,d,verify_weights=not attempts);anchors=[]
  for physical in a['anchors']:
   path=w['references'][physical]['image_path'];enc=encode(model,processor,path)
   enc.update(authority=bind(AUTH),recipe=recipe,physical_row=physical,image_sha256=bind(path)['sha256']);save(d/f'anchor{physical}.pt',enc)
   metrics=compare_tokens(old[physical],enc['tokens']);mask=enc['image_mask'];image_metrics=compare_tokens(old[physical][mask],enc['tokens'][mask]) if len(old[physical])==len(mask) else {'passed':False}
   anchors.append(dict(physical_row=physical,metrics=metrics,image_metrics=image_metrics,prefix=enc['prefix'],suffix=enc['suffix'],encoded=bind(d/f'anchor{physical}.pt')))
   print({'recipe':recipe['name'],'anchor':physical,**metrics},flush=True)
  attempt=dict(recipe=recipe,anchors=anchors,passed=all(x['metrics']['passed'] for x in anchors));write(d/'validation.json',attempt);attempts.append(attempt)
  del model,processor;gc.collect();torch.cuda.empty_cache()
  if attempt['passed']:chosen=attempt;break
 selected=chosen or next(x for x in attempts if x['recipe']==a['fallback_recipe'])
 assert len({(x['prefix'],x['suffix']) for x in selected['anchors']})==1
 v=dict(status='ENCODER_RECIPE_FROZEN',authority=bind(AUTH),mode='reuse_verified' if chosen else 'reencode_gallery',recipe=selected['recipe'],prefix=selected['anchors'][0]['prefix'],suffix=selected['anchors'][0]['suffix'],attempts=attempts,selection_uses='ENCODING_PARITY_ONLY_NO_LABELS')
 write(OUT/'compatibility.json',v)
 if chosen:write(OUT/'gallery/validation.json',dict(status='BASE_GALLERY_READY',authority=bind(AUTH),gallery_cache=a['gallery_cache'],compatibility=bind(OUT/'compatibility.json'),count=5413))
 print({'compatibility_mode':v['mode'],'recipe':v['recipe']},flush=True);return 0

def gallery_encode(a,w,index,budget):
 import torch
 comp=read(OUT/'compatibility.json');assert comp['authority']==bind(AUTH) and comp['mode']=='reencode_gallery'
 d=OUT/'gallery/shards'/f'{index:02d}';d.mkdir(parents=True,exist_ok=True);lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if index:assert read(OUT/'gallery/shards/00/validation.json')['status']=='BASE_GALLERY_SHARD_PASS'
 model,processor=load_encoder(a,comp['recipe'],d);start=time.monotonic();receipts=[]
 selected=next(x for x in comp['attempts'] if x['recipe']==comp['recipe']);anchor_map={x['physical_row']:x for x in selected['anchors']}
 for physical in range(index,5413,50):
  td=OUT/'gallery/tokens';tp=td/f'{physical:04d}.pt';vp=td/f'{physical:04d}.json';path=w['references'][physical]['image_path']
  if vp.exists():v=read(vp);assert v['authority']==bind(AUTH);checked(v['tokens']);receipts.append(bind(vp));continue
  if time.monotonic()-start>budget:return 75
  enc=encode(model,processor,path);assert (enc['prefix'],enc['suffix'])==(comp['prefix'],comp['suffix'])
  enc.update(authority=bind(AUTH),recipe=comp['recipe'],physical_row=physical,image_sha256=bind(path)['sha256']);parity=None
  if physical in anchor_map:
   cached=torch.load(checked(anchor_map[physical]['encoded']),weights_only=True,map_location='cpu');parity=compare_tokens(cached['tokens'],enc['tokens']);assert parity['passed'],('NEW_ENCODER_PARITY',physical,parity)
  save(tp,enc);write(vp,dict(authority=bind(AUTH),physical_row=physical,tokens=bind(tp),image_sha256=enc['image_sha256'],anchor_parity=parity));receipts.append(bind(vp))
  print({'gallery':physical,'seconds':time.monotonic()-start},flush=True)
 write(d/'validation.json',dict(status='BASE_GALLERY_SHARD_PASS',authority=bind(AUTH),tokens=receipts));return 0

def gallery_join(a,w):
 import torch
 comp=read(OUT/'compatibility.json');assert comp['authority']==bind(AUTH)
 indices=[]
 for index in range(50):
  v=read(OUT/'gallery/shards'/f'{index:02d}'/'validation.json');assert v['authority']==bind(AUTH) and v['status']=='BASE_GALLERY_SHARD_PASS'
  indices.extend(read(checked(b))['physical_row'] for b in v['tokens'])
 assert sorted(indices)==list(range(5413))
 bank=[]
 for physical in range(5413):
  v=read(OUT/'gallery/tokens'/f'{physical:04d}.json');t=torch.load(checked(v['tokens']),weights_only=True,map_location='cpu')
  assert t['authority']==bind(AUTH) and t['physical_row']==physical and t['recipe']==comp['recipe']
  assert t['tokens'].ndim==2 and t['tokens'].shape[1]==128 and torch.isfinite(t['tokens']).all()
  bank.append(t['tokens'])
 dest=OUT/'gallery/gallery_tokens.pt';save(dest,dict(passage_emb=bank,indices=list(range(5413)),setids=[Path(x['image_path']).stem.strip() for x in w['references']]))
 write(OUT/'gallery/validation.json',dict(status='BASE_GALLERY_READY',authority=bind(AUTH),gallery_cache=bind(dest),compatibility=bind(OUT/'compatibility.json'),count=5413));return 0

def score(a,w,index,budget):
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(8);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
 d=OUT/'native/shards'/f'{index:02d}';d.mkdir(parents=True,exist_ok=True);lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if index:assert read(OUT/'native/shards/00/validation.json')['status']=='BASE_NATIVE_SHARD_PASS'
 comp=read(OUT/'compatibility.json');assert comp['authority']==bind(AUTH)
 gv=read(OUT/'gallery/validation.json');assert gv['authority']==bind(AUTH) and gv['status']=='BASE_GALLERY_READY'
 gallery=torch.load(checked(gv['gallery_cache']) if index==0 else gv['gallery_cache']['path'],weights_only=True,map_location='cpu',mmap=True)['passage_emb']
 model,processor=load_encoder(a,comp['recipe'],d);pre,post=comp['prefix'],comp['suffix']
 if index==0:
  anchors=[]
  for physical in a['anchors']:
   enc=encode(model,processor,w['references'][physical]['image_path']);metrics=compare_tokens(gallery[physical],enc['tokens']);assert metrics['passed'],('CACHE_MISMATCH',physical,metrics)
   anchors.append(dict(physical_row=physical,**metrics))
  write(d/'gallery_parity.json',dict(authority=bind(AUTH),anchors=anchors,gallery_validation=bind(OUT/'gallery/validation.json')))
 lengths=torch.tensor([len(x) for x in gallery],device='cuda');maxlen=int(lengths.max());refs=torch.zeros((5413,maxlen,128),dtype=torch.float64,device='cuda')
 for i,x in enumerate(gallery):refs[i,:len(x)]=x.to('cuda',dtype=torch.float64)
 start=time.monotonic();seals=[]
 for row in [x for x in w['records'] if x['execution_ordinal']%50==index]:
  qd=OUT/'native/queries'/f"query{row['execution_ordinal']:03d}";vp=qd/'validation.json';qd.mkdir(parents=True,exist_ok=True)
  if vp.exists():v=read(vp);assert v['authority']==bind(AUTH);checked(v['payload']);seals.append(bind(vp));continue
  if time.monotonic()-start>budget:return 75
  tp=OUT/'tokens'/(row['query_id']+'.pt')
  if tp.exists():enc=torch.load(tp,weights_only=True);assert enc['image_sha256']==row['image_sha256'] and enc['authority']==bind(AUTH)
  else:
   assert bind(row['image_path'])['sha256']==row['image_sha256'];enc=encode(model,processor,row['image_path']);enc.update(authority=bind(AUTH),query_id=row['query_id'],image_sha256=row['image_sha256']);save(tp,enc)
  assert (enc['prefix'],enc['suffix'])==(pre,post)
  query=enc['tokens'].to('cuda',dtype=torch.float64);raw=[];part=qd/'partial.json'
  if part.exists():v=read(part);assert v['authority']==bind(AUTH);raw=v['scores']
  with torch.inference_mode():
   for pos in range(len(raw),5413,8):
    sim=torch.einsum('nd,csd->cns',query,refs[pos:pos+8]);mask=torch.arange(maxlen,device='cuda')[None,:]>=lengths[pos:pos+8,None];sim.masked_fill_(mask[:,None,:],-torch.inf);raw.extend(sim.max(2).values.sum(1).cpu().tolist())
    if pos%256==0 or len(raw)==5413 or time.monotonic()-start>budget:write_mutable(part,dict(authority=bind(AUTH),scores=raw))
    if len(raw)<5413 and time.monotonic()-start>budget:return 75
   order=sorted(range(5413),key=lambda j:(-raw[j],j));seen=set();rank=[]
   for j in order:
    identity=w['references'][j]['identity']
    if identity not in seen:seen.add(identity);rank.append(j)
   axis=sorted(rank[:128]);winner=axis.index(rank[0]);q=F.normalize(query[enc['image_mask'].to('cuda')],dim=1);stats=[];traces=[]
   for j in axis:
    r=F.normalize(refs[j,pre:int(lengths[j])-post],dim=1);sim=q@r.T;qt=sim.topk(2,dim=1);rt=sim.topk(2,dim=0)
    stats.append([float(qt.values[:,0].mean()),float(rt.values[0].mean()),float((qt.values[:,0]-qt.values[:,1]).mean()),float((rt.values[0]-rt.values[1]).mean()),float(sim.mean())]);traces.append(dict(physical_row=j,query_argmax=qt.indices[:,0].cpu(),reference_argmax=rt.indices[0].cpu(),query_top1=qt.values[:,0].cpu(),reference_top1=rt.values[0].cpu()))
   errors={}
   for j in sorted({0,rank[0],rank[127]}):
    expected=float((query.cpu().numpy()@gallery[j].double().numpy().T).max(1).sum());errors[str(j)]=abs(expected-raw[j]);assert errors[str(j)]<2e-9
  trace=qd/'trace.pt';save(trace,dict(authority=bind(AUTH),query_id=row['query_id'],traces=traces))
  payload=dict(authority=bind(AUTH),query_id=row['query_id'],execution_ordinal=row['execution_ordinal'],image_sha256=row['image_sha256'],query_tokens=bind(tp),candidate_physical_rows=axis,raw_scores=[raw[j] for j in axis],winner=winner,challenger_positions=[i for i in range(128) if i!=winner],statistics=stats,gallery_scores=raw,ranked_physical_rows=rank,intermediate=bind(trace),label_reads=0)
  write(qd/'payload.json',payload);write(vp,dict(status='BASE_NATIVE_QUERY_PASS',authority=bind(AUTH),payload=bind(qd/'payload.json'),numpy_errors=errors));seals.append(bind(vp));print({'query':row['execution_ordinal'],'seconds':time.monotonic()-start},flush=True)
 write(d/'validation.json',dict(status='BASE_NATIVE_SHARD_PASS',authority=bind(AUTH),queries=seals));return 0
def write_mutable(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def native_rows():
 a=read(AUTH);rows=[]
 for i in range(50):
  v=read(OUT/'native/shards'/f'{i:02d}'/'validation.json');assert v['authority']==bind(AUTH)
  for b in v['queries']:rows.append(read(checked(read(checked(b))['payload'])))
 rows.sort(key=lambda x:x['execution_ordinal']);assert [x['execution_ordinal'] for x in rows]==list(range(593));return rows
def quality_prepare(a,w):
 from PIL import Image
 from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
 native=native_rows();orig={x['query_id']:x for x in read(checked(a['original_quality']))['records']};extra={x['query_id']:x for x in read(checked(a['native_colpali_quality']))['records']}
 catalog=read(ROOT/'results/rc_h593_feature_fusion_cache_v1/catalog.json');items={x['item']['path']:x['item'] for x in catalog['images'].values()};profile=read(checked(a['quality_profile']));proc=read(checked(profile['sources']['processor']));query={x['query_id']:x for x in w['records']};references={};records=[]
 for row in native:
  qid=row['query_id'];o=orig[qid];m=dict(zip(o['candidate_physical_rows'],o['mass']));e=extra[qid]
  for j,x in zip(e['candidate_physical_rows'],e['mass']):
   if j in m:assert abs(m[j]-x)<1e-10
   m[j]=x
  axis=row['candidate_physical_rows']
  for j in axis:
   if str(j) in references:continue
   path=w['references'][j]['image_path']
   if path in items:im=dict(items[path])
   else:
    with Image.open(path) as img:h,ww=img.height,img.width
    hh,width=smart_resize(h,ww,factor=28,min_pixels=proc['min_pixels'],max_pixels=proc['max_pixels']);im=dict(path=path,image_sha256=bind(path)['sha256'],grid=[hh//28,width//28],frame='DECODED_RAW_BEFORE_EXIF')
   references[str(j)]=im
  records.append(dict(query_id=qid,execution_ordinal=row['execution_ordinal'],query_item=query[qid]['query_item'],candidate_physical_rows=axis,query_tokens=row['query_tokens'],query_receipt=bind(OUT/'native/queries'/f"query{row['execution_ordinal']:03d}"/'validation.json'),image_sha256=row['image_sha256'],reuse_mass={str(j):m[j] for j in axis if j in m},missing=[j for j in axis if j not in m],reuse_source=dict(original=o['operator_payload'],colpali=e['quality_source'])))
 write(OUT/'quality/workers.json',dict(records=records,references=references,labels_included=False));write(OUT/'quality/coverage.json',dict(reused=sum(len(x['reuse_mass']) for x in records),missing=sum(len(x['missing']) for x in records),pairs=593*128));print(read(OUT/'quality/coverage.json'),flush=True)
def quality(stage,index,budget):
 import rc_colqwen_base_quality_core_v1 as Q
 Q.OUT=OUT/'quality';Q.AUTH=AUTH;Q.PROFILE=PROFILE;a=read(AUTH);counts=read(OUT/'quality/coverage.json');a.update(reused_pairs=counts['reused'],missing_pairs=counts['missing']);w=read(OUT/'quality/workers.json')
 if stage=='quality':return Q.worker(a,w,index,budget)
 Q.join(a,w);return 0
def head_prepare(a):
 v=read(OUT/'quality/validation.json');assert v['status']=='COLQWEN_BASE_NATIVE_ALL_QUALITY_PASS';qs={x['query_id']:x for x in read(checked(v['workers']))['records']};rows=native_rows()
 for r in rows:r['mass']=qs[r['query_id']]['mass']
 write(OUT/'head/rows.json',dict(records=rows));h={k:a[k] for k in ['gallery','split','curator','models','candidate_source','scope','evidence']};h['sources']=a['sources'];h['parent']=bind(AUTH);h['rows']=bind(OUT/'head/rows.json');h['folds']=copy.deepcopy(a['folds']);labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']};byid={r['query_id']:r for r in rows}
 for f,b in h['folds'].items():
  roles={r['query_id']:r['identity'] for r in read(checked(b['train_roles']))['records']};b['effective_train_query_ids']=[q for q in b['train_query_ids'] if any(labels[j]==roles[q] for j in byid[q]['candidate_physical_rows'])]
 write(HEAD_AUTH,h)
def head(stage,index):
 import rc_colqwen_base_head_core_v1 as H
 H.OUT=OUT/'head';H.AUTH=HEAD_AUTH;H.load_rows=lambda a:read(checked(a['rows']))['records'];a=read(HEAD_AUTH)
 if stage=='fit':return H.fit(a,index,450)
 H.join(a);return 0
def submit(stage,dep=None,array=None):
 gpu=stage in ['compat','gallery','score','quality'];cmd=['sbatch','--parsable']
 if dep:cmd+=['--dependency=afterok:'+dep]
 if array:cmd+=['--array='+array]
 cmd+=[str(GPU if gpu else CPU),stage];p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60);job=p.stdout.strip().split(';')[0];assert job.isdigit();return job
def advance(stage,index):
 job=os.environ['SLURM_JOB_ID'];path=OUT/'dispatch'/f'{job}_{stage}.json'
 if path.exists():return
 jobs={}
 if stage=='compat':
  comp=read(OUT/'compatibility.json')
  jobs['next']=submit('score' if comp['mode']=='reuse_verified' else 'gallery',job,'0')
 elif stage=='gallery' and index==0:
  assert read(OUT/'gallery/shards/00/validation.json')['status']=='BASE_GALLERY_SHARD_PASS'
  jobs['gallery_rest']=submit('gallery',job,'1-49%50');jobs['gallery_join']=submit('gallery-join',jobs['gallery_rest'])
 elif stage=='gallery-join':jobs['score_pilot']=submit('score',job,'0')
 elif stage=='score' and index==0:
  assert read(OUT/'native/shards/00/validation.json')['status']=='BASE_NATIVE_SHARD_PASS';jobs['score_rest']=submit('score',job,'1-49%50');jobs['quality_prepare']=submit('quality-prepare',jobs['score_rest'])
 elif stage=='quality-prepare':
  jobs['quality_pilot']=submit('quality',job,'0');jobs['quality_rest']=submit('quality',jobs['quality_pilot'],'1-49%50');jobs['quality_join']=submit('quality-join',jobs['quality_rest'])
 elif stage=='quality-join':
  jobs['fit']=submit('fit',job,'0-4%5');jobs['join']=submit('join',jobs['fit'])
 write(path,dict(stage=stage,previous=job,jobs=jobs));print({'submitted':jobs},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')));p.add_argument('--budget',type=int,default=660);x=p.parse_args()
 if x.stage=='prepare':prepare()
 elif x.stage.startswith('advance-'):advance(x.stage[len('advance-'):],x.index)
 else:
  a,w=guard()
  if x.stage=='compat':sys.exit(compatibility(a,w))
  elif x.stage=='gallery':sys.exit(gallery_encode(a,w,x.index,x.budget))
  elif x.stage=='gallery-join':sys.exit(gallery_join(a,w))
  elif x.stage=='score':sys.exit(score(a,w,x.index,x.budget))
  elif x.stage=='quality-prepare':quality_prepare(a,w)
  elif x.stage=='quality':sys.exit(quality(x.stage,x.index,x.budget))
  elif x.stage=='quality-join':quality(x.stage,x.index,x.budget);head_prepare(a)
  elif x.stage in ['fit','join']:sys.exit(head(x.stage,x.index))
  else:raise ValueError(x.stage)
