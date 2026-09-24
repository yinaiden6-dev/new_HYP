#!/usr/bin/env python3
"""Explicit untrained-retrieval base control; native C128 and candidate-bound M."""
import argparse,copy,fcntl,hashlib,json,math,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parents[2]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_colpali_mass_transfer_v1 as U
read,write,bind,checked=U.read,U.write,U.bind,U.checked
OUT=ROOT/'results/rc_colqwen_base_native_v1';AUTH=ROOT/'registry/rc_colqwen_base_native_authority_v1_20260924.json'
MODEL=WORK/'models/downloaded_models/colqwen2.5-base';GALLERY=WORK/'colqwen/difficult/raw_gallery/cache/colqwen_gallery_emb_difficult.pt'
HEAD_AUTH=ROOT/'registry/rc_colqwen_base_head_authority_v1_20260924.json'
GPU=ROOT/'slurm/rc_colqwen_base_native_gpu_v1.sbatch';CPU=ROOT/'slurm/rc_colqwen_base_native_cpu_v1.sbatch'
PROFILE=ROOT/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
def save(p,v):
 import torch
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def prepare():
 import torch
 assert not AUTH.exists();parent=read(ROOT/'registry/rc_colpali_native_mass_head_authority_v1_20260923.json');g=read(checked(parent['gallery']))['records']
 d=torch.load(GALLERY,map_location='cpu',weights_only=True,mmap=True)
 assert len(d['passage_emb'])==len(g)==5413 and d['indices']==list(range(5413))
 assert [Path(x['image_path']).stem.strip() for x in g]==d['setids'],'GALLERY_ORDER'
 conf=read(MODEL/'config.json');assert conf['hidden_size']==2048 and conf['architectures']==['ColQwen2_5']
 catalog=read(ROOT/'results/rc_h593_feature_fusion_cache_v1/catalog.json');queries=[]
 for q in catalog['queries']:
  im=catalog['images'][q['query_image_key']]['item'];queries.append(dict(query_id=q['query_id'],execution_ordinal=q['execution_ordinal'],image_path=im['path'],image_sha256=im['image_sha256'],query_item=im))
 assert len(queries)==593
 write(OUT/'workers.json',dict(records=queries,references=g,labels_included=False,gallery_identity_is_public=True))
 sources=[WORK/'.venv-colpali/lib/python3.13/site-packages/colpali_engine/models/qwen2_5/colqwen2_5/modeling_colqwen2_5.py',WORK/'.venv-colpali/lib/python3.13/site-packages/colpali_engine/models/qwen2_5/colqwen2_5/processing_colqwen2_5.py',Path(__file__),ROOT/'programs/rc_colqwen_base_quality_core_v1.py',ROOT/'programs/rc_colqwen_base_head_core_v1.py',ROOT/'programs/run_rc_six_cause_loss_binding_v1.py',GPU,CPU,ROOT/'plan/RC_COLQWEN_BASE_NATIVE_V1_20260924.md']
 a=dict(sources=[bind(p) for p in sources],workers=bind(OUT/'workers.json'),gallery_cache=bind(GALLERY),model=str(MODEL),model_config=[bind(p) for p in MODEL.glob('*.json')],model_weights=[bind(p) for p in MODEL.glob('*.safetensors')],gallery=parent['gallery'],split=parent['split'],curator=parent['curator'],folds=parent['folds'],quality_profile=bind(PROFILE),shards=50,queries=593,models=['CONTENT7_COST1','CONTENT7_CE','MASS5_COST1','MASS5_CE'],anchors=[0,2706,5412],model_scope='Qwen2.5-VL-3B-Instruct plus checkpoint fixed untrained retrieval projection; no retrieval LoRA',candidate_source='Own ColQwen base FP64 full5413 MaxSim, public identity dedup, natural C128',scope='Base content with retrained simplified MASS5, not full local seven-parameter transfer',evidence='Opened H593 grouped OOF5; no target insertion; not externally confirmed',native_colpali_quality=bind(ROOT/'results/rc_colpali_native_quality_v1/ready_workers.json'),original_quality=bind(ROOT/'results/rc_colpali_mass_transfer_v1/workers.json'))
 write(AUTH,a);print({'prepared':True,'model':a['model_scope'],'queries':593,'gallery':5413},flush=True)
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
def score(a,w,index,budget):
 import torch
 import torch.nn.functional as F
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 torch.set_num_threads(8);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
 d=OUT/'native/shards'/f'{index:02d}';d.mkdir(parents=True,exist_ok=True);lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if index:assert read(OUT/'native/shards/00/validation.json')['status']=='BASE_NATIVE_SHARD_PASS'
 for b in a['model_config']:checked(b)
 if index==0:
  checked(a['gallery_cache'])
  for b in a['model_weights']:checked(b)
 gallery=torch.load(GALLERY,weights_only=True,map_location='cpu',mmap=True)['passage_emb'];mapping={r'^model\.':'language_model.'}
 model,info=ColQwen2_5.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',key_mapping=mapping,output_loading_info=True)
 clean={k:list(info.get(k,[])) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}
 write(d/'model_loading.json',dict(authority=bind(AUTH),issues=clean,key_mapping=mapping))
 assert not any(clean.values()),clean
 model=model.to('cuda').eval();processor=ColQwen2_5_Processor.from_pretrained(MODEL,local_files_only=True)
 anchors=[];pre=post=None
 for physical in (a['anchors'] if index==0 else []):
  enc=encode(model,processor,w['references'][physical]['image_path']);old=gallery[physical].double();new=enc['tokens'].double();assert old.shape==new.shape
  cosine=F.cosine_similarity(old,new,dim=1);l2=float(torch.linalg.vector_norm(old-new)/torch.linalg.vector_norm(old));assert float(cosine.min())>=.999 and l2<=.03,('CACHE_MISMATCH',physical,float(cosine.min()),l2)
  if pre is not None:assert (pre,post)==(enc['prefix'],enc['suffix'])
  pre,post=enc['prefix'],enc['suffix'];anchors.append(dict(physical_row=physical,min_cosine=float(cosine.min()),relative_l2=l2,image_sha256=bind(w['references'][physical]['image_path'])['sha256']))
 if index==0:write(OUT/'compatibility.json',dict(authority=bind(AUTH),anchors=anchors,prefix=pre,suffix=post,status='BASE_GALLERY_ANCHORS_PASS'))
 else:
  comp=read(OUT/'compatibility.json');assert comp['authority']==bind(AUTH);pre,post=comp['prefix'],comp['suffix']
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
 gpu=stage in ['score','quality'];cmd=['sbatch','--parsable']
 if dep:cmd+=['--dependency=afterok:'+dep]
 if array:cmd+=['--array='+array]
 cmd+=[str(GPU if gpu else CPU),stage];p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60);job=p.stdout.strip().split(';')[0];assert job.isdigit();return job
def advance(stage,index):
 job=os.environ['SLURM_JOB_ID'];path=OUT/'dispatch'/f'{job}_{stage}.json'
 if path.exists():return
 jobs={}
 if stage=='score' and index==0:
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
  if x.stage=='score':sys.exit(score(a,w,x.index,x.budget))
  elif x.stage=='quality-prepare':quality_prepare(a,w)
  elif x.stage=='quality':sys.exit(quality(x.stage,x.index,x.budget))
  elif x.stage=='quality-join':quality(x.stage,x.index,x.budget);head_prepare(a)
  elif x.stage in ['fit','join']:sys.exit(head(x.stage,x.index))
  else:raise ValueError(x.stage)
