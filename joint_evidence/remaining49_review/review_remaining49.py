"""Export private visual review cards and cached-score controls; no inference."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import sys,json,csv,math,re,hashlib,tarfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1')
V=P/'background_causal_v1/joint_image_ocr_v1';D=V/'remaining54_v1';A=D/'roma_coordinates'
OUT=D/'remaining49_v1';DATA=Path('/mnt/lsdf_zyin/dataset');CACHE=Path('/mnt/lsdf_zyin/cache/ocr')
sys.path.insert(0,str(P))
from experiment_utils import Ranking,zscore,metrics
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def oriented(p):
 with Image.open(p) as src:
  ex=src.getexif();o=ex.get(274,1);im=src.copy()
  m={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}
  if o in m:im=im.transpose(m[o])
  im=im.convert('RGB');im.info.clear();return im
def fit(im,w,h):
 im=im.copy();im.thumbnail((w,h),Image.Resampling.LANCZOS);return im
def table(name,key):
 with (DATA/'lists'/f'{name}.csv').open() as f:return {r[key]:r for r in csv.DictReader(f)}

def main():
 OUT.mkdir(exist_ok=True)
 cohort=read(P/'cohort.json');qids=[r['query'] for r in cohort];qi={q:i for i,q in enumerate(qids)}
 viewer=read(P/'release/joint_evidence/CASE_VIEWER_DATA.json');vr={r['query']:r for r in viewer['rows']}
 cases=[r for r in viewer['rows'] if not r['query_face_excluded'] and r['coordinate_evidence']['factorial']['repaired_image_repaired_ocr']['rank']>1];assert len(cases)==49
 targets18=[r['query'] for r in cases if r['coordinate_evidence']['image']['rank']==1];assert len(targets18)==18
 protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'n':1117,'target49':[r['query'] for r in cases],'target18':targets18,'frozen_primary':'repaired_image_repaired_ocr=1058','images':'Original EXIF-oriented pixels, confirmed human subject bbox; no SAM, no color correction. Private review derivatives only.','controls':'Predeclared cached-score interface interventions evaluated on all1117; no new encoder/OCR inference, no per-query routing or truth insertion. Diagnostic contribution pairs selected after scoring. No replacement of the published1058 arm.','arms':['primary','image_mix_no_ocr','direct_repaired_image_current_ocr','remove_legacy_full_renorm','remove_subject_renorm','keep_old_image_scale','remove_text_keep_code','remove_code_keep_text']}
 write(OUT/'PROTOCOL.json',protocol)
 photos=table('photos','photo_id');refs=table('references','reference_id')
 ids=[r['reference_id'] for r in sorted(refs.values(),key=lambda x:int(x['row']))];sha=[refs[r]['sha256'] for r in ids];rid={r:j for j,r in enumerate(ids)}
 assert len(ids)==5405;rk=Ranking(ids,sha,cohort)
 policy=read(P/'SUBJECT_REGION_POLICY.json');assert policy['selection']=='reviewed_human_boxes_only' and not policy['sam_allowed'];boxes=read(policy['remote_source'])['records']
 qo=read(CACHE/'photos.json');ro=read(CACHE/'references.json');merged=read(D/'OCR_MERGED.json')
 import torch
 torch.set_num_threads(2)
 ib=torch.load('/mnt/lsdf_zyin/cache/scores/colnomic7b/image.pt',map_location='cpu',mmap=True,weights_only=True)
 assert ib['query_ids']==qids and ib['reference_ids']==ids
 old=ib['scores'].numpy().astype(float);sub=np.load(V/'prior_subject_scores.npy');oldnew=np.load(V/'selected_image_scores.npy');repair=np.load(A/'image_selected_scores.npy');inc=np.load(V/'incremental_image_scores.npy');final=np.load(A/'repaired_image_repaired_ocr_scores.npy')
 assert np.max(abs(inc-(.4*old+.2*sub+.4*oldnew)))<1e-7
 newinc=inc+.4*(repair-oldnew)
 oldtext=np.load(V/'production_numeric_z.npy');oldcode=np.load(D/'code_bonus.npy');text=oldtext.copy();code=oldcode.copy()
 tok=lambda s:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',s.lower(),flags=re.UNICODE)
 docs=[set(tok(ro[r]['text'])) for r in ids];post=defaultdict(list)
 for j,doc in enumerate(docs):
  for word in doc:post[word].append(j)
 weights={w:(np.array(js),np.full(len(js),math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5)),dtype=np.float32)) for w,js in post.items()}
 audit={r['query']:r for r in read(D/'SELECTIVE_OCR_MATCHES.json') if r['threshold']==0}
 for q,a in audit.items():
  i=qi[q]
  if not a['text'].strip():continue
  raw=np.zeros(len(ids),dtype=np.float32)
  for w,count in Counter(tok(a['text'])).items():
   if w in weights:ix,val=weights[w];raw[ix]+=count*val
  text[i]=zscore(raw)
  percode=defaultdict(dict)
  for m in a['matches']:
   js=m['references'];idf=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1)
   for r in js:percode[m['query_code']][rid[r]]=max(percode[m['query_code']].get(rid[r],0),idf)
  code[i]=0
  for m in percode.values():
   for j,val in m.items():code[i,j]+=val
 blendedtext=.5*(.75*oldtext+.25*text);blendedcode=.75*oldcode+.25*code;currentocr=blendedtext+blendedcode
 replay=.5*zscore(newinc)+currentocr;error=float(np.max(abs(final-replay)))
 print('REPLAY_COMPONENTS',json.dumps({'image_raw_reconstruction':float(np.max(abs(inc-(.4*old+.2*sub+.4*oldnew)))),'old_joint':float(np.max(abs(np.load(D/'fixed1051_scores.npy')-(.5*zscore(inc)+.5*oldtext+oldcode)))),'new_ocr':float(np.max(abs(np.load(D/'selected_ocr_scores.npy')-(.5*zscore(inc)+currentocr)))),'final':error}),flush=True)
 assert error<1e-7,error
 br,bw,_=rk.all(final);assert (br==1).sum()==1058
 scopes={'all':np.arange(len(cohort)),'validation':np.array([i for i,r in enumerate(cohort) if r['split']=='val']),'test':np.array([i for i,r in enumerate(cohort) if r['split']=='test']),'retained':np.array([i for i,q in enumerate(qids) if not vr[q]['query_face_excluded']]),'target49':np.array([qi[r['query']] for r in cases]),'target18':np.array([qi[q] for q in targets18])}
 arms={'primary':final,'image_mix_no_ocr':.5*zscore(newinc),'direct_repaired_image_current_ocr':.5*zscore(repair)+currentocr,'remove_legacy_full_renorm':.5*zscore(.2*sub+.4*repair)+currentocr,'remove_subject_renorm':.5*zscore(.4*old+.4*repair)+currentocr,'keep_old_image_scale':.5*(newinc-inc.mean(1,keepdims=True))/inc.std(1,keepdims=True)+currentocr,'remove_text_keep_code':.5*zscore(newinc)+blendedcode,'remove_code_keep_text':.5*zscore(newinc)+blendedtext}
 results={};rankrows={};pairs=[]
 for name,s in arms.items():
  ranks,winners,_=rk.all(s);results[name]={k:metrics(ranks,cohort,ix,br) for k,ix in scopes.items()};rankrows[name]={'ranks':ranks.tolist(),'winners':winners.tolist()}
  # Independent scalar check for all current49; original full score reconstruction checked all1117.
  for r in cases:
   i=qi[r['query']];seen=set();rank=0
   for j in np.argsort(-s[i],kind='stable'):
    if sha[j] in seen:continue
    seen.add(sha[j]);rank+=1
    if sha[j] in cohort[i]['positive_sha256']:break
   assert rank==ranks[i]
 for r in cases:
  q=r['query'];i=qi[q];f=rk.one(final[i],i);j=rid[f['positive']];k=rid[f['negative']];sd=newinc[i].std();gap=lambda x:float(x[i,j]-x[i,k])
  parts={'legacy_full':.5*.4*gap(old)/sd,'legacy_subject':.5*.2*gap(sub)/sd,'repaired_image':.5*.4*gap(repair)/sd,'text':gap(blendedtext),'code':gap(blendedcode)}
  assert abs(sum(parts.values())-f['margin'])<1e-7
  imagegap=parts['legacy_full']+parts['legacy_subject']+parts['repaired_image'];stage='old_image_overturns_repaired_image' if gap(repair)>0 and imagegap<=0 else 'ocr_overturns_image_mix' if imagegap>0 and f['margin']<0 else 'image_mix_overturns_positive_ocr' if parts['text']+parts['code']>0 else 'both_image_and_ocr_nonpositive'
  b=boxes[q];assert b['source']=='human' and b['status']=='confirmed';x,y,X,Y=b['bbox'];oldlines=[l for l in qo[q]['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y]
  pair=dict(query=q,positive=f['positive'],negative=f['negative'],rank=f['rank'],margin=f['margin'],pure_repaired_image_rank=r['coordinate_evidence']['image']['rank'],parts=parts,image_mix_gap=imagegap,pure_repaired_pair_gap=gap(repair),stage=stage,controls={n:rankrows[n]['ranks'][i] for n in arms},box=b,photo=photos[q],references={t:refs[t] for t in [f['positive'],f['negative']]},old_subject_ocr=oldlines,new_ocr=audit.get(q),positive_ocr=ro[f['positive']],negative_ocr=ro[f['negative']],existing_evidence=r['coordinate_evidence'].get('diagnosis'),visibility_review=r.get('visibility_review'))
  pairs.append(pair)
 write(OUT/'SCORE_AUDIT.json',dict(status='passed',queries=1117,replay_max_error=error,controls=results,rankrows=rankrows,queries_order=qids,target49=[r['query'] for r in cases],target18=targets18,cases=pairs,independent_target_rank_checks=49*len(arms),limits='Fixed global diagnostics on repeatedly inspected historical data; success counts are not a validation-selected new method. Score-stage attribution is not pixel/encoder causality.'))
 print('SCORE AUDIT',json.dumps({k:{s:v[s]['correct'] for s in ['all','validation','target49','target18']}for k,v in results.items()}),flush=True)
 assets=OUT/'private_assets';cards=OUT/'cards';assets.mkdir(exist_ok=True);cards.mkdir(exist_ok=True)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
 for n,r in enumerate(pairs,1):
  q=r['query'];im=oriented(DATA/photos[q]['path']);assert im.size==(r['box']['width'],r['box']['height']);box=r['box']['bbox'];crop=im.crop(tuple(round(v)for v in box));full=im.copy();ImageDraw.Draw(full).rectangle(box,outline='red',width=max(4,im.width//250))
  images=[full,crop,oriented(DATA/refs[r['positive']]['path']),oriented(DATA/refs[r['negative']]['path'])]
  labels=['Query + human box','Query subject',r['positive']+' CORRECT',r['negative']+' WRONG']
  widths=[420,620,720,720];canvas=Image.new('RGB',(sum(widths),930),'white');draw=ImageDraw.Draw(canvas);draw.text((10,8),f'{n:02} {q} | image rank {r["pure_repaired_image_rank"]} | final rank {r["rank"]}',font=font,fill='black');ox=0
  for k,(pic,label,w) in enumerate(zip(images,labels,widths)):
   draw.text((ox+8,45),label,font=font,fill='black');small=fit(pic,w-16,835);canvas.paste(small,(ox+(w-small.width)//2,85+(835-small.height)//2));ox+=w
   assetname=q+('_context' if k==0 else '_subject') if k<2 else (r['positive'] if k==2 else r['negative']);target=assets/(assetname+'.jpg')
   if not target.exists():fit(pic,3000,3000).save(target,quality=96)
  canvas.save(cards/f'{n:02}_{q}.jpg',quality=94)
  print('CARD',n,q,flush=True)
 # Export-only archive stays private; no full-image public release is implied.
 with tarfile.open(OUT/'REVIEW49_PRIVATE.tar.gz','w:gz') as tar:
  for f in sorted(OUT.rglob('*')):
   if f.is_file() and f.suffix not in ['.gz','.log'] and f.name!='REVIEW49_PRIVATE.tar.gz':tar.add(f,arcname=str(f.relative_to(OUT)),recursive=False)
 write(OUT/'DELIVERY.json',{'status':'complete','archive_sha256':hashlib.sha256((OUT/'REVIEW49_PRIVATE.tar.gz').read_bytes()).hexdigest(),'cards':49,'private_assets':len(list(assets.glob('*.jpg'))),'full_original_upload_authorized':False})
 print('DONE',flush=True)

if __name__=='__main__':main()
