"""Evaluate the complete observable-trigger OCR rerun, including success controls."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,time,json,math,re,hashlib,argparse
import numpy as np,torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,zscore,Ranking,metrics
ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
protocol=dict(trigger='OCR_TRIGGER.json',thresholds=[0,.8,.9],weights=[0,.25,.5,1],code_update=[False,True],score='fixed1051 + weight*0.5*(new_text_z-old_text_z) + weight*(new_code_bonus-old_code_bonus) if code_update; code branch stays fixed otherwise',selection='All 173 original validation queries: package macro Top1 then macro MRR then smaller weight, lower confidence threshold and no code update.',empty_text_policy='If no recognized line remains at the fixed threshold, preserve old text and code scores.',scope='Exploratory historical full-gallery evaluation; all triggered successes and failures included. Human subject boxes, no hand-read text.',runtime='Existing pinned OCR models, CPU Paddle3 runtime; historical runtime parity not presumed.')
write(D/'SELECTIVE_OCR_EVAL_PROTOCOL.json',protocol)
plan=read(D/'OCR_TRIGGER.json');expected={x['query'] for x in plan['records']};assert len(expected)==379
started=time.time()
while True:
 merged={}
 try:
  for f in sorted(D.glob('new_ocr[0-9]*.json')):
   for q,x in read(f).items():
    if q in merged:assert merged[q]==x
    merged[q]=x
 except json.JSONDecodeError:
  if not args.wait:raise
  time.sleep(10);continue
 assert set(merged)<=expected
 if set(merged)==expected:break
 if not args.wait:raise RuntimeError(f'Incomplete OCR {len(merged)}/{len(expected)}')
 if time.time()-started>12*3600:raise RuntimeError('OCR completion wait exceeded 12 hours')
 time.sleep(30)
write(D/'OCR_MERGED.json',merged)
c=read(P/'cohort.json');qids=[x['query'] for x in c];qi={q:i for i,q in enumerate(qids)}
rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);ids=rb['ids'];rk=Ranking(ids,rb['sha256'],c)
fixed=np.load(D/'fixed1051_scores.npy');oldcode=np.load(D/'code_bonus.npy');oldtext=np.load(V/'production_numeric_z.npy');base,_,_=rk.all(fixed);assert int((base==1).sum())==1051
excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries']);target=np.array([i for i,x in enumerate(c) if x['query'] not in excluded and base[i]>1]);assert len(target)==54
scopes={'all':np.arange(len(c)),'validation':np.array([i for i,x in enumerate(c) if x['split']=='val']),'test':np.array([i for i,x in enumerate(c) if x['split']=='test']),'retained':np.array([i for i,x in enumerate(c) if x['query'] not in excluded]),'target54':target,'triggered':np.array([qi[q] for q in expected])}
qo=read(R/'cache/ocr/photos.json');ro=read(R/'cache/ocr/references.json');policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records']
tok=lambda s:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',s.lower(),flags=re.UNICODE)
wordpost=defaultdict(list)
for j,r in enumerate(ids):
 for w in set(tok(ro[r]['text'])):wordpost[w].append(j)
wordweights={w:(np.array(js),np.full(len(js),math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5)),dtype=np.float32)) for w,js in wordpost.items()}
def textscore(s):
 # Match the historical producer's FP32 term weights and sequential accumulation.
 a=np.zeros(len(ids),dtype=np.float32)
 for w,count in Counter(tok(s)).items():
  if w in wordweights:ix,v=wordweights[w];a[ix]+=count*v
 return a
def codes(text):
 out=set()
 for line in text.upper().splitlines():
  cand=re.findall(r'[A-Z0-9]+(?:[-./][A-Z0-9]+)*',line)
  if re.fullmatch(r'[A-Z0-9\s./-]{5,24}',line.strip()):cand.append(line)
  for s in cand:
   s=re.sub('[^A-Z0-9]','',s);n=sum(x.isdigit() for x in s)
   if 5<=len(s)<=18 and n>=5 and len(s)-n<=4 and len(set(s))>1:out.add(s)
 return out
posts=defaultdict(set)
for j,r in enumerate(ids):
 for code in codes(ro[r]['text']):posts[code].add(j)
posts={s:js for s,js in posts.items() if len(js)<=len(ids)*.01};deletions=defaultdict(set)
for s in posts:
 if len(s)>=7:
  deletions[s].add(s)
  for k in range(len(s)):deletions[s[:k]+s[k+1:]].add(s)
def edit(a,b):
 if abs(len(a)-len(b))>1:return False
 if len(a)==len(b):return sum(x!=y for x,y in zip(a,b))<=1
 if len(a)>len(b):a,b=b,a
 i=j=e=0
 while i<len(a) and j<len(b):
  if a[i]==b[j]:i+=1;j+=1
  else:e+=1;j+=1
  if e>1:return False
 return True
def bonus(s):
 out=np.zeros(len(ids));matches=[]
 for code in sorted(codes(s)):
  cand={code} if code in posts else set()
  if len(code)>=8:
   cand.update(deletions.get(code,set()))
   for k in range(len(code)):cand.update(deletions.get(code[:k]+code[k+1:],set()))
   cand={x for x in cand if min(len(x),len(code))>=8 and edit(code,x)}
  a=np.zeros(len(ids))
  for x in sorted(cand):
   js=sorted(posts[x]);idf=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1);a[js]=np.maximum(a[js],idf);matches.append(dict(query_code=code,reference_code=x,references=[ids[j] for j in js]))
  out+=a
 return out,matches
with np.load(P/'text_scores.npz') as old:
 assert old['query_ids'].tolist()==qids and old['reference_ids'].tolist()==ids
 rawold=old['numeric']
maxtext=maxcode=0.;audit=[];replay=[]
for q in qids:
 i=qi[q];x,y,X,Y=boxes[q]['bbox'];s='\n'.join(l['text'] for l in qo[q]['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y)
 te=float(abs(textscore(s)-rawold[i]).max());b,_=bonus(s);ce=float(abs(b-oldcode[i]).max())
 maxtext=max(maxtext,te);maxcode=max(maxcode,ce);replay.append(dict(query=q,text_max_error=te,code_max_error=ce))
write(D/'OCR_BASELINE_REPLAY.json',dict(status='passed' if maxtext==0 and maxcode<1e-7 else 'failed',queries=len(qids),text_max_error=maxtext,code_max_error=maxcode,score_precision='Historical FP32 IDF term arrays and FP32 sequential accumulation; no tolerance relaxation.',previous_fp64_max_error=0.0005077575324321515,checks=replay))
assert maxtext==0 and maxcode<1e-7,(maxtext,maxcode)
choices=[];curve=[];stored={}
for threshold in protocol['thresholds']:
 text=oldtext.copy();code=oldcode.copy();changed=[]
 for q,r in merged.items():
  i=qi[q];s='\n'.join(l['text'] for l in r['ocr']['lines'] if l['confidence']>=threshold)
  if not s.strip():continue
  text[i]=zscore(textscore(s));code[i],matches=bonus(s);changed.append(q)
  audit.append(dict(query=q,threshold=threshold,text=s,codes=sorted(codes(s)),matches=matches))
 stored[threshold]=(text,code)
 for weight in protocol['weights']:
  for code_update in protocol['code_update']:
   score=fixed+weight*.5*(text-oldtext)
   if code_update:score=score+weight*(code-oldcode)
   rr,_,_=rk.all(score);vm=metrics(rr,c,scopes['validation'],base);name=f'confidence{threshold:g}/weight{weight:g}/code{int(code_update)}'
   choices.append(dict(name=name,threshold=threshold,weight=weight,code_update=code_update,validation=vm))
   for scope,ix in scopes.items():curve.append(dict(method=name,threshold=threshold,weight=weight,code_update=code_update,scope=scope,**metrics(rr,c,ix,base)))
best=max(choices,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['weight'],-x['threshold'],-int(x['code_update'])))
text,code=stored[best['threshold']];score=fixed+best['weight']*.5*(text-oldtext)
if best['code_update']:score=score+best['weight']*(code-oldcode)
rr,ww,_=rk.all(score);np.save(D/'selected_ocr_scores.npy',score)
rows=[dict(query=q,baseline_rank=int(base[i]),selected_rank=int(rr[i]),winner=ww[i],triggered=q in expected) for i,q in enumerate(qids)]
write(D/'SELECTIVE_OCR_SUMMARY.json',dict(status='complete',selected=best,metrics={s:metrics(rr,c,ix,base) for s,ix in scopes.items()},repairs=[qids[i] for i in range(len(c)) if base[i]>1 and rr[i]==1],harms=[qids[i] for i in range(len(c)) if base[i]==1 and rr[i]>1],coverage=len(merged),checks=dict(original_text_reconstruction_max_error=maxtext,original_code_reconstruction_max_error=maxcode,untriggered_rows_unchanged=bool(np.array_equal(score[[i for i,q in enumerate(qids) if q not in expected]],fixed[[i for i,q in enumerate(qids) if q not in expected]]))),runtime=protocol['runtime']))
write(D/'SELECTIVE_OCR_MATCHES.json',audit);csvwrite(D/'selective_ocr_curve.csv',curve);csvwrite(D/'selective_ocr_cases1117.csv',rows)
print('OCR EVALUATION DONE',json.dumps(read(D/'SELECTIVE_OCR_SUMMARY.json')),flush=True)
