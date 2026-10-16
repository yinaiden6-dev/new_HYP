"""Full-gallery identifier diagnostics, automatic controls, and query-only exclusion."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,sys,re,math,hashlib
import numpy as np,torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin')
sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,zscore,Ranking,metrics
torch.set_num_threads(2)
protocol=read(D/'protocol.json');exclude=set(protocol['excluded_queries'])
c=read(P/'cohort.json');qi={r['query']:i for i,r in enumerate(c)}
rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
ids=rb['ids'];sha=rb['sha256'];ri={r:i for i,r in enumerate(ids)};rk=Ranking(ids,sha,c)
joint=np.load(V/'incremental_joint_scores.npy');image=np.load(V/'incremental_image_scores.npy');zi=zscore(image)
baseline,_,_=rk.all(joint);assert (baseline==1).sum()==1041
with np.load(P/'text_scores.npz') as t:rawtext=t['numeric'].astype(float)
qo=read(R/'cache/ocr/photos.json');ro=read(R/'cache/ocr/references.json');policy=read(P/'SUBJECT_REGION_POLICY.json')
assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records']
qtexts=[]
for r in c:
 x,y,X,Y=boxes[r['query']]['bbox']
 qtexts.append('\n'.join(l['text'] for l in qo[r['query']]['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y))
def codes(text):
 result=set()
 for line in text.upper().splitlines():
  # Whitespace is only collapsed for lines composed entirely of a short code.
  candidates=re.findall(r'[A-Z0-9]+(?:[-./][A-Z0-9]+)*',line)
  if re.fullmatch(r'[A-Z0-9\s./-]{5,24}',line.strip()):candidates.append(line)
  for s in candidates:
   s=re.sub('[^A-Z0-9]','',s);digits=sum(ch.isdigit() for ch in s)
   if 5<=len(s)<=18 and digits>=5 and len(s)-digits<=4 and len(set(s))>1:result.add(s)
 return result
qcodes=[codes(t) for t in qtexts];rcodes=[codes(ro[r]['text']) for r in ids]
posts=defaultdict(set)
for j,cc in enumerate(rcodes):
 for code in cc:posts[code].add(j)
posts={s:js for s,js in posts.items() if len(js)<=len(ids)*.01}
deletions=defaultdict(set)
for s in posts:
 if len(s)>=7:
  deletions[s].add(s)
  for k in range(len(s)):deletions[s[:k]+s[k+1:]].add(s)
def one_edit(a,b):
 if abs(len(a)-len(b))>1:return False
 if len(a)==len(b):return sum(x!=y for x,y in zip(a,b))<=1
 if len(a)>len(b):a,b=b,a
 i=j=errors=0
 while i<len(a) and j<len(b):
  if a[i]==b[j]:i+=1;j+=1
  else:errors+=1;j+=1
  if errors>1:return False
 return True
def match(cc,mode):
 js=set();pairs=[]
 for s in sorted(cc):
  candidates={s} if s in posts else set()
  if mode=='one_edit_min8' and len(s)>=8:
   candidates.update(deletions.get(s,set()))
   for k in range(len(s)):candidates.update(deletions.get(s[:k]+s[k+1:],set()))
   candidates={t for t in candidates if min(len(s),len(t))>=8 and one_edit(s,t)}
  for t in sorted(candidates):
   js.update(posts[t]);pairs.append(dict(query_code=s,reference_code=t,references=[ids[j] for j in sorted(posts[t])]))
 return js,pairs
scopes={'all':np.arange(len(c)),'query_face_sufficient':np.array([i for i,r in enumerate(c) if r['query'] not in exclude]),
        'validation':np.array([i for i,r in enumerate(c) if r['split']=='val']),
        'validation_query_face_sufficient':np.array([i for i,r in enumerate(c) if r['split']=='val' and r['query'] not in exclude]),
        'test':np.array([i for i,r in enumerate(c) if r['split']=='test']),
        'test_query_face_sufficient':np.array([i for i,r in enumerate(c) if r['split']=='test' and r['query'] not in exclude])}
choices=[];bonuses={};matchrows=[]
for mode in protocol['automatic_code_match']['modes']:
 bonus=np.zeros(joint.shape,dtype=np.float32)
 for i,cc in enumerate(qcodes):
  js,pp=match(cc,mode)
  if js:bonus[i,list(js)]=1
  if pp:matchrows.append(dict(query=c[i]['query'],mode=mode,codes=sorted(cc),matches=pp))
 bonuses[mode]=bonus
 for w in protocol['automatic_code_match']['weights']:
  rr,_,_=rk.all(joint+w*bonus)
  choices.append(dict(mode=mode,weight=w,ranks=rr,validation=metrics(rr,c,scopes['validation_query_face_sufficient'],baseline)))
chosen=max(choices,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['weight'],x['mode']=='exact'))
write(D/'AUTOMATIC_SELECTION.json',{k:v for k,v in chosen.items() if k!='ranks'})
curve=[]
for arm in choices:
 for scope,ix in scopes.items():curve.append(dict(mode=arm['mode'],weight=arm['weight'],scope=scope,**metrics(arm['ranks'],c,ix,baseline)))
csvwrite(D/'automatic_curve.csv',curve);write(D/'automatic_matches.json',matchrows)
selected_scores=joint+chosen['weight']*bonuses[chosen['mode']]
perquery=[]
for i,r in enumerate(c):
 z=rk.one(selected_scores[i],i)
 perquery.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],excluded=r['query'] in exclude,baseline_rank=int(baseline[i]),rank=z['rank'],positive=z['positive'],negative=z['negative'],margin=z['margin'],code_candidates=' | '.join(sorted(qcodes[i]))))
csvwrite(D/'automatic_cases1117.csv',perquery)

# Diagnostic intervention: supply a verified query reading, never change reference OCR.
tok=lambda t:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',t.lower(),flags=re.UNICODE)
docs=[set(tok(ro[r]['text'])) for r in ids];words=defaultdict(list)
for j,dd in enumerate(docs):
 for w in dd:words[w].append(j)
weights={w:(np.array(js),math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))) for w,js in words.items()}
manual=[]
for q,code in protocol['identifier_diagnostics'].items():
 i=qi[q];z=rk.one(joint[i],i);p,n=ri[z['positive']],ri[z['negative']]
 norm=next(iter(codes(code)));already=norm in qcodes[i];add=np.zeros(len(ids))
 if not already:
  for w,f in Counter(tok(code)).items():
   if w in weights:ix,weight=weights[w];add[ix]+=f*weight
 newraw=rawtext[i]+add
 fresh=.5*(zi[i]+zscore(newraw))
 fixed=joint[i]+(.5*add/rawtext[i].std() if rawtext[i].std() else 0)
 targets,pp=match({norm},'exact');direct=np.zeros(len(ids));direct[list(targets)]=1
 interventions={}
 for label,s in [('verified_numeric_fresh',fresh),('verified_numeric_fixed',fixed)]:interventions[label]=rk.one(s,i)
 for w in protocol['automatic_code_match']['weights']:interventions['verified_whole_code_bonus_'+str(w)]=rk.one(joint[i]+w*direct,i)
 manual.append(dict(query=q,code=code,already_in_query_ocr=already,baseline=z,existing_query_codes=sorted(qcodes[i]),reference_positive_codes=sorted(rcodes[p]),reference_negative_codes=sorted(rcodes[n]),exact_reference_matches=pp,interventions=interventions,uses_verified_reading=True))
write(D/'MANUAL_READING_DIAGNOSTICS.json',manual)

# Same query exclusion for all configurations for which per-query rankings exist.
def cr(path):
 with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
methods={}
im=cr(V/'image_cases1117.csv')
for k in im[0]:
 if k.endswith('_rank'):methods['image/'+k[:-5]]=np.array([int(r[k]) for r in im])
old=cr(P/'release/all1117_attribution.csv')
for k in ['original_rank','center_presence_rank','numeric_rank','selected_image_none_rank','selected_original_fresh_rank','selected_original_fixed_rank','selected_center_presence_fresh_rank','selected_center_presence_fixed_rank','selected_numeric_fresh_rank','selected_numeric_fixed_rank']:
 methods['prior/'+k]=np.array([int(r[k]) for r in old])
cross=defaultdict(dict)
for r in cr(V/'ocr_joint_per_query.csv'):cross[r['image']+'/'+r['ocr']+'/'+r['normalization']][r['query']]=int(r['rank'])
for k,rr in cross.items():methods['cross/'+k]=np.array([rr[r['query']] for r in c])
methods['final_incremental']=baseline;methods['automatic_identifier_selected']=chosen['ranks']
summary=[]
for name,rr in methods.items():
 for scope,ix in scopes.items():summary.append(dict(method=name,scope=scope,**metrics(rr,c,ix)))
csvwrite(D/'query_exclusion_results.csv',summary)
write(D/'SUMMARY.json',dict(protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),excluded_queries=sorted(exclude),gallery_rows=len(ids),gallery_unique=len(set(sha)),
 baseline={scope:metrics(baseline,c,ix) for scope,ix in scopes.items()},automatic_selected={k:v for k,v in chosen.items() if k!='ranks'},
 automatic={scope:metrics(chosen['ranks'],c,ix,baseline) for scope,ix in scopes.items()},
 selected_repairs=[c[i]['query'] for i in np.flatnonzero((baseline>1)&(chosen['ranks']==1))],selected_harms=[c[i]['query'] for i in np.flatnonzero((baseline==1)&(chosen['ranks']>1))],
 manual_readings_not_used_by_automatic=True,test_labels_not_used_for_selection=True,matching_codes=len(posts),comparison_methods=len(methods)))
print(json.dumps(read(D/'SUMMARY.json')),flush=True)
