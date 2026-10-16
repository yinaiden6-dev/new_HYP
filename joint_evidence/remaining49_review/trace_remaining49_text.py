"""Decompose cached current-pair OCR scores without changing inference."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,re,math
import numpy as np
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1')
V=P/'background_causal_v1/joint_image_ocr_v1';D=V/'remaining54_v1';O=D/'remaining49_v1'
read=lambda p:json.loads(Path(p).read_text())
a=read(O/'SCORE_AUDIT.json');cohort=read(P/'cohort.json');qi={r['query']:i for i,r in enumerate(cohort)}
ro=read('/mnt/lsdf_zyin/cache/ocr/references.json');ids=sorted(ro,key=lambda x:int(x.split('_')[-1]));assert len(ids)==5405
tok=lambda s:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',s.lower(),flags=re.UNICODE)
docs=[set(tok(ro[r]['text']))for r in ids];ri={r:i for i,r in enumerate(ids)};post=defaultdict(list)
for j,d in enumerate(docs):
 for w in d:post[w].append(j)
weights={w:(np.array(js),np.float32(math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))))for w,js in post.items()}
oldz=np.load(V/'production_numeric_z.npy');oldcode=np.load(D/'code_bonus.npy');oldraw=np.load(P/'text_scores.npz')['numeric']
oldmatches={r['query']:r for r in read(V/'identifier_layout_v1/automatic_matches.json')if r['mode']=='one_edit_min8'}
out=[]
for c in a['cases']:
 q=c['query'];i=qi[q];p,n=ri[c['positive']],ri[c['negative']];new=c['new_ocr'];use=bool(new and new['text'].strip());terms=defaultdict(lambda:dict(old=0.,new=0.,old_count=0,new_count=0))
 checks={};fallback=0.
 for kind,txt,factor in [('old','\n'.join(l['text']for l in c['old_subject_ocr']),.375 if use else .5),('new',new['text']if use else '',.125 if use else 0.)]:
  counts=Counter(tok(txt));raw=np.zeros(len(ids),np.float32)
  for w,count in counts.items():
   if w in weights:js,wt=weights[w];raw[js]+=count*wt
  sd=float(raw.std(dtype=float));lex=0.
  for w,count in counts.items():
   if w not in weights:continue
   sign=int(w in docs[p])-int(w in docs[n]);v=factor*count*float(weights[w][1])*sign/sd if sd>1e-5 else 0.
   if sign and factor:terms[w][kind]=v;terms[w][kind+'_count']=count;lex+=v
  actual=factor*(oldz[i,p]-oldz[i,n])if kind=='old'else factor*(float(raw[p])-float(raw[n]))/sd if sd>1e-5 else 0.
  residual=float(actual-lex);checks[kind]={'std':sd,'lexical_gap':lex,'actual_gap':float(actual),'rounding_or_fallback_residual':residual}
  if kind=='old':
   err=float(abs(raw-oldraw[i]).max());assert err<3e-4,(q,err);checks[kind]['raw_replay_max_error']=err
   if sd>1e-5:assert abs(residual)<3e-5,(q,residual)
   else:fallback=residual
 code_terms=[];code_sum=0.
 for kind,items,factor in [('old',oldmatches.get(q,{}).get('matches',[]),.75 if use else 1.),('new',new.get('matches',[])if use else [],.25 if use else 0.)]:
  groups=defaultdict(list)
  for m in items:groups[m['query_code']].append(m)
  gap=0.
  for qc,ms in groups.items():
   pg=ng=0.;sources=[]
   for m in ms:
    js=m['references'];idf=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1)
    if c['positive']in js:pg=max(pg,idf)
    if c['negative']in js:ng=max(ng,idf)
    if c['positive']in js or c['negative']in js:sources.append({'query_code':qc,'reference_code':m['reference_code'],'supports_positive':c['positive']in js,'supports_negative':c['negative']in js,'reference_count':len(js)})
   v=factor*(pg-ng);gap+=v
   if v:code_terms.append({'source':kind,'query_code':qc,'contribution':v,'matches':sources})
  if kind=='old':assert abs(gap-factor*(oldcode[i,p]-oldcode[i,n]))<1e-8
  code_sum+=gap
 assert abs(code_sum-c['parts']['code'])<1e-8,(q,code_sum,c['parts']['code'])
 termlist=[dict(word=w,total=v['old']+v['new'],**v)for w,v in terms.items()]
 textsum=sum(t['total']for t in termlist)+fallback
 assert abs(textsum-c['parts']['text'])<3e-5,(q,textsum,c['parts']['text'])
 out.append(dict(query=q,text_checks=checks,text_replay_error=textsum-c['parts']['text'],fallback=fallback,negative_terms=sorted([t for t in termlist if t['total']<0],key=lambda t:t['total']),positive_terms=sorted([t for t in termlist if t['total']>0],key=lambda t:-t['total']),code_terms=sorted(code_terms,key=lambda t:t['contribution']),code_replay_error=code_sum-c['parts']['code']))
(O/'TEXT_TRACE.json').write_text(json.dumps({'status':'passed','cases':out,'code_checks':len(out),'max_text_replay_error':max(abs(x['text_replay_error'])for x in out)},ensure_ascii=False,indent=2)+'\n')
print('DONE',len(out),max(abs(x['text_replay_error'])for x in out),flush=True)
