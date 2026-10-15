"""Predeclared global OCR robustness controls on cached full-gallery scores.

No manual codes, case labels or diagnostic positives enter inference. The fixed
historical validation split selects one arm before test metrics are inspected.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, math, re, sys, hashlib
import numpy as np
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1')
V=P/'background_causal_v1/joint_image_ocr_v1';D=V/'remaining54_v1';A=D/'roma_coordinates';O=D/'remaining49_v1/robustness'
sys.path.insert(0,str(P))
from experiment_utils import Ranking,zscore,metrics,read,write
def gtins(text):
 out=set()
 for line in text.splitlines():
  for match in re.finditer(r'(?<!\d)\d[\d\s\-\"\'.,]*\d(?!\d)',line):
   digits=re.sub(r'\D','',match.group())
   if len(digits) not in (12,13,14) or len(set(digits))<3:continue
   body=digits[:-1][::-1];check=(-sum(int(c)*(3 if k%2==0 else 1)for k,c in enumerate(body)))%10
   if check==int(digits[-1]):out.add(digits.zfill(14))
 return sorted(out)
def run():
 O.mkdir(parents=True,exist_ok=True)
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),n=1117,
  purpose='Test whether rare/sparse lexical OCR terms and short numeric aliases dominate useful image/code signals.',
  text_cap=[None,2.,1.,.5],image_direct_fraction=[0.,.5,1.],code_modes=['current','min8','min8_plus_exact_gtin'],
  rule='Cap each query term IDF*TF / original query score SD before the original 0.5 text weight; do not renormalize capped terms. Centering constants do not affect ranks. min8 keeps code query strings with at least 8 alphanumeric characters, excluding numeric strings suffixed LBS/IBS/1BS/BS/MG/MCG/ML/KG/OZ/LB. GTIN: only 12/13/14 digit automatic OCR strings passing checksum, >=3 distinct digits; zero-pad to14; exact gallery match; IDF bonus with max across query codes.',
  image='Convex combination of standardized existing mixed-image score and standardized repaired-image score.',
  selection='Validation macro Top1, then macro MRR, then fewer modifications, then declaration order. Baseline included. Freeze selected arm before reporting test metrics.',
  limits='Hypotheses informed by inspected failures. Historical validation/test reused; exploratory retrospective audit, not a fresh blind test.',
  no_manual_values=True,no_sam=True,no_truth_in_candidates=True)
 write(O/'PROTOCOL.json',protocol)
 cohort=read(P/'cohort.json');qids=[r['query']for r in cohort];qi={q:i for i,q in enumerate(qids)}
 audit=read(D/'remaining49_v1/SCORE_AUDIT.json');viewer={r['query']:r for r in read(P/'release/joint_evidence/CASE_VIEWER_DATA.json')['rows']}
 import csv,torch
 torch.set_num_threads(2)
 with open('/mnt/lsdf_zyin/dataset/lists/references.csv')as f:refs=sorted(list(csv.DictReader(f)),key=lambda r:int(r['row']))
 ids=[r['reference_id']for r in refs];ri={r:i for i,r in enumerate(ids)};sha=[r['sha256']for r in refs];rk=Ranking(ids,sha,cohort)
 ro=read('/mnt/lsdf_zyin/cache/ocr/references.json');qo=read('/mnt/lsdf_zyin/cache/ocr/photos.json')
 pol=read(P/'SUBJECT_REGION_POLICY.json');assert pol['selection']=='reviewed_human_boxes_only' and not pol['sam_allowed'];boxes=read(pol['remote_source'])['records']
 oldtexts=[]
 for q in qids:
  b=boxes[q];assert b['source']=='human' and b['status']=='confirmed';x,y,X,Y=b['bbox']
  oldtexts.append('\n'.join(l['text']for l in qo[q]['lines']if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y))
 newrows={r['query']:r for r in read(D/'SELECTIVE_OCR_MATCHES.json')if r['threshold']==0}
 oldmatches={r['query']:r['matches']for r in read(V/'identifier_layout_v1/automatic_matches.json')if r['mode']=='one_edit_min8'}
 newtexts=[newrows[q]['text']if q in newrows and newrows[q]['text'].strip()else oldtexts[i]for i,q in enumerate(qids)]
 tok=lambda s:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',s.lower(),flags=re.UNICODE)
 post=defaultdict(list)
 for j,r in enumerate(ids):
  for w in set(tok(ro[r]['text'])):post[w].append(j)
 weights={w:(np.array(js),np.float32(math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))))for w,js in post.items()}
 oldz=np.load(V/'production_numeric_z.npy');oldraw=np.load(P/'text_scores.npz')['numeric']
 shape=oldz.shape;texts={cap:np.zeros(shape)for cap in protocol['text_cap']};code8=np.zeros(shape);gtin=np.zeros(shape);codescan=[];maxerr=0.
 refgtin=defaultdict(list)
 for r in ids:
  for c in gtins(ro[r]['text']):refgtin[c].append(r)
 def code_matrix(matches):
  per=defaultdict(dict)
  for m in matches:
   clean=re.sub(r'[^A-Za-z0-9]','',m['query_code']).upper()
   if len(clean)<8 or re.fullmatch(r'\d+(?:LBS|IBS|1BS|BS|MG|MCG|ML|KG|OZ|LB)',clean):continue
   js=m['references'];idf=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1)
   for r in js:per[m['query_code']][ri[r]]=max(per[m['query_code']].get(ri[r],0),idf)
  out=np.zeros(len(ids))
  for ps in per.values():
   for j,val in ps.items():out[j]+=val
  return out
 for i,q in enumerate(qids):
  hasnew=q in newrows and bool(newrows[q]['text'].strip())
  for kind,txt,factor in [('old',oldtexts[i],.375 if hasnew else .5),('new',newtexts[i],.125 if hasnew else 0.)]:
   if not factor:continue
   counts=Counter(tok(txt));raw=np.zeros(len(ids),np.float32)
   for w,n in counts.items():
    if w in weights:js,idf=weights[w];raw[js]+=n*idf
   if kind=='old':maxerr=max(maxerr,float(np.max(abs(raw-oldraw[i]))))
   sd=raw.std(dtype=float)
   texts[None][i]+=factor*(oldz[i]if kind=='old'else zscore(raw))
   for cap in (2.,1.,.5):
    if sd<=1e-5:texts[cap][i]+=factor*(oldz[i]if kind=='old'else zscore(raw));continue
    for w,n in counts.items():
     if w in weights:js,idf=weights[w];texts[cap][i,js]+=factor*min(n*float(idf)/sd,cap)
  oldc=code_matrix(oldmatches.get(q,[]));newc=code_matrix(newrows[q]['matches'])if hasnew else oldc
  code8[i]=.75*oldc+.25*newc
  qcs=sorted(set(gtins(oldtexts[i])+gtins(newtexts[i])));hits=[]
  for c in qcs:
   js=refgtin.get(c,[])
   if js:
    val=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1)
    ix=[ri[r]for r in js];gtin[i,ix]=np.maximum(gtin[i,ix],val);hits.append(dict(code=c,references=js,bonus=val))
  codescan.append(dict(query=q,old=gtins(oldtexts[i]),new=gtins(newtexts[i]),matches=hits))
 assert maxerr<3e-4,maxerr
 inc=np.load(V/'incremental_image_scores.npy');repair=np.load(A/'image_selected_scores.npy');oldnew=np.load(V/'selected_image_scores.npy');mix=.5*zscore(inc+.4*(repair-oldnew));direct=.5*zscore(repair)
 final=np.load(A/'repaired_image_repaired_ocr_scores.npy');currentcode=final-mix-texts[None]
 baseline,_,_=rk.all(final);assert (baseline==1).sum()==1058
 scopes={'all':np.arange(len(cohort)),'validation':np.array([i for i,r in enumerate(cohort)if r['split']=='val']),'test':np.array([i for i,r in enumerate(cohort)if r['split']=='test']),'retained':np.array([i for i,q in enumerate(qids)if not viewer[q]['query_face_excluded']]),'target49':np.array([qi[q]for q in audit['target49']]),'target18':np.array([qi[q]for q in audit['target18']])}
 modes={'current':currentcode,'min8':code8,'min8_plus_exact_gtin':code8+gtin}
 # Freeze ranking selection using validation only; held-out labels not used here.
 grid=[];best=None;winner=None;armranks={};armspecs={}
 for cap in protocol['text_cap']:
  for alpha in protocol['image_direct_fraction']:
   for mode in protocol['code_modes']:
    name=f'cap_{cap}_direct_{alpha}_code_{mode}';score=(1-alpha)*mix+alpha*direct+texts[cap]+modes[mode]
    ranks,winners,margins=rk.all(score);val=metrics(ranks,cohort,scopes['validation'],baseline)
    mods=int(cap is not None)+int(alpha>0)+int(mode!='current');key=(val['macro_accuracy'],val['macro_mrr'],-mods)
    spec=dict(name=name,cap=cap,alpha=alpha,code_mode=mode,validation=val);grid.append(spec);armranks[name]=ranks;armspecs[name]=spec
    if best is None or key>best:best=key;winner=name
    print('VAL',name,val['correct'],round(val['macro_accuracy'],6),flush=True)
 selected=armspecs[winner];write(O/'SELECTION.json',dict(status='frozen_before_test_summary',selected=selected,key=best,validation_grid=grid))
 results={}
 for name,ranks in armranks.items():results[name]={'spec':armspecs[name],'metrics':{k:metrics(ranks,cohort,ix,baseline)for k,ix in scopes.items()}}
 cap=selected['cap'];alpha=selected['alpha'];mode=selected['code_mode'];score=(1-alpha)*mix+alpha*direct+texts[cap]+modes[mode]
 sr,sw,sm=rk.all(score);np.save(O/'selected_scores.npy',score)
 # Independent stable full-gallery traversal: all1117, SHA duplicate handling.
 for i,s in enumerate(score):
  seen=set();rank=0
  for j in np.argsort(-s,kind='stable'):
   if sha[j]in seen:continue
   seen.add(sha[j]);rank+=1
   if sha[j]in cohort[i]['positive_sha256']:break
  assert rank==sr[i],(qids[i],rank,sr[i])
 cases=[]
 for q in audit['target49']:
  i=qi[q];pair=rk.one(score[i],i);j,k=ri[pair['positive']],ri[pair['negative']]
  parts={'mixed_image':float((1-alpha)*(mix[i,j]-mix[i,k])),'direct_image':float(alpha*(direct[i,j]-direct[i,k])),'ordinary_text':float(texts[cap][i,j]-texts[cap][i,k]),'identifier':float(modes[mode][i,j]-modes[mode][i,k])}
  assert abs(sum(parts.values())-pair['margin'])<1e-7
  cases.append(dict(query=q,selected=pair,parts=parts,controls={name:int(rr[i])for name,rr in armranks.items()},codes=codescan[i]))
 write(O/'RESULTS.json',dict(status='verified',selected=selected,selection=results[winner]['metrics'],grid=results,query_order=qids,selected_ranks=sr.tolist(),selected_winners=sw.tolist(),baseline_ranks=baseline.tolist(),cases=cases,repairs=[qids[i]for i in np.where((baseline>1)&(sr==1))[0]],harms=[qids[i]for i in np.where((baseline==1)&(sr>1))[0]],raw_text_replay_max_error=maxerr,independent_rank_checks=len(cohort),gtin_query_matches=sum(bool(r['matches'])for r in codescan),gtin_reference_codes=len(refgtin),limits=protocol['limits']))
 write(O/'AUTOMATIC_GTIN.json',dict(reference_codes=dict(refgtin),queries=codescan))
 print('SELECTED',winner,json.dumps(results[winner]['metrics']),flush=True)
if __name__=='__main__':run()
