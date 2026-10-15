"""Full 2x2 image-readout/OCR interface audit, including pure repaired image."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,csv,re,math
import numpy as np
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1');V=P/'background_causal_v1/joint_image_ocr_v1';D=V/'remaining54_v1';A=D/'roma_coordinates';O=D/'remaining49_v1/robustness'
sys.path.insert(0,str(P))
from experiment_utils import read,write,Ranking,zscore,metrics
def main():
 c=read(P/'cohort.json');qids=[r['query']for r in c];qi={q:i for i,q in enumerate(qids)}
 with open('/mnt/lsdf_zyin/dataset/lists/references.csv')as f:refs=sorted(list(csv.DictReader(f)),key=lambda r:int(r['row']))
 ids=[r['reference_id']for r in refs];ri={r:i for i,r in enumerate(ids)};sha=[r['sha256']for r in refs];rk=Ranking(ids,sha,c)
 audit=read(D/'remaining49_v1/SCORE_AUDIT.json');ac={r['query']:r for r in audit['cases']};final=np.load(A/'repaired_image_repaired_ocr_scores.npy');repair=np.load(A/'image_selected_scores.npy');inc=np.load(V/'incremental_image_scores.npy');oldnew=np.load(V/'selected_image_scores.npy');mix=.5*zscore(inc+.4*(repair-oldnew));direct=.5*zscore(repair);ocr=final-mix
 arms={'repaired_image_only':direct,'mixed_image_only':mix,'repaired_image_ocr':direct+ocr,'mixed_image_ocr':final};br,_,_=rk.all(final)
 viewer={r['query']:r for r in read(P/'release/joint_evidence/CASE_VIEWER_DATA.json')['rows']}
 scopes={'all':np.arange(len(c)),'validation':np.array([i for i,r in enumerate(c)if r['split']=='val']),'test':np.array([i for i,r in enumerate(c)if r['split']=='test']),'retained':np.array([i for i,q in enumerate(qids)if not viewer[q]['query_face_excluded']]),'target49':np.array([qi[q]for q in audit['target49']]),'target18':np.array([qi[q]for q in audit['target18']])}
 results={};ranks={}
 for name,s in arms.items():
  r,w,m=rk.all(s);results[name]={'metrics':{k:metrics(r,c,v,br)for k,v in scopes.items()}};ranks[name]=dict(ranks=r.tolist(),winners=w.tolist())
 ro=read('/mnt/lsdf_zyin/cache/ocr/references.json');tok=lambda s:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',s.lower(),flags=re.UNICODE);docs=[set(tok(ro[r]['text']))for r in ids];post=defaultdict(list)
 for j,d in enumerate(docs):
  for t in d:post[t].append(j)
 wt={w:(np.array(js),np.float32(math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))))for w,js in post.items()}
 cases=[]
 for q in audit['target49']:
  i=qi[q];pairs={k:rk.one(s[i],i)for k,s in arms.items()};p=pairs['repaired_image_ocr'];j,k=ri[p['positive']],ri[p['negative']];textterms=defaultdict(float);src=ac[q];new=src['new_ocr'];use=bool(new and new['text'].strip())
  for txt,factor in [('\n'.join(l['text']for l in src['old_subject_ocr']),.375 if use else .5),(new['text']if use else '',.125 if use else 0.)]:
   raw=np.zeros(len(ids),np.float32);counts=Counter(tok(txt))
   for w,n in counts.items():
    if w in wt:ix,idf=wt[w];raw[ix]+=n*idf
   sd=raw.std(dtype=float)
   if sd>1e-5:
    for w,n in counts.items():
     if w in wt:textterms[w]+=factor*n*float(wt[w][1])*(int(w in docs[j])-int(w in docs[k]))/sd
  parts=dict(repaired_image=float(direct[i,j]-direct[i,k]),ocr=float(ocr[i,j]-ocr[i,k]));assert abs(sum(parts.values())-p['margin'])<1e-7
  type='both_single_additions_fail'if pairs['mixed_image_only']['rank']>1 and p['rank']>1 else'legacy_image_only'if pairs['mixed_image_only']['rank']>1 else'ocr_only'if p['rank']>1 else'joint_interaction'
  cases.append(dict(query=q,target18=q in audit['target18'],pairs=pairs,mechanism_when_repaired_image_correct=type if pairs['repaired_image_only']['rank']==1 else None,direct_image_ocr_pair_parts=parts,direct_pair_negative_terms=sorted([dict(word=w,contribution=v)for w,v in textterms.items()if v<0],key=lambda r:r['contribution'])[:10]))
 write(O/'INTERFACE_FACTORIAL.json',dict(status='verified',protocol='Both image sources crossed with OCR off/on; fixed existing scores, all1117 full-gallery ranks. Interventions diagnose score interfaces, not whether layout itself is encoded.',grid=results,query_order=qids,rankrows=ranks,cases=cases,target18_types=dict(Counter(x['mechanism_when_repaired_image_correct']for x in cases if x['target18'])),primary_unchanged=1058))
 print('INTERFACE',[(k,v['metrics']['all']['correct'])for k,v in results.items()]);print('18',dict(Counter(x['mechanism_when_repaired_image_correct']for x in cases if x['target18'])));print('C576',[x for x in cases if x['query']=='photo_c576c0c30f74cd4f'])
if __name__=='__main__':main()
