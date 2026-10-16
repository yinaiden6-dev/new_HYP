"""Full-gallery barcode evidence ablation with validation-only selection."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import sys,json,csv,math
import numpy as np
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1');D=P/'background_causal_v1/joint_image_ocr_v1/remaining54_v1';O=D/'remaining49_v1/robustness'
sys.path.insert(0,str(P))
from experiment_utils import read,write,Ranking,metrics
from robust_remaining49 import gtins
def main():
 protocol=dict(time=datetime.now(timezone.utc).isoformat(),sources=['decoded_query','decoded_or_ocr_query'],weights=[0.,.5,1.,2.,4.],reference='all exact valid GTINs from decoded pixels OR cached automatic OCR; SHA aliases retained for evaluation',score='baseline1058 plus max normalized IDF over matching exact GTINs; no query labels or candidate insertion',selection='validation macro accuracy, macro MRR, then smaller weight, then decoded-only; baseline included',limits='Exploratory historical data; no manual barcode transcription.')
 write(O/'BARCODE_SCORE_PROTOCOL.json',protocol)
 d=read(O/'BARCODE_DECODE.json');assert d['status']=='complete' and len(d['records'])==6447
 c=read(P/'cohort.json');qids=[r['query']for r in c];qi={q:i for i,q in enumerate(qids)}
 with open('/mnt/lsdf_zyin/dataset/lists/references.csv')as f:refs=sorted(list(csv.DictReader(f)),key=lambda r:int(r['row']))
 ids=[r['reference_id']for r in refs];ri={r:i for i,r in enumerate(ids)};sha=[r['sha256']for r in refs];rk=Ranking(ids,sha,c)
 auto=read(O/'AUTOMATIC_GTIN.json');post=defaultdict(set,{k:set(v)for k,v in auto['reference_codes'].items()});oq={r['query']:set(r['old']+r['new'])for r in auto['queries']};dq={};decoded_byid={r['id']:r for r in d['records']}
 for r in d['records']:
  codes=set(x for z in r.get('codes',[])for x in gtins(z['text']))
  if r['kind']=='query':dq[r['id']]=codes
  else:
   for code in codes:post[code].update(r['aliases'])
 base=np.load(D/'roma_coordinates/repaired_image_repaired_ocr_scores.npy');br,_,_=rk.all(base);assert (br==1).sum()==1058
 viewer={r['query']:r for r in read(P/'release/joint_evidence/CASE_VIEWER_DATA.json')['rows']};audit=read(D/'remaining49_v1/SCORE_AUDIT.json')
 scopes={'all':np.arange(len(c)),'validation':np.array([i for i,r in enumerate(c)if r['split']=='val']),'test':np.array([i for i,r in enumerate(c)if r['split']=='test']),'retained':np.array([i for i,q in enumerate(qids)if not viewer[q]['query_face_excluded']]),'target49':np.array([qi[q]for q in audit['target49']]),'target18':np.array([qi[q]for q in audit['target18']])}
 bonuses={};matches={};grid=[];ranksall={};winnersall={};best=None;selected=None
 for source in protocol['sources']:
  bonus=np.zeros(base.shape);ms=[]
  for i,q in enumerate(qids):
   codes=dq.get(q,set())if source=='decoded_query'else dq.get(q,set())|oq[q];hits=[]
   for code in sorted(codes):
    js=sorted(post.get(code,[]))
    if js:
     val=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1);ix=[ri[j]for j in js];bonus[i,ix]=np.maximum(bonus[i,ix],val);hits.append(dict(code=code,references=js,bonus=val))
   ms.append(dict(query=q,decoded=sorted(dq.get(q,set())),ocr=sorted(oq[q]),matches=hits))
  bonuses[source]=bonus;matches[source]=ms
  for weight in protocol['weights']:
   name=f'{source}_weight_{weight}';r,w,m=rk.all(base+weight*bonus);v=metrics(r,c,scopes['validation'],br);spec=dict(name=name,source=source,weight=weight,validation=v);grid.append(spec);ranksall[name]=r;winnersall[name]=w;key=(v['macro_accuracy'],v['macro_mrr'],-weight,int(source=='decoded_query'))
   if best is None or key>best:best=key;selected=spec
 write(O/'BARCODE_SELECTION.json',dict(selected=selected,grid=grid,selected_before_test_summary=True))
 results={x['name']:dict(spec=x,metrics={k:metrics(ranksall[x['name']],c,ix,br)for k,ix in scopes.items()})for x in grid}
 score=base+selected['weight']*bonuses[selected['source']];rr,ww,mm=rk.all(score);np.save(O/'barcode_selected_scores.npy',score)
 for i,s in enumerate(score):
  seen=set();rank=0
  for j in np.argsort(-s,kind='stable'):
   if sha[j]in seen:continue
   seen.add(sha[j]);rank+=1
   if sha[j]in c[i]['positive_sha256']:break
  assert rank==rr[i]
 cases=[]
 for q in audit['target49']:
  i=qi[q];cases.append(dict(query=q,decode=decoded_byid.get(q),matches={s:matches[s][i]for s in matches},selected=rk.one(score[i],i),controls={n:int(r[i])for n,r in ranksall.items()}))
 write(O/'BARCODE_RESULTS.json',dict(status='verified',selected=selected,selection=results[selected['name']]['metrics'],grid=results,cases=cases,query_order=qids,selected_ranks=rr.tolist(),selected_winners=ww.tolist(),repairs=[qids[i]for i in np.where((br>1)&(rr==1))[0]],harms=[qids[i]for i in np.where((br==1)&(rr>1))[0]],independent_rank_checks=len(c),decoded_queries=sum(bool(v)for v in dq.values()),decoded_references=sum(r['kind']=='reference'and bool(r.get('codes'))for r in d['records']),decode_errors=[r for r in d['records']if r['status']!='ok'],total_decoded_images=len(d['records']),matches=matches[selected['source']],limits=protocol['limits']))
 write(O/'BARCODE_RANKS1117.json',dict(queries=qids,methods={n:dict(ranks=r.tolist(),winners=winnersall[n].tolist())for n,r in ranksall.items()}))
 print('BARCODE_SELECTED',selected,results[selected['name']]['metrics'],flush=True)
if __name__=='__main__':main()
