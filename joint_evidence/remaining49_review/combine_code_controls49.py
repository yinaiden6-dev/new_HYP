"""Fixed composition of prior old-code retention and new automatic GTIN evidence."""
from pathlib import Path
from collections import defaultdict
import sys,csv,math
import numpy as np
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1');D=P/'background_causal_v1/joint_image_ocr_v1/remaining54_v1';O=D/'remaining49_v1/robustness';A=D/'roma_coordinates'
sys.path.insert(0,str(P))
from experiment_utils import read,write,Ranking,metrics
from robust_remaining49 import gtins
def main():
 write(O/'COMPOSED_PROTOCOL.json',dict(type='fixed exploratory composition, not validation-selected',arms=['retained_old_codes','retained_old_codes_plus_decoded_weight1','retained_old_codes_plus_union_weight4'],n=1117,no_truth_in_inference=True,no_manual_strings=True,weights='Inherited from separately documented fixed controls; no tuning in this experiment.'))
 cohort=read(P/'cohort.json');qids=[r['query']for r in cohort];qi={q:i for i,q in enumerate(qids)}
 with open('/mnt/lsdf_zyin/dataset/lists/references.csv')as f:refs=sorted(list(csv.DictReader(f)),key=lambda r:int(r['row']))
 ids=[r['reference_id']for r in refs];ri={r:i for i,r in enumerate(ids)};sha=[r['sha256']for r in refs];rk=Ranking(ids,sha,cohort)
 auto=read(O/'AUTOMATIC_GTIN.json');post=defaultdict(set,{k:set(v)for k,v in auto['reference_codes'].items()});oq={r['query']:set(r['old']+r['new'])for r in auto['queries']};dq={}
 for r in read(O/'BARCODE_DECODE.json')['records']:
  codes=set(x for z in r.get('codes',[])for x in gtins(z['text']))
  if r['kind']=='query':dq[r['id']]=codes
  else:
   for code in codes:post[code].update(r['aliases'])
 prior=np.load(A/'repaired_image_repaired_ocr_scores.npy');retained=np.load(A/'repaired_image_retained_ocr_scores.npy');br,_,_=rk.all(prior)
 def bonus(union):
  out=np.zeros(prior.shape)
  for i,q in enumerate(qids):
   for code in dq.get(q,set())|(oq[q]if union else set()):
    js=sorted(post.get(code,[]))
    if js:
     val=math.log(1+(len(ids)-len(js)+.5)/(len(js)+.5))/math.log(len(ids)+1);ix=[ri[r]for r in js];out[i,ix]=np.maximum(out[i,ix],val)
  return out
 arms={'retained_old_codes':retained,'retained_old_codes_plus_decoded_weight1':retained+bonus(False),'retained_old_codes_plus_union_weight4':retained+4*bonus(True)}
 audit=read(D/'remaining49_v1/SCORE_AUDIT.json');viewer={r['query']:r for r in read(P/'release/joint_evidence/CASE_VIEWER_DATA.json')['rows']}
 scopes={'all':np.arange(len(cohort)),'validation':np.array([i for i,r in enumerate(cohort)if r['split']=='val']),'test':np.array([i for i,r in enumerate(cohort)if r['split']=='test']),'retained':np.array([i for i,q in enumerate(qids)if not viewer[q]['query_face_excluded']]),'target49':np.array([qi[q]for q in audit['target49']]),'target18':np.array([qi[q]for q in audit['target18']])}
 results={};ranks={}
 for name,score in arms.items():
  rr,ww,mm=rk.all(score)
  for i,s in enumerate(score):
   seen=set();rank=0
   for j in np.argsort(-s,kind='stable'):
    if sha[j]in seen:continue
    seen.add(sha[j]);rank+=1
    if sha[j]in cohort[i]['positive_sha256']:break
   assert rank==rr[i]
  results[name]={'metrics':{k:metrics(rr,cohort,v,br)for k,v in scopes.items()},'repairs':[qids[i]for i in np.where((br>1)&(rr==1))[0]],'harms':[qids[i]for i in np.where((br==1)&(rr>1))[0]]};ranks[name]=dict(ranks=rr.tolist(),winners=ww.tolist())
 write(O/'COMPOSED_RESULTS.json',dict(status='verified',grid=results,query_order=qids,rankrows=ranks,independent_rank_checks=3*len(cohort),primary_unchanged=1058,limits='Fixed exploratory controls; historical splits inspected, not new validation-selected method.'))
 print({k:v['metrics']for k,v in results.items()},flush=True)
if __name__=='__main__':main()
