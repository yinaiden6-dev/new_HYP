"""Independent stable-sort rank validation of the fixed weight1 automatic control."""
from pathlib import Path
from collections import defaultdict
import json,csv,sys,math,hashlib
import numpy as np,torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,metrics
torch.set_num_threads(2);c=read(P/'cohort.json');ids0=torch.load(R/'cache/scores/colnomic7b/image.pt',map_location='cpu',mmap=True,weights_only=True);ids=ids0['reference_ids'];sha=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)['sha256'];ri={r:i for i,r in enumerate(ids)};qi={r['query']:i for i,r in enumerate(c)}
score=np.load(V/'incremental_joint_scores.npy');bonus=np.zeros_like(score);matches=read(D/'automatic_matches.json');perq={}
for r in matches:
 if r['mode']!='one_edit_min8':continue
 perq[r['query']]=r['matches'];groups=defaultdict(list)
 for p in r['matches']:groups[p['query_code']].append(p)
 for code,pp in groups.items():
  values=np.zeros(len(ids))
  for p in pp:
   js=[ri[x] for x in p['references']];n=len(js);idf=math.log(1+(len(ids)-n+.5)/(n+.5))/math.log(len(ids)+1);values[js]=np.maximum(values[js],idf)
  bonus[qi[r['query']]]+=values
with (D/'weighted_automatic_cases.csv').open(encoding='utf-8-sig') as f:expected={r['query']:int(r['rank']) for r in csv.DictReader(f) if r['mode']=='one_edit_min8' and float(r['weight'])==1}
rows=[];ranks=[];base=[];exclude=set(read(D/'protocol.json')['excluded_queries'])
for i,r in enumerate(c):
 def independent(values):
  seen=set();rank=0;positive=None;negative=None;prank=None
  for j in np.argsort(-values,kind='stable'):
   if sha[j] in seen:continue
   seen.add(sha[j]);rank+=1
   if sha[j] in r['positive_sha256']:
    if positive is None:positive=int(j);prank=rank
   elif negative is None:negative=int(j)
   if positive is not None and negative is not None:return prank,positive,negative
  raise AssertionError(r['query'])
 old,_,_=independent(score[i]);rank,p,n=independent(score[i]+bonus[i]);assert rank==expected[r['query']]
 ranks.append(rank);base.append(old);evidence=[]
 for pair in perq.get(r['query'],[]):
  ps=ids[p] in pair['references'];ns=ids[n] in pair['references']
  if ps!=ns:evidence.append(dict(query_code=pair['query_code'],reference_code=pair['reference_code'],favors='positive' if ps else 'negative'))
 rows.append(dict(query=r['query'],split=r['split'],excluded=r['query'] in exclude,baseline_rank=old,rank=rank,positive=ids[p],negative=ids[n],raw_margin=float(score[i,p]-score[i,n]),code_margin=float(bonus[i,p]-bonus[i,n]),final_margin=float(score[i,p]+bonus[i,p]-score[i,n]-bonus[i,n]),code_evidence=json.dumps(evidence)))
csvwrite(D/'weighted_fixed1_cases1117.csv',rows);ranks=np.array(ranks);base=np.array(base)
scopes={'all':np.arange(len(c)),'retained_queries':np.array([i for i,r in enumerate(c) if r['query'] not in exclude]),'test':np.array([i for i,r in enumerate(c) if r['split']=='test']),'retained_test':np.array([i for i,r in enumerate(c) if r['split']=='test' and r['query'] not in exclude])}
write(D/'IDENTIFIER_VALIDATION.json',dict(status='passed',independent_stable_sort_rank_checks=1117,formula='original final1041 scores + IDF-weighted one-edit-min8 code evidence, fixed weight1',selected_by_validation=False,fullscore_sha256=hashlib.sha256((V/'incremental_joint_scores.npy').read_bytes()).hexdigest(),matches_sha256=hashlib.sha256((D/'automatic_matches.json').read_bytes()).hexdigest(),scopes={k:metrics(ranks,c,ix,base) for k,ix in scopes.items()},repairs=[r for r in rows if r['baseline_rank']>1 and r['rank']==1],harms=[r for r in rows if r['baseline_rank']==1 and r['rank']>1]))
print('VALIDATED',int((ranks==1).sum()),flush=True)
