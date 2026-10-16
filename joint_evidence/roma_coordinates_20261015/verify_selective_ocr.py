"""Independent selected-score aggregation and SHA-deduplicated ranking replay."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,re,math
import numpy as np
import torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin')
sys.path.insert(0,str(P))
from experiment_utils import read,write,zscore
torch.set_num_threads(2)
c=read(P/'cohort.json');qids=[x['query'] for x in c]
rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
ids=rb['ids'];sha=rb['sha256'];ri={r:i for i,r in enumerate(ids)}
ro=read(R/'cache/ocr/references.json');qo=read(R/'cache/ocr/photos.json')
boxes=read(Path(read(P/'SUBJECT_REGION_POLICY.json')['remote_source']))['records']
s=read(D/'SELECTIVE_OCR_SUMMARY.json');chosen=s['selected'];w=chosen['weight']
fixed=np.load(D/'fixed1051_scores.npy');selected=np.load(D/'selected_ocr_scores.npy')
oldtext=np.load(V/'production_numeric_z.npy');oldcode=np.load(D/'code_bonus.npy')
audit={x['query']:x for x in read(D/'SELECTIVE_OCR_MATCHES.json') if x['threshold']==chosen['threshold']}
post=defaultdict(list);tok=lambda t:re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+',t.lower(),re.UNICODE)
for j,r in enumerate(ids):
 for word in set(tok(ro[r]['text'])):post[word].append(j)
newtext=oldtext.copy();newcode=oldcode.copy()
for i,q in enumerate(qids):
 if q not in audit:continue
 a=audit[q];raw=np.zeros(len(ids),dtype=np.float32)
 for word,count in Counter(tok(a['text'])).items():
  ix=post.get(word,[])
  if ix:raw[ix]+=count*np.full(len(ix),math.log(1+(len(ids)-len(ix)+.5)/(len(ix)+.5)),dtype=np.float32)
 newtext[i]=zscore(raw)
 # Reuse the recorded matching decisions; independently replay their max/sum reduction.
 percode={}
 for match in a['matches']:
  v=percode.setdefault(match['query_code'],np.zeros(len(ids)))
  ix=[ri[r] for r in match['references']]
  value=math.log(1+(len(ids)-len(ix)+.5)/(len(ix)+.5))/math.log(len(ids)+1)
  v[ix]=np.maximum(v[ix],value)
 newcode[i]=sum(percode.values(),start=np.zeros(len(ids)))
dt=.5*w*(newtext-oldtext);dc=w*(newcode-oldcode) if chosen['code_update'] else np.zeros_like(fixed)
replay=fixed+dt+dc;maxerr=float(np.max(np.abs(replay-selected)));assert maxerr<1e-12,maxerr

def rank(score,i):
 seen=set();n=0;positive=set(c[i]['positive_sha256'])
 for j in sorted(range(len(ids)),key=lambda j:(-float(score[j]),j)):
  if sha[j] in seen:continue
  seen.add(sha[j]);n+=1
  if sha[j] in positive:return n
 raise AssertionError('No positive in gallery')

import csv
with (D/'selective_ocr_cases1117.csv').open(encoding='utf-8-sig') as f:cases=list(csv.DictReader(f))
assert [x['query'] for x in cases]==qids
diagnostics=[];repairs=[];harms=[]
for i,q in enumerate(qids):
 br=rank(fixed[i],i);nr=rank(selected[i],i)
 assert br==int(cases[i]['baseline_rank']) and nr==int(cases[i]['selected_rank'])
 assert ids[int(np.argmax(selected[i]))]==cases[i]['winner']
 if br>1 and nr==1:repairs.append(q)
 elif br==1 and nr>1:harms.append(q)
 else:continue
 positive=[j for j in range(len(ids)) if sha[j] in set(c[i]['positive_sha256'])]
 p=min(positive,key=lambda j:(-float(selected[i,j]),j))
 n=int(np.argmax(fixed[i] if nr==1 else selected[i]))
 assert sha[n] not in set(c[i]['positive_sha256'])
 x,y,X,Y=boxes[q]['bbox'];oldsubject='\n'.join(l['text'] for l in qo[q]['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y)
 matches=[dict(query_code=m['query_code'],reference_code=m['reference_code'],positive=ids[p] in m['references'],negative=ids[n] in m['references'],gallery_rows=len(m['references'])) for m in audit.get(q,{}).get('matches',[]) if ids[p] in m['references'] or ids[n] in m['references']]
 diagnostics.append(dict(query=q,baseline_rank=br,selected_rank=nr,positive=ids[p],negative=ids[n],baseline_pair_margin=float(fixed[i,p]-fixed[i,n]),text_delta_margin=float(dt[i,p]-dt[i,n]),code_delta_margin=float(dc[i,p]-dc[i,n]),selected_pair_margin=float(selected[i,p]-selected[i,n]),text_update_only_rank=rank(fixed[i]+dt[i],i),code_update_only_rank=rank(fixed[i]+dc[i],i),old_subject_text=oldsubject,new_subject_text=audit.get(q,{}).get('text',''),relevant_code_matches=matches))
assert repairs==s['repairs'] and harms==s['harms']
write(D/'SELECTIVE_OCR_INDEPENDENT_VERIFICATION.json',dict(status='passed',queries=1117,selected_score_max_error=maxerr,independent_sha_rank_checks=1117,repairs=repairs,harms=harms,changed_case_diagnostics=diagnostics,scope='Independent word-score and code-bonus aggregation and scalar full-gallery SHA-deduplicated rank replay. Code string matching decisions are reused, not independently verified; OCR transcription truth is not inferred from retrieval success. Counterfactual text/code-only ranks are diagnostic, not per-case routing.'))
print(json.dumps(dict(status='passed',selected_score_max_error=maxerr,repairs=repairs,harms=harms,diagnostics=diagnostics)),flush=True)
