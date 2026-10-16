"""Truth-selected OCR line deletion with same-count random controls, full-gallery ranks."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from collections import Counter,defaultdict
import re,json,math,time,hashlib
import numpy as np,torch
from experiment_utils import read,write,csvwrite,zscore,Ranking,metrics
B=Path(__file__).resolve().parent;ROOT=Path('/mnt/lsdf_zyin');start=time.time()
torch.set_num_threads(2)
cohort=read(B/'cohort.json');I=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',weights_only=True,map_location='cpu')
qids=I['query_ids'];rids=I['reference_ids'];image=I['scores'].numpy();zi=zscore(image)
rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',weights_only=True,map_location='cpu')
ranker=Ranking(rids,rb['sha256'],cohort);ri={r:i for i,r in enumerate(rids)}
qo=read(ROOT/'cache/ocr/photos.json');ro=read(ROOT/'cache/ocr/references.json')
boxes=read(ROOT/'results/colnomic_full_audit/20261013_resume/foreground_harms_review/source_snapshots/subject_boxes_after_user_review.json')['records']
fusion=torch.load(ROOT/'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt',map_location='cpu',weights_only=True)['scores'].numpy()
saved={'original':fusion,'center_presence':np.load(ROOT/'results/colnomic_multiobject_diagnosis/20261014/experiments/center_presence_all1117_scores.npz')['fusion'],'numeric':np.load(ROOT/'results/colnomic_multiobject_diagnosis/20261014/numeric_tokenization/scores.npz')['fusion']}
result=[];evidence=[];summaries={};textscores={};errors={}
for method in ['original','center_presence','numeric']:
 pattern=r'\d+(?:\.\d+)?|[^\W\d_]+' if method=='numeric' else r'[^\W_]+'
 def tok(t):return re.findall(pattern,t.lower(),flags=re.UNICODE)
 docs=[Counter(tok(ro[r]['text'])) for r in rids];dl=np.array([sum(d.values()) for d in docs]);avg=dl.mean();post=defaultdict(list)
 for j,d in enumerate(docs):
  for w,tf in d.items():post[w].append((j,tf))
 weights={}
 for w,v in post.items():
  js=np.array([a for a,b in v]);tf=np.array([b for a,b in v]);idf=math.log(1+(len(rids)-len(v)+.5)/(len(v)+.5))
  value=idf*tf*2.2/(tf+1.2*(.25+.75*dl[js]/avg)) if method=='original' else np.full(len(js),idf)
  weights[w]=(js,value.astype(np.float32))
 text=np.zeros_like(image);methodrows=[];maxerr=0.
 for i,q in enumerate(qids):
  lines=qo[q]['lines'];keep=list(range(len(lines)))
  if method!='original':
   x,y,X,Y=boxes[q]['bbox'];keep=[j for j,l in enumerate(lines) if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y]
  counters=[Counter(tok(lines[j]['text'])) for j in keep]
  # Exactly reproduce production sequential accumulation of query-frequency terms.
  total=Counter(tok(qo[q]['text'])) if method=='original' else Counter(tok('\n'.join(lines[j]['text'] for j in keep)))
  assert total==sum(counters,Counter()),(method,q,'OCR full text differs from line concatenation')
  for w,qf in total.items():
   if w in weights:js,val=weights[w];text[i,js]+=qf*val
  base=.5*(zi[i]+zscore(text[i])) if total else zi[i]
  err=float(np.abs(base-saved[method][i]).max());maxerr=max(maxerr,err);assert err<3e-5,(method,q,err)
  br=ranker.one(base,i);assert br['rank']==int(cohort[i][method+'_rank']),(q,method,br)
  p,n=ri[br['positive']],ri[br['negative']]
  ls=np.zeros((len(keep),len(rids)),np.float64)
  for j,terms in enumerate(counters):
   for w,qf in terms.items():
    if w in weights:js,val=weights[w];ls[j,js]+=qf*val
  # Production sums FP32 term totals, while the attribution sums FP64 line parts.
  # Bound accumulation drift relative to score scale, then rescore remaining terms exactly.
  assert np.allclose(ls.sum(0),text[i],atol=1e-4,rtol=5e-6),(method,q,float(np.abs(ls.sum(0)-text[i]).max()))
  gaps=ls[:,p]-ls[:,n];bad=np.flatnonzero(gaps< -1e-7)
  singles=np.array([int(np.argmin(gaps))]) if len(bad) else np.array([],dtype=int)
  for j,g in enumerate(gaps):
   evidence.append(dict(query=q,method=method,line=keep[j],text=lines[keep[j]]['text'],positive=br['positive'],negative=br['negative'],raw_pair_margin=float(g),selected_as_negative=bool(g< -1e-7),inside_subject=method!='original' or bool(boxes[q]['bbox'][0]<=(lines[keep[j]]['bbox'][0]+lines[keep[j]]['bbox'][2])/2<=boxes[q]['bbox'][2] and boxes[q]['bbox'][1]<=(lines[keep[j]]['bbox'][1]+lines[keep[j]]['bbox'][3])/2<=boxes[q]['bbox'][3])))
  variants=[('none',-1,np.array([],dtype=int)),('worst_one',-1,singles),('all_negative',-1,bad)]
  for target,selection in [('random_one',singles),('random_all',bad)]:
   for seed in range(10):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{q}/{method}/{target}/{seed}/20261014'.encode()).digest()[:8],'little'))
    variants.append((target,seed,rng.choice(len(keep),len(selection),replace=False)))
  for variant,seed,drop in variants:
   # Fixed-normalization contrast isolates contribution removal; an additive mean shift does not affect ranking.
   sd=float(text[i].std(dtype=np.float64));fixed=base-.5*ls[drop].sum(0)/sd if sd>0 else base
   drop_set=set(drop);remaining=sum((counters[j] for j in range(len(keep)) if j not in drop_set),Counter())
   new=np.zeros(len(rids),np.float32)
   for word,qf in remaining.items():
    if word in weights:js,val=weights[word];new[js]+=qf*val
   fresh=.5*(zi[i]+zscore(new)) if remaining else zi[i]
   for normalization,score in [('fixed',fixed),('fresh',fresh)]:
    rr=ranker.one(score,i)
    # New gallery challengers are retained, not merely the old negative candidate.
    rec=dict(query=q,package_id=cohort[i]['package_id'],split=cohort[i]['split'],background=cohort[i]['background'],method=method,variant=variant,seed=seed,normalization=normalization,deleted_lines=len(drop),baseline_rank=br['rank'],baseline_positive=br['positive'],baseline_negative=br['negative'],**rr,fixed_pair_margin=float(score[p]-score[n]),repair=br['rank']>1 and rr['rank']==1,harm=br['rank']==1 and rr['rank']>1)
    methodrows.append(rec)
  if (i+1)%150==0:print(method,i+1,'sec',round(time.time()-start),flush=True)
 textscores[method]=text;errors[method]=maxerr;result.extend(methodrows)
 for variant in ['none','worst_one','all_negative','random_one','random_all']:
  for norm in ['fixed','fresh']:
   seeds=range(10) if variant.startswith('random') else [-1]
   for seed in seeds:
    rs=[r for r in methodrows if r['variant']==variant and r['normalization']==norm and r['seed']==seed];assert len(rs)==1117
    ranks=np.array([r['rank'] for r in rs]);baseline=np.array([r['baseline_rank'] for r in rs])
    summaries[f'{method}/{variant}/{norm}/{seed}']=metrics(ranks,cohort,baseline=baseline)
 csvwrite(B/f'ocr_{method}_interventions.csv',methodrows)
np.savez_compressed(B/'text_scores.npz',query_ids=np.array(qids),reference_ids=np.array(rids),**textscores)
csvwrite(B/'ocr_line_evidence.csv',evidence)
write(B/'ocr_summary.json',dict(summaries=summaries,baseline_replay_errors=errors,seconds=time.time()-start,rows=len(result),lines=len(evidence),limits=['Truth-selected line deletion is an oracle diagnostic, not a deployable rule.','Negative pair contribution does not by itself establish OCR transcription error.','Fresh and fixed text normalization reported separately.']))
print('DONE OCR',len(result),errors,flush=True)
