"""Prespecified validation-only weight selection, held-out historical evaluation and controls."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import time
import numpy as np,torch
from experiment_utils import read,write,csvwrite,zscore,Ranking,metrics
B=Path(__file__).resolve().parent;ROOT=Path('/mnt/lsdf_zyin');start=time.time();torch.set_num_threads(2)
cohort=read(B/'cohort.json');protocol=read(B/'protocol.json');names=protocol['methods']
I=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',weights_only=True,map_location='cpu');image=I['scores'].numpy().astype(float)
rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',weights_only=True,map_location='cpu')
ranker=Ranking(I['reference_ids'],rb['sha256'],cohort);zi=zscore(image)
scores=np.empty((len(cohort),len(names),len(rb['ids'])),np.float32);maxerr=0
for gpu in range(4):
 m=read(B/f'token_scores_gpu{gpu}.json');scores[m['query_indices']]=np.load(B/f'token_scores_gpu{gpu}.npy');maxerr=max(maxerr,m['max_baseline_error'])
assert np.abs(scores[:,0]-image).max()<3e-6
fusion=torch.load(ROOT/'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt',weights_only=True,map_location='cpu')['scores'].numpy().astype(float)
fusions={'original':fusion,'center_presence':np.load(ROOT/'results/colnomic_multiobject_diagnosis/20261014/experiments/center_presence_all1117_scores.npz')['fusion'].astype(float),'numeric':np.load(ROOT/'results/colnomic_multiobject_diagnosis/20261014/numeric_tokenization/scores.npz')['fusion'].astype(float)}
zt={k:2*v-zi for k,v in fusions.items()};sd=image.std(1,keepdims=True);mu=image.mean(1,keepdims=True)
scopes={'all':np.arange(len(cohort))}
for s in ['train','val','test']:scopes[s]=np.array([i for i,r in enumerate(cohort) if r['split']==s])
for s in ['single_object','multi_object']:scopes[s]=np.array([i for i,r in enumerate(cohort) if r['background']==s])
scopes['four_methods_success825']=np.array([i for i,r in enumerate(cohort) if all(r[k+'_correct']=='True' for k in ['image','original','center_presence','numeric'])])
baseline={k:ranker.all(v)[0] for k,v in dict(image=image,**fusions).items()}
for k,ranks in baseline.items():assert np.array_equal(ranks,[int(r[k+'_rank']) for r in cohort]),k
configs=[('image','none')]+[(k,n) for k in fusions for n in ['fresh','fixed']]
curves=[];per=[];selected={};selectedranks={};allresults={}
def run(method,weight,channel,norm):
 a=(1-weight)*image+weight*scores[:,names.index(method)]
 if channel!='image':a=.5*((zscore(a) if norm=='fresh' else (a-mu)/sd)+zt[channel])
 ranks,top,gap=ranker.all(a)
 return ranks,top,gap
def save(method,w,channel,norm,ranks,top,gap,kind):
 tag=f'{channel}/{norm}/{method}/{w:g}';allresults[tag]=ranks
 for scope,ids in scopes.items():curves.append(dict(channel=channel,normalization=norm,method=method,weight=w,kind=kind,scope=scope,**metrics(ranks,cohort,ids,baseline[channel])))
 for i,r in enumerate(cohort):per.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],channel=channel,normalization=norm,method=method,weight=w,kind=kind,rank=int(ranks[i]),top1=top[i],margin=float(gap[i]),baseline_rank=int(baseline[channel][i]),repair=baseline[channel][i]>1 and ranks[i]==1,harm=baseline[channel][i]==1 and ranks[i]>1))
for channel,norm in configs:
 candidates=[]
 for w in protocol['weights']:
  ranks,top,gap=run('center',w,channel,norm);m=metrics(ranks,cohort,scopes['val']);candidates.append((m['macro_accuracy'],m['macro_mrr'],-w,w,ranks,top,gap));save('center',w,channel,norm,ranks,top,gap,'weight_curve')
 best=max(candidates,key=lambda x:x[:3]);w=best[3];key=channel+'/'+norm;selected[key]=dict(weight=w,validation_macro_accuracy=best[0],validation_macro_mrr=best[1],scopes={s:metrics(best[4],cohort,ids,baseline[channel]) for s,ids in scopes.items()})
 selectedranks[key]=best[4]
 for method in names[2:]:
  weights=[w] if method.startswith('random') else sorted(set([w,1]))
  for ww in weights:
   ranks,top,gap=run(method,ww,channel,norm);save(method,ww,channel,norm,ranks,top,gap,'random_control' if method.startswith('random') else 'mask_sensitivity')
 print('EVALUATED',key,'chosen',w,'all',selected[key]['scopes']['all'],flush=True)
# Cluster bootstrap compares package-average accuracies; no photo independence assumption.
bootstrap={}
for key,ranks in selectedranks.items():
 ch=key.split('/')[0];out={}
 for scope in ['all','test','val']:
  ids=scopes[scope];packages=sorted({cohort[i]['package_id'] for i in ids});delta=[]
  for p in packages:
   ii=[i for i in ids if cohort[i]['package_id']==p];delta.append(float(np.mean(ranks[ii]==1)-np.mean(baseline[ch][ii]==1)))
  rng=np.random.default_rng(20261014);delta=np.array(delta);boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
  out[scope]=dict(macro_accuracy_change=float(delta.mean()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),identities=len(delta))
 bootstrap[key]=out
# Feasibility of ANY nonnegative convex mix of full and regional readout. This is an oracle diagnostic.
oracle=[]
for method in ['center','overlap50','fully_inside','area_fraction']:
 region=scores[:,names.index(method)].astype(float)
 for i,r in enumerate(cohort):
  possible=[];negative=~ranker.rawpos[i]
  for p in np.flatnonzero(ranker.rawpos[i]):
   d=image[i,p]-image[i,negative];v=region[i,p]-region[i,negative]-d
   flat=np.abs(v)<1e-12
   if np.any(flat&(d<0)):continue
   lo=max(0.,float(np.max(-d[v>1e-12]/v[v>1e-12],initial=-np.inf)));hi=min(1.,float(np.min(-d[v< -1e-12]/v[v< -1e-12],initial=np.inf)))
   if lo<hi:
    mid=(lo+hi)/2;rr=ranker.one((1-mid)*image[i]+mid*region[i],i)
    if rr['rank']==1:possible.append((lo,hi,str(ranker.rids[p])))
  oracle.append(dict(query=r['query'],method=method,baseline_rank=int(baseline['image'][i]),oracle_convex_mix_can_rank1=bool(possible),feasible_intervals=possible))
csvwrite(B/'token_metrics.csv',curves);csvwrite(B/'token_per_query.csv',per)
write(B/'token_oracle_intervals.json',oracle)
write(B/'token_summary.json',dict(selected=selected,cluster_bootstrap=bootstrap,baseline={k:metrics(v,cohort) for k,v in baseline.items()},max_baseline_replay_error=maxerr,scoring_queries=1117,references=5405,unique_references=len(ranker.usha),seconds=time.time()-start,oracle_counts={m:sum(r['method']==m and r['baseline_rank']>1 and r['oracle_convex_mix_can_rank1'] for r in oracle) for m in ['center','overlap50','fully_inside','area_fraction']},limits=protocol['oracle_warning']))
print('DONE TOKEN EVALUATION',round(time.time()-start,1),flush=True)
