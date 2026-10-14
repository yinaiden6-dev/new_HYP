"""Oracle nonnegative query-token weights: primal witness and dual upper bound."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,time,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np,torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'solver_dependencies'))
from scipy.optimize import linprog
from experiment_utils import read,write
ROOT=Path('/mnt/lsdf_zyin')
def solve(D):
 nvars=D.shape[1];initial=np.argsort(D.mean(1))[:24];active=list(map(int,initial))
 for iteration in range(100):
  d=D[active];c=np.r_[np.zeros(nvars),-1.];a=np.c_[-d,np.ones(len(d))]
  res=linprog(c,A_ub=a,b_ub=np.zeros(len(d)),A_eq=np.array([np.r_[np.ones(nvars),0.]]),b_eq=[1],bounds=[(0,None)]*nvars+[(None,None)],method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'threads':1})
  assert res.success,res.message
  w=res.x[:-1];gaps=D@w;lower=float(gaps.min());target=float(res.x[-1]);alpha=-res.ineqlin.marginals
  assert abs(alpha.sum()-1)<1e-6 and alpha.min()>-1e-7
  upper=float(np.max(alpha@d));gap=upper-lower
  if gap<5e-7:
   return dict(lower=lower,upper=upper,duality_gap=gap,iterations=iteration+1,active=len(active)),w,np.array(active),alpha
  more=[int(j) for j in np.argsort(gaps)[:64] if gaps[j]<target-1e-8 and j not in active]
  assert more,(lower,upper,target);active.extend(more)
 raise RuntimeError('Cutting-plane limit reached')
def worker(gpu,null=False):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False
 start=time.time();cohort=read(B/'cohort.json');z=np.load(B/'masks.npz');out=B/'weight_witnesses';out.mkdir(exist_ok=True)
 qb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True)
 rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True)
 old=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True)
 refs=[x[x.float().norm(dim=1)>0].float() for x in rb['embeddings']];blocks=[]
 for lo in range(0,len(refs),16):
  rr=refs[lo:lo+16];size=max(map(len,rr));a=torch.zeros((len(rr),size,128),device=gpu);mask=torch.zeros((len(rr),size),device=gpu,dtype=torch.bool)
  for j,v in enumerate(rr):a[j,:len(v)]=v.to(gpu);mask[j,:len(v)]=True
  blocks.append((lo,a.transpose(1,2).contiguous(),mask))
 ids=[i for i,r in enumerate(cohort) if r['image_correct']=='False'][gpu::4];results=[]
 for k,i in enumerate(ids):
  r=cohort[i];qid=r['query'];idx=z[qid+'__query_indices'];q=qb['embeddings'][i][idx].float().to(gpu)
  maxima=np.empty((len(refs),len(idx)),np.float32)
  for lo,rr,mask in blocks:
   dots=torch.matmul(q[None],rr);dots.masked_fill_(~mask[:,None,:],-torch.inf);maxima[lo:lo+len(rr)]=dots.amax(2).cpu().numpy()
  assert np.abs(maxima.mean(1,dtype=np.float64)-old['scores'][i].numpy()).max()<3e-6
  positives=np.array([j for j,s in enumerate(rb['sha256']) if s in r['positive_sha256']]);neg=np.array([j for j,s in enumerate(rb['sha256']) if s not in r['positive_sha256']])
  p=int(positives[np.argmax(maxima[positives].mean(1))]);assert np.allclose(maxima[positives],maxima[p],atol=2e-7,rtol=0),'Duplicate positive SHA has different features; solve each separately.'
  if null:
   rng=np.random.default_rng(int.from_bytes(hashlib.sha256((qid+'/null-20261014').encode()).digest()[:8],'little'))
   unique={}
   for j in neg:unique.setdefault(rb['sha256'][j],int(j))
   chosen=rng.choice(list(unique.values()),size=5,replace=False)
   baseline=maxima.mean(1,dtype=np.float64);ordered=[];seen=set()
   for j in np.argsort(-baseline,kind='stable'):
    h=rb['sha256'][j]
    if h not in seen:ordered.append(int(j));seen.add(h)
   at=next(j for j,x in enumerate(ordered) if x in positives)
   neighbor=ordered[at+1] if at+1<len(ordered) else ordered[at-1]
   for label,target in [('random_'+str(j),int(tar)) for j,tar in enumerate(chosen)]+[('rank_neighbor',neighbor)]:
    other=np.array([j for j,s in enumerate(rb['sha256']) if s!=rb['sha256'][target]])
    D=maxima[target].astype(float)[None,:]-maxima[other].astype(float);summary,w,active,alpha=solve(D)
    status='strict_rank1_witness' if summary['lower']>1e-6 else 'certified_no_strict_rank1' if summary['upper']< -1e-6 else 'numerical_tie_boundary'
    results.append(dict(query=qid,control=label,target=rb['ids'][target],target_baseline_rank=next(j+1 for j,x in enumerate(ordered) if rb['sha256'][x]==rb['sha256'][target]),status=status,**summary))
   if (k+1)%10==0:print('NULL',gpu,k+1,'/',len(ids),'SEC',round(time.time()-start),flush=True)
   continue
  masks={'all':np.ones(len(idx),dtype=bool),'subject_template':z[qid+'__foreground_center']|z[qid+'__template']}
  for mode,mask in masks.items():
   D=maxima[p,mask].astype(float)[None,:]-maxima[neg][:,mask].astype(float)
   summary,w,active,alpha=solve(D);ww=np.zeros(len(idx));ww[mask]=w;score=maxima@ww
   # Strict margin witness independent of the optimization model's auxiliary t.
   witness_gap=float(score[p]-score[neg].max());assert abs(witness_gap-summary['lower'])<2e-7
   status='strict_rank1_witness' if summary['lower']>1e-6 else 'certified_no_strict_rank1' if summary['upper']< -1e-6 else 'numerical_tie_boundary'
   path=out/f'{qid}__{mode}.npz';np.savez_compressed(path,query_raw_indices=idx,query_weights=ww,positive_reference_index=p,negative_reference_indices=neg[active],dual_weights=alpha)
   results.append(dict(query=qid,package_id=r['package_id'],split=r['split'],mode=mode,baseline_rank=int(r['image_rank']),status=status,positive=rb['ids'][p],top1=rb['ids'][int(score.argmax())],nonzero_weights=int((ww>1e-8).sum()),**summary,witness_file=str(path.relative_to(B))))
  if (k+1)%10==0:print('GPU',gpu,'DONE',k+1,'/',len(ids),'SEC',round(time.time()-start),flush=True);write(B/f'weight_results_gpu{gpu}.json',results)
 write(B/f'weight_{"null" if null else "results"}_gpu{gpu}.json',results);print('FINISHED',gpu,len(ids),round(time.time()-start),flush=True)
if __name__=='__main__':
 null='--null-control' in sys.argv
 if null:write(B/'weight_null_protocol.json',dict(scope='Same247 original image failures, all-query-token support.',controls='Five deterministic uniformly sampled wrong reference SHAs per query; one wrong SHA immediately below the true SHA in original rank, or above if last. Same oracle objective and solver.',purpose='Test whether truth-target feasibility is specific or arbitrary wrong targets can also be promoted; descriptive conditional control, not a causal information proof.'))
 else:write(B/'weight_feasibility_protocol.json',dict(scope='All247 original image failures;870 successes already have uniform-weight witnesses.',optimization='For the sole positive reference SHA, maximize minimum margin over every wrong reference with w>=0,sum(w)=1. All tokens and subject+template support. Cutting plane, full-gallery primal verification and convex mixture of negative constraints as independent dual bound.',tolerance=1e-6,limits=['Truth-dependent per-photo oracle; no deployable method or generalization claim.','Positive witness does not establish semantically meaningful identity coding; nuisance correlations can be exploited.','Negative upper bound rules out this query-weighted MaxSim family only, not all matchers, intermediate layers or encoder information.']))
 mp.set_start_method('spawn');ps=[mp.Process(target=worker,args=(g,null)) for g in range(4)]
 for p in ps:p.start()
 for p in ps:p.join()
 assert all(p.exitcode==0 for p in ps),[p.exitcode for p in ps]
 if null:
  results=sum([read(B/f'weight_null_gpu{g}.json') for g in range(4)],[]);write(B/'weight_null_results.json',results);sys.exit(0)
 results=sum([read(B/f'weight_results_gpu{g}.json') for g in range(4)],[]);write(B/'weight_feasibility_results.json',results)
 from collections import Counter
 write(B/'weight_feasibility_summary.json',dict(counts={mode:dict(Counter(r['status'] for r in results if r['mode']==mode)) for mode in ['all','subject_template']},cases=247,solves=len(results),maximum_duality_gap=max(r['duality_gap'] for r in results)))
