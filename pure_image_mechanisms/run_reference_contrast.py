"""Label-free image-token weights from variation across reference candidates."""
import os,sys,time,hashlib,argparse
from pathlib import Path
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,Ranking,metrics
NAMES=['full','global_std','hard20_std']+['hard20_shuffle_%02d'%s for s in range(10)]
def worker(gpu,nworkers):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False;t0=time.time()
 c=read(P/'cohort.json');z=np.load(P/'masks.npz');qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True);rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True);old=torch.load(R/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True)
 first=np.unique(np.array(rb['sha256']),return_index=True)[1];refs=[r[r.float().norm(dim=1)>0].float() for r in rb['embeddings']];blocks=[]
 for lo in range(0,len(refs),16):
  rr=refs[lo:lo+16];n=max(map(len,rr));a=torch.zeros((len(rr),n,128),device=gpu);mask=torch.zeros((len(rr),n),device=gpu,dtype=torch.bool)
  for j,r in enumerate(rr):a[j,:len(r)]=r.to(gpu);mask[j,:len(r)]=True
  blocks.append((lo,a.transpose(1,2).contiguous(),mask))
 ids=list(range(gpu,len(c),nworkers));out=np.lib.format.open_memmap(O/f'reference_contrast_gpu{gpu}.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(NAMES),len(refs)));audit=[]
 (O/'contrast_weights').mkdir(exist_ok=True)
 for k,i in enumerate(ids):
  qid=c[i]['query'];idx=z[qid+'__query_indices'];q=qb['embeddings'][i][idx].float().to(gpu);m=torch.empty((len(q),len(refs)),device=gpu)
  for lo,rr,mask in blocks:d=q[None]@rr;d.masked_fill_(~mask[:,None,:],-torch.inf);m[:,lo:lo+len(rr)]=d.amax(2).T
  base=m.mean(0);assert float((base.cpu()-old['scores'][i]).abs().max())<3e-6
  # Weighting uses one canonical row/SHA globally; top20 follows exact baseline SHA ranking.
  order=np.argsort(-base.cpu().numpy(),kind='stable');top=[];seen=set()
  for j in order:
   if rb['sha256'][j] in seen:continue
   seen.add(rb['sha256'][j]);top.append(int(j))
   if len(top)==20:break
  top=np.array(top)
  g=m[:,first].std(1,unbiased=False);h=m[:,top].std(1,unbiased=False);g=g/g.sum() if g.sum()>0 else torch.ones_like(g)/len(g);h=h/h.sum() if h.sum()>0 else torch.ones_like(h)/len(h)
  ww=[torch.ones_like(h)/len(h),g,h];vis=z[qid+'__visual'];hcpu=h.cpu().numpy()
  for seed in range(10):
   rng=np.random.default_rng(int(hashlib.sha256(f'{qid}/{seed}/contrast'.encode()).hexdigest()[:12],16));x=hcpu.copy()
   for group in [np.flatnonzero(vis),np.flatnonzero(~vis)]:x[group]=rng.permutation(x[group])
   ww.append(torch.from_numpy(x).to(gpu))
  out[k]=(torch.stack(ww)@m).cpu().numpy();np.savez_compressed(O/'contrast_weights'/f'{qid}.npz',global_std=g.cpu().numpy(),hard20_std=hcpu,top20=top,query_indices=idx)
  audit.append(dict(query=qid,positive_in_top20=any(rb['sha256'][j] in c[i]['positive_sha256'] for j in top),max_weight=float(h.max()),effective_tokens=float(1/(h*h).sum())))
  if k%40==0:out.flush();print('CONTRAST',gpu,k+1,len(ids),round(time.time()-t0,1),flush=True)
 out.flush();write(O/f'reference_contrast_gpu{gpu}.json',dict(indices=ids,methods=NAMES,seconds=time.time()-t0,audit=audit))
def summarize(nworkers):
 c=read(P/'cohort.json');rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);rk=Ranking(rb['ids'],rb['sha256'],c);scores=np.empty((len(c),len(NAMES),len(rb['ids'])),np.float32)
 for g in range(nworkers):m=read(O/f'reference_contrast_gpu{g}.json');scores[m['indices']]=np.load(O/f'reference_contrast_gpu{g}.npy')
 val=[i for i,r in enumerate(c) if r['split']=='val'];test=[i for i,r in enumerate(c) if r['split']=='test'];base=np.array([int(r['image_rank']) for r in c]);selected={};rows=[];curves=[]
 for name,j in [('global_std',1),('hard20_std',2)]:
  choices=[]
  for weight in [0,.1,.25,.5,.75,1]:
   rr,tt,mm=rk.all((1-weight)*scores[:,0]+weight*scores[:,j]);v=metrics(rr,c,val,base);choices.append((v['macro_accuracy'],v['macro_mrr'],-weight,weight,rr,tt,mm))
   for scope,ids in [('all',None),('val',val),('test',test)]:curves.append(dict(method=name,weight=weight,scope=scope,**metrics(rr,c,ids,base)))
  best=max(choices,key=lambda x:x[:3]);weight,rr,tt,mm=best[3:];selected[name]=dict(weight=weight,all=metrics(rr,c,baseline=base),val=metrics(rr,c,val,base),test=metrics(rr,c,test,base))
  for i,r in enumerate(c):rows.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],method=name,weight=weight,rank=int(rr[i]),top1=str(tt[i]),margin=float(mm[i]),original_rank=int(base[i])))
 weight=selected['hard20_std']['weight'];nulls=[]
 for j,name in enumerate(NAMES[3:],start=3):
  rr,tt,mm=rk.all((1-weight)*scores[:,0]+weight*scores[:,j]);nulls.append(dict(method=name,weight=weight,**metrics(rr,c,baseline=base)))
  for i,r in enumerate(c):rows.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],method=name,weight=weight,rank=int(rr[i]),top1=str(tt[i]),margin=float(mm[i]),original_rank=int(base[i])))
 csvwrite(O/'reference_contrast_per_query.csv',rows);csvwrite(O/'reference_contrast_curves.csv',curves);write(O/'reference_contrast_summary.json',dict(selected=selected,nulls=nulls));print(selected,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--summary',action='store_true');a=ap.parse_args();summarize(a.workers) if a.summary else worker(a.gpu,a.workers)
