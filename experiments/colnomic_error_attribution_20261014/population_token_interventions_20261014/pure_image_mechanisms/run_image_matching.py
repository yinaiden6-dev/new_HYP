"""Full-gallery image-only directional and many-to-one scoring controls; no OCR."""
import os,sys,time,argparse
from pathlib import Path
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,Ranking,metrics
NAMES=['full','reverse','balanced','visual_forward','visual_reverse','visual_balanced']
def balanced(values,indices,nref):
 count=torch.zeros((len(values),nref),device=values.device,dtype=values.dtype);count.scatter_add_(1,indices,torch.ones_like(values));w=1/count.gather(1,indices);return (values*w).sum(1)/w.sum(1)
def worker(gpu,nworkers):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False
 c=read(P/'cohort.json');z=np.load(P/'masks.npz');rv=np.load(P/'reference_visual_masks.npz');t0=time.time()
 qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True);rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True);old=torch.load(R/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True)
 refs=[x[x.float().norm(dim=1)>0].float() for x in rb['embeddings']];blocks=[]
 for lo in range(0,len(refs),12):
  rr=refs[lo:lo+12];n=max(map(len,rr));a=torch.zeros((len(rr),n,128),device=gpu);mask=torch.zeros((len(rr),n),device=gpu,dtype=torch.bool);vm=mask.clone()
  for j,v in enumerate(rr):a[j,:len(v)]=v.to(gpu);mask[j,:len(v)]=True;vm[j,:len(v)]=torch.from_numpy(rv[rb['ids'][lo+j]]).to(gpu)
  assert vm.any(1).all();blocks.append((lo,a.transpose(1,2).contiguous(),mask,vm))
 ids=list(range(gpu,len(c),nworkers));out=np.lib.format.open_memmap(O/f'image_matching_gpu{gpu}.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(NAMES),len(refs)));maxerr=0
 for k,i in enumerate(ids):
  qid=c[i]['query'];idx=z[qid+'__query_indices'];q=qb['embeddings'][i][idx].float().to(gpu);v=torch.from_numpy(z[qid+'__visual']).to(gpu)
  for lo,rr,mask,vm in blocks:
   d=torch.matmul(q[None],rr);d.masked_fill_(~mask[:,None,:],-torch.inf);m,j=d.max(2);rev=d.max(1).values;rev=rev.masked_fill(~mask,0).sum(1)/mask.sum(1)
   vals=[m.mean(1),rev,balanced(m,j,rr.shape[2])]
   dv=d[:,v,:].clone();dv.masked_fill_(~vm[:,None,:],-torch.inf);mv,jv=dv.max(2);rvv=dv.max(1).values;rvv=rvv.masked_fill(~vm,0).sum(1)/vm.sum(1)
   vals += [mv.mean(1),rvv,balanced(mv,jv,rr.shape[2])]
   out[k,:,lo:lo+len(rr)]=torch.stack(vals).cpu().numpy()
  err=float(np.abs(out[k,0]-old['scores'][i].numpy()).max());assert err<3e-6;maxerr=max(maxerr,err)
  if k%25==0:out.flush();print('MATCHING',gpu,k+1,len(ids),round(time.time()-t0,1),flush=True)
 out.flush();write(O/f'image_matching_gpu{gpu}.json',dict(indices=ids,methods=NAMES,maxerr=maxerr,seconds=time.time()-t0));print('MATCHING DONE',gpu,flush=True)
def summarize(nworkers):
 c=read(P/'cohort.json');rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True);ranker=Ranking(rb['ids'],rb['sha256'],c);scores=np.empty((len(c),len(NAMES),len(rb['ids'])),np.float32)
 for g in range(nworkers):r=read(O/f'image_matching_gpu{g}.json');scores[r['indices']]=np.load(O/f'image_matching_gpu{g}.npy')
 base=np.array([int(r['image_rank']) for r in c]);val=[i for i,r in enumerate(c) if r['split']=='val'];test=[i for i,r in enumerate(c) if r['split']=='test'];fail=np.flatnonzero(base>1);rows=[];curves=[];selected={};decisions={}
 for mode,j in [(n,j) for j,n in enumerate(NAMES) if j]:
  options=[]
  for w in [0,.1,.25,.5,.75,1]:
   a=(1-w)*scores[:,0]+w*scores[:,j];rr,tt,mm=ranker.all(a);v=metrics(rr,c,val,base);options.append((v['macro_accuracy'],v['macro_mrr'],-w,w,rr,tt,mm))
   for scope,ids in [('all',None),('val',val),('test',test),('image_failures247',fail)]:curves.append(dict(method=mode,weight=w,scope=scope,**metrics(rr,c,ids,base)))
  best=max(options,key=lambda x:x[:3]);w,rr,tt,mm=best[3:];selected[mode]=dict(weight=w,all=metrics(rr,c,baseline=base),val=metrics(rr,c,val,base),test=metrics(rr,c,test,base),original247=metrics(rr,c,fail,base))
  for i,r in enumerate(c):rows.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],method=mode,weight=w,rank=int(rr[i]),top1=str(tt[i]),margin=float(mm[i]),original_rank=int(base[i]),repair=bool(base[i]>1 and rr[i]==1),harm=bool(base[i]==1 and rr[i]>1)))
  decisions[mode]=rr
 csvwrite(O/'pure_matching_curves.csv',curves);csvwrite(O/'pure_matching_per_query.csv',rows);write(O/'pure_matching_summary.json',selected);np.savez_compressed(O/'pure_matching_selected_ranks.npz',**decisions);print(selected,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--summary',action='store_true');a=ap.parse_args();summarize(a.workers) if a.summary else worker(a.gpu,a.workers)
