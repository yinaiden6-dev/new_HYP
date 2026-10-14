import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import time,multiprocessing as mp
from pathlib import Path
import numpy as np,torch
from experiment_utils import read,write,csvwrite,Ranking,metrics
B=Path(__file__).resolve().parent;ROOT=Path('/mnt/lsdf_zyin')
def worker(gpu):
 torch.set_num_threads(2);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False
 c=read(B/'cohort.json');z=np.load(B/'masks.npz');rv=np.load(B/'reference_visual_masks.npz');names=read(B/'matching_controls_protocol.json')['methods'];start=time.time()
 qb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True);rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True)
 old=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True)
 refs=[x[x.float().norm(dim=1)>0].float() for x in rb['embeddings']];blocks=[]
 for lo in range(0,len(refs),16):
  rr=refs[lo:lo+16];size=max(map(len,rr));a=torch.zeros((len(rr),size,128),device=gpu);mask=torch.zeros((len(rr),size),device=gpu,dtype=torch.bool);vm=mask.clone()
  for j,v in enumerate(rr):
   a[j,:len(v)]=v.to(gpu);mask[j,:len(v)]=True;vv=rv[rb['ids'][lo+j]];assert len(vv)==len(v);vm[j,:len(v)]=torch.from_numpy(vv).to(gpu)
  blocks.append((lo,a.transpose(1,2).contiguous(),mask,vm))
 ids=list(range(gpu,len(c),4));scores=np.lib.format.open_memmap(B/f'matching_scores_gpu{gpu}.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(names),len(refs)));flags=[];maxerr=0
 for k,i in enumerate(ids):
  qid=c[i]['query'];idx=z[qid+'__query_indices'];q=qb['embeddings'][i][idx].float().to(gpu)
  v=z[qid+'__visual'];t=z[qid+'__template'];fg=z[qid+'__foreground_center'];bg=v&~fg;frac=z[qid+'__box_overlap_fraction']
  # If a diagnostic subgroup is empty, use full-query scores and explicitly flag fallback.
  masks=[np.ones(len(idx)),fg,bg,t,v,fg.astype(float)*v.sum()/max(1,fg.sum())+t,((frac>0)&v)|t]
  for j,m in enumerate(masks):
   if not m.sum():masks[j]=np.ones(len(idx));flags.append(dict(query=qid,method=names[j],fallback='empty diagnostic subset; use full'))
  ws=np.asarray(masks,dtype=float);ws/=ws.sum(1,keepdims=True);w=torch.from_numpy(ws).to(gpu)
  other=np.asarray([np.ones(len(idx)),fg|t,v],dtype=float);other/=other.sum(1,keepdims=True);wo=torch.from_numpy(other).to(gpu)
  for lo,rr,mask,vm in blocks:
   d=torch.matmul(q[None],rr);d.masked_fill_(~mask[:,None,:],-torch.inf);m=d.amax(2).double();scores[k,:7,lo:lo+len(rr)]=(w@m.T).cpu().numpy()
   d.masked_fill_(~vm[:,None,:],-torch.inf);m=d.amax(2).double();scores[k,7:,lo:lo+len(rr)]=(wo@m.T).cpu().numpy()
  err=float(np.abs(scores[k,0]-old['scores'][i].numpy()).max());assert err<3e-6;maxerr=max(maxerr,err)
 scores.flush();write(B/f'matching_scores_gpu{gpu}.json',dict(query_indices=ids,methods=names,max_error=maxerr,fallbacks=flags,seconds=time.time()-start));print('MATCH DONE',gpu,flush=True)
if __name__=='__main__':
 mp.set_start_method('spawn');ps=[mp.Process(target=worker,args=(g,)) for g in range(4)]
 for p in ps:p.start()
 for p in ps:p.join()
 assert all(p.exitcode==0 for p in ps),[p.exitcode for p in ps]
 c=read(B/'cohort.json');rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',weights_only=True,map_location='cpu');ranker=Ranking(rb['ids'],rb['sha256'],c)
 names=read(B/'matching_controls_protocol.json')['methods'];scores=np.empty((len(c),len(names),len(rb['ids'])),np.float32)
 for gpu in range(4):m=read(B/f'matching_scores_gpu{gpu}.json');scores[m['query_indices']]=np.load(B/f'matching_scores_gpu{gpu}.npy')
 old=np.array([int(r['image_rank']) for r in c]);tp=list(__import__('csv').DictReader((B/'token_per_query.csv').open(encoding='utf-8-sig')))
 harmed={r['query'] for r in tp if r['channel']=='image' and r['method']=='center' and r['weight']=='1' and r['harm']=='True'};assert len(harmed)==31
 rows=[];summary={}
 for k,name in enumerate(names):
  ranks,top,margin=ranker.all(scores[:,k]);summary[name]=dict(all=metrics(ranks,c,baseline=old),prior31=metrics(ranks,c,[i for i,r in enumerate(c) if r['query'] in harmed],old))
  for i,r in enumerate(c):rows.append(dict(query=r['query'],package_id=r['package_id'],split=r['split'],method=name,rank=int(ranks[i]),top1=top[i],margin=float(margin[i]),original_rank=int(old[i]),prior_center_harm=r['query'] in harmed))
 csvwrite(B/'matching_controls_per_query.csv',rows);write(B/'matching_controls_summary.json',summary)
