"""GPU replay of frozen MaxSim and prespecified regional readouts, no encoder calls."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,json,time,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
B=Path(__file__).resolve().parent
ROOT=Path('/mnt/lsdf_zyin')
def worker(gpu,workers):
 torch.set_num_threads(2);torch.cuda.set_device(gpu)
 torch.backends.cuda.matmul.allow_tf32=False
 start=time.time();protocol=json.loads((B/'protocol.json').read_text());names=protocol['methods']
 cohort=json.loads((B/'cohort.json').read_text());z=np.load(B/'masks.npz')
 qbank=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True)
 rbank=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True)
 original=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True)
 assert qbank['ids']==original['query_ids']==[r['query'] for r in cohort]
 assert rbank['ids']==original['reference_ids']
 refs=[x[x.float().norm(dim=1)>0].float() for x in rbank['embeddings']]
 # Preload reference blocks once; padding is excluded before the MaxSim maximum.
 blocks=[]
 for lo in range(0,len(refs),16):
  rr=refs[lo:lo+16];size=max(map(len,rr));a=torch.zeros((len(rr),size,128),device=gpu);mask=torch.zeros((len(rr),size),device=gpu,dtype=torch.bool)
  for j,v in enumerate(rr):a[j,:len(v)]=v.to(gpu);mask[j,:len(v)]=True
  blocks.append((lo,a.transpose(1,2).contiguous(),mask))
 ids=list(range(gpu,len(cohort),workers));dest=B/f'token_scores_gpu{gpu}.npy'
 scores=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(ids),len(names),len(refs)))
 errors=[];counts=[]
 for k,i in enumerate(ids):
  r=cohort[i];qid=r['query'];raw=qbank['embeddings'][i];idx=z[qid+'__query_indices'];q=raw[idx].float().to(gpu)
  actual=hashlib.sha256(r['token_dtype'].encode()+json.dumps(list(raw.shape),separators=(',',':')).encode()+raw.contiguous().numpy().tobytes()).hexdigest()
  assert actual==r['token_sha256'],qid
  v=z[qid+'__visual'];t=z[qid+'__template'];c=z[qid+'__foreground_center'];f=z[qid+'__box_overlap_fraction']
  ws=[np.ones(len(idx)),c+t,((f>=.5)&v)+t,((f>=1-1e-8)&v)+t,f+t,(v&~c)+t]
  vi=np.flatnonzero(v)
  for seed in range(10):
   rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{qid}/{seed}/20261014'.encode()).digest()[:8],'little'))
   take=rng.choice(vi,int(c.sum()),replace=False);w=t.astype(float);w[take]=1;ws.append(w)
  weights=np.asarray(ws,dtype=np.float32);assert (weights.sum(1)>0).all()
  weights/=weights.sum(1,keepdims=True);w=torch.from_numpy(weights).to(gpu)
  for lo,rr,mask in blocks:
   dots=torch.matmul(q[None],rr);dots.masked_fill_(~mask[:,None,:],-torch.inf)
   maxima=dots.amax(2);s=torch.matmul(w,maxima.T)
   scores[k,:,lo:lo+len(rr)]=s.cpu().numpy()
  err=float(np.abs(scores[k,0]-original['scores'][i].numpy()).max());assert err<3e-6,(qid,err)
  errors.append(err);counts.append(dict(query=qid,visual=int(v.sum()),foreground=int(c.sum()),template=int(t.sum())))
  if (k+1)%20==0:print(json.dumps(dict(gpu=gpu,done=k+1,total=len(ids),seconds=round(time.time()-start,1),max_error=max(errors))),flush=True);scores.flush()
 scores.flush()
 (B/f'token_scores_gpu{gpu}.json').write_text(json.dumps(dict(query_indices=ids,methods=names,counts=counts,max_baseline_error=max(errors),elapsed_seconds=time.time()-start,torch=torch.__version__,device=torch.cuda.get_device_name(gpu)),indent=2))
 print('DONE',gpu,len(ids),max(errors),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
 mp.set_start_method('spawn');ps=[mp.Process(target=worker,args=(g,args.workers)) for g in range(args.workers)]
 for p in ps:p.start()
 for p in ps:p.join()
 assert all(p.exitcode==0 for p in ps),[p.exitcode for p in ps]
