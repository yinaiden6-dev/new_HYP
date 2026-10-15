"""Independent CPU NumPy validation of candidate-derived weights and score samples."""
import sys,time,hashlib
from pathlib import Path
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write
torch.set_num_threads(2);t0=time.time();c=read(P/'cohort.json');z=np.load(P/'masks.npz');qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True);rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);first=np.unique(rb['sha256'],return_index=True)[1];checks=[]
for i in [0,247,1116]:
 qid=c[i]['query'];q=qb['embeddings'][i][z[qid+'__query_indices']].float().numpy();m=np.empty((len(q),len(rb['ids'])),np.float32)
 for j,r in enumerate(rb['embeddings']):
  a=r[r.float().norm(dim=1)>0].float().numpy();m[:,j]=(q@a.T).max(1)
 top=[];seen=set()
 for j in np.argsort(-m.mean(0,dtype=np.float64),kind='stable'):
  if rb['sha256'][j] in seen:continue
  seen.add(rb['sha256'][j]);top.append(int(j))
  if len(top)==20:break
 saved=np.load(O/'contrast_weights'/f'{qid}.npz');assert np.array_equal(top,saved['top20'])
 g=m[:,first].std(1,dtype=np.float64);h=m[:,top].std(1,dtype=np.float64);g/=g.sum();h/=h.sum();assert max(abs(g-saved['global_std']).max(),abs(h-saved['hard20_std']).max())<2e-6
 ww=[np.ones(len(q))/len(q),g,h];vis=z[qid+'__visual']
 for seed in range(10):
  rng=np.random.default_rng(int(hashlib.sha256(f'{qid}/{seed}/contrast'.encode()).hexdigest()[:12],16));w=h.copy()
  for group in [np.flatnonzero(vis),np.flatnonzero(~vis)]:w[group]=rng.permutation(w[group])
  assert np.allclose([w[vis].sum(),w[~vis].sum()],[h[vis].sum(),h[~vis].sum()]);ww.append(w)
 worker=i%2;ix=read(O/f'reference_contrast_gpu{worker}.json')['indices'].index(i);actual=np.load(O/f'reference_contrast_gpu{worker}.npy',mmap_mode='r')[ix];expected=np.array(ww)@m;err=float(abs(expected-actual).max());assert err<3e-6;checks.append(dict(query=qid,all_reference_scores=expected.size,max_absolute_error=err));print('CHECKED',qid,err,flush=True)
write(O/'REFERENCE_CONTRAST_VALIDATION.json',dict(status='passed',queries=len(checks),score_values=sum(r['all_reference_scores'] for r in checks),checks=checks,seconds=time.time()-t0))
