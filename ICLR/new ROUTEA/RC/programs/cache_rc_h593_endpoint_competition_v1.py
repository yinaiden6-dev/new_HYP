#!/usr/bin/env python3
"""Unlabelled full-reference same-support competition from qualified natural tokens/maps."""
import argparse,json,os,sys,subprocess,uuid
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import cache_rc_new_hyp593_features_v1 as C
OUT=ROOT/'results/rc_h593_endpoint_competition_v1'
AUTH=ROOT/'registry/rc_h593_endpoint_competition_cache_authority_v1_20260911.json'

def guard(stage):
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 if stage!='preflight':
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');pf=M.read(OUT/'cache_preflight.json');M.need(pf['status']=='H593_J_POOL_AND_SERIALIZATION_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH),'CACHE_PREFLIGHT')
 def audit(e,args):
  if e!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  s=os.path.abspath(os.fsdecode(args[0])).lower();M.need(not any(x in s for x in ('curator_roles','role_shards','/roles/fold','target_join','rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/fits/','/reports/')),'CACHE_LABEL_AND_DIAGNOSTIC_BARRIER')
 sys.addaudithook(audit)

def normalize(x):
 x=x.double();return x/x.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12)

def pool(free,weights):
 b=torch.stack([(free*w[None]).sum(1)/w.sum().clamp_min(1e-12) for w in weights]);other=b.clone();other.fill_diagonal_(-torch.inf);return b,b.diag().clone(),b.diag()-other.amax(1)

def compute(kind,shard,replay):
 rows=[]
 for w in C.selected(kind,shard):
  q,r,refs=C.pair(w);axis=q['candidate_physical_rows'];M.need(len(axis)==128 and axis==sorted(set(axis)),'FULL_C128');a=[];weights=[];qv=normalize(q['query_tokens'])
  for i,c in enumerate(r['candidates']):
   ref=refs[axis[i]];M.need(c['candidate_position']==i and c['physical_row']==axis[i] and c['reference_tokens_sha256']==ref['tokens_sha256'],'REFERENCE_ALIGNMENT');sim=qv@normalize(ref['tokens']).T
   if replay:
    m=sim.numpy();idx=np.argmax(m,axis=1);free=torch.from_numpy(m[np.arange(len(m)),idx].copy())
   else:free=sim.max(1).values
   wq=c['query_visibility'];M.need(wq.dtype==torch.float64 and wq.shape==(len(qv),) and bool(torch.isfinite(wq).all()) and bool(((wq>=0)&(wq<=1)).all()),'WEIGHT_DOMAIN');a.append(free);weights.append(wq)
  b,f,j=pool(torch.stack(a),torch.stack(weights));M.need(bool(torch.isfinite(b).all()) and bool(torch.isfinite(j).all()),'FINITE_COMPETITION')
  rows.append(dict(query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],source_image_sha256=w['source_image_sha256'],candidate_physical_rows=axis,B=b,F=f,J=j))
  print(json.dumps(dict(event='J_CACHE_REPLAY' if replay else 'J_CACHE_QUERY_READY',query_id=w['query_id'])),flush=True)
 return rows

def write_shard(kind,shard,replay,nonce):
 folder=OUT/'cache'/kind/f'shard{shard:02d}';rows=compute(kind,shard,replay);M.need(bool(rows),'NONEMPTY_SHARD')
 if replay:
  M.need(nonce==os.environ.get('H593_J_NONCE'),'FRESH_REPLAY');rec=M.read(folder/'receipt.json');old=torch.load(M.checked(rec['payload']),map_location='cpu',weights_only=True,mmap=True);M.need(len(old['records'])==len(rows),'ROW_COUNT')
  for x,y in zip(old['records'],rows):
   M.need({k:v for k,v in x.items() if k not in ('B','F','J')}=={k:v for k,v in y.items() if k not in ('B','F','J')},'METADATA_REPLAY')
   for key in ('B','F','J'):M.need(torch.equal(x[key],y[key]),'INDEPENDENT_FULL_AXIS_POOL_BITS')
  M.write(folder/'validation.json',dict(status='H593_J_UNLABELLED_NUMPY_POSITION_REPLAY_PASS',payload=rec['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),count=len(rows),fresh_nonce=nonce))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_CACHE')
  with p.open('xb') as f:torch.save(dict(records=rows,authority=M.bind(AUTH)),f);f.flush();os.fsync(f.fileno())
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH),count=len(rows)))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--kind',kind,'--shard',str(shard),'--nonce',nonce],env=dict(os.environ,H593_J_NONCE=nonce,CUDA_VISIBLE_DEVICES=''),check=True)

def preflight():
 free=torch.tensor([[.75,.75],[1.,0.],[0.,1.]],dtype=torch.float64);weights=torch.ones_like(free);b,f,j=pool(free,weights);M.need(j.tolist()==[.25,-.25,-.25],'POOL_BEFORE_MAX');b,f,j=pool(free,torch.zeros_like(weights));M.need(torch.equal(j,torch.zeros_like(j)),'ZERO_SUPPORT')
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'cache_synthetic.pt'
 with p.open('xb') as stream:torch.save(dict(B=b,F=f,J=j),stream)
 x=torch.load(p,map_location='cpu',weights_only=True,mmap=True);M.need(torch.equal(x['J'],j),'SERIALIZATION')
 M.write(OUT/'cache_preflight.json',dict(status='H593_J_POOL_AND_SERIALIZATION_PREFLIGHT_PASS',authority=M.bind(AUTH),natural_tensor_reads=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['produce','replay','preflight']);ap.add_argument('--kind',choices=['reuse','missing'],default='reuse');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(a.stage)
 if a.stage=='preflight':preflight()
 else:write_shard(a.kind,a.shard,a.stage=='replay',a.nonce)
