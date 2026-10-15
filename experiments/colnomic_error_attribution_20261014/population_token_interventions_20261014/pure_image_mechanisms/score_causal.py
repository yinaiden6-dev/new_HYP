"""Full-gallery causal-arm ranks plus fixed-candidate, fixed-argmax decomposition."""
import sys,time,argparse
from pathlib import Path
import numpy as np,torch
O=Path(__file__).resolve().parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,Ranking
SCOPES=['all','subject_template','subject_visual','background_visual','template']
def masks(item):
 v=item['valid'].numpy();fg=item['foreground'].numpy();t=item['template'].numpy();out=[v,fg|t,fg,v&~fg&~t,t]
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',default='cohort62.json');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--arms');ap.add_argument('--stream',action='store_true');ap.add_argument('--suffix',default='');a=ap.parse_args();torch.set_num_threads(3);torch.cuda.set_device(a.gpu);torch.backends.cuda.matmul.allow_tf32=False
 cohort=read(O/a.cohort);cohort=cohort[:a.limit] if a.limit else cohort;arms=a.arms.split(',') if a.arms else read(O/'protocol.json')['arms'];rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);ri={r:i for i,r in enumerate(rb['ids'])};ranker=Ranking(rb['ids'],rb['sha256'],cohort)
 refs=[r[r.float().norm(dim=1)>0].float() for r in rb['embeddings']];blocks=[]
 for lo in range(0,len(refs),16):
  rr=refs[lo:lo+16];nn=max(map(len,rr));dev='cpu' if a.stream else a.gpu;v=torch.zeros((len(rr),nn,128),device=dev);valid=torch.zeros((len(rr),nn),device=dev,dtype=torch.bool)
  for j,r in enumerate(rr):v[j,:len(r)]=r.to(dev);valid[j,:len(r)]=True
  blocks.append((lo,v.transpose(1,2).contiguous(),valid))
 label=str(len(cohort))+a.suffix;rows=[];pairs=[];stages=[];parities=[];t0=time.time();allout=np.lib.format.open_memmap(O/f'causal_scores_{label}.npy',mode='w+',dtype=np.float32,shape=(len(cohort),len(arms),len(SCOPES),len(refs)))
 for i,r in enumerate(cohort):
  qid=r['query'];orig=torch.load(O/'embeddings'/f'{qid}__original.pt',map_location='cpu',mmap=True,weights_only=True);q0=orig['embedding'].float().to(a.gpu);ms=masks(orig);ww=np.array([m.astype(float)/m.sum() if m.sum() else np.zeros(len(m)) for m in ms]);w=torch.tensor(ww,device=a.gpu,dtype=torch.float32)
  pos=r['image_positive'];negatives=list(dict.fromkeys([r['image_negative'],r['old_center_top1']]));negatives=[n for n in negatives if rb['sha256'][ri[n]] not in r['positive_sha256']];candidates=[pos]+negatives
  original_dots={n:q0@refs[ri[n]].to(a.gpu).T for n in candidates};argmax={n:d.argmax(1) for n,d in original_dots.items()};original_max={n:d.max(1).values for n,d in original_dots.items()}
  for ai,arm in enumerate(arms):
   item=torch.load(O/'embeddings'/f'{qid}__{arm}.pt',map_location='cpu',mmap=True,weights_only=True);q=item['embedding'].float().to(a.gpu);assert q.shape==q0.shape
   for lo,rr,valid in blocks:
    if a.stream:rr=rr.to(a.gpu);valid=valid.to(a.gpu)
    d=torch.matmul(q[None],rr);d.masked_fill_(~valid[:,None,:],-torch.inf);maxs=d.amax(2);allout[i,ai,:,lo:lo+len(rr)]=(w@maxs.T).cpu().numpy()
   for si,s in enumerate(SCOPES):
    ranked=ranker.one(allout[i,ai,si],i) if ms[si].any() else dict(rank=None,top1=None,positive=None,negative=None,margin=None)
    rows.append(dict(query=qid,role=r['role'],package_id=r['package_id'],arm=arm,scope=s,old_image_rank=int(r['image_rank']),old_center_rank=int(r['old_center_rank']),**ranked))
   cur={};locked={};matches={}
   for n in candidates:
    d=q@refs[ri[n]].to(a.gpu).T;cur[n],matches[n]=d.max(1);locked[n]=d[torch.arange(len(q),device=a.gpu),argmax[n]]
   pairarrays={}
   for n in negatives:
    delta=cur[pos]-cur[n];lockdelta=locked[pos]-locked[n];old=original_max[pos]-original_max[n]
    for si,s in enumerate(SCOPES):
     if not ms[si].any():continue
     weight=w[si];free=float(weight@delta);lock=float(weight@lockdelta);oldmargin=float(weight@old)
     pairs.append(dict(query=qid,role=r['role'],arm=arm,scope=s,positive=pos,negative=n,original_margin=oldmargin,free_margin=free,locked_margin=lock,fixed_match_representation_effect=lock-oldmargin,rematching_effect=free-lock,total_margin_change=free-oldmargin,positive_score=float(weight@cur[pos]),negative_score=float(weight@cur[n]),positive_locked=float(weight@locked[pos]),negative_locked=float(weight@locked[n]),positive_changed_match_fraction=float(weight@(matches[pos]!=argmax[pos]).float()),negative_changed_match_fraction=float(weight@(matches[n]!=argmax[n]).float())))
    pairarrays[n+'__delta']=delta.cpu().numpy();pairarrays[n+'__locked_delta']=lockdelta.cpu().numpy();pairarrays[n+'__positive_argmax']=matches[pos].cpu().numpy();pairarrays[n+'__negative_argmax']=matches[n].cpu().numpy()
   np.savez_compressed(O/'traces'/f'{qid}__{arm}__pair.npz',**pairarrays)
   tr=np.load(O/'traces'/f'{qid}__{arm}.npz');visualfg=tr['visual_foreground'];unchanged=~tr['visual_input_changed'];posidx=tr['positions']
   for region,sel in [('subject_unmodified_input',visualfg&unchanged),('subject_changed_input',visualfg&~unchanged),('background_unmodified_input',~visualfg&unchanged),('background_changed_input',~visualfg&~unchanged)]:
    for stage in ['vision','preprojection','embedding']:
     vals=tr[stage+'_cosine_change'] if stage=='vision' else tr[stage+'_cosine_change'][posidx];l2=tr[stage+'_l2_change'] if stage=='vision' else tr[stage+'_l2_change'][posidx]
     stages.append(dict(query=qid,role=r['role'],arm=arm,region=region,stage=stage,n=int(sel.sum()),cosine_change_mean=float(vals[sel].mean()) if sel.any() else None,l2_change_mean=float(l2[sel].mean()) if sel.any() else None,l2_changed_tokens=int((l2[sel]>1e-6).sum())))
  parities.append(dict(orig['record']['parity'],role=r['role'],old_image_rank=int(r['image_rank']),fresh_image_rank=rows[-len(arms)*len(SCOPES)]['rank']))
  if i%5==0:allout.flush();print('SCORED',i+1,len(cohort),round(time.time()-t0,1),flush=True)
 allout.flush();csvwrite(O/f'causal_ranks_{label}.csv',rows);csvwrite(O/f'causal_pair_decomposition_{label}.csv',pairs);csvwrite(O/f'causal_stage_changes_{label}.csv',stages);write(O/f'causal_parity_{label}.json',parities);write(O/f'causal_scoring_done_{label}.json',dict(queries=len(cohort),arms=arms,scopes=SCOPES,rankings=len(rows),seconds=time.time()-t0))
if __name__=='__main__':main()
