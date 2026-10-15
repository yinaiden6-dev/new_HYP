"""Independent CPU score replay and trace/processor checks; does not encode a model."""
import sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageFilter
O=Path(__file__).resolve().parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write
torch.set_num_threads(2)
def th(t):
 t=t.detach().cpu().contiguous();return hashlib.sha256(str(t.dtype).encode()+json.dumps(list(t.shape),separators=(',',':')).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()
def main(tag):
 start=time.time();n=int(tag.split('_')[0]);cohort=read(O/f'cohort{n}.json');done=read(O/f'causal_scoring_done_{tag}.json');arms=done['arms'];scopes=done['scopes'];scores=np.load(O/f'causal_scores_{tag}.npy',mmap_mode='r')
 qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True);rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);qi={q:i for i,q in enumerate(qb['ids'])};ri={q:i for i,q in enumerate(rb['ids'])};parity=sham=unchanged=0
 for r in cohort:
  q=r['query'];b=torch.load(O/'embeddings'/f'{q}__original.pt',map_location='cpu',mmap=True,weights_only=True);s=torch.load(O/'embeddings'/f'{q}__lossless_sham.pt',map_location='cpu',mmap=True,weights_only=True);assert torch.equal(b['embedding'],qb['embeddings'][qi[q]]) and torch.equal(b['embedding'],s['embedding']);parity+=1;sham+=1
  for arm in ['far_background_gray','far_background_blur']:
   tr=np.load(O/'traces'/f'{q}__{arm}.npz');assert not np.any(tr['visual_input_changed']&tr['visual_foreground']);unchanged+=1
  if n==62 and 'matched_subject_gray' in arms:
   a=read(O/'traces'/f'{q}__matched_background_gray.json');d=read(O/'traces'/f'{q}__matched_subject_gray.json');assert a['mask_pixels']==d['mask_pixels']
 print('PARITY',n,flush=True)
 checks=[];maxerr=0
 for i in sorted(set([0,1,2,n//3,n//2,n-1])):
  r=cohort[i];q=r['query'];b=torch.load(O/'embeddings'/f'{q}__original.pt',map_location='cpu',mmap=True,weights_only=True);v=b['valid'].numpy();f=b['foreground'].numpy();t=b['template'].numpy();masks=[v,f|t,f,v&~f&~t,t]
  for ai,arm in enumerate(arms):
   item=torch.load(O/'embeddings'/f'{q}__{arm}.pt',map_location='cpu',mmap=True,weights_only=True);qq=item['embedding'].double()
   for j in sorted(set([ri[r['image_positive']],ri[r['image_negative']],ri[r['old_center_top1']],(i*997+13)%len(ri)])):
    rr=rb['embeddings'][j];rr=rr[rr.float().norm(dim=1)>0].double();best=(qq@rr.T).max(1).values.numpy()
    for si,mask in enumerate(masks):
     if not mask.any():continue
     expected=float(best[mask].mean());actual=float(scores[i,ai,si,j]);err=abs(expected-actual);assert err<2e-6,(q,arm,j,si,err);maxerr=max(maxerr,err);checks.append(err)
 print('SCORES',len(checks),maxerr,flush=True)
 # Rebuild actual processor inputs on a subset with a separately written pixel edit.
 from colpali_engine.models import ColQwen2_5_Processor
 from transformers import AutoImageProcessor
 config=read(R/'configs/baselines.json')['models']['colnomic7b'];processor=ColQwen2_5_Processor.from_pretrained(R/config['checkpoint'],local_files_only=True);processor.image_processor=AutoImageProcessor.from_pretrained(R/config['checkpoint'],local_files_only=True,backend='torchvision');pixelchecks=[]
 for i in sorted(set([0,1,2,n//2,n-1])):
  r=cohort[i];q=r['query'];record=read(O/'traces'/f'{q}__original.json');im=Image.open(R/r['image_spec']['image_path']).convert('RGB');a=np.array(im);H,W=a.shape[:2];gh,gw=np.array(record['image_grid_thw'][1:])//2;x0,y0,x1,y1=record['geometry']['raw_box'];xx=np.arange(W)+.5;yy=np.arange(H)+.5
  far=~((xx[None,:]>=x0-2*W/gw)&(xx[None,:]<x1+2*W/gw)&(yy[:,None]>=y0-2*H/gh)&(yy[:,None]<y1+2*H/gh));base=processor.process_images([im])
  for arm in ['original','far_background_gray','far_background_blur']:
   out=a.copy()
   if arm!='original':
    replace=np.array(im.filter(ImageFilter.GaussianBlur(float(max(W/gw,H/gh))))) if arm.endswith('blur') else np.full_like(a,127);out[far]=replace[far]
   inp=processor.process_images([Image.fromarray(out)]);rec=read(O/'traces'/f'{q}__{arm}.json');assert hashlib.sha256(out.tobytes()).hexdigest()==rec['raw_rgb_sha256']
   for k,t in inp.items():
    if isinstance(t,torch.Tensor):assert th(t)==rec['input_sha256'][k],(q,arm,k)
   pv=inp['pixel_values'];changed=(base['pixel_values']!=pv).reshape(-1,pv.shape[-1]).any(1).reshape(-1,4).any(1).numpy();tr=np.load(O/'traces'/f'{q}__{arm}.npz');assert np.array_equal(changed,tr['visual_input_changed']),(q,arm,changed.shape,tr['visual_input_changed'].shape);pixelchecks.append(dict(query=q,arm=arm,changed_subject_patches=int((changed&tr['visual_foreground']).sum())))
 write(O/f'CAUSAL_VALIDATION_{tag}.json',dict(status='passed',original_old_bitwise=parity,sham_bitwise=sham,unchanged_foreground_arm_checks=unchanged,independent_float64_score_values=len(checks),max_absolute_score_error=maxerr,independent_pixel_processor_checks=pixelchecks,seconds=time.time()-start));print('VALIDATED',tag,flush=True)
if __name__=='__main__':main(sys.argv[1])
