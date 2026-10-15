"""Pixel interventions with fresh-original parity, stage traces and locked matches."""
import os,sys,time,json,hashlib,argparse,io,platform,importlib.metadata
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageFilter
R=Path('/mnt/lsdf_zyin');O=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(O.parent)]
from evaluation.encoders import Encoder
from experiment_utils import read,write
from scripts.encode_colnomic_images import token_hash

def raw_box(box,orient,W,H):
 DW,DH=(H,W) if orient in [5,6,7,8] else (W,H)
 x=np.array([box[0],box[2],box[0],box[2]])/DW;y=np.array([box[1],box[1],box[3],box[3]])/DH
 xx,yy={1:(x,y),2:(1-x,y),3:(1-x,1-y),4:(x,1-y),5:(y,x),6:(y,1-x),7:(1-y,1-x),8:(1-y,x)}[orient]
 return [float(xx.min()*W),float(yy.min()*H),float(xx.max()*W),float(yy.max()*H)]
def rect_mask(W,H,b):
 x0,y0,x1,y1=b;return (np.arange(W)[None,:]+.5>=x0)&(np.arange(W)[None,:]+.5<x1)&(np.arange(H)[:,None]+.5>=y0)&(np.arange(H)[:,None]+.5<y1)
def spatial_sample(mask,n,seed):
 # Contiguous runs in a reproducibly shifted raster, not salt-and-pepper noise.
 idx=np.flatnonzero(mask);rng=np.random.default_rng(seed);offset=int(rng.integers(len(idx))) if len(idx) else 0
 out=np.zeros(mask.size,bool);out[np.roll(idx,-offset)[:n]]=True;return out.reshape(mask.shape)
def make_masks(W,H,b,gh,gw,qid):
 subject=rect_mask(W,H,b);padx=2*W/gw;pady=2*H/gh
 expanded=rect_mask(W,H,[b[0]-padx,b[1]-pady,b[2]+padx,b[3]+pady]);far=~expanded
 interior=rect_mask(W,H,[b[0]+W/gw,b[1]+H/gh,b[2]-W/gw,b[3]-H/gh]);fallback=False
 if not interior.any():interior=subject.copy();fallback=True
 n=min(int(far.sum()),int(interior.sum()),int(subject.sum()*.1));seed=int(hashlib.sha256(qid.encode()).hexdigest()[:12],16)
 return dict(far_background_gray=far,far_background_blur=far,all_background_gray=~subject,boundary_ring_gray=expanded&~subject,matched_background_gray=spatial_sample(far,n,seed),matched_subject_gray=spatial_sample(interior,n,seed)),dict(raw_box=b,guard_merged_tokens=2,matched_pixels=n,interior_fallback=fallback,subject_pixels=int(subject.sum()),far_pixels=int(far.sum()))
def features(encoder,im,captured):
 inp=encoder.processor.process_images([im]);host={k:v.detach().cpu().clone() for k,v in inp.items() if isinstance(v,torch.Tensor)}
 att=host['input_ids'][0][host['attention_mask'][0].bool()];pos=torch.where(att==encoder.processor.tokenizer.convert_tokens_to_ids(encoder.processor.image_token))[0]
 out=encoder.encode_prepared(inp)[0][0];v=captured.pop('vision');h=captured.pop('preprojection');assert len(v)==len(pos) and len(h)==len(out)
 return dict(embedding=out,vision=v,preprojection=h,positions=pos,inputs=host)
def verify_merge_order(processor,gh,gw):
 colors=np.zeros((gh,gw,3),np.uint8)
 for y in range(gh):
  for x in range(gw):colors[y,x]=[(x*7)%256,(y*9)%256,((x+y)*5)%256]
 canvas=np.repeat(np.repeat(colors,28,axis=0),28,axis=1);inp=processor.process_images([Image.fromarray(canvas)])
 assert inp['image_grid_thw'][0].tolist()==[1,gh*2,gw*2]
 patch=inp['pixel_values'].reshape(-1,4,3,2,14,14).mean(dim=(1,3,4,5)).numpy()
 ip=processor.image_processor;expect=(colors.reshape(-1,3).astype(np.float32)*ip.rescale_factor-np.array(ip.image_mean))/np.array(ip.image_std)
 err=float(np.abs(patch-expect).max());assert err<1e-5,('merge order mismatch',err)
 return dict(grid=[gh,gw],synthetic_patch_color_validation=True,max_error=err,definition='Four adjacent raw14patches in processor sequence map to one visual28patch, row-major merged order.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--cohort',default='cohort62.json');ap.add_argument('--arms');a=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261014)
 cohort=read(O/a.cohort);cohort=cohort[:a.limit] if a.limit else cohort;cohort=[r for i,r in enumerate(cohort) if i%a.shards==a.shard]
 config=read(R/'configs/baselines.json')['models']['colnomic7b'];(O/'embeddings').mkdir(exist_ok=True);(O/'traces').mkdir(exist_ok=True);(O/'previews').mkdir(exist_ok=True)
 qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True);qi={q:i for i,q in enumerate(qb['ids'])}
 arms=a.arms.split(',') if a.arms else read(O/'protocol.json')['arms'];todo=[r for r in cohort if any(not (O/'embeddings'/f'{r["query"]}__{arm}.pt').exists() for arm in arms)]
 if not todo:print('ALL ALREADY ENCODED',flush=True);return
 runtime=dict(packages={p:importlib.metadata.version(p) for p in ['torch','transformers','colpali-engine','peft','accelerate','Pillow','torchvision']},python=platform.python_version(),gpu_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),model_config=config,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest())
 write(O/f'runtime_shard{a.shard}.json',runtime);t0=time.time();encoder=Encoder('colnomic7b',config,R,'cuda:0')
 from transformers import AutoImageProcessor
 encoder.processor.image_processor=AutoImageProcessor.from_pretrained(R/config['checkpoint'],local_files_only=True,backend='torchvision')
 model=encoder.model.get_base_model();captured={}
 def vh(module,args,out):captured['vision']=out.pooler_output.detach().cpu().clone()
 def ph(module,args):captured['preprojection']=args[0].detach().cpu().clone()[0]
 hooks=[model.visual.register_forward_hook(vh),model.custom_text_proj.register_forward_pre_hook(ph)]
 print('MODEL LOADED',round(time.time()-t0,2),len(todo),'queries',flush=True)
 try:
  for r in todo:
   q=r['query'];path=R/r['image_spec']['image_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_spec']['image_sha256']
   with Image.open(path) as s:orient=int(s.getexif().get(274,1));im=s.convert('RGB')
   W,H=im.size;raw=np.asarray(im);base=features(encoder,im,captured);grid=base['inputs']['image_grid_thw'][0].tolist();gh,gw=grid[1]//2,grid[2]//2
   prior=O/'embeddings'/f'{q}__original.pt'
   if prior.exists():assert torch.equal(torch.load(prior,map_location='cpu',mmap=True,weights_only=True)['embedding'],base['embedding']),'Cross-run original changed'
   checkfile=O/'traces'/f'merge_order_{gh}_{gw}.json'
   if not checkfile.exists():write(checkfile,verify_merge_order(encoder.processor,gh,gw))
   assert gh*gw==len(base['positions']);b=raw_box(r['subject_box_display'],orient,W,H);mm,geom=make_masks(W,H,b,gh,gw,q)
   pos=base['positions'].numpy();valid=base['embedding'].float().norm(dim=1).numpy()>0;v=np.zeros(len(valid),bool);v[pos]=True
   cx=(np.arange(gw)+.5)*W/gw;cy=(np.arange(gh)+.5)*H/gh;fgxy=((cy[:,None]>=b[1])&(cy[:,None]<=b[3])&(cx[None,:]>=b[0])&(cx[None,:]<=b[2])).reshape(-1)
   fg=np.zeros(len(valid),bool);fg[pos]=fgxy;tpl=valid&~v
   old=qb['embeddings'][qi[q]];same=old.shape==base['embedding'].shape
   parity=dict(query=q,same_shape=same,bitwise_equal=bool(same and torch.equal(old,base['embedding'])),max_absolute_error=float((old.float()-base['embedding'].float()).abs().max()) if same else None,mean_cosine=float(torch.nn.functional.cosine_similarity(old.float(),base['embedding'].float(),dim=1).mean()) if same else None)
   write(O/'traces'/f'{q}__parity.json',parity)
   for arm in arms:
    target=O/'embeddings'/f'{q}__{arm}.pt'
    if target.exists():continue
    start=time.time();mask=np.zeros((H,W),bool) if arm in ['original','lossless_sham'] else mm[arm]
    if arm=='original':value=base;arr=raw
    else:
     arr=raw.copy()
     if arm=='lossless_sham':
      buf=io.BytesIO();im.save(buf,format='PNG');buf.seek(0);edit=Image.open(buf).convert('RGB');assert np.array_equal(np.asarray(edit),raw)
     else:
      replacement=np.asarray(im.filter(ImageFilter.GaussianBlur(radius=max(W/gw,H/gh)))) if arm=='far_background_blur' else np.full_like(raw,127)
      arr[mask]=replacement[mask];assert np.array_equal(arr[~mask],raw[~mask]);edit=Image.fromarray(arr)
     value=features(encoder,edit,captured)
    for key in ['input_ids','attention_mask','image_grid_thw']:assert torch.equal(base['inputs'][key],value['inputs'][key]),key
    inputdiff=(base['inputs']['pixel_values']!=value['inputs']['pixel_values']).reshape(-1,base['inputs']['pixel_values'].shape[-1]).any(1)
    # Processor supplies merge-group ordered raw14x14 patches, four per final28x28 visual token.
    visual_changed=inputdiff.reshape(-1,4).any(1).numpy();assert len(visual_changed)==len(pos)
    if arm=='lossless_sham':assert torch.equal(base['embedding'],value['embedding']) and not visual_changed.any(),'Sham repeat is not exact'
    diffs={}
    for stage in ['vision','preprojection','embedding']:
     x=base[stage].float();y=value[stage].float();diffs[stage+'_cosine_change']=(1-torch.nn.functional.cosine_similarity(x,y,dim=1)).numpy();diffs[stage+'_l2_change']=(y-x).norm(dim=1).numpy()
    rec=dict(query=q,role=r['role'],arm=arm,raw_size=[W,H],exif_orientation=orient,image_grid_thw=grid,geometry=geom,mask_pixels=int(mask.sum()),changed_pixels=int(np.any(arr!=raw,axis=2).sum()),raw_rgb_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),input_sha256={k:token_hash(t) for k,t in value['inputs'].items()},stage_sha256={k:token_hash(value[k]) for k in ['vision','preprojection','embedding']},parity=parity,encoder_seconds=time.time()-start)
    torch.save(dict(record=rec,embedding=value['embedding'],positions=value['positions'],valid=torch.from_numpy(valid),foreground=torch.from_numpy(fg),template=torch.from_numpy(tpl),vision=value['vision'] if arm in ['original','far_background_gray','far_background_blur'] else None,preprojection=value['preprojection'] if arm in ['original','far_background_gray','far_background_blur'] else None),target)
    np.savez_compressed(O/'traces'/f'{q}__{arm}.npz',visual_input_changed=visual_changed,visual_foreground=fgxy,positions=pos,**diffs)
    write(O/'traces'/f'{q}__{arm}.json',rec)
    if r['role']=='harm' and arm in ['original','far_background_gray','far_background_blur','boundary_ring_gray','matched_background_gray','matched_subject_gray']:
     preview=Image.fromarray(arr);preview.thumbnail((480,480));preview.save(O/'previews'/f'{q}__{arm}.jpg',quality=88)
    print('ENCODED',a.shard,q,arm,round(time.time()-t0,1),'parity',parity['bitwise_equal'],flush=True)
 finally:
  for h in hooks:h.remove()
  encoder.release()
 write(O/f'encoding_done_{Path(a.cohort).stem}_shard{a.shard}.json',dict(queries=len(cohort),arms=arms,seconds=time.time()-t0,script_sha256=runtime['script_sha256']))
if __name__=='__main__':main()
