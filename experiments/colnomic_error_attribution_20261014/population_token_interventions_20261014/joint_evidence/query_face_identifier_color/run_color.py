"""Fixed-pair color/identifier interventions on model input tensors, no synthetic text."""
from pathlib import Path
import sys,os,time,hashlib,itertools
import numpy as np,torch
from PIL import Image
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent;P=O.parent;R=Path('/mnt/lsdf_zyin')
sys.path[:0]=[str(O/'input_detail_v3'),str(P),str(R)]
from experiment_utils import read,write,csvwrite
from run_input_detail import unpack,pack,prepared
from evaluation.encoders import Encoder
from evaluation.regions import open_photo,crop_region
from transformers import AutoImageProcessor
from scripts.encode_colnomic_images import token_hash

torch.set_num_threads(2);torch.manual_seed(20261015);torch.backends.cuda.matmul.allow_tf32=False
spec=read(D/'protocol.json')['layout_case'];q=spec['query'];refs=[spec['positive'],spec['negative']]
qo=read(R/'cache/ocr/photos.json');ro=read(R/'cache/ocr/references.json')
policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records']
cfg=read(R/'configs/baselines.json')['models']['colnomic7b'];enc=Encoder('colnomic7b',cfg,R,'cuda:0')
enc.processor.image_processor=AutoImageProcessor.from_pretrained(R/cfg['checkpoint'],local_files_only=True,backend='torchvision')
proc=enc.processor;proc.image_processor.size['longest_edge']=602112
mean=torch.tensor(proc.image_processor.image_mean)[:,None,None];std=torch.tensor(proc.image_processor.image_std)[:,None,None]
(D/'color_embeddings').mkdir(exist_ok=True);(D/'color_previews').mkdir(exist_ok=True)
def encode(inp):return enc.encode_prepared(prepared(inp))[0][0]
def rgb(inp):return unpack(inp['pixel_values'],inp['image_grid_thw'][0].tolist())*std+mean
def prepared_rgb(base,pixels):
 t=prepared(base);t['pixel_values']=pack((pixels-mean)/std).reshape_as(base['pixel_values']);return t
def gray(pixels):return (pixels*torch.tensor([.2126,.7152,.0722])[:,None,None]).sum(0,keepdim=True).expand_as(pixels).clone()
def box_pixels(box,size):
 W,H=size;return tuple(round(v*[W/978,H/1358,W/978,H/1358][i]) for i,v in enumerate(box))
def hide(inp,box):
 x,y,X,Y=box_pixels(box,(inp.shape[2],inp.shape[1]));p=inp[:,y:Y,x:X];bg=inp[:,max(0,y-8):y,x:X].reshape(3,-1).median(1).values
 inp[:,y:Y,x:X]=bg[:,None,None]
def side_palette(pixels):
 x,y,X,Y=box_pixels((55,430,145,890),(pixels.shape[2],pixels.shape[1]));p=pixels[:,y:Y,x:X].reshape(3,-1)
 light=p.mean(0);return p[:,light<=torch.quantile(light,.5)].median(1).values,p[:,light>=torch.quantile(light,.99)].median(1).values
def recolor_side(pixels,source,target):
 result=pixels.clone();bg,fg=source;BG,FG=target;direction=(fg-bg);den=direction.square().sum()
 for box in [(26,335,181,1002),(450,335,607,1002)]:
  x,y,X,Y=box_pixels(box,(pixels.shape[2],pixels.shape[1]));part=pixels[:,y:Y,x:X]
  alpha=((part-bg[:,None,None])*direction[:,None,None]).sum(0)/den
  result[:,y:Y,x:X]=BG[:,None,None]+alpha.clamp(0,1)[None]*(FG-BG)[:,None,None]
 return result
try:
 qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True)
 oldq=qb['embeddings'][qb['ids'].index(q)];baseinputs={};embs={};records={};pixels={}
 for rid in refs:
  with Image.open(R/ro[rid]['image_path']) as src:im=src.convert('RGB')
  baseinputs[rid]=proc.process_images([im]);pixels[rid]=rgb(baseinputs[rid])
 palettes={r:side_palette(pixels[r]) for r in refs}
 # Audit side-panel alignment using per-image normalized contrast, not absolute color.
 masks=[];layout=[]
 for rid in refs:
  p=pixels[rid];x,y,X,Y=box_pixels((55,430,145,890),(p.shape[2],p.shape[1]));bg,fg=palettes[rid];v=fg-bg
  alpha=((p[:,y:Y,x:X]-bg[:,None,None])*v[:,None,None]).sum(0)/v.square().sum();m=alpha>.5;masks.append(m)
  yy,xx=torch.where(m);layout.append(dict(reference=rid,glyph_bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],ink_pixels=int(m.sum())))
 layout_iou=float((masks[0]&masks[1]).sum()/(masks[0]|masks[1]).sum())
 for ix,rid in enumerate(refs):
  base=baseinputs[rid];original=encode(base);again=encode(prepared_rgb(base,pixels[rid]));repack_max=float((again-original).abs().max())
  # Exact raw pack roundtrip is the no-op control; RGB inverse may round the last bit.
  raw=unpack(base['pixel_values'],base['image_grid_thw'][0].tolist());roundtrip=prepared(base);roundtrip['pixel_values']=pack(raw).reshape_as(base['pixel_values']);assert torch.equal(roundtrip['pixel_values'],base['pixel_values']);assert torch.equal(encode(roundtrip),original)
  rec=dict(repeated_bitwise_equal=True,rgb_roundtrip_embedding_max_difference=repack_max,grid=base['image_grid_thw'].tolist(),arms={})
  for color,identifier in itertools.product(['original','gray','swap_side_colors'],['visible','hidden']):
   arm=color+'__'+identifier;p=pixels[rid].clone()
   if color=='gray':p=gray(p)
   elif color=='swap_side_colors':p=recolor_side(p,palettes[rid],palettes[refs[1-ix]])
   if identifier=='hidden':
    hide(p,(240,1020,390,1110));hide(p,(700,115,785,142))
   inp=base if arm=='original__visible' else prepared_rgb(base,p)
   e=original if arm=='original__visible' else encode(inp);embs[rid,arm]=e
   rec['arms'][arm]=dict(embedding_hash=token_hash(e),input_hash=token_hash(inp['pixel_values']),changed_input_values=int((inp['pixel_values']!=base['pixel_values']).sum()))
   torch.save(dict(embedding=e),D/'color_embeddings'/f'{rid}__{arm}.pt')
   Image.fromarray((p.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).save(D/'color_previews'/f'{rid}__{arm}.png')
   print('REF',rid,arm,flush=True)
  sham=pixels[rid].clone();hide(sham,(240,1170,390,1260));si=prepared_rgb(base,sham);embs[rid,'blank_sham']=encode(si)
  rec['arms']['blank_sham']=dict(embedding_hash=token_hash(embs[rid,'blank_sham']),changed_input_values=int((si['pixel_values']!=base['pixel_values']).sum()))
  records[rid]=rec
 with Image.open(R/qo[q]['image_path']) as src:rawphoto=src.convert('RGB')
 crop,bounds=crop_region(open_photo(R/qo[q]['image_path']),boxes[q]['bbox'],margin=.05)
 qemb={};qrecords={}
 for view,im in [('full',rawphoto),('human_crop',crop)]:
  inp=proc.process_images([im]);e=encode(inp);qemb[view,'original']=e;qemb[view,'gray']=encode(prepared_rgb(inp,gray(rgb(inp))))
  qrecords[view]=dict(grid=inp['image_grid_thw'].tolist(),output_hash=token_hash(e),historical_query_equal=torch.equal(e,oldq) if view=='full' else None)
  if view=='full':assert torch.equal(e,oldq),'Original query does not reproduce cache'
 results=[];arrays={}
 for (view,qcolor),qe in qemb.items():
  for arm in list(records[refs[0]]['arms']):
   cp=(qe.float()@embs[refs[0],arm].float().T).max(1).values;cn=(qe.float()@embs[refs[1],arm].float().T).max(1).values
   delta=cp-cn;key=view+'__'+qcolor+'__'+arm;arrays[key]=delta.numpy()
   results.append(dict(query=q,view=view,query_color=qcolor,reference_arm=arm,positive_score=float(cp.mean()),negative_score=float(cn.mean()),margin=float(delta.mean()),positive_token_fraction=float((delta>0).float().mean()),tokens=len(delta)))
 csvwrite(D/'color_pairs.csv',results);np.savez_compressed(D/'color_token_deltas.npz',**arrays)
 write(D/'COLOR_SUMMARY.json',dict(query=q,positive=refs[0],negative=refs[1],user_correction=spec['user_correction'],layout_audit=layout,normalized_side_text_mask_iou=layout_iou,reference_records=records,query_records=qrecords,subject_box=boxes[q],crop_bounds=bounds,results=results,limits=['Fixed-pair diagnostic, not a deployable full-gallery repair.','Reference side color swap preserves per-pixel text alpha and position; other package faces are unchanged.','Grayscale also changes contrast; interpret alongside side-color swap.','Reference identifier hiding includes barcode and printed code; remaining variant text on unseen faces is not removed.']))
 print('COLOR DONE',flush=True)
finally:enc.release()
