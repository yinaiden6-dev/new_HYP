"""Frozen-feature, visible-face correspondence diagnostic; manual polygon is explicit."""
from pathlib import Path
import sys
import numpy as np,torch
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path[:0]=[str(P),str(R)]
from experiment_utils import read,write,csvwrite
from evaluation.regions import open_photo,crop_region
from colpali_engine.models import ColQwen2_5_Processor
from transformers import AutoImageProcessor
torch.set_num_threads(2)
q='photo_fbc7b6d852b1125b';refs=['reference_003333','reference_003331'];s=read(D/'COLOR_SUMMARY.json');cfg=read(R/'configs/baselines.json')['models']['colnomic7b']
proc=ColQwen2_5_Processor.from_pretrained(R/cfg['checkpoint'],local_files_only=True);proc.image_processor=AutoImageProcessor.from_pretrained(R/cfg['checkpoint'],local_files_only=True,backend='torchvision');proc.image_processor.size['longest_edge']=602112
ro=read(R/'cache/ocr/references.json');qo=read(R/'cache/ocr/photos.json');b=s['subject_box']['bbox'];crop,bounds=crop_region(open_photo(R/qo[q]['image_path']),b,margin=.05)
# Polygon reviewed on the original-resolution tight human crop. It covers the broad colored side and excludes the lower barcode face.
localpoly=[(295,128),(660,183),(430,1610),(10,1550)]
poly=[(x+b[0]-bounds[0],y+b[1]-bounds[1]) for x,y in localpoly]
def tokens(im):
 inp=proc.process_images([im]);ids=inp['input_ids'][0][inp['attention_mask'][0].bool()];pos=torch.where(ids==proc.tokenizer.convert_tokens_to_ids(proc.image_token))[0]
 h,w=inp['image_grid_thw'][0][1:].tolist();h//=2;w//=2
 xx,yy=np.meshgrid((np.arange(w)+.5)*im.width/w,(np.arange(h)+.5)*im.height/h)
 return len(ids),pos.numpy(),xx.ravel(),yy.ravel()
def within(x,y,polygon):
 yes=np.zeros(x.shape,bool)
 for i in range(len(polygon)):
  xa,ya=polygon[i];xb,yb=polygon[(i+1)%len(polygon)]
  yes^=((ya>y)!=(yb>y))&(x<(xb-xa)*(y-ya)/(yb-ya+1e-20)+xa)
 return yes
N,pos,x,y=tokens(crop);side=np.zeros(N,bool);side[pos]=within(x,y,poly);visual=np.zeros(N,bool);visual[pos]=True
cb=torch.load(R/'cache/foreground/human_boxes/0a7a1ed078cd491c/photo_embeddings.pt',map_location='cpu',mmap=True,weights_only=True);qe=cb['embeddings'][cb['ids'].index(q)].float();assert len(qe)==N
from scripts.encode_colnomic_images import token_hash
assert token_hash(cb['embeddings'][cb['ids'].index(q)])==s['query_records']['human_crop']['output_hash']
groups={'all':np.ones(N,bool),'visible_colored_side':side,'other_visual':visual&~side,'template':~visual}
refgroups={};emb={}
for rid in refs:
 im=Image.open(R/ro[rid]['image_path']).convert('RGB');n,p,xx,yy=tokens(im)
 left=(xx>26)&(xx<181)&(yy>335)&(yy<1002);right=(xx>450)&(xx<607)&(yy>335)&(yy<1002)
 code=(xx>240)&(xx<390)&(yy>1020)&(yy<1110)
 masks={'all':np.ones(n,bool)}
 for name,mask in [('same_visible_side',left|right),('identifier_face',code),('left_side',left)]:
  masks[name]=np.zeros(n,bool);masks[name][p]=mask
 masks['other_faces_and_template']=~masks['same_visible_side']&~masks['identifier_face'];refgroups[rid]=masks
 for arm in ['original__visible','original__hidden','gray__visible','swap_side_colors__visible']:
  emb[rid,arm]=torch.load(D/'color_embeddings'/f'{rid}__{arm}.pt',map_location='cpu',weights_only=True)['embedding'].float()
rows=[];matches=[];deltas={}
for arm in ['original__visible','original__hidden','gray__visible','swap_side_colors__visible']:
 for refread in ['all','same_visible_side','left_side']:
  vals=[]
  for rid in refs:
   ix=np.flatnonzero(refgroups[rid][refread]);val,arg=(qe@emb[rid,arm][ix].T).max(1);vals.append(val)
   if refread=='all':
    for qgroup,mask in groups.items():
     mm=ix[arg.numpy()[mask]]
     matches.append(dict(reference=rid,arm=arm,query_group=qgroup,tokens=int(mask.sum()),matched_same_side_fraction=float(refgroups[rid]['same_visible_side'][mm].mean()),matched_identifier_fraction=float(refgroups[rid]['identifier_face'][mm].mean()),matched_other_fraction=float(refgroups[rid]['other_faces_and_template'][mm].mean())))
  delta=(vals[0]-vals[1]).numpy();deltas[arm+'__'+refread]=delta
  for group,mask in groups.items():rows.append(dict(arm=arm,reference_readout=refread,query_readout=group,tokens=int(mask.sum()),margin=float(delta[mask].mean()),contribution_to_all_mean=float(delta[mask].sum()/len(delta))))
csvwrite(D/'color_region_scores.csv',rows);csvwrite(D/'color_region_matches.csv',matches);np.savez_compressed(D/'color_region_deltas.npz',**deltas,query_side_mask=side,query_visual_mask=visual)
preview=crop.copy();ImageDraw.Draw(preview).polygon(poly,outline='red',width=5);preview.save(D/'color_previews/verified_query_side.png')
write(D/'COLOR_REGIONS.json',dict(query=q,polygon_in_margin_crop=poly,polygon_in_tight_human_crop=localpoly,query_side_tokens=int(side.sum()),query_total_tokens=N,results=rows,matches=matches,scope='Manual visible-face diagnostic only; no test-truth-driven face selector is claimed as deployable. Frozen feature filtering is not equivalent to pixel isolation.'))
print('REGION TRACE DONE',flush=True)
