"""Context control: re-encode only corresponding side faces after fixed geometric rectification."""
from pathlib import Path
import sys
import numpy as np,torch
from PIL import Image,ImageOps
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path[:0]=[str(P),str(R)]
from experiment_utils import read,write,csvwrite
from evaluation.encoders import Encoder
from evaluation.regions import open_photo,crop_region
from transformers import AutoImageProcessor
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
s=read(D/'COLOR_SUMMARY.json');reg=read(D/'COLOR_REGIONS.json');q=s['query'];refs=[s['positive'],s['negative']];qo=read(R/'cache/ocr/photos.json');ro=read(R/'cache/ocr/references.json')
crop,bounds=crop_region(open_photo(R/qo[q]['image_path']),s['subject_box']['bbox'],margin=.05)
pts=np.array(reg['polygon_in_margin_crop'],np.float32);target=np.array([[0,0],[154,0],[154,666],[0,666]],np.float32)
matrix=[];values=[]
for (u,v),(x,y) in zip(target,pts):
 matrix.extend([[u,v,1,0,0,0,-x*u,-x*v],[0,0,0,u,v,1,-y*u,-y*v]]);values.extend([x,y])
coeff=np.linalg.solve(np.array(matrix,dtype=float),np.array(values,dtype=float));homography=np.linalg.inv(np.r_[coeff,1].reshape(3,3))
face=crop.transform((155,667),Image.Transform.PERSPECTIVE,tuple(coeff),resample=Image.Resampling.BICUBIC)
inputs={'query_face_color':face,'query_face_gray':ImageOps.grayscale(face).convert('RGB')}
for rid in refs:
 im=Image.open(R/ro[rid]['image_path']).convert('RGB')
 for side,box in [('left',(26,335,181,1002)),('right',(450,335,605,1002))]:
  f=im.crop(box);inputs[rid+'_'+side+'_color']=f;inputs[rid+'_'+side+'_gray']=ImageOps.grayscale(f).convert('RGB')
for name,im in inputs.items():im.save(D/'color_previews'/('face_'+name+'.png'))
cfg=read(R/'configs/baselines.json')['models']['colnomic7b'];enc=Encoder('colnomic7b',cfg,R,'cuda:0');enc.processor.image_processor=AutoImageProcessor.from_pretrained(R/cfg['checkpoint'],local_files_only=True,backend='torchvision');enc.processor.image_processor.size['longest_edge']=602112
emb={};metadata={}
try:
 for name,im in inputs.items():
  inp=enc.processor.process_images([im]);e=enc.encode_prepared(inp)[0][0];emb[name]=e.float();metadata[name]=dict(tokens=len(e),size=im.size,median_rgb=np.median(np.asarray(im).reshape(-1,3),axis=0).tolist());torch.save(dict(embedding=e),D/'color_embeddings'/('face_'+name+'.pt'));print('FACE',name,flush=True)
 cb=torch.load(R/'cache/foreground/human_boxes/0a7a1ed078cd491c/photo_embeddings.pt',map_location='cpu',mmap=True,weights_only=True);emb['query_existing_crop']=cb['embeddings'][cb['ids'].index(q)].float()
 rows=[]
 for qmode in ['query_existing_crop','query_face_color','query_face_gray']:
  for side in ['left','right']:
   for color in ['color','gray']:
    p=emb[qmode]@emb[refs[0]+'_'+side+'_'+color].T;n=emb[qmode]@emb[refs[1]+'_'+side+'_'+color].T
    rows.append(dict(query_mode=qmode,reference_side=side,reference_color=color,margin=float((p.max(1).values-n.max(1).values).mean())))
 csvwrite(D/'isolated_face_scores.csv',rows);write(D/'ISOLATED_FACES.json',dict(query=q,polygon=pts.tolist(),target_size=[155,667],homography=homography.tolist(),metadata=metadata,results=rows,scope='Manual same-face crop and rectification diagnostic. Identifiers and other faces absent; no automatic face selector or population-level repair claimed.'))
finally:enc.release()
