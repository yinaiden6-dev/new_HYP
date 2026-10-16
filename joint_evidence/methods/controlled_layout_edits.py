"""Deterministic glyph relocation with common background-edit sham and held identities."""
import re,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,rows,csvout,sha


def prepare_source(path,record,spec,line_index):
 with Image.open(path) as image:
  if int(image.getexif().get(274,1))!=1:raise ValueError('Reference EXIF orientation must be 1')
  a=np.asarray(image.convert('RGB')).copy()
 line=record['lines'][line_index];ox1,oy1,ox2,oy2=np.rint(line['bbox']).astype(int)
 px1=max(0,ox1-8);py1=max(0,oy1-8);px2=min(a.shape[1],ox2+8);py2=min(a.shape[0],oy2+8)
 local=a[py1:py2,px1:px2];binary=(local.min(2)<245).astype(np.uint8)
 n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
 selected=np.zeros(binary.shape,bool)
 for i in range(1,n):
  x,y=centers[i]+[px1,py1]
  if ox1-2<=x<=ox2+2 and oy1-2<=y<=oy2+2 and stats[i,cv2.CC_STAT_AREA]>=2:selected|=labels==i
 if selected.sum()<20:raise ValueError('No reliable glyph')
 ys,xs=np.nonzero(selected)
 if min(xs.min(),ys.min(),selected.shape[1]-1-xs.max(),selected.shape[0]-1-ys.max())<1:raise ValueError('Glyph exceeds expanded OCR neighborhood')
 x1=px1+xs.min()-1;x2=px1+xs.max()+2;y1=py1+ys.min()-1;y2=py1+ys.max()+2
 foreground=selected[y1-py1:y2-py1,x1-px1:x2-px1]
 previous=[l for l in record['lines'] if len(re.findall(r'\w+',l['text']))>=3 and
  0<y1-l['bbox'][3]<5*(y2-y1) and abs(l['bbox'][0]-x1)<.025*a.shape[1]]
 if not previous:raise ValueError('No preceding paragraph line')
 previous=min(previous,key=lambda l:y1-l['bbox'][3])
 step=a.shape[1]/spec['layout']['width'];shifts=[int(round(k*step)) for k in [1,2,3,4]]
 band_right=x2+shifts[-1]
 if band_right>min(a.shape[1],previous['bbox'][2]+4):raise ValueError('Insufficient paragraph whitespace')
 band=a[y1:y2,x1:band_right];outside=band[:,x2-x1:]
 if np.any(outside.min(2)<245):raise ValueError('Destination overlaps another mark or glyph')
 polygon=np.asarray(line['polygon']);edges=np.roll(polygon,-1,axis=0)-polygon;edge=edges[np.linalg.norm(edges,axis=1).argmax()]
 if abs(edge[1])>.08*np.linalg.norm(edge):raise ValueError('Rotated word')
 return a,(x1,y1,x2,y2),foreground,shifts,step


def render(a,bbox,foreground,shifts):
 x1,y1,x2,y2=bbox;right=x2+max(shifts);base=a.copy()
 # All sham and relocated arms share this exact background intervention.
 band=base[y1:y2,x1:right];band[band.min(2)>=245]=255
 source_region=base[y1:y2,x1:x2];source_region[foreground]=255
 glyph=a[y1:y2,x1:x2].copy();variants={}
 for label,shift in [('sham',0)]+[('shift_'+str(k),s) for k,s in zip([1,2,3,4],shifts)]:
  b=base.copy();region=b[y1:y2,x1+shift:x2+shift];region[foreground]=glyph[foreground]
  assert np.array_equal(region[foreground],glyph[foreground])
  changed=(b!=a).any(2);allowed=np.zeros(a.shape[:2],bool);allowed[y1:y2,x1:right]=True
  assert not np.any(changed&~allowed)
  variants[label]=b
 sham=variants['sham'];sham_colors,sham_counts=np.unique(sham[y1:y2,x1:right].reshape(-1,3),axis=0,return_counts=True)
 for b in variants.values():
  colors,counts=np.unique(b[y1:y2,x1:right].reshape(-1,3),axis=0,return_counts=True)
  assert np.array_equal(colors,sham_colors) and np.array_equal(counts,sham_counts)
 return variants


def main():
 folder=OUT/'stage4_interventions';images=folder/'images';images.mkdir(parents=True,exist_ok=True)
 refs={r['reference_id']:r for r in rows(ROOT/'dataset/lists/references.csv')}
 meta={r['id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['references']}
 ro=read(ROOT/'cache/ocr/references.json');photos=rows(ROOT/'dataset/lists/photos.csv')
 package_splits={r['package_id']:s for s in ['train','val','test'] for r in rows(ROOT/f'dataset/lists/{s}.csv')}
 positives={r['package_id']:r['reference_id'] for r in rows(ROOT/'dataset/lists/positives.csv')}
 selected=[];rejected=[]
 # Select sources by OCR/whitespace, never by frozen embedding responses or retrieval scores.
 for split in ['train','val','test']:
  count=0
  for package,rid in sorted(positives.items()):
   if package_splits[package]!=split:continue
   path=ROOT/'dataset'/refs[rid]['path']
   for line_index,line in enumerate(ro[rid]['lines']):
    if not re.fullmatch(r'[A-Za-z]{5,20}[.!]',line['text'].strip()):continue
    try:prepared=prepare_source(path,ro[rid],meta[rid],line_index)
    except ValueError as error:rejected.append(dict(reference_id=rid,line=line_index,reason=str(error)));continue
    selected.append(dict(reference_id=rid,package_id=package,split=split,line_index=line_index,text=line['text'],prepared=prepared,path=path))
    count+=1;break
   if count==2:break
  print('Layout sources',split,count,flush=True)
 # User-selected details case is an external causal focal case, excluded from probe fitting/selection.
 rid='reference_000233';line_index=60;prepared=prepare_source(ROOT/'dataset'/refs[rid]['path'],ro[rid],meta[rid],line_index)
 selected.append(dict(reference_id=rid,package_id='external_reference_000233',split='focal',line_index=line_index,text='details.',prepared=prepared,path=ROOT/'dataset'/refs[rid]['path']))
 manifest=[dict(id='reference_000229_fresh_base',path=str(ROOT/'dataset'/refs['reference_000229']['path']),source_id='reference_000229',arm='base',split='focal')]
 validations=[];source_records=[]
 for item in selected:
  rid=item['reference_id'];a,bbox,foreground,shifts,step=item['prepared'];variants=render(a,bbox,foreground,shifts)
  records=[dict(id=rid+'_fresh_base',path=str(item['path']),source_id=rid,arm='base',split=item['split'],package_id=item['package_id'])]
  canvas=Image.new('RGB',(600,180*len(variants)),'white');draw=ImageDraw.Draw(canvas)
  for k,(arm,b) in enumerate(variants.items()):
   path=images/(rid+'_'+arm+'.png');Image.fromarray(b).save(path)
   shift=0 if arm=='sham' else shifts[int(arm[-1])-1]
   x1,y1,x2,y2=bbox;word_bbox=[int(x1+shift),int(y1),int(x2+shift),int(y2)]
   records.append(dict(id=rid+'_'+arm,path=str(path),source_id=rid,arm=arm,split=item['split'],package_id=item['package_id'],
    word_bbox=word_bbox,layout_class=None if arm=='sham' else int(int(arm[-1])>=3)))
   zoom=Image.fromarray(b).crop((max(0,x1-5),max(0,y1-35),min(a.shape[1],x2+max(shifts)+5),min(a.shape[0],y2+35)))
   zoom.thumbnail((580,145));canvas.paste(zoom,(10,k*180+25));draw.text((10,k*180+5),arm,fill='black')
  canvas.save(folder/(rid+'_controlled_layouts.png'))
  manifest.extend(records)
  validations.append(dict(reference_id=rid,word=item['text'],source_bbox=[int(v) for v in bbox],pixel_shifts=shifts,raw_patch_width=step,
   all_pixels_outside_common_edit_band_unchanged=True,foreground_glyph_RGB_pixels_identical=True,
   sham_and_all_layout_arms_have_identical_RGB_histograms=True,other_text_color_capacity_unchanged=True,
   glyph_foreground_pixels=int(foreground.sum()),source_image_sha256=sha(item['path']),
   layout_classes='shift1/2 vs shift3/4; every class arm edited, sham separate'))
  source_records.append({k:v for k,v in item.items() if k not in ['prepared','path']})
 # Fresh query is used to audit device/encoding drift, cached query remains primary for fixed-gallery tests.
 qid='photo_41de3eb383934ec3';photo=next(r for r in photos if r['photo_id']==qid)
 manifest.append(dict(id=qid+'_fresh_base',path=str(ROOT/'dataset'/photo['path']),source_id=qid,arm='base',split='focal'))
 write(folder/'sources.json',source_records);write(folder/'pixel_validation.json',validations);csvout(folder/'rejected_sources.csv',rejected)
 write(folder/'encoding_manifest.json',manifest)
 # The already-loading worker reads this manifest after model initialization.
 write(OUT/'encoding_profile_manifest.json',manifest)
 print('Controlled layout images ready, total encodings',len(manifest),flush=True)

if __name__=='__main__':main()
