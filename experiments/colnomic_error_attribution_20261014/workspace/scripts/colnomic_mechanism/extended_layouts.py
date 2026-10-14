"""All eligible identities: move an isolated word in ORIGINAL generated photos.

Sources are selected by provenance, OCR, confirmed subject bounds and blank pixels,
never by baseline correctness or by the user-selected inspection photographs.
"""
import sys,re,math
from pathlib import Path
from collections import Counter
import numpy as np,torch,cv2
from PIL import Image,ImageOps,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,read,write,sha,csvout

INVERSE={1:None,2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,
 4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_90,
 7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_270}


def to_raw(array,orientation):
 image=Image.fromarray(array)
 if INVERSE[orientation] is not None:image=image.transpose(INVERSE[orientation])
 return np.asarray(image).copy()


def raw_bbox(display_bbox,display_shape,orientation):
 # Integer rectangle masks make EXIF inverse accounting exact, including mirrored cases.
 x1,y1,x2,y2=display_bbox;mask=np.zeros(display_shape[:2],np.uint8);mask[y1:y2,x1:x2]=255
 raw=to_raw(np.repeat(mask[:,:,None],3,2),orientation)[:,:,0];ys,xs=np.nonzero(raw)
 return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]


def prepare(a,line,spec,box,other_lines,max_tilt=.08):
 if not re.fullmatch(r'[A-Za-z]{4,20}[.!]?',line['text'].strip()):raise ValueError('not an isolated alphabetic word')
 if float(line['confidence'])<.95:raise ValueError('low OCR confidence')
 polygon=np.asarray(line['polygon']);center=polygon.mean(0)
 if not ((center>=box[:2])&(center<box[2:])).all():raise ValueError('outside confirmed subject')
 edges=np.roll(polygon,-1,axis=0)-polygon;edge=edges[np.linalg.norm(edges,axis=1).argmax()]
 if abs(edge[1])>max_tilt*np.linalg.norm(edge):raise ValueError('word not horizontal in displayed frame')
 ox1,oy1,ox2,oy2=np.rint(line['bbox']).astype(int);px1=max(0,ox1-8);py1=max(0,oy1-8);px2=min(a.shape[1],ox2+8);py2=min(a.shape[0],oy2+8)
 local=a[py1:py2,px1:px2];gray=cv2.cvtColor(local,cv2.COLOR_RGB2GRAY)
 threshold,binary=cv2.threshold(gray,0,1,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
 border=np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]
 if np.median(border)<threshold:binary=1-binary
 n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8);selected=np.zeros(binary.shape,bool)
 for i in range(1,n):
  x,y=centers[i]+[px1,py1]
  if ox1-2<=x<=ox2+2 and oy1-2<=y<=oy2+2 and stats[i,cv2.CC_STAT_AREA]>=2:selected|=labels==i
 if selected.sum()<20:raise ValueError('unreliable foreground')
 ys,xs=np.nonzero(selected)
 if min(xs.min(),ys.min(),selected.shape[1]-1-xs.max(),selected.shape[0]-1-ys.max())<1:raise ValueError('glyph crosses OCR neighborhood')
 x1=max(0,px1+xs.min()-3);x2=min(a.shape[1],px1+xs.max()+4);y1=max(0,py1+ys.min()-3);y2=min(a.shape[0],py1+ys.max()+4)
 if y2-y1<10:raise ValueError('word too small')
 l=spec['layout']
 step=spec['raw_height']/l['height'] if spec['exif_orientation']>=5 else spec['raw_width']/l['width']
 # Two edited classes; one/two patch widths. Unlike v1, no four-patch whitespace requirement.
 shifts=[max(1,int(round(.25*step))),max(2,int(round(.75*step)))];right=x2+max(shifts)
 if right>min(a.shape[1],int(box[2])):raise ValueError('insufficient subject whitespace')
 destination=a[y1:y2,x2:right].reshape(-1,3);low=np.quantile(destination,.01,axis=0);high=np.quantile(destination,.99,axis=0);background=np.rint(np.median(destination,axis=0)).astype(np.uint8)
 if np.any(high-low>24):raise ValueError('destination not uniform paper')
 if np.any(np.abs(destination.astype(float)-background.astype(float)).max(1)>35):raise ValueError('destination contains other marks')
 for other in other_lines:
  if other is line:continue
  b=np.array(other['bbox'])
  if min(b[2],right)>max(b[0],x1) and min(b[3],y2)>max(b[1],y1):raise ValueError('edit band overlaps another OCR line')
 return (x1,y1,x2,y2),background,shifts


def render_blocks(a,bbox,background,shifts):
 """Full glyph rectangle transfer, common neutral canvas, no-layout editor controls."""
 x1,y1,x2,y2=bbox;right=x2+max(shifts);base=a.copy();base[y1:y2,x1:right]=background
 block=a[y1:y2,x1:x2].copy();variants={}
 for arm,offset in [('sham',0),('shift_1',shifts[0]),('shift_2',shifts[1])]:
  b=base.copy();b[y1:y2,x1+offset:x2+offset]=block;variants[arm]=b
  assert np.array_equal(b[y1:y2,x1+offset:x2+offset],block)
 sham=variants['sham'];colors,counts=np.unique(sham[y1:y2,x1:right].reshape(-1,3),axis=0,return_counts=True)
 for b in variants.values():
  c,n=np.unique(b[y1:y2,x1:right].reshape(-1,3),axis=0,return_counts=True);assert np.array_equal(c,colors) and np.array_equal(n,counts)
 # A deterministic blank paper block retains its interpolated edge shade; letters stay in place.
 left=np.median(block[:,:2],axis=1);right_colors=np.median(block[:,-2:],axis=1)
 blank=np.rint(left[:,None,:]*(1-np.linspace(0,1,x2-x1)[None,:,None])+right_colors[:,None,:]*np.linspace(0,1,x2-x1)[None,:,None]).astype(np.uint8)
 for k,offset in enumerate(shifts,1):
  b=sham.copy();start=max(x1+offset,x2)
  if start<x2+offset:b[y1:y2,start:x2+offset]=blank[:,start-(x1+offset):]
  assert np.array_equal(b[y1:y2,x1:x2],block);variants['editor_'+str(k)]=b
 allowed=np.zeros(a.shape[:2],bool);allowed[y1:y2,x1:x2+max(shifts)]=True
 for b in variants.values():assert np.array_equal(b[~allowed],a[~allowed])
 return variants


def main():
 torch.set_num_threads(2);cv2.setNumThreads(2);d=data();folder=OUT/'stage4_extended_photos_v3';images=folder/'images';images.mkdir(parents=True,exist_ok=True)
 protocol=dict(version=3,source_population='all dataset original photos whose ORIGINAL cache provenance is generated',
  source_selection='one first eligible photo/word per package, deterministic package/photo/OCR order; no correctness, labels beyond fixed split, focused-photo lists or embedding responses used',
  eligibility='isolated alphabetic OCR word, confidence >=.95, horizontal <=.08, height >=10px, foreground >=20px, inside subject, uniform blank destination 1%-99% spread<=24; light/dark glyph polarity detected; no other OCR line in band',
  arms=['sham common background at original location','shift_1 .25 patch width','shift_2 .75 patch width','editor_1/2 background-patch control with glyph position unchanged'],
  class_definition='shift_1=0, shift_2=1; BOTH classes edited; source identity held out',
  scope='isolated-word horizontal position, not asserted to be paragraph-final words or natural instance identities',
  original_features='cached original-query embedding reused, fresh base encoded only to verify exact recipe before edit interpretation',
  encoder='automatic LoRA FP32, default torchvision processor; recent generated-photo parity verified in separate diagnosis',
  planned_minimum_test_identities=10,historical_test_exposure=True,maximum_training_identities=20,maximum_validation_identities=8,
  background_change_declared='edit-band paper normalized identically for sham/shift arms; full glyph rectangle RGB and pixels outside band unchanged; editor controls check movable background artifacts')
 path=folder/'protocol.json'
 if path.exists():assert read(path)==protocol
 else:write(path,protocol)
 supplement='--supplement' in sys.argv
 sources=read(folder/'sources.json') if supplement else [];manifest=read(folder/'encoding_manifest.json') if supplement else []
 rejected=[];validations=read(folder/'pixel_validation.json') if supplement else []
 if supplement:
  write(folder/'angle_supplement_protocol.json',dict(reason='primary eligibility yielded 9 test identities, planned minimum 10',
   primary_sources_frozen=True,new_sources_from_unrepresented_identities_only=True,maximum_supplemental_tilt=.15,
   no_probe_predictions_used=True,stopping='first additional eligible test identity in deterministic order; train/val caps already reached'))
 candidates={}
 for i,qid in enumerate(d['qb']['ids']):
  if d['qb']['provenance'][i]['kind']=='generated':candidates.setdefault(d['photos'][qid]['package_id'],[]).append((qid,i))
 for package in sorted(candidates):
  if supplement and sum(r['split']=='test' for r in sources)>=protocol['planned_minimum_test_identities']:break
  if any(r['package_id']==package for r in sources):continue
  split=d['splits'][candidates[package][0][0]]
  limit=protocol['maximum_training_identities'] if split=='train' else protocol['maximum_validation_identities'] if split=='val' else None
  if limit is not None and sum(r['split']==split for r in sources)>=limit:continue
  selected=None
  for qid,qi in sorted(candidates[package]):
   ocr=d['qo'][qid];box=np.array(d['boxes'][qid]['bbox']);spec=d['metadata']['photos'][qi];path=ROOT/d['qb']['paths'][qi]
   assert sha(path)==d['qb']['sha256'][qi]
   with Image.open(path) as im:
    raw=np.asarray(im.convert('RGB'));a=np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
   assert a.shape[:2]==(ocr['height'],ocr['width'])
   for line_index,line in enumerate(ocr['lines']):
    if not re.fullmatch(r'[A-Za-z]{4,20}[.!]?',line['text'].strip()):continue
    try:bbox,bg,shifts=prepare(a,line,spec,box,ocr['lines'],.15 if supplement else .08);variants=render_blocks(a,bbox,bg,shifts)
    except ValueError as e:rejected.append(dict(package_id=package,photo_id=qid,line=line_index,reason=str(e)));continue
    selected=(qid,qi,line_index,line['text'],path,raw,a,spec,bbox,bg,shifts,variants);break
   if selected is not None:break
  if selected is None:continue
  qid,qi,line_index,text,path,raw,a,spec,bbox,bg,shifts,variants=selected;orientation=spec['exif_orientation'];assert np.array_equal(to_raw(a,orientation),raw)
  source=dict(photo_id=qid,package_id=package,group_id=d['groups'][package],split=d['splits'][qid],word=text,line_index=line_index,
   image_sha256=sha(path),source_sha256=d['qb']['source_sha256'][qi],cache_row=qi,word_bbox_display=list(map(int,bbox)),pixel_shifts=shifts,exif_orientation=orientation,
   maximum_tilt=.15 if supplement else .08,supplemental_source=supplement)
  sources.append(source);manifest.append(dict(id=qid+'_fresh_base',path=str(path),source_id=qid,arm='base',split=source['split'],package_id=package))
  canvas=Image.new('RGB',(600,750),'white');draw=ImageDraw.Draw(canvas)
  for k,(arm,b) in enumerate(variants.items()):
   array=to_raw(b,orientation);p=images/(qid+'_'+arm+'.png');exif=Image.Exif();exif[274]=orientation;Image.fromarray(array).save(p,exif=exif)
   offset=shifts[int(arm[-1])-1] if arm.startswith('shift_') else 0;x1,y1,x2,y2=bbox;rb=raw_bbox((x1+offset,y1,x2+offset,y2),a.shape,orientation)
   manifest.append(dict(id=qid+'_'+arm,path=str(p),source_id=qid,arm=arm,split=source['split'],package_id=package,word_bbox=rb,
    layout_class=int(arm=='shift_2') if arm.startswith('shift_') else None))
   zoom=Image.fromarray(b).crop((max(0,x1-8),max(0,y1-28),min(a.shape[1],x2+max(shifts)+8),min(a.shape[0],y2+28)));zoom.thumbnail((580,120));canvas.paste(zoom,(10,k*150+25));draw.text((10,k*150+3),arm,fill='black')
  preview=folder/'previews'/(qid+'.png');preview.parent.mkdir(exist_ok=True);canvas.save(preview)
  validations.append(dict(photo_id=qid,glyph_rectangle_shape=[int(bbox[3]-bbox[1]),int(bbox[2]-bbox[0])],unchanged_outside_edit_band=True,
   original_raw_display_inverse_exact=True,glyph_RGB_conserved=True,sham_and_shift_histograms_identical=True,
   other_text_capacity_and_outside_colors_unchanged=True,edit_band_background_RGB=bg.tolist(),editor_controls_do_not_move_word=True))
  print('Source selected',package,source['split'],qid,text,flush=True)
 write(folder/'sources.json',sources);write(folder/'encoding_manifest.json',manifest);write(folder/'pixel_validation.json',validations);csvout(folder/('supplement_rejected_sources.csv' if supplement else 'rejected_sources.csv'),rejected)
 summary=dict(identities_per_split={s:sum(r['split']==s for r in sources) for s in ['train','val','test']},unique_sources=len(sources),encoding_records=len(manifest),
  focuses_used_for_selection=False,source_selection_complete=True)
 write(folder/'source_summary.json',summary);assert summary['identities_per_split']['test']>=protocol['planned_minimum_test_identities'],summary
 print(summary,flush=True)

if __name__=='__main__':main()
