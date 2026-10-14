"""PP-OCRv6 inference and incremental JSON cache (run with the Paddle environment)."""
import argparse
import json
import os
from pathlib import Path
import time
import unicodedata

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
import numpy as np
from PIL import Image, ImageOps

from .data import file_hash, read_csv, read_json, write_json


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split()).casefold()


def inverse_rotation(points, k, width, height):
    p = np.asarray(points,dtype=np.float32).copy()
    x,y=p[:,0].copy(),p[:,1].copy()
    if k == 1:
        p[:,0],p[:,1]=width-1-y,x
    elif k == 2:
        p[:,0],p[:,1]=width-1-x,height-1-y
    elif k == 3:
        p[:,0],p[:,1]=y,height-1-x
    return p


def deduplicate(lines, config):
    import cv2
    selected=[]
    for line in sorted(lines,key=lambda x:x['confidence'],reverse=True):
        a=cv2.convexHull(np.array(line['polygon'],dtype=np.float32))
        area_a=cv2.contourArea(a)
        if area_a <= 0:
            continue
        duplicate=False
        for old in selected:
            ba,bb=line['bbox'],old['bbox']
            if ba[2] <= bb[0] or bb[2] <= ba[0] or ba[3] <= bb[1] or bb[3] <= ba[1]:
                continue
            b=cv2.convexHull(np.array(old['polygon'],dtype=np.float32))
            area_b=cv2.contourArea(b)
            intersection,_=cv2.intersectConvexConvex(a,b)
            if intersection/max(area_a+area_b-intersection,1e-8) >= config['dedup_polygon_iou'] or \
               intersection/max(min(area_a,area_b),1e-8) >= config['dedup_polygon_containment']:
                duplicate=True;break
        if not duplicate:
            selected.append(line)
    return sorted(selected,key=lambda x:((x['bbox'][1]+x['bbox'][3])/2,x['bbox'][0]))


class OCR:
    def __init__(self, root, device='gpu:0'):
        self.root=Path(root)
        self.config=read_json(self.root/'configs/ppocrv6.json')
        model_spec=read_json(self.root/'configs/ppocrv6_models.json')
        pinned_files=[f for model in model_spec['models'] for f in model['files']]
        pinned_files+=model_spec['runtime_optimization']['files']
        for item in pinned_files:
            path=self.root/'weights/ppocrv6_medium'/item['path']
            if file_hash(path)!=item['sha256']:
                raise ValueError(f'OCR weight or graph differs from pinned revision: {path}')
        from paddlex.inference.models.text_recognition.processors import CTCLabelDecode
        original=CTCLabelDecode.__call__

        def gpu_ctc(decoder,pred,return_word_box=False,**kwargs):
            if len(pred) != 2 or np.asarray(pred[0]).ndim != 2:
                return original(decoder,pred,return_word_box=return_word_box,**kwargs)
            if return_word_box:
                raise ValueError('Word boxes are not part of the released OCR recipe')
            indices,scores=(np.asarray(p) for p in pred)
            if not np.issubdtype(indices.dtype,np.integer):
                indices,scores=scores,indices
            decoded=decoder.decode(indices,scores,is_remove_duplicate=True,return_word_box=False)
            return [x[0] for x in decoded],[x[1] for x in decoded]

        CTCLabelDecode.__call__=gpu_ctc
        from paddleocr import PaddleOCR
        keys=('text_detection_model_name','text_recognition_model_name',
              'text_recognition_batch_size','use_doc_orientation_classify','use_doc_unwarping',
              'use_textline_orientation','text_det_limit_side_len','text_det_limit_type',
              'text_det_thresh','text_det_box_thresh','text_det_unclip_ratio','text_rec_score_thresh')
        weights=self.root/'weights/ppocrv6_medium'
        self.pipeline=PaddleOCR(**{k:self.config[k] for k in keys},device=device,
            engine=self.config['engine'],precision=self.config['precision'],enable_hpi=False,use_tensorrt=False,
            text_detection_model_dir=str(weights/self.config['text_detection_model_name']),
            text_recognition_model_dir=str(weights/self.config['text_recognition_weights_subdir']),
            enable_mkldnn=False)

    def recognize(self, row, image_path):
        started=time.monotonic()
        with Image.open(image_path) as source:
            stored_size=list(source.size)
            orientation=int(source.getexif().get(274,1))
            rgb=np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
        h,w=rgb.shape[:2]
        regions=[(0,0,w,h)]
        if max(w,h)>self.config['tile_trigger_long_side']:
            size=self.config['tile_size'];step=size-self.config['tile_overlap']
            xs=sorted(set(list(range(0,max(w-size,0)+1,step))+[max(w-size,0)]))
            ys=sorted(set(list(range(0,max(h-size,0)+1,step))+[max(h-size,0)]))
            regions += [(x,y,min(x+size,w),min(y+size,h)) for y in ys for x in xs]
        lines=[]
        for x0,y0,x1,y1 in regions:
            tile=rgb[y0:y1,x0:x1]
            for angle in self.config['rotations_ccw']:
                k=angle//90
                bgr=np.ascontiguousarray(np.rot90(tile,k)[:,:,::-1])
                for result in self.pipeline.predict(bgr):
                    for text,confidence,poly in zip(result['rec_texts'],result['rec_scores'],result['rec_polys'],strict=True):
                        if float(confidence)<self.config['minimum_retained_confidence'] or not text.strip():
                            continue
                        p=inverse_rotation(poly,k,x1-x0,y1-y0)+[x0,y0]
                        p[:,0]=np.clip(p[:,0],0,w-1);p[:,1]=np.clip(p[:,1],0,h-1)
                        lines.append(dict(text=str(text),normalized_text=normalize(text),confidence=float(confidence),
                            polygon=p.tolist(),bbox=[float(p[:,0].min()),float(p[:,1].min()),
                                                    float(p[:,0].max()),float(p[:,1].max())]))
        lines=deduplicate(lines,self.config)
        entry=dict(image_path='dataset/'+row['path'],image_sha256=row['sha256'],width=w,height=h,
            stored_size=stored_size,exif_orientation=orientation,
            coordinate_system='EXIF-transposed image, pixel centers, top-left origin',
            text='\n'.join(x['text'] for x in lines),normalized_text='\n'.join(x['normalized_text'] for x in lines),lines=lines)
        return entry,time.monotonic()-started


def ensure(root,population,device='gpu:0',only_ids=None):
    root=Path(root)
    rows=read_csv(root/f'dataset/lists/{population}.csv')
    path=root/f'cache/ocr/{population}.json'
    bank=read_json(path) if path.exists() else {}
    config=read_json(root/'configs/ppocrv6.json')
    metadata_path=root/'cache/ocr/metadata.json'
    metadata=read_json(metadata_path) if metadata_path.exists() else {}
    models=read_json(root/'configs/ppocrv6_models.json')
    if metadata.get('config',config)!=config or metadata.get('models',models)!=models:
        bank={}
    key='photo_id' if population=='photos' else 'reference_id'
    engine=None;seconds=0.;generated=0
    for row in rows:
        identifier=row[key]
        if only_ids is not None and identifier not in only_ids:
            continue
        if file_hash(root/'dataset'/row['path']) != row['sha256']:
            raise ValueError(f'Image content differs from manifest: {identifier}')
        entry=bank.get(identifier,{})
        if entry.get('image_sha256')==row['sha256'] and isinstance(entry.get('text'),str):
            # Moving an unchanged image does not change OCR pixels or text.
            entry['image_path']='dataset/'+row['path']
            continue
        if engine is None:
            engine=OCR(root,device)
        bank[identifier],elapsed=engine.recognize(row,root/'dataset'/row['path'])
        seconds+=elapsed;generated+=1
        print(f'OCR {identifier}: {len(bank[identifier]["lines"])} lines, {elapsed:.2f}s',flush=True)
    bank={r[key]:bank[r[key]] for r in rows if r[key] in bank}
    write_json(path,bank)
    metadata.update(model=config['model'],config=config,image_path_base='release root',
                    models=read_json(root/'configs/ppocrv6_models.json'))
    metadata.setdefault('populations',{})[population]=dict(images=len(bank),
        lines=sum(len(x['lines']) for x in bank.values()),empty_ids=[k for k,v in bank.items() if not v['text'].strip()],
        file=path.name,sha256=file_hash(path))
    if generated:
        metadata.setdefault('incremental_runs',[]).append(dict(population=population,generated=generated,
            inference_seconds=seconds,script_sha256=file_hash(__file__)))
    write_json(metadata_path,metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--population',choices=['photos','references'],required=True)
    p.add_argument('--device',default='gpu:0')
    p.add_argument('--only-id',action='append')
    a=p.parse_args();ensure(a.root,a.population,a.device,a.only_id)
