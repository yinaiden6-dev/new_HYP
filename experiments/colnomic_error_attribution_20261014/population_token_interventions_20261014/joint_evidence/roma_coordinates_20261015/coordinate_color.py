"""Color correspondence control computed from saved RoMa coordinates, no new model."""
from pathlib import Path
from functools import lru_cache
import sys,time,json,argparse
import numpy as np
import cv2
from PIL import Image,ImageOps
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;OUT=D/'roma_coordinates'
sys.path.insert(0,str(P));from experiment_utils import read,write

@lru_cache(maxsize=256)
def load(path):
    with Image.open(path) as im:
        orientation=int(im.getexif().get(274,1));rgb=im.copy()
        op={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}.get(orientation)
        if op is not None:rgb=rgb.transpose(op)
        rgb=rgb.convert('RGB');rgb.info.clear();rgb.thumbnail((384,384))
        a=np.asarray(rgb,dtype=np.float32)/255
    return cv2.cvtColor(a,cv2.COLOR_RGB2LAB)

def sample(a,xy):
    h,w=a.shape[:2];mp=(np.asarray(xy,dtype=np.float32)*[w,h]-.5).astype(np.float32)
    return cv2.remap(a,mp[:,0,None],mp[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]

def run(wait):
    cv2.setNumThreads(1);plan=read(OUT/'PLAN.json');qs={q['query']:q for q in plan['queries']}
    dest=OUT/'COLOR.json';done=read(dest)['pairs'] if dest.exists() else {};started=time.time()
    protocol=dict(input='Original EXIF-oriented RGB; existing human query box; no OCR and no identity label.',resize='Longest side <=384, PIL thumbnail default bicubic; Lab through OpenCV RGBfloat32.',score='Confidence-weighted exp(-Lab_ab_distance/20) for saved query->reference64x64 matches inside subject; cyclic error<=0.05 and overlap>=0.5; chroma reliability=min(max(query_chroma,reference_chroma)/30,1). If no colored reliable correspondence,0.',control='Same samples/weights, reference colors half-grid rolled before distance.',selection='Prespecified weight grid and173validation only; additional exploratory family.')
    while True:
        count=0
        for t in plan['tasks']:
            key=t['query']+'/'+str(t['row']);p=OUT/'pairs'/t['query']/f"{t['row']:06}.npz"
            if key in done or not p.with_suffix('.json').exists():continue
            q=plan['query_records'][t['query']];r=plan['references'][str(t['row'])]
            with np.load(p) as f:xy=f['grid_xy'];wx=f['grid_warp'];confidence=f['grid_confidence'];cycle=f['grid_cycle_error']
            W,H=q['raw_width'],q['raw_height']
            if q['exif_orientation'] in [5,6,7,8]:W,H=H,W
            box=np.array(qs[t['query']]['box'])/[W,H,W,H]
            valid=((xy>=box[:2])&(xy<=box[2:])).all(1)&((wx>=0)&(wx<=1)).all(1)&(confidence>=.5)&(cycle<=.05)
            a=sample(load(q['path']),xy)[:,1:];b=sample(load(r['path']),wx)[:,1:]
            chroma=np.maximum(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1));w=confidence*valid*np.clip(chroma/30,0,1)*(chroma>10)
            delta=np.linalg.norm(a-b,axis=1);dr=np.linalg.norm(a-np.roll(b,len(b)//2,axis=0),axis=1)
            den=max(w.sum(),1e-12);C=float((w*np.exp(-delta/20)).sum()/den);Cr=float((w*np.exp(-dr/20)).sum()/den)
            done[key]=dict(C=C,Croll=Cr,reliable_colored_points=int((w>0).sum()),weight=float(w.sum()));count+=1
        if count or not dest.exists():
            tmp=dest.with_suffix('.tmp');write(tmp,dict(protocol=protocol,pairs=done,complete=len(done)==len(plan['tasks']),count=len(done),expected=len(plan['tasks'])));tmp.replace(dest)
            print('COLOR',len(done),len(plan['tasks']),flush=True)
        if len(done)==len(plan['tasks']) or not wait:break
        if time.time()-started>16*3600:raise TimeoutError('Coordinate color wait exceeded16h')
        time.sleep(45)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');a=ap.parse_args();run(a.wait)
