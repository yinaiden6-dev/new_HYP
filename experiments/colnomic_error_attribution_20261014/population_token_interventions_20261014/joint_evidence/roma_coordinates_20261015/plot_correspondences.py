"""Diagnostic image figures, actual frozen MaxSim and fresh RoMa point links."""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;OUT=D/'roma_coordinates'
sys.path.insert(0,str(P));from experiment_utils import read
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
plan=read(OUT/'PLAN.json');qs={q['query']:q for q in plan['queries']}

def fit(canvas,path,rect):
    x,y,w,h=rect
    with Image.open(path) as im:im=ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((w,h));x+=(w-im.width)//2;y+=(h-im.height)//2;canvas.paste(im,(x,y))
    return np.array([x,y]),np.array(im.size)

special=['photo_02fb3c06e1fb70e5','photo_0913bcc35e9beb5a','photo_0a0e75a3b7797609','photo_fbc7b6d852b1125b']
for qid in special:
    spec=qs[qid];q=plan['query_records'][qid];pair=spec['diagnostic_pairs']['joint'];im=Image.new('RGB',(1840,1000),'white');d=ImageDraw.Draw(im)
    d.text((20,10),qid+' | original human subject box (cyan); correct/wrong reference fixed',font=font,fill='black')
    for row,(role,j) in enumerate(pair.items()):
        r=plan['references'][str(j)];path=OUT/'pairs'/qid/f'{j:06}.npz'
        if not path.exists():continue
        rec=read(path.with_suffix('.json'))
        with np.load(path) as f:a={k:f[k] for k in ['q_xy','r_xy','maxsim_arg','subject_mask','grid_xy','grid_warp','grid_confidence','grid_cycle_error']}
        W,H=q['raw_width'],q['raw_height']
        if q['exif_orientation'] in [5,6,7,8]:W,H=H,W
        box=np.array(spec['box'])/[W,H,W,H]
        for col,method in enumerate(['ColNomic MaxSim','RoMa correspondence']):
            x=col*920;y=55+row*465
            d.text((x+15,y),role.upper()+' '+r['id']+' | '+method,font=font,fill='#156b3b' if role=='positive' else '#a33328')
            oq,sq=fit(im,q['path'],(x+10,y+35,430,380));orr,sr=fit(im,r['path'],(x+470,y+35,430,380))
            d=ImageDraw.Draw(im);bb=np.r_[oq+sq*box[:2],oq+sq*box[2:]];d.rectangle(tuple(bb),outline='#00a8c8',width=3)
            if col==0:
                idx=np.flatnonzero(a['subject_mask']);idx=idx[np.linspace(0,len(idx)-1,min(32,len(idx))).astype(int)]
                xx=a['q_xy'][idx];yy=a['r_xy'][a['maxsim_arg'][idx]]
                detail=f"Patch mean MaxSim={rec['features']['MaxSim']:.4f}; same32 subject locations"
            else:
                xx=a['grid_xy'];yy=a['grid_warp'];valid=((xx>=box[:2])&(xx<=box[2:])).all(1)&(a['grid_confidence']>=.5)&(a['grid_cycle_error']<=.05)&((yy>=0)&(yy<=1)).all(1)
                idx=np.flatnonzero(valid);n=len(idx)
                if n:idx=idx[np.linspace(0,n-1,min(32,n)).astype(int)]
                xx=xx[idx];yy=yy[idx];detail=f"Reliable points={n}; geometry={rec['features']['G']:.4f}; overlap={rec['features']['M']:.4f}"
            for k,(u,v) in enumerate(zip(xx,yy)):
                pa=oq+sq*u;pb=orr+sr*v;color=['#b92327','#225da9','#138844','#9736a1'][k%4]
                d.line((*pa,*pb),fill=color,width=1);d.ellipse((pa[0]-2,pa[1]-2,pa[0]+2,pa[1]+2),fill=color);d.ellipse((pb[0]-2,pb[1]-2,pb[0]+2,pb[1]+2),fill=color)
            d.text((x+15,y+421),detail,font=small,fill='black')
    im.save(OUT/(qid+'_correspondences.png'))
    print('FIGURE',qid,flush=True)
