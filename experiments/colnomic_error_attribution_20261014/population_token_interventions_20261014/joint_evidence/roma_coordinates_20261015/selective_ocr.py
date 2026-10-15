"""Automatic observable trigger; rerun OCR without knowing query truth or failure list."""
from pathlib import Path
import sys,os,json,math,hashlib,argparse
import numpy as np
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');sys.path[:0]=[str(P),str(R)]
from experiment_utils import read,write
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
ap=argparse.ArgumentParser();ap.add_argument('--plan-only',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);args=ap.parse_args()
c=read(P/'cohort.json');qo=read(R/'cache/ocr/photos.json');policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records'];scores=np.load(D/'fixed1051_scores.npy');top=np.partition(scores,-2,axis=1)[:,-2:];gaps=top.max(1)-top.min(1)
plan=[]
for i,r in enumerate(c):
 q=r['query'];b=boxes[q]['bbox'];x,y,X,Y=b;lines=[l for l in qo[q]['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2<=X and y<=(l['bbox'][1]+l['bbox'][3])/2<=Y]
 orientations=[0,0,0,0]
 for l in lines:
  a,z=l['polygon'][:2];angle=math.degrees(math.atan2(z[1]-a[1],z[0]-a[0]));orientations[int(round(angle/90))%4]+=len(l['text'])
 direction=int(np.argmax(orientations));ratio=orientations[direction]/max(1,sum(orientations));rotation=90*direction if ratio>=.6 else 0
 low_long=any(len(l['text'])>=60 and l['confidence']<.8 for l in lines)
 low_sparse=len(lines)<=5 and any(l['confidence']<.8 for l in lines)
 reasons=[]
 if gaps[i]<=.1:reasons.append('top2_score_gap_le_0.1')
 if rotation:reasons.append('dominant_nonupright_text')
 if low_long:reasons.append('long_line_confidence_lt_0.8')
 if low_sparse:reasons.append('sparse_low_confidence_text')
 if reasons:plan.append(dict(query=q,rotation=rotation,reasons=reasons,gap=float(gaps[i]),box=b))
if args.plan_only or args.shard==0:write(D/'OCR_TRIGGER.json',dict(rule='Score gap<=0.1 OR dominant nonupright text OR long(>=60 chars) OCR line confidence<0.8 OR <=5 subject lines with confidence<0.8. Existing raw top2-reference gap intentionally fixed; duplicates may cause extra reruns.',selected=len(plan),total=1117,uses_ground_truth=False,records=plan))
print('PLAN',len(plan),flush=True)
if args.plan_only:sys.exit()
from evaluation.ocr import OCR
from evaluation.regions import open_photo
engine=OCR(R,'cpu');(D/'ocr_inputs').mkdir(exist_ok=True);dest=D/f'new_ocr{args.shard}.json';done=read(dest) if dest.exists() else {}
for k,r in enumerate(plan):
 if k%args.shards!=args.shard:continue
 q=r['query']
 if q in done:continue
 im=open_photo(R/qo[q]['image_path']).crop(tuple(map(int,r['box']))).rotate(r['rotation'],expand=True);path=D/'ocr_inputs'/(q+'.png');im.save(path)
 out,seconds=engine.recognize(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),path)
 done[q]=dict(query=q,rotation=r['rotation'],reasons=r['reasons'],image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ocr=out,seconds=seconds);write(dest,done)
 print('OCR',k+1,len(plan),q,round(seconds,2),flush=True)
print('OCR DONE',flush=True)
