"""Automatic barcode decoding; human subject crops, no selected true references."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,csv,time
import cv2,numpy as np
from PIL import Image,ImageOps
P=Path('/mnt/lsdf_zyin/results/colnomic_population_interventions/20261014_v1');D=P/'background_causal_v1/joint_image_ocr_v1/remaining54_v1';O=D/'remaining49_v1/robustness'
sys.path.insert(0,str(P))
from experiment_utils import read,write
from review_remaining49 import oriented
cv2.setNumThreads(1)
def table(n,k):
 with open('/mnt/lsdf_zyin/dataset/lists/'+n+'.csv')as f:return {r[k]:r for r in csv.DictReader(f)}
def main():
 photos=table('photos','photo_id');refs=table('references','reference_id');cohort=read(P/'cohort.json');policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(policy['remote_source'])['records']
 rows=[]
 for r in cohort:
  q=r['query'];b=boxes[q];assert b['source']=='human' and b['status']=='confirmed';rows.append(dict(id=q,kind='query',path=photos[q]['path'],box=b['bbox']))
 seen={}
 for r,x in refs.items():seen.setdefault(x['sha256'],dict(id=r,kind='reference',path=x['path'],aliases=[],box=None))['aliases'].append(r)
 rows+=list(seen.values());benchmark='--benchmark'in sys.argv
 if benchmark:rows=rows[:12]+rows[-12:]
 output=O/('BARCODE_BENCHMARK.json'if benchmark else 'BARCODE_DECODE.json');prior=[r for r in read(output)['records']if r['status']=='ok']if output.exists()else [];done={r['id']for r in prior}
 write(O/'BARCODE_PROTOCOL.json',dict(detector='OpenCV barcode_BarcodeDetector',query='confirmed human subject rectangle, EXIF oriented',reference='all5330 SHA unique full references',resolution_max=1800,workers=4,no_manual_strings=True,no_candidate_truth=True))
 def work(r):
  started=time.time()
  try:
   im=oriented(Path('/mnt/lsdf_zyin/dataset')/r['path'])
   if r['box']is not None:im=im.crop(tuple(round(v)for v in r['box']))
   size=im.size;im.thumbnail((1800,1800),Image.Resampling.LANCZOS);img=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
   ok,decoded,types,points=cv2.barcode_BarcodeDetector().detectAndDecodeWithType(img)
   return dict(id=r['id'],kind=r['kind'],aliases=r.get('aliases',[]),codes=[dict(text=c,type=t)for c,t in zip(decoded,types)if c],detected_points=points.tolist()if points is not None else [],original_crop_size=size,input_size=im.size,seconds=time.time()-started,status='ok')
  except Exception as e:return dict(id=r['id'],kind=r['kind'],status='error',error=str(e),seconds=time.time()-started)
 start=time.time()
 with ThreadPoolExecutor(max_workers=4)as pool:
  futures=[pool.submit(work,r)for r in rows if r['id']not in done]
  for f in as_completed(futures):
   prior.append(f.result())
   if len(prior)%50==0:write(output,dict(status='running',total=len(rows),records=prior));print('DECODE',len(prior),len(rows),sum(bool(r.get('codes'))for r in prior),round(time.time()-start),flush=True)
 write(output,dict(status='complete',total=len(rows),elapsed_seconds=time.time()-start,records=prior));print('DONE',len(prior),sum(bool(r.get('codes'))for r in prior),time.time()-start,flush=True)
if __name__=='__main__':main()
