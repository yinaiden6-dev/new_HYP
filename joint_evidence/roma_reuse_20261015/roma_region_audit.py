"""Inspect cached overlap support for repaired/harmed photographs, never run RoMa."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from PIL import Image
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');C=R/'cache/content_box_ce_polish18/deployment/ee69890fef908bb6a170';OUT=D/'roma_reuse'
sys.path.insert(0,str(P));from experiment_utils import read,write,Ranking
torch.set_num_threads(2)
c=read(P/'cohort.json');qi={x['query']:i for i,x in enumerate(c)};policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records']
qb=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True)
rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);ids=rb['ids'];sha=rb['sha256'];ri={x:j for j,x in enumerate(ids)};rk=Ranking(ids,sha,c)
m=read(C/'manifest.json');qm={x['id']:x for x in m['queries']}
for i,row in enumerate(c):assert qm[row['query']]['sha256']==qb['sha256'][i]
names={'fusion_fixed_0.2':(OUT,'S_previous_fixed_0.2',np.load(D/'fixed1051_scores.npy')),'image_selected':(OUT/'pure_selected_image','selected',np.load(V/'selected_image_scores.npy'))}
rows=[];small={};recorded={};meta=[]
for family,(folder,arm,scores) in names.items():
 s=read(folder/'SUMMARY.json')['results'][arm];pred=np.load(folder/(arm+'.npz'));pmap=dict(zip(pred['query_ids'].tolist(),pred['top1'].tolist()))
 targets=sorted(set(s['repairs']+s['harms']+['photo_0a0e75a3b7797609']))
 for q in targets:
  i=qi[q];d=rk.one(scores[i],i);p=d['positive'];n=pmap[q] if q in s['harms'] else d['negative'];pairs={}
  assert sha[ri[p]] in set(c[i]['positive_sha256']) and sha[ri[n]] not in set(c[i]['positive_sha256'])
  for role,rid in [('positive',p),('negative',n)]:
   j=ri[rid];axis=qm[q]['candidate_physical_rows'];aliases=[k for k in axis if sha[k]==sha[j]]
   if not aliases:pairs[role]=dict(reference=rid,available=False);continue
   j=aliases[0];path=C/'pairs'/q/f'{j:06}.npz';key=q+'__'+ids[j]
   if key not in recorded:
    with np.load(path) as f:
     assert str(f['query_tokens_sha256'].item())==qm[q]['image_tokens_sha256']
     assert str(f['reference_tokens_sha256'].item())==m['references'][str(j)]['image_tokens_sha256']
     a=f['overlap_AB'].copy();b=f['overlap_BA'].copy();c4=f['c4'].tolist();fields=f.files
    box=boxes[q];assert box['source']=='human' and box['status']=='confirmed'
    W,H=qm[q]['raw_width'],qm[q]['raw_height']
    if qm[q]['exif_orientation'] in [5,6,7,8]:W,H=H,W
    x,y,X,Y=box['bbox'];h,w=a.shape;x0=int(np.floor(x/W*w));y0=int(np.floor(y/H*h));x1=int(np.ceil(X/W*w));y1=int(np.ceil(Y/H*h));x0,x1=np.clip([x0,x1],0,w);y0,y1=np.clip([y0,y1],0,h)
    inside=float(a[y0:y1,x0:x1].sum(dtype=np.float64));total=float(a.sum(dtype=np.float64));area=(x1-x0)*(y1-y0)
    recorded[key]=dict(reference=ids[j],available=True,visibility_score=c4[0],overlap_mass=c4[1],query_overlap_inside_fraction=inside/max(total,1e-30),query_mean_inside=inside/max(area,1),query_mean_outside=(total-inside)/max(a.size-area,1),subject_area_fraction=area/a.size,query_sha256=qm[q]['sha256'],reference_sha256=sha[j],native_pair_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),box=box['bbox'],oriented_size=[W,H],cache_fields=fields)
    small[key+'__query']=np.asarray(Image.fromarray(a).resize((256,256),Image.Resampling.BOX));small[key+'__reference']=np.asarray(Image.fromarray(b).resize((256,256),Image.Resampling.BOX))
   pairs[role]=recorded[key]
  rows.append(dict(query=q,family=family,status='repair' if q in s['repairs'] else 'harm' if q in s['harms'] else 'user_color_example',**pairs))
summary={}
for family in names:
 for status in ['repair','harm']:
  xx=[x for x in rows if x['family']==family and x['status']==status and x['positive']['available'] and x['negative']['available']]
  summary[family+'/'+status]=dict(pairs=len(xx),positive_overlap_majority_outside_subject=sum(x['positive']['query_overlap_inside_fraction']<.5 for x in xx),negative_overlap_majority_outside_subject=sum(x['negative']['query_overlap_inside_fraction']<.5 for x in xx),positive_mean_inside_fraction=float(np.mean([x['positive']['query_overlap_inside_fraction'] for x in xx])),negative_mean_inside_fraction=float(np.mean([x['negative']['query_overlap_inside_fraction'] for x in xx])))
write(OUT/'REGION_AUDIT.json',dict(status='passed',summary=summary,rows=rows,all_query_SHA_matches=True,limits=['Human bounding rectangles, not SAM masks.','Confidence-map fractions are descriptive, not a causal intervention or the exact pooled-token score contribution.','Full-photo RoMa can score other objects or shared generic design details.','Compressed preview maps are for display only; metrics use original 1280x1280 float32 maps.']))
np.savez_compressed(OUT/'overlap_previews.npz',**small)
print('REGION AUDIT',json.dumps(summary),flush=True)
