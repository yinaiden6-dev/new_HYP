"""Accept a RoMa winner switch only with subject-localized support; truth-free gate.

Designed after observing the preceding repair/harm audits, hence exploratory.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');C=R/'cache/content_box_ce_polish18/deployment/ee69890fef908bb6a170';OUT=D/'roma_reuse';sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,metrics
torch.set_num_threads(2)
write(OUT/'SUBJECT_GATE_PROTOCOL.json',dict(designed_after_inspecting_region_audit=True,primary_threshold=.5,threshold_sensitivity=[0,.25,.5,.75,.9,1],rule='If RoMa proposes a different Top1, accept its same-Top10 permutation only when at least threshold of proposed winner query overlap confidence lies in the existing human subject box. Otherwise keep every baseline rank. If winner is unchanged, preserve the proposed permutation.',baseline_arms={'fusion':'fixed1051 + previously fixed S beta0.2','image':'image944 + validation-selected M beta0.5'},controls=['Accept based on subject rectangle area fraction only at identical threshold.','Roll overlap map by half height and half width, retain same values; gate at identical threshold.'],selection='Full173 validation macro Top1 then macro MRR, then smaller threshold; do not select using target54 or test. Fixed0.5 also reported independently.',no_ground_truth_at_inference=True,no_SAM=True,no_neural_forward=True,limitation='Human subject boxes remain external annotations; retrospective exploratory result, not untouched benchmark. Spatial support gating is not correspondence-constrained matching.'))
c=read(P/'cohort.json');qids=[x['query'] for x in c];qi={q:i for i,q in enumerate(qids)};policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records'];m=read(C/'manifest.json');qm={x['id']:x for x in m['queries']};rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);ri={x:j for j,x in enumerate(rb['ids'])};sha=rb['sha256']
excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries']);fixed=np.load(OUT/'baseline.npz')['ranks'];target=np.array([i for i,q in enumerate(qids) if q not in excluded and fixed[i]>1]);assert len(target)==54
scopes={'all':np.arange(len(c)),'validation':np.array([i for i,x in enumerate(c) if x['split']=='val']),'test':np.array([i for i,x in enumerate(c) if x['split']=='test']),'retained':np.array([i for i,q in enumerate(qids) if q not in excluded]),'target54':target}
cache=read(OUT/'SUBJECT_GATE_SUPPORT.json') if (OUT/'SUBJECT_GATE_SUPPORT.json').exists() else {};summary={};curves=[];allcases=[]
for family,folder,arm in [('fusion',OUT,'S_previous_fixed_0.2'),('image',OUT/'pure_selected_image','selected')]:
 base=np.load(folder/'baseline.npz');prop=np.load(folder/(arm+'.npz'));br=base['ranks'];pr=prop['ranks'];bt=base['top1'];pt=prop['top1'];changed=bt!=pt;support=np.ones(len(c));area=np.ones(len(c));rolled=np.ones(len(c))
 for count,i in enumerate(np.flatnonzero(changed)):
  q=qids[i];rid=str(pt[i]);j=ri[rid];axis=qm[q]['candidate_physical_rows'];aliases=[k for k in axis if sha[k]==sha[j]];assert len(aliases)==1;j=aliases[0];key=q+'__'+rb['ids'][j]
  if key not in cache:
   path=C/'pairs'/q/f'{j:06}.npz'
   with np.load(path) as f:
    a=f['overlap_AB'].copy();assert str(f['query_tokens_sha256'].item())==qm[q]['image_tokens_sha256'];assert str(f['reference_tokens_sha256'].item())==m['references'][str(j)]['image_tokens_sha256']
   box=boxes[q];assert box['source']=='human' and box['status']=='confirmed';W,H=qm[q]['raw_width'],qm[q]['raw_height']
   if qm[q]['exif_orientation'] in [5,6,7,8]:W,H=H,W
   x,y,X,Y=box['bbox'];h,w=a.shape;x0,x1=np.clip([int(np.floor(x/W*w)),int(np.ceil(X/W*w))],0,w);y0,y1=np.clip([int(np.floor(y/H*h)),int(np.ceil(Y/H*h))],0,h);total=float(a.sum(dtype=np.float64));shuffled=np.roll(a,(h//2,w//2),axis=(0,1))
   cache[key]=dict(query=q,reference=rb['ids'][j],support=float(a[y0:y1,x0:x1].sum(dtype=np.float64))/max(total,1e-30),area=(x1-x0)*(y1-y0)/a.size,rolled=float(shuffled[y0:y1,x0:x1].sum(dtype=np.float64))/max(total,1e-30),box=box['bbox'],query_sha256=qm[q]['sha256'],reference_sha256=sha[j],pair=str(path))
  x=cache[key];support[i]=x['support'];area[i]=x['area'];rolled[i]=x['rolled']
  if count%20==0:print('GATE SUPPORT',family,count,int(changed.sum()),flush=True);write(OUT/'SUBJECT_GATE_SUPPORT.json',cache)
 write(OUT/'SUBJECT_GATE_SUPPORT.json',cache)
 proposals=[]
 def evaluate(feature,values,threshold):
  accept=(~changed)|(values>=threshold);rr=np.where(accept,pr,br);tt=np.where(accept,pt,bt);return rr,tt,accept
 for feature,values in [('subject_overlap',support),('area_only',area),('rolled_overlap',rolled)]:
  for threshold in [0,.25,.5,.75,.9,1]:
   rr,tt,accept=evaluate(feature,values,threshold);v=metrics(rr,c,scopes['validation'],br)
   proposals.append(dict(feature=feature,threshold=threshold,validation=v))
   for scope,ix in scopes.items():curves.append(dict(family=family,feature=feature,threshold=threshold,scope=scope,accepted_switches=int((changed[ix]&accept[ix]).sum()),**metrics(rr,c,ix,br)))
 best=max([x for x in proposals if x['feature']=='subject_overlap'],key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['threshold']))
 arms={'subject_fixed0.5':('subject_overlap',support,.5),'subject_validation_selected':('subject_overlap',support,best['threshold']),'area_fixed0.5':('area_only',area,.5),'roll_fixed0.5':('rolled_overlap',rolled,.5)};res={}
 for name,(feature,values,threshold) in arms.items():
  rr,tt,accept=evaluate(feature,values,threshold);res[name]=dict(feature=feature,threshold=threshold,accepted_switches=int((changed&accept).sum()),metrics={s:metrics(rr,c,ix,br) for s,ix in scopes.items()},repairs=[qids[i] for i in range(len(c)) if br[i]>1 and rr[i]==1],harms=[qids[i] for i in range(len(c)) if br[i]==1 and rr[i]>1])
  np.savez_compressed(OUT/f'gate_{family}_{name}.npz',ranks=rr,top1=tt,query_ids=qids,accepted=accept,changed=changed)
  for i,q in enumerate(qids):allcases.append(dict(query=q,family=family,method=name,baseline_rank=int(br[i]),proposal_rank=int(pr[i]),rank=int(rr[i]),accepted=bool(accept[i]),winner=str(tt[i]),proposed_winner=str(pt[i]),support=float(support[i]) if changed[i] else '',area=float(area[i]) if changed[i] else '',rolled=float(rolled[i]) if changed[i] else ''))
 summary[family]=dict(proposed_switches=int(changed.sum()),selection=best,results=res)
write(OUT/'SUBJECT_GATE_SUMMARY.json',dict(status='complete',families=summary,limitations=read(OUT/'SUBJECT_GATE_PROTOCOL.json')['limitation']));csvwrite(OUT/'subject_gate_curve.csv',curves);csvwrite(OUT/'subject_gate_cases1117.csv',allcases)
print('SUBJECT GATE DONE',json.dumps({k:{n:v['metrics']['all'] for n,v in s['results'].items()} for k,s in summary.items()}),flush=True)
