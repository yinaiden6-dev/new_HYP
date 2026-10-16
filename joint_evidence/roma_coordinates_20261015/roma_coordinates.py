"""Frozen RoMa precise correspondence completion; natural candidates, no OCR input.

All coordinates use the EXIF-oriented image, normalized to [0,1]. Patch
coordinates are transformed from the actual cached ColNomic input frame.
Diagnostic true pairs are explicitly separate from deployable candidates.
"""
from pathlib import Path
import argparse,sys,json,time,hashlib,importlib.util,os
import numpy as np
import torch
from torch.nn import functional as F
import cv2

D=Path(__file__).resolve().parent; V=D.parent; P=V.parent.parent
R=Path('/mnt/lsdf_zyin'); C=R/'cache/content_box_ce_polish18/deployment/ee69890fef908bb6a170'
OUT=D/'roma_coordinates';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(P))
from experiment_utils import read,write,Ranking
spec=importlib.util.spec_from_file_location('original_roma_deployment',R/'scripts/run_content_box_ce_polish18.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def top_unique(a,sha,n=10):
    out=[];seen=set()
    for j in np.argsort(-a,kind='stable'):
        if sha[j] in seen:continue
        seen.add(sha[j]);out.append(int(j))
        if len(out)==n:break
    return out

def prepare():
    torch.set_num_threads(2);banks=old.Banks(False);c=read(P/'cohort.json')
    ids=banks.r['ids'];sha=banks.r['sha256'];rk=Ranking(ids,sha,c)
    scores={'image':np.load(V/'selected_image_scores.npy'),'joint':np.load(D/'fixed1051_scores.npy')}
    ranks={k:rk.all(v)[0] for k,v in scores.items()}
    assert int((ranks['image']==1).sum())==944 and int((ranks['joint']==1).sum())==1051
    excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries'])
    target=[i for i,x in enumerate(c) if x['query'] not in excluded and ranks['joint'][i]>1]
    assert len(target)==54
    policy=read(P/'SUBJECT_REGION_POLICY.json');assert policy['sam_allowed'] is False
    boxes=read(policy['remote_source'])['records']
    m=old.relocate_manifest(read(C/'manifest.json'),R);qm={q['id']:q for q in m['queries']}
    refs=m['references'];ri={x:j for j,x in enumerate(ids)};needed=set();tasks=[];queries=[]
    special=['photo_02fb3c06e1fb70e5','photo_0913bcc35e9beb5a','photo_0a0e75a3b7797609','photo_fbc7b6d852b1125b']
    for i,x in enumerate(c):
        q=x['query'];assert qm[q]['bank_index']==i and banks.q['ids'][i]==q
        assert qm[q]['sha256']==banks.q['sha256'][i]
        b=boxes[q];assert b['source']=='human' and b['status']=='confirmed'
        top={k:top_unique(s[i],sha) for k,s in scores.items()};union=set(sum(top.values(),[]));diag=set();pairs={}
        if i in target or q in special:
            for k,s in scores.items():
                pair=rk.one(s[i],i);pairs[k]={role:ri[pair[role]] for role in ['positive','negative']}
                diag.update(pairs[k].values())
        priority=0 if q in special else 1 if i in target else 2
        for j in sorted(union|diag):
            needed.add(j);tasks.append(dict(query=q,query_index=i,row=j,priority=priority,diagnostic=j in diag,candidate=j in union))
        queries.append(dict(query=q,index=i,top=top,diagnostic_pairs=pairs,box=b['bbox'],target54=i in target,priority=priority))
    missing=needed-{int(k) for k in refs}
    if missing:
        from colpali_engine.models import ColQwen2_5_Processor
        banks.processor=ColQwen2_5_Processor.from_pretrained(R/'weights/colnomic7b/adapter',local_files_only=True)
        for j in sorted(missing):
            r=dict(id=ids[j],physical_row=j,bank_index=j,path=str(R/banks.r['paths'][j]),sha256=sha[j])
            r.update(banks.inspect(r,j,True));refs[str(j)]=r
    for j in needed:assert refs[str(j)]['id']==ids[j] and refs[str(j)]['sha256']==sha[j]
    tasks.sort(key=lambda t:(t['priority'],t['query_index'],not t['diagnostic'],t['row']))
    protocol=dict(version=1,candidates='Union of independent image944 and joint1051 natural SHA-unique Top10; each reranker only uses its own original ten. Diagnostic positives never inserted.',
        model=m['recipe']['roma'],frame='EXIF-oriented normalized [0,1], historical token geometry transformed from DECODED_RAW_BEFORE_EXIF',
        human_boxes_only=True,no_ocr=True,no_colnomic_forward=True,no_training=True,
        save='Native FP32 bidirectional warp/confidence for fixed diagnostic pairs; FP32 native-map samples at EVERY frozen token center plus a fixed64x64 grid for EVERY pair.',
        homography='RANSAC seed17, source points inside human box, overlap>=0.5, cycle distance<=0.05, threshold0.02 in normalized reference coordinates; no labels. If <4 or failed, geometry score0 and feature score falls back to visibility.',
        lambda_distance=.1,rank_weights=[0,.05,.1,.2,.5,1,2,5],gates=[0,.5],
        selection='All original173validation: macroTop1, macroMRR, smaller weight, fewer added constraints. Historical exploratory only.',
        controls=['visibility alone','geometry alone','feature MaxSim with no coordinate penalty','dense warp constraint','single homography constraint','half-query-grid rolled warp and homography'],
        limits=['RoMa provides extra visual features; a gain is not proof ColNomic itself encodes all layout.','A single homography is a planar approximation; foldouts/multiple faces violate it.','Historical test set already inspected.'])
    payload=dict(protocol=protocol,source_manifest_sha=old.file_hash(C/'manifest.json'),queries=queries,query_records=qm,references={str(j):refs[str(j)] for j in sorted(needed)},tasks=tasks)
    payload['signature']=old.digest(dict(protocol=protocol,queries=queries,tasks=tasks,source=payload['source_manifest_sha']))
    old.atomic_json(OUT/'PLAN.json',payload)
    print(json.dumps(dict(tasks=len(tasks),diagnostic_pairs=sum(t['diagnostic'] for t in tasks),target54_pairs=sum(t['priority']<=1 for t in tasks),references=len(needed),new_reference_metadata=len(missing),signature=payload['signature'])),flush=True)

def sample(a,xy):
    """a NHWC, xy Nx2 in oriented normalized [0,1]; native align_corners=False."""
    grid=torch.as_tensor(xy,dtype=torch.float32,device=a.device)[None,:,None,:]*2-1
    return F.grid_sample(a.float().permute(0,3,1,2),grid,align_corners=False,padding_mode='border')[0,:, :,0].T

def centers(g):
    b=g.cell_boxes_xyxy.numpy();return (b[:,:2]+b[:,2:])/2

def inside(xy,box):
    return ((xy>=box[:2])&(xy<=box[2:])).all(1)

def transform(H,xy):
    v=np.c_[xy,np.ones(len(xy))]@H.T
    with np.errstate(divide='ignore',invalid='ignore'):out=v[:,:2]/v[:,2:]
    return np.nan_to_num(out,nan=1e3,posinf=1e3,neginf=-1e3)

def get_features(result,q,r,qspec,banks,core,processor_sha):
    aq=old.geometry(q,processor_sha);ar=old.geometry(r,processor_sha)
    qxy=centers(aq);rxy=centers(ar);Hq,Wq=aq.oriented_size_hw
    box=np.array(qspec['box'],float)/np.array([Wq,Hq,Wq,Hq]);mask=inside(qxy,box)
    assert mask.any(),q['id']
    qt=banks.image_tokens(q).float();rt=banks.image_tokens(r,True).float()
    qt=F.normalize(qt,dim=1).cuda();rt=F.normalize(rt,dim=1).cuda();sim=qt@rt.T
    qa=(sample(result['warp_AB'],qxy)+1)/2;ra=(sample(result['warp_BA'],rxy)+1)/2
    confq=sample(result['overlap_AB'],qxy)[:,0];confr=sample(result['overlap_BA'],rxy)[:,0]
    cycle=(sample(result['warp_BA'],qa)+1)/2-torch.as_tensor(qxy,device='cuda',dtype=torch.float32)
    grid=(np.stack(np.meshgrid((np.arange(64)+.5)/64,(np.arange(64)+.5)/64),-1).reshape(-1,2))
    warp=(sample(result['warp_AB'],grid)+1)/2;confidence=sample(result['overlap_AB'],grid)[:,0]
    cyc=(sample(result['warp_BA'],warp)+1)/2-torch.as_tensor(grid,device='cuda',dtype=torch.float32)
    gridw=warp.cpu().numpy();gridc=confidence.cpu().numpy();gridcycle=cyc.norm(dim=1).cpu().numpy()
    keep=inside(grid,box)&(gridc>=.5)&(gridcycle<=.05)&((gridw>=0)&(gridw<=1)).all(1)
    H=None
    if keep.sum()>=4:
        cv2.setRNGSeed(17);H,_=cv2.findHomography(grid[keep],gridw[keep],cv2.RANSAC,.02,maxIters=2000,confidence=.995)
        if H is not None and (not np.isfinite(H).all() or np.linalg.cond(H)>1e8):H=None
    maps={k:result[k][0,...,0].float().cpu().numpy() for k in ['overlap_AB','overlap_BA']}
    u=core['cell_means'](maps['overlap_AB'],aq).float().cuda();v=core['cell_means'](maps['overlap_BA'],ar).float().cuda()
    mass=(u.mean()*v.mean()).sqrt();weighted=sim*v[None]
    free,argfree=sim.max(1);local,arg=weighted.max(1)
    denom=u.sum().clamp_min(1e-12);pooled=lambda z:float((mass*(z*u).sum()/denom).item())
    penalized=lambda xy: (weighted-.1*torch.cdist(torch.as_tensor(xy,device='cuda',dtype=torch.float32),torch.as_tensor(rxy,device='cuda',dtype=torch.float32))).max(1)
    dense,ad=penalized(qa);rolled,ard=penalized(qa.roll(max(1,len(qa)//2),0))
    hp=transform(H,qxy) if H is not None else qa.cpu().numpy()
    hom,ah=penalized(hp);homroll,ahr=penalized(np.roll(hp,max(1,len(hp)//2),axis=0))
    if H is None:hom=local;homroll=local
    homerror=np.linalg.norm(transform(H,grid)-gridw,axis=1) if H is not None else np.full(len(grid),1e3)
    support=inside(grid,box);geo=float((gridc[support]*(homerror[support]<=.02)*(gridcycle[support]<=.05)).mean())
    h,w=maps['overlap_AB'].shape;x,y,X,Y=box
    x0,y0=np.floor(np.array([x,y])*[w,h]).astype(int);x1,y1=np.ceil(np.array([X,Y])*[w,h]).astype(int)
    x0,x1=np.clip([x0,x1],0,w);y0,y1=np.clip([y0,y1],0,h)
    fraction=float(maps['overlap_AB'][y0:y1,x0:x1].sum(dtype=np.float64)/max(maps['overlap_AB'].sum(dtype=np.float64),1e-30))
    feats=dict(M=float(mass),S=pooled(local),G=geo,D=pooled(dense),H=pooled(hom),Droll=pooled(rolled),Hroll=pooled(homroll),MaxSim=float(free.mean()),subject_fraction=fraction,homography_valid=H is not None,homography_points=int(keep.sum()),subject_token_fraction=float(mask.mean()))
    arrays=dict(q_xy=qxy,r_xy=rxy,warp_q_to_r=qa.cpu().numpy(),warp_r_to_q=ra.cpu().numpy(),overlap_q=confq.cpu().numpy(),overlap_r=confr.cpu().numpy(),cycle_q=cycle.cpu().numpy(),u=u.cpu().numpy(),v=v.cpu().numpy(),subject_mask=mask,homography=np.full((3,3),np.nan) if H is None else H,grid_xy=grid,grid_warp=gridw,grid_confidence=gridc,grid_cycle_error=gridcycle,grid_homography_error=homerror,
        maxsim=free.cpu().numpy(),maxsim_arg=argfree.cpu().numpy(),visibility_local=local.cpu().numpy(),visibility_arg=arg.cpu().numpy(),dense_local=dense.cpu().numpy(),dense_arg=ad.cpu().numpy(),homography_local=hom.cpu().numpy(),homography_arg=ah.cpu().numpy(),dense_roll_local=rolled.cpu().numpy(),homography_roll_local=homroll.cpu().numpy())
    return feats,arrays,maps

def run(args):
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_device(0)
    plan=read(OUT/'PLAN.json');m=read(C/'manifest.json');processor_sha=old.recipe_input_hash(m,'weights/colnomic7b/adapter/preprocessor_config.json')
    banks=old.Banks(False);core,_,_=old.math_functions();matcher=old.OriginalRoMa(core)
    qs={q['query']:q for q in plan['queries']};started=time.time();done=0
    tasks=[t for n,t in enumerate(plan['tasks']) if n%args.shards==args.shard]
    if args.limit:tasks=tasks[:args.limit]
    for t in tasks:
        q=plan['query_records'][t['query']];r=plan['references'][str(t['row'])]
        folder=OUT/'pairs'/q['id'];dest=folder/f"{t['row']:06}.npz";receipt=dest.with_suffix('.json')
        if receipt.exists():
            assert read(receipt)['signature']==plan['signature'];done+=1;continue
        ts=time.time();matcher.f.start(q['sha256'],r['sha256']);matcher.fine.start(q['sha256'],r['sha256'])
        with torch.inference_mode():
            result=matcher.model.match(matcher.image(q),matcher.image(r))
            feats,arrays,maps=get_features(result,q,r,qs[q['id']],banks,core,processor_sha)
        assert all(np.isfinite(a).all() for k,a in arrays.items() if k!='homography')
        replay=None;cached=C/'pairs'/q['id']/f"{t['row']:06}.npz"
        if cached.exists() and (done<4 or t['diagnostic']):
            with np.load(cached) as oldpair:
                replay={k:float(np.abs(maps[k]-oldpair[k]).max()) for k in maps}
        if t['diagnostic']:
            for k in ['warp_AB','warp_BA','overlap_AB','overlap_BA']:arrays['native_'+k]=result[k][0].float().cpu().numpy()
        old.save_npz(dest,arrays)
        rec=dict(**t,signature=plan['signature'],features=feats,seconds=time.time()-ts,query_sha=q['sha256'],reference_sha=r['sha256'],query_tokens_sha=q['image_tokens_sha256'],reference_tokens_sha=r['image_tokens_sha256'],native_overlap_replay_maxerror=replay,file_sha256=old.file_hash(dest),coordinate_frame=plan['protocol']['frame'])
        old.atomic_json(receipt,rec);done+=1
        old.atomic_json(OUT/f'worker{args.shard}.json',dict(status='running',completed=done,total=len(tasks),elapsed=time.time()-started,last=rec))
        print(json.dumps(dict(shard=args.shard,done=done,total=len(tasks),query=q['id'],row=t['row'],seconds=round(rec['seconds'],2),features=feats,replay=replay)),flush=True)
        del result,arrays,maps
    old.atomic_json(OUT/f'worker{args.shard}.json',dict(status='complete',completed=done,total=len(tasks),elapsed=time.time()-started))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--limit',type=int);args=ap.parse_args()
    prepare() if args.mode=='prepare' else run(args)
