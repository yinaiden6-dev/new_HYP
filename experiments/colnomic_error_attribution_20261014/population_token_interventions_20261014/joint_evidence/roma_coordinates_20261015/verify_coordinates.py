"""Independent NumPy coordinate and score replay, not the producer reduction."""
from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np
import torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');OUT=D/'roma_coordinates'
sys.path.insert(0,str(P));from experiment_utils import read,write

def bilinear(a,xy):
    h,w=a.shape[:2];p=np.array(xy,dtype=float)*[w,h]-.5;p[:,0]=np.clip(p[:,0],0,w-1);p[:,1]=np.clip(p[:,1],0,h-1)
    lo=np.floor(p).astype(int);hi=np.minimum(lo+1,[w-1,h-1]);u=(p-lo)[:,0,None];v=(p-lo)[:,1,None]
    return (1-u)*(1-v)*a[lo[:,1],lo[:,0]]+u*(1-v)*a[lo[:,1],hi[:,0]]+(1-u)*v*a[hi[:,1],lo[:,0]]+u*v*a[hi[:,1],hi[:,0]]

def run(wait,partial):
    torch.set_num_threads(2);plan=read(OUT/'PLAN.json');started=time.time()
    while len(list((OUT/'pairs').glob('*/*.json')))<len(plan['tasks']):
        if partial:break
        if not wait:raise RuntimeError('Not complete')
        if time.time()-started>16*3600:raise TimeoutError('Completion16h')
        time.sleep(45)
    q=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True)
    r=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
    rows=[];hashes=0
    for t in plan['tasks']:
        path=OUT/'pairs'/t['query']/f"{t['row']:06}.npz";meta=path.with_suffix('.json')
        if not meta.exists():continue
        rec=read(meta);assert rec['signature']==plan['signature'];assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['file_sha256'];hashes+=1
        if not t['diagnostic']:continue
        qa=plan['query_records'][t['query']];ra=plan['references'][str(t['row'])]
        qt=q['embeddings'][qa['bank_index']][qa['layout']['start']:qa['layout']['start']+qa['layout']['count']].float().numpy().astype(float)
        rt=r['embeddings'][ra['bank_index']][ra['layout']['start']:ra['layout']['start']+ra['layout']['count']].float().numpy().astype(float)
        qt/=np.maximum(np.linalg.norm(qt,axis=1,keepdims=True),1e-12);rt/=np.maximum(np.linalg.norm(rt,axis=1,keepdims=True),1e-12);sim=qt@rt.T
        with np.load(path) as z:
            assert z['q_xy'].shape==(len(qt),2) and z['r_xy'].shape==(len(rt),2)
            warp=(bilinear(z['native_warp_AB'],z['q_xy'])+1)/2
            werr=float(abs(warp-z['warp_q_to_r']).max());cerr=float(abs(bilinear(z['native_overlap_AB'],z['q_xy'])[:,0]-z['overlap_q']).max())
            # Producer grid_sample uses FP32 while this reference uses FP64 arithmetic.
            assert werr<2e-4 and cerr<2e-4,(t,werr,cerr)
            u=z['u'].astype(float);v=z['v'].astype(float);mass=np.sqrt(u.mean()*v.mean());weighted=sim*v[None]
            local=weighted.max(1);dense=(weighted-.1*np.linalg.norm(z['warp_q_to_r'][:,None,:]-z['r_xy'][None,:,:],axis=2)).max(1)
            emax=float(abs(sim.max(1)-z['maxsim']).max());eS=float(abs(local-z['visibility_local']).max());eD=float(abs(dense-z['dense_local']).max())
            # torch.cdist's FP32 matrix-product implementation can lose precision
            # near zero distance (sqrt of a cancellation residual). Check the
            # final weighted score tightly, not just individual unweighted cells.
            assert max(emax,eS)<2e-5 and eD<1e-4,(t,emax,eS,eD)
            score=float(mass*np.dot(u,local)/max(u.sum(),1e-12));assert abs(score-rec['features']['S'])<2e-6
            dscore=float(mass*np.dot(u,dense)/max(u.sum(),1e-12));derr=abs(dscore-rec['features']['D']);assert derr<2e-6,(t,derr)
            H=z['homography']
            if np.isfinite(H).all():
                hp=np.c_[z['q_xy'],np.ones(len(qt))]@H.T;hp=hp[:,:2]/hp[:,2:]
                hom=(weighted-.1*np.linalg.norm(hp[:,None,:]-z['r_xy'][None,:,:],axis=2)).max(1)
            else:hom=local
            hscore=float(mass*np.dot(u,hom)/max(u.sum(),1e-12));herr=abs(hscore-rec['features']['H']);assert herr<2e-6,(t,herr)
            rows.append(dict(query=t['query'],row=t['row'],warp_error=werr,confidence_error=cerr,maxsim_error=emax,visibility_error=eS,dense_penalty_error=eD,score_error=abs(score-rec['features']['S']),dense_score_error=derr,homography_score_error=herr))
    result=dict(status='partial_pass' if partial else 'passed',files_sha_verified=hashes,expected_files=len(plan['tasks']),native_diagnostic_pairs_verified=len(rows),expected_diagnostic_pairs=sum(t['diagnostic'] for t in plan['tasks']),checks=rows,
        scope='Every completed pair archive hash checked; all diagnostic native coordinate fields replayed with independent NumPy bilinear interpolation and cosine/distance reductions. FP32 cdist local tolerance1e-4; final pooled score tolerance2e-6. This verifies calculations within stated numerical tolerance, not mechanism uniqueness.')
    if not partial:assert hashes==len(plan['tasks']) and len(rows)==result['expected_diagnostic_pairs']
    write(OUT/('PARTIAL_VERIFICATION.json' if partial else 'INDEPENDENT_VERIFICATION.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--partial',action='store_true');a=ap.parse_args();run(a.wait,a.partial)
