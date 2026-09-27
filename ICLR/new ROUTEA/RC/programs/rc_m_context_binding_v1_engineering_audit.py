#!/usr/bin/env python3
"""Read-only first-wave engineering audit; no inference or scientific selection."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
ROOT=RC/'results/rc_m_context_binding_v1'
ARMS=['NATIVE','X_PLUS__001','X_PLUS__110','X_PLUS__111','X_MINUS__111','Y_PLUS__110','Y_PLUS__111']
SHARDS=[0,1,2,3]


def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())
def load(b):
    assert binding(b['path'])==b
    return torch.load(b['path'],weights_only=True,mmap=True,map_location='cpu')


def main():
    p=read(ROOT/'protocol.json');pb=binding(ROOT/'protocol.json')
    state=load(p['head_state'])['head'];weight=state['norm.weight'];bias=state['norm.bias']
    import run_rc_m_phase_newgroups_cpu_v1 as geometry_only
    records=[];errors={k:0. for k in ['LN','variance','covariance','mean_mass','pool','common_confidence_RMS']}
    for shard in SHARDS:
        w=p['workers'][shard];folder=ROOT/'pairs'/f'{shard:03d}'
        cap=load(w['capture']);features=[load(b)['features'] for b in w['features']]
        geometries=geometry_only.geometries(w)
        invariant=read(folder/'input_invariants.json');assert invariant['protocol']==pb and len(invariant['records'])==32
        assert all(x['inverse_roll_bit_exact'] and x['selected_Gram_relative_error']<1e-12 for x in invariant['records'])
        assert all(x['full_P_FFT_amplitude_relative_error'] is None or x['full_P_FFT_amplitude_relative_error']<1e-12 for x in invariant['records'])
        for arm in ARMS:
            receipt=read(folder/(arm+'.json'));assert receipt['protocol']==pb
            z=load(receipt['payload']);ln=load(receipt['intermediates'])
            assert z['protocol']==ln['protocol']==pb and z['position']==w['position']
            shift=(0,0) if arm=='NATIVE' else tuple(p['directions'][arm.split('__')[0]])
            bits='000' if arm=='NATIVE' else arm.split('__')[1]
            masses=[]
            for side in [0,1]:
                a=[torch.roll(t,shift,(1,2)) if bits[0]=='1' else t for t in features[side]]
                j=cap['sides'][side]['J'];pp=cap['sides'][side]['P']
                if bits[1]=='1':j=torch.roll(j,shift,(1,2))
                if bits[2]=='1':pp=torch.roll(pp,shift,(1,2))
                s=(a[-1]+j)+pp;original=(features[side][-1]+cap['sides'][side]['J'])+cap['sides'][side]['P']
                expected=F.layer_norm(s.reshape(1,-1,1024),(1024,),weight,bias,eps=1e-5).reshape_as(s)
                err=float((ln['LN_sum'][side]-expected).abs().max());errors['LN']=max(errors['LN'],err);assert err==0
                if bits=='111':
                    assert torch.equal(s,torch.roll(original,shift,(1,2)))
                    nl=F.layer_norm(original.reshape(1,-1,1024),(1024,),weight,bias,eps=1e-5).reshape_as(s)
                    assert torch.equal(ln['LN_sum'][side],torch.roll(nl,shift,(1,2)))
                x=[a[-1].double(),j.double(),pp.double()];mu=[t.mean(-1) for t in x]
                saved=z['sides'][side]['preLN']
                for i,key in enumerate(['var_A1','var_J','var_P']):
                    val=x[i].square().mean(-1)-mu[i].square();err=float((val-saved[key]).abs().max())
                    errors['variance']=max(errors['variance'],err);assert err<1e-10
                for i,k,key in [(0,1,'cov_AJ'),(0,2,'cov_AP'),(1,2,'cov_JP')]:
                    val=(x[i]*x[k]).mean(-1)-mu[i]*mu[k];err=float((val-saved[key]).abs().max())
                    errors['covariance']=max(errors['covariance'],err);assert err<1e-10
                conf=z['sides'][side]['confidence']
                if arm=='NATIVE':assert torch.equal(conf,cap['sides'][side]['confidence'])
                probability=conf[0,...,0].sigmoid().double().numpy();height,width=probability.shape;expected_pool=[]
                for box,valid in zip(geometries[side].cell_boxes_xyxy,geometries[side].valid_patch_mask):
                    if not valid:expected_pool.append(0.);continue
                    left,top,right,bottom=map(float,box)
                    l=max(0,min(width-1,math.floor(left*width)));t=max(0,min(height-1,math.floor(top*height)))
                    r=max(l+1,min(width,math.ceil(right*width)));b=max(t+1,min(height,math.ceil(bottom*height)))
                    expected_pool.append(float(probability[t:b,l:r].mean()))
                err=float(np.max(abs(np.asarray(expected_pool)-z['sides'][side]['weights'].numpy())))
                errors['pool']=max(errors['pool'],err);assert err<2e-12
                masses.append(math.fsum(expected_pool)/len(expected_pool))
                if bits=='111':
                    unrolled=torch.roll(conf,(-4*shift[0],-4*shift[1]),(1,2))
                    rms=float((unrolled.double()-cap['sides'][side]['confidence'].double()).square().mean().sqrt())
                    err=abs(rms-z['sides'][side]['common_grid_diagnostic']['inverse_rolled_confidence_RMS'])
                    errors['common_confidence_RMS']=max(errors['common_confidence_RMS'],err);assert err<1e-12
            err=abs(math.sqrt(masses[0]*masses[1])-z['M']);errors['mean_mass']=max(errors['mean_mass'],err);assert err<2e-12
            records.append(dict(shard=shard,index=w['index'],arm=arm,receipt=binding(folder/(arm+'.json'))))
        print(json.dumps(dict(shard=shard,checked_arms=len(ARMS),errors=errors)),flush=True)
    result=dict(status='CONTEXT_BINDING_FIRST_WAVE_ENGINEERING_REVIEW_PASS',reviewer_code=binding(__file__),protocol=pb,
        fixed_shards=SHARDS,fixed_arms=ARMS,checked_candidate_arms=len(records),sides=2*len(records),
        independent_max_errors=errors,records=records,native_confidence_replay_exact=True,
        common_SUM_and_actual_LN_equivariance_exact=True,scientific_partial_group_claim=False,
        scope='Read-only engineering review of the first four scheduled shards; no effect-size or accuracy conclusion.',
        encoder_matcher_or_DPT_forwards=0,experiment_outputs_changed=False)
    path=ROOT/'engineering_review/first_wave.json';path.parent.mkdir(exist_ok=True)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(result['status'],flush=True)


if __name__=='__main__':main()
