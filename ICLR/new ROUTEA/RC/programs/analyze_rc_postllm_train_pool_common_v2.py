#!/usr/bin/env python3
"""Frozen query-independent M direction from TRAIN16 patch distributions.

No fitting: preserve each TRAIN query's per-patch 16-D bottleneck distribution,
evaluate GELU before pooling patches, then average the 16 query means equally.
Only the pooled response direction is query-independent; ordinary content and
the constant-M adapter path remain query-dependent. This is not a scalar scorer.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import json
import numpy as np
import torch
from torch.nn import functional as F
from rc_postllm_m_backend_v1 import load_projection, predict_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
from analyze_rc_postllm_signal_decomposition_v2 import (
    bind, checked, read, write, decision, reference_stats, parse_rows, projected)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_postllm_train_pool_common_v2'
SOURCE = ROOT/'results/rc_postllm_m_signal_decomposition_v2'
POST = ROOT/'results/rc_postllm_m_v1'
AUTH = ROOT/'registry/rc_postllm_m_v1_authority_20260924.json'
MANIFEST = ROOT/'results/rc_prellm_m_adapter_v2/input_manifest.json'
SNAPSHOT = POST/'POST_REAL/snapshots/0128.pt'
ARMS = ['TRAIN_POOL_COMMON','IDEAL_SELF_COMMON',
        'TRAIN_POOL_COMMON_FIXED_ARGMAX','IDEAL_SELF_COMMON_FIXED_ARGMAX']


def save_npz(path, **values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**values)
    os.replace(tmp,path)


def prepare_protocol():
    a=read(AUTH);m=read(MANIFEST)
    sources=[bind(p) for p in [AUTH,MANIFEST,SNAPSHOT,Path(__file__),
        ROOT/'programs/rc_postllm_m_backend_v1.py',ROOT/'programs/rc_prellm_m_adapter_v1.py',
        ROOT/'programs/rc_prellm_m_scaled_adapter_v4.py',
        ROOT/'programs/analyze_rc_postllm_signal_decomposition_v2.py']]
    protocol={'schema':2,'sources':sources,'model':'frozen POST_REAL128',
        'parent_sidecar_source':bind(ROOT/'programs/analyze_rc_postllm_train_pool_common_v1.py'),
        'change_from_v1':'Only upstream residual-decomposition binding/output version changes; pooled nonlinear computation and quantization bridge unchanged.',
        'head':'same frozen joint INTERNAL3; no refit',
        'pool_query_ids':[r['query_id'] for r in m['train_rows']],
        'probe_query_ids':[r['query_id'] for r in m['probe_rows']],
        'all_row_order':[r['query_id'] for r in m['train_rows']+m['probe_rows']],
        'definition':'t_qp=W_content LN(H_qp)+b_down; c=g*(logM-train_mu)/train_sigma; latent_q(c)=mean_p[GELU(t_qp+w_M*c)-GELU(t_qp)]; latent_POOL(c)=mean_q_in_TRAIN16 latent_q(c); direction=s*W_up*latent. No GELU(mean).',
        'base':'actual BF16 hidden after same POST_REAL adapter with standardized M=0',
        'pool_weighting':'equal1/16 per query; each query has equal1/Nq per patch',
        'arms':ARMS,'fixed_argmax':'constant-M best reference token from existing signal-decomposition result',
        'new_training':False,'new_llm_or_roma_forwards':0,'held_fit':False,
        'scope':'opened eight probe queries; retain entire natural C128; descriptive frozen mechanism test'}
    assert len(protocol['pool_query_ids'])==16
    assert not set(protocol['pool_query_ids']) & set(protocol['probe_query_ids'])
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=protocol:
        raise ValueError('sidecar protocol changed since checkpoint')
    write(OUT/'protocol.json',protocol)
    return a,m,bind(OUT/'protocol.json')


def cache_row(row):
    seal=read(ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/row['query_id']/'validation.json')
    return torch.load(checked(seal['payload']),map_location='cpu',weights_only=True),seal['payload']


def latent_tokens(hidden, small):
    return F.linear(F.layer_norm(hidden.float(),(3584,)),small.down.weight[:,:-1],small.down.bias)


def latent_response(t, conditions, small):
    # Preserve every patch's nonlinear activation before reducing patches.
    response=F.gelu(t[None,:,:]+conditions[:,None,None]*small.down.weight[:,-1][None,None,:])
    return (response-F.gelu(t)[None,:,:]).mean(dim=1)


@torch.no_grad()
def build_pool(m, small, protocol):
    path=OUT/'train_pool_latents.npz'
    sealpath=OUT/'train_pool_validation.json'
    if path.exists() and sealpath.exists():
        seal=read(sealpath)
        if seal['protocol']!=protocol:
            raise ValueError('pool protocol mismatch')
        checked(seal['payload'])
        with np.load(path,allow_pickle=False) as z:
            ids=z['query_ids'].tolist();offset=z['offsets'];t=z['latent_tokens']
            assert ids==[r['query_id'] for r in m['train_rows']]
            result=[torch.from_numpy(t[offset[i]:offset[i+1]].copy()) for i in range(16)]
        return result,bind(sealpath)
    arrays=[];ids=[];offsets=[0];bindings=[]
    for row in m['train_rows']:
        cache,binding=cache_row(row)
        t=latent_tokens(cache['hidden'][cache['image_mask']],small)
        arrays.append(t);ids.append(row['query_id']);offsets.append(offsets[-1]+len(t));bindings.append(binding)
        del cache
    save_npz(path,query_ids=np.asarray(ids),offsets=np.asarray(offsets,np.int64),
        latent_tokens=torch.cat(arrays).numpy(),patch_counts=np.diff(offsets),
        query_pool_weights=np.ones(16,np.float64)/16)
    write(sealpath,{'status':'TRAIN16_PER_PATCH_LATENTS_PRESERVED','protocol':protocol,
        'payload':bind(path),'query_ids':ids,'patch_counts':np.diff(offsets).tolist(),
        'source_cache_bindings':bindings,'held_query_count':0,
        'no_GELU_of_mean':True,'equal_query_weights':True})
    return arrays,bind(sealpath)


def curves(arrays, self_t, masses, small, gain):
    c=(torch.tensor(masses,dtype=torch.float32).clamp_min(small.mass_epsilon).log()-small.mass_log_mean)/small.mass_log_std*gain
    pool=torch.zeros((len(masses),16),dtype=torch.float32)
    own=torch.zeros_like(pool)
    for start in range(0,len(masses),16):
        cc=c[start:start+16]
        means=[latent_response(t,cc,small) for t in arrays]
        pool[start:start+16]=torch.stack(means).mean(0)
        own[start:start+16]=latent_response(self_t,cc,small)
    pool_d=F.linear(pool,small.up.weight)*small.residual_scale
    self_d=F.linear(own,small.up.weight)*small.residual_scale
    return c,pool,own,pool_d,self_d


@torch.no_grad()
def validate_direct(m, small, arrays, gain, protocol):
    path=OUT/'pool_direct_validation.json'
    if path.exists():
        if read(path)['protocol']!=protocol:raise ValueError('direct validation protocol mismatch')
        return
    mass=m['train_rows'][0]['M'][0]
    c=(torch.tensor(mass).clamp_min(small.mass_epsilon).log()-small.mass_log_mean)/small.mass_log_std*gain
    # Actual original adapter-up hook, one predeclared engineering mass, all16TRAIN.
    captured=[]
    hook=small.up.register_forward_hook(lambda module,args,out:captured.append(out.detach().clone()))
    direct=[];analytic=[];checks=[]
    try:
        for row,t in zip(m['train_rows'],arrays):
            cache,_=cache_row(row);h=cache['hidden'][cache['image_mask']]
            small.conditioning='constant';small(h,mass);base=captured.pop()*small.residual_scale
            small.conditioning='real';small(h,mass);real=captured.pop()*small.residual_scale
            d=(real-base).mean(0)
            lat=latent_response(t,c.reshape(1),small)[0]
            expected=F.linear(lat,small.up.weight)*small.residual_scale
            err=float((d-expected).abs().max())
            rel=float((d-expected).norm()/d.norm().clamp_min(1e-12))
            if err>1e-5 or rel>1e-4:
                raise ValueError('direct actual-up vs latent pooled residual differs')
            checks.append({'query_id':row['query_id'],'max_abs_error':err,'relative_L2_error':rel})
            direct.append(d);analytic.append(expected)
    finally:hook.remove();small.conditioning='real'
    directmean=torch.stack(direct).mean(0);analyticmean=torch.stack(analytic).mean(0)
    write(path,{'status':'ALL16_ACTUAL_ADAPTER_UP_VS_NONLINEAR_LATENT_POOL_PASS',
        'protocol':protocol,'engineering_M':mass,'query_checks':checks,
        'pool_max_abs_error':float((directmean-analyticmean).abs().max()),
        'pool_relative_L2_error':float((directmean-analyticmean).norm()/directmean.norm().clamp_min(1e-12)),
        'held_queries':0,'new_training':False,'new_encoder_forwards':0})


def summarize():
    files=sorted((OUT/'queries').glob('*/result.json'))
    if not files:return
    labels={r['query_id']:r for r in read(POST/'result.json')['rows']}
    rows=[read(p) for p in files];totals={}
    for split in ['train_engineering','probe']:
        rr=[x for x in rows if x['split']==split]
        if not rr:continue
        totals[split]={}
        for arm in ARMS+['ACTUAL_COMMON_BRIDGE','CPU_NATIVE_REFERENCE','CPU_CONSTANT_REFERENCE']:
            cor=[];resc=[];br=[];changes=[]
            for row in rr:
                q=row['query_id'];lab=labels[q];d=row['arms'][arm]['decision']
                good=d['prediction_identity']==lab['identity'];raw=lab['correct']['RAW']
                if good:cor.append(q)
                if good and not raw:resc.append(q)
                if raw and not good:br.append(q)
                if d['prediction_identity']!=lab['selected']['POST_REAL']:changes.append(q)
            totals[split][arm]={'n':len(rr),'correct':len(cor),'rescues':resc,'breaks':br,
                'changed_vs_sealed_GPU_POST_REAL':changes}
    probes=sum(r['split']=='probe' for r in rows)
    write(OUT/'summary.json',{'status':'ALL8_PROBE_COMPLETE' if probes==8 else 'PARTIAL_ROWS_COMPLETE',
        'completed_probe_queries':probes,'completed_rows':len(rows),'summary':totals,
        'result_bindings':[bind(p) for p in files],'opened_label_source':bind(POST/'result.json'),
        'interpretation':'Tests whether query-independent TRAIN-pooled M direction preserves opened-probe correction. The base content remains query-dependent; not a pure-scalar system or spatial-attention proof.',
        'new_training':False,'held_fit':False,'new_llm_or_roma_forwards':0})


@torch.no_grad()
def process(index,row,model,small,head,pool,a,protocol,pool_binding,started,budget):
    folder=OUT/'queries'/row['query_id'];final=folder/'result.json'
    if final.exists():
        if read(final)['protocol']!=protocol:raise ValueError('completed row protocol mismatch')
        return True
    refrow_path=SOURCE/'queries'/row['query_id']/'result.json'
    if not refrow_path.exists():raise ValueError(f'upstream CPU residual decomposition missing: {refrow_path}')
    refrow=read(refrow_path)
    assert refrow['query_id']==row['query_id'] and refrow['candidate_ids']==row['candidate_ids']
    assert refrow['M']==row['M'] and refrow['arms']['NATIVE']['decision']['theta']==list(head)
    cache,cache_binding=cache_row(row);h=cache['hidden'][cache['image_mask']]
    small.conditioning='constant';h0=small(h,row['M'][0]);small.conditioning='real'
    t=latent_tokens(h,small)
    tick=time.monotonic();cs,pool_l,self_l,pool_d,self_d=curves(pool,t,row['M'],small,a['condition_gain'])
    curve_seconds=time.monotonic()-tick
    curvepath=folder/'condition_curves.npz'
    save_npz(curvepath,M=np.asarray(row['M'],np.float64),scaled_condition=cs.numpy(),
        pool_latent_response=pool_l.numpy(),self_latent_response=self_l.numpy(),
        pool_direction=pool_d.numpy(),self_direction=self_d.numpy())
    completed=[];rowstarted=time.monotonic()
    for i in range(128):
        path=folder/'candidates'/f'{i:04d}.json'
        if path.exists():
            rec=read(path)
            if rec['protocol']!=protocol or rec['candidate_position']!=i:raise ValueError('resume mismatch')
            checked(rec['patch_evidence']);completed.append(rec);continue
        if time.monotonic()-started>=budget:
            write(folder/'progress.json',{'status':'PARTIAL_BUDGET','row_index':index,'candidates':len(completed),
                'total':128,'protocol':protocol});return False
        tick=time.monotonic()
        source_candidate=read(checked(refrow['candidate_results'][i]))
        assert source_candidate['candidate_position']==i and source_candidate['M']==row['M'][i]
        with np.load(checked(source_candidate['patch_evidence']),allow_pickle=False) as z:
            k=z['token_arms'].tolist().index('CONSTANT')
            oldloc=torch.from_numpy(z['best_reference_token'][k].copy())
            actual_common=torch.from_numpy(z['common_hidden_vector'].copy())
        ref=F.normalize(torch.load(checked(row['reference_tokens'][i]),map_location='cpu',weights_only=True)['tokens'].double(),dim=-1)
        fixed_ref=ref[oldloc]
        scores={};stats={};max_values=[];max_indices=[];fixed_values=[]
        for arm,direction in [('TRAIN_POOL_COMMON',pool_d[i]),('IDEAL_SELF_COMMON',self_d[i])]:
            changed=(h0.float()+direction).to(torch.bfloat16)
            tok=projected(model,cache,changed);unit=F.normalize(tok.double(),dim=-1)
            val,idx=(unit@ref.T).max(1)
            fixed=(unit*fixed_ref).sum(1)
            if float((val-fixed).min()) < -1e-12:raise ValueError('fixed assignment exceeded MaxSim')
            scores[arm]=float(val.mean());scores[arm+'_FIXED_ARGMAX']=float(fixed.mean())
            stats[arm]={**reference_stats(idx,len(ref)),
                'argmax_changed_fraction_vs_constant':float((idx!=oldloc).double().mean()),
                'free_minus_fixed_mean':float((val-fixed).mean()),
                'direction_norm':float(direction.norm())}
            max_values.append(val.numpy());max_indices.append(idx.numpy());fixed_values.append(fixed.numpy())
        payload=folder/'patch_evidence'/f'{i:04d}.npz'
        save_npz(payload,arms=np.asarray(ARMS[:2]),maxsim_by_patch=np.stack(max_values),
            best_reference_token=np.stack(max_indices),fixed_argmax_by_patch=np.stack(fixed_values),
            constant_best_reference_token=oldloc.numpy(),pool_direction=pool_d[i].numpy(),
            ideal_self_direction=self_d[i].numpy(),actual_BF16_common_direction=actual_common.numpy(),
            pool_latent=pool_l[i].numpy(),ideal_self_latent=self_l[i].numpy())
        own=self_d[i];pd=pool_d[i]
        dotden=float(pd.norm()*own.norm())
        bridge={'ideal_self_vs_actual_BF16_common_direction_L2':float((own-actual_common).norm()),
            'actual_BF16_common_direction_norm':float(actual_common.norm()),
            'ideal_self_vs_actual_COMMON_L_difference':scores['IDEAL_SELF_COMMON']-source_candidate['scores']['COMMON_ONLY'],
            'pool_self_direction_cosine':float(torch.dot(pd,own))/dotden if dotden>0 else None,
            'pool_self_direction_L2':float((pd-own).norm())}
        rec={'protocol':protocol,'query_id':row['query_id'],'candidate_position':i,
            'candidate_id':row['candidate_ids'][i],'M':row['M'][i],'scaled_condition':float(cs[i]),
            'scores':scores,'stats':stats,'quantization_bridge':bridge,
            'source_candidate':refrow['candidate_results'][i],'patch_evidence':bind(payload),
            'candidate_seconds':time.monotonic()-tick}
        write(path,rec);completed.append(rec)
        if (i+1)%32==0:
            print(json.dumps({'row':index,'query_id':row['query_id'],'candidates':i+1,
                'candidate_seconds':time.monotonic()-rowstarted,'curve_seconds':curve_seconds}),flush=True)
    arms={arm:{'L':[c['scores'][arm] for c in completed]} for arm in ARMS}
    for arm in ARMS:arms[arm]['decision']=decision(row,arms[arm]['L'],head)
    for arm,old in [('ACTUAL_COMMON_BRIDGE','COMMON_ONLY'),('CPU_NATIVE_REFERENCE','NATIVE'),('CPU_CONSTANT_REFERENCE','CONSTANT')]:
        arms[arm]=refrow['arms'][old]
    write(final,{'status':'FROZEN_TRAIN_POOL_COMMON_ROW_COMPLETE','protocol':protocol,
        'query_id':row['query_id'],'row_index':index,'split':'probe' if index>=16 else 'train_engineering',
        'candidate_ids':row['candidate_ids'],'M':row['M'],'raw_scores':row['raw_scores'],
        'arms':arms,'cache_binding':cache_binding,'pool_binding':pool_binding,'curve_binding':bind(curvepath),
        'source_decomposition':bind(refrow_path),'source_CPU_GPU_anchors':refrow['cpu_gpu_anchors'],
        'curve_seconds':curve_seconds,'candidate_seconds_sum':sum(c['candidate_seconds'] for c in completed),
        'candidate_results':[bind(folder/'candidates'/f'{i:04d}.json') for i in range(128)],
        'held_fit':False,'new_training':False,'new_llm_or_roma_forwards':0})
    print(json.dumps({'row_complete':index,'curve_seconds':curve_seconds,
        'candidate_seconds_sum':sum(c['candidate_seconds'] for c in completed)}),flush=True)
    summarize();return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',default='16-23',help='indices in fixed TRAIN16 then PROBE8; default probe8')
    ap.add_argument('--budget',type=float,default=480)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args()
    if not 1<=args.threads<=2 or args.budget<=0:ap.error('threads1..2; budgetpositive')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.summarize_only:summarize();return
    started=time.monotonic();a,m,protocol=prepare_protocol()
    cp=torch.load(SNAPSHOT,map_location='cpu',weights_only=True)
    assert cp['step']==128 and cp['arm']=='POST_REAL' and cp['authority']==bind(AUTH)
    small=ScaledQualityResidualAdapter(hidden_size=3584,bottleneck=16,condition_gain=a['condition_gain'])
    small.load_state_dict(cp['adapter']);small.eval().requires_grad_(False)
    model=load_projection(a,'cpu');write(OUT/'projection_loading.json',model.report)
    with torch.no_grad():
        pool,pool_binding=build_pool(m,small,protocol)
        validate_direct(m,small,pool,a['condition_gain'],protocol)
        rows=m['train_rows']+m['probe_rows'];selected=parse_rows(args.rows,len(rows))
        for index in selected:
            if not process(index,rows[index],model,small,cp['head'].double().tolist(),pool,a,protocol,
                pool_binding,started,args.budget):
                print(json.dumps({'status':'PARTIAL_BUDGET','row':index,'elapsed_seconds':time.monotonic()-started}),flush=True);return
    summarize();print(json.dumps({'status':'SELECTED_ROWS_COMPLETE','rows':selected,
        'elapsed_seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':main()
