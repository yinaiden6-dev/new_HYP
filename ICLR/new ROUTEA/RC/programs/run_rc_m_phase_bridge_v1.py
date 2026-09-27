#!/usr/bin/env python3
"""Frozen, CPU-only same-intervention replay; no fitting or upstream forwards."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from run_rc_h593_simple_explanations_v1 import read, write, bind, checked, sym
from run_rc_postllm_h593_v1 import Inputs
from rc_postllm_m_backend_v1 import load_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
from analyze_rc_postllm_signal_decomposition_v2 import projected, decision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_m_phase_external_internal_bridge_v1'
POST = ROOT/'results/rc_postllm_h593_decomposition_v1/protocol.json'
SOURCE = ROOT/'results/rc_mass_discrimination_source_v1/result.json'
PLAN = ROOT/'plan/RC_M_PHASE_EXTERNAL_INTERNAL_BRIDGE_V1_20260927.md'


def worlds(masses, original):
    base = np.log(np.asarray(masses['NATIVE'], np.float64))
    assert np.isfinite(base).all()
    values = {'NATIVE': np.exp(base), 'ORIGINAL_HR1': np.asarray(original, np.float64)}
    closure = 0.
    for arm, x in masses.items():
        if arm == 'NATIVE':
            continue
        lm = np.log(np.asarray(x, np.float64))
        shift = float((lm-base).mean())
        values[arm+'/FULL'] = np.asarray(x, np.float64)
        values[arm+'/SCALE'] = np.exp(base+shift)
        values[arm+'/RELATIVE'] = np.exp(lm-shift)
        closure = max(closure, float(np.max(np.abs(
            np.log(values[arm+'/SCALE'])+np.log(values[arm+'/RELATIVE'])-base-lm))))
    assert all(np.isfinite(v).all() and (v > 0).all() and (v <= 1).all() for v in values.values())
    assert closure < 1e-12
    return values, closure


def external(row, masses):
    """Recompute ALL descendants including spatial-control contrast features."""
    w, c = row['winner'], np.asarray(row['challengers'])
    m = np.asarray(masses, np.float64)
    source = np.asarray(row['native_scores'], np.float64)
    scaled = source * (m/np.asarray(row['original_M']))[:, None]
    s, q, r = scaled.T
    lc = s/np.maximum(m, 1e-12)
    raw = np.asarray(row['native_X'])[:, 0]
    x = np.stack([raw, sym(s[c],s[w]), sym(m[c],m[w]), sym(lc[c],lc[w]),
                  sym((s-q)[c],(s-q)[w]), sym((s-r)[c],(s-r)[w])], 1)
    free = np.asarray(row['free_content'])
    y = np.stack([raw, sym((m*free)[c],(m*free)[w]), sym(m[c],m[w]),
                  sym(free[c],free[w]), np.zeros(127), np.zeros(127)],1)
    answer = {}
    for name, xx, theta in [('NATIVE7',x,row['native_theta']),('M_FREE',y,row['simple_theta'])]:
        t = np.asarray(theta)
        z = xx @ t[:-1]+t[-1]
        allz = np.zeros(128); allz[c] = z
        best = int(c[int(z.argmax())]); pick = best if float(z.max()) > 0 else w
        answer[name] = {'features':xx.tolist(),'logits128':allz.tolist(),'prediction_position':pick}
    return answer


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source = read(SOURCE)
    assert source['queries'] == 8 and source['endpoint'] == 'COARSE'
    pp = read(POST)
    manifest = read(checked(pp['manifest']))
    byq = {r['query_id']:r for r in manifest['rows']}
    folds = {r['query_id']:r['fold'] for r in pp['rows']}
    cachepath = ROOT/'results/rc_h593_simple_explanations_v1/cache.json'
    cache = read(cachepath); bycache = {r['query_id']:r for r in cache['rows']}
    ea = read(ROOT/'registry/rc_h593_simple_explanations_authority_v1_20260924.json')
    records = []; errors = []
    for src in source['phase_rows']:
        q = src['query_id']; row=byq[q]; ext=bycache[q]; fold=folds[q]
        assert src['axis'] == ext['axis'] == row['candidate_ids']
        payload_binding=ea['sources'][ext['execution_ordinal']]['payload']
        native=read(checked(payload_binding))['modes']['NATIVE']
        e = dict(query_id=q,fold=fold,winner=ext['winner'],challengers=ext['challengers'],
                 original_M=ext['mass'],free_content=ext['free_content'],native_X=ext['native_X'],
                 native_scores=[[v[k] for k in ('real_score','query_control_score','reference_control_score')]
                                for v in native['scores']],
                 native_theta=[float.fromhex(v) for v in cache['native_parameters'][fold]['COST1']],
                 simple_theta=[float.fromhex(v) for v in cache['prior_product_parameters'][fold]['COST1']],
                 source=payload_binding)
        assert np.max(np.abs(np.asarray(row['M'])-ext['mass'])) < 1e-14
        assert ext['winner'] == row['winner_index']
        values,closure=worlds(src['mass_by_arm'],row['M'])
        native_err=float(np.max(np.abs(np.asarray(external(e,row['M'])['NATIVE7']['features'])-ext['native_X'])))
        assert native_err < 2e-10
        errors.append(native_err)
        records.append(dict(query_id=q,fold=fold,index=src['index'],post_row=row,external=e,
                            masses={k:v.tolist() for k,v in values.items()},log_decomposition_error=closure))
    inputs_path=OUT/'inputs.json'
    inputs=dict(rows=records,labels_included=False)
    if inputs_path.exists(): assert read(inputs_path)==inputs
    else: write(inputs_path,inputs)
    sources=[Path(__file__),ROOT/'programs/join_rc_m_phase_bridge_v1.py',PLAN,
             ROOT/'programs/rc_prellm_m_scaled_adapter_v4.py',ROOT/'programs/rc_prellm_m_adapter_v1.py',
             ROOT/'programs/rc_postllm_m_backend_v1.py',ROOT/'programs/run_rc_postllm_h593_v1.py',
             ROOT/'programs/analyze_rc_postllm_signal_decomposition_v2.py',
             ROOT/'programs/run_rc_h593_simple_explanations_v1.py']
    p=dict(status='FROZEN_PHASE_BRIDGE_PANEL8',inputs=bind(inputs_path),source=bind(SOURCE),
           post_protocol=bind(POST),authority=pp['authority'],manifest=pp['manifest'],endpoints=pp['endpoints'],
           code_sources=[bind(s) for s in sources],queries=8,candidates=128,
           max_external_native_feature_error=max(errors),
           worlds=list(records[0]['masses'])+['CONSTANT'],training=False,gpu=False,
           primary='Mean of four LOCAL output contrasts minus four GLOBAL output contrasts; same COARSE NATIVE baseline.',
           decomposition='logM shift across128 candidates plus zero-mean relative change, 2-factor exact Shapley',
           known_limit='Internal only receives M by construction. Recovery is a replay check, not independent proof of sufficiency.')
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==p
    else: write(OUT/'protocol.json',p)
    print(json.dumps(dict(status=p['status'],queries=8,worlds=len(p['worlds']),native_error=max(errors))),flush=True)


def load_context(index):
    p=read(OUT/'protocol.json');pb=bind(OUT/'protocol.json')
    for b in p['code_sources']: checked(b)
    item=read(checked(p['inputs']))['rows'][index]
    a=read(checked(p['authority']))
    cp=torch.load(checked(p['endpoints'][str(item['fold'])]['snapshot']),map_location='cpu',weights_only=True)
    assert cp['arm']=='POST_REAL' and cp['authority']==p['authority'] and cp['input_scope']['manifest']==p['manifest']
    adapter=ScaledQualityResidualAdapter(hidden_size=3584,bottleneck=a['bottleneck'],
                   condition_gain=a['condition_gain'],residual_scale=a['residual_scale'])
    adapter.load_state_dict(cp['adapter']);adapter.eval().requires_grad_(False)
    bank=Inputs(p['authority'],'cpu',cpu_gb=2,gpu_gb=.5)
    cache=bank.cache_cpu(item['post_row']);refs=bank.refs(item['post_row'])
    projection=load_projection(a,'cpu')
    return p,pb,item,adapter,cp['head'].double().tolist(),cache,refs,projection


@torch.no_grad()
def compute_candidate(ctx,pos):
    p,pb,item,adapter,head,cache,refs,projection=ctx
    hidden=cache['hidden'][cache['image_mask']]
    vals=[];locs=[];ms=[];scores=[]
    for arm in p['worlds']:
        constant=arm=='CONSTANT';adapter.conditioning='constant' if constant else 'real'
        mass=0. if constant else item['masses'][arm][pos]
        tokens=projected(projection,cache,adapter(hidden,mass))
        unit=F.normalize(tokens.double(),dim=-1)
        best,idx=(unit@refs[pos].T).max(1)
        vals.append(best.numpy());locs.append(idx.numpy().astype(np.int32));ms.append(mass)
        scores.append(float(best.mean()))
    return np.stack(vals),np.stack(locs),np.asarray(ms),np.asarray(scores)


def worker(index,budget,threads,limit):
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    start=time.monotonic();folder=OUT/'queries'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=load_context(index);p,pb,item,adapter,head,cache,refs,projection=ctx
        if (folder/'result.json').exists():
            assert read(folder/'result.json')['protocol']==pb
            return 0
        completed=[]; fresh=0
        for pos in range(128):
            path=folder/'candidates'/f'{pos:03d}.json'
            if path.exists():
                rec=read(path);assert rec['protocol']==pb and rec['position']==pos;checked(rec['patch_evidence'])
                completed.append(rec);continue
            if time.monotonic()-start > budget-20 or (limit and fresh>=limit):
                write(folder/'progress.json',dict(query_id=item['query_id'],pairs=len(completed),total=128,protocol=pb))
                return 75
            tick=time.monotonic();patch,locations,mass,scores=compute_candidate(ctx,pos)
            data=folder/'patch'/f'{pos:03d}.npz';data.parent.mkdir(parents=True,exist_ok=True)
            tmp=data.with_name(data.name+f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f: np.savez_compressed(f,worlds=np.asarray(p['worlds']),M=mass,
                                      maxsim=patch,argmax=locations,L=scores)
            os.replace(tmp,data)
            rec=dict(protocol=pb,query_id=item['query_id'],position=pos,candidate_id=item['post_row']['candidate_ids'][pos],
                     patch_evidence=bind(data),scores=scores.tolist(),seconds=time.monotonic()-tick)
            write(path,rec);completed.append(rec);fresh+=1
            print(json.dumps(dict(index=index,pairs=pos+1,total=128,seconds=rec['seconds'])),flush=True)
        values=np.asarray([x['scores'] for x in completed]).T
        arms={}
        for i,arm in enumerate(p['worlds']):
            post=decision(item['post_row'],values[i],head)
            z=np.zeros(128);z[item['post_row']['challenger_positions']]=post['logits']
            arms[arm]=dict(L=values[i].tolist(),POST=dict(logits128=z.tolist(),prediction_position=post['prediction_position']))
            if arm!='CONSTANT': arms[arm]['external']=external(item['external'],item['masses'][arm])
        original=read(ROOT/'results/rc_postllm_h593_decomposition_v1/queries'/item['query_id']/'result.json')
        anchors={}
        for new,old in [('ORIGINAL_HR1','NATIVE'),('CONSTANT','CONSTANT')]:
            e=float(np.max(np.abs(np.asarray(arms[new]['L'])-original['arms'][old]['L'])))
            same=arms[new]['POST']['prediction_position']==original['arms'][old]['decision']['prediction_position']
            assert e<2e-10 and same,('ORIGINAL_REPLAY',e,same)
            anchors[new]=dict(max_L_error=e,same_decision=same)
        write(folder/'result.json',dict(status='PHASE_BRIDGE_ROW_COMPLETE',protocol=pb,query_id=item['query_id'],fold=item['fold'],
              worlds=arms,head=head,anchors=anchors,elapsed_seconds=time.monotonic()-start,
              native_hidden_source=item['post_row'].get('hidden_validation'),
              candidate_results=[bind(folder/'candidates'/f'{i:03d}.json') for i in range(128)]))
        return 0


def pool(budget,workers,threads):
    def launch(i):
        cmd=[sys.executable,str(Path(__file__)),'worker','--index',str(i),'--budget',str(budget),'--threads',str(threads)]
        path=OUT/'worker_logs'/f'{i:02d}.txt';path.parent.mkdir(exist_ok=True)
        with path.open('a') as f: return subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
    todo=[i for i in range(8) if not (OUT/'queries'/f'{i:02d}'/'result.json').exists()]
    start=time.monotonic();codes=[]
    # At most one batch per allocation; no second full-budget wave overruns Slurm.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        codes=list(ex.map(launch,todo[:workers]))
    if any(c not in (0,75) for c in codes): return 1
    done=sum((OUT/'queries'/f'{i:02d}'/'result.json').exists() for i in range(8))
    write(OUT/'progress.json',dict(complete_queries=done,total=8,worker_codes=codes,seconds=time.monotonic()-start))
    return 0 if done==8 else 75


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','worker','pool'])
    ap.add_argument('--index',type=int,default=0);ap.add_argument('--budget',type=float,default=450)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='worker': sys.exit(worker(a.index,a.budget,a.threads,a.limit))
    else: sys.exit(pool(a.budget,a.workers,a.threads))
