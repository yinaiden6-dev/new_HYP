#!/usr/bin/env python3
"""Fixed pooled-feature readouts on the existing 71-query natural-C128 panel.

No encoder, learned readout, threshold selection, or matcher replay. Compute
and seal all new scores before the identity join. This is an opened panel.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_pair_relation_localization_v1/feature_readout'
CACHE = ROOT/'results/rc_h593_feature_fusion_cache_v1'
SNAPSHOT = ROOT/'results/rc_h71_feature_fusion_pilot_v1/snapshot.json'
PARENT = ROOT/'registry/rc_h593_feature_fusion_train_authority_v2_20260922.json'
GRID = 8
TEMP = .1


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''): h.update(b)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    assert bind(b['path']) == b, ('SHA_DRIFT', b['path'])
    return Path(b['path'])


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text() == s, ('IMMUTABLE', p)
    else:
        tmp = p.with_suffix(p.suffix+'.tmp'); tmp.write_text(s); tmp.replace(p)


def prepare():
    parent = read(PARENT); snap = read(SNAPSHOT)
    assert [r['execution_ordinal'] for r in snap['records']] == list(range(71))
    write(OUT/'spec.json', dict(
        status='FIXED_OPENED_PANEL_DESCRIPTIVE_READOUT_SPEC', program=bind(__file__),
        snapshot=bind(SNAPSHOT), ready=bind(CACHE/'ready.json'), parent=bind(PARENT),
        identities=parent['join_sources']['curator'], gallery=parent['public_sources']['gallery'],
        historical_result=bind(ROOT/'results/rc_h71_feature_fusion_pilot_v1/result.json'),
        panel='Existing execution ordinals 0..70, all 128 natural candidates each',
        pooling='Valid patch centers assigned to fixed 8x8 image-coordinate bins; equal patch average inside each occupied bin; no empty-bin token',
        sources={'COL':'Original ColNomic128 pooled then per-cell L2',
                 'ROMA_COARSE':'coarse_11 and coarse_17 each raw1024 pooled then separately L2; concatenate divided by sqrt2; no random compression'},
        mean_descriptor='Per-source mean of valid ORIGINAL patches, separately L2 per coarse component; cosine',
        readouts=['forward_maxsim','reverse_maxsim','bidirectional_maxsim','global_mean_cosine',
                  'mutual_top1_fraction','unique_hit_coverage','soft_cycle'],
        soft_temperature=TEMP, grid=GRID, dtype='float32 scores; native M and full Col L preserved from cache',
        controls=['Same underlying pooled tokens for all local content/structure statistics',
                  'Structure summaries contain no position after fixed pooling',
                  'All new scores sealed before new identity join; no labels used by scoring'],
        observation='Already opened panel; existing result first row was inspected for schema before this spec. This is not a fresh preregistered test.',
        no_training=True, no_new_encoders=True, no_new_roma_forward=True,
        limitations=['Pooled approximation, not full token exact MaxSim',
                    'Fixed generic readouts cannot establish absence of information or optimality',
                    'Comparing scores does not causally attribute native M to these statistics',
                    'Single-image encoders followed by pair comparison differ from independent scalar SINGLE arm',
                    'Argmax ranking diagnostic is different from frozen COST1 HOLD/SWITCH'],
        relation_match_diagnostic='Within each query sort by coarse bidirectional MaxSim; adjacent pairs with absolute score gap <=0.01, original candidate order resolves ties; descriptive correlation of signed M/structure differences only',
        cpu_threads=4))


def norm(x):
    return F.normalize(x, dim=-1)


def pool(block, mask, bins):
    x = block.float()[mask]
    sums = torch.zeros(GRID*GRID, x.shape[1]); sums.index_add_(0, bins, x)
    n = torch.bincount(bins, minlength=GRID*GRID)
    keep = n > 0
    return norm(sums[keep]/n[keep,None]), norm(x.mean(0))


def load_image(bound):
    p = torch.load(checked(bound), map_location='cpu', weights_only=True, mmap=True)
    mask = p['valid_patch_mask'].bool()
    centers = ((p['cell_boxes_xyxy'][:,:2]+p['cell_boxes_xyxy'][:,2:])/2)[mask]
    xy = torch.floor(centers*GRID).long().clamp(0,GRID-1)
    bins = xy[:,1]*GRID+xy[:,0]
    col, cm = pool(p['original_colnomic_tokens'],mask,bins)
    parts = [pool(p['components'][k],mask,bins) for k in ('coarse_11','coarse_17')]
    coarse = torch.cat([x[0] for x in parts],1)/math.sqrt(2)
    gm = torch.cat([x[1] for x in parts])/math.sqrt(2)
    return {'COL':(col,cm), 'ROMA_COARSE':(coarse,gm)}


def stats(q, r):
    qt,qg = q; rt,rg = r; sim = qt@rt.T
    mx, qi = sim.max(1); my, ri = sim.max(0)
    nq,nr = sim.shape
    mutual_q = ri[qi] == torch.arange(nq)
    mutual_r = qi[ri] == torch.arange(nr)
    a = torch.softmax(sim/TEMP,1); b = torch.softmax(sim/TEMP,0)
    return dict(forward_maxsim=float(mx.mean()), reverse_maxsim=float(my.mean()),
                bidirectional_maxsim=float((mx.mean()+my.mean())/2),
                global_mean_cosine=float(qg@rg),
                mutual_top1_fraction=float((mutual_q.float().mean()+mutual_r.float().mean())/2),
                unique_hit_coverage=float((len(qi.unique())/nr+len(ri.unique())/nq)/2),
                soft_cycle=float((a*b).sum()/math.sqrt(nq*nr))), sim


def numpy_check(q,r,observed):
    qn,qg = [x.double().numpy() for x in q]; rn,rg = [x.double().numpy() for x in r]
    s=qn@rn.T; qi=s.argmax(1); ri=s.argmax(0); nq,nr=s.shape
    e=np.exp(s/TEMP-(s/TEMP).max(1,keepdims=True)); a=e/e.sum(1,keepdims=True)
    e=np.exp(s/TEMP-(s/TEMP).max(0,keepdims=True)); b=e/e.sum(0,keepdims=True)
    expected=dict(forward_maxsim=float(s.max(1).mean()),reverse_maxsim=float(s.max(0).mean()),
                  bidirectional_maxsim=float((s.max(1).mean()+s.max(0).mean())/2),global_mean_cosine=float(qg@rg),
                  mutual_top1_fraction=float(((ri[qi]==np.arange(nq)).mean()+(qi[ri]==np.arange(nr)).mean())/2),
                  unique_hit_coverage=float((len(set(qi))/nr+len(set(ri))/nq)/2),
                  soft_cycle=float((a*b).sum()/math.sqrt(nq*nr)))
    err={k:abs(v-observed[k]) for k,v in expected.items()}
    assert max(err.values())<2e-6, err
    return max(err.values())


def score():
    sp=read(OUT/'spec.json'); checked(sp['program']); snap=read(checked(sp['snapshot']))
    ready=read(checked(sp['ready'])); records=snap['records']; bank={}; start=time.monotonic()
    def get(k):
        if k not in bank: bank[k]=load_image(ready['features'][k])
        return bank[k]
    results=[]; errors=[]; used=set()
    for m in records:
        dest=OUT/'scores'/f"query{m['execution_ordinal']:03d}.json"
        if dest.exists():
            p=read(dest); assert p['spec']==bind(OUT/'spec.json'); results.append(bind(dest)); continue
        p=torch.load(checked(m['payload']),map_location='cpu',weights_only=True)
        assert p['query_id']==m['query_id'] and len(p['pairs'])==128
        q=get(p['query_image_key']); used.add(p['query_image_key'])
        raw=list(map(float,p['candidate_raw_scores'])); native=[float(v['native_c4'][1]) for v in p['pairs']]
        values={'RAW':raw,'NATIVE_M':native}; diag=[]
        if p['query_id'] in snap['cached_initial']:
            old=torch.load(checked(snap['cached_initial'][p['query_id']]),map_location='cpu',weights_only=True)
            assert old['query_id']==p['query_id']; values['COL_FULL_FORWARD_MAXSIM']=old['C4'][:,0].tolist()
        for j,pair in enumerate(p['pairs']):
            r=get(pair['image_key']); used.add(pair['image_key'])
            for source in ('COL','ROMA_COARSE'):
                s,sim=stats(q[source],r[source])
                for name,x in s.items(): values.setdefault(source+'/'+name,[]).append(x)
                if m['execution_ordinal'] in (0,35,70) and j==0:
                    errors.append(numpy_check(q[source],r[source],s))
                if j==0: diag.append(dict(source=source,nq=sim.shape[0],nr=sim.shape[1]))
        assert all(len(v)==128 and np.isfinite(v).all() for v in values.values())
        write(dest,dict(spec=bind(OUT/'spec.json'),query_id=p['query_id'],execution_ordinal=m['execution_ordinal'],
                        source=m['payload'],candidate_physical_rows=p['candidate_physical_rows'],winner=p['winner'],
                        scores=values,geometry_counts=diag,new_identity_label_reads=0))
        results.append(bind(dest))
        if m['execution_ordinal']%10==0: print(dict(done=len(results),loaded_images=len(bank),seconds=time.monotonic()-start),flush=True)
    write(OUT/'score_seal.json',dict(status='ALL71_C128_SCORES_SEALED',spec=bind(OUT/'spec.json'),queries=71,candidates=9088,
                                      predictions=results,new_identity_label_reads=0,used_image_count=len(used),
                                      numpy_stat_checks=len(errors),max_numpy_error=max(errors,default=0.)))


def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12: return None
    return float(np.corrcoef(x,y)[0,1])


def summarize():
    sp=read(OUT/'spec.json'); seal=read(OUT/'score_seal.json'); assert seal['spec']==bind(OUT/'spec.json')
    predictions=[read(checked(b)) for b in seal['predictions']]
    roles={r['query_id']:r for r in read(checked(sp['identities']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(sp['gallery']))['records']}
    hist={r['query_id']:r for r in read(checked(sp['historical_result']))['rows']}
    rows=[]; methods=list(predictions[0]['scores']); matched=[]
    for p in predictions:
        qid=p['query_id']; role=roles[qid]; old=hist[qid]; cand=p['candidate_physical_rows']; w=p['winner']
        targets=[i for i,c in enumerate(cand) if labels[c]==role['identity']]; assert len(targets)<=1
        row=dict(query_id=qid,component=role['component'],target_position=targets[0] if targets else None,
                 raw_correct=old['correct']['RAW'],original_cost1_correct=old['correct']['ORIGINAL_COST1'],scores={})
        mass=np.array(p['scores']['NATIVE_M'])
        for method in methods:
            v=np.array(p['scores'][method]); selected=int(v.argmax()); iscorrect=labels[cand[selected]]==role['identity']
            item=dict(selected_physical_row=cand[selected],top1_correct=iscorrect,correlation_with_M=corr(v,mass))
            if targets:
                t=targets[0]; wrong=np.arange(128)!=t
                item.update(target_minus_raw=float(v[t]-v[w]),target_minus_max_wrong=float(v[t]-v[wrong].max()),
                            target_rank_min=int((v>v[t]).sum()+1),target_rank_max=int((v>=v[t]).sum()))
            row['scores'][method]=item
        rows.append(row)
        base=np.array(p['scores']['ROMA_COARSE/bidirectional_maxsim']); order=np.argsort(base,kind='stable')
        for i,j in zip(order[:-1],order[1:]):
            if base[j]-base[i] <= .01:
                rec=dict(query_id=qid,component=role['component'],i=int(i),j=int(j),maxsim_gap=float(base[j]-base[i]),M_gap=float(mass[j]-mass[i]))
                for metric in ('global_mean_cosine','mutual_top1_fraction','unique_hit_coverage','soft_cycle'):
                    v=p['scores']['ROMA_COARSE/'+metric];rec[metric+'_gap']=v[j]-v[i]
                matched.append(rec)
    raw_correct=sum(r['raw_correct'] for r in rows); rescued=[r for r in rows if not r['raw_correct'] and r['original_cost1_correct']]
    summary={}
    for method in methods:
        present=[r for r in rows if r['target_position'] is not None]
        groups=defaultdict(list); rhos=defaultdict(list); margin_groups=defaultdict(list)
        for r in rows:
            groups[r['component']].append(float(r['scores'][method]['top1_correct'])-float(r['raw_correct']))
            c=r['scores'][method]['correlation_with_M']
            if c is not None:rhos[r['component']].append(c)
            if r['target_position'] is not None:margin_groups[r['component']].append(r['scores'][method]['target_minus_max_wrong'])
        summary[method]=dict(correct=sum(r['scores'][method]['top1_correct'] for r in rows),n=71,
                             rescue_vs_RAW=sum(not r['raw_correct'] and r['scores'][method]['top1_correct'] for r in rows),
                             breaks_vs_RAW=sum(r['raw_correct'] and not r['scores'][method]['top1_correct'] for r in rows),
                             original_cost1_rescues_target_above_raw=sum(r['scores'][method]['target_minus_raw']>0 for r in rescued),
                             original_cost1_rescues_top1=sum(r['scores'][method]['top1_correct'] for r in rescued),
                             target_above_all_wrong=sum(r['scores'][method]['target_minus_max_wrong']>0 for r in present),
                             mean_target_rank=float(np.mean([r['scores'][method]['target_rank_min'] for r in present])),
                             group_equal_top1_delta_vs_RAW=float(np.mean([np.mean(v) for v in groups.values()])),
                             group_equal_mean_target_margin=float(np.mean([np.mean(v) for v in margin_groups.values()])),
                             group_equal_mean_within_query_correlation_with_M=float(np.mean([np.mean(v) for v in rhos.values()])) if rhos else None)
    matchsummary={}
    for key in ('global_mean_cosine_gap','mutual_top1_fraction_gap','unique_hit_coverage_gap','soft_cycle_gap'):
        querygroups=defaultdict(list)
        for x in matched:querygroups[x['query_id']].append(x)
        groupcorr=defaultdict(list)
        for qid,rr in querygroups.items():
            c=corr([x[key] for x in rr],[x['M_gap'] for x in rr])
            if c is not None:groupcorr[roles[qid]['component']].append(c)
        matchsummary[key]=dict(pooled_descriptive_correlation=corr([x[key] for x in matched],[x['M_gap'] for x in matched]),
                               group_equal_mean_within_query_correlation=float(np.mean([np.mean(v) for v in groupcorr.values()])) if groupcorr else None)
    write(OUT/'matched_candidates.json',dict(spec=seal['spec'],records=matched))
    write(OUT/'result.json',dict(status='FIXED_FEATURE_READOUT_71_COMPLETE',spec=seal['spec'],score_seal=bind(OUT/'score_seal.json'),
                                  queries=71,groups=len({r['component'] for r in rows}),candidates=9088,
                                  target_present=sum(r['target_position'] is not None for r in rows),RAW=raw_correct,
                                  ORIGINAL_COST1=sum(r['original_cost1_correct'] for r in rows),original_cost1_rescues=len(rescued),
                                  summary=summary,matched_candidates=len(matched),matched_summary=matchsummary,rows=rows,
                                  evidence='Fixed pooled-feature ranking and descriptive correlations; no native-matcher causal attribution'))
    table=['|固定读出|正确/71|救/损 vs RAW|原COST1救回中 target > RAW|','|---|---:|---:|---:|']
    for k,s in summary.items():table.append(f"|{k}|{s['correct']}|{s['rescue_vs_RAW']}/{s['breaks_vs_RAW']}|{s['original_cost1_rescues_target_above_raw']}/{len(rescued)}|")
    report='\n'.join(['# 固定单图特征读出：已有71张自然C128面板','',
        '所有9,088个候选均参与。固定8×8有效patch池化，无训练、无模型前向；先封存新分数再做标签join。',
        'ROMA_COARSE使用保存的单图coarse_11/17各1024维；两个块分别归一化后等权合并。COL使用原128维。全局均值读出使用原patch均值。',
        '局部内容与结构统计使用完全相同池化token；除构建池化单元外不读坐标。纯argmax是读出诊断，不是COST1的HOLD/SWITCH策略。','',
        f"RAW={raw_correct}/71，原七参数COST1={sum(r['original_cost1_correct'] for r in rows)}/71，原救回={len(rescued)}；groups={len({r['component'] for r in rows})}。",'',
        *table,'','结构差值与M差值的相关性仅为描述，不能证明M来自该结构，也不能由某个固定读出失败推断冻结特征缺信息。',
        '71张此前已打开；本轮冻结前为查schema读取过既有结果首行，不宣称全盲预注册。',
        '原六臂SINGLE对query/reference分别出标量再相乘，其同query相对M对比消去query公共因子；它不覆盖这里的单图向量配对比较。',''])
    (OUT/'report_zh.md').write_text(report)
    print(json.dumps({k:{x:s[x] for x in ('correct','rescue_vs_RAW','breaks_vs_RAW','original_cost1_rescues_target_above_raw')} for k,s in summary.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','score','join']); args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    {'prepare':prepare,'score':score,'join':summarize}[args.stage]()
