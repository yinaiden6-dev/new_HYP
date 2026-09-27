#!/usr/bin/env python3
"""Independent arithmetic audit, then labels and mechanism contrasts."""
from __future__ import annotations
import argparse
import json
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from run_rc_m_phase_bridge_v1 import ROOT, OUT, read, write, bind, checked, load_context


def internal_logits(row,L,theta):
    raw=np.asarray(row['raw_scores'],np.float64);L=np.asarray(L,np.float64)
    w=row['winner_index'];others=row['challenger_positions'];z=np.zeros(128)
    std=float(np.sqrt(np.mean((raw-raw.mean())**2)))
    for c in others:
        z[c]=theta[0]*(raw[c]-raw[w])/max(std,1e-12)+theta[1]*(L[c]-L[w])/(abs(L[c])+abs(L[w])+1e-12)+theta[2]
    return z


@torch.no_grad()
def independent_patch(ctx,pos,arm):
    p,pb,item,ad,head,cache,refs,proj=ctx
    h=cache['hidden'][cache['image_mask']]
    if arm=='CONSTANT': condition=torch.zeros((len(h),1),dtype=torch.float32)
    else:
        m=torch.as_tensor(item['masses'][arm][pos],dtype=torch.float32)
        c=((m.clamp_min(ad.mass_epsilon).log()-ad.mass_log_mean)/ad.mass_log_std)*ad.condition_gain
        condition=c.expand(len(h),1)
    x=torch.cat([F.layer_norm(h.float(),(3584,)),condition],dim=1)
    residual=F.linear(F.gelu(F.linear(x,ad.down.weight,ad.down.bias)),ad.up.weight,ad.up.bias)*ad.residual_scale
    hidden=cache['hidden'].clone();hidden[cache['image_mask']]=h+residual.to(h.dtype)
    z=F.linear(hidden,proj.weight,proj.bias)+F.linear(F.linear(hidden,proj.lora_a),proj.lora_b)*proj.scaling
    z=z/z.norm(dim=-1,keepdim=True);z=z*cache['batch']['attention_mask'].unsqueeze(-1)
    z=z[cache['image_mask']].double();z=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    sim=torch.mm(z,refs[pos].t());values,indices=torch.max(sim,dim=1)
    return values.numpy(),indices.numpy()


def audit(index):
    folder=OUT/'queries'/f'{index:02d}';path=folder/'result.json'
    receipt=folder/'independent_validation.json'
    if receipt.exists():
        v=read(receipt);assert v['result']==bind(path) and v['protocol']==bind(OUT/'protocol.json')
        for b in v['candidate_results']: checked(b)
        for b in v['patches']: checked(b)
        return v
    ctx=load_context(index);p,pb,item,*_=ctx
    result=read(path);assert result['protocol']==pb and result['query_id']==item['query_id']
    L=np.zeros((len(p['worlds']),128));maximum=0.;forward_error=0.;patches=[]
    for pos,b in enumerate(result['candidate_results']):
        rec=read(checked(b));assert rec['protocol']==pb and rec['position']==pos
        assert rec['candidate_id']==item['post_row']['candidate_ids'][pos]
        patches.append(rec['patch_evidence'])
        with np.load(checked(rec['patch_evidence']),allow_pickle=False) as data:
            assert data['worlds'].tolist()==p['worlds']
            assert np.isfinite(data['maxsim']).all()
            assert data['argmax'].shape==data['maxsim'].shape
            assert (data['argmax']>=0).all() and (data['argmax']<len(ctx[6][pos])).all()
            for j,arm in enumerate(p['worlds']):
                expected_m=0. if arm=='CONSTANT' else item['masses'][arm][pos]
                assert float(data['M'][j])==expected_m
            avg=data['maxsim'].mean(axis=1);L[:,pos]=avg
            maximum=max(maximum,float(np.max(abs(avg-np.asarray(rec['scores'])))),float(np.max(abs(avg-data['L']))))
            if pos in (0,64):
                for j,arm in enumerate(p['worlds']):
                    vals,locs=independent_patch(ctx,pos,arm)
                    forward_error=max(forward_error,float(np.max(abs(vals-data['maxsim'][j]))))
                    assert np.array_equal(locs,data['argmax'][j])
    sys.path.insert(0,str(ROOT/'src'))
    from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
    e=item['external'];w=e['winner'];c=e['challengers'];row=item['post_row']
    for j,arm in enumerate(p['worlds']):
        resultarm=result['worlds'][arm]
        maximum=max(maximum,float(np.max(abs(L[j]-resultarm['L']))))
        z=internal_logits(row,L[j],result['head'])
        maximum=max(maximum,float(np.max(abs(z-resultarm['POST']['logits128']))))
        for model in ('NATIVE7','M_FREE') if arm!='CONSTANT' else ():
            m=np.asarray(item['masses'][arm]);native=np.asarray(e['native_scores']);oldm=np.asarray(e['original_M'])
            candidates={}
            for k in range(128):
                if model=='NATIVE7': s,q,r=native[k]*(m[k]/oldm[k])
                else: s=q=r=m[k]*e['free_content'][k]
                candidates[k]=dict(real_score=s,visibility_mass=m[k],query_control_score=q,reference_control_score=r)
            features=np.stack([candidate_feature(row['raw_scores'],candidates,k,w).numpy() for k in c])
            theta=e['native_theta'] if model=='NATIVE7' else e['simple_theta']
            score=np.zeros(128);score[c]=features@np.asarray(theta[:-1])+theta[-1]
            maximum=max(maximum,float(np.max(abs(score-resultarm['external'][model]['logits128']))))
        all_models={'POST':resultarm['POST'],**resultarm.get('external',{})}
        for rr in all_models.values():
            zz=np.asarray(rr['logits128']);best=c[int(np.argmax(zz[c]))]
            assert rr['prediction_position']==(best if zz[best]>0 else w)
    assert maximum<2e-10 and forward_error<2e-10,(maximum,forward_error)
    v=dict(status='PHASE_BRIDGE_INDEPENDENT_ROW_PASS',protocol=pb,result=bind(path),query_id=item['query_id'],
           maximum_arithmetic_error=maximum,maximum_forward_error=forward_error,
           forward_positions=[0,64],forward_worlds=p['worlds'],candidate_results=result['candidate_results'],patches=patches)
    write(receipt,v);return v


def group_summary(rows):
    groups={}
    for r in rows: groups.setdefault(r['component'],[]).append(r['value'])
    x=np.asarray([np.mean(v) for v in groups.values()])
    if not len(x): return {'groups':0}
    rng=np.random.default_rng(20260927);draw=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(groups=len(x),queries=len(rows),mean=float(x.mean()),
                exploratory95=np.quantile(draw,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum()))


def join(budget):
    start=time.monotonic();p=read(OUT/'protocol.json');pb=bind(OUT/'protocol.json')
    for b in p['code_sources']: checked(b)
    missing=[i for i in range(8) if not (OUT/'queries'/f'{i:02d}'/'result.json').exists()]
    if missing: print(json.dumps(dict(waiting=missing)),flush=True);return 75
    audits=[]
    for i in range(8):
        if time.monotonic()-start>budget-30: return 75
        audits.append(audit(i));print(json.dumps(dict(validated_query=i)),flush=True)
    # All candidate predictions passed before labels are read for the summaries.
    source=read(checked(p['source']));labels={r['query_id']:r for r in source['phase_rows']}
    inputs=read(checked(p['inputs']))['rows'];query_stats=[];decompositions=[];decision_counts={}
    global_arms=[k for k in labels[inputs[0]['query_id']]['mass_by_arm'] if k.startswith('GLOBAL')]
    local_arms=[k for k in labels[inputs[0]['query_id']]['mass_by_arm'] if k.startswith('LOCAL')]
    for i,item in enumerate(inputs):
        result=read(OUT/'queries'/f'{i:02d}'/'result.json');lab=labels[item['query_id']]
        # Selection of wrong competitor was sealed by unchanged free content in the source analysis.
        if not lab['target_present']: continue
        t=lab['target_position'];w=lab['arm_metrics']['NATIVE']['strongest_position'];identity=item['post_row']['candidate_identities'][t]
        assert item['post_row']['candidate_identities'][w]!=identity
        metrics={}
        for arm,v in result['worlds'].items():
            if arm=='CONSTANT': continue
            lm=np.log(item['masses'][arm]);l=np.asarray(v['L'])
            measures={'logM_target_wrong':float(lm[t]-lm[w]),'POST_content_target_wrong':float(l[t]-l[w])}
            for model,rr in {'POST':v['POST'],**v['external']}.items():
                z=np.asarray(rr['logits128']);measures[model+'_target_wrong']=float(z[t]-z[w])
                measures[model+'_target_HOLD']=float(z[t]);measures[model+'_correct']=float(item['post_row']['candidate_identities'][rr['prediction_position']]==identity)
            metrics[arm]=measures
        contrasts={}
        for metric in metrics['NATIVE']:
            if metric.endswith('_correct'): continue
            contrasts[metric]={kind:float(np.mean([metrics[k+'/'+kind][metric] for k in local_arms])-
                                         np.mean([metrics[k+'/'+kind][metric] for k in global_arms]))
                               for kind in ['FULL','SCALE','RELATIVE']}
            for phase in global_arms+local_arms:
                b=metrics['NATIVE'][metric];f=metrics[phase+'/FULL'][metric]
                s=metrics[phase+'/SCALE'][metric];r=metrics[phase+'/RELATIVE'][metric]
                ps=.5*((s-b)+(f-r));pr=.5*((r-b)+(f-s))
                assert abs(ps+pr-(f-b))<1e-10
                decompositions.append(dict(query_id=item['query_id'],component=lab['component'],phase=phase,metric=metric,
                                           scale=ps,relative=pr,interaction=f-s-r+b,total=f-b))
        query_stats.append(dict(query_id=item['query_id'],component=lab['component'],fold=item['fold'],
                                target_position=t,fixed_wrong_position=w,metrics=metrics,contrasts=contrasts))
    summary={}
    for metric in query_stats[0]['contrasts']:
        summary[metric]={kind:group_summary([dict(component=r['component'],value=r['contrasts'][metric][kind]) for r in query_stats])
                         for kind in ('FULL','SCALE','RELATIVE')}
    # Reproduce the already sealed upstream LOCAL-GLOBAL result, independent of downstream outcomes.
    prior=source['phase_contrasts']['strongest_content_logM_gap']['LOCAL_minus_GLOBAL']['group_equal_mean']
    assert abs(summary['logM_target_wrong']['FULL']['mean']-prior)<1e-12
    validation=dict(status='PHASE_BRIDGE_PANEL8_INDEPENDENT_PASS',protocol=pb,rows=8,
                    pair_worlds=8*128*len(p['worlds']),audits=[bind(OUT/'queries'/f'{i:02d}'/'independent_validation.json')for i in range(8)],
                    max_arithmetic_error=max(v['maximum_arithmetic_error']for v in audits),
                    max_forward_error=max(v['maximum_forward_error']for v in audits),
                    labels_read_after_all_candidate_audits=True)
    output=dict(status='PHASE_BRIDGE_PANEL8_COMPLETE',protocol=pb,queries=8,target_present=len(query_stats),
                summary_LOCAL_minus_GLOBAL=summary,rows=query_stats,decomposition=decompositions,
                scope='Opened eight-query panel, original natural C128 and held-fold fixed heads. COARSE interventions; HR1-trained readouts.',
                limitation='Small exploratory panel; M-only model recovery is structural. No unique causal explanation, sufficiency, or new accuracy claim.')
    write(OUT/'validation.json',validation);write(OUT/'result.json',output)
    lines=['# 同一相位干预贯通外部与内部：结果','',
           '8张预定面板、7张目标在原自然C128；所有端点固定，不训练、不新增GPU。下表为LOCAL−GLOBAL：负数表示正确身份相对固定内容错误对手的优势减小。不同模型logit单位不同，不比较绝对大小。','',
           '|读出|完整变化|仅共同尺度|仅相对变化|完整变化探索性95%区间|','|---|---:|---:|---:|---|']
    for key in ['logM_target_wrong','POST_content_target_wrong','NATIVE7_target_wrong','M_FREE_target_wrong','POST_target_wrong']:
        s=summary[key];lines.append(f"|{key}|{s['FULL']['mean']:.8f}|{s['SCALE']['mean']:.8f}|{s['RELATIVE']['mean']:.8f}|{s['FULL']['exploratory95']}|")
    lines+=['','## 解释边界','',
            '- 同一相位干预已通过相同候选轴送入冻结外部和内部模型；结果不预设两条路径同方向。',
            '- SCALE保持M的所有候选排序和log比值。若内部输出仍改变，定位的是绝对标度敏感性，而非新增身份信息。',
            '- RELATIVE保持query的平均logM，检验候选对比变化；这是数值接口干预，不声称对应自然图像。',
            '- Shapley分解仅针对本次定义的共同/相对两因素，不是唯一视觉因果分解。',
            '- 小面板已打开；独立核算不等于独立外部验证，区间未作多重比较校正。',
            '- 全部逐patch、候选、头输出、来源SHA与二因素分解保存在结果目录。',
            '',f"结果：`{OUT}`；验收：`{validation['status']}`。"]
    report=ROOT/'reports/REPORT_M_PHASE_EXTERNAL_INTERNAL_BRIDGE_V1_20260927.md'
    report.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status=output['status'],summary=summary)),flush=True)
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=float,default=480);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);sys.exit(join(a.budget))
