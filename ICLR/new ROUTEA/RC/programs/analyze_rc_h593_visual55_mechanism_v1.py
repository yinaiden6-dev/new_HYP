#!/usr/bin/env python3
"""Explain frozen visual interventions from existing candidate and head caches."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_h593_cached_visual_trend_v1 as B

OUT=ROOT/'results/rc_h593_visual55_mechanism_v1'
FEATURES=('RAW','S','M','L','Q','R','bias')


def aggregate(rows, getter):
    groups=sorted({r['component'] for r in rows})
    values=[statistics.mean(getter(r) for r in rows if r['component']==g) for g in groups]
    return dict(mean=statistics.mean(values),negative_groups=sum(v<0 for v in values),
        positive_groups=sum(v>0 for v in values),groups=len(groups),
        leave_one_group_out=[min((sum(values)-v)/(len(values)-1) for v in values),max((sum(values)-v)/(len(values)-1) for v in values)])


def main():
    import torch
    torch.set_num_threads(1)
    a,snap=B.guard(); v=B.read(B.OUT/'validation.json'); result=B.read(B.checked(v['result']))
    old={r['query_id']:r for r in B.read(B.checked(a['old_result']))['rows']}
    labels={r['physical_row']:r['identity'] for r in B.read(B.checked(a['gallery']))['records']}
    rows=[]; sources=[]; max_mass_closure=0.; max_head_closure=0.
    for rec in snap['records']:
        vis=B.read(B.checked(rec['visual'])); rr=old[rec['query_id']]; axis=vis['candidate_physical_rows']
        targets=[i for i,x in enumerate(axis) if labels[x]==rr['identity']]
        if not targets:continue
        target=targets[0]; wrong=[i for i in range(128) if i!=target]
        if 'reused_prediction' in rec: pb=rec['reused_prediction']
        else:pb=B.read(B.OUT/f"query{rec['index']:03d}/validation.json")['payload']
        pred=B.read(B.checked(pb)); sources.extend([pb,pred['features'],rec['visual']])
        feature=torch.load(B.checked(pred['features']),map_location='cpu',weights_only=True)
        theta=feature['theta'].tolist(); assert theta==[float.fromhex(x) for x in pred['head']['theta_hex']]
        challenger=pred['row']['challenger_positions']; winner=pred['row']['winner']
        def vectors(path):
            values=[[0.]*7 for _ in range(128)]
            for i,x in zip(challenger,feature['X'][path].tolist()):values[i]=[xx*tt for xx,tt in zip(x,theta[:-1])]+[theta[-1]]
            return values
        def logits(path):
            vals=[0.]*128
            for i,z in zip(challenger,pred['models'][path]['logits_hex']):vals[i]=float.fromhex(z)
            return vals
        native_path='FULL_UV/native/HR1'; native_vectors=vectors(native_path); native_z=logits(native_path)
        head_wrong=max(wrong,key=lambda i:native_z[i]); native_head_margin=native_z[target]-native_z[head_wrong]
        native_M=vis['masses']['NATIVE']['HR1']; mass_wrong=max(wrong,key=lambda i:native_M[i])
        row=dict(index=rec['index'],query_id=rec['query_id'],original_query_id=rr['original_query_id'],component=rr['component'],
            target_physical_row=axis[target],native_max_M_wrong=axis[mass_wrong],native_max_head_wrong=axis[head_wrong],arms={})
        for arm in B.ARMS:
            masses=vis['masses'][arm]['HR1']; next_mass_wrong=max(wrong,key=lambda i:masses[i])
            assert min(native_M[target],masses[target],native_M[mass_wrong],masses[mass_wrong],masses[next_mass_wrong])>0
            lt=math.log(masses[target]/native_M[target]); lw=math.log(masses[mass_wrong]/native_M[mass_wrong]); shift=math.log(masses[mass_wrong]/masses[next_mass_wrong])
            gap=math.log(masses[target]/masses[next_mass_wrong])-math.log(native_M[target]/native_M[mass_wrong])
            error=abs(gap-(lt-lw+shift));max_mass_closure=max(error,max_mass_closure);assert error<2e-12
            detail=dict(target_log_ratio=lt,fixed_wrong_log_ratio=lw,strongest_wrong_log_ratio=math.log(masses[next_mass_wrong]/native_M[mass_wrong]),
                mean_wrong_log_ratio=math.log(sum(masses[i] for i in wrong)/sum(native_M[i] for i in wrong)),
                target_vs_fixed_wrong_change=lt-lw,new_wrong_competition_term=shift,total_log_ratio_change=gap,
                max_M_wrong_changed=next_mass_wrong!=mass_wrong,new_max_M_wrong=axis[next_mass_wrong],heads={})
            for mode in ('M_ONLY','FULL_UV'):
                path=mode+'/visual/'+arm+'/HR1'; vz=vectors(path); z=logits(path); next_wrong=max(wrong,key=lambda i:z[i])
                contribution=[vz[target][j]-vz[head_wrong][j]-native_vectors[target][j]+native_vectors[head_wrong][j] for j in range(7)]
                competition=z[head_wrong]-z[next_wrong]
                total=z[target]-z[next_wrong]-native_head_margin
                error=abs(total-(sum(contribution)+competition));max_head_closure=max(max_head_closure,error);assert error<2e-10
                assert abs(contribution[0])<2e-12 and abs(contribution[6])<2e-12
                detail['heads'][mode]=dict(feature_terms=dict(zip(FEATURES,contribution)),fixed_head_wrong=axis[head_wrong],new_head_wrong=axis[next_wrong],
                    head_wrong_changed=next_wrong!=head_wrong,competition_term=competition,total_margin_change=total)
            row['arms'][arm]=detail
        rows.append(row)
    assert len(rows)==52 and len({r['component'] for r in rows})==31
    summary={}
    fields=('target_log_ratio','fixed_wrong_log_ratio','strongest_wrong_log_ratio','mean_wrong_log_ratio','target_vs_fixed_wrong_change','new_wrong_competition_term','total_log_ratio_change')
    for arm in B.ARMS:
        s={f:aggregate(rows,lambda r:r['arms'][arm][f]) for f in fields}
        s['max_M_wrong_changed_queries']=sum(r['arms'][arm]['max_M_wrong_changed'] for r in rows)
        for field in fields[:4]:s[field]['geometric_ratio']=math.exp(s[field]['mean'])
        s['heads']={}
        for mode in ('M_ONLY','FULL_UV'):
            s['heads'][mode]={f:aggregate(rows,lambda r:r['arms'][arm]['heads'][mode]['feature_terms'][f]) for f in FEATURES}
            for f in ('competition_term','total_margin_change'):s['heads'][mode][f]=aggregate(rows,lambda r:r['arms'][arm]['heads'][mode][f])
            s['heads'][mode]['wrong_changed_queries']=sum(r['arms'][arm]['heads'][mode]['head_wrong_changed'] for r in rows)
        summary[arm]=s
    result=dict(status='VISUAL55_EXISTING_EVIDENCE_DECOMPOSITION_COMPLETE',source_result=v['result'],snapshot=a['snapshot'],sources=sources,
        all_queries=55,target_present_queries=52,groups=31,summary=summary,rows=rows,new_gpu_forwards=0,new_training=False,
        selection='Target-present subset of fixed55; three target-absent cases excluded only from mechanistic target comparisons',
        decomposition='Delta log(Mtarget/Mmaxwrong) = target log change - fixed-native-wrong log change + log(Mfixedwrong_new/Mmaxwrong_new)',
        head_decomposition='Fixed-native strongest wrong head candidate: seven weighted feature differences plus new-competitor term; no refit',
        scope='Conditional interventions on RoMa input with original ColNomic contents. Highest-M wrong is distinct from highest-head-score wrong. No causal claim about a unique semantic visual factor.')
    B.write(OUT/'result.json',result)
    B.write(OUT/'validation.json',dict(status='MASS_AND_HEAD_DECOMPOSITION_CLOSURE_PASS',result=B.bind(OUT/'result.json'),
        max_log_mass_closure_error=max_mass_closure,max_head_margin_closure_error=max_head_closure,
        raw_and_bias_terms_unchanged=True,zero_mass_floor_used=False,script=B.bind(__file__)))
    lines=['# 55张已有结果：质量尺度、候选竞争与原头通路','', '本轮只分析已有55张，不按准确率显著性扩采。机制比较使用目标在C128的52张、31组；原折COST1、原ColNomic内容、C128与阈值固定。', '',
        '表中倍数为干预后/原图的按组等权几何比，1为不变。固定错误候选指原图M最高的错误reference；新最高错误允许换成其他reference，尚不是小头最终选择的错误候选。', '',
        '|干预|正确reference M|原最高M错误reference M|干预后最高错误M|错误候选平均M|最高M错误换人|','|---|---:|---:|---:|---:|---:|']
    names={'Q_GRAY':'query灰度','R_GRAY':'reference灰度','Q_LOWPASS':'query低通','R_LOWPASS':'reference低通','Q_SHUFFLE':'query打乱','R_SHUFFLE':'reference打乱'}
    for arm,s in summary.items():
        nums=[s[f]['geometric_ratio'] for f in fields[:4]]
        lines.append('|'+names[arm]+'|'+'|'.join(f'{v:.3f}×' for v in nums)+f"|{s['max_M_wrong_changed_queries']}/52|")
    lines+=['','## 质量区分退化怎样发生','',
        '灰度并不主要表现为正确reference的M大幅下降：query灰度下target约不变，但干预后最强错误M上升；reference灰度兼有target略降和错误竞争增强。不能由最终准确率不变说颜色无用。','',
        '低通和query打乱时，正确reference与原先最强错误reference的M都明显下降；但另一些错误reference变得更有竞争力。只跟踪原先一个错误候选会漏掉整个C128中对手的替换。这是M相对区分力下降的重要组成，不能简单归结为整体M变小。','',
        'reference打乱下，正确reference的M保留比例高于原先最高M错误reference；若只看固定那一对，平均区分反而改善。但错误候选替换后，总体target/最强wrong比值仍下降。它直接说明必须把固定对手变化与候选集合竞争分开。','',
        '这些是受控视觉干预下的机制证据，不自动证明原始未干预错误也全由同一原因导致。分块打乱带来新边缘，低通同时破坏多种细节，仍不能给文字、logo、纹理分配独立因果贡献。','',
        '## 如何经原头传递（固定原生最强错误对手）','',
        '以下为target相对固定原生最强错误head候选的margin变化贡献，按组等权。竞争项是干预后最强错误head候选变化带来的额外项，非新的模型参数。各项之和经过逐query校验，精确恢复margin总变化。','',
        '|干预/通路|S贡献|M贡献|L贡献|Q贡献|R贡献|对手替换项|总margin变化|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        for mode in ('M_ONLY','FULL_UV'):
            h=s['heads'][mode]
            lines.append('|'+names[arm]+'/'+mode+'|'+'|'.join(f"{h[k]['mean']:+.4f}" for k in ('S','M','L','Q','R','competition_term','total_margin_change'))+'|')
    lines+=['','RAW项与bias贡献在本对照中为零。M_ONLY改变M及所有依赖其缩放的特征，并非只修改小头中M这一列；完整u/v还会改变自由内容的权重和归一化局部相似度。贡献分解只解释这个冻结头，不等于特征独立不可替代性。','',
        '索引70继续既定收尾。是否以后补128，只应由具体未解决机制问题是否缺少样本/干预决定，不能把准确率p值未显著作为自动触发条件。']
    (ROOT/'reports/REPORT_H593_VISUAL55_MECHANISM_DECOMPOSITION_20260923.md').write_text('\n'.join(lines)+'\n')
    for arm,s in summary.items():
        print(arm, {f:round(s[f]['mean'],5) for f in fields},'head_M_ONLY', {f:round(s['heads']['M_ONLY'][f]['mean'],5) for f in ('S','M','L','Q','R','competition_term','total_margin_change')},flush=True)
    print('CLOSURE',max_mass_closure,max_head_closure,flush=True)


if __name__=='__main__':main()
