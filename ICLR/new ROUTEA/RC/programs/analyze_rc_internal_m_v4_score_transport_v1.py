#!/usr/bin/env python3
"""Descriptive TRAIN/PROBE score audit, without fitting or model selection."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT = R / 'rc_internal_m_v4_probe_v1/score_transport_audit'


def load(p): return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def verify(b):
    assert bind(b['path']) == b, b['path']
    return Path(b['path'])


def rank(x, t): return 1 + int((x > x[t]).sum())


def main():
    probe = R/'rc_internal_m_v4_probe_v1'
    pv = load(probe/'validation.json');pr = load(verify(pv['result']))
    a = load(verify(pv['authority']));m = load(verify(a['manifest']))
    tv = load(R/'rc_internal_m_condition_scale_v4/validation.json');tr=load(verify(tv['result']))
    for b in load(verify(pr['prelabel_seal']))['predictions']:verify(b)
    for arm in ('PRE_REAL','PRE_SHUFFLED'):
        ep=load(R/'rc_internal_m_condition_scale_v4'/arm/'endpoints/0128/validation.json')
        for b in ep['predictions']:verify(b)
    labels={r['query_id']:r for r in pr['rows']}
    training_mass=np.concatenate([r['M'] for r in m['train_rows']])
    log_bounds=[float(np.log(np.maximum(training_mass,1e-8)).min()),float(np.log(np.maximum(training_mass,1e-8)).max())]
    outputs=[];summaries={}
    for split,inputs in [('TRAIN16',m['train_rows']),('PROBE8',m['probe_rows'])]:
        part=[]
        for row in inputs:
            q=row['query_id'];isprobe=split=='PROBE8'
            root=probe/'predictions' if isprobe else R/'rc_internal_m_condition_scale_v4/PRE_REAL/endpoints/0128'
            modes=('REAL','REAL_CONSTANT','REAL_SHUFFLED') if isprobe else ('native','constant','shuffled')
            pred=[load(root/mode/(q+'.json')) for mode in modes]
            L,C,S=[np.asarray(p['L']) for p in pred]
            parity=R/'rc_prellm_m_adapter_v2/encoder_cache'/q/'parity.json'
            cache_receipt=load(parity.parent/'validation.json');verify(cache_receipt['parity'])
            original=np.asarray(load(parity)['fresh_L0']);mass=np.asarray(row['M']);lm=np.log(np.maximum(mass,1e-8))
            identity=labels[q]['identity'] if isprobe else row['target_id']
            targets=[i for i,x in enumerate(row['candidate_identities']) if x==identity]
            assert len(targets)<=1
            t=targets[0] if targets else None
            paths={'ORIGINAL_L':original,'REAL_L':L,'REAL_CONSTANT_L':C,'REAL_SHUFFLED_L':S}
            direct={k:dict(position=int(v.argmax()),identity=row['candidate_identities'][int(v.argmax())],
                      correct=row['candidate_identities'][int(v.argmax())]==identity,
                      target_rank=rank(v,t) if t is not None else None) for k,v in paths.items()}
            delta=L-C;shuffled_delta=S-C; assigned_shuffled=np.roll(lm,-1)
            rec=dict(split=split,query_id=q,target_identity=identity,target_position=t,
                source_parity=cache_receipt['parity'],direct_content_readout=direct,
                joint_prediction=pred[0]['decision']['prediction_identity'],
                joint_correct=pred[0]['decision']['prediction_identity']==identity,
                mass_log_bounds=[float(lm.min()),float(lm.max())],
                masses_outside_train_range=int(((lm<log_bounds[0])|(lm>log_bounds[1])).sum()),
                correlation_real_delta_with_logM=float(np.corrcoef(delta,lm)[0,1]),
                correlation_shuffled_delta_with_assigned_logM=float(np.corrcoef(shuffled_delta,assigned_shuffled)[0,1]),
                mean_constant_minus_original=float((C-original).mean()),
                std_constant_minus_original=float((C-original).std()),
                mean_real_minus_original=float((L-original).mean()),
                min_real_minus_constant=float(delta.min()),max_real_minus_constant=float(delta.max()))
            if t is not None:
                rec['target_scores']={k:float(v[t]) for k,v in paths.items()}
                rec['target_real_minus_original']=float((L-original)[t])
                rec['nontarget_mean_real_minus_original']=float(np.delete(L-original,t).mean())
            part.append(rec)
        cor=[r['correlation_real_delta_with_logM'] for r in part]
        summary=dict(queries=len(part),target_present=sum(r['target_position'] is not None for r in part),
                     content_correct={k:sum(r['direct_content_readout'][k]['correct'] for r in part) for k in paths},
                     joint_correct=sum(r['joint_correct'] for r in part),
                     logM_delta_query_correlation=dict(min=float(np.min(cor)),median=float(np.median(cor)),max=float(np.max(cor))),
                     probe_or_train_masses_outside_train_range=sum(r['masses_outside_train_range'] for r in part),
                     content_gain_queries=[r['query_id'] for r in part if r['direct_content_readout']['REAL_L']['correct'] and not r['direct_content_readout']['ORIGINAL_L']['correct']],
                     content_break_queries=[r['query_id'] for r in part if not r['direct_content_readout']['REAL_L']['correct'] and r['direct_content_readout']['ORIGINAL_L']['correct']],
                     real_content_correct_but_joint_wrong=[r['query_id'] for r in part if r['direct_content_readout']['REAL_L']['correct'] and not r['joint_correct']])
        summaries[split]=summary;outputs+=part
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='DESCRIPTIVE_SCORE_TRANSPORT_AUDIT_PASS',program=bind(__file__),sources=[pv['result'],tv['result'],a['manifest']],
                rows=outputs,summary=summaries,training_logM_bounds=log_bounds,
                fitting_updates=0,new_encoder_inference=0,new_roma_inference=0,
                limitations=['Direct L argmax is post-hoc diagnostic, not a preselected replacement model.',
                    'Correlation is descriptive within each query; candidates are not independent statistical replicates.',
                    'High correlation does not prove a scalar-only explanation, semantic improvement, or attention localization.',
                    'The primary inference remains 4/8; eight opened probes over six components do not establish generalization.'])
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 内部 M V4：质量输入、内容排序与动作读出的进一步分析','',
           '全部使用封存数据，未训练、未选阈值、未执行新模型推理。以下L直接取第一只用于事后诊断，不替换原定主模型。','',
           '| 口径 | TRAIN16 | PROBE8 |','|---|---:|---:|']
    for label,key in [('原始自由内容L0第一名','ORIGINAL_L'),('适配器真实M后的内容第一名','REAL_L'),('同一适配器内部M换常量后的内容第一名','REAL_CONSTANT_L')]:
        lines.append(f"| {label} | {summaries['TRAIN16']['content_correct'][key]}/16 | {summaries['PROBE8']['content_correct'][key]}/8 |")
    lines+=['| 原定联合训练决策头 | 10/16 | 4/8 |','| TRAIN16冻结读出重训头 | 11/16 | 4/8 |',
            '| TRAIN16外部加性/乘积头 | 12/16 | 6/8 |','',
            '## 已分清的现象','',
            '1. 原始内容与恒定条件不能混为一个基线。婴儿药目标原始L0排名23，真实M后2；同一REAL适配器改常量为39。cefdinir原始2，真实1，常量3。',
            '2. 内部内容输出的直接第一名在probe多认对1张，但原定动作头取消了这次纠错。在TRAIN也有2张内容第一名正确、联合头仍错误。因此当前损失并非全部发生在表示端。',
            '3. 真实M相对于同适配器常量M引起的分数改变量，与logM在各query的128候选之间强正相关。',
            f"   TRAIN16相关系数中位数{summaries['TRAIN16']['logM_delta_query_correlation']['median']:.4f}；PROBE8为{summaries['PROBE8']['logM_delta_query_correlation']['median']:.4f}，范围{summaries['PROBE8']['logM_delta_query_correlation']['min']:.4f}–{summaries['PROBE8']['logM_delta_query_correlation']['max']:.4f}。",
            '   这支持质量数值已通过内部路径影响候选相对分数；不能据此断言学会了文字/logo定位，也不能把1024个候选当成1024个独立样本。',
            '4. 婴儿药目标分数比原始只增加约0.00054，而其余候选平均降低约0.01234。该例排名改善主要发生于相对竞争，不宜说成目标内容被大幅增强。',
            '5. 现有证据允许一种更简单的解释：内部适配器在内容分数中表达了部分质量校准。尚未证明这比外部校准学到了新的patch选择规律。',
            '', '下一步若继续开发，应该围绕保持有用的条件化排序、学习与其匹配的接受规则设计TRAIN内对照；不能根据这8张挑阈值，也不能把事后5/8包装成新独立成功。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()
