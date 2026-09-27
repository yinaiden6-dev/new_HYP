#!/usr/bin/env python3
"""Read-only, no-inference second review of sealed complete context-binding data."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_context_binding_v1'
OUT = ROOT / 'independent_interpretation'
REPORT = RC / 'reports/REPORT_M_CONTEXT_BINDING_INDEPENDENT_INTERPRETATION_20260927.md'
DIRS = ['X_PLUS', 'X_MINUS', 'Y_PLUS', 'Y_MINUS']
BITS = [''.join(b) for b in itertools.product('01', repeat=3)]
CONTRASTS = {
    'aligned_vs_misaligned': {'000': .5, '111': .5, '001': -.5, '110': -.5},
    'P_fixed_context_shift': {'110': 1., '000': -1.},
    'common_grid_shift': {'111': 1., '000': -1.},
    'P_only_shift': {'001': 1., '000': -1.},
    'J_P_interaction_A_fixed': {'011': 1., '010': -1., '001': -1., '000': 1.},
    'J_P_interaction_A_shifted': {'111': 1., '110': -1., '101': -1., '100': 1.},
    'A_J_P_interaction': {'111': 1., '110': -1., '101': -1., '100': 1.,
                         '011': -1., '010': 1., '001': 1., '000': -1.},
}


def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def check(b):
    assert bind(b['path']) == b
    return Path(b['path'])
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if p.exists(): assert p.read_text() == s
    else: p.write_text(s)
def stats(a):
    a = np.asarray(a, dtype=np.float64)
    idx = np.random.default_rng(20260927).integers(0, len(a), size=(10000, len(a)))
    return {'groups': len(a), 'mean': float(a.mean()), 'positive_groups': int((a > 0).sum()),
            'negative_groups': int((a < 0).sum()),
            'exploratory_bootstrap95': np.quantile(a[idx].mean(1), [.025, .975]).tolist()}
def effects(v):
    bydir = {}
    for d in DIRS:
        y = {b: v['NATIVE' if b == '000' else d + '__' + b] for b in BITS}
        bydir[d] = {c: math.fsum(w*y[b] for b,w in weights.items()) for c,weights in CONTRASTS.items()}
    mean = {c: math.fsum(bydir[d][c] for d in DIRS)/4 for c in CONTRASTS}
    return mean, bydir
def close(a,b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        # Exactly-zero contrast signs can differ under an independent summation
        # order. Per-group sign differences are separately bounded below.
        return max([close(a[k],b[k]) for k in a if k not in ['positive_groups','negative_groups']] + [0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([close(x,y) for x,y in zip(a,b)] + [0.])
    err = abs(float(a)-float(b)); assert err < 2e-12
    return err


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--upstream-only',action='store_true'); args=ap.parse_args()
    req=[ROOT/'upstream_validation.json']+([] if args.upstream_only else [ROOT/'validation.json'])
    if not all(p.exists() for p in req): raise SystemExit(75)
    uv=read(req[0]); assert uv['status']=='CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS'
    up=read(check(uv['result'])); protocol=read(check(uv['protocol']))
    assert up['groups']==9 and up['pairs']==18 and up['arms']==29 and uv['candidate_arms']==522
    assert len({r['component'] for r in up['rows']})==9
    sources=[bind(Path(__file__)),bind(req[0]),uv['result'],uv['protocol']]
    invariant=[]
    for b in uv['pair_validations']:
        pv=read(check(b)); assert pv['status']=='CONTEXT_BINDING_PAIR_PASS' and len(pv['arms'])==29
        iv=read(check(pv['input_invariants'])); sources += [b,pv['input_invariants']]
        assert len(iv['records'])==32
        for r in iv['records']:
            assert r['inverse_roll_bit_exact'] and r['selected_Gram_relative_error']<1e-12
            if r['full_P_FFT_amplitude_relative_error'] is not None:
                assert r['full_P_FFT_amplitude_relative_error']<1e-12
            invariant.append(r)
    metrics={}; direction={}; rows=[]; error=0.; grid=[]; lns=[]; sign_roundoff=[]
    for r in up['rows']:
        t=r['target']; w=r['fixed_wrong']; arms=list(t)
        vals={'target_logM':{a:math.log(t[a]['M']) for a in arms},
              'fixed_wrong_logM':{a:math.log(w[a]['M']) for a in arms},
              'logM_gap':{a:math.log(t[a]['M'])-math.log(w[a]['M']) for a in arms},
              'dense_logM_gap':{a:math.log(t[a]['dense_M'])-math.log(w[a]['dense_M']) for a in arms}}
        for role,data in [('target',t),('fixed_wrong',w)]:
            for side in range(2):
                for name in ['cov_AJ','cov_AP','cov_JP','var_sum_actual','context_P_cosine']:
                    vals[f'{role}_side{side}_{name}']={a:data[a]['sides'][side]['preLN_means'][name] for a in arms}
                vals[f'{role}_side{side}_LN_to_native_cosine']={a:data[a]['sides'][side]['LN_to_native_cosine_mean'] for a in arms}
                for d in DIRS:
                    x=data[d+'__111']['sides'][side]['common_grid_diagnostic']
                    grid.append(dict(index=r['index'],role=role,side=side,direction=d,value=x))
                for b in ['001','110','111']:
                    lns.append(dict(index=r['index'],role=role,side=side,bits=b,
                        mean_native_cosine=math.fsum(data[d+'__'+b]['sides'][side]['LN_to_native_cosine_mean'] for d in DIRS)/4))
        er={}; dr={}
        for m,v in vals.items():
            er[m],dr[m]=effects(v); error=max(error,close(er[m],r['effects'][m]))
            for c,x in er[m].items():
                y=r['effects'][m][c]
                if np.sign(x)!=np.sign(y):
                    assert max(abs(x),abs(y))<2e-12
                    sign_roundoff.append(dict(index=r['index'],metric=m,contrast=c,independent=x,source=y))
        rows.append(dict(index=r['index'],query_id=r['query_id'],component=r['component'],effects=er,directions=dr))
    for m in rows[0]['effects']:
        metrics[m]={c:stats([r['effects'][m][c] for r in rows]) for c in CONTRASTS}
        direction[m]={d:{c:stats([r['directions'][m][d][c] for r in rows]) for c in CONTRASTS} for d in DIRS}
    error=max(error,close(metrics,up['summary']))
    data=dict(status='CONTEXT_BINDING_SECOND_INTERPRETATION_UPSTREAM_PASS',sources=sources,
        groups=9,upstream_effect_summary=metrics,direction_summaries=direction,rows=rows,
        inverse_roll_checks=len(invariant),maximum_invariant_Gram_relative_error=max(x['selected_Gram_relative_error'] for x in invariant),
        maximum_invariant_P_FFT_relative_error=max(x['full_P_FFT_amplitude_relative_error'] or 0 for x in invariant),
        common_grid_diagnostics=grid,LN_native_direction_diagnostics=lns,
        near_zero_sign_roundoff=sign_roundoff,
        independent_effect_and_bootstrap_max_error=error,scope=protocol['boundary'],new_forward_or_fit=False,
        upstream_tensor_verification='Prior sealed upstream validator verifies payload/LN SHA and independent M pooling; this second review verifies its result/validation/protocol and all 18 invariant JSON bindings, then independently recomputes scalar contrasts and group statistics.')
    if args.upstream_only:
        save(OUT/'upstream_review.json',data); print(json.dumps({'status':data['status'],'logM':metrics['logM_gap']},ensure_ascii=False));return
    bv=read(req[1]); assert bv['status']=='CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS'
    br=read(check(bv['result'])); assert br['groups']==9 and br['upstream']==uv['result']
    sources.extend([bind(req[1]),bv['result']])
    bridge={}; side_effects={}; bridge_rows=[]
    for b in bv['bridge_query_validations']:
        v=read(check(b)); check(v['result']); sources += [b,v['result']]
    assert [r['index'] for r in br['rows']]==[r['index'] for r in rows]
    for r in br['rows']:
        er={m:effects({a:v[m] for a,v in r['arms'].items()})[0] for m in r['effects']}
        error=max(error,close(er,r['effects']))
        er_side={role:{m:effects({a:v[role][m] for a,v in r['individual_endpoints'].items()})[0]
                      for m in br['individual_summary'][role]} for role in ['target','fixed_wrong']}
        bridge_rows.append(dict(index=r['index'],effects=er,side_effects=er_side))
    for m in br['gap_summary']:
        bridge[m]={c:stats([r['effects'][m][c] for r in bridge_rows]) for c in CONTRASTS}
    for role in ['target','fixed_wrong']:
        side_effects[role]={m:{c:stats([r['side_effects'][role][m][c] for r in bridge_rows]) for c in CONTRASTS}
                            for m in br['individual_summary'][role]}
    error=max(error,close(bridge,br['gap_summary']),close(side_effects,br['individual_summary']),
              close(bridge['logM_gap'],metrics['logM_gap']))
    data.update(status='CONTEXT_BINDING_SECOND_INTERPRETATION_COMPLETE_PASS',bridge_gap_summary=bridge,
                bridge_individual_summary=side_effects,bridge_rows=bridge_rows,independent_effect_and_bootstrap_max_error=error)
    save(OUT/'result.json',data)
    def fmt(v):
        lo,hi=v['exploratory_bootstrap95']; return f"{v['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]；{v['positive_groups']}/9正"
    text=['# A/J/P上下文配准：完整九组的独立解释','',
      '基于9个query，每个query固定target/wrong两个候选，共18个query-reference pairs，各29臂全部验收，并完成同组冻结外部／POST桥接；这里是9组target/wrong比较，不是18组。未新增前向、未拟合、未改封存结果。四方向先在各组内平均，统计单位为九个已打开的机制开发组。区间为探索性组bootstrap，不是独立确认。','',
      '## 核心结果：支持水平、比例区分与内容利用必须分开','',
      f"主配准对比令target logM {metrics['target_logM']['aligned_vs_misaligned']['mean']:+.6f}、wrong logM {metrics['fixed_wrong_logM']['aligned_vs_misaligned']['mean']:+.6f}，两者9/9组均正；但logM gap为{fmt(bridge['logM_gap']['aligned_vs_misaligned'])}，未证实比例区分稳定增强。",
      f"与此同时，原始M gap为{fmt(bridge['M_gap']['aligned_vs_misaligned'])}。这不能被省略，也不能笼统概括为没有任何身份倾向：相近的百分比变化作用于不同初始M值，可产生不同的绝对增量。若为共同乘法因子k，log(kMt)-log(kMw)不变，但原始差缩放为k(Mt-Mw)；这里没有证明变化恰好是共同k。",
      f"同一干预的POST内容gap为{fmt(bridge['POST_content_gap']['aligned_vs_misaligned'])}，POST action gap为{fmt(bridge['POST_action_gap']['aligned_vs_misaligned'])}。因此有内部内容差的正向信号，但不能升级为最终动作已稳定改善，更不能替代全C128纠错计数。",
      'POST读取每个候选的绝对logM及已有内容，经同一适配器得到内容表示；它不直接复制logM gap。相近的M比例变化可在不同内容及初始标度处产生不同的响应，不能由此唯一归因到某个patch或某一非线性部件。','',
      '## 对比究竟分开了什么','',
      '- P固定、仅A/J同移（110−000）：P完整张量及其网格布局不动；它直接检验保持P自身完全相同，改变输入上下文是否仍改变结果。但A/J相对解码网格的位置也改变。',
      '- 对齐/错位配对差（000+111−001−110）/2：平衡两种绝对摆放，消去标量端点上纯加性的上下文摆放项与P摆放项。非零值表示当前接口的跨分量交互；不能直接命名为唯一语义绑定。',
      '- 111−000共同移位：保持逐patch元组与SUM/LN共同移位；测量完整DPT及池化的网格/边界敏感性。它不是应当为零的负对照，也不能在存在交互时用相减彻底排除边界。',
      '- 条件J×P与三阶交互：检验J/P配准是否依赖A的摆放。不同交互不能相加为唯一贡献率。','',
      '| 对比 | target logM | wrong logM | target−wrong logM | 原始M gap | dense均值构成的gap |',
      '|---|---|---|---|---|---|']
    for c in CONTRASTS:
        items=[metrics[m][c] for m in ['target_logM','fixed_wrong_logM','logM_gap']]+[bridge['M_gap'][c],metrics['dense_logM_gap'][c]]
        text.append('| '+c+' | '+' | '.join(fmt(x) for x in items)+' |')
    text += ['', '## 同一次干预的外部与内部传播','',
      '| 对比 | NATIVE7 action gap | M_FREE action gap | POST内容gap | POST action gap |',
      '|---|---|---|---|---|']
    for c in CONTRASTS:
        text.append('| '+c+' | '+' | '.join(fmt(bridge[m][c]) for m in ['NATIVE7_action_gap','M_FREE_action_gap','POST_content_gap','POST_action_gap'])+' |')
    text += ['', '上述传播只替换M及其代数派生量；原外部局部形状/内容固定，POST保持同一冻结适配器及tokens。只改变固定target/wrong；第三RAW赢家保留原M/L，不是全C128排名实验。不同下游端点允许方向不一致，不能只挑正向支持统一解释。','',
      '## 输入不变量与归一化边界','',
      f"全部{len(invariant)}个逆移位检查通过；32通道Gram最大相对误差{data['maximum_invariant_Gram_relative_error']:.3g}，P全通道Fourier幅度最大相对误差{data['maximum_invariant_P_FFT_relative_error']:.3g}。",
      '循环移位保留各场的向量集合、Gram、Fourier幅度及环绕相关结构，但不保留相对A/J或有限解码网格的绝对摆放。因此若输出改变，只能否定这些被保留的场内统计已充分决定输出，不能否定所有P结构的价值。',
      'DPT先接收A0与A1+J+P：改变配准会改变Cov(A,J)、Cov(A,P)、Cov(J,P)、SUM方差及归一化方向；每patch中间量已保存。共同移位的SUM与实际LayerNorm等变在上游验收中逐元素验证，但完整DPT可不等变。交叉项变化是已定位的接口后果，尚不是各项对M的唯一中介因果。',
      '保存的共同移位confidence RMS是原始DPT confidence logit误差，不是0–1支持权重误差。dense均值gap同样跨0，只限制exact token池化独有解释，不能排除完整confidence场中仍有被全局均值压缩丢失的局部身份信息。',
      '此接口只观察总和，任何保持A0和实际SUM完全相同的J/P补偿改变在下游不可区分。这是识别唯一J/P来源的结构边界。','',
      '## 独立复核','',
      f"独立重算全部七种对比、四方向、九组统计与bootstrap，并逐项核对上游和桥接logM闭合；最大误差{error:.3g}。",
      '完整逐组、逐方向、绝对target/wrong端点、LN方向、网格误差及源SHA保存在独立JSON。没有依据部分组或某个方向选择结论。',
      f"结果：`{(OUT/'result.json').relative_to(RC)}`。",'']
    s='\n'.join(text)
    if REPORT.exists():assert REPORT.read_text()==s
    else:REPORT.write_text(s)
    save(OUT/'validation.json',dict(status=data['status'],result=bind(OUT/'result.json'),report=bind(REPORT),groups=9,
        source_upstream_validation=bind(req[0]),source_bridge_validation=bind(req[1]),max_error=error,new_forward_or_fit=False))
    print(json.dumps({'status':data['status'],'max_error':error,'report':str(REPORT)},ensure_ascii=False))


if __name__=='__main__':main()
