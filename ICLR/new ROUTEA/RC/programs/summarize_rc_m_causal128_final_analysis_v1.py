#!/usr/bin/env python3
"""Join the completed attribution chain and two bounded cache-only add-ons.

No fitting, model inference, data selection, or edits to sealed experiments.
"""
import csv
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
BASE = RC / 'results/rc_m_causal128_attribution_chain_v1'
OUT = BASE / 'final_analysis_v1'
REPORT = RC / 'reports/REPORT_M_CAUSAL128_FINAL_ANALYSIS_20260927.md'
REQUIRED = {
    'paths/cache_validation.json': 'M_CAUSAL128_EXTRACT_ALL128_PASS',
    'paths/validation.json': 'M_CAUSAL128_COMPENSATION_ALL5_INDEPENDENT_PASS',
    'pair_structure/validation.json': 'M_CAUSAL128_PAIR_STRUCTURE_PASS',
    'bridge/validation.json': 'CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS',
    'bridge/joint_pairs/validation.json': 'JOINT_RESTORATION_FIXED14_BRIDGE_PASS',
    'phase_replication/validation.json': 'PHASE_NEWGROUP_REPLICATION_JOIN_PASS',
    'final_summary/validation.json': 'M_CAUSAL128_ATTRIBUTION_SUMMARY_PASS',
    'property_compensation_fixed_pairs_v1/validation.json': 'PROPERTY_FIXED_PAIR_COMPENSATION_INDEPENDENT_PASS',
    'newgroup_property_bridge_v1/validation.json': 'NEWGROUP_PROPERTY_BRIDGE_INDEPENDENT_PASS',
}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def collect(obj, found):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
            assert len(obj['sha256']) == 64
            prior = found.setdefault(obj['path'], obj['sha256'])
            assert prior == obj['sha256'], ('INCONSISTENT_BINDING', obj['path'])
        for value in obj.values():
            collect(value, found)
    elif isinstance(obj, list):
        for value in obj:
            collect(value, found)


def stat(s):
    lo, hi = s['exploratory_bootstrap95']
    return f"{s['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]"


def main():
    OUT.mkdir(exist_ok=True)
    seals, receipts = {}, {}
    for name, expected in REQUIRED.items():
        value = read(BASE / name)
        assert value['status'] == expected, (name, value['status'])
        collect(value, seals)
        receipts[name] = dict(status=value['status'], **bind(BASE / name))
    independent_reviews = {}
    for name, expected in {
        'independent_property_review_v1/validation.json': 'PROPERTY_NEWGROUP_BRIDGE_AND_COMPENSATION_SECOND_REVIEW_PASS',
        'independent_property_review_v1/post_absolute_input_validation.json': 'POST_ABSOLUTE_M_ACTUAL_INPUT_AND_PATCH_REPLAY_PASS',
    }.items():
        value = read(BASE / name)
        assert value['status'] == expected
        collect(value, seals)
        independent_reviews[name] = dict(status=value['status'], **bind(BASE / name))
    for path, sha in seals.items():
        assert bind(path)['sha256'] == sha, ('BOUND_ARTIFACT_CHANGED', path)
    phase = read(BASE / 'phase_replication/result.json')
    bridge = read(BASE / 'newgroup_property_bridge_v1/result.json')
    comp = read(BASE / 'property_compensation_fixed_pairs_v1/result.json')
    assert phase['target_present_groups'] == bridge['groups'] == comp['test_groups'] == 9
    assert bridge['arms'] * bridge['pairs'] == 324
    assert bridge['full_C128_predictions_computed'] is False and comp['full_C128_action_test'] is False
    effects = []
    for endpoint, contrasts in bridge['gap_summary'].items():
        for contrast, value in contrasts.items():
            lo, hi = value['exploratory_bootstrap95']
            effects.append(dict(endpoint=endpoint, contrast=contrast, mean=value['mean'],
                exploratory_low=lo, exploratory_high=hi, positive_groups=value['positive_groups'], groups=value['groups']))
    with (OUT / 'endpoint_effects.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(effects[0])); writer.writeheader(); writer.writerows(effects)
    costs = []
    for cell, value in comp['evaluations'].items():
        costs.append(dict(cell=cell, frozen_loss=value['FROZEN_GLOBAL']['mean_group_loss'],
            refit_loss=value['REFIT']['mean_group_loss'], raw_l_loss=value['RAW_L_ONLY']['mean_group_loss'],
            positive_groups=value['REFIT']['positive_mean_margin_groups'], groups=9))
    with (OUT / 'property_compensation.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(costs[0]));writer.writeheader();writer.writerows(costs)
    primary = 'phase_restoration_after_amplitude_permutation'
    native = 'phase_restoration_native_amplitude'
    amp = 'amplitude_restoration_under_local_phase'
    summ = bridge['gap_summary']
    lines = [
        '# M归因最终分析：哪些性质起作用，如何传播，何时可被补偿', '',
        '**本轮冻结的128主链、9组性质复验、同模型传播和有界补偿诊断均已完成，9类验收全部通过。** 日期：2026-09-27。', '',
        '本轮目的不是刷新准确率，而是解释M中的候选区分从何而来，以及它如何被外部头和内部表示调制利用。计算结果包含正向、反向和未决证据，完整保留。', '',
        '## 1. 先回答：是否在query与reference之间建立上下文关系', '',
        '是。RoMa的双图特征交互产生上下文J；每个query patch对reference位置的匹配分布，经正弦/余弦位置特征取期望得到P。P既包含每个patch的分布特征，也包含多个patch之间的相对关系。冻结预测器读取这些输入，产生两侧支持权重，汇总为M。ColNomic继续负责自由内容匹配。', '',
        '关键不在于给已有pairwise计算改名字。本轮额外隔离了：在每个patch的谱功率和范数不变时，跨patch相对编码的改变是否会改变候选区分；仅让数值幅度恢复是否也能达到同样作用。', '',
        '## 2. 面板、模型和干预边界', '',
        '| 分析 | 数据与固定条件 | 端点 |', '|---|---|---|',
        '| J/P入口及同族补偿 | H593执行面板128图、原组件五折、ColNomic自然C128；120图target在候选内 | 45组候选分差；46组决策；4入口世界×5折重训 |',
        '| 性质开发复验 | 旧7组、14固定候选、18干预臂 | 幅度与相位结构的条件效应 |',
        '| 性质分组复验 | F71机制开发之外的新9组、18固定候选、324候选干预输出 | 正确候选减预先固定错误候选的logM差 |',
        '| 新9组下游传播 | 同一原折冻结NATIVE7、M_FREE、POST_REAL；18臂全部传播 | 外部action差、内部内容差与action差 |',
        '| 性质补偿 | 旧7组TRAIN、新9组TEST；四个四维反对称读出及RAW/L对照 | 分组平均排序损失；无HOLD、无完整C128准确率 |', '',
        '这128图不是历史99/128的EVAL面板。新9组已参与历史H593评测，属于机制开发组之外的复验，不是未接触外部确认。四个轴/符号重复不增加独立组数，区间均为探索性组bootstrap，未作普遍显著性或唯一因果声明。', '',
        'GLOBAL给所有P patch施加同一相位旋转，保留所有patch之间内积；LOCAL给不同patch施加不同符号的旋转，保留每patch谱功率但改变跨patch关系。AMP_PERMUTE保留范数分布和向量方向，改变幅度与patch的绑定。“恢复相位”指LOCAL→GLOBAL，不是恢复原生NATIVE P。A/J固定时，这些干预也可能改变P与内容上下文的相容性，不能唯一解释成真实物体几何。', '',
        '## 3. 最强的正向闭环：幅度仍打乱，恢复相对结构', '',
        '| 新9组端点 | 组均值与探索性95%区间 | 正向组 |', '|---|---:|---:|',
    ]
    for key, label in [('logM_gap','正确−错误logM'),('NATIVE7_action_gap','原外部NATIVE7决策差'),
                       ('M_FREE_action_gap','自由内容×M外部决策差'),('POST_content_gap','原内部POST内容差'),('POST_action_gap','原内部POST决策差')]:
        v=summ[key][primary]; lines.append(f"| {label} | {stat(v)} | {v['positive_groups']}/9 |")
    lines += ['',
        '这条闭环是在同一次干预、同一组候选、同一原折冻结模型中得到的：相对编码改变→M候选区分改变→外部及内部端点改变。它支持相对结构的作用；不要求重新生成区域，也不要求下游逐patch硬对应。新9组传播独立NumPy核算的最大误差为3.56e−15。', '',
        '主128面板还显示上下文依赖：有J时加入P，组平均logM分差+0.613814；无J直接入口时加入P为−0.076750。P本身也由双图交互构造，因此删除J入口不等于删除所有双图信息，不能把J和P说成相互独立的因果源。', '',
        '## 4. 必须保留的反例：不是M增大就有效，也不是所有效应同向传播', '',
        '新9组原幅度下恢复相对结构，正确logM平均下降0.019992、错误下降0.037592，区分因此改善。恢复原幅度绑定则使正确与错误logM分别增加0.261316、0.243839，双方都9/9组上升，但身份分差只有4/9组正向，区间含0。可明显改变M数值的性质，不等于稳定帮助身份的性质。', '',
        '| 新9组条件 | logM差效应 | 外部NATIVE7差效应 | 内部POST内容差效应 |', '|---|---:|---:|---:|',
    ]
    for key,label in [(native,'原幅度下GLOBAL−LOCAL'),(amp,'LOCAL下恢复幅度绑定')]:
        lines.append('| '+label+' | '+' | '.join(stat(summ[m][key]) for m in ['logM_gap','NATIVE7_action_gap','POST_content_gap'])+' |')
    lines += ['',
        '**原幅度条件中，logM候选分差改善，但POST内容分差反向。** 所以不能说内部和外部对所有上游改变都有等价或单调的响应。内部模块读取每个候选的标准化绝对logM及patch内容，再经残差、投影、归一化形成表示；同一个M分差可以对应不同的绝对水平和内容，产生不同的变化。幅度恢复在内部有正向平均作用，也限制了“幅度无用”的说法。此反例支持对背景与内容的条件依赖，但尚未把绝对尺度、内容和各非线性项分别隔离，不能唯一归咎其中一项。', '',
        '针对反向结果，另行hook核对了324/324个候选干预的真实适配器输入：条件列精确等于该候选自己的标准化logM，query内容列跨臂不变；重新读取缓存hidden并经冻结POST模块回放，所有patch MaxSim和argmax完全一致。这排除了候选串位、漏传M或汇总符号错误，不把真实反例归为工程故障。', '',
        '这轮得到的是共同输入通路及其条件作用：外部通过已有内容证据的组合与候选比较，内部通过内容表示条件化使用M。不是两种实现数学等价，不是同一空间mask的恢复，也不是所有干预的唯一中介分解。', '',
        '## 5. 删除后能否补偿：已充分求解的有限模型族', '',
        '全部尺度和5个参数向量在读取新9组M之前封存。旧7组上使用固定ridge=0.001的严格凸排序目标，解析Newton与独立标量梯度核算均通过；最大梯度2.27e−12，训练目标最优差上界4.49e−21。这里已排除“这个目标还没优化完”，但不能排除小样本泛化和模型族限制。', '',
        '| 性质单元 | 冻结GLOBAL头损失 | 本单元重训损失 | RAW/L对照损失 | 重训正分差组 |', '|---|---:|---:|---:|---:|',
    ]
    for row in costs:
        lines.append(f"| {row['cell']} | {row['frozen_loss']:.6f} | {row['refit_loss']:.6f} | {row['raw_l_loss']:.6f} | {row['positive_groups']}/9 |")
    lines += ['',
        '原幅度下LOCAL单元重训后仍比GLOBAL多0.020685排序损失，探索性区间[0.002876, 0.045092]；当前读出没有补回相对结构扰动。但幅度置换单元仍可有效排序，重训损失甚至低于原幅度GLOBAL均值；相应剩余差区间含0，不能据此宣称新最佳或普遍等价。所有单元重训正均值分差均7/9组，不能拿连续损失改变量冒充检索正确数提升。', '',
        '因此，当前可以报告：某些相对结构作用在这一读出族和分组检验中未被补偿，而原幅度绑定不是维持此固定配对排序能力的唯一途径。不能推出物理信息不可替代，也没有重训RoMa或验证完整C128性质必要性。', '',
        '## 6. 四步完成情况与本轮结论', '',
        '| 步骤 | 已完成 | 仍不能据此声称 |', '|---|---|---|',
        '| 控制其他因素、干预特定性质 | 谱功率保留的相位对照、幅度绑定置换、J/P四格 | 真实物理几何的唯一干预 |',
        '| 交叉恢复并追踪传播 | 旧7组及新9组同模型传播；target/wrong单独效应 | 所有上游效应完全由M唯一中介 |',
        '| 删除后允许补偿 | 20项入口补偿；5个充分拟合的性质读出诊断 | 所有可能模型均无法替代 |',
        '| 未参与机制选择的组复验 | 新9组单列，不与旧组混合；保留反例 | 未接触外部数据上的普遍证明 |', '',
        '**最终可成立的论点：配对上下文使对应分布编码中的相对结构具有候选区分作用；这部分作用能经M压缩，在明确的幅度条件下同时传入外部校准和内部表示调制。幅度与相对结构的作用可分离，具体下游响应还依赖M绝对水平和内容；当前证据支持一组条件机制，不支持唯一因果解。**', '',
        '这比“M有用”“pairwise有用”更具体，也保留了哪些解释不成立。论文可提出可检验的接口与性质结论；区域精度、身份充分性、所有性质不可替代均不在本轮证明范围。', '',
        '## 7. 文件与复现', '',
        '- [详细实现解释与证据边界](REPORT_M_CAUSAL128_INTERPRETATION_REVIEW_20260927.md)',
        '- [主128封存汇总](../results/rc_m_causal128_attribution_chain_v1/final_summary/report.md)',
        '- [新9组上游性质复验](../results/rc_m_causal128_attribution_chain_v1/phase_replication/report.md)',
        '- [新9组同模型传播](../results/rc_m_causal128_attribution_chain_v1/newgroup_property_bridge_v1/report.md)',
        '- [有界性质补偿](../results/rc_m_causal128_attribution_chain_v1/property_compensation_fixed_pairs_v1/report.md)',
        '- [独立补偿及上下游闭合复核](../results/rc_m_causal128_attribution_chain_v1/independent_property_review_v1/validation.json)',
        '- [内部绝对M入口与324项patch回放复核](../results/rc_m_causal128_attribution_chain_v1/independent_property_review_v1/post_absolute_input_validation.json)',
        '- [机器可读总结果](../results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/result.json)',
        '- [全部端点效应表](../results/rc_m_causal128_attribution_chain_v1/final_analysis_v1/endpoint_effects.csv)', '',
        '本轮最后两个补充项仅用现有缓存，未新增RoMa/视觉编码器/LLM前向。accelerated上申请的GPU仅用于调度，CPU执行且CUDA_VISIBLE_DEVICES为空。主相位复验需要的缺失前向此前已在CPU完成并保存，不能把整个历史链误写成从未运行匹配器。', '',
        '运行本文件对应汇总器可重新核对9类receipt与绑定结果，重建表格和此报告。实验源码、协议、逐候选标量结果、头参数、分析表和报告进入GitHub；原图、模型权重、大型张量缓存、err/out日志不进入。原封存final_summary保留当时的范围描述，新增两个诊断在本报告单独补齐，未回写旧实验。', '',
    ]
    REPORT.write_text('\n'.join(lines))
    result = dict(status='M_CAUSAL128_FINAL_ANALYSIS_COMPLETE',required_receipts=receipts,
        independent_reviews=independent_reviews,
        completed_receipts=len(receipts),verified_direct_artifact_bindings=len(seals),
        experiment_scope='Frozen 128-query main attribution chain plus locked nine-group property replication, frozen propagation, and bounded seven-to-nine-group property readout compensation.',
        datasets=dict(main_queries=128,natural_candidates=128,original_folds=5,old_property_groups=7,new_property_groups=9),
        endpoint_effects=effects,property_compensation=costs,
        relative_structure_effect_supported_conditionally=True,
        all_interventions_same_sign_in_external_internal=False,
        unique_cause_claim=False,identity_sufficiency_claim=False,physical_geometry_recovery_claim=False,
        property_full_C128_necessity_claim=False,untouched_external_confirmation_claim=False,
        analysis_code=bind(__file__),report=bind(REPORT))
    write('result.json',result)
    write('receipt_inventory.json',receipts)
    write('validation.json',dict(status='M_CAUSAL128_FINAL_ANALYSIS_ALL9_PASS',result=bind(OUT/'result.json'),
        report=bind(REPORT),code=bind(__file__),required_receipts=receipts,
        independent_reviews=independent_reviews,
        verified_direct_artifact_bindings=len(seals),
        tables=[bind(OUT/'endpoint_effects.csv'),bind(OUT/'property_compensation.csv')],
        no_new_model_computation=True,no_new_fit_or_data_selection=True))
    print(json.dumps(dict(status='M_CAUSAL128_FINAL_ANALYSIS_ALL9_PASS',receipts=len(receipts),bindings=len(seals))))


if __name__ == '__main__':
    main()
