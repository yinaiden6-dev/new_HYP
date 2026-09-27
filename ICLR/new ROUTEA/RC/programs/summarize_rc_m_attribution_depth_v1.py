#!/usr/bin/env python3
"""Join completed attribution receipts into a descriptive, non-optimizing report."""
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
UP = RC / 'results/rc_m_context_binding_v1'
DOWN = RC / 'results/rc_post_m_level_gap_v1'
COARSE = RC / 'results/rc_m_coarse_path_correction_audit_v1'
ROOT = RC / 'results/rc_m_attribution_depth_v1'


def read(path):
    return json.loads(path.read_text())


def bind(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def cell(value):
    lo, hi = value['exploratory_bootstrap95']
    return f"{value['mean']:+.7f} [{lo:+.7f}, {hi:+.7f}]"


def main():
    checks = {
        UP / 'upstream_validation.json': 'CONTEXT_BINDING_UPSTREAM_INDEPENDENT_PASS',
        UP / 'validation.json': 'CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS',
        DOWN / 'validation.json': 'POST_M_LEVEL_GAP_INDEPENDENT_PASS',
        DOWN / 'second_arithmetic_validation.json': 'POST_M_LEVEL_GAP_SECOND_ARITHMETIC_PASS',
        COARSE / 'validation.json': 'COARSE_PATH_FULL128_CORRECTION_ACCOUNTING_INDEPENDENT_PASS',
        ROOT / 'validation.json': 'M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS',
    }
    for path, status in checks.items():
        receipt = read(path)
        assert receipt['status'] == status, (path, receipt['status'])
        if 'result' in receipt:
            assert bind(Path(receipt['result']['path'])) == receipt['result']
    upstream = read(UP / 'upstream_result.json')
    bridge = read(UP / 'result.json')
    downstream = read(DOWN / 'result.json')
    coarse = read(COARSE / 'result.json')
    assert upstream['groups'] == bridge['groups'] == 9
    assert len(downstream['rows']) == 120 and coarse['queries'] == 128
    sources = [bind(p) for p in [*checks, UP / 'upstream_result.json', UP / 'result.json',
                                DOWN / 'result.json', COARSE / 'result.json', Path(__file__)]]
    lines = [
        '# M归因深化：配准、共同水平与实际纠错的收口', '',
        '两条新增实验链和独立核算均完成。这里的“完成”指已冻结的有限实验，不表示所有原系统收益已有唯一根因。没有训练新模型，没有新增准确率纪录。', '',
        '## 已经推进了什么', '',
        '上游证明了输出支持依赖内容与对应编码的配合：两侧支持同时提高，logM比例分差未显示稳定改善，但原始M绝对分差与POST内容分差出现正向信号。这三种端点不能混用。下游拆出了旧反向现象的共同水平与相对差距来源。已有完整C128结果也已逐query核算，粗通路的实际纠错作用不是空白。', '',
        '## 上游：提高支持强度，不自动提高身份区分', '',
        '固定9个已打开身份组，每个query比较target与一个固定wrong，共18个query–reference候选对；四个方向在组内平均，29臂共522次CPU双向DPT缓存回放。所有编码器、匹配器和参数保持冻结。P完整不变的对照与保留P向量集/通道Gram/Fourier幅度的对照同时保留。', '',
        '| 配准对照 | target logM | wrong logM | target−wrong logM |',
        '|---|---:|---:|---:|',
    ]
    labels = {'aligned_vs_misaligned': '对齐相对错位（主对比）',
              'P_fixed_context_shift': 'P完整固定，仅移动A/J',
              'common_grid_shift': 'A/J/P共同移动',
              'J_P_interaction_A_fixed': 'A固定时J×P交互',
              'J_P_interaction_A_shifted': 'A移动时J×P交互'}
    for key, label in labels.items():
        lines.append('| ' + label + ' | ' + ' | '.join(cell(upstream['summary'][m][key])
                     for m in ('target_logM', 'fixed_wrong_logM', 'logM_gap')) + ' |')
    lines += ['',
        '主对比两侧logM变化均9/9为正；logM分差区间跨0，不等于证明无效或等价，更不等于所有尺度上的身份分差均无改善。完整DPT不是平移等变网络，共同移动对照必须保留。整场sigmoid均值构成的log分差也跨0，这只限制“本现象由exact token池化独自造成”的解释，不能排除完整支持图含有被均值压缩掉的身份信息。',
        '本轮在query网格移动编码，旧GLOBAL/LOCAL则旋转每块对应坐标的编码相位，两者不是同一干预；不能由本轮近零分差抹掉旧相位干预的正向证据。', '',
        '## 同一次上游干预怎样进入外部与内部模型', '',
        '固定原折NATIVE7、M_FREE及POST_REAL，仅沿M及代数派生量传播，原局部权重形状/内容保持固定。完整置信度场已保存，但此处不宣称全部上游变化的总效应。以下仍是两个固定候选的分差，不是全C128准确率。', '',
        '| 对比 | 外部NATIVE7动作差 | 外部M_FREE动作差 | POST内容差 | POST动作差 |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, label in labels.items():
        lines.append('| ' + label + ' | ' + ' | '.join(cell(bridge['gap_summary'][m][key])
                     for m in ('NATIVE7_action_gap', 'M_FREE_action_gap', 'POST_content_gap', 'POST_action_gap')) + ' |')
    primary = 'aligned_vs_misaligned'
    lines += ['',
        '主对比的原始M分差变化为 ' + cell(bridge['gap_summary']['M_gap'][primary]) +
        '，POST内容分差为 ' + cell(bridge['gap_summary']['POST_content_gap'][primary]) +
        '；两者探索区间均在0以上。但POST动作分差区间仍跨0，不能把内容端点的变化宣布为稳定新增纠错。',
        'logMt−logMw衡量比例区分，Mt−Mw衡量绝对差；相近比例变化作用在不同原始M水平上，也可能改变绝对差。这说明不能用logM分差单独代表所有下游可利用的变化。这里没有把原始M差认定为POST变化的唯一中介。条件J×P对比中的POST正向结果完整保留，但属于预定次级对比，不能替代主对比或独立确认。',
        '这组结果与下面的共同水平/相对差距检验一起，支持“不同读出利用M的不同方面”：外部动作主要依赖相对证据，内部内容调制还会响应绝对工作水平。它不是把所有上游性质唯一分解的证明。', '',
        '## 下游：共同水平可以帮助，也可以抵消相对优势', '',
        '主坐标μ=(logMt+logMw)/2、δ=logMt−logMw。表中为同一冻结POST的内容分差变化；两项是四格有限对照的对称分配，不是唯一因果份额。方括号为探索性按组bootstrap区间。', '',
        '| 条件 | 共同水平分量 | 相对差距分量 | 总变化 |', '|---|---:|---:|---:|']
    panels = [('phase9/log_M/native_amplitude', '原幅度相位对照，9组'),
              ('phase9/log_M/permuted_amplitude', '置换幅度相位对照，9组'),
              ('jp128/log_M/P_with_J', '保留J恢复P，120张/45组'),
              ('jp128/log_M/J_without_P', '没有P恢复J，120张/45组'),
              ('jp128/log_M/P_without_J', '没有J恢复P，120张/45组')]
    for scope, label in panels:
        s = downstream['summary'][scope]['POST_content_gap']
        assert abs(s['symmetric_mu']['mean'] + s['symmetric_delta']['mean'] - s['total']['mean']) < 1e-12
        lines.append('| ' + label + ' | ' + ' | '.join(cell(s[k]) for k in ('symmetric_mu', 'symmetric_delta', 'total')) + ' |')
    lines += ['',
        '旧反向现象的负共同水平分量超过正相对差距点估计；后者区间含0，不能说每个query皆如此。另一方面，P with J中的共同水平分量明确为正，因此不能统一删除绝对M。J without P近零是两路抵消，不等于内部完全未利用M。',
        '两套坐标全部报告。J with P的log-M交叉有21张M>1，未裁剪/未计算非法输入；99张合法子集与全部120端点分开。logit对照覆盖全量，但M不是身份概率；贡献/交互会随坐标改变。部分干预输入超原TRAIN范围，因此不能把数值合法写成分布内验证。',
        '本面板93张RAW=target、27张RAW=fixed_wrong，没有第三RAW锚。外部对共同缩放近似不变是这种配对下归一化公式的既有性质，本轮只是复核；内部还依赖候选绝对M工作水平和内容。二者共享M接口，不是等价读出。', '',
        '![POST共同水平与相对差距的有限分解](figures/m_attribution_depth_20260927/post_M_level_gap_three_panels.png)', '',
        '## 粗通路已经承担了哪些原有纠错', '',
        'H593执行清单前128张，原held折、ColNomic自然C128；RAW93/128，target在候选中120/128。不是旧EVAL128，也不是全593。独立重算三模型×五世界×128共1920组完整候选分数和0阈值HOLD/SWITCH。', '',
        '| 模型 | M来源 | 正确/128 | 救回RAW | 误伤RAW | 原HR1救回保留 |',
        '|---|---|---:|---:|---:|---:|']
    worlds = {'ORIGINAL_HR1': '原HR1', 'A1J1P1': '完整COARSE', 'A1J1P0': '去P', 'A1J0P1': '去J'}
    for model in ('NATIVE7', 'M_FREE', 'POST'):
        for world, label in worlds.items():
            s = coarse['summary']['FULL128'][model][world]
            lines.append(f"| {model} | {label} | {s['correct']} | {s['rescue_vs_RAW']['count']} | {s['break_vs_RAW']['count']} | {s['original_HR1_rescues_retained']['count']}/{s['original_HR1_rescue_count']} |")
    lines += ['',
        '恢复集合有重叠，不能相加成独立贡献比例。POST的HR1→COARSE漂移单列；同样的救回数也可能是不同query集合。这里沿M传播粗通路删除，原局部形状固定，不等于删除整条RoMa作用。“去J/没有J”只删除预测器的直接J输入；P仍由原匹配上下文生成，不等于去掉全部双图交互信息。', '',
        '## 判断与剩余问题', '',
        '上游与下游都可能尚未充分兑现可用信息，但本轮没有训练修复模型，不能保证优化后准确率提高。上游还需分开输入信息不足、预测器读出不足及压缩损失；当前两种全局均值不能代表完整空间场。下游已明确绝对水平与相对证据的条件作用，修复必须使用推理可见信息，不能借用诊断中的target/wrong真值。',
        '更细的相位、幅度、配准性质各自承担多少原生全C128救回仍未闭合。本轮量化的是固定接口的条件作用和粗通路的实际纠错集合，不是全593收益的唯一根因。没有新增QRR、所有权、分割或外部迁移主张。', '',
        '详细数据与独立解读：`results/rc_m_context_binding_v1/`、`results/rc_post_m_level_gap_v1/`、`results/rc_m_coarse_path_correction_audit_v1/`。全部可用中间张量留在workspace，Git导出保留源码、协议、标量结果、来源哈希与科学图。', '']
    report = RC / 'reports/REPORT_M_ATTRIBUTION_DEPTH_FINAL_20260927.md'
    report.write_text('\n'.join(lines))
    out = ROOT / 'final_analysis_v1'
    out.mkdir(exist_ok=True)
    result = dict(status='M_ATTRIBUTION_DEPTH_FINAL_REPORT_VERIFIED', sources=sources,
                  report=bind(report), complete_frozen_experiments=True,
                  unique_cause_claim=False, new_accuracy_result=False)
    (out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], report=str(report))))


if __name__ == '__main__':
    main()
