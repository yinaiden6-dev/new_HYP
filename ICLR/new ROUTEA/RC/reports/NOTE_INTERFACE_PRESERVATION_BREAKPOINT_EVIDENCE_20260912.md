# 接口信息保留：可作机制贡献的证据与缺口

2026-09-12重新核对历史原始结果、完成记录和源码。结论：这是一个已有证据的接口机制贡献切入点，可用于解释为何某些P分解并未保留旧系统能力；尚无证据将它称为超过当前最强头的新识别突破或所有历史失败的根因。

## 已有的三层证据

| 证据 | 数据/操作 | 已核实结果 |
| --- | --- | --- |
| 可恢复性反例 | 相同query/reference tokens、相同采样wr，改变未采样reference visibility | 完整M和旧score分别.5/1；8项已有合成测试本轮重执行通过 |
| 原计算依赖M | 旧FROZEN_C，RAW C128，opened difficult90；固定L/Lq/Lr，令M=1并重建相关输入 | 69→61，原8次救回全失，90条均HOLD；34560比值逐bit保持 |
| 无损接口恢复 | 同旧FROZEN_C，TRAIN32+EVAL32，完整RAW C128 | 8192pairs、32768scalars、8128logits全bit一致；64预测全部保持，0训练/0encoder或RoMa forward |

回放中EVAL是RAW25→27/32，TRAIN是27→28/32。TRAIN28不是EVAL28。69/90并未在该8192pair接口回放里重新执行；它来自同FROZEN_C的独立opened回归干预。8192是query-reference对数，不是独立图片数。

质量列置零也已做过：旧FROZEN_C EVAL27→25、difficult69→61；当前ec7新EVAL128的M列置零99→88，原12次救回消失。这些固定head列消融支持计算依赖，不等同完整移除所有质量信息，更不是旧压缩P接口的直接性能测试。

## 必须保留的两个限定

1. 合成反例证明采样数值接口一般不能恢复原统计量；它不是自然数据完整atom对象相等的碰撞频率统计，也没有证明每次压缩必然改变分类。原模型的错误决策也会被无损接口保留。
2. 真实修复保留完整双侧soft visibility、均值/M及canonical轴，由V恢复完整reference weighted MaxSim，再接回旧头。不是在失败P/GISC上“只加一个M”。旧P-only科学NO-GO没有被改写。

## 尚未量化的比较

尚未找到在同一失败P/GISC/读出、同训练、同query/C128/内容匹配/支持/聚合下，仅恢复缺失全局质量统计后，逐query带来多少救回/损失的完整对照。也没有将原接口、旧压缩接口和局部修复在当前ec7的28/99或593五折上组成同一匹配表。当前六特征头本来已含M，因此不能用旧接口缺M解释今天28/99的全部平台期。

若进一步验证，应事先严格定义受限数值接口及缺失字段的可执行读出；不能以随意置零充当“真实旧P”。质量统计近似与硬内容对应应分成独立因素，不把恢复完整计算图的收益全部归给某个标量。最低统计包括原score误差、action不一致率、原rescue保留率、修复的救回/损失，分TRAIN/EVAL并保持head和候选来源一致。

## 论文中可写与不可写

可写：reference假设的模块接口必须保留后续身份判定实际使用的软质量与内容信息；仅有几何支持或硬对应，并不足以保证继承原系统能力。本文定位一个具体违反此条件的接口，并无损修复了原计算。

不可写：此结论首次提出了充分信息原理；已经证明新HYP普遍成立；只补M即可修好所有P失败；无损恢复27等于超过原28/99；接口回放已覆盖593；恢复原模型即保证真实身份正确。

固定模型的输入充分性已有研究，例如[SIS, AISTATS2019](https://proceedings.mlr.press/v89/carter19a.html)。本文具体价值应来自机制定位、可重放对照与信息保留约束，不将基础函数不可恢复性论证重新包装成新普遍定理。

## 原始证据

- tests/test_reference_visibility_pv_lossless_v1.py:145
- reports/REPORT_RC_ORIGINAL_ROMA_SUCCESS_AND_P_INFORMATION_LOSS_20260909.md
- reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md:41
- results/rc_frozen_roma_visibility_mass_factor_v1/result.json
- results/rc_original_raw_visibility_pv_replay_v1/result.json
- registry/rc_original_raw_visibility_pv_replay_completion_v1_20260909.json

回放独立验证不是该目录另一个validation.json：生产程序--phase validate在独立进程比较原缓存标量与完整action后生成result.json；完成记录及prejoin/source SHA闭合。本轮未重跑自然数据回放、未提交新试验。
