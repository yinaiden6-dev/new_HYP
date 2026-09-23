# new HYP：特征与匹配质量的贡献定位

核查时间：2026-09-22 15:41 UTC。本文依据现有源码、已封存结果及当前执行回执；不修改在运行的科学协议或训练程序。

## 已定位的接口

原系统的直接路径是：

```text
RoMa 粗、细视觉特征 → 两图匹配与细化 → overlap → query/reference token 权重
                                                        ↓
冻结 ColNomic tokens → 内容相似度 → 质量加权评分 → retrieval-only 训练的小头 → HOLD/SWITCH
```

`programs/materialize_rc_original7_train128_roma_v1.py:323–329` 从 match 输出提取 overlap_AB/BA 的 cell_means，再把原 query/reference tokens 交给评分器。`programs/run_romav2_colnomic_visibility_xf_six_case_v1.py:34–35` 的评分器只接收 q、r、wq、wr。因此原系统没有将 RoMa embeddings 直接融合到 ColNomic token，也没有通过 RoMa 更新 ColNomic 主干。

但 RoMa 特征是匹配质量的上游来源。“质量权重是直接接口”不等于“上游特征没有贡献”，更不等于“普通额外特征无法替代匹配器”。overlap 是模型预测的匹配质量证据，不是身份正确概率，也不是已确认的空间 ownership。

## 已有结果支持到哪里

来源：`results/rc_h593_quality_operator_eval_v1/result.json`，SHA256 `aa6a2919419c4d1a6c95d31e03169f795fa67d718271f0267ad06574fc78c935`；独立验证 `QUALITY_OPERATOR_EVAL_ALL_COUNTS_PASS`。

原 H593 分组五折 OOF、自然 RAW C128、相同 HOLD/SWITCH。分母593含23个目标不在候选中的样本；这是已反复使用的内部开发面板。

|机制|COST1 正确/593|对 RAW 救回/损失|
|---|---:|---:|
|RAW 基线|426|0/0|
|完整质量加权、原头|481|58/3|
|整体质量 M × 自由 MaxSim、原头|441|15/0|
|整体质量 M × 自由 MaxSim、同协议重训头|481|57/2|

最后一臂取消 query 局部加权和 reference 权重参与 MaxSim，仍使用完整 RoMa 产生的 M。相对完整 COST1 是4救回/4损失、共21个预测身份改变；相同正确总数不等于相同决策或统计等价。结果支持整体配对质量与内容的校准可以达到本面板相同正确数；不支持局部信息普遍无用，也尚未隔离粗特征、细特征、粗匹配器、细化器的独立贡献。

## 已在执行的定位对照

沿用 `plan/RC_H593_FEATURE_FUSION_TRAIN_V2_20260922.md`；四种来源 × 两种评分，另有无适配器对照。来源为 COL_ONLY、COARSE、FINE、COARSE_FINE；评分为 FREE、ROMA_WEIGHTED。所有适配器均4096参数，query/reference共享，原七参数形式头，主干冻结，原折、两个种子、COST1/CE及预算固定。

|比较|能定位的问题|解释限制|
|---|---|---|
|同来源 FREE 对比 ROMA_WEIGHTED|在该融合方式下，匹配产生的权重是否仍有额外价值|两臂分别训练，表示会共同变化；这是训练流程的整体效果|
|COARSE/FINE/COARSE_FINE 的 FREE 对比 COL_ONLY 的 FREE|无两图匹配器时，上游视觉特征是否能帮助内容编码|必须保留同参数量 COL_ONLY 和无适配器对照，不能把额外训练本身算作特征收益|
|上述来源各自的 ROMA_WEIGHTED 对比 COL_ONLY 的 ROMA_WEIGHTED|原有质量证据之外，直接特征融合是否还能增加收益|负结果只限制本次固定投影和低秩输出门控，不证明特征无信息|
|当前坐标精度干预|内部采样精度改变是否影响 overlap 和检索|不能替代删除细化阶段，也不直接测真实对应误差|

“无匹配器”臂仍读取 RoMa 使用的预训练单图编码器；不能称为完全不依赖其预训练。当前是输出 token 端融合，尚不是进入 ColNomic LLM 前的融合。

如需把质量的产生进一步分解为粗匹配和细化贡献，另需冻结 matcher-only overlap 与 full-refinement overlap 的阶段对照，控制预处理、双向权重映射、候选、训练及头，并保留阶段输出。此对照尚未提交，不能用当前坐标实验冒充。

## 当前真实进度

- 全量特征缓存合格：5324个唯一图片/几何条目，21个分片回执；原始 tokens、池化粗细特征及投影留存。
- 完整 C128 训练容量检查合格；不是自然准确率结果。
- `5157532_0` 已正常完成首片。日志记录 FREE 初始化输入准备和初始头拟合后保存，checkpoint 的联合训练 `step=0/2000`。应称“初始化完成”，不能称已完成一次融合更新。
- `5157555_[1–46%46]` 实时核查均在 accelerated 因 Priority 排队。共享 dispatcher 心跳正常，完整配置验证为0/200；没有新的融合准确率。
- 本次继续利用现有对照定位，没有重复提交一组相同实验。预测全部封存并通过核验后才进行规定的 held 标签汇总。

最终需要分别报告：质量输出能否替代局部权重、上游特征能否直接改善内容、匹配器在融合后是否仍有附加价值。它们是相关但不同的命题。
