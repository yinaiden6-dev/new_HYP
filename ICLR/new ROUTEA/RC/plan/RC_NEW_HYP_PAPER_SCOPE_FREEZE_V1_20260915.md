# new HYP：当前论文收口与未来工作范围

2026-09-15，Europe/Berlin。依据用户本轮决定：当前研究主线收口；ownership保留；完整过目不忘系统及新编码器训练作为未来发展。本文固定后续写作范围，不修改既有实验、模型、判据或结果。

## 当前贡献

论文统一使用 **new HYP：Reference-conditioned Joint Evidence Calibration for Fine-grained Retrieval**。候选reference提供身份比较的视觉依据；匹配质量与内容相容性形成联合证据，检索监督学习共享的候选比较及纠错规则。当前贡献包括机制形式化、已验证的失效原因与训练改进，以及冻结头的跨数据集收益。

具体实现为：冻结ColNomic全库检索产生自然C128，冻结RoMa提供候选条件的软可见性，ColNomic在完整reference视觉token上进行加权MaxSim，六统计/七参数头比较全部127 challenger，最高logit大于0则SWITCH，否则HOLD。HOLD保持RAW首选。模型无关的机制描述与RoMa/ColNomic的实际实现、权重和预训练来源均应披露。

任务新增训练仅使用query-reference身份/正负检索关系，不引入任务内框、mask、点或对应关系监督。基础模型的原始预训练不被重新称为retrieval-only。无需任务内空间标注的价值可以报告；未经测量的标注人时节省比例不作结论。

训练改进COST4→COST1保持编码器、证据接口、头结构和推理阈值，将训练中压制错误challenger的代价系数由4改为1，再学习共享参数。它修复了所检验场景中“已有正确最高challenger却选择HOLD”的一部分错误，不能被概括为所有历史失败只有一个原因。

## 已封存的跨数据集证据

下表使用同一批固定full-H593商品训练头；主模型COST1、次模型CE；每个数据集均无新增训练或校准。各数据集全库RAW取自然C128，再执行相同127-challenger HOLD/SWITCH。所有分母和证据级别单独保留。

| 数据与候选来源 | RAW | COST4 | 主COST1 | 次CE | COST1相对RAW救/损 | 证据范围 |
|---|---:|---:|---:|---:|---:|---|
| GroZi480；旧5413物理reference追加120张 | 321/480 | 326/480 | 355/480 | 367/480 | 34/0 | 预定范围内外部确认，27来源视频分组 |
| RPC600；独立200-reference gallery | 192/600 | 194/600 | 207/600 | 217/600 | 18/3 | 单商品跨相机外部确认，SKU分组统计 |
| ISIC537；独立390-reference gallery | 466/537 | 469/537 | 500/537 | 513/537 | 34/0 | 已打开且历史选择过队列的跨域探索，346患者分组 |

正确候选证据绑定的贡献有干预支持：COST1绑定打乱后，三组分别为313/480、174/600、414/537。不能将此候选级检验等同于严格空间ownership或细粒度位置绑定的证明。

ISIC主COST1五项预定探索比较通过。相对COST4新救回31张均已是COST4最高challenger但被HOLD。ISIC仍保持`positive_exploratory_transfer_signal=true`、`untouched_external_GO_claimed=false`；不升级为未触碰医学确认或疾病诊断结果。

旧28/32、99/128及其它旧panel、H593 OOF各保留自己的模型和分组协议。表内full-H593头的外部收益不能改写旧固定面板成绩。CE保持预定次模型角色。

## 当前编码器的定位

ColNomic是已经支撑本轮机制与收益验证的冻结内容编码器，但不把它作为完整过目不忘系统的最终表示方案。用户判断其作为冻结token编码器的适配性仍不足，本轮据此将更适合细粒度实例识别的编码器学习列为未来目标。

论文可说明冻结表示无法由本轮七参数校准直接学习新的视觉区分线索，reference信息不足和候选召回缺失也限制可纠错范围。现有成果不构成“ColNomic普遍不适用”或“所有剩余错误均由编码器造成”的受控证明；替换或训练编码器的净增仍属于未来待验证结果。

## 保留方向

| 方向 | 当前定位 | 未来要回答的问题 |
|---|---|---|
| Ownership | 保留为独立扩展，不作为当前论文收口前置条件 | 证据是否确实属于目标物体，能否支持多物体与背景混淆下的空间解释 |
| 完整过目不忘系统 | 长期系统目标 | reference持续登记、图库增长后的旧身份保持、未知对象拒识及复杂场景识别 |
| 新的可训练编码器 | 后续技术目标 | 用检索监督提升相似实例的可辨识表示，检验对跨域、召回和已有正确结果的影响 |

当前论文以已验证范围内的new HYP机制和技术增益为完成目标。旧空间P-only失败保留为历史结果，不因新机制成功改写。形式化性质与经验验证分别陈述，不宣称普遍正确或零错误定理。

后续默认推进论文整理、图表和复现材料；以上未来方向不自动触发新的训练或任务提交。本次范围收口不等于论文文本、文献核对和投稿材料已经全部完成。

## 原始结果与报告

- [GroZi完整报告](../reports/REPORT_NEW_HYP_GROZI120_EXTERNAL_CONFIRMATION_V1_20260913.md)，[原结果](../results/rc_new_hyp_grozi120_external_v1/result.json)，SHA256 `0bee519b388b4e54209a843199ba34285282ec95ec5941ce307b9b9165e426fa`。
- [RPC完整报告](../reports/REPORT_NEW_HYP_RPC_EXTERNAL_CONFIRMATION_V1_20260914.md)，[原结果](../results/rc_new_hyp_rpc_transfer_v1/result.json)，SHA256 `6679415e8301b3b80dd49e57b70ebb7d48606659d294869c79e28fedb486bd08`。
- [ISIC完整报告](../reports/REPORT_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md)，[原结果](../results/rc_new_hyp_isic_transfer_v1/result.json)，SHA256 `6af5bfbafd36f9438372b2c5c1fff3e2f154ffa9c9af84000278809b14ea0d46`。
- [固定头清单](../results/rc_new_hyp_external_head_freeze_v1/bundle.json)。
- [既有形式化定义](../reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md)：数学条件与历史实验结论保留，写作时结合本次范围及后续已封存结果更新。
