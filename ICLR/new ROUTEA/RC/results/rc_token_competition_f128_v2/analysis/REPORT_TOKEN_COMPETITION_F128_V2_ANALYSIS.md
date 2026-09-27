# TOKEN competition V2：F128 开发面板完整分析

本报告只在完整 independent join PASS 且输入 SHA/来源绑定全部复核后生成。主结果使用预先固定的 zero 阈值；继承阈值单独列作次要结果。

- **TOKEN_QR 对 B_CAL**：净增加 5 个正确决策（+3.91 个百分点）；配对 identity-cluster 95% CI 未完全高于 0，尚不能据此确认稳定增益。
- **TOKEN_QRR_ANCHOR 对 B_CAL**：净增加 5 个正确决策（+3.91 个百分点）；配对 identity-cluster 95% CI 未完全高于 0，尚不能据此确认稳定增益。
- **TOKEN_QRR_MULTI 对 B_CAL**：净增加 3 个正确决策（+2.34 个百分点）；配对 identity-cluster 95% CI 未完全高于 0，尚不能据此确认稳定增益。
- **TOKEN_QRR_MULTI 对 TOKEN_QRR_ANCHOR**：净减少 2 个正确决策（-1.56 个百分点），当前配置不支持优于对照的主张。

## 主结果：fixed0

模型为 seed0 的三个 native-token residual reader，叠加 SAME F128 分阶段冻结的 18 维 B_CAL；候选来自冻结的自然 ColNomic C128，全部 128 个候选均可选择。F128 是已打开的开发面板，保持原分组五折 OOF。

| 模型 | 对照 | 正确 /128 | 救回 | 破坏 | 净变化 | 改变决策 | 对照正确样本损失率 | 配对增益 95% CI（百分点） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TOKEN_QR | B_CAL (103/128) | 108 | 9 | 4 | +5 | 20 | 3.88% | [-0.86, 9.24] |
| TOKEN_QRR_ANCHOR | B_CAL (103/128) | 108 | 9 | 4 | +5 | 20 | 3.88% | [-1.53, 9.57] |
| TOKEN_QRR_MULTI | B_CAL (103/128) | 106 | 7 | 4 | +3 | 17 | 3.88% | [-2.29, 6.96] |
| TOKEN_QRR_MULTI | TOKEN_QRR_ANCHOR (108/128) | 106 | 0 | 2 | -2 | 6 | 1.85% | [-4.00, 0.00] |

救回指对照错误而该模型正确；破坏指对照正确而该模型错误；净变化 = 救回 − 破坏。对照正确样本损失率完整纳入已有正确样本的回归损失。改变决策也包含两者都错误的情况。CI 以目标 identity 为重采样簇、10,000 次 bootstrap、随机种子 20260927；单位是准确率差，表中转为百分点。它是开发面板上的描述性不确定性估计，不是跨 seed、跨数据集确认，也未做多重比较校正。

## 折间一致性与训练选择

| 模型 | 对照 | 五折净变化（fold0→4） | 正 / 零 / 负折数 |
|---|---|---|---|
| TOKEN_QR | B_CAL | +1, +2, +0, -1, +3 | 3 / 1 / 1 |
| TOKEN_QRR_ANCHOR | B_CAL | +1, +3, +0, -1, +2 | 3 / 1 / 1 |
| TOKEN_QRR_MULTI | B_CAL | +0, +3, +0, -1, +1 | 2 / 2 / 1 |
| TOKEN_QRR_MULTI | TOKEN_QRR_ANCHOR | -1, +0, +0, +0, -1 | 0 / 3 / 2 |

五折净变化仅用于检查增益是否集中于少数折，不把五折当成五次独立复现。

| 模型 | 主 epoch（fold0→4） | CE 最优 epoch | epoch0 回退数 /5 | reader 参数量 |
|---|---|---|---:|---:|
| TOKEN_QR | 5, 7, 7, 6, 6 | 5, 7, 7, 6, 6 | 0 | 11744 |
| TOKEN_QRR_ANCHOR | 5, 6, 5, 6, 6 | 5, 6, 5, 6, 6 | 0 | 11760 |
| TOKEN_QRR_MULTI | 5, 5, 4, 5, 4 | 5, 5, 4, 5, 4 | 0 | 11760 |

选择规则在训练前冻结：仅 inner-validation fixed0 正确数严格超过 B_CAL 的 epoch 合格，从合格 epoch 中取 CE 最低者，平局取最早者；无合格者回退 epoch0，精确保留 B_CAL。best-CE epoch 是选择诊断，不能直接替换主 epoch；outer refit 只执行选定 epoch 数。没有使用这里的 held 结果重新挑选 arm、epoch、阈值或 seed。

## whole-C128 成员覆盖与目标深度

目标 identity 在 C128 内：**120/128 (93.75%)**；缺失 8。按冻结 B_CAL 全 C128 logit 排名，目标在 top2 的有 108，在 rank3–128 的有 12。
成员覆盖是当前固定候选池的上限，独立于 reader 的排序/动作能力。目标深度采用 logit 降序、相同分数按 physical row 升序，仅作诊断；实际 HOLD/SWITCH 仍使用已冻结的唯一最大 challenger 及严格阈值规则，平局 HOLD。

| 模型（fixed0） | 目标 B_CAL 深度 | 查询数 | 对照错误数 | 救回 | 破坏 | 净变化 |
|---|---|---:|---:|---:|---:|---:|
| TOKEN_QR | top2 | 108 | 5 | 4 | 4 | +0 |
| TOKEN_QR | rank3_128 | 12 | 12 | 5 | 0 | +5 |
| TOKEN_QR | absent | 8 | 8 | 0 | 0 | +0 |
| TOKEN_QRR_ANCHOR | top2 | 108 | 5 | 4 | 4 | +0 |
| TOKEN_QRR_ANCHOR | rank3_128 | 12 | 12 | 5 | 0 | +5 |
| TOKEN_QRR_ANCHOR | absent | 8 | 8 | 0 | 0 | +0 |
| TOKEN_QRR_MULTI | top2 | 108 | 5 | 4 | 4 | +0 |
| TOKEN_QRR_MULTI | rank3_128 | 12 | 12 | 3 | 0 | +3 |
| TOKEN_QRR_MULTI | absent | 8 | 8 | 0 | 0 | +0 |

rank3–128 的救回可证明当前模型确实纠正过较深候选，但不能单独证明新的候选召回、空间所有权或因果定位能力。缺失组在该固定候选轴上无法救回。逐查询表保留目标深度、动作、候选 residual 和所选 rival，便于检查具体行为。

## 次要结果：继承 B_CAL 阈值

继承阈值直接沿用 SAME F128 各折 B_CAL 阈值，没有新拟合。以下结果与 fixed0 分开解释，不据此改换主 operating point。

| 模型 | 对照 | 正确 /128 | 救回 / 破坏 / 净变化 | 配对增益 95% CI（百分点） |
|---|---|---:|---|---|
| TOKEN_QR | B_CAL (93/128) | 101 | 11 / 3 / +8 | [0.92, 12.06] |
| TOKEN_QRR_ANCHOR | B_CAL (93/128) | 102 | 14 / 5 / +9 | [0.74, 14.00] |
| TOKEN_QRR_MULTI | B_CAL (93/128) | 105 | 13 / 1 / +12 | [3.82, 15.94] |
| TOKEN_QRR_MULTI | TOKEN_QRR_ANCHOR (102/128) | 105 | 5 / 2 / +3 | [-1.54, 6.72] |

## 能支持什么，尚不能支持什么

**最直接的隔离比较是 MULTI 对 ANCHOR。** 两者参数结构、初始化规则、native token 输入、B_CAL、损失、训练及选择规则相同，预先定义的差别是 target-free rival graph：ANCHOR 使用 RAW anchor；MULTI 在每个候选处额外加入冻结 B_CAL 最强的非自身、非 RAW rival，去重后取均值。训练轨迹和最终选择 epoch 可以随这一结构干预而变化。其当前 fixed0 结论为：净减少 2 个正确决策（-1.56 个百分点），当前配置不支持优于对照的主张。

**QR 是辅助结构对照。** 它使用相同局部输入，但独立候选读出与联合比较读出的参数结构不同，仅近似匹配有效容量；因此 QR 与 joint reader 的差别不能宣称是完全等参数的单因素因果结论。

**不能将与旧 QRR 的差别唯一归因于压缩。** native-token V2 同时改变局部内容表示、匹配/支持交互、读出结构及候选竞争机制等因素。本报告没有将旧 QRR 数值混入这组主配对比较；若要证明压缩是唯一原因，仍需同输入、同结构、同训练/选择规则的受控压缩消融。

**保留 original M。** 原 common features、M 与阶段 RMS、18 维 B_CAL 都保持原绑定；当前改变的是 native-token residual 读出与竞争。这些结果不构成 M 重设计的证据，也不自动证明 P/HYP、空间归属或全图库候选召回改善。

**证据范围仅为 seed0、已打开的 F128 开发面板。** 这不是旧 EVAL128，不是 H593 正式结果替换，也没有外部数据确认。不从本表选择后续最佳 arm 后再把同一面板称为 untouched confirmation。

## 完整性与交付

最终 gate：`TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS`。程序复核所有显式 SHA 绑定，包括 protocol/source、15 个独立 receipt、各 fit artifact、baseline、证据 manifest、trace 与最终汇总，并从最终绑定 curator/gallery 复核 identity 判定、官方计数、配对统计与 CI。该程序不替代原来的模型数值 replay。

交付：[结构化摘要](analysis_summary.json)、[分折表](per_fold.csv)、[逐查询诊断](per_query_diagnostics.csv)。15 份 model.pt 及 checkpoint/trace 大文件保持本地；此目录只含分析摘要和表格，不复制模型权重。
