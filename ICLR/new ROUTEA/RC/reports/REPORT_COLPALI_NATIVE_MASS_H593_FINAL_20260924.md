# ColPali 自有 C128：H593 五折最终结果

日期：2026-09-24。50/50 评分分片、5/5 折训练与最终汇总完成。独立复核通过。

## 协议

冻结 ColPali tokens，针对完整 5413 张 reference 图库进行 FP64 MaxSim，按身份去重，使用 ColPali 自己的自然 C128。H593 五折分组留出，每张 query 恰有一次留出预测，共 64 个 component。目标进入 C128 为 467/593，未进入的 126 张仍计入分母，没有插入 target。

本次迁移的是整体 RoMa 质量 M 与自由图像内容匹配 L 的 MASS5 简化头，不是原局部七参数头。MASS5 输入为 RAW 挑战者相对差距及 M×L、M、L 三项相对证据，另有偏置。CONTENT7 对照使用 RAW 差距与五项纯内容统计及偏置。两种头分别按 COST1、CE 在各折 TRAIN 重训，均 2000 步。

## 结果

| 方法 | 正确/593 | R@1 | 相对 RAW 救回 | 相对 RAW 损失 | C128 MRR |
|---|---:|---:|---:|---:|---:|
| COLPALI_RAW_C128 | 283/593 | 47.72% | 0 | 0 | 0.542559 |
| COLPALI_IMAGE_L | 283/593 | 47.72% | 1 | 1 | 0.542119 |
| CONTENT7_COST1 | 286/593 | 48.23% | 3 | 0 | 0.562214 |
| CONTENT7_CE | 294/593 | 49.58% | 16 | 5 | 0.565426 |
| MASS5_COST1 | 348/593 | 58.68% | 65 | 0 | 0.657496 |
| MASS5_COST1_CBIND | 252/593 | 42.50% | 1 | 32 | 0.477665 |
| MASS5_CE | 365/593 | 61.55% | 93 | 11 | 0.669927 |
| MASS5_CE_CBIND | 120/593 | 20.24% | 2 | 165 | 0.311163 |

## 解释与边界

- MASS5_COST1 相对 RAW 净增 65，保住全部原正确；相对独立重训 CONTENT7_COST1 救回 64、损失 2，净增 62。不能把收益简单归因于多训练了一个头。
- MASS5_CE 相对 RAW 救回 93、损失 11，净增 82。COST1 和 CE 体现不同的纠错与误伤取舍。
- CBIND 在推理时将同一 query 的 128 个候选 M 循环平移 64 位，保持内容统计、RAW 及训练参数不变。COST1 从 348 降至 252，CE 从 365 降至 120，支持正确候选绑定的重要性；该单一错配对照不能独立分解全部视觉来源。
- 原汇总的 component 等权 bootstrap 95% 区间：RAW→MASS5_COST1 为 [0.0892, 0.1741]；RAW→MASS5_CE 为 [0.1186, 0.2299]；CONTENT7_COST1→MASS5_COST1 为 [0.0781, 0.1654]。这些是组均值差区间，不是逐图准确率差区间。
- 支持同一药盒数据上跨检索器、经重训后的机制有效性。不是跨数据集确认，不是原头参数零样本迁移，也没有证明 ColPali tokens 永远无法自行读出相同信息。
- 仍使用 RoMa 提供 M；不是去掉 RoMa 的学生模型。不能将 ColPali 自有 C128 的成绩与 ColNomic 的不同候选池成绩直接解释为机制强弱。

## 完成与复核

Slurm：CPU 评分 5160936_1–49 全部 COMPLETED，41–52 秒；原 GPU 评分分片 0 为 5160875_0；五折训练 5160877_0–4 全部 COMPLETED，各 56 秒；汇总 5160878 COMPLETED，35 秒。

独立复核覆盖 authority/result/fold payload SHA256、五折 query 无重复且覆盖 593 张、全部计数、救回/损失、MRR、component 均值差，以及候选缺失样本没有正确预测。区间读取原汇总，本轮未重做 bootstrap。CPU/GPU 执行差异已在迁移报告中记录，通过数值容差与决策一致性检验，不宣称 bit-exact。

## 文件

- [最终原始结果](../results/rc_colpali_native_mass_head_v1/result.json)
- [最终验收](../results/rc_colpali_native_mass_head_v1/validation.json)
- [独立复核](rc_colpali_native_mass_final_independent_check_20260924.json)
- [CPU 执行迁移报告](REPORT_COLPALI_NATIVE_SCORE_CPU_MIGRATION_20260924.md)
- [冻结科学协议](../registry/rc_colpali_native_mass_head_authority_v1_20260923.json)
