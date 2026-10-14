# B03 · photo_bd7ba38f8275e85a

按当前评测真值分析，未新增身份人工复核。

包装 `package_015`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000934`，**纯图像实际 Top-1 错误参考** `reference_002762`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 7 |
| 正确参考 MaxSim | +0.595697 |
| 错误参考 MaxSim | +0.612269 |
| 正确减错误的分差 | -0.016572 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 162 个。正向贡献 +0.036589，负向贡献 -0.053161，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.007919 | -0.010374 |
| nonvisual | 11 | +0.000143 | +0.009502 |
| subject_nontext | 76 | -0.003979 | -0.038271 |
| subject_text | 86 | -0.004817 | -0.040948 |

最主要负差额来自 `background`，贡献 -0.007919。只保留主体 query token 后，本对分差为 -0.039692。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 387 | 43.1% | True | +6.466864 | 75.6% |
| 错误 | 6 | 13.6% | False | 不可评估 | 不可评估 |

主体中 86 个 token 与 OCR 行相交，55 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 294 | subject_text | -0.466044 | Questors or corament? | Questons or comamendr? | Questions or comments? |
| 412 | subject_text | -0.445505 | MADE10 | V6001648 TEMH200000002710 | Made in India |
| 281 | background | -0.438910 |  |  | glycolate, stearic acid |
| 287 | background | -0.432266 |  | Questons or comamendr? | Questions or comments? |
| 283 | background | -0.418091 |  | "This product ls not manubuctured or distribubed by | Questions or comments? |
| 295 | subject_text | -0.394607 | Snct 1-77-7- M -F1 8  EST  E M PST | cuntct a Poleon Contrd Cant dght aw (1-800-222-1222) |  |
| 292 | subject_text | -0.393750 | Questors or corament? |  | glycolate, stearic acid |
| 300 | background | -0.381733 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 627 | background | +0.368249 |  |  | whether a drug contains acetaminophen, ask a |
| 452 | subject_text | +0.365602 | CONTAINS NO ASPIRIN |  | do not take more than directed (see overdose |
| 396 | subject_text | +0.365161 | 35 Wabaryiow Bivi Panalp-peny, NU 07054 | 35 Waterview Bhvd Parslppany, NJ 07054 |  |
| 455 | subject_nontext | +0.358426 |  |  | Liver waming: This product contains acetaminophen. |
| 634 | background | +0.341014 |  |  | whether a drug contains acetaminophen, ask a |
| 580 | background | +0.333399 |  |  | whether a drug contains acetaminophen, ask a |
| 476 | subject_nontext | +0.328397 |  | 200000002710 |  |
| 393 | subject_text | +0.328107 | 35 Wabaryiow Bivi Panalp-peny, NU 07054 | (A division of Graunulae USA, Inc.) | AAA Pharmaceutical, Inc. |

## 同包装成功对照

`photo_4917ad5f3fac9f55` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.016572 | +0.052914 |
| 只保留主体分差 | -0.039692 | +0.109073 |
| 主体 token 数 | 162 | 50 |

[成功对照 token 表](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
