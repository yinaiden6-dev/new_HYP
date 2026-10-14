# B03 · photo_7c65ac75f6293f15

按当前评测真值分析，未新增身份人工复核。

包装 `package_087`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_005035`，**纯图像实际 Top-1 错误参考** `reference_002159`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 6 |
| 正确参考 MaxSim | +0.535830 |
| 错误参考 MaxSim | +0.544223 |
| 正确减错误的分差 | -0.008393 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 240 个。正向贡献 +0.043793，负向贡献 -0.052186，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 528 | -0.023825 | -0.035151 |
| nonvisual | 11 | -0.000366 | -0.025930 |
| subject_nontext | 197 | +0.007289 | +0.028823 |
| subject_text | 43 | +0.008509 | +0.154158 |

最主要负差额来自 `background`，贡献 -0.023825。只保留主体 query token 后，本对分差为 +0.051279。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 97 | 44.1% | True | +7.676831 | 75.2% |
| 错误 | 6 | 27.1% | False | 不可评估 | 不可评估 |

主体中 43 个 token 与 OCR 行相交，13 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 535 | subject_nontext | -0.481860 |  |  |  |
| 516 | background | -0.450425 |  |  |  |
| 422 | background | -0.422449 |  |  |  |
| 517 | background | -0.422149 |  |  |  |
| 358 | background | -0.399851 |  | Add the remalning water and shake to |  |
| 521 | background | -0.389102 |  |  |  |
| 636 | background | -0.371291 |  | Suspension, USP |  |
| 424 | background | -0.367782 |  | Add the remalning water and shake to |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 560 | subject_nontext | +0.556786 |  |  |  |
| 431 | subject_text | +0.522884 | Aurobindo Pharma USA, Inc. |  |  |
| 463 | subject_text | +0.512763 | Aurobindo Pharma USA, Inc. |  |  |
| 442 | subject_nontext | +0.456677 |  |  |  |
| 399 | subject_text | +0.446519 | Aurobindo Pharma USA, Inc. |  |  |
| 593 | background | +0.443304 |  |  | Distributed by: CVS Pharmacy, Inc. |
| 527 | subject_text | +0.431227 | Aurobindo Pharma USA, Inc. | 279 Princeton-Hightstown Road |  |
| 465 | subject_text | +0.407289 | Made In India | Made In Indla |  |

## 同包装成功对照

`photo_d4b8067813323602` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008393 | +0.150583 |
| 只保留主体分差 | +0.051279 | +0.261814 |
| 主体 token 数 | 240 | 230 |

[成功对照 token 表](../photo_d4b8067813323602/tokens.csv) · [对照摘要](../photo_d4b8067813323602/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
