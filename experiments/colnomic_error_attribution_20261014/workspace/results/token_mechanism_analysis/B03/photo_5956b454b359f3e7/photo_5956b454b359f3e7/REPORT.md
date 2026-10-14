# B03 · photo_5956b454b359f3e7

按当前评测真值分析，未新增身份人工复核。

包装 `package_036`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_004768`，**纯图像实际 Top-1 错误参考** `reference_004769`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.555008 |
| 错误参考 MaxSim | +0.559852 |
| 正确减错误的分差 | -0.004844 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 110 个。正向贡献 +0.016216，负向贡献 -0.021060，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 610 | -0.004760 | -0.005704 |
| nonvisual | 11 | -0.000016 | -0.001081 |
| subject_nontext | 29 | -0.000386 | -0.009721 |
| subject_text | 81 | +0.000318 | +0.002868 |

最主要负差额来自 `background`，贡献 -0.004760。只保留主体 query token 后，本对分差为 -0.000451。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 562 | 50.2% | True | +8.339603 | 79.4% |
| 错误 | 106 | 23.4% | True | +7.562435 | 75.0% |

主体中 81 个 token 与 OCR 行相交，63 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 374 | subject_text | -0.385707 | A/Victoria/4897/2022 IVR-238 (H1N1), A/Croatia/10136RV/ |  | per 0.7 mL dose, in the recommended ratio of 60 mcg HA each, |
| 369 | subject_text | -0.253580 | 2023 X-425A (H3N2), and B/Michigan/01/2021 (a B/Austria/ |  |  |
| 59 | background | -0.186480 |  |  |  |
| 250 | background | -0.185079 |  |  |  |
| 379 | background | -0.169981 |  |  | per 0.7 mL dose, in the recommended ratio of 60 mcg HA each, |
| 689 | background | -0.166808 |  |  |  |
| 27 | background | -0.165020 |  |  |  |
| 230 | background | -0.161009 |  | GTIN: 00349281425509 | and B/Michigan/01/2021 (a B/Austria/1359417/2021-like |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 553 | background | +0.197169 |  | For Intramuscular Use |  |
| 173 | background | +0.193450 |  | For Intramuscular Use |  |
| 328 | subject_text | +0.178354 | *Triton® X-100 - Registered trademark of Union Carbide, Co., USA. | Swiftwater, PA 18370 USA | Co., USA. |
| 673 | background | +0.156550 |  |  |  |
| 348 | subject_text | +0.156117 | *Triton® X-100 - Registered trademark of Union Carbide, Co., USA. | Swiftwater, PA 18370 USA | Co., USA. |
| 485 | background | +0.141201 |  |  |  |
| 418 | background | +0.139841 |  | For Intramuscular Use | Formula |
| 531 | background | +0.139771 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| (HA) per 0.5 mL dose, in the recommended ratio of 15 mcg | 10 | -0.000062 |
| See full prescribing information for additional details. | 13 | +0.000364 |
| 10 single-dose prefilled syringes 0.5 mL each | 8 | +0.000070 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_3c8abd32bcea6cfa` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004844 | +0.037636 |
| 只保留主体分差 | -0.000451 | +0.073213 |
| 主体 token 数 | 110 | 90 |

[成功对照 token 表](../photo_3c8abd32bcea6cfa/tokens.csv) · [对照摘要](../photo_3c8abd32bcea6cfa/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
