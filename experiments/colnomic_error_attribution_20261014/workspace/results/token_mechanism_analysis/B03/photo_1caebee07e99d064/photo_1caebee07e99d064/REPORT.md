# B03 · photo_1caebee07e99d064

按当前评测真值分析，未新增身份人工复核。

包装 `package_080`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003367`，**纯图像实际 Top-1 错误参考** `reference_002253`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.537375 |
| 错误参考 MaxSim | +0.538452 |
| 正确减错误的分差 | -0.001077 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 168 个。正向贡献 +0.031546，负向贡献 -0.032623，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 600 | -0.008490 | -0.011023 |
| nonvisual | 11 | -0.000412 | -0.029169 |
| subject_nontext | 80 | +0.003132 | +0.030499 |
| subject_text | 88 | +0.004692 | +0.041538 |

最主要负差额来自 `background`，贡献 -0.008490。只保留主体 query token 后，本对分差为 +0.036282。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 160 | 62.2% | True | +6.589423 | 66.5% |
| 错误 | 5 | 0.3% | False | 不可评估 | 不可评估 |

主体中 88 个 token 与 OCR 行相交，59 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 549 | background | -0.469495 | STERILE |  | STERILE |
| 743 | background | -0.405123 |  | congestion relief | for the relief of redness of the eye due to minor |
| 644 | background | -0.363652 | Eye Drops | No Drip | for the relief of redness of the eye due to minor |
| 575 | background | -0.356661 |  | adults and children 6 to under 12 years of age (with | STERILE |
| 113 | background | -0.333254 | I at an a siney d ee ar paleseea's |  |  |
| 666 | background | -0.324726 |  |  |  |
| 591 | background | -0.305098 |  | sneezing or an increase in nasal discharge may occur |  |
| 193 | background | -0.301134 | FRESHENS | No Drip |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 397 | subject_text | +0.464319 | *CVS/pharmacy Nasal Mist is not | ASAL L | Naphazoline HCI 0.03% |
| 441 | subject_text | +0.461459 | Woonsocket, RI 02895 |  | ©2019 CVS/pharmacy |
| 682 | background | +0.454456 |  | 1 FL OZ (30 mL) | 0.5 FL 0Z (15 mL) |
| 532 | background | +0.441845 |  |  | IS BROKEN OR MISSING. |
| 155 | background | +0.428440 | (milbemycin oxime/praziquantel) | ASAL L | GLYCERIN 0.5% |
| 335 | subject_text | +0.381505 | Inc., distributor of Afrin® No Drip | Afrin® No Drip Original* |  |
| 655 | background | +0.377766 |  |  |  |
| 623 | background | +0.375370 |  | Afrin® No Drip Original* | to avoid contamination, do not touch tip of container to any |

## 同包装成功对照

`photo_174967c2577c6dc9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001077 | +0.094335 |
| 只保留主体分差 | +0.036282 | +0.190819 |
| 主体 token 数 | 168 | 175 |

[成功对照 token 表](../photo_174967c2577c6dc9/tokens.csv) · [对照摘要](../photo_174967c2577c6dc9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
