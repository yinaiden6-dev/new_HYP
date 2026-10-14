# B03 · photo_062fafe5f481faca

按当前评测真值分析，未新增身份人工复核。

包装 `package_119`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001971`，**纯图像实际 Top-1 错误参考** `reference_001973`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.758038 |
| 错误参考 MaxSim | +0.760889 |
| 正确减错误的分差 | -0.002851 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 231 个。正向贡献 +0.027931，负向贡献 -0.030782，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 537 | -0.000097 | -0.000141 |
| nonvisual | 11 | -0.000160 | -0.011356 |
| subject_nontext | 95 | +0.000276 | +0.002260 |
| subject_text | 136 | -0.002868 | -0.016430 |

最主要负差额来自 `subject_text`，贡献 -0.002868。只保留主体 query token 后，本对分差为 -0.008744。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 290 | 74.7% | True | +6.138643 | 63.6% |
| 错误 | 43 | 17.1% | True | +6.056289 | 85.6% |

主体中 136 个 token 与 OCR 行相交，32 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 17 | background | -0.465384 |  | (*capsule-shapedtablets) |  |
| 425 | background | -0.421009 |  |  |  |
| 528 | subject_text | -0.403711 | ("capsule-shapedtablets) |  | CAPLETS* |
| 431 | subject_text | -0.357139 | CONT AINS 2 MEDICINES - Acetaminophen + Ituprofen • DUALACTION |  |  |
| 721 | background | -0.350306 |  |  |  |
| 163 | background | -0.347216 |  |  |  |
| 73 | background | -0.338700 |  | •Acetaminophen+Ibuprofen •DUALACTION | Acto ingrodlent (it euch capix) |
| 16 | background | -0.336689 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 461 | subject_text | +0.351552 | 144 | 144 | 216 |
| 493 | subject_text | +0.326196 | 144 | 144 | 216 |
| 71 | background | +0.322543 |  | 144 | 216 |
| 462 | subject_text | +0.287290 | 144 | 144 |  |
| 212 | subject_nontext | +0.247747 |  |  |  |
| 267 | subject_nontext | +0.242654 |  |  |  |
| 750 | background | +0.232826 |  | snachbeedngwaningapplistoyou |  |
| 438 | subject_text | +0.227567 | CONT AINS 2 MEDICINES - Acetaminophen + Ituprofen • DUALACTION |  | -Acetaminophen + Ibuprofen |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| ACE TAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | 50 | -0.002276 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_daac4b2576b9e9f5` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002851 | +0.013929 |
| 只保留主体分差 | -0.008744 | +0.027074 |
| 主体 token 数 | 231 | 170 |

[成功对照 token 表](../photo_daac4b2576b9e9f5/tokens.csv) · [对照摘要](../photo_daac4b2576b9e9f5/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
