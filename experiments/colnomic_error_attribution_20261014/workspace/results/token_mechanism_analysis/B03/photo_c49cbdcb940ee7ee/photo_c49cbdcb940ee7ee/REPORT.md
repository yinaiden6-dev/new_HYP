# B03 · photo_c49cbdcb940ee7ee

按当前评测真值分析，未新增身份人工复核。

包装 `package_040`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001112`，**纯图像实际 Top-1 错误参考** `reference_003822`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 74 |
| 正确参考 MaxSim | +0.427893 |
| 错误参考 MaxSim | +0.444713 |
| 正确减错误的分差 | -0.016819 |
| F03 的正确参考排名（辅助对照） | 49 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 84 个。正向贡献 +0.017280，负向贡献 -0.034099，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 684 | -0.017986 | -0.020484 |
| nonvisual | 11 | +0.000247 | +0.017496 |
| subject_nontext | 17 | +0.001126 | +0.051596 |
| subject_text | 67 | -0.000206 | -0.002400 |

最主要负差额来自 `background`，贡献 -0.017986。只保留主体 query token 后，本对分差为 +0.008528。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 111 | 66.6% | True | +8.306452 | 78.0% |
| 错误 | 6 | 1.2% | False | 不可评估 | 不可评估 |

主体中 67 个 token 与 OCR 行相交，36 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 501 | subject_text | -0.332955 | ·Sneezing·Runny Nose·itchy, Watery Eyes·Itchy Throotor Nose | Sneezing ·Runny Nose •Itchy, Watery Eyes·Itchy Throat or Nose | • Relief of sneezing; runny nose; itchy, watery eyes & itchy throat or nose |
| 502 | subject_text | -0.303510 | ·Sneezing·Runny Nose·itchy, Watery Eyes·Itchy Throotor Nose | ·Sneezing ·Runny Nose - Itchy, Watery Eyes · Itchy Throat or Nose 45 TABLETS | • Relief of sneezing; runny nose; itchy, watery eyes & itchy throat or nose |
| 474 | subject_text | -0.217033 | Actual Sze | Actual Size | ACTUAL SIZE |
| 465 | subject_text | -0.199096 | Originalprescriptionstrength | 24 Hour relief of: | Indoor & Outdoor Allergies |
| 714 | background | -0.184189 | Dresdner Straße | ·Sneezing ·Runny Nose - Itchy, Watery Eyes · Itchy Throat or Nose 45 TABLETS | 200 WILMOT RD., DEERFIELD, IL 60015 |
| 96 | background | -0.167796 |  | be careful when driving a motor vehicle or operating | Ask a doctor before use if you have liver or kidney disease. |
| 481 | background | -0.166745 |  |  |  |
| 542 | background | -0.166432 | WIE | Questions or comments? 1-877-290-4008 | grain (wheat, barley or rye) |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 385 | background | +0.541910 |  |  | Allergy Relief |
| 386 | background | +0.523469 | TONRT |  | Allergy Relief |
| 400 | subject_text | +0.521571 | Signature | Signature | upper respiratory allergies: |
| 401 | subject_text | +0.487832 | Signature | Signature | When using this product |
| 434 | subject_nontext | +0.485642 |  | Original prescription strength | children under 6 |
| 433 | subject_text | +0.435666 | SELECT | Original prescription strength | ITEM 329218 W10024-0522-L |
| 432 | subject_text | +0.395948 | SELECT | SELECT | When using this product |
| 464 | subject_text | +0.379485 | Originalprescriptionstrength | Original prescription strength |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| TABLETS USP, 10 mg - Antihistamine | 6 | -0.000023 |
| 45 TABLETS | 3 | +0.000368 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_2df687d8a2f89f40` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.016819 | +0.019497 |
| 只保留主体分差 | +0.008528 | +0.043249 |
| 主体 token 数 | 84 | 312 |

[成功对照 token 表](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
