# B03 · photo_fd5974ccf5d29b5a

按当前评测真值分析，未新增身份人工复核。

包装 `package_028`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001012`，**纯图像实际 Top-1 错误参考** `reference_003684`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.548676 |
| 错误参考 MaxSim | +0.551140 |
| 正确减错误的分差 | -0.002464 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 220 个。正向贡献 +0.052476，负向贡献 -0.054941，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 548 | -0.025678 | -0.036501 |
| nonvisual | 11 | -0.000858 | -0.060784 |
| subject_nontext | 50 | +0.002423 | +0.037746 |
| subject_text | 170 | +0.021649 | +0.099203 |

最主要负差额来自 `background`，贡献 -0.025678。只保留主体 query token 后，本对分差为 +0.085235。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 469 | 66.8% | True | +4.616679 | 55.9% |
| 错误 | 11 | 20.7% | True | +11.748874 | 97.6% |

主体中 170 个 token 与 OCR 行相交，100 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 767 | background | -0.580942 |  | the owner of the registered trademark COLACE®. |  |
| 586 | background | -0.553174 | Irritated |  |  |
| 489 | background | -0.510598 | Walgreens | If pregnant or breast-feeding, ask a health | Walgreens |
| 520 | background | -0.484150 | w/19 | Inactive ingredients: gelatin (bovine), | STERILE 0.34 FL OZ (10 mL) |
| 115 | background | -0.475997 | PerriconeMD |  |  |
| 8 | background | -0.473761 | Kor uet, | Softener | Eye Itch Relief |
| 392 | background | -0.470650 |  | relieves occasionall constipation (irregularity) | Eye Itch Relief |
| 138 | background | -0.466289 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 511 | background | +0.530713 |  |  | 100% SATISFACTION GUARANTEED walgreens.com |
| 375 | subject_text | +0.472877 | glycerin, polyethylene glycol, purified water | Inactive ingredients: gelatin (bovine), |  |
| 500 | subject_text | +0.449359 | each softgel contains: sodium 7 mg. Very | Stool Softener | Actual Size |
| 542 | subject_text | +0.440514 | SAN ANTONIO, TEXAS 78204 | PRODUCT OF ROMANIA |  |
| 340 | subject_text | +0.434062 | each softgel contains: sodium 7 mg. Very | Docusate Sodium, 100 mg |  |
| 342 | subject_text | +0.424318 | Inactive ingredients: gelatin (bovine), |  | glycerin, hydrochloric acid and/or |
| 407 | subject_text | +0.421436 | glycerin, polyethylene glycol, purified water | Inactive ingredients: gelatin (bovine), |  |
| 546 | background | +0.415222 |  | edible ink, FD&C red #40, FD&C yellow #6 | 3651 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| each softgel contains: sodium 7 mg. Very | 15 | +0.003990 |
| sorbitol. Contains less than 1% of D&C red #33, | 19 | +0.003166 |
| 100% | 4 | +0.000442 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_b35bafd0e0a8761b` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002464 | +0.068501 |
| 只保留主体分差 | +0.085235 | +0.225275 |
| 主体 token 数 | 220 | 210 |

[成功对照 token 表](../photo_b35bafd0e0a8761b/tokens.csv) · [对照摘要](../photo_b35bafd0e0a8761b/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
