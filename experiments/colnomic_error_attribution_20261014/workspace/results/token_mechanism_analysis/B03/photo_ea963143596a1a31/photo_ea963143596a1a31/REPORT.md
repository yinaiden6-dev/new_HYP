# B03 · photo_ea963143596a1a31

按当前评测真值分析，未新增身份人工复核。

包装 `package_029`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001023`，**纯图像实际 Top-1 错误参考** `reference_002714`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 13 |
| 正确参考 MaxSim | +0.536344 |
| 错误参考 MaxSim | +0.547692 |
| 正确减错误的分差 | -0.011348 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 56 个。正向贡献 +0.044887，负向贡献 -0.056235，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 664 | -0.015442 | -0.017000 |
| nonvisual | 11 | -0.000617 | -0.041025 |
| subject_nontext | 24 | +0.001216 | +0.037026 |
| subject_text | 32 | +0.003495 | +0.079850 |

最主要负差额来自 `background`，贡献 -0.015442。只保留主体 query token 后，本对分差为 +0.061497。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 715 | 44.3% | True | +10.246733 | 80.0% |
| 错误 | 4 | 11.0% | False | 不可评估 | 不可评估 |

主体中 32 个 token 与 OCR 行相交，27 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 421 | background | -0.448294 |  | Do not use if powch |  |
| 693 | background | -0.428031 |  | Do not use if powch | childon ncio a yc uo no mo thana tre gn-ofirto n |
| 591 | background | -0.421795 |  | Drug Facts (continued) |  |
| 420 | background | -0.416595 |  | Do not use if powch | childon ncio a yc uo no mo thana tre gn-ofirto n |
| 627 | background | -0.409659 |  | Do not use if powch | childon ncio a yc uo no mo thana tre gn-ofirto n |
| 443 | background | -0.403991 |  | Do not use if powch | childon ncio a yc uo no mo thana tre gn-ofirto n |
| 521 | background | -0.400916 |  | Do not use if powch | childon ncio a yc uo no mo thana tre gn-ofirto n |
| 697 | background | -0.398936 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 332 | background | +0.472407 |  |  | Toolhparl |
| 352 | subject_text | +0.429339 | Drug Facts (continued) | Drug Facts (continued) | Toolhparl |
| 261 | background | +0.422881 |  |  |  |
| 431 | subject_text | +0.403322 | taking the blood thinning drug warlarin | do not exceed recommended dosage |  |
| 374 | subject_text | +0.391799 | taldng a prescription monoamine cdas Inhbltor | MAOl drug. If you do not know if your prascription drug contains an | fl uoalde tooth past à |
| 272 | background | +0.388636 |  |  |  |
| 383 | background | +0.374415 |  |  |  |
| 582 | background | +0.370889 |  | I you have ever had an allergle reacton to this product or any of Its |  |

## 同包装成功对照

`photo_d65410305cff7c7c` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.011348 | +0.074831 |
| 只保留主体分差 | +0.061497 | +0.106121 |
| 主体 token 数 | 56 | 56 |

[成功对照 token 表](../photo_d65410305cff7c7c/tokens.csv) · [对照摘要](../photo_d65410305cff7c7c/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
