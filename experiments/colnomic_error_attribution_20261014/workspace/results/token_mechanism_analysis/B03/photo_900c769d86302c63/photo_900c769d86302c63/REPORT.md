# B03 · photo_900c769d86302c63

按当前评测真值分析，未新增身份人工复核。

包装 `package_013`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000929`，**纯图像实际 Top-1 错误参考** `reference_002207`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 19 |
| 正确参考 MaxSim | +0.615544 |
| 错误参考 MaxSim | +0.643012 |
| 正确减错误的分差 | -0.027469 |
| F03 的正确参考排名（辅助对照） | 19 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 225 个。正向贡献 +0.019211，负向贡献 -0.046680，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 495 | -0.021494 | -0.031741 |
| nonvisual | 11 | -0.000311 | -0.020689 |
| subject_nontext | 185 | -0.003197 | -0.012634 |
| subject_text | 40 | -0.002466 | -0.045068 |

最主要负差额来自 `background`，贡献 -0.021494。只保留主体 query token 后，本对分差为 -0.018400。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 88 | 26.1% | True | +9.824098 | 78.8% |
| 错误 | 7 | 21.4% | False | 不可评估 | 不可评估 |

主体中 40 个 token 与 OCR 行相交，23 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 513 | subject_text | -0.455401 | Wooneocket, Ri 02895 | 1G71R 17 C4 | CVS Pharmacy. Inc. |
| 312 | subject_nontext | -0.442035 |  |  | PAPER |
| 374 | subject_nontext | -0.415930 |  | Healthcare, L.P., distributor of Flonase® Allergy Relief. |  |
| 35 | background | -0.414030 |  |  |  |
| 494 | subject_text | -0.409506 | CVS Pharmacy, Inc. |  |  |
| 122 | background | -0.377720 |  | for more than 2 months a year. Be sure to read the Quick Start Guide and Question & | PAPER |
| 574 | subject_text | -0.371677 | CVS.com/returnpolicy |  | CVS.com/returnpolicy |
| 612 | subject_nontext | -0.366579 |  |  | PAPER |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 87 | background | +0.618228 |  |  |  |
| 290 | subject_nontext | +0.466156 |  |  |  |
| 229 | subject_nontext | +0.408700 |  |  |  |
| 316 | subject_nontext | +0.350348 |  |  |  |
| 616 | subject_nontext | +0.331081 |  |  | Drug Facts |
| 23 | background | +0.328011 |  |  |  |
| 476 | subject_nontext | +0.279771 |  |  |  |
| 292 | subject_nontext | +0.262497 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100%money | 5 | -0.000182 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7247f008f2067eb2` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.027469 | +0.288426 |
| 只保留主体分差 | -0.018400 | +0.326057 |
| 主体 token 数 | 225 | 375 |

[成功对照 token 表](../photo_7247f008f2067eb2/tokens.csv) · [对照摘要](../photo_7247f008f2067eb2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
