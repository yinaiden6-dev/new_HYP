# B03 · photo_1c020a9f93e2a55c

按当前评测真值分析，未新增身份人工复核。

包装 `package_056`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001208`，**纯图像实际 Top-1 错误参考** `reference_001209`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.570512 |
| 错误参考 MaxSim | +0.576862 |
| 正确减错误的分差 | -0.006349 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 105 个。正向贡献 +0.015246，负向贡献 -0.021596，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 615 | -0.007824 | -0.009299 |
| nonvisual | 11 | -0.000366 | -0.024331 |
| subject_nontext | 62 | +0.001215 | +0.014326 |
| subject_text | 43 | +0.000626 | +0.010634 |

最主要负差额来自 `background`，贡献 -0.007824。只保留主体 query token 后，本对分差为 +0.012814。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 103 | 28.3% | True | +9.721372 | 84.6% |
| 错误 | 58 | 20.9% | True | +12.521361 | 86.7% |

主体中 43 个 token 与 OCR 行相交，29 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 159 | background | -0.259677 |  |  |  |
| 147 | background | -0.257449 |  |  |  |
| 119 | background | -0.203366 |  | Original |  |
| 106 | background | -0.172282 |  |  |  |
| 262 | background | -0.170626 |  | preparation is used causes discomfort, or if a rash or irritation should develop, discontinue use and |  |
| 247 | background | -0.160171 |  |  |  |
| 110 | background | -0.152642 |  | ANAL-ESE |  |
| 183 | background | -0.146958 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 299 | background | +0.300959 |  |  | ANALESE |
| 279 | background | +0.189666 |  |  |  |
| 355 | subject_text | +0.163235 | Active Ingredient: Benzocaine 7.5% |  | ANALESE |
| 560 | background | +0.155117 |  |  | Inactive Ingredients: PEG-8, Glycerin, PEG-75, Sodium Saccharin, Flavor, Red 40 |
| 346 | background | +0.144657 |  |  |  |
| 115 | background | +0.142953 |  |  | ANALESE |
| 297 | background | +0.134427 |  |  | ANALESE |
| 309 | background | +0.121836 |  |  | ANALESE |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Active Ingredient: Benzocaine 7.5% | 6 | +0.000157 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_8c6978ca27f16e82` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006349 | +0.000086 |
| 只保留主体分差 | +0.012814 | +0.002965 |
| 主体 token 数 | 105 | 144 |

[成功对照 token 表](../photo_8c6978ca27f16e82/tokens.csv) · [对照摘要](../photo_8c6978ca27f16e82/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
