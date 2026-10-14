# B03 · photo_04a4b44735fe9ba8

按当前评测真值分析，未新增身份人工复核。

包装 `package_120`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_001983`，**纯图像实际 Top-1 错误参考** `reference_001939`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.759517 |
| 错误参考 MaxSim | +0.773493 |
| 正确减错误的分差 | -0.013976 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 304 个。正向贡献 +0.022219，负向贡献 -0.036195，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 464 | -0.007263 | -0.012194 |
| nonvisual | 11 | -0.000040 | -0.002817 |
| subject_nontext | 100 | -0.001063 | -0.008282 |
| subject_text | 204 | -0.005610 | -0.021422 |

最主要负差额来自 `background`，贡献 -0.007263。只保留主体 query token 后，本对分差为 -0.017099。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 379 | 65.5% | True | +2.133823 | 47.0% |
| 错误 | 105 | 38.5% | True | +5.505915 | 60.1% |

主体中 204 个 token 与 OCR 行相交，69 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 591 | subject_text | -0.348332 | WHEN TAKEN AS DIRECTED SEE CRUG FACTS PANEL |  | *WHEN TAKEN AS DIRECTED. |
| 60 | background | -0.307123 |  | Actual |  |
| 59 | background | -0.298116 |  | Actual |  |
| 636 | subject_nontext | -0.273903 |  |  |  |
| 29 | background | -0.273841 |  | Loratadine tablets USP 10 mg, antihistamine |  |
| 102 | background | -0.254627 |  | Loratadine tablets USP 10 mg, antihistamine |  |
| 103 | background | -0.254146 |  | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL |  |
| 589 | subject_text | -0.251573 | WHEN TAKEN AS DIRECTED SEE CRUG FACTS PANEL | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL | • No synthetic colors |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 358 | background | +0.396071 |  |  |  |
| 365 | subject_text | +0.391402 | Compare to the active ingredient in Claritin | Compare to the active ingredient in Claritin |  |
| 397 | subject_text | +0.384259 | 24 hour |  |  |
| 364 | subject_nontext | +0.322046 |  | Compare to the active ingredient in Claritin |  |
| 378 | subject_nontext | +0.283935 |  | Compare to the active ingredient in Claritin |  |
| 141 | subject_text | +0.276907 | Walgreensfree+pure | Walgreensfree+pure | Walgreens |
| 410 | subject_nontext | +0.252512 |  | Compare to the active ingredient in Claritin |  |
| 677 | background | +0.247316 |  |  | Compare to the active ingredient in Claritin® Tablets" |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Loratadine tablets USP 10 mg, antihistamine | 25 | -0.001304 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7e0b721a9c25b555` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013976 | +0.008533 |
| 只保留主体分差 | -0.017099 | +0.001896 |
| 主体 token 数 | 304 | 208 |

[成功对照 token 表](../photo_7e0b721a9c25b555/tokens.csv) · [对照摘要](../photo_7e0b721a9c25b555/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
