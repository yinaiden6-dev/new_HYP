# B03 · photo_0295d53bc16d3fa3

按当前评测真值分析，未新增身份人工复核。

包装 `package_101`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000412`，**纯图像实际 Top-1 错误参考** `reference_000028`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.755652 |
| 错误参考 MaxSim | +0.770603 |
| 正确减错误的分差 | -0.014952 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 200 个。正向贡献 +0.027056，负向贡献 -0.042007，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 520 | -0.007337 | -0.010314 |
| nonvisual | 11 | +0.000089 | +0.005924 |
| subject_nontext | 63 | +0.000182 | +0.002113 |
| subject_text | 137 | -0.007886 | -0.042077 |

最主要负差额来自 `subject_text`，贡献 -0.007886。只保留主体 query token 后，本对分差为 -0.028158。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 488 | 63.5% | True | +5.600950 | 61.0% |
| 错误 | 8 | 12.9% | False | 不可评估 | 不可评估 |

主体中 137 个 token 与 OCR 行相交，112 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 392 | subject_text | -0.571687 | weight, either mixed with food or placed | weight, either mixed with food or placed | dose of 0.045 mg/lb (0.1 mg/kg) body |
| 32 | background | -0.432999 |  | 5 mg/mL Solution for Injection | 0.09 mg/lb (0.2 mg/kg) body weight |
| 532 | subject_nontext | -0.393776 |  | 5 mg/mL Solution for Injection | 0.09 mg/lb (0.2 mg/kg) body weight |
| 352 | subject_text | -0.376696 | (SQ), followed, after 24 hours, by | 5 mg/mL Solution for Injection | 0.09 mg/lb (0.2 mg/kg) body weight |
| 339 | background | -0.371991 |  |  | 0.09 mg/lb (0.2 mg/kg) body weight |
| 39 | background | -0.362435 |  | Non-steroidal anti-inflammatory drug for |  |
| 393 | subject_text | -0.358090 | weight, either mixed with food or placed | weight, either mixed with food or placed | dose of 0.045 mg/lb (0.1 mg/kg) body |
| 349 | subject_text | -0.349169 | (SQ), followed, after 24 hours, by | meloxicam oral suspension at the daily |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 537 | subject_nontext | +0.450521 |  |  |  |
| 296 | subject_nontext | +0.425704 |  |  |  |
| 197 | subject_nontext | +0.411391 |  |  | Caution: Federal law restricts this |
| 301 | background | +0.403667 |  | Meloxivet | Repeated use of meloxicam in cats |
| 529 | subject_nontext | +0.341316 |  |  |  |
| 311 | subject_text | +0.319292 | Dogs: Meloxivet Injection should be | Meloxivet |  |
| 536 | subject_nontext | +0.302663 |  |  |  |
| 305 | background | +0.285369 |  | Meloxivet |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 0.09 mg/lb (0.2 mg/kg) body weight | 8 | +0.000007 |
| dose of 0.045 mg/lb (0.1 mg/kg) body | 16 | -0.002537 |
| Injection to cats at a dose of 0,14 mg/lb | 16 | -0.001054 |
| (0.3 mg/kg) body weight, Use of | 7 | -0.000883 |
| of dosing, the use of a 1 mL graduated | 16 | -0.001557 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7bc115b787ac85b4` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014952 | +0.049822 |
| 只保留主体分差 | -0.028158 | +0.056720 |
| 主体 token 数 | 200 | 171 |

[成功对照 token 表](../photo_7bc115b787ac85b4/tokens.csv) · [对照摘要](../photo_7bc115b787ac85b4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
