# B03 · photo_368eda8ed55ea25d

按当前评测真值分析，未新增身份人工复核。

包装 `package_017`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000936`，**纯图像实际 Top-1 错误参考** `reference_004378`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.675685 |
| 错误参考 MaxSim | +0.682115 |
| 正确减错误的分差 | -0.006430 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 360 个。正向贡献 +0.057797，负向贡献 -0.064226，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 360 | -0.001920 | -0.003898 |
| nonvisual | 11 | -0.000252 | -0.016766 |
| subject_nontext | 96 | -0.004067 | -0.030968 |
| subject_text | 264 | -0.000191 | -0.000529 |

最主要负差额来自 `subject_nontext`，贡献 -0.004067。只保留主体 query token 后，本对分差为 -0.008646。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1251 | 86.8% | True | +4.346096 | 53.9% |
| 错误 | 9 | 8.5% | False | 不可评估 | 不可评估 |

主体中 264 个 token 与 OCR 行相交，141 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 269 | subject_text | -0.604934 | Ask a doctor bafore ume if you have |  | Ask a doctor before use If you have |
| 311 | subject_text | -0.514612 | These could be eigne of a serlous liness. |  | or occurs with fever, rash, or persistent |
| 124 | background | -0.512012 |  |  | store between 20-25°C (68-77°F) |
| 271 | subject_text | -0.500287 | Ask a doctor bafore ume if you have | peraistent or chronic cough such as occurs with smoking, asthma, chronic bronchitis, or emphysema | persistent or chronic cough such as occurs |
| 619 | background | -0.499468 |  |  | store between 20-25°C (68-77°F) |
| 316 | subject_text | -0.490602 | cough lasts more than 7 days, comes back, or occury with fever, rash, or porslatent headache. |  Tamper evident: Do not use lf carton is open or i printed seal on biister Is broken or inissing. |  |
| 630 | background | -0.484693 |  |  | store between 20-25°C (68-77°F) |
| 267 | subject_text | -0.477864 | Ask a doctor bafore ume if you have | cough lasts more than 7 days, comos back, or occurs with fover, rash, or perslatent hoadache. | Ask a doctor before use If you have |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 115 | background | +0.573249 |  | CVS |  |
| 554 | subject_text | +0.569635 | One CVS Drive, Woonsocket, Ri 02896 |  |  |
| 196 | subject_text | +0.561544 | Active ingredlent (in each extended-release bHayer tablet) | Active ingredient (in each extended-release bHayar tablet) |  |
| 56 | background | +0.541945 |  | CVS |  |
| 492 | subject_text | +0.539763 | Mon-Fri 8:00 AN EST to 5:00 PM PST. |  | 5181813 |
| 575 | subject_text | +0.524795 | CVS.com 1-80D-SHOP CVS | Made In Indla | Questions? (1-800-406-7984) |
| 615 | background | +0.520356 |  |  |  |
| 692 | background | +0.519722 |  |  | 5181813 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Gualfeneein 1200 mg | 7 | +0.000203 |
| adulks and chlidren 12 yeaus of age and ovar: 1 tablet every 12 hours. Do not axceed 2 tablets In | 14 | +0.000281 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_f0d980d139f16444` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006430 | +0.015203 |
| 只保留主体分差 | -0.008646 | +0.018352 |
| 主体 token 数 | 360 | 384 |

[成功对照 token 表](../photo_f0d980d139f16444/tokens.csv) · [对照摘要](../photo_f0d980d139f16444/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
