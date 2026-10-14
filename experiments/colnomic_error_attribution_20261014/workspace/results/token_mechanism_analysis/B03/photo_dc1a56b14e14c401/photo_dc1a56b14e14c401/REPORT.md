# B03 · photo_dc1a56b14e14c401

按当前评测真值分析，未新增身份人工复核。

包装 `package_021`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000973`，**纯图像实际 Top-1 错误参考** `reference_003637`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 34 |
| 正确参考 MaxSim | +0.645858 |
| 错误参考 MaxSim | +0.688517 |
| 正确减错误的分差 | -0.042660 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 190 个。正向贡献 +0.021669，负向贡献 -0.064328，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.031803 | -0.043864 |
| nonvisual | 11 | -0.000112 | -0.007455 |
| subject_nontext | 36 | +0.001605 | +0.032593 |
| subject_text | 154 | -0.012350 | -0.058622 |

最主要负差额来自 `background`，贡献 -0.031803。只保留主体 query token 后，本对分差为 -0.041339。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 845 | 71.4% | True | +7.649467 | 77.3% |
| 错误 | 11 | 30.3% | True | +6.655049 | 62.5% |

主体中 154 个 token 与 OCR 行相交，130 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 534 | subject_text | -0.478863 | you have difficulty ewallowing | have stomach pain that does not get better | it feels like the pill is stuck in your throat |
| 613 | background | -0.437551 |  |  |  |
| 307 | background | -0.427586 |  |  |  |
| 339 | background | -0.425837 |  | you have had high blood proesurs, hesut dasass, | you have a history of stomach problems, such as heartburn |
| 303 | background | -0.418160 |  | Wanings |  |
| 393 | subject_text | -0.406626 | under a doctor's caus for any verloue cond ition | do not exceed 2 caplets in | Ask a doctor or pharmacist before use if you are |
| 533 | subject_text | -0.400906 | you have difficulty ewallowing | take with food or milk lf stomach upeet occure | it feels like the pill is stuck in your throat |
| 295 | subject_text | -0.377012 | right before or after heart surgary |  | right before or after heart surgery |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 318 | subject_nontext | +0.375333 |  |  | 24 Caplets (CAPSULE-SHAPED TABLETS) |
| 407 | background | +0.354227 |  | Ask a deckor or paaranacat batere use if you are | 24 Caplets (CAPSULE-SHAPED TABLETS) |
| 593 | background | +0.318983 |  | Ask a deckor or paaranacat batere use if you are | 24 Caplets (CAPSULE-SHAPED TABLETS) |
| 312 | subject_text | +0.316203 | the stomach bleeding wauming appilas to you |  | more than directed or for longer than directed. |
| 358 | subject_nontext | +0.310848 |  |  |  |
| 403 | background | +0.294431 |  | Ask a deckor or paaranacat batere use if you are | If pregnant or breast-feeding, ask a health professional |
| 268 | background | +0.261482 |  |  |  |
| 402 | background | +0.251951 |  | Ask a deckor or paaranacat batere use if you are |  |

## 同包装成功对照

`photo_38d1d33620ab1ea1` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.042660 | +0.050937 |
| 只保留主体分差 | -0.041339 | +0.121179 |
| 主体 token 数 | 190 | 54 |

[成功对照 token 表](../photo_38d1d33620ab1ea1/tokens.csv) · [对照摘要](../photo_38d1d33620ab1ea1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
