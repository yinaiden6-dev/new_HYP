# B03 · photo_63d8b363d3a72f0f

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_002203`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.528687 |
| 错误参考 MaxSim | +0.530070 |
| 正确减错误的分差 | -0.001383 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 195 个。正向贡献 +0.023773，负向贡献 -0.025155，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 573 | -0.004253 | -0.005782 |
| nonvisual | 11 | -0.000110 | -0.007757 |
| subject_nontext | 72 | +0.000033 | +0.000355 |
| subject_text | 123 | +0.002947 | +0.018665 |

最主要负差额来自 `background`，贡献 -0.004253。只保留主体 query token 后，本对分差为 +0.011904。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 152 | 47.1% | True | +5.960571 | 64.9% |
| 错误 | 25 | 3.2% | False | 不可评估 | 不可评估 |

主体中 123 个 token 与 OCR 行相交，89 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 524 | background | -0.399908 |  | Treats Frequent Heartburn | • Do not lie flat or bend over soon |
| 556 | background | -0.369846 |  | • Raise the head of your bed | • Do not lie flat or bend over soon |
| 300 | background | -0.352390 |  | Do not eat late at night or just before bedtime. | • Do not lie flat or bend over soon |
| 323 | background | -0.309927 | Bupivacalne Hydrochloride |  |  |
| 400 | subject_text | -0.307945 | fried foods, chocolate, caffeine, alcohol and | Do not eat late at night or just before bedtime. |  |
| 755 | background | -0.274555 |  |  |  |
| 564 | subject_text | -0.249105 | Wear loose-fitting clothing around | • Raise the head of your bed |  |
| 14 | background | -0.248180 | Not  |  | adults 18 years of age and older |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 311 | subject_text | +0.449681 | PLASTIC | PLASTIC | magnesium carbonate, methacrylicacid |
| 343 | subject_text | +0.403080 | PLASTIC | PLASTIC |  |
| 215 | background | +0.361461 |  | PAPER |  |
| 368 | subject_text | +0.349731 | fried foods, chocolate, caffeine, alcohol and | fried foods, chocolate, caffeine, alcohol and | fatty and fried foods, chocolate, caffeine, |
| 407 | subject_text | +0.344566 | PAPER | PAPER |  |
| 431 | subject_text | +0.307029 | cause heartbum, such as rich, spicy, fatty and |  | to cause heartburn, such as rich, spicy, |
| 149 | background | +0.301468 |  |  | 24 |
| 24 | background | +0.281900 | TALETS USP, 10 m- Atmine | PAPER |  |

## 同包装成功对照

`photo_ab6fd7a33337ccee` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001383 | +0.005803 |
| 只保留主体分差 | +0.011904 | +0.021804 |
| 主体 token 数 | 195 | 209 |

[成功对照 token 表](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
