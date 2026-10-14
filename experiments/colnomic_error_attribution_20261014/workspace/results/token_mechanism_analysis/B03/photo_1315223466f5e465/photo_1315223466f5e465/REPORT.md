# B03 · photo_1315223466f5e465

按当前评测真值分析，未新增身份人工复核。

包装 `package_099`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000797`，**纯图像实际 Top-1 错误参考** `reference_000796`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.591154 |
| 错误参考 MaxSim | +0.594244 |
| 正确减错误的分差 | -0.003090 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 156 个。正向贡献 +0.019698，负向贡献 -0.022788，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.003296 | -0.004195 |
| nonvisual | 11 | -0.000061 | -0.004285 |
| subject_nontext | 65 | -0.001420 | -0.017021 |
| subject_text | 91 | +0.001686 | +0.014437 |

最主要负差额来自 `background`，贡献 -0.003296。只保留主体 query token 后，本对分差为 +0.001329。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 31 | 24.0% | True | +11.650102 | 69.0% |
| 错误 | 132 | 33.5% | True | +6.339561 | 66.7% |

主体中 91 个 token 与 OCR 行相交，45 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 133 | background | -0.544727 |  | Drug Facts (continued) |  |
| 470 | subject_nontext | -0.253923 |  | Kids Ear |  |
| 502 | subject_nontext | -0.253421 |  |  |  |
| 309 | subject_text | -0.207332 | Made in Switzerland | Manufactured by: Similasan AG, Switzerland | unpainted/uncoated area |
| 751 | background | -0.153395 |  |  |  |
| 214 | subject_nontext | -0.150454 |  | Kids Ear |  |
| 58 | background | -0.145119 |  |  | Similasan |
| 688 | background | -0.143412 |  | + |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 276 | subject_text | +0.321077 | Corporation, Highlands Ranch, CO 80129 | Farb/Lackaussparung |  |
| 587 | background | +0.305066 |  |  | Corporation, Highlands Ranch, CO 80129 |
| 291 | background | +0.194186 |  |  |  |
| 766 | background | +0.180806 |  |  | Corporation, Highlands Ranch, CO 80129 |
| 407 | subject_text | +0.175050 | Farb/Lackaussparung | Farb/Lackaussparung | Farti/Lackaussparung |
| 340 | subject_text | +0.160596 | Corporation, Highlands Ranch, CO 80129 | Corporation, Highlands Ranch, CO 80129 | Corporation, Highlands Ranch, CO 80129 |
| 170 | background | +0.159353 |  | Similasan | Similasan |
| 555 | background | +0.156500 |  | Made in Switzerland | 0m9011021g |

## 同包装成功对照

`photo_15676d7248e6a493` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003090 | +0.062646 |
| 只保留主体分差 | +0.001329 | +0.137862 |
| 主体 token 数 | 156 | 140 |

[成功对照 token 表](../photo_15676d7248e6a493/tokens.csv) · [对照摘要](../photo_15676d7248e6a493/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
