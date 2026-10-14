# B03 · photo_441809a846ca8a24

按当前评测真值分析，未新增身份人工复核。

包装 `package_019`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000943`，**纯图像实际 Top-1 错误参考** `reference_003224`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.615537 |
| 错误参考 MaxSim | +0.629358 |
| 正确减错误的分差 | -0.013820 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 160 个。正向贡献 +0.011619，负向贡献 -0.025439，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 560 | -0.009208 | -0.012020 |
| nonvisual | 11 | -0.000265 | -0.017580 |
| subject_nontext | 123 | -0.002731 | -0.016233 |
| subject_text | 37 | -0.001616 | -0.031932 |

最主要负差额来自 `background`，贡献 -0.009208。只保留主体 query token 后，本对分差为 -0.019864。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 5 | 8.7% | False | 不可评估 | 不可评估 |
| 错误 | 6 | 1.7% | False | 不可评估 | 不可评估 |

主体中 37 个 token 与 OCR 行相交，2 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 391 | subject_text | -0.307375 | AsperFlex | AsperFlex | AsperFlex |
| 85 | background | -0.300665 |  |  |  |
| 88 | background | -0.172535 |  |  |  |
| 222 | background | -0.157808 |  |  |  |
| 36 | background | -0.150050 |  |  |  |
| 553 | subject_nontext | -0.148833 |  |  |  |
| 80 | background | -0.147403 |  |  |  |
| 411 | subject_text | -0.145823 | AsperFlex | AsperFlex | AsperFlex |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 397 | background | +0.164572 |  |  |  |
| 588 | background | +0.156205 |  |  |  |
| 204 | background | +0.156042 |  |  |  |
| 385 | background | +0.132173 |  |  |  |
| 259 | background | +0.114360 |  |  |  |
| 7 | background | +0.113567 |  |  | Phryaoplc Acd Phiyvrylpyroldow FD Pogjipusben Sodue Ttyaryiac Tetsrc kcd Titarian Daiics, Veter |
| 162 | background | +0.111581 |  | Flexible Fabric | Flexible Fabric |
| 538 | background | +0.096257 |  |  | use ony  dieted. Fesd and follow al drections and vaningt an this coron do ner alow contast witn the eys |

## 同包装成功对照

`photo_12144e4f9cd970e7` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013820 | +0.013077 |
| 只保留主体分差 | -0.019864 | +0.011894 |
| 主体 token 数 | 160 | 432 |

[成功对照 token 表](../photo_12144e4f9cd970e7/tokens.csv) · [对照摘要](../photo_12144e4f9cd970e7/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
