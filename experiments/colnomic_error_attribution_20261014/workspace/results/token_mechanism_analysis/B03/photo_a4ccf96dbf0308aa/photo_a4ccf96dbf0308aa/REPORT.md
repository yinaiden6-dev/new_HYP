# B03 · photo_a4ccf96dbf0308aa

按当前评测真值分析，未新增身份人工复核。

包装 `package_005`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000841`，**纯图像实际 Top-1 错误参考** `reference_003913`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.556639 |
| 错误参考 MaxSim | +0.561956 |
| 正确减错误的分差 | -0.005317 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 138 个。正向贡献 +0.041918，负向贡献 -0.047235，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 582 | -0.006717 | -0.008437 |
| nonvisual | 11 | +0.000403 | +0.026753 |
| subject_nontext | 110 | +0.002081 | +0.013831 |
| subject_text | 28 | -0.001083 | -0.028286 |

最主要负差额来自 `background`，贡献 -0.006717。只保留主体 query token 后，本对分差为 +0.005285。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 154 | 22.0% | True | +7.949409 | 68.9% |
| 错误 | 21 | 1.0% | False | 不可评估 | 不可评估 |

主体中 28 个 token 与 OCR 行相交，12 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 570 | subject_nontext | -0.462527 |  | 1300 Morris Drive, Chesterbrook, PA 19087 |  |
| 390 | subject_text | -0.426899 | 1300 Morris Drive, Chesterbrook, PA 19087 | 1300 Morris Drive, Chesterbrook, PA 19087 |  |
| 403 | background | -0.424934 |  | AAT |  |
| 384 | background | -0.419882 |  | 1300 Morris Drive, Chesterbrook, PA 19087 |  |
| 160 | background | -0.398441 |  | 1300 Morris Drive, Chesterbrook, PA 19087 |  |
| 457 | background | -0.392939 |  | AAT |  |
| 264 | background | -0.388676 |  | 1300 Morris Drive, Chesterbrook, PA 19087 |  |
| 577 | background | -0.375745 |  | 1300 Morris Drive, Chesterbrook, PA 19087 |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 105 | background | +0.315616 |  |  |  |
| 245 | background | +0.310141 |  |  |  |
| 4 | background | +0.306353 |  |  |  |
| 165 | background | +0.301386 |  |  |  |
| 5 | background | +0.291596 |  |  |  |
| 103 | background | +0.285932 |  |  |  |
| 564 | background | +0.281458 |  |  | Allergy |
| 71 | background | +0.273594 |  |  |  |

## 同包装成功对照

`photo_5c30d65aff48a1e8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005317 | +0.164843 |
| 只保留主体分差 | +0.005285 | +0.198777 |
| 主体 token 数 | 138 | 115 |

[成功对照 token 表](../photo_5c30d65aff48a1e8/tokens.csv) · [对照摘要](../photo_5c30d65aff48a1e8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
