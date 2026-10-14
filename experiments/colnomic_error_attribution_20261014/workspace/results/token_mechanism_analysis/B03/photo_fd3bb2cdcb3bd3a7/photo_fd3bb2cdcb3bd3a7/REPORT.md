# B03 · photo_fd3bb2cdcb3bd3a7

按当前评测真值分析，未新增身份人工复核。

包装 `package_090`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000265`，**纯图像实际 Top-1 错误参考** `reference_003310`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 24 |
| 正确参考 MaxSim | +0.486505 |
| 错误参考 MaxSim | +0.526780 |
| 正确减错误的分差 | -0.040275 |
| F03 的正确参考排名（辅助对照） | 34 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 116 个。正向贡献 +0.049185，负向贡献 -0.089460，相加等于原图像分差。与原评分缓存的最大误差 3e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 652 | -0.059246 | -0.070786 |
| nonvisual | 11 | -0.000276 | -0.019549 |
| subject_nontext | 37 | +0.000472 | +0.009929 |
| subject_text | 79 | +0.018775 | +0.185136 |

最主要负差额来自 `background`，贡献 -0.059246。只保留主体 query token 后，本对分差为 +0.129251。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 10 | 5.1% | False | 不可评估 | 不可评估 |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 79 个 token 与 OCR 行相交，14 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 530 | background | -0.695353 | BEES |  | BEES |
| 536 | background | -0.680526 | TOOTHPASTE |  | Multi-Care |
| 763 | background | -0.668646 |  | JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC |  |
| 562 | background | -0.656629 |  | JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC |  |
| 532 | background | -0.654691 | BEES | ONCE A MONTH | Complete |
| 498 | background | -0.650360 | BURT'S |  | BURT'S |
| 597 | background | -0.648898 |  | ONCE A MONTH |  |
| 313 | subject_nontext | -0.647880 |  |  | Multi-Care |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 337 | subject_text | +0.559950 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 305 | subject_text | +0.547630 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 304 | subject_text | +0.527265 | Comfortis (spinosad) | Comfortis | Complete |
| 338 | subject_text | +0.522637 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 273 | subject_text | +0.517800 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 458 | background | +0.511342 |  | Comfortis™(spinosad) |  |
| 334 | subject_text | +0.502134 | Comfortis (spinosad) | Comfortis |  |
| 302 | subject_text | +0.498188 | Comfortis (spinosad) | Comfortis™(spinosad) |  |

## 同包装成功对照

`photo_d68d74e6a16e45bf` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.040275 | +0.179158 |
| 只保留主体分差 | +0.129251 | +0.387085 |
| 主体 token 数 | 116 | 247 |

[成功对照 token 表](../photo_d68d74e6a16e45bf/tokens.csv) · [对照摘要](../photo_d68d74e6a16e45bf/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
