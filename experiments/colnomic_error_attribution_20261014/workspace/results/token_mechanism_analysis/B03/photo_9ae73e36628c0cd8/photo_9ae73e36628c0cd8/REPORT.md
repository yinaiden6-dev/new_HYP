# B03 · photo_9ae73e36628c0cd8

按当前评测真值分析，未新增身份人工复核。

包装 `package_095`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000377`，**纯图像实际 Top-1 错误参考** `reference_000375`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.735614 |
| 错误参考 MaxSim | +0.735618 |
| 正确减错误的分差 | -0.000004 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 140 个。正向贡献 +0.023247，负向贡献 -0.023251，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 580 | -0.000703 | -0.000886 |
| nonvisual | 11 | -0.000079 | -0.005273 |
| subject_nontext | 77 | -0.000383 | -0.003635 |
| subject_text | 63 | +0.001161 | +0.013469 |

最主要负差额来自 `background`，贡献 -0.000703。只保留主体 query token 后，本对分差为 +0.004062。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 111 | 46.1% | True | +7.985817 | 66.9% |
| 错误 | 79 | 23.5% | True | +6.993505 | 62.6% |

主体中 63 个 token 与 OCR 行相交，28 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 91 | background | -0.349227 |  |  | Product of Japan |
| 318 | subject_nontext | -0.215482 |  |  |  |
| 160 | background | -0.189410 |  |  | Product of Japan |
| 573 | background | -0.180681 |  | If you have any questions concerning the use of this product, contact your veterinarian. | If you have any questions concerning the use of this product, contact your veterinarian. |
| 270 | subject_nontext | -0.173888 |  |  |  |
| 437 | subject_nontext | -0.167840 |  | Storage Conditions: INTERCEPTOR should be stored at room temperature, |  |
| 165 | background | -0.167626 |  |  |  |
| 646 | background | -0.164485 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 514 | subject_nontext | +0.340240 |  |  |  |
| 46 | background | +0.292264 |  |  | Place treatment reminder stickers on your calendar for each of the months following the |
| 685 | background | +0.263368 |  |  |  |
| 93 | background | +0.231010 |  |  | Manufactured for: Elanco US Inc. Greenfield, IN 46140, USA |
| 314 | subject_text | +0.224121 | Veterinarian'simpriothere |  | Veterinarian's imprint here |
| 652 | background | +0.210210 |  |  |  |
| 59 | background | +0.199408 |  |  | Place treatment reminder stickers on your calendar for each of the months following the |
| 479 | background | +0.196927 |  |  | 5-100 1he |

## 同包装成功对照

`photo_5d0a29a6c57b4010` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000004 | +0.022873 |
| 只保留主体分差 | +0.004062 | +0.062275 |
| 主体 token 数 | 140 | 130 |

[成功对照 token 表](../photo_5d0a29a6c57b4010/tokens.csv) · [对照摘要](../photo_5d0a29a6c57b4010/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
