# B03 · photo_e1cecb781eaf3b03

按当前评测真值分析，未新增身份人工复核。

包装 `package_080`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003367`，**纯图像实际 Top-1 错误参考** `reference_004600`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.585068 |
| 错误参考 MaxSim | +0.587363 |
| 正确减错误的分差 | -0.002296 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 368 个。正向贡献 +0.030790，负向贡献 -0.033086，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 400 | -0.005050 | -0.009834 |
| nonvisual | 11 | -0.000214 | -0.015126 |
| subject_nontext | 158 | -0.000293 | -0.001444 |
| subject_text | 210 | +0.003260 | +0.012094 |

最主要负差额来自 `background`，贡献 -0.005050。只保留主体 query token 后，本对分差为 +0.006282。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 554 | 33.0% | True | +8.009024 | 70.5% |
| 错误 | 52 | 17.0% | True | +7.838048 | 74.1% |

主体中 210 个 token 与 OCR 行相交，136 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 398 | subject_text | -0.451930 | congestion to recur or worsen | adult supervision): 2 or 3 sprays in each nostril not | temporary discomfort such as buming, stinging, sneezing |
| 573 | subject_text | -0.409367 | head, insert nozzle into nostril. Fully depress rim with a | firm, even stroke and sniff deeply. Wipe nozzle clean | clockwise. To spray, squeeze bottle quickly and firmly. Do |
| 408 | background | -0.367563 |  | temporarily relieves nasal congestion due to: |  |
| 263 | subject_nontext | -0.365953 |  | temporarily relieves nasal congestion due to: | upper respiratory allergies |
| 373 | subject_text | -0.339722 | When using this product |  | When using this product |
| 656 | subject_nontext | -0.339534 |  |  |  |
| 410 | background | -0.336538 |  |  |  |
| 705 | subject_nontext | -0.326778 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 641 | subject_text | +0.520269 | SAFETY SEALED: Do not use If Imprinted seal on bollle | SAFETY SEALED: Do not use If Imprinted seal on bottle | Other information |
| 621 | subject_nontext | +0.481662 |  | SAFETY SEALED: Do not use If Imprinted seal on bottle |  |
| 642 | subject_text | +0.458895 | Is broken or missing | SAFETY SEALED: Do not use If Imprinted seal on bottle | Keep out of reach of children. If swallowed, get medical |
| 666 | subject_nontext | +0.445436 |  |  |  |
| 640 | subject_text | +0.444811 | SAFETY SEALED: Do not use If Imprinted seal on bollle | SAFETY SEALED: Do not use If Imprinted seal on bottle | distributor of Afrin®. |
| 664 | subject_text | +0.444217 | Is broken or missing |  | Keep out of reach of children. If swallowed, get medical |
| 634 | subject_nontext | +0.379695 |  |  | Other information |
| 467 | subject_text | +0.364373 | Pharmacy | CVS | Drug Facts |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Oxymetazoline hydrochloride 0.05%... Nasal Decongestant | 10 | +0.000184 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_587142eeed6488be` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002296 | +0.047033 |
| 只保留主体分差 | +0.006282 | +0.064485 |
| 主体 token 数 | 368 | 276 |

[成功对照 token 表](../photo_587142eeed6488be/tokens.csv) · [对照摘要](../photo_587142eeed6488be/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
