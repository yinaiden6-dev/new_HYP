# B03 · photo_7e79bf355aa7c38b

按当前评测真值分析，未新增身份人工复核。

包装 `package_001`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000820`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 1574 |
| 正确参考 MaxSim | +0.489893 |
| 错误参考 MaxSim | +0.566064 |
| 正确减错误的分差 | -0.076171 |
| F03 的正确参考排名（辅助对照） | 19 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 180 个。正向贡献 +0.026195，负向贡献 -0.102366，相加等于原图像分差。与原评分缓存的最大误差 3e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 588 | -0.081100 | -0.107444 |
| nonvisual | 11 | -0.000174 | -0.012353 |
| subject_nontext | 48 | -0.001784 | -0.028952 |
| subject_text | 132 | +0.006887 | +0.040644 |

最主要负差额来自 `background`，贡献 -0.081100。只保留主体 query token 后，本对分差为 +0.022085。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 50 | 56.0% | True | +9.219704 | 78.2% |
| 错误 | 8 | 17.0% | True | +15.137406 | 97.7% |

主体中 132 个 token 与 OCR 行相交，77 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 192 | background | -0.648343 | SAL L |  | ASAL L |
| 451 | background | -0.647343 | rug may take 1 to 4 days for fl |  | 7 |
| 267 | background | -0.638505 | Softener |  | 7 |
| 383 | background | -0.637243 |  |  |  |
| 444 | subject_text | -0.628641 | moving upine front & back ot me neck. Use in tme m orning tor |  | 7 |
| 159 | background | -0.628554 | SAL L |  | ASAL L |
| 381 | subject_nontext | -0.625422 |  |  | 7 |
| 483 | background | -0.610161 | frequent heartburn |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 497 | subject_text | +0.443766 | rects the appearance of uneven skin tone & sun dannge. |  | Afrin® No Drip Original* |
| 472 | subject_text | +0.408849 | Stop use nd aak a doclof lf rasn co |  |  |
| 276 | subject_text | +0.375981 | Uses - Hetps prevent sun bum. • I used as drected with other sun |  |  |
| 465 | subject_text | +0.351689 | rects the appearance of uneven skin tone & sun dannge. |  | Warnings |
| 471 | subject_text | +0.326209 | • Do not use on damaged or broten sxn |  |  |
| 476 | subject_text | +0.313402 | moving upine front & back ot me neck. Use in tme m orning tor |  |  |
| 239 | subject_text | +0.292806 | This whipped.lightwright moisturizing therapy visibly |  | ASAL L |
| 496 | subject_text | +0.288611 | smmooths wrinkkes creases & crepey skin texture as it or | This whipped,lightweight moisturizing therapy visibly |  |

## 同包装成功对照

`photo_41892aea27698341` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.076171 | +0.072699 |
| 只保留主体分差 | +0.022085 | +0.265455 |
| 主体 token 数 | 180 | 238 |

[成功对照 token 表](../photo_41892aea27698341/tokens.csv) · [对照摘要](../photo_41892aea27698341/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
