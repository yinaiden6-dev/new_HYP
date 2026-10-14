# B03 · photo_158d2a9a0ee0d554

按当前评测真值分析，未新增身份人工复核。

包装 `package_045`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_004853`，**纯图像实际 Top-1 错误参考** `reference_004794`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 7 |
| 正确参考 MaxSim | +0.537883 |
| 错误参考 MaxSim | +0.546934 |
| 正确减错误的分差 | -0.009051 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 114 个。正向贡献 +0.028931，负向贡献 -0.037982，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 606 | -0.006764 | -0.008159 |
| nonvisual | 11 | -0.000004 | -0.000282 |
| subject_nontext | 59 | -0.001801 | -0.022319 |
| subject_text | 55 | -0.000482 | -0.006401 |

最主要负差额来自 `background`，贡献 -0.006764。只保留主体 query token 后，本对分差为 -0.014640。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 346 | 26.8% | True | +12.067099 | 80.7% |
| 错误 | 14 | 5.2% | False | 不可评估 | 不可评估 |

主体中 55 个 token 与 OCR 行相交，36 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 356 | background | -0.268598 |  |  |  |
| 273 | subject_text | -0.219842 | Store at 20° to 25℃ (68° to 77°F) [see USP Controlled Room Temperature]. |  |  |
| 97 | background | -0.206277 |  |  |  |
| 611 | subject_text | -0.202637 | Permetrin ream, 5% |  |  |
| 177 | background | -0.199908 |  |  |  |
| 153 | background | -0.197954 |  |  |  |
| 320 | background | -0.197201 |  | into the skin from the head to the soles of the feet. | QUINNOVA |
| 244 | background | -0.189044 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 474 | subject_text | +0.471572 | Dosage: Thoroughly massage Permethrin Cream, 5% |  |  |
| 468 | background | +0.404913 |  |  |  |
| 454 | subject_text | +0.384379 | into the skin from the head to the soles of the feet. |  |  |
| 488 | background | +0.323987 |  |  |  |
| 151 | background | +0.316793 |  |  |  |
| 455 | subject_nontext | +0.311293 |  |  |  |
| 459 | background | +0.301526 |  | Store at 20° to 25C (68° to 77°F) [see USP Controlled Room Temperature] |  |
| 158 | background | +0.276537 |  |  | Selenium sulfide, dim ethicone, ethylparaben, glyoerin, |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| pufied water, and sodium hydroxide. Formaldehyde 1 mg (0.1%) is added | 11 | -0.000248 |
| Active Ingredient: Each gram contains permethrin 50 mg (5%). | 10 | -0.000018 |
| Dosage: Thoroughly massage Permethrin Cream, 5% | 7 | +0.001251 |
| Permetrin ream, 5% | 2 | -0.000324 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_e96b3f54227d2e31` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009051 | +0.111446 |
| 只保留主体分差 | -0.014640 | +0.145520 |
| 主体 token 数 | 114 | 105 |

[成功对照 token 表](../photo_e96b3f54227d2e31/tokens.csv) · [对照摘要](../photo_e96b3f54227d2e31/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
