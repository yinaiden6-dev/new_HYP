# B03 · photo_16ff2e9baedfa1da

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_079`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003333`，**纯图像实际 Top-1 错误参考** `reference_002079`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 10 |
| 正确参考 MaxSim | +0.695801 |
| 错误参考 MaxSim | +0.707696 |
| 正确减错误的分差 | -0.011895 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 310 个。正向贡献 +0.038803，负向贡献 -0.050698，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 410 | -0.005430 | -0.009681 |
| nonvisual | 11 | -0.000532 | -0.035367 |
| subject_nontext | 84 | +0.000620 | +0.005397 |
| subject_text | 226 | -0.006553 | -0.021197 |

最主要负差额来自 `subject_text`，贡献 -0.006553。只保留主体 query token 后，本对分差为 -0.013991。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 585 | 65.5% | True | +5.861406 | 60.5% |
| 错误 | 5 | 25.1% | False | 不可评估 | 不可评估 |

主体中 226 个 token 与 OCR 行相交，162 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 335 | subject_text | -0.559247 | medical help or contact a Poison Control Center right away |  | apply liberally 15 minutes |
| 313 | subject_text | -0.542690 | medical help or contact a Poison Control Center right away |  | swallowed, get medical help or |
| 436 | subject_text | -0.496126 | limit time in the sun, especially from 10 a.m-2 p.m |  | 10 a.m. - 2 p.m. |
| 438 | subject_text | -0.495290 | limit time in the sun, especially from 10 a.m-2 p.m | sweating • Sun Protection Measures: Spending | 10 a.m. - 2 p.m. |
| 301 | background | -0.471037 |  | medical help or contact a Poison Control Center right away. | Keep out of reach of |
| 180 | background | -0.446268 |  | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (Ci 77491, CI | PEG-8, Cyclomethicone, |
| 366 | background | -0.427807 |  |  | sunscreen if swimming or |
| 362 | background | -0.416468 |  |  | sunscreen if swimming or |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 174 | subject_nontext | +0.544727 |  |  |  |
| 611 | subject_text | +0.541281 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 | children under 6 months: Ask |
| 48 | background | +0.514448 |  |  |  |
| 600 | background | +0.509373 |  | Questions or comments?1-855-353-7320 | Other information |
| 572 | subject_text | +0.487833 | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (C 77491. | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (Ci 77491, CI | Di-PPG-3 Myristyl Ether |
| 571 | subject_text | +0.460427 | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (C 77491. | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (Ci 77491, CI | Carica Papaya Extract, |
| 665 | background | +0.445351 |  |  |  |
| 612 | subject_text | +0.440340 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Octinoxate 6.0% | 7 | +0.000583 |
| Octisalate 4.0% | 4 | +0.000866 |
| Oxybenzone 3.5% | 8 | +0.001133 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_51f7a3bb218030b8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.011895 | +0.221243 |
| 只保留主体分差 | -0.013991 | +0.262020 |
| 主体 token 数 | 310 | 319 |

[成功对照 token 表](../photo_51f7a3bb218030b8/tokens.csv) · [对照摘要](../photo_51f7a3bb218030b8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
