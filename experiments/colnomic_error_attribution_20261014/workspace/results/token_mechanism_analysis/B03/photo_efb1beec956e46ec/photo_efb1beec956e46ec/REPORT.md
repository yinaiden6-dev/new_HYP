# B03 · photo_efb1beec956e46ec

按当前评测真值分析，未新增身份人工复核。

包装 `package_116`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000047`，**纯图像实际 Top-1 错误参考** `reference_000050`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.634226 |
| 错误参考 MaxSim | +0.634662 |
| 正确减错误的分差 | -0.000436 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 270 个。正向贡献 +0.020686，负向贡献 -0.021121，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 498 | -0.002941 | -0.004601 |
| nonvisual | 11 | +0.000001 | +0.000082 |
| subject_nontext | 124 | -0.001044 | -0.006557 |
| subject_text | 146 | +0.003548 | +0.018932 |

最主要负差额来自 `background`，贡献 -0.002941。只保留主体 query token 后，本对分差为 +0.007226。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 270 | 55.7% | True | +6.431585 | 62.3% |
| 错误 | 260 | 54.9% | True | +5.574291 | 61.7% |

主体中 146 个 token 与 OCR 行相交，43 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 540 | subject_nontext | -0.308514 |  |  |  |
| 693 | subject_nontext | -0.284069 |  |  | (Taenia pisiformis). |
| 131 | background | -0.280634 | ACETAMOPHE250mg &BUPROEN (NSAD)125mgTABLET |  |  |
| 662 | subject_text | -0.261870 | la), Whipworms (Trichuris vulpis), and Tapeworms |  | worms(Ancylostoma caninum, Uncinaria stenocepha- |
| 48 | background | -0.247436 | 60mg per | 30lb |  |
| 663 | subject_text | -0.228485 | la), Whipworms (Trichuris vulpis), and Tapeworms | la), Whipworms (Trichuris vulpis), and Tapeworms | worms(Ancylostoma caninum, Uncinaria stenocepha- |
| 59 | background | -0.220438 |  | Active ingredients: Fenbendazole Granules | (fenbendazole) |
| 43 | background | -0.192771 | 160 mg | Active ingredients: Fenbendazole Granules |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 337 | subject_text | +0.498501 | Canine | Canine | Canine |
| 209 | background | +0.434135 |  | Canine |  |
| 591 | subject_text | +0.319874 | 30lb | 30lb | 40lb |
| 333 | subject_text | +0.316988 | Canine | Canine | Canine |
| 585 | background | +0.293905 |  | 30lb | 40lb |
| 592 | subject_text | +0.273209 | 30lb | 30lb |  |
| 50 | background | +0.252674 | 60mg per | Canine |  |
| 311 | subject_nontext | +0.223479 |  | Dewormer | Canine |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 22.2%(222 mg/g) | 4 | +0.000442 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_0be78636d1b29c9f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000436 | +0.008719 |
| 只保留主体分差 | +0.007226 | +0.018212 |
| 主体 token 数 | 270 | 378 |

[成功对照 token 表](../photo_0be78636d1b29c9f/tokens.csv) · [对照摘要](../photo_0be78636d1b29c9f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
