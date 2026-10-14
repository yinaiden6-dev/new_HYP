# B03 · photo_55b61213a0c8f1a9

本查询与相同正负图像已有身份歧义标记，不能直接作为模型失败证据。

包装 `package_079`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003333`，**纯图像实际 Top-1 错误参考** `reference_003332`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.684773 |
| 错误参考 MaxSim | +0.685523 |
| 正确减错误的分差 | -0.000750 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 同一候选对的身份歧义待复核 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 96 个。正向贡献 +0.012705，负向贡献 -0.013456，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 624 | -0.000585 | -0.000686 |
| nonvisual | 11 | -0.000044 | -0.002945 |
| subject_nontext | 12 | -0.000004 | -0.000271 |
| subject_text | 84 | -0.000116 | -0.001012 |

最主要负差额来自 `background`，贡献 -0.000585。只保留主体 query token 后，本对分差为 -0.000919。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 284 | 57.5% | True | +6.363790 | 62.5% |
| 错误 | 284 | 57.2% | True | +6.074802 | 61.5% |

主体中 84 个 token 与 OCR 行相交，66 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 517 | background | -0.206055 |  | Questions or comments?1-855-353-7320 |  |
| 438 | background | -0.147049 |  |  |  |
| 714 | background | -0.139815 |  |  | For external use only • Do not use on damaged |
| 196 | background | -0.134587 |  |  |  |
| 351 | subject_text | -0.133894 | medical help or contact a Poison Control Center right away |  |  |
| 150 | background | -0.124122 |  |  |  |
| 106 | background | -0.110819 |  |  |  |
| 484 | background | -0.110378 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 194 | background | +0.265166 |  |  |  |
| 512 | subject_text | +0.223741 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 |
| 181 | background | +0.214636 |  |  |  |
| 692 | background | +0.162039 |  |  |  |
| 496 | subject_text | +0.154539 | Glycol, Fragrance (Partum) May Contain: ron Oxodes C 77491 |  |  |
| 379 | background | +0.148143 |  |  |  |
| 715 | background | +0.145718 |  | For external use only • Do not use on damaged |  |
| 334 | subject_text | +0.143725 | or broken skin  When using this product keep | For external use only • Do not use on damaged | For external use only • Do not use on damaged |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Octinoxate 6.0% | 3 | +0.000004 |
| Octisalate 4.0%. | 3 | +0.000007 |
| Oxybenzone 3.5% | 3 | -0.000012 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_184d6f956513c5d8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000750 | +0.002232 |
| 只保留主体分差 | -0.000919 | +0.002171 |
| 主体 token 数 | 96 | 104 |

[成功对照 token 表](../photo_184d6f956513c5d8/tokens.csv) · [对照摘要](../photo_184d6f956513c5d8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
