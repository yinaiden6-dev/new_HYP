# B03 · photo_759614539794e4a6

按当前评测真值分析，未新增身份人工复核。

包装 `package_047`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001139`，**纯图像实际 Top-1 错误参考** `reference_004450`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 215 |
| 正确参考 MaxSim | +0.579680 |
| 错误参考 MaxSim | +0.651142 |
| 正确减错误的分差 | -0.071462 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 200 个。正向贡献 +0.030163，负向贡献 -0.101625，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 520 | -0.051024 | -0.071728 |
| nonvisual | 11 | -0.000222 | -0.014721 |
| subject_nontext | 66 | -0.005013 | -0.055518 |
| subject_text | 134 | -0.015204 | -0.082942 |

最主要负差额来自 `background`，贡献 -0.051024。只保留主体 query token 后，本对分差为 -0.073892。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 688 | 67.5% | True | +9.219427 | 71.9% |
| 错误 | 57 | 5.3% | False | 不可评估 | 不可评估 |

主体中 134 个 token 与 OCR 行相交，103 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 392 | subject_text | -0.546644 | This product is not manufact ured or distr buted by Reckitt | PARENTS | *This product is not manufactured or |
| 248 | subject_text | -0.531193 | adults and children 12 years and older: 1 tablet | PARENTS |  |
| 555 | subject_nontext | -0.506491 |  |  |  |
| 595 | background | -0.481888 |  | *This product is not manufactured or distributed by Reckitt | SARL, owner of the registered trademark |
| 381 | background | -0.476096 |  | stea rate, microcrystalline cellulose, silicon dioxide | Questions or comments? |
| 629 | background | -0.470065 |  | Other information |  |
| 391 | subject_text | -0.466922 | This product is not manufact ured or distr buted by Reckitt |  | *This product is not manufactured or |
| 414 | subject_text | -0.463101 | Benckiser LLC, owner of the registered trademark Mucinex. | *This product is not manufactured or distributed by Reckitt | SARL, owner of the registered trademark |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 425 | background | +0.493192 |  | PARENTS | SARL, owner of the registered trademark |
| 461 | background | +0.424174 |  | www.meijer.com |  |
| 416 | subject_text | +0.386236 | Benckiser LLC, owner of the registered trademark Mucinex. | COMPARETOMAXIMUM | stearate, medium chain triglycerides, |
| 453 | subject_text | +0.375729 | www.meijer.com | meijer. | undesired reaction or side effect, |
| 653 | background | +0.368450 |  | meijer. | Or we'll replace it or give you |
| 716 | background | +0.361902 |  | meijer. | undesired reaction or side effect, |
| 535 | subject_text | +0.355429 | www.mejer.com/ | meijer. | Or we'll replace it or give you |
| 235 | subject_text | +0.350041 | this product can be administ ered without regard for | this product can be administered without regard for |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| every 12 hours; not more than 2 tablets in 24 hours | 13 | -0.001544 |
| each tablet contains: magnesium 25 mg | 8 | -0.002102 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_02c11dd14bf4116f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.071462 | +0.248502 |
| 只保留主体分差 | -0.073892 | +0.288708 |
| 主体 token 数 | 200 | 220 |

[成功对照 token 表](../photo_02c11dd14bf4116f/tokens.csv) · [对照摘要](../photo_02c11dd14bf4116f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
