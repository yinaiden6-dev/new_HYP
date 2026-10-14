# B03 · photo_15ce8c42c548e678

按当前评测真值分析，未新增身份人工复核。

包装 `package_047`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001139`，**纯图像实际 Top-1 错误参考** `reference_002438`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 14 |
| 正确参考 MaxSim | +0.602764 |
| 错误参考 MaxSim | +0.625514 |
| 正确减错误的分差 | -0.022750 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.029424，负向贡献 -0.052174，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.023355 | -0.026347 |
| nonvisual | 11 | +0.000101 | +0.006695 |
| subject_nontext | 21 | -0.000479 | -0.016665 |
| subject_text | 51 | +0.000983 | +0.014096 |

最主要负差额来自 `background`，贡献 -0.023355。只保留主体 query token 后，本对分差为 +0.005124。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 591 | 47.4% | True | +6.512260 | 73.6% |
| 错误 | 26 | 4.4% | False | 不可评估 | 不可评估 |

主体中 51 个 token 与 OCR 行相交，36 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 122 | background | -0.484345 |  | Do not use for children under 12 years of age |  |
| 252 | background | -0.455517 |  | Do not use for children under 12 years of age |  |
| 173 | background | -0.410922 |  | Do not use for children under 12 years of age |  |
| 135 | background | -0.399613 |  | GUARANTEE | OUR QUALITY |
| 306 | background | -0.391689 |  | Do not use for children under 12 years of age |  |
| 242 | background | -0.374314 |  | Do not use for children under 12 years of age |  |
| 302 | background | -0.341843 |  | Do not use for children under 12 years of age |  |
| 301 | background | -0.333964 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.478207 | BOTTLE | BOTTLE |  |
| 482 | background | +0.455354 |  | BOTTLE | colloidal silicon dioxide, magnesium starate,maltodextrin, |
| 164 | background | +0.454266 |  | BOTTLE |  |
| 196 | background | +0.404261 |  | this product can be administered without regard for |  |
| 167 | background | +0.393689 |  | BOTTLE | Keep out of reach of children. |
| 255 | background | +0.363656 |  | every 12 hours; not more than 2 tabletsin 24 hours | Chest |
| 670 | background | +0.362971 |  | BOTTLE | PID 334641 |
| 651 | background | +0.344073 |  |  | PAPER PLASTIC |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| every 12 hours; n ot more than 2 tablets in 24 hours | 5 | +0.000027 |
| each tablet contains: magnesium 25 mg | 4 | +0.000264 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_12ddea4ffa36f7a8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.022750 | +0.147505 |
| 只保留主体分差 | +0.005124 | +0.183796 |
| 主体 token 数 | 72 | 84 |

[成功对照 token 表](../photo_12ddea4ffa36f7a8/tokens.csv) · [对照摘要](../photo_12ddea4ffa36f7a8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
