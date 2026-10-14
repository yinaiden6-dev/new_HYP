# B03 · photo_85e17d28180026cd

按当前评测真值分析，未新增身份人工复核。

包装 `package_092`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000304`，**纯图像实际 Top-1 错误参考** `reference_000471`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.541170 |
| 错误参考 MaxSim | +0.544795 |
| 正确减错误的分差 | -0.003624 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 56 个。正向贡献 +0.029718，负向贡献 -0.033342，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 664 | -0.004987 | -0.005490 |
| nonvisual | 11 | -0.000344 | -0.022838 |
| subject_nontext | 20 | -0.000175 | -0.006412 |
| subject_text | 36 | +0.001882 | +0.038216 |

最主要负差额来自 `background`，贡献 -0.004987。只保留主体 query token 后，本对分差为 +0.022277。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 502 | 44.0% | True | +9.888396 | 77.1% |
| 错误 | 4 | 5.5% | False | 不可评估 | 不可评估 |

主体中 36 个 token 与 OCR 行相交，32 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 92 | background | -0.347603 |  |  |  |
| 401 | background | -0.258073 |  | For Large Dogs |  |
| 592 | background | -0.246067 |  |  |  |
| 405 | background | -0.228465 |  | (praziquantel/pyrantel pamoate/febantel) |  |
| 88 | background | -0.210586 |  |  |  |
| 722 | background | -0.203365 |  |  |  |
| 225 | background | -0.196959 |  |  |  |
| 526 | background | -0.193761 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 219 | background | +0.551860 |  |  | Revalor-XS |
| 453 | subject_text | +0.514426 | & Veterinary Support at 888-545-5973 |  |  |
| 332 | subject_text | +0.512482 | Body Wt. (lbs.) |  |  |
| 307 | background | +0.449838 |  |  |  |
| 303 | background | +0.446697 |  |  |  |
| 462 | background | +0.439179 |  |  |  |
| 334 | subject_text | +0.397200 | Drontal Plus Taste Tabs Tablets for |  |  |
| 39 | background | +0.379496 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| contains: 136.0 mg praziquantell 136,0 mg | 3 | -0.000260 |
| further details. | 2 | +0.000251 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_f0ed0b2bc867b9df` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003624 | +0.135493 |
| 只保留主体分差 | +0.022277 | +0.158538 |
| 主体 token 数 | 56 | 56 |

[成功对照 token 表](../photo_f0ed0b2bc867b9df/tokens.csv) · [对照摘要](../photo_f0ed0b2bc867b9df/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
