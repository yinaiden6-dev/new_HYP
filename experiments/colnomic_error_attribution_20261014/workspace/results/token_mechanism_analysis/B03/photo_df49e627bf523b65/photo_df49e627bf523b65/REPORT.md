# B03 · photo_df49e627bf523b65

按当前评测真值分析，未新增身份人工复核。

包装 `package_113`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000157`，**纯图像实际 Top-1 错误参考** `reference_000150`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.566440 |
| 错误参考 MaxSim | +0.569017 |
| 正确减错误的分差 | -0.002577 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 60 个。正向贡献 +0.024800，负向贡献 -0.027377，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 660 | -0.001908 | -0.002113 |
| nonvisual | 11 | -0.000045 | -0.002992 |
| subject_nontext | 18 | -0.000765 | -0.031060 |
| subject_text | 42 | +0.000140 | +0.002443 |

最主要负差额来自 `background`，贡献 -0.001908。只保留主体 query token 后，本对分差为 -0.007608。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 588 | 43.6% | True | +13.179481 | 83.9% |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 42 个 token 与 OCR 行相交，36 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 432 | subject_text | -0.278120 | 29130  06 |  | DIRECTIONS |
| 530 | background | -0.273723 |  | *See package onsert for dosing instructionsfor other |  |
| 495 | subject_nontext | -0.222917 |  |  |  |
| 305 | background | -0.217627 |  |  |  |
| 539 | background | -0.209839 |  | *See package onsert for dosing instructionsfor other |  |
| 548 | background | -0.204574 |  |  |  |
| 364 | background | -0.197361 |  |  |  |
| 72 | background | -0.195112 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 410 | background | +0.406149 |  | TAKE TIME |  |
| 439 | background | +0.389251 |  | www.VetOne.net |  |
| 412 | subject_text | +0.376173 | TAKE TIME | TAKE TIME |  |
| 436 | background | +0.369065 |  | www.VetOne.net |  |
| 258 | background | +0.335750 |  |  | of drug per Ib of body weight per day, administering more |
| 371 | subject_text | +0.295688 | the recommended number of treatments, and/or |  | of drug per Ib of body weight per day, administering more |
| 611 | background | +0.224273 |  | TAKE TIME |  |
| 215 | background | +0.210416 |  | STORAGE: Store at room temperature 15º-25C | Refer to Package Onsert for Complete Directions. |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| antibiotic oxytetracycline Each mL contain3 200 mg | 5 | -0.000093 |
| and on a wAu basis, 40.0% 2-pyrrolidone, 5.0% | 4 | -0.000058 |
| 62-pyrrolidone, 5.0% | 3 | +0.000014 |
| 1.8% magnesium oxide. | 4 | -0.000245 |
| 0 2% sodium | 1 | +0.000042 |
| exceeding 10 mL subcutaneously per injection site in | 5 | +0.000266 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_145709484be1d522` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002577 | +0.067790 |
| 只保留主体分差 | -0.007608 | +0.042408 |
| 主体 token 数 | 60 | 60 |

[成功对照 token 表](../photo_145709484be1d522/tokens.csv) · [对照摘要](../photo_145709484be1d522/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
