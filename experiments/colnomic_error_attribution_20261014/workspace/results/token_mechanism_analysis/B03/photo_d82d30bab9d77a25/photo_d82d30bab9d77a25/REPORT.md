# B03 · photo_d82d30bab9d77a25

按当前评测真值分析，未新增身份人工复核。

包装 `package_033`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001031`，**纯图像实际 Top-1 错误参考** `reference_004077`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.680635 |
| 错误参考 MaxSim | +0.700656 |
| 正确减错误的分差 | -0.020021 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 112 个。正向贡献 +0.033506，负向贡献 -0.053527，相加等于原图像分差。与原评分缓存的最大误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 608 | -0.025395 | -0.030533 |
| nonvisual | 11 | -0.000091 | -0.006035 |
| subject_nontext | 37 | +0.001429 | +0.028237 |
| subject_text | 75 | +0.004036 | +0.039339 |

最主要负差额来自 `background`，贡献 -0.025395。只保留主体 query token 后，本对分差为 +0.035671。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 229 | 54.8% | True | +4.958322 | 57.1% |
| 错误 | 50 | 11.5% | False | 不可评估 | 不可评估 |

主体中 75 个 token 与 OCR 行相交，64 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 403 | background | -0.534822 |  |  | narrow angle glaucoma. |
| 367 | background | -0.521478 |  |  | Tetrahydrozoline HCl 0.05%.Redness reliever |
| 607 | background | -0.520699 |  |  |  |
| 427 | background | -0.511250 |  | Children under 6 years of |  |
| 407 | background | -0.510615 |  | Uses for the relief of redness of the eyes due to | narrow angle glaucoma. |
| 613 | background | -0.509589 |  |  |  |
| 422 | background | -0.507588 |  |  |  |
| 457 | subject_text | -0.491326 | becomes cloudy to avoid contamination, do not | touch tip of container to any surface | surface to avoid contamination |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 536 | subject_text | +0.466483 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | do not touch tip of container to any |
| 535 | subject_text | +0.463585 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | Colirio Ocusan® works |
| 534 | subject_text | +0.428546 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | NW  3 |
| 454 | subject_text | +0.423918 | touch tip of container to any surface |  | ON THE BOTTLE IS BROKEN OR MISSING |
| 537 | subject_text | +0.410247 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | do not touch tip of container to any |
| 532 | subject_text | +0.409988 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | Drug Facts (continued) |
| 111 | background | +0.380120 |  | Ask a doctor before use if you have |  |
| 318 | subject_nontext | +0.367532 |  | DO NOT USE IF CARTONIS OPEN OR BLUE NECK |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Dextran 70 0.1% | 3 | -0.000180 |
| Polyethylene glycol 400 1% | 4 | -0.000189 |
| Povidone 1 %.. | 3 | -0.000201 |
| Tetrahydrozoline HCl 0.05% | 4 | -0.000086 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_3f61e3e16c427164` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.020021 | +0.032079 |
| 只保留主体分差 | +0.035671 | +0.066621 |
| 主体 token 数 | 112 | 108 |

[成功对照 token 表](../photo_3f61e3e16c427164/tokens.csv) · [对照摘要](../photo_3f61e3e16c427164/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
