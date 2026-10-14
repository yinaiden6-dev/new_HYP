# B03 · photo_a0dda9c59f0160fe

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_004360`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.579243 |
| 错误参考 MaxSim | +0.598058 |
| 正确减错误的分差 | -0.018815 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 276 个。正向贡献 +0.019214，负向贡献 -0.038029，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 492 | -0.011033 | -0.017468 |
| nonvisual | 11 | -0.000111 | -0.007880 |
| subject_nontext | 40 | -0.001486 | -0.028931 |
| subject_text | 236 | -0.006186 | -0.020419 |

最主要负差额来自 `background`，贡献 -0.011033。只保留主体 query token 后，本对分差为 -0.021652。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 227 | 68.2% | True | +7.787549 | 69.7% |
| 错误 | 10 | 30.4% | True | +17.524031 | 97.1% |

主体中 236 个 token 与 OCR 行相交，186 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 262 | background | -0.550775 | CVS | 73460 UW C3 | See inside for more details or |
| 230 | background | -0.512014 |  | 73460 UW C3 | See inside for more details or |
| 263 | background | -0.501586 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | See inside for more details or |
| 294 | background | -0.481856 | pharmacy | *This product is not manufactured or distrbuted by GlaxoSmithKline |  |
| 84 | background | -0.389603 |  |  |  |
| 535 | subject_text | -0.380534 | 6 hours. Do not use more than 20 lozenges per day.  it is |  | it is important to complete treatment. If you feel you |
| 270 | subject_nontext | -0.377742 |  |  |  |
| 252 | background | -0.364242 |  | USES, DIRECTIONS AND WARNINGS. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 599 | subject_text | +0.287084 | 6 hours. Do not use more than 20 lozenges per day.  it is |  | 1 lozenge every |
| 434 | subject_text | +0.266643 | tingling sensation occasionally move the lozenge from one | Do not chew or swallow lozenge. you may feel a warm or | you may feel a warm or tingling sensation |
| 506 | subject_text | +0.259983 | keep vial tightly closed and protect from light | Other informationstore at 20-25℃ (68-77°F) | do not use more than one lozenge at a time or continuously |
| 545 | background | +0.209876 |  |  |  |
| 396 | subject_text | +0.195612 | first cigarette more than 30 minutes after waking up, use | If you smoke your first cigarette WITHIN | do not eat or drink 15 minutes before using or while the |
| 284 | subject_nontext | +0.190418 |  |  | Active ingredient (in each lozenge) |
| 538 | subject_text | +0.189162 | keep vial tightly closed and protect from light | hon | you get symptoms of nicotine overdose such as nausea, vomiting, dizziness, diarrhea, weakness |
| 232 | background | +0.188231 | 24125514 | 094 09 0036 R00 C-002262-01-008-0000 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| waking up, use 4 mg nicotine lozenge if you smoke your | 22 | +0.000474 |
| 2 mg nicotine lozenge aocording to the following 12 week | 19 | -0.000163 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_80eb2d50ed90738e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.018815 | +0.045373 |
| 只保留主体分差 | -0.021652 | +0.094385 |
| 主体 token 数 | 276 | 273 |

[成功对照 token 表](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
