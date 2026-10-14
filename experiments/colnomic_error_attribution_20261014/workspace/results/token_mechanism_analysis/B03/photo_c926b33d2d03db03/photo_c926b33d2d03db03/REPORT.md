# B03 · photo_c926b33d2d03db03

按当前评测真值分析，未新增身份人工复核。

包装 `package_087`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_005035`，**纯图像实际 Top-1 错误参考** `reference_005156`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.577873 |
| 错误参考 MaxSim | +0.579481 |
| 正确减错误的分差 | -0.001608 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 242 个。正向贡献 +0.020239，负向贡献 -0.021847，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 526 | +0.000887 | +0.001314 |
| nonvisual | 11 | -0.000204 | -0.014432 |
| subject_nontext | 192 | -0.003160 | -0.012822 |
| subject_text | 50 | +0.000869 | +0.013537 |

最主要负差额来自 `subject_nontext`，贡献 -0.003160。只保留主体 query token 后，本对分差为 -0.007376。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 93 | 34.1% | True | +6.840065 | 71.3% |
| 错误 | 31 | 4.2% | False | 不可评估 | 不可评估 |

主体中 50 个 token 与 OCR 行相交，14 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 267 | subject_nontext | -0.340243 |  |  |  |
| 273 | subject_nontext | -0.272949 |  | Made In Indla |  |
| 18 | background | -0.214410 |  |  |  |
| 50 | background | -0.198355 |  | Suspension, USP |  |
| 694 | background | -0.152382 |  | 6586221860 |  |
| 728 | background | -0.150378 | 3 | prescribing Information. This bottle | 400 mg |
| 220 | background | -0.148502 |  | Rx only |  |
| 508 | subject_text | -0.143374 | UIZ | 6586221860 |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 376 | subject_nontext | +0.264039 |  |  |  |
| 191 | background | +0.229952 |  |  |  |
| 239 | background | +0.224939 |  |  | 279 Princeton-Hightstown Road |
| 366 | subject_text | +0.187289 | Aurobindo Pharma USA. Inc. |  | Aurobindo Pharma USA, Inc. |
| 263 | background | +0.184378 |  |  |  |
| 600 | subject_text | +0.181918 | Water |  |  |
| 236 | background | +0.169353 |  |  |  |
| 722 | background | +0.160543 |  | Code: TS/DRUGS/78/1996 |  |

## 同包装成功对照

`photo_af9240a1c073e28a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001608 | +0.128714 |
| 只保留主体分差 | -0.007376 | +0.186546 |
| 主体 token 数 | 242 | 288 |

[成功对照 token 表](../photo_af9240a1c073e28a/tokens.csv) · [对照摘要](../photo_af9240a1c073e28a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
