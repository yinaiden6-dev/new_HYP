# B03 · photo_80f2fdc29081fedd

按当前评测真值分析，未新增身份人工复核。

包装 `package_065`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003233`，**纯图像实际 Top-1 错误参考** `reference_003666`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 297 |
| 正确参考 MaxSim | +0.562527 |
| 错误参考 MaxSim | +0.614916 |
| 正确减错误的分差 | -0.052390 |
| F03 的正确参考排名（辅助对照） | 22 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 88 个。正向贡献 +0.007883，负向贡献 -0.060272，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 632 | -0.045090 | -0.052153 |
| nonvisual | 11 | -0.000175 | -0.011646 |
| subject_nontext | 16 | -0.001401 | -0.063988 |
| subject_text | 72 | -0.005724 | -0.058116 |

最主要负差额来自 `background`，贡献 -0.045090。只保留主体 query token 后，本对分差为 -0.059184。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 356 | 45.8% | True | +14.392938 | 87.1% |
| 错误 | 3 | 0.0% | False | 不可评估 | 不可评估 |

主体中 72 个 token 与 OCR 行相交，64 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 58 | background | -0.316202 |  |  |  |
| 357 | subject_text | -0.285404 | i    |  | speechleg swellingpain getsworse or lasts |
| 60 | background | -0.283565 |  |  | rash blisters If an allergic reaction occurs, stop |
| 156 | background | -0.259659 |  |  |  |
| 249 | background | -0.246523 |  |  |  |
| 280 | background | -0.234169 |  |  |  |
| 386 | background | -0.233502 |  |  | better you have sym ptoms of heart problems or |
| 378 | subject_nontext | -0.231213 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 257 | background | +0.358066 |  |  | Qualityguaranteed oryourmoneyback. |
| 377 | subject_text | +0.311288 | i    |  |  |
| 190 | background | +0.269052 |  |  |  |
| 430 | subject_text | +0.204167 | stomach bleeding waming applies to you | Ask a doctor before use if |  |
| 397 | subject_text | +0.131369 | pam |  | DO NOT USE IF PRINTED FOL UNDER CAP |
| 552 | background | +0.125390 |  | you have high blood pressure, heart disease, liver | applies to you you have problems or serious side |
| 132 | background | +0.114171 |  |  |  |
| 214 | background | +0.108393 |  |  | rash blisters If an allergic reaction occurs, stop |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Aspirin 325 mg (NSAIO) | 2 | -0.000200 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_59498b6acfbfbf4f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.052390 | +0.107231 |
| 只保留主体分差 | -0.059184 | +0.145442 |
| 主体 token 数 | 88 | 78 |

[成功对照 token 表](../photo_59498b6acfbfbf4f/tokens.csv) · [对照摘要](../photo_59498b6acfbfbf4f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
