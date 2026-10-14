# B03 · photo_bc5564b0e01eeb18

按当前评测真值分析，未新增身份人工复核。

包装 `package_085`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000229`，**纯图像实际 Top-1 错误参考** `reference_000233`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.541852 |
| 错误参考 MaxSim | +0.543848 |
| 正确减错误的分差 | -0.001996 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 136 个。正向贡献 +0.020258，负向贡献 -0.022254，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 632 | -0.001035 | -0.001275 |
| nonvisual | 11 | -0.000074 | -0.005235 |
| subject_nontext | 29 | -0.001024 | -0.027510 |
| subject_text | 107 | +0.000137 | +0.000995 |

最主要负差额来自 `background`，贡献 -0.001035。只保留主体 query token 后，本对分差为 -0.005083。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 189 | 58.0% | True | +6.711442 | 61.8% |
| 错误 | 55 | 33.3% | True | +7.353903 | 65.2% |

主体中 107 个 token 与 OCR 行相交，74 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 504 | subject_text | -0.408278 | East Brunswick, NJ 08816. | Contact No: 1-844-227-6687 |  |
| 443 | subject_text | -0.274143 | August 2024 | Code: 4206162/TS/DRUGS/2023 |  |
| 343 | subject_nontext | -0.248309 |  |  |  |
| 280 | subject_nontext | -0.240779 |  |  |  |
| 345 | subject_text | -0.233403 | Contact No:1-844-227-6687 |  | Code: 4206162/TS/DRUGS/2023 |
| 382 | background | -0.232950 | 50.1-100 lbs |  | 50 mL |
| 500 | subject_text | -0.213826 | more than 90 days after initial puncture |  |  |
| 474 | subject_text | -0.207852 | Code:4206162/TS/DRUGS/2023 |  | August 2024 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 354 | background | +0.403134 |  NG MANDG LLG |  |  |
| 270 | subject_text | +0.367741 | The Use of this Product in Cats may result |  | The Use of this Product in Cats may result in |
| 408 | subject_text | +0.361686 | East Brunswick, NJ 08816. |  |  |
| 173 | background | +0.320777 |  | puncture a maximum of 20 times. Any |  |
| 375 | subject_text | +0.300485 | Cronus Pharma LLC, |  |  |
| 367 | subject_text | +0.268055 | in Retinal Toxicity. Keep out of reach of | The Use of this Product in Cats may result |  |
| 274 | subject_text | +0.219809 | Use within 90 days of first puncture and |  | a maximum of 50 times. Any product remaining |
| 498 | subject_text | +0.212104 | (59-86°F). |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| complete details. | 4 | -0.000227 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_20594074f229bd8f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001996 | +0.001343 |
| 只保留主体分差 | -0.005083 | +0.000349 |
| 主体 token 数 | 136 | 99 |

[成功对照 token 表](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
