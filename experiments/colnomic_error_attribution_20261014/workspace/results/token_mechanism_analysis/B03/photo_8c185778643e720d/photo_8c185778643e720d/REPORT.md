# B03 · photo_8c185778643e720d

按当前评测真值分析，未新增身份人工复核。

包装 `package_010`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000009`，**纯图像实际 Top-1 错误参考** `reference_000051`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.670221 |
| 错误参考 MaxSim | +0.672128 |
| 正确减错误的分差 | -0.001907 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 132 个。正向贡献 +0.016777，负向贡献 -0.018684，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 588 | -0.001971 | -0.002451 |
| nonvisual | 11 | +0.000025 | +0.001653 |
| subject_nontext | 70 | -0.000302 | -0.003154 |
| subject_text | 62 | +0.000341 | +0.004022 |

最主要负差额来自 `background`，贡献 -0.001971。只保留主体 query token 后，本对分差为 +0.000217。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 27 | 6.7% | False | 不可评估 | 不可评估 |
| 错误 | 25 | 6.8% | False | 不可评估 | 不可评估 |

主体中 62 个 token 与 OCR 行相交，51 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_nontext | -0.370230 |  |  |  |
| 357 | subject_nontext | -0.309669 |  |  |  |
| 614 | background | -0.192200 |  |  |  |
| 35 | background | -0.182849 |  | Zone lot et expiration |  |
| 372 | subject_nontext | -0.161724 |  |  |  |
| 494 | subject_text | -0.156188 | BROAD SPECTRUM PARASITICIDE | BROAD SPECTRUM PARASITICIDE |  |
| 673 | background | -0.153354 |  | (milbemycin oxime) | MilbeGuard |
| 343 | background | -0.149895 |  | Zone lot et expiration | pour position seulement |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 378 | subject_nontext | +0.422873 |  |  |  |
| 236 | background | +0.290777 |  |  | Zone lot et expiration |
| 327 | subject_nontext | +0.290132 |  |  |  |
| 517 | subject_nontext | +0.218441 |  |  |  |
| 449 | subject_text | +0.215873 | For dogs 26-50 Ibs. | For dogs 26-50 Ibs. | 26-50bs |
| 149 | background | +0.149163 |  | Recommended for cats | 26-50bs |
| 646 | background | +0.143917 |  | For dogs 26-50 Ibs. | 26-50bs |
| 582 | background | +0.133002 |  |  | 51-100bs |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Net contents: 6 tablets. 11.5 mg each | 5 | +0.000216 |
| 11mg | 2 | +0.000008 |
| For cats 6.1-12 lbs | 5 | +0.000204 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_841307b9e72b2c9d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001907 | +0.002730 |
| 只保留主体分差 | +0.000217 | +0.006111 |
| 主体 token 数 | 132 | 136 |

[成功对照 token 表](../photo_841307b9e72b2c9d/tokens.csv) · [对照摘要](../photo_841307b9e72b2c9d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
