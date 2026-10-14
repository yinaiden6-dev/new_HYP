# B03 · photo_6063de8dd48b934a

按当前评测真值分析，未新增身份人工复核。

包装 `package_010`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000009`，**纯图像实际 Top-1 错误参考** `reference_000051`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.770433 |
| 错误参考 MaxSim | +0.775634 |
| 正确减错误的分差 | -0.005201 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 140 个。正向贡献 +0.023424，负向贡献 -0.028625，相加等于原图像分差。与原评分缓存的最大误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 580 | -0.005285 | -0.006661 |
| nonvisual | 11 | -0.000018 | -0.001194 |
| subject_nontext | 51 | -0.001252 | -0.017950 |
| subject_text | 89 | +0.001355 | +0.011125 |

最主要负差额来自 `background`，贡献 -0.005285。只保留主体 query token 后，本对分差为 +0.000534。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 489 | 54.8% | True | +5.557323 | 56.1% |
| 错误 | 420 | 53.4% | True | +6.500940 | 60.4% |

主体中 89 个 token 与 OCR 行相交，58 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 393 | subject_text | -0.413542 | To report adverse effects, access medical information, or to obtain | If you have any questions concerning the use of this product, contact | To report adverse effects, access medical information, or to obtain |
| 117 | background | -0.390779 |  | should be stored at room temperature, between 68 and 77 F (20 -25 C). | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 617 | background | -0.297187 |  |  | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 114 | background | -0.296776 |  |  | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 427 | background | -0.280225 |  |  | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 614 | background | -0.240838 |  |  | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 132 | background | -0.239488 |  | should be stored at room temperature, between 68 and 77 F (20 -25 C). |  |
| 312 | subject_nontext | -0.231334 |  | should be stored at room temperature, between 68 and 77 F (20 -25 C). |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 369 | background | +0.458111 |  | Place treatment reminder stickers on your calendar for each of the months | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 101 | background | +0.319523 |  |  |  |
| 289 | background | +0.287210 |  |  |  |
| 34 | background | +0.279097 |  |  | Endlosed insert describes use, directions and precautions to follow when |
| 390 | background | +0.230780 |  | additionall product information, ca 1-800-999-0297. | To report adverse effects, access medical information, or to obtain |
| 354 | subject_text | +0.221687 | Endosed insert describes use, directions and precautions to follow when | Endlosed insert describes use, directions and precautions to follow when | If you have any questions concerning the use of this product, contact |
| 352 | subject_text | +0.216705 | Endosed insert describes use, directions and precautions to follow when |  | Endlosed insert describes use, directions and precautions to follow when |
| 345 | background | +0.209100 |  |  | Endlosed insert describes use, directions and precautions to follow when |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 2-10 lbs | 1 | -0.000020 |
| 11-25lbs | 2 | -0.000051 |
| 12.1-25 lbs | 3 | -0.000093 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_8e714c29254155f0` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005201 | +0.010363 |
| 只保留主体分差 | +0.000534 | +0.012353 |
| 主体 token 数 | 140 | 130 |

[成功对照 token 表](../photo_8e714c29254155f0/tokens.csv) · [对照摘要](../photo_8e714c29254155f0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
