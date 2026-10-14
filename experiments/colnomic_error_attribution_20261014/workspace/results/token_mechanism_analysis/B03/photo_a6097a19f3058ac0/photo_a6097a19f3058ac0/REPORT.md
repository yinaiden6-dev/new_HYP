# B03 · photo_a6097a19f3058ac0

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_004359`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.601052 |
| 错误参考 MaxSim | +0.603045 |
| 正确减错误的分差 | -0.001993 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 135 个。正向贡献 +0.029804，负向贡献 -0.031796，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 633 | -0.006504 | -0.008004 |
| nonvisual | 11 | +0.000273 | +0.019319 |
| subject_nontext | 30 | -0.000543 | -0.014094 |
| subject_text | 105 | +0.004781 | +0.035473 |

最主要负差额来自 `background`，贡献 -0.006504。只保留主体 query token 后，本对分差为 +0.024458。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 257 | 52.6% | True | +3.848449 | 55.6% |
| 错误 | 24 | 10.4% | False | 不可评估 | 不可评估 |

主体中 105 个 token 与 OCR 行相交，73 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | background | -0.439242 | Pharmacy |  |  |
| 383 | background | -0.380023 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 352 | background | -0.351263 | Pharmacy | If you are pregnant or breast-feeding, only use this medicine |  |
| 675 | background | -0.323954 | 1020 Vienna |  |  |
| 479 | background | -0.318846 |  | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 384 | background | -0.312309 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 447 | background | -0.309173 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 446 | background | -0.308413 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 536 | subject_text | +0.515643 | R | up&up |  |
| 472 | subject_text | +0.435891 | R | up&up |  |
| 504 | subject_text | +0.435074 | R | up&up |  |
| 514 | background | +0.384217 |  | up&up |  |
| 362 | background | +0.340432 | MAGNES | Inactive ingredients acesulfame potassium, calcium |  |
| 630 | background | +0.259402 |  | R |  |
| 361 | background | +0.257208 | CALCIU |  | Woonsocket, RI 02895 |
| 482 | background | +0.253623 |  | 370030275458 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 2 mg (Nicotine) | 9 | -0.000134 |
| Polacrllex Lozenge, 4 mg | 4 | -0.000061 |
| 2mg | 2 | -0.000054 |
| 20 LOZENGES, 2 mg EACH | 7 | +0.000367 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_d2377339b5c178d4` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001993 | +0.006482 |
| 只保留主体分差 | +0.024458 | +0.030441 |
| 主体 token 数 | 135 | 135 |

[成功对照 token 表](../photo_d2377339b5c178d4/tokens.csv) · [对照摘要](../photo_d2377339b5c178d4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
