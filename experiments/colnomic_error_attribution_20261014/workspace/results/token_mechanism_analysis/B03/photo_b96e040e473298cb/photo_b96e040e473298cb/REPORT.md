# B03 · photo_b96e040e473298cb

按当前评测真值分析，未新增身份人工复核。

包装 `package_050`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003087`，**纯图像实际 Top-1 错误参考** `reference_000822`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.636424 |
| 错误参考 MaxSim | +0.640633 |
| 正确减错误的分差 | -0.004209 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 126 个。正向贡献 +0.029440，负向贡献 -0.033650，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 594 | -0.004533 | -0.005578 |
| nonvisual | 11 | +0.000064 | +0.004222 |
| subject_nontext | 65 | -0.001661 | -0.018682 |
| subject_text | 61 | +0.001921 | +0.023018 |

最主要负差额来自 `background`，贡献 -0.004533。只保留主体 query token 后，本对分差为 +0.001506。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 287 | 36.1% | True | +7.210629 | 72.4% |
| 错误 | 91 | 5.5% | False | 不可评估 | 不可评估 |

主体中 61 个 token 与 OCR 行相交，44 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 472 | subject_nontext | -0.407401 |  |  |  |
| 91 | background | -0.401436 |  | VERSIÓN EN ESPAÑOL EN EL INTERIOR DE LA CAJA |  |
| 52 | background | -0.386448 |  |  |  |
| 61 | background | -0.384158 |  | Inactive ingredients |  |
| 471 | subject_text | -0.354292 | FDA Registered Facility | FDA Registered Facility | FDA Registered Facility |
| 74 | background | -0.335779 |  | Active ingredients | Net Contents 4 FL OZ / 118 mL |
| 461 | background | -0.320993 |  | FDA Registered Facility | FDA Registered Facility |
| 473 | subject_nontext | -0.313018 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 254 | subject_text | +0.319608 | If pregnant or breast-feeding, ask a health professional |  |  |
| 353 | subject_text | +0.256021 | store between 20-25°C(68-77°F) | store between 20-25°℃(68-77°F) | Brosolvan |
| 248 | background | +0.244542 |  |  |  |
| 273 | subject_text | +0.224392 | Center right away. Quick medical attention is critical |  |  |
| 226 | background | +0.222158 |  |  | Chest congestion / Mucus |
| 576 | background | +0.216596 |  |  | Contenido neto 118 mL / 4 FL OZ |
| 566 | background | +0.208823 |  |  |  |
| 165 | background | +0.208164 |  | Drug Facts |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 20 mL in dosing cup | 2 | +0.000203 |
| each 20 mL contains: | 2 | -0.000031 |
| sodium 20 mg | 2 | -0.000031 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_8bd2c2ece5146d67` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004209 | +0.183034 |
| 只保留主体分差 | +0.001506 | +0.256081 |
| 主体 token 数 | 126 | 130 |

[成功对照 token 表](../photo_8bd2c2ece5146d67/tokens.csv) · [对照摘要](../photo_8bd2c2ece5146d67/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
