# B03 · photo_19aa4d08f51deacc

按当前评测真值分析，未新增身份人工复核。

包装 `package_068`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001359`，**纯图像实际 Top-1 错误参考** `reference_003785`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 8 |
| 正确参考 MaxSim | +0.686495 |
| 错误参考 MaxSim | +0.711477 |
| 正确减错误的分差 | -0.024982 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 243 个。正向贡献 +0.040713，负向贡献 -0.065695，相加等于原图像分差。与原评分缓存的最大误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 477 | -0.022922 | -0.035128 |
| nonvisual | 11 | +0.000286 | +0.019029 |
| subject_nontext | 78 | +0.003316 | +0.031076 |
| subject_text | 165 | -0.005662 | -0.025086 |

最主要负差额来自 `background`，贡献 -0.022922。只保留主体 query token 后，本对分差为 -0.007058。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1186 | 70.0% | True | +8.221926 | 65.3% |
| 错误 | 9 | 14.3% | False | 不可评估 | 不可评估 |

主体中 165 个 token 与 OCR 行相交，121 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 502 | background | -0.674235 |  |  | Store at room temperature 68°F to 77°F (20° to 25°C) |
| 508 | background | -0.603502 |  |  | Store at room temperature 68°F to 77°F (20° to 25°C) |
| 694 | background | -0.598495 |  |  | Store at room temperature 68°F to 77°F (20° to 25°C) |
| 431 | subject_text | -0.566854 | a Poison Control right away. |  | If swallowed, get medical help or contact a Poison Control |
| 402 | background | -0.553758 |  |  | again within a few days, excessive skin irritation occurs. |
| 693 | background | -0.544884 |  |  | Store at room temperature 68°F to 77°F (20° to 25°C) |
| 225 | background | -0.510092 |  |  | WARNINGS: For external use only! |
| 503 | background | -0.509877 |  |  | Store at room temperature 68°F to 77°F (20° to 25°C) |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 582 | background | +0.612580 |  | AVON |  |
| 586 | background | +0.515783 |  |  | Alpha-Tocopherol Acetate (Vitamin E), Aminomethylpropanol, |
| 451 | subject_text | +0.506278 | Directionsuses only as directed | Directionsuses only as directed | Children under 12 years of age: ask a doctor before use. |
| 592 | subject_nontext | +0.485598 |  |  |  |
| 71 | background | +0.463094 |  |  |  |
| 452 | subject_text | +0.455445 | Directionsuses only as directed | Directionsuses only as directed | Children under 12 years of age: ask a doctor before use. |
| 576 | subject_nontext | +0.442708 |  |  |  |
| 571 | subject_text | +0.434088 | Call toll free 1-800-FOR-AVON |  | For questions call: +1 (239) 236-8158 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Camphor 4 % | 5 | +0.000208 |
| Menthol 10% | 3 | -0.000520 |
| Methyl salicylate 30 % | 4 | -0.000783 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_049efdffe7a60987` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.024982 | +0.174451 |
| 只保留主体分差 | -0.007058 | +0.216241 |
| 主体 token 数 | 243 | 286 |

[成功对照 token 表](../photo_049efdffe7a60987/tokens.csv) · [对照摘要](../photo_049efdffe7a60987/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
