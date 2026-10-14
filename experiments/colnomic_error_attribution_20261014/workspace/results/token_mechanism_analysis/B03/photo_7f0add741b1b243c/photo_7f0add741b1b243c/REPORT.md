# B03 · photo_7f0add741b1b243c

按当前评测真值分析，未新增身份人工复核。

包装 `package_084`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_004975`，**纯图像实际 Top-1 错误参考** `reference_004974`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.521760 |
| 错误参考 MaxSim | +0.524079 |
| 正确减错误的分差 | -0.002320 |
| F03 的正确参考排名（辅助对照） | 4 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 144 个。正向贡献 +0.020224，负向贡献 -0.022544，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 624 | -0.008349 | -0.010423 |
| nonvisual | 11 | -0.000170 | -0.012038 |
| subject_nontext | 62 | +0.001524 | +0.019146 |
| subject_text | 82 | +0.004675 | +0.044417 |

最主要负差额来自 `background`，贡献 -0.008349。只保留主体 query token 后，本对分差为 +0.033536。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 51.6% | True | +15.626101 | 88.4% |
| 错误 | 22 | 4.6% | False | 不可评估 | 不可评估 |

主体中 82 个 token 与 OCR 行相交，28 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 101 | background | -0.449936 |  |  |  |
| 584 | background | -0.418141 | STERILE |  |  |
| 583 | background | -0.414183 | STERILE |  | Sterile, Nonpyrogenic |
| 35 | background | -0.291420 |  |  |  |
| 622 | background | -0.223182 | 30 |  |  |
| 419 | background | -0.219279 | Mase in inda |  |  |
| 374 | subject_text | -0.142374 | Preservative-Free | Preservative-Free | PRESERVATIVE FREE |
| 112 | background | -0.139844 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 342 | subject_text | +0.439103 | 10 x 10 mL Single-Dose Vials |  | 25 x 30 mL Single-Dose Vials |
| 310 | subject_text | +0.347774 | 10 x 10 mL Single-Dose Vials |  | 25 x 30 mL Single-Dose Vials |
| 630 | subject_nontext | +0.301236 |  |  |  |
| 423 | background | +0.279300 | 10ml/0.33floz | 10 x 10 mL Single-Dose Vials | 75 mg per 30 mL |
| 435 | subject_text | +0.267360 | 0.25% 25 mg/10 mL (2.5 mg/mL) | 0.25% 25 mg/10 mL (2.5 mg/mL) |  |
| 243 | subject_text | +0.219326 | 0.25% 25 mg/10 mL (2.5 mg/mL) |  |  |
| 275 | subject_text | +0.216208 | 0.25% 25 mg/10 mL (2.5 mg/mL) |  | 75 mg per 30 mL |
| 461 | background | +0.215829 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 0.25% 25 mg/10 mL (2.5 mg/mL) | 21 | +0.001289 |
| 10 x 10 mL Single-Dose Vials | 7 | +0.001165 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_709e8a3ee42539bc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002320 | +0.001374 |
| 只保留主体分差 | +0.033536 | +0.030093 |
| 主体 token 数 | 144 | 153 |

[成功对照 token 表](../photo_709e8a3ee42539bc/tokens.csv) · [对照摘要](../photo_709e8a3ee42539bc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
