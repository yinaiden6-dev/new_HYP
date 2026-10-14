# B03 · photo_4d9a6959fb508a57

按当前评测真值分析，未新增身份人工复核。

包装 `package_017`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000936`，**纯图像实际 Top-1 错误参考** `reference_004358`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.792161 |
| 错误参考 MaxSim | +0.793717 |
| 正确减错误的分差 | -0.001556 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 384 个。正向贡献 +0.026093，负向贡献 -0.027649，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 336 | -0.001953 | -0.004249 |
| nonvisual | 11 | -0.000080 | -0.005334 |
| subject_nontext | 223 | -0.000546 | -0.001790 |
| subject_text | 161 | +0.001024 | +0.004648 |

最主要负差额来自 `background`，贡献 -0.001953。只保留主体 query token 后，本对分差为 +0.000909。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 475 | 67.5% | True | +5.346841 | 60.4% |
| 错误 | 99 | 25.8% | True | +5.674855 | 69.1% |

主体中 161 个 token 与 OCR 行相交，60 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 257 | subject_nontext | -0.327835 |  |  |  |
| 11 | background | -0.315887 |  |  |  |
| 301 | subject_nontext | -0.292574 |  |  |  |
| 12 | background | -0.264565 |  |  |  |
| 282 | subject_nontext | -0.246264 |  | Active ingredient (in each extended-release bHayar tablet) |  |
| 143 | background | -0.244442 |  | Compare to the active | 12 |
| 390 | subject_text | -0.210462 | Immedlate & extended release |  | • Immediate & extended-release |
| 300 | subject_nontext | -0.198594 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 630 | background | +0.587202 |  | 28 (2 X 14) EXTENDED-RELEASE | 4S155 17 C2 |
| 570 | subject_text | +0.584103 | 28 (2 X 14) EXTENDED-RELEASE | 28 (2 X 14) EXTENDED-RELEASE | 4S155 17 C2 |
| 550 | subject_text | +0.497768 | 28 (2 X 14) EXTENDED-RELEASE | 28 (2 X 14) EXTENDED-RELEASE | 4S155 17 C2 |
| 568 | subject_text | +0.488621 | 28 (2 X 14) EXTENDED-RELEASE | 28 (2 X 14) EXTENDED-RELEASE | MAXIMUM STRENGTH |
| 628 | background | +0.458787 |  |  |  |
| 548 | subject_text | +0.446514 | 28 (2 X 14) EXTENDED-RELEASE |  |  |
| 569 | subject_text | +0.431854 | 28 (2 X 14) EXTENDED-RELEASE |  | MAXIMUM STRENGTH |
| 580 | subject_nontext | +0.429930 |  | 28 (2 X 14) EXTENDED-RELEASE |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| TABLETS, 1200 mg | 7 | -0.000509 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_f0d980d139f16444` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001556 | +0.016429 |
| 只保留主体分差 | +0.000909 | +0.028860 |
| 主体 token 数 | 384 | 384 |

[成功对照 token 表](../photo_f0d980d139f16444/tokens.csv) · [对照摘要](../photo_f0d980d139f16444/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
