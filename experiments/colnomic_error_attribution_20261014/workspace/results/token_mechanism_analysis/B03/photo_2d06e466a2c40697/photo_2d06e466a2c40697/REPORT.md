# B03 · photo_2d06e466a2c40697

按当前评测真值分析，未新增身份人工复核。

包装 `package_083`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000221`，**纯图像实际 Top-1 错误参考** `reference_000219`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.684979 |
| 错误参考 MaxSim | +0.691710 |
| 正确减错误的分差 | -0.006731 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 128 个。正向贡献 +0.016792，负向贡献 -0.023524，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 592 | -0.004441 | -0.005484 |
| nonvisual | 11 | -0.000081 | -0.005389 |
| subject_nontext | 42 | -0.000466 | -0.008106 |
| subject_text | 86 | -0.001743 | -0.014819 |

最主要负差额来自 `background`，贡献 -0.004441。只保留主体 query token 后，本对分差为 -0.012616。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1193 | 60.1% | True | +7.022996 | 64.8% |
| 错误 | 1165 | 59.7% | True | +6.320117 | 61.9% |

主体中 86 个 token 与 OCR 行相交，72 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 314 | subject_text | -0.331617 | Dosage and Administration: Bravecto should be | Dosage and Administration: Bravecto should be | Dosage and Administration: Bravecto should be |
| 275 | subject_text | -0.292695 | for the treatment and prevention of flea infestations | For complete product information, |  |
| 315 | subject_text | -0.290429 | Dosage and Administration: Bravecto should be |  | Dosage and Administration: Bravecto should be |
| 695 | background | -0.273047 |  | READ PACKAGE INSERT FOR APPLICATION INSTRUCTIONS |  |
| 333 | subject_text | -0.271664 | americanum ticks. Treatment with Bravecto may begin |  |  |
| 716 | background | -0.240258 |  |  |  |
| 328 | background | -0.234875 |  |  |  |
| 419 | background | -0.209517 |  | soap and water immediately after use of the product. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 12 | background | +0.259382 |  |  |  |
| 96 | background | +0.216477 |  |  |  |
| 337 | subject_nontext | +0.193634 |  | americonum ticks. Treatment with Bravecto may begin |  |
| 35 | background | +0.184455 |  |  |  |
| 458 | background | +0.180646 |  |  | www.bravecto.com. For technical assistance or to report |
| 331 | subject_text | +0.160923 | americanum ticks. Treatment with Bravecto may begin | without interruption. |  |
| 71 | background | +0.155846 |  |  |  |
| 407 | background | +0.147060 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| to provide a minimum dose of 11.4 mg/ib (25 mg/kg) | 7 | -0.001267 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_613ba5f27d0c5cfe` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006731 | +0.010421 |
| 只保留主体分差 | -0.012616 | +0.025950 |
| 主体 token 数 | 128 | 112 |

[成功对照 token 表](../photo_613ba5f27d0c5cfe/tokens.csv) · [对照摘要](../photo_613ba5f27d0c5cfe/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
