# B03 · photo_5964b5e1bd6ef958

按当前评测真值分析，未新增身份人工复核。

包装 `package_081`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003368`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 657 |
| 正确参考 MaxSim | +0.476274 |
| 错误参考 MaxSim | +0.545671 |
| 正确减错误的分差 | -0.069397 |
| F03 的正确参考排名（辅助对照） | 18 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 156 个。正向贡献 +0.039705，负向贡献 -0.109102，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.066963 | -0.085236 |
| nonvisual | 11 | -0.001328 | -0.094073 |
| subject_nontext | 109 | -0.006270 | -0.044812 |
| subject_text | 47 | +0.005165 | +0.085606 |

最主要负差额来自 `background`，贡献 -0.066963。只保留主体 query token 后，本对分差为 -0.005519。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 53 | 23.3% | True | +10.342774 | 76.1% |
| 错误 | 9 | 2.9% | False | 不可评估 | 不可评估 |

主体中 47 个 token 与 OCR 行相交，19 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 693 | background | -0.643253 |  |  | 7 |
| 703 | background | -0.618148 |  |  | 7 |
| 745 | background | -0.614889 |  |  |  |
| 297 | background | -0.610800 | CVS |  | CVS |
| 685 | background | -0.610053 |  |  | 7 |
| 616 | background | -0.607698 |  |  | 7 |
| 709 | background | -0.605354 |  |  | 7 |
| 458 | background | -0.595653 | NASAL AL | TIIN cen | NASAL MIST |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 338 | subject_nontext | +0.519880 |  |  |  |
| 628 | subject_nontext | +0.485965 |  | ElizabethArden | temporarily relieves sinus congestion and pressure |
| 465 | subject_nontext | +0.464518 |  |  |  |
| 65 | background | +0.461112 |  |  |  |
| 536 | subject_text | +0.458512 | PLACE UPC HERE | PLACE UPC HERE |  |
| 309 | subject_nontext | +0.457546 |  |  | or down throat |
| 468 | subject_text | +0.450932 | ©Elizabeth Arden, Inc. | New York, NY 10003 | more often than every 10 to 12 hours. Do not exceed |
| 307 | subject_nontext | +0.450509 |  |  | or down throat |

## 同包装成功对照

`photo_9872ed71c7af7565` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.069397 | +0.043611 |
| 只保留主体分差 | -0.005519 | +0.176978 |
| 主体 token 数 | 156 | 225 |

[成功对照 token 表](../photo_9872ed71c7af7565/tokens.csv) · [对照摘要](../photo_9872ed71c7af7565/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
