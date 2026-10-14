# B03 · photo_a2635073f4550a08

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_002508`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 352 |
| 正确参考 MaxSim | +0.577506 |
| 错误参考 MaxSim | +0.609676 |
| 正确减错误的分差 | -0.032170 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 84 个。正向贡献 +0.022746，负向贡献 -0.054916，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 636 | -0.034452 | -0.039599 |
| nonvisual | 11 | -0.000608 | -0.040384 |
| subject_nontext | 31 | +0.001782 | +0.042012 |
| subject_text | 53 | +0.001109 | +0.015291 |

最主要负差额来自 `background`，贡献 -0.034452。只保留主体 query token 后，本对分差为 +0.025152。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 47.5% | True | +5.868239 | 72.8% |
| 错误 | 5 | 9.8% | False | 不可评估 | 不可评估 |

主体中 53 个 token 与 OCR 行相交，43 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 489 | background | -0.399531 |  | more than 4,000 mg of acetaminophen In 24 |  |
| 616 | background | -0.396985 |  | more than 4,000 mg of acetaminophen In 24 |  |
| 297 | subject_nontext | -0.391333 |  |  |  |
| 296 | subject_nontext | -0.374478 |  | 24 GELCAPS |  |
| 609 | background | -0.373656 |  |  |  |
| 538 | background | -0.368817 |  | more than 4,000 mg of acetaminophen In 24 |  |
| 664 | background | -0.367755 |  | more than 4,000 mg of acetaminophen In 24 |  |
| 498 | background | -0.364591 |  | more than 4,000 mg of acetaminophen In 24 |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 454 | subject_text | +0.569069 | MADE IN INDIA | MADE IN INDIA |  |
| 394 | subject_text | +0.499153 | Extra Strength TYLENOL® PM Rapid Release Gels |  | Kroger |
| 453 | subject_nontext | +0.485108 |  | MADE IN INDIA | 9 |
| 455 | subject_text | +0.456445 | MADE IN INDIA | MADE IN INDIA | Drug Facts (continued) |
| 493 | subject_nontext | +0.445717 |  |  |  |
| 395 | subject_text | +0.437495 | Extra Strength TYLENOL® PM Rapid Release Gels |  |  |
| 89 | background | +0.376289 |  | RAPID RELEASE | Sore throat warning: If sore throat is severe, persists for more than 2 days, is accompanied or followed by |
| 492 | subject_nontext | +0.351766 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 2 | -0.000294 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_375b67daeeeeff5f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.032170 | +0.095409 |
| 只保留主体分差 | +0.025152 | +0.179207 |
| 主体 token 数 | 84 | 66 |

[成功对照 token 表](../photo_375b67daeeeeff5f/tokens.csv) · [对照摘要](../photo_375b67daeeeeff5f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
