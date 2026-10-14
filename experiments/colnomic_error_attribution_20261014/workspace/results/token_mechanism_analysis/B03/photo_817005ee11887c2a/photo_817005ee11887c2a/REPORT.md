# B03 · photo_817005ee11887c2a

按当前评测真值分析，未新增身份人工复核。

包装 `package_015`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000934`，**纯图像实际 Top-1 错误参考** `reference_004350`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 9 |
| 正确参考 MaxSim | +0.694740 |
| 错误参考 MaxSim | +0.734357 |
| 正确减错误的分差 | -0.039617 |
| F03 的正确参考排名（辅助对照） | 8 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 81 个。正向贡献 +0.017799，负向贡献 -0.057416，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 639 | -0.034112 | -0.039023 |
| nonvisual | 11 | +0.000113 | +0.007516 |
| subject_nontext | 31 | -0.000404 | -0.009527 |
| subject_text | 50 | -0.005214 | -0.076228 |

最主要负差额来自 `background`，贡献 -0.034112。只保留主体 query token 后，本对分差为 -0.050700。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 86 | 34.5% | True | +9.425407 | 87.5% |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 50 个 token 与 OCR 行相交，39 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 72 | background | -0.521548 |  |  |  |
| 431 | subject_text | -0.449141 | 咖 |  | For the temporary relief of minor arthritis pain |
| 132 | background | -0.446888 |  |  |  |
| 74 | background | -0.431682 |  |  |  |
| 242 | background | -0.425119 |  | :1+3/4 X 1+3/4 X 3+3/8 |  |
| 552 | background | -0.423397 |  |  |  |
| 121 | background | -0.413582 |  | promenetul sad merstusl crnape | For the temporary relief of minor arthritis pain |
| 394 | subject_text | -0.402961 | Pain Reliever/Fever Reducer | Pain Reliever/Fever Reducer |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 290 | subject_text | +0.321554 | 8 Hour | 8 Hour |  |
| 156 | background | +0.298894 |  |  |  |
| 294 | subject_nontext | +0.268853 |  |  |  |
| 316 | subject_nontext | +0.265707 |  |  |  |
| 698 | background | +0.265607 |  |  | do not take more than 6 geltabs |
| 437 | subject_nontext | +0.236868 |  |  |  |
| 170 | background | +0.233408 |  |  |  |
| 681 | background | +0.233023 |  |  | Lasts up to 8 hours |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Tablets USP, 650 mg | 7 | -0.001401 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_4917ad5f3fac9f55` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.039617 | +0.059583 |
| 只保留主体分差 | -0.050700 | +0.119657 |
| 主体 token 数 | 81 | 50 |

[成功对照 token 表](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
