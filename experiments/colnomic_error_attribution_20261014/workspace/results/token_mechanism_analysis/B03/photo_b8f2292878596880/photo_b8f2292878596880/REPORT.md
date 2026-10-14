# B03 · photo_b8f2292878596880

按当前评测真值分析，未新增身份人工复核。

包装 `package_063`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003207`，**纯图像实际 Top-1 错误参考** `reference_003438`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.615017 |
| 错误参考 MaxSim | +0.620220 |
| 正确减错误的分差 | -0.005203 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 210 个。正向贡献 +0.044965，负向贡献 -0.050168，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 510 | -0.013126 | -0.018814 |
| nonvisual | 11 | -0.000096 | -0.006362 |
| subject_nontext | 40 | +0.001537 | +0.028097 |
| subject_text | 170 | +0.006482 | +0.027872 |

最主要负差额来自 `background`，贡献 -0.013126。只保留主体 query token 后，本对分差为 +0.027915。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 503 | 65.1% | True | +2.441439 | 47.4% |
| 错误 | 7 | 15.5% | False | 不可评估 | 不可评估 |

主体中 170 个 token 与 OCR 行相交，118 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 46 | background | -0.497665 |  | Moist Moisture Cream |  |
| 228 | subject_text | -0.438245 | advised to stop using it immediately. If you keep using it, the symptoms will get |  |  |
| 327 | subject_nontext | -0.421579 |  | Active Ingredient; Dimethicone 1% | Inactive Ingredients |
| 106 | background | -0.388472 |  |  |  |
| 265 | background | -0.388203 |  |  |  |
| 422 | background | -0.382846 |  | Active Ingredient; Dimethicone 1% |  |
| 80 | background | -0.372590 |  |  |  |
| 329 | subject_text | -0.370510 | iractive ingredients: | Active Ingredient; Dimethicone 1% | Inactive Ingredients |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 658 | background | +0.564964 |  |  |  |
| 421 | subject_nontext | +0.557615 |  | 100 | (Rosemary) Leaf Extract, Sericin, Aesculus Hippocastanum (Horse Chestnut) |
| 538 | background | +0.537585 |  |  |  |
| 119 | background | +0.510998 |  |  |  |
| 601 | background | +0.501923 |  | Polyisobutene Propylparaben Sodium Polyacrylate Sodium Hyaluronate Trisodium EDTA |  |
| 420 | subject_text | +0.488390 | Polyiscoulene Propylbaraben Sodium Polyacryale Sodum Hyaluronate Trisodium EDTA | Polyisobutene Propylparaben Sodium Polyacrylate Sodium Hyaluronate Trisodium EDTA |  |
| 529 | background | +0.488015 |  |  |  |
| 317 | subject_text | +0.459402 | [Keep out of reach of Children] Keep out of reach of babies and children |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Active Ingredient: Dimethicone 1% | 12 | -0.001922 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_43eca55aded800ef` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005203 | +0.078525 |
| 只保留主体分差 | +0.027915 | +0.091871 |
| 主体 token 数 | 210 | 224 |

[成功对照 token 表](../photo_43eca55aded800ef/tokens.csv) · [对照摘要](../photo_43eca55aded800ef/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
