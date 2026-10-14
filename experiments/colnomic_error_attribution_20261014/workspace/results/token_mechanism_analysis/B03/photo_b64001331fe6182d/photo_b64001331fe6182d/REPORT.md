# B03 · photo_b64001331fe6182d

按当前评测真值分析，未新增身份人工复核。

包装 `package_054`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001401`，**纯图像实际 Top-1 错误参考** `reference_001021`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.666551 |
| 错误参考 MaxSim | +0.684670 |
| 正确减错误的分差 | -0.018119 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 190 个。正向贡献 +0.037589，负向贡献 -0.055708，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.014553 | -0.020072 |
| nonvisual | 11 | +0.000387 | +0.025734 |
| subject_nontext | 96 | -0.002837 | -0.021601 |
| subject_text | 94 | -0.001116 | -0.008680 |

最主要负差额来自 `background`，贡献 -0.014553。只保留主体 query token 后，本对分差为 -0.015209。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 415 | 66.4% | True | +9.351061 | 82.4% |
| 错误 | 7 | 5.0% | False | 不可评估 | 不可评估 |

主体中 94 个 token 与 OCR 行相交，52 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 371 | subject_text | -0.509061 | You do not have to drink any of the |  | of the glass. |
| 276 | subject_text | -0.447739 | 24 |  | do not exceed 8 tablets in 24 hours |
| 290 | subject_text | -0.441286 | Other information • each tablet contains: scdum 567 mg | READ ALL WARNINGS AND DIRECTIONS BEFORE USE |  |
| 389 | subject_nontext | -0.430709 |  |  | are in the water. |
| 289 | subject_text | -0.411651 | Other information • each tablet contains: scdum 567 mg |  |  |
| 52 | background | -0.411069 |  |  |  |
| 118 | background | -0.403258 |  | Heartburn |  |
| 558 | background | -0.382435 |  | fully dissolve 2 tablets in 4 ounces of water before taking |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 508 | subject_text | +0.566162 | Made in | Made in | NDC 30142-421-68 |
| 710 | background | +0.476060 |  | 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) | DO NOT USE IF PRINTED PACKETS |
| 510 | subject_text | +0.472179 | 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) | 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) | DO NOT USE IF PRINTED PACKETS |
| 515 | subject_text | +0.420837 | 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) |  | (2) tablets in |
| 499 | background | +0.395846 |  | Made in |  |
| 489 | subject_text | +0.374978 | Lot | Made in |  |
| 16 | background | +0.330683 |  | 815556020033 | 36 EFFERVESCENT |
| 519 | background | +0.325499 |  | 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| I tackas eveny 4 502% 0 88 | 3 | -0.000042 |
| 2% | 2 | -0.000181 |
| Other information • each tablet contains: scdum 567 mg | 8 | -0.001476 |
| • Place two (2) tablets in 4 oz of | 5 | -0.000186 |
| 36 POUCHES OF 2 EFFERVESCENT TABLETS (72 TABLETS) | 9 | +0.001722 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_3431544b3be0914a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.018119 | +0.063075 |
| 只保留主体分差 | -0.015209 | +0.071579 |
| 主体 token 数 | 190 | 171 |

[成功对照 token 表](../photo_3431544b3be0914a/tokens.csv) · [对照摘要](../photo_3431544b3be0914a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
