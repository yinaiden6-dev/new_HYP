# B03 · photo_02fb3c06e1fb70e5

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 189 |
| 正确参考 MaxSim | +0.487882 |
| 错误参考 MaxSim | +0.575597 |
| 正确减错误的分差 | -0.087715 |
| F03 的正确参考排名（辅助对照） | 48 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 209 个。正向贡献 +0.044109，负向贡献 -0.131825，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 559 | -0.102431 | -0.142744 |
| nonvisual | 11 | -0.001600 | -0.113339 |
| subject_nontext | 164 | +0.007518 | +0.035711 |
| subject_text | 45 | +0.008798 | +0.152307 |

最主要负差额来自 `background`，贡献 -0.102431。只保留主体 query token 后，本对分差为 +0.060815。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 97 | 33.3% | True | +11.146492 | 79.2% |
| 错误 | 5 | 21.2% | False | 不可评估 | 不可评估 |

主体中 45 个 token 与 OCR 行相交，21 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 627 | background | -0.643976 |  |  | congestion relief |
| 713 | background | -0.627520 |  | 094 09 0036 R00 C-002262-01-008-0000 | 1 FL OZ (30 mL) |
| 619 | background | -0.606379 | congestion relief | Usereduces withdrawal symptoms, including nicotine | congestion relief |
| 600 | background | -0.597234 |  |  | 7 |
| 696 | background | -0.592110 |  | Lozenge | 7 |
| 555 | background | -0.590414 | 12 HOUR PUMP MIST |  |  |
| 631 | background | -0.590299 |  |  | 7 |
| 455 | background | -0.589283 | SAL L |  | NASAL MIST |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 399 | subject_text | +0.553704 | source where proof of age cannet be verified | source where proof of age cannot be verified | Shake well before use. Before using the first time, |
| 408 | subject_nontext | +0.527680 |  | source where proof of age cannot be verified |  |
| 439 | subject_nontext | +0.508045 |  | source where proof of age cannot be verified | SAFETY SEALED: Do not use If Imprinted seal on bottle |
| 461 | subject_text | +0.493667 | not for sale to those under 18 years of age | source where proof of age cannot be verified | No Drip |
| 494 | subject_text | +0.491673 | proof of age required |  | 1-800-shop-CVS |
| 664 | background | +0.483553 |  | source where proof of age cannot be verified | *CVS/pharmacy® Nasal Mist is not |
| 493 | subject_text | +0.479061 | not for sale to those under 18 years of age | source where proof of age cannot be verified | ASAL L |
| 573 | background | +0.469639 |  | ACTUAL SIZE | 050428588352 |

## 同包装成功对照

`photo_80eb2d50ed90738e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.087715 | +0.132694 |
| 只保留主体分差 | +0.060815 | +0.260755 |
| 主体 token 数 | 209 | 273 |

[成功对照 token 表](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
