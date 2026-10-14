# B03 · photo_01a2f5c5bef357c2

按当前评测真值分析，未新增身份人工复核。

包装 `package_010`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000009`，**纯图像实际 Top-1 错误参考** `reference_000051`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.774305 |
| 错误参考 MaxSim | +0.787015 |
| 正确减错误的分差 | -0.012711 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 330 个。正向贡献 +0.023654，负向贡献 -0.036365，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 390 | -0.004882 | -0.009151 |
| nonvisual | 11 | -0.000101 | -0.006711 |
| subject_nontext | 134 | -0.000859 | -0.004687 |
| subject_text | 196 | -0.006868 | -0.025615 |

最主要负差额来自 `subject_text`，贡献 -0.006868。只保留主体 query token 后，本对分差为 -0.017117。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 605 | 58.3% | True | +5.740839 | 64.2% |
| 错误 | 553 | 58.1% | True | +6.352719 | 61.0% |

主体中 196 个 token 与 OCR 行相交，101 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 416 | subject_text | -0.456096 | Storage Conditions: MILBEGUARD• (milbemycin oxime) Flavored Tablets | should be stored at room temperature, between 68 and 77 F (20 -25 C). | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |
| 350 | subject_text | -0.440328 | To report adverse effects, access medical information, or to obtain | If you have any questions concerning the use of this product, contact | To report adverse effects, access medical information, or to obtain |
| 349 | subject_text | -0.438287 | To report adverse effects, access medical information, or to obtain |  | To report adverse effects, access medical information, or to obtain |
| 311 | subject_text | -0.415767 | Place treatment reminder stickers on your calendar for each of the months |  | administering MILBEGUARD* (milbemycin oxime) Fllavored Tablets. |
| 304 | background | -0.393015 |  | If you have any questions concerning the use of this product, contact | administering MILBEGUARD* (milbemycin oxime) Fllavored Tablets. |
| 489 | subject_text | -0.378858 | 2-10lbs | Recommended for dogs | 2-10bs |
| 313 | subject_text | -0.367752 | Place treatment reminder stickers on your calendar for each of the months | Place treatment reminder stickers on your calendar for each of the months | administering MILBEGUARD* (milbemycin oxime) Fllavored Tablets. |
| 411 | subject_text | -0.339412 | Storage Conditions: MILBEGUARD• (milbemycin oxime) Flavored Tablets |  | Storage Conditions: MILBEGUARD* (milbemycin oxime) Fllavored Tablets |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 384 | background | +0.333711 |  |  | MilbeGuard" is a registered trademark of Ceve Santé Animale S.A. |
| 92 | background | +0.309696 |  |  | and puppies 11 - 25 lbs. |
| 570 | background | +0.282256 |  |  | MilbeGuard" is a registered trademark of Ceve Santé Animale S.A. |
| 285 | background | +0.269449 |  | Endlosed insert describes use, directions and precautions to follow when | If you have any questions concerning the use of this product, contact |
| 627 | background | +0.264517 |  | location out of reach of dogs, cats, and other animals to prevent accidental | location out of reach of dogs, cats, and other animals to prevent accidental |
| 628 | background | +0.263432 |  | location out of reach of dogs, cats, and other animals to prevent accidental | location out of reach of dogs, cats, and other animals to prevent accidental |
| 364 | background | +0.251748 |  | To report adverse effects, access medical information, or to obtain | To report adverse effects, access medical information, or to obtain |
| 95 | background | +0.239280 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 2-10lbs | 3 | -0.000633 |
| 11-25lbs | 3 | -0.000194 |
| 12.1-25 lbs | 4 | -0.000409 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_557f7ff982ed4de2` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.012711 | +0.007404 |
| 只保留主体分差 | -0.017117 | +0.021342 |
| 主体 token 数 | 330 | 130 |

[成功对照 token 表](../photo_557f7ff982ed4de2/tokens.csv) · [对照摘要](../photo_557f7ff982ed4de2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
