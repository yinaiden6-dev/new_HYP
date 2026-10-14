# B03 · photo_86a37b0da0dd5ece

按当前评测真值分析，未新增身份人工复核。

包装 `package_044`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001133`，**纯图像实际 Top-1 错误参考** `reference_002524`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 12 |
| 正确参考 MaxSim | +0.604649 |
| 错误参考 MaxSim | +0.615220 |
| 正确减错误的分差 | -0.010572 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 112 个。正向贡献 +0.042170，负向贡献 -0.052741，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 608 | -0.009146 | -0.010997 |
| nonvisual | 11 | -0.000092 | -0.006128 |
| subject_nontext | 31 | +0.000730 | +0.017220 |
| subject_text | 81 | -0.002063 | -0.018622 |

最主要负差额来自 `background`，贡献 -0.009146。只保留主体 query token 后，本对分差为 -0.008701。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 563 | 56.7% | True | +13.994468 | 91.7% |
| 错误 | 8 | 11.5% | False | 不可评估 | 不可评估 |

主体中 81 个 token 与 OCR 行相交，76 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 444 | background | -0.372856 |  |  |  |
| 130 | background | -0.353898 |  |  |  |
| 445 | background | -0.334627 |  |  |  |
| 62 | background | -0.315856 |  |  |  |
| 202 | background | -0.313848 |  | Purpose |  |
| 388 | background | -0.304565 |  |  | PAPER |
| 52 | background | -0.278472 |  |  |  |
| 97 | background | -0.267540 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 517 | subject_text | +0.651988 | LEADER? | LEADER? | PAPER |
| 101 | background | +0.641080 |  | LEADER? |  |
| 91 | background | +0.628753 |  | LEADER? |  |
| 711 | background | +0.561925 |  |  | Cold, Flu & Hay Fever Multi-Symptom Relief |
| 219 | background | +0.509068 |  | LEADER |  |
| 220 | background | +0.504200 |  | LEADER | Drug Facts (continued) |
| 531 | subject_nontext | +0.496054 |  | LEADER | Cold, Flu & Hay Fever Multi-Symptom Relief |
| 518 | subject_text | +0.486013 | LEADER? | LEADER | Cold, Flu & Hay Fever Multi-Symptom Relief |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen 160 mg. | 3 | -0.000066 |
| Active ingredient (in each 5 mL) | 5 | -0.000618 |
| Acetamlnophen 160 mg per 5 mL | 7 | +0.001497 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_e156cd0eb0edd234` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010572 | +0.092669 |
| 只保留主体分差 | -0.008701 | +0.130791 |
| 主体 token 数 | 112 | 98 |

[成功对照 token 表](../photo_e156cd0eb0edd234/tokens.csv) · [对照摘要](../photo_e156cd0eb0edd234/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
