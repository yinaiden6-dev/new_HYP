# B03 · photo_31521050e001b9b4

按当前评测真值分析，未新增身份人工复核。

包装 `package_097`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000792`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 23 |
| 正确参考 MaxSim | +0.550223 |
| 错误参考 MaxSim | +0.581044 |
| 正确减错误的分差 | -0.030820 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 184 个。正向贡献 +0.056124，负向贡献 -0.086945，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 584 | -0.053593 | -0.071488 |
| nonvisual | 11 | -0.000915 | -0.064783 |
| subject_nontext | 45 | -0.000171 | -0.002963 |
| subject_text | 139 | +0.023859 | +0.133713 |

最主要负差额来自 `background`，贡献 -0.053593。只保留主体 query token 后，本对分差为 +0.100287。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 281 | 67.3% | True | +6.641054 | 67.8% |
| 错误 | 12 | 23.6% | True | +13.095529 | 94.4% |

主体中 139 个 token 与 OCR 行相交，97 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 220 | background | -0.606865 | ASAL AL | STERILE • HOMEOPATHIC | NASAL MIST |
| 190 | background | -0.592714 | ASAL AL | STERILE • HOMEOPATHIC | ASAL L |
| 95 | background | -0.589118 | No Drip | Active ingredients ....Purpose | No Drip |
| 157 | background | -0.588948 | ASAL AL | Inactive ingredients: | MIST |
| 125 | background | -0.586374 | ASAL AL |  | NASAL MIST |
| 156 | background | -0.578929 | ASAL AL | ORG1012-F |  |
| 186 | background | -0.572564 | congestion relief |  | Fast, powerful |
| 416 | background | -0.570141 | skin reactions. Sympto | STERILE • HOMEOPATHIC | 7 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 370 | subject_text | +0.551456 | you might have a serious underlying | you might have a serious underlying | ASAL L |
| 428 | subject_text | +0.495467 | yellow, or green discharge that forms a | yellow, or green discharge that forms a | No Drip |
| 347 | subject_text | +0.491422 | sulphate (as preservative), Sodium nitrate |  |  |
| 401 | subject_text | +0.480771 | you might have a serious underlying | you might have a serious underlying | ASAL L |
| 526 | subject_text | +0.467523 | in vision | changes in vision occur |  |
| 166 | background | +0.459609 | symptoms with Similasan Kids | According to homeopathic principles, the active |  |
| 508 | subject_text | +0.457151 | (1-800-WALGREENS (1-800-925-4733) | Walgreens | head, insert nozzle into nostril. Fully depress rim with a |
| 357 | background | +0.456641 | www.SimilasanUSA.com | Compare to Similasan® Irritated | 050428588352 |

## 同包装成功对照

`photo_53000d6acda74399` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.030820 | +0.042773 |
| 只保留主体分差 | +0.100287 | +0.227551 |
| 主体 token 数 | 184 | 176 |

[成功对照 token 表](../photo_53000d6acda74399/tokens.csv) · [对照摘要](../photo_53000d6acda74399/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
