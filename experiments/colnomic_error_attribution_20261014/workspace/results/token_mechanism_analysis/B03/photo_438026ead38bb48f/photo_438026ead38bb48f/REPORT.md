# B03 · photo_438026ead38bb48f

按当前评测真值分析，未新增身份人工复核。

包装 `package_029`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001023`，**纯图像实际 Top-1 错误参考** `reference_001326`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.630491 |
| 错误参考 MaxSim | +0.664528 |
| 正确减错误的分差 | -0.034037 |
| F03 的正确参考排名（辅助对照） | 6 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 99 个。正向贡献 +0.024268，负向贡献 -0.058305，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 621 | -0.029677 | -0.034933 |
| nonvisual | 11 | -0.000042 | -0.002767 |
| subject_nontext | 69 | -0.001171 | -0.012403 |
| subject_text | 30 | -0.003148 | -0.076702 |

最主要负差额来自 `background`，贡献 -0.029677。只保留主体 query token 后，本对分差为 -0.031887。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 72 | 23.8% | True | +4.101495 | 58.6% |
| 错误 | 34 | 15.7% | True | +10.358634 | 85.9% |

主体中 30 个 token 与 OCR 行相交，11 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 53 | background | -0.527683 |  |  |  |
| 232 | background | -0.524581 |  | MADE IN MEXICO |  |
| 295 | subject_text | -0.513394 | Bayer, the Bayer Cross and Alka-Seltzer Plus | Bayer |  |
| 353 | subject_nontext | -0.481168 |  |  |  |
| 242 | background | -0.462405 |  |  |  |
| 233 | background | -0.461048 |  | MADE IN MEXICO |  |
| 352 | subject_text | -0.449903 | MADE IN MEXICO |  |  |
| 666 | background | -0.443481 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 253 | subject_nontext | +0.453919 |  | 20 EFFERVESCENT TABLETS | Whippany, NJ 07981 |
| 266 | background | +0.329091 |  | 20 EFFERVESCENT TABLETS | PARENTS |
| 143 | background | +0.328568 |  | 20 EFFERVESCENT TABLETS |  |
| 634 | background | +0.325083 |  | 20 EFFERVESCENT TABLETS |  |
| 434 | background | +0.305309 |  | 20 EFFERVESCENT TABLETS |  |
| 114 | background | +0.304237 |  | 20 EFFERVESCENT TABLETS | Whippany, NJ 07981 |
| 633 | background | +0.295281 |  | 20 EFFERVESCENT TABLETS | PARENTS |
| 291 | subject_nontext | +0.242400 |  |  | Whippany, NJ 07981 |

## 同包装成功对照

`photo_079284acc905a73f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.034037 | +0.011720 |
| 只保留主体分差 | -0.031887 | +0.031164 |
| 主体 token 数 | 99 | 120 |

[成功对照 token 表](../photo_079284acc905a73f/tokens.csv) · [对照摘要](../photo_079284acc905a73f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
