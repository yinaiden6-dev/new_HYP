# B03 · photo_8a62e5cd6902b0ae

按当前评测真值分析，未新增身份人工复核。

包装 `package_029`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001023`，**纯图像实际 Top-1 错误参考** `reference_001310`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.665417 |
| 错误参考 MaxSim | +0.681381 |
| 正确减错误的分差 | -0.015964 |
| F03 的正确参考排名（辅助对照） | 5 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 143 个。正向贡献 +0.025713，负向贡献 -0.041677，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 577 | -0.011106 | -0.014070 |
| nonvisual | 11 | +0.000287 | +0.019098 |
| subject_nontext | 86 | -0.002962 | -0.025176 |
| subject_text | 57 | -0.002184 | -0.028005 |

最主要负差额来自 `background`，贡献 -0.011106。只保留主体 query token 后，本对分差为 -0.026304。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 41 | 40.3% | True | +5.657842 | 74.5% |
| 错误 | 18 | 4.6% | False | 不可评估 | 不可评估 |

主体中 57 个 token 与 OCR 行相交，14 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 138 | background | -0.440777 |  |  | is torn or open. |
| 449 | subject_text | -0.417108 | is torn or open. | Is tom or open. | is torn or open. |
| 702 | background | -0.411327 |  |  | Cold |
| 634 | background | -0.401140 |  | 20 EFFERVESCENT TABLETS |  |
| 473 | background | -0.367479 |  | 3 or more alcoholic drinks every day while using this product |  |
| 314 | subject_nontext | -0.352889 |  |  |  |
| 709 | background | -0.341659 |  |  |  |
| 636 | background | -0.327910 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 357 | subject_text | +0.487428 | &Flu | &Flu | Cold |
| 58 | background | +0.448605 |  | &Flu |  |
| 57 | background | +0.426841 |  | &Flu |  |
| 158 | background | +0.382889 |  | &Flu |  |
| 38 | background | +0.378043 |  | &Flu |  |
| 240 | background | +0.375211 |  | &Flu |  |
| 557 | background | +0.365606 |  | &Flu | Cold |
| 98 | background | +0.354375 |  | &Flu |  |

## 同包装成功对照

`photo_079284acc905a73f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.015964 | +0.005185 |
| 只保留主体分差 | -0.026304 | +0.015135 |
| 主体 token 数 | 143 | 120 |

[成功对照 token 表](../photo_079284acc905a73f/tokens.csv) · [对照摘要](../photo_079284acc905a73f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
