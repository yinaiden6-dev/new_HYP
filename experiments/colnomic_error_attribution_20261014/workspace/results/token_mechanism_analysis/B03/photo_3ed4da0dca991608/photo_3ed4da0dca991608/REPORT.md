# B03 · photo_3ed4da0dca991608

按当前评测真值分析，未新增身份人工复核。

包装 `package_051`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001190`，**纯图像实际 Top-1 错误参考** `reference_001630`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.549976 |
| 错误参考 MaxSim | +0.557639 |
| 正确减错误的分差 | -0.007663 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 35 个。正向贡献 +0.040987，负向贡献 -0.048650，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 685 | -0.008263 | -0.008818 |
| nonvisual | 11 | -0.000930 | -0.061829 |
| subject_nontext | 6 | -0.000174 | -0.021153 |
| subject_text | 29 | +0.001704 | +0.042945 |

最主要负差额来自 `background`，贡献 -0.008263。只保留主体 query token 后，本对分差为 +0.031957。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 116 | 27.9% | True | +7.316689 | 71.4% |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 29 个 token 与 OCR 行相交，20 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 463 | background | -0.339565 |  |  | •Supervse children under 6 years of age when using this product to avoid |
| 578 | background | -0.332738 |  |  |  |
| 659 | background | -0.328046 |  |  |  |
| 564 | background | -0.308961 |  |  |  |
| 614 | background | -0.301953 |  |  |  |
| 447 | background | -0.290190 |  | PRODUCT NAME: |  |
| 663 | background | -0.289149 |  |  |  |
| 720 | background | -0.284782 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 214 | background | +0.491937 |  | DIRECTIONS OF SAFE USE |  |
| 334 | subject_text | +0.472787 | DIRECTIONS OFSAFEUSE | DIRECTIONS OF SAFE USE |  |
| 353 | subject_text | +0.455828 | and drythelace | DIRECTIONS OF SAFE USE |  |
| 328 | background | +0.451747 |  | DIRECTIONS OF SAFE USE |  |
| 515 | background | +0.439711 |  | DIRECTIONS OF SAFE USE |  |
| 211 | background | +0.416979 |  | DIRECTIONS OF SAFE USE |  |
| 528 | background | +0.416357 |  | DIRECTIONS OF SAFE USE |  |
| 25 | background | +0.402302 |  |  |  |

## 同包装成功对照

`photo_567000412151b626` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007663 | +0.311472 |
| 只保留主体分差 | +0.031957 | +0.295747 |
| 主体 token 数 | 35 | 42 |

[成功对照 token 表](../photo_567000412151b626/tokens.csv) · [对照摘要](../photo_567000412151b626/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
