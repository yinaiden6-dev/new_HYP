# B03 · photo_c8920e14924534af

按当前评测真值分析，未新增身份人工复核。

包装 `package_084`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_004975`，**纯图像实际 Top-1 错误参考** `reference_004976`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.533403 |
| 错误参考 MaxSim | +0.540485 |
| 正确减错误的分差 | -0.007082 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 88 个。正向贡献 +0.016658，负向贡献 -0.023739，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 680 | -0.004543 | -0.005205 |
| nonvisual | 11 | -0.000018 | -0.001299 |
| subject_nontext | 42 | -0.000957 | -0.017745 |
| subject_text | 46 | -0.001564 | -0.026481 |

最主要负差额来自 `background`，贡献 -0.004543。只保留主体 query token 后，本对分差为 -0.022312。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 174 | 34.8% | True | +8.739888 | 70.1% |
| 错误 | 138 | 32.6% | True | +7.950884 | 64.8% |

主体中 46 个 token 与 OCR 行相交，30 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 27 | background | -0.586312 |  |  |  |
| 595 | background | -0.375274 |  |  |  |
| 313 | background | -0.277955 |  |  |  |
| 499 | subject_text | -0.266511 | Mfg. Lic. No.: 36/MD/AP/2013/F/G |  |  |
| 753 | background | -0.254780 |  |  |  |
| 435 | subject_text | -0.245666 | Mfg. Lic. No.: 36/MD/AP/2013/F/G |  |  |
| 289 | background | -0.244861 |  |  |  |
| 353 | background | -0.244220 | COIE |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 310 | subject_text | +0.303651 | 2005 Northstar Healthcare Holdings Ltd. |  |  |
| 120 | background | +0.235485 | Pharmacy |  | Manufactured by: |
| 172 | background | +0.217392 |  |  | Manufactured by: |
| 152 | background | +0.215289 | Pharmacy |  | Manufactured by: |
| 119 | background | +0.210507 | Pharmacy |  | Manufactured by: |
| 502 | subject_text | +0.199624 | 37260332710116 | N3 |  |
| 238 | background | +0.192705 |  |  | Iss.: 06/2024 |
| 151 | background | +0.189454 | Pharmacy |  | Manufactured by: |

## 同包装成功对照

`photo_45b5970f1ebe4f19` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007082 | +0.001765 |
| 只保留主体分差 | -0.022312 | +0.004502 |
| 主体 token 数 | 88 | 99 |

[成功对照 token 表](../photo_45b5970f1ebe4f19/tokens.csv) · [对照摘要](../photo_45b5970f1ebe4f19/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
