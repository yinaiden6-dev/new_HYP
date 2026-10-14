# B03 · photo_8c54b7644f2e05b2

按当前评测真值分析，未新增身份人工复核。

包装 `package_084`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_004975`，**纯图像实际 Top-1 错误参考** `reference_004976`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 10 |
| 正确参考 MaxSim | +0.540843 |
| 错误参考 MaxSim | +0.555692 |
| 正确减错误的分差 | -0.014849 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 165 个。正向贡献 +0.014539，负向贡献 -0.029389，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 603 | -0.009063 | -0.011708 |
| nonvisual | 11 | -0.000175 | -0.012374 |
| subject_nontext | 91 | -0.001992 | -0.017053 |
| subject_text | 74 | -0.003620 | -0.038104 |

最主要负差额来自 `background`，贡献 -0.009063。只保留主体 query token 后，本对分差为 -0.026494。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 192 | 44.3% | True | +6.553124 | 66.7% |
| 错误 | 152 | 42.0% | True | +6.187395 | 62.8% |

主体中 74 个 token 与 OCR 行相交，44 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 35 | background | -0.536095 |  |  |  |
| 367 | subject_text | -0.415421 | Telangana-502281,INDIA |  |  |
| 273 | subject_text | -0.358975 | Memphis, TN 38141. |  |  |
| 368 | subject_text | -0.357787 | Telangana-502281,INDIA |  |  |
| 306 | subject_text | -0.324521 | Manufactured by: |  |  |
| 275 | subject_text | -0.310771 | Memphis, TN 38141. | Memphis, TN 38141. |  |
| 528 | subject_text | -0.298871 | WZ | NDC 72603-327-10 |  |
| 304 | subject_text | -0.276797 | Manufactured by: |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 567 | subject_text | +0.350080 | 2005 Northstar Healthcare Holdings Ltd. |  | © 2005 Northstar Healthcare Holdings Ltd. |
| 566 | subject_text | +0.327269 | 726031132710111 | © 2005 Northstar Healthcare Holdings Ltd. | © 2005 Northstar Healthcare Holdings Ltd. |
| 366 | subject_nontext | +0.222618 |  |  |  |
| 18 | background | +0.217190 | Perficore |  |  |
| 564 | subject_text | +0.185740 | 726031132710111 |  | © 2005 Northstar Healthcare Holdings Ltd. |
| 374 | subject_text | +0.183111 | Mfg. Lic. No.: 36/MD/AP/2013/F/G |  |  |
| 335 | subject_text | +0.175914 | Aspiro Pharma Limited | Aspiro Pharma Limited | Manufactured by: |
| 763 | background | +0.150188 |  |  |  |

## 同包装成功对照

`photo_709e8a3ee42539bc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014849 | +0.008916 |
| 只保留主体分差 | -0.026494 | +0.026366 |
| 主体 token 数 | 165 | 153 |

[成功对照 token 表](../photo_709e8a3ee42539bc/tokens.csv) · [对照摘要](../photo_709e8a3ee42539bc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
