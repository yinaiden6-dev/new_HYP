# B03 · photo_f2049cd8617674da

按当前评测真值分析，未新增身份人工复核。

包装 `package_055`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_004780`，**纯图像实际 Top-1 错误参考** `reference_004779`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.734768 |
| 错误参考 MaxSim | +0.735318 |
| 正确减错误的分差 | -0.000550 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 136 个。正向贡献 +0.024455，负向贡献 -0.025005，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 584 | -0.000468 | -0.000585 |
| nonvisual | 11 | -0.000087 | -0.005796 |
| subject_nontext | 76 | +0.001161 | +0.011171 |
| subject_text | 60 | -0.001157 | -0.014090 |

最主要负差额来自 `subject_text`，贡献 -0.001157。只保留主体 query token 后，本对分差为 +0.000026。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 230 | 48.7% | True | +5.441197 | 57.8% |
| 错误 | 206 | 50.4% | True | +6.659075 | 57.8% |

主体中 60 个 token 与 OCR 行相交，30 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 311 | subject_text | -0.557473 | When reconstituted with | • (1) Single-use vial containing lyophilized powder | When reconstituted with |
| 185 | background | -0.418097 |  |  | When reconstituted with |
| 198 | background | -0.293331 |  |  | When reconstituted with |
| 715 | background | -0.285646 |  | contains 1.2% sucrose, 2.4% mannitol, | contains 1.2% sucrose, 2.4% mannitol, |
| 505 | background | -0.280294 |  |  | When reconstituted with |
| 201 | background | -0.278872 |  |  | When reconstituted with |
| 691 | background | -0.277714 |  |  |  |
| 335 | subject_text | -0.273818 | contains 1.2% sucrose, 2.4% mannitol, |  | contains 1.2% sucrose, 2.4% mannitol, |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 240 | background | +0.459188 |  |  |  |
| 120 | background | +0.411698 |  |  |  |
| 111 | background | +0.405153 |  |  |  |
| 254 | subject_nontext | +0.265079 |  |  |  |
| 554 | subject_text | +0.243393 | 71104-933-01 | 2000 IU Nominal |  |
| 555 | subject_text | +0.219122 | 71104-933-01 | 371104-933-014 | 3 71104-911-01 2 |
| 130 | background | +0.211000 |  |  |  |
| 249 | background | +0.208529 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 5 mL liquid diluent, the product | 10 | -0.001500 |
| contains 1.2% sucrose, 2.4% mannitol, | 6 | -0.000525 |
| 0.01%polysorbate 20. | 6 | +0.000144 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_dd1b9056edc18f04` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000550 | +0.016457 |
| 只保留主体分差 | +0.000026 | +0.022613 |
| 主体 token 数 | 136 | 119 |

[成功对照 token 表](../photo_dd1b9056edc18f04/tokens.csv) · [对照摘要](../photo_dd1b9056edc18f04/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
