# B03 · photo_93ed25f14350d103

按当前评测真值分析，未新增身份人工复核。

包装 `package_021`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000973`，**纯图像实际 Top-1 错误参考** `reference_000992`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 109 |
| 正确参考 MaxSim | +0.606016 |
| 错误参考 MaxSim | +0.654963 |
| 正确减错误的分差 | -0.048947 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 190 个。正向贡献 +0.022720，负向贡献 -0.071667，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.032502 | -0.044829 |
| nonvisual | 11 | -0.000054 | -0.003598 |
| subject_nontext | 63 | -0.004922 | -0.057111 |
| subject_text | 127 | -0.011469 | -0.066015 |

最主要负差额来自 `background`，贡献 -0.032502。只保留主体 query token 后，本对分差为 -0.063063。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 787 | 65.7% | True | +8.740107 | 71.3% |
| 错误 | 18 | 20.5% | True | +7.442135 | 76.5% |

主体中 127 个 token 与 OCR 行相交，101 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 240 | background | -0.432824 |  | profosalonal betora usa. it ls especialy importunt | professional before use. It is especially |
| 259 | subject_nontext | -0.432501 |  |  | the last 3 months of pregnancy unless |
| 461 | background | -0.415666 |  |  tamporarily ralleves minor aches and paina due to: |  |
| 111 | background | -0.412354 |  |  | professional before use. It is especially |
| 256 | subject_text | -0.384828 | not to use naproxon sod lum during the last 3 |  | the last 3 months of pregnancy unless |
| 474 | subject_text | -0.378893 | Inactive ingrodionts FD&C bluo #2 | Drug Facts | Inactive ingredients FD&C blue #2 |
| 408 | background | -0.375630 |  |  |  |
| 472 | subject_text | -0.369443 | Inactive ingrodionts FD&C bluo #2 | Drug Facts | Inactive ingredients FD&C blue #2 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 234 | subject_text | +0.468977 | pragnant or kreast-feedkeg, ask x health | Drug Facts (contlnued) |  |
| 104 | background | +0.354542 |  |  |  |
| 107 | background | +0.342294 |  |  | stomach bleeding. The chance is higher if you |
| 233 | subject_text | +0.261428 | pragnant or kreast-feedkeg, ask x health | Drug Facts (contlnued) |  |
| 369 | background | +0.234671 |  |  |  |
| 137 | background | +0.234123 |  |  |  |
| 232 | subject_text | +0.224606 | pragnant or kreast-feedkeg, ask x health | Drug Facts (contlnued) |  |
| 138 | background | +0.214273 |  |  | professional before use. It is especially |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| each caplst courtains: sodium 20 mg | 10 | -0.001461 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_38d1d33620ab1ea1` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.048947 | +0.049277 |
| 只保留主体分差 | -0.063063 | +0.093175 |
| 主体 token 数 | 190 | 54 |

[成功对照 token 表](../photo_38d1d33620ab1ea1/tokens.csv) · [对照摘要](../photo_38d1d33620ab1ea1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
