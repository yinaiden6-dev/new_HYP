# B03 · photo_dce2ba688bc88c07

按当前评测真值分析，未新增身份人工复核。

包装 `package_080`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003367`，**纯图像实际 Top-1 错误参考** `reference_004600`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.575695 |
| 错误参考 MaxSim | +0.578261 |
| 正确减错误的分差 | -0.002567 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 198 个。正向贡献 +0.025589，负向贡献 -0.028156，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 570 | -0.004364 | -0.005964 |
| nonvisual | 11 | -0.000337 | -0.023837 |
| subject_nontext | 29 | +0.000904 | +0.024294 |
| subject_text | 169 | +0.001230 | +0.005669 |

最主要负差额来自 `background`，贡献 -0.004364。只保留主体 query token 后，本对分差为 +0.008397。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 472 | 75.4% | True | +5.308870 | 58.2% |
| 错误 | 30 | 30.1% | True | +6.778179 | 66.0% |

主体中 169 个 token 与 OCR 行相交，121 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 401 | subject_text | -0.421882 | directed. Frequent or prolonged use may cause nasal | 12 HOUR PUMP MIST | temporary discomfort such as buming, stinging, sneezing |
| 370 | subject_text | -0.394721 | sneezing or an increase in nasal discharge may occur | sneezing or an increase in nasal discharge may occur |  |
| 251 | subject_text | -0.337798 | firm, even stroke and sniff deeply. Wipe nozzle clean | firm, even stroke and sniff deeply. Wipe nozzle clean | clockwise. To spray, squeeze bottle quickly and firmly. Do |
| 740 | background | -0.329959 |  |  |  |
| 428 | subject_text | -0.294973 | conmon coldhay fever |  |  |
| 727 | background | -0.275500 | ATIONS |  |  |
| 416 | background | -0.270484 | RT'S |  |  |
| 410 | subject_text | -0.247929 | nozzle between first and second fingers. Without tilting |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 445 | subject_nontext | +0.416586 |  |  | Keep out of reach of children. If swallowed, get medical |
| 431 | subject_text | +0.404213 | thyrold diseasediabetes | Ask a doctor before use If you have | diabetes |
| 444 | subject_text | +0.396497 | SAFETY SEALED: Do not use Il Imprinted seat on botlle | SAFETY SEALED: Do not use If Imprinted seal on bottle | distributor of Afrin®. |
| 412 | subject_text | +0.393246 | SAFETY SEALED: Do not use Il Imprinted seat on botlle |  | Secure cap after use. |
| 572 | background | +0.377142 |  | SAFETY SEALED: Do not use If Imprinted seal on bottle | Secure cap after use. |
| 637 | background | +0.337571 |  |  |  |
| 731 | background | +0.315949 |  |  |  |
| 764 | background | +0.299773 |  | SAFETY SEALED: Do not use If Imprinted seal on bottle | DECONGESTANT |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Oxymetazoline hydrochloride 0.05%... Nasal Decongestant | 16 | -0.001083 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_174967c2577c6dc9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002567 | +0.030309 |
| 只保留主体分差 | +0.008397 | +0.090859 |
| 主体 token 数 | 198 | 175 |

[成功对照 token 表](../photo_174967c2577c6dc9/tokens.csv) · [对照摘要](../photo_174967c2577c6dc9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
