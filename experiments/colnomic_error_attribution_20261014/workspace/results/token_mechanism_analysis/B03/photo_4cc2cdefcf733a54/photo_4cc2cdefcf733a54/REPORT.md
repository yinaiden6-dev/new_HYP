# B03 · photo_4cc2cdefcf733a54

按当前评测真值分析，未新增身份人工复核。

包装 `package_057`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001410`，**纯图像实际 Top-1 错误参考** `reference_002830`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.643764 |
| 错误参考 MaxSim | +0.648059 |
| 正确减错误的分差 | -0.004294 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.026533，负向贡献 -0.030828，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.004395 | -0.004958 |
| nonvisual | 11 | -0.000209 | -0.013910 |
| subject_nontext | 46 | +0.000416 | +0.006616 |
| subject_text | 26 | -0.000106 | -0.002993 |

最主要负差额来自 `background`，贡献 -0.004395。只保留主体 query token 后，本对分差为 +0.003146。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 335 | 29.1% | True | +5.921352 | 71.8% |
| 错误 | 5 | 3.8% | False | 不可评估 | 不可评估 |

主体中 26 个 token 与 OCR 行相交，21 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 333 | subject_text | -0.262772 | Active Ingredient (in each tablet) | ANTACID |  |
| 174 | background | -0.252489 |  |  | Ask a doctor or phamacist before use if you are |
| 713 | background | -0.240049 |  |  |  |
| 572 | background | -0.239945 |  |  |  |
| 569 | background | -0.238392 |  |  |  |
| 570 | background | -0.237412 |  | Other Information:Tamper Evident. Do not use if packet is torn, cut or opened | complications during delivery |
| 435 | subject_nontext | -0.230747 |  |  | If pregnant or breast-feeding, ask a health |
| 558 | background | -0.229329 |  |  | Ask a doctor or phamacist before use if you are |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 76 | background | +0.364886 |  |  |  |
| 509 | background | +0.317621 |  |  |  |
| 512 | background | +0.225641 |  | Active Ingredient (in each tablet) |  |
| 721 | background | +0.223440 |  |  | reaction to any other pain reliever/fever reducer |
| 506 | background | +0.219677 |  | Active Ingredient (in each tablet) |  |
| 313 | subject_text | +0.208758 | Active Ingredient (in each tablet) |  |  |
| 327 | background | +0.206555 |  | Active Ingredient (in each tablet) |  |
| 196 | background | +0.206321 |  | away, Prompt medical attention is critical for adults as well as for children even if you do not notice any signs or |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Calcium Carbonate 420 mg | 2 | -0.000106 |
| De net:  Take more than 18 tablets in a 24 hour period, or use the maximum dosage of this product for more than 2 | 8 | -0.000130 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_49600b91aa952055` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004294 | +0.181495 |
| 只保留主体分差 | +0.003146 | +0.219175 |
| 主体 token 数 | 72 | 56 |

[成功对照 token 表](../photo_49600b91aa952055/tokens.csv) · [对照摘要](../photo_49600b91aa952055/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
