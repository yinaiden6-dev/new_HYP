# B03 · photo_b05f3285c039c3be

按当前评测真值分析，未新增身份人工复核。

包装 `package_022`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000976`，**纯图像实际 Top-1 错误参考** `reference_004062`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2135 |
| 正确参考 MaxSim | +0.497155 |
| 错误参考 MaxSim | +0.542030 |
| 正确减错误的分差 | -0.044875 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 80 个。正向贡献 +0.014988，负向贡献 -0.059863，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 640 | -0.042723 | -0.048797 |
| nonvisual | 11 | -0.000020 | -0.001311 |
| subject_nontext | 53 | -0.000164 | -0.002256 |
| subject_text | 27 | -0.001969 | -0.053304 |

最主要负差额来自 `background`，贡献 -0.042723。只保留主体 query token 后，本对分差为 -0.019485。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 147 | 33.2% | True | +8.406170 | 74.7% |
| 错误 | 9 | 13.5% | False | 不可评估 | 不可评估 |

主体中 27 个 token 与 OCR 行相交，7 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 95 | background | -0.572027 |  |  |  |
| 279 | background | -0.543043 |  | Kea t of naach of cids. In cae ol verdos, get medcal he or conact a Polaon Contrl Canter rght ay. 1-800-222-1222 |  |
| 105 | background | -0.534145 |  |  |  |
| 414 | subject_text | -0.516294 | Thia product a not manufactured or datributed by Mcel Conaumar Halare, Dion  | * Thia product la not manufacturad or datrlbuted by Mchell Conaumar Healthcare, Divtalon of |  |
| 36 | background | -0.501324 |  |  |  |
| 108 | background | -0.493255 |  |  |  |
| 423 | background | -0.487535 |  |  |  |
| 276 | background | -0.483436 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 439 | background | +0.268149 |  | Allergy Relief |  |
| 53 | background | +0.247605 |  | Allergy Relief |  |
| 240 | background | +0.245644 |  | Allergy Relief |  |
| 353 | subject_text | +0.241599 | McNell-PPC Inc., dbtribuior of ZyrtacTableta. | Allergy Relief |  |
| 113 | background | +0.219724 |  | Allergy Relief |  |
| 243 | background | +0.218622 |  | Allergy Relief |  |
| 247 | background | +0.204840 |  | Allergy Relief |  |
| 110 | background | +0.202786 |  | Allergy Relief |  |

## 同包装成功对照

`photo_0cf112e57c488bf5` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.044875 | +0.214800 |
| 只保留主体分差 | -0.019485 | +0.329544 |
| 主体 token 数 | 80 | 90 |

[成功对照 token 表](../photo_0cf112e57c488bf5/tokens.csv) · [对照摘要](../photo_0cf112e57c488bf5/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
