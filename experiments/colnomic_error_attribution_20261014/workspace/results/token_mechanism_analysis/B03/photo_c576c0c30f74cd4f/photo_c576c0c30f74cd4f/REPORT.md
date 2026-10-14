# B03 · photo_c576c0c30f74cd4f

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003300`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 10 |
| 正确参考 MaxSim | +0.699610 |
| 错误参考 MaxSim | +0.713393 |
| 正确减错误的分差 | -0.013783 |
| F03 的正确参考排名（辅助对照） | 10 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 85 个。正向贡献 +0.015186，负向贡献 -0.028969，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 635 | -0.010762 | -0.012389 |
| nonvisual | 11 | -0.000168 | -0.011149 |
| subject_nontext | 22 | -0.000352 | -0.011709 |
| subject_text | 63 | -0.002501 | -0.029016 |

最主要负差额来自 `background`，贡献 -0.010762。只保留主体 query token 后，本对分差为 -0.024536。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 486 | 56.0% | True | +6.548046 | 67.1% |
| 错误 | 496 | 56.2% | True | +6.559706 | 64.7% |

主体中 63 个 token 与 OCR 行相交，53 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 371 | subject_text | -0.411104 | NON-COMEDOGENIC | TESTÉ DERMATOLOGIQUEMENT | HYPOALLERGENIC |
| 172 | background | -0.338279 |  |  |  |
| 546 | background | -0.279826 |  |  | HYPOALLERGENIC |
| 552 | subject_text | -0.269992 | TESTÉ SOUS CONTROLE | TESTÉ DERMATOLOGIQUEMENT | HYPOALLERGENIC |
| 78 | background | -0.253973 |  |  | HYPOALLERGENIC |
| 512 | subject_text | -0.252113 | POUR DES RETOUCHES. |  |  |
| 292 | subject_text | -0.232508 | PORE-CLOGGING OILS |  |  |
| 372 | subject_text | -0.231619 | NON-COMEDOGENIC |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 422 | background | +0.193185 |  |  |  |
| 428 | background | +0.164045 |  |  | CE FOND DE TEINT LIQUIDE |
| 145 | background | +0.157338 |  | CE FOND DE TEINT LIQUIDE |  |
| 358 | background | +0.156949 |  | OR FOR TOUCH-UPS. |  |
| 423 | background | +0.147050 |  |  |  |
| 613 | background | +0.144214 |  | CE FOND DE TEINT LIQUIDE |  |
| 123 | background | +0.139961 |  |  |  |
| 177 | background | +0.132396 |  |  |  |

## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
